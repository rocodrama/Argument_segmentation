import os
import argparse
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from pathlib import Path
from PIL import Image
from torchvision import transforms
import numpy as np

# SMP 라이브러리
import segmentation_models_pytorch as smp
from torch.utils.data import Dataset

# ------------------------------
# 1. 데이터셋 클래스 (최적화: 1채널 & 정규화 제거)
# ------------------------------
class MedicalMaskDataset(Dataset):
    def __init__(self, img_dir, mask_dir, size=512):
        self.img_dir = Path(img_dir)
        self.mask_dir = Path(mask_dir)
        self.size = size
        
        self.images = sorted([f.name for f in self.img_dir.iterdir() if f.suffix.lower() in ['.jpg', '.png', '.jpeg', '.tiff', '.bmp']])
        
        # [수정 1] ImageNet 정규화 제거 -> 0~1 스케일링만 수행 (흑백 텍스처 보존)
        self.transform = transforms.Compose([
            transforms.Resize((size, size)),
            transforms.ToTensor(), 
            # transforms.Normalize(...) <-- 제거함
        ])
        
        self.mask_transform = transforms.Compose([
            transforms.Resize((size, size), interpolation=transforms.InterpolationMode.NEAREST),
            transforms.ToTensor()
        ])

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        img_name = self.images[idx]
        img_path = self.img_dir / img_name
        
        # [수정 2] 이미지를 흑백(L)으로 로드
        image = Image.open(img_path).convert("L")
        
        mask_name = img_name  
        mask_path = self.mask_dir / mask_name
        
        # 확장자 매칭 로직
        if not mask_path.exists():
             for ext in ['.png', '.jpg', '.jpeg', '.tiff', '.bmp']:
                 test_path = self.mask_dir / (Path(img_name).stem + ext)
                 if test_path.exists():
                     mask_path = test_path
                     break
        
        if not mask_path.exists():
            print(f"⚠️ 경고: 마스크 파일을 찾을 수 없음 - {mask_path}")
            mask = Image.new("L", image.size, 0)
        else:
            mask = Image.open(mask_path).convert("L")
        
        return self.transform(image), self.mask_transform(mask)

# ------------------------------
# 2. 모델 선택 헬퍼 (최적화: 가중치 None, 1채널)
# ------------------------------
def get_model(model_name, encoder='resnet34', in_channels=1, classes=1):
    print(f"🏗️ Model: {model_name} | Backbone: {encoder} | Input: {in_channels}ch (Grayscale) | Pretrained: None")
    
    models = {
        'Unet': smp.Unet,
        'UnetPlusPlus': smp.UnetPlusPlus,
        'DeepLabV3Plus': smp.DeepLabV3Plus,
        'MAnet': smp.MAnet,
        'FPN': smp.FPN,
        'PAN': smp.PAN
    }
    
    if model_name not in models:
        raise ValueError(f"지원하지 않는 모델명입니다. 가능한 모델: {list(models.keys())}")
    
    return models[model_name](
        encoder_name=encoder, 
        encoder_weights=None,  # [수정 3] ImageNet 가중치 사용 안 함 (의료 영상 전용 학습)
        in_channels=in_channels, 
        classes=classes
    )

# ------------------------------
# 3. 학습 함수
# ------------------------------
def train(args):
    device = torch.device(f"cuda:{args.gpu}" if torch.cuda.is_available() else "cpu")
    os.makedirs(args.output_dir, exist_ok=True)
    
    print("📂 데이터 로딩 중... (Mode: Grayscale)")
    train_dataset = MedicalMaskDataset(img_dir=args.train_img, mask_dir=args.train_mask, size=args.size)
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers)
    
    val_loader = None
    if args.val_img and args.val_mask:
        val_dataset = MedicalMaskDataset(img_dir=args.val_img, mask_dir=args.val_mask, size=args.size)
        val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers)
        print(f"📊 Data Count: Train={len(train_dataset)}, Val={len(val_dataset)}")
    else:
        print(f"📊 Data Count: Train={len(train_dataset)} (Validation 없음)")

    # 모델 로드 (in_channels=1)
    model = get_model(args.model, encoder=args.encoder, in_channels=1).to(device)
    
    # [수정 4] Loss 함수 변경: Dice + BCE 결합
    criterion_dice = smp.losses.DiceLoss(mode='binary', from_logits=True)
    criterion_bce = nn.BCEWithLogitsLoss()
    
    optimizer = optim.AdamW(model.parameters(), lr=args.lr)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs, eta_min=1e-6)

    best_iou = 0.0
    
    for epoch in range(args.epochs):
        model.train()
        train_loss = 0.0
        
        print(f"\n[Epoch {epoch+1}/{args.epochs}] Training...")
        
        for i, (images, masks) in enumerate(train_loader):
            images = images.to(device)
            masks = masks.to(device)
            
            optimizer.zero_grad()
            logits = model(images)
            
            # Hybrid Loss 계산 (BCE로 전체 윤곽 + Dice로 세부 겹침)
            loss_dice = criterion_dice(logits, masks)
            loss_bce = criterion_bce(logits, masks)
            loss = 0.5 * loss_dice + 0.5 * loss_bce
            
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item()
        
        avg_train_loss = train_loss / len(train_loader)
        
        # --- Validation & Save ---
        if val_loader:
            model.eval()
            val_loss = 0.0
            
            tp_list, fp_list, fn_list, tn_list = [], [], [], []
            
            with torch.no_grad():
                for images, masks in val_loader:
                    images = images.to(device)
                    masks = masks.to(device)
                    
                    logits = model(images)
                    
                    # Val Loss도 동일하게 계산
                    l_dice = criterion_dice(logits, masks)
                    l_bce = criterion_bce(logits, masks)
                    loss = 0.5 * l_dice + 0.5 * l_bce
                    val_loss += loss.item()
                    
                    pred_mask = (torch.sigmoid(logits) > 0.5).long()
                    tp, fp, fn, tn = smp.metrics.get_stats(pred_mask, masks.long(), mode='binary', threshold=0.5)
                    
                    tp_list.append(tp); fp_list.append(fp)
                    fn_list.append(fn); tn_list.append(tn)
            
            iou_score = smp.metrics.iou_score(
                torch.cat(tp_list), torch.cat(fp_list), 
                torch.cat(fn_list), torch.cat(tn_list), 
                reduction="micro-imagewise"
            )
            
            avg_val_loss = val_loss / len(val_loader)
            
            print(f"  >> Result: Train Loss={avg_train_loss:.4f} | Val Loss={avg_val_loss:.4f} | Val IoU={iou_score:.4f}")
            
            if iou_score > best_iou:
                best_iou = iou_score
                best_save_path = os.path.join(args.output_dir, f"best_{args.model}_{args.encoder}.pth")
                torch.save(model.state_dict(), best_save_path)
                print(f"  🏆 New Best IoU! Saved: {best_save_path}")
            
            # Latest 저장
            latest_save_path = os.path.join(args.output_dir, f"latest_{args.model}_{args.encoder}.pth")
            torch.save(model.state_dict(), latest_save_path)
                
        else:
            print(f"  >> Result: Train Loss={avg_train_loss:.4f}")
            latest_save_path = os.path.join(args.output_dir, f"latest_{args.model}_{args.encoder}.pth")
            torch.save(model.state_dict(), latest_save_path)
        
        scheduler.step()

    print("Training Complete.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    
    parser.add_argument("--train_img", type=str, required=True, help="학습 이미지 폴더 경로")
    parser.add_argument("--train_mask", type=str, required=True, help="학습 마스크 폴더 경로")
    parser.add_argument("--val_img", type=str, default=None, help="검증 이미지 폴더 경로")
    parser.add_argument("--val_mask", type=str, default=None, help="검증 마스크 폴더 경로")
    
    parser.add_argument("--output_dir", type=str, default="seg_results", help="모델 저장 경로")
    
    parser.add_argument("--model", type=str, default="Unet", choices=['Unet', 'UnetPlusPlus', 'DeepLabV3Plus', 'MAnet', 'FPN'])
    parser.add_argument("--encoder", type=str, default="resnet34")
    
    parser.add_argument("--size", type=int, default=512)
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--gpu", type=int, default=0)
    
    args = parser.parse_args()
    train(args)