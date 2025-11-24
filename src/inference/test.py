import os
import argparse
import torch
import numpy as np
import pandas as pd  # CSV 저장을 위해 추가
from pathlib import Path
from PIL import Image
from tqdm import tqdm
from torchvision import transforms
import segmentation_models_pytorch as smp
from torch.utils.data import DataLoader, Dataset

# ------------------------------
# 1. 데이터셋 클래스
# ------------------------------
class MedicalTestDataset(Dataset):
    def __init__(self, img_dir, mask_dir, size=512):
        self.img_dir = Path(img_dir)
        self.mask_dir = Path(mask_dir)
        self.size = size
        
        self.images = sorted([f.name for f in self.img_dir.iterdir() if f.suffix.lower() in ['.jpg', '.png', '.jpeg', '.tiff', '.bmp']])
        
        self.transform = transforms.Compose([
            transforms.Resize((size, size)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
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
        
        image = Image.open(img_path).convert("RGB")
        
        mask_name = img_name
        mask_path = self.mask_dir / mask_name
        
        if not mask_path.exists():
             for ext in ['.png', '.jpg', '.jpeg', '.tiff', '.bmp']:
                 test_path = self.mask_dir / (Path(img_name).stem + ext)
                 if test_path.exists():
                     mask_path = test_path
                     break
        
        if not mask_path.exists():
            mask = Image.new("L", image.size, 0)
        else:
            mask = Image.open(mask_path).convert("L")

        return self.transform(image), self.mask_transform(mask), img_name

# ------------------------------
# 2. 모델 로드 함수
# ------------------------------
def load_model(model_name, encoder, checkpoint_path, device):
    print(f"🏗️ Loading Model: {model_name} (Backbone: {encoder})")
    
    if model_name == 'Unet':
        model = smp.Unet(encoder_name=encoder, encoder_weights=None, in_channels=3, classes=1)
    elif model_name == 'UnetPlusPlus':
        model = smp.UnetPlusPlus(encoder_name=encoder, encoder_weights=None, in_channels=3, classes=1)
    elif model_name == 'DeepLabV3Plus':
        model = smp.DeepLabV3Plus(encoder_name=encoder, encoder_weights=None, in_channels=3, classes=1)
    elif model_name == 'MAnet':
        model = smp.MAnet(encoder_name=encoder, encoder_weights=None, in_channels=3, classes=1)
    elif model_name == 'FPN':
        model = smp.FPN(encoder_name=encoder, encoder_weights=None, in_channels=3, classes=1)
    else:
        raise ValueError(f"Unknown model: {model_name}")

    model.load_state_dict(torch.load(checkpoint_path, map_location=device))
    model.to(device)
    model.eval()
    return model

# ------------------------------
# 3. 메인 테스트 함수
# ------------------------------
def test(args):
    device = torch.device(f"cuda:{args.gpu}" if torch.cuda.is_available() else "cpu")
    os.makedirs(args.output_dir, exist_ok=True)
    
    test_dataset = MedicalTestDataset(img_dir=args.test_img, mask_dir=args.test_mask, size=args.size)
    test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False, num_workers=4)
    
    print(f"📊 Test Data: {len(test_dataset)}장")
    
    model = load_model(args.model, args.encoder, args.weights, device)
    
    # 전체 통계 변수
    tp_total, fp_total, fn_total, tn_total = 0, 0, 0, 0
    
    # CSV 저장을 위한 리스트
    results_list = []
    
    print("🚀 Testing & Saving Results...")
    with torch.no_grad():
        for image, mask, img_name in tqdm(test_loader):
            image = image.to(device)
            mask = mask.to(device)
            
            logits = model(image)
            pr_mask = torch.sigmoid(logits)
            pr_mask_binary = (pr_mask > 0.5).float()
            
            # --- 이미지별 Metric 계산 ---
            tp, fp, fn, tn = smp.metrics.get_stats(pr_mask_binary.long(), mask.long(), mode='binary', threshold=0.5)
            
            # 텐서 값을 스칼라로 변환
            tp_val = tp.item()
            fp_val = fp.item()
            fn_val = fn.item()
            tn_val = tn.item()
            
            # 개별 이미지 IoU, Dice 계산
            iou_img = tp_val / (tp_val + fp_val + fn_val + 1e-7)
            dice_img = 2 * tp_val / (2 * tp_val + fp_val + fn_val + 1e-7)
            acc_img = (tp_val + tn_val) / (tp_val + tn_val + fp_val + fn_val + 1e-7)
            
            # 결과 리스트에 추가
            results_list.append({
                "Image": img_name[0],
                "IoU": round(iou_img, 4),
                "Dice": round(dice_img, 4),
                "Accuracy": round(acc_img, 4)
            })
            
            # 전체 통계 누적
            tp_total += tp_val
            fp_total += fp_val
            fn_total += fn_val
            tn_total += tn_val
            
            # --- 시각화 저장 ---
            img_vis = image.squeeze().cpu().numpy().transpose(1, 2, 0)
            img_vis = img_vis * np.array([0.229, 0.224, 0.225]) + np.array([0.485, 0.456, 0.406])
            img_vis = (img_vis * 255).clip(0, 255).astype(np.uint8)
            
            gt_vis = mask.squeeze().cpu().numpy()
            gt_vis = (gt_vis * 255).astype(np.uint8)
            gt_vis = np.stack([gt_vis]*3, axis=-1)
            
            pr_vis = pr_mask_binary.squeeze().cpu().numpy()
            pr_vis = (pr_vis * 255).astype(np.uint8)
            pr_vis_rgb = np.stack([pr_vis]*3, axis=-1)
            
            combined = np.hstack([img_vis, gt_vis, pr_vis_rgb])
            save_path = os.path.join(args.output_dir, img_name[0])
            Image.fromarray(combined).save(save_path)

    # --- 최종 Metric 계산 (Micro Average) ---
    iou_total = tp_total / (tp_total + fp_total + fn_total + 1e-7)
    f1_total = 2 * tp_total / (2 * tp_total + fp_total + fn_total + 1e-7)
    acc_total = (tp_total + tn_total) / (tp_total + tn_total + fp_total + fn_total + 1e-7)
    
    # CSV 마지막 줄에 평균 추가
    results_list.append({
        "Image": "AVERAGE",
        "IoU": round(iou_total, 4),
        "Dice": round(f1_total, 4),
        "Accuracy": round(acc_total, 4)
    })
    
    # CSV 파일 저장
    csv_path = os.path.join(args.output_dir, "test_metrics.csv")
    df = pd.DataFrame(results_list)
    df.to_csv(csv_path, index=False)
    
    print("\n" + "="*30)
    print(f"🏆 Final Test Results ({args.model} - {args.encoder})")
    print(f"   IoU (Jaccard): {iou_total:.4f}")
    print(f"   Dice (F1)    : {f1_total:.4f}")
    print(f"   Pixel Acc    : {acc_total:.4f}")
    print("="*30)
    print(f"💾 Result Images saved to: {args.output_dir}")
    print(f"📄 Metrics CSV saved to: {csv_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--test_img", type=str, required=True, help="테스트 이미지 폴더")
    parser.add_argument("--test_mask", type=str, required=True, help="테스트 마스크 폴더")
    parser.add_argument("--output_dir", type=str, default="test_results", help="결과 이미지 저장 경로")
    
    parser.add_argument("--model", type=str, required=True, choices=['Unet', 'UnetPlusPlus', 'DeepLabV3Plus', 'MAnet', 'FPN'])
    parser.add_argument("--encoder", type=str, default="resnet34")
    parser.add_argument("--weights", type=str, required=True, help="학습된 .pth 파일 경로")
    
    parser.add_argument("--size", type=int, default=512)
    parser.add_argument("--gpu", type=int, default=0)
    
    args = parser.parse_args()
    test(args)