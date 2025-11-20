import os
import argparse
import torch
import numpy as np
from pathlib import Path
from PIL import Image
from tqdm import tqdm
from torchvision import transforms
import segmentation_models_pytorch as smp
from torch.utils.data import DataLoader, Dataset

# ------------------------------
# 1. 데이터셋 클래스 (학습과 동일한 정규화 필수)
# ------------------------------
class MedicalTestDataset(Dataset):
    def __init__(self, img_dir, mask_dir, size=512):
        self.img_dir = Path(img_dir)
        self.mask_dir = Path(mask_dir)
        self.size = size
        
        self.images = sorted([f.name for f in self.img_dir.iterdir() if f.suffix.lower() in ['.jpg', '.png', '.jpeg', '.tiff', '.bmp']])
        
        # 학습 때와 동일한 정규화 적용
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
        
        # 이미지 로드
        image = Image.open(img_path).convert("RGB")
        
        # 마스크 로드 (파일명 매칭 로직은 학습 코드와 동일하게 맞추세요)
        mask_name = img_name
        mask_path = self.mask_dir / mask_name
        
        if not mask_path.exists():
             # 확장자가 다를 경우 예외 처리
             for ext in ['.png', '.jpg', '.jpeg', '.tiff', '.bmp']:
                 test_path = self.mask_dir / (Path(img_name).stem + ext)
                 if test_path.exists():
                     mask_path = test_path
                     break
        
        if not mask_path.exists():
            # 마스크가 없으면 평가 불가 -> 0으로 채움 (또는 에러 처리)
            mask = Image.new("L", image.size, 0)
        else:
            mask = Image.open(mask_path).convert("L")

        return self.transform(image), self.mask_transform(mask), img_name

# ------------------------------
# 2. 모델 로드 함수
# ------------------------------
def load_model(model_name, encoder, checkpoint_path, device):
    print(f"🏗️ Loading Model: {model_name} (Backbone: {encoder})")
    
    # SMP 모델 생성
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

    # 가중치 로드
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
    
    # 데이터셋 로드
    test_dataset = MedicalTestDataset(img_dir=args.test_img, mask_dir=args.test_mask, size=args.size)
    test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False, num_workers=4) # 테스트는 배치 1 권장
    
    print(f"📊 Test Data: {len(test_dataset)}장")
    
    # 모델 로드
    model = load_model(args.model, args.encoder, args.weights, device)
    
    # Metric 통계 변수
    tp_total, fp_total, fn_total, tn_total = 0, 0, 0, 0
    
    print("🚀 Testing & Saving Results...")
    with torch.no_grad():
        for image, mask, img_name in tqdm(test_loader):
            image = image.to(device)
            mask = mask.to(device)
            
            # 추론
            logits = model(image)
            pr_mask = torch.sigmoid(logits) # 확률값 (0~1)
            pr_mask_binary = (pr_mask > 0.5).float() # 이진화 (0 or 1)
            
            # 통계 집계 (TP, FP, FN, TN)
            tp, fp, fn, tn = smp.metrics.get_stats(pr_mask_binary.long(), mask.long(), mode='binary', threshold=0.5)
            tp_total += tp.sum().item()
            fp_total += fp.sum().item()
            fn_total += fn.sum().item()
            tn_total += tn.sum().item()
            
            # --- 결과 이미지 저장 (시각화) ---
            # 1. 원본 이미지 복원 (Normalize 역변환)
            # mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
            img_vis = image.squeeze().cpu().numpy().transpose(1, 2, 0)
            img_vis = img_vis * np.array([0.229, 0.224, 0.225]) + np.array([0.485, 0.456, 0.406])
            img_vis = (img_vis * 255).clip(0, 255).astype(np.uint8)
            
            # 2. GT 마스크
            gt_vis = mask.squeeze().cpu().numpy()
            gt_vis = (gt_vis * 255).astype(np.uint8)
            gt_vis = np.stack([gt_vis]*3, axis=-1) # RGB 3채널로 변경 (병합 위해)
            
            # 3. 예측 마스크
            pr_vis = pr_mask_binary.squeeze().cpu().numpy()
            pr_vis = (pr_vis * 255).astype(np.uint8)
            # 예측 부분은 빨간색 틴트를 주거나 그냥 흑백으로 표시
            pr_vis_rgb = np.stack([pr_vis]*3, axis=-1)
            
            # 4. 나란히 붙이기 [원본 | 정답 | 예측]
            combined = np.hstack([img_vis, gt_vis, pr_vis_rgb])
            
            # 저장
            save_path = os.path.join(args.output_dir, img_name[0])
            Image.fromarray(combined).save(save_path)

    # --- 최종 Metric 계산 ---
    # IoU = TP / (TP + FP + FN)
    iou = tp_total / (tp_total + fp_total + fn_total + 1e-7)
    # F1 (Dice) = 2*TP / (2*TP + FP + FN)
    f1_score = 2 * tp_total / (2 * tp_total + fp_total + fn_total + 1e-7)
    # Accuracy
    accuracy = (tp_total + tn_total) / (tp_total + tn_total + fp_total + fn_total + 1e-7)
    
    print("\n" + "="*30)
    print(f"🏆 Final Test Results ({args.model} - {args.encoder})")
    print(f"   IoU (Jaccard): {iou:.4f}")
    print(f"   Dice (F1)    : {f1_score:.4f}")
    print(f"   Pixel Acc    : {accuracy:.4f}")
    print("="*30)
    print(f"💾 Result Images saved to: {args.output_dir}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    # 데이터 경로
    parser.add_argument("--test_img", type=str, required=True, help="테스트 이미지 폴더")
    parser.add_argument("--test_mask", type=str, required=True, help="테스트 마스크 폴더")
    parser.add_argument("--output_dir", type=str, default="test_results", help="결과 이미지 저장 경로")
    
    # 모델 설정 (학습때와 동일해야 함)
    parser.add_argument("--model", type=str, required=True, choices=['Unet', 'UnetPlusPlus', 'DeepLabV3Plus', 'MAnet', 'FPN'])
    parser.add_argument("--encoder", type=str, default="resnet34")
    parser.add_argument("--weights", type=str, required=True, help="학습된 .pth 파일 경로")
    
    parser.add_argument("--size", type=int, default=512)
    parser.add_argument("--gpu", type=int, default=0)
    
    args = parser.parse_args()
    test(args)