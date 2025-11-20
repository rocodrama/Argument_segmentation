import os
import argparse
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
from tqdm import tqdm
import numpy as np

# 평가 라이브러리
import lpips
from torchmetrics.image import PeakSignalNoiseRatio, StructuralSimilarityIndexMeasure, FrechetInceptionDistance, LearnedPerceptualImagePatchSimilarity
from torchmetrics.classification import MulticlassJaccardIndex, MulticlassAccuracy

# ------------------------------
# 1. 사용자 정의 세그멘테이션 모델 (수정 필요)
# ------------------------------
class SegmentationEvaluator(nn.Module):
    def __init__(self, num_classes=2):
        super().__init__()
        # TODO: 여기에 실제로 학습된 세그멘테이션 모델을 로드하세요.
        # 예: self.model = torch.load("my_best_segmentation_model.pth")
        # 아래는 예시용 더미 모델 (ResNet50-DeepLabV3 등)
        from torchvision.models.segmentation import deeplabv3_resnet50
        self.model = deeplabv3_resnet50(num_classes=num_classes) 
        # self.model.load_state_dict(...) # 가중치 로드 필수!
        
    def forward(self, x):
        # 모델에 따라 출력 형식이 다를 수 있음 (dict['out'] 또는 tensor)
        out = self.model(x)
        if isinstance(out, dict):
            out = out['out']
        return out

# ------------------------------
# 2. 데이터셋 정의
# ------------------------------
class EvaluationDataset(Dataset):
    def __init__(self, real_dir, fake_dir, mask_dir=None, size=512):
        self.real_dir = real_dir
        self.fake_dir = fake_dir
        self.mask_dir = mask_dir
        self.size = size
        
        # 파일명 매칭 (확장자 무시)
        self.filenames = sorted([f for f in os.listdir(fake_dir) if f.lower().endswith(('.png', '.jpg', '.jpeg'))])
        
        self.transform = transforms.Compose([
            transforms.Resize((size, size)),
            transforms.ToTensor(),
            # Normalize는 모델에 따라 다름. 여기선 [0, 1] 유지하거나 필요시 추가
        ])
        
        # 마스크는 최근접 이웃(Nearest)으로 리사이즈해야 값이 안 깨짐
        self.mask_transform = transforms.Compose([
            transforms.Resize((size, size), interpolation=transforms.InterpolationMode.NEAREST),
            transforms.ToTensor() # 값은 그대로 0, 1 유지
        ])

    def __len__(self):
        return len(self.filenames)

    def __getitem__(self, idx):
        fname = self.filenames[idx]
        
        # 1. Fake Image (Generated)
        fake = Image.open(os.path.join(self.fake_dir, fname)).convert("RGB")
        fake = self.transform(fake)
        
        # 2. Real Image (Ground Truth)
        real_path = os.path.join(self.real_dir, fname)
        if os.path.exists(real_path):
            real = Image.open(real_path).convert("RGB")
            real = self.transform(real)
        else:
            real = torch.zeros_like(fake) # 없을 경우 대비

        # 3. Mask (Ground Truth for Segmentation)
        mask = torch.tensor([])
        if self.mask_dir:
            mask_path = os.path.join(self.mask_dir, fname)
            if os.path.exists(mask_path):
                mask_img = Image.open(mask_path).convert("L") # Grayscale
                mask = self.mask_transform(mask_img)
                # [0, 1] 범위의 텐서를 정수 클래스 인덱스(0, 1)로 변환
                mask = (mask * 255).long().squeeze(0)
                mask = (mask > 128).long() # 이진 마스크 가정
            
        return {"fake": fake, "real": real, "mask": mask}

# ------------------------------
# 3. 메인 평가 함수
# ------------------------------
def evaluate(args):
    device = torch.device(f"cuda:{args.gpu}" if torch.cuda.is_available() else "cpu")
    print(f"🚀 평가 시작 (Device: {device})")
    
    # --- Metrics 초기화 ---
    psnr = PeakSignalNoiseRatio().to(device)
    ssim = StructuralSimilarityIndexMeasure(data_range=1.0).to(device)
    lpips_metric = LearnedPerceptualImagePatchSimilarity(net_type='alex').to(device)
    fid = FrechetInceptionDistance(feature=2048).to(device)
    
    # Segmentation Metrics (Optional)
    use_seg = args.mask_dir is not None
    if use_seg:
        print("ℹ️ Segmentation Accuracy 평가를 포함합니다.")
        seg_model = SegmentationEvaluator(num_classes=2).to(device)
        seg_model.eval()
        iou = MulticlassJaccardIndex(num_classes=2).to(device)
        acc = MulticlassAccuracy(num_classes=2).to(device)
    
    # --- DataLoader ---
    dataset = EvaluationDataset(args.real_dir, args.fake_dir, args.mask_dir, args.size)
    dataloader = DataLoader(dataset, batch_size=args.batch_size, num_workers=4)
    
    # --- Evaluation Loop ---
    psnr_scores, ssim_scores, lpips_scores = [], [], []
    
    print("📊 데이터 처리 및 Metric 계산 중...")
    for batch in tqdm(dataloader):
        fake_imgs = batch['fake'].to(device) # [B, 3, H, W]
        real_imgs = batch['real'].to(device)
        
        # 1. Image Quality Metrics (Pairwise)
        # 값을 [0, 1] 범위로 클램핑 (안전을 위해)
        fake_imgs = torch.clamp(fake_imgs, 0, 1)
        real_imgs = torch.clamp(real_imgs, 0, 1)

        psnr_scores.append(psnr(fake_imgs, real_imgs).item())
        ssim_scores.append(ssim(fake_imgs, real_imgs).item())
        lpips_scores.append(lpips_metric(fake_imgs * 2 - 1, real_imgs * 2 - 1).item()) # LPIPS는 [-1, 1] 권장
        
        # 2. FID Update (Distribution)
        # FID는 uint8 [0, 255] 입력을 기대함
        fake_uint8 = (fake_imgs * 255).byte()
        real_uint8 = (real_imgs * 255).byte()
        
        fid.update(real_uint8, real=True)
        fid.update(fake_uint8, real=False)
        
        # 3. Segmentation Accuracy
        if use_seg:
            target_masks = batch['mask'].to(device) # [B, H, W]
            if target_masks.numel() > 0:
                with torch.no_grad():
                    # Fake Image -> Seg Model -> Pred Mask
                    seg_out = seg_model(fake_imgs) # [B, 2, H, W]
                    pred_masks = torch.argmax(seg_out, dim=1)
                    
                    iou.update(pred_masks, target_masks)
                    acc.update(pred_masks, target_masks)

    # --- 결과 집계 ---
    final_fid = fid.compute().item()
    avg_psnr = np.mean(psnr_scores)
    avg_ssim = np.mean(ssim_scores)
    avg_lpips = np.mean(lpips_scores)
    
    print("\n" + "="*40)
    print(f"🏆 최종 평가 결과 ({args.name})")
    print("="*40)
    print(f"1. [품질]   FID   (↓): {final_fid:.4f}  (낮을수록 좋음)")
    print(f"2. [지각]   LPIPS (↓): {avg_lpips:.4f}  (낮을수록 좋음)")
    print(f"3. [구조]   SSIM  (↑): {avg_ssim:.4f}   (1에 가까울수록 좋음)")
    print(f"4. [픽셀]   PSNR  (↑): {avg_psnr:.2f} dB")
    
    if use_seg:
        final_miou = iou.compute().item()
        final_acc = acc.compute().item()
        print("-" * 40)
        print(f"5. [의미]   mIoU  (↑): {final_miou:.4f}")
        print(f"6. [의미]   Acc   (↑): {final_acc:.4f}")
    print("="*40)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--name", type=str, default="Model Evaluation", help="평가 이름")
    parser.add_argument("--real_dir", type=str, required=True, help="Ground Truth 이미지 폴더")
    parser.add_argument("--fake_dir", type=str, required=True, help="생성된 이미지 폴더")
    parser.add_argument("--mask_dir", type=str, default=None, help="Segmentation 평가용 마스크 폴더 (선택)")
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--size", type=int, default=512)
    parser.add_argument("--gpu", type=int, default=0)
    
    args = parser.parse_args()
    evaluate(args)