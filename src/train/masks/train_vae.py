import os
import argparse
from pathlib import Path
import numpy as np
from PIL import Image
from tqdm import tqdm

import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from torchvision.utils import save_image

from diffusers import AutoencoderKL

# ------------------------------
# 유틸리티: PSNR 계산 함수
# ------------------------------
def calculate_psnr(img1, img2):
    """
    이미지 배치(Batch) 간의 평균 PSNR 계산
    img1, img2: [B, C, H, W], Range: [0, 1]
    """
    mse = torch.mean((img1 - img2) ** 2, dim=[1, 2, 3])
    # 0으로 나누기 방지
    if mse.item() == 0:
        return 100.0 
    return 20 * torch.log10(1.0 / torch.sqrt(mse)).mean().item()

# ------------------------------
# 1. 데이터셋 정의
# ------------------------------
class MaskDataset(Dataset):
    def __init__(self, mask_dir, size=512):
        self.mask_dir = Path(mask_dir)
        self.size = size
        self.masks = sorted([
            f for f in os.listdir(mask_dir) 
            if f.lower().endswith(('.png', '.jpg', '.jpeg', '.tiff', '.bmp'))
        ])
        
        self.transform = transforms.Compose([
            transforms.Resize((size, size), interpolation=transforms.InterpolationMode.NEAREST),
            transforms.ToTensor(),
            transforms.Normalize([0.5], [0.5])
        ])

    def __len__(self):
        return len(self.masks)

    def __getitem__(self, index):
        path = self.mask_dir / self.masks[index]
        try:
            img = Image.open(path).convert("L")
            return self.transform(img)
        except Exception as e:
            print(f"Error loading {path}: {e}")
            return torch.zeros(1, self.size, self.size)

# ------------------------------
# 2. 학습 메인 함수
# ------------------------------
def train_vae(args):
    device = torch.device(f"cuda:{args.gpu}" if torch.cuda.is_available() else "cpu")
    
    # 디렉토리 설정
    os.makedirs(args.output_dir, exist_ok=True)
    image_log_dir = os.path.join(args.output_dir, 'samples')
    ckpt_dir = os.path.join(args.output_dir, 'checkpoints')
    best_model_dir = os.path.join(args.output_dir, 'best_vae') # Best 모델 저장 경로
    
    os.makedirs(image_log_dir, exist_ok=True)
    os.makedirs(ckpt_dir, exist_ok=True)

    # 모델 초기화
    vae = AutoencoderKL(
        in_channels=1,
        out_channels=1,
        down_block_types=["DownEncoderBlock2D"] * 4,
        up_block_types=["UpDecoderBlock2D"] * 4,
        block_out_channels=[128, 256, 512, 512],
        latent_channels=4,
        layers_per_block=2,
        act_fn="silu",
        norm_num_groups=32,
        sample_size=args.resolution
    ).to(device)

    optimizer = torch.optim.AdamW(vae.parameters(), lr=args.lr)

    dataset = MaskDataset(args.input_dir, size=args.resolution)
    train_loader = DataLoader(
        dataset, 
        batch_size=args.batch_size, 
        shuffle=True, 
        num_workers=args.num_workers,
        pin_memory=True
    )

    print(f"VAE 학습 시작")
    print(f" - 데이터: {len(dataset)}장")
    print(f" - Epochs: {args.epochs}")

    # Best PSNR 기록용 변수 초기화
    best_psnr = 0.0

    for epoch in range(args.epochs):
        vae.train()
        epoch_loss = 0.0
        epoch_recon_loss = 0.0
        
        progress_bar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{args.epochs}")
        
        for step, batch in enumerate(progress_bar):
            images = batch.to(device)

            # Forward
            posterior = vae.encode(images).latent_dist
            z = posterior.sample()
            reconstruction = vae.decode(z).sample

            # Loss
            recon_loss = F.l1_loss(reconstruction, images)
            kl_loss = posterior.kl().mean()
            loss = recon_loss + (args.kl_weight * kl_loss)

            # Backward
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            epoch_loss += loss.item()
            epoch_recon_loss += recon_loss.item()
            progress_bar.set_postfix({"L1": recon_loss.item()})

        # --- Epoch 종료 후 평가 ---
        avg_loss = epoch_loss / len(train_loader)
        avg_recon = epoch_recon_loss / len(train_loader)
        
        # PSNR 계산 (간이 평가: 학습 데이터의 마지막 배치 사용)
        # *엄밀한 평가를 위해선 별도의 Validation Set을 로더로 만들어야 하지만, 
        # VAE 특성상 학습 데이터 복원력만 봐도 충분히 판단 가능합니다.
        with torch.no_grad():
            orig_norm = (images / 2 + 0.5).clamp(0, 1)
            recon_norm = (reconstruction / 2 + 0.5).clamp(0, 1)
            current_psnr = calculate_psnr(orig_norm, recon_norm)

        print(f"[Epoch {epoch+1}] Loss: {avg_loss:.5f} | PSNR: {current_psnr:.2f} dB (Best: {best_psnr:.2f} dB)")

        # Best Model 저장
        if current_psnr > best_psnr:
            best_psnr = current_psnr
            print(f"🏆 최고 성능 갱신! ({best_psnr:.2f} dB) -> 모델 저장 중...")
            vae.save_pretrained(best_model_dir)
            
            # Best 순간의 이미지도 따로 저장해두면 좋습니다.
            comparison = torch.cat([orig_norm[:4], recon_norm[:4]], dim=0)
            save_image(comparison, os.path.join(image_log_dir, f"best_sample_psnr{best_psnr:.1f}.png"), nrow=4)

        # 주기적 저장 (Backup 용)
        if (epoch + 1) % args.save_interval == 0:
            save_path = os.path.join(ckpt_dir, f"vae_epoch_{epoch+1}")
            vae.save_pretrained(save_path)
            
            comparison = torch.cat([orig_norm[:4], recon_norm[:4]], dim=0)
            save_image(comparison, os.path.join(image_log_dir, f"epoch_{epoch+1:04d}.png"), nrow=4)

    print(f"학습 완료! 최종 Best PSNR: {best_psnr:.2f} dB")
    print(f"Best Model 저장 위치: {best_model_dir}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_dir", type=str, required=True, help="마스크 이미지 폴더")
    parser.add_argument("--output_dir", type=str, default="vae_result", help="결과 저장 경로")
    parser.add_argument("--resolution", type=int, default=512)
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--kl_weight", type=float, default=1e-6)
    parser.add_argument("--save_interval", type=int, default=5)
    parser.add_argument("--gpu", type=int, default=0)
    
    args = parser.parse_args()
    train_vae(args)