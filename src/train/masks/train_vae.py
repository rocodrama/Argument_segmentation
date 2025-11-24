import os
import argparse
from pathlib import Path
import numpy as np
from PIL import Image

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
    이미지 배치(Batch) 간의 평균 PSNR 계산 (수정됨)
    """
    # 전체 평균(Scalar)을 구해서 차원 문제를 방지합니다.
    mse = torch.mean((img1 - img2) ** 2)
    
    # 0으로 나누기 방지 (아주 작은 값 더하기)
    if mse.item() <= 1e-10:
        return 100.0 
        
    psnr = 20 * torch.log10(1.0 / torch.sqrt(mse))
    return psnr.item()

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

    print(f"Start VAE Training")
    print(f" - Data Size: {len(dataset)}")
    print(f" - Epochs: {args.epochs}")

    # Best PSNR 기록용 변수 초기화
    best_psnr = 0.0

    for epoch in range(args.epochs):
        vae.train()
        epoch_loss = 0.0
        epoch_recon_loss = 0.0
        
        # Epoch 시작 로그
        print(f"\n[Epoch {epoch+1}/{args.epochs}] Start")
        
        for step, batch in enumerate(train_loader):
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

            # 중간 로그 출력 (매 50 step 마다)
            if (step + 1) % 50 == 0:
                print(f"  Step [{step+1}/{len(train_loader)}] L1 Loss: {recon_loss.item():.5f}")

        # --- Epoch 종료 후 평가 ---
        avg_loss = epoch_loss / len(train_loader)
        
        # PSNR 계산 (학습 데이터의 마지막 배치 사용)
        with torch.no_grad():
            orig_norm = (images / 2 + 0.5).clamp(0, 1)
            recon_norm = (reconstruction / 2 + 0.5).clamp(0, 1)
            current_psnr = calculate_psnr(orig_norm, recon_norm)

        # 결과 출력 (이모지 제거)
        print(f"Done Epoch {epoch+1} | Loss: {avg_loss:.5f} | PSNR: {current_psnr:.2f} dB (Best: {best_psnr:.2f} dB)")

        # Best Model 저장
        if current_psnr > best_psnr:
            best_psnr = current_psnr
            print(f"New Best PSNR ({best_psnr:.2f} dB) -> Saving model...")
            vae.save_pretrained(best_model_dir)
            
            comparison = torch.cat([orig_norm[:4], recon_norm[:4]], dim=0)
            save_image(comparison, os.path.join(image_log_dir, f"best_sample_psnr{best_psnr:.1f}.png"), nrow=4)

        # 주기적 저장
        if (epoch + 1) % args.save_interval == 0:
            save_path = os.path.join(ckpt_dir, f"vae_epoch_{epoch+1}")
            vae.save_pretrained(save_path)
            
            comparison = torch.cat([orig_norm[:4], recon_norm[:4]], dim=0)
            save_image(comparison, os.path.join(image_log_dir, f"epoch_{epoch+1:04d}.png"), nrow=4)

    print(f"Training Complete. Final Best PSNR: {best_psnr:.2f} dB")
    print(f"Best Model Path: {best_model_dir}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_dir", type=str, required=True, help="Mask image folder")
    parser.add_argument("--output_dir", type=str, default="vae_result", help="Output directory")
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