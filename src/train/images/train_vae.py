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
    mse = torch.mean((img1 - img2) ** 2) # dim 인자 제거
    if mse.item() == 0:
        return 100.0 
    return 20 * torch.log10(1.0 / torch.sqrt(mse)).item()

# ------------------------------
# 1. 데이터셋 정의
# ------------------------------
class ImageDataset(Dataset):
    def __init__(self, img_dir, size=512):
        self.img_dir = Path(img_dir)
        self.size = size
        
        # 지원 확장자 목록
        extensions = {'.png', '.jpg', '.jpeg', '.tiff', '.bmp'}
        
        # 하위 폴더를 포함하지 않고 해당 폴더만 검색 (기존 로직 유지)
        # 만약 하위 폴더까지 찾으려면 os.walk 또는 Path.rglob 사용 필요
        self.images = sorted([
            f for f in os.listdir(img_dir) 
            if os.path.splitext(f)[-1].lower() in extensions
        ])
        
        self.transform = transforms.Compose([
            transforms.Resize((size, size), interpolation=transforms.InterpolationMode.NEAREST),
            transforms.ToTensor(),
            transforms.Normalize([0.5], [0.5]) # [-1, 1] 범위로 정규화
        ])

    def __len__(self):
        return len(self.images)

    def __getitem__(self, index):
        path = self.img_dir / self.images[index]
        try:
            # VAE 모델 구조에 따라 채널 수 맞춤 (기본 3채널)
            img = Image.open(path).convert("RGB")
            return self.transform(img)
        except Exception as e:
            print(f"Error loading {path}: {e}")
            return torch.zeros(3, self.size, self.size)

# ------------------------------
# 2. 학습 메인 함수
# ------------------------------
def train_vae(args):
    device = torch.device(f"cuda:{args.gpu}" if torch.cuda.is_available() else "cpu")
    
    # 디렉토리 설정
    os.makedirs(args.output_dir, exist_ok=True)
    image_log_dir = os.path.join(args.output_dir, 'samples')
    best_model_dir = os.path.join(args.output_dir, 'best_vae') 
    
    os.makedirs(image_log_dir, exist_ok=True)
    os.makedirs(best_model_dir, exist_ok=True)

    # --- 모델 초기화 또는 불러오기 ---
    if args.resume:
        if not os.path.exists(args.resume):
            raise FileNotFoundError(f"Resume path not found: {args.resume}")
        
        print(f"Resuming training from: {args.resume}")
        # 저장된 설정과 가중치를 로드 (AutoencoderKL 구조 자동 인식)
        vae = AutoencoderKL.from_pretrained(args.resume)
    else:
        print("Initializing new VAE model...")
        # Stable Diffusion V1.4/1.5와 동일한 구조 (Latent Channel = 4)
        vae = AutoencoderKL(
            in_channels=3,
            out_channels=3,
            down_block_types=["DownEncoderBlock2D"] * 4,
            up_block_types=["UpDecoderBlock2D"] * 4,
            block_out_channels=[128, 256, 512, 512],
            latent_channels=4,
            layers_per_block=2,
            act_fn="silu",
            norm_num_groups=32,
            sample_size=args.resolution
        )
    
    vae = vae.to(device)

    optimizer = torch.optim.AdamW(vae.parameters(), lr=args.lr)
    
    # [AMP] GradScaler 초기화 (최신 버전 대응)
    # torch.cuda.amp.GradScaler() -> torch.amp.GradScaler('cuda')
    scaler = torch.amp.GradScaler('cuda')

    # --- 데이터 로더 ---
    dataset = ImageDataset(args.input_dir, size=args.resolution)
    if len(dataset) == 0:
        raise ValueError(f"No images found in {args.input_dir}")

    train_loader = DataLoader(
        dataset, 
        batch_size=args.batch_size, 
        shuffle=True, 
        num_workers=args.num_workers,
        pin_memory=True
    )

    print(f"Start VAE Training")
    print(f" - Data Size: {len(dataset)}")
    print(f" - Batch Size: {args.batch_size} (Accum: {args.gradient_accumulation_steps})")

    best_psnr = 0.0
    global_step = 0

    for epoch in range(args.epochs):
        vae.train()
        epoch_loss = 0.0
        epoch_recon_loss = 0.0
        
        print(f"\n[Epoch {epoch+1}/{args.epochs}] Start")
        
        optimizer.zero_grad()

        for step, batch in enumerate(train_loader):
            images = batch.to(device)

            # [AMP] Autocast 적용 (device_type 명시)
            with torch.amp.autocast('cuda'):
                # Forward Pass
                posterior = vae.encode(images).latent_dist
                z = posterior.sample()
                reconstruction = vae.decode(z).sample

                # Loss 계산
                recon_loss = F.mse_loss(reconstruction, images, reduction='mean')
                kl_loss = posterior.kl().mean()
                
                loss = recon_loss + (args.kl_weight * kl_loss)
                
                # [Accumulation] Loss 나누기
                loss = loss / args.gradient_accumulation_steps

            # [AMP] Backward
            scaler.scale(loss).backward()

            # [Accumulation] Step 수행
            if (step + 1) % args.gradient_accumulation_steps == 0:
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad()
                global_step += 1

            # 로깅 (원래 스케일로 복원)
            current_loss = loss.item() * args.gradient_accumulation_steps
            epoch_loss += current_loss
            epoch_recon_loss += recon_loss.item()
            
            # 중간 로그 출력 (매 50 step 마다)
            if (step + 1) % 50 == 0:
                print(f"  Step [{step+1}/{len(train_loader)}] Loss: {current_loss:.5f}")

        # --- Epoch 종료 후 평가 ---
        avg_loss = epoch_loss / len(train_loader)
        
        # 간이 평가 (마지막 배치의 PSNR 확인)
        with torch.no_grad():
            # 시각화를 위해 [-1, 1] -> [0, 1]
            orig_norm = (images / 2 + 0.5).clamp(0, 1)
            recon_norm = (reconstruction / 2 + 0.5).clamp(0, 1)
            current_psnr = calculate_psnr(orig_norm, recon_norm)

        print(f"Done Epoch {epoch+1} | Total Loss: {avg_loss:.5f} | PSNR: {current_psnr:.2f} dB (Best: {best_psnr:.2f} dB)")

        # Best Model 저장 (PSNR 기준)
        if current_psnr > best_psnr:
            best_psnr = current_psnr
            print(f"New Best PSNR ({best_psnr:.2f} dB) -> Saving model...")
            vae.save_pretrained(best_model_dir)
            
            # 비교 이미지 저장
            comparison = torch.cat([orig_norm[:4], recon_norm[:4]], dim=0)
            save_image(comparison, os.path.join(image_log_dir, f"best_sample_psnr{best_psnr:.1f}.png"), nrow=4)

        # [수정됨] 주기적 저장 (이미지 샘플 + 모델 체크포인트)
        if (epoch + 1) % args.save_interval == 0:
            # 1. 비교 이미지 저장
            comparison = torch.cat([orig_norm[:4], recon_norm[:4]], dim=0)
            save_image(comparison, os.path.join(image_log_dir, f"epoch_{epoch+1:04d}.png"), nrow=4)
            
            # 2. 체크포인트 폴더 저장
            checkpoint_path = os.path.join(args.output_dir, f"checkpoint_epoch_{epoch+1}")
            vae.save_pretrained(checkpoint_path)
            print(f"Checkpoint saved: {checkpoint_path}")

    print(f"Training Complete. Final Best PSNR: {best_psnr:.2f} dB")
    print(f"Saved Path: {best_model_dir}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_dir", type=str, required=True, help="Input image folder")
    parser.add_argument("--output_dir", type=str, default="vae_result", help="Output directory")
    
    # 추가된 Resume 옵션
    parser.add_argument("--resume", type=str, default=None, help="Path to the checkpoint folder to resume from")
    
    # 학습 설정
    parser.add_argument("--resolution", type=int, default=512)
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--gradient_accumulation_steps", type=int, default=4)
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--kl_weight", type=float, default=1e-6)
    
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--save_interval", type=int, default=5)
    parser.add_argument("--gpu", type=int, default=0)
    
    args = parser.parse_args()
    train_vae(args)
