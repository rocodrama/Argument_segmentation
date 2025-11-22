import os
import argparse
from pathlib import Path
from PIL import Image
import numpy as np

import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from torchvision.utils import save_image

from diffusers import AutoencoderKL, UNet2DModel, DDPMScheduler

# Stable Diffusion 표준 Scaling Factor
SD_SCALING_FACTOR = 0.18215

# ------------------------------
# 1. 유틸리티: PSNR 계산
# ------------------------------
def calculate_psnr(img1, img2):
    """
    이미지 품질 평가 (Peak Signal-to-Noise Ratio)
    img1, img2: [B, C, H, W], Range: [-1, 1] or [0, 1]
    """
    # [-1, 1] 범위라면 [0, 1]로 변환
    if img1.min() < 0:
        img1 = (img1 / 2 + 0.5).clamp(0, 1)
    if img2.min() < 0:
        img2 = (img2 / 2 + 0.5).clamp(0, 1)
        
    mse = torch.mean((img1 - img2) ** 2)
    if mse == 0:
        return 100.0
    return 20 * torch.log10(1.0 / torch.sqrt(mse)).item()

# ------------------------------
# 2. Dataset (Paired: Mask + Image)
# ------------------------------
class PairedDataset(Dataset):
    def __init__(self, root_dir, split='train', size=512):
        self.root_dir = Path(root_dir) / split
        self.size = size
        
        self.mask_dir = self.root_dir / 'masks'
        self.image_dir = self.root_dir / 'images'
        
        if not self.mask_dir.exists() or not self.image_dir.exists():
             print(f"Warning: {split} directory not found at {self.root_dir}")
             self.mask_files = []
             self.image_files = []
        else:
            # 파일명 매칭 (확장자 필터링)
            exts = ['.jpg', '.png', '.jpeg', '.tiff', '.bmp']
            self.mask_files = sorted([f for f in self.mask_dir.iterdir() if f.suffix.lower() in exts])
            self.image_files = sorted([f for f in self.image_dir.iterdir() if f.suffix.lower() in exts])

            assert len(self.mask_files) == len(self.image_files), f"[{split}] Mask and Image counts mismatch."

        self.transform = transforms.Compose([
            transforms.Resize((size, size), interpolation=transforms.InterpolationMode.NEAREST),
            transforms.ToTensor(),
            transforms.Normalize([0.5], [0.5]) # [-1, 1]
        ])

    def __len__(self):
        return len(self.mask_files)

    def __getitem__(self, idx):
        mask_path = self.mask_files[idx]
        mask = Image.open(mask_path).convert("RGB")
        
        img_path = self.image_files[idx]
        image = Image.open(img_path).convert("RGB")
        
        return {
            'mask': self.transform(mask), 
            'image': self.transform(image)
        }

# ------------------------------
# 3. 학습 메인 함수
# ------------------------------
def train(args):
    device = torch.device(f"cuda:{args.gpu}" if torch.cuda.is_available() else "cpu")
    
    # 결과 저장 경로 생성
    os.makedirs(args.output_dir, exist_ok=True)
    sample_dir = os.path.join(args.output_dir, "samples")
    os.makedirs(sample_dir, exist_ok=True)

    # --- [Step 1] 두 개의 VAE 로드 ---
    print("Loading VAE models...")
    
    # (1) Image VAE (Target용: 정답 이미지를 인코딩/디코딩)
    if args.image_vae_path:
        print(f" -> Loading Image VAE from: {args.image_vae_path}")
        vae_image = AutoencoderKL.from_pretrained(args.image_vae_path).to(device)
    else:
        raise ValueError("Please provide --image_vae_path")

    # (2) Mask VAE (Condition용: 입력 마스크를 인코딩)
    if args.mask_vae_path:
        print(f" -> Loading Mask VAE from: {args.mask_vae_path}")
        vae_mask = AutoencoderKL.from_pretrained(args.mask_vae_path).to(device)
    else:
        raise ValueError("Please provide --mask_vae_path")

    # VAE는 학습하지 않고 고정(Freeze)
    vae_image.requires_grad_(False)
    vae_image.eval()
    vae_mask.requires_grad_(False)
    vae_mask.eval()

    # --- [Step 2] UNet 초기화 ---
    # 입력 채널 = Noisy Image Latent(4) + Mask Latent(4) = 8
    unet = UNet2DModel(
        sample_size=args.resolution // 8,
        in_channels=8, 
        out_channels=4, # Image Latent의 Noise 예측
        layers_per_block=2,
        block_out_channels=(128, 256, 512, 512),
        down_block_types=("DownBlock2D", "DownBlock2D", "AttnDownBlock2D", "DownBlock2D"),
        up_block_types=("UpBlock2D", "AttnUpBlock2D", "UpBlock2D", "UpBlock2D"),
    ).to(device)

    noise_scheduler = DDPMScheduler(num_train_timesteps=1000)
    optimizer = torch.optim.AdamW(unet.parameters(), lr=args.lr)
    scaler = torch.amp.GradScaler('cuda')

    # --- [Step 3] 데이터 로더 (Train & Val) ---
    print("Preparing Datasets...")
    
    # Train Loader
    train_dataset = PairedDataset(args.data_dir, split='train', size=args.resolution)
    if len(train_dataset) == 0:
        raise ValueError("No training data found.")
        
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers, pin_memory=True)
    
    # Val Loader
    val_dataset = PairedDataset(args.data_dir, split='val', size=args.resolution)
    if len(val_dataset) > 0:
        val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers, pin_memory=True)
        print(f"   - Train Size: {len(train_dataset)}")
        print(f"   - Val Size:   {len(val_dataset)}")
    else:
        print("   - Warning: No validation data found. Validation loop will be skipped.")
        val_loader = None

    print(f"Start Training for {args.epochs} epochs")

    # --- [Step 4] 학습 루프 ---
    global_step = 0
    
    for epoch in range(args.epochs):
        # ==========================
        #      Training Loop
        # ==========================
        unet.train()
        epoch_loss = 0.0
        
        print(f"\n[Epoch {epoch+1}/{args.epochs}]")
        optimizer.zero_grad()

        for step, batch in enumerate(train_loader):
            masks = batch['mask'].to(device)
            images = batch['image'].to(device)
            bs = images.shape[0]

            with torch.amp.autocast('cuda'):
                # 1. Encoding (Image -> Image VAE, Mask -> Mask VAE)
                with torch.no_grad():
                    latents = vae_image.encode(images).latent_dist.sample() * SD_SCALING_FACTOR
                    mask_latents = vae_mask.encode(masks).latent_dist.sample() * SD_SCALING_FACTOR

                # 2. Add Noise
                noise = torch.randn_like(latents)
                timesteps = torch.randint(0, noise_scheduler.config.num_train_timesteps, (bs,), device=device).long()
                noisy_latents = noise_scheduler.add_noise(latents, noise, timesteps)

                # 3. Concatenate [Noisy Image Latent, Mask Latent]
                unet_input = torch.cat([noisy_latents, mask_latents], dim=1)
                
                # 4. Predict Noise
                noise_pred = unet(unet_input, timesteps).sample
                
                # 5. Loss Calculation
                loss = F.mse_loss(noise_pred, noise)
                loss_accum = loss / args.gradient_accumulation_steps

            # Backward
            scaler.scale(loss_accum).backward()

            if (step + 1) % args.gradient_accumulation_steps == 0:
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad()
                global_step += 1
            
            current_loss = loss.item()
            epoch_loss += current_loss
            
            if (step + 1) % 50 == 0:
                print(f"  [Train] Step {step+1}/{len(train_loader)} Loss: {current_loss:.5f}")

        avg_train_loss = epoch_loss / len(train_loader)

        # ==========================
        #      Validation Loop
        # ==========================
        avg_val_loss = 0.0
        
        if val_loader:
            unet.eval()
            val_loss_sum = 0.0
            with torch.no_grad():
                for batch in val_loader:
                    masks = batch['mask'].to(device)
                    images = batch['image'].to(device)
                    bs = images.shape[0]

                    with torch.amp.autocast('cuda'):
                        latents = vae_image.encode(images).latent_dist.sample() * SD_SCALING_FACTOR
                        mask_latents = vae_mask.encode(masks).latent_dist.sample() * SD_SCALING_FACTOR
                        
                        noise = torch.randn_like(latents)
                        timesteps = torch.randint(0, noise_scheduler.config.num_train_timesteps, (bs,), device=device).long()
                        noisy_latents = noise_scheduler.add_noise(latents, noise, timesteps)
                        
                        unet_input = torch.cat([noisy_latents, mask_latents], dim=1)
                        noise_pred = unet(unet_input, timesteps).sample
                        
                        v_loss = F.mse_loss(noise_pred, noise)
                        val_loss_sum += v_loss.item()
            
            avg_val_loss = val_loss_sum / len(val_loader)
            print(f"  >> End Epoch {epoch+1} | Train Loss: {avg_train_loss:.5f} | Val Loss: {avg_val_loss:.5f}")
        else:
            print(f"  >> End Epoch {epoch+1} | Train Loss: {avg_train_loss:.5f}")


        # ==========================
        #    Save & Sampling
        # ==========================
        if (epoch + 1) % args.save_interval == 0:
            print(f"  Saving checkpoint and sampling...")
            save_path = os.path.join(args.output_dir, f"unet_epoch_{epoch+1}")
            unet.save_pretrained(save_path)
            
            # Sampling (Validation 데이터 중 하나 선택)
            unet.eval()
            sample_batch = next(iter(val_loader)) if val_loader else next(iter(train_loader))
            
            sample_mask = sample_batch['mask'][:1].to(device)
            sample_gt = sample_batch['image'][:1].to(device)

            with torch.amp.autocast('cuda'):
                with torch.no_grad():
                    # Mask Encoding (Mask VAE)
                    sample_mask_latent = vae_mask.encode(sample_mask).latent_dist.sample() * SD_SCALING_FACTOR
                    
                    # Random Start Noise
                    latents = torch.randn(1, 4, 64, 64).to(device)
                    
                    # Denoising Process
                    for t in noise_scheduler.timesteps:
                        input_latents = torch.cat([latents, sample_mask_latent], dim=1)
                        model_output = unet(input_latents, t).sample
                        latents = noise_scheduler.step(model_output, t, latents).prev_sample

                    # Decoding (Image VAE)
                    decoded_img = vae_image.decode(latents / SD_SCALING_FACTOR).sample
                    
                    # Calculate PSNR (Generated vs Ground Truth)
                    current_psnr = calculate_psnr(sample_gt, decoded_img)
                    print(f"  [Sample Quality] PSNR: {current_psnr:.2f} dB")

                    # Save Visualization: [Mask] | [Generated] | [Target]
                    vis = torch.cat([sample_mask, decoded_img, sample_gt], dim=3)
                    vis = (vis / 2 + 0.5).clamp(0, 1).float() 
                    save_image(vis, os.path.join(sample_dir, f"val_epoch_{epoch+1}_psnr{current_psnr:.1f}.png"))

    print("Training Complete.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    
    # 데이터셋 경로 (train/val 폴더 포함)
    parser.add_argument("--data_dir", type=str, required=True, help="Root dataset path containing 'train' and 'val' folders")
    parser.add_argument("--output_dir", type=str, default="ldm_dual_vae_result")
    
    # 두 개의 VAE 모델 경로 (필수)
    parser.add_argument("--mask_vae_path", type=str, required=True, help="Path to the pre-trained Mask VAE folder")
    parser.add_argument("--image_vae_path", type=str, required=True, help="Path to the pre-trained Image VAE folder")
    
    # 학습 하이퍼파라미터
    parser.add_argument("--resolution", type=int, default=512)
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--gradient_accumulation_steps", type=int, default=4)
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--epochs", type=int, default=200)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--save_interval", type=int, default=10)
    parser.add_argument("--gpu", type=int, default=0)
    
    args = parser.parse_args()
    train(args)
