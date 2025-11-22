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

# SD VAE 표준 Scaling Factor
SD_SCALING_FACTOR = 0.18215

# ------------------------------
# 1. Dataset (Paired: Mask + Image)
# ------------------------------
class PairedDataset(Dataset):
    def __init__(self, root_dir, split='train', size=512):
        self.root_dir = Path(root_dir) / split
        self.size = size
        
        self.mask_dir = self.root_dir / 'masks'
        self.image_dir = self.root_dir / 'images'
        
        if not self.mask_dir.exists() or not self.image_dir.exists():
             # split 폴더가 없을 경우 대비
             raise FileNotFoundError(f"Directory not found: {self.mask_dir} or {self.image_dir}")

        # 파일명 매칭
        self.mask_files = sorted([f for f in self.mask_dir.iterdir() if f.suffix in ['.jpg', '.png', '.jpeg', '.tiff', '.bmp']])
        self.image_files = sorted([f for f in self.image_dir.iterdir() if f.suffix in ['.jpg', '.png', '.jpeg', '.tiff', '.bmp']])

        assert len(self.mask_files) == len(self.image_files), f"[{split}] Mask and Image counts mismatch."

        self.transform = transforms.Compose([
            transforms.Resize((size, size), interpolation=transforms.InterpolationMode.NEAREST),
            transforms.ToTensor(),
            transforms.Normalize([0.5], [0.5])
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
# 2. 학습 메인 함수
# ------------------------------
def train(args):
    device = torch.device(f"cuda:{args.gpu}" if torch.cuda.is_available() else "cpu")
    os.makedirs(args.output_dir, exist_ok=True)
    sample_dir = os.path.join(args.output_dir, "samples")
    os.makedirs(sample_dir, exist_ok=True)

    # --- 1. 두 개의 Pretrained VAE 로드 ---
    print("Loading VAE models...")
    
    # (1) Image VAE (Target)
    if args.image_vae_path:
        print(f" -> Loading Image VAE from: {args.image_vae_path}")
        vae_image = AutoencoderKL.from_pretrained(args.image_vae_path).to(device)
    else:
        raise ValueError("Please provide --image_vae_path")

    # (2) Mask VAE (Condition)
    if args.mask_vae_path:
        print(f" -> Loading Mask VAE from: {args.mask_vae_path}")
        vae_mask = AutoencoderKL.from_pretrained(args.mask_vae_path).to(device)
    else:
        raise ValueError("Please provide --mask_vae_path")

    vae_image.requires_grad_(False)
    vae_image.eval()
    vae_mask.requires_grad_(False)
    vae_mask.eval()

    # --- 2. UNet 초기화 ---
    unet = UNet2DModel(
        sample_size=args.resolution // 8,
        in_channels=8, # 4 (Noisy Latent) + 4 (Mask Latent)
        out_channels=4,
        layers_per_block=2,
        block_out_channels=(128, 256, 512, 512),
        down_block_types=("DownBlock2D", "DownBlock2D", "AttnDownBlock2D", "DownBlock2D"),
        up_block_types=("UpBlock2D", "AttnUpBlock2D", "UpBlock2D", "UpBlock2D"),
    ).to(device)

    noise_scheduler = DDPMScheduler(num_train_timesteps=1000)
    optimizer = torch.optim.AdamW(unet.parameters(), lr=args.lr)
    scaler = torch.amp.GradScaler('cuda')

    # --- 3. 데이터 로더 (Train & Val) ---
    print("Preparing Datasets...")
    
    # Train Dataset
    train_dataset = PairedDataset(args.data_dir, split='train', size=args.resolution)
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers)
    
    # Val Dataset (추가됨)
    try:
        val_dataset = PairedDataset(args.data_dir, split='val', size=args.resolution)
        val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers)
        print(f"   - Train Size: {len(train_dataset)}")
        print(f"   - Val Size:   {len(val_dataset)}")
    except Exception as e:
        print(f"Warning: Could not load validation set. ({e})")
        val_loader = None

    print(f"Start Training for {args.epochs} epochs")

    # --- 4. 학습 루프 ---
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
                # Encoding
                with torch.no_grad():
                    latents = vae_image.encode(images).latent_dist.sample() * SD_SCALING_FACTOR
                    mask_latents = vae_mask.encode(masks).latent_dist.sample() * SD_SCALING_FACTOR

                # Add Noise
                noise = torch.randn_like(latents)
                timesteps = torch.randint(0, noise_scheduler.config.num_train_timesteps, (bs,), device=device).long()
                noisy_latents = noise_scheduler.add_noise(latents, noise, timesteps)

                # Concatenate & Predict
                unet_input = torch.cat([noisy_latents, mask_latents], dim=1)
                noise_pred = unet(unet_input, timesteps).sample
                
                loss = F.mse_loss(noise_pred, noise)
                loss_accum = loss / args.gradient_accumulation_steps

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
            print(f"  Saving checkpoint and validation sample...")
            save_path = os.path.join(args.output_dir, f"unet_epoch_{epoch+1}")
            unet.save_pretrained(save_path)
            
            # Validation Sampling
            unet.eval()
            # 검증 데이터셋에서 하나의 샘플을 가져옴 (없으면 학습 데이터 사용)
            sample_batch = next(iter(val_loader)) if val_loader else next(iter(train_loader))
            
            sample_mask = sample_batch['mask'][:1].to(device) # [1, 3, H, W]
            sample_gt = sample_batch['image'][:1].to(device)  # GT 비교용

            with torch.amp.autocast('cuda'):
                with torch.no_grad():
                    # Mask Encoding
                    sample_mask_latent = vae_mask.encode(sample_mask).latent_dist.sample() * SD_SCALING_FACTOR
                    
                    # Random Noise
                    latents = torch.randn(1, 4, 64, 64).to(device)
                    
                    # Denoising
                    for t in noise_scheduler.timesteps:
                        input_latents = torch.cat([latents, sample_mask_latent], dim=1)
                        model_output = unet(input_latents, t).sample
                        latents = noise_scheduler.step(model_output, t, latents).prev_sample

                    # Decoding
                    decoded_img = vae_image.decode(latents / SD_SCALING_FACTOR).sample
                    
                    # 시각화: [Input Mask] | [Generated] | [Ground Truth]
                    vis = torch.cat([sample_mask, decoded_img, sample_gt], dim=3)
                    vis = (vis / 2 + 0.5).clamp(0, 1).float() 
                    save_image(vis, os.path.join(sample_dir, f"val_epoch_{epoch+1}.png"))

    print("Training Complete.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", type=str, required=True, help="Dataset folder (must contain 'train' and 'val' subfolders)")
    parser.add_argument("--output_dir", type=str, default="ldm_dual_vae_result")
    
    parser.add_argument("--mask_vae_path", type=str, required=True, help="Path to trained Mask VAE")
    parser.add_argument("--image_vae_path", type=str, required=True, help="Path to trained Image VAE")
    
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
