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
            exts = ['.jpg', '.png', '.jpeg', '.tiff', '.bmp']
            self.mask_files = sorted([f for f in self.mask_dir.iterdir() if f.suffix.lower() in exts])
            self.image_files = sorted([f for f in self.image_dir.iterdir() if f.suffix.lower() in exts])

            assert len(self.mask_files) == len(self.image_files), f"[{split}] Mask and Image counts mismatch."

        # 이미지(RGB, 3ch)용 변환기
        self.image_transform = transforms.Compose([
            transforms.Resize((size, size), interpolation=transforms.InterpolationMode.NEAREST),
            transforms.ToTensor(),
            transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5]) 
        ])

        # 마스크(Grayscale, 1ch)용 변환기
        self.mask_transform = transforms.Compose([
            transforms.Resize((size, size), interpolation=transforms.InterpolationMode.NEAREST),
            transforms.ToTensor(),
            transforms.Normalize([0.5], [0.5]) 
        ])

    def __len__(self):
        return len(self.mask_files)

    def __getitem__(self, idx):
        mask_path = self.mask_files[idx]
        img_path = self.image_files[idx]

        # 마스크는 Grayscale("L")로 로드
        mask = Image.open(mask_path).convert("L")
        # 이미지는 RGB로 로드
        image = Image.open(img_path).convert("RGB")
        
        return {
            'mask': self.mask_transform(mask),   
            'image': self.image_transform(image) 
        }

# ------------------------------
# 3. 학습 메인 함수
# ------------------------------
def train(args):
    device = torch.device(f"cuda:{args.gpu}" if torch.cuda.is_available() else "cpu")
    
    os.makedirs(args.output_dir, exist_ok=True)
    # Best Sample 저장용 폴더
    best_sample_dir = os.path.join(args.output_dir, "best_samples")
    os.makedirs(best_sample_dir, exist_ok=True)

    # --- [Step 1] 두 개의 VAE 로드 ---
    print("Loading VAE models...")
    
    if args.image_vae_path:
        print(f" -> Loading Image VAE from: {args.image_vae_path}")
        vae_image = AutoencoderKL.from_pretrained(args.image_vae_path).to(device)
    else:
        raise ValueError("Please provide --image_vae_path")

    if args.mask_vae_path:
        print(f" -> Loading Mask VAE from: {args.mask_vae_path}")
        vae_mask = AutoencoderKL.from_pretrained(args.mask_vae_path).to(device)
    else:
        raise ValueError("Please provide --mask_vae_path")

    vae_image.requires_grad_(False)
    vae_image.eval()
    vae_mask.requires_grad_(False)
    vae_mask.eval()

    # --- [Step 2] UNet 초기화 (Resume 지원) ---
    if args.resume:
        print(f"Resuming training from checkpoint: {args.resume}")
        # 저장된 폴더에서 모델 구조와 가중치를 자동으로 로드
        unet = UNet2DModel.from_pretrained(args.resume).to(device)
    else:
        print("Initializing new UNet model...")
        unet = UNet2DModel(
            sample_size=args.resolution // 8,
            in_channels=8, 
            out_channels=4, 
            layers_per_block=2,
            block_out_channels=(128, 256, 512, 512),
            down_block_types=("DownBlock2D", "DownBlock2D", "AttnDownBlock2D", "DownBlock2D"),
            up_block_types=("UpBlock2D", "AttnUpBlock2D", "UpBlock2D", "UpBlock2D"),
        ).to(device)

    noise_scheduler = DDPMScheduler(num_train_timesteps=1000)
    optimizer = torch.optim.AdamW(unet.parameters(), lr=args.lr)
    scaler = torch.amp.GradScaler('cuda')

    # --- [Step 3] 데이터 로더 ---
    print("Preparing Datasets...")
    
    train_dataset = PairedDataset(args.data_dir, split='train', size=args.resolution)
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers, pin_memory=True)
    
    val_dataset = PairedDataset(args.data_dir, split='val', size=args.resolution)
    if len(val_dataset) > 0:
        val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers, pin_memory=True)
        print(f"   - Train Size: {len(train_dataset)}")
        print(f"   - Val Size:   {len(val_dataset)}")
    else:
        print("   - Warning: No validation data found.")
        val_loader = None

    print(f"Start Training for {args.epochs} epochs")

    # --- [Step 4] 학습 루프 ---
    global_step = 0
    best_val_loss = float('inf') # Best Loss 추적용

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
                with torch.no_grad():
                    latents = vae_image.encode(images).latent_dist.sample() * SD_SCALING_FACTOR
                    mask_latents = vae_mask.encode(masks).latent_dist.sample() * SD_SCALING_FACTOR

                noise = torch.randn_like(latents)
                timesteps = torch.randint(0, noise_scheduler.config.num_train_timesteps, (bs,), device=device).long()
                noisy_latents = noise_scheduler.add_noise(latents, noise, timesteps)

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

            # --- ★ Best Model Save & Sampling ★ ---
            if avg_val_loss < best_val_loss:
                print(f"  ★ New Best Val Loss! ({best_val_loss:.5f} -> {avg_val_loss:.5f})")
                best_val_loss = avg_val_loss
                
                # 1. Best Model 저장
                best_save_path = os.path.join(args.output_dir, "best_unet")
                unet.save_pretrained(best_save_path)
                print(f"     Saved Best Model to: {best_save_path}")
                
                # 2. Best 일 때만 샘플 이미지 생성
                print(f"     Generating Best Sample...")
                sample_batch = next(iter(val_loader))
                sample_mask = sample_batch['mask'][:1].to(device)
                sample_gt = sample_batch['image'][:1].to(device)

                with torch.amp.autocast('cuda'):
                    with torch.no_grad():
                        sample_mask_latent = vae_mask.encode(sample_mask).latent_dist.sample() * SD_SCALING_FACTOR
                        latents = torch.randn(1, 4, 64, 64).to(device)
                        
                        for t in noise_scheduler.timesteps:
                            input_latents = torch.cat([latents, sample_mask_latent], dim=1)
                            model_output = unet(input_latents, t).sample
                            latents = noise_scheduler.step(model_output, t, latents).prev_sample

                        decoded_img = vae_image.decode(latents / SD_SCALING_FACTOR).sample
                        
                        sample_mask_vis = sample_mask.repeat(1, 3, 1, 1)
                        current_psnr = calculate_psnr(sample_gt, decoded_img)
                        
                        vis = torch.cat([sample_mask_vis, decoded_img, sample_gt], dim=3)
                        vis = (vis / 2 + 0.5).clamp(0, 1).float() 
                        
                        save_name = f"best_loss{best_val_loss:.4f}_psnr{current_psnr:.1f}_ep{epoch+1}.png"
                        save_image(vis, os.path.join(best_sample_dir, save_name))
                        print(f"     Saved Best Sample: {save_name} (PSNR: {current_psnr:.2f} dB)")

        # --- 주기적 Checkpoint 백업 & 샘플링 (추가됨) ---
        if (epoch + 1) % args.save_interval == 0:
            # 1. 모델 저장
            checkpoint_path = os.path.join(args.output_dir, f"checkpoint_epoch_{epoch+1}")
            unet.save_pretrained(checkpoint_path)
            print(f"  [Backup] Generating sample for epoch {epoch+1}...")
            
            # 2. 샘플 이미지 생성
            unet.eval()
            # Validation 셋이 없으면 Train 셋에서 하나 가져오기
            loader = val_loader if val_loader else train_loader
            sample_batch = next(iter(loader))
            
            sample_mask = sample_batch['mask'][:1].to(device)
            sample_gt = sample_batch['image'][:1].to(device)

            with torch.amp.autocast('cuda'):
                with torch.no_grad():
                    # Mask Latent
                    sample_mask_latent = vae_mask.encode(sample_mask).latent_dist.sample() * SD_SCALING_FACTOR
                    
                    # Random Noise
                    latents = torch.randn(1, 4, 64, 64).to(device)
                    
                    # Diffusion Process
                    for t in noise_scheduler.timesteps:
                        input_latents = torch.cat([latents, sample_mask_latent], dim=1)
                        model_output = unet(input_latents, t).sample
                        latents = noise_scheduler.step(model_output, t, latents).prev_sample

                    # Decode
                    decoded_img = vae_image.decode(latents / SD_SCALING_FACTOR).sample
                    
                    # Visualize [Mask(3ch) | Generated | GT]
                    sample_mask_vis = sample_mask.repeat(1, 3, 1, 1)
                    vis = torch.cat([sample_mask_vis, decoded_img, sample_gt], dim=3)
                    vis = (vis / 2 + 0.5).clamp(0, 1).float() 
                    
                    # Save
                    sample_save_path = os.path.join(args.output_dir, f"sample_epoch_{epoch+1}.png")
                    save_image(vis, sample_save_path)
                    print(f"  [Backup] Checkpoint & Sample saved: {checkpoint_path}")
            
            unet.train() # 다시 학습 모드로 복귀

    print("Training Complete.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", type=str, required=True)
    parser.add_argument("--output_dir", type=str, default="ldm_dual_vae_result")
    parser.add_argument("--mask_vae_path", type=str, required=True)
    parser.add_argument("--image_vae_path", type=str, required=True)
    
    # Resume 옵션 추가
    parser.add_argument("--resume", type=str, default=None, help="Path to the UNet folder to resume from")
    
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