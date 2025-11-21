import os
import argparse
import json
from pathlib import Path
import numpy as np
from PIL import Image

import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from torchvision.utils import save_image

from diffusers import AutoencoderKL, UNet2DModel, DDPMScheduler

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
# 2. 유틸리티
# ------------------------------
def compute_snr_scale(vae, dataloader, device, output_dir, num_batches=10):
    """
    Latent Scaling Factor를 계산하고 파일로 저장합니다.
    """
    print("Calculating Latent Scaling Factor...")
    vae.eval()
    latents = []
    with torch.no_grad():
        for i, batch in enumerate(dataloader):
            if i >= num_batches: break
            imgs = batch.to(device)
            posterior = vae.encode(imgs).latent_dist
            latents.append(posterior.sample())
    
    latents = torch.cat(latents, dim=0)
    std = latents.std().item()
    scale_factor = 1.0 / std
    
    print(f"   -> Measured std: {std:.4f}, Recommended Scaling Factor: {scale_factor:.4f}")
    
    # Scaling Factor 파일 저장 (txt 파일)
    save_path = os.path.join(output_dir, "scaling_factor.txt")
    with open(save_path, "w") as f:
        f.write(str(scale_factor))
    print(f"Saved Scaling Factor to: {save_path}")
    
    return scale_factor

def save_checkpoint(output_dir, epoch, unet, optimizer, global_step, is_best=False):
    if is_best:
        save_path = os.path.join(output_dir, "best_unet")
    else:
        save_path = os.path.join(output_dir, f"checkpoint-{epoch}", "unet")
    
    os.makedirs(save_path, exist_ok=True)
    unet.save_pretrained(save_path)
    
    if not is_best:
        state_path = os.path.join(output_dir, f"checkpoint-{epoch}", "training_state.pt")
        torch.save({
            "epoch": epoch,
            "global_step": global_step,
            "optimizer_state_dict": optimizer.state_dict(),
        }, state_path)
    
    msg = "Best Model" if is_best else "Checkpoint"
    print(f"Saved {msg}: {save_path}")

def load_checkpoint(resume_path, unet, optimizer):
    print(f"Loading checkpoint: {resume_path}")
    unet_path = os.path.join(resume_path, "unet")
    loaded_unet = UNet2DModel.from_pretrained(unet_path)
    unet.load_state_dict(loaded_unet.state_dict())
    
    state_path = os.path.join(resume_path, "training_state.pt")
    if os.path.exists(state_path):
        state = torch.load(state_path, map_location="cpu")
        optimizer.load_state_dict(state["optimizer_state_dict"])
        return state["epoch"] + 1, state["global_step"]
    return 0, 0

# ------------------------------
# 3. 평가 함수 (Validation Loop)
# ------------------------------
@torch.no_grad()
def validate(unet, vae, val_loader, noise_scheduler, scaling_factor, device):
    unet.eval()
    total_val_loss = 0.0
    num_batches = 0
    
    for batch in val_loader:
        clean_images = batch.to(device)
        bs = clean_images.shape[0]

        with torch.cuda.amp.autocast():
            # VAE Encoding
            posterior = vae.encode(clean_images).latent_dist
            latents = posterior.sample() * scaling_factor

            # Add Noise
            noise = torch.randn_like(latents)
            timesteps = torch.randint(0, noise_scheduler.config.num_train_timesteps, (bs,), device=device).long()
            noisy_latents = noise_scheduler.add_noise(latents, noise, timesteps)

            # Predict
            noise_pred = unet(noisy_latents, timesteps).sample

            # Validation Loss
            loss = F.mse_loss(noise_pred, noise)
            
        total_val_loss += loss.item()
        num_batches += 1

    return total_val_loss / num_batches

# ------------------------------
# 4. 학습 메인 함수
# ------------------------------
def train_ldm(args):
    device = torch.device(f"cuda:{args.gpu}" if torch.cuda.is_available() else "cpu")
    os.makedirs(args.output_dir, exist_ok=True)
    sample_dir = os.path.join(args.output_dir, "samples")
    os.makedirs(sample_dir, exist_ok=True)

    # --- VAE 로드 ---
    print("Loading VAE...")
    vae = AutoencoderKL.from_pretrained(args.vae_path).to(device)
    vae.requires_grad_(False)
    vae.eval()

    # --- UNet 초기화 ---
    latent_channels = vae.config.latent_channels
    unet = UNet2DModel(
        sample_size=args.resolution // 8,
        in_channels=latent_channels,
        out_channels=latent_channels,
        layers_per_block=2,
        block_out_channels=(128, 256, 512, 512),
        down_block_types=("DownBlock2D", "DownBlock2D", "AttnDownBlock2D", "DownBlock2D"),
        up_block_types=("UpBlock2D", "AttnUpBlock2D", "UpBlock2D", "UpBlock2D"),
    ).to(device)

    noise_scheduler = DDPMScheduler(num_train_timesteps=1000)
    optimizer = torch.optim.AdamW(unet.parameters(), lr=args.lr)

    # GradScaler 초기화 (Mixed Precision Training용)
    scaler = torch.cuda.amp.GradScaler()

    # --- 데이터 로더 ---
    train_dataset = MaskDataset(args.train_dir, size=args.resolution)
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers, pin_memory=True)
    
    val_loader = None
    if args.val_dir:
        val_dataset = MaskDataset(args.val_dir, size=args.resolution)
        val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers, pin_memory=True)
        print(f"Validation Set Loaded: {len(val_dataset)} images")

    # --- Scaling Factor 계산 및 저장 ---
    scale_file = os.path.join(args.output_dir, "scaling_factor.txt")
    if args.resume and os.path.exists(scale_file):
        with open(scale_file, "r") as f:
            scaling_factor = float(f.read().strip())
        print(f"Loaded Scaling Factor: {scaling_factor:.4f}")
    else:
        scaling_factor = compute_snr_scale(vae, train_loader, device, args.output_dir)

    # --- Resume ---
    start_epoch = 0
    global_step = 0
    if args.resume:
        start_epoch, global_step = load_checkpoint(args.resume, unet, optimizer)

    print(f"Start LDM Training: Epoch {start_epoch} ~ {args.epochs}")
    
    best_val_loss = float('inf')

    for epoch in range(start_epoch, args.epochs):
        unet.train()
        train_loss = 0.0
        
        print(f"\n[Epoch {epoch+1}/{args.epochs}] Start")
        
        for step, batch in enumerate(train_loader):
            clean_images = batch.to(device)
            bs = clean_images.shape[0]

            optimizer.zero_grad()

            # Autocast Context Manager 적용
            with torch.cuda.amp.autocast():
                with torch.no_grad():
                    posterior = vae.encode(clean_images).latent_dist
                    latents = posterior.sample() * scaling_factor

                noise = torch.randn_like(latents)
                timesteps = torch.randint(0, noise_scheduler.config.num_train_timesteps, (bs,), device=device).long()
                noisy_latents = noise_scheduler.add_noise(latents, noise, timesteps)

                # Model Prediction
                noise_pred = unet(noisy_latents, timesteps).sample
                loss = F.mse_loss(noise_pred, noise)

            # Scaler를 이용한 Backward & Step
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

            train_loss += loss.item()
            global_step += 1
            
            # 로그 출력 (매 50 step 마다)
            if (step + 1) % 50 == 0:
                print(f"  Step [{step+1}/{len(train_loader)}] Loss: {loss.item():.5f}")

        avg_train_loss = train_loss / len(train_loader)
        
        # --- Validation 수행 ---
        if val_loader:
            avg_val_loss = validate(unet, vae, val_loader, noise_scheduler, scaling_factor, device)
            print(f"Done Epoch {epoch+1} | Train Loss: {avg_train_loss:.5f} | Val Loss: {avg_val_loss:.5f}")
            
            if avg_val_loss < best_val_loss:
                best_val_loss = avg_val_loss
                save_checkpoint(args.output_dir, epoch+1, unet, optimizer, global_step, is_best=True)
        else:
            print(f"Done Epoch {epoch+1} | Train Loss: {avg_train_loss:.5f}")

        # --- 주기적 저장 및 샘플링 ---
        if (epoch + 1) % args.save_interval == 0:
            save_checkpoint(args.output_dir, epoch+1, unet, optimizer, global_step, is_best=False)
            
            print("  Generating sample...")
            unet.eval()
            # Sampling 시에도 autocast 적용
            with torch.cuda.amp.autocast():
                with torch.no_grad():
                    sample_noise = torch.randn(4, latent_channels, 64, 64).to(device)
                    # Sampling loop (tqdm 제거)
                    for t in noise_scheduler.timesteps:
                        model_output = unet(sample_noise, t).sample
                        sample_noise = noise_scheduler.step(model_output, t, sample_noise).prev_sample
                    
                    images_decoded = vae.decode(sample_noise / scaling_factor).sample
                    images_decoded = (images_decoded / 2 + 0.5).clamp(0, 1)
                    
            save_path = os.path.join(sample_dir, f"sample_epoch_{epoch+1:04d}.png")
            save_image(images_decoded.float(), save_path, nrow=2)
            print(f"  Sample saved: {save_path}")

    print(f"Training Complete. Best Val Loss: {best_val_loss:.5f}")
    print(f"Inference Scaling Factor: {scaling_factor:.4f} (File saved: {scale_file})")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--train_dir", type=str, required=True, help="Train mask folder")
    parser.add_argument("--val_dir", type=str, default=None, help="Val mask folder (optional)")
    parser.add_argument("--vae_path", type=str, required=True, help="Pretrained VAE folder")
    parser.add_argument("--output_dir", type=str, default="ldm_result")
    parser.add_argument("--resume", type=str, default=None)
    
    parser.add_argument("--resolution", type=int, default=512)
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--save_interval", type=int, default=10)
    parser.add_argument("--gpu", type=int, default=0)
    
    args = parser.parse_args()
    train_ldm(args)