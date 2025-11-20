import os
import argparse
from pathlib import Path
from tqdm import tqdm

import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from torchvision.utils import save_image

from diffusers import AutoencoderKL, UNet2DModel, DDPMScheduler

# Stable Diffusion VAE의 표준 Scaling Factor
SD_SCALING_FACTOR = 0.18215

# ------------------------------
# 1. 데이터셋 정의 (RGB 변환)
# ------------------------------
class RGBMaskDataset(Dataset):
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
            transforms.Normalize([0.5], [0.5]) # [0, 1] -> [-1, 1]
        ])

    def __len__(self):
        return len(self.masks)

    def __getitem__(self, index):
        path = self.mask_dir / self.masks[index]
        try:
            # 중요: SD VAE는 3채널 입력을 받으므로 RGB로 변환
            img = Image.open(path).convert("RGB")
            return self.transform(img)
        except Exception as e:
            print(f"Error loading {path}: {e}")
            return torch.zeros(3, self.size, self.size)

# ------------------------------
# 2. 유틸리티 함수
# ------------------------------
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
    
    msg = "🏆 Best Model" if is_best else "💾 Checkpoint"
    print(f"{msg} 저장 완료: {save_path}")

def load_checkpoint(resume_path, unet, optimizer):
    print(f"🔄 체크포인트 로드 중: {resume_path}")
    unet_path = os.path.join(resume_path, "unet")
    try:
        loaded_unet = UNet2DModel.from_pretrained(unet_path)
        unet.load_state_dict(loaded_unet.state_dict())
    except Exception as e:
        print(f"❌ 모델 로드 실패: {e}")
        return 0, 0
    
    state_path = os.path.join(resume_path, "training_state.pt")
    if os.path.exists(state_path):
        state = torch.load(state_path, map_location="cpu")
        optimizer.load_state_dict(state["optimizer_state_dict"])
        return state["epoch"] + 1, state["global_step"]
    return 0, 0

# ------------------------------
# 3. 평가 (Validation)
# ------------------------------
@torch.no_grad()
def validate(unet, vae, val_loader, noise_scheduler, device):
    unet.eval()
    total_loss = 0.0
    num_batches = 0
    
    for batch in val_loader:
        clean_images = batch.to(device)
        bs = clean_images.shape[0]

        # VAE Encoding (RGB Images)
        latents = vae.encode(clean_images).latent_dist.sample() * SD_SCALING_FACTOR

        # Add Noise
        noise = torch.randn_like(latents)
        timesteps = torch.randint(0, noise_scheduler.config.num_train_timesteps, (bs,), device=device).long()
        noisy_latents = noise_scheduler.add_noise(latents, noise, timesteps)

        # Predict
        noise_pred = unet(noisy_latents, timesteps).sample
        loss = F.mse_loss(noise_pred, noise)
        
        total_loss += loss.item()
        num_batches += 1

    return total_loss / num_batches

# ------------------------------
# 4. 학습 메인 함수
# ------------------------------
def train_ldm(args):
    device = torch.device(f"cuda:{args.gpu}" if torch.cuda.is_available() else "cpu")
    os.makedirs(args.output_dir, exist_ok=True)
    sample_dir = os.path.join(args.output_dir, "samples")
    os.makedirs(sample_dir, exist_ok=True)

    # --- 1. Pretrained VAE 로드 (HuggingFace) ---
    print(f"❄️ Pretrained VAE 로드 중: {args.model_id} ...")
    vae = AutoencoderKL.from_pretrained(args.model_id, subfolder="vae").to(device)
    vae.requires_grad_(False)
    vae.eval()

    # --- 2. UNet 초기화 ---
    # SD VAE는 Latent Channel이 4입니다.
    unet = UNet2DModel(
        sample_size=args.resolution // 8,  # 512 / 8 = 64
        in_channels=4,                     # VAE Latent Channels
        out_channels=4,
        layers_per_block=2,
        block_out_channels=(128, 256, 512, 512),
        down_block_types=("DownBlock2D", "DownBlock2D", "AttnDownBlock2D", "DownBlock2D"),
        up_block_types=("UpBlock2D", "AttnUpBlock2D", "UpBlock2D", "UpBlock2D"),
    ).to(device)

    noise_scheduler = DDPMScheduler(num_train_timesteps=1000)
    optimizer = torch.optim.AdamW(unet.parameters(), lr=args.lr)

    # --- 3. 데이터 로더 ---
    train_dataset = RGBMaskDataset(args.train_dir, size=args.resolution)
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers, pin_memory=True)
    
    val_loader = None
    if args.val_dir:
        val_dataset = RGBMaskDataset(args.val_dir, size=args.resolution)
        val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers, pin_memory=True)
        print(f"✅ Validation Set: {len(val_dataset)}장")

    # --- 4. Resume ---
    start_epoch = 0
    global_step = 0
    if args.resume:
        start_epoch, global_step = load_checkpoint(args.resume, unet, optimizer)

    print(f"🚀 학습 시작: Epoch {start_epoch} ~ {args.epochs}")
    best_val_loss = float('inf')

    for epoch in range(start_epoch, args.epochs):
        unet.train()
        train_loss = 0.0
        
        progress_bar = tqdm(train_loader, desc=f"Epoch {epoch+1}")
        
        for step, batch in enumerate(progress_bar):
            clean_images = batch.to(device)
            bs = clean_images.shape[0]

            # VAE Encoding
            with torch.no_grad():
                latents = vae.encode(clean_images).latent_dist.sample() * SD_SCALING_FACTOR

            # Forward Diffusion
            noise = torch.randn_like(latents)
            timesteps = torch.randint(0, noise_scheduler.config.num_train_timesteps, (bs,), device=device).long()
            noisy_latents = noise_scheduler.add_noise(latents, noise, timesteps)

            # Prediction & Backward
            noise_pred = unet(noisy_latents, timesteps).sample
            loss = F.mse_loss(noise_pred, noise)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            train_loss += loss.item()
            global_step += 1
            progress_bar.set_postfix({"Loss": loss.item()})

        # --- Evaluation ---
        avg_train_loss = train_loss / len(train_loader)
        
        if val_loader:
            avg_val_loss = validate(unet, vae, val_loader, noise_scheduler, device)
            print(f"[Epoch {epoch+1}] Train: {avg_train_loss:.5f} | Val: {avg_val_loss:.5f}")
            
            # Best Model Save
            if avg_val_loss < best_val_loss:
                best_val_loss = avg_val_loss
                save_checkpoint(args.output_dir, epoch+1, unet, optimizer, global_step, is_best=True)
        else:
            print(f"[Epoch {epoch+1}] Train: {avg_train_loss:.5f}")

        # --- Periodic Save & Sampling ---
        if (epoch + 1) % args.save_interval == 0:
            save_checkpoint(args.output_dir, epoch+1, unet, optimizer, global_step, is_best=False)
            
            # Sampling Check
            unet.eval()
            with torch.no_grad():
                sample_noise = torch.randn(4, 4, 64, 64).to(device)
                for t in tqdm(noise_scheduler.timesteps, desc="Sampling", leave=False):
                    out = unet(sample_noise, t).sample
                    sample_noise = noise_scheduler.step(out, t, sample_noise).prev_sample
                
                # Decode
                images = vae.decode(sample_noise / SD_SCALING_FACTOR).sample
                images = (images / 2 + 0.5).clamp(0, 1)
                
                # 저장 (컬러로 나올 수 있으나 마스크 데이터면 흑백에 가깝게 나옴)
                save_path = os.path.join(sample_dir, f"sample_epoch_{epoch+1:04d}.png")
                save_image(images, save_path, nrow=2)

    print(f"🎉 학습 종료! Best Val Loss: {best_val_loss:.5f}")

if __name__ == "__main__":
    import argparse
    from PIL import Image # PIL import가 누락될 수 있어 안전하게 추가
    
    parser = argparse.ArgumentParser()
    parser.add_argument("--train_dir", type=str, required=True)
    parser.add_argument("--val_dir", type=str, default=None)
    parser.add_argument("--output_dir", type=str, default="ldm_sd_result")
    parser.add_argument("--resume", type=str, default=None)
    parser.add_argument("--model_id", type=str, default="CompVis/stable-diffusion-v1-4", help="HuggingFace Model ID")
    
    parser.add_argument("--resolution", type=int, default=512)
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--save_interval", type=int, default=10)
    parser.add_argument("--gpu", type=int, default=0)
    
    args = parser.parse_args()
    train_ldm(args)