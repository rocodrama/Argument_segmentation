import os
import argparse
from pathlib import Path
from PIL import Image

import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from torchvision.utils import save_image

from diffusers import AutoencoderKL, UNet2DModel, DDPMScheduler

# ------------------------------
# 1. 데이터셋 (Paired: Mask + Image)
# ------------------------------
class PairedDataset(Dataset):
    def __init__(self, root_dir, split='train', size=512):
        self.root_dir = Path(root_dir) / split
        self.size = size
        
        self.mask_dir = self.root_dir / 'masks'
        self.image_dir = self.root_dir / 'images'
        
        # 파일명 매칭
        self.mask_files = sorted([f for f in self.mask_dir.iterdir() if f.suffix.lower() in ['.jpg', '.png', '.jpeg', '.tiff', '.bmp']])
        self.image_files = sorted([f for f in self.image_dir.iterdir() if f.suffix.lower() in ['.jpg', '.png', '.jpeg', '.tiff', '.bmp']])

        assert len(self.mask_files) == len(self.image_files), "Mask and Image file counts do not match."

        self.transform = transforms.Compose([
            transforms.Resize((size, size), interpolation=transforms.InterpolationMode.NEAREST),
            transforms.ToTensor(),
            transforms.Normalize([0.5], [0.5])
        ])

    def __len__(self):
        return len(self.mask_files)

    def __getitem__(self, idx):
        # VAE는 3채널 입력을 기대하므로 RGB로 변환
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
def train_ldm(args):
    device = torch.device(f"cuda:{args.gpu}" if torch.cuda.is_available() else "cpu")
    os.makedirs(args.output_dir, exist_ok=True)
    sample_dir = os.path.join(args.output_dir, "samples")
    os.makedirs(sample_dir, exist_ok=True)

    # --- [핵심] 직접 학습시킨 VAE 로드 ---
    print(f"Loading local VAE: {args.vae_path}")
    try:
        # 로컬 폴더 경로를 직접 지정
        vae = AutoencoderKL.from_pretrained(args.vae_path).to(device)
    except OSError:
        # 만약 구조가 다르다면 subfolder 시도
        print("Failed to load directly, trying subfolder='vae'...")
        vae = AutoencoderKL.from_pretrained(args.vae_path, subfolder="vae").to(device)
        
    vae.requires_grad_(False)
    vae.eval()
    
    # Latent Channel 확인 (보통 4)
    latent_dim = vae.config.latent_channels 
    print(f"VAE loaded (Latent Dim: {latent_dim})")

    # --- UNet 초기화 ---
    # 입력: Noisy Latent(4) + Mask Latent(4) = 8채널
    unet = UNet2DModel(
        sample_size=args.resolution // 8,  # 512 / 8 = 64
        in_channels=latent_dim * 2,        # 4 + 4 = 8
        out_channels=latent_dim,           # 예측은 Noise(4)만 함
        layers_per_block=2,
        block_out_channels=(128, 256, 512, 512),
        down_block_types=("DownBlock2D", "DownBlock2D", "AttnDownBlock2D", "DownBlock2D"),
        up_block_types=("UpBlock2D", "AttnUpBlock2D", "UpBlock2D", "UpBlock2D"),
    ).to(device)

    noise_scheduler = DDPMScheduler(num_train_timesteps=1000)
    optimizer = torch.optim.AdamW(unet.parameters(), lr=args.lr)

    # [AMP] GradScaler
    scaler = torch.cuda.amp.GradScaler()

    # --- 데이터 로더 ---
    train_dataset = PairedDataset(args.data_dir, split='train', size=args.resolution)
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers)
    
    # Scaling Factor 설정
    SCALING_FACTOR = 0.18215 

    print(f"Start LDM Training: {len(train_dataset)} images")
    print(f" - Batch Size: {args.batch_size} (Accum: {args.gradient_accumulation_steps})")

    global_step = 0

    for epoch in range(args.epochs):
        unet.train()
        epoch_loss = 0.0
        
        print(f"\n[Epoch {epoch+1}/{args.epochs}] Start")
        
        optimizer.zero_grad()

        for step, batch in enumerate(train_loader):
            masks = batch['mask'].to(device)
            images = batch['image'].to(device)
            bs = images.shape[0]

            # [AMP] Autocast (Forward 전체)
            with torch.cuda.amp.autocast():
                # A. VAE Encoding (Mask & Image)
                with torch.no_grad():
                    # Target Image -> Latent
                    latents = vae.encode(images).latent_dist.sample() * SCALING_FACTOR
                    # Condition Mask -> Latent
                    mask_latents = vae.encode(masks).latent_dist.sample() * SCALING_FACTOR

                # B. 노이즈 추가
                noise = torch.randn_like(latents)
                timesteps = torch.randint(0, noise_scheduler.config.num_train_timesteps, (bs,), device=device).long()
                noisy_latents = noise_scheduler.add_noise(latents, noise, timesteps)

                # C. 조건 결합 (Concatenation) -> [Batch, 8, 64, 64]
                unet_input = torch.cat([noisy_latents, mask_latents], dim=1)

                # D. 예측
                noise_pred = unet(unet_input, timesteps).sample
                
                # E. Loss
                loss = F.mse_loss(noise_pred, noise)
                loss = loss / args.gradient_accumulation_steps

            # [AMP] Backward
            scaler.scale(loss).backward()

            # [Accumulation] Update
            if (step + 1) % args.gradient_accumulation_steps == 0:
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad()
                global_step += 1

            current_loss = loss.item() * args.gradient_accumulation_steps
            epoch_loss += current_loss
            
            # 중간 로그 출력 (매 50 step 마다)
            if (step + 1) % 50 == 0:
                print(f"  Step [{step+1}/{len(train_loader)}] Loss: {current_loss:.5f}")

        # --- 주기적 저장 및 샘플링 ---
        if (epoch + 1) % args.save_interval == 0:
            print(f"  Saving checkpoint and sample...")
            # 모델 저장
            save_path = os.path.join(args.output_dir, f"unet_epoch_{epoch+1}")
            unet.save_pretrained(save_path)
            
            # 샘플링 테스트
            unet.eval()
            with torch.cuda.amp.autocast():
                with torch.no_grad():
                    # 첫 번째 배치의 마스크 하나만 가져와서 테스트
                    sample_mask = masks[:1] 
                    sample_mask_latent = vae.encode(sample_mask).latent_dist.sample() * SCALING_FACTOR
                    
                    # 랜덤 노이즈 시작
                    latents = torch.randn(1, latent_dim, 64, 64).to(device)
                    
                    # Sampling Loop (tqdm 제거)
                    for t in noise_scheduler.timesteps:
                        # Mask Latent를 계속 힌트로 줌 (Guided Generation)
                        input_latents = torch.cat([latents, sample_mask_latent], dim=1)
                        
                        model_output = unet(input_latents, t).sample
                        latents = noise_scheduler.step(model_output, t, latents).prev_sample

                    # Decoding
                    decoded_img = vae.decode(latents / SCALING_FACTOR).sample
                    
                    # [Mask | Generated | Original] 비교 저장
                    vis = torch.cat([sample_mask, decoded_img, images[:1]], dim=3)
                    vis = (vis / 2 + 0.5).clamp(0, 1).float()
                    save_image(vis, os.path.join(sample_dir, f"val_epoch_{epoch+1}.png"))

    print("LDM Training Complete.")
    print(f"Final Model Saved: {args.output_dir}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", type=str, required=True, help="Dataset folder (train/masks, train/images)")
    parser.add_argument("--vae_path", type=str, required=True, help="Pretrained VAE folder path")
    parser.add_argument("--output_dir", type=str, default="ldm_result")
    
    parser.add_argument("--resolution", type=int, default=512)
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--gradient_accumulation_steps", type=int, default=4)
    
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--save_interval", type=int, default=10)
    parser.add_argument("--gpu", type=int, default=0)
    parser.add_argument("--num_workers", type=int, default=4)
    
    args = parser.parse_args()
    train_ldm(args)