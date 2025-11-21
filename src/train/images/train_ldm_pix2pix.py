import os
import argparse
from pathlib import Path
from tqdm import tqdm
from PIL import Image

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
        
        # 파일명 매칭
        self.mask_files = sorted([f for f in self.mask_dir.iterdir() if f.suffix in ['.jpg', '.png', '.jpeg', '.tiff', '.bmp']])
        self.image_files = sorted([f for f in self.image_dir.iterdir() if f.suffix in ['.jpg', '.png', '.jpeg', '.tiff', '.bmp']])

        assert len(self.mask_files) == len(self.image_files), "Mask와 Image 파일 수가 다릅니다."

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

    # --- 1. Pretrained VAE 로드 (Frozen) ---
    print(f"❄️ VAE 로드 중 ({args.model_id})...")
    vae = AutoencoderKL.from_pretrained(args.model_id, subfolder="vae").to(device)
    vae.requires_grad_(False)
    vae.eval()

    # --- 2. UNet 초기화 ---
    # 입력 채널 = Noisy Latent(4) + Mask Latent(4) = 8
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

    # [추가] GradScaler 초기화 (Mixed Precision)
    scaler = torch.cuda.amp.GradScaler()

    # --- 3. 데이터 로더 ---
    train_dataset = PairedDataset(args.data_dir, split='train', size=args.resolution)
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers)
    
    print(f"🚀 Pix2Pix LDM 학습 시작: {len(train_dataset)}장")
    print(f"   - Accumulation Steps: {args.gradient_accumulation_steps}")
    print(f"   - Effective Batch Size: {args.batch_size * args.gradient_accumulation_steps}")

    # --- 4. 학습 루프 ---
    global_step = 0
    
    for epoch in range(args.epochs):
        unet.train()
        epoch_loss = 0.0
        
        progress_bar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{args.epochs}")
        
        # Optimizer 초기화 (accumulation 시작 전)
        optimizer.zero_grad()

        for step, batch in enumerate(progress_bar):
            masks = batch['mask'].to(device)
            images = batch['image'].to(device)
            bs = images.shape[0]

            # [수정] Autocast 적용 (Forward 전체)
            with torch.cuda.amp.autocast():
                # A. Latent Encoding
                with torch.no_grad():
                    latents = vae.encode(images).latent_dist.sample() * SD_SCALING_FACTOR
                    mask_latents = vae.encode(masks).latent_dist.sample() * SD_SCALING_FACTOR

                # B. 노이즈 추가
                noise = torch.randn_like(latents)
                timesteps = torch.randint(0, noise_scheduler.config.num_train_timesteps, (bs,), device=device).long()
                noisy_latents = noise_scheduler.add_noise(latents, noise, timesteps)

                # C. 조건 결합 (Concatenation) [Batch, 8, 64, 64]
                unet_input = torch.cat([noisy_latents, mask_latents], dim=1)

                # D. 예측 및 Loss
                noise_pred = unet(unet_input, timesteps).sample
                loss = F.mse_loss(noise_pred, noise)
                
                # [추가] Gradient Accumulation을 위해 Loss 나누기
                loss = loss / args.gradient_accumulation_steps

            # E. Backward (Scaler 사용)
            scaler.scale(loss).backward()

            # [추가] 지정된 스텝마다 업데이트 수행
            if (step + 1) % args.gradient_accumulation_steps == 0:
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad()
                global_step += 1

            # 로깅을 위해 loss 복원
            current_loss = loss.item() * args.gradient_accumulation_steps
            epoch_loss += current_loss
            progress_bar.set_postfix({"Loss": current_loss})

        # --- 주기적 저장 및 샘플링 ---
        if (epoch + 1) % args.save_interval == 0:
            # 모델 저장
            save_path = os.path.join(args.output_dir, f"unet_epoch_{epoch+1}")
            unet.save_pretrained(save_path)
            
            # 샘플링 테스트
            unet.eval()
            # 샘플링도 autocast 적용 (속도 향상)
            with torch.cuda.amp.autocast():
                with torch.no_grad():
                    sample_mask = masks[:1] # [1, 3, 512, 512]
                    sample_mask_latent = vae.encode(sample_mask).latent_dist.sample() * SD_SCALING_FACTOR
                    
                    latents = torch.randn(1, 4, 64, 64).to(device)
                    
                    for t in tqdm(noise_scheduler.timesteps, desc="Sampling", leave=False):
                        input_latents = torch.cat([latents, sample_mask_latent], dim=1)
                        model_output = unet(input_latents, t).sample
                        latents = noise_scheduler.step(model_output, t, latents).prev_sample

                    # Decoding
                    decoded_img = vae.decode(latents / SD_SCALING_FACTOR).sample
                    
                    # 시각화 (FP32로 변환 후 저장)
                    vis = torch.cat([sample_mask, decoded_img], dim=3)
                    vis = (vis / 2 + 0.5).clamp(0, 1).float() 
                    save_image(vis, os.path.join(sample_dir, f"val_epoch_{epoch+1}.png"))

    print("🎉 학습 완료!")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", type=str, required=True, help="split_dataset_physical 결과 폴더")
    parser.add_argument("--output_dir", type=str, default="ldm_pix2pix_result")
    parser.add_argument("--model_id", type=str, default="CompVis/stable-diffusion-v1-4")
    parser.add_argument("--resolution", type=int, default=512)
    
    # 배치 관련 설정
    parser.add_argument("--batch_size", type=int, default=4, help="GPU당 배치 사이즈")
    parser.add_argument("--gradient_accumulation_steps", type=int, default=4, help="업데이트 전 누적할 스텝 수")
    
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--epochs", type=int, default=200)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--save_interval", type=int, default=10)
    parser.add_argument("--gpu", type=int, default=0)
    
    args = parser.parse_args()
    train(args)