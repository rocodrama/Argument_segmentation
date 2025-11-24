import os
import argparse
import torch
import numpy as np
from PIL import Image
from tqdm import tqdm
from pathlib import Path

from diffusers import AutoencoderKL, UNet2DModel, DDPMScheduler

def load_scaling_factor(unet_path, override_factor=None):
    """
    Scaling Factor를 로드합니다.
    1. override_factor가 있으면 우선 사용
    2. UNet 경로의 상위 폴더에서 scaling_factor.txt 탐색
    3. 없으면 기본값 0.18215 반환 (Stable Diffusion 기본값)
    """
    if override_factor is not None:
        print(f"Using provided Scaling Factor: {override_factor}")
        return override_factor

    # UNet 경로가 .../best_unet 또는 .../checkpoint-X/unet 일 수 있음
    # 따라서 부모의 부모 디렉토리까지 탐색
    potential_paths = [
        Path(unet_path).parent / "scaling_factor.txt",
        Path(unet_path).parent.parent / "scaling_factor.txt"
    ]

    for p in potential_paths:
        if p.exists():
            with open(p, "r") as f:
                factor = float(f.read().strip())
            print(f"Loaded Scaling Factor from {p}: {factor}")
            return factor
    
    print("Warning: scaling_factor.txt not found. Using default 0.18215.")
    return 0.18215

@torch.no_grad()
def generate_masks(args):
    device = torch.device(f"cuda:{args.gpu}" if torch.cuda.is_available() else "cpu")
    os.makedirs(args.output_dir, exist_ok=True)

    print(f"Loading models...")
    # 1. VAE 로드
    vae = AutoencoderKL.from_pretrained(args.vae_path).to(device)
    vae.eval()

    # 2. UNet 로드
    unet = UNet2DModel.from_pretrained(args.unet_path).to(device)
    unet.eval()

    # 3. Scheduler 설정 (학습과 동일하게 DDPM 사용)
    # 빠른 생성을 원하면 DDIMScheduler로 교체 가능
    scheduler = DDPMScheduler(num_train_timesteps=1000)

    # 4. Scaling Factor 설정
    scaling_factor = load_scaling_factor(args.unet_path, args.scaling_factor)

    print(f"Start generating {args.num_samples} images...")
    
    # Latent 차원 계산 (Resolution / 8)
    h = w = args.resolution // 8
    c = unet.config.in_channels  # 보통 4
    
    # 배치 단위로 생성
    num_batches = (args.num_samples + args.batch_size - 1) // args.batch_size
    
    generated_count = 0

    for i in range(num_batches):
        # 현재 배치 크기 계산 (마지막 배치는 작을 수 있음)
        current_bs = min(args.batch_size, args.num_samples - generated_count)
        
        # 1. 랜덤 노이즈 생성 (Latent Space)
        latents = torch.randn(current_bs, c, h, w).to(device)

        # 2. Denoising Loop (Diffusion Process)
        # tqdm으로 진행상황 표시
        for t in tqdm(scheduler.timesteps, desc=f"Batch {i+1}/{num_batches}"):
            # 모델 예측 (Noise Prediction)
            model_output = unet(latents, t).sample
            
            # Scheduler Step (Noise 제거)
            latents = scheduler.step(model_output, t, latents).prev_sample

        # 3. Decode (VAE)
        # Scaling Factor 역연산
        latents = latents / scaling_factor
        decoded_images = vae.decode(latents).sample

        # 4. Post-processing
        # [-1, 1] -> [0, 1] -> [0, 255]
        decoded_images = (decoded_images / 2 + 0.5).clamp(0, 1)
        decoded_images = decoded_images.cpu().permute(0, 2, 3, 1).numpy()
        decoded_images = (decoded_images * 255).round().astype("uint8")

        # 5. 저장
        for img_array in decoded_images:
            # (H, W, C) -> (H, W) for Grayscale if C=1
            if img_array.shape[2] == 1:
                img_array = img_array.squeeze(2)
                img = Image.fromarray(img_array, mode="L")
            else:
                img = Image.fromarray(img_array)
                
            save_name = f"generated_{generated_count:05d}.png"
            img.save(os.path.join(args.output_dir, save_name))
            generated_count += 1

    print(f"Done! Generated images saved to {args.output_dir}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    # 경로 설정
    parser.add_argument("--unet_path", type=str, required=True, help="Path to the trained UNet folder (e.g., ldm_result/best_unet)")
    parser.add_argument("--vae_path", type=str, required=True, help="Path to the trained VAE folder")
    parser.add_argument("--output_dir", type=str, default="generated_masks", help="Directory to save generated images")
    
    # 생성 옵션
    parser.add_argument("--num_samples", type=int, default=10, help="Total number of images to generate")
    parser.add_argument("--batch_size", type=int, default=4, help="Batch size for generation")
    parser.add_argument("--resolution", type=int, default=512, help="Image resolution (must match training)")
    parser.add_argument("--scaling_factor", type=float, default=None, help="Override scaling factor (optional)")
    parser.add_argument("--gpu", type=int, default=0)
    
    args = parser.parse_args()
    generate_masks(args)