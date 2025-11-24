import os
import argparse
import torch
from PIL import Image
from tqdm import tqdm
from pathlib import Path

from diffusers import AutoencoderKL, UNet2DModel, DDPMScheduler
from torchvision import transforms

# Stable Diffusion 표준 Scaling Factor
SD_SCALING_FACTOR = 0.18215

def generate(args):
    device = torch.device(f"cuda:{args.gpu}" if torch.cuda.is_available() else "cpu")
    os.makedirs(args.output_dir, exist_ok=True)

    # ------------------------------
    # 1. 모델 로드 (LDM 구성요소)
    # ------------------------------
    print("Loading LDM models...")
    
    # (1) Mask VAE (Condition 인코딩용)
    print(f" -> Loading Mask VAE: {args.mask_vae_path}")
    vae_mask = AutoencoderKL.from_pretrained(args.mask_vae_path).to(device)
    vae_mask.eval()

    # (2) Image VAE (Result 디코딩용)
    print(f" -> Loading Image VAE: {args.image_vae_path}")
    vae_image = AutoencoderKL.from_pretrained(args.image_vae_path).to(device)
    vae_image.eval()

    # (3) UNet (Noise Prediction Model)
    print(f" -> Loading UNet: {args.unet_path}")
    unet = UNet2DModel.from_pretrained(args.unet_path).to(device)
    unet.eval()

    # (4) Scheduler (Diffusion 스케줄러)
    # 학습때 DDPM을 썼으므로 생성도 DDPM (오래 걸리지만 품질 좋음)
    scheduler = DDPMScheduler(num_train_timesteps=1000)
    # 만약 속도를 높이고 싶다면 아래 주석을 풀고 DDIM 등을 쓰세요 (단, import 필요)
    # from diffusers import DDIMScheduler
    # scheduler = DDIMScheduler(num_train_timesteps=1000) 
    # scheduler.set_timesteps(50) # 50 step만 사용

    # ------------------------------
    # 2. 입력 마스크 준비
    # ------------------------------
    input_path = Path(args.input_path)
    if input_path.is_dir():
        valid_exts = ['.jpg', '.png', '.jpeg', '.tiff', '.bmp']
        mask_files = sorted([f for f in input_path.iterdir() if f.suffix.lower() in valid_exts])
    else:
        mask_files = [input_path]

    if not mask_files:
        print(f"No image files found in {args.input_path}")
        return

    print(f"Found {len(mask_files)} masks to process.")

    # 전처리: 학습 때와 동일하게 (Grayscale -> Resize -> Tensor -> Norm)
    mask_transform = transforms.Compose([
        transforms.Resize((args.resolution, args.resolution), interpolation=transforms.InterpolationMode.NEAREST),
        transforms.ToTensor(),
        transforms.Normalize([0.5], [0.5]) 
    ])

    # ------------------------------
    # 3. Diffusion 생성 루프
    # ------------------------------
    h = w = args.resolution // 8 # Latent Size (512 -> 64)
    
    for mask_file in tqdm(mask_files, desc="Generating Images"):
        # (1) 마스크 로드
        mask_pil = Image.open(mask_file).convert("L")
        mask_tensor = mask_transform(mask_pil).unsqueeze(0).to(device) # [1, 1, H, W]

        with torch.no_grad():
            # (2) Condition Encoding (Mask -> Latent)
            mask_latent = vae_mask.encode(mask_tensor).latent_dist.sample() * SD_SCALING_FACTOR
            
            # (3) Random Noise 생성 (Start from Pure Noise)
            latents = torch.randn(1, 4, h, w).to(device)
            
            # (4) Denoising Loop (Reverse Process)
            # 노이즈를 조금씩 제거하며 이미지를 구체화
            for t in scheduler.timesteps:
                # 입력: [현재 노이즈 상태, 마스크 조건]
                input_latents = torch.cat([latents, mask_latent], dim=1)
                
                # UNet이 노이즈 예측
                noise_pred = unet(input_latents, t).sample
                
                # 스케줄러가 노이즈 제거 (Previous Sample 계산)
                latents = scheduler.step(noise_pred, t, latents).prev_sample

            # (5) Decode (Latent -> Image)
            decoded_img = vae_image.decode(latents / SD_SCALING_FACTOR).sample
            
            # (6) 저장
            decoded_img = (decoded_img / 2 + 0.5).clamp(0, 1)
            decoded_img = decoded_img.cpu().permute(0, 2, 3, 1).numpy()[0]
            decoded_img = (decoded_img * 255).round().astype("uint8")
            
            result_pil = Image.fromarray(decoded_img)
            
            save_name = f"{mask_file.stem}.png"
            save_path = os.path.join(args.output_dir, save_name)
            result_pil.save(save_path)

    print(f"Generation Complete! Results saved to: {args.output_dir}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    
    parser.add_argument("--input_path", type=str, required=True, help="Path to input mask or folder")
    parser.add_argument("--output_dir", type=str, default="generated_results")
    
    # 학습된 모델 경로 (train_ldm_pix2pix.py의 결과물)
    parser.add_argument("--unet_path", type=str, required=True, help="Path to trained UNet (e.g. best_unet)")
    parser.add_argument("--mask_vae_path", type=str, required=True, help="Path to Mask VAE")
    parser.add_argument("--image_vae_path", type=str, required=True, help="Path to Image VAE")
    
    parser.add_argument("--resolution", type=int, default=512)
    parser.add_argument("--gpu", type=int, default=0)
    
    args = parser.parse_args()
    generate(args)