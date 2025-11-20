import os
import argparse
import torch
from diffusers import AutoencoderKL, UNet2DModel, DDPMScheduler
from torchvision.utils import save_image
from torchvision.transforms import Grayscale
from tqdm import tqdm

# SD VAE 표준 Scaling Factor
SD_SCALING_FACTOR = 0.18215

def generate_masks(args):
    device = torch.device(f"cuda:{args.gpu}" if torch.cuda.is_available() else "cpu")
    os.makedirs(args.output_dir, exist_ok=True)

    print(f"🚀 SD-VAE LDM 추론 시작")
    print(f" - UNet 경로: {args.unet_path}")
    print(f" - Scaling Factor: {SD_SCALING_FACTOR} (Fixed)")

    # 1. 모델 로드
    # 1-1. Pretrained VAE
    vae = AutoencoderKL.from_pretrained(args.model_id, subfolder="vae").to(device)
    
    # 1-2. 학습된 UNet
    unet = UNet2DModel.from_pretrained(args.unet_path).to(device)
    
    # 1-3. 스케줄러
    scheduler = DDPMScheduler(num_train_timesteps=1000)

    unet.eval()
    vae.eval()

    # 2. 생성 루프
    num_batches = (args.num_samples + args.batch_size - 1) // args.batch_size
    
    for i in range(num_batches):
        curr_batch_size = min(args.batch_size, args.num_samples - i * args.batch_size)
        
        # A. 랜덤 노이즈 (SD VAE는 Latent Channel이 4)
        latents = torch.randn(
            (curr_batch_size, 4, args.resolution // 8, args.resolution // 8),
            device=device
        )

        # B. Denoising
        for t in tqdm(scheduler.timesteps, desc=f"Batch {i+1}/{num_batches}"):
            with torch.no_grad():
                noise_pred = unet(latents, t).sample
                latents = scheduler.step(noise_pred, t, latents).prev_sample

        # C. Decoding (Scale Factor 적용)
        latents = latents / SD_SCALING_FACTOR
        with torch.no_grad():
            images = vae.decode(latents).sample

        # D. 후처리 및 저장
        images = (images / 2 + 0.5).clamp(0, 1)
        
        # SD VAE는 출력이 RGB(3채널)이므로, 마스크 용도라면 흑백(1채널)으로 변환하는게 좋음
        if args.save_grayscale:
            # R,G,B 평균을 내서 1채널로 만듦
            images = images.mean(dim=1, keepdim=True)

        for j, img in enumerate(images):
            idx = i * args.batch_size + j
            save_path = os.path.join(args.output_dir, f"generated_{idx:04d}.png")
            save_image(img, save_path)

    print(f"🎉 생성 완료! 저장 위치: {args.output_dir}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--unet_path", type=str, required=True, help="학습된 UNet 폴더 (예: ldm_sd_result/best_unet)")
    parser.add_argument("--model_id", type=str, default="CompVis/stable-diffusion-v1-4", help="VAE 모델 ID")
    parser.add_argument("--output_dir", type=str, default="generated_samples_sd")
    parser.add_argument("--num_samples", type=int, default=10)
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--resolution", type=int, default=512)
    parser.add_argument("--save_grayscale", action='store_true', help="결과를 흑백으로 저장하려면 사용")
    parser.add_argument("--gpu", type=int, default=0)
    
    args = parser.parse_args()
    generate_masks(args)