import os
import argparse
import torch
from diffusers import AutoencoderKL, UNet2DModel, DDPMScheduler
from torchvision.utils import save_image
from tqdm import tqdm

def generate_masks(args):
    device = torch.device(f"cuda:{args.gpu}" if torch.cuda.is_available() else "cpu")
    os.makedirs(args.output_dir, exist_ok=True)

    print(f"🚀 Custom LDM 추론 시작")
    print(f" - VAE 경로: {args.vae_path}")
    print(f" - UNet 경로: {args.unet_path}")
    print(f" - Scaling Factor: {args.scale_factor}")

    # 1. 모델 로드
    # 1-1. VAE (직접 학습한 모델)
    try:
        vae = AutoencoderKL.from_pretrained(args.vae_path).to(device)
    except:
        print("❌ 오류: VAE 경로를 확인하세요.")
        return

    # 1-2. UNet (LDM 학습 결과)
    unet = UNet2DModel.from_pretrained(args.unet_path).to(device)
    
    # 1-3. 스케줄러 (학습 때와 동일한 설정)
    scheduler = DDPMScheduler(num_train_timesteps=1000)

    unet.eval()
    vae.eval()

    # 2. 생성 루프
    # Latent Channel 수는 VAE 설정에서 가져옴 (보통 4)
    latent_channels = vae.config.latent_channels
    
    # 배치 단위로 생성
    num_batches = (args.num_samples + args.batch_size - 1) // args.batch_size
    
    for i in range(num_batches):
        # 현재 배치의 크기 계산 (마지막 배치는 작을 수 있음)
        curr_batch_size = min(args.batch_size, args.num_samples - i * args.batch_size)
        
        # A. 랜덤 노이즈 생성 (Latent Space)
        # 512x512 -> 64x64 (f=8)
        latents = torch.randn(
            (curr_batch_size, latent_channels, args.resolution // 8, args.resolution // 8),
            device=device
        )

        # B. Denoising Process (Reverse Diffusion)
        for t in tqdm(scheduler.timesteps, desc=f"Generating Batch {i+1}/{num_batches}"):
            with torch.no_grad():
                # 노이즈 예측
                noise_pred = unet(latents, t).sample
                # 노이즈 제거 (Step)
                latents = scheduler.step(noise_pred, t, latents).prev_sample

        # C. VAE Decoding
        # Scaling Factor로 나누어줘야 원래 스케일로 돌아옴
        latents = latents / args.scale_factor
        
        with torch.no_grad():
            images = vae.decode(latents).sample

        # D. 저장 ([-1, 1] -> [0, 1])
        images = (images / 2 + 0.5).clamp(0, 1)
        
        for j, img in enumerate(images):
            idx = i * args.batch_size + j
            save_path = os.path.join(args.output_dir, f"generated_{idx:04d}.png")
            save_image(img, save_path)

    print(f"🎉 생성 완료! 저장 위치: {args.output_dir}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--unet_path", type=str, required=True, help="학습된 UNet 폴더 (예: ldm_result/best_unet)")
    parser.add_argument("--vae_path", type=str, required=True, help="학습된 VAE 폴더 (예: vae_result/best_vae)")
    parser.add_argument("--output_dir", type=str, default="generated_samples_custom")
    parser.add_argument("--num_samples", type=int, default=10, help="생성할 이미지 수")
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--resolution", type=int, default=512)
    # 중요: 학습 로그에 찍혔던 '권장 Scaling Factor' 값을 여기에 넣으세요.
    parser.add_argument("--scale_factor", type=float, default=1.0, help="Latent Scaling Factor (학습 로그 참고)")
    parser.add_argument("--gpu", type=int, default=0)
    
    args = parser.parse_args()
    generate_masks(args)