import os
import argparse
import torch
from diffusers import AutoencoderKL, UNet2DModel, DDPMScheduler
from torchvision.utils import save_image

def generate_masks(args):
    device = torch.device(f"cuda:{args.gpu}" if torch.cuda.is_available() else "cpu")
    os.makedirs(args.output_dir, exist_ok=True)

    print(f"Start Custom LDM Inference")
    print(f" - VAE Path: {args.vae_path}")
    print(f" - UNet Path: {args.unet_path}")
    print(f" - Scaling Factor: {args.scale_factor}")

    # 1. 모델 로드
    # 1-1. VAE
    try:
        vae = AutoencoderKL.from_pretrained(args.vae_path).to(device)
    except:
        print("Error: Check VAE path.")
        return

    # 1-2. UNet
    unet = UNet2DModel.from_pretrained(args.unet_path).to(device)
    
    # 1-3. 스케줄러
    scheduler = DDPMScheduler(num_train_timesteps=1000)

    unet.eval()
    vae.eval()

    # 2. 생성 루프
    latent_channels = vae.config.latent_channels
    
    # 배치 단위로 생성
    num_batches = (args.num_samples + args.batch_size - 1) // args.batch_size
    
    for i in range(num_batches):
        curr_batch_size = min(args.batch_size, args.num_samples - i * args.batch_size)
        print(f"\n[Batch {i+1}/{num_batches}] Generating {curr_batch_size} images...")
        
        # A. 랜덤 노이즈 생성 (Latent Space)
        latents = torch.randn(
            (curr_batch_size, latent_channels, args.resolution // 8, args.resolution // 8),
            device=device
        )

        # B. Denoising Process (Reverse Diffusion)
        # tqdm 대신 enumerate를 사용하여 진행상황 체크
        for step_index, t in enumerate(scheduler.timesteps):
            with torch.no_grad():
                noise_pred = unet(latents, t).sample
                latents = scheduler.step(noise_pred, t, latents).prev_sample
            
            # 200 스텝마다 진행 로그 출력
            if (step_index + 1) % 200 == 0:
                print(f"  Denoising Step {step_index + 1}/{len(scheduler.timesteps)}")

        # C. VAE Decoding
        latents = latents / args.scale_factor
        
        with torch.no_grad():
            images = vae.decode(latents).sample

        # D. 저장
        images = (images / 2 + 0.5).clamp(0, 1)
        
        for j, img in enumerate(images):
            idx = i * args.batch_size + j
            save_path = os.path.join(args.output_dir, f"generated_{idx:04d}.png")
            save_image(img, save_path)
            print(f"  Saved: {save_path}")

    print(f"Generation Complete. Output Directory: {args.output_dir}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--unet_path", type=str, required=True, help="Trained UNet path")
    parser.add_argument("--vae_path", type=str, required=True, help="Trained VAE path")
    parser.add_argument("--output_dir", type=str, default="generated_samples_custom")
    parser.add_argument("--num_samples", type=int, default=10, help="Number of samples to generate")
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--resolution", type=int, default=512)
    parser.add_argument("--scale_factor", type=float, default=1.0, help="Latent Scaling Factor")
    parser.add_argument("--gpu", type=int, default=0)
    
    args = parser.parse_args()
    generate_masks(args)