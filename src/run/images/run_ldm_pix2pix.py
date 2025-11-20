import os
import argparse
import torch
from PIL import Image
from tqdm import tqdm
from torchvision import transforms
from torchvision.utils import save_image
from diffusers import AutoencoderKL, UNet2DModel, DDPMScheduler

SD_SCALING_FACTOR = 0.18215

def generate(args):
    device = torch.device(f"cuda:{args.gpu}" if torch.cuda.is_available() else "cpu")
    os.makedirs(args.output_dir, exist_ok=True)

    print(f"🚀 Pix2Pix LDM 추론 시작")
    print(f" - UNet: {args.unet_path}")
    
    # 1. 모델 로드
    vae = AutoencoderKL.from_pretrained(args.model_id, subfolder="vae").to(device)
    unet = UNet2DModel.from_pretrained(args.unet_path).to(device)
    scheduler = DDPMScheduler(num_train_timesteps=1000)

    unet.eval()
    vae.eval()

    # 2. 데이터 준비
    transform = transforms.Compose([
        transforms.Resize((args.resolution, args.resolution), interpolation=transforms.InterpolationMode.NEAREST),
        transforms.ToTensor(),
        transforms.Normalize([0.5], [0.5])
    ])

    input_files = sorted([f for f in os.listdir(args.input_dir) if f.lower().endswith(('.png', '.jpg', '.jpeg'))])
    
    # 3. 생성 루프
    for filename in tqdm(input_files, desc="Generating"):
        mask_path = os.path.join(args.input_dir, filename)
        mask_img = Image.open(mask_path).convert("RGB")
        mask_tensor = transform(mask_img).unsqueeze(0).to(device) # [1, 3, 512, 512]

        # Mask Encoding (Condition)
        with torch.no_grad():
            mask_latent = vae.encode(mask_tensor).latent_dist.sample() * SD_SCALING_FACTOR
        
        # Random Noise Start
        latents = torch.randn(1, 4, args.resolution // 8, args.resolution // 8).to(device)

        # Denoising Loop
        for t in tqdm(scheduler.timesteps, leave=False):
            with torch.no_grad():
                # 매 스텝마다 Condition(Mask Latent)을 붙여줌
                model_input = torch.cat([latents, mask_latent], dim=1)
                
                # 예측
                noise_pred = unet(model_input, t).sample
                
                # 스텝 업데이트
                latents = scheduler.step(noise_pred, t, latents).prev_sample

        # Decoding
        with torch.no_grad():
            image = vae.decode(latents / SD_SCALING_FACTOR).sample

        # 저장
        image = (image / 2 + 0.5).clamp(0, 1)
        save_path = os.path.join(args.output_dir, filename)
        save_image(image, save_path)

    print(f"🎉 생성 완료: {args.output_dir}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_dir", type=str, required=True, help="입력 마스크 폴더")
    parser.add_argument("--output_dir", type=str, default="generated_ldm_pix2pix")
    parser.add_argument("--unet_path", type=str, required=True, help="학습된 UNet 폴더")
    parser.add_argument("--model_id", type=str, default="CompVis/stable-diffusion-v1-4")
    parser.add_argument("--resolution", type=int, default=512)
    parser.add_argument("--gpu", type=int, default=0)
    
    args = parser.parse_args()
    generate(args)