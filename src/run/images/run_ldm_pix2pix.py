import os
import argparse
import torch
from PIL import Image
from torchvision import transforms
from torchvision.utils import save_image
from diffusers import AutoencoderKL, UNet2DModel, DDPMScheduler

SD_SCALING_FACTOR = 0.18215

def generate(args):
    device = torch.device(f"cuda:{args.gpu}" if torch.cuda.is_available() else "cpu")
    os.makedirs(args.output_dir, exist_ok=True)

    print(f"Start Pix2Pix LDM Inference")
    print(f" - UNet: {args.unet_path}")
    
    # 1. 모델 로드
    try:
        vae = AutoencoderKL.from_pretrained(args.model_id, subfolder="vae").to(device)
    except:
        # 로컬 경로나 다른 구조일 경우 대비
        vae = AutoencoderKL.from_pretrained(args.model_id).to(device)
        
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
    total_files = len(input_files)
    print(f"Found {total_files} mask images.")
    
    # 3. 생성 루프
    for i, filename in enumerate(input_files):
        print(f"[{i+1}/{total_files}] Processing: {filename}")
        
        mask_path = os.path.join(args.input_dir, filename)
        mask_img = Image.open(mask_path).convert("RGB")
        mask_tensor = transform(mask_img).unsqueeze(0).to(device) # [1, 3, 512, 512]

        # Mask Encoding (Condition)
        with torch.no_grad():
            mask_latent = vae.encode(mask_tensor).latent_dist.sample() * SD_SCALING_FACTOR
        
        # Random Noise Start
        latents = torch.randn(1, 4, args.resolution // 8, args.resolution // 8).to(device)

        # Denoising Loop (tqdm 제거)
        for step_index, t in enumerate(scheduler.timesteps):
            with torch.no_grad():
                # 매 스텝마다 Condition(Mask Latent)을 붙여줌
                model_input = torch.cat([latents, mask_latent], dim=1)
                
                # 예측
                noise_pred = unet(model_input, t).sample
                
                # 스텝 업데이트
                latents = scheduler.step(noise_pred, t, latents).prev_sample
            
            # (옵션) 너무 오래 걸리면 진행상황 출력 (매 200 스텝)
            if (step_index + 1) % 200 == 0:
                 print(f"  Step {step_index + 1}/1000")

        # Decoding
        with torch.no_grad():
            image = vae.decode(latents / SD_SCALING_FACTOR).sample

        # 저장
        image = (image / 2 + 0.5).clamp(0, 1)
        save_path = os.path.join(args.output_dir, filename)
        save_image(image, save_path)
        print(f"  Saved: {save_path}")

    print(f"Generation Complete: {args.output_dir}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_dir", type=str, required=True, help="Input mask folder")
    parser.add_argument("--output_dir", type=str, default="generated_ldm_pix2pix")
    parser.add_argument("--unet_path", type=str, required=True, help="Trained UNet folder")
    parser.add_argument("--model_id", type=str, default="CompVis/stable-diffusion-v1-4")
    parser.add_argument("--resolution", type=int, default=512)
    parser.add_argument("--gpu", type=int, default=0)
    
    args = parser.parse_args()
    generate(args)