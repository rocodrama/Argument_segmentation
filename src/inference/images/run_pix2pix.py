import os
import argparse
import torch
from PIL import Image
from pathlib import Path
from tqdm import tqdm
from torchvision import transforms

# 모델 정의 파일 임포트 (같은 폴더에 pix2pix_model.py가 있어야 함)
from pix2pix_model import UnetGenerator

def generate(args):
    device = torch.device(f"cuda:{args.gpu}" if torch.cuda.is_available() else "cpu")
    os.makedirs(args.output_dir, exist_ok=True)

    # ------------------------------
    # 1. 모델 로드
    # ------------------------------
    print(f"Loading Generator from: {args.checkpoint}")
    
    # 학습 코드와 동일한 구조로 초기화 (Input 3ch -> Output 3ch, 512px=Downs 9)
    netG = UnetGenerator(input_nc=3, output_nc=3, num_downs=9, ngf=64)
    
    # 가중치 로드
    checkpoint = torch.load(args.checkpoint, map_location=device)
    netG.load_state_dict(checkpoint)
    
    netG.to(device)
    netG.eval()

    # ------------------------------
    # 2. 데이터 준비
    # ------------------------------
    input_path = Path(args.input_path)
    if input_path.is_dir():
        valid_exts = ['.jpg', '.png', '.jpeg', '.tiff', '.bmp']
        mask_files = sorted([f for f in input_path.iterdir() if f.suffix.lower() in valid_exts])
    else:
        mask_files = [input_path]

    if not mask_files:
        print(f"No images found in {args.input_path}")
        return

    print(f"Found {len(mask_files)} masks to process.")

    # 전처리: 학습 때와 동일 (Resize -> ToTensor -> Normalize)
    transform = transforms.Compose([
        transforms.Resize((args.resolution, args.resolution), interpolation=transforms.InterpolationMode.NEAREST),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])

    # ------------------------------
    # 3. 생성 루프
    # ------------------------------
    with torch.no_grad():
        for mask_file in tqdm(mask_files, desc="Generating"):
            # 이미지 로드 (RGB)
            mask = Image.open(mask_file).convert('RGB')
            mask_tensor = transform(mask).unsqueeze(0).to(device) # [1, 3, H, W]
            
            # Inference
            fake_image = netG(mask_tensor)
            
            # Post-processing (Denormalize -1~1 -> 0~255)
            fake_image = (fake_image + 1) / 2.0
            fake_image = fake_image.clamp(0, 1)
            
            # Tensor -> PIL -> Save
            fake_image_cpu = fake_image.cpu().squeeze(0).permute(1, 2, 0).numpy() # [H, W, C]
            fake_image_pil = Image.fromarray((fake_image_cpu * 255).astype('uint8'))
            
            save_name = f"{mask_file.stem}.png"
            save_path = os.path.join(args.output_dir, save_name)
            fake_image_pil.save(save_path)

    print(f"Generation Complete! Saved to {args.output_dir}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    
    # 입력 경로 (마스크 이미지 파일 또는 폴더)
    parser.add_argument('--input_path', type=str, required=True, help='Path to input mask image or folder')
    
    # 모델 체크포인트 경로 (.pth 파일)
    parser.add_argument('--checkpoint', type=str, required=True, help='Path to trained netG model file (.pth)')
    
    # 출력 폴더
    parser.add_argument('--output_dir', type=str, default='generated_gan_results')
    
    # 옵션
    parser.add_argument('--resolution', type=int, default=512)
    parser.add_argument('--gpu', type=int, default=0)
    
    args = parser.parse_args()
    generate(args)