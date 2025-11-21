import os
import argparse
import torch
from torchvision import transforms
from torchvision.utils import save_image
from PIL import Image
from pathlib import Path

# 모델 정의 파일 임포트
from pix2pix_model import UnetGenerator

def generate(args):
    device = torch.device(f"cuda:{args.gpu}" if torch.cuda.is_available() else "cpu")
    
    # 결과 저장 폴더 생성
    os.makedirs(args.output_dir, exist_ok=True)

    print(f"Start Pix2Pix (512x512) Inference")
    print(f" - Model Path: {args.model_path}")
    print(f" - Input Dir: {args.input_dir}")

    # 1. 모델 초기화
    # 512x512 학습 모델이므로 num_downs=9 설정
    netG = UnetGenerator(input_nc=3, output_nc=3, num_downs=9, ngf=64).to(device)
    
    # 2. 가중치 로드
    if not os.path.exists(args.model_path):
        print(f"Error: Model file not found: {args.model_path}")
        return

    try:
        checkpoint = torch.load(args.model_path, map_location=device)
        # state_dict 처리
        if isinstance(checkpoint, dict) and 'state_dict' in checkpoint:
             netG.load_state_dict(checkpoint['state_dict'])
        else:
             netG.load_state_dict(checkpoint)
        print("Model weights loaded successfully.")
    except Exception as e:
        print(f"Error loading model: {e}")
        return

    netG.eval()

    # 3. 전처리 정의
    transform = transforms.Compose([
        transforms.Resize((args.size, args.size)), # 512
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])

    # 4. 이미지 생성 루프
    input_files = sorted([
        f for f in os.listdir(args.input_dir) 
        if f.lower().endswith(('.png', '.jpg', '.jpeg', '.tiff', '.bmp'))
    ])
    
    total_files = len(input_files)

    if total_files == 0:
        print("No image files found in input directory.")
        return

    print(f"Found {total_files} images. Starting generation...")

    for i, filename in enumerate(input_files):
        print(f"[{i+1}/{total_files}] Processing: {filename}")
        input_path = os.path.join(args.input_dir, filename)
        
        try:
            # 이미지 로드 (RGB 변환)
            mask = Image.open(input_path).convert('RGB') 
            
            # 전처리 및 배치 차원 추가 [1, 3, 512, 512]
            input_tensor = transform(mask).unsqueeze(0).to(device)

            # 추론
            with torch.no_grad():
                fake_image = netG(input_tensor)

            # 후처리: [-1, 1] -> [0, 1]
            fake_image = (fake_image + 1) / 2.0
            
            # 저장
            save_path = os.path.join(args.output_dir, filename)
            save_image(fake_image, save_path)
            print(f"  Saved: {save_path}")
            
        except Exception as e:
            print(f"  Failed ({filename}): {e}")

    print(f"Generation Complete. Output Directory: {args.output_dir}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--input_dir', type=str, required=True, help="Input mask folder")
    parser.add_argument('--output_dir', type=str, default='generated_pix2pix_512', help="Output folder")
    parser.add_argument('--model_path', type=str, required=True, help="Trained Generator (.pth)")
    parser.add_argument('--size', type=int, default=512, help="Image size")
    parser.add_argument('--gpu', type=int, default=0)
    
    args = parser.parse_args()
    generate(args)