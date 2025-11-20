import os
import argparse
import torch
from torchvision import transforms
from torchvision.utils import save_image
from PIL import Image
from pathlib import Path
from tqdm import tqdm

# 모델 정의 파일 임포트
from pix2pix_model import UnetGenerator

def generate(args):
    device = torch.device(f"cuda:{args.gpu}" if torch.cuda.is_available() else "cpu")
    
    # 결과 저장 폴더 생성
    os.makedirs(args.output_dir, exist_ok=True)

    print(f"🚀 Pix2Pix (512x512) 추론 시작")
    print(f" - 모델 경로: {args.model_path}")
    print(f" - 입력 폴더: {args.input_dir}")

    # 1. 모델 초기화
    # 중요: 512x512 학습 모델이므로 num_downs=9로 설정해야 합니다.
    netG = UnetGenerator(input_nc=3, output_nc=3, num_downs=9, ngf=64).to(device)
    
    # 2. 가중치 로드
    if not os.path.exists(args.model_path):
        print(f"❌ 오류: 모델 파일을 찾을 수 없습니다: {args.model_path}")
        return

    try:
        checkpoint = torch.load(args.model_path, map_location=device)
        # state_dict 처리 (혹시 모를 dict 구조 대응)
        if isinstance(checkpoint, dict) and 'state_dict' in checkpoint:
             netG.load_state_dict(checkpoint['state_dict'])
        else:
             netG.load_state_dict(checkpoint)
        print("✅ 모델 가중치 로드 완료")
    except Exception as e:
        print(f"❌ 모델 로드 중 에러 발생: {e}")
        return

    netG.eval() # 평가 모드 (Dropout, BatchNorm 고정)

    # 3. 전처리 정의 (학습과 동일하게)
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

    if len(input_files) == 0:
        print("⚠️ 입력 폴더에 이미지 파일이 없습니다.")
        return

    for filename in tqdm(input_files, desc="Generating Images"):
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
            
        except Exception as e:
            print(f"⚠️ 파일 처리 실패 ({filename}): {e}")

    print(f"🎉 생성 완료! 결과물 위치: {args.output_dir}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--input_dir', type=str, required=True, help="입력 마스크 폴더 경로")
    parser.add_argument('--output_dir', type=str, default='generated_pix2pix_512', help="결과 저장 경로")
    parser.add_argument('--model_path', type=str, required=True, help="학습된 Generator 모델 파일 (.pth)")
    # 512 모델이므로 기본값 512
    parser.add_argument('--size', type=int, default=512, help="이미지 크기")
    parser.add_argument('--gpu', type=int, default=0)
    
    args = parser.parse_args()
    generate(args)