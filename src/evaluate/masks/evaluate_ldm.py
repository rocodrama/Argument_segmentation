import argparse
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
import os
from pathlib import Path

# torch_fidelity 라이브러리 임포트
from torch_fidelity import calculate_metrics

class RGBMaskDataset(Dataset):
    """
    평가를 위해 Grayscale 마스크를 RGB로 변환하여 로드하는 데이터셋
    (InceptionV3 모델은 3채널 입력을 요구하기 때문)
    """
    def __init__(self, root_dir):
        self.root_dir = Path(root_dir)
        self.files = sorted([
            f for f in os.listdir(root_dir) 
            if f.lower().endswith(('.png', '.jpg', '.jpeg', '.tiff', '.bmp'))
        ])
        self.transform = transforms.Compose([
            transforms.Resize((299, 299)), # InceptionV3 입력 크기
            transforms.ToTensor(),
            # Grayscale -> RGB (채널 복제)
            transforms.Lambda(lambda x: x.repeat(3, 1, 1) if x.shape[0] == 1 else x), 
            transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
        ])

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        img_path = self.root_dir / self.files[idx]
        # 'L' 모드로 열어서 transform에서 RGB로 복제
        img = Image.open(img_path).convert("L") 
        return self.transform(img)

def evaluate(args):
    print(f"📊 평가 시작...")
    print(f" - 실제 데이터 경로: {args.real_dir}")
    print(f" - 생성 데이터 경로: {args.fake_dir}")

    # torch-fidelity는 경로 문자열을 직접 입력받아 내부적으로 처리할 수도 있고,
    # 커스텀 데이터셋을 받을 수도 있습니다. 
    # 여기서는 흑백->RGB 변환이 필요하므로 Wrapper를 씌우는게 안전하지만,
    # 편의상 torch-fidelity의 기능을 활용해 경로를 직접 넘기는 방식을 먼저 시도합니다.
    # (만약 내부적으로 채널 에러가 나면 위 데이터셋 클래스를 활용해야 합니다.)
    
    # FID 및 IS 계산
    metrics_dict = calculate_metrics(
        input1=args.real_dir, 
        input2=args.fake_dir, 
        cuda=True, 
        isc=True, # Inception Score 계산
        fid=True, # FID 계산
        kid=False,
        verbose=True,
    )

    print("\n" + "="*30)
    print(f"🏆 평가 결과")
    print(f" - FID (낮을수록 좋음): {metrics_dict['frechet_inception_distance']:.4f}")
    print(f" - IS  (높을수록 좋음): {metrics_dict['inception_score_mean']:.4f}")
    print("="*30)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--real_dir", type=str, required=True, help="원본(Train/Test) 마스크 이미지 폴더")
    parser.add_argument("--fake_dir", type=str, required=True, help="LDM이 생성한 이미지 폴더")
    args = parser.parse_args()
    
    # 데이터가 충분한지 확인 (적어도 수천 장 권장, 최소 수백 장)
    real_count = len(os.listdir(args.real_dir))
    fake_count = len(os.listdir(args.fake_dir))
    
    if real_count < 100 or fake_count < 100:
        print("⚠️ 경고: 데이터 수가 너무 적으면 점수가 부정확할 수 있습니다. (권장: 1000장 이상)")
        
    evaluate(args)