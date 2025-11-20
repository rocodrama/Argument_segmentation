import os
import argparse
import glob
import shutil
import random
from pathlib import Path
from tqdm import tqdm

def parse_ratios(ratio_str):
    """비율 문자열 파싱 (예: '8:1:1')"""
    try:
        parts = [float(p) for p in ratio_str.split(':')]
        total = sum(parts)
        return [p / total for p in parts]
    except:
        raise argparse.ArgumentTypeError("비율 형식 오류. '8:1:1' 또는 '0.8:0.1:0.1' 형태로 입력하세요.")

def split_physical(input_dir, out_dir, ratios, seed=42, move_files=False):
    # 1. 설정 및 시드 고정
    random.seed(seed)
    action_name = "이동(Move)" if move_files else "복사(Copy)"
    
    input_path = Path(input_dir)
    out_path = Path(out_dir)
    
    images_dir = input_path / 'images'
    masks_dir = input_path / 'masks'

    if not images_dir.exists() or not masks_dir.exists():
        print(f"❌ 오류: 입력 폴더 안에 'images'와 'masks' 폴더가 있어야 합니다.")
        return

    # 2. 파일 쌍 찾기
    print("🔍 파일 쌍 매칭 중...")
    # 지원 확장자
    exts = ['*.jpg', '*.jpeg', '*.png', '*.tiff', '*.tif', '*.bmp']
    image_files = []
    for ext in exts:
        image_files.extend(images_dir.glob(ext))
    
    # 마스크 매칭을 위한 맵핑
    mask_map = {}
    for ext in exts:
        for m in masks_dir.glob(ext):
            # 마스크 파일명에서 확장자 제거
            m_stem = m.stem
            # '_mask'가 있다면 제거한 이름도 키로 사용 (매칭 유연성)
            if m_stem.endswith('_mask'):
                key = m_stem[:-5] # '_mask' 제거
            else:
                key = m_stem
            mask_map[key] = m

    pairs = []
    for img_path in image_files:
        stem = img_path.stem
        if stem in mask_map:
            pairs.append((img_path, mask_map[stem]))
    
    print(f"✅ 총 {len(pairs)}쌍의 데이터 발견.")
    if len(pairs) == 0:
        return

    # 3. 셔플 및 분할
    random.shuffle(pairs)
    
    n_total = len(pairs)
    train_r, val_r, test_r = ratios
    
    n_train = int(n_total * train_r)
    n_val = int(n_total * val_r)
    # 나머지는 테스트
    
    splits = {
        'train': pairs[:n_train],
        'val': pairs[n_train:n_train+n_val],
        'test': pairs[n_train+n_val:]
    }

    # 4. 물리적 복사/이동 수행
    print(f"🚀 데이터셋 물리적 분할 시작 ({action_name})...")
    
    for split_name, split_pairs in splits.items():
        if not split_pairs:
            continue
            
        # 타겟 디렉토리 생성 (예: output/train/images, output/train/masks)
        target_img_dir = out_path / split_name / 'images'
        target_mask_dir = out_path / split_name / 'masks'
        
        os.makedirs(target_img_dir, exist_ok=True)
        os.makedirs(target_mask_dir, exist_ok=True)
        
        print(f"📂 {split_name.upper()} 셋 처리 중 ({len(split_pairs)}장)...")
        
        for img_src, mask_src in tqdm(split_pairs):
            # 파일명 유지
            img_dst = target_img_dir / img_src.name
            mask_dst = target_mask_dir / mask_src.name
            
            if move_files:
                shutil.move(str(img_src), str(img_dst))
                shutil.move(str(mask_src), str(mask_dst))
            else:
                shutil.copy2(str(img_src), str(img_dst))
                shutil.copy2(str(mask_src), str(mask_dst))

    print("\n✨ 작업 완료!")
    print(f"결과 폴더 구조: {out_dir}/[train|val|test]/[images|masks]")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="데이터셋을 train/val/test 폴더로 물리적으로 분할합니다.")
    parser.add_argument("--input", type=str, required=True, help="원본 데이터 폴더 (내부에 images, masks 포함)")
    parser.add_argument("--out", type=str, required=True, help="결과가 저장될 폴더")
    parser.add_argument("--ratio", type=str, default="8:1:1", help="분할 비율 (예: 8:1:1)")
    parser.add_argument("--seed", type=int, default=42, help="랜덤 시드")
    parser.add_argument("--move", action="store_true", help="파일을 복사하지 않고 이동시킵니다 (주의!)")
    
    args = parser.parse_args()
    
    ratios = parse_ratios(args.ratio)
    split_physical(args.input, args.out, ratios, args.seed, args.move)