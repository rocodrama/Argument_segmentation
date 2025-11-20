import os
import argparse
import glob
import csv
from sklearn.model_selection import train_test_split
import numpy as np
from typing import List, Tuple, Dict

def parse_ratios(ratio_str: str) -> List[float]:
    """
    비율 문자열(예: '1:1:2' 또는 '0.1:0.1:0.8')을 파싱하여 합이 1이 되도록 정규화합니다.
    """
    try:
        parts = [float(p) for p in ratio_str.split(':')]
    except ValueError:
        raise argparse.ArgumentTypeError(f"비율 형식 오류: '{ratio_str}'. ':'로 구분된 숫자여야 합니다.")

    if not all(p >= 0 for p in parts):
        raise argparse.ArgumentTypeError("비율은 음수가 될 수 없습니다.")

    ratio_sum = sum(parts)
    if ratio_sum == 0:
        raise argparse.ArgumentTypeError("비율의 합은 0이 될 수 없습니다.")

    # 합이 1이 되도록 정규화
    normalized_ratios = [p / ratio_sum for p in parts]
    
    if len(normalized_ratios) != 3:
        raise argparse.ArgumentTypeError("Train, Val, Test 세 개의 비율이 필요합니다. (예: '0.8:0.1:0.1')")

    return normalized_ratios

def find_paired_data(input_dir: str) -> List[Tuple[str, str]]:
    """
    images 및 masks 폴더에서 동일한 이름을 가진 파일 쌍을 찾아 경로 리스트를 반환합니다.
    """
    images_dir = os.path.join(input_dir, 'images')
    masks_dir = os.path.join(input_dir, 'masks')
    
    IMAGE_EXTENSIONS = ('*.jpg', '*.jpeg', '*.png', '*.tiff', '*.tif', '*.gif', '*.bmp', '*.webp')
    
    # 1. 이미지 파일 목록 수집
    all_image_paths = []
    for ext in IMAGE_EXTENSIONS:
        all_image_paths.extend(glob.glob(os.path.join(images_dir, ext)))

    image_map = {} # Key: 기본 이름, Value: 이미지 경로
    for img_path in all_image_paths:
        base_name = os.path.splitext(os.path.basename(img_path))[0]
        image_map[base_name] = img_path

    # 2. 마스크 파일 쌍 매칭
    paired_data_map = {} # Key: 기본 이름, Value: (이미지 경로, 마스크 경로)
    all_mask_paths = []
    for ext in IMAGE_EXTENSIONS:
        all_mask_paths.extend(glob.glob(os.path.join(masks_dir, ext)))

    for mask_path in all_mask_paths:
        mask_base_name_full = os.path.splitext(os.path.basename(mask_path))[0]
        
        # Case 1: 이름이 일치하는 경우
        if mask_base_name_full in image_map:
            paired_data_map[mask_base_name_full] = (image_map[mask_base_name_full], mask_path)
            
        # Case 2: '_mask' 접미사를 제외하고 이름이 일치하는 경우
        elif mask_base_name_full.endswith('_mask'):
            base_name_no_mask = mask_base_name_full[:-len('_mask')]
            if base_name_no_mask in image_map:
                paired_data_map[base_name_no_mask] = (image_map[base_name_no_mask], mask_path)

    # 3. 최종 쌍 데이터 리스트 생성 (기본 이름으로 정렬)
    all_pairs = sorted(list(paired_data_map.values()))
    
    if not all_pairs:
        print("오류: images 폴더와 masks 폴더에서 쌍을 이루는 파일을 찾을 수 없습니다.")
        return []

    print(f"✅ 총 {len(all_pairs)}쌍의 이미지/마스크 데이터를 찾았습니다.")
    return all_pairs


def split_and_save_csv(input_dir: str, out_dir: str, train_ratio: float, val_ratio: float, test_ratio: float):
    """
    파일 쌍을 찾아 지정된 비율로 분할하고 파일 목록을 CSV로 저장합니다.
    """
    all_pairs = find_paired_data(input_dir)
    if not all_pairs:
        return

    total_count = len(all_pairs)

    # 1. 데이터 분할 수행
    
    # Val과 Test의 합 비율 계산
    test_size_val = val_ratio + test_ratio
    
    # 1-1. Train과 (Val+Test) 분리
    if test_size_val == 0:
        # Val, Test 비율이 모두 0이면 모두 Train으로
        train_paths = all_pairs
        val_paths = []
        test_paths = []
    else:
        train_paths, val_test_paths = train_test_split(
            all_pairs, 
            test_size=test_size_val, 
            random_state=42, 
            shuffle=True
        )
        
        # 1-2. Val과 Test 분리
        if val_ratio == 0:
            # Val 비율이 0이면 val_test_paths 전체가 Test
            val_paths = []
            test_paths = val_test_paths
        elif test_ratio == 0:
            # Test 비율이 0이면 val_test_paths 전체가 Val
            val_paths = val_test_paths
            test_paths = []
        else:
            # Val/Test가 모두 존재하면 비율에 맞춰 분리
            relative_test_ratio = test_ratio / test_size_val
            val_paths, test_paths = train_test_split(
                val_test_paths, 
                test_size=relative_test_ratio, 
                random_state=42, 
                shuffle=True
            )

    # 2. 분할 결과 저장

    os.makedirs(out_dir, exist_ok=True)
    
    splits: Dict[str, List[Tuple[str, str]]] = {
        'train': train_paths, 
        'val': val_paths, 
        'test': test_paths
    }
    
    for split_name, data_list in splits.items():
        output_file = os.path.join(out_dir, f'{split_name}.csv')
        
        # CSV 파일에 데이터 저장
        with open(output_file, 'w', newline='', encoding='utf-8') as f:
            writer = csv.writer(f)
            
            # 헤더(Header) 추가
            writer.writerow(['image_path', 'mask_path']) 
            
            # 데이터 쓰기
            for img_path, mask_path in data_list:
                writer.writerow([img_path, mask_path])
        
        print(f"💾 {split_name.upper()} 데이터셋 {len(data_list)}쌍 CSV 저장 완료: {output_file}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="이미지 및 마스크 쌍 데이터셋을 train, val, test로 분할하고 CSV 파일로 저장합니다.")
    
    parser.add_argument(
        '--input', 
        type=str, 
        required=True, 
        help="원본 데이터의 상위 폴더 경로 (내부에 'images', 'masks' 폴더가 있어야 함)"
    )
    parser.add_argument(
        '--out', 
        type=str, 
        required=True, 
        help="분할된 파일 목록(*.csv)이 저장될 폴더 경로"
    )
    parser.add_argument(
        '--train', 
        type=parse_ratios, 
        default='0.8:0.1:0.1', 
        metavar='TRAIN:VAL:TEST_RATIO',
        help="Train, Val, Test의 비율을 지정합니다. (예: '1:1:2' 또는 '0.8:0.1:0.1')"
    )

    args = parser.parse_args()

    # 입력 비율 파싱 (parse_ratios에서 이미 정규화되어 train, val, test 순으로 반환됨)
    train_r, val_r, test_r = args.train
    
    # 입력 경로 유효성 검사
    input_valid = os.path.isdir(args.input) and \
                  os.path.isdir(os.path.join(args.input, 'images')) and \
                  os.path.isdir(os.path.join(args.input, 'masks'))
                  
    if not input_valid:
        print(f"❌ 오류: --input 경로 '{args.input}'를 찾을 수 없거나, 내부에 'images' 및 'masks' 폴더가 모두 존재하지 않습니다.")
    else:
        split_and_save_csv(args.input, args.out, train_r, val_r, test_r)