import os
import argparse
import numpy as np
import pandas as pd
from PIL import Image
from tqdm import tqdm

def convert_images_to_npy(input_dir: str, index_file: str, out_dir: str):
    """
    CSV 인덱스 파일을 읽어 이미지와 마스크 파일을 NumPy 배열로 변환하여 저장합니다.
    """
    
    # 1. 경로 설정 및 유효성 검사
    if not os.path.isdir(input_dir):
        print(f"❌ 오류: 입력 폴더 '{input_dir}'를 찾을 수 없습니다.")
        return
    if not os.path.isfile(index_file):
        print(f"❌ 오류: 인덱스 파일 '{index_file}'을 찾을 수 없습니다.")
        return

    os.makedirs(out_dir, exist_ok=True)
    
    # 2. 인덱스 파일(CSV) 로드
    try:
        df = pd.read_csv(index_file)
    except Exception as e:
        print(f"❌ 오류: CSV 파일 '{index_file}'을 읽는 데 실패했습니다. 오류: {e}")
        return

    if 'image_path' not in df.columns or 'mask_path' not in df.columns:
        print("❌ 오류: CSV 파일에 'image_path' 또는 'mask_path' 컬럼이 없습니다.")
        return

    # 3. 데이터 로드 및 변환 준비
    
    # NumPy 배열을 저장할 리스트 초기화
    image_list = []
    mask_list = []
    
    # 로드 및 변환할 파일 수
    total_files = len(df)
    
    print(f"✅ 인덱스 파일에서 총 {total_files}쌍의 파일 경로를 로드했습니다.")
    
    # 4. 파일 반복 및 변환
    # tqdm을 사용하여 진행 상황을 표시합니다.
    for index, row in tqdm(df.iterrows(), total=total_files, desc="Converting to NumPy"):
        image_relative_path = row['image_path']
        mask_relative_path = row['mask_path']
        
        # 파일 경로를 절대 경로로 조합 (CSV 파일에 절대 경로가 저장되어 있다고 가정)
        # 만약 CSV에 상대 경로가 저장되어 있다면, os.path.join(input_dir, image_relative_path)와 같이 수정 필요
        img_path = image_relative_path
        mask_path = mask_relative_path
        
        try:
            # 4-1. 이미지 파일 로드 및 전처리
            img = Image.open(img_path).convert('RGB') # RGB로 강제 변환 (3채널)
            img_array = np.array(img, dtype=np.float32) 
            
            # (선택적) 정규화: 0-255 -> 0-1
            img_array /= 255.0 
            
            # 4-2. 마스크 파일 로드 및 전처리
            mask = Image.open(mask_path).convert('L') # Grayscale (흑백)로 변환 (1채널)
            mask_array = np.array(mask, dtype=np.uint8)
            
            # 마스크 이진화 (0 또는 1/255로 간주. 여기서는 0, 1로 변환한다고 가정)
            # 마스크 파일을 0 또는 1로 변환하는 일반적인 방법 (0이 아닌 모든 픽셀을 1로):
            mask_array = (mask_array > 0).astype(np.uint8)
            
            image_list.append(img_array)
            mask_list.append(mask_array)
            
        except FileNotFoundError:
            print(f"\n⚠️ 경고: 파일을 찾을 수 없습니다. 이미지: {img_path} 또는 마스크: {mask_path}")
        except Exception as e:
            print(f"\n❌ 오류: 파일 처리 중 오류 발생 - {e} (이미지: {img_path})")

    if not image_list:
        print("경고: 변환된 파일이 없습니다. 경로를 확인해주세요.")
        return
        
    # 5. 최종 NumPy 배열 생성
    X_data = np.stack(image_list, axis=0) # 이미지 데이터 (Features)
    Y_data = np.stack(mask_list, axis=0) # 마스크 데이터 (Labels)
    
    # 6. NumPy 파일 저장
    
    # 파일 이름은 인덱스 파일 이름에서 확장자를 제거하고 따옵니다. (예: train.csv -> train)
    base_name = os.path.splitext(os.path.basename(index_file))[0]
    
    x_out_path = os.path.join(out_dir, f'X_{base_name}.npy')
    y_out_path = os.path.join(out_dir, f'Y_{base_name}.npy')

    np.save(x_out_path, X_data)
    np.save(y_out_path, Y_data)

    print("\n" + "="*50)
    print(f"🎉 NumPy 변환 완료!")
    print(f"   X (이미지) 저장 경로: {x_out_path}, Shape: {X_data.shape}")
    print(f"   Y (마스크) 저장 경로: {y_out_path}, Shape: {Y_data.shape}")
    print("="*50)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="CSV 인덱스 파일을 기반으로 이미지를 NumPy 배열로 변환합니다.")
    
    parser.add_argument(
        '--input', 
        type=str, 
        required=True, 
        help="원본 데이터의 상위 폴더 경로 (현재 스크립트에서는 CSV에 절대 경로가 있다고 가정하고 사용되지 않음)"
    )
    parser.add_argument(
        '--index', 
        type=str, 
        required=True, 
        help="분할된 데이터셋 목록이 포함된 CSV 파일 경로 (예: './splits/train.csv')"
    )
    parser.add_argument(
        '--out', 
        type=str, 
        required=True, 
        help="NumPy 배열 파일이 저장될 폴더 경로 (예: './npy_data')"
    )

    args = parser.parse_args()

    convert_images_to_npy(args.input, args.index, args.out)