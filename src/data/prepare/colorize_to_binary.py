import os
import argparse
import glob
from PIL import Image
import numpy as np

def convert_to_binary(input_dir, out_dir, threshold, output_value):
    """
    지정된 폴더의 모든 이미지 파일을 이진화하여 0 또는 output_value(기본값 255)로 변환하여 저장합니다.
    """
    # 1. 지원하는 이미지 확장자 목록
    IMAGE_EXTENSIONS = ['*.jpg', '*.jpeg', '*.png', '*.tiff', '*.tif', '*.gif', '*.bmp', '*.webp']

    # 2. 입력 폴더 내 모든 파일 경로 수집
    mask_paths = []
    for ext in IMAGE_EXTENSIONS:
        # 서브폴더는 포함하지 않고, 지정된 디렉토리의 파일만 찾습니다.
        mask_paths.extend(glob.glob(os.path.join(input_dir, ext)))

    if not mask_paths:
        print(f"경고: 입력 폴더 '{input_dir}'에서 변환할 이미지 파일을 찾을 수 없습니다.")
        return

    # 3. 출력 디렉토리 생성
    os.makedirs(out_dir, exist_ok=True)
    print(f"출력 폴더 생성: {out_dir}")

    # 4. 파일 처리 및 변환
    print(f"총 {len(mask_paths)}개의 파일을 변환합니다. (Threshold: {threshold}, Output Value: {output_value})")
    
    for i, mask_path in enumerate(mask_paths):
        file_name = os.path.basename(mask_path)
        out_path = os.path.join(out_dir, file_name)

        try:
            # 1. 이미지 로드
            img = Image.open(mask_path).convert('L')  # 이미지를 8비트 흑백(Grayscale)으로 변환
            img_array = np.array(img)

            # 2. 임계값 기준으로 이진화 수행
            # 임계값보다 크거나 같으면 True (1), 작으면 False (0)
            binary_array = (img_array >= threshold)

            # 3. 원하는 출력 값(0 또는 output_value)으로 변환
            # True(1)는 output_value로, False(0)는 0으로 변환
            final_array = binary_array.astype(np.uint8) * output_value
            
            # 4. NumPy 배열을 다시 PIL 이미지로 변환
            binary_img = Image.fromarray(final_array)
            
            # 5. PNG로 저장 (PNG는 바이너리 이미지 저장에 적합)
            # 원본 파일명과 동일하게 저장하며, PNG 확장자로 저장됨
            # Note: 원본이 JPG일 경우에도 PNG로 저장됩니다.
            binary_img.save(out_path, 'PNG')
            
            print(f"[{i+1}/{len(mask_paths)}]: '{file_name}' 변환 완료.")
            
        except Exception as e:
            print(f"오류: 파일 '{file_name}' 처리 중 오류 발생: {e}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="컬러 마스크 파일을 바이너리(0 또는 255) 마스크로 변환합니다.")
    
    parser.add_argument(
        '--input', 
        type=str, 
        required=True, 
        help="원본 마스크 파일들이 있는 폴더 경로 (예: './data/masks')"
    )
    parser.add_argument(
        '--out', 
        type=str, 
        required=True, 
        help="변환된 바이너리 파일이 저장될 폴더 경로 (예: './output/binary_masks')"
    )
    parser.add_argument(
        '--threshold', 
        type=int, 
        default=128, 
        help="이진화를 위한 임계값 (0-255). 이 값보다 크거나 같은 픽셀은 배경이 아닌 영역으로 간주됩니다. (기본값: 128)"
    )
    parser.add_argument(
        '--option', 
        type=int, 
        default=255, 
        choices=[1, 255],
        help="배경이 아닌 영역을 표시할 값. 0과 이 값으로만 구성됩니다. (선택: 1 또는 255, 기본값: 255)"
    )

    args = parser.parse_args()

    # 입력 경로 유효성 확인
    if not os.path.isdir(args.input):
        print(f"오류: --input 경로 '{args.input}'를 찾을 수 없습니다.")
    else:
        convert_to_binary(args.input, args.out, args.threshold, args.option)