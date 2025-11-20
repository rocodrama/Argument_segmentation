import os
import argparse
import glob
from PIL import Image

def resize_images(input_dir, out_dir, target_x, target_y):
    """
    입력 폴더의 모든 이미지를 target_x * target_y 해상도로 리사이즈하여 
    출력 폴더에 원본 폴더 구조를 유지하며 저장합니다.
    """
    # 1. 지원하는 이미지 확장자 목록 (PIL이 지원하는 대부분의 형식)
    IMAGE_EXTENSIONS = ('*.jpg', '*.jpeg', '*.png', '*.tiff', '*.tif', '*.gif', '*.bmp', '*.webp')
    
    # 목표 해상도 튜플
    target_size = (target_x, target_y)
    print(f"이미지를 {target_size[0]}x{target_size[1]} 해상도로 리사이즈합니다.")

    # 2. 출력 디렉토리 생성 (필요한 경우)
    os.makedirs(out_dir, exist_ok=True)
    
    # 3. 입력 디렉토리를 재귀적으로 탐색
    # os.walk를 사용하여 모든 하위 디렉토리와 파일을 순회합니다.
    processed_count = 0
    
    for root, _, files in os.walk(input_dir):
        # 현재 처리 중인 폴더: 입력 폴더를 기준으로 상대 경로 계산
        relative_path = os.path.relpath(root, input_dir)
        
        # 출력 폴더의 대응되는 경로 설정
        output_sub_dir = os.path.join(out_dir, relative_path)
        
        # 출력 하위 폴더 생성
        os.makedirs(output_sub_dir, exist_ok=True)

        for file_name in files:
            # 파일 확장자를 확인하여 이미지 파일인지 검사
            if file_name.lower().endswith(tuple(ext.lstrip('*') for ext in IMAGE_EXTENSIONS)):
                input_path = os.path.join(root, file_name)
                output_path = os.path.join(output_sub_dir, file_name)
                
                try:
                    # 1. 이미지 로드
                    img = Image.open(input_path)
                    
                    # 2. 이미지 리사이즈
                    # Image.LANCZOS는 고품질 리샘플링 필터입니다.
                    resized_img = img.resize(target_size, Image.Resampling.LANCZOS)
                    
                    # 3. 리사이즈된 이미지 저장
                    resized_img.save(output_path)
                    
                    processed_count += 1
                    print(f"처리 완료: {input_path} -> {output_path} ({img.size} -> {resized_img.size})")

                except Exception as e:
                    print(f"오류: 파일 '{input_path}' 처리 중 오류 발생: {e}")

    if processed_count == 0:
        print("경고: 지정된 폴더에서 처리된 이미지 파일을 찾지 못했습니다.")
    else:
        print(f"\n✅ 이미지 리사이즈 작업이 완료되었습니다. 총 {processed_count}개의 파일을 처리했습니다.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="입력 폴더의 모든 이미지 파일을 원하는 해상도로 리사이즈하여 폴더 구조를 유지하며 저장합니다.")
    
    parser.add_argument(
        '--input', 
        type=str, 
        required=True, 
        help="원본 이미지 파일들이 있는 상위 폴더 경로 (예: './data/high_res')"
    )
    parser.add_argument(
        '--out', 
        type=str, 
        required=True, 
        help="리사이즈된 파일이 저장될 상위 폴더 경로 (예: './output/low_res')"
    )
    parser.add_argument(
        '--x', 
        type=int, 
        required=True, 
        help="리사이즈할 목표 해상도의 가로 길이 (픽셀)"
    )
    parser.add_argument(
        '--y', 
        type=int, 
        required=True, 
        help="리사이즈할 목표 해상도의 세로 길이 (픽셀)"
    )

    args = parser.parse_args()

    # 입력 경로 유효성 확인
    if not os.path.isdir(args.input):
        print(f"오류: --input 경로 '{args.input}'를 찾을 수 없습니다.")
    elif args.x <= 0 or args.y <= 0:
        print("오류: 가로(--x) 및 세로(--y) 해상도는 0보다 큰 값이어야 합니다.")
    else:
        resize_images(args.input, args.out, args.x, args.y)