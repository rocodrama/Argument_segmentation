import os
import argparse
import glob
from PIL import Image

def rename_and_save_image_pairs(input_images_dir, input_masks_dir, out_dir):
    """
    images 및 masks 폴더의 파일 쌍을 찾아 번호를 매겨 지정된 출력 폴더에 PNG로 저장합니다.
    """
    # 1. 지원하는 모든 이미지 확장자 목록
    # Pillow (PIL)가 지원하는 대부분의 형식이나, 필요에 따라 더 추가/제거 가능
    IMAGE_EXTENSIONS = ['*.jpg', '*.jpeg', '*.png', '*.tiff', '*.tif', '*.gif', '*.bmp', '*.webp']

    # 2. images 폴더 내 모든 파일 경로 수집
    image_paths = []
    for ext in IMAGE_EXTENSIONS:
        image_paths.extend(glob.glob(os.path.join(input_images_dir, ext)))

    # 3. 파일 이름 쌍을 위한 딕셔너리 생성
    # Key: 파일 기본 이름 (확장자 제외)
    # Value: images 경로
    image_map = {}
    for img_path in image_paths:
        # 확장자 없는 파일 이름 (예: 'asd')
        base_name = os.path.splitext(os.path.basename(img_path))[0]
        image_map[base_name] = img_path

    # 4. masks 폴더 내 모든 파일 경로 수집
    mask_paths = []
    for ext in IMAGE_EXTENSIONS:
        mask_paths.extend(glob.glob(os.path.join(input_masks_dir, ext)))

    # 5. 마스크 파일의 기본 이름과 쌍을 찾습니다.
    # Key: 이미지 파일 기본 이름 (예: 'asd')
    # Value: 마스크 파일 경로
    mask_map = {}
    for mask_path in mask_paths:
        mask_base_name = os.path.splitext(os.path.basename(mask_path))[0]

        # 5-1. Case 1: 마스크 이름이 이미지 이름과 동일 (예: 'asd.png')
        if mask_base_name in image_map:
            mask_map[mask_base_name] = mask_path
        
        # 5-2. Case 2: 마스크 이름에 '_mask'가 포함된 경우 (예: 'asd_mask.png')
        # '_mask' 부분을 제거한 이름이 images에 있는지 확인
        if mask_base_name.endswith('_mask'):
            potential_image_name = mask_base_name[:-len('_mask')]
            if potential_image_name in image_map:
                mask_map[potential_image_name] = mask_path
    
    # 6. 최종 쌍 리스트 생성 및 정렬
    # image_map과 mask_map 모두에 존재하는 기본 이름만 사용
    # 정렬하여 출력 일련번호가 일관되게 나오도록 함
    paired_base_names = sorted(list(set(image_map.keys()) & set(mask_map.keys())))

    if not paired_base_names:
        print("경고: images 폴더와 masks 폴더에서 쌍을 이루는 파일을 찾을 수 없습니다.")
        return

    # 7. 출력 디렉토리 구조 생성
    out_images_dir = os.path.join(out_dir, 'images')
    out_masks_dir = os.path.join(out_dir, 'masks')
    os.makedirs(out_images_dir, exist_ok=True)
    os.makedirs(out_masks_dir, exist_ok=True)
    print(f"출력 폴더 생성: {out_images_dir} 및 {out_masks_dir}")

    # 8. 파일 저장 및 이름 변경
    for i, base_name in enumerate(paired_base_names, 1):
        image_src = image_map[base_name]
        mask_src = mask_map[base_name]
        
        # 새 파일 이름: 1, 2, 3, ... (PNG로 강제 변환)
        new_name = f"{i}.png"
        
        image_dest = os.path.join(out_images_dir, new_name)
        mask_dest = os.path.join(out_masks_dir, new_name)
        
        # 이미지 파일 로드 및 PNG로 저장
        try:
            # 이미지 저장
            Image.open(image_src).save(image_dest, 'PNG')
            
            # 마스크 파일 로드 및 PNG로 저장
            Image.open(mask_src).save(mask_dest, 'PNG')
            
            print(f"[{i}/{len(paired_base_names)}]: '{base_name}' 쌍 저장 완료. -> '{new_name}'")
        except Exception as e:
            print(f"오류: 파일 '{base_name}' 처리 중 오류 발생: {e}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="이미지 및 마스크 쌍의 이름을 일련번호로 변경하고 PNG로 저장합니다.")
    parser.add_argument(
        '--input-images', 
        type=str, 
        required=True, 
        help="원본 이미지 파일이 있는 폴더 경로 (예: './data/images')"
    )
    parser.add_argument(
        '--input-masks', 
        type=str, 
        required=True, 
        help="원본 마스크 파일이 있는 폴더 경로 (예: './data/masks')"
    )
    parser.add_argument(
        '--out_dir', 
        type=str, 
        required=True, 
        help="결과 파일(images/ 및 masks/)이 저장될 상위 폴더 경로"
    )

    args = parser.parse_args()

    # 입력 경로가 유효한지 확인
    if not os.path.isdir(args.input_images):
        print(f"오류: --input-images 경로 '{args.input_images}'를 찾을 수 없습니다.")
    elif not os.path.isdir(args.input_masks):
        print(f"오류: --input-masks 경로 '{args.input_masks}'를 찾을 수 없습니다.")
    else:
        rename_and_save_image_pairs(args.input_images, args.input_masks, args.out_dir)