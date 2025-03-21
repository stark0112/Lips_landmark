import torch
import cv2
import numpy as np
import albumentations as A
import os
from tqdm import tqdm
import random

# ✅ 기존 데이터 로드
data_path = "merged_data_celeba2.pt"
aug_data_path = "merged_data_celeba2_aug.pt"
data = torch.load(data_path)

# ✅ BlazeFace 증강 방식 적용
transformations = {
    "flip": A.HorizontalFlip(p=0.5),  
    "rotate": A.ShiftScaleRotate(shift_limit=0, scale_limit=0, rotate_limit=15, p=0.5),  
    "scale": A.RandomScale(scale_limit=0.1, p=0.5),  
    "brightness": A.RandomBrightnessContrast(brightness_limit=0.1, contrast_limit=0, p=0.5),  
    "shift": A.ShiftScaleRotate(shift_limit=0.05, scale_limit=0, rotate_limit=0, p=0.5),  
    "blur": A.GaussianBlur(blur_limit=3, p=0.1),  
}

save_dir = r"C:\Users\User\Desktop\Lips_Landmark\archive\Part 22\Part 22"
os.makedirs(save_dir, exist_ok=True)

# ✅ 새로운 데이터 저장 리스트 (원본 데이터 유지)
augmented_data = list(data)

# ✅ 마지막 파일 번호 찾기
existing_ids = [int(os.path.basename(img_path).split(".")[0]) for img_path, _, _ in data]
last_id = max(existing_ids)

# ✅ 증강 적용 (BlazeFace 설정)
for i, (image_path, bbox, landmarks) in enumerate(tqdm(data, total=len(data))):
    image = cv2.imread(image_path)
    if image is None:
        print(f"❌ 이미지 로드 실패: {image_path}")
        continue

    h, w, _ = image.shape
    x1, y1, x2, y2 = bbox  
    # 🔹 **BBox 변환: Pascal VOC 형식 유지** ✅ 픽셀 단위
    bbox_alb = [float(x1), float(y1), float(x2), float(y2)]
    
    # 🔹 **랜드마크 변환: 정규화 해제**
    landmarks_pixel_array = np.array(landmarks, dtype=np.float32).reshape(-1, 2) * [w, h]

    # ✅ **리스트 변환 (numpy → list of tuples)**
    landmarks_pixel_list = [tuple(pt) for pt in landmarks_pixel_array.tolist()]

    # 랜덤하게 적용할 증강 선택 (BlazeFace 확률 기준)
    applied_transforms = [t for t in transformations.values() if random.random() < t.p]

    if applied_transforms:
        last_id += 1
        new_image_id = f"{last_id:06d}.jpg"
        new_image_path = os.path.join(save_dir, new_image_id)

        # ✅ **증강 파이프라인 생성**
        transform = A.Compose(
            applied_transforms,
            bbox_params=A.BboxParams(format="pascal_voc", label_fields=[]),  # remove_invisible=False 삭제
            keypoint_params=A.KeypointParams(format="xy")
        )

        try:
            # ✅ **증강 적용**
            aug = transform(image=image, bboxes=[bbox_alb], keypoints=landmarks_pixel_list)

            # ✅ **변환된 값 가져오기**
            aug_image = aug["image"]
            aug_bbox = list(map(int, aug["bboxes"][0]))  
            aug_landmarks_pixel = np.array(aug["keypoints"], dtype=np.float32)

            # ✅ **증강된 이미지 크기 확인**
            h_new, w_new, _ = aug_image.shape  
            # ✅ **랜드마크를 다시 정규화 (0~1 범위)**
            aug_landmarks = (aug_landmarks_pixel / [w_new, h_new]).flatten().tolist()

            # ✅ **증강된 이미지 저장**
            cv2.imwrite(new_image_path, aug_image)

        except Exception as e:
            print(f"❌ 증강 중 오류 발생: {image_path}, 오류: {e}")
            continue  # 🔥 오류 발생 시 저장 X

        # ✅ **오류 없이 성공한 경우에만 저장**
        augmented_data.append((new_image_path, aug_bbox, aug_landmarks))
        print(last_id)
        # ✅ **1000개 단위로 진행 상황 확인**
        if i % 500 == 0:
            print(f"✅ {i}번째 증강 데이터 저장 완료: {new_image_path}")

# ✅ **증강된 데이터 저장**
torch.save(augmented_data, aug_data_path)
print(f"✅ 증강된 데이터 저장 완료: {aug_data_path}")
