import torch
import cv2
import matplotlib.pyplot as plt
import os

# ✅ 증강된 데이터 로드
aug_data_path = "merged_data_celeba2_aug.pt"
augmented_data = torch.load(aug_data_path)

# ✅ 확인하고 싶은 이미지 파일명 설정 (예: "252000.jpg")
search_filename = "378007.jpg"  # 원하는 이미지 파일명을 입력하세요

# ✅ 특정 이미지 검색
matched_samples = [data for data in augmented_data if search_filename in os.path.basename(data[0])]

if not matched_samples:
    print(f"❌ '{search_filename}'을 포함하는 이미지가 데이터셋에 없습니다.")
else:
    for image_path, bbox, landmarks in matched_samples:
        # ✅ 이미지 로드
        image = cv2.imread(image_path)
        if image is None:
            print(f"❌ 이미지 로드 실패: {image_path}")
            continue

        h, w, _ = image.shape
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        # ✅ Bounding Box (BBox) 그리기 (초록색)
        x1, y1, x2, y2 = map(int, bbox)

        # 🔹 이미지 범위 초과 방지
        x1, y1, x2, y2 = max(0, x1), max(0, y1), min(w - 1, x2), min(h - 1, y2)

        cv2.rectangle(image_rgb, (x1, y1), (x2, y2), (0, 255, 0), 2)
        print(f"📌 Bounding Box: ({x1}, {y1}) → ({x2}, {y2})")

        # ✅ 랜드마크 그리기 (빨간색)
        for i in range(0, len(landmarks), 2):
            lm_x = int(landmarks[i] * w)  # 정규화된 값 → 픽셀 값 변환
            lm_y = int(landmarks[i + 1] * h)
            lm_x, lm_y = max(0, min(w - 1, lm_x)), max(0, min(h - 1, lm_y))  # 범위 초과 방지
            cv2.circle(image_rgb, (lm_x, lm_y), 3, (255, 0, 0), -1)

        # ✅ 결과 시각화
        plt.figure(figsize=(6, 6))
        plt.imshow(image_rgb)
        plt.axis("off")
        plt.title(f"📷 {os.path.basename(image_path)}")
        plt.show()
