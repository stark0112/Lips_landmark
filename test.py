import os
import cv2
import numpy as np
import pandas as pd
import random
import matplotlib.pyplot as plt

# ✅ CelebA 데이터 경로 설정
CELEBA_IMAGE_ROOT = r"C:\Users\User\Desktop\Lips_Landmark\archive"
CELEBA_ANNOTATION_FILE = r"C:\Users\User\Desktop\Lips_Landmark\archive\list_bbox_celeba.csv"

# ✅ CelebA 이미지 파일 찾기 (중첩 폴더 문제 해결)
def find_celeba_images(root_dir):
    image_paths = {}
    for part in os.listdir(root_dir):
        part_path = os.path.join(root_dir, part)
        if os.path.isdir(part_path):
            for subdir in os.listdir(part_path):
                subdir_path = os.path.join(part_path, subdir)
                if os.path.isdir(subdir_path):
                    for img in os.listdir(subdir_path):
                        if img.endswith(".jpg"):
                            image_paths[img] = os.path.join(subdir_path, img)
    return image_paths

celeba_image_paths = find_celeba_images(CELEBA_IMAGE_ROOT)

# ✅ CelebA 바운딩 박스 불러오기
def load_celeba_annotations(annotation_file, image_paths):
    df = pd.read_csv(annotation_file)
    annotations = {}
    for _, row in df.iterrows():
        image_name, x1, y1, width, height = row["image_id"], row["x_1"], row["y_1"], row["width"], row["height"]
        x2 = x1 + width
        y2 = y1 + height
        if image_name in image_paths:
            annotations[image_name] = (image_paths[image_name], x1, y1, x2, y2)
    return annotations

# ✅ 데이터 불러오기
celeba_data = load_celeba_annotations(CELEBA_ANNOTATION_FILE, celeba_image_paths)
print(f"✅ CelebA 데이터 수: {len(celeba_data)}")

# ✅ 이미지 + 바운딩 박스 시각화 함수
def visualize_celeba_samples(data, num_samples=5):
    sample_images = random.sample(list(data.keys()), num_samples)  # 랜덤 샘플 선택

    fig, axes = plt.subplots(1, num_samples, figsize=(15, 5))

    for ax, image_name in zip(axes, sample_images):
        image_path, x1, y1, x2, y2 = data[image_name]
        image = cv2.imread(image_path)
        if image is None:
            continue
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        # 바운딩 박스 그리기
        ax.add_patch(plt.Rectangle((x1, y1), x2 - x1, y2 - y1, fill=False, color='red', linewidth=2))

        ax.imshow(image)
        ax.axis("off")

    plt.show()

# ✅ 랜덤 이미지 5개 확인 (CelebA 데이터셋)
visualize_celeba_samples(celeba_data, num_samples=5)
