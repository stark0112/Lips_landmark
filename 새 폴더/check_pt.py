import torch

# ✅ 데이터 로드
aug_data_path = "merged_data_celeba2_aug.pt"
augmented_data = torch.load(aug_data_path)

# ✅ 데이터셋 개수 출력
print(f"✅ 데이터셋 개수: {len(augmented_data)}개")

# ✅ 데이터 형태 확인 (첫 번째 샘플)
first_sample = augmented_data[0]
print("\n✅ 데이터 구조 예시:")
print(f"  ├── 이미지 경로: {first_sample[0]}")
print(f"  ├── Bounding Box: {first_sample[1]}")  # 바운딩 박스 좌표
print(f"  └── 랜드마크 (정규화됨): {first_sample[2]}")  # 랜드마크 좌표 (정규화됨)

# ✅ 데이터셋 일부 샘플 출력 (마지막 5개)
print("\n✅ 마지막 5개 샘플:")
for image_path, bbox, landmarks in augmented_data[-100:]:
    print(f"📷 이미지 경로: {image_path}")
    print(f"   ├── Bounding Box: {bbox}")
    print(f"   └── 랜드마크: {landmarks[:6]} ... (총 {len(landmarks)}개)")
    print("-" * 80)
