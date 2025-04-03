import os
import cv2
import torch
import mediapipe as mp
from tqdm import tqdm

# ✅ 설정
base_dir = r"C:\Users\User\Desktop\Lips_Landmark"
merged_folder = os.path.join(base_dir, "merged_images")
cropped_lips_dir = os.path.join(base_dir, "cropped_lips")
os.makedirs(cropped_lips_dir, exist_ok=True)

output_pt_file = os.path.join(base_dir, "lips_3d_landmarks_new2.pt")

# ✅ Mediapipe FaceMesh 설정
mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh(static_image_mode=True, max_num_faces=1, refine_landmarks=True)

# ✅ 입술 인덱스 (Mediapipe 기준)
LIPS_LANDMARKS = [
    61, 146, 91, 181, 84, 17, 314, 405, 321, 375,  # Upper outer
    78, 191, 80, 81, 82, 13, 312, 311, 310, 415,  # Lower outer
    95, 88, 178, 87, 14, 317, 402, 318, 324, 308,  # Upper inner
    185, 40, 39, 37, 0, 267, 269, 270, 409, 291   # Lower inner
]

# ✅ 상수 설정
TARGET_SIZE = 256
MARGIN = 25

# ✅ 기존 landmark 데이터 불러오기
if os.path.exists(output_pt_file):
    landmark_data = torch.load(output_pt_file)
    print(f"📥 기존 데이터 {len(landmark_data)}개 로드됨")
else:
    landmark_data = []
    print("⚠️ 기존 .pt 파일 없음, 새로 생성 시작")

processed_count = len(landmark_data)

# ✅ merged_images 폴더 처리
image_files = [f for f in os.listdir(merged_folder) if f.lower().endswith(('.jpg', '.png'))]

for image_file in tqdm(image_files, desc="Processing merged_images"):
    image_path = os.path.join(merged_folder, image_file)

    image = cv2.imread(image_path)
    if image is None:
        continue

    h, w, _ = image.shape
    rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    results = face_mesh.process(rgb_image)
    if not results.multi_face_landmarks:
        continue

    face_landmarks = results.multi_face_landmarks[0]
    lips_points = [(int(face_landmarks.landmark[idx].x * w),
                    int(face_landmarks.landmark[idx].y * h),
                    face_landmarks.landmark[idx].z) for idx in LIPS_LANDMARKS]

    min_x = max(min(p[0] for p in lips_points) - MARGIN, 0)
    max_x = min(max(p[0] for p in lips_points) + MARGIN, w)
    min_y = max(min(p[1] for p in lips_points) - MARGIN, 0)
    max_y = min(max(p[1] for p in lips_points) + MARGIN, h)

    cropped_w, cropped_h = max_x - min_x, max_y - min_y
    if cropped_w < 20 or cropped_h < 20:
        continue

    cropped_lips = image[min_y:max_y, min_x:max_x]
    cropped_lips_resized = cv2.resize(cropped_lips, (TARGET_SIZE, TARGET_SIZE))

    # 저장 파일명
    cropped_image_name = f"{processed_count + 1:06d}.jpg"
    cropped_image_path = os.path.join(cropped_lips_dir, cropped_image_name)
    cv2.imwrite(cropped_image_path, cropped_lips_resized)

    # 랜드마크 정규화
    normalized_landmarks = [
        (
            round((p[0] - min_x) / cropped_w * TARGET_SIZE, 6),
            round((p[1] - min_y) / cropped_h * TARGET_SIZE, 6),
            round(p[2], 6)
        )
        for p in lips_points
    ]

    # 데이터 저장
    landmark_data.append((cropped_image_path, normalized_landmarks))
    processed_count += 1

# ✅ 결과 저장
torch.save(landmark_data, output_pt_file)
print(f"✅ 총 {len(landmark_data)}개의 데이터가 저장되었습니다 → {output_pt_file}")
