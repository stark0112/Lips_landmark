import torch
import cv2
import mediapipe as mp

# ✅ Mediapipe 초기화
mp_face_detection = mp.solutions.face_detection
mp_face_mesh = mp.solutions.face_mesh

# ✅ 기존 데이터 로드
old_data_path = "merged_data_celeba.pt"  # 기존 데이터 파일 경로
new_data_path = "merged_data_celeba2.pt"  # 새로 저장할 파일 경로
data = torch.load(old_data_path, map_location="cpu")

# ✅ 새로운 데이터 저장용 리스트
new_data = []

# ✅ BlazeFace & FaceMesh 모델 초기화
with mp_face_detection.FaceDetection(model_selection=1, min_detection_confidence=0.5) as face_detection, \
     mp_face_mesh.FaceMesh(static_image_mode=True, max_num_faces=1, refine_landmarks=True, min_detection_confidence=0.5) as face_mesh:

    # ✅ 기존 데이터 순회
    for i, (image_path, _, _) in enumerate(data):
        image = cv2.imread(image_path)
        if image is None:
            print(f"❌ 이미지 로드 실패: {image_path}")
            continue

        h, w, _ = image.shape
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)  # OpenCV는 BGR, Mediapipe는 RGB 사용

        # 🔹 1️⃣ BlazeFace를 사용하여 BBox 추출
        face_results = face_detection.process(image_rgb)
        if not face_results.detections:
            print(f"❌ 얼굴 검출 실패: {image_path}")
            continue

        detection = face_results.detections[0]  # 첫 번째 얼굴만 사용
        bboxC = detection.location_data.relative_bounding_box
        x1 = int(bboxC.xmin * w)
        y1 = int(bboxC.ymin * h)
        x2 = x1 + int(bboxC.width * w)
        y2 = y1 + int(bboxC.height * h)
        bbox = [x1, y1, x2, y2]  # ✅ (x1, y1, x2, y2) 픽셀 단위 유지

        # 🔹 2️⃣ FaceMesh를 사용하여 랜드마크 추출
        face_mesh_results = face_mesh.process(image_rgb)
        if not face_mesh_results.multi_face_landmarks:
            print(f"❌ 랜드마크 검출 실패: {image_path}")
            continue

        face_landmarks = face_mesh_results.multi_face_landmarks[0]

        # 🔹 3️⃣ 새로운 랜드마크 구성 (7개 포인트, 0~1 정규화된 값)
        landmarks = []
        key_indices = {
            "오른쪽 위 눈꺼풀": 159,
            "오른쪽 아래 눈꺼풀": 145,
            "왼쪽 위 눈꺼풀": 386,
            "왼쪽 아래 눈꺼풀": 374,
            "코": 1,
            "오른쪽 입꼬리": 61,
            "왼쪽 입꼬리": 291,
            "윗입술": 0,
            "아랫입술": 17,
        }

        # 🔹 오른쪽 눈 중앙 계산 (정규화된 값)
        right_eye_x = (face_landmarks.landmark[159].x + face_landmarks.landmark[145].x) / 2
        right_eye_y = (face_landmarks.landmark[159].y + face_landmarks.landmark[145].y) / 2
        landmarks.extend([right_eye_x, right_eye_y])

        # 🔹 왼쪽 눈 중앙 계산 (정규화된 값)
        left_eye_x = (face_landmarks.landmark[386].x + face_landmarks.landmark[374].x) / 2
        left_eye_y = (face_landmarks.landmark[386].y + face_landmarks.landmark[374].y) / 2
        landmarks.extend([left_eye_x, left_eye_y])

        # 🔹 나머지 랜드마크 추가 (정규화된 값)
        for key in ["코", "오른쪽 입꼬리", "왼쪽 입꼬리", "윗입술", "아랫입술"]:
            idx = key_indices[key]
            x, y = face_landmarks.landmark[idx].x, face_landmarks.landmark[idx].y
            landmarks.extend([x, y])

        # 🔹 4️⃣ 새로운 데이터 저장 (✅ 기존 형식 유지)
        new_data.append((image_path, bbox, landmarks))

        # 🔹 진행 상황 출력
        if i % 500 == 0:
            print(f"🔄 진행 중: {i}/{len(data)}")
       
# ✅ 최종 데이터 저장
torch.save(new_data, new_data_path)
print(f"✅ 새로운 데이터 저장 완료: {new_data_path}")
