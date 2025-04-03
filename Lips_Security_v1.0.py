import cv2
import mediapipe as mp
import numpy as np
import time
from fastdtw import fastdtw
from scipy.spatial.distance import euclidean

# --- 입술, 기준점 인덱스 설정 (MediaPipe FaceMesh 기준) ---
LIPS_IDX_40 = [
    61, 185, 40, 39, 37, 0, 267, 269, 270, 409,
    291, 375, 321, 405, 314, 17, 84, 181, 91, 146,
    78, 95, 88, 178, 87, 14, 317, 402, 318, 324,
    308, 415, 310, 311, 312, 13, 82, 81, 80, 191
]
EYE_LEFT_IDX = 33      # 왼쪽 눈
EYE_RIGHT_IDX = 263    # 오른쪽 눈
NOSE_TIP_IDX = 1       # 코 끝

# --- 입술 3D 정규화 함수 ---
def normalize_lips_3d(points, ref_points):
    center = np.mean(points, axis=0)   # 중심 정렬
    points -= center

    # 좌표계 정의 (x축: 입꼬리, y축: 입 위아래, z축: 수직 방향)
    x_axis = points[10] - points[0]
    x_axis /= np.linalg.norm(x_axis)
    y_axis = points[29] - points[19]
    y_axis /= np.linalg.norm(y_axis)
    z_axis = np.cross(x_axis, y_axis)
    z_axis /= np.linalg.norm(z_axis)
    y_axis = np.cross(z_axis, x_axis)
    y_axis /= np.linalg.norm(y_axis)
    y_axis *= -1

    # 회전 정렬
    R = np.stack([x_axis, y_axis, z_axis], axis=1)
    points = points @ R

    # 눈-코 사이 거리로 스케일 정규화
    eye_left = ref_points[EYE_LEFT_IDX]
    eye_right = ref_points[EYE_RIGHT_IDX]
    nose = ref_points[NOSE_TIP_IDX]
    scale = (np.linalg.norm(eye_left - eye_right) + np.linalg.norm(nose - (eye_left + eye_right)/2)) / 2
    if scale > 0:
        points /= scale
    return points

# --- 발화 구간 추출 (움직임이 큰 구간만 추출) ---
def extract_active_window(sequence, window_size=15):
    sequence = np.array(sequence)
    diffs = np.linalg.norm(np.diff(sequence, axis=0), axis=1)
    scores = np.convolve(diffs, np.ones(window_size), mode='valid')
    start_idx = np.argmax(scores)
    end_idx = start_idx + window_size
    return sequence[start_idx:end_idx]

# --- 일정 시간 동안 시퀀스 수집 (등록 / 인증 시) ---
def collect_sequence(duration=5):
    sequence = []
    start_time = time.time()
    while time.time() - start_time < duration:
        ret, frame = cap.read()
        if not ret: continue

        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = face_mesh.process(rgb)

        if results.multi_face_landmarks:
            landmarks = results.multi_face_landmarks[0]
            all_landmarks = np.array([[lm.x, lm.y, lm.z] for lm in landmarks.landmark])
            lips = np.array([all_landmarks[i] for i in LIPS_IDX_40])
            normalized = normalize_lips_3d(lips, all_landmarks)
            sequence.append(normalized.flatten())

        cv2.imshow("Webcam", frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    return extract_active_window(sequence)

# --- 두 시퀀스 간 DTW 거리 계산 ---
def compare_dtw(seq1, seq2):
    distance, _ = fastdtw(seq1, seq2, dist=euclidean)
    return distance

# --- 등록된 여러 시퀀스와의 평균 DTW 거리 계산 ---
def compare_multiple_dtw(test_seq, registered_seqs):
    distances = [compare_dtw(reg_seq, test_seq) for reg_seq in registered_seqs]
    return np.mean(distances), distances

# --- 등록된 시퀀스 기반 임계값 계산 (Mean + k*Std 방식) ---
def calculate_threshold(seqs):
    pair_dists = [compare_dtw(seqs[i], seqs[j]) for i in range(len(seqs)) for j in range(i+1, len(seqs))]
    threshold = np.mean(pair_dists) + 1.5 * np.std(pair_dists)  # ← 여기서 1.5 조정 가능
    print(f"🧠 자동 설정 임계값: {threshold:.2f} (mean + 1.5 * std)")
    return threshold

# --- MediaPipe 초기화 ---
mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh(refine_landmarks=True)
cap = cv2.VideoCapture(0)

# --- 전역 변수 ---
registered_sequences = []
thresh = None
font = cv2.FONT_HERSHEY_SIMPLEX
MAX_REGISTER_COUNT = 3

# --- 메인 루프 ---
while cap.isOpened():
    ret, frame = cap.read()
    if not ret: break

    h, w, _ = frame.shape
    display = frame.copy()
    key = cv2.waitKey(1) & 0xFF

    # 랜드마크 시각화
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = face_mesh.process(rgb)

    if results.multi_face_landmarks:
        landmarks = results.multi_face_landmarks[0]
        all_landmarks = np.array([[lm.x, lm.y, lm.z] for lm in landmarks.landmark])
        lips = np.array([all_landmarks[i] for i in LIPS_IDX_40])
        normalized = normalize_lips_3d(lips, all_landmarks)

        # 원본 프레임 상 시각화
        for i in LIPS_IDX_40:
            x = int(landmarks.landmark[i].x * w)
            y = int(landmarks.landmark[i].y * h)
            cv2.circle(display, (x, y), 2, (255, 0, 0), -1)

        # 정규화된 입술 시각화 (왼쪽 상단)
        anchor = (100, 100)
        scale = 80
        for pt in normalized:
            x = int(pt[0] * scale + anchor[0])
            y = int(pt[1] * scale + anchor[1])
            cv2.circle(display, (x, y), 2, (0, 0, 255), -1)

    # --- 키 이벤트 처리 ---
    if key == ord('r'):
        if len(registered_sequences) < MAX_REGISTER_COUNT:
            print(f"🟢 등록 시작 ({len(registered_sequences)+1}/{MAX_REGISTER_COUNT})")
            seq = collect_sequence()
            registered_sequences.append(seq)
            print("✅ 등록 완료")
            if len(registered_sequences) == MAX_REGISTER_COUNT:
                thresh = calculate_threshold(registered_sequences)
        else:
            print("❗ 등록은 최대 3회까지만 가능합니다. 초기화하려면 'C' 키를 누르세요.")

    elif key == ord('a'):
        print("🟡 인증 시작 (3초)")
        test_sequence = collect_sequence()
        if len(registered_sequences) == MAX_REGISTER_COUNT and thresh is not None:
            avg_dist, dists = compare_multiple_dtw(test_sequence, registered_sequences)
            print(f"📏 DTW 거리들: {[round(d, 2) for d in dists]}")
            print(f"📏 평균 거리: {avg_dist:.2f} (임계값: {thresh:.2f})")

            if avg_dist <= thresh:
                cv2.putText(display, "PASS ✅", (30, 60), font, 1.5, (0, 255, 0), 3)
            elif avg_dist <= thresh * 1.2:
                cv2.putText(display, "PASS (within 20%) ✅", (30, 60), font, 1.5, (0, 200, 100), 3)
            else:
                cv2.putText(display, "FAIL ❌", (30, 60), font, 1.5, (0, 0, 255), 3)
        else:
            print("❌ 등록이 부족합니다. 먼저 등록을 완료해주세요.")

    elif key == ord('c'):
        registered_sequences = []
        thresh = None
        print("♻️ 등록 초기화 완료")

    elif key == ord('q'):
        break

    cv2.imshow("Webcam", display)

# --- 종료 ---
cap.release()
cv2.destroyAllWindows()
