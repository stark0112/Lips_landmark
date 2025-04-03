import os
import cv2
import numpy as np
from fastdtw import fastdtw
from scipy.spatial.distance import euclidean
import mediapipe as mp
from sklearn.metrics import roc_curve, accuracy_score, precision_score, recall_score, f1_score
import matplotlib.pyplot as plt

# --- 설정 ---
LIPS_IDX_40 = [61, 185, 40, 39, 37, 0, 267, 269, 270, 409,
               291, 375, 321, 405, 314, 17, 84, 181, 91, 146,
               78, 95, 88, 178, 87, 14, 317, 402, 318, 324,
               308, 415, 310, 311, 312, 13, 82, 81, 80, 191]
EYE_LEFT_IDX, EYE_RIGHT_IDX, NOSE_TIP_IDX = 33, 263, 1

ROOT = r"C:\\Users\\jyk\\Desktop\\LIP_SECURITY"
REGISTER_DIR = os.path.join(ROOT, "플렉시스테라마임(김재영)", "train")
TEST_SETS = {
    "같은사람_같은단어": os.path.join(ROOT, "플렉시스테라마임(김재영)", "test"),
    "같은사람_다른단어": os.path.join(ROOT, "안드로이드딥러닝(김재영)", "test"),
    "다른사람_같은단어": os.path.join(ROOT, "플렉시스테라마임(오재복)", "test"),
    "다른사람_다른단어": os.path.join(ROOT, "테라마임화이팅(오재복)", "test")
}

# --- 초기화 ---
mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh(refine_landmarks=True)

# --- 정규화 함수 ---
def normalize_lips_3d(points, ref_points):
    center = np.mean(points, axis=0)
    points -= center
    x_axis = points[10] - points[0]
    y_axis = points[29] - points[19]
    x_axis /= np.linalg.norm(x_axis)
    y_axis /= np.linalg.norm(y_axis)
    z_axis = np.cross(x_axis, y_axis); z_axis /= np.linalg.norm(z_axis)
    y_axis = np.cross(z_axis, x_axis); y_axis /= np.linalg.norm(y_axis); y_axis *= -1
    R = np.stack([x_axis, y_axis, z_axis], axis=1)
    points = points @ R
    eye_left, eye_right, nose = ref_points[EYE_LEFT_IDX], ref_points[EYE_RIGHT_IDX], ref_points[NOSE_TIP_IDX]
    scale = (np.linalg.norm(eye_left - eye_right) + np.linalg.norm(nose - (eye_left + eye_right) / 2)) / 2
    return points / scale if scale > 0 else points

# --- 시퀀스 추출 ---
def extract_sequence_from_video(path, rotate=True, window_size=15):
    sequence = []
    cap = cv2.VideoCapture(path)
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret: break
        if rotate: frame = cv2.rotate(frame, cv2.ROTATE_90_CLOCKWISE)
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = face_mesh.process(rgb)
        if results.multi_face_landmarks:
            lm = results.multi_face_landmarks[0].landmark
            all_landmarks = np.array([[p.x, p.y, p.z] for p in lm])
            lips = np.array([all_landmarks[i] for i in LIPS_IDX_40])
            normalized = normalize_lips_3d(lips, all_landmarks)
            sequence.append(normalized.flatten())
    cap.release()
    if len(sequence) < window_size: return None
    diffs = np.linalg.norm(np.diff(sequence, axis=0), axis=1)
    scores = np.convolve(diffs, np.ones(window_size), mode='valid')
    start = np.argmax(scores)
    return np.array(sequence[start:start+window_size])

# --- 거리 계산 ---
def compare_dtw(seq1, seq2):
    dist, _ = fastdtw(seq1, seq2, dist=euclidean)
    return dist

# --- EER 기반 임계값 계산 ---
def get_eer_threshold(distances, labels):
    fpr, tpr, thresholds = roc_curve(labels, -np.array(distances))  # 작을수록 유사하므로 부호 반전
    fnr = 1 - tpr
    eer_idx = np.argmin(np.abs(fpr - fnr))
    eer_thresh = -thresholds[eer_idx]  # 부호 복원
    eer = fpr[eer_idx]

    # --- 시각화 ---
    plt.plot(fpr, tpr, label='ROC Curve')
    plt.plot([0,1],[0,1],'--',alpha=0.3)
    plt.scatter(fpr[eer_idx], tpr[eer_idx], color='red', label=f"EER = {eer:.2f}")
    plt.title("ROC Curve & EER")
    plt.xlabel("False Positive Rate"); plt.ylabel("True Positive Rate")
    plt.legend(); plt.grid(True); plt.show()

    return eer_thresh, eer

# --- 등록 ---
print("📥 등록 시작")
registered_sequences = []
for f in sorted(os.listdir(REGISTER_DIR)):
    if f.endswith((".mp4", ".mov")):
        seq = extract_sequence_from_video(os.path.join(REGISTER_DIR, f))
        if seq is not None: registered_sequences.append(seq)
if len(registered_sequences) < 3:
    print("❌ 등록 실패: 3개 이상 필요"); exit()

# --- 테스트 ---
print("\n📊 테스트 결과")
all_distances, all_labels = [], []

for label, test_path in TEST_SETS.items():
    print(f"\n📂 {label}")
    is_positive = (label == "같은사람_같은단어")
    for f in sorted(os.listdir(test_path)):
        if not f.endswith(".mp4"): continue
        path = os.path.join(test_path, f)
        test_seq = extract_sequence_from_video(path)
        if test_seq is None:
            print(f"⚠️ {f} - 시퀀스 부족"); continue
        dists = [compare_dtw(reg, test_seq) for reg in registered_sequences]
        avg = np.mean(dists)
        all_distances.append(avg)
        all_labels.append(1 if is_positive else 0)
        print(f"{f} → 평균 거리: {avg:.2f}")

# --- 성능 평가 ---
eer_thresh, eer = get_eer_threshold(all_distances, all_labels)
y_pred = [1 if d <= eer_thresh else 0 for d in all_distances]

print(f"\n📈 EER Threshold: {eer_thresh:.2f} | EER: {eer:.3f}")
print("\n📌 EER 기준 성능 지표")
print(f"Accuracy : {accuracy_score(all_labels, y_pred):.2f}")
print(f"Precision: {precision_score(all_labels, y_pred):.2f}")
print(f"Recall   : {recall_score(all_labels, y_pred):.2f}")
print(f"F1 Score : {f1_score(all_labels, y_pred):.2f}")
