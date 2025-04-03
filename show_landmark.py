import os
import torch
import random
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import numpy as np

# ✅ 경로
base_dir = r"C:\Users\User\Desktop\Lips_Landmark"
pt_file_path = os.path.join(base_dir, "lips_3d_landmarks_new2.pt")
data = torch.load(pt_file_path)

# ✅ 입 안쪽 인덱스
inner_indices = list(range(10, 20)) + list(range(20, 30)) + [10, 20, 19, 29]

def expand_landmarks_60(landmarks):
    """입 안쪽 랜드마크 사이에 중간점 20개 추가 (총 60개)"""
    landmarks = torch.tensor(landmarks, dtype=torch.float32)
    mids = []
    # upper inner (10~19)
    for i in range(10, 19):
        mids.append((landmarks[i] + landmarks[i+1]) / 2.0)
    # lower inner (20~29)
    for i in range(20, 29):
        mids.append((landmarks[i] + landmarks[i+1]) / 2.0)
    # extra edges
    mids.append((landmarks[20] + landmarks[10]) / 2.0)
    mids.append((landmarks[19] + landmarks[29]) / 2.0)
    mids = torch.stack(mids, dim=0)
    return torch.cat([landmarks, mids], dim=0)

def plot_3d_landmarks(landmarks, title="3D Landmarks", color="green"):
    """3D 랜드마크 시각화"""
    fig = plt.figure(figsize=(6, 6))
    ax = fig.add_subplot(111, projection='3d')
    xs, ys, zs = landmarks[:, 0], landmarks[:, 1], landmarks[:, 2]
    ax.scatter(xs, ys, zs, c=color, s=20)
    ax.set_title(title)
    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.set_zlabel("Z")
    ax.view_init(elev=10, azim=-80)  # 보기 각도
    plt.tight_layout()
    plt.show()

# ✅ 샘플 시각화
idx = random.randint(0, len(data)-1)
img_path, gt_landmarks = data[idx]
gt_landmarks = torch.tensor(gt_landmarks)

# 원본 40개
plot_3d_landmarks(gt_landmarks, title=f"[GT] 3D Landmarks (40개) - #{idx}", color="green")

# 확장된 60개
landmarks_60 = expand_landmarks_60(gt_landmarks)
plot_3d_landmarks(landmarks_60, title=f"[GT] 3D Landmarks (60개, midpoints) - #{idx}", color="orange")
