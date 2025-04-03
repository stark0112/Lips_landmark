import os
import torch
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from PIL import Image
import numpy as np
from torchvision import transforms
from train_3d_lanmark_model import Lip3DLandmarkLite128Net  # ✅ 모델 정의가 포함된 파일에서 import (또는 직접 포함)

# 🔹 경로 설정
base_dir = r"C:\Users\User\Desktop\Lips_Landmark"
pt_file = os.path.join(base_dir, "lips_3d_landmarks_60.pt")
model_path = os.path.join(base_dir, "best_lip_3d_landmark_model3.pth")

# 🔹 이미지 전처리
transform = transforms.Compose([
    transforms.Resize((128, 128)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5]*3, std=[0.5]*3)
])

# 🔹 모델 로드
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = Lip3DLandmarkLite128Net().to(device)
model.load_state_dict(torch.load(model_path, map_location=device))
model.eval()

# 🔹 데이터 로드
data = torch.load(pt_file)

# 🔹 랜덤 샘플 선택
idx = torch.randint(0, len(data), (1,)).item()
img_path, gt_landmarks = data[idx]
gt_landmarks = torch.tensor(gt_landmarks)

# 🔹 이미지 로드
img_full_path = os.path.join(base_dir, img_path)
img = Image.open(img_full_path).convert("RGB")
img_tensor = transform(img).unsqueeze(0).to(device)

# 🔹 모델 추론
with torch.no_grad():
    pred_landmarks = model(img_tensor)[0].cpu()

# ✅ 2D 시각화
img_vis = np.array(img.resize((256, 256)))  # 시각화용
gt_landmarks_2d = gt_landmarks[:, :2]
pred_landmarks_2d = pred_landmarks[:, :2]

plt.figure(figsize=(6, 6))
plt.imshow(img_vis)
plt.scatter(gt_landmarks_2d[:, 0], gt_landmarks_2d[:, 1], c='lime', label='GT', s=10)
plt.scatter(pred_landmarks_2d[:, 0], pred_landmarks_2d[:, 1], c='red', label='Pred', s=10, marker='x')
plt.legend()
plt.title(f"2D Landmark Comparison - Sample #{idx}")
plt.axis("off")
plt.tight_layout()
plt.show()

# ✅ 3D 시각화
fig = plt.figure(figsize=(8, 6))
ax = fig.add_subplot(111, projection='3d')

# Ground Truth
ax.scatter(gt_landmarks[:, 0], gt_landmarks[:, 1], gt_landmarks[:, 2], c='green', label="GT", s=20)
# Prediction
ax.scatter(pred_landmarks[:, 0], pred_landmarks[:, 1], pred_landmarks[:, 2], c='red', label="Pred", s=20, marker='^')

ax.set_title(f"3D Landmark Comparison - Sample #{idx}")
ax.set_xlabel("X")
ax.set_ylabel("Y")
ax.set_zlabel("Z")
ax.view_init(elev=10, azim=-70)
ax.legend()
plt.tight_layout()
plt.show()
