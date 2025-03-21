import torch
import cv2
import numpy as np
import matplotlib.pyplot as plt
import torch.nn as nn
import torch.nn.functional as F

# ✅ GPU 설정
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"✅ Using device: {device}")

# ✅ 5. BlazeFace 모델 정의 (BBox + 신뢰도 + 랜드마크 예측)
class DepthwiseSeparableConv(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=5, stride=1, padding=2):
        super().__init__()
        self.depthwise = nn.Conv2d(in_channels, in_channels, kernel_size, stride, padding, groups=in_channels, bias=False)
        self.pointwise = nn.Conv2d(in_channels, out_channels, 1, bias=False)
        self.bn = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        x = self.depthwise(x)
        x = self.pointwise(x)
        x = self.bn(x)
        return self.relu(x)

class BlazeBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1, num_layers=1, dropout_prob=0.3):
        super().__init__()
        layers = [DepthwiseSeparableConv(in_channels, out_channels, stride=stride)]
        for _ in range(num_layers - 1):
            layers.append(DepthwiseSeparableConv(out_channels, out_channels))
        self.block = nn.Sequential(*layers)
        self.skip_connection = nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride, bias=False) if stride > 1 else None
        self.dropout = nn.Dropout(p=dropout_prob)

    def forward(self, x):
        residual = self.skip_connection(x) if self.skip_connection else x
        x = self.block(x)
        x = x + residual
        x = self.dropout(x)
        return F.relu(x)

class BlazeFace(nn.Module):
    def __init__(self, num_anchors=4):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 32, kernel_size=5, stride=2, padding=2)

        self.blaze_block1 = BlazeBlock(32, 32, num_layers=5)
        self.blaze_block2 = BlazeBlock(32, 48, stride=2)
        self.blaze_block3 = BlazeBlock(48, 48, num_layers=6)
        self.blaze_block4 = BlazeBlock(48, 96, stride=2)
        self.blaze_block5 = BlazeBlock(96, 96, num_layers=6)
        self.blaze_block6 = BlazeBlock(96, 192, stride=2)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = self.blaze_block1(x)
        x = self.blaze_block2(x)
        x = self.blaze_block3(x)
        x = self.blaze_block4(x)
        x = self.blaze_block5(x)
        x = self.blaze_block6(x)
        return x
    
class BlazeFaceFull(nn.Module):
    def __init__(self, num_anchors=4):
        super().__init__()
        self.backbone = BlazeFace()

        # 🔹 바운딩 박스 예측 헤드
        self.bbox_head = nn.Conv2d(192, num_anchors * 4, kernel_size=1)  # 🔥 num_anchors 적용
        self.conf_head = nn.Conv2d(192, num_anchors, kernel_size=1)  # 🔥 신뢰도 예측
        
        # 🔹 랜드마크 서브 네트워크 추가
        self.landmark_subnet = nn.Sequential(
            nn.Conv2d(192, 128, kernel_size=3, padding=1),  # 추가적인 Feature 학습
            nn.ReLU(),
            nn.Conv2d(128, num_anchors * 7 * 2, kernel_size=1)  # 최종 랜드마크 예측
        )

    def forward(self, x):
        x = self.backbone(x)

        # 🔹 바운딩 박스 예측
        bboxes = self.bbox_head(x)  # (batch, num_anchors * 4, H, W)
        bboxes = bboxes.view(x.shape[0], -1, 4)  # 🔥 (batch, num_anchors, 4)

        # 🔹 신뢰도 예측
        confidences = torch.sigmoid(self.conf_head(x)).view(x.shape[0], -1)  # 🔥 (batch, num_anchors)

        # 🔹 랜드마크 예측
        landmarks = self.landmark_subnet(x)  # (batch, num_anchors * 14, H, W)
        landmarks = landmarks.view(x.shape[0], -1, 7 * 2)  # 🔥 (batch, num_anchors, 14)

        return bboxes, confidences, landmarks

# ✅ 모델 불러오기
model = BlazeFaceFull().to(device)

# ✅ BlazeFace 모델 불러오기 (strict=False 추가)
model.load_state_dict(torch.load("blazeface_full_best.pth", map_location=device), strict=False)
model.eval()
print("✅ 모델 로드 완료!")

# ✅ 평가할 이미지 경로
image_path = r"C:\Users\User\Desktop\Lips_Landmark\archive\Part 3\Part 3\020026.jpg" # 테스트할 이미지 파일 경로

# ✅ 이미지 전처리 함수
def preprocess_image(image_path, img_size=128):
    image = cv2.imread(image_path)
    if image is None:
        print(f"❌ 이미지 로드 실패: {image_path}")
        return None, None

    original_h, original_w = image.shape[:2]
    image_resized = cv2.resize(image, (img_size, img_size))
    image_tensor = torch.tensor(image_resized, dtype=torch.float32).permute(2, 0, 1) / 255.0
    return image_tensor.unsqueeze(0).to(device), (original_w, original_h)

# ✅ 이미지 불러오기
image_tensor, original_size = preprocess_image(image_path)

# ✅ 모델 추론 실행
if image_tensor is not None:
    with torch.no_grad():
        pred_bboxes, pred_confs, pred_landmarks = model(image_tensor)

    # 🔹 가장 신뢰도 높은 바운딩 박스 선택
    best_idx = torch.argmax(pred_confs, dim=1).item()
    best_conf = pred_confs[0, best_idx].item()

    # ✅ 신뢰도 값 출력
    print(f"📌 Best Confidence Score: {best_conf}")

    # 🔹 바운딩 박스 & 랜드마크 복원 (정규화 해제)
    w, h = original_size
    best_bbox = torch.clamp(pred_bboxes[0, best_idx].cpu(), 0, 1).numpy()
    best_landmarks = torch.clamp(pred_landmarks[0, best_idx].cpu(), 0, 1).numpy()

    x1, y1, x2, y2 = best_bbox
    x1, x2 = int(x1 * w), int(x2 * w)
    y1, y2 = int(y1 * h), int(y2 * h)

    best_landmarks = best_landmarks.reshape(-1, 2)
    best_landmarks = [(int(lx * w), int(ly * h)) for lx, ly in best_landmarks]

    # ✅ 시각화
    image = cv2.imread(image_path)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    # 바운딩 박스 그리기
    cv2.rectangle(image, (x1, y1), (x2, y2), (0, 255, 0), 2)

    # 랜드마크 그리기
    for (lx, ly) in best_landmarks:
        cv2.circle(image, (lx, ly), 2, (0, 0, 255), -1)

    # 이미지 출력
    plt.figure(figsize=(8, 8))
    plt.imshow(image)
    plt.axis("off")
    plt.show()