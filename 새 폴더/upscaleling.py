import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
import os
from PIL import Image
from tqdm import tqdm  # 🔹 tqdm 라이브러리 추가 (진행률 표시)

# 🔹 Depthwise Separable Convolution 정의
class DepthwiseSeparableConv(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1):
        super().__init__()
        self.depthwise = nn.Conv2d(in_channels, in_channels, kernel_size=kernel_size, 
                                   stride=stride, padding=kernel_size//2, groups=in_channels, bias=False)
        self.pointwise = nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False)
        self.bn = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU6(inplace=True)
    
    def forward(self, x):
        x = self.depthwise(x)
        x = self.pointwise(x)
        x = self.bn(x)
        return self.relu(x)

# 🔹 Inverted Residual Block (IRB) 정의
class InvertedResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels, expansion=6):
        super().__init__()
        hidden_dim = in_channels * expansion
        self.use_residual = in_channels == out_channels
        
        self.block = nn.Sequential(
            nn.Conv2d(in_channels, hidden_dim, kernel_size=1, bias=False),
            nn.BatchNorm2d(hidden_dim),
            nn.ReLU6(inplace=True),

            DepthwiseSeparableConv(hidden_dim, hidden_dim, stride=1),
            nn.BatchNorm2d(hidden_dim),
            nn.ReLU6(inplace=True),

            nn.Conv2d(hidden_dim, out_channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(out_channels)
        )
    
    def forward(self, x):
        if self.use_residual:
            return x + self.block(x)  # Skip Connection 적용
        else:
            return self.block(x)

# 🔹 네트워크 정의
class Lip3DLandmarkNet(nn.Module):
    def __init__(self):
        super().__init__()

        # 초기 다운샘플링 단계
        self.conv1 = DepthwiseSeparableConv(3, 16, stride=2)  # 256 → 128
        self.conv2 = DepthwiseSeparableConv(16, 32, stride=2) # 128 → 64
        self.conv3 = DepthwiseSeparableConv(32, 64, stride=2) # 64 → 32
        self.conv4 = DepthwiseSeparableConv(64, 128, stride=2) # 32 → 16

        # IRB 블록 6개 적용
        self.irb1 = InvertedResidualBlock(128, 128)
        self.irb2 = InvertedResidualBlock(128, 128)
        self.irb3 = InvertedResidualBlock(128, 128)
        self.irb4 = InvertedResidualBlock(128, 256)
        self.irb5 = InvertedResidualBlock(256, 256)
        self.irb6 = InvertedResidualBlock(256, 256)

        # Global Average Pooling & FC Layer
        self.global_pool = nn.AdaptiveAvgPool2d(1)  # (B, C, H, W) → (B, C, 1, 1)
        self.fc = nn.Linear(256, 120)  # 40개의 (x, y, z) 좌표를 출력

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.conv4(x)

        x = self.irb1(x)
        x = self.irb2(x)
        x = self.irb3(x)
        x = self.irb4(x)
        x = self.irb5(x)
        x = self.irb6(x)

        x = self.global_pool(x)
        x = torch.flatten(x, 1)  # (B, 256, 1, 1) → (B, 256)
        x = self.fc(x)  # (B, 120)

        return x.view(-1, 40, 3)  # (B, 40, 3) 형태로 변환 (x, y, z)

# 2️⃣ 모델 불러오기
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = Lip3DLandmarkNet().to(device)
model.load_state_dict(torch.load("best_lip_3d_landmark_model.pth", map_location=device))
model.eval()  # 🔹 추론(테스트) 모드로 변경

# 3️⃣ 이미지 전처리 함수 정의
def preprocess_image(image_path):
    transform = transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
    ])
    image = Image.open(image_path).convert("RGB")
    image = transform(image).unsqueeze(0)  # (1, 3, 256, 256) 배치 차원 추가
    return image.to(device)

# 4️⃣ 랜드마크 예측 함수
def predict_landmarks(image_path):
    image = preprocess_image(image_path)
    with torch.no_grad():
        landmarks = model(image).cpu().numpy()  # (1, 40, 3) 형태
    return landmarks[0]  # (40, 3)

# 5️⃣ 예측 결과 시각화 함수
def visualize_landmarks(image_path, landmarks):
    image = Image.open(image_path)
    img_w, img_h = image.size  # 원본 이미지 크기 가져오기

    # 랜드마크 좌표 변환 (정규화 해제)
    landmarks[:, 0] = (landmarks[:, 0] + 1) * (img_w / 2)  # x 좌표 변환
    landmarks[:, 1] = (landmarks[:, 1] + 1) * (img_h / 2)  # y 좌표 변환

    # 이미지 및 랜드마크 표시
    plt.imshow(image)
    plt.scatter(landmarks[:, 0], landmarks[:, 1], c='red', marker='o')
    plt.title("Predicted 3D Lip Landmarks (Projected to 2D)")
    plt.show()

# 6️⃣ 테스트 이미지 예측 및 시각화
image_path = "test_lip_image.jpg"  # 🔹 테스트할 이미지 경로
predicted_landmarks = predict_landmarks(image_path)
visualize_landmarks(image_path, predicted_landmarks)