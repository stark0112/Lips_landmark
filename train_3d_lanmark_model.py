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

class Lip3DLandmarkDataset(Dataset):
    def __init__(self, data_dir, transform=None):
        self.data_dir = data_dir
        self.transform = transform

        # 🔸 .pt 파일 불러오기
        self.data_list = torch.load(os.path.join(data_dir, "lips_3d_landmarks.pt"))

    def __len__(self):
        return len(self.data_list)

    def __getitem__(self, idx):
        # 3개의 값이 저장된 경우 → (이미지 경로, 랜드마크, 원본 크기)
        img_path, landmarks, _ = self.data_list[idx]  # 원본 크기 (_) 무시

        img = Image.open(os.path.join(self.data_dir, img_path)).convert("RGB")

        if self.transform:
            img = self.transform(img)

        # 🔹 3D 랜드마크 좌표를 Tensor로 변환
        landmarks = torch.tensor(landmarks, dtype=torch.float32)  # (40, 3)

        return img, landmarks
    
# 🔹 학습 루프 (최적의 모델 저장 & 진행률 표시)
def train_model(model, train_loader, val_loader, criterion, optimizer, num_epochs=50):
    best_val_loss = float('inf')  # 초기 최적 검증 손실 설정
    model.train()

    for epoch in range(num_epochs):
        running_loss = 0.0
        train_loader_tqdm = tqdm(train_loader, desc=f"Epoch [{epoch+1}/{num_epochs}] Training", leave=False)

        for batch_idx, (images, landmarks) in enumerate(train_loader_tqdm):
            images, landmarks = images.to(device, non_blocking=True), landmarks.to(device, non_blocking=True)

            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, landmarks)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()

            # 🔹 10개 배치마다 평균 손실 출력
            if batch_idx % 10 == 0:
                avg_loss = running_loss / (batch_idx + 1)
                train_loader_tqdm.set_postfix(loss=avg_loss)

        # 🔹 검증 단계 (진행률 표시)
        model.eval()
        val_loss = 0.0
        val_loader_tqdm = tqdm(val_loader, desc=f"Epoch [{epoch+1}/{num_epochs}] Validation", leave=False)

        with torch.no_grad():
            for images, landmarks in val_loader_tqdm:
                images, landmarks = images.to(device, non_blocking=True), landmarks.to(device, non_blocking=True)
                outputs = model(images)
                val_loss += criterion(outputs, landmarks).item()

        val_loss /= len(val_loader)

        # 🔹 에포크별 손실 출력
        print(f"\n✅ Epoch [{epoch+1}/{num_epochs}] - Train Loss: {running_loss/len(train_loader):.6f}, Val Loss: {val_loss:.6f}")

        # 🔹 최적의 모델 저장 (검증 손실이 줄어들 때만 저장)
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(), "best_lip_3d_landmark_model.pth")
            print(f"✅ 새로운 최적 모델 저장: best_lip_3d_landmark_model.pth (Val Loss: {best_val_loss:.6f})")

        model.train()

    print("✅ 최종 학습 완료!")

# 🔹 실행 코드
if __name__ == '__main__':
    # 🔹 데이터 로딩
    data_dir = r"C:\Users\User\Desktop\Lips_Landmark"
    transform = transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
    ])

    train_dataset = Lip3DLandmarkDataset(data_dir=data_dir, transform=transform)
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True, num_workers=2, pin_memory=True)

    val_dataset = Lip3DLandmarkDataset(data_dir=data_dir, transform=transform)
    val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False, num_workers=2, pin_memory=True)

    # 🔹 모델 및 학습 설정
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = Lip3DLandmarkNet().to(device)
    criterion = nn.SmoothL1Loss().to(device)
    optimizer = optim.Adam(model.parameters(), lr=0.001, weight_decay=1e-5)

    # 🔹 모델 학습 실행
    train_model(model, train_loader, val_loader, criterion, optimizer, num_epochs=50)
