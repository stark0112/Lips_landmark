import os
import random
from PIL import Image
from tqdm import tqdm

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms

# ------------------------------
# 모델 정의
# ------------------------------
class DepthwiseSeparableConv(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_channels, in_channels, 3, stride, 1, groups=in_channels, bias=False),
            nn.BatchNorm2d(in_channels),
            nn.Hardswish(inplace=True),
            nn.Conv2d(in_channels, out_channels, 1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.Hardswish(inplace=True)
        )

    def forward(self, x):
        return self.block(x)

class InvertedResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels, expansion=4):
        super().__init__()
        hidden = in_channels * expansion
        self.use_res = in_channels == out_channels
        self.block = nn.Sequential(
            nn.Conv2d(in_channels, hidden, 1, bias=False),
            nn.BatchNorm2d(hidden),
            nn.Hardswish(inplace=True),
            nn.Conv2d(hidden, hidden, 3, 1, 1, groups=hidden, bias=False),
            nn.BatchNorm2d(hidden),
            nn.Hardswish(inplace=True),
            nn.Conv2d(hidden, out_channels, 1, bias=False),
            nn.BatchNorm2d(out_channels)
        )

    def forward(self, x):
        return x + self.block(x) if self.use_res else self.block(x)

class Lip3DLandmarkLite128Net(nn.Module):
    def __init__(self):
        super().__init__()
        self.stem = nn.Sequential(
            nn.Conv2d(3, 16, 3, 2, 1, bias=False),
            nn.BatchNorm2d(16),
            nn.Hardswish(inplace=True),
            DepthwiseSeparableConv(16, 32, stride=2),
        )
        self.irbs = nn.Sequential(
            InvertedResidualBlock(32, 64, expansion=2),
            InvertedResidualBlock(64, 64, expansion=2),
            InvertedResidualBlock(64, 96, expansion=3),
            InvertedResidualBlock(96, 96, expansion=3),
            InvertedResidualBlock(96, 128, expansion=4),
            InvertedResidualBlock(128, 128, expansion=4),
        )
        self.head = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(128, 60 * 3)
        )

    def forward(self, x):
        x = self.stem(x)
        x = self.irbs(x)
        x = self.head(x)
        return x.view(-1, 60, 3)

# ------------------------------
# Loss 함수 (Z축 가중치 포함)
# ------------------------------
class WeightedWingLoss(nn.Module):
    def __init__(self, w=10.0, epsilon=2.0, z_weight=5.0):
        super().__init__()
        self.w = w
        self.epsilon = epsilon
        self.C = w - w * torch.log(torch.tensor(1 + w / epsilon))
        self.z_weight = z_weight

    def forward(self, pred, target):
        diff = pred - target
        abs_diff = torch.abs(diff)
        loss = torch.where(
            abs_diff < self.w,
            self.w * torch.log(1 + abs_diff / self.epsilon),
            abs_diff - self.C.to(abs_diff.device)
        )
        loss[:, :, 2] *= self.z_weight  # z 가중치 적용
        return loss.mean()

# ------------------------------
# 데이터셋 정의
# ------------------------------
class Lip3DLandmarkDataset(Dataset):
    def __init__(self, data_dir, transform=None):
        self.data_dir = data_dir
        self.transform = transform
        self.data_list = torch.load(os.path.join(data_dir, "lips_3d_landmarks_60.pt"))

    def __len__(self):
        return len(self.data_list)

    def __getitem__(self, idx):
        max_tries = 10
        for _ in range(max_tries):
            img_path, landmarks = self.data_list[idx]
            full_path = os.path.join(self.data_dir, img_path)
            if os.path.exists(full_path):
                img = Image.open(full_path).convert("RGB")
                if self.transform:
                    img = self.transform(img)

                landmarks = torch.tensor(landmarks, dtype=torch.float32)
                landmarks[:, :2] = landmarks[:, :2] / 2.0  # x,y 정규화 (128 기준)
                z = landmarks[:, 2]
                z -= z.mean()             # 중심 정렬
                z *= 100.0                # 스케일 확대
                landmarks[:, 2] = z
                return img, landmarks
            idx = random.randint(0, len(self.data_list) - 1)
        raise FileNotFoundError("유효한 이미지를 찾을 수 없습니다.")

# ------------------------------
# 학습 루프
# ------------------------------
def train_model(model, train_loader, val_loader, criterion, optimizer, num_epochs=50):
    best_val_loss = float('inf')
    model.train()

    for epoch in range(num_epochs):
        running_loss = 0.0
        train_loader_tqdm = tqdm(train_loader, desc=f"Epoch [{epoch+1}/{num_epochs}] Training", leave=False)

        for batch_idx, (images, landmarks) in enumerate(train_loader_tqdm):
            images, landmarks = images.to(device), landmarks.to(device)
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, landmarks)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()

            if batch_idx % 10 == 0:
                avg_loss = running_loss / (batch_idx + 1)
                train_loader_tqdm.set_postfix(loss=avg_loss)

        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for images, landmarks in tqdm(val_loader, desc=f"Epoch [{epoch+1}/{num_epochs}] Validation", leave=False):
                images, landmarks = images.to(device), landmarks.to(device)
                outputs = model(images)
                val_loss += criterion(outputs, landmarks).item()
        val_loss /= len(val_loader)

        print(f"\n✅ Epoch [{epoch+1}/{num_epochs}] - Train Loss: {running_loss/len(train_loader):.6f}, Val Loss: {val_loss:.6f}")

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(), "best_lip_3d_landmark_model3.pth")
            print(f"✅ 최적 모델 저장됨 (Val Loss: {best_val_loss:.6f})")

        model.train()

# ------------------------------
# 실행부
# ------------------------------
if __name__ == '__main__':
    data_dir = r"C:\\Users\\User\\Desktop\\Lips_Landmark"
    transform = transforms.Compose([
        transforms.Resize((128, 128)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5]*3, std=[0.5]*3)
    ])

    train_dataset = Lip3DLandmarkDataset(data_dir, transform)
    val_dataset = Lip3DLandmarkDataset(data_dir, transform)

    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True, num_workers=2, pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False, num_workers=2, pin_memory=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = Lip3DLandmarkLite128Net().to(device)
    criterion = WeightedWingLoss(w=10.0, epsilon=2.0, z_weight=5.0).to(device)
    optimizer = optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-5)

    train_model(model, train_loader, val_loader, criterion, optimizer, num_epochs=50)
