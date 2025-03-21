import os
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torchvision
import cv2
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm

# ✅ 1. GPU 설정
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
torch.backends.cudnn.benchmark = True
print(f"✅ Using device: {device}")

# ✅ 2. 데이터셋 로드 (Bounding Box + Confidence + Landmarks)
merged_data = torch.load("merged_data_celeba2_aug.pt", map_location=device)

# ✅ 3. 데이터셋 클래스 정의
class FaceDataset(Dataset):
    def __init__(self, data, img_size=128):
        self.data = data
        self.img_size = img_size

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        try:
            image_path, bbox, landmarks = self.data[idx]

            # 🔹 데이터가 None이거나 길이가 부족하면 건너뛰기
            if not image_path or len(bbox) != 4 or len(landmarks) != 14:
                print(f"🚨 잘못된 데이터 샘플: {image_path} → 건너뜀")
                return torch.zeros(3, self.img_size, self.img_size), torch.zeros(4), torch.zeros(14)

            # 🔹 이미지 로드
            image = cv2.imread(image_path)
            if image is None:
                print(f"❌ 이미지 로드 실패: {image_path} → 건너뜀")
                return torch.zeros(3, self.img_size, self.img_size), torch.zeros(4), torch.zeros(14)

            original_h, original_w = image.shape[:2]

            # 🔹 이미지 전처리
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            image = cv2.resize(image, (self.img_size, self.img_size))
            image = torch.tensor(image, dtype=torch.float32).permute(2, 0, 1) / 255.0

            # 🔹 바운딩 박스 정규화 (0~1)
            bbox = torch.tensor([
                bbox[0] / original_w, bbox[1] / original_h, 
                bbox[2] / original_w, bbox[3] / original_h
            ], dtype=torch.float32)

            # 🔹 랜드마크는 이미 정규화된 상태
            landmarks = torch.tensor(landmarks, dtype=torch.float32)

            return image, bbox, landmarks

        except Exception as e:
            print(f"🚨 데이터셋 오류: {e} → 건너뜀")
            return torch.zeros(3, self.img_size, self.img_size), torch.zeros(4), torch.zeros(14)


# ✅ 4. DataLoader 설정
def get_dataloader(batch_size=64):
    dataset = FaceDataset(merged_data)
    return DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=4, pin_memory=True, persistent_workers=True)

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
        self.landmark_head = nn.Conv2d(192, num_anchors * 7 * 2, kernel_size=1)  # 🔥 랜드마크 예측

    def forward(self, x):
        x = self.backbone(x)

        # 🔹 바운딩 박스 예측
        bboxes = self.bbox_head(x)  # (batch, num_anchors * 4, H, W)
        bboxes = bboxes.view(x.shape[0], -1, 4)  # 🔥 (batch, num_anchors, 4)

        # 🔹 신뢰도 예측
        confidences = torch.sigmoid(self.conf_head(x)).view(x.shape[0], -1)  # 🔥 (batch, num_anchors)

        # 🔹 랜드마크 예측
        landmarks = self.landmark_head(x)  # (batch, num_anchors * 14, H, W)
        landmarks = landmarks.view(x.shape[0], -1, 7 * 2)  # 🔥 (batch, num_anchors, 14)

        return bboxes, confidences, landmarks






if __name__ == "__main__":
    model = BlazeFaceFull().to(device)

    optimizer = optim.Adam(model.parameters(), lr=0.001)
    criterion_bbox = nn.SmoothL1Loss().to(device)
    criterion_conf = nn.BCELoss().to(device)
    criterion_landmark = nn.SmoothL1Loss().to(device)

    dataloader = get_dataloader(batch_size=64)
    num_epochs = 50
    best_loss = float('inf')

    for epoch in range(num_epochs):
        total_loss = 0.0
        model.train()

        with tqdm(dataloader, desc=f"Epoch {epoch+1}/{num_epochs}") as tbar:
            for images, gt_bboxes, gt_landmarks in tbar:
                images, gt_bboxes, gt_landmarks = images.to(device), gt_bboxes.to(device), gt_landmarks.to(device)
                optimizer.zero_grad()

                pred_bboxes, pred_confs, pred_landmarks = model(images)

    
                bbox_loss = criterion_bbox(pred_bboxes, gt_bboxes.unsqueeze(1).expand_as(pred_bboxes))
                conf_loss = criterion_conf(pred_confs, torch.ones_like(pred_confs, device=device))
                landmark_loss = criterion_landmark(pred_landmarks, gt_landmarks.unsqueeze(1).expand_as(pred_landmarks))
                batch_loss = bbox_loss + conf_loss + landmark_loss

                batch_loss.backward()
                optimizer.step()
                total_loss += batch_loss.item()
                tbar.set_postfix(loss=batch_loss.item())

        print(f"📢 Epoch [{epoch+1}/{num_epochs}], Total Loss: {total_loss:.4f}")

        if total_loss < best_loss:
            best_loss = total_loss
            torch.save(model.state_dict(), "blazeface_full_best.pth")
            print("✅ New best model saved!")

    print("🎯 전체 모델 학습 완료! 모델 저장됨.")


