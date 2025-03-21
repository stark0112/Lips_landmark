import os
import cv2
import torch
import numpy as np
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torchvision
import matplotlib.pyplot as plt
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm

# ✅ 1. GPU 설정 (CUDA 활성화)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
torch.backends.cudnn.benchmark = True
print(f"✅ Using device: {device}")

# ✅ 2. 데이터 로드
celeba_data = torch.load("total_data_celeba.pt", map_location=device)

# ✅ 3. 데이터셋 클래스
class FaceDataset(Dataset):
    def __init__(self, data, img_size=128):
        self.data = data
        self.img_size = img_size

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        image_path, x1, y1, x2, y2 = self.data[idx]

        image = cv2.imread(image_path)
        if image is None:
            return torch.zeros(3, self.img_size, self.img_size), torch.zeros(4)

        original_h, original_w = image.shape[:2]

        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image = cv2.resize(image, (self.img_size, self.img_size))
        image = torch.tensor(image, dtype=torch.float32).permute(2, 0, 1) / 255.0

        bbox = torch.tensor([
            x1 / original_w, y1 / original_h, x2 / original_w, y2 / original_h
        ], dtype=torch.float32)

        return image, bbox

# ✅ 4. DataLoader 설정
def get_dataloader(batch_size=64):
    dataset = FaceDataset(celeba_data)
    return DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=4, pin_memory=True, persistent_workers=True)

# ✅ 5. BlazeFace 모델 정의
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

        self.bbox_head = nn.Conv2d(192, num_anchors * 4, kernel_size=1)
        self.conf_head = nn.Conv2d(192, num_anchors, kernel_size=1)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = self.blaze_block1(x)
        x = self.blaze_block2(x)
        x = self.blaze_block3(x)
        x = self.blaze_block4(x)
        x = self.blaze_block5(x)
        x = self.blaze_block6(x)

        bboxes = self.bbox_head(x)
        confidences = torch.sigmoid(self.conf_head(x))

        return bboxes.view(bboxes.shape[0], -1, 4), confidences.view(confidences.shape[0], -1)

# ✅ 6. 모델 학습 (conf_loss 개선 반영)
if __name__ == "__main__":
    model = BlazeFace().to(device)
    dataloader = get_dataloader(batch_size=64)
    
    criterion_bbox = nn.SmoothL1Loss().to(device)
    criterion_conf = nn.BCELoss().to(device)
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=3)

    num_epochs = 30
    best_loss = float('inf')

    for epoch in range(num_epochs):
        total_loss = 0.0
        model.train()

        with tqdm(dataloader, desc=f"Epoch {epoch+1}/{num_epochs}") as tbar:
            for batch_idx, (images, gt_bboxes) in enumerate(tbar):
                images, gt_bboxes = images.to(device, non_blocking=True), gt_bboxes.to(device, non_blocking=True)
                optimizer.zero_grad()
                pred_bboxes, pred_confs = model(images)

                # ✅ conf_loss 개선된 부분
                conf_target = torch.zeros_like(pred_confs, device=device)  # 모든 앵커를 배경(0)으로 설정
                positive_idxs, _ = match_anchors(gt_bboxes, anchors)  # GT와 매칭된 앵커 인덱스 찾기
                conf_target.scatter_(1, positive_idxs.unsqueeze(1), 1.0)  # GT와 매칭된 앵커만 1로 설정

                bbox_loss = criterion_bbox(pred_bboxes, gt_bboxes.unsqueeze(1).expand_as(pred_bboxes))
                conf_loss = criterion_conf(pred_confs, conf_target)
                batch_loss = bbox_loss + conf_loss

                batch_loss.backward()
                optimizer.step()
                total_loss += batch_loss.item()
                tbar.set_postfix(loss=batch_loss.item())

        scheduler.step(total_loss)
        print(f"📢 Epoch [{epoch+1}/{num_epochs}], Total Loss: {total_loss:.4f}")

        if total_loss < best_loss:
            best_loss = total_loss
            torch.save(model.state_dict(), "blazeface_best.pth")
            print("✅ New best model saved!")

    print("🎯 학습 완료! 모델 저장됨.")
