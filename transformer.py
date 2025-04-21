# ✅ Triplet Loss 기반 Transformer AutoEncoder 학습 코드 (with collapse 방지 개선)

import os
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torch.nn.utils.rnn import pad_sequence
from tqdm import tqdm
from glob import glob
import csv
import math
import wandb

# --- 하이퍼파라미터 --- #
input_dim = 120
emb_dim = 128
num_heads = 4
num_layers = 4
dropout = 0.1
batch_size = 32
lr = 0.001
epochs = 80
alpha = 0.4
margin = 0.02
min_pos_dist = 0.0004

TRIPLET_CSV = r"C:\\Users\\User\\Desktop\\data\\triplet_csv\\triplet_index.csv"
ROOT_DATA = r"C:\\Users\\User\\Desktop\\data"

wandb.init(project="transformer_lips_triplet")

# --- 모델 정의 --- #
class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=1024):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float32).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(1)
        self.register_buffer('pe', pe)

    def forward(self, x):
        return x + self.pe[:x.size(0)]

class TransformerAutoEncoder(nn.Module):
    def __init__(self, input_dim, emb_dim, num_heads, num_layers, dropout=0.1):
        super().__init__()
        self.cls_token = nn.Parameter(torch.randn(1, 1, emb_dim))
        self.enc_embedding = nn.Linear(input_dim, emb_dim)
        self.pos_encoder = PositionalEncoding(emb_dim)
        self.encoder = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(emb_dim, num_heads, 2048, dropout),
            num_layers=num_layers
        )
        self.emb_proj = nn.Sequential(  # 🟩 collapse 방지용 projection
            nn.LayerNorm(emb_dim),
            nn.Dropout(0.3),
            nn.Linear(emb_dim, emb_dim),
            nn.ReLU(),
            nn.Linear(emb_dim,emb_dim)
        )
        self.dec_embedding = nn.Linear(emb_dim, emb_dim)
        self.decoder = nn.TransformerDecoder(
            nn.TransformerDecoderLayer(emb_dim, num_heads, 2048, dropout),
            num_layers=num_layers
        )
        self.dec_fc = nn.Linear(emb_dim, input_dim)

    def forward(self, x):
        B, T, _ = x.shape
        x = self.enc_embedding(x)
        cls = self.cls_token.repeat(B, 1, 1)
        x = torch.cat([cls, x], dim=1).transpose(0, 1)
        x = self.pos_encoder(x)
        enc_output = self.encoder(x)
        emb = self.emb_proj(enc_output[0])
        dec_input = self.dec_embedding(emb).unsqueeze(1).repeat(1, T, 1).transpose(0, 1)
        dec_output = self.decoder(dec_input, enc_output)
        out = self.dec_fc(dec_output.transpose(0, 1))
        return out, emb

# --- Dataset --- #
class TripletLipDataset(Dataset):
    def __init__(self, csv_path, root_dir):
        with open(csv_path, 'r') as f:
            reader = csv.reader(f)
            next(reader)
            self.triplets = list(reader)
        self.root_dir = root_dir
        self.split_folders = [os.path.join(root_dir, f) for f in os.listdir(root_dir) if f.startswith("split_sentence")]

    def _find_full_path(self, rel_path):
        for folder in self.split_folders:
            full_path = os.path.join(folder, rel_path)
            if os.path.exists(full_path):
                return full_path
        raise FileNotFoundError(f"❌ 파일을 찾을 수 없습니다: {rel_path}")

    def __len__(self):
        return len(self.triplets)

    def __getitem__(self, idx):
        anchor_rel, pos_rel, neg_rel = self.triplets[idx]
        a = np.load(self._find_full_path(anchor_rel))
        p = np.load(self._find_full_path(pos_rel))
        n = np.load(self._find_full_path(neg_rel))
        return self._to_tensor(a), self._to_tensor(p), self._to_tensor(n)

    def _to_tensor(self, arr):
        return torch.tensor(arr, dtype=torch.float32)

def collate_fn_triplet(batch):
    anchors = pad_sequence([b[0] for b in batch], batch_first=True)
    positives = pad_sequence([b[1] for b in batch], batch_first=True)
    negatives = pad_sequence([b[2] for b in batch], batch_first=True)
    return anchors, positives, negatives

# --- Triplet Loss --- #
def triplet_loss(anchor, positive, negative, margin=0.03, min_pos_dist=0.004):
    anchor = F.normalize(anchor, dim=1)
    positive = F.normalize(positive, dim=1)
    negative = F.normalize(negative, dim=1)
    pos_dist = F.pairwise_distance(anchor, positive)
    neg_dist = F.pairwise_distance(anchor, negative)
    loss = F.relu(pos_dist - neg_dist + margin) + F.relu(min_pos_dist - pos_dist) * 0.5
    return loss.mean()

# --- 학습 --- #
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = TransformerAutoEncoder(input_dim, emb_dim, num_heads, num_layers, dropout).to(device)
optimizer = optim.Adam(model.parameters(), lr=lr)
criterion = nn.MSELoss()

train_dataset = TripletLipDataset(TRIPLET_CSV, ROOT_DATA)
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, collate_fn=collate_fn_triplet)

def train():
    best_triplet_loss = float('inf')
    for epoch in range(1, epochs + 1):
        model.train()
        total_loss, total_recon, total_triplet = 0, 0, 0

        for anchor, positive, negative in tqdm(train_loader, desc=f"[Epoch {epoch}]"):
            anchor, positive, negative = anchor.to(device), positive.to(device), negative.to(device)
            optimizer.zero_grad()
            rec_a, emb_a = model(anchor)
            rec_p, emb_p = model(positive)
            rec_n, emb_n = model(negative)

            loss_recon = criterion(rec_a, anchor) + criterion(rec_p, positive) + criterion(rec_n, negative)
            loss_triplet = triplet_loss(emb_a, emb_p, emb_n, margin, min_pos_dist)
            loss = loss_recon + alpha * loss_triplet
            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            total_recon += loss_recon.item()
            total_triplet += loss_triplet.item()

        avg_total_loss = total_loss / len(train_loader)
        avg_recon_loss = total_recon / len(train_loader)
        avg_triplet_loss = total_triplet / len(train_loader)

        wandb.log({"epoch": epoch, "train_loss": avg_total_loss,
                   "recon_loss": avg_recon_loss, "triplet_loss": avg_triplet_loss})

        print(f"[Epoch {epoch}] Loss: {avg_total_loss:.4f} | Recon: {avg_recon_loss:.4f} | Triplet: {avg_triplet_loss:.4f}")

        if avg_triplet_loss < best_triplet_loss:
            best_triplet_loss = avg_triplet_loss
            save_path = os.path.join(ROOT_DATA, "transformer_triplet_best.pth")
            torch.save(model.state_dict(), save_path)
            print(f"✅ Best model saved at epoch {epoch} (triplet_loss={avg_triplet_loss:.4f})")

if __name__ == '__main__':
    train()
