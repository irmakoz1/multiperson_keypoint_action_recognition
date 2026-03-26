#!/usr/bin/env python3
"""
Baseline for Hypothesis 2: Separate spatial (per frame) + temporal transformer modeling.
Uses MPOSEFeatureExtractor, spatial GraphSAGE per frame, then a temporal transformer.
"""

import os
import sys
import argparse
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torch_geometric.nn import SAGEConv
from tqdm import tqdm
from sklearn.metrics import classification_report, balanced_accuracy_score
from sklearn.utils.class_weight import compute_class_weight
import json
from pathlib import Path
import logging
import mpose


# Add project root
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../..')))
from src.encoder.mpose_encoder import MPOSEFeatureExtractor

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

COCO_CONNECTIONS = [
    (0,1),(0,2),(1,3),(2,4),(5,6),(5,7),(7,9),(6,8),(8,10),
    (5,11),(6,12),(11,12),(11,13),(13,15),(12,14),(14,16)
]

# ----------------------------
# Spatial GraphSAGE (applied per frame)
# ----------------------------
class SpatialGraphSAGE(nn.Module):
    def __init__(self, in_dim, hidden_dims, num_joints, skeleton_connections, dropout=0.3):
        super().__init__()
        self.num_joints = num_joints
        # Build spatial edge index
        edge_list = []
        for (i, j) in skeleton_connections:
            edge_list.append((i, j))
            edge_list.append((j, i))
        self.register_buffer('edge_index', torch.tensor(edge_list, dtype=torch.long).t().contiguous())

        dims = [in_dim] + hidden_dims
        self.convs = nn.ModuleList()
        self.norms = nn.ModuleList()
        self.projs = nn.ModuleList()
        for i in range(len(dims)-1):
            self.convs.append(SAGEConv(dims[i], dims[i+1], aggr='mean'))
            self.norms.append(nn.BatchNorm1d(dims[i+1]))
            if dims[i+1] != dims[i]:
                self.projs.append(nn.Linear(dims[i], dims[i+1]))
            else:
                self.projs.append(nn.Identity())
        self.dropout = dropout

    def forward(self, x):
        """
        x: (batch * T, J, D)
        Returns: (batch * T, J, hidden_dim) after layers
        """
        batch = x.size(0)
        N = self.num_joints
        offsets = torch.arange(batch, device=x.device) * N
        edge_index_batch = self.edge_index.repeat(1, batch)
        offsets_per_edge = offsets.repeat_interleave(self.edge_index.size(1))
        edge_index_batch = edge_index_batch + offsets_per_edge.unsqueeze(0)
        x_flat = x.view(-1, x.size(-1))  # (batch * N, D)

        for conv, norm, proj in zip(self.convs, self.norms, self.projs):
            x_res = proj(x_flat)
            x_flat = conv(x_flat, edge_index_batch)
            x_flat = norm(x_flat)
            x_flat = F.relu(x_flat)
            x_flat = x_flat + x_res
            x_flat = F.dropout(x_flat, p=self.dropout, training=self.training)

        x_flat = x_flat.view(batch, N, -1)
        return x_flat

# ----------------------------
# Positional Encoding for Temporal Transformer
# ----------------------------
class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=100):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-np.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe.unsqueeze(0))

    def forward(self, x):
        return x + self.pe[:, :x.size(1), :]

# ----------------------------
# Combined Model: Extractor + Spatial GraphSAGE + Temporal Transformer
# ----------------------------
class SpatialTemporalTransformerModel(nn.Module):
    def __init__(self,
                 num_joints=17,
                 extractor_params=None,
                 spatial_hidden_dims=[128, 256, 128],
                 transformer_hidden_dim=256,
                 transformer_num_heads=8,
                 transformer_num_layers=4,
                 dropout=0.3,
                 num_actions=20,
                 temporal_window=30,
                 skeleton_connections=None):
        super().__init__()
        self.temporal_window = temporal_window
        self.num_joints = num_joints

        # Feature extractor
        if extractor_params is None:
            extractor_params = {
                'num_joints': num_joints,
                'joint_type_embedding_dim': 16,
                'max_angles': 3,
                'use_angles': True,
                'use_velocities': True,
                'use_relative_pos': True,
                'use_confidence': True,
                'use_temporal': False,
                'output_dim': 64,
                'normalize': 'torso',
                'augmentations': {},
                'min_torso_height': 10.0,
                'max_norm_range': 5.0,
                'fallback_to_bbox': True,
            }
        self.extractor = MPOSEFeatureExtractor(**extractor_params)

        # Spatial GraphSAGE (per frame)
        self.spatial_gnn = SpatialGraphSAGE(
            in_dim=64,
            hidden_dims=spatial_hidden_dims,
            num_joints=num_joints,
            skeleton_connections=skeleton_connections or COCO_CONNECTIONS,
            dropout=dropout
        )

        # Temporal Transformer
        self.temporal_transformer = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(
                d_model=spatial_hidden_dims[-1],
                nhead=transformer_num_heads,
                dim_feedforward=spatial_hidden_dims[-1]*4,
                dropout=dropout,
                batch_first=True
            ),
            num_layers=transformer_num_layers
        )
        self.pos_encoder = PositionalEncoding(spatial_hidden_dims[-1], max_len=temporal_window+1)
        self.cls_token = nn.Parameter(torch.randn(1, 1, spatial_hidden_dims[-1]))
        self.dropout = nn.Dropout(dropout)

        # Classifier (same as unified GraphSAGE: LayerNorm, Linear(128), BatchNorm, ReLU, Dropout, Linear(num_actions))
        self.classifier = nn.Sequential(
            nn.LayerNorm(spatial_hidden_dims[-1]),
            nn.Linear(spatial_hidden_dims[-1], 128),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(128, num_actions)
        )

    def forward(self, x_seq):
        """
        x_seq: (batch, T, J, C) with C=5 (x,y,conf,dx,dy)
        """
        batch, T, J, _ = x_seq.shape
        assert T == self.temporal_window

        # 1. Extract per‑frame joint embeddings
        out = self.extractor(x_seq, apply_augmentations=self.training)
        joint_emb = out['joint_embeddings']  # (batch, T, J, D) D=64

        # 2. Apply spatial GraphSAGE to all frames
        joint_emb_flat = joint_emb.view(batch * T, J, -1)
        spatial_out = self.spatial_gnn(joint_emb_flat)  # (batch*T, J, H_spatial)
        # Pool over joints (mean) -> (batch*T, H_spatial)
        frame_vec = spatial_out.mean(dim=1)
        # Reshape to (batch, T, H_spatial)
        frame_seq = frame_vec.view(batch, T, -1)

        # 3. Temporal Transformer
        # Add CLS token
        cls_tokens = self.cls_token.expand(batch, -1, -1)  # (batch, 1, H_spatial)
        x = torch.cat([cls_tokens, frame_seq], dim=1)      # (batch, T+1, H_spatial)
        x = self.pos_encoder(x)                            # add positional encoding
        x = self.temporal_transformer(x)                   # (batch, T+1, H_spatial)
        cls_out = x[:, 0, :]                               # (batch, H_spatial)

        # 4. Classify
        logits = self.classifier(cls_out)
        probs = F.softmax(logits, dim=-1)
        return {'logits': logits, 'probs': probs}

# ----------------------------
# Dataset (same as before)
# ----------------------------
class MPOSESequenceDataset(Dataset):
    def __init__(self, X, y, window_size=30, num_joints=17):
        self.window_size = window_size
        self.num_joints = num_joints
        self.X = []
        self.y = []
        for seq, label in zip(X, y):
            T = seq.shape[0]
            if T < window_size:
                pad = window_size - T
                seq = np.pad(seq, ((0, pad), (0,0), (0,0)), mode='constant')
            elif T > window_size:
                seq = seq[:window_size]
            if seq.shape[1] > num_joints:
                seq = seq[:, :num_joints, :]
            if seq.shape[2] < 5:
                pad = np.zeros((seq.shape[0], seq.shape[1], 5 - seq.shape[2]), dtype=np.float32)
                seq = np.concatenate([seq, pad], axis=-1)
            self.X.append(seq)
            self.y.append(label)

    def __len__(self):
        return len(self.y)

    def __getitem__(self, idx):
        return torch.from_numpy(self.X[idx]).float(), torch.tensor(self.y[idx], dtype=torch.long)

# ----------------------------
# Training utilities
# ----------------------------
def train_epoch(model, loader, optimizer, criterion, device):
    model.train()
    total_loss = 0
    correct = 0
    total = 0
    for seq, labels in tqdm(loader, desc="Training"):
        seq = seq.to(device)
        labels = labels.to(device)
        optimizer.zero_grad()
        out = model(seq)
        loss = criterion(out['logits'], labels)
        loss.backward()
        optimizer.step()
        total_loss += loss.item() * seq.size(0)
        preds = out['logits'].argmax(dim=1)
        correct += (preds == labels).sum().item()
        total += labels.size(0)
    return total_loss / total, correct / total

def evaluate(model, loader, criterion, device):
    model.eval()
    total_loss = 0
    all_preds = []
    all_labels = []
    with torch.no_grad():
        for seq, labels in tqdm(loader, desc="Validating"):
            seq = seq.to(device)
            labels = labels.to(device)
            out = model(seq)
            loss = criterion(out['logits'], labels)
            total_loss += loss.item() * seq.size(0)
            preds = out['logits'].argmax(dim=1)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
    acc = (np.array(all_preds) == np.array(all_labels)).mean()
    bal_acc = balanced_accuracy_score(all_labels, all_preds)
    report = classification_report(all_labels, all_preds, zero_division=0)
    return total_loss / len(loader.dataset), acc, bal_acc, report

# ----------------------------
# Main
# ----------------------------
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--window_size', type=int, default=20)
    parser.add_argument('--batch_size', type=int, default=64)
    parser.add_argument('--epochs', type=int, default=100)
    parser.add_argument('--lr', type=float, default=1e-3)
    parser.add_argument('--weight_decay', type=float, default=1e-4)
    parser.add_argument('--spatial_hidden_dims', type=int, nargs='+', default=[128, 256, 128])
    parser.add_argument('--transformer_hidden_dim', type=int, default=256)
    parser.add_argument('--transformer_num_heads', type=int, default=8)
    parser.add_argument('--transformer_num_layers', type=int, default=4)
    parser.add_argument('--dropout', type=float, default=0.3)
    parser.add_argument('--device', default='cuda' if torch.cuda.is_available() else 'cpu')
    parser.add_argument('--output_dir', default='./spatial_temporal_transformer_baseline')
    parser.add_argument('--split', type=int, default=1)
    parser.add_argument('--num_workers', type=int, default=1)
    args = parser.parse_args()

    device = torch.device(args.device)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    logger.info("Loading MPOSE2021 dataset (PoseNet, with velocities)...")
    data_dir='./data/mpose'
    dataset = mpose.MPOSE(pose_extractor='posenet',
                    split=args.split,
                    preprocess='scale_and_center',
                    velocities=True,
                    remove_zip=False)
    X_train, y_train, X_test, y_test = dataset.get_data()

    y_train = np.array(y_train)
    y_test = np.array(y_test)

    train_dataset = MPOSESequenceDataset(X_train, y_train, args.window_size, num_joints=17)
    test_dataset = MPOSESequenceDataset(X_test, y_test, args.window_size, num_joints=17)

    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True,
                              num_workers=args.num_workers, pin_memory=(args.device=='cuda'))
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False,
                             num_workers=args.num_workers, pin_memory=(args.device=='cuda'))

    num_classes = len(np.unique(y_train))
    logger.info(f"Train samples: {len(train_dataset)}, Test samples: {len(test_dataset)}")
    logger.info(f"Number of action classes: {num_classes}")

    # Class weights
    class_weights = compute_class_weight('balanced', classes=np.unique(y_train), y=y_train)
    class_weights = torch.clamp(torch.tensor(class_weights, dtype=torch.float32), min=0.2, max=2.0).to(device)
    logger.info(f"Class weights: {class_weights}")

    # Model
    model = SpatialTemporalTransformerModel(
        num_joints=17,
        spatial_hidden_dims=args.spatial_hidden_dims,
        transformer_hidden_dim=args.transformer_hidden_dim,
        transformer_num_heads=args.transformer_num_heads,
        transformer_num_layers=args.transformer_num_layers,
        dropout=args.dropout,
        num_actions=num_classes,
        temporal_window=args.window_size,
        skeleton_connections=COCO_CONNECTIONS
    ).to(device)

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    criterion = nn.CrossEntropyLoss(weight=class_weights)

    # Training loop
    best_val_acc = 0
    patience = 30
    patience_counter = 0
    history = {'train_loss': [], 'train_acc': [], 'val_loss': [], 'val_acc': [],'bal_acc':[]}

    for epoch in range(args.epochs):
        logger.info(f"\nEpoch {epoch+1}/{args.epochs}")
        train_loss, train_acc = train_epoch(model, train_loader, optimizer, criterion, device)
        val_loss, val_acc, bal_acc, report = evaluate(model, test_loader, criterion, device)

        history['train_loss'].append(train_loss)
        history['train_acc'].append(train_acc)
        history['val_loss'].append(val_loss)
        history['val_acc'].append(val_acc)
        history['bal_acc'].append(bal_acc)


        logger.info(f"Train loss: {train_loss:.4f}, acc: {train_acc:.4f}")
        logger.info(f"Val loss: {val_loss:.4f}, acc: {val_acc:.4f}, balanced acc: {bal_acc:.4f}")

        if bal_acc > best_val_acc:
            best_val_acc = bal_acc
            torch.save(model.state_dict(), output_dir / 'best_model.pth')
            logger.info("New best model saved.")
            patience_counter = 0
        else:
            patience_counter += 1
            if patience_counter >= patience:
                logger.info(f"Early stopping after {epoch+1} epochs")
                break

    # Save final model and history
    torch.save(model.state_dict(), output_dir / 'final_model.pth')
    with open(output_dir / 'history.json', 'w') as f:
        json.dump(history, f, indent=2)

    # Final evaluation on best model
    model.load_state_dict(torch.load(output_dir / 'best_model.pth'))
    _, final_acc, final_bal_acc, final_report = evaluate(model, test_loader, criterion, device)
    logger.info(f"Final test accuracy: {final_acc:.4f}, balanced: {final_bal_acc:.4f}")
    logger.info(f"Classification report:\n{final_report}")

if __name__ == '__main__':
    main()