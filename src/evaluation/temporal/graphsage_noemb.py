#!/usr/bin/env python3
"""
Train spatio‑temporal GraphSAGE on raw normalized keypoints (no feature extractor).
Uses torso normalization per frame, then projects to 64‑dim node features.
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

# Add project root to import helpers (if needed)
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))
# We don't import extractor; we'll define our own normalization and projection.

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Skeleton connections for COCO 17 joints
COCO_CONNECTIONS = [
    (0,1),(0,2),(1,3),(2,4),(5,6),(5,7),(7,9),(6,8),(8,10),
    (5,11),(6,12),(11,12),(11,13),(13,15),(12,14),(14,16)
]

# ----------------------------------------------------------------------
# Multi‑head Attention Pooling
# ----------------------------------------------------------------------
class MultiHeadPooling(nn.Module):
    def __init__(self, in_dim: int, num_heads: int = 4):
        super().__init__()
        self.num_heads = num_heads
        self.attn = nn.ModuleList([nn.Linear(in_dim, 1) for _ in range(num_heads)])

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        heads = []
        for a in self.attn:
            scores = F.softmax(a(x), dim=1)
            head = (x * scores).sum(dim=1)
            heads.append(head)
        return torch.cat(heads, dim=1)

# ----------------------------------------------------------------------
# Spatio‑temporal GraphSAGE on raw keypoints
# ----------------------------------------------------------------------
class SpatioTemporalGraphSAGERaw(nn.Module):
    def __init__(self,
                 num_joints: int = 17,
                 node_feature_dim: int = 3,          # (x,y,conf) after normalization
                 projected_dim: int = 64,            # project to this dim before GraphSAGE
                 graphsage_hidden_dims: list = [128, 256, 128],
                 num_actions: int = 20,
                 dropout: float = 0.3,
                 skeleton_connections: list = None,
                 temporal_window: int = 30,
                 num_attention_heads: int = 4):
        super().__init__()
        self.temporal_window = temporal_window
        self.num_joints = num_joints

        # Linear projection from raw normalized coordinates to node features
        self.projection = nn.Linear(node_feature_dim, projected_dim)

        # Skeleton connections
        if skeleton_connections is None:
            skeleton_connections = COCO_CONNECTIONS
        self.skeleton_connections = skeleton_connections

        # Build spatio‑temporal edge index
        self.register_buffer('edge_index', self._build_spatio_temporal_edges())

        # GraphSAGE layers
        dims = [projected_dim] + graphsage_hidden_dims
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
        self.node_attention = MultiHeadPooling(dims[-1], num_heads=num_attention_heads)
        self.classifier = nn.Sequential(
            nn.Linear(dims[-1] * num_attention_heads, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(128, num_actions)
        )

    def _build_spatio_temporal_edges(self):
        num_joints = self.num_joints
        T = self.temporal_window

        # Spatial edges per frame
        spatial_edges = []
        for t in range(T):
            offset = t * num_joints
            for (i, j) in self.skeleton_connections:
                spatial_edges.append((offset + i, offset + j))
                spatial_edges.append((offset + j, offset + i))
        # Temporal edges (same joint across consecutive frames)
        temporal_edges = []
        for t in range(T - 1):
            for j in range(num_joints):
                i = t * num_joints + j
                j_next = (t + 1) * num_joints + j
                temporal_edges.append((i, j_next))
                temporal_edges.append((j_next, i))

        all_edges = spatial_edges + temporal_edges
        edge_index = torch.tensor(all_edges, dtype=torch.long).t().contiguous()
        return edge_index

    def forward(self, x_seq):
        """
        x_seq: (batch, T, J, C) where C = 3 (x, y, confidence) – already normalized.
        """
        batch, T, J, C = x_seq.shape
        assert T == self.temporal_window
        assert C == 3, f"Expected 3 channels (normalized x,y,conf), got {C}"

        # Project raw features to node dimension
        x_flat = x_seq.view(batch * T, J, C)               # (B*T, J, C)
        node_feats = self.projection(x_flat)               # (B*T, J, D)
        node_feats = node_feats.view(batch, T * J, -1)     # (B, N, D)

        # Build batched graph
        N = T * J
        x_batch = node_feats.view(-1, node_feats.size(-1)) # (B*N, D)
        offsets = torch.arange(batch, device=x_seq.device) * N
        edge_index_batch = self.edge_index.repeat(1, batch)
        offsets_per_edge = offsets.repeat_interleave(self.edge_index.size(1))
        edge_index_batch = edge_index_batch + offsets_per_edge.unsqueeze(0)

        # GraphSAGE layers
        x = x_batch
        for conv, norm, proj in zip(self.convs, self.norms, self.projs):
            x_res = proj(x)
            x = conv(x, edge_index_batch)
            x = norm(x)
            x = F.relu(x)
            x = x + x_res
            x = F.dropout(x, p=self.dropout, training=self.training)

        # Split and pool
        node_emb_list = x.split(N, dim=0)
        pooled = []
        for node_emb in node_emb_list:
            attn_out = self.node_attention(node_emb.unsqueeze(0))
            pooled.append(attn_out)
        pooled = torch.cat(pooled, dim=0)

        logits = self.classifier(pooled)
        probs = F.softmax(logits, dim=-1)
        return {'logits': logits, 'probs': probs}

# ----------------------------------------------------------------------
# Dataset with torso normalization (same as raw transformer)
# ----------------------------------------------------------------------
class MPOSESequenceDataset(Dataset):
    def __init__(self, X, y, window_size=30, min_torso_height=10.0, max_norm_range=5.0, num_joints=17):
        self.window_size = window_size
        self.min_torso_height = min_torso_height
        self.max_norm_range = max_norm_range
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
            # Truncate joints
            if seq.shape[1] > num_joints:
                seq = seq[:, :num_joints, :]
            # Normalize each frame
            norm_seq = []
            for frame in seq:
                norm_frame = self.normalize_frame(frame)
                # Keep only first 3 channels (x,y,conf)
                if norm_frame.shape[1] > 3:
                    norm_frame = norm_frame[:, :3]
                norm_seq.append(norm_frame)
            self.X.append(np.stack(norm_seq, axis=0))
            self.y.append(label)

    def normalize_frame(self, keypoints):
        """Torso normalization for a single frame (J, C)."""
        left_shoulder = 5
        right_shoulder = 6
        left_hip = 11
        right_hip = 12

        pos = keypoints[:, :2]
        conf = keypoints[:, 2:3]   # confidence (third channel)

        # Torso normalization
        if (conf[left_shoulder] > 0 and conf[right_shoulder] > 0 and
            conf[left_hip] > 0 and conf[right_hip] > 0):
            shoulder_mid = (pos[left_shoulder] + pos[right_shoulder]) / 2
            hip_mid = (pos[left_hip] + pos[right_hip]) / 2
            torso_height = np.linalg.norm(shoulder_mid - hip_mid)
            if torso_height > self.min_torso_height:
                center = (shoulder_mid + hip_mid) / 2
                norm_pos = (pos - center) / torso_height
                norm_pos = np.clip(norm_pos, -self.max_norm_range, self.max_norm_range)
                return np.concatenate([norm_pos, conf], axis=1)

        # Fallback: bounding‑box normalization
        min_xy = np.min(pos, axis=0)
        max_xy = np.max(pos, axis=0)
        bbox_center = (min_xy + max_xy) / 2
        bbox_size = np.max(max_xy - min_xy)
        if bbox_size > 0:
            norm_pos = (pos - bbox_center) / bbox_size
            norm_pos = np.clip(norm_pos, -self.max_norm_range, self.max_norm_range)
            return np.concatenate([norm_pos, conf], axis=1)
        else:
            return keypoints[:, :3]   # return original if all fails

    def __len__(self):
        return len(self.y)

    def __getitem__(self, idx):
        return torch.from_numpy(self.X[idx]).float(), torch.tensor(self.y[idx], dtype=torch.long)
# ----------------------------------------------------------------------
# Training utilities
# ----------------------------------------------------------------------
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

# ----------------------------------------------------------------------
# Main
# ----------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--window_size', type=int, default=20)
    parser.add_argument('--batch_size', type=int, default=64)
    parser.add_argument('--epochs', type=int, default=100)
    parser.add_argument('--lr', type=float, default=1e-3)
    parser.add_argument('--weight_decay', type=float, default=1e-4)
    parser.add_argument('--hidden_dims', type=int, nargs='+', default=[128, 256, 128])
    parser.add_argument('--num_heads', type=int, default=4)
    parser.add_argument('--dropout', type=float, default=0.3)
    parser.add_argument('--device', default='cuda' if torch.cuda.is_available() else 'cpu')
    parser.add_argument('--output_dir', default='./graphsage_raw')
    parser.add_argument('--split', type=int, default=1)
    parser.add_argument('--num_workers', type=int, default=1)
    args = parser.parse_args()

    device = torch.device(args.device)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Load MPOSE2021 raw keypoints (no velocities)
    logger.info("Loading MPOSE2021 dataset (raw keypoints, no velocities)...")
    dataset = mpose.MPOSE(pose_extractor='posenet',
                          split=args.split,
                          preprocess=None,          # we do our own normalization
                          velocities=False,
                          remove_zip=False)
    X_train, y_train, X_test, y_test = dataset.get_data()

    y_train = np.array(y_train)
    y_test = np.array(y_test)

    # Create datasets (will apply torso normalization)
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
    model = SpatioTemporalGraphSAGERaw(
        num_joints=17,
        node_feature_dim=3,
        projected_dim=64,
        graphsage_hidden_dims=args.hidden_dims,
        num_actions=num_classes,
        dropout=args.dropout,
        skeleton_connections=COCO_CONNECTIONS,
        temporal_window=args.window_size,
        num_attention_heads=args.num_heads
    ).to(device)

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    criterion = nn.CrossEntropyLoss(weight=class_weights)

    # Training loop
    best_val_acc = 0
    patience = 10
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