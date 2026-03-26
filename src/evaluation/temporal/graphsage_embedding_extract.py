#!/usr/bin/env python3
"""
Train spatio‑temporal GraphSAGE with MPOSEFeatureExtractor on MPOSE2021.
Uses the 5‑channel MPOSE dataset (x,y,conf,dx,dy).
python src\evaluation\temporal\graphsage_with_preprocessing.py --window_size 20 --batch_size 64 --epochs 100 --num_workers 2
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
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
import json
from pathlib import Path
import logging
from mpose import MPOSE

# Add project root
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../..')))
from src.encoder.mpose_encoder import MPOSEFeatureExtractor

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Skeleton connections for COCO 17 joints
COCO_CONNECTIONS = [
    (0,1),(0,2),(1,3),(2,4),(5,6),(5,7),(7,9),(6,8),(8,10),
    (5,11),(6,12),(11,12),(11,13),(13,15),(12,14),(14,16)
]

# Multi‑head attention pooling
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

# Spatio‑temporal GraphSAGE with MPOSEFeatureExtractor
class SpatioTemporalGraphSAGEWithExtractor(nn.Module):
    def __init__(self,
                 num_joints: int = 17,
                 joint_embedding_dim: int = 64,
                 graphsage_hidden_dims: list = [128, 256, 128],
                 num_actions: int = 20,
                 dropout: float = 0.3,
                 skeleton_connections: list = None,
                 temporal_window: int = 30,
                 num_attention_heads: int = 4,
                 extractor_params: dict = None):
        super().__init__()
        self.temporal_window = temporal_window
        self.num_joints = num_joints

        # MPOSE feature extractor (5‑channel input)
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
                'output_dim': joint_embedding_dim,
                'normalize': 'torso',
                'augmentations': {},
                'min_torso_height': 10.0,
                'max_norm_range': 5.0,
                'fallback_to_bbox': True,
            }
        self.extractor = MPOSEFeatureExtractor(**extractor_params)

        # Skeleton connections
        if skeleton_connections is None:
            skeleton_connections = COCO_CONNECTIONS
        self.skeleton_connections = skeleton_connections

        # Build spatio‑temporal edge index
        self.register_buffer('edge_index', self._build_spatio_temporal_edges())

        # GraphSAGE layers
        dims = [joint_embedding_dim] + graphsage_hidden_dims
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
        spatial_edges = []
        for t in range(T):
            offset = t * num_joints
            for (i, j) in self.skeleton_connections:
                spatial_edges.append((offset + i, offset + j))
                spatial_edges.append((offset + j, offset + i))
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
        batch_size, T, J, C = x_seq.shape
        assert T == self.temporal_window, f"Expected {self.temporal_window} frames, got {T}"
        # Extract joint embeddings (batch, T, J, D)
        out = self.extractor(x_seq, apply_augmentations=self.training)
        joint_emb = out['joint_embeddings']  # (B, T, J, D)
        node_feats = joint_emb.view(batch_size, T * J, -1)  # (B, N, D)

        # Build batched graph
        N = T * J
        x_batch = node_feats.view(-1, node_feats.size(-1))  # (B*N, D)
        offsets = torch.arange(batch_size, device=x_seq.device) * N
        edge_index_batch = self.edge_index.repeat(1, batch_size)
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
        pooled = torch.cat(pooled, dim=0)   # (batch, embedding_dim)

        logits = self.classifier(pooled)
        probs = F.softmax(logits, dim=-1)
        return {'logits': logits, 'probs': probs, 'embedding': pooled, 'joint_embeddings': joint_emb}

# Dataset (MPOSE with 5 channels)
class MPOSESequenceDataset(Dataset):
    def __init__(self, X, y, window_size=30):
        self.window_size = window_size
        self.X = []
        self.y = []
        for seq, label in zip(X, y):
            T = seq.shape[0]
            if T < window_size:
                pad = window_size - T
                seq = np.pad(seq, ((0, pad), (0,0), (0,0)), mode='constant')
            elif T > window_size:
                seq = seq[:window_size]
            # Ensure we have 5 channels (if not, pad with zeros)
            if seq.shape[2] < 5:
                pad = np.zeros((seq.shape[0], seq.shape[1], 5 - seq.shape[2]), dtype=np.float32)
                seq = np.concatenate([seq, pad], axis=-1)
            self.X.append(seq)
            self.y.append(label)

    def __len__(self):
        return len(self.y)

    def __getitem__(self, idx):
        return torch.from_numpy(self.X[idx]).float(), torch.tensor(self.y[idx], dtype=torch.long)

# Training utilities
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

def extract_embeddings(model, loader, device):
    """Extract embeddings and labels from all samples in loader."""
    model.eval()
    embeddings = []
    labels = []
    with torch.no_grad():
        for seq, label in tqdm(loader, desc="Extracting embeddings"):
            seq = seq.to(device)
            out = model(seq)
            embeddings.append(out['embedding'].cpu().numpy())
            labels.append(label.numpy())
    embeddings = np.concatenate(embeddings, axis=0)
    labels = np.concatenate(labels, axis=0)
    return embeddings, labels

def visualize_embeddings(embeddings, labels, output_dir, class_names=None, pca_components=2, run_tsne=True):
    """Run PCA and t‑SNE, save plots."""
    # PCA
    pca = PCA(n_components=pca_components)
    emb_pca = pca.fit_transform(embeddings)
    plt.figure(figsize=(10, 8))
    scatter = plt.scatter(emb_pca[:, 0], emb_pca[:, 1], c=labels, cmap='tab20', s=5, alpha=0.6)
    plt.colorbar(scatter, label='Class ID')
    plt.title(f'PCA of GraphSAGE embeddings\nExplained variance: {pca.explained_variance_ratio_.sum():.2f}')
    plt.xlabel('PC1')
    plt.ylabel('PC2')
    plt.savefig(output_dir / 'pca_embeddings.png', dpi=150, bbox_inches='tight')
    plt.close()
    logger.info(f"PCA plot saved to {output_dir / 'pca_embeddings.png'}")

    if run_tsne:
        # Subsample if too many points
        if len(embeddings) > 5000:
            idx = np.random.choice(len(embeddings), 5000, replace=False)
            emb_tsne = embeddings[idx]
            labels_tsne = labels[idx]
        else:
            emb_tsne = embeddings
            labels_tsne = labels
        tsne = TSNE(n_components=2, random_state=42, perplexity=30, n_iter=1000)
        emb_tsne_2d = tsne.fit_transform(emb_tsne)
        plt.figure(figsize=(10, 8))
        scatter = plt.scatter(emb_tsne_2d[:, 0], emb_tsne_2d[:, 1], c=labels_tsne, cmap='tab20', s=5, alpha=0.6)
        plt.colorbar(scatter, label='Class ID')
        plt.title('t‑SNE of GraphSAGE embeddings')
        plt.savefig(output_dir / 'tsne_embeddings.png', dpi=150, bbox_inches='tight')
        plt.close()
        logger.info(f"t‑SNE plot saved to {output_dir / 'tsne_embeddings.png'}")

def get_class_names(dataset):
    """Retrieve class names from MPOSE dataset."""
    labels_info = dataset.get_labels()
    if isinstance(labels_info, dict):
        # Invert to label -> name
        class_names = [None] * (max(labels_info.values()) + 1)
        for name, idx in labels_info.items():
            class_names[idx] = name
        return class_names
    else:
        return None

# Main
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
    parser.add_argument('--output_dir', default='./graphsage_temporal')
    parser.add_argument('--split', type=int, default=1)
    parser.add_argument('--num_workers', type=int, default=4)
    parser.add_argument('--extract_embeddings', action='store_true', help='Extract embeddings after training and visualize with PCA/t‑SNE')
    parser.add_argument('--pca_components', type=int, default=2, help='Number of PCA components')
    parser.add_argument('--tsne', action='store_true', default=True, help='Run t‑SNE (default True)')
    args = parser.parse_args()

    device = torch.device(args.device)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Load MPOSE2021 (PoseNet, 5 channels)
    logger.info("Loading MPOSE2021 dataset (PoseNet, with velocities)...")
    data_dir='./data/mpose'
    dataset = MPOSE(pose_extractor='posenet',
                split=args.split,
                preprocess='scale_and_center',
                velocities=True,
                remove_zip=False)
    X_train, y_train, X_test, y_test = dataset.get_data()

    # Convert labels to numpy
    y_train = np.array(y_train)
    y_test = np.array(y_test)

    # Get class names
    class_names = get_class_names(dataset)

    # Create datasets
    train_dataset = MPOSESequenceDataset(X_train, y_train, args.window_size)
    test_dataset = MPOSESequenceDataset(X_test, y_test, args.window_size)

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
    model = SpatioTemporalGraphSAGEWithExtractor(
        num_joints=17,
        joint_embedding_dim=64,
        graphsage_hidden_dims=args.hidden_dims,
        num_actions=num_classes,
        dropout=args.dropout,
        temporal_window=args.window_size,
        num_attention_heads=args.num_heads,
        skeleton_connections=COCO_CONNECTIONS
    ).to(device)

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    criterion = nn.CrossEntropyLoss(weight=class_weights)

    # Training loop
    best_val_acc = 0
    patience = 30
    patience_counter = 0
    history = {'train_loss': [], 'train_acc': [], 'val_loss': [], 'val_acc': [],'bal_acc'=[]}

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

    # Final evaluation
    model.load_state_dict(torch.load(output_dir / 'best_model.pth'))
    _, final_acc, final_bal_acc, final_report = evaluate(model, test_loader, criterion, device)
    logger.info(f"Final test accuracy: {final_acc:.4f}, balanced: {final_bal_acc:.4f}")
    logger.info(f"Classification report:\n{final_report}")

    # Extract and visualize embeddings if requested
    if args.extract_embeddings:
        logger.info("Extracting embeddings from test set...")
        embeddings, labels = extract_embeddings(model, test_loader, device)
        logger.info(f"Embeddings shape: {embeddings.shape}")
        visualize_embeddings(embeddings, labels, output_dir, class_names=class_names,
                             pca_components=args.pca_components, run_tsne=args.tsne)

if __name__ == '__main__':
    main()