#!/usr/bin/env python3
"""
Train a temporal transformer on raw keypoints (with torso normalization) for MPOSE2021.
python transformer_raw.py

"""

import os
import sys
import argparse
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from sklearn.metrics import balanced_accuracy_score, classification_report
from sklearn.utils.class_weight import compute_class_weight
from pathlib import Path
import json
import matplotlib.pyplot as plt
from datetime import datetime
from tqdm import tqdm
import mpose

# -------------------- Dataset (with torso normalization) --------------------
class MPOSESequenceDataset(Dataset):
    """
    Loads sequences of keypoints, applies torso normalization per frame,
    and flattens each frame to a vector of length 51.
    """
    def __init__(self, X, y, seq_len=20, min_torso_height=10.0, max_norm_range=5.0):
        self.seq_len = seq_len
        self.min_torso_height = min_torso_height
        self.max_norm_range = max_norm_range
        self.X = []
        self.y = []
        for seq, label in zip(X, y):
            T = seq.shape[0]
            if T < seq_len:
                pad = seq_len - T
                seq = np.pad(seq, ((0, pad), (0,0), (0,0)), mode='constant')
            elif T > seq_len:
                seq = seq[:seq_len]
            # Normalize each frame
            norm_seq = []
            for frame in seq:
                norm_frame = self.normalize_frame(frame)
                # Flatten to (J*3)
                norm_seq.append(norm_frame.flatten())
            self.X.append(np.stack(norm_seq, axis=0))
            self.y.append(label)

    def normalize_frame(self, keypoints):
        """Torso normalization (same as before) for a single frame (J, 3)."""
        left_shoulder = 5
        right_shoulder = 6
        left_hip = 11
        right_hip = 12

        pos = keypoints[:, :2]
        conf = keypoints[:, 2]

        # Check if torso joints have confidence
        if (conf[left_shoulder] > 0 and conf[right_shoulder] > 0 and
            conf[left_hip] > 0 and conf[right_hip] > 0):
            shoulder_mid = (pos[left_shoulder] + pos[right_shoulder]) / 2
            hip_mid = (pos[left_hip] + pos[right_hip]) / 2
            torso_height = np.linalg.norm(shoulder_mid - hip_mid)
            if torso_height > self.min_torso_height:
                center = (shoulder_mid + hip_mid) / 2
                norm_pos = (pos - center) / torso_height
                norm_pos = np.clip(norm_pos, -self.max_norm_range, self.max_norm_range)
                return np.concatenate([norm_pos, conf[:, np.newaxis]], axis=1)

        # Fallback: bounding‑box normalization
        min_xy = np.min(pos, axis=0)
        max_xy = np.max(pos, axis=0)
        bbox_center = (min_xy + max_xy) / 2
        bbox_size = np.max(max_xy - min_xy)
        if bbox_size > 0:
            norm_pos = (pos - bbox_center) / bbox_size
            norm_pos = np.clip(norm_pos, -self.max_norm_range, self.max_norm_range)
            return np.concatenate([norm_pos, conf[:, np.newaxis]], axis=1)
        else:
            # Very unlikely: return original
            return keypoints

    def __len__(self):
        return len(self.y)

    def __getitem__(self, idx):
        # X[idx] shape (seq_len, 51), y[idx] scalar
        return torch.from_numpy(self.X[idx]).float(), torch.tensor(self.y[idx], dtype=torch.long)

# -------------------- Temporal Transformer Model --------------------
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

class TemporalTransformerClassifier(nn.Module):
    """
    Transformer that processes a sequence of frame‑level vectors.
    Input: (batch, seq_len, input_dim)
    Output: logits over num_classes.
    """
    def __init__(self, input_dim, hidden_dim, num_classes, seq_len,
                 num_heads=8, num_layers=4, dropout=0.3):
        super().__init__()
        self.input_proj = nn.Linear(input_dim, hidden_dim)
        # Important: max_len must be >= seq_len+1 to accommodate CLS token
        self.pos_encoder = PositionalEncoding(hidden_dim, max_len=seq_len+1)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden_dim, nhead=num_heads,
            dim_feedforward=hidden_dim*4, dropout=dropout, batch_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.cls_token = nn.Parameter(torch.randn(1, 1, hidden_dim))
        # Improved classifier (like GraphSAGE)
        self.classifier = nn.Sequential(
            nn.LayerNorm(hidden_dim),
            nn.Linear(hidden_dim, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(128, num_classes)
        )
        self.seq_len = seq_len

    def forward(self, x):
        # x: (batch, seq_len, input_dim)
        B = x.size(0)
        x = self.input_proj(x)                     # (B, seq_len, hidden)
        cls_tokens = self.cls_token.expand(B, -1, -1)
        x = torch.cat([cls_tokens, x], dim=1)      # (B, seq_len+1, hidden)
        x = self.pos_encoder(x)
        x = self.transformer(x)
        cls_out = x[:, 0, :]                       # (B, hidden)
        logits = self.classifier(cls_out)
        return logits

# -------------------- Helper functions --------------------
def plot_training_history(history, save_path):
    epochs = range(1, len(history['train_loss']) + 1)
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))
    ax1.plot(epochs, history['train_loss'], 'b-', label='Train Loss')
    ax1.plot(epochs, history['val_loss'], 'r-', label='Val Loss')
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Loss')
    ax1.set_title('Loss')
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    ax2.plot(epochs, history['train_acc'], 'b-', label='Train Acc')
    ax2.plot(epochs, history['val_acc'], 'r-', label='Val Acc')
    if 'val_bal_acc' in history and history['val_bal_acc']:
        ax2.plot(epochs, history['val_bal_acc'], 'g--', label='Balanced Acc')
    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('Accuracy')
    ax2.set_title('Accuracy')
    ax2.legend()
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()
    print(f"Training curves saved to {save_path}")

# -------------------- Main --------------------
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--seq_len', type=int, default=20, help='Number of frames per sequence')
    parser.add_argument('--batch_size', type=int, default=64)
    parser.add_argument('--epochs', type=int, default=100)
    parser.add_argument('--lr', type=float, default=1e-3)
    parser.add_argument('--hidden_dim', type=int, default=128)
    parser.add_argument('--num_heads', type=int, default=8)
    parser.add_argument('--num_layers', type=int, default=4)
    parser.add_argument('--dropout', type=float, default=0.3)
    parser.add_argument('--device', default='cuda' if torch.cuda.is_available() else 'cpu')
    parser.add_argument('--patience', type=int, default=30)
    parser.add_argument('--output_dir', default='./transformer_raw_mpose')
    parser.add_argument('--split', type=int, default=1, help='MPOSE2021 split (1,2,3)')
    args = parser.parse_args()

    device = torch.device(args.device)
    print(f"Using device: {device}")

    # Create output directory
    exp_name = f"transformer_raw_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    output_dir = Path(args.output_dir) / exp_name
    output_dir.mkdir(parents=True, exist_ok=True)
    print(f"Results will be saved to {output_dir}")

    # Save arguments
    with open(output_dir / 'args.json', 'w') as f:
        json.dump(vars(args), f, indent=2)

    # -------------------- Load MPOSE2021 --------------------
    import mpose as MPOSE


    print("Loading MPOSE2021 dataset...")
    data_dir='./data/mpose'
    dataset =mpose.MPOSE(pose_extractor='posenet',
                    split=args.split,
                    preprocess='scale_and_center',
                    remove_zip=False)
    X_train, y_train, X_test, y_test = dataset.get_data()
    num_classes = len(np.unique(y_train))
    print(f"Number of action classes: {num_classes}")
    print(f"Train samples: {len(X_train)}, Test samples: {len(X_test)}")

    # Create datasets and loaders
    train_dataset = MPOSESequenceDataset(X_train, y_train, seq_len=args.seq_len)
    test_dataset = MPOSESequenceDataset(X_test, y_test, seq_len=args.seq_len)
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=0)
    val_loader   = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False, num_workers=0)

    # -------------------- Model --------------------
    input_dim = 17 * 3  # 51
    model = TemporalTransformerClassifier(
        input_dim=input_dim,
        hidden_dim=args.hidden_dim,
        num_classes=num_classes,
        seq_len=args.seq_len,
        num_heads=args.num_heads,
        num_layers=args.num_layers,
        dropout=args.dropout
    ).to(device)

    # Class weights (balanced)
    class_weights = compute_class_weight('balanced', classes=np.unique(y_train), y=y_train)
    class_weights = torch.clamp(torch.tensor(class_weights, dtype=torch.float32), min=0.2, max=2.0).to(device)
    criterion = nn.CrossEntropyLoss(weight=class_weights)

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)

    # -------------------- Training loop --------------------
    best_bal_acc = 0.0
    best_model_state = None
    patience_counter = 0
    history = {
        'train_loss': [],
        'train_acc': [],
        'val_loss': [],
        'val_acc': [],
        'val_bal_acc': []
    }

    for epoch in range(args.epochs):
        # Training
        model.train()
        total_loss = 0
        correct = 0
        total = 0
        loop = tqdm(train_loader, desc=f'Epoch {epoch+1}/{args.epochs}', leave=False)
        for seq, labels in loop:
            seq = seq.to(device)          # (B, T, 51)
            labels = labels.to(device)
            optimizer.zero_grad()
            logits = model(seq)
            loss = criterion(logits, labels)
            loss.backward()
            optimizer.step()

            total_loss += loss.item() * seq.size(0)
            preds = logits.argmax(dim=1)
            correct += (preds == labels).sum().item()
            total += labels.size(0)
            loop.set_postfix(loss=loss.item(), acc=correct/total)

        train_loss = total_loss / len(train_loader.dataset)
        train_acc = correct / total

        # Validation
        model.eval()
        val_loss = 0
        all_preds = []
        all_labels = []
        with torch.no_grad():
            for seq, labels in val_loader:
                seq = seq.to(device)
                labels = labels.to(device)
                logits = model(seq)
                loss = criterion(logits, labels)
                val_loss += loss.item() * seq.size(0)
                preds = logits.argmax(dim=1)
                all_preds.extend(preds.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())
        val_loss /= len(val_loader.dataset)
        val_acc = (np.array(all_preds) == np.array(all_labels)).mean()
        bal_acc = balanced_accuracy_score(all_labels, all_preds)

        # Record
        history['train_loss'].append(train_loss)
        history['train_acc'].append(train_acc)
        history['val_loss'].append(val_loss)
        history['val_acc'].append(val_acc)
        history['val_bal_acc'].append(bal_acc)

        if (epoch+1) % 5 == 0 or epoch == 0:
            print(f"Epoch {epoch+1:3d}: Train Loss={train_loss:.4f}, Train Acc={train_acc:.4f} | "
                  f"Val Loss={val_loss:.4f}, Val Acc={val_acc:.4f}, Bal Acc={bal_acc:.4f}")

        # Save best model
        if bal_acc > best_bal_acc:
            best_bal_acc = bal_acc
            best_model_state = model.state_dict().copy()
            patience_counter = 0
            print(f"  -> New best model! Balanced Acc = {best_bal_acc:.4f}")
        else:
            patience_counter += 1
            if patience_counter >= args.patience:
                print(f"Early stopping at epoch {epoch+1}")
                break

    # Save best model and history
    torch.save(best_model_state, output_dir / 'best_model.pth')
    with open(output_dir / 'training_history.json', 'w') as f:
        json.dump(history, f)
    plot_training_history(history, output_dir / 'training_curves.png')

    # Final evaluation on best model
    model.load_state_dict(best_model_state)
    model.eval()
    all_preds = []
    all_labels = []
    with torch.no_grad():
        for seq, labels in val_loader:
            seq = seq.to(device)
            labels = labels.to(device)
            logits = model(seq)
            preds = logits.argmax(dim=1)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
    final_acc = (np.array(all_preds) == np.array(all_labels)).mean()
    final_bal_acc = balanced_accuracy_score(all_labels, all_preds)
    print(f"\nFinal test accuracy: {final_acc:.4f}, balanced accuracy: {final_bal_acc:.4f}")
    print("\nClassification report:")
    print(classification_report(all_labels, all_preds, zero_division=0))

    print(f"\nAll results saved to {output_dir}")

if __name__ == '__main__':
    main()