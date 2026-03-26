#!/usr/bin/env python3
"""
Load a trained GraphSAGE model and visualize test embeddings with PCA and t‑SNE.
Compute silhouette score, Davies‑Bouldin, and Calinski‑Harabasz.
Usage:
python src/utils/embedding_extract_temporal.py \
    --model_path outputs/temporal/graphsage_temporal/best_model.pth \
    --split 1 \
    --embedding_key joint_embeddings \
    --pooling last_frame
"""

import argparse
import numpy as np
import torch
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.metrics import silhouette_score, davies_bouldin_score, calinski_harabasz_score
from torch.utils.data import DataLoader
from pathlib import Path
import sys
import os

# Add project root to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))
from src.encoder.mpose_encoder import MPOSEFeatureExtractor
from src.evaluation.temporal.graphsage_with_preprocessing import SpatioTemporalGraphSAGEWithExtractor
from src.evaluation.temporal.graphsage_with_preprocessing import MPOSESequenceDataset
import mpose

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_path', required=True, help='Path to best_model.pth')
    parser.add_argument('--window_size', type=int, default=20, help='Temporal window used during training')
    parser.add_argument('--batch_size', type=int, default=64)
    parser.add_argument('--split', type=int, default=1, help='MPOSE2021 split (1,2,3)')
    parser.add_argument('--device', default='cuda' if torch.cuda.is_available() else 'cpu')
    parser.add_argument('--output_dir', default='./embeddings_vis', help='Directory to save plots and results')
    parser.add_argument('--no_tsne', action='store_true', help='Skip t‑SNE (faster)')
    parser.add_argument('--embedding_key', default='joint_embeddings',
                        help='Key in model output to use as embedding (e.g., "joint_embeddings", "logits", "probs")')
    parser.add_argument('--pooling', default='last_frame', choices=['mean', 'last_frame', 'first_frame'],
                        help='Only used if embedding_key is "joint_embeddings". How to pool: '
                             '"mean": average over joints and frames, '
                             '"last_frame": use final frame and average over joints, '
                             '"first_frame": use first frame and average over joints')
    args = parser.parse_args()

    device = torch.device(args.device)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Load MPOSE2021 test set
    dataset = mpose.MPOSE(pose_extractor='posenet',
                          split=args.split,
                          preprocess='scale_and_center',
                          velocities=True,
                          remove_zip=False)
    _, _, X_test, y_test = dataset.get_data()
    y_test = np.array(y_test)
    num_classes = len(np.unique(y_test))
    print(f"Test set size: {len(X_test)}, number of classes: {num_classes}")

    # Create dataset and loader
    test_dataset = MPOSESequenceDataset(X_test, y_test, args.window_size)
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False, num_workers=0)

    # Build model (same architecture as training)
    model = SpatioTemporalGraphSAGEWithExtractor(
        num_joints=17,
        joint_embedding_dim=64,
        graphsage_hidden_dims=[128, 256, 128],
        num_actions=num_classes,
        dropout=0.3,
        temporal_window=args.window_size,
        num_attention_heads=4,
        skeleton_connections=None  # uses default COCO connections
    ).to(device)

    # Load trained weights
    checkpoint = torch.load(args.model_path, map_location=device)
    model.load_state_dict(checkpoint)
    model.eval()

    # Extract embeddings
    embeddings = []
    labels = []
    used_key = args.embedding_key
    with torch.no_grad():
        for seq, lbl in test_loader:
            seq = seq.to(device)
            out = model(seq)
            if not embeddings:  # first batch: print all output keys
                print("Output keys from model:", list(out.keys()))
                if args.embedding_key not in out:
                    print(f"Warning: Requested embedding key '{args.embedding_key}' not found. Available: {list(out.keys())}")
                    # Fallback to first suitable key
                    if 'joint_embeddings' in out:
                        used_key = 'joint_embeddings'
                    elif 'logits' in out:
                        used_key = 'logits'
                    elif 'probs' in out:
                        used_key = 'probs'
                    else:
                        raise KeyError(f"No suitable embedding key found. Available: {list(out.keys())}")
                    print(f"Falling back to '{used_key}'")
                else:
                    used_key = args.embedding_key

            data = out[used_key]

            if used_key == 'joint_embeddings':
                # data shape: (B, T, J, D)
                data_np = data.cpu().numpy()
                if args.pooling == 'mean':
                    emb = data_np.mean(axis=(1, 2))
                elif args.pooling == 'last_frame':
                    emb = data_np[:, -1, :, :].mean(axis=1)
                elif args.pooling == 'first_frame':
                    emb = data_np[:, 0, :, :].mean(axis=1)
                else:
                    raise ValueError(f"Unknown pooling: {args.pooling}")
            else:
                # Assume it's already a vector embedding per sample (B, D)
                emb = data.cpu().numpy()

            embeddings.append(emb)
            labels.append(lbl.numpy())

    embeddings = np.concatenate(embeddings, axis=0)
    labels = np.concatenate(labels, axis=0)
    print(f"Extracted embeddings shape: {embeddings.shape} from key '{used_key}'")
    if used_key == 'joint_embeddings':
        print(f"Pooling method: {args.pooling}")

    # ----- Clustering metrics -----
    if num_classes < 2:
        print("Skipping clustering metrics because there is only one class.")
    else:
        # Silhouette score
        try:
            sil_score = silhouette_score(embeddings, labels)
            print(f"Silhouette Score: {sil_score:.4f}")
        except Exception as e:
            print(f"Could not compute silhouette score: {e}")
            sil_score = None

        # Davies-Bouldin (using standard hyphen)
        try:
            db_score = davies_bouldin_score(embeddings, labels)
            print(f"Davies-Bouldin Score: {db_score:.4f}")
        except Exception as e:
            print(f"Could not compute Davies-Bouldin score: {e}")
            db_score = None

        # Calinski-Harabasz
        try:
            ch_score = calinski_harabasz_score(embeddings, labels)
            print(f"Calinski-Harabasz Score: {ch_score:.2f}")
        except Exception as e:
            print(f"Could not compute Calinski-Harabasz score: {e}")
            ch_score = None

        # Save metrics to a text file with UTF-8 encoding
        metrics_file = output_dir / f"metrics_{used_key}.txt"
        if used_key == 'joint_embeddings':
            metrics_file = output_dir / f"metrics_{used_key}_{args.pooling}.txt"
        # Use utf-8 encoding to handle any special characters
        with open(metrics_file, 'w', encoding='utf-8') as f:
            f.write(f"Embedding key: {used_key}\n")
            if used_key == 'joint_embeddings':
                f.write(f"Pooling method: {args.pooling}\n")
            f.write(f"Number of samples: {len(embeddings)}\n")
            f.write(f"Number of classes: {num_classes}\n")
            if sil_score is not None:
                f.write(f"Silhouette Score: {sil_score:.4f}\n")
            if db_score is not None:
                f.write(f"Davies-Bouldin Score: {db_score:.4f}\n")   # regular hyphen
            if ch_score is not None:
                f.write(f"Calinski-Harabasz Score: {ch_score:.2f}\n")
        print(f"Metrics saved to {metrics_file}")

    # ----- PCA -----
    pca = PCA(n_components=2)
    emb_pca = pca.fit_transform(embeddings)
    plt.figure(figsize=(10, 8))
    scatter = plt.scatter(emb_pca[:, 0], emb_pca[:, 1], c=labels, cmap='tab20', s=5, alpha=0.6)
    plt.colorbar(scatter, label='Class ID')
    title = f'PCA of GraphSAGE embeddings (key={used_key})'
    if used_key == 'joint_embeddings':
        title += f', pooling={args.pooling}'
    title += f'\nExplained variance: {pca.explained_variance_ratio_.sum():.2f}'
    plt.title(title)
    plt.xlabel('PC1')
    plt.ylabel('PC2')
    outfile = output_dir / f"pca_{used_key}"
    if used_key == 'joint_embeddings':
        outfile = output_dir / f"pca_{used_key}_{args.pooling}.png"
    else:
        outfile = output_dir / f"pca_{used_key}.png"
    plt.savefig(outfile, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"PCA plot saved to {outfile}")

    # ----- t‑SNE -----
    if not args.no_tsne:
        # Subsample if too many points (t‑SNE is slow for large datasets)
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
        title = f't‑SNE of GraphSAGE embeddings (key={used_key})'
        if used_key == 'joint_embeddings':
            title += f', pooling={args.pooling}'
        plt.title(title)
        outfile = output_dir / f"tsne_{used_key}"
        if used_key == 'joint_embeddings':
            outfile = output_dir / f"tsne_{used_key}_{args.pooling}.png"
        else:
            outfile = output_dir / f"tsne_{used_key}.png"
        plt.savefig(outfile, dpi=150, bbox_inches='tight')
        plt.close()
        print(f"t‑SNE plot saved to {outfile}")

if __name__ == '__main__':
    main()