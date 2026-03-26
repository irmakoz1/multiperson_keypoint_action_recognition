
#!/usr/bin/env python3
"""
Explore MPOSE2021 dataset statistics, class distributions, and imbalance metrics.
Uses the actual class names from the dataset.
"""
from mpose import MPOSE
from mpose.utils import plot_pose
import os
import argparse
import numpy as np
import matplotlib.pyplot as plt
from collections import Counter
import random
import pandas as pd
import warnings
import mpose

def compute_imbalance_metrics(class_counts):
    """Compute imbalance metrics given a dictionary class -> count."""
    counts = np.array(list(class_counts.values()))
    min_count = counts.min()
    max_count = counts.max()
    mean_count = counts.mean()
    median_count = np.median(counts)
    imbalance_ratio = max_count / min_count
    # Gini coefficient
    sorted_counts = np.sort(counts)
    n = len(sorted_counts)
    sum_total = np.sum(sorted_counts)
    gini = (2 * np.sum((np.arange(1, n+1) * sorted_counts))) / (n * sum_total) - (n + 1) / n
    return {
        "min": min_count,
        "max": max_count,
        "mean": mean_count,
        "median": median_count,
        "imbalance_ratio": imbalance_ratio,
        "gini": gini,
    }

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--split', type=int, default=1, help='MPOSE2021 split (1,2,3)')
    parser.add_argument('--output_dir', default='./mpose_stats', help='Directory to save plots')
    parser.add_argument('--seed', type=int, default=42, help='Random seed')
    parser.add_argument('--imbalance_only', action='store_true', help='Only print imbalance metrics (skip plots)')
    args = parser.parse_args()

    random.seed(args.seed)
    np.random.seed(args.seed)

    os.makedirs(args.output_dir, exist_ok=True)

    print("Loading MPOSE2021 dataset...")
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        data_dir='./data/mpose'
        dataset = MPOSE(pose_extractor='openpose',
                split=args.split,
                preprocess='scale_and_center',velocities=True,
                remove_zip=False)
    X_train, y_train, X_test, y_test = dataset.get_data()

    # Convert to numpy arrays
    y_train = np.array(y_train)
    y_test = np.array(y_test)

    all_labels = np.concatenate([y_train, y_test])
    num_classes = len(np.unique(all_labels))

    # Get class name mapping (label -> name)
    labels_info = dataset.get_labels()
    # labels_info is a dict: name -> label. Invert it to label -> name
    if isinstance(labels_info, dict):
        # If keys are names, values are labels
        class_name_map = {v: k for k, v in labels_info.items()}
    else:
        # If it's a list, assume order
        class_name_map = {i: str(labels_info[i]) for i in range(len(labels_info))}
    print(f"Class mapping: {class_name_map}")

    # Create a list of class names in order of label
    class_names = [class_name_map[i] for i in range(num_classes)]

    print(f"Total classes: {num_classes}")
    print(f"Train samples: {len(X_train)}")
    print(f"Test samples: {len(X_test)}")
    print(f"Total samples: {len(X_train) + len(X_test)}")

    # Class distributions
    train_counts = Counter(y_train)
    test_counts = Counter(y_test)

    # Print per‑class counts with names
    print("\nClass distribution (train):")
    for cls in sorted(train_counts.keys()):
        print(f"  {class_names[cls]}: {train_counts[cls]}")
    print("\nClass distribution (test):")
    for cls in sorted(test_counts.keys()):
        print(f"  {class_names[cls]}: {test_counts[cls]}")

    # Imbalance metrics
    train_imbalance = compute_imbalance_metrics(train_counts)
    test_imbalance = compute_imbalance_metrics(test_counts)
    print("\nImbalance metrics (train):")
    for k, v in train_imbalance.items():
        print(f"  {k}: {v:.2f}" if isinstance(v, float) else f"  {k}: {v}")
    print("\nImbalance metrics (test):")
    for k, v in test_imbalance.items():
        print(f"  {k}: {v:.2f}" if isinstance(v, float) else f"  {k}: {v}")

    if args.imbalance_only:
        return

    # Save class distribution to CSV
    rows = []
    for cls in range(num_classes):
        rows.append({
            'class_name': class_names[cls],
            'class_id': cls,
            'train_count': train_counts.get(cls, 0),
            'test_count': test_counts.get(cls, 0)
        })
    df = pd.DataFrame(rows)
    df.to_csv(os.path.join(args.output_dir, f'class_distribution_split{args.split}.csv'), index=False)
    print(f"\nClass distribution saved to {args.output_dir}/class_distribution_split{args.split}.csv")

    # Plot class distribution with names
    plt.figure(figsize=(15, 6))
    plt.subplot(1, 2, 1)
    plt.bar(range(num_classes), [train_counts.get(i, 0) for i in range(num_classes)], alpha=0.7)
    plt.xlabel('Class')
    plt.ylabel('Number of samples')
    plt.title('Train class distribution')
    plt.xticks(range(num_classes), class_names, rotation=90, fontsize=8)

    plt.subplot(1, 2, 2)
    plt.bar(range(num_classes), [test_counts.get(i, 0) for i in range(num_classes)], alpha=0.7, color='orange')
    plt.xlabel('Class')
    plt.ylabel('Number of samples')
    plt.title('Test class distribution')
    plt.xticks(range(num_classes), class_names, rotation=90, fontsize=8)
    plt.tight_layout()
    plt.savefig(os.path.join(args.output_dir, f'class_distribution_split{args.split}.png'), dpi=150)
    plt.close()

    # Sequence length distribution
    train_lengths = [seq.shape[0] for seq in X_train]
    test_lengths = [seq.shape[0] for seq in X_test]

    plt.figure(figsize=(12, 5))
    plt.subplot(1, 2, 1)
    plt.hist(train_lengths, bins=30, alpha=0.7, label='Train')
    plt.xlabel('Number of frames')
    plt.ylabel('Frequency')
    plt.title('Train sequence lengths')
    plt.legend()
    plt.subplot(1, 2, 2)
    plt.hist(test_lengths, bins=30, alpha=0.7, color='orange', label='Test')
    plt.xlabel('Number of frames')
    plt.ylabel('Frequency')
    plt.title('Test sequence lengths')
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(args.output_dir, f'seq_lengths_split{args.split}.png'), dpi=150)
    plt.close()

    print(f"\nMean train length: {np.mean(train_lengths):.2f} frames")
    print(f"Mean test length: {np.mean(test_lengths):.2f} frames")

    # Per‑class average frame length
    class_lengths_train = {cls: [] for cls in range(num_classes)}
    for seq, label in zip(X_train, y_train):
        class_lengths_train[label].append(seq.shape[0])
    class_lengths_train_mean = {cls: np.mean(lengths) for cls, lengths in class_lengths_train.items() if lengths}

    plt.figure(figsize=(15, 6))
    plt.bar(class_lengths_train_mean.keys(), class_lengths_train_mean.values())
    plt.xlabel('Class')
    plt.ylabel('Average frames')
    plt.title('Average sequence length per class (train)')
    plt.xticks(range(num_classes), class_names, rotation=90, fontsize=8)
    plt.tight_layout()
    plt.savefig(os.path.join(args.output_dir, f'avg_seq_len_per_class_split{args.split}.png'), dpi=150)
    plt.close()

    # Average confidence
    all_conf_train = []
    for seq in X_train:
        all_conf_train.extend(seq[:, :, 2].flatten())
    avg_conf_train = np.mean(all_conf_train)
    print(f"\nAverage keypoint confidence in train set: {avg_conf_train:.3f}")

    print(f"\nPlots and data saved to {args.output_dir}")

if __name__ == '__main__':
    main()