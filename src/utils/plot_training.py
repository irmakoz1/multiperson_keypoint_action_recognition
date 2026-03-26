#!/usr/bin/env python3
"""
Plot training curves (loss, accuracy, balanced accuracy) from a GraphSAGE model's history.json.
Usage:
    python plot_training_curves.py --base_dir ./outputs/temporal/graphsage_temporal --best
    python plot_training_curves.py --base_dir ./outputs/temporal/graphsage_temporal --best --output curves.png
"""

import argparse
import json
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path
import sys

def find_best_model(base_dir):
    """Find the model with highest balanced accuracy in its history."""
    best_bal_acc = -1
    best_dir = None
    base_path = Path(base_dir)
    for hist_path in base_path.rglob('history.json'):
        with open(hist_path, 'r') as f:
            history = json.load(f)
        # Balanced accuracy could be under 'bal_acc' or 'val_bal_acc'
        bal_list = history.get('bal_acc', history.get('val_bal_acc', []))
        if bal_list:
            max_bal = max(bal_list)
            if max_bal > best_bal_acc:
                best_bal_acc = max_bal
                best_dir = hist_path.parent
    return best_dir, best_bal_acc

def plot_training_curves(history, output_file, title=None):
    """Plot loss and accuracy from a history dict."""
    # Determine available keys
    train_loss = history.get('train_loss', [])
    val_loss = history.get('val_loss', [])
    train_acc = history.get('train_acc', [])
    val_acc = history.get('val_acc', [])
    # Balanced accuracy may be under 'bal_acc' or 'val_bal_acc'
    bal_acc = history.get('bal_acc', history.get('val_bal_acc', []))

    epochs = range(1, len(train_loss) + 1) if train_loss else None

    if not train_loss and not train_acc:
        print("No loss or accuracy data found in history.")
        return

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))

    # Plot loss
    if train_loss:
        ax1.plot(epochs, train_loss, 'b-', label='Train Loss', linewidth=2)
        if val_loss:
            ax1.plot(epochs, val_loss, 'r-', label='Val Loss', linewidth=2)
        ax1.set_xlabel('Epoch')
        ax1.set_ylabel('Loss')
        ax1.set_title('Loss')
        ax1.legend()
        ax1.grid(True, alpha=0.3)

    # Plot accuracy
    if train_acc:
        ax2.plot(epochs, train_acc, 'b-', label='Train Acc', linewidth=2)
        if val_acc:
            ax2.plot(epochs, val_acc, 'r-', label='Val Acc', linewidth=2)
        if bal_acc:
            ax2.plot(epochs, bal_acc, 'g--', label='Balanced Acc', linewidth=2)
        ax2.set_xlabel('Epoch')
        ax2.set_ylabel('Accuracy')
        ax2.set_title('Accuracy')
        ax2.legend()
        ax2.grid(True, alpha=0.3)

    # Main title if provided
    if title:
        fig.suptitle(title, fontsize=14)

    plt.tight_layout()
    plt.savefig(output_file, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Plot saved to {output_file}")

def main():
    parser = argparse.ArgumentParser(description='Plot training curves from GraphSAGE history.')
    parser.add_argument('--model_dir', type=str, help='Path to model directory containing history.json')
    parser.add_argument('--base_dir', type=str, default='./outputs',
                        help='Base directory to search for models (used with --best)')
    parser.add_argument('--best', action='store_true',
                        help='Automatically select the model with highest balanced accuracy')
    parser.add_argument('--output', type=str, default='training_curves.png',
                        help='Output image file name')
    parser.add_argument('--title', type=str, help='Optional title for the plot')
    args = parser.parse_args()

    if args.model_dir:
        model_dir = Path(args.model_dir)
        hist_path = model_dir / 'history.json'
        if not hist_path.exists():
            print(f"Error: {hist_path} not found.")
            sys.exit(1)
        with open(hist_path, 'r') as f:
            history = json.load(f)
        plot_training_curves(history, args.output, title=args.title or f"Training curves – {model_dir.name}")
    elif args.best:
        best_dir, best_acc = find_best_model(args.base_dir)
        if best_dir is None:
            print("No model with balanced accuracy found.")
            sys.exit(1)
        print(f"Best model: {best_dir} (Balanced Acc: {best_acc:.4f})")
        with open(best_dir / 'history.json', 'r') as f:
            history = json.load(f)
        plot_training_curves(history, args.output, title=args.title or f"Training curves – Best model ({best_dir.name})")
    else:
        print("Please provide either --model_dir or --best.")
        sys.exit(1)

if __name__ == '__main__':
    main()