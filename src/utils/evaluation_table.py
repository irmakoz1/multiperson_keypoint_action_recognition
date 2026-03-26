#!/usr/bin/env python3
"""
Collect training results from all model runs and create a summary table.
Handles different naming of balanced accuracy and history files.
Usage:
python src/utils/evaluation_table.py --base_dir ./outputs/temporal --output results.csv
"""

import os
import sys
import argparse
import json
from pathlib import Path
import pandas as pd
from tabulate import tabulate

def find_history_files(base_dir):
    """Return list of (path_to_history_file, model_dir) pairs."""
    base_path = Path(base_dir)
    history_files = []
    # Recursively search for history.json and training_history.json
    for fname in ['history.json', 'training_history.json']:
        for path in base_path.rglob(fname):
            history_files.append((path, path.parent))
    return history_files

def extract_metrics(history_path, model_dir, debug=False):
    """Extract metrics and configuration from a model directory."""
    with open(history_path, 'r') as f:
        history = json.load(f)

    if debug:
        print(f"\n=== {model_dir} ===")
        print(f"  History file: {history_path}")
        print(f"  Keys: {list(history.keys())}")
        # Print summary for each key
        for k in history:
            if isinstance(history[k], list) and len(history[k]) > 0:
                print(f"  {k}: length {len(history[k])}, max: {max(history[k])}, last: {history[k][-1]}")
            elif isinstance(history[k], (int, float)):
                print(f"  {k}: {history[k]}")

    # Balanced accuracy – priority: bal_acc (GraphSAGE) > val_bal_acc (transformers) > others
    best_val_bal_acc = None
    bal_source = None
    for key in ['bal_acc', 'val_bal_acc', 'val_balanced_accuracy', 'balanced_accuracy']:
        if key in history and history[key]:
            val = history[key]
            if isinstance(val, list) and val:
                best_val_bal_acc = max(val)
            elif isinstance(val, (int, float)):
                best_val_bal_acc = val
            else:
                continue
            bal_source = key
            break

    # Regular accuracy
    best_val_acc = None
    if 'val_acc' in history and history['val_acc']:
        val = history['val_acc']
        best_val_acc = max(val) if isinstance(val, list) else val
    best_train_acc = None
    if 'train_acc' in history and history['train_acc']:
        val = history['train_acc']
        best_train_acc = max(val) if isinstance(val, list) else val
    best_val_loss = None
    if 'val_loss' in history and history['val_loss']:
        val = history['val_loss']
        best_val_loss = min(val) if isinstance(val, list) else val

    # Final test metrics
    final_metrics_path = model_dir / 'final_metrics.json'
    test_bal_acc = None
    if final_metrics_path.exists():
        with open(final_metrics_path, 'r') as f:
            final = json.load(f)
            test_bal_acc = final.get('balanced_accuracy') or final.get('bal_acc')
        if debug:
            print(f"  Final metrics: test_bal_acc = {test_bal_acc}")

    # Load configuration
    args_path = model_dir / 'args.json'
    config = {}
    if args_path.exists():
        with open(args_path, 'r') as f:
            config = json.load(f)
        if debug:
            print(f"  Args keys: {list(config.keys())}")

    # Infer model type from directory name (case-insensitive)
    dir_name = model_dir.name.lower()
    model_type = 'Unknown'
    if 'graphsage' in dir_name:
        if 'raw' in dir_name or 'no_extractor' in dir_name:
            model_type = 'Unified GraphSAGE (raw)'
        else:
            model_type = 'Unified GraphSAGE (extractor)'
    elif 'transformer' in dir_name:
        if 'raw' in dir_name:
            model_type = 'Temporal Transformer (raw)'
        elif 'extractor' in dir_name or 'mpose' in dir_name:
            model_type = 'Temporal Transformer (extractor)'
        else:
            model_type = 'Temporal Transformer'
    elif 'spatial_temporal_lstm' in dir_name:
        model_type = 'Separate Spatial+LSTM (extractor)'
    elif 'spatial_temporal_transformer' in dir_name:
        model_type = 'Separate Spatial+Transformer (extractor)'
    elif 'lstm' in dir_name:
        model_type = 'LSTM (raw)'

    # Extract parameters
    window_size = config.get('window_size')
    if window_size is None:
        window_size = config.get('seq_len', 'N/A')
    batch_size = config.get('batch_size', 'N/A')
    hidden_dim = config.get('hidden_dim')
    if hidden_dim is None:
        hidden_dim = config.get('lstm_hidden_dim', 'N/A')
    num_layers = config.get('num_layers')
    if num_layers is None:
        num_layers = config.get('num_lstm_layers', 'N/A')
    num_heads = config.get('num_heads', 'N/A')
    dropout = config.get('dropout', 'N/A')

    return {
        'model_dir': str(model_dir),
        'model_type': model_type,
        'window_size': window_size,
        'batch_size': batch_size,
        'hidden_dim': hidden_dim,
        'num_layers': num_layers,
        'num_heads': num_heads,
        'dropout': dropout,
        'best_train_acc': best_train_acc,
        'best_val_acc': best_val_acc,
        'best_val_bal_acc': best_val_bal_acc,
        'bal_source': bal_source,
        'test_bal_acc': test_bal_acc,
        'best_val_loss': best_val_loss,
    }

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--base_dir', type=str, default='./outputs',
                        help='Base directory containing model subdirectories')
    parser.add_argument('--output', type=str, default='results.csv',
                        help='Output CSV file name')
    parser.add_argument('--no_print', action='store_true',
                        help='Do not print table to console')
    parser.add_argument('--debug', action='store_true',
                        help='Print debug information for each model directory')
    args = parser.parse_args()

    base_path = Path(args.base_dir)
    if not base_path.exists():
        print(f"Base directory {base_path} does not exist.")
        sys.exit(1)

    history_files = find_history_files(base_path)
    if not history_files:
        print("No history.json or training_history.json files found.")
        sys.exit(1)

    results = []
    for history_path, model_dir in history_files:
        print(f"Processing {model_dir}")
        metrics = extract_metrics(history_path, model_dir, debug=args.debug)
        if metrics:
            results.append(metrics)

    if not results:
        print("No valid metrics found.")
        return

    df = pd.DataFrame(results)
    # Sort by best_val_bal_acc if exists, else best_val_acc
    if 'best_val_bal_acc' in df.columns:
        df = df.sort_values('best_val_bal_acc', ascending=False, na_position='last')
    else:
        df = df.sort_values('best_val_acc', ascending=False, na_position='last')

    # Save CSV
    df.to_csv(args.output, index=False)
    print(f"\nResults saved to {args.output}")

    if not args.no_print:
        # Columns to display
        display_cols = ['model_type', 'window_size', 'batch_size', 'hidden_dim', 'num_layers',
                        'num_heads', 'dropout', 'best_train_acc', 'best_val_acc',
                        'best_val_bal_acc', 'bal_source', 'test_bal_acc', 'best_val_loss']
        display_df = df[[c for c in display_cols if c in df.columns]].copy()
        # Format floats
        float_cols = ['best_train_acc', 'best_val_acc', 'best_val_bal_acc', 'test_bal_acc', 'best_val_loss']
        for col in float_cols:
            if col in display_df.columns:
                display_df[col] = display_df[col].map(lambda x: f'{x:.4f}' if pd.notna(x) else '')
        display_df = display_df.fillna('')
        print("\n" + tabulate(display_df, headers='keys', tablefmt='psql', showindex=False))

if __name__ == '__main__':
    main()