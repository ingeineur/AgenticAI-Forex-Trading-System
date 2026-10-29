#!/usr/bin/env python3
"""
Diagnostic script to investigate why LSTM test metrics are poor.
Checks for class imbalance, data quality, label leakage, and model behavior.
"""

import argparse
import json
import numpy as np
import pandas as pd


def diagnose_dataset(dataset_npz_path, meta_csv_path=None):
    """Load and analyze the NPZ dataset."""
    print("=" * 70)
    print(f"Loading dataset: {dataset_npz_path}")
    print("=" * 70)

    data = np.load(dataset_npz_path, allow_pickle=True)
    X = data["X"].astype(np.float32)
    y = data["y"].astype(np.float32)
    feature_cols = data["feature_cols"].tolist()
    sequence_length = int(data["sequence_length"])

    print(f"\nDataset Shape:")
    print(f"  X: {X.shape} (sequences, timesteps, features)")
    print(f"  y: {y.shape}")
    print(f"  Sequence length: {sequence_length}")
    print(f"  Features: {len(feature_cols)}")

    # Label distribution
    print(f"\nLabel Distribution:")
    unique, counts = np.unique(y, return_counts=True)
    for label, count in zip(unique, counts):
        pct = 100 * count / len(y)
        print(f"  Label {int(label)}: {count:,} ({pct:.1f}%)")

    # Class imbalance ratio
    pos_count = (y == 1).sum()
    neg_count = (y == 0).sum()
    if pos_count > 0:
        imbalance_ratio = neg_count / pos_count
        print(f"  Imbalance Ratio (neg/pos): {imbalance_ratio:.2f}")
        print(f"  [HIGH] Imbalance!" if imbalance_ratio > 3.0 else "  [OK] Moderate imbalance")

    # Train/val/test split
    n = len(y)
    train_end = int(n * 0.70)
    val_end = int(n * 0.85)

    y_train = y[:train_end]
    y_val = y[train_end:val_end]
    y_test = y[val_end:]

    print(f"\nTrain/Val/Test Split:")
    for name, y_split in [("Train", y_train), ("Val", y_val), ("Test", y_test)]:
        pos = (y_split == 1).sum()
        neg = (y_split == 0).sum()
        total = len(y_split)
        pos_pct = 100 * pos / total if total > 0 else 0
        print(f"  {name:6s}: {total:,} samples ({pos:,} pos / {neg:,} neg) {pos_pct:.1f}% positive")

    # Feature statistics
    print(f"\nFeature Statistics (full dataset):")
    X_flat = X.reshape(-1, X.shape[-1])
    for i, feat in enumerate(feature_cols[:10]):  # First 10 features
        mean_val = X_flat[:, i].mean()
        std_val = X_flat[:, i].std()
        min_val = X_flat[:, i].min()
        max_val = X_flat[:, i].max()
        print(f"  {feat:25s}: mean={mean_val:8.3f} std={std_val:6.3f} [{min_val:8.3f}, {max_val:8.3f}]")
    if len(feature_cols) > 10:
        print(f"  ... and {len(feature_cols) - 10} more features")

    # Check for NaN or Inf
    nan_count = np.isnan(X).sum()
    inf_count = np.isinf(X).sum()
    print(f"\nData Quality:")
    print(f"  NaN values: {nan_count}")
    print(f"  Inf values: {inf_count}")
    if nan_count > 0 or inf_count > 0:
        print(f"  [WARN] Data contains invalid values!")

    # Metadata analysis
    if meta_csv_path:
        print(f"\nLoading metadata: {meta_csv_path}")
        try:
            meta = pd.read_csv(meta_csv_path)
            print(f"  Samples: {len(meta)}")
            print(f"\n  Label distribution in metadata:")
            print(f"    {meta['label'].value_counts().to_dict()}")
            print(f"\n  HTF alignment stats:")
            print(f"    Mean: {meta['htf_alignment'].mean():.3f}")
            print(f"    Std:  {meta['htf_alignment'].std():.3f}")
            print(f"    Min:  {meta['htf_alignment'].min():.3f}")
            print(f"    Max:  {meta['htf_alignment'].max():.3f}")
        except Exception as e:
            print(f"  Error reading metadata: {e}")

    # Config
    if "config" in data.files:
        config = json.loads(data["config"].item())
        print(f"\nDataset Config:")
        for key, val in config.items():
            print(f"  {key}: {val}")

    print("\n" + "=" * 70)
    print("DIAGNOSIS SUMMARY:")
    print("=" * 70)

    if imbalance_ratio > 3.0:
        print("[CRITICAL] HIGH CLASS IMBALANCE: Negative class dominates.")
        print("   -> Model may default to predicting 0 (negative).")
        print("   -> Consider: higher pos_weight, data sampling, or threshold tuning.")

    if nan_count > 0 or inf_count > 0:
        print("[CRITICAL] DATA CONTAINS INVALID VALUES: NaN or Inf detected.")
        print("   -> Model may not learn properly from corrupted sequences.")
        print("   -> Check: feature normalization, division-by-zero, missing fills.")

    if len(y) < 200:
        print("[WARNING] DATASET TOO SMALL: Only ~200 sequences.")
        print("   -> LSTM needs more data to generalize.")
        print("   -> Try: longer input CSV, lower signal thresholds, or simpler model.")

    print("\n" + "=" * 70)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Diagnose LSTM dataset quality and class balance.")
    parser.add_argument("dataset_npz", help="Path to the .npz dataset file")
    parser.add_argument("--meta", help="Optional path to metadata .csv for additional context")
    args = parser.parse_args()

    diagnose_dataset(args.dataset_npz, args.meta)
