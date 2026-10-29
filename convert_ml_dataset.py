#!/usr/bin/env python3
"""Train LSTM with ML signals dataset (compatible with existing train_lstm_price_action.py)"""

import numpy as np
import sys

def convert_dataset():
    """Convert new dataset format to old format expected by train_lstm_price_action.py"""
    
    import sys
    npz_file = sys.argv[1] if len(sys.argv) > 1 else "dataset_3r3r.npz"
    print(f"Loading {npz_file}...")
    data = np.load(npz_file)
    
    X_train = data["X_train"]
    y_train = data["y_train"]
    X_test = data["X_test"]
    y_test = data["y_test"]
    
    # Combine and save in old format
    X = np.vstack([X_train, X_test])
    y = np.hstack([y_train, y_test])
    
    print(f"Dataset shape: {X.shape}")
    print(f"Labels: {len(y)}")
    print(f"Win rate: {y.mean():.1%}")
    print()
    
    # Create in old format
    print("Saving in legacy format for train_lstm_price_action.py...")
    base = npz_file.replace('.npz', '')
    outfile = f"{base}_legacy.npz"
    
    np.savez(
        outfile,
        X=X,
        y=y,
        feature_cols=np.array(["open", "high", "low", "close", "volume"]),
        sequence_length=5,
    )
    
    print(f"Saved to {outfile}")
    print()
    print(f"Next: python train_lstm_price_action.py {outfile}")

if __name__ == "__main__":
    convert_dataset()
