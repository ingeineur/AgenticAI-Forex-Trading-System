import argparse
import json

import numpy as np


def require_torch():
    try:
        import torch
        from torch import nn
        from torch.utils.data import DataLoader, TensorDataset
    except ImportError as exc:
        raise SystemExit(
            "PyTorch is required for the LSTM trainer. Install it in your trading Python environment, "
            "then rerun this script. See https://pytorch.org/get-started/locally/ for the correct command."
        ) from exc
    return torch, nn, DataLoader, TensorDataset


class LSTMClassifier:
    def __init__(self, nn, input_size, hidden_size=64, num_layers=2, dropout=0.2):
        class _Model(nn.Module):
            def __init__(self):
                super().__init__()
                self.lstm = nn.LSTM(
                    input_size=input_size,
                    hidden_size=hidden_size,
                    num_layers=num_layers,
                    batch_first=True,
                    dropout=dropout if num_layers > 1 else 0.0,
                )
                self.head = nn.Sequential(
                    nn.LayerNorm(hidden_size),
                    nn.Linear(hidden_size, hidden_size // 2),
                    nn.ReLU(),
                    nn.Dropout(dropout),
                    nn.Linear(hidden_size // 2, 1),
                )

            def forward(self, x):
                output, _ = self.lstm(x)
                last = output[:, -1, :]
                return self.head(last).squeeze(-1)

        self.model = _Model()


def standardize_time_split(X_train, X_val, X_test):
    mean = X_train.reshape(-1, X_train.shape[-1]).mean(axis=0)
    std = X_train.reshape(-1, X_train.shape[-1]).std(axis=0)
    std = np.where(std < 1e-8, 1.0, std)
    return (
        (X_train - mean) / std,
        (X_val - mean) / std,
        (X_test - mean) / std,
        mean.astype(np.float32),
        std.astype(np.float32),
    )


def binary_metrics(y_true, y_prob, threshold):
    y_pred = (y_prob >= threshold).astype(np.float32)
    tp = ((y_pred == 1) & (y_true == 1)).sum()
    fp = ((y_pred == 1) & (y_true == 0)).sum()
    tn = ((y_pred == 0) & (y_true == 0)).sum()
    fn = ((y_pred == 0) & (y_true == 1)).sum()
    precision = tp / max(tp + fp, 1)
    recall = tp / max(tp + fn, 1)
    accuracy = (tp + tn) / max(len(y_true), 1)
    selected = y_prob >= threshold
    selected_winrate = y_true[selected].mean() if selected.any() else np.nan
    return {
        "threshold": threshold,
        "accuracy": float(accuracy),
        "precision": float(precision),
        "recall": float(recall),
        "selected_trades": int(selected.sum()),
        "selected_winrate": float(selected_winrate) if not np.isnan(selected_winrate) else None,
    }


def main():
    parser = argparse.ArgumentParser(description="Train an LSTM classifier for price-action setup scoring.")
    parser.add_argument("dataset_npz")
    parser.add_argument("--model-output", default="lstm_price_action.pt")
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--hidden-size", type=int, default=64)
    parser.add_argument("--num-layers", type=int, default=2)
    parser.add_argument("--dropout", type=float, default=0.2)
    parser.add_argument("--lr", type=float, default=0.001)
    parser.add_argument("--threshold", type=float, default=0.60)
    args = parser.parse_args()

    torch, nn, DataLoader, TensorDataset = require_torch()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    data = np.load(args.dataset_npz, allow_pickle=True)
    X = data["X"].astype(np.float32)
    y = data["y"].astype(np.float32)
    feature_cols = data["feature_cols"].tolist()
    sequence_length = int(data["sequence_length"])
    config = json.loads(data["config"].item()) if "config" in data.files else {}

    if len(y) < 100:
        raise SystemExit(
            f"Only {len(y)} labeled sequences were found. Use more history or loosen the "
            "price-action filters before training an LSTM."
        )

    n = len(y)
    train_end = int(n * 0.70)
    val_end = int(n * 0.85)

    X_train, y_train = X[:train_end], y[:train_end]
    X_val, y_val = X[train_end:val_end], y[train_end:val_end]
    X_test, y_test = X[val_end:], y[val_end:]

    X_train, X_val, X_test, mean, std = standardize_time_split(X_train, X_val, X_test)

    train_ds = TensorDataset(torch.tensor(X_train), torch.tensor(y_train))
    val_x = torch.tensor(X_val).to(device)
    val_y = torch.tensor(y_val).to(device)
    test_x = torch.tensor(X_test).to(device)

    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=False)

    model = LSTMClassifier(
        nn,
        input_size=X.shape[-1],
        hidden_size=args.hidden_size,
        num_layers=args.num_layers,
        dropout=args.dropout,
    ).model.to(device)

    pos = float(y_train.sum())
    neg = float(len(y_train) - pos)
    pos_weight = torch.tensor([neg / max(pos, 1.0)], dtype=torch.float32).to(device)
    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)

    best_val_loss = float("inf")
    best_state = None
    patience = 6
    stale_epochs = 0

    for epoch in range(1, args.epochs + 1):
        model.train()
        train_losses = []
        for batch_x, batch_y in train_loader:
            batch_x = batch_x.to(device)
            batch_y = batch_y.to(device)
            optimizer.zero_grad()
            logits = model(batch_x)
            loss = criterion(logits, batch_y)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            train_losses.append(loss.item())

        model.eval()
        with torch.no_grad():
            val_logits = model(val_x)
            val_loss = criterion(val_logits, val_y).item()
            val_prob = torch.sigmoid(val_logits).cpu().numpy()

        metrics = binary_metrics(y_val, val_prob, args.threshold)
        print(
            f"epoch={epoch:03d} train_loss={np.mean(train_losses):.4f} "
            f"val_loss={val_loss:.4f} selected={metrics['selected_trades']} "
            f"selected_winrate={metrics['selected_winrate']}"
        )

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_state = {key: value.detach().cpu().clone() for key, value in model.state_dict().items()}
            stale_epochs = 0
        else:
            stale_epochs += 1
            if stale_epochs >= patience:
                print("Early stopping.")
                break

    model.load_state_dict(best_state)
    model.eval()
    with torch.no_grad():
        test_prob = torch.sigmoid(model(test_x)).cpu().numpy()

    print("\nTest metrics")
    for threshold in [0.50, args.threshold, 0.70, 0.80]:
        print(binary_metrics(y_test, test_prob, threshold))

    artifact = {
        "model_state": model.state_dict(),
        "feature_cols": feature_cols,
        "sequence_length": sequence_length,
        "mean": mean,
        "std": std,
        "model_config": {
            "input_size": X.shape[-1],
            "hidden_size": args.hidden_size,
            "num_layers": args.num_layers,
            "dropout": args.dropout,
            "threshold": args.threshold,
        },
        "dataset_config": config,
    }
    torch.save(artifact, args.model_output)
    print(f"\nSaved model to {args.model_output}")


if __name__ == "__main__":
    main()
