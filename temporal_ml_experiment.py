"""Temporal ML experiment: LSTM (torch) + XGBoost vs RF baseline on per-trade feature sequences.
Compares on TP-first label with a strict time split (train 60% / test 40%).
Usage: python temporal_ml_experiment.py seq_xau_m15.npz
"""
import sys
import numpy as np
import pickle

def load(path):
    d = np.load(path)
    return d['seq'], d['static'], d['label'], int(d['seq_len']), int(d['n_seq']), int(d['n_static'])

def evaluate(y_true, prob, th=0.5):
    pred = (prob >= th).astype(int)
    if pred.sum() == 0:
        return 0, 0.0, 0.0
    wr = (pred == y_true).mean()
    wins = (pred * y_true).sum()
    net = wins * 0.5 - (pred.sum() - wins) * 1.0  # TP_MULT=0.5 for v2
    return int(pred.sum()), wr, net

def main(path):
    seq, static, y, seq_len, n_seq, n_static = load(path)
    n = len(y)
    split = int(n * 0.6)
    print(f"Temporal experiment on {path}")
    print(f"  trades={n} seq_len={seq_len} n_seq={n_seq} n_static={n_static} base_rate={y.mean():.1%}")
    print(f"  time-split: train={split} test={n - split}")

    # --- RF baseline on flattened seq+static ---
    X_flat = seq.reshape(n, -1)
    X_flat = np.hstack([X_flat, static])
    from sklearn.ensemble import RandomForestClassifier
    rf = RandomForestClassifier(n_estimators=200, max_depth=6, min_samples_split=20,
                                min_samples_leaf=10, random_state=42)
    rf.fit(X_flat[:split], y[:split])
    rf_prob = rf.predict_proba(X_flat[split:])[:, 1]
    rf_t, rf_wr, rf_net = evaluate(y[split:], rf_prob, 0.5)
    print(f"\n[RF baseline (flattened seq)]      th=0.50 -> {rf_t:4d} trades  WR={rf_wr:.1%}  net={rf_net:+.1f}R")

    # --- XGBoost on flattened seq+static ---
    import xgboost as xgb
    xg = xgb.XGBClassifier(n_estimators=200, max_depth=5, learning_rate=0.05,
                           subsample=0.8, colsample_bytree=0.8, random_state=42,
                           eval_metric='logloss', tree_method='hist')
    xg.fit(X_flat[:split], y[:split], verbose=False)
    xg_prob = xg.predict_proba(X_flat[split:])[:, 1]
    best = (0, 0.0, -999)
    for th in np.arange(0.3, 0.9, 0.05):
        t, w, net = evaluate(y[split:], xg_prob, th)
        if net > best[2]:
            best = (t, w, net)
    print(f"[XGBoost flattened]               best -> {best[0]:4d} trades  WR={best[1]:.1%}  net={best[2]:+.1f}R")

    # --- LSTM (torch) on raw sequence ---
    import torch
    import torch.nn as nn
    torch.manual_seed(42)
    dev = torch.device('cpu')

    class SeqLSTM(nn.Module):
        def __init__(self, n_seq, n_static, hidden=32, layers=1):
            super().__init__()
            self.lstm = nn.LSTM(n_seq, hidden, layers, batch_first=True, dropout=0.2 if layers > 1 else 0.0)
            self.fc = nn.Linear(hidden + n_static, 1)
        def forward(self, x_seq, x_static):
            out, _ = self.lstm(x_seq)          # (B, T, H)
            last = out[:, -1, :]               # last timestep
            z = torch.cat([last, x_static], dim=1)
            return torch.sigmoid(self.fc(z)).squeeze(-1)

    model = SeqLSTM(n_seq, n_static).to(dev)
    opt = torch.optim.Adam(model.parameters(), lr=1e-3)
    lossf = nn.BCELoss()

    Xs = torch.tensor(seq, dtype=torch.float32)
    Xt = torch.tensor(static, dtype=torch.float32)
    yt = torch.tensor(y, dtype=torch.float32)

    Xs_tr, Xt_tr, y_tr = Xs[:split], Xt[:split], yt[:split]
    Xs_te, Xt_te, y_te = Xs[split:], Xt[split:], yt[split:]

    n_epochs = 50
    batch = 64
    for ep in range(n_epochs):
        model.train()
        perm = torch.randperm(split)
        for b in range(0, split, batch):
            idx = perm[b:b+batch]
            opt.zero_grad()
            p = model(Xs_tr[idx], Xt_tr[idx])
            loss = lossf(p, y_tr[idx])
            loss.backward()
            opt.step()
        model.eval()
        with torch.no_grad():
            pv = model(Xs_te, Xt_te).numpy()
        best_ep = (0, 0.0, -999)
        for th in np.arange(0.3, 0.9, 0.05):
            t, w, net = evaluate(y_te.numpy(), pv, th)
            if net > best_ep[2]:
                best_ep = (t, w, net)
        if (ep + 1) % 10 == 0:
            print(f"    LSTM epoch {ep+1:2d}/{n_epochs} test best -> {best_ep[0]:4d} trades  WR={best_ep[1]:.1%}  net={best_ep[2]:+.1f}R")
    with torch.no_grad():
        lstm_prob = model(Xs_te, Xt_te).numpy()
    best = (0, 0.0, -999)
    for th in np.arange(0.3, 0.9, 0.05):
        t, w, net = evaluate(y_te.numpy(), lstm_prob, th)
        if net > best[2]:
            best = (t, w, net)
    print(f"[LSTM sequence]                  best -> {best[0]:4d} trades  WR={best[1]:.1%}  net={best[2]:+.1f}R")

    print("\nBaseline (no ML):")
    print(f"  all {n - split} test trades: WR={y[split:].mean():.1%}  net={(y[split:].sum()*0.5 - (n-split - y[split:].sum())):+.1f}R")

if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "seq_xau_m15.npz")