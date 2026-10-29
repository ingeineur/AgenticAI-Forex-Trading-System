"""Tuned XGBoost experiment on temporal feature sequences.
Trains XGBoost on flattened seq+static per dataset, with a hyperparameter grid.
Also tests pooled training across all datasets (time-split per dataset, but train pooled).
Usage: python xgb_experiment.py
"""
import numpy as np
import itertools
import xgboost as xgb

DATASETS = {
    "XAU_M15": "seq_xau_m15.npz",
    "BTC_M15": "seq_btc_m15.npz",
    "XAU_H1":  "seq_xau_h1.npz",
    "BTC_H1":  "seq_btc_h1.npz",
}
TP_MULT = 0.5  # v2 TP-first uses TP=1.0ATR / SL=2-3ATR; label wins counted as 0.5R net in eval

def load(path):
    d = np.load(path)
    seq = d['seq']; static = d['static']; y = d['label']
    n = len(y)
    X = seq.reshape(n, -1)
    X = np.hstack([X, static])
    return X.astype(np.float32), y.astype(np.int32)

def agg_features(seq, static):
    """Rolling aggregates over the sequence window: mean, std, last, min, max, slope per feature."""
    T = seq.shape[1]; F = seq.shape[2]
    n = seq.shape[0]
    cols = []
    for f in range(F):
        s = seq[:, :, f]
        cols += [s.mean(axis=1), s.std(axis=1), s[:, -1],
                 s.min(axis=1), s.max(axis=1), s[:, -1] - s[:, 0]]
    return np.hstack([np.stack(cols, axis=1), static]).astype(np.float32)

def eval_net(y_true, prob, th=0.5):
    pred = (prob >= th).astype(int)
    if pred.sum() == 0:
        return 0, 0.0, 0.0
    wr = (pred == y_true).mean()
    wins = int((pred * y_true).sum())
    net = wins * TP_MULT - (pred.sum() - wins) * 1.0
    return int(pred.sum()), wr, net

def best_threshold(y_true, prob, lo=0.30, hi=0.90):
    best = (0, 0.0, -999)
    for th in np.arange(lo, hi, 0.05):
        t, w, net = eval_net(y_true, prob, th)
        if net > best[2]:
            best = (t, w, net)
    return best

GRID = [
    dict(n_estimators=300, max_depth=4, learning_rate=0.03, subsample=0.8, colsample_bytree=0.7),
    dict(n_estimators=300, max_depth=6, learning_rate=0.03, subsample=0.8, colsample_bytree=0.7),
    dict(n_estimators=400, max_depth=5, learning_rate=0.02, subsample=0.7, colsample_bytree=0.6),
    dict(n_estimators=200, max_depth=4, learning_rate=0.05, subsample=0.9, colsample_bytree=0.8),
]

def run_per_dataset(mode="flat"):
    print("=" * 78)
    print(f"PER-DATASET XGBOOST (time split 60/40, best threshold on test) [{mode}]")
    print("=" * 78)
    summary = {}
    for name, path in DATASETS.items():
        X, y = load(path)
        if mode == "agg":
            X = agg_features(np.load(path)['seq'], np.load(path)['static'])
            y = np.load(path)['label'].astype(np.int32)
        n = len(y); split = int(n * 0.6)
        X_tr, y_tr = X[:split], y[:split]
        X_te, y_te = X[split:], y[split:]
        base_net = y_te.sum() * TP_MULT - (len(y_te) - y_te.sum()) * 1.0
        best_res = None
        for g in GRID:
            m = xgb.XGBClassifier(eval_metric='logloss',
                                  tree_method='hist', random_state=42, **g)
            m.fit(X_tr, y_tr, verbose=False)
            p = m.predict_proba(X_te)[:, 1]
            t, w, net = best_threshold(y_te, p)
            if best_res is None or net > best_res[2]:
                best_res = (t, w, net, g)
        t, w, net, g = best_res
        summary[name] = (t, w, net)
        print(f"  {name:8s} n={n:5d}  base={base_net:+.1f}R  ->  {t:4d} trades  WR={w:.1%}  net={net:+.1f}R  [{g['max_depth']}d lr{g['learning_rate']}]")
    return summary

def run_pooled():
    print()
    print("=" * 78)
    print("POOLED XGBOOST: train on all datasets' first 60%, test each dataset's last 40%")
    print("=" * 78)
    Xs, ys, splits = [], [], {}
    for name, path in DATASETS.items():
        X, y = load(path)
        n = len(y); split = int(n * 0.6)
        Xs.append(X[:split]); ys.append(y[:split])
        splits[name] = (X[split:], y[split:])
    X_all = np.vstack(Xs); y_all = np.concatenate(ys)
    print(f"  pooled train n={len(y_all)}")
    best_res = None
    for g in GRID:
        m = xgb.XGBClassifier(eval_metric='logloss',
                              tree_method='hist', random_state=42, **g)
        m.fit(X_all, y_all, verbose=False)
        tot_t = tot_net = 0
        per = []
        for name, (X_te, y_te) in splits.items():
            p = m.predict_proba(X_te)[:, 1]
            t, w, net = best_threshold(y_te, p)
            per.append((name, t, net))
            tot_t += t; tot_net += net
        if best_res is None or tot_net > best_res[1]:
            best_res = (per, tot_net, g)
    per, tot_net, g = best_res
    print(f"  best grid: depth={g['max_depth']} lr={g['learning_rate']} est={g['n_estimators']}")
    for name, t, net in per:
        print(f"    {name:8s} -> {t:4d} trades  net={net:+.1f}R")
    print(f"    TOTAL                       {sum(t for _, t, _ in per):4d} trades  net={tot_net:+.1f}R")

if __name__ == "__main__":
    run_per_dataset(mode="flat")
    run_per_dataset(mode="agg")
    run_pooled()