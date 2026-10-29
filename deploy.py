#!/usr/bin/env python3
"""
XAUUSD M15 Live Signal Generator - Deployment
Generates trading signals using ML + LSTM pipeline (3R/3R structure)
"""

import numpy as np
import pandas as pd
import pickle
import argparse
import json
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')


def atr(arr, period=14):
    tr = np.maximum(np.maximum(
        arr['high'] - arr['low'],
        np.abs(arr['high'] - np.roll(arr['close'], 1))
    ), np.abs(arr['low'] - np.roll(arr['close'], 1)))
    return pd.Series(tr).rolling(period).mean().to_numpy()


def create_ml_features(arr):
    """Create ML features from structured array (20 features)"""
    n = len(arr)
    eps = 1e-9
    o, h, l, c, v = arr['open'], arr['high'], arr['low'], arr['close'], arr['volume']
    
    feat = np.zeros((n, 20), dtype=np.float32)
    atr_vals = atr(arr, 14)
    
    feat[:, 0] = np.abs(c - o) / (atr_vals + eps)
    feat[:, 1] = (h - l) / (atr_vals + eps)
    feat[:, 2] = (h - np.maximum(o, c)) / (atr_vals + eps)
    feat[:, 3] = (np.minimum(o, c) - l) / (atr_vals + eps)
    ema20 = pd.Series(c).ewm(span=20, adjust=False).mean().to_numpy()
    ema50 = pd.Series(c).ewm(span=50, adjust=False).mean().to_numpy()
    feat[:, 4] = (c - ema20) / (atr_vals + eps)
    feat[:, 5] = (c - ema50) / (atr_vals + eps)
    feat[:, 6] = np.diff(ema20, prepend=ema20[0]) / (atr_vals + eps)
    feat[:, 7] = np.diff(ema50, prepend=ema50[0]) / (atr_vals + eps)
    vol_ma = pd.Series(v).rolling(20).mean().to_numpy()
    feat[:, 8] = v / (vol_ma + eps)
    feat[:, 9] = pd.Series(c).pct_change().rolling(20).std().to_numpy()
    feat[:, 10] = pd.Series(c).pct_change().fillna(0.0).to_numpy()
    feat[:, 11] = c / np.roll(c, 5) - 1.0
    feat[:, 12] = (np.roll(c, 1) > np.roll(o, 1)).astype(float)
    feat[:, 13] = (np.roll(c, 2) > np.roll(o, 2)).astype(float)
    feat[:, 14] = c > (pd.Series(h).rolling(5).max() + pd.Series(l).rolling(5).min()).to_numpy() / 2.0
    feat[:, 15] = atr_vals / (c + eps)
    feat[:, 16] = ((c - o) > 0).astype(float)
    feat[:, 17] = np.abs(c - o) / (h - l + eps)
    recent_high = pd.Series(h).rolling(10).max().to_numpy()
    recent_low = pd.Series(l).rolling(10).min().to_numpy()
    feat[:, 18] = (c - recent_low) / (recent_high - recent_low + eps)
    feat[:, 19] = v / (pd.Series(v).rolling(50).mean().to_numpy() + eps)
    
    return feat, atr_vals


def make_lstm_sequence(df_lookback):
    """Convert 5-bar lookback window to LSTM sequence"""
    o = df_lookback['open'].to_numpy()
    h = df_lookback['high'].to_numpy()
    l = df_lookback['low'].to_numpy()
    c = df_lookback['close'].to_numpy()
    v = df_lookback['volume'].to_numpy()
    
    current_close = c[-1]
    if current_close <= 0:
        return None
    
    o_norm = (o / current_close - 1.0) * 100
    h_norm = (h / current_close - 1.0) * 100
    l_norm = (l / current_close - 1.0) * 100
    c_norm = (c / current_close - 1.0) * 100
    v_norm = v / (np.mean(v) + 1e-9)
    
    return np.column_stack([o_norm, h_norm, l_norm, c_norm, v_norm]).astype(np.float32)


def load_models():
    """Load ML model and LSTM model"""
    lstm_mean = None
    lstm_std = None
    
    print("Loading ML model...")
    if not Path("ml_pipeline_model.pkl").exists():
        print("ERROR: ml_pipeline_model.pkl not found")
        return None, None, None, None
    
    with open("ml_pipeline_model.pkl", "rb") as f:
        ml_artifact = pickle.load(f)
    
    print(f"  ML threshold: {ml_artifact.get('threshold', 0.55):.2f}")
    
    print("Loading LSTM model...")
    if not Path("lstm_price_action.pt").exists():
        print("  WARNING: lstm_price_action.pt not found (run ML-only)")
        return ml_artifact, None, None, None
    
    try:
        import torch
        class LSTMNet(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.lstm = torch.nn.LSTM(5, 64, 2, batch_first=True, dropout=0.2)
                self.head = torch.nn.Sequential(
                    torch.nn.LayerNorm(64),
                    torch.nn.Linear(64, 32),
                    torch.nn.ReLU(),
                    torch.nn.Dropout(0.2),
                    torch.nn.Linear(32, 1),
                )
            def forward(self, x):
                out, _ = self.lstm(x)
                return self.head(out[:, -1, :]).squeeze(-1)
        
        lstm_model = LSTMNet()
        state = torch.load("lstm_price_action.pt", map_location="cpu", weights_only=False)
        lstm_model.load_state_dict(state['model_state'])
        lstm_model.eval()
        
        lstm_mean = state.get('mean')
        lstm_std = state.get('std')
        
        print("  LSTM model loaded")
        return ml_artifact, lstm_model, lstm_mean, lstm_std
    except Exception as e:
        print(f"  WARNING: Could not load LSTM: {e}")
        return ml_artifact, None, None, None


def generate_signals(csv_path, ml_artifact, lstm_model, lstm_mean, lstm_std,
                     ml_threshold=0.55, lstm_threshold=0.50, lookback=5):
    """Generate trading signals from OHLCV CSV"""
    print(f"\nReading data from {csv_path}...")
    df = pd.read_csv(csv_path)
    if "timestamp" not in df.columns and "time" in df.columns:
        df = df.rename(columns={"time": "timestamp"})
    df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True)
    df = df.sort_values("timestamp").reset_index(drop=True)
    
    print(f"  {len(df):,} bars loaded")
    
    # ML features
    dtype = [('open', 'f8'), ('high', 'f8'), ('low', 'f8'), ('close', 'f8'), ('volume', 'f8')]
    arr = np.zeros(len(df), dtype=dtype)
    for col in ['open', 'high', 'low', 'close', 'volume']:
        arr[col] = df[col].to_numpy()
    
    ml_features, atr_vals = create_ml_features(arr)
    
    # ML predictions
    model = ml_artifact['model']
    ml_probs = model.predict_proba(ml_features)[:, 1]
    ml_signals = ml_probs >= ml_threshold
    
    print(f"  ML signals (prob >= {ml_threshold:.2f}): {ml_signals.sum():,}")
    
    # Generate signals
    signals = []
    
    for i in np.where(ml_signals)[0]:
        if i < lookback:
            continue
        
        lookback_df = df.iloc[i - lookback:i]
        seq = make_lstm_sequence(lookback_df)
        if seq is None:
            continue
        
        lstm_prob = None
        if lstm_model is not None and lstm_mean is not None and lstm_std is not None:
            try:
                import torch
                seq_norm = (seq - lstm_mean) / lstm_std
                seq_tensor = torch.from_numpy(seq_norm).unsqueeze(0).float()
                with torch.no_grad():
                    logit = lstm_model(seq_tensor)
                    lstm_prob = torch.sigmoid(logit).item()
            except Exception as e:
                print(f"    LSTM inference error: {e}")
        
        final_prob = lstm_prob if lstm_prob is not None else ml_probs[i]
        lstm_approved = lstm_prob is None or lstm_prob >= lstm_threshold
        
        signals.append({
            "timestamp": str(df["timestamp"].iloc[i]),
            "bar_index": int(i),
            "close": float(df["close"].iloc[i]),
            "ml_probability": float(ml_probs[i]),
            "lstm_probability": float(lstm_prob) if lstm_prob is not None else None,
            "final_probability": float(final_prob),
            "lstm_approved": bool(lstm_approved),
            "signal_type": "BUY" if final_prob >= ml_threshold else "NEUTRAL",
            "atr": float(atr_vals[i]) if not np.isnan(atr_vals[i]) else 0.0,
        })
    
    high_conf = [s for s in signals if s["final_probability"] >= ml_threshold and s["lstm_approved"]]
    
    print(f"  ML-only signals: {len([s for s in signals if s['ml_probability'] >= ml_threshold])}")
    print(f"  ML+LSTM approved: {len(high_conf)}")
    
    if signals:
        probs = np.array([s["final_probability"] for s in signals])
        print(f"  Prob range: {probs.min():.3f} - {probs.max():.3f}")
    
    return signals, df


def backtest(ml_artifact, lstm_model, lstm_mean, lstm_std,
             ml_threshold=0.55, lstm_threshold=0.50, test_fraction=0.2):
    """Backtest on out-of-sample data only"""
    print("\n" + "="*60)
    print("OUT-OF-SAMPLE BACKTEST: ML+LSTM XAUUSD M15 (3R/3R)")
    print("="*60)
    
    signals, df = generate_signals("XAUUSD_M15.csv", ml_artifact, lstm_model, lstm_mean, lstm_std,
                                     ml_threshold, lstm_threshold)
    
    if not signals:
        print("No signals generated")
        return
    
    # Filter to test period only (last test_fraction of bars)
    test_start = int(len(df) * (1 - test_fraction))
    test_signals = [s for s in signals if s['bar_index'] >= test_start]
    
    print(f"  Total signals: {len(signals)}")
    print(f"  Test-period signals (last {test_fraction:.0%}): {len(test_signals)}")
    print()
    
    if len(test_signals) < 5:
        print("  Too few test signals for meaningful stats")
        return
    
    # Simulate trades
    h = df['high'].to_numpy()
    l = df['low'].to_numpy()
    c = df['close'].to_numpy()
    atr_vals = pd.Series(h - l).rolling(14).mean().to_numpy()
    
    trades = []
    for signal in test_signals:
        idx = signal['bar_index']
        entry = c[idx]
        atr_val = atr_vals[idx]
        if np.isnan(atr_val) or atr_val <= 0:
            continue
        
        tp = entry + 3.0 * atr_val
        sl = entry - 3.0 * atr_val
        
        for j in range(idx + 1, min(idx + 50, len(df))):
            if l[j] <= sl:
                trades.append({'outcome': 'loss', 'pnl': -3.0, 'bars': j - idx})
                break
            if h[j] >= tp:
                trades.append({'outcome': 'win', 'pnl': 3.0, 'bars': j - idx})
                break
    
    if not trades:
        print("  No resolved trades")
        return
    
    wins = sum(1 for t in trades if t['outcome'] == 'win')
    losses = len(trades) - wins
    wr = wins / len(trades)
    total_pnl = sum(t['pnl'] for t in trades)
    avg_bars = np.mean([t['bars'] for t in trades])
    
    max_consec_losses = 0
    cur_losses = 0
    for t in trades:
        if t['outcome'] == 'loss':
            cur_losses += 1
            max_consec_losses = max(max_consec_losses, cur_losses)
        else:
            cur_losses = 0
    
    print(f"OUT-OF-SAMPLE BACKTEST RESULTS:")
    print(f"  Period: {df['timestamp'].iloc[test_start]} to {df['timestamp'].iloc[-1]}")
    print(f"  Trades: {len(trades)}")
    print(f"  Wins: {wins} ({wr:.1%})")
    print(f"  Losses: {losses} ({(1-wr):.1%})")
    print(f"  Total PnL: {total_pnl:.2f}R")
    print(f"  Avg PnL/trade: {total_pnl/len(trades):.2f}R")
    print(f"  Max consecutive losses: {max_consec_losses}")
    print(f"  Avg bars to resolve: {avg_bars:.1f} (~{avg_bars/4:.1f} hrs)")
    print()
    
    print("="*60)
    verdict = "TRADEABLE" if wr >= 0.60 else "PROMISING" if wr >= 0.50 else "NOT YET"
    print(f"VERDICT: {verdict}")
    print("="*60)
    
    # Summary for quick reference
    print()
    print("DEPLOYMENT CONFIG:")
    print(f"  ML threshold: {ml_threshold:.2f}")
    print(f"  LSTM threshold: {lstm_threshold:.2f}")
    print(f"  TP: 3.0 * ATR")
    print(f"  SL: 3.0 * ATR")
    print(f"  Expected WR: {wr:.1%}")
    print(f"  Expected EV: {total_pnl/len(trades):.2f}R per trade")
    print()
    print("RISK MANAGEMENT:")
    print(f"  Max cons. losses: {max_consec_losses}")
    print(f"  Recommended risk: 0.5% per trade (survive {max_consec_losses+1} losses)")
    print(f"  Stop trading after: {max_consec_losses * 2} consecutive losses")
    print()
    
    return trades


def main():
    parser = argparse.ArgumentParser(description="XAUUSD M15 Signal Generator")
    parser.add_argument("csv", nargs="?", help="CSV data file for signal generation")
    parser.add_argument("--backtest", action="store_true", help="Backtest on historical data")
    parser.add_argument("--ml-threshold", type=float, default=0.55)
    parser.add_argument("--lstm-threshold", type=float, default=0.50)
    parser.add_argument("--output", help="Save signals to JSON")
    args = parser.parse_args()
    
    print("="*60)
    print("XAUUSD M15 SIGNAL GENERATOR (3R/3R)")
    print("="*60)
    
    ml_artifact, lstm_model, lstm_mean, lstm_std = load_models()
    if ml_artifact is None:
        return
    
    if args.backtest:
        backtest(ml_artifact, lstm_model, lstm_mean, lstm_std, args.ml_threshold, args.lstm_threshold)
    
    elif args.csv:
        signals, _ = generate_signals(args.csv, ml_artifact, lstm_model, lstm_mean, lstm_std,
                                       args.ml_threshold, args.lstm_threshold)
        if args.output:
            out = [{'timestamp': s['timestamp'], 'close': s['close'],
                    'ml_prob': round(s['ml_probability'], 4),
                    'lstm_prob': round(s['lstm_probability'], 4) if s['lstm_probability'] else None,
                    'signal': s['signal_type']} for s in signals]
            with open(args.output, "w") as f:
                json.dump(out, f, indent=2)
            print(f"\nSaved {len(out)} signals to {args.output}")
    
    else:
        print("\nUsage:")
        print("  python deploy.py --backtest")
        print("  python deploy.py new_data.csv")
        print("  python deploy.py new_data.csv --output signals.json")


if __name__ == "__main__":
    main()
