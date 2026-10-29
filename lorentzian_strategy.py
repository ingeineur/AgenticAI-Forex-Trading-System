"""
Lorentzian Classification Strategy
ML-based: Approximate Nearest Neighbors + Lorentzian Distance
Pine Script reference: Lorentzian_Classification.txt

Usage:
   python lorentzian_strategy.py XAUUSD --backtest
   python lorentzian_strategy.py XAUUSD --backtest --tf h1
   python lorentzian_strategy.py XAUUSD --train-ml
   python lorentzian_strategy.py XAUUSD --backtest --ml-threshold 0.5
"""
import pandas as pd
import numpy as np
import argparse
import time
import logging
import sys
import os
import math
import pickle
from datetime import datetime, timezone, timedelta
from ta.volatility import AverageTrueRange
from ta.trend import ADXIndicator, CCIIndicator
from ta.momentum import RSIIndicator

eps = 1e-9

# ---------- ARGS ----------
parser = argparse.ArgumentParser(description="Lorentzian Classification Strategy")
parser.add_argument("symbol", default="XAUUSD", nargs="?", help="Symbol (XAUUSD, BTCUSD)")
parser.add_argument("--bars", type=int, default=5000, help="Bars to load")
parser.add_argument("--risk", type=float, default=10, help="Risk per trade in USD")
parser.add_argument("--max-orders", type=int, default=2, help="Max concurrent positions")
parser.add_argument("--lookback", type=int, default=50, help="Lookback bars (unused)")
parser.add_argument("--max-wait", type=int, default=10, help="Max bars (unused)")
parser.add_argument("--watch", action="store_true", help="Live monitoring mode")
parser.add_argument("--trade", action="store_true", help="Auto-execute with --watch")
parser.add_argument("--backtest", action="store_true", help="CSV backtest mode")
parser.add_argument("--interval", type=int, default=20, help="Poll interval in seconds")
parser.add_argument("--tf", choices=["m15", "h1"], default="m15", help="Timeframe")
parser.add_argument("--tp-mult", type=float, default=0.5, help="TP as multiple of 1R")
parser.add_argument("--sl-atr", type=float, default=0.0, help="SL as ATR multiple (0 = fixed USD)")
parser.add_argument("--tp-atr", type=float, default=0.0, help="TP as ATR multiple (0 = use --tp-mult)")
parser.add_argument("--train-ml", action="store_true", help="Train Random Forest on Lorentzian signals")
parser.add_argument("--ml-path", type=str, default=None, help="Path to load/save ML model")
parser.add_argument("--ml-threshold", type=float, default=0.0, help="Min ML probability (0.0 = disabled)")
# Lorentzian-specific params
parser.add_argument("--neighbors", type=int, default=8, help="Number of ANN neighbors (K)")
parser.add_argument("--max-bars-back", type=int, default=2000, help="Max history bars for ANN search")
parser.add_argument("--feature-count", type=int, default=5, choices=[2,3,4,5], help="Number of features for distance")
parser.add_argument("--hold-bars", type=int, default=4, help="Hold period in bars before exit")
parser.add_argument("--use-kernel-filter", action="store_true", default=True, help="Enable kernel regression filter")
parser.add_argument("--use-ema-filter", action="store_true", default=False, help="Enable EMA trend filter")
parser.add_argument("--use-sma-filter", action="store_true", default=False, help="Enable SMA trend filter")
parser.add_argument("--ema-period", type=int, default=200, help="EMA trend period")
parser.add_argument("--sma-period", type=int, default=200, help="SMA trend period")
parser.add_argument("--kernel-h", type=int, default=8, help="Kernel lookback window")
parser.add_argument("--kernel-r", type=float, default=8.0, help="Kernel relative weighting")
parser.add_argument("--kernel-x", type=int, default=25, help="Kernel regression level")
parser.add_argument("--use-dynamic-exits", action="store_true", default=False, help="Use kernel-based exits")
args = parser.parse_args()

SYMBOL = args.symbol.upper()
TF = args.tf
BARS = args.bars
RISK_USD = args.risk
MAX_ORDERS = args.max_orders
INTERVAL = args.interval
IS_BACKTEST = args.backtest
IS_WATCH = args.watch
IS_TRADE = args.trade
TP_MULT = args.tp_mult
SL_ATR = args.sl_atr
TP_ATR = args.tp_atr if args.tp_atr > 0 else args.sl_atr
TRAIN_ML = args.train_ml
ML_PATH = args.ml_path or f"lorentzian_ml_{SYMBOL}_{TF}.pkl"
ML_THRESHOLD = args.ml_threshold
K_NEIGHBORS = args.neighbors
MAX_BARS_BACK = args.max_bars_back
FEATURE_COUNT = args.feature_count
HOLD_BARS = args.hold_bars
USE_KERNEL_FILTER = args.use_kernel_filter
USE_EMA_FILTER = args.use_ema_filter
USE_SMA_FILTER = args.use_sma_filter
EMA_PERIOD = args.ema_period
SMA_PERIOD = args.sma_period
KERNEL_H = args.kernel_h
KERNEL_R = args.kernel_r
KERNEL_X = args.kernel_x
USE_DYNAMIC_EXITS = args.use_dynamic_exits

logging.basicConfig(filename=f"{SYMBOL}_lorentzian.log", level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s")

USD1_TO_PRICE = {"XAUUSD": 1, "BTCUSD": 100}
CSV_MAP = {
    "XAUUSD": {"m15": "XAUUSD_M15_combined.csv", "h1": "XAUUSD_H1_combined.csv"},
    "BTCUSD": {"m15": "BTCUSD_15_combined.csv", "h1": "BTCUSD_H1_combined.csv"},
}

# ---------- INDICATORS ----------
def add_indicators(df):
    df['body'] = abs(df['close'] - df['open'])
    df['range'] = df['high'] - df['low']
    atr14 = AverageTrueRange(df['high'], df['low'], df['close'], 14)
    df['atr'] = atr14.average_true_range()
    df['avg_vol'] = df['tick_volume'].rolling(20).mean()
    # Features for Lorentzian
    df['rsi14'] = RSIIndicator(df['close'], 14).rsi()
    df['rsi9'] = RSIIndicator(df['close'], 9).rsi()
    # WaveTrend: esa = ema(hlc3, 10), d = ema(abs(hlc3-esa), 10), ci = (hlc3-esa)/(0.015*d), tci = ema(ci, 11)
    hlc3 = (df['high'] + df['low'] + df['close']) / 3
    esa = hlc3.ewm(span=10, adjust=False).mean()
    d = (hlc3 - esa).abs().ewm(span=10, adjust=False).mean()
    ci = (hlc3 - esa) / (0.015 * d + eps)
    df['wt'] = ci.ewm(span=11, adjust=False).mean()
    # CCI
    df['cci'] = CCIIndicator(df['high'], df['low'], df['close'], 20).cci()
    # ADX
    adx = ADXIndicator(df['high'], df['low'], df['close'], 20)
    df['adx'] = adx.adx()
    return df

# ---------- KERNEL REGRESSION ----------
def rational_quadratic_kernel(source, h, r, x):
    """Nadaraya-Watson estimator with Rational Quadratic kernel."""
    n = len(source)
    result = np.full(n, np.nan)
    for i in range(h, n):
        start = i - h
        w = np.zeros(h)
        for j in range(h):
            dist = (x - i + start + j) ** 2
            alpha = max(x, h)
            w[j] = (1 + dist / (2 * r * alpha)) ** (-r)
        w_sum = np.sum(w)
        if w_sum > 0:
            result[i] = np.sum(w * source[start:i]) / w_sum
    return result

def gaussian_kernel(source, h, x):
    """Nadaraya-Watson estimator with Gaussian kernel."""
    n = len(source)
    result = np.full(n, np.nan)
    for i in range(h, n):
        start = i - h
        w = np.zeros(h)
        for j in range(h):
            dist = (x - i + start + j) ** 2
            w[j] = np.exp(-0.5 * dist / (h ** 2))
        w_sum = np.sum(w)
        if w_sum > 0:
            result[i] = np.sum(w * source[start:i]) / w_sum
    return result

# ---------- ANN PREDICTION ----------
def compute_lorentzian_distance(f1, f2):
    """Lorentzian distance = sum(ln(1 + |f1_i - f2_i|))"""
    return np.sum(np.log(1 + np.abs(f1 - f2)))

def ann_prediction(features, close, i, K, max_bars_back, feature_count, hold_bars):
    """Approximate Nearest Neighbors prediction for bar i."""
    start = max(hold_bars, i - max_bars_back)
    if i - start < K * 4:
        return 0
    indices = np.arange(start, i, 4)
    if len(indices) < 2:
        return 0
    f_i = features[i, :feature_count]
    # Vectorized Lorentzian distances
    f_hist = features[indices, :feature_count]
    diffs = np.abs(f_hist - f_i)
    distances = np.sum(np.log(1 + diffs), axis=1)
    # Keep K nearest
    if len(distances) > K:
        nearest_idx = np.argpartition(distances, K)[:K]
        nearest_dist = distances[nearest_idx]
        sorted_order = np.argsort(nearest_dist)
        nearest_idx = nearest_idx[sorted_order]
    else:
        nearest_idx = np.arange(len(distances))
    # Labels: sign of close[j+hold_bars] - close[j]
    hist_bars = indices[nearest_idx]
    labels = np.sign(close[np.minimum(hist_bars + hold_bars, len(close)-1)] - close[hist_bars])
    labels = np.clip(labels, -1, 1)
    prediction = int(np.sum(labels))
    return prediction

# ---------- SIGNAL GENERATION ----------
def get_lorentzian_signal(df, features, close, i, K, max_bars_back, feature_count, hold_bars):
    """Get raw ML signal (-1, 0, +1) at bar i."""
    pred = ann_prediction(features, close, i, K, max_bars_back, feature_count, hold_bars)
    if pred > 0:
        return 1
    elif pred < 0:
        return -1
    return 0

# ---------- ML FEATURES FOR RF ----------
def extract_ml_features_lz(o, h, l, c, body, atr_val, vol, avg_vol, atr_arr, ts_arr,
                           features_arr, i, direction, prediction_raw, sl_dist):
    """Extract feature vector for a Lorentzian trade."""
    f = np.zeros(24, dtype=np.float32)
    f[0] = atr_val / (c[i] + eps)
    f[1] = vol[i] / (avg_vol[i] + eps)
    f[2] = prediction_raw / 8.0  # normalized prediction strength
    f[3] = features_arr[i, 0] / 100.0  # RSI14
    f[4] = features_arr[i, 1] / 100.0  # WT
    f[5] = features_arr[i, 2] / 100.0  # CCI
    f[6] = features_arr[i, 3] / 100.0  # ADX
    f[7] = features_arr[i, 4] / 100.0  # RSI9
    ma50 = np.mean(c[max(0, i-50):i]) if i >= 50 else c[i]
    f[8] = (c[i] - ma50) / (atr_val + eps)
    ma200 = np.mean(c[max(0, i-200):i]) if i >= 200 else c[i]
    f[9] = (c[i] - ma200) / (atr_val + eps)
    f[10] = (ma50 - ma200) / (c[i] + eps)
    f[11] = 1.0 if direction == "BUY" else 0.0
    f[12] = body[i] / (atr_val + eps)
    f[13] = (h[i] - l[i]) / (atr_val + eps)
    entry_ts = pd.Timestamp(ts_arr[i])
    hour = entry_ts.hour if hasattr(entry_ts, 'hour') else 12
    f[14] = math.cos(2 * math.pi * hour / 24)
    f[15] = math.sin(2 * math.pi * hour / 24)
    atr50 = np.nanmean(atr_arr[max(0, i-50):i+1])
    f[16] = atr_val / (atr50 + eps)
    f[17] = sl_dist / (atr_val + eps)
    f[18] = vol[i] / (avg_vol[i] + eps)
    f[19] = atr_arr[i] / (atr_arr[max(0, i-50)] + eps)
    f[20] = c[i] > ma50 and 1.0 or 0.0
    f[21] = c[i] > ma200 and 1.0 or 0.0
    f[22] = 1.0 if direction == "BUY" else -1.0
    f[23] = prediction_raw  # raw prediction sum
    return f

def train_ml_lorentzian(all_features, all_labels):
    """Train Random Forest on collected Lorentzian trade data."""
    from sklearn.ensemble import RandomForestClassifier
    X = np.array(all_features, dtype=np.float32)
    y = np.array(all_labels, dtype=np.int32)
    n_trades = len(y)
    pos_ratio = y.mean()
    print(f"\n  ML Training: {n_trades} trades ({pos_ratio:.1%} win rate)")
    if n_trades < 100:
        print("  Too few trades, skipping"); return None
    split = int(n_trades * 0.7)
    X_train, y_train = X[:split], y[:split]
    X_test, y_test = X[split:], y[split:]
    model = RandomForestClassifier(
        n_estimators=200, max_depth=6, min_samples_split=20,
        min_samples_leaf=10, class_weight="balanced", random_state=42
    )
    model.fit(X_train, y_train)
    test_prob = model.predict_proba(X_test)[:, 1]
    best_th, best_net = 0.5, -999
    tp = TP_ATR if TP_ATR > 0 else TP_MULT
    for th in np.arange(0.3, 0.85, 0.05):
        preds = (test_prob >= th).astype(int)
        if preds.sum() == 0: continue
        net = (preds * y_test).sum() * tp - (preds * (1 - y_test)).sum() * 1
        if net > best_net: best_net, best_th = net, th
    test_preds = (test_prob >= best_th).astype(int)
    test_wr = (test_preds == y_test).mean()
    test_trades = test_preds.sum()
    test_wins = (test_preds * y_test).sum()
    test_net = test_wins * tp - (test_trades - test_wins) * 1
    feat_names = [
        "atr/close", "vol_ratio", "pred_str", "rsi14", "wt", "cci", "adx", "rsi9",
        "d_ma50", "d_ma200", "trend", "is_buy", "body/atr", "range/atr",
        "hr_cos", "hr_sin", "atr_trend", "risk_dist", "entry_vol", "atr_change",
        "abv_ma50", "abv_ma200", "dir_sign", "raw_pred"
    ]
    fi = sorted(zip(feat_names, model.feature_importances_), key=lambda x: -x[1])
    print(f"  Time-split: train={split} trades, test={n_trades-split}")
    print(f"  Test WR: {y_test.mean():.1%} baseline -> {test_wr:.1%} @ th={best_th:.2f} ({test_trades} tr, {test_net:+.1f}R)")
    print(f"\n  Top 10 features:")
    for name, imp in fi[:10]:
        print(f"    {name:16s}  {imp:.3f}")
    artifact = {"model": model, "threshold": best_th, "features": feat_names}
    with open(ML_PATH, "wb") as f:
        pickle.dump(artifact, f)
    print(f"  Model saved to {ML_PATH}")
    return model

# ---------- BACKTEST ----------
def run_backtest():
    print(f"\n{'='*60}")
    print(f"  LORENTZIAN CLASSIFICATION BACKTEST")
    print(f"  {SYMBOL} TF={TF} K={K_NEIGHBORS} Hold={HOLD_BARS}b")
    risk_label = f"SL={SL_ATR:.1f}ATR TP={TP_ATR:.1f}ATR" if SL_ATR > 0 else f"TP={TP_MULT}R"
    print(f"  {risk_label}")
    print(f"{'='*60}")
    csv_path = CSV_MAP.get(SYMBOL, {}).get(TF)
    if not csv_path: print(f"  ERROR: No CSV for {SYMBOL}/{TF}"); return
    if not os.path.exists(csv_path): print(f"  ERROR: {csv_path} not found"); return
    df = pd.read_csv(csv_path)
    if 'volume' in df.columns and 'tick_volume' not in df.columns:
        df['tick_volume'] = df['volume']
    if 'real_volume' in df.columns: df['volume'] = df['real_volume']
    df['timestamp'] = pd.to_datetime(df['timestamp'], utc=True)
    df = df.sort_values('timestamp').reset_index(drop=True)
    if BARS and BARS < len(df): df = df.tail(BARS).reset_index(drop=True)
    print(f"  Bars: {len(df)} ({df['timestamp'].iloc[0]} -> {df['timestamp'].iloc[-1]})")
    df = add_indicators(df)
    n = len(df)
    if n < 500: print("  Too few bars"); return

    o = df['open'].values.astype(float)
    h = df['high'].values.astype(float)
    l = df['low'].values.astype(float)
    c = df['close'].values.astype(float)
    body = df['body'].values.astype(float)
    atr = df['atr'].values.astype(float)
    vol = df['tick_volume'].values.astype(float)
    avg_vol = df['avg_vol'].values.astype(float)
    ts = df['timestamp'].values

    # Build feature matrix for ANN
    features = np.column_stack([
        df['rsi14'].values.astype(float),
        df['wt'].values.astype(float),
        df['cci'].values.astype(float),
        df['adx'].values.astype(float),
        df['rsi9'].values.astype(float),
    ])

    # Kernel regression
    kernel_raw = rational_quadratic_kernel(c, KERNEL_H, KERNEL_R, KERNEL_X)
    kernel_lag = gaussian_kernel(c, KERNEL_H - 2, KERNEL_X) if KERNEL_H > 2 else kernel_raw.copy()

    # Trend filters
    ema_period_val = min(EMA_PERIOD, n-1)
    sma_period_val = min(SMA_PERIOD, n-1)
    ema = pd.Series(c).ewm(span=ema_period_val, adjust=False).mean().values
    sma = pd.Series(c).rolling(sma_period_val).mean().values

    one_r = RISK_USD * USD1_TO_PRICE.get(SYMBOL, 1)
    rwd = one_r * TP_MULT

    trades = wins = losses = 0
    ml_collect = []
    ml_model = None
    if TRAIN_ML:
        print(f"  ML training mode: collecting features from all trades")
    elif ML_THRESHOLD > 0 and os.path.exists(ML_PATH):
        with open(ML_PATH, "rb") as f:
            artifact = pickle.load(f)
        ml_model = artifact["model"]
        print(f"  ML filter: loaded {ML_PATH}, threshold={ML_THRESHOLD:.2f}")
    elif ML_THRESHOLD > 0:
        print(f"  WARNING: {ML_PATH} not found, ML filter disabled")

    print(f"  Scanning {n} bars...")

    # State for signal-based exits
    in_trade = False
    direction = None
    entry_bar = 0
    entry_price = 0
    prev_signal = 0
    sl_price = 0
    tp_price = 0

    train_start = max(HOLD_BARS + 1, MAX_BARS_BACK + 1, 200)

    for i in range(train_start, n):
        if np.isnan(atr[i]) or atr[i] <= 0:
            continue

        # Get raw Lorentzian signal
        signal = get_lorentzian_signal(df, features, c, i, K_NEIGHBORS, MAX_BARS_BACK, FEATURE_COUNT, HOLD_BARS)
        signal_changed = signal != 0 and signal != prev_signal
        prev_signal_was_zero = prev_signal == 0

        # Filters
        is_bullish_kernel = False
        is_bearish_kernel = False
        if USE_KERNEL_FILTER and not np.isnan(kernel_raw[i]) and i > 0 and not np.isnan(kernel_raw[i-1]):
            is_bullish_kernel = kernel_raw[i-1] < kernel_raw[i]
            is_bearish_kernel = kernel_raw[i-1] > kernel_raw[i]
        else:
            is_bullish_kernel = signal > 0
            is_bearish_kernel = signal < 0

        is_ema_up = c[i] > ema[i] if USE_EMA_FILTER and not np.isnan(ema[i]) else True
        is_ema_down = c[i] < ema[i] if USE_EMA_FILTER and not np.isnan(ema[i]) else True
        is_sma_up = c[i] > sma[i] if USE_SMA_FILTER and not np.isnan(sma[i]) else True
        is_sma_down = c[i] < sma[i] if USE_SMA_FILTER and not np.isnan(sma[i]) else True

        is_new_buy = signal == 1 and signal_changed and is_bullish_kernel and is_ema_up and is_sma_up
        is_new_sell = signal == -1 and signal_changed and is_bearish_kernel and is_ema_down and is_sma_down

        # --- ENTRY ---
        if not in_trade:
            if is_new_buy:
                # Compute SL/TP
                if SL_ATR > 0:
                    sl_dist = atr[i] * SL_ATR
                    tp_dist = atr[i] * TP_ATR
                else:
                    sl_dist = one_r; tp_dist = rwd
                entry_price = c[i]
                direction = "BUY"
                entry_bar = i
                in_trade = True
                sl_price = entry_price - sl_dist
                tp_price = entry_price + tp_dist
                # ML filter
                if ml_model is not None:
                    feat = extract_ml_features_lz(o, h, l, c, body, atr[i], vol, avg_vol, atr, ts,
                        features, i, "BUY", signal, sl_dist)
                    prob = ml_model.predict_proba(feat.reshape(1, -1))[0, 1]
                    if prob < ML_THRESHOLD:
                        in_trade = False; continue
                if TRAIN_ML:
                    feat = extract_ml_features_lz(o, h, l, c, body, atr[i], vol, avg_vol, atr, ts,
                        features, i, "BUY", signal, sl_dist)
                    ml_collect.append((feat, -1))  # label filled at exit

            elif is_new_sell:
                if SL_ATR > 0:
                    sl_dist = atr[i] * SL_ATR
                    tp_dist = atr[i] * TP_ATR
                else:
                    sl_dist = one_r; tp_dist = rwd
                entry_price = c[i]
                direction = "SELL"
                entry_bar = i
                in_trade = True
                sl_price = entry_price + sl_dist
                tp_price = entry_price - tp_dist
                if ml_model is not None:
                    feat = extract_ml_features_lz(o, h, l, c, body, atr[i], vol, avg_vol, atr, ts,
                        features, i, "SELL", signal, sl_dist)
                    prob = ml_model.predict_proba(feat.reshape(1, -1))[0, 1]
                    if prob < ML_THRESHOLD:
                        in_trade = False; continue
                if TRAIN_ML:
                    feat = extract_ml_features_lz(o, h, l, c, body, atr[i], vol, avg_vol, atr, ts,
                        features, i, "SELL", signal, sl_dist)
                    ml_collect.append((feat, -1))  # label filled at exit

        # --- EXIT ---
        if in_trade:
            bars_held = i - entry_bar
            exit_now = False
            exit_reason = ""
            exit_price = c[i]

            # SL/TP hit (only if ATR-based or fixed risk)
            if SL_ATR > 0 or direction is not None:
                if direction == "BUY":
                    if l[i] <= sl_price:
                        exit_now = True; exit_reason = "SL"; exit_price = sl_price
                    elif h[i] >= tp_price:
                        exit_now = True; exit_reason = "TP"; exit_price = tp_price
                else:
                    if h[i] >= sl_price:
                        exit_now = True; exit_reason = "SL"; exit_price = sl_price
                    elif l[i] <= tp_price:
                        exit_now = True; exit_reason = "TP"; exit_price = tp_price

            # Strict exits: 4-bar hold or signal flip
            if not exit_now:
                if USE_DYNAMIC_EXITS:
                    if direction == "BUY" and is_bearish_kernel:
                        exit_now = True; exit_reason = "KERNEL"
                    elif direction == "SELL" and is_bullish_kernel:
                        exit_now = True; exit_reason = "KERNEL"
                else:
                    is_last_signal_buy = prev_signal == 1
                    is_last_signal_sell = prev_signal == -1
                    if bars_held >= HOLD_BARS:
                        exit_now = True; exit_reason = "HOLD"
                    elif direction == "BUY" and is_new_sell:
                        exit_now = True; exit_reason = "FLIP"
                    elif direction == "SELL" and is_new_buy:
                        exit_now = True; exit_reason = "FLIP"

            if exit_now:
                in_trade = False
                trades += 1
                if direction == "BUY":
                    hit = exit_price >= entry_price
                else:
                    hit = exit_price <= entry_price
                if hit: wins += 1
                else: losses += 1
                # Update ML training label
                if TRAIN_ML and len(ml_collect) > 0 and ml_collect[-1][1] == -1:
                    ml_collect[-1] = (ml_collect[-1][0], int(hit))

        prev_signal = signal if signal != 0 else prev_signal

    print(f"\n  RESULTS ({SYMBOL} {TF})")
    print(f"  {'='*40}")
    if trades:
        wr = wins / trades
        pnl = wins * (TP_ATR if TP_ATR > 0 else TP_MULT) - losses * 1
        print(f"  Trades:  {trades:,}")
        print(f"  Wins:    {wins:,}   Losses: {losses:,}")
        print(f"  WR:      {wr:.1%}")
        print(f"  PnL:     {pnl:+,.0f}R")
    else:
        print("  No trades")

    if TRAIN_ML and len(ml_collect) >= 100:
        all_features = [x[0] for x in ml_collect if x[1] != -1]
        all_labels = [x[1] for x in ml_collect if x[1] != -1]
        if len(all_features) >= 100:
            train_ml_lorentzian(all_features, all_labels)


# ---------- LIVE ----------
def run_live():
    print(f"\n{'='*60}")
    print(f"  LORENTZIAN CLASSIFICATION LIVE ({TF})")
    print(f"  {SYMBOL} --- Polling every {INTERVAL}s")
    if IS_TRADE: print("  Trade mode: ACTIVE")
    else: print("  Monitor only (add --trade)")
    print(f"{'='*60}")
    import MetaTrader5 as mt5
    if not mt5.initialize(): print("  ERROR: MT5 init failed"); return
    acc, pwd, srv = int(os.environ["MT5_ACCOUNT"]), os.environ["MT5_PASSWORD"], "Pepperstone-Demo"
    if not mt5.login(acc, pwd, srv):
        print("  Login failed...")
        if not mt5.login(int(os.environ["MT5_ACCOUNT"]), os.environ["MT5_PASSWORD"], "Pepperstone-Demo"):
            print("  ERROR: Login failed"); mt5.shutdown(); return
    print(f"  Logged in: {acc}")
    mt5.symbol_select(SYMBOL, True)

    live_ml_model = None
    if ML_THRESHOLD > 0 and os.path.exists(ML_PATH):
        with open(ML_PATH, "rb") as f:
            artifact = pickle.load(f)
        live_ml_model = artifact["model"]
        print(f"  ML filter: loaded {ML_PATH}, threshold={ML_THRESHOLD:.2f}")

    last_sig_key = None
    LOGGED = 0
    while True:
        try:
            tf_mt5 = mt5.TIMEFRAME_M15 if TF == "m15" else mt5.TIMEFRAME_H1
            rates = mt5.copy_rates_from_pos(SYMBOL, tf_mt5, 0, BARS)
            if rates is None: time.sleep(INTERVAL); continue
            df = pd.DataFrame(rates)
            df['timestamp'] = pd.to_datetime(df['time'], unit='s')
            df = add_indicators(df)
            n = len(df)
            if n < 500: time.sleep(INTERVAL); continue

            features = np.column_stack([
                df['rsi14'].values.astype(float),
                df['wt'].values.astype(float),
                df['cci'].values.astype(float),
                df['adx'].values.astype(float),
                df['rsi9'].values.astype(float),
            ])
            c_arr = df['close'].values.astype(float)
            atr_arr = df['atr'].values.astype(float)
            o_arr = df['open'].values.astype(float)
            h_arr = df['high'].values.astype(float)
            l_arr = df['low'].values.astype(float)
            body_arr = np.abs(c_arr - o_arr)
            vol_arr = df['tick_volume'].values.astype(float)
            avg_vol_arr = df['avg_vol'].values.astype(float)
            ts_arr = df['timestamp'].values

            signal = get_lorentzian_signal(df, features, c_arr, n-1, K_NEIGHBORS, MAX_BARS_BACK, FEATURE_COUNT, HOLD_BARS)
            setup = None
            if signal == 1:
                setup = {"direction": "BUY", "entry_price": c_arr[-1], "signal_ts": df.iloc[-1]['timestamp'],
                         "atr": atr_arr[-1], "signal": signal, "sig_source": "LORENTZIAN"}
            elif signal == -1:
                setup = {"direction": "SELL", "entry_price": c_arr[-1], "signal_ts": df.iloc[-1]['timestamp'],
                         "atr": atr_arr[-1], "signal": signal, "sig_source": "LORENTZIAN"}

            if setup and live_ml_model is not None:
                ej = n - 1
                sl_d = atr_arr[ej] * SL_ATR if SL_ATR > 0 else RISK_USD * USD1_TO_PRICE.get(SYMBOL, 1)
                feat = extract_ml_features_lz(o_arr, h_arr, l_arr, c_arr, body_arr, atr_arr[ej],
                    vol_arr, avg_vol_arr, atr_arr, ts_arr, features, ej,
                    setup['direction'], signal, sl_d)
                prob = live_ml_model.predict_proba(feat.reshape(1, -1))[0, 1]
                setup['ml_prob'] = prob
                if prob < ML_THRESHOLD:
                    setup = None

            if setup:
                key = f"{setup['direction']}_{setup['entry_price']:.1f}"
                if key != last_sig_key or LOGGED == 0:
                    print_setup_lz(setup); LOGGED += 1; last_sig_key = key
                    if IS_TRADE:
                        n_pos = count_positions(mt5, ["LZ"])
                        if n_pos < MAX_ORDERS:
                            result = place_mt5_trade_lz(mt5, setup)
                            if result:
                                print(f"  >>> ORDER: retcode={result.retcode}")
                                logging.info(f"Trade: {setup['direction']} {SYMBOL} retcode={result.retcode}")
                        else: print(f"  Max orders ({MAX_ORDERS}) reached")
        except KeyboardInterrupt: print("\nStopped"); mt5.shutdown(); break
        except Exception as e: print(f"  Error: {e}")
        time.sleep(INTERVAL)
    mt5.shutdown()

def print_setup_lz(setup):
    print(f"\n{'='*60}")
    print(f"  LORENTZIAN SIGNAL: {setup['sig_source']}")
    print(f"{'='*60}")
    print(f"  Symbol:    {SYMBOL}")
    print(f"  Direction: {setup['direction']}")
    print(f"  Time:      {setup['signal_ts']}")
    print(f"  Entry:     {setup['entry_price']:.2f}")
    print(f"  ATR:       {setup['atr']:.2f}")
    print(f"  Signal:    {setup['signal']}")
    if 'ml_prob' in setup:
        print(f"  ML Prob:   {setup['ml_prob']:.2%}")
    print(f"{'='*60}\n")

def count_positions(mt5, keywords=None):
    if mt5 is None: return 0
    positions = mt5.positions_get(symbol=SYMBOL)
    if positions is None: return 0
    if keywords:
        return sum(1 for p in positions if all(k in (p.comment or "") for k in keywords))
    return len(positions)

def place_mt5_trade_lz(mt5, setup):
    direction = setup['direction']
    tick = mt5.symbol_info_tick(SYMBOL)
    info = mt5.symbol_info(SYMBOL)
    if tick is None or info is None: print("  ERROR: No symbol data"); return None
    price = tick.ask if direction == "BUY" else tick.bid
    atr_val = setup['atr']
    if SL_ATR > 0:
        sl_dist = atr_val * SL_ATR
        tp_dist = atr_val * TP_ATR
    else:
        sl_dist = RISK_USD * USD1_TO_PRICE.get(SYMBOL, 1)
        tp_dist = sl_dist * TP_MULT
    if direction == "BUY":
        sl = price - sl_dist; tp = price + tp_dist
    else:
        sl = price + sl_dist; tp = price - tp_dist
    ts = info.trade_tick_size or 0.01
    sl = round(sl / ts) * ts; tp = round(tp / ts) * ts
    min_dist = info.trade_stops_level * info.point if info.trade_stops_level else 0
    if min_dist > 0:
        if direction == "BUY":
            if price - sl < min_dist: sl = price - min_dist
            if tp - price < min_dist: tp = price + min_dist
        else:
            if sl - price < min_dist: sl = price + min_dist
            if price - tp < min_dist: tp = price - min_dist
    order_type = mt5.ORDER_TYPE_BUY if direction == "BUY" else mt5.ORDER_TYPE_SELL
    comment = f"LZ {direction[:1]} {setup.get('signal',0)}"
    request = {
        "action": mt5.TRADE_ACTION_DEAL, "symbol": SYMBOL, "volume": 0.01,
        "type": order_type, "price": price, "sl": sl, "tp": tp,
        "deviation": 20, "magic": 1016, "comment": comment,
        "type_time": mt5.ORDER_TIME_GTC, "type_filling": mt5.ORDER_FILLING_IOC,
    }
    return mt5.order_send(request)

# ---------- MAIN ----------
if __name__ == "__main__":
    if IS_BACKTEST or TRAIN_ML:
        run_backtest()
    elif IS_WATCH:
        run_live()
    else:
        print("Use --backtest or --watch")
        print(f"  python lorentzian_strategy.py {SYMBOL} --backtest")
