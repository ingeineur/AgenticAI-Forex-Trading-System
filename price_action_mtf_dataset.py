import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd


BASE_FEATURES = [
    "ret_1",
    "body_atr",
    "range_atr",
    "upper_wick_atr",
    "lower_wick_atr",
    "close_to_ema20_atr",
    "close_to_ema50_atr",
    "ema20_slope_atr",
    "ema50_slope_atr",
    "volume_ratio",
    "spread_atr",
    "sweep_long",
    "sweep_short",
    "breakout_long",
    "breakout_short",
    "engulf_long",
    "engulf_short",
    "displacement_up",
    "displacement_down",
]

HTF_FRAMES = {
    "H1": "1h",
    "H4": "4h",
    "D1": "1D",
}


def read_ohlcv(path):
    df = pd.read_csv(path)
    if "timestamp" not in df.columns and "time" in df.columns:
        df = df.rename(columns={"time": "timestamp"})
    required = {"timestamp", "open", "high", "low", "close"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Missing required columns: {sorted(missing)}")

    df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True)
    if "volume" not in df.columns:
        if "tick_volume" in df.columns:
            df["volume"] = df["tick_volume"]
        else:
            df["volume"] = 0.0
    if "spread" not in df.columns:
        df["spread"] = 0.0
    if "real_volume" not in df.columns:
        df["real_volume"] = 0.0

    df = df.sort_values("timestamp").drop_duplicates("timestamp").reset_index(drop=True)
    return df


def atr(df, period=14):
    high = df["high"]
    low = df["low"]
    close = df["close"]
    tr = pd.concat(
        [
            high - low,
            (high - close.shift()).abs(),
            (low - close.shift()).abs(),
        ],
        axis=1,
    ).max(axis=1)
    return tr.rolling(period).mean()


def add_base_price_action(df, swing_lookback=20, range_lookback=20):
    df = df.copy()
    eps = 1e-9

    df["atr"] = atr(df)
    df["ret_1"] = df["close"].pct_change().fillna(0.0)
    df["body"] = (df["close"] - df["open"]).abs()
    df["range"] = df["high"] - df["low"]
    df["upper_wick"] = df["high"] - df[["open", "close"]].max(axis=1)
    df["lower_wick"] = df[["open", "close"]].min(axis=1) - df["low"]

    df["body_atr"] = df["body"] / (df["atr"] + eps)
    df["range_atr"] = df["range"] / (df["atr"] + eps)
    df["upper_wick_atr"] = df["upper_wick"] / (df["atr"] + eps)
    df["lower_wick_atr"] = df["lower_wick"] / (df["atr"] + eps)

    df["ema20"] = df["close"].ewm(span=20, adjust=False).mean()
    df["ema50"] = df["close"].ewm(span=50, adjust=False).mean()
    df["close_to_ema20_atr"] = (df["close"] - df["ema20"]) / (df["atr"] + eps)
    df["close_to_ema50_atr"] = (df["close"] - df["ema50"]) / (df["atr"] + eps)
    df["ema20_slope_atr"] = df["ema20"].diff(5) / (df["atr"] + eps)
    df["ema50_slope_atr"] = df["ema50"].diff(5) / (df["atr"] + eps)

    df["volume_ma"] = df["volume"].rolling(20).mean()
    df["volume_ratio"] = df["volume"] / (df["volume_ma"] + eps)
    df["spread_atr"] = df["spread"] / (df["atr"] + eps)

    swing_high = df["high"].rolling(swing_lookback).max().shift(1)
    swing_low = df["low"].rolling(swing_lookback).min().shift(1)
    df["swing_high"] = swing_high
    df["swing_low"] = swing_low

    df["sweep_short"] = (df["high"] > swing_high) & (df["close"] < swing_high)
    df["sweep_long"] = (df["low"] < swing_low) & (df["close"] > swing_low)

    range_high = df["high"].rolling(range_lookback).max().shift(1)
    range_low = df["low"].rolling(range_lookback).min().shift(1)
    range_size = range_high - range_low
    range_ma = range_size.rolling(range_lookback).mean()
    compression = range_size / (range_ma + eps)

    df["range_high"] = range_high
    df["range_low"] = range_low
    df["range_compression"] = compression
    df["breakout_long"] = (df["close"] > range_high) & (compression < 0.9)
    df["breakout_short"] = (df["close"] < range_low) & (compression < 0.9)

    prev_open = df["open"].shift(1)
    prev_close = df["close"].shift(1)
    df["engulf_long"] = (
        (df["close"] > df["open"])
        & (prev_close < prev_open)
        & (df["close"] > prev_open)
        & (df["open"] < prev_close)
        & (df["body_atr"] > 0.6)
    )
    df["engulf_short"] = (
        (df["close"] < df["open"])
        & (prev_close > prev_open)
        & (df["close"] < prev_open)
        & (df["open"] > prev_close)
        & (df["body_atr"] > 0.6)
    )

    df["displacement_up"] = (df["close"] > df["open"]) & (df["range_atr"] > 1.25)
    df["displacement_down"] = (df["close"] < df["open"]) & (df["range_atr"] > 1.25)

    for col in [
        "sweep_long",
        "sweep_short",
        "breakout_long",
        "breakout_short",
        "engulf_long",
        "engulf_short",
        "displacement_up",
        "displacement_down",
    ]:
        df[col] = df[col].astype(float)

    df["candidate_long"] = (
        (df["sweep_long"] > 0)
        | (df["breakout_long"] > 0)
        | (df["engulf_long"] > 0)
        | (df["displacement_up"] > 0)
    )
    df["candidate_short"] = (
        (df["sweep_short"] > 0)
        | (df["breakout_short"] > 0)
        | (df["engulf_short"] > 0)
        | (df["displacement_down"] > 0)
    )
    df["setup_direction"] = np.select(
        [df["candidate_long"], df["candidate_short"]],
        [1, -1],
        default=0,
    )
    return df


def resample_ohlcv(df, rule):
    indexed = df.set_index("timestamp")
    htf = indexed.resample(rule, label="right", closed="right").agg(
        {
            "open": "first",
            "high": "max",
            "low": "min",
            "close": "last",
            "volume": "sum",
            "spread": "mean",
            "real_volume": "sum",
        }
    )
    htf = htf.dropna(subset=["open", "high", "low", "close"]).reset_index()
    return htf


def htf_features(htf, prefix):
    htf = htf.copy()
    eps = 1e-9
    htf[f"{prefix}_atr"] = atr(htf)
    htf[f"{prefix}_ema20"] = htf["close"].ewm(span=20, adjust=False).mean()
    htf[f"{prefix}_ema50"] = htf["close"].ewm(span=50, adjust=False).mean()
    htf[f"{prefix}_trend"] = np.sign(htf[f"{prefix}_ema20"] - htf[f"{prefix}_ema50"])
    htf[f"{prefix}_ema20_slope_atr"] = htf[f"{prefix}_ema20"].diff(3) / (htf[f"{prefix}_atr"] + eps)
    htf[f"{prefix}_close_to_ema20_atr"] = (htf["close"] - htf[f"{prefix}_ema20"]) / (htf[f"{prefix}_atr"] + eps)
    htf[f"{prefix}_swing_high"] = htf["high"].rolling(20).max()
    htf[f"{prefix}_swing_low"] = htf["low"].rolling(20).min()
    htf[f"{prefix}_dist_to_swing_high_atr"] = (htf[f"{prefix}_swing_high"] - htf["close"]) / (htf[f"{prefix}_atr"] + eps)
    htf[f"{prefix}_dist_to_swing_low_atr"] = (htf["close"] - htf[f"{prefix}_swing_low"]) / (htf[f"{prefix}_atr"] + eps)
    htf[f"{prefix}_ret"] = htf["close"].pct_change().fillna(0.0)
    keep = [
        "timestamp",
        f"{prefix}_trend",
        f"{prefix}_ema20_slope_atr",
        f"{prefix}_close_to_ema20_atr",
        f"{prefix}_dist_to_swing_high_atr",
        f"{prefix}_dist_to_swing_low_atr",
        f"{prefix}_ret",
    ]
    return htf[keep]


def add_htf_context(base):
    out = base.copy()
    for name, rule in HTF_FRAMES.items():
        features = htf_features(resample_ohlcv(base, rule), name)
        out = pd.merge_asof(
            out.sort_values("timestamp"),
            features.sort_values("timestamp"),
            on="timestamp",
            direction="backward",
        )

    direction = out["setup_direction"].replace(0, np.nan)
    trend_cols = [f"{name}_trend" for name in HTF_FRAMES]
    out["htf_alignment"] = out[trend_cols].mul(direction, axis=0).mean(axis=1).fillna(0.0)

    for name in HTF_FRAMES:
        out[f"{name}_return_corr"] = (
            out["ret_1"].rolling(32).corr(out[f"{name}_ret"].fillna(0.0)).fillna(0.0)
        )

    return out


def generate_labels(df, tp_mult=2.0, sl_mult=1.0, max_horizon=50):
    labels = np.full(len(df), np.nan)
    directions = df["setup_direction"].to_numpy()
    high = df["high"].to_numpy()
    low = df["low"].to_numpy()
    close = df["close"].to_numpy()
    atr_values = df["atr"].to_numpy()

    for i in range(len(df) - max_horizon):
        direction = directions[i]
        if direction == 0 or np.isnan(atr_values[i]) or atr_values[i] <= 0:
            continue

        entry = close[i]
        if direction == 1:
            tp = entry + tp_mult * atr_values[i]
            sl = entry - sl_mult * atr_values[i]
            for j in range(i + 1, i + max_horizon + 1):
                if low[j] <= sl:
                    labels[i] = 0
                    break
                if high[j] >= tp:
                    labels[i] = 1
                    break
        elif direction == -1:
            tp = entry - tp_mult * atr_values[i]
            sl = entry + sl_mult * atr_values[i]
            for j in range(i + 1, i + max_horizon + 1):
                if high[j] >= sl:
                    labels[i] = 0
                    break
                if low[j] <= tp:
                    labels[i] = 1
                    break

    return labels


def build_sequences(df, feature_cols, sequence_length):
    X = []
    y = []
    meta_rows = []
    values = df[feature_cols].replace([np.inf, -np.inf], np.nan).fillna(0.0).to_numpy(dtype=np.float32)
    labels = df["label"].to_numpy()

    for i in range(sequence_length - 1, len(df)):
        if np.isnan(labels[i]):
            continue
        if df["setup_direction"].iloc[i] == 0:
            continue
        X.append(values[i - sequence_length + 1 : i + 1])
        y.append(labels[i])
        meta_rows.append(
            {
                "timestamp": df["timestamp"].iloc[i],
                "label": int(labels[i]),
                "setup_direction": int(df["setup_direction"].iloc[i]),
                "close": float(df["close"].iloc[i]),
                "atr": float(df["atr"].iloc[i]),
                "htf_alignment": float(df["htf_alignment"].iloc[i]),
            }
        )

    if not X:
        raise ValueError("No labeled sequences were generated. Try a larger input file or lower the filters.")

    return np.stack(X), np.asarray(y, dtype=np.float32), pd.DataFrame(meta_rows)


def main():
    parser = argparse.ArgumentParser(description="Build an LSTM-ready price-action, multi-timeframe dataset.")
    parser.add_argument("input_csv")
    parser.add_argument("--output", default="price_action_mtf_sequences.npz")
    parser.add_argument("--meta-output", default="price_action_mtf_meta.csv")
    parser.add_argument("--sequence-length", type=int, default=64)
    parser.add_argument("--tp-mult", type=float, default=2.0)
    parser.add_argument("--sl-mult", type=float, default=1.0)
    parser.add_argument("--max-horizon", type=int, default=50)
    args = parser.parse_args()

    df = read_ohlcv(args.input_csv)
    df = add_base_price_action(df)
    df = add_htf_context(df)
    df["label"] = generate_labels(df, args.tp_mult, args.sl_mult, args.max_horizon)

    htf_feature_cols = []
    for name in HTF_FRAMES:
        htf_feature_cols.extend(
            [
                f"{name}_trend",
                f"{name}_ema20_slope_atr",
                f"{name}_close_to_ema20_atr",
                f"{name}_dist_to_swing_high_atr",
                f"{name}_dist_to_swing_low_atr",
                f"{name}_ret",
                f"{name}_return_corr",
            ]
        )

    feature_cols = BASE_FEATURES + htf_feature_cols + ["setup_direction", "htf_alignment"]
    X, y, meta = build_sequences(df, feature_cols, args.sequence_length)

    np.savez_compressed(
        args.output,
        X=X,
        y=y,
        feature_cols=np.asarray(feature_cols),
        sequence_length=np.asarray(args.sequence_length),
        config=np.asarray(
            json.dumps(
                {
                    "input_csv": str(Path(args.input_csv)),
                    "tp_mult": args.tp_mult,
                    "sl_mult": args.sl_mult,
                    "max_horizon": args.max_horizon,
                }
            )
        ),
    )
    meta.to_csv(args.meta_output, index=False)

    print(f"Saved {len(y)} sequences to {args.output}")
    print(f"Saved metadata to {args.meta_output}")
    print(f"Positive label rate: {y.mean():.3f}")
    print(f"Features: {len(feature_cols)}")


if __name__ == "__main__":
    main()

