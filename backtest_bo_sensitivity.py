#!/usr/bin/env python3
"""Backtest: breakout + liquidity sweep reversal signals."""
import pandas as pd, numpy as np, math, itertools, sys, argparse
from ta.volatility import AverageTrueRange

eps = 1e-9; SMALL_VAL = -1e-9
MAX_WAIT = 15

parser = argparse.ArgumentParser(description="Backtest breakout + reversal strategy")
parser.add_argument("--fractal-strength", type=int, default=2, help="Fractal strength (default 2)")
parser.add_argument("--lookback", type=int, default=50, help="Lookback bars for fractal S/R (default 50)")
args = parser.parse_args()
FRACTAL_STRENGTH = args.fractal_strength
LOOKBACK = args.lookback

CONFIG = {
    "XAUUSD": {"risk_usd": 10, "usd1_to_price": 1, "risk_to_reward": 0.5, "csv": "XAUUSD_M15_combined.csv"},
    "BTCUSD": {"risk_usd": 10, "usd1_to_price": 100, "risk_to_reward": 0.4, "csv": "BTCUSD_15_combined.csv"},
}

def add_indicators(df):
    df['body'] = (df['close'] - df['open']).abs()
    atr = AverageTrueRange(df['high'], df['low'], df['close'], 14)
    df['atr'] = atr.average_true_range()
    df['ema50'] = df['close'].ewm(span=50).mean()
    df['ema200'] = df['close'].ewm(span=200).mean()
    return df

def fast_fractals(highs, lows, strength=2):
    n = len(highs); fh = np.zeros(n, dtype=bool); fl = np.zeros(n, dtype=bool)
    s = strength
    for i in range(s, n - s):
        fh[i] = all(highs[i-k] < highs[i] and highs[i+k] < highs[i] for k in range(1, s+1))
        fl[i] = all(lows[i-k] > lows[i] and lows[i+k] > lows[i] for k in range(1, s+1))
    return fh, fl

def resample_h1(m15):
    m15 = m15.copy(); m15['time'] = m15['timestamp']
    return m15.resample('1h', on='time').agg({
        'open': 'first', 'high': 'max', 'low': 'min',
        'close': 'last', 'tick_volume': 'sum',
    }).dropna().reset_index()

def precompute_h1_state(h1):
    n = len(h1)
    fh, fl = fast_fractals(h1['high'].values, h1['low'].values, FRACTAL_STRENGTH)
    bias_s = np.full(n, "NEUTRAL", dtype=object)
    bias_r = np.full(n, "NEUTRAL", dtype=object)
    h1_sup = np.full(n, np.nan); h1_res = np.full(n, np.nan)
    last_fh = np.nan; last_fl = np.nan
    for i in range(n):
        if fh[i]: last_fh = h1['high'].iloc[i]
        if fl[i]: last_fl = h1['low'].iloc[i]
        h1_res[i] = last_fh; h1_sup[i] = last_fl
    for i in range(3, n):
        r = h1.iloc[i-3:i+1]
        bull = sum(r['close'] > r['open'])
        bear = 4 - bull
        net = r['close'].iloc[-1] - r['open'].iloc[0]
        if bull >= 3 and net > 0: bias_s[i] = "BULLISH"
        elif bear >= 3 and net < 0: bias_s[i] = "BEARISH"
        else: bias_s[i] = "NEUTRAL"
        if bull >= 2 and net > 0: bias_r[i] = "BULLISH"
        elif bear >= 2 and net < 0: bias_r[i] = "BEARISH"
        else: bias_r[i] = "NEUTRAL"
    return bias_s, bias_r, h1_sup, h1_res

def calculate_score(row, df, i, support, resistance, direction):
    body = row['body']
    if body <= SMALL_VAL: return -5, 0.0
    score = 0.0; cr = row['high'] - row['low']; atr = row['atr'] if row['atr'] > 0 else eps
    ema_dist = (row['ema50'] - row['ema200']) / atr
    if direction == "BUY": score += max(-2.0, min(2.0, ema_dist * 4))
    else: score += max(-2.0, min(2.0, -ema_dist * 4))
    disp = row['high'] - resistance if direction == "BUY" else support - row['low']
    gap_ratio = disp / cr if cr > 0 else 0; gap_ratio = max(0.0, gap_ratio)
    if gap_ratio < 0.2: return -5, gap_ratio
    score += min(2.0, gap_ratio * 2)
    avg_body = df['body'].rolling(20).mean().iloc[i]
    br = body / avg_body if avg_body > 0 else 1.0; score += min(1.5, br * 0.5)
    avg_vol = df['tick_volume'].rolling(20).mean().iloc[i]
    vr = row['tick_volume'] / avg_vol if avg_vol > 0 else 1.0; score += min(1.0, vr * 0.33)
    uw = row['high'] - max(row['close'], row['open'])
    lw = min(row['close'], row['open']) - row['low']
    ur = uw / body if body > 0 else 0; lr = lw / body if body > 0 else 0
    if direction == "BUY" and ur > 0.5: score -= (ur - 0.5) * 3
    elif direction == "SELL" and lr > 0.5: score -= (lr - 0.5) * 3
    btr = body / cr if cr > 0 else 0
    if btr > 0.6: score += 1.0
    b_dist = row['close'] - resistance if direction == "BUY" else support - row['close']
    score += min(1.0, max(0.0, b_dist / atr * 5))
    return score, gap_ratio

def score_to_prob(s): return max(0.0, min(1.0, (s + 2) / 9))

# --- Reversal detection ---
def detect_sweep(m15, i, level, direction, atr_buf=0.1):
    row = m15.iloc[i]
    buffer = row['atr'] * atr_buf
    if direction == "BUY": return row['low'] < (level - buffer)
    else: return row['high'] > (level + buffer)

def detect_reversal(m15, i, support, resistance):
    row = m15.iloc[i]; close = row['close']
    sweep_buy = detect_sweep(m15, i, support, "BUY")
    if not sweep_buy and i >= 1: sweep_buy = detect_sweep(m15, i-1, support, "BUY")
    if sweep_buy and close > support: return "BUY", support
    sweep_sell = detect_sweep(m15, i, resistance, "SELL")
    if not sweep_sell and i >= 1: sweep_sell = detect_sweep(m15, i-1, resistance, "SELL")
    if sweep_sell and close < resistance: return "SELL", resistance
    return None, None

# --- Backtest ---
param_grid = list(itertools.product([True, False], [0], [0.5]))

for sym in ["XAUUSD", "BTCUSD"]:
    cfg = CONFIG[sym]
    risk = cfg['risk_usd'] * cfg['usd1_to_price']
    rwd = cfg['risk_to_reward'] * risk

    print(f"  Loading {sym}...")
    m15 = pd.read_csv(cfg['csv'])
    m15['timestamp'] = pd.to_datetime(m15['timestamp'], utc=True)
    m15 = m15.sort_values('timestamp').reset_index(drop=True)

    # Filter last 6 months only
    cutoff = m15['timestamp'].max() - pd.Timedelta(days=183)
    m15 = m15[m15['timestamp'] >= cutoff].reset_index(drop=True)
    print(f"  Last 6 months: {m15['timestamp'].min()} to {m15['timestamp'].max()} ({len(m15)} bars)")

    h1 = resample_h1(m15)
    m15 = add_indicators(m15) if not m15.empty else m15
    n = len(m15)

    print(f"  Pre-computing H1 states ({len(h1)} H1 bars)...")
    h1_bias_s, h1_bias_r, h1_sup, h1_res = precompute_h1_state(h1)
    h1_idx_map = np.searchsorted(h1['time'].values, m15['timestamp'].values, side='right') - 1

    print(f"  Pre-computing M15 S/R...")
    m15_sup = np.full(n, np.nan); m15_res = np.full(n, np.nan)
    for i in range(n):
        sub = m15.iloc[max(0,i-LOOKBACK):i+1]
        ph, pl = fast_fractals(sub['high'].values, sub['low'].values, FRACTAL_STRENGTH)
        r_idx = np.where(ph)[0]; s_idx = np.where(pl)[0]
        if len(r_idx) > 0: m15_res[i] = sub['high'].iloc[r_idx[-1]]
        if len(s_idx) > 0: m15_sup[i] = sub['low'].iloc[s_idx[-1]]
        if np.isnan(m15_res[i]): m15_res[i] = sub['high'].max()
        if np.isnan(m15_sup[i]): m15_sup[i] = sub['low'].min()

    # Track breakout vs reversal separately, plus a raw baseline (no filters)
    res_total = {p: {"t":0,"w":0} for p in param_grid}
    res_breakout = {p: {"t":0,"w":0} for p in param_grid}
    res_reversal = {p: {"t":0,"w":0} for p in param_grid}
    res_raw = {"t":0, "w":0}  # raw breakout, no bias/score/room/reversal
    skipped_stats = {"bias": 0, "score": 0, "gap": 0, "zone": 0}

    print(f"  Scanning {n} bars for breakouts + reversals...")
    for i in range(100, n - 1):
        row = m15.iloc[i]
        if np.isnan(row['atr']) or row['atr'] <= 0: continue
        support = m15_sup[i]; resistance = m15_res[i]
        if np.isnan(support) or np.isnan(resistance) or resistance <= 0 or support <= 0: continue

        hi = h1_idx_map[i]
        if hi < 4 or hi >= len(h1_bias_s): continue

        o, c = row['open'], row['close']; buf = row['atr'] * 0.1
        bu = c > (resistance + buf) and o < resistance
        bd = c < (support - buf) and o > support

        direction = None; signal_type = None; zone_center = None

        # Check breakout first
        if bu: direction = "BUY"; signal_type = "breakout"; zone_center = resistance
        elif bd: direction = "SELL"; signal_type = "breakout"; zone_center = support

        # If no breakout, check reversal
        if direction is None:
            rev_dir, rev_zone = detect_reversal(m15, i, support, resistance)
            if rev_dir is not None:
                direction = rev_dir
                signal_type = "reversal"
                zone_center = rev_zone

        if direction is None: continue

        h1s = h1_bias_s[hi]; h1r = h1_bias_r[hi]

        # For breakout: need H1 bias aligned (relaxed)
        if signal_type == "breakout":
            if direction == "BUY" and h1r == "BEARISH": skipped_stats["bias"]+=1; continue
            if direction == "SELL" and h1r == "BULLISH": skipped_stats["bias"]+=1; continue

        # For reversal: need strict H1 alignment (per user spec)
        if signal_type == "reversal":
            if direction == "BUY" and h1s != "BULLISH": skipped_stats["bias"]+=1; continue
            if direction == "SELL" and h1s != "BEARISH": skipped_stats["bias"]+=1; continue

        # Score (use zone_center as the breached level)
        score, gap_r = calculate_score(row, m15, i, zone_center, zone_center, direction)
        if score <= -5: skipped_stats["gap"]+=1; continue
        pv = score_to_prob(score)
        if pv < 0.5: skipped_stats["score"]+=1; continue

        # Zone entry
        zl = zone_center - row['atr'] * 0.25; zh = zone_center + row['atr'] * 0.25
        entry_j = None
        for j in range(i + 1, min(i + MAX_WAIT + 1, n)):
            mid = (m15.iloc[j]['high'] + m15.iloc[j]['low']) / 2
            if zl <= mid <= zh: entry_j = j; break
        if entry_j is None: skipped_stats["zone"]+=1; continue

        ep_vals = m15.iloc[entry_j]; ep = (ep_vals['high'] + ep_vals['low']) / 2
        if direction == "BUY": sl = ep - risk; tp = ep + rwd
        else: sl = ep + risk; tp = ep - rwd

        if direction == "BUY":
            ht = m15['high'].iloc[entry_j+1:].ge(tp).any()
            hs = m15['low'].iloc[entry_j+1:].le(sl).any()
        else:
            ht = m15['low'].iloc[entry_j+1:].le(tp).any()
            hs = m15['high'].iloc[entry_j+1:].ge(sl).any()

        for (h1f, rr, pt), r in [((True, 0, 0.5), res_total[(True, 0, 0.5)]),
                                   ((False, 0, 0.5), res_total[(False, 0, 0.5)])]:
            # Bias filter for combo
            bv = h1s if h1f else h1r
            if signal_type == "breakout":
                if direction == "BUY":
                    if h1f and bv != "BULLISH": continue
                    if not h1f and bv == "BEARISH": continue
                else:
                    if h1f and bv != "BEARISH": continue
                    if not h1f and bv == "BULLISH": continue
            else:  # reversal
                if direction == "BUY" and bv != "BULLISH": continue
                if direction == "SELL" and bv != "BEARISH": continue
            if ht: r["t"]+=1; r["w"]+=1
            elif hs: r["t"]+=1
            # also track by type
            bt = res_breakout[(h1f, rr, pt)] if signal_type == "breakout" else res_reversal[(h1f, rr, pt)]
            if ht: bt["t"]+=1; bt["w"]+=1
            elif hs: bt["t"]+=1

        # Raw baseline: breakout only, no bias/score/room/reversal
        if signal_type == "breakout":
            if ht: res_raw["t"]+=1; res_raw["w"]+=1
            elif hs: res_raw["t"]+=1

    # Print combined results
    print(f"\n{'='*70}\n  {sym} — Combined Breakout + Reversal\n{'='*70}")
    print(f"  Skipped: bias={skipped_stats['bias']} gap={skipped_stats['gap']} score={skipped_stats['score']} zone={skipped_stats['zone']}")
    print(f"\n{'Bias':>5s} {'SigType':>9s} {'Trades':>7s} {'WR':>7s} {'EV':>9s} {'PnL':>10s}")
    print("-" * 55)
    for h1f, lbl in [(True, "S3"), (False, "R2")]:
        for sig_type, src in [("All", res_total), ("BO", res_breakout), ("REV", res_reversal)]:
            r = src[(h1f, 0, 0.5)]
            if r["t"] == 0: continue
            wr = r["w"] / r["t"]; ev = wr * rwd - (1-wr) * risk
            pnl = r["w"] * rwd - (r["t"] - r["w"]) * risk
            print(f"{lbl:>5s} {sig_type:>9s} {r['t']:>7d} {wr:>7.1%} {ev:>+9.2f}R {pnl:>+10,.0f}R")
    # Raw baseline
    if res_raw["t"] > 0:
        w = res_raw["w"]; wr = w / res_raw["t"]
        ev = wr * rwd - (1-wr) * risk
        pnl = w * rwd - (res_raw["t"] - w) * risk
        print(f"  Raw breakout (no filters): {res_raw['t']:>7d}  {wr:>7.1%}  {ev:>+9.2f}R  {pnl:>+10,.0f}R")
