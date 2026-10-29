"""
Multi-timeframe trend detection using EMA/RSI/MACD/ADX/structure scoring.

Implements DetectMarketTrend and AnalyseTimeframe from the algo specs.

Usage:
    python trend.py XAUUSD
    python trend.py BTCUSD
"""
import MetaTrader5 as mt5
import pandas as pd
import numpy as np
import sys
from datetime import datetime

SYMBOL = sys.argv[1].upper() if len(sys.argv) > 1 else "XAUUSD"
USE_D1 = "--d1" in sys.argv

TF_MAP = {
    "M15": mt5.TIMEFRAME_M15,
    "H1":  mt5.TIMEFRAME_H1,
    "H4":  mt5.TIMEFRAME_H4,
    "D1":  mt5.TIMEFRAME_D1,
}
TF_BARS = {"D1": 200, "H4": 200, "H1": 200, "M15": 200}

if USE_D1:
    TF_ORDER = ["D1", "H4", "H1", "M15"]
    TF_WEIGHTS = {"D1": 0.40, "H4": 0.30, "H1": 0.20, "M15": 0.10}
else:
    TF_ORDER = ["H4", "H1", "M15"]
    TF_WEIGHTS = {"H4": 0.50, "H1": 0.35, "M15": 0.15}


def ema(arr, period):
    n = len(arr)
    out = np.zeros(n)
    out[0] = arr[0]
    k = 2 / (period + 1)
    for i in range(1, n):
        out[i] = arr[i] * k + out[i-1] * (1 - k)
    return out


def rsi(arr, period=14):
    n = len(arr)
    gain = np.zeros(n)
    loss = np.zeros(n)
    for i in range(1, n):
        d = arr[i] - arr[i-1]
        gain[i] = d if d > 0 else 0
        loss[i] = -d if d < 0 else 0
    avg_g = np.zeros(n); avg_l = np.zeros(n)
    avg_g[period] = gain[1:period+1].mean()
    avg_l[period] = loss[1:period+1].mean()
    for i in range(period+1, n):
        avg_g[i] = (avg_g[i-1] * (period-1) + gain[i]) / period
        avg_l[i] = (avg_l[i-1] * (period-1) + loss[i]) / period
    rs = np.divide(avg_g, avg_l, out=np.ones(n)*50, where=avg_l!=0)
    rsi_vals = 100 - 100 / (1 + rs)
    rsi_vals[:period] = 50
    return rsi_vals


def macd(arr, fast=12, slow=26, signal=9):
    e_fast = ema(arr, fast)
    e_slow = ema(arr, slow)
    macd_line = e_fast - e_slow
    sig_line = ema(macd_line, signal)
    return macd_line, sig_line


def adx(h, l, c, period=14):
    n = len(c)
    plus_di = np.zeros(n); minus_di = np.zeros(n); adx_vals = np.zeros(n)
    tr = np.zeros(n); up_move = np.zeros(n); down_move = np.zeros(n)
    for i in range(1, n):
        tr[i] = max(h[i]-l[i], abs(h[i]-c[i-1]), abs(l[i]-c[i-1]))
        up_move[i] = h[i] - h[i-1]
        down_move[i] = l[i-1] - l[i]
    atr = ema(tr, period)
    smoothed_up = ema(np.maximum(up_move, 0), period)
    smoothed_dn = ema(np.maximum(down_move, 0), period)
    for i in range(1, n):
        if atr[i] > 0:
            plus_di[i] = 100 * smoothed_up[i] / atr[i]
            minus_di[i] = 100 * smoothed_dn[i] / atr[i]
    dx = np.zeros(n)
    for i in range(1, n):
        s = plus_di[i] + minus_di[i]
        if s > 0:
            dx[i] = 100 * abs(plus_di[i] - minus_di[i]) / s
    adx_vals = ema(dx, period)
    return adx_vals, plus_di, minus_di


def bollinger(c, period=20, std=2):
    mid = np.convolve(c, np.ones(period)/period, mode='same')
    mid[:period-1] = c[:period-1]
    rolling_std = np.zeros(len(c))
    for i in range(period-1, len(c)):
        rolling_std[i] = c[i-period+1:i+1].std()
    upper = mid + std * rolling_std
    lower = mid - std * rolling_std
    bw = (upper - lower) / mid * 100
    return mid, upper, lower, bw


def swing_structure(h, l, c=None, strength=2):
    """Detect HH, HL, LH, LL. Returns (hh, hl, lh, ll)."""
    n = len(h)
    sh = np.zeros(n, bool); sl = np.zeros(n, bool)
    for i in range(strength, n-strength):
        sh[i] = all(h[i] >= h[i-k] and h[i] > h[i+k] for k in range(1, strength+1))
        sl[i] = all(l[i] <= l[i-k] and l[i] < l[i+k] for k in range(1, strength+1))
    hi = np.where(sh)[0]; li = np.where(sl)[0]
    hh = hl = lh = ll = False
    if len(hi) >= 2:
        if h[hi[-1]] > h[hi[-2]]: hh = True
        elif h[hi[-1]] < h[hi[-2]]: lh = True
    if len(li) >= 2:
        if l[li[-1]] > l[li[-2]]: hl = True
        elif l[li[-1]] < l[li[-2]]: ll = True
    return hh, hl, lh, ll


def analyse_timeframe(o, h, l, c):
    """AnalyseTimeframe: score the trend for one timeframe."""
    n = len(c)

    # Indicators
    ema20 = ema(c, 20)
    ema50 = ema(c, 50)
    ema200 = ema(c, 200)
    rsi_vals = rsi(c)
    macd_line, sig_line = macd(c)
    adx_vals, plus_di, minus_di = adx(h, l, c)
    _, _, _, bb_width = bollinger(c)

    # EMA slope (over last 5 bars)
    slope = (ema20[-1] - ema20[-6]) / (ema20[-6] + 1e-9) * 100 if n >= 6 else 0
    slope_threshold = 0.1

    # Market structure
    hh, hl, lh, ll = swing_structure(h, l)

    # Scores
    bull = bear = side = 0

    # EMA Alignment
    if ema20[-1] > ema50[-1] > ema200[-1]:
        bull += 2
    elif ema20[-1] < ema50[-1] < ema200[-1]:
        bear += 2
    else:
        side += 1

    # EMA Slope
    if slope > slope_threshold:
        bull += 2
    elif slope < -slope_threshold:
        bear += 2
    else:
        side += 1

    # ADX
    if adx_vals[-1] > 25:
        if plus_di[-1] > minus_di[-1]:
            bull += 2
        else:
            bear += 2
    else:
        side += 2

    # RSI
    if rsi_vals[-1] > 60:
        bull += 1
    elif rsi_vals[-1] < 40:
        bear += 1
    else:
        side += 1

    # MACD
    if macd_line[-1] > sig_line[-1]:
        bull += 1
    else:
        bear += 1

    # Market Structure
    if hh and hl:
        bull += 3
    elif lh and ll:
        bear += 3
    else:
        side += 2

    # Bollinger Width (consolidation)
    narrow_threshold = bb_width[-50:].mean() * 0.3 if n >= 50 else 0
    if bb_width[-1] < narrow_threshold:
        side += 2

    # Final trend
    if bull > bear and bull > side:
        trend = "BULLISH"
    elif bear > bull and bear > side:
        trend = "BEARISH"
    else:
        trend = "SIDEWAYS"

    total = bull + bear + side
    confidence = max(bull, bear, side) / total if total > 0 else 0

    return {
        "trend": trend,
        "confidence": confidence,
        "bull": bull, "bear": bear, "side": side,
        "ema20": ema20[-1], "ema50": ema50[-1], "ema200": ema200[-1],
        "rsi": rsi_vals[-1], "adx": adx_vals[-1],
        "hh": hh, "hl": hl, "lh": lh, "ll": ll,
        "macd_hist": float(macd_line[-1] - sig_line[-1]),
        "macd_hist_prev": float(macd_line[-2] - sig_line[-2]) if len(macd_line) > 1 else 0,
    }


def per_bar_structure(h, l, strength=2):
    """Return per-bar HH/HL/LH/LL flags: is the latest swing higher/lower than the prior.
    Flags are point-in-time at each index i (only uses data up to i)."""
    n = len(h)
    sh = np.zeros(n, bool); sl = np.zeros(n, bool)
    for i in range(strength, n - strength):
        sh[i] = all(h[i] >= h[i-k] and h[i] > h[i+k] for k in range(1, strength+1))
        sl[i] = all(l[i] <= l[i-k] and l[i] < l[i+k] for k in range(1, strength+1))
    # For each index, track the last two confirmed swing highs / lows up to that index
    hi = np.where(sh)[0]; li = np.where(sl)[0]
    hh = np.zeros(n, bool); hl = np.zeros(n, bool); lh = np.zeros(n, bool); ll = np.zeros(n, bool)
    last2 = None; last1 = None
    hi_pos = 0
    for i in range(n):
        if hi_pos < len(hi) and hi[hi_pos] == i:
            last2 = last1; last1 = i
            hi_pos += 1
        if last2 is not None and last1 is not None:
            if h[last1] > h[last2]: hh[i] = True
            elif h[last1] < h[last2]: lh[i] = True
    last2 = None; last1 = None
    li_pos = 0
    for i in range(n):
        if li_pos < len(li) and li[li_pos] == i:
            last2 = last1; last1 = i
            li_pos += 1
        if last2 is not None and last1 is not None:
            if l[last1] > l[last2]: hl[i] = True
            elif l[last1] < l[last2]: ll[i] = True
    return hh, hl, lh, ll


def analyse_tf_arrays(o, h, l, c, strength=2):
    """Compute full indicator arrays for a timeframe. Returns a dict of arrays."""
    ema20 = ema(c, 20)
    ema50 = ema(c, 50)
    ema200 = ema(c, 200)
    rsi_vals = rsi(c)
    macd_line, sig_line = macd(c)
    adx_vals, plus_di, minus_di = adx(h, l, c)
    _, _, _, bb_width = bollinger(c)
    hh, hl, lh, ll = per_bar_structure(h, l, strength)
    return {
        "ema20": ema20, "ema50": ema50, "ema200": ema200,
        "rsi": rsi_vals, "adx": adx_vals, "plus_di": plus_di, "minus_di": minus_di,
        "macd_line": macd_line, "sig_line": sig_line, "bb_width": bb_width,
        "hh": hh, "hl": hl, "lh": lh, "ll": ll,
    }


def snapshot_at(arrs, i):
    """AnalyseTimeframe-style score dict evaluated at a single bar index i (point-in-time)."""
    n = len(arrs["ema20"])
    ema20 = arrs["ema20"]; ema50 = arrs["ema50"]; ema200 = arrs["ema200"]
    rsi_vals = arrs["rsi"]; adx_vals = arrs["adx"]
    plus_di = arrs["plus_di"]; minus_di = arrs["minus_di"]
    macd_line = arrs["macd_line"]; sig_line = arrs["sig_line"]
    bb_width = arrs["bb_width"]
    hh = arrs["hh"]; hl = arrs["hl"]; lh = arrs["lh"]; ll = arrs["ll"]

    slope = (ema20[i] - ema20[max(0, i-5)]) / (ema20[max(0, i-5)] + 1e-9) * 100
    slope_threshold = 0.1

    bull = bear = side = 0
    if ema20[i] > ema50[i] > ema200[i]:
        bull += 2
    elif ema20[i] < ema50[i] < ema200[i]:
        bear += 2
    else:
        side += 1

    if slope > slope_threshold:
        bull += 2
    elif slope < -slope_threshold:
        bear += 2
    else:
        side += 1

    if adx_vals[i] > 25:
        if plus_di[i] > minus_di[i]:
            bull += 2
        else:
            bear += 2
    else:
        side += 2

    if rsi_vals[i] > 60:
        bull += 1
    elif rsi_vals[i] < 40:
        bear += 1
    else:
        side += 1

    if macd_line[i] > sig_line[i]:
        bull += 1
    else:
        bear += 1

    if hh[i] and hl[i]:
        bull += 3
    elif lh[i] and ll[i]:
        bear += 3
    else:
        side += 2

    narrow_threshold = bb_width[max(0, i-50):i+1].mean() * 0.3 if i >= 50 else 0
    if bb_width[i] < narrow_threshold:
        side += 2

    if bull > bear and bull > side:
        trend = "BULLISH"
    elif bear > bull and bear > side:
        trend = "BEARISH"
    else:
        trend = "SIDEWAYS"

    total = bull + bear + side
    confidence = max(bull, bear, side) / total if total > 0 else 0

    return {
        "trend": trend,
        "confidence": confidence,
        "bull": bull, "bear": bear, "side": side,
        "ema20": ema20[i], "ema50": ema50[i], "ema200": ema200[i],
        "rsi": rsi_vals[i], "adx": adx_vals[i],
        "hh": hh[i], "hl": hl[i], "lh": lh[i], "ll": ll[i],
        "macd_hist": float(macd_line[i] - sig_line[i]),
        "macd_hist_prev": float(macd_line[max(0, i-1)] - sig_line[max(0, i-1)]),
    }


def compute_reversal_probability(results, current_tf="H1"):
    """Score 0-100% probability of a near-term trend reversal across all TFs."""
    score = 0.0; total_w = 0.0
    weights = {"D1": 0.30, "H4": 0.30, "H1": 0.25, "M15": 0.15}
    for tf_name, w in weights.items():
        if tf_name not in results:
            continue
        r = results[tf_name]
        s = 0.0; n = 0
        trend = r.get("trend", "SIDEWAYS")
        rsi = r.get("rsi", 50)
        adx = r.get("adx", 20)
        hh, hl, lh, ll = r.get("hh", False), r.get("hl", False), r.get("lh", False), r.get("ll", False)
        mh = r.get("macd_hist", 0)
        mh_prev = r.get("macd_hist_prev", 0)

        # RSI extreme (0-30%)
        if trend == "BULLISH" and rsi > 70:
            s += min((rsi - 70) / 20 * 30, 30)
            n += 30
        elif trend == "BEARISH" and rsi < 30:
            s += min((30 - rsi) / 20 * 30, 30)
            n += 30
        else:
            n += 30

        # ADX weakening (0-20%)
        if adx > 20 and adx < 30:
            s += (adx - 20) / 10 * 20
            n += 20
        else:
            n += 20

        # Structure flip risk (0-30%)
        if (trend == "BULLISH" and lh and not hh) or (trend == "BEARISH" and hl and not ll):
            s += 30
            n += 30
        elif (trend == "BULLISH" and lh) or (trend == "BEARISH" and hl):
            s += 15
            n += 30
        else:
            n += 30

        # MACD momentum fade (0-20%)
        if (mh > 0 and mh < mh_prev) or (mh < 0 and mh > mh_prev):
            s += 20
            n += 20
        else:
            n += 20

        tf_pct = s / n * 100 if n > 0 else 50
        score += w * tf_pct
        total_w += w

    return round(score / total_w, 1) if total_w > 0 else 50.0


def detect_market_trend(offset_hours=0):
    """DetectMarketTrend: load all TFs with optional hour offset, analyse, weighted vote."""
    _TF_DIV = {"M15": 0.25, "H1": 1, "H4": 4, "D1": 24}
    results = {}
    for tf_name in TF_ORDER:
        tf = TF_MAP[tf_name]
        bars = TF_BARS[tf_name]
        start_pos = 1 + int(offset_hours / _TF_DIV[tf_name]) if offset_hours > 0 else 1
        rates = mt5.copy_rates_from_pos(SYMBOL, tf, start_pos, bars)
        if rates is None or len(rates) < 200:
            print(f"  WARNING: {tf_name} insufficient data ({len(rates) if rates is not None else 0} bars)")
            results[tf_name] = {"trend": "SIDEWAYS", "confidence": 0, "bull": 0, "bear": 0, "side": 0}
            continue
        df = pd.DataFrame(rates)
        o = df['open'].values.astype(float)
        h = df['high'].values.astype(float)
        l = df['low'].values.astype(float)
        c = df['close'].values.astype(float)
        results[tf_name] = analyse_timeframe(o, h, l, c)

    # Weighted vote
    bull_score = bear_score = side_score = 0.0
    for tf_name in TF_ORDER:
        r = results[tf_name]
        w = TF_WEIGHTS[tf_name]
        if r["trend"] == "BULLISH":
            bull_score += w * r["confidence"]
        elif r["trend"] == "BEARISH":
            bear_score += w * r["confidence"]
        else:
            side_score += w * r["confidence"]

    scores = {"BULLISH": bull_score, "BEARISH": bear_score, "SIDEWAYS": side_score}
    total = bull_score + bear_score + side_score
    bull_pct = bull_score / total * 100 if total > 0 else 0
    bear_pct = bear_score / total * 100 if total > 0 else 0
    side_pct = side_score / total * 100 if total > 0 else 0
    final_trend = max(scores, key=scores.get)
    if max(bull_pct, bear_pct) <= 50:
        final_trend = "SIDEWAYS"

    # Pullback detection (only when D1 is included)
    pullback = False
    tb = {k: results[k]["trend"] for k in TF_ORDER}
    if USE_D1:
        if tb["D1"] == "BULLISH" and tb["H4"] == "BULLISH" and tb["H1"] == "BULLISH" and tb["M15"] == "BEARISH":
            pullback = True
        if tb["D1"] == "BEARISH" and tb["H4"] == "BEARISH" and tb["H1"] == "BEARISH" and tb["M15"] == "BULLISH":
            pullback = True

    # Reversal detection (only when D1 is included)
    reversal = False
    if USE_D1 and tb["D1"] != tb["H4"] and tb["H4"] == tb["H1"] and tb["H1"] == tb["M15"]:
        reversal = True

    # Print
    print(f"\n  {'='*60}")
    print(f"  TREND DETECTION — {SYMBOL}  ({'/'.join(TF_ORDER)})")
    print(f"  {'='*60}")
    print(f"  {'TF':>5s}  {'TREND':>9s}  {'CONF':>6s}  {'EMA20':>8s}  {'EMA50':>8s}  {'RSI':>5s}  {'ADX':>5s}  {'HH':>3s}  {'HL':>3s}  {'LH':>3s}  {'LL':>3s}")
    print(f"  {'-'*65}")
    for tf_name in TF_ORDER:
        r = results[tf_name]
        hh_s = 'V' if r.get('hh') else '.'; hl_s = 'V' if r.get('hl') else '.'
        lh_s = 'V' if r.get('lh') else '.'; ll_s = 'V' if r.get('ll') else '.'
        e20 = f"{r.get('ema20', 0):.0f}"
        e50 = f"{r.get('ema50', 0):.0f}"
        print(f"  {tf_name:>5s}  {r['trend']:>9s}  {r['confidence']:.2%}  "
              f"{e20:>8s}  {e50:>8s}  {r.get('rsi', 0):.0f}  {r.get('adx', 0):.0f}  "
              f"{hh_s:>3s}  {hl_s:>3s}  {lh_s:>3s}  {ll_s:>3s}")
    print(f"  {'-'*65}")
    print(f"  Scores:    BULL={bull_pct:.0f}%  BEAR={bear_pct:.0f}%  SIDE={side_pct:.0f}%")
    print(f"  Final:     {final_trend}")
    print(f"  Pullback:  {pullback}")
    print(f"  Reversal:  {reversal}")
    rev_score = compute_reversal_probability(results)
    print(f"  RVS:       {rev_score:.0f}%")
    print(f"  {'='*60}")
    return final_trend, results, rev_score


def trend_from_df(df, resample_tf_fn, current_tf=None, use_d1=False):
    """Trend scoring from resampled df (backtest mode, no MT5 needed).
    Returns (is_accumulating, dominant_bias, tf_bias)."""
    m15 = resample_tf_fn(df, '15min')
    h1  = resample_tf_fn(df, '1h')
    h4  = resample_tf_fn(df, '4h')
    d1  = resample_tf_fn(df, '1D') if use_d1 else None
    weights = {"D1": 0.40, "H4": 0.30, "H1": 0.20, "M15": 0.10} if use_d1 else {"H4": 0.50, "H1": 0.35, "M15": 0.15}
    tfs = [("D1", d1.tail(200)), ("H4", h4.tail(200)), ("H1", h1.tail(200)), ("M15", m15.tail(200))] if use_d1 else \
           [("M15", m15.tail(200)), ("H1", h1.tail(200)), ("H4", h4.tail(200))]
    bull = bear = side = 0.0
    tf_results = {}
    for tf_name, tf_df in tfs:
        if len(tf_df) < 20:
            return (True, None, None)
        o = tf_df['open'].values.astype(float); h = tf_df['high'].values.astype(float)
        l = tf_df['low'].values.astype(float); c = tf_df['close'].values.astype(float)
        try:
            r = analyse_timeframe(o, h, l, c)
        except:
            return (True, None, None)
        tf_results[tf_name] = r
        w = weights[tf_name]
        if r["trend"] == "BULLISH": bull += w * r["confidence"]
        elif r["trend"] == "BEARISH": bear += w * r["confidence"]
        else: side += w * r["confidence"]
    tf_bias = None
    if current_tf and current_tf in tf_results:
        tr = tf_results[current_tf]["trend"]
        tf_bias = tr if tr != "SIDEWAYS" else None
    total = bull + bear + side
    bp = bull / total * 100 if total > 0 else 0; bep = bear / total * 100 if total > 0 else 0
    rev_score = compute_reversal_probability(tf_results, current_tf or "H1")
    if max(bp, bep) <= 50:
        return (True, None, tf_bias, rev_score)
    return (False, "BULLISH" if bp > bep else "BEARISH", tf_bias, rev_score)


def get_trend(symbol, current_tf=None, use_d1=False, offset_hours=0):
    """Returns (is_accumulating, dominant_bias, tf_bias, reversal_score).
    offset_hours: analyse data from N hours ago (0 = latest)."""
    import os
    global SYMBOL, USE_D1, TF_ORDER, TF_WEIGHTS
    SYMBOL = symbol
    USE_D1 = use_d1
    TF_ORDER = ["D1", "H4", "H1", "M15"] if use_d1 else ["H4", "H1", "M15"]
    TF_WEIGHTS = {"D1": 0.40, "H4": 0.30, "H1": 0.20, "M15": 0.10} if use_d1 else {"H4": 0.50, "H1": 0.35, "M15": 0.15}
    mt5.symbol_select(SYMBOL, True)
    old_out = sys.stdout
    sys.stdout = open(os.devnull, 'w')
    final_trend, results, rev_score = detect_market_trend(offset_hours=offset_hours)
    sys.stdout = old_out
    tf_bias = None
    if current_tf and current_tf in results:
        r = results[current_tf]["trend"]
        tf_bias = r if r != "SIDEWAYS" else None
    if final_trend == "SIDEWAYS":
        return (True, None, tf_bias, rev_score)
    return (False, final_trend, tf_bias, rev_score)


if __name__ == "__main__":
    import argparse
    _p = argparse.ArgumentParser()
    _p.add_argument("symbol", nargs="?", default="XAUUSD")
    _p.add_argument("--d1", action="store_true")
    _p.add_argument("--offset", type=int, default=0, help="Hours to offset (0 = latest)")
    _a = _p.parse_args()
    SYMBOL = _a.symbol.upper()
    USE_D1 = _a.d1
    if not mt5.initialize():
        print("MT5 init failed"); sys.exit(1)
    ac, pw, sv = int(os.environ["MT5_ACCOUNT"]), os.environ["MT5_PASSWORD"], "PepperstoneBS-MT5-Live01"
    if not mt5.login(ac, pw, sv):
        print("Login failed"); mt5.shutdown(); sys.exit(1)
    mt5.symbol_select(SYMBOL, True)
    detect_market_trend(offset_hours=_a.offset)
    mt5.shutdown()
