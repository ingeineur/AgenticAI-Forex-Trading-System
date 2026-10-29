"""
SMC Strategy: CHoCH/BOS → FVG/OB + EQH/EQL → Premium/Discount
Based on LuxAlgo Smart Money Concepts methodology.

Rules:
  1. SELL at premium zone only when EQH (Equal Highs) is detected
  2. SELL at OB or FVG only after CHoCH or BOS, price retraces to zone
  3. BUY at discount zone only when EQL (Equal Lows) is detected
  4. BUY at OB or FVG only after CHoCH or BOS, price retraces to zone

Usage:
  python smc_strategy.py XAUUSD --backtest
  python smc_strategy.py XAUUSD --backtest --tf h1
  python smc_strategy.py XAUUSD --train-ml             # Train ML model
  python smc_strategy.py XAUUSD --backtest --ml-threshold 0.5
  python smc_strategy.py XAUUSD --watch --trade
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

eps = 1e-9
BULLISH = 1
BEARISH = -1
NEUTRAL = 0
SIG_STRUCTURE = "STRUCTURE"  # CHoCH/BOS → FVG/OB
SIG_LIQUIDITY = "LIQUIDITY"  # EQH/EQL → Premium/Discount

parser = argparse.ArgumentParser(description="SMC Strategy: CHoCH/BOS + EQH/EQL + Zone Entry")
parser.add_argument("symbol", default="XAUUSD", nargs="?", help="Symbol (XAUUSD, BTCUSD, etc)")
parser.add_argument("--bars", type=int, default=5000, help="Bars to load")
parser.add_argument("--risk", type=float, default=10, help="Risk per trade in USD")
parser.add_argument("--max-orders", type=int, default=2, help="Max concurrent positions")
parser.add_argument("--lookback", type=int, default=50, help="Lookback bars for swing detection")
parser.add_argument("--max-wait", type=int, default=10, help="Max bars to wait for zone entry")
parser.add_argument("--watch", action="store_true", help="Live monitoring mode")
parser.add_argument("--trade", action="store_true", help="Auto-execute with --watch")
parser.add_argument("--backtest", action="store_true", help="CSV backtest mode")
parser.add_argument("--session", choices=["london", "ny", "any"], default="any", help="Session filter")
parser.add_argument("--volume-filter", action="store_true", help="Enable volume filter (1.2x avg)")
parser.add_argument("--tp-mult", type=float, default=0.5, help="TP as multiple of 1R")
parser.add_argument("--interval", type=int, default=20, help="Poll interval in seconds")
parser.add_argument("--fractal-strength", type=int, default=2, help="Fractal strength for swing detection")
parser.add_argument("--entry-mode", choices=["fvg", "ob", "pd", "all", "structure", "liquidity"], default="all", help="Zone entry type filter")
parser.add_argument("--tf", choices=["m15", "h1"], default="m15", help="Timeframe for backtest/live")
parser.add_argument("--eqh-threshold", type=float, default=0.1, help="EQH/EQL threshold as multiple of ATR")
parser.add_argument("--train-ml", action="store_true", help="Train Random Forest model from backtest trades")
parser.add_argument("--ml-path", type=str, default=None, help="Path to load/save ML model")
parser.add_argument("--ml-threshold", type=float, default=0.0, help="Min ML probability to take trade (0.0 = disabled)")
parser.add_argument("--supertrend", action="store_true", help="Enable SuperTrend confluence filter")
parser.add_argument("--st-atr", type=int, default=10, help="SuperTrend ATR period")
parser.add_argument("--st-mult", type=float, default=3.0, help="SuperTrend ATR multiplier")
parser.add_argument("--pvt-filter", action="store_true", help="Enable PVT divergence filter")
parser.add_argument("--sl-atr", type=float, default=0.0, help="SL as ATR multiple (0 = use fixed USD risk)")
parser.add_argument("--tp-atr", type=float, default=0.0, help="TP as ATR multiple (0 = use --tp-mult)")
parser.add_argument("--orderflow", action="store_true", help="Enable orderflow delta filter")
parser.add_argument("--of-mode", choices=["ohlc", "tickrule"], default="ohlc", help="Orderflow method: ohlc=OHLC approx, tickrule=bid tick-rule (live, experimental)")
parser.add_argument("--of-lookback", type=int, default=-1, help="Bars to average delta for orderflow filter (default: 5 H1 / 10 M15)")
parser.add_argument("--swing-tf", choices=["tf", "h4"], default="tf", help="Timeframe for swing detection: tf=current, h4=resampled H4")
parser.add_argument("--require-sweep", action="store_true", help="Only take SMC entries when price swept an H4 swing level first")
parser.add_argument("--market-only", action="store_true", help="Skip limit orders, only enter via market when price is near zone")
parser.add_argument("--max-signal-age", type=int, default=48, help="Max bars since structure break (48 = H1 2 days / M15 12 hours)")
parser.add_argument("--show-zones", action="store_true", help="Print all potential zones within max signal age")
parser.add_argument("--ob-sweep", action="store_true", help="Require sweep-and-reject of swing level before OB zone entry")
parser.add_argument("--instance-id", type=str, default="SMC", help="Unique instance ID for trade comments and PnL tracking")
parser.add_argument("--volume", type=float, default=0.01, help="Order volume / lot size")
parser.add_argument("--d1", action="store_true", help="Include D1 timeframe in trend scoring")
parser.add_argument("--rvs-threshold", type=int, default=0, help="Skip trades when reversal probability exceeds this (0=disabled)")
args = parser.parse_args()

SYMBOL = args.symbol.upper()
TF = args.tf
BARS = args.bars
RISK_USD = args.risk
MAX_ORDERS = args.max_orders
LOOKBACK = args.lookback
MAX_WAIT = args.max_wait
USE_VOLUME = args.volume_filter
TP_MULT = args.tp_mult
SESSION = args.session
INTERVAL = args.interval
IS_BACKTEST = args.backtest
IS_WATCH = args.watch
IS_TRADE = args.trade
FRACTAL_STR = args.fractal_strength
ENTRY_MODE = args.entry_mode
EQH_THRESHOLD = args.eqh_threshold
TRAIN_ML = args.train_ml
ML_PATH = args.ml_path or f"smc_ml_{SYMBOL}_{TF}.pkl"
ML_THRESHOLD = args.ml_threshold
USE_ST = args.supertrend
ST_ATR = args.st_atr
ST_MULT = args.st_mult
USE_PVT = args.pvt_filter
SL_ATR = args.sl_atr
TP_ATR = args.tp_atr if args.tp_atr > 0 else args.sl_atr
USE_OF = args.orderflow
OF_MODE = args.of_mode
OF_LOOKBACK = args.of_lookback if args.of_lookback > 0 else (10 if TF == "m15" else 5)
SWING_TF = args.swing_tf
REQUIRE_SWEEP = args.require_sweep
MARKET_ONLY = args.market_only
MAX_SIGNAL_AGE = args.max_signal_age
SHOW_ZONES = args.show_zones
OB_SWEEP = args.ob_sweep
INSTANCE_ID = args.instance_id
VOLUME = args.volume
USE_D1 = args.d1
RVS_THRESHOLD = args.rvs_threshold

logging.basicConfig(filename=f"{SYMBOL}_{TF}_smc.log", level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s")

USD1_TO_PRICE = {"XAUUSD": 1, "BTCUSD": 100}
CSV_MAP = {
    "XAUUSD": {"m15": "XAUUSD_M15_combined.csv", "h1": "XAUUSD_H1_combined.csv"},
    "BTCUSD": {"m15": "BTCUSD_15_combined.csv", "h1": "BTCUSD_H1_combined.csv"},
}

def add_indicators(df):
    df['body'] = abs(df['close'] - df['open'])
    df['range'] = df['high'] - df['low']
    atr = AverageTrueRange(df['high'], df['low'], df['close'], 14)
    df['atr'] = atr.average_true_range()
    df['atr200'] = df['close'].rolling(200).std()  # proxy for ATR(200)
    df['avg_vol'] = df['tick_volume'].rolling(20).mean()
    return df

def compute_supertrend(h, l, c, period=10, multiplier=3.0):
    n = len(c)
    hl2 = (h + l) / 2
    tr = np.maximum(h - l, np.maximum(np.abs(h - np.roll(c, 1)), np.abs(l - np.roll(c, 1))))
    tr[0] = h[0] - l[0]
    atr = np.zeros(n)
    atr[0] = tr[0]
    for i in range(1, n):
        atr[i] = (atr[i-1] * (period - 1) + tr[i]) / period
    up = hl2 - multiplier * atr
    dn = hl2 + multiplier * atr
    trend = np.ones(n)
    st_up, st_dn = np.zeros(n), np.zeros(n)
    st_up[0], st_dn[0] = up[0], dn[0]
    for i in range(1, n):
        if np.isnan(up[i]) or np.isnan(dn[i]):
            trend[i] = trend[i-1]; st_up[i], st_dn[i] = st_up[i-1], st_dn[i-1]; continue
        st_up[i] = max(up[i], st_up[i-1]) if c[i-1] > st_up[i-1] else up[i]
        st_dn[i] = min(dn[i], st_dn[i-1]) if c[i-1] < st_dn[i-1] else dn[i]
        if trend[i-1] == -1 and c[i] > st_dn[i-1]: trend[i] = 1
        elif trend[i-1] == 1 and c[i] < st_up[i-1]: trend[i] = -1
        else: trend[i] = trend[i-1]
    return trend, st_up, st_dn

def compute_pvt(c, vol):
    """Price-Volume Trend: cumulative sum of vol * (close change) / prev_close."""
    pvt = np.zeros(len(c), dtype=float)
    for i in range(1, len(c)):
        if abs(c[i-1]) > 1e-9:
            pvt[i] = pvt[i-1] + vol[i] * (c[i] - c[i-1]) / c[i-1]
    return pvt

def pvt_divergence(h, l, pvt, idx, direction, lookback=30):
    """Check PVT divergence at sweep bar idx."""
    start = max(0, idx - lookback)
    if direction == "SELL":
        window_h = h[start:idx+1]
        window_pvt = pvt[start:idx+1]
        peak_i = np.argmax(window_h)
        if len(window_h) - 1 - peak_i > 2: return False
        pvt_here = window_pvt[-1]
        pvt_prior_max = np.max(window_pvt[:max(1, peak_i)])
        return pvt_here < pvt_prior_max * 0.999
    else:
        window_l = l[start:idx+1]
        window_pvt = pvt[start:idx+1]
        trough_i = np.argmin(window_l)
        if len(window_l) - 1 - trough_i > 2: return False
        pvt_here = window_pvt[-1]
        pvt_prior_min = np.min(window_pvt[:max(1, trough_i)])
        return pvt_here > pvt_prior_min * 1.001

# ---------- ORDER FLOW ----------
def compute_orderflow_approx(df):
    """Approximate buy/sell volume from OHLC + tick_volume (for backtest)."""
    df = df.copy()
    total = df['high'] - df['low'] + eps
    df['buy_vol'] = df['tick_volume'] * (df['close'] - df['low']) / total
    df['sell_vol'] = df['tick_volume'] * (df['high'] - df['close']) / total
    df['delta'] = df['buy_vol'] - df['sell_vol']
    df['cum_delta'] = df['delta'].cumsum()
    return df

def compute_orderflow_live(df, mt5, tf, n_bars=200):
    """Fetch MT5 ticks and compute delta per bar using bid tick-rule or OHLC approx."""
    if OF_MODE == "ohlc":
        return compute_orderflow_approx(df)
    rates = mt5.copy_rates_from_pos(SYMBOL, tf, 0, n_bars)
    if rates is None or len(rates) < 2:
        return compute_orderflow_approx(df)
    bar_times = [r['time'] for r in rates]
    ticks = mt5.copy_ticks_from(SYMBOL, bar_times[0], 200000)
    if ticks is None or len(ticks) == 0:
        return compute_orderflow_approx(df)
    buy_count = np.zeros(n_bars)
    sell_count = np.zeros(n_bars)
    total_ticks = np.zeros(n_bars)
    tick_idx = 0
    prev_bid = None
    for bar_idx in range(n_bars - 1):
        bar_start = bar_times[bar_idx]
        bar_end = bar_times[bar_idx + 1]
        while tick_idx < len(ticks) and ticks[tick_idx]['time'] < bar_start:
            tick_idx += 1
        while tick_idx < len(ticks) and ticks[tick_idx]['time'] < bar_end:
            t = ticks[tick_idx]
            bid = t['bid']
            total_ticks[bar_idx] += 1
            if bid > 0 and prev_bid is not None:
                if bid > prev_bid:
                    buy_count[bar_idx] += 1
                elif bid < prev_bid:
                    sell_count[bar_idx] += 1
            prev_bid = bid
            tick_idx += 1
    offset = len(df) - n_bars
    if offset < 0:
        buy_count = np.pad(buy_count[-len(df):], (len(df) - n_bars, 0), 'constant')
        sell_count = np.pad(sell_count[-len(df):], (len(df) - n_bars, 0), 'constant')
        total_ticks = np.pad(total_ticks[-len(df):], (len(df) - n_bars, 0), 'constant')
    df['buy_vol'] = buy_count[-len(df):] if offset >= 0 else buy_count
    df['sell_vol'] = sell_count[-len(df):] if offset >= 0 else sell_count
    df['total_ticks'] = total_ticks[-len(df):] if offset >= 0 else total_ticks
    df['delta'] = df['buy_vol'] - df['sell_vol']
    df['cum_delta'] = df['delta'].cumsum()
    return df

MIN_OF_TICKS = 50  # minimum ticks per bar to trust delta

def orderflow_ok(df, idx, direction, lookback=3):
    """Check if orderflow delta confirms trade direction."""
    if 'delta' not in df.columns:
        return True
    start = max(0, idx - lookback + 1)
    if 'total_ticks' in df.columns:
        min_ticks = MIN_OF_TICKS * lookback
        if df['total_ticks'].iloc[start:idx+1].sum() < min_ticks:
            return True  # skip filter if insufficient tick data
    avg_delta = df['delta'].iloc[start:idx+1].mean()
    if direction == "SELL":
        return avg_delta < 0
    else:
        return avg_delta > 0

def is_high_fractal(highs, i, strength=2):
    for k in range(1, strength + 1):
        if highs[i] <= highs[i - k] or highs[i] <= highs[i + k]:
            return False
    return True


def is_low_fractal(lows, i, strength=2):
    for k in range(1, strength + 1):
        if lows[i] >= lows[i - k] or lows[i] >= lows[i + k]:
            return False
    return True

def get_fractal_snr(df, strength=2):
    highs = df["high"].values
    lows = df["low"].values

    resistance = None
    support = None

    for i in range(len(df) - strength - 1, strength - 1, -1):
        if resistance is None and is_high_fractal(highs, i, strength):
            resistance = highs[i]

        if support is None and is_low_fractal(lows, i, strength):
            support = lows[i]

        if resistance is not None and support is not None:
            break

    return support, resistance

def get_swing_points(highs, lows, strength=2):
    """Fractal swing: center > all left/right neighbors (same as is_high_fractal)."""
    n = len(highs); sh = np.zeros(n, dtype=bool); sl = np.zeros(n, dtype=bool)
    for i in range(strength, n - strength):
        sh[i] = all(highs[i] > highs[i-k] and highs[i] > highs[i+k] for k in range(1, strength+1))
        sl[i] = all(lows[i] < lows[i-k] and lows[i] < lows[i+k] for k in range(1, strength+1))
    
    return sh, sl

def resample_tf(df, rule):
    d = df.copy()
    d['time'] = d['timestamp']
    return d.resample(rule, on='time').agg({
        'open': 'first', 'high': 'max', 'low': 'min',
        'close': 'last', 'tick_volume': 'sum',
    }).dropna().reset_index()

def resample_h4(m15):
    return resample_tf(m15, '4h')

def classify_bias(h, l):
    sh, sl = get_swing_points(h, l, FRACTAL_STR)
    si = np.where(sh | sl)[0]
    if len(si) < 2: return "SIDEWAYS"
    i0, i1 = si[-2], si[-1]
    p0 = h[i0] if sh[i0] else l[i0]
    p1 = h[i1] if sh[i1] else l[i1]
    if p1 > p0: return "BULLISH"
    if p1 < p0: return "BEARISH"
    return "SIDEWAYS"

def atr_contracting_check(h, l, c, period=14):
    tr = np.maximum(h - l, np.maximum(np.abs(h - np.roll(c, 1)), np.abs(l - np.roll(c, 1))))
    tr[0] = h[0] - l[0]
    n = len(tr); atr_loc = np.zeros(n); atr_loc[0] = tr[0]
    for i in range(1, n): atr_loc[i] = (atr_loc[i-1] * (period - 1) + tr[i]) / period
    recent = atr_loc[-5:].mean()
    avg = atr_loc[-min(period, n):].mean()
    return recent < avg * 0.7

from trend import get_trend, trend_from_df  # scoring-based trend detection

LONDON_OPEN_START, LONDON_OPEN_END = 7, 10
NY_OPEN_START, NY_OPEN_END = 12, 15

def session_ok(hour_utc):
    if SESSION == "any": return True
    if SESSION == "london": return LONDON_OPEN_START <= hour_utc < LONDON_OPEN_END
    if SESSION == "ny": return NY_OPEN_START <= hour_utc < NY_OPEN_END
    return True

def _last_swing_high_before(df, idx):
    sub = df.iloc[max(0, idx - 30):idx]
    if len(sub) < 3: return None
    c = sub['close'].values
    sh, _ = get_swing_points(c, c, FRACTAL_STR)
    si = np.where(sh)[0]
    return c[si[-1]] if len(si) > 0 else None

def _last_swing_low_before(df, idx):
    sub = df.iloc[max(0, idx - 30):idx]
    if len(sub) < 3: return None
    c = sub['close'].values
    _, sl = get_swing_points(c, c, FRACTAL_STR)
    si = np.where(sl)[0]
    return c[si[-1]] if len(si) > 0 else None

def find_fvg(df, i, direction, max_lookahead=5, min_atr_factor=0.1):
    """Bullish FVG (BUY): middle candle bullish, BOS up, gap between wicks 1 and 3.
    Bearish FVG (SELL): middle candle bearish, BOS down, gap between wicks 1 and 3.
    Returns None if the gap has already been filled by subsequent candles."""
    ref_atr = df.iloc[i]['atr'] if 'atr' in df.columns else 0
    min_gap = ref_atr * min_atr_factor if ref_atr > 0 else 0
    n = len(df)
    for gap_start in range(i+1, min(i+max_lookahead, n-2)):
        gap_end = gap_start + 2
        if gap_end >= n:
            break
        c0 = df.iloc[gap_start]
        c1 = df.iloc[gap_start + 1]
        c2 = df.iloc[gap_end]
        if direction == "BUY":
            if c1['close'] <= c1['open']: continue
            prev_sw = _last_swing_high_before(df, gap_start)
            if prev_sw is None or max(c0['close'], c1['close'], c2['close']) <= prev_sw:
                continue
            gap = c2['low'] - c0['high']
            if gap > min_gap and (gap_end + 1 >= n or df['low'].iloc[gap_end+1:].min() > c0['high']):
                return c0['high'], c2['low']
        else:
            if c1['close'] >= c1['open']: continue
            prev_sw = _last_swing_low_before(df, gap_start)
            if prev_sw is None or min(c0['close'], c1['close'], c2['close']) >= prev_sw:
                continue
            gap = c0['low'] - c2['high']
            if gap > min_gap and (gap_end + 1 >= n or df['high'].iloc[gap_end+1:].max() < c0['low']):
                return c2['high'], c0['low']
    return None, None

def find_order_block(highs, lows, swing_idx, break_idx, direction):
    start = min(swing_idx, break_idx) + 1
    end = max(swing_idx, break_idx)
    if end - start < 1: return None, None, None
    if direction == "BUY":
        idx = np.argmin(lows[start:end+1]) + start
    else:
        idx = np.argmax(highs[start:end+1]) + start
    return lows[idx], highs[idx], "OB"

def get_premium_discount_zones(swing_high, swing_low):
    if swing_high is None or swing_low is None: return None, None, None
    prem_bottom, prem_top = 0.95*swing_high + 0.05*swing_low, swing_high
    eq = (swing_high + swing_low) / 2
    disc_bottom, disc_top = swing_low, 0.05*swing_high + 0.95*swing_low
    return (prem_bottom, prem_top), eq, (disc_bottom, disc_top)

def is_price_in_zone(mid, zl, zh, sigma, pad):
    return (zl - sigma*pad) <= mid <= (zh + sigma*pad)

COMMENT_PREFIX = "SMC"

def count_positions(mt5, keywords=None):
    if mt5 is None: return 0
    count = 0
    positions = mt5.positions_get(symbol=SYMBOL)
    if positions:
        if keywords:
            count += sum(1 for p in positions if all(k in (p.comment or "") for k in keywords))
        else:
            count += len(positions)
    orders = mt5.orders_get(symbol=SYMBOL)
    if orders:
        if keywords:
            count += sum(1 for o in orders if all(k in (o.comment or "") for k in keywords))
        else:
            count += len(orders)
    return count

# ---------- ML FEATURES ----------
def extract_ml_features_smc(o, h, l, c, body, atr_val, vol, avg_vol, atr_arr, ts_arr,
                            origin_idx, entry_j, direction, zone_source, sig_source,
                            one_r, is_acc, dom_bias):
    """Extract feature vector for an SMC trade. Returns numpy array of 25 features."""
    f = np.zeros(25, dtype=np.float32)
    origin_idx = max(0, origin_idx)
    f[0] = atr_val / (c[entry_j] + eps)            # 0: atr/close
    f[1] = vol[origin_idx] / (avg_vol[origin_idx] + eps) if not np.isnan(avg_vol[origin_idx]) else 1.0
    f[2] = (entry_j - origin_idx) / 10.0           # 2: bars waited
    ma50 = np.mean(c[max(0, entry_j-50):entry_j]) if entry_j >= 50 else c[entry_j]
    f[3] = (c[entry_j] - ma50) / (atr_val + eps)   # 3: dist_ema50
    ma200 = np.mean(c[max(0, entry_j-200):entry_j]) if entry_j >= 200 else c[entry_j]
    f[4] = (c[entry_j] - ma200) / (atr_val + eps)  # 4: dist_ema200
    f[5] = (ma50 - ma200) / (c[entry_j] + eps)     # 5: trend
    f[6] = 1.0 if direction == "BUY" else 0.0      # 6: is_buy
    f[7] = 1.0 if zone_source in ("FVG",) else 0.0
    f[8] = 1.0 if zone_source in ("OB",) else 0.0
    f[9] = 1.0 if zone_source in ("Premium", "Discount") else 0.0  # pd zone
    f[10] = 1.0 if sig_source == "LIQUIDITY" else 0.0               # sig_type
    f[11] = 1.0 if sig_source == "STRUCTURE" else 0.0
    f[12] = body[origin_idx] / (atr_val + eps)     # 12: body/atr
    f[13] = (h[origin_idx] - l[origin_idx]) / (atr_val + eps)  # 13: range/atr
    f[14] = 1.0 if c[entry_j] > o[entry_j] else 0.0            # 14: entry bullish
    entry_ts = pd.Timestamp(ts_arr[entry_j])
    hour = entry_ts.hour if hasattr(entry_ts, 'hour') else 12
    f[15] = math.cos(2 * math.pi * hour / 24)      # 15: hr_cos
    f[16] = math.sin(2 * math.pi * hour / 24)      # 16: hr_sin
    atr50 = np.nanmean(atr_arr[max(0, entry_j-50):entry_j+1])
    f[17] = atr_val / (atr50 + eps)                 # 17: atr_trend
    f[18] = one_r / (atr_val + eps)                 # 18: risk_dist
    f[19] = vol[entry_j] / (avg_vol[entry_j] + eps) if not np.isnan(avg_vol[entry_j]) else 1.0
    f[20] = atr_arr[entry_j] / (atr_arr[max(0, origin_idx)] + eps)  # 20: atr_change
    f[21] = 1.0 if zone_source == "FVG" else 0.0   # 21: is_fvg (aux)
    f[22] = 1.0 if (dom_bias == BULLISH and direction == "BUY") or (dom_bias == BEARISH and direction == "SELL") else 0.0
    f[23] = 1.0 if is_acc else 0.0
    f[24] = 1.0 if dom_bias == BULLISH else -1.0 if dom_bias == BEARISH else 0.0
    return f

def train_ml_from_backtest_smc(all_features, all_labels):
    """Train Random Forest on collected SMC trade data, print results, save model."""
    from sklearn.ensemble import RandomForestClassifier
    X = np.array(all_features, dtype=np.float32)
    y = np.array(all_labels, dtype=np.int32)
    n_trades = len(y)
    pos_ratio = y.mean()
    print(f"\n  ML Training: {n_trades} trades ({pos_ratio:.1%} win rate)")
    if n_trades < 100:
        print("  Too few trades for ML, skipping"); return None
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
    for th in np.arange(0.3, 0.85, 0.05):
        preds = (test_prob >= th).astype(int)
        if preds.sum() == 0: continue
        net = (preds * y_test).sum() * TP_MULT - (preds * (1 - y_test)).sum() * 1
        if net > best_net: best_net, best_th = net, th
    test_preds = (test_prob >= best_th).astype(int)
    test_wr = (test_preds == y_test).mean()
    test_trades = test_preds.sum()
    test_wins = (test_preds * y_test).sum()
    test_net = test_wins * TP_MULT - (test_trades - test_wins) * 1
    feat_names = [
        "atr/close", "vol_ratio", "bars_waited", "dist_ema50", "dist_ema200",
        "trend", "is_buy", "zone_fvg", "zone_ob", "zone_pd",
        "sig_liq", "sig_struct", "body/atr", "range/atr", "entry_bull",
        "hr_cos", "hr_sin", "atr_trend", "risk_dist", "entry_vol",
        "atr_change", "is_fvg", "trend_aligned", "is_accum", "trend_bias"
    ]
    fi = sorted(zip(feat_names, model.feature_importances_), key=lambda x: -x[1])
    print(f"  Time-split: train={split} trades, test={n_trades-split} trades")
    print(f"  Test split WR: {y_test.mean():.1%} (baseline)")
    print(f"  Best threshold: {best_th:.2f} -> {test_trades} trades, {test_wr:.1%} WR, {test_net:+.1f}R net")
    print(f"\n  Top 10 features:")
    for name, imp in fi[:10]:
        print(f"    {name:16s}  {imp:.3f}")
    artifact = {"model": model, "threshold": best_th, "features": feat_names}
    with open(ML_PATH, "wb") as f:
        pickle.dump(artifact, f)
    print(f"\n  Model saved to {ML_PATH}")
    return model

def run_backtest():
    print(f"\n{'='*60}")
    print(f"  SMC STRATEGY BACKTEST")
    st_str = " ST" if USE_ST else ""
    of_str = " OF" if USE_OF else ""
    if SL_ATR > 0:
        print(f"  {SYMBOL} TF={TF} -- Session={SESSION} SL={SL_ATR:.1f}ATR TP={TP_ATR:.1f}ATR Entry={ENTRY_MODE}{st_str}{of_str}")
    else:
        print(f"  {SYMBOL} TF={TF} -- Session={SESSION} TP={TP_MULT}R Entry={ENTRY_MODE}{st_str}{of_str}")
    print(f"{'='*60}")
    csv_path = CSV_MAP.get(SYMBOL, {}).get(TF)
    if not csv_path: print(f"  ERROR: No CSV for {SYMBOL}/{TF}"); return
    import os
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
    if USE_OF:
        df = compute_orderflow_approx(df)
        print(f"  Orderflow filter: enabled (OHLC approx, lookback={OF_LOOKBACK})")
    n = len(df)
    if n < 300: print("  Too few bars"); return

    o = df['open'].values.astype(float)
    h = df['high'].values.astype(float)
    l = df['low'].values.astype(float)
    c = df['close'].values.astype(float)
    body = df['body'].values.astype(float)
    atr = df['atr'].values.astype(float)
    atr200 = df['atr200'].values.astype(float)
    ts = df['timestamp'].values
    vol = df['tick_volume'].values.astype(float)
    avg_vol = df['avg_vol'].values.astype(float)

    sh, sl = get_swing_points(h, l, FRACTAL_STR)

    st_trend = None
    if USE_ST:
        st_trend, _, _ = compute_supertrend(h, l, c, ST_ATR, ST_MULT)
        print(f"  SuperTrend filter: period={ST_ATR} mult={ST_MULT}")

    pvt_arr = None
    if USE_PVT:
        pvt_arr = compute_pvt(c, vol)
        print(f"  PVT divergence filter: enabled")

    one_r = RISK_USD * USD1_TO_PRICE.get(SYMBOL, 1)
    rwd = one_r * TP_MULT
    trades = wins = losses = 0
    zone_counts = {"FVG": 0, "OB": 0, "Premium": 0, "Discount": 0}
    ml_collect = []
    ml_model = None
    if TRAIN_ML:
        print(f"  ML training mode: collecting features from all trades")
    elif ML_THRESHOLD > 0 and os.path.exists(ML_PATH):
        with open(ML_PATH, "rb") as f:
            artifact = pickle.load(f)
        ml_model = artifact["model"]
        if ml_model.n_features_in_ != 25:
            print(f"  ERROR: model has {ml_model.n_features_in_} features, expected 22. Retrain.")
            logging.warning(f"  ERROR: model has {ml_model.n_features_in_} features, expected 22. Retrain.")
            ml_model = None
        if ml_model:
            print(f"  ML filter: loaded {ML_PATH}, threshold={ML_THRESHOLD:.2f}")
            logging.info(f"  ML filter: loaded {ML_PATH}, threshold={ML_THRESHOLD:.2f}")
    elif ML_THRESHOLD > 0:
        print(f"  WARNING: {ML_PATH} not found, ML filter disabled")
    sig_counts = {"CHoCH_BULL": 0, "CHoCH_BEAR": 0, "BOS_BULL": 0, "BOS_BEAR": 0, "EQH": 0, "EQL": 0}

    # Swing tracking
    active_sh_idx = -1; active_sh_price = None; active_sh_crossed = False
    prev_sh_price = None
    active_sl_idx = -1; active_sl_price = None; active_sl_crossed = False
    prev_sl_price = None
    trend = NEUTRAL

    # EQH/EQL state
    eqh_detected_at = -1   # bar index where EQH was detected
    eqh_level = None        # the price level of the EQH (swing high price)
    eql_detected_at = -1
    eql_level = None

    sigs = []  # pending signals: {type, direction, sig_source, break_type, zone_info, ...}

    _is_acc, _dom_str, _, _ = trend_from_df(df, resample_tf, use_d1=USE_D1)
    ACC_STATE = (_is_acc, BULLISH if _dom_str == "BULLISH" else BEARISH if _dom_str == "BEARISH" else NEUTRAL)
    print(ACC_STATE)

    print(f"  Scanning {n} bars...")

    for i in range(250, n):
        if np.isnan(atr[i]) or atr[i] <= 0: continue
        hour_utc = pd.Timestamp(ts[i]).hour
        if not session_ok(hour_utc): continue

        conf = i - FRACTAL_STR
        if conf > 0 and conf < n:
            # --- New swing high confirmed ---
            if sh[conf]:
                prev_sh_price = active_sh_price
                active_sh_price = h[conf]
                active_sh_idx = conf
                active_sh_crossed = False

                # EQH detection: new swing high is close to previous swing high
                if (prev_sh_price is not None
                    and abs(active_sh_price - prev_sh_price) < EQH_THRESHOLD * atr200[conf]
                    and ENTRY_MODE in ("all", "liquidity", "pd")):
                    eqh_detected_at = conf
                    eqh_level = active_sh_price
                    sig_counts["EQH"] += 1

            # --- New swing low confirmed ---
            if sl[conf]:
                prev_sl_price = active_sl_price
                active_sl_price = l[conf]
                active_sl_idx = conf
                active_sl_crossed = False

                # EQL detection: new swing low is close to previous swing low
                if (prev_sl_price is not None
                    and abs(active_sl_price - prev_sl_price) < EQH_THRESHOLD * atr200[conf]
                    and ENTRY_MODE in ("all", "liquidity", "pd")):
                    eql_detected_at = conf
                    eql_level = active_sl_price
                    sig_counts["EQL"] += 1

        # Trend: based on last two swing highs or lows
        if active_sh_price is not None and prev_sh_price is not None:
            trend = BULLISH if active_sh_price > prev_sh_price else BEARISH
        elif active_sl_price is not None and prev_sl_price is not None:
            trend = BULLISH if active_sl_price > prev_sl_price else BEARISH

        # --- Structure break detection (CHoCH/BOS) ---
        structure_check = False
        is_cho_ch = False
        break_dir = None
        swing_idx = -1
        swing_price = None

        if not active_sh_crossed and active_sh_idx >= 0 and i - active_sh_idx < LOOKBACK:
            if c[i] > active_sh_price:
                is_cho_ch = (trend == BEARISH)
                if is_cho_ch: sig_counts["CHoCH_BULL"] += 1
                else: sig_counts["BOS_BULL"] += 1
                structure_check = True
                break_dir = "BUY"
                swing_idx = active_sh_idx
                swing_price = active_sh_price
                active_sh_crossed = True

        if not active_sl_crossed and active_sl_idx >= 0 and i - active_sl_idx < LOOKBACK:
            if c[i] < active_sl_price:
                is_cho_ch = (trend == BULLISH)
                if is_cho_ch: sig_counts["CHoCH_BEAR"] += 1
                else: sig_counts["BOS_BEAR"] += 1
                structure_check = True
                break_dir = "SELL"
                swing_idx = active_sl_idx
                swing_price = active_sl_price
                active_sl_crossed = True

        if structure_check and ENTRY_MODE in ("all", "structure", "fvg", "ob"):
            is_acc_back, h4_bias = ACC_STATE
            
            if is_acc_back: continue
            if is_cho_ch:
                continue  # skip reversals (CHoCH)
            # Add BOS (continuation) signal
            sigs.append({
                "type": SIG_STRUCTURE, "direction": break_dir,
                "break_type": "BOS",
                "swing_idx": swing_idx, "swing_price": swing_price,
                "break_idx": i, "atr": atr[i],
            })

        # --- Check pending signals for zone entry ---
        j = 0
        while j < len(sigs):
            sig = sigs[j]
            if i <= sig['break_idx']: j += 1; continue
            if i - sig['break_idx'] > MAX_WAIT: sigs.pop(j); continue

            direction = sig['direction']
            mid = (h[i] + l[i]) / 2
            sigma = sig['atr']
            entered = False

            if 'zones' not in sig:
                zones = []
                if sig['type'] == SIG_STRUCTURE:
                    # FVG + OB for structure signals
                    if ENTRY_MODE in ("all", "structure", "fvg"):
                        fvg_l, fvg_h = find_fvg(df, sig['break_idx'], direction)
                        if fvg_l is not None: zones.append((fvg_l, fvg_h, "FVG"))
                    if ENTRY_MODE in ("all", "structure", "ob"):
                        ob_l, ob_h, _ = find_order_block(h, l, sig['swing_idx'], sig['break_idx'], direction)
                        if ob_l is not None: zones.append((ob_l, ob_h, "OB"))
                # Liquidity signals already have zones pre-set from detection
                sig['zones'] = zones if 'zones' not in sig else sig.get('zones', [])

            for zl, zh, zs in sig['zones']:
                if is_price_in_zone(mid, zl, zh, sigma, 0.25):
                    entry_price = mid
                    if SL_ATR > 0:
                        sd = atr[i] * SL_ATR
                        td = atr[i] * TP_ATR
                    else:
                        sd = one_r; td = rwd
                    sl_price = entry_price - sd if direction == "BUY" else entry_price + sd
                    tp_price = entry_price + td if direction == "BUY" else entry_price - td
                    end = min(i + 50, n)
                    if direction == "BUY":
                        hit_tp = h[i+1:end].max() >= tp_price if i+1 < end else False
                    else:
                        hit_tp = l[i+1:end].min() <= tp_price if i+1 < end else False
                    # PVT divergence filter
                    if USE_PVT and pvt_arr is not None and sig['type'] == SIG_STRUCTURE:
                        if not pvt_divergence(h, l, pvt_arr, sig['break_idx'], direction):
                            continue
                    # Orderflow filter
                    if USE_OF and not orderflow_ok(df, i, direction, OF_LOOKBACK):
                        continue
                    # ML filter
                    if ml_model is not None:
                        feat = extract_ml_features_smc(o, h, l, c, body, atr[i], vol, avg_vol, atr, ts,
                            sig['break_idx'], i, sig['direction'], zs, "STRUCTURE", one_r, is_acc_back, h4_bias)
                        prob = ml_model.predict_proba(feat[:ml_model.n_features_in_].reshape(1, -1))[0, 1]
                        if prob < ML_THRESHOLD:
                            continue
                    if USE_ST and st_trend is not None and st_trend[i] != (-1 if sig['direction'] == "SELL" else 1):
                        continue
                    zone_counts[zs] += 1
                    if TRAIN_ML:
                        feat = extract_ml_features_smc(o, h, l, c, body, atr[i], vol, avg_vol, atr, ts,
                            sig['break_idx'], i, sig['direction'], zs, "STRUCTURE", one_r, is_acc_back, h4_bias)
                        ml_collect.append((feat, int(hit_tp)))
                    trades += 1
                    if hit_tp: wins += 1
                    else: losses += 1
                    entered = True
                    sigs.pop(j)
                    break

            if not entered: j += 1

        # --- Liquidity signal: EQH detected → check premium zone for SELL ---
        if eqh_detected_at >= 0 and ENTRY_MODE in ("all", "liquidity", "pd"):
            prem, _, _ = get_premium_discount_zones(active_sh_price, active_sl_price)
            if prem is not None:
                mid = (h[i] + l[i]) / 2
                if is_price_in_zone(mid, prem[0], prem[1], atr[i], 0.25):
                    direction = "SELL"
                    entry_price = mid
                    if SL_ATR > 0:
                        sd = atr[i] * SL_ATR
                        td = atr[i] * TP_ATR
                    else:
                        sd = one_r; td = rwd
                    sl_price = entry_price + sd
                    tp_price = entry_price - td
                    end = min(i + 50, n)
                    hit_tp = l[i+1:end].min() <= tp_price if i+1 < end else False
                    # PVT divergence filter
                    if USE_PVT and pvt_arr is not None:
                        if not pvt_divergence(h, l, pvt_arr, eqh_detected_at, "SELL"):
                            continue
                    # Orderflow filter
                    if USE_OF and not orderflow_ok(df, i, "SELL", OF_LOOKBACK):
                        continue
                    # ML filter
                    if ml_model is not None:
                        feat = extract_ml_features_smc(o, h, l, c, body, atr[i], vol, avg_vol, atr, ts,
                            eqh_detected_at, i, "SELL", "Premium", "LIQUIDITY", one_r, is_acc_back, h4_bias)
                        prob = ml_model.predict_proba(feat[:ml_model.n_features_in_].reshape(1, -1))[0, 1]
                        if prob < ML_THRESHOLD:
                            continue
                    if USE_ST and st_trend is not None and st_trend[i] != -1:
                        continue
                    zone_counts["Premium"] += 1
                    if TRAIN_ML:
                        feat = extract_ml_features_smc(o, h, l, c, body, atr[i], vol, avg_vol, atr, ts,
                            eqh_detected_at, i, "SELL", "Premium", "LIQUIDITY", one_r, is_acc_back, h4_bias)
                        ml_collect.append((feat, int(hit_tp)))
                    trades += 1
                    if hit_tp: wins += 1
                    else: losses += 1
                    eqh_detected_at = -1  # consumed

        # --- Liquidity signal: EQL detected → check discount zone for BUY ---
        if eql_detected_at >= 0 and ENTRY_MODE in ("all", "liquidity", "pd"):
            _, _, disc = get_premium_discount_zones(active_sh_price, active_sl_price)
            if disc is not None:
                mid = (h[i] + l[i]) / 2
                if is_price_in_zone(mid, disc[0], disc[1], atr[i], 0.25):
                    direction = "BUY"
                    entry_price = mid
                    if SL_ATR > 0:
                        sd = atr[i] * SL_ATR
                        td = atr[i] * TP_ATR
                    else:
                        sd = one_r; td = rwd
                    sl_price = entry_price - sd
                    tp_price = entry_price + td
                    end = min(i + 50, n)
                    hit_tp = h[i+1:end].max() >= tp_price if i+1 < end else False
                    # PVT divergence filter
                    if USE_PVT and pvt_arr is not None:
                        if not pvt_divergence(h, l, pvt_arr, eql_detected_at, "BUY"):
                            continue
                    # Orderflow filter
                    if USE_OF and not orderflow_ok(df, i, "BUY", OF_LOOKBACK):
                        continue
                    # ML filter
                    if ml_model is not None:
                        feat = extract_ml_features_smc(o, h, l, c, body, atr[i], vol, avg_vol, atr, ts,
                            eql_detected_at, i, "BUY", "Discount", "LIQUIDITY", one_r, is_acc_back, h4_bias)
                        prob = ml_model.predict_proba(feat[:ml_model.n_features_in_].reshape(1, -1))[0, 1]
                        if prob < ML_THRESHOLD:
                            continue
                    if USE_ST and st_trend is not None and st_trend[i] != 1:
                        continue
                    zone_counts["Discount"] += 1
                    if TRAIN_ML:
                        feat = extract_ml_features_smc(o, h, l, c, body, atr[i], vol, avg_vol, atr, ts,
                            eql_detected_at, i, "BUY", "Discount", "LIQUIDITY", one_r, is_acc_back, h4_bias)
                        ml_collect.append((feat, int(hit_tp)))
                    trades += 1
                    if hit_tp: wins += 1
                    else: losses += 1
                    eql_detected_at = -1  # consumed

    print(f"\n  RESULTS ({SYMBOL} {TF})")
    print(f"  {'='*40}")
    print(f"  Signals: {sig_counts}")
    if trades:
        wr = wins / trades
        ev = wr * TP_MULT - (1 - wr) * 1
        pnl = wins * TP_MULT - losses * 1
        print(f"  Trades:  {trades:,}")
        print(f"  Wins:    {wins:,}   Losses: {losses:,}")
        print(f"  WR:      {wr:.1%}")
        print(f"  EV:      {ev:+.2f}R")
        print(f"  PnL:     {pnl:+,.0f}R")
        print(f"  Zones:   {zone_counts}")
    else:
        print("  No trades")

    if TRAIN_ML and len(ml_collect) >= 100:
        all_features = [x[0] for x in ml_collect]
        all_labels = [x[1] for x in ml_collect]
        train_ml_from_backtest_smc(all_features, all_labels)

def place_mt5_trade(mt5, setup):
    direction = setup['direction']
    tick = mt5.symbol_info_tick(SYMBOL)
    info = mt5.symbol_info(SYMBOL)
    if tick is None or info is None: print("  ERROR: No symbol data"); return None
    zone_boundary = setup.get('zone_low' if direction == "BUY" else 'zone_high')
    entry_p = setup.get('entry_price')
    if zone_boundary is not None and zone_boundary > 0 and entry_p is not None:
        limit_price = (zone_boundary + entry_p) / 2
    else:
        limit_price = zone_boundary or entry_p
    atr_val = setup.get('atr', 0)
    if SL_ATR > 0 and atr_val > 0:
        sl_dist = atr_val * SL_ATR
        tp_dist = atr_val * TP_ATR
    else:
        sl_dist = RISK_USD * USD1_TO_PRICE.get(SYMBOL, 1)
        tp_dist = sl_dist * TP_MULT
    THRESH = 0.2 * sl_dist
    is_market = False
    if direction == "BUY":
        dist_to_zone = tick.ask - limit_price
        if dist_to_zone <= THRESH:
            is_market = True
        elif MARKET_ONLY:
            print(f"  SKIP: buy zone {limit_price:.1f} not within market range ({dist_to_zone:.2f})")
            return None
        elif dist_to_zone > 3 * atr_val:
            print(f"  SKIP: buy zone {limit_price:.1f} too far below ask {tick.ask:.1f} ({dist_to_zone:.2f})")
            return None
        # else: limit order at zone_low
    else:
        dist_to_zone = limit_price - tick.bid
        if dist_to_zone <= THRESH:
            is_market = True
        elif MARKET_ONLY:
            print(f"  SKIP: sell zone {limit_price:.1f} not within market range ({dist_to_zone:.2f})")
            return None
        elif dist_to_zone > 3 * atr_val:
            print(f"  SKIP: sell zone {limit_price:.1f} too far above bid {tick.bid:.1f} ({dist_to_zone:.2f})")
            return None
        # else: limit order at zone_high
    entry_price = tick.ask if (direction == "BUY" and is_market) else (tick.bid if (direction == "SELL" and is_market) else limit_price)
    mode = "MARKET" if is_market else "LIMIT"
    print(f"  >>> {mode} {direction} at {entry_price:.2f} (zone: {limit_price:.2f}, sl_dist: {sl_dist:.4f})")
    if direction == "BUY":
        sl = entry_price - sl_dist; tp = entry_price + tp_dist
    else:
        sl = entry_price + sl_dist; tp = entry_price - tp_dist
    ts = info.trade_tick_size or 0.01
    sl = round(sl / ts) * ts; tp = round(tp / ts) * ts
    min_dist = info.trade_stops_level * info.point if info.trade_stops_level else 0
    if min_dist > 0:
        if direction == "BUY":
            if entry_price - sl < min_dist: sl = entry_price - min_dist
            if tp - entry_price < min_dist: tp = entry_price + min_dist
        else:
            if sl - entry_price < min_dist: sl = entry_price + min_dist
            if entry_price - tp < min_dist: tp = entry_price - min_dist
    if is_market:
        order_type = mt5.ORDER_TYPE_BUY if direction == "BUY" else mt5.ORDER_TYPE_SELL
        action = mt5.TRADE_ACTION_DEAL
    else:
        order_type = mt5.ORDER_TYPE_BUY_LIMIT if direction == "BUY" else mt5.ORDER_TYPE_SELL_LIMIT
        action = mt5.TRADE_ACTION_PENDING
    #comment = f"SMC {setup.get('zone_source','?')} {setup.get('sig_source','?')}"
    comment = f"{INSTANCE_ID} {TF} SMC {setup.get('zone_source','?')}"
    request = {
        "action": action, "symbol": SYMBOL, "volume": VOLUME,
        "type": order_type, "price": entry_price, "sl": sl, "tp": tp,
        "deviation": 20, "magic": 1016, "comment": comment,
        "type_time": mt5.ORDER_TIME_GTC, "type_filling": mt5.ORDER_FILLING_IOC,
    }
    return mt5.order_send(request)

def print_setup(setup):
    print(f"\n{'='*60}")
    print(f"  SMC SIGNAL: {setup.get('sig_source','?')} {setup.get('break_type','?')}")
    print(f"{'='*60}")
    for k, v in setup.items():
        if k != 'zone_low' and k != 'zone_high':
            print(f"  {k}: {v}")
    zl = setup.get('zone_low'); zh = setup.get('zone_high')
    if zl is not None and zh is not None:
        print(f"  zone_low: {zl:.2f}")
        print(f"  zone_high: {zh:.2f}")
    st = setup.get('st_trend')
    if st is not None:
        label = "BULLISH" if st == 1 else "BEARISH" if st == -1 else "NONE"
        print(f"  supertrend: {label}")
    of_delta = setup.get('delta')
    of_cum = setup.get('cum_delta')
    of_avg = setup.get('avg_delta')
    if of_delta is not None:
        parts = [f"  delta:     {of_delta:+.0f}"]
        if of_avg is not None:
            parts.append(f"avg_delta: {of_avg:+.0f}")
        if of_cum is not None:
            parts.append(f"cum_delta: {of_cum:+.0f}")
        print("   ".join(parts))
    print(f"{'='*60}\n")

def scan_for_setups(df, return_all=False):
    """Scan recent bars for any valid setup (CHoCH/BOS → FVG/OB or EQH/EQL → P/D).
    When return_all=True, returns a list of all valid setups (for zone display)."""
    n = len(df)
    if n < 200: return None
    N = min(100, n // 2)
    highs = df['high'].values; lows = df['low'].values; closes = df['close'].values
    atr_arr = df['atr'].values; atr200 = df['atr200'].values; ts = df['timestamp'].values
    sh, sl = get_swing_points(highs, lows, FRACTAL_STR)

    # Pre-compute PVT if enabled
    pvt_scan = None
    if USE_PVT:
        vol_a = df['tick_volume'].values.astype(float)
        pvt_scan = compute_pvt(closes.astype(float), vol_a)

    # Pre-compute SuperTrend if enabled
    st_scan = None
    if USE_ST:
        st_scan, _, _ = compute_supertrend(highs.astype(float), lows.astype(float), closes.astype(float), ST_ATR, ST_MULT)

    all_setups = []

    # Walk through to get latest swing state
    active_sh_idx=-1; active_sh_price=None; prev_sh_price=None; active_sh_crossed=False
    active_sl_idx=-1; active_sl_price=None; prev_sl_price=None; active_sl_crossed=False
    trend = NEUTRAL
    eqh_active = False; eql_active = False

    if SWING_TF == "h4":
        h4 = resample_h4(df.iloc[max(0, n-300):n])
        if len(h4) >= 6:
            h4_h = h4['high'].values; h4_l = h4['low'].values
            h4_times = h4['time'].values
            h4_sh, h4_sl = get_swing_points(h4_h, h4_l, FRACTAL_STR)
            h4_si = np.where(h4_sh)[0]; h4_sli = np.where(h4_sl)[0]
            if len(h4_si) >= 2:
                prev_sh_price = h4_h[h4_si[-2]]
                active_sh_price = h4_h[h4_si[-1]]
                h4_t = pd.Timestamp(h4_times[h4_si[-1]])
                active_sh_idx = (df['timestamp'] >= h4_t).idxmax() if h4_t <= ts[-1] else n - 1
                atr_h4 = h4['close'].values.ptp() / max(len(h4), 1)
                if abs(active_sh_price - prev_sh_price) < EQH_THRESHOLD * atr_h4:
                    eqh_active = True
            if len(h4_sli) >= 2:
                prev_sl_price = h4_l[h4_sli[-2]]
                active_sl_price = h4_l[h4_sli[-1]]
                h4_t = pd.Timestamp(h4_times[h4_sli[-1]])
                active_sl_idx = (df['timestamp'] >= h4_t).idxmax() if h4_t <= ts[-1] else n - 1
                atr_h4 = h4['close'].values.ptp() / max(len(h4), 1)
                if abs(active_sl_price - prev_sl_price) < EQH_THRESHOLD * atr_h4:
                    eql_active = True
        _is_a, _dom_s, _tf_bias, _rvs = get_trend(SYMBOL, TF.upper(), use_d1=USE_D1)
        is_acc, h4_bias = _is_a, BULLISH if _dom_s == "BULLISH" else BEARISH if _dom_s == "BEARISH" else NEUTRAL
        h4_bias = h4_bias or NEUTRAL
        tf_bias = BULLISH if _tf_bias == "BULLISH" else BEARISH if _tf_bias == "BEARISH" else NEUTRAL
    else:
        for i in range(FRACTAL_STR, n - FRACTAL_STR):
            if sh[i]:
                prev_sh_price = active_sh_price
                active_sh_price = highs[i]; active_sh_idx = i; active_sh_crossed = False
                if prev_sh_price and abs(active_sh_price - prev_sh_price) < EQH_THRESHOLD * atr200[i]:
                    eqh_active = True
            if sl[i]:
                prev_sl_price = active_sl_price
                active_sl_price = lows[i]; active_sl_idx = i; active_sl_crossed = False
                if prev_sl_price and abs(active_sl_price - prev_sl_price) < EQH_THRESHOLD * atr200[i]:
                    eql_active = True

        _is_a, _dom_s, _tf_bias, _rvs = get_trend(SYMBOL, TF.upper(), use_d1=USE_D1)
        is_acc, h4_bias = _is_a, BULLISH if _dom_s == "BULLISH" else BEARISH if _dom_s == "BEARISH" else NEUTRAL
        tf_bias = BULLISH if _tf_bias == "BULLISH" else BEARISH if _tf_bias == "BEARISH" else NEUTRAL

    trend = h4_bias
    _sw_key = (active_sh_price, active_sl_price, h4_bias)
    if not hasattr(scan_for_setups, '_last_sw_key') or scan_for_setups._last_sw_key != _sw_key:
        scan_for_setups._last_sw_key = _sw_key
        sh_s = f"R={active_sh_price:.1f}" if active_sh_price else "R=None"
        sl_s = f"S={active_sl_price:.1f}" if active_sl_price else "S=None"
        print(f"  SWING {sh_s} {sl_s} bias={h4_bias} TF-bias={tf_bias}")
        logging.info(f"SWING {sh_s} {sl_s} bias={h4_bias} TF-bias={tf_bias}")

    # Pre-compute H4 swing levels for --require-sweep
    h4_swing_high = None
    h4_swing_low = None
    if REQUIRE_SWEEP:
        h4_s = resample_h4(df.iloc[max(0, n-300):n])
        if len(h4_s) >= 6:
            h4_h = h4_s['high'].values; h4_l = h4_s['low'].values
            h4_sh, h4_sl = get_swing_points(h4_h, h4_l, FRACTAL_STR)
            h4_si = np.where(h4_sh)[0]; h4_sli = np.where(h4_sl)[0]
            if len(h4_si) > 0: h4_swing_high = h4_h[h4_si[-1]]
            if len(h4_sli) > 0: h4_swing_low = h4_l[h4_sli[-1]]

    prem, eq, disc = get_premium_discount_zones(active_sh_price, active_sl_price)

    # Check EQH/EQL liquidity entries first (direct zone check, no MAX_WAIT)
    if eqh_active and prem is not None and ENTRY_MODE in ("all", "liquidity", "pd"):
        for i in range(n - 5, n):
            mid = (highs[i] + lows[i]) / 2
            sigma = atr_arr[i] if not np.isnan(atr_arr[i]) else 1
            if is_price_in_zone(mid, prem[0], prem[1], sigma, 0.25):
                if is_acc or h4_bias == BULLISH or tf_bias == BULLISH or tf_bias == NEUTRAL: continue
                if RVS_THRESHOLD > 0 and _rvs is not None and _rvs > RVS_THRESHOLD: continue
                if USE_PVT and pvt_scan is not None:
                    if not pvt_divergence(highs, lows, pvt_scan, i, "SELL"):
                        continue
                if USE_ST and st_scan is not None and st_scan[i] != -1:
                    continue
                if USE_OF and 'delta' in df.columns and not orderflow_ok(df, i, "SELL", OF_LOOKBACK):
                    continue
                if REQUIRE_SWEEP and (h4_swing_high is None or max(highs[max(0,i-2):i+1]) <= h4_swing_high):
                    continue
                all_setups.append({
                    "direction": "SELL", "entry_price": mid,
                    "sig_source": "LIQUIDITY", "break_type": "EQH",
                    "zone_source": "Premium", "zone_low": prem[0], "zone_high": prem[1],
                    "atr": sigma, "signal_ts": df.iloc[i]['timestamp'],
                    "st_trend": int(st_scan[i]) if st_scan is not None else None,
                    "delta": float(df['delta'].iloc[i]) if 'delta' in df.columns else 0,
                    "cum_delta": float(df['cum_delta'].iloc[i]) if 'cum_delta' in df.columns else 0,
                    "avg_delta": float(df['delta'].iloc[max(0,i-OF_LOOKBACK+1):i+1].mean()) if 'delta' in df.columns else 0,
                    "is_acc": is_acc, "dom_bias": h4_bias,
                })
                break

    if eql_active and disc is not None and ENTRY_MODE in ("all", "liquidity", "pd"):
        for i in range(n - 5, n):
            mid = (highs[i] + lows[i]) / 2
            sigma = atr_arr[i] if not np.isnan(atr_arr[i]) else 1
            if is_price_in_zone(mid, disc[0], disc[1], sigma, 0.25):
                if is_acc or h4_bias == BEARISH or tf_bias == BEARISH or tf_bias == NEUTRAL: continue
                if RVS_THRESHOLD > 0 and _rvs is not None and _rvs > RVS_THRESHOLD: continue
                if USE_PVT and pvt_scan is not None:
                    if not pvt_divergence(highs, lows, pvt_scan, i, "BUY"):
                        continue
                if USE_ST and st_scan is not None and st_scan[i] != 1:
                    continue
                if USE_OF and 'delta' in df.columns and not orderflow_ok(df, i, "BUY", OF_LOOKBACK):
                    continue
                if REQUIRE_SWEEP and (h4_swing_low is None or min(lows[max(0,i-2):i+1]) >= h4_swing_low):
                    continue
                all_setups.append({
                    "direction": "BUY", "entry_price": mid,
                    "sig_source": "LIQUIDITY", "break_type": "EQL",
                    "zone_source": "Discount", "zone_low": disc[0], "zone_high": disc[1],
                    "atr": sigma, "signal_ts": df.iloc[i]['timestamp'],
                    "st_trend": int(st_scan[i]) if st_scan is not None else None,
                    "delta": float(df['delta'].iloc[i]) if 'delta' in df.columns else 0,
                    "cum_delta": float(df['cum_delta'].iloc[i]) if 'cum_delta' in df.columns else 0,
                    "avg_delta": float(df['delta'].iloc[max(0,i-OF_LOOKBACK+1):i+1].mean()) if 'delta' in df.columns else 0,
                    "is_acc": is_acc, "dom_bias": h4_bias,
                })
                break

    # Check CHoCH/BOS → FVG/OB entries
    for i in range(max(FRACTAL_STR+1, n-N), n-1):
        if np.isnan(atr_arr[i]) or atr_arr[i] <= 0: continue
        hour_utc = pd.Timestamp(ts[i]).hour
        if not session_ok(hour_utc): continue
        if (n - 1) - i > MAX_SIGNAL_AGE: continue

        # Bullish break
        if active_sh_price and closes[i] > active_sh_price and (SWING_TF == "h4" or i - active_sh_idx < LOOKBACK):
            is_cho_ch = (trend == BEARISH) or (tf_bias == BEARISH)
            if is_cho_ch or is_acc or tf_bias == NEUTRAL: continue
            if RVS_THRESHOLD > 0 and _rvs is not None and _rvs > RVS_THRESHOLD: continue
            zones = []
            if ENTRY_MODE in ("all", "structure", "fvg"):
                fvg_l, fvg_h = find_fvg(df, i, "BUY")
                if fvg_l is not None: zones.append((fvg_l, fvg_h, "FVG"))
            if ENTRY_MODE in ("all", "structure", "ob"):
                ob_l, ob_h, _ = find_order_block(highs, lows, active_sh_idx, i, "BUY")
                if ob_l is not None: zones.append((ob_l, ob_h, "OB"))
            if not zones: continue
            entry_found = False
            for j in range(i+1, min(i+MAX_WAIT+1, n)):
                if entry_found: break
                mid = (highs[j] + lows[j]) / 2
                sigma = atr_arr[i]
                for zl, zh, zs in zones:
                    if is_price_in_zone(mid, zl, zh, sigma, 0.25):
                        if zs == "OB" and OB_SWEEP and active_sl_price is not None:
                            swept_ok = any(lows[k] < active_sl_price and closes[k] > active_sl_price for k in range(max(0,j-2),j+1))
                            if not swept_ok: continue
                        if USE_PVT and pvt_scan is not None:
                            if not pvt_divergence(highs, lows, pvt_scan, i, "BUY"):
                                continue
                        if USE_ST and st_scan is not None and st_scan[j] != 1:
                            continue
                        if USE_OF and 'delta' in df.columns and not orderflow_ok(df, j, "BUY", OF_LOOKBACK):
                            continue
                        if REQUIRE_SWEEP and (h4_swing_low is None or min(lows[max(0,j-2):j+1]) >= h4_swing_low):
                            continue
                        all_setups.append({
                            "direction": "BUY", "entry_price": mid,
                            "sig_source": "STRUCTURE", "break_type": "CHoCH" if is_cho_ch else "BOS",
                            "swing_price": active_sh_price, "zone_source": zs,
                            "zone_low": zl, "zone_high": zh, "atr": sigma,
                            "signal_ts": df.iloc[j]['timestamp'],
                            "st_trend": int(st_scan[j]) if st_scan is not None else None,
                            "delta": float(df['delta'].iloc[j]) if 'delta' in df.columns else 0,
                            "cum_delta": float(df['cum_delta'].iloc[j]) if 'cum_delta' in df.columns else 0,
                            "avg_delta": float(df['delta'].iloc[max(0,j-OF_LOOKBACK+1):j+1].mean()) if 'delta' in df.columns else 0,
                            "is_acc": is_acc, "dom_bias": h4_bias,
                        })
                        entry_found = True
                        break

        # Bearish break
        if active_sl_price and closes[i] < active_sl_price and (SWING_TF == "h4" or i - active_sl_idx < LOOKBACK):
            is_cho_ch = (trend == BULLISH) or (tf_bias == BULLISH)
            if is_cho_ch or is_acc or tf_bias == NEUTRAL: continue
            if RVS_THRESHOLD > 0 and _rvs is not None and _rvs > RVS_THRESHOLD: continue
            zones = []
            if ENTRY_MODE in ("all", "structure", "fvg"):
                fvg_l, fvg_h = find_fvg(df, i, "SELL")
                if fvg_l is not None: zones.append((fvg_l, fvg_h, "FVG"))
            if ENTRY_MODE in ("all", "structure", "ob"):
                ob_l, ob_h, _ = find_order_block(highs, lows, active_sl_idx, i, "SELL")
                if ob_l is not None: zones.append((ob_l, ob_h, "OB"))
            if not zones: continue
            entry_found = False
            for j in range(i+1, min(i+MAX_WAIT+1, n)):
                if entry_found: break
                mid = (highs[j] + lows[j]) / 2
                sigma = atr_arr[i]
                for zl, zh, zs in zones:
                    if is_price_in_zone(mid, zl, zh, sigma, 0.25):
                        if zs == "OB" and OB_SWEEP and active_sh_price is not None:
                            swept_ok = any(highs[k] > active_sh_price and closes[k] < active_sh_price for k in range(max(0,j-2),j+1))
                            if not swept_ok: continue
                        if USE_PVT and pvt_scan is not None:
                            if not pvt_divergence(highs, lows, pvt_scan, i, "SELL"):
                                continue
                        if USE_ST and st_scan is not None and st_scan[j] != -1:
                            continue
                        if USE_OF and 'delta' in df.columns and not orderflow_ok(df, j, "SELL", OF_LOOKBACK):
                            continue
                        if REQUIRE_SWEEP and (h4_swing_high is None or max(highs[max(0,j-2):j+1]) <= h4_swing_high):
                            continue
                        all_setups.append({
                            "direction": "SELL", "entry_price": mid,
                            "sig_source": "STRUCTURE", "break_type": "CHoCH" if is_cho_ch else "BOS",
                            "swing_price": active_sl_price, "zone_source": zs,
                            "zone_low": zl, "zone_high": zh, "atr": sigma,
                            "signal_ts": df.iloc[j]['timestamp'],
                            "st_trend": int(st_scan[j]) if st_scan is not None else None,
                            "delta": float(df['delta'].iloc[j]) if 'delta' in df.columns else 0,
                            "cum_delta": float(df['cum_delta'].iloc[j]) if 'cum_delta' in df.columns else 0,
                            "avg_delta": float(df['delta'].iloc[max(0,j-OF_LOOKBACK+1):j+1].mean()) if 'delta' in df.columns else 0,
                            "is_acc": is_acc, "dom_bias": h4_bias,
                        })
                        entry_found = True
                        break
    if not all_setups:
        return None
    if return_all:
        return all_setups
    return all_setups[0]

def _log_entry(setup, result):
    _bias_map = {1: "BULLISH", -1: "BEARISH", 0: "NEUTRAL"}
    dom = setup.get('dom_bias')
    dom_s = _bias_map.get(dom, str(dom or '?'))
    parts = [
        f"dir={setup['direction']}",
        f"src={setup.get('sig_source','?')}",
        f"bk={setup.get('break_type','?')}",
        f"zone={setup.get('zone_source','?')}",
        f"entry={setup.get('entry_price','?'):.2f}",
        f"zl={setup.get('zone_low','?'):.2f}",
        f"zh={setup.get('zone_high','?'):.2f}",
        f"atr={setup.get('atr',0):.2f}",
        f"trend={dom_s}",
        f"acc={setup.get('is_acc','?')}",
        f"retcode={result.retcode}",
    ]
    if 'ml_prob' in setup:
        parts.append(f"ml={setup['ml_prob']:.2%}")
    if 'delta' in setup:
        parts.append(f"delta={setup['delta']:.0f}")
    return " | ".join(parts)

def run_live():
    print(f"\n{'='*60}")
    st_s = " ST" if USE_ST else ""
    pvt_s = " PVT" if USE_PVT else ""
    of_s = " OF" if USE_OF else ""
    ml_s = f" ML@{ML_THRESHOLD}" if ML_THRESHOLD > 0 else ""
    atr_s = f" SL={SL_ATR:.1f}ATR TP={TP_ATR:.1f}ATR" if SL_ATR > 0 else ""
    print(f"  SMC STRATEGY LIVE ({TF}){st_s}{pvt_s}{of_s}{ml_s}{atr_s}")
    print(f"  {SYMBOL} --- Polling every {INTERVAL}s")
    if IS_TRADE: print("  Trade mode: ACTIVE")
    else: print("  Monitor only (add --trade)")
    print(f"{'='*60}")
    import MetaTrader5 as mt5
    if not mt5.initialize(): print("  ERROR: MT5 init failed"); return
    acc, pwd, srv = int(os.environ["MT5_ACCOUNT"]), os.environ["MT5_PASSWORD"], "PepperstoneBS-MT5-Live01"
    if not mt5.login(acc, pwd, srv):
        print("  Login failed, trying demo...")
        if not mt5.login(int(os.environ["MT5_ACCOUNT"]), os.environ["MT5_PASSWORD"], "PepperstoneBS-MT5-Live01"):
            print("  ERROR: Login failed"); mt5.shutdown(); return
    print(f"  Logged in: {acc}")
    start_msg = f"STARTED id={INSTANCE_ID} symbol={SYMBOL} tf={TF} entry={ENTRY_MODE} risk={RISK_USD}"
    extra = ""
    if USE_ST: extra += " ST"
    if USE_PVT: extra += " PVT"
    if USE_OF: extra += f" OF({OF_MODE},{OF_LOOKBACK})"
    if ML_THRESHOLD > 0: extra += f" ML@{ML_THRESHOLD}"
    if SL_ATR > 0: extra += f" SL={SL_ATR}ATR TP={TP_ATR}ATR"
    if MARKET_ONLY: extra += " MARKET_ONLY"
    if OB_SWEEP: extra += " OB_SWEEP"
    if REQUIRE_SWEEP: extra += " REQ_SWEEP"
    start_msg += extra
    print(f"  {start_msg}")
    logging.info(start_msg)
    mt5.symbol_select(SYMBOL, True)
    live_ml_model = None
    if ML_THRESHOLD > 0 and os.path.exists(ML_PATH):
        with open(ML_PATH, "rb") as f:
            artifact = pickle.load(f)
        live_ml_model = artifact["model"]
        if live_ml_model.n_features_in_ != 25:
            print(f"  ERROR: model has {live_ml_model.n_features_in_} features, expected 22. Retrain.")
            logging.warning(f"  ERROR: model has {live_ml_model.n_features_in_} features, expected 22. Retrain.")
            live_ml_model = None
        if live_ml_model:
            print(f"  ML filter: loaded {ML_PATH}, threshold={ML_THRESHOLD:.2f}")
            logging.info(f"  ML filter: loaded {ML_PATH}, threshold={ML_THRESHOLD:.2f}")
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
            if USE_OF:
                df = compute_orderflow_live(df, mt5, tf_mt5)
            setup = scan_for_setups(df)

            if SHOW_ZONES:
                all_zones = scan_for_setups(df, return_all=True)
                if all_zones:
                    print(f"\n  Zones in range ({len(all_zones)}):")
                    for z in all_zones:
                        z_dir = z['direction']; z_zs = z['zone_source']
                        z_zl = z['zone_low']; z_zh = z['zone_high']
                        z_src = z.get('sig_source', '')
                        z_d = f"delta={z['delta']:+.0f}" if 'delta' in z else ""
                        print(f"    {z_dir:4s} {z_zs:6s}  [{z_zl:.2f}-{z_zh:.2f}]  "
                              f"{z_src}  {z_d}")
                else:
                    print(f"  No active zones")

            if setup and live_ml_model is not None:
                o_a = df['open'].values.astype(float)
                h_a = df['high'].values.astype(float)
                l_a = df['low'].values.astype(float)
                c_a = df['close'].values.astype(float)
                body_a = abs(c_a - o_a)
                atr_a = df['atr'].values.astype(float)
                vol_a = df['tick_volume'].values.astype(float)
                avg_vol_a = df['avg_vol'].values.astype(float)
                ts_a = df['timestamp'].values
                ej = len(df) - 1
                # Find origin index: for structure, use last break_idx; for liquidity, use current
                origin = max(0, len(df) - 10)
                _is_a = setup.get('is_acc', False); _dom_b = setup.get('dom_bias', NEUTRAL)
                feat = extract_ml_features_smc(o_a, h_a, l_a, c_a, body_a, atr_a[ej], vol_a, avg_vol_a, atr_a, ts_a,
                    origin, ej, setup['direction'], setup['zone_source'], setup['sig_source'],
                    RISK_USD * USD1_TO_PRICE.get(SYMBOL, 1), _is_a, _dom_b)
                prob = live_ml_model.predict_proba(feat[:live_ml_model.n_features_in_].reshape(1, -1))[0, 1]
                setup['ml_prob'] = prob
                if prob < ML_THRESHOLD:
                    logging.info(f"********** Setup {setup['direction']} is skipped due to probability < threshold: {prob} < {ML_THRESHOLD} ***************")
                    continue
            ## --- Re-validate pending orders (disabled) ---
            #if IS_TRADE:
            #    pending = mt5.orders_get(symbol=SYMBOL)
            #    if pending:
            #        for o in pending:
            #            if "SMC" not in (o.comment or ""): continue
            #            keep = False
            #            if setup:
            #                zp = setup.get('zone_low' if setup['direction'] == "BUY" else 'zone_high')
            #                same_dir = (o.type == mt5.ORDER_TYPE_BUY_LIMIT and setup['direction'] == "BUY") or \
            #                           (o.type == mt5.ORDER_TYPE_SELL_LIMIT and setup['direction'] == "SELL")
            #                if same_dir: keep = True
            #            #if not keep:
            #                #mt5.order_send({"action": mt5.TRADE_ACTION_REMOVE, "order": o.ticket})
            #                #print(f"  Cancelled stale order #{o.ticket}")
            #                #logging.info(f"Cancelled order #{o.ticket}")
            # --- Place new orders ---
            if setup:
                limit_p = setup.get('zone_low' if setup['direction'] == "BUY" else 'zone_high', setup['entry_price'])
                key = f"{setup['direction']}_{setup['sig_source']}_{limit_p:.1f}"
                if key != last_sig_key or LOGGED == 0:
                    print_setup(setup); LOGGED += 1; last_sig_key = key
                    if IS_TRADE:
                        n_pos = count_positions(mt5, ["SMC"])
                        if n_pos < MAX_ORDERS:
                            result = place_mt5_trade(mt5, setup)
                            if result:
                                print(f"  >>> ORDER: retcode={result.retcode}")
                                logging.info(_log_entry(setup, result))
                        else: print(f"  Max orders ({MAX_ORDERS}) reached")
        except KeyboardInterrupt: print("\nStopped"); mt5.shutdown(); break
        except Exception as e: print(f"  Error: {e}")
        time.sleep(INTERVAL)
    mt5.shutdown()

if __name__ == "__main__":
    if IS_BACKTEST or TRAIN_ML: run_backtest()
    elif IS_WATCH: run_live()
    else: print("Use --backtest or --watch\n  python smc_strategy.py XAUUSD --backtest")
