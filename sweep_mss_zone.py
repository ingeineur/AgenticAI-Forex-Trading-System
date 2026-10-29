"""
Sweep -> Structure Shift -> Zone Entry Strategy
Price action + liquidity based, 90%+ WR target, low frequency.

Usage:
   python sweep_mss_zone.py XAUUSD --backtest                    # CSV backtest (M15)
   python sweep_mss_zone.py XAUUSD --backtest --tf h1            # H1 backtest
   python sweep_mss_zone.py XAUUSD --train-ml                    # Train ML model on backtest
   python sweep_mss_zone.py XAUUSD --backtest --ml-threshold 0.6 # Filter with trained model
   python sweep_mss_zone.py XAUUSD --watch                       # MT5 live monitor
   python sweep_mss_zone.py XAUUSD --watch --trade               # MT5 auto-execute
   python sweep_mss_zone.py BTCUSD --backtest                    # BTCUSD backtest
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

SMALL_VAL = -1e-9
eps = 1e-9
D1_LOOKBACK = 30  # ~30 days of daily bars for D1 level context

# ---------- ARGS ----------
parser = argparse.ArgumentParser(description="Sweep -> MSS -> Zone Entry Strategy")
parser.add_argument("symbol", default="XAUUSD", nargs="?", help="Symbol (XAUUSD, BTCUSD, etc)")
parser.add_argument("--bars", type=int, default=5000, help="Bars to fetch/load")
parser.add_argument("--risk", type=float, default=10, help="Risk per trade in USD")
parser.add_argument("--max-orders", type=int, default=2, help="Max concurrent positions")
parser.add_argument("--fractal-strength", type=int, default=2, help="Fractal strength for swing detection")
parser.add_argument("--lookback", type=int, default=50, help="Lookback bars for S/R")
parser.add_argument("--max-wait", type=int, default=10, help="Max bars to wait for zone entry")
parser.add_argument("--watch", action="store_true", help="Live monitoring mode")
parser.add_argument("--trade", action="store_true", help="Auto-execute with --watch")
parser.add_argument("--backtest", action="store_true", help="CSV backtest mode")
parser.add_argument("--session", choices=["london", "ny", "any"], default="any", help="Session filter")
parser.add_argument("--volume-filter", action="store_true", help="Enable volume filter (1.2x avg)")
parser.add_argument("--tp-mult", type=float, default=0.5, help="TP as multiple of 1R (ignored if --tp-atr set)")
parser.add_argument("--sl-atr", type=float, default=0.0, help="SL as ATR multiple (0 = use fixed USD risk)")
parser.add_argument("--tp-atr", type=float, default=0.0, help="TP as ATR multiple (0 = use --tp-mult)")
parser.add_argument("--interval", type=int, default=20, help="Poll interval in seconds")
parser.add_argument("--tf", choices=["m15", "h1"], default="m15", help="Timeframe for backtest/live")
parser.add_argument("--train-ml", action="store_true", help="Train Random Forest model from backtest trades")
parser.add_argument("--ml-path", type=str, default=None, help="Path to load/save ML model")
parser.add_argument("--ml-threshold", type=float, default=0.0, help="Min ML probability to take trade (0.0 = disabled)")
parser.add_argument("--supertrend", action="store_true", help="Enable SuperTrend confluence filter")
parser.add_argument("--st-atr", type=int, default=10, help="SuperTrend ATR period")
parser.add_argument("--st-mult", type=float, default=3.0, help="SuperTrend ATR multiplier")
parser.add_argument("--pvt-filter", action="store_true", help="Enable PVT divergence filter")
parser.add_argument("--orderflow", action="store_true", help="Enable orderflow delta filter")
parser.add_argument("--of-mode", choices=["ohlc", "tickrule"], default="ohlc", help="Orderflow method: ohlc=OHLC approx, tickrule=bid tick-rule (live, experimental)")
parser.add_argument("--of-lookback", type=int, default=-1, help="Bars to average delta for orderflow filter (default: 5 H1 / 10 M15)")
parser.add_argument("--market-only", action="store_true", help="Skip limit orders, only enter via market when price is near zone")
parser.add_argument("--max-signal-age", type=int, default=48, help="Max bars since sweep (48 = H1 2 days / M15 12 hours)")
parser.add_argument("--show-zones", action="store_true", help="Print all potential zones within max signal age")
parser.add_argument("--ob-sweep", action="store_true", help="Require sweep-and-reject of swing level before OB zone entry")
parser.add_argument("--skip-swept", action="store_true", help="Disable Swept zone entries (only FVG + OB)")
parser.add_argument("--instance-id", type=str, default="MSS", help="Unique instance ID for trade comments and PnL tracking")
parser.add_argument("--volume", type=float, default=0.01, help="Order volume / lot size")
parser.add_argument("--d1", action="store_true", help="Include D1 timeframe in trend scoring")
parser.add_argument("--rvs-threshold", type=int, default=0, help="Skip trades when reversal probability exceeds this (0=disabled)")
parser.add_argument("--label-bars", type=int, default=50, help="Bars ahead to check TP touch for the ML label")
args = parser.parse_args()

SYMBOL = args.symbol.upper()
TF = args.tf
BARS = args.bars
RISK_USD = args.risk
MAX_ORDERS = args.max_orders
FRACTAL_STR = args.fractal_strength
LOOKBACK = args.lookback
MAX_WAIT = args.max_wait
USE_VOLUME = args.volume_filter
TP_MULT = args.tp_mult
SL_ATR = args.sl_atr
TP_ATR = args.tp_atr if args.tp_atr > 0 else args.sl_atr  # default tp_atr = sl_atr if not set
TRAIN_ML = args.train_ml
ML_PATH = args.ml_path or f"sweep_ml_{SYMBOL}_{TF}.pkl"
USE_ST = args.supertrend
ST_ATR = args.st_atr
ST_MULT = args.st_mult
ML_THRESHOLD = args.ml_threshold
USE_PVT = args.pvt_filter
USE_OF = args.orderflow
OF_MODE = args.of_mode
OF_LOOKBACK = args.of_lookback if args.of_lookback > 0 else (10 if TF == "m15" else 5)
MARKET_ONLY = args.market_only
MAX_SIGNAL_AGE = args.max_signal_age
SHOW_ZONES = args.show_zones
OB_SWEEP = args.ob_sweep
SKIP_SWEPT = args.skip_swept
INSTANCE_ID = args.instance_id
VOLUME = args.volume
USE_D1 = args.d1
RVS_THRESHOLD = args.rvs_threshold
LABEL_BARS = args.label_bars
SESSION = args.session
INTERVAL = args.interval
IS_BACKTEST = args.backtest
IS_WATCH = args.watch
IS_TRADE = args.trade

logging.basicConfig(
    filename=f"{SYMBOL}_{TF}_sweep_mss.log",
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
)

USD1_TO_PRICE = {"XAUUSD": 1, "BTCUSD": 100, "NAS100": 30}
CSV_MAP = {
    "XAUUSD": {"m15": "XAUUSD_M15_combined.csv", "h1": "XAUUSD_H1_combined.csv"},
    "BTCUSD": {"m15": "BTCUSD_15_combined.csv", "h1": "BTCUSD_H1_combined.csv"},
    "NAS100": {"m15": "NAS100_M15_combined.csv", "h1": "NAS100_H1_combined.csv"},
}

# ---------- DATA ----------
def add_indicators(df):
    df['body'] = abs(df['close'] - df['open'])
    df['range'] = df['high'] - df['low']
    atr = AverageTrueRange(df['high'], df['low'], df['close'], 14)
    df['atr'] = atr.average_true_range()
    df['ema50'] = df['close'].ewm(span=50).mean()
    df['ema200'] = df['close'].ewm(span=200).mean()
    # Rolling volume average
    df['avg_vol'] = df['tick_volume'].rolling(20).mean()
    return df

def compute_supertrend(h, l, c, period=10, multiplier=3.0):
    """Compute SuperTrend trend direction (1=bullish, -1=bearish) using Wilder/RMA ATR."""
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
    st_up = np.zeros(n)
    st_dn = np.zeros(n)
    st_up[0] = up[0]; st_dn[0] = dn[0]
    for i in range(1, n):
        if np.isnan(up[i]) or np.isnan(dn[i]):
            trend[i] = trend[i-1]; st_up[i] = st_up[i-1]; st_dn[i] = st_dn[i-1]; continue
        st_up[i] = max(up[i], st_up[i-1]) if c[i-1] > st_up[i-1] else up[i]
        st_dn[i] = min(dn[i], st_dn[i-1]) if c[i-1] < st_dn[i-1] else dn[i]
        if trend[i-1] == -1 and c[i] > st_dn[i-1]:
            trend[i] = 1
        elif trend[i-1] == 1 and c[i] < st_up[i-1]:
            trend[i] = -1
        else:
            trend[i] = trend[i-1]
    return trend, st_up, st_dn

def compute_pvt(c, vol):
    """Price-Volume Trend: cumulative sum of vol * (close change) / prev_close."""
    pvt = np.zeros(len(c), dtype=float)
    for i in range(1, len(c)):
        if abs(c[i-1]) > eps:
            pvt[i] = pvt[i-1] + vol[i] * (c[i] - c[i-1]) / c[i-1]
    return pvt

def pvt_divergence(h, l, pvt, idx, direction, lookback=30):
    """Check PVT divergence at sweep bar idx.
    SELL: bearish div = price new high, PVT lower than prior peak → reversal down.
    BUY:  bullish div  = price new low,  PVT higher than prior trough → reversal up."""
    start = max(0, idx - lookback)
    if direction == "SELL":
        window_h = h[start:idx+1]
        window_pvt = pvt[start:idx+1]
        peak_i = np.argmax(window_h)
        if len(window_h) - 1 - peak_i > 2:
            return False  # peak not at/near sweep
        pvt_here = window_pvt[-1]
        pvt_prior_max = np.max(window_pvt[:max(1, peak_i)])
        return pvt_here < pvt_prior_max * 0.999  # bearish div
    else:
        window_l = l[start:idx+1]
        window_pvt = pvt[start:idx+1]
        trough_i = np.argmin(window_l)
        if len(window_l) - 1 - trough_i > 2:
            return False
        pvt_here = window_pvt[-1]
        pvt_prior_min = np.min(window_pvt[:max(1, trough_i)])
        return pvt_here > pvt_prior_min * 1.001  # bullish div

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

def get_swing_points(highs, lows, strength=2):
    n = len(highs)
    swing_highs = np.zeros(n, dtype=bool)
    swing_lows = np.zeros(n, dtype=bool)
    for i in range(strength, n - strength):
        swing_highs[i] = all(highs[i] > highs[i-k] and highs[i] > highs[i+k] for k in range(1, strength+1))
        swing_lows[i] = all(lows[i] < lows[i-k] and lows[i] < lows[i+k] for k in range(1, strength+1))
    return swing_highs, swing_lows

def resample_tf(df, rule):
    d = df.copy()
    d['time'] = d['timestamp']
    return d.resample(rule, on='time').agg({
        'open': 'first', 'high': 'max', 'low': 'min',
        'close': 'last', 'tick_volume': 'sum',
    }).dropna().reset_index()

def resample_h4(m15):
    return resample_tf(m15, '4h')

def resample_d1(df):
    return resample_tf(df, '1D')

def classify_bias(h, l):
    sh, sl = get_swing_points(h, l, FRACTAL_STR)
    si = np.where(sh | sl)[0]
    if len(si) < 2:
        return "SIDEWAYS"
    i0, i1 = si[-2], si[-1]
    p0 = h[i0] if sh[i0] else l[i0]
    p1 = h[i1] if sh[i1] else l[i1]
    if p1 > p0: return "BULLISH"
    if p1 < p0: return "BEARISH"
    return "SIDEWAYS"

def atr_contracting_check(h, l, c, period=14):
    tr = np.maximum(h - l, np.maximum(np.abs(h - np.roll(c, 1)), np.abs(l - np.roll(c, 1))))
    tr[0] = h[0] - l[0]
    n = len(tr); atr_arr_local = np.zeros(n); atr_arr_local[0] = tr[0]
    for i in range(1, n): atr_arr_local[i] = (atr_arr_local[i-1] * (period - 1) + tr[i]) / period
    recent = atr_arr_local[-5:].mean()
    avg = atr_arr_local[-min(period, n):].mean()
    return recent < avg * 0.7

from trend import get_trend, trend_from_df, analyse_timeframe, compute_reversal_probability, analyse_tf_arrays, snapshot_at  # scoring-based trend detection


# ---------- LEVEL DETECTION ----------
def as_np_ts(val):
    """Convert to UTC numpy datetime64[s] for array comparison."""
    ts = pd.Timestamp(val)
    if ts.tz is None:
        ts = ts.tz_localize('UTC')
    return np.datetime64(ts.to_numpy(), 's')

def get_asian_range(df, date):
    """Return (high, low) of Asian session (00:00-06:00 UTC) for the given date."""
    ts_np = df['timestamp'].values.astype('datetime64[s]')
    day_start = as_np_ts(date).astype('datetime64[D]')
    day_end = day_start + np.timedelta64(6, 'h')
    idx = np.where((ts_np >= day_start) & (ts_np < day_end))[0]
    if len(idx) < 2:
        return None, None
    h = df['high'].values.astype(float)
    l = df['low'].values.astype(float)
    return float(h[idx].max()), float(l[idx].min())

def get_prev_day_high_low(df, bar_time):
    """Return (high, low) of previous full day."""
    ts_np = df['timestamp'].values.astype('datetime64[s]')
    cur_midnight = as_np_ts(bar_time).astype('datetime64[D]')
    day_start = cur_midnight - np.timedelta64(1, 'D')
    day_end = cur_midnight
    idx = np.where((ts_np >= day_start) & (ts_np < day_end))[0]
    if len(idx) < 5:
        return None, None
    h = df['high'].values.astype(float)
    l = df['low'].values.astype(float)
    return float(h[idx].max()), float(l[idx].min())

def get_tf_swing_levels(df, n_bars=30):
    """Get the most recent swing high and low on the current timeframe (fractal strength)."""
    sub = df.tail(n_bars)
    if len(sub) < FRACTAL_STR * 2 + 1:
        return None, None
    highs = sub['high'].values
    lows = sub['low'].values
    sh, sl = get_swing_points(highs, lows, FRACTAL_STR)
    swing_high = None
    swing_low = None
    for i in range(len(highs) - FRACTAL_STR - 1, FRACTAL_STR - 1, -1):
        if swing_high is None and sh[i]:
            swing_high = highs[i]
        if swing_low is None and sl[i]:
            swing_low = lows[i]
        if swing_high is not None and swing_low is not None:
            break
    return swing_low, swing_high

def get_h4_swing_levels(df, n_bars=30):
    """Return lists of 4H swing highs and lows (fractal strength) within the last n_bars 4H bars."""
    h4 = resample_h4(df).tail(n_bars)
    if len(h4) < FRACTAL_STR * 2 + 1:
        return [], []
    highs = h4['high'].values
    lows = h4['low'].values
    sh, sl = get_swing_points(highs, lows, FRACTAL_STR)
    swing_highs = [highs[i] for i in range(FRACTAL_STR, len(highs) - FRACTAL_STR) if sh[i]]
    swing_lows = [lows[i] for i in range(FRACTAL_STR, len(lows) - FRACTAL_STR) if sl[i]]
    return swing_lows, swing_highs

def get_bsl_levels(df, bar_ts):
    """Buy-side liquidity levels (where longs exit / shorts have stops)."""
    levels = []
    pdh, _ = get_prev_day_high_low(df, bar_ts)
    if pdh: levels.append(("PDH", pdh))
    _, swing_h = get_tf_swing_levels(df)
    if swing_h: levels.append(("Swing_H", swing_h))
    return levels

def get_ssl_levels(df, bar_ts):
    """Sell-side liquidity levels (where shorts exit / longs have stops)."""
    levels = []
    _, pdl = get_prev_day_high_low(df, bar_ts)
    if pdl: levels.append(("PDL", pdl))
    swing_l, _ = get_tf_swing_levels(df)
    if swing_l: levels.append(("Swing_L", swing_l))
    return levels

# ---------- SESSION FILTER ----------
LONDON_OPEN_START = 7   # 07:00 UTC
LONDON_OPEN_END = 10    # 10:00 UTC
NY_OPEN_START = 12       # 12:00 UTC
NY_OPEN_END = 15         # 15:00 UTC

def session_ok(hour_utc):
    if SESSION == "any":
        return True
    if SESSION == "london":
        return LONDON_OPEN_START <= hour_utc < LONDON_OPEN_END
    if SESSION == "ny":
        return NY_OPEN_START <= hour_utc < NY_OPEN_END
    return True

# ---------- SWEEP DETECTION ----------
def detect_sweep(df, i, level, direction):
    """Check if candle i swept through a level.
    direction = 'BUY' means price swept BELOW a support (SSL)
    direction = 'SELL' means price swept ABOVE a resistance (BSL)"""
    row = df.iloc[i]
    buffer = row['atr'] * 0.1
    if direction == "BUY":  # sweep below support
        return row['low'] < (level - buffer)
    else:  # sweep above resistance
        return row['high'] > (level + buffer)

def sweep_quality_ok(df, i, level, direction):
    """Check rejection quality: close back >50% of body inside the level."""
    row = df.iloc[i]
    body = row['body']
    if body <= eps:
        return True  # doji, let MSS decide
    if direction == "BUY":  # swept below, closed above -> bullish
        # How much of body is above the level?
        penetration = level - row['low']  # how far below level it went
        close_above_level = row['close'] - level
        if close_above_level <= 0:
            return False  # didn't close back above
        body_inside_ratio = close_above_level / body
        return body_inside_ratio > 0.5
    else:  # swept above, closed below -> bearish
        penetration = row['high'] - level
        close_below_level = level - row['close']
        if close_below_level <= 0:
            return False
        body_inside_ratio = close_below_level / body
        return body_inside_ratio > 0.5

# ---------- MSS DETECTION ----------
def find_last_swing(df, i, direction, lookback=10):
    """Find the most recent swing high (SELL) or low (BUY) before index i."""
    sub = df.iloc[max(0, i-lookback):i]
    if len(sub) < FRACTAL_STR * 2 + 1:
        return None, None
    highs = sub['high'].values
    lows = sub['low'].values
    sh, sl = get_swing_points(highs, lows, FRACTAL_STR)
    if direction == "BUY":
        # Need a swing low to break above
        idx = np.where(sl)[0]
        if len(idx) == 0:
            return None, None
        return "swing_low", float(sub['low'].iloc[idx[-1]])
    else:
        # Need a swing high to break below
        idx = np.where(sh)[0]
        if len(idx) == 0:
            return None, None
        return "swing_high", float(sub['high'].iloc[idx[-1]])

def detect_mss(df, i, sweep_dir, max_lookahead=3):
    """Detect Market Structure Shift within max_lookahead candles after the sweep.
    For sweep BUY (swept low): price must break above the most recent swing low.
    For sweep SELL (swept high): price must break below the most recent swing high."""
    lookahead_end = min(i + max_lookahead + 1, len(df))
    if lookahead_end <= i + 1:
        return False, None, None
    sweep_row = df.iloc[i]
    ahead = df.iloc[i+1:lookahead_end]
    if len(ahead) == 0:
        return False, None, None
    if sweep_dir == "BUY":
        # Find most recent swing low before sweep
        sw_type, sw_price = find_last_swing(df, i, "BUY")
        if sw_price is None:
            # Fallback: use the sweep candle high as reference
            sw_price = sweep_row['open']  # approximate
        # Check if any candle ahead breaks above sw_price
        for j in range(i+1, lookahead_end):
            if df.iloc[j]['high'] > sw_price:
                return True, f"MSS_BULL", sw_price
        return False, None, sw_price
    else:
        sw_type, sw_price = find_last_swing(df, i, "SELL")
        if sw_price is None:
            sw_price = sweep_row['open']
        for j in range(i+1, lookahead_end):
            if df.iloc[j]['low'] < sw_price:
                return True, f"MSS_BEAR", sw_price
        return False, None, sw_price

# ---------- FVG DETECTION ----------
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

# ---------- ORDER BLOCK DETECTION ----------
def find_order_block(df, i, direction, max_lookahead=5):
    """Find an Order Block from the MSS move.
    An OB is the last bullish candle before the impulsive move (for BUY)
    or last bearish candle before the impulsive down move (for SELL)."""
    for j in range(i+1, min(i+max_lookahead, len(df)-1)):
        c = df.iloc[j]
        if direction == "BUY":
            if c['close'] > c['open']:
                prev = df.iloc[j-1]
                zone_low = min(prev['close'], prev['open'])
                zone_high = max(prev['close'], prev['open'])
                return zone_low, zone_high, "OB"
        else:
            if c['close'] < c['open']:
                prev = df.iloc[j-1]
                zone_low = min(prev['close'], prev['open'])
                zone_high = max(prev['close'], prev['open'])
                return zone_low, zone_high, "OB"
    return None, None, None

# ---------- ZONE CHECK ----------
def is_price_in_zone(df, j, zone_low, zone_high, atr, pad=0.25):
    """Check if price at bar j is within zone ±0.25 ATR."""
    mid = (df.iloc[j]['high'] + df.iloc[j]['low']) / 2
    zl = zone_low - atr * pad
    zh = zone_high + atr * pad
    return zl <= mid <= zh, mid

# ---------- SIGNAL GENERATION ----------
COMMENT_PREFIX = "LIQ"

def count_positions(mt5, keywords=None):
    if mt5 is None:
        return 0
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

def scan_for_setups(df, return_all=False):
    """Scan the recent bars for any valid sweep + MSS + zone setup.
    When return_all=True, returns a list of all valid setups (for zone display)."""
    n = len(df)
    if n < 200:
        return None
    N = 50
    setups = []
    highs = df['high'].values.astype(float)
    lows = df['low'].values.astype(float)
    closes = df['close'].values.astype(float)
    # Pre-compute SuperTrend if enabled
    st_trend_scan = None
    if USE_ST:
        h_a = df['high'].values.astype(float)
        l_a = df['low'].values.astype(float)
        c_a = df['close'].values.astype(float)
        st_trend_scan, _, _ = compute_supertrend(h_a, l_a, c_a, ST_ATR, ST_MULT)
    # Pre-compute PVT if enabled
    pvt_scan = None
    if USE_PVT:
        c_a = df['close'].values.astype(float)
        v_a = df['tick_volume'].values.astype(float)
        pvt_scan = compute_pvt(c_a, v_a)
    # Compute accumulation state once (not per sweep candle)
    ACC_IS, ACC_DOM, ACC_TF, ACC_RVS = get_trend(SYMBOL, TF.upper(), use_d1=USE_D1)

    # Print swing levels for debugging (only on change)
    swl, swh = get_tf_swing_levels(df)
    _sw_key = (swh, swl, ACC_DOM)
    if not hasattr(scan_for_setups, '_last_sw_key') or scan_for_setups._last_sw_key != _sw_key:
        scan_for_setups._last_sw_key = _sw_key
        sh_s = f"R={swh:.1f}" if swh else "R=None"; sl_s = f"S={swl:.1f}" if swl else "S=None"
        print(f"  SWING {sh_s} {sl_s} trend={ACC_DOM} tf_trend={ACC_TF}")
        logging.info(f"SWING {sh_s} {sl_s} trend={ACC_DOM} tf_trend={ACC_TF}")
    #print(f"accumulating: {ACC_IS}   ------  trend: {ACC_DOM}")

    # Scan last N bars for sweep candles
    for i in range(n - N - 3, n - 2):
        row = df.iloc[i]
        if np.isnan(row['atr']) or row['atr'] <= 0:
            continue
        hour_utc = pd.Timestamp(row['timestamp']).hour
        if not session_ok(hour_utc):
            continue
        # Get all BSL and SSL levels
        bar_ts = row['timestamp']
        bsl_levels = get_bsl_levels(df, bar_ts)
        ssl_levels = get_ssl_levels(df, bar_ts)
        # Check for SELL setups (sweep BSL = above resistance)
        for lev_name, level in bsl_levels:
            if level is None or level <= 0:
                continue
            swept = detect_sweep(df, i, level, "SELL")
            if not swept and i >= 1:
                swept = detect_sweep(df, i-1, level, "SELL")
            if not swept:
                continue
            # Quality check: close back inside
            if not sweep_quality_ok(df, i if swept else i-1, level, "SELL"):
                continue
            sweep_idx = i if swept else i-1
            if (n - 1) - sweep_idx > MAX_SIGNAL_AGE: continue
            # Volume filter
            if USE_VOLUME:
                row_s = df.iloc[sweep_idx]
                if row_s['tick_volume'] < 1.2 * (row_s.get('avg_vol') or 1):
                    continue
            # MSS check
            mss_ok, mss_type, mss_price = detect_mss(df, sweep_idx, "SELL")
            if not mss_ok:
                continue
            # PVT divergence filter
            if USE_PVT and pvt_scan is not None:
                h_a = df['high'].values.astype(float)
                l_a = df['low'].values.astype(float)
                if not pvt_divergence(h_a, l_a, pvt_scan, sweep_idx, "SELL"):
                    continue
            # Accumulation check
            if ACC_IS or ACC_DOM == "BULLISH":
                continue
            if ACC_TF is None or ACC_TF == "BULLISH":
                continue
            if RVS_THRESHOLD > 0 and ACC_RVS is not None and ACC_RVS > RVS_THRESHOLD:
                continue
            # Zone = swept level retest
            zone_low = level - row['atr'] * 0.25
            zone_high = level + row['atr'] * 0.25
            # Also check FVG
            fvg_low, fvg_high = find_fvg(df, sweep_idx, "SELL")
            # OB
            ob_low, ob_high, ob_type = find_order_block(df, sweep_idx, "SELL")
            # Check if price is currently in any zone (FVG > OB > Swept)
            for j in range(n - 5, n):
                row_j = df.iloc[j]
                atr = row_j['atr'] if row_j['atr'] > 0 else row['atr']
                close_j = row_j['close']
                mid = (row_j['high'] + row_j['low']) / 2
                zl_pad = zone_low - atr * 0.25
                zh_pad = zone_high + atr * 0.25
                in_zone = False
                if fvg_low is not None and fvg_high is not None:
                    zl_fvg = fvg_low - atr * 0.25
                    zh_fvg = fvg_high + atr * 0.25
                    if zl_fvg <= mid <= zh_fvg:
                        in_zone = True; zone_source = "FVG"; zl_use, zh_use = fvg_low, fvg_high
                if not in_zone and ob_low is not None:
                    zl_ob = ob_low - atr * 0.25
                    zh_ob = ob_high + atr * 0.25
                    if zl_ob <= mid <= zh_ob:
                        if OB_SWEEP:
                            swept_ok = False
                            for k in range(max(0, j-2), j+1):
                                if highs[k] > level and closes[k] < level:
                                    swept_ok = True; break
                            if not swept_ok: continue
                        in_zone = True; zone_source = "OB"; zl_use, zh_use = ob_low, ob_high
                if not SKIP_SWEPT and not in_zone and zl_pad <= close_j <= zh_pad:
                    in_zone = True; zone_source = "Swept"; zl_use, zh_use = zone_low, zone_high
                    mid = close_j
                if in_zone and (not USE_ST or st_trend_scan is None or st_trend_scan[j] == -1):
                    if USE_OF and 'delta' in df.columns and not orderflow_ok(df, j, "SELL", OF_LOOKBACK):
                        continue
                    setup = {
                        "direction": "SELL",
                        "entry_price": mid,
                        "sweep_level": level,
                        "sweep_idx": sweep_idx,
                        "level_name": lev_name,
                        "mss_type": mss_type,
                        "zone_source": zone_source,
                        "zone_low": zl_use,
                        "zone_high": zh_use,
                        "atr": atr,
                        "signal_ts": df.iloc[j]['timestamp'],
                        "filters_passed": {
                            "session": True,
                            "volume": True if not USE_VOLUME else (row_s['tick_volume'] >= 1.2 * (row_s.get('avg_vol') or 1)),
                            "mss": True,
                            "zone": True,
                            "supertrend": USE_ST,
                            "orderflow": USE_OF,
                        },
                        "delta": float(df['delta'].iloc[j]) if 'delta' in df.columns else 0,
                        "cum_delta": float(df['cum_delta'].iloc[j]) if 'cum_delta' in df.columns else 0,
                        "avg_delta": float(df['delta'].iloc[max(0,j-OF_LOOKBACK+1):j+1].mean()) if 'delta' in df.columns else 0,
                        "is_acc": ACC_IS, "dom_bias": ACC_DOM,
                    }
                    setups.append(setup)
        # Check for BUY setups (sweep SSL = below support)
        for lev_name, level in ssl_levels:
            if level is None or level <= 0:
                continue
            swept = detect_sweep(df, i, level, "BUY")
            if not swept and i >= 1:
                swept = detect_sweep(df, i-1, level, "BUY")
            if not swept:
                continue
            if not sweep_quality_ok(df, i if swept else i-1, level, "BUY"):
                continue
            sweep_idx = i if swept else i-1
            if (n - 1) - sweep_idx > MAX_SIGNAL_AGE: continue
            if USE_VOLUME:
                row_s = df.iloc[sweep_idx]
                if row_s['tick_volume'] < 1.2 * (row_s.get('avg_vol') or 1):
                    continue
            mss_ok, mss_type, mss_price = detect_mss(df, sweep_idx, "BUY")
            if not mss_ok:
                continue
            # PVT divergence filter
            if USE_PVT and pvt_scan is not None:
                h_a = df['high'].values.astype(float)
                l_a = df['low'].values.astype(float)
                if not pvt_divergence(h_a, l_a, pvt_scan, sweep_idx, "BUY"):
                    continue
            # Accumulation check
            if ACC_IS or ACC_DOM == "BEARISH":
                continue
            if ACC_TF is None or ACC_TF == "BEARISH":
                continue
            if RVS_THRESHOLD > 0 and ACC_RVS is not None and ACC_RVS > RVS_THRESHOLD:
                continue
            zone_low = level - row['atr'] * 0.25
            zone_high = level + row['atr'] * 0.25
            fvg_low, fvg_high = find_fvg(df, sweep_idx, "BUY")
            ob_low, ob_high, ob_type = find_order_block(df, sweep_idx, "BUY")
            for j in range(n - 5, n):
                row_j = df.iloc[j]
                atr = row_j['atr'] if row_j['atr'] > 0 else row['atr']
                close_j = row_j['close']
                mid = (row_j['high'] + row_j['low']) / 2
                zl_pad = zone_low - atr * 0.25
                zh_pad = zone_high + atr * 0.25
                in_zone = False
                if fvg_low is not None and fvg_high is not None:
                    zl_fvg = fvg_low - atr * 0.25
                    zh_fvg = fvg_high + atr * 0.25
                    if zl_fvg <= mid <= zh_fvg:
                        in_zone = True; zone_source = "FVG"; zl_use, zh_use = fvg_low, fvg_high
                if not in_zone and ob_low is not None:
                    zl_ob = ob_low - atr * 0.25
                    zh_ob = ob_high + atr * 0.25
                    if zl_ob <= mid <= zh_ob:
                        if OB_SWEEP:
                            swept_ok = False
                            for k in range(max(0, j-2), j+1):
                                if lows[k] < level and closes[k] > level:
                                    swept_ok = True; break
                            if not swept_ok: continue
                        in_zone = True; zone_source = "OB"; zl_use, zh_use = ob_low, ob_high
                if not SKIP_SWEPT and not in_zone and zl_pad <= close_j <= zh_pad:
                    in_zone = True; zone_source = "Swept"; zl_use, zh_use = zone_low, zone_high
                    mid = close_j
                if in_zone and (not USE_ST or st_trend_scan is None or st_trend_scan[j] == 1):
                    if USE_OF and 'delta' in df.columns and not orderflow_ok(df, j, "BUY", OF_LOOKBACK):
                        continue
                    setup = {
                        "direction": "BUY",
                        "entry_price": mid,
                        "sweep_level": level,
                        "sweep_idx": sweep_idx,
                        "level_name": lev_name,
                        "mss_type": mss_type,
                        "zone_source": zone_source,
                        "zone_low": zl_use,
                        "zone_high": zh_use,
                        "atr": atr,
                        "signal_ts": df.iloc[j]['timestamp'],
                        "filters_passed": {"session": True, "volume": True, "mss": True, "zone": True, "supertrend": USE_ST, "orderflow": USE_OF},
                        "delta": float(df['delta'].iloc[j]) if 'delta' in df.columns else 0,
                        "cum_delta": float(df['cum_delta'].iloc[j]) if 'cum_delta' in df.columns else 0,
                        "avg_delta": float(df['delta'].iloc[max(0,j-OF_LOOKBACK+1):j+1].mean()) if 'delta' in df.columns else 0,
                        "is_acc": ACC_IS, "dom_bias": ACC_DOM,
                    }
                    setups.append(setup)
    # Deduplicate: keep the setup with the nearest zone match
    if not setups:
        return None
    # Pick the most recent best setup (prefer later sweep, better zone proximity)
    setups.sort(key=lambda s: (-s['sweep_idx'], abs(s['entry_price'] - s['sweep_level'])))
    if return_all:
        return setups
    return setups[0]

# ---------- RISK MANAGEMENT ----------
def place_mt5_trade(mt5, setup):
    direction = setup['direction']
    tick = mt5.symbol_info_tick(SYMBOL)
    info = mt5.symbol_info(SYMBOL)
    if tick is None or info is None:
        print("  ERROR: No symbol data"); return None

    zone_boundary = setup.get('zone_low' if direction == "BUY" else 'zone_high')
    entry_p = setup.get('entry_price')
    if zone_boundary is not None and zone_boundary > 0 and entry_p is not None:
        limit_price = (zone_boundary + entry_p) / 2
    else:
        limit_price = zone_boundary or entry_p
    atr_val = setup['atr']
    if SL_ATR > 0:
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
    entry_price = tick.ask if (direction == "BUY" and is_market) else (tick.bid if (direction == "SELL" and is_market) else limit_price)
    mode = "MARKET" if is_market else "LIMIT"
    print(f"  >>> {mode} {direction} at {entry_price:.2f} (zone: {limit_price:.2f}, sl_dist: {sl_dist:.4f})")
    if direction == "BUY":
        sl = entry_price - sl_dist
        tp = entry_price + tp_dist
    else:
        sl = entry_price + sl_dist
        tp = entry_price - tp_dist
    ts = info.trade_tick_size or 0.01
    sl = round(sl / ts) * ts
    tp = round(tp / ts) * ts
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
    comment = f"{INSTANCE_ID} {TF} MSS {setup['level_name']} {setup['zone_source']}"
    request = {
        "action": action,
        "symbol": SYMBOL,
        "volume": VOLUME,
        "type": order_type,
        "price": entry_price,
        "sl": sl,
        "tp": tp,
        "deviation": 20,
        "magic": 1015,
        "comment": comment,
        "type_time": mt5.ORDER_TIME_GTC,
        "type_filling": mt5.ORDER_FILLING_IOC,
    }
    result = mt5.order_send(request)
    return result

# ---------- PRINT SETUP ----------
def print_setup(setup):
    print(f"\n{'='*60}")
    print(f"  SWEEP -> MSS -> ZONE ENTRY SIGNAL")
    print(f"{'='*60}")
    print(f"  Symbol:    {SYMBOL}")
    print(f"  Direction: {setup['direction']}")
    print(f"  Time:      {setup['signal_ts']}")
    print(f"  Entry:     {setup['entry_price']:.2f}")
    print(f"  Swept:     {setup['level_name']} @ {setup['sweep_level']:.2f}")
    print(f"  Zone:      {setup['zone_source']} [{setup['zone_low']:.2f} - {setup['zone_high']:.2f}]")
    print(f"  MSS:       {setup['mss_type']}")
    print(f"  ATR:       {setup['atr']:.2f}")
    ep = setup['entry_price']
    atr_val = setup['atr']
    if SL_ATR > 0:
        sl_dist = atr_val * SL_ATR
        tp_dist = atr_val * TP_ATR
    else:
        sl_dist = RISK_USD * USD1_TO_PRICE.get(SYMBOL, 1)
        tp_dist = sl_dist * TP_MULT
    if setup['direction'] == "BUY":
        sl_p = ep - sl_dist; tp_p = ep + tp_dist
    else:
        sl_p = ep + sl_dist; tp_p = ep - tp_dist
    print(f"  SL:        {sl_p:.2f}   TP: {tp_p:.2f}")
    limit_p = setup.get('zone_low' if setup['direction'] == "BUY" else 'zone_high', ep)
    print(f"  Limit:     {limit_p:.2f}  (zone {'low' if setup['direction'] == 'BUY' else 'high'})")
    st = setup.get('st_trend')
    if st is not None:
        label = "BULLISH" if st == 1 else "BEARISH" if st == -1 else "NONE"
        print(f"  ST Trend:  {label}")
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
    print(f"  Filters:   {setup['filters_passed']}")
    if 'ml_prob' in setup:
        print(f"  ML Prob:   {setup['ml_prob']:.2%}")
    print(f"{'='*60}\n")

# ---------- ML FEATURES ----------
class MlContext:
    """Point-in-time multi-timeframe context for ML features (no look-ahead).

    Precomputes indicator arrays on the current TF and on resampled H1/H4/D1.
    Each current-TF bar is aligned to its latest *completed* HTF bar via
    searchsorted, so no future information is used.
    """
    def __init__(self, df, o, h, l, c, atr_arr, ts, tp_dist_arr=None):
        self.n = len(c)
        self.atr = atr_arr
        self.ts = np.asarray(ts).astype('datetime64[s]') if ts is not None else None
        self.cur = analyse_tf_arrays(o, h, l, c, FRACTAL_STR)
        self.htf = {}
        for name, rule in (("H1", "1h"), ("H4", "4h"), ("D1", "1D")):
            rs = resample_tf(df, rule)
            ro = rs['open'].values.astype(float)
            rh = rs['high'].values.astype(float)
            rl = rs['low'].values.astype(float)
            rc = rs['close'].values.astype(float)
            rt = rs['time'].values.astype('datetime64[s]')
            self.htf[name] = {
                "arrs": analyse_tf_arrays(ro, rh, rl, rc, FRACTAL_STR),
                "time": rt,
                "high": rh,
                "low": rl,
            }
        # D1 swing points for S/R features
        d1 = self.htf["D1"]
        self.d1_sh, self.d1_sl = get_swing_points(d1["high"], d1["low"], FRACTAL_STR)
        self.d1_time = d1["time"]
        self.d1_high = d1["high"]
        self.d1_low = d1["low"]

    def _completed_idx(self, name, i):
        """Index of last completed HTF bar (fully closed before current bar i)."""
        htf = self.htf[name]
        t = self.ts[i]
        return int(np.searchsorted(htf["time"], t, side='right')) - 2

    def snapshot(self, name, i):
        """AnalyseTimeframe-style dict for TF at current bar i (point-in-time)."""
        if name == TF.upper():
            return snapshot_at(self.cur, i)
        if name not in self.htf:
            return None
        j = self._completed_idx(name, i)
        if j < 0:
            return None
        return snapshot_at(self.htf[name]["arrs"], j)

    def bias(self, name, i):
        """+1 / 0 / -1 trend bias for a TF at current bar i."""
        sn = self.snapshot(name, i)
        if sn is None:
            return 0.0
        t = sn["trend"]
        return 1.0 if t == "BULLISH" else (-1.0 if t == "BEARISH" else 0.0)

    def reversal_probability(self, i):
        """True multi-TF reversal probability (0-1), point-in-time at bar i."""
        results = {}
        for name in ("M15", "H1", "H4", "D1"):
            sn = self.snapshot(name, i)
            if sn is not None:
                results[name] = sn
        if not results:
            return 0.5
        return float(compute_reversal_probability(results, TF.upper())) / 100.0

    def rsi_norm(self, i):
        return float(np.clip((self.cur["rsi"][i] - 50) / 50.0, -1.0, 1.0))

    def macd_hist_atr(self, i):
        mh = float(self.cur["macd_line"][i] - self.cur["sig_line"][i])
        return float(np.clip(mh / (self.atr[i] + eps), -2.0, 2.0))

    def d1_levels(self, i, tp_dist=0.0, ref_price=None):
        """Confirmed D1 swing distances relative to reference price (entry or close[i]).
        Returns (resistance_above, support_below, touch_count, block_buy, block_sell).
        Point-in-time: only swings confirmed before bar i (j + FRACTAL_STR < cur)."""
        t = self.ts[i]
        cur = int(np.searchsorted(self.d1_time, t, side='right')) - 2
        if cur < 0:
            return np.nan, np.nan, 0.0, 0.0, 0.0
        max_j = cur - FRACTAL_STR
        start_j = max(0, max_j - D1_LOOKBACK)
        c_i = ref_price if ref_price is not None else self._close[i]
        atr_i = self.atr[i] if not np.isnan(self.atr[i]) else 0.0
        res = sup = None
        touch = 0
        block_buy = block_sell = 0
        for j in range(max_j, start_j - 1, -1):
            if self.d1_sh[j]:
                if self.d1_high[j] > c_i:
                    d = self.d1_high[j] - c_i
                    if res is None or d < res:
                        res = d
                    if tp_dist > 0 and c_i < self.d1_high[j] <= c_i + tp_dist:
                        block_buy = 1
                    if atr_i > 0 and abs(self.d1_high[j] - c_i) <= atr_i:
                        touch += 1
            if self.d1_sl[j]:
                if self.d1_low[j] < c_i:
                    d = c_i - self.d1_low[j]
                    if sup is None or d < sup:
                        sup = d
                    if tp_dist > 0 and c_i - tp_dist <= self.d1_low[j] < c_i:
                        block_sell = 1
                    if atr_i > 0 and abs(self.d1_low[j] - c_i) <= atr_i:
                        touch += 1
        return (res if res is not None else np.nan,
                sup if sup is not None else np.nan,
                float(min(touch, 5.0)), float(block_buy), float(block_sell))


def extract_ml_features(o, h, l, c, body, atr, vol, avg_vol, atr_arr, ts_arr,
                        sweep_idx, entry_j, direction, zone_source, level_name,
                        one_r, ctx, entry_price=None, tp_dist=None):
    """Extract feature vector for a trade setup. Returns numpy array of 30 features.
    All higher-TF / trend / reversal features are point-in-time via ctx."""
    f = np.zeros(30, dtype=np.float32)
    # 0: ATR / close (normalized volatility)
    f[0] = atr / (c[entry_j] + eps)
    # 1: sweep volume ratio
    f[1] = vol[sweep_idx] / (avg_vol[sweep_idx] + eps)
    # 2: bars waited from sweep to entry
    f[2] = (entry_j - sweep_idx) / 10.0
    # 3: entry price distance from MA50
    ma50 = np.mean(c[max(0, entry_j-50):entry_j]) if entry_j >= 50 else c[entry_j]
    f[3] = (c[entry_j] - ma50) / (atr + eps)
    # 4: entry price distance from MA200
    ma200 = np.mean(c[max(0, entry_j-200):entry_j]) if entry_j >= 200 else c[entry_j]
    f[4] = (c[entry_j] - ma200) / (atr + eps)
    # 5: trend strength (ma50 - ma200) / close
    f[5] = (ma50 - ma200) / (c[entry_j] + eps)
    # 6: is_buy
    f[6] = 1.0 if direction == "BUY" else 0.0
    # 7-8: zone one-hot (Swept implied by all-zero)
    f[7] = 1.0 if zone_source == "FVG" else 0.0
    f[8] = 1.0 if zone_source == "OB" else 0.0
    # 9: level type one-hot (PD; Swing implied by all-zero)
    f[9] = 1.0 if "PD" in level_name else 0.0
    # 10: sweep candle body / atr
    f[10] = body[sweep_idx] / (atr + eps)
    # 11: sweep candle range / atr
    f[11] = (h[sweep_idx] - l[sweep_idx]) / (atr + eps)
    # 12: entry candle is bullish
    f[12] = 1.0 if c[entry_j] > o[entry_j] else 0.0
    # 13: entry hour (cosine)
    entry_ts = pd.Timestamp(ts_arr[entry_j])
    hour = entry_ts.hour if hasattr(entry_ts, 'hour') else 12
    f[13] = math.cos(2 * math.pi * hour / 24)
    # 14: entry hour (sine)
    f[14] = math.sin(2 * math.pi * hour / 24)
    # 15: atr trend (atr at entry vs 50 bars ago)
    atr50 = np.nanmean(atr_arr[max(0, entry_j-50):entry_j+1])
    f[15] = atr / (atr50 + eps)
    # 16: risk distance normalized
    f[16] = one_r / (atr + eps)
    # 17: entry volume ratio
    f[17] = vol[entry_j] / (avg_vol[entry_j] + eps)
    # 18: sweep to entry atr change
    f[18] = atr_arr[entry_j] / (atr_arr[sweep_idx] + eps)
    # 19: D1 trend bias (point-in-time)
    f[19] = ctx.bias("D1", entry_j)
    # 20: H4 trend bias (point-in-time)
    f[20] = ctx.bias("H4", entry_j)
    # 21: HTF alignment with trade direction (avg of D1/H4 agreement)
    dir_sign = 1.0 if direction == "BUY" else -1.0
    agree_d1 = 1.0 if dir_sign * f[19] > 0 else 0.0
    agree_h4 = 1.0 if dir_sign * f[20] > 0 else 0.0
    f[21] = (agree_d1 + agree_h4) / 2.0
    # 22: multi-TF reversal probability (point-in-time)
    f[22] = ctx.reversal_probability(entry_j)
    # 23-24: opposing / supporting D1 distance (ATR-normalized, capped 5)
    d1_res, d1_sup, touch_cnt, block_buy, block_sell = ctx.d1_levels(entry_j, tp_dist or 0.0, entry_price)
    if direction == "BUY":
        opp_d = d1_res; sup_d = d1_sup
    else:
        opp_d = d1_sup; sup_d = d1_res
    f[23] = min(opp_d / (atr + eps), 5.0) if not np.isnan(opp_d) else 5.0
    f[24] = min(sup_d / (atr + eps), 5.0) if not np.isnan(sup_d) else 5.0
    # 25: entry at strong D1 level (within 0.5 ATR of a confirmed D1 level)
    d1_near = min(opp_d, sup_d) if not np.isnan(opp_d) and not np.isnan(sup_d) else (opp_d if not np.isnan(opp_d) else sup_d)
    f[25] = 1.0 if not np.isnan(d1_near) and d1_near <= 0.5 * (atr + eps) else 0.0
    # 26: D1 level touch count (reaction density, capped 5)
    f[26] = touch_cnt
    # 27: TP path blocked by a confirmed D1 level between entry and TP
    f[27] = block_buy if direction == "BUY" else block_sell
    # 28: RSI norm (current TF, point-in-time)
    f[28] = ctx.rsi_norm(entry_j)
    # 29: MACD histogram / ATR (current TF, point-in-time)
    f[29] = ctx.macd_hist_atr(entry_j)
    return f

def train_ml_from_backtest(all_features, all_labels):
    """Train Random Forest on collected trade data, print results, save model."""
    from sklearn.ensemble import RandomForestClassifier
    X = np.array(all_features, dtype=np.float32)
    y = np.array(all_labels, dtype=np.int32)
    n_trades = len(y)
    pos_ratio = y.mean()
    print(f"\n  ML Training: {n_trades} trades ({pos_ratio:.1%} win rate)")
    if n_trades < 100:
        print("  Too few trades for ML, skipping training")
        return None
    # Time-split: first 70% train, last 30% test
    split = int(n_trades * 0.7)
    X_train, y_train = X[:split], y[:split]
    X_test, y_test = X[split:], y[split:]
    model = RandomForestClassifier(
        n_estimators=200, max_depth=6, min_samples_split=20,
        min_samples_leaf=10, class_weight="balanced", random_state=42
    )
    model.fit(X_train, y_train)
    test_prob = model.predict_proba(X_test)[:, 1]
    # Find best threshold on test set
    best_th = 0.5
    best_net = -999
    for th in np.arange(0.3, 0.85, 0.05):
        preds = (test_prob >= th).astype(int)
        if preds.sum() == 0:
            continue
        net = (preds * y_test).sum() * TP_MULT - (preds * (1 - y_test)).sum() * 1
        if net > best_net:
            best_net = net
            best_th = th
    # Evaluate at best threshold
    test_preds = (test_prob >= best_th).astype(int)
    test_wr = (test_preds == y_test).mean()
    test_trades = test_preds.sum()
    test_wins = (test_preds * y_test).sum()
    test_net = test_wins * TP_MULT - (test_trades - test_wins) * 1
    # Feature importances
    feat_names = [
        "atr/close", "vol_ratio", "bars_waited", "dist_ema50", "dist_ema200",
        "trend", "is_buy", "zone_fvg", "zone_ob", "lv_pd",
        "body/atr", "range/atr", "entry_bull", "hr_cos", "hr_sin",
        "atr_trend", "risk_dist", "entry_vol", "atr_change",
        "d1_trend", "h4_trend", "htf_aligned", "reversal_probability",
        "d1_opp_dist_atr", "d1_sup_dist_atr", "entry_at_d1_level", "d1_touch_count",
        "tp_path_blocked", "rsi_norm", "macd_hist_atr"
    ]
    fi = sorted(zip(feat_names, model.feature_importances_), key=lambda x: -x[1])
    print(f"  Time-split: train={split} trades, test={n_trades - split} trades")
    print(f"  Test split WR: {y_test.mean():.1%} (baseline)")
    print(f"  Best threshold: {best_th:.2f} -> {test_trades} trades, {test_wr:.1%} WR, {test_net:+.1f}R net")
    print(f"\n  Top 10 features:")
    for name, imp in fi[:10]:
        print(f"    {name:16s}  {imp:.3f}")
    # Save model
    artifact = {"model": model, "threshold": best_th, "features": feat_names}
    with open(ML_PATH, "wb") as f:
        pickle.dump(artifact, f)
    print(f"\n  Model saved to {ML_PATH}")
    return model

# ---------- BACKTEST ----------
def run_backtest():
    print(f"\n{'='*60}")
    print(f"  SWEEP -> MSS -> ZONE ENTRY BACKTEST")
    if SL_ATR > 0:
        st_str = " ST" if USE_ST else ""
        print(f"  {SYMBOL} -- Session={SESSION} Volume={'ON' if USE_VOLUME else 'OFF'} SL={SL_ATR:.1f}ATR TP={TP_ATR:.1f}ATR{st_str}")
    else:
        st_str = " ST" if USE_ST else ""
        pvt_str = " PVT" if USE_PVT else ""
        of_str = " OF" if USE_OF else ""
        print(f"  {SYMBOL} -- Session={SESSION} Volume={'ON' if USE_VOLUME else 'OFF'} TP={TP_MULT}R{st_str}{pvt_str}{of_str}")
    print(f"{'='*60}")
    csv_path = CSV_MAP.get(SYMBOL, {}).get(TF)
    if not csv_path:
        print(f"  ERROR: No CSV mapping for {SYMBOL}/{TF}"); return
    import os
    if not os.path.exists(csv_path):
        print(f"  ERROR: {csv_path} not found"); return
    df = pd.read_csv(csv_path)
    # Normalize column names: volume -> tick_volume, real_volume -> volume
    if 'volume' in df.columns and 'tick_volume' not in df.columns:
        df['tick_volume'] = df['volume']
    if 'real_volume' in df.columns:
        df['volume'] = df['real_volume']
    df['timestamp'] = pd.to_datetime(df['timestamp'], utc=True)
    df = df.sort_values('timestamp').reset_index(drop=True)
    if BARS and BARS < len(df):
        df = df.tail(BARS).reset_index(drop=True)
    print(f"  Bars: {len(df)} ({df['timestamp'].iloc[0]} -> {df['timestamp'].iloc[-1]})")
    df = add_indicators(df)
    if USE_OF:
        df = compute_orderflow_approx(df)
        print(f"  Orderflow filter: enabled (OHLC approx, lookback={OF_LOOKBACK})")
    n = len(df)
    if n < 300: print("  Too few bars"); return

    # Pre-compute numpy arrays for speed
    o = df['open'].values.astype(float)
    h = df['high'].values.astype(float)
    l = df['low'].values.astype(float)
    c = df['close'].values.astype(float)
    body = df['body'].values.astype(float)
    atr = df['atr'].values.astype(float)
    vol = df['tick_volume'].values.astype(float)
    avg_vol = df['avg_vol'].values.astype(float)
    ts = df['timestamp'].values

    # SuperTrend
    st_trend = None
    if USE_ST:
        st_trend, _, _ = compute_supertrend(h, l, c, ST_ATR, ST_MULT)
        print(f"  SuperTrend filter: period={ST_ATR} mult={ST_MULT}")

    # PVT divergence
    pvt_arr = None
    if USE_PVT:
        pvt_arr = compute_pvt(c, vol)
        print(f"  PVT divergence filter: enabled")

    # Pre-compute daily boundaries for PDH/PDL
    ts_np = df['timestamp'].values
    dates_val = pd.Series(df['timestamp']).dt.date.values
    uniq_dates = np.unique(dates_val)
    pdhl_cache = {}
    for d in uniq_dates:
        start_np = pd.Timestamp(d).tz_localize('UTC').to_numpy()
        end24 = start_np + np.timedelta64(24, 'h')
        idx2 = np.where((ts_np >= start_np) & (ts_np < end24))[0]
        if len(idx2) >= 5:
            pdhl_cache[d] = (float(h[idx2].max()), float(l[idx2].min()))

    # Pre-compute current-TF swing points (for sweep detection)
    sh, sl = get_swing_points(h, l, FRACTAL_STR)

    # BSL/SSL levels per bar: PDH/PDL + most recent current-TF swing (sweep detection)
    bar_bsl = [{} for _ in range(n)]
    bar_ssl = [{} for _ in range(n)]
    for i in range(n):
        d = dates_val[i]
        pd_ts = pd.Timestamp(d).tz_localize('UTC') - timedelta(days=1)
        pd_date = pd_ts.date()
        if pd_date in pdhl_cache:
            pdh, pdl = pdhl_cache[pd_date]
            bar_bsl[i]['PDH'] = pdh; bar_ssl[i]['PDL'] = pdl
        for j in range(i-1, max(i-LOOKBACK, -1), -1):
            if sh[j] and 'Swing_H' not in bar_bsl[i]:
                bar_bsl[i]['Swing_H'] = h[j]
            if sl[j] and 'Swing_L' not in bar_ssl[i]:
                bar_ssl[i]['Swing_L'] = l[j]
            if 'Swing_H' in bar_bsl[i] and 'Swing_L' in bar_ssl[i]:
                break

    # Point-in-time multi-TF context for ML features (built only if ML active)
    ml_ctx = None

    # Pre-compute MSS candidate flags
    mss_bull = np.zeros(n, dtype=bool)
    mss_bear = np.zeros(n, dtype=bool)
    for i in range(100, n-4):
        # Check if candle i could be a MSS confirmation for a prior sweep
        # Bullish MSS: price just broke above a recent swing low
        sw_type, sw_price = find_last_swing(df, i, "BUY")
        if sw_price and h[i] > sw_price:
            mss_bull[i] = True
        sw_type, sw_price = find_last_swing(df, i, "SELL")
        if sw_price and l[i] < sw_price:
            mss_bear[i] = True

    print(f"  Pre-computed levels and MSS flags")
    print(f"  Scanning {n} bars for sweep setups...")

    one_r = RISK_USD * USD1_TO_PRICE.get(SYMBOL, 1)
    rwd = one_r * TP_MULT
    trades = wins = losses = 0
    zl_counts = {"Swept": 0, "FVG": 0, "OB": 0}

    # ML setup
    ml_model = None
    ml_collect = []  # (features, label) for training
    if TRAIN_ML:
        print(f"  ML training mode: collecting features from all trades")
    elif ML_THRESHOLD > 0 and os.path.exists(ML_PATH):
        with open(ML_PATH, "rb") as f:
            artifact = pickle.load(f)
        ml_model = artifact["model"]
        if ml_model.n_features_in_ != 30:
            print(f"  WARNING: model has {ml_model.n_features_in_} features, expected 30. Retrain.")
            logging.warning(f"  WARNING: model has {ml_model.n_features_in_} features, expected 30. Retrain.")
            ml_model = None
        if ml_model:
            print(f"  ML filter: loaded {ML_PATH}, threshold={ML_THRESHOLD:.2f}")
            logging.info(f"  ML filter: loaded {ML_PATH}, threshold={ML_THRESHOLD:.2f}")
    elif ML_THRESHOLD > 0:
        print(f"  WARNING: {ML_PATH} not found, ML filter disabled")
        logging.warning(f"  WARNING: {ML_PATH} not found, ML filter disabled")

    if TRAIN_ML or ml_model is not None:
        ml_ctx = MlContext(df, o, h, l, c, atr, ts)
        ml_ctx._close = c

    # Walk through each bar as potential sweep bar
    for i in range(200, n - MAX_WAIT - 2):
        if np.isnan(atr[i]) or atr[i] <= 0:
            continue
        hour_utc = pd.Timestamp(ts[i]).hour
        if not session_ok(hour_utc):
            continue

        bsl = bar_bsl[i]
        ssl = bar_ssl[i]

        # --- SELL: sweep BSL (above resistance) ---
        for lev_name, level in bsl.items():
            if level is None or level <= 0: continue
            sweep_idx = None
            for check_i in (i, i-1):
                if check_i < 0: break
                buf = atr[check_i] * 0.1 if not np.isnan(atr[check_i]) else 0
                if h[check_i] > (level + buf):
                    # Close back inside: close < level and >50% body inside
                    if c[check_i] < level:
                        pct_inside = (level - c[check_i]) / body[check_i] if body[check_i] > 0 else 0
                        if pct_inside > 0.5:
                            sweep_idx = check_i; break
            if sweep_idx is None: continue
            if USE_VOLUME and not np.isnan(avg_vol[sweep_idx]) and avg_vol[sweep_idx] > 0:
                if vol[sweep_idx] < 1.2 * avg_vol[sweep_idx]: continue
            # MSS: must see bearish MSS within [sweep_idx+1, sweep_idx+4]
            mss_found = False
            for k in range(sweep_idx+1, min(sweep_idx+5, n-1)):
                if mss_bear[k]:
                    # Check: did this MSS candle break the relevant swing high?
                    sw_type, sw_price = find_last_swing(df, k, "SELL")
                    if sw_price and l[k] < sw_price:
                        mss_found = True; break
            if not mss_found: continue
            # PVT divergence filter
            if USE_PVT and pvt_arr is not None:
                if not pvt_divergence(h, l, pvt_arr, sweep_idx, "SELL"):
                    continue
            # Trend filter skipped in backtest (global snapshot is broken; live applies it per-poll)
            # Zone entry
            zl_sw = level - atr[sweep_idx] * 0.25
            zh_sw = level + atr[sweep_idx] * 0.25
            fvg_l, fvg_h = find_fvg(df, sweep_idx, "SELL")
            ob_l, ob_h, _ = find_order_block(df, sweep_idx, "SELL")
            entry_j = None; ep = None; zs = "Swept"
            for j in range(sweep_idx + 1, min(sweep_idx + MAX_WAIT + 1, n)):
                mid = (h[j] + l[j]) / 2
                if fvg_l is not None and fvg_l <= mid <= fvg_h: entry_j = j; ep = mid; zs = "FVG"; break
                if ob_l is not None and ob_l <= mid <= ob_h: entry_j = j; ep = mid; zs = "OB"; break
                if not SKIP_SWEPT and zl_sw <= c[j] <= zh_sw: entry_j = j; ep = c[j]; zs = "Swept"; break
            if entry_j is None: continue
            if USE_ST and st_trend is not None and st_trend[entry_j] != -1:
                continue
            if USE_OF and not orderflow_ok(df, entry_j, "SELL", OF_LOOKBACK):
                continue
            if SL_ATR > 0:
                sl_dist = atr[entry_j] * SL_ATR
                tp_dist = atr[entry_j] * TP_ATR
            else:
                sl_dist = one_r; tp_dist = rwd
            sl_p = ep + sl_dist; tp_p = ep - tp_dist
            end = min(entry_j + LABEL_BARS, n)
            ht = l[entry_j+1:end].min() <= tp_p
            hs = h[entry_j+1:end].max() >= sl_p
            # ML filter
            if ml_model is not None:
                feat = extract_ml_features(o, h, l, c, body, atr[entry_j], vol, avg_vol, atr, ts,
                    sweep_idx, entry_j, "SELL", zs, lev_name, sl_dist, ml_ctx, ep, tp_dist)
                prob = round(ml_model.predict_proba(feat[:ml_model.n_features_in_].reshape(1, -1))[0, 1], 2)
                if prob < ML_THRESHOLD:
                    logging.info(f"********** Setup is skipped due to probability < threshold: {prob} < {ML_THRESHOLD} ***************")
                    continue
            zl_counts[zs] += 1
            if TRAIN_ML:
                feat = extract_ml_features(o, h, l, c, body, atr[entry_j], vol, avg_vol, atr, ts,
                    sweep_idx, entry_j, "SELL", zs, lev_name, sl_dist, ml_ctx, ep, tp_dist)
                ml_collect.append((feat, int(ht)))
            trades += 1
            if ht: wins += 1
            else: losses += 1

        # --- BUY: sweep SSL (below support) ---
        for lev_name, level in ssl.items():
            if level is None or level <= 0: continue
            sweep_idx = None
            for check_i in (i, i-1):
                if check_i < 0: break
                buf = atr[check_i] * 0.1 if not np.isnan(atr[check_i]) else 0
                if l[check_i] < (level - buf):
                    if c[check_i] > level:
                        pct_inside = (c[check_i] - level) / body[check_i] if body[check_i] > 0 else 0
                        if pct_inside > 0.5:
                            sweep_idx = check_i; break
            if sweep_idx is None: continue
            if USE_VOLUME and not np.isnan(avg_vol[sweep_idx]) and avg_vol[sweep_idx] > 0:
                if vol[sweep_idx] < 1.2 * avg_vol[sweep_idx]: continue
            mss_found = False
            for k in range(sweep_idx+1, min(sweep_idx+5, n-1)):
                if mss_bull[k]:
                    sw_type, sw_price = find_last_swing(df, k, "BUY")
                    if sw_price and h[k] > sw_price:
                        mss_found = True; break
            if not mss_found: continue
            # PVT divergence filter
            if USE_PVT and pvt_arr is not None:
                if not pvt_divergence(h, l, pvt_arr, sweep_idx, "BUY"):
                    continue
            # Trend filter skipped in backtest (global snapshot is broken; live applies it per-poll)
            zl_sw = level - atr[sweep_idx] * 0.25
            zh_sw = level + atr[sweep_idx] * 0.25
            fvg_l, fvg_h = find_fvg(df, sweep_idx, "BUY")
            ob_l, ob_h, _ = find_order_block(df, sweep_idx, "BUY")
            entry_j = None; ep = None; zs = "Swept"
            for j in range(sweep_idx + 1, min(sweep_idx + MAX_WAIT + 1, n)):
                mid = (h[j] + l[j]) / 2
                if fvg_l is not None and fvg_l <= mid <= fvg_h: entry_j = j; ep = mid; zs = "FVG"; break
                if ob_l is not None and ob_l <= mid <= ob_h: entry_j = j; ep = mid; zs = "OB"; break
                if not SKIP_SWEPT and zl_sw <= c[j] <= zh_sw: entry_j = j; ep = c[j]; zs = "Swept"; break
            if entry_j is None: continue
            if USE_ST and st_trend is not None and st_trend[entry_j] != 1:
                continue
            if USE_OF and not orderflow_ok(df, entry_j, "BUY", OF_LOOKBACK):
                continue
            if SL_ATR > 0:
                sl_dist = atr[entry_j] * SL_ATR
                tp_dist = atr[entry_j] * TP_ATR
            else:
                sl_dist = one_r; tp_dist = rwd
            sl_p = ep - sl_dist; tp_p = ep + tp_dist
            end = min(entry_j + LABEL_BARS, n)
            ht = h[entry_j+1:end].max() >= tp_p
            hs = l[entry_j+1:end].min() <= sl_p
            # ML filter
            if ml_model is not None:
                feat = extract_ml_features(o, h, l, c, body, atr[entry_j], vol, avg_vol, atr, ts,
                    sweep_idx, entry_j, "BUY", zs, lev_name, sl_dist, ml_ctx, ep, tp_dist)
                prob = round(ml_model.predict_proba(feat[:ml_model.n_features_in_].reshape(1, -1))[0, 1], 2)
                if prob < ML_THRESHOLD:
                    logging.info(f"********** Setup is skipped due to probability < threshold: {prob} < {ML_THRESHOLD} ***************")
                    continue
            zl_counts[zs] += 1
            if TRAIN_ML:
                feat = extract_ml_features(o, h, l, c, body, atr[entry_j], vol, avg_vol, atr, ts,
                    sweep_idx, entry_j, "BUY", zs, lev_name, sl_dist, ml_ctx, ep, tp_dist)
                ml_collect.append((feat, int(ht)))
            trades += 1
            if ht: wins += 1
            else: losses += 1

    print(f"\n  RESULTS ({SYMBOL})")
    print(f"  {'='*40}")
    if trades:
        wr = wins / trades; ev = wr * TP_MULT - (1 - wr) * 1
        pnl = wins * TP_MULT - losses * 1
        print(f"  Trades:  {trades:,}")
        print(f"  Wins:    {wins:,}   Losses: {losses:,}")
        print(f"  WR:      {wr:.1%}")
        print(f"  EV:      {ev:+.2f}R")
        print(f"  PnL:     {pnl:+,.0f}R")
        print(f"  Zones:   {zl_counts}")
    else:
        print("  No trades")

    # ML training at end
    if TRAIN_ML and len(ml_collect) >= 100:
        all_features = [x[0] for x in ml_collect]
        all_labels = [x[1] for x in ml_collect]
        train_ml_from_backtest(all_features, all_labels)

# ---------- LIVE ----------
def _log_entry(setup, result):
    _bias_map = {1: "BULLISH", -1: "BEARISH", 0: "NEUTRAL"}
    dom = setup.get('dom_bias')
    dom_s = _bias_map.get(dom, dom or '?')
    parts = [
        f"dir={setup['direction']}",
        f"zone={setup.get('zone_source','?')}",
        f"entry={setup.get('entry_price','?'):.2f}",
        f"zl={setup.get('zone_low','?'):.2f}",
        f"zh={setup.get('zone_high','?'):.2f}",
        f"atr={setup.get('atr',0):.2f}",
        f"swept={setup.get('level_name','?')}@{setup.get('sweep_level','?'):.2f}",
        f"mss={setup.get('mss_type','?')}",
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
    print(f"  SWEEP -> MSS -> ZONE ENTRY LIVE{st_s}{pvt_s}{of_s}{ml_s}")
    print(f"  {SYMBOL} --- Watching every {INTERVAL}s")
    if IS_TRADE: print("  Trade mode: ACTIVE")
    else: print("  Monitor only (add --trade)")
    print(f"{'='*60}")
    import MetaTrader5 as mt5
    if not mt5.initialize():
        print("  ERROR: MT5 init failed"); return
    acc = int(os.environ["MT5_ACCOUNT"]); pwd = os.environ["MT5_PASSWORD"]; srv = "PepperstoneBS-MT5-Live01"
    if not mt5.login(acc, pwd, srv):
        print("  Login failed, trying demo...")
        if not mt5.login(int(os.environ["MT5_ACCOUNT"]), os.environ["MT5_PASSWORD"], "PepperstoneBS-MT5-Live01"):
            print("  ERROR: Login failed"); mt5.shutdown(); return
    print(f"  Logged in: {acc}")
    start_msg = f"STARTED id={INSTANCE_ID} symbol={SYMBOL} tf={TF} risk={RISK_USD}"
    extra = ""
    if USE_ST: extra += " ST"
    if USE_PVT: extra += " PVT"
    if USE_OF: extra += f" OF({OF_MODE},{OF_LOOKBACK})"
    if ML_THRESHOLD > 0: extra += f" ML@{ML_THRESHOLD}"
    if SL_ATR > 0: extra += f" SL={SL_ATR}ATR TP={TP_ATR}ATR"
    if MARKET_ONLY: extra += " MARKET_ONLY"
    if OB_SWEEP: extra += " OB_SWEEP"
    start_msg += extra
    print(f"  {start_msg}")
    logging.info(start_msg)
    mt5.symbol_select(SYMBOL, True)
    # Load ML model if configured
    live_ml_model = None
    if ML_THRESHOLD > 0 and os.path.exists(ML_PATH):
        with open(ML_PATH, "rb") as f:
            artifact = pickle.load(f)
        live_ml_model = artifact["model"]

        if live_ml_model.n_features_in_ != 30:
            print(f"  ERROR: model has {live_ml_model.n_features_in_} features, expected 30. Retrain.")
            logging.warning(f"  ERROR: model has {live_ml_model.n_features_in_} features, expected 30. Retrain.")
            live_ml_model = None
        if live_ml_model:
            print(f"  ML filter: loaded {ML_PATH}, threshold={ML_THRESHOLD:.2f}")
            logging.info(f"  ML filter: loaded {ML_PATH}, threshold={ML_THRESHOLD:.2f}")
    last_sweep_key = None
    LOGGED_SETUPS = 0
    while True:
        try:
            tf_mt5 = mt5.TIMEFRAME_M15 if TF == "m15" else mt5.TIMEFRAME_H1
            rates = mt5.copy_rates_from_pos(SYMBOL, tf_mt5, 0, BARS)
            if rates is None:
                time.sleep(INTERVAL); continue
            df = pd.DataFrame(rates)
            df['timestamp'] = pd.to_datetime(df['time'], unit='s')
            df = add_indicators(df)
            if USE_OF:
                df = compute_orderflow_live(df, mt5, tf_mt5)
            setup = scan_for_setups(df)
            if setup and live_ml_model is not None:
                # Compute ML probability for this setup
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
                # Point-in-time multi-TF context (same as backtest)
                _ctx_live = MlContext(df, o_a, h_a, l_a, c_a, atr_a, ts_a)
                _ctx_live._close = c_a
                ep_live = setup['entry_price']
                if SL_ATR > 0:
                    tp_dist_live = atr_a[ej] * TP_ATR
                else:
                    tp_dist_live = RISK_USD * USD1_TO_PRICE.get(SYMBOL, 1) * TP_MULT
                feat = extract_ml_features(o_a, h_a, l_a, c_a, body_a, atr_a[ej], vol_a, avg_vol_a, atr_a, ts_a,
                    setup['sweep_idx'], ej, setup['direction'], setup['zone_source'],
                    setup['level_name'], RISK_USD * USD1_TO_PRICE.get(SYMBOL, 1), _ctx_live, ep_live, tp_dist_live)
                prob = round(live_ml_model.predict_proba(feat[:live_ml_model.n_features_in_].reshape(1, -1))[0, 1], 2)
                setup['ml_prob'] = prob
                if prob < ML_THRESHOLD:
                    logging.info(f"********** Setup {setup['direction']} is skipped due to probability < threshold: {prob} < {ML_THRESHOLD} ***************")
                    continue  # skip this setup

            # Show all zones in range (if enabled)
            if SHOW_ZONES:
                all_zones = scan_for_setups(df, return_all=True)
                if all_zones:
                    print(f"\n  Zones in range ({len(all_zones)}):")
                    for z in all_zones:
                        z_dir = z['direction']; z_zs = z['zone_source']
                        z_zl = z['zone_low']; z_zh = z['zone_high']
                        z_ln = z['level_name']; z_sl = z['sweep_level']
                        z_d = f"delta={z['delta']:+.0f}" if 'delta' in z else ""
                        print(f"    {z_dir:4s} {z_zs:6s}  [{z_zl:.2f}-{z_zh:.2f}]  "
                              f"swept={z_ln} @{z_sl:.2f}  {z_d}")
                else:
                    print(f"  No active zones")

            ## --- Re-validate pending orders (disabled) ---
            #if IS_TRADE:
            #    pending = mt5.orders_get(symbol=SYMBOL)
            #    if pending:
            #        for o in pending:
            #            if "LIQ" not in (o.comment or ""): continue
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
                key = f"{setup['direction']}_{setup['level_name']}_{limit_p:.1f}"
                if key != last_sweep_key or LOGGED_SETUPS == 0:
                    print_setup(setup)
                    LOGGED_SETUPS += 1
                    last_sweep_key = key
                    if IS_TRADE:
                        n_pos = count_positions(mt5, ["MSS"])
                        if n_pos < MAX_ORDERS:
                            result = place_mt5_trade(mt5, setup)
                            if result:
                                print(f"  >>> ORDER: retcode={result.retcode}")
                                logging.info(_log_entry(setup, result))
                        else:
                            print(f"  Max orders ({MAX_ORDERS}) reached, skipping")
        except KeyboardInterrupt:
            print("\nStopped"); mt5.shutdown(); break
        except Exception as e:
            print(f"  Error: {e}")
            time.sleep(INTERVAL)
        time.sleep(INTERVAL)
    mt5.shutdown()

# ---------- MAIN ----------
if __name__ == "__main__":
    if IS_BACKTEST or TRAIN_ML:
        run_backtest()
    elif IS_WATCH:
        run_live()
    else:
        print("  Use --backtest to backtest or --watch for live monitoring")
        print(f"  Example: python sweep_mss_zone.py {SYMBOL} --backtest")
