import os
import MetaTrader5 as mt5
import pandas as pd
import time
from datetime import datetime
from ta.trend import EMAIndicator, MACD
from ta.momentum import RSIIndicator
from ta.volatility import AverageTrueRange
import logging
import sys
import math
import argparse

SMALL_VAL = -1e-9

def parse_args():
    parser = argparse.ArgumentParser(description="Breakout + Reversal Strategy")
    parser.add_argument("symbol", help="Symbol e.g. XAUUSD, BTCUSD")
    parser.add_argument("bars", type=int, help="Number of M15 bars to fetch")
    parser.add_argument("max_orders", type=int, help="Max open orders")
    parser.add_argument("risk_usd", type=float, help="Risk per trade in USD")
    parser.add_argument("lookback", type=int, help="Lookback bars for fractal S/R")
    parser.add_argument("fractal_strength", type=int, help="Fractal strength (candles each side)")
    parser.add_argument("--bias", action="store_true", help="Enable H1 bias filter (reduces trades, no WR gain)")
    parser.add_argument("--reversal", action="store_true", help="Enable liquidity sweep reversal signals (adds ~15%% trades)")
    parser.add_argument("--score", action="store_true", help="Enable probability scoring filter (removes ~30%% trades, no WR gain)")
    return parser.parse_args()

args = parse_args()
SYMBOL = args.symbol
TIMEFRAME = mt5.TIMEFRAME_M15
TIMEFRAME_H1 = mt5.TIMEFRAME_H1
BARS = args.bars
MAX_OPEN_ORDERS = args.max_orders
RISK_USD = args.risk_usd
LOOKBACK = args.lookback
FRACTAL_STRENGTH = args.fractal_strength
USE_BIAS = args.bias
USE_REVERSAL = args.reversal
USE_SCORE = args.score

comment_parts = ["BO"]
if USE_BIAS: comment_parts.append("bias")
if USE_REVERSAL: comment_parts.append("rev")
if USE_SCORE: comment_parts.append("logic")
COMMENT = " ".join(comment_parts)

logging.basicConfig(
    filename=f"{SYMBOL}_trade.log",
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
)

MAGICS = {
    "M1": 1001,
    "M5": 1005,
    "M15": 1015,
    "H1": 1240,
}
SMALL_VAL = -1e-9

# -------------------------
# STATE CLASS
# -------------------------
class PositionState:
    def __init__(self):
        self.support = SMALL_VAL
        self.resistance = SMALL_VAL
        self.active_bo_sell = False
        self.active_bo_buy = False
        self.sell_support_zone = SMALL_VAL
        self.buy_resistance_zone = SMALL_VAL
        self.active = True

    def reset(self):
        #self.support = SMALL_VAL
        #self.resistance = SMALL_VAL
        self.active = True

state = PositionState()

def initialize_mt5():
    if not mt5.initialize():
        raise Exception("MT5 initialization failed")

    logging.info("✅ MT5 connected")    
    #account = int(os.environ["MT5_ACCOUNT"])
    #password = os.environ["MT5_PASSWORD"]
    #server = "Pepperstone-MT5-Live01"
    account = int(os.environ["MT5_ACCOUNT"])
    password = os.environ["MT5_PASSWORD"]
    server = "Pepperstone-Demo"

    if not mt5.login(account, password, server):
        logging.info("❌ Login failed")
    else:
        logging.info("✅ Logged in")

def get_data(tf):
    rates = mt5.copy_rates_from_pos(SYMBOL, tf, 0, BARS)
    df = pd.DataFrame(rates)
    df['time'] = pd.to_datetime(df['time'], unit='s')
    return df

def add_indicators(df):
    df['ema50'] = df['close'].ewm(span=50).mean()
    df['ema200'] = df['close'].ewm(span=200).mean()
    df['body'] = abs(df['close'] - df['open'])
    atr = AverageTrueRange(df['high'], df['low'], df['close'], 14)
    df['atr'] = atr.average_true_range()
    return df

def detect_fractals(df):
    highs = df['high']
    lows = df['low']

    df['fractal_high'] = (
        (highs.shift(2) < highs.shift(1)) &
        (highs.shift(1) < highs) &
        (highs > highs.shift(-1)) &
        (highs.shift(-1) > highs.shift(-2))
    )

    df['fractal_low'] = (
        (lows.shift(2) > lows.shift(1)) &
        (lows.shift(1) > lows) &
        (lows < lows.shift(-1)) &
        (lows.shift(-1) < lows.shift(-2))
    )

    return df

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

def get_recent_levels(df, lookback=50):
    recent = df.tail(lookback)

    resistance = recent[recent['fractal_high']]['high'].max()
    support = recent[recent['fractal_low']]['low'].min()

    resistance = 0.0 if math.isnan(resistance) else float(resistance)
    support = 0.0 if math.isnan(support) else float(support)

    return support, resistance

def detect_breakout(open, close, support, resistance, atr):
    buffer = atr * 0.1

    breakout_up = close > (resistance + buffer) and open < resistance
    breakout_down = close < (support - buffer) and open > support

    print(f"Detect breakout-up close: {close} open: {open} resistance: {resistance} buffer: {buffer}")
    print(f"Detect breakout-down close: {close} open: {open} support: {support} buffer: {buffer}")

    #logging.info(f"Detect breakout-up close: {close} open: {open} resistance: {resistance} buffer: {buffer}")
    #logging.info(f"Detect breakout-down close: {close} open: {open} support: {support} buffer: {buffer}")

    return breakout_up, breakout_down

def calculate_score(df, i, support, resistance, direction, h1_bias):
    row = df.iloc[i]
    body = row['body']
    if body <= SMALL_VAL:
        return -5, 0.0

    score = 0.0
    candle_range = row['high'] - row['low']
    atr = row['atr'] if row['atr'] > 0 else 1e-9

    # 1. Trend alignment: EMA50/200 distance normalized by ATR (-2 to +2)
    ema_dist = (row['ema50'] - row['ema200']) / atr
    if direction == "BUY":
        score += max(-2.0, min(2.0, ema_dist * 4))
    else:
        score += max(-2.0, min(2.0, -ema_dist * 4))

    # 2. Breakout gap ratio as continuous factor (0 to +2)
    buy_displacement = row['high'] - resistance
    sell_displacement = support - row['low']
    gap_ratio = buy_displacement / candle_range if direction == "BUY" else sell_displacement / candle_range
    gap_ratio = max(0.0, gap_ratio)
    if gap_ratio < 0.2:
        return -5, gap_ratio  # only hard-reject if barely any displacement
    score += min(2.0, gap_ratio * 2)

    # 3. Body momentum relative to rolling average (0 to +1.5)
    avg_body = df['body'].rolling(20).mean().iloc[i]
    body_ratio = body / avg_body if avg_body > 0 else 1.0
    score += min(1.5, body_ratio * 0.5)

    # 4. Volume confirmation (0 to +1)
    avg_vol = df['tick_volume'].rolling(20).mean().iloc[i]
    vol_ratio = row['tick_volume'] / avg_vol if avg_vol > 0 else 1.0
    score += min(1.0, vol_ratio * 0.33)

    # 5. Wick rejection — gradual penalty instead of hard cut
    upper_wick = row['high'] - max(row['close'], row['open'])
    lower_wick = min(row['close'], row['open']) - row['low']
    upper_ratio = upper_wick / body if body > 0 else 0
    lower_ratio = lower_wick / body if body > 0 else 0
    if direction == "BUY" and upper_ratio > 0.5:
        score -= (upper_ratio - 0.5) * 3
    elif direction == "SELL" and lower_ratio > 0.5:
        score -= (lower_ratio - 0.5) * 3

    # 6. Body quality: strong body relative to full range is bullish (+0 to +1)
    body_to_range = body / candle_range if candle_range > 0 else 0
    if body_to_range > 0.6:
        score += 1.0

    # 7. Breakout distance beyond level in ATR terms (0 to +1)
    if direction == "BUY":
        b_dist = row['close'] - resistance
    else:
        b_dist = support - row['close']
    b_dist_atr = b_dist / atr
    score += min(1.0, max(0, b_dist_atr * 5))

    return score, gap_ratio

def score_to_probability(score):
    # Score range: ~-2 to ~+9.5 → map to [0, 1]
    return max(0.0, min(1.0, (score + 2) / 9))

def get_h1_data(bars=50):
    rates = mt5.copy_rates_from_pos(SYMBOL, TIMEFRAME_H1, 0, bars)
    if rates is None: return None
    df = pd.DataFrame(rates)
    df['time'] = pd.to_datetime(df['time'], unit='s')
    df['body'] = abs(df['close'] - df['open'])
    return df

def detect_h1_bias(df_h1, periods=4, strict=False):
    """Returns 'BULLISH', 'BEARISH', or 'NEUTRAL' from last N H1 candles.
    strict=True: require 3/4 candles aligned.
    strict=False (relaxed): require 2/4 candles aligned (default)."""
    recent = df_h1.tail(periods)
    bullish = sum(recent['close'] > recent['open'])
    bearish = periods - bullish
    net_change = recent['close'].iloc[-1] - recent['open'].iloc[0]
    req = 3 if strict else 2
    if bullish >= req and net_change > 0:
        return "BULLISH"
    elif bearish >= req and net_change < 0:
        return "BEARISH"
    return "NEUTRAL"

def get_h1_fractal_snr(df_h1, strength=2):
    df = df_h1.copy()
    df = detect_fractals(df)
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

def detect_liquidity_sweep(df, i, level, direction, atr_buffer=0.1):
    """Check if candle i swept through a level (below support or above resistance)."""
    row = df.iloc[i]
    buffer = row['atr'] * atr_buffer
    if direction == "BUY":  # sweep below support
        return row['low'] < (level - buffer)
    else:  # sweep above resistance
        return row['high'] > (level + buffer)

def detect_reversal(df, i, support, resistance):
    """Check for liquidity sweep + reversal close.
    Returns ('BUY','reversal'), ('SELL','reversal'), (None,None), or ('BUY','breakout'), ('SELL','breakout')."""
    row = df.iloc[i]
    close = row['close']
    buffer = row['atr'] * 0.1

    # --- Reversal BUY: H1 bullish, sweep below support, close above support ---
    sweep_buy = detect_liquidity_sweep(df, i, support, "BUY")
    if not sweep_buy and i >= 1:
        sweep_buy = detect_liquidity_sweep(df, i - 1, support, "BUY")
    if sweep_buy and close > support:
        return "BUY", "reversal"

    # --- Reversal SELL: H1 bearish, sweep above resistance, close below resistance ---
    sweep_sell = detect_liquidity_sweep(df, i, resistance, "SELL")
    if not sweep_sell and i >= 1:
        sweep_sell = detect_liquidity_sweep(df, i - 1, resistance, "SELL")
    if sweep_sell and close < resistance:
        return "SELL", "reversal"

    return None, None

def is_price_in_zone(symbol, state, atr, zone_center):
    tick = mt5.symbol_info_tick(symbol)
    if tick is None:
        raise Exception("Failed to get tick data")
    tick = mt5.symbol_info_tick(SYMBOL)
    mid_price = (tick.ask + tick.bid) / 2
    zone_low = zone_center - atr * 0.25
    zone_high = zone_center + atr * 0.25
    return zone_low <= mid_price <= zone_high

def generate_signal(df, state, h1_df, h1_support, h1_resistance, lookback=50):
    i = len(df) - 2
    open = df['open'][i]
    close = df['close'][i]

    support, resistance = get_fractal_snr_tv(df, FRACTAL_STRENGTH, lookback)
    if support is None or resistance is None:
        support, resistance = get_recent_levels(df, lookback)

    h1_bias = None
    if USE_BIAS or USE_REVERSAL:
        h1_bias = detect_h1_bias(h1_df, periods=4, strict=False)
        print(f"H1 bias: {h1_bias} (last 4 H1 candles)")

    direction = None
    signal_type = None
    zone_center = None

    # --- 1. Check breakout signals ---
    breakout_up, breakout_down = detect_breakout(open, close, support, resistance, df['atr'][i])
    if breakout_up:
        bias_ok = (not USE_BIAS) or (h1_bias in ("BULLISH", "NEUTRAL"))
        if bias_ok:
            direction = "BUY"
            signal_type = "breakout"
            zone_center = resistance
            print(f"BO BUY: breakout above resistance={resistance}" + (f", H1 bias={h1_bias}" if USE_BIAS else ""))
    elif breakout_down:
        bias_ok = (not USE_BIAS) or (h1_bias in ("BEARISH", "NEUTRAL"))
        if bias_ok:
            direction = "SELL"
            signal_type = "breakout"
            zone_center = support
            print(f"BO SELL: breakout below support={support}" + (f", H1 bias={h1_bias}" if USE_BIAS else ""))

    # --- 2. If no breakout, check reversal signals ---
    if direction is None and USE_REVERSAL:
        rev_dir, rev_type = detect_reversal(df, i, support, resistance)
        if rev_dir == "BUY":
            bias_ok = (not USE_BIAS) or (h1_bias == "BULLISH")
            if bias_ok:
                direction = "BUY"
                signal_type = "reversal"
                zone_center = support
                print(f"REV BUY: sweep below support={support}, close above" + (f", H1 bullish" if USE_BIAS else ""))
        elif rev_dir == "SELL":
            bias_ok = (not USE_BIAS) or (h1_bias == "BEARISH")
            if bias_ok:
                direction = "SELL"
                signal_type = "reversal"
                zone_center = resistance
                print(f"REV SELL: sweep above resistance={resistance}, close below" + (f", H1 bearish" if USE_BIAS else ""))

    score = None
    probability = None

    if direction == "BUY":
        state.active_bo_buy = True
    elif direction == "SELL":
        state.active_bo_sell = True

    signal = direction if direction else None

    if direction:
        print(f"signal_type={signal_type} dir={direction} zone_center={zone_center:.2f}")
        if USE_SCORE:
            score, gap_ratio = calculate_score(df, i, zone_center, zone_center, direction, h1_bias)
            probability = score_to_probability(score)
            print(f"  score: {score:.2f}  probability: {probability:.1%}")
            if probability < 0.5:
                signal = None
                print(f"  -> REJECTED by scoring filter")
        print()
    else:
        print(f"No signal — breakout" + (" or reversal" if USE_REVERSAL else "") + " conditions not met\n")

    result = {
        "signal": signal,
        "support": support,
        "resistance": resistance,
        "h1_bias": h1_bias,
        "h1_support": h1_support,
        "h1_resistance": h1_resistance,
        "signal_type": signal_type,
        "zone_center": zone_center,
        "score": score,
        "probability": probability,
    }
    return result

USD1_TO_PRICE_DICT = {
    "XAUUSD": 1,
    "BTCUSD": 100,
    "GBPJPY": 0.1
}

RISK_TO_REWARD_DICT = {
    "XAUUSD": 0.5,
    "BTCUSD": 0.4,
    "GBPJPY": 0.4
}

def calculate_trade_by_usd(symbol, direction):
    tick = mt5.symbol_info_tick(symbol)

    if tick is None:
        raise Exception("❌ No tick data")

    entry = tick.ask if direction == "BUY" else tick.bid
    
    # --- RISK ---
    risk = RISK_USD * USD1_TO_PRICE_DICT[symbol]
    reward = RISK_TO_REWARD_DICT[symbol] * risk

    # --- SL / TP ---
    if direction == "BUY":
        sl = entry - risk
        tp = entry + reward

    else:
        sl = entry + risk
        tp = entry - reward

    return sl, tp

def count_bo_positions():
    positions = mt5.positions_get(symbol=SYMBOL)
    if positions is None: return 0
    keywords = COMMENT.split()[1:]  # ["bias", "rev", "logic"] etc, empty if none
    if not keywords:
        return sum(1 for p in positions if p.comment.startswith("BO"))
    return sum(1 for p in positions if all(k in p.comment for k in keywords))

def place_trade(signal, sl, tp):
    n_current = count_bo_positions()

    while n_current < MAX_OPEN_ORDERS:
        n_current += 1
        lot = 0.01
        price = mt5.symbol_info_tick(SYMBOL).ask if signal == "BUY" else mt5.symbol_info_tick(SYMBOL).bid

        order_type = mt5.ORDER_TYPE_BUY if signal == "BUY" else mt5.ORDER_TYPE_SELL

        request = {
            "action": mt5.TRADE_ACTION_DEAL,
            "symbol": SYMBOL,
            "volume": lot,
            "type": order_type,
            "price": price,
            "sl": sl,
            "tp": tp,
            "deviation": 20,
            "magic": 1015,
            "comment": COMMENT,
            "type_time": mt5.ORDER_TIME_GTC,
            "type_filling": mt5.ORDER_FILLING_IOC,
        }

        result = mt5.order_send(request)
        print(result)
        logging.info(result)

def update_state(result):

    if state.resistance != result['resistance'] or state.support != result['support']:
        # allows new order for the new resistance
        state.active = True

    if state.resistance != result['resistance']:
        print(f"new resistance detected: {result['resistance']}")
        logging.info(f"new resistance detected: {result['resistance']}")
        state.resistance = float(result['resistance'])
        
        # reset the current buy zone because the resistance level has expired
        state.buy_resistance_zone = SMALL_VAL
        state.active_bo_buy = False

        logging.info(f"resistance buy zone has been reset: {state.buy_resistance_zone} active bo buy: {state.active_bo_buy}")
        
        print(result)
        logging.info(result)
    
    if  state.support != result['support']:
        print(f"new support detected {result['support']}")
        logging.info(f"new support detected {result['support']}")
        state.support = float(result['support'])

        # reset the current sell zone because the support level has expired
        state.sell_support_zone = SMALL_VAL
        state.active_bo_sell = False

        logging.info(f"support sell zone has been reset: {state.sell_support_zone} active bo sell: {state.active_bo_sell}")

        print(result)
        logging.info(result)

def get_fractal_snr_tv_lookback(df, lookback=None, strength=2):
    if lookback:
        df = df.tail(lookback)

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

def get_fractal_snr_tv(df, strength=2, lookback=None):
    if lookback:
        df = df.tail(lookback)

    highs = df["high"].values
    lows = df["low"].values

    resistance = None
    support = None

    # start far enough to avoid index error
    start = len(df) - (strength + 1)

    for i in range(start, strength, -1):

        if resistance is None and is_high_fractal(highs, i, strength):
            resistance = highs[i]

        if support is None and is_low_fractal(lows, i, strength):
            support = lows[i]

        if resistance is not None and support is not None:
            break

    return support, resistance

def run():
    initialize_mt5()

    while True:
        df = get_data(mt5.TIMEFRAME_M15)
        df = add_indicators(df)
        df = detect_fractals(df)
        i = len(df) - 2

        # Fetch H1 data only if bias or reversal is enabled
        h1_df = h1_support = h1_resistance = None
        if USE_BIAS or USE_REVERSAL:
            h1_df = get_h1_data(bars=50)
            if h1_df is not None:
                h1_support, h1_resistance = get_h1_fractal_snr(h1_df, FRACTAL_STRENGTH)
        
        result = generate_signal(df, state, h1_df, h1_support, h1_resistance, LOOKBACK)
        
        # update state with an active signal and set zone
        if result["signal"] and state.active:
            logging.info(result)
            print(result)

            if state.active_bo_buy or state.active_bo_sell:
                zone_center = result.get("zone_center")
                if zone_center is not None:
                    # Store the zone center for entry checking
                    if result["signal"] == "BUY":
                        state.buy_resistance_zone = zone_center
                    else:
                        state.sell_support_zone = zone_center
                    signal_label = result.get("signal_type", "breakout").upper()
                    print(f"  [{signal_label}] Zone set @ {zone_center:.2f}")

            state.active = False

        # update latest support and resistance
        update_state(result)

        # check for buy/sell zone if there's an active signal
        if state.active_bo_buy or state.active_bo_sell:
            zone = state.buy_resistance_zone if state.active_bo_buy else state.sell_support_zone
            if zone > SMALL_VAL and is_price_in_zone(SYMBOL, state, df['atr'][i], zone):
                print("******************** Price is in the buy/sell zone ********************")
                logging.info("******************** Price is in the buy/sell zone ********************")

                signal = "BUY" if state.active_bo_buy else "SELL"
                sl, tp = calculate_trade_by_usd(SYMBOL, signal)
                place_trade(signal, sl, tp)
                
                print("******************** resetting zones ********************")
                logging.info("******************** resetting zones ********************")
                
                state.active_bo_buy = False
                state.active_bo_sell = False
                state.sell_support_zone = SMALL_VAL
                state.buy_resistance_zone = SMALL_VAL

        time.sleep(10)

if __name__ == "__main__":
    run()
