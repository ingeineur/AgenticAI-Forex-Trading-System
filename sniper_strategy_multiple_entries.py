import os
import MetaTrader5 as mt5
import pandas as pd
import time
from datetime import datetime
from ta.trend import EMAIndicator, MACD
from ta.momentum import RSIIndicator
from ta.volatility import AverageTrueRange
from ai_analyses import analyze_with_llm, analyze_with_mistral
#from test_ollama import analyze_with_llm
import logging
import sys

logging.basicConfig(
    filename=f"{sys.argv[1]}_trade.log",
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
)

# -------------------------
# CONFIG
# -------------------------
SYMBOL = sys.argv[1]
TIMEFRAMES = {
    #"M1": mt5.TIMEFRAME_M1,
    "M5": mt5.TIMEFRAME_M5,
    "M15": mt5.TIMEFRAME_M15,
    #"H1": mt5.TIMEFRAME_H1
}

SMALL_VAL = -1e-9
MAX_RISK_PCT = 0.01
MIN_SL = float(sys.argv[2])
MIN_SL_USD = float(sys.argv[2])
ATR_MULTIPLIER = 1.8
CROSS_TOLERANCE = 0.3
MAX_OPEN_ORDERS = int(sys.argv[8])
DECISION_THRESHOLD_PERCENT = int(sys.argv[9])

ASSET_CLASS = sys.argv[3]

LLM_LABEL = sys.argv[6]

MAGICS = {
    "M1": 1001,
    "M5": 1005,
    "M15": 1015,
    "H1": 1240,
}

USD1_TO_PRICE_DICT = {
    "XAUUSD": 1,
    "BTCUSD": 100,
    "GBPJPY": 0.1
}

RISK_TO_REWARD_DICT = {
    "XAUUSD": 0.6,
    "BTCUSD": 0.6,
    "GBPJPY": 1
}

MAX_CROSS_PERIOD_DICT = {
    "M1": 60,
    "M5": 24,
    "M15": 10
}

# -------------------------
# STATE CLASS
# -------------------------
class PositionState:
    def __init__(self, timeframe):
        self.timeframe = timeframe
        self.magic = MAGICS[timeframe]

        self.last_closed = 0.0
        self.cross_price = SMALL_VAL
        self.cross_ema9 = SMALL_VAL
        self.cross_ema21 = SMALL_VAL
        self.cross_periods = 0.0
        self.zone_low = 0.0
        self.zone_height = 0.0
        self.num_open_orders = 0

    def reset(self):
        self.last_closed = 0.0
        self.cross_price = SMALL_VAL
        self.cross_ema9 = SMALL_VAL
        self.cross_ema21 = SMALL_VAL
        self.cross_periods = 0.0
        self.zone_low = 0.0
        self.zone_height = 0.0
        self.num_open_orders = 0

states = {}
states["buy"] = {tf: PositionState(tf) for tf in TIMEFRAMES}
states["sell"] = {tf: PositionState(tf) for tf in TIMEFRAMES}
#states = {tf: PositionState(tf) for tf in TIMEFRAMES}

# -------------------------
# MT5 INIT
# -------------------------
if not mt5.initialize():
    logging.info("❌ MT5 init failed")
    quit()

logging.info("✅ MT5 connected")

account = int(os.environ["MT5_ACCOUNT"])
password = os.environ["MT5_PASSWORD"]
server = "Pepperstone-Demo"

if not mt5.login(account, password, server):
    logging.info("❌ Login failed")
else:
    logging.info("✅ Logged in")

# -------------------------
# DATA
# -------------------------
def get_data(timeframe, bars=200):
    rates = mt5.copy_rates_from_pos(SYMBOL, timeframe, 0, bars)
    df = pd.DataFrame(rates)
    df['time'] = pd.to_datetime(df['time'], unit='s')
    return df


# -------------------------
# INDICATORS
# -------------------------
def add_vwap(df):
    tp = (df['high'] + df['low'] + df['close']) / 3
    vol = df['tick_volume']
    df['vwap'] = (tp * vol).cumsum() / vol.cumsum()
    return df

def TEMA(series, period):
    ema1 = series.ewm(span=period, adjust=False).mean()
    ema2 = ema1.ewm(span=period, adjust=False).mean()
    ema3 = ema2.ewm(span=period, adjust=False).mean()

    return 3 * ema1 - 3 * ema2 + ema3

def add_indicators(df):
    df['ema9'] = EMAIndicator(df['close'], 9).ema_indicator()
    df['ema21'] = EMAIndicator(df['close'], 21).ema_indicator()

    df['tema9'] = TEMA(df['close'], 9)
    df['tema21'] = TEMA(df['close'], 21)

    macd = MACD(df['close'])
    df['macd'] = macd.macd()
    df['macd_signal'] = macd.macd_signal()

    df['rsi'] = RSIIndicator(df['close'], 14).rsi()

    atr = AverageTrueRange(df['high'], df['low'], df['close'], 14)
    df['atr'] = atr.average_true_range()

    df['vol_avg'] = df['tick_volume'].rolling(20).mean()

    df = add_vwap(df)
    return df


# -------------------------
# SCORE SYSTEM
# -------------------------
def compute_score(last):
    bScore = 0
    rScore = 0

    if last['close'] > last['vwap']: bScore += 1
    else: rScore += 1

    if last['rsi'] > 50: bScore += 1
    else: rScore += 1

    if last['macd'] > last['macd_signal']: bScore += 1
    else: rScore += 1

    if last['ema9'] > last['ema21']: bScore += 1
    else: rScore += 1

    if last['tick_volume'] > last['vol_avg']:
        if last['close'] > last['open']:
            bScore += 1
        else:
            rScore += 1

    bullPct = (bScore / 5) * 100
    bearPct = (rScore / 5) * 100

    return bullPct, bearPct


# -------------------------
# SIGNAL
# -------------------------
def update_ema_cross(tf, df, last, state, ema_crossed, temaEnabled):
    # update last detected cross
    #logging.info(f"***************updating ema cross state: ema21 - {state.cross_ema21} ema9 - {state.cross_ema9}*****************")
    '''if state.cross_price > 0.0 and (state.cross_ema21 > 0.0 and state.cross_ema9 > 0.0):
        logging.info(f"*********************updating periods --->:{tf} {state.cross_periods} {state.cross_price} {state.cross_ema21} {state.cross_ema9}")
      
        if state.cross_periods > MAX_CROSS_PERIOD_DICT[tf]:
            logging.info("***************Reseting state cross period")
            state.cross_periods = 0
            state.cross_price = SMALL_VAL
            state.cross_ema9 = SMALL_VAL
            state.cross_ema21 = SMALL_VAL

        elif state.last_closed != last["close"]:
            logging.info(f"--{state.last_closed} <----> {last["close"]}--")
            state.cross_periods += 1'''

    if ema_crossed:
        #reset values
        state.cross_periods = 0
        state.last_closed = last['close']
        logging.info(f"********TIMEFRAME {tf}: detected ema crossed: ema9: {last['ema9']}, ema21: {last['ema21']} last closed {state.last_closed}********")

        if temaEnabled is True:
            state.cross_price = last["close"]
            state.cross_ema9 = last["tema9"]
            state.cross_ema21 = last["tema21"]
        else: 
            state.cross_price = last["close"]
            state.cross_ema9 = last["ema9"]
            state.cross_ema21 = last["ema21"]

def detect_ema_cross(df, temaEnabled):
    prev = df.iloc[-3]
    last = df.iloc[-2]

    buy_cross = prev['ema9'] < prev['ema21'] and last['ema9'] > last['ema21']
    sell_cross = prev['ema9'] > prev['ema21'] and last['ema9'] < last['ema21']

    tema_buy_cross = prev['tema9'] < prev['tema21'] and last['tema9'] > last['tema21']
    tema_sell_cross = prev['tema9'] > prev['tema21'] and last['tema9'] < last['tema21']

    bullPct, bearPct = compute_score(last)

    if temaEnabled is True:
        triggerBuy = tema_buy_cross and bullPct >= 30
        triggerSell = tema_sell_cross and bearPct >= 30
        return triggerBuy, triggerSell
    
    triggerBuy = buy_cross and bullPct >= 30
    triggerSell = sell_cross and bearPct >= 30

    if triggerBuy:
        logging.info("**************EMA BUY CROSS DETECTED*********")
    elif triggerSell:
        logging.info("**************EMA SELL CROSS DETECTED*********")

    return triggerBuy, triggerSell

def generate_signal(df, temaEnabled, state):
    prev = df.iloc[-3]
    last = df.iloc[-2]

    buy_cross = prev['ema9'] < prev['ema21'] and last['ema9'] > last['ema21']
    sell_cross = prev['ema9'] > prev['ema21'] and last['ema9'] < last['ema21']

    tema_buy_cross = prev['tema9'] < prev['tema21'] and last['tema9'] > last['tema21']
    tema_sell_cross = prev['tema9'] > prev['tema21'] and last['tema9'] < last['tema21']

    bullPct, bearPct = compute_score(df)

    if temaEnabled is True:
        triggerBuy = tema_buy_cross and bullPct >= 30
        triggerSell = tema_sell_cross and bearPct >= 30
        return triggerBuy, triggerSell, bullPct, bearPct, last['tema9'], last['tema21']
    
    triggerBuy = buy_cross and bullPct >= 30
    triggerSell = sell_cross and bearPct >= 30
    return triggerBuy, triggerSell, bullPct, bearPct, last['ema9'], last['ema21']

# -------------------------
# TRADE CALC
# -------------------------

def get_pip_size(symbol):
    info = mt5.symbol_info(symbol)
    
    # Common convention
    if "JPY" in symbol:
        return 0.001
    elif "XAU" in symbol:
        return 0.1   # or 0.01 depending on broker
    elif "BTC" in symbol:
        return 1.0   # adjust if needed
    else:
        return 0.0001

def calculate_trade_pips(entry, direction, sl_pips=50, rr=1.0, symbol="XAUUSD"):
    pip = get_pip_size(symbol)

    risk = sl_pips * pip
    reward = risk * rr

    logging.info(f"SL (pips): {sl_pips}, pip size: {pip}, risk: {risk}")

    if direction == "BUY":
        sl = entry - risk
        tp = entry + reward
    else:
        sl = entry + risk
        tp = entry - reward

    return entry, sl, [tp]
    
def calculate_trade(entry, atr, direction):
    risk = max(MIN_SL, atr * ATR_MULTIPLIER)
    
    # cap risk by %
    if risk / entry > MAX_RISK_PCT:
        risk = entry * MAX_RISK_PCT
    
    logging.info(f"Calculated risk {risk}")

    if direction == "BUY":
        sl = entry - risk
        tps = [entry + risk * i for i in range(1, 6)]
    else:
        sl = entry + risk
        tps = [entry - risk * i for i in range(1, 6)]

    return entry, sl, tps


# -------------------------
# FRACTAL DETECTION
# -------------------------
def compute_fractals(df: pd.DataFrame):
    highs = df["high"].values
    lows = df["low"].values

    fractal_highs = []
    fractal_lows = []

    for i in range(2, len(df) - 2):
        # Fractal High (Resistance)
        if (
            highs[i] > highs[i - 1]
            and highs[i] > highs[i - 2]
            and highs[i] > highs[i + 1]
            and highs[i] > highs[i + 2]
        ):
            fractal_highs.append((i, highs[i]))

        # Fractal Low (Support)
        if (
            lows[i] < lows[i - 1]
            and lows[i] < lows[i - 2]
            and lows[i] < lows[i + 1]
            and lows[i] < lows[i + 2]
        ):
            fractal_lows.append((i, lows[i]))

    return fractal_highs, fractal_lows


# -------------------------
# FILTER LEVELS (remove duplicates / noise)
# -------------------------
def filter_levels(levels, min_distance):
    if not levels:
        return []

    filtered = [levels[0]]

    for idx, price in levels[1:]:
        if all(abs(price - f_price) > min_distance for _, f_price in filtered):
            filtered.append((idx, price))

    return filtered


# -------------------------
# GET NEAREST S/R
# -------------------------
def get_fractal_snr(df, current_price):
    # Use only closed candles
    df = df.iloc[:-1]

    fractal_highs, fractal_lows = compute_fractals(df)

    # Sort by price proximity (optional)
    fractal_highs = sorted(fractal_highs, key=lambda x: x[1])
    fractal_lows = sorted(fractal_lows, key=lambda x: x[1])

    # --- Filter noise (use ATR or fixed distance) ---
    df = add_indicators(df)
    atr = df["atr"].iloc[-1]

    fractal_highs = filter_levels(fractal_highs, min_distance=atr * 0.5)
    fractal_lows = filter_levels(fractal_lows, min_distance=atr * 0.5)

    # -------------------------
    # FIND NEAREST LEVELS
    # -------------------------
    resistance = None
    support = None

    # nearest resistance ABOVE price
    for _, price in fractal_highs:
        if price > current_price:
            resistance = price
            break

    # nearest support BELOW price
    for _, price in reversed(fractal_lows):
        if price < current_price:
            support = price
            break

    return support, resistance

def is_high_fractal(highs, i):
    return (
        highs[i] > highs[i - 1] and
        highs[i] > highs[i - 2] and
        highs[i] > highs[i + 1] and
        highs[i] > highs[i + 2]
    )

def is_low_fractal(lows, i):
    return (
        lows[i] < lows[i - 1] and
        lows[i] < lows[i - 2] and
        lows[i] < lows[i + 1] and
        lows[i] < lows[i + 2]
    )

def get_fractal_snr_tv(df):
    highs = df["high"].values
    lows = df["low"].values

    resistance = None
    support = None

    # iterate backwards (like TradingView "latest line wins")
    for i in range(len(df) - 3, 2, -1):
        if resistance is None and is_high_fractal(highs, i):
            resistance = highs[i]

        if support is None and is_low_fractal(lows, i):
            support = lows[i]

        if resistance is not None and support is not None:
            break

    return support, resistance


def calculate_trade_by_usd(df, symbol, direction):
    tick = mt5.symbol_info_tick(symbol)

    if tick is None:
        raise Exception("❌ No tick data")

    entry = tick.ask if direction == "BUY" else tick.bid
    mid = (tick.ask + tick.bid) / 2

    # --- RISK ---
    risk = MIN_SL_USD * USD1_TO_PRICE_DICT[symbol]
    reward = RISK_TO_REWARD_DICT[symbol] * risk

    # --- S/R based on mid (important) ---
    sup, res = get_fractal_snr_tv(df)
    logging.info(f"************ price: {entry} support: {sup} resistence: {res} *****************")

    logging.info(f"price: {entry} | mid: {mid} | S: {sup} | R: {res}")

    spread = tick.ask - tick.bid
    buffer = spread * 1.2

    # --- SL / TP ---
    if direction == "BUY":
        sl = entry - risk
        tp_raw = entry + reward
        tp = res if res else tp_raw

    else:
        sl = entry + risk
        tp_raw = entry - reward
        tp = sup if sup else tp_raw

    return entry, sl, [tp]

# -------------------------
# SYNC MT5
# -------------------------
def sync_state(intent, state, extra_status):
    positions = mt5.positions_get(symbol=SYMBOL)

    #match = None
    count = 0
    if positions:
        for p in positions:
            if p.magic == state.magic and p.comment == f"{intent}{state.timeframe}-{extra_status}":
                count += 1
                #match = p
                #break

    state.num_open_orders = count
    #if state.num_open_orders <= 0:
        #logging.info(f"[{state.timeframe}] 🔄 All orders (MT5 closed)")
    #    state.reset()


# -------------------------
# ORDER
# -------------------------
def validate_price(signal, price):
    tick = mt5.symbol_info_tick(SYMBOL)

    if signal == "BUY":
        # ❗ ensure valid price
        if price >= tick.ask:
            logging.info(f"[{state.timeframe}] ⚠️ BUY LIMIT invalid (above market)")
            return None

    else:
        if price <= tick.bid:
            logging.info(f"[{state.timeframe}] ⚠️ SELL LIMIT invalid (below market)")
            return None
    
    return price
    
def place_order(state, signal, entry, sl, tps, extra_status):
    if state.num_open_orders >= MAX_OPEN_ORDERS:
        return

    i = state.num_open_orders
    while i < MAX_OPEN_ORDERS:
        i += 1
        tick = mt5.symbol_info_tick(SYMBOL)
        price = tick.ask if signal == "BUY" else tick.bid
        order_type = mt5.ORDER_TYPE_BUY if signal == "BUY" else mt5.ORDER_TYPE_SELL

        intent = "B" if signal == "BUY" else "S"

        request = {
            "action": mt5.TRADE_ACTION_DEAL,
            "symbol": SYMBOL,
            "volume": float(sys.argv[7]),
            "type": order_type,
            "price": price,
            "sl": sl,
            "tp": tps[0],
            "magic": 1000 + int(state.timeframe.replace("M", "")),  # M1=1001, M5=1005, etc,
            "comment": f"{intent}{state.timeframe}-{extra_status}",
            "type_time": mt5.ORDER_TIME_GTC,
            "type_filling": mt5.ORDER_FILLING_IOC,
            "deviation": 10,
        }

        result = mt5.order_send(request)

        if result is None:
            logging.info(f"[{state.timeframe}] ❌ order_send returned None:", mt5.last_error())
            return
        
        if result.retcode == mt5.TRADE_RETCODE_DONE:
            logging.info(f"[{state.timeframe}] ✅ {signal} opened")
        else:
            logging.info(f"[{state.timeframe}] ❌ Order failed:", result.retcode)
            logging.info("retcode:", result.retcode)
            logging.info("comment:", result.comment)
            logging.info("request:", request)

def normalize_price(symbol, price):
    info = mt5.symbol_info(symbol)
    return round(price, info.digits)

def place_limit_order(state, signal, entry, sl, tps, extra_status):
    if state.num_open_orders >= MAX_OPEN_ORDERS:
        return

    price = validate_price(signal, entry)

    if price is None:
        return
    
    order_type = mt5.ORDER_TYPE_BUY_LIMIT if signal == "BUY" else mt5.ORDER_TYPE_SELL_LIMIT
    
    request = {
        "action": mt5.TRADE_ACTION_PENDING,
        "symbol": SYMBOL,
        "volume": float(sys.argv[7]),
        "type": order_type,
        "price": normalize_price(SYMBOL, price),
        "sl": sl,
        "tp": tps[0],
        "magic": 2000 + int(state.timeframe.replace("M", "")),
        "comment": f"{state.timeframe}-{extra_status}-L",
        "type_time": mt5.ORDER_TIME_GTC,
        "type_filling": mt5.ORDER_FILLING_RETURN,
    }

    result = mt5.order_send(request)

    if result is None:
        logging.info(f"[{state.timeframe}] ❌ limit order_send None:", mt5.last_error())
        return

    if result.retcode == mt5.TRADE_RETCODE_DONE:
        logging.info(f"[{state.timeframe}] 📌 LIMIT {signal} placed @ {price}")
    else:
        logging.info(f"[{state.timeframe}] ❌ LIMIT failed:", result.retcode, result.comment)

# -------------------------
# LOGGING
# -------------------------
def log_trade(tf, signal, entry, sl, tps, bullPct, bearPct, tema9, tema21, temaEnabled, aiEnabled, decision):
    log = {
        "time": datetime.now(),
        "tf": tf,
        "signal": signal,
        "entry": entry,
        "sl": sl,
        "tp1": tps[0],
        "bullPct": bullPct,
        "bearPct": bearPct,
        "ema9": tema9,
        "ema21": tema21,
        "tema": temaEnabled,
        "AI": aiEnabled,
        "LLM": LLM_LABEL,
        "Decision": decision,
    }

    logging.info("📊", log)
    pd.DataFrame([log]).to_csv(f"{SYMBOL}_trade_log.csv", mode='a', header=False, index=False)


# -------------------------
# LLM 
# -------------------------
def count_distinct_touches(df, level, atr, kind="resistance", lookback=100):
    buffer = atr * 0.3
    touches = 0
    in_zone = False

    for i in range(len(df) - lookback, len(df)):
        high = df["high"].iloc[i]
        low = df["low"].iloc[i]

        if kind == "resistance":
            is_touch = (level - buffer) <= high <= (level + buffer)
        else:
            is_touch = (level - buffer) <= low <= (level + buffer)

        if is_touch and not in_zone:
            touches += 1
            in_zone = True
        elif not is_touch:
            in_zone = False

    return touches

def build_llm_packet(intent, symbol, asset_class, tf, df, bullPct, bearPct, state):
    
    tick = mt5.symbol_info_tick(symbol)

    if tick is None:
        raise Exception("❌ Failed to get tick data")

    tick = mt5.symbol_info_tick(SYMBOL)
    mid_price = (tick.ask + tick.bid) / 2

    # -------------------------
    # HTF DATA (H1)
    # -------------------------
    df_htf = get_data(timeframe=mt5.TIMEFRAME_H1, bars=200)
    df_htf = add_indicators(df_htf)

    htf_sup , htf_res = get_fractal_snr_tv(df_htf)
    htf = df_htf.iloc[-2]        # closed candle
    htf_prev = df_htf.iloc[-3]
    
    res_touches = count_distinct_touches(df_htf, htf_res, htf['atr'], kind="resistance")
    sup_touches = count_distinct_touches(df_htf, htf_sup, htf['atr'], "support")

    distance_to_res = (htf_res - mid_price) / mid_price
    distance_to_sup = (mid_price - htf_sup) / mid_price

    logging.info(f"LLM Packet***************htf-support: {htf_sup} htf-resistance {htf_res}************** price: {mid_price} dist {distance_to_res} : {distance_to_sup}")

    is_near_res = bool(distance_to_res < 0.001)   # 0.01%
    is_near_sup = bool(distance_to_sup < 0.001)

    # -------------------------
    # LTF DATA (your trading TF)
    # -------------------------
    last = df.iloc[-2]           # closed candle
    prev = df.iloc[-3]

    # -------------------------
    # LIVE PRICE (for zone entry)
    # -------------------------
    tick = mt5.symbol_info_tick(symbol)

    if tick is None:
        raise Exception("❌ Failed to get tick data")

    current_price = tick.ask if intent == "BUY" else tick.bid

    # -------------------------
    # SAFE VALUES
    # -------------------------
    cross_gap = abs(state.cross_ema9 - state.cross_ema21)
    cross_gap = cross_gap if cross_gap != 0 else 1e-6
    ema21_slope_from_cross = float(last["ema21"] - state.cross_ema21)

    current_gap = abs(last["ema9"] - last["ema21"])

    # -------------------------
    # ZONE CHECK
    # -------------------------
    #in_zone = state.zone_low <= current_price <= state.zone_high

    # -------------------------
    # STRUCTURE CHECK
    # -------------------------
    if intent == "BUY":
        structure_holding = last["low"] > prev["low"]
    else:
        structure_holding = last["high"] < prev["high"]

    # -------------------------
    # PACKET
    # -------------------------
    packet = {
        # --- META ---
        "asset_class": asset_class,
        "symbol": symbol,
        "timeframe": tf,
        "intent": intent,

        # --- PRICE ---
        "current_price": float(current_price),
        "close": float(last["close"]),

        # --- ZONE ---
        #"zone_low": float(state.zone_low),
        #"zone_high": float(state.zone_high),
        #"is_in_zone": bool(in_zone),

        # --- CROSS REFERENCE ---
        "cross_ema9": float(state.cross_ema9),
        "cross_ema21": float(state.cross_ema21),
        "cross_ema_gap": float(cross_gap),
        "ema21_slope_from_cross": ema21_slope_from_cross,

        # --- LTF INDICATORS ---
        "ema9": float(last["ema9"]),
        "ema21": float(last["ema21"]),
        "ema_gap": float(current_gap),
        "ema_gap_vs_cross": float(current_gap / cross_gap),
        "ema_slope": float(last["ema9"] - prev["ema9"]),

        "rsi": float(last["rsi"]),
        "macd": float(last["macd"]),
        "macd_signal": float(last["macd_signal"]),
        "atr": float(last["atr"]),

        # --- SCORES ---
        "bullPct": float(bullPct),
        "bearPct": float(bearPct),

        # --- CONDITIONS ---
        "is_rsi_neutral": bool(45 <= last["rsi"] <= 55),
        "is_low_volatility": bool(
            last["atr"] < df["atr"].rolling(20).mean().iloc[-2]  # FIXED
        ),

        # --- VWAP ---
        "vwap": float(last["vwap"]),
        "is_above_vwap": bool(last["close"] > last["vwap"]),
        "vwap_distance_pct": float(
            (last["close"] - last["vwap"]) / last["vwap"] * 100
        ),
        "vwap_distance_atr": float(
            abs(last["close"] - last["vwap"]) / last["atr"]
        ),

        # --- STRUCTURE ---
        "structure_holding": bool(structure_holding),

        # --- HTF ---
        "htf_resistance": float(htf_res),
        "htf_support": float(htf_sup),
        "is_near_htf_res": is_near_res,
        "is_near_htf_sup": is_near_sup,
        "distance_to_htf_res": float(distance_to_res),
        "distance_to_htf_sup": float(distance_to_sup),
        "htf_res_touches": res_touches,
        "htf_sup_touches": sup_touches,

        "htf_ema9": float(htf["ema9"]),
        "htf_ema21": float(htf["ema21"]),
        "htf_trend_up": bool(htf["ema9"] > htf["ema21"]),
        "htf_ema_slope": float(htf["ema9"] - htf_prev["ema9"]),
        "htf_rsi": float(htf["rsi"]),
        "htf_above_vwap": bool(htf["close"] > htf["vwap"]),
    }

    return packet

def is_price_in_zone(signal, symbol, last, cross_ema21):

    tick = mt5.symbol_info_tick(symbol)

    if tick is None:
        raise Exception("❌ Failed to get tick data")

    tick = mt5.symbol_info_tick(SYMBOL)
    mid_price = (tick.ask + tick.bid) / 2

    zone_mid = cross_ema21  # more stable than price
    #zone_size = last["atr"] * 0.2  # tunable

    if signal == "BUY":
        zone_low = zone_mid - last["atr"] * 0.25
        zone_high = zone_mid + last["atr"] * 0.25
    else:
        zone_low = zone_mid - last["atr"] * 0.25
        zone_high = zone_mid + last["atr"] * 0.25

    #logging.info(f"zone low: {zone_low} high: {zone_high} price: {mid_price}")

    return zone_low <= mid_price <= zone_high

def eval_execute_order(df, tf, state, intent):
    if state.num_open_orders < MAX_OPEN_ORDERS:
        current = df.iloc[-1]

        # calculate score for current price 
        bullPct, bearPct = compute_score(current)

        decision = 0
        if ai_enabled is True:
            # assemble market data for llm
            market_data = build_llm_packet(intent, SYMBOL, ASSET_CLASS, tf, df, bullPct, bearPct, state)
            logging.info(market_data)
            print(market_data)

            if LLM_LABEL == "mist":
                decision = analyze_with_mistral(market_data)
            else:
                decision = analyze_with_llm(market_data)
            logging.info(f"AI decision: {decision} %")
        
        if decision >= DECISION_THRESHOLD_PERCENT:
            #entry, sl, tps = calculate_trade(current['close'], current['atr'], intent)
            entry, sl, tps = calculate_trade_by_usd(df, SYMBOL, intent)
            place_order(state, intent, entry, sl, tps, extra_status)
            log_trade(tf, intent, entry, sl, tps, bullPct, bearPct, current["ema9"], current["ema21"], tema_enabled, ai_enabled, decision)
        else:
            #entry, sl, tps = calculate_trade(current['close'], current['atr'], "BUY")
            entry, sl, tps = calculate_trade_by_usd(df, SYMBOL, intent)
            log_trade(tf, f"REJECT {intent}", entry, sl, tps, bullPct, bearPct, current["ema9"], current["ema21"], tema_enabled, ai_enabled, decision)

# -------------------------
# MAIN LOOP
# -------------------------
def str_to_bool(val):
    return str(val).lower() in ("true", "1", "yes", "y", "t")

tema_enabled = str_to_bool(sys.argv[4])
ai_enabled = str_to_bool(sys.argv[5])
logging.info(f"{SYMBOL} {ASSET_CLASS} AI: {ai_enabled} TEMA: {tema_enabled} - {LLM_LABEL}")

extra_status = ""
if ai_enabled:
    extra_status = "AI"

if tema_enabled:
    extra_status += "TM"

extra_status += f"-{LLM_LABEL}"

while True:
    try:
        for tf, mtf in TIMEFRAMES.items():
            buy_state = states["buy"][tf]
            sell_state = states["sell"][tf]

            # 1. Sync MT5
            sync_state("B", buy_state, extra_status)
            sync_state("S", sell_state, extra_status)

            # 2. Data
            df = get_data(mtf)
            df = add_indicators(df)

            last = df.iloc[-2]
            current = df.iloc[-1]

            # 3. detect ema crossed
            ema_buy_crossed, ema_sell_crossed = detect_ema_cross(df, tema_enabled)
            
            # 4. Update both sell and buy state's ema crossed signal
            update_ema_cross(tf, df, last, buy_state, ema_buy_crossed, tema_enabled)
            update_ema_cross(tf, df, last, sell_state, ema_sell_crossed, tema_enabled)

            if buy_state.cross_price > 0.0 and buy_state.cross_ema21 > 0.0 and buy_state.cross_ema9 > 0.0:
                # 5. Check current price if it enters the zone buy
                in_zone = is_price_in_zone("BUY", SYMBOL, last, buy_state.cross_ema21)
                
                # 6. valid buy entry
                if in_zone:
                    ema_gap_now = abs(current["ema9"] - current["ema21"])
                    ema_gap_cross = abs(buy_state.cross_ema9 - buy_state.cross_ema21)
                    valid_buy_entry = (current["ema9"] > current["ema21"]) and (ema_gap_now >= ema_gap_cross * 0.8)
                    logging.info(f"{tf}: detected valid buy entry {valid_buy_entry}")
                    
                    if valid_buy_entry:
                        eval_execute_order(df, tf, buy_state, "BUY")

            # saved the last closed
            buy_state.last_closed = last['close']

            if sell_state.cross_price > 0.0 and sell_state.cross_ema21 > 0 and sell_state.cross_ema9 > 0:
                # Check current price if it enters the zone sell
                in_zone = is_price_in_zone("SELL", SYMBOL, last, sell_state.cross_ema21)
                
                # valid sell entry
                if in_zone:
                    ema_gap_now = abs(current["ema9"] - current["ema21"])
                    ema_gap_cross = abs(sell_state.cross_ema9 - sell_state.cross_ema21)
                    valid_sell_entry = (current["ema9"] < current["ema21"]) and (ema_gap_now >= ema_gap_cross * 0.8)
                    logging.info(f"{tf}: detected valid sell entry {valid_sell_entry}")
                    
                    if valid_sell_entry:
                        eval_execute_order(df, tf, sell_state, "SELL")

            # saved the last closed
            sell_state.last_closed = last['close']
            
        time.sleep(10)

    except Exception as e:
        logging.info("❌ ERROR:", e)
        time.sleep(5)