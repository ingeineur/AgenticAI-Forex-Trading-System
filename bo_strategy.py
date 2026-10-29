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

SYMBOL = sys.argv[1]
TIMEFRAME = mt5.TIMEFRAME_M15
BARS = int(sys.argv[2])
MAX_OPEN_ORDERS = int(sys.argv[3])
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
        self.active = True
        self.num_open_orders = 0

    def reset(self):
        self.num_open_orders = 0
        self.support = SMALL_VAL
        self.resistance = SMALL_VAL
        self.active = True

state = PositionState()

def initialize_mt5():
    if not mt5.initialize():
        raise Exception("MT5 initialization failed")

    logging.info("✅ MT5 connected")    
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

def get_fractal_snr(df):
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

def get_recent_levels(df, lookback=50):
    recent = df.tail(lookback)

    resistance = recent[recent['fractal_high']]['high'].max()
    support = recent[recent['fractal_low']]['low'].min()

    return support, resistance

def detect_breakout(price, support, resistance, buffer=0.1):
    breakout_up = price > resistance + buffer
    breakout_down = price < support - buffer
    return breakout_up, breakout_down

def calculate_score(df, i):
    score = 0

    # Trend
    if df['ema50'][i] > df['ema200'][i]:
        score += 2
    else:
        score -= 1

    # Momentum
    body = df['body'][i]
    avg_body = df['body'].rolling(20).mean()[i]
    if body > avg_body:
        score += 2

    # Volume
    if df['tick_volume'][i] > df['tick_volume'].rolling(20).mean()[i]:
        score += 1

    # Wick rejection (fake breakout filter)
    upper_wick = df['high'][i] - max(df['close'][i], df['open'][i])
    lower_wick = min(df['close'][i], df['open'][i]) - df['low'][i]

    if upper_wick > body:
        score -= 2
    if lower_wick > body:
        score -= 2

    return score

def score_to_probability(score):
    return max(0, min(1, (score + 5) / 10))

def generate_signal(df):
    i = len(df) - 2
    price = df['close'][i]

    support, resistance = get_recent_levels(df)

    breakout_up, breakout_down = detect_breakout(price, support, resistance)

    if breakout_up:
        logging.info("breakout up")
    elif breakout_down:
        logging.info("breakout down")

    score = calculate_score(df, i)
    probability = score_to_probability(score)

    signal = None

    if breakout_up and probability > 0.7:
        signal = "BUY"
    elif breakout_down and probability > 0.7:
        signal = "SELL"

    return {
        "signal": signal,
        "score": score,
        "probability": probability,
        "support": support,
        "resistance": resistance
    }

USD1_TO_PRICE_DICT = {
    "XAUUSD": 1,
    "BTCUSD": 100,
    "GBPJPY": 0.1
}

RISK_TO_REWARD_DICT = {
    "XAUUSD": 0.5,
    "BTCUSD": 0.5,
    "GBPJPY": 0.5
}

def calculate_trade_by_usd(symbol, direction):
    tick = mt5.symbol_info_tick(symbol)

    if tick is None:
        raise Exception("❌ No tick data")

    entry = tick.ask if direction == "BUY" else tick.bid
    
    # --- RISK ---
    risk = 15 * USD1_TO_PRICE_DICT[symbol]
    reward = RISK_TO_REWARD_DICT[symbol] * risk

    # --- SL / TP ---
    if direction == "BUY":
        sl = entry - risk
        tp = entry + reward

    else:
        sl = entry + risk
        tp = entry - reward

    return sl, tp

def place_trade(signal, sl, tp):
    positions = mt5.positions_get(symbol=SYMBOL)
    i = len(positions)

    while i < MAX_OPEN_ORDERS:
        i += 1
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
            "magic": 1015,  # M1=1001, M5=1005, etc,
            "comment": "Breakout Strategy",
            "type_time": mt5.ORDER_TIME_GTC,
            "type_filling": mt5.ORDER_FILLING_IOC,
        }

        result = mt5.order_send(request)
        print(result)

def update_state(result):
    if state.resistance != result['resistance'] or state.support != result['support']:
        print("new SNR detected")
        logging.info("new SNR detected")
        state.reset()
        state.resistance = result['resistance']
        state.support = result['support']

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

def run():
    initialize_mt5()

    while True:
        df = get_data(mt5.TIMEFRAME_M15)
        df = add_indicators(df)
        df = detect_fractals(df)
        
        #support, resistance = get_fractal_snr_tv(get_data(mt5.TIMEFRAME_H1))
        #print(f"support {support} resistance: {resistance}")
        result = generate_signal(df)

        # update strategy state
        update_state(result)
        logging.info(result)

        if result["signal"] and state.active:
            print(result)
            sl, tp = calculate_trade_by_usd(SYMBOL, result["signal"])
            place_trade(result["signal"], sl, tp)
            state.active = False

        time.sleep(10)

if __name__ == "__main__":
    run()