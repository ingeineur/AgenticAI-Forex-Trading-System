import os
import MetaTrader5 as mt5
import pandas as pd
import time
from datetime import datetime
from ta.trend import EMAIndicator, MACD
from ta.momentum import RSIIndicator
from ta.volatility import AverageTrueRange
#from ai_analyses import analyze_with_llm, analyze_with_mistral
#from test_ollama import analyze_with_llm
import logging
import sys
import math

logging.basicConfig(
    filename=f"{sys.argv[1]}_trade.log",
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
)

SYMBOL = sys.argv[1]
TIMEFRAME = mt5.TIMEFRAME_M15
BARS = int(sys.argv[2])
MAX_OPEN_ORDERS = int(sys.argv[3])
RISK_USD = float(sys.argv[4])
LOOKBACK = int(sys.argv[5])
FRACTAL_STRENGTH = int(sys.argv[6])

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
        self.active = True

class BmBState:
    def __init__(self):
        self.support = SMALL_VAL
        self.resistance = SMALL_VAL
        self.active_bo_sell = False
        self.active_bo_buy = False
        self.active_buy = False
        self.active_sell = False

state = PositionState()
bmb_state = BmBState()

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

    resistance = 0.0 if math.isnan(resistance) else float(resistance)
    support = 0.0 if math.isnan(support) else float(support)

    return support, resistance

def detect_breakout(open, close, support, resistance, atr):
    buffer = atr * 0.1

    breakout_up = resistance > SMALL_VAL and close > (resistance + buffer) and open < resistance
    breakout_down = support > SMALL_VAL and close < (support - buffer) and open > support

    print(f"Detect breakout-up close: {close} open: {open} resistance: {resistance} buffer: {buffer}")
    print(f"Detect breakout-down close: {close} open: {open} support: {support} buffer: {buffer}")

    return breakout_up, breakout_down

def calculate_score(df, i, support, resistance, direction):

    row = df.iloc[i]
    score = 0

    risk = RISK_USD * USD1_TO_PRICE_DICT[SYMBOL]
    reward = RISK_TO_REWARD_DICT[SYMBOL] * risk
    
    atr_ratio = row['atr']/reward
    print(f"-----> ATR: {row['atr']} reward: {reward} ATR RATIO: {atr_ratio}")
    
    body = row['body']
    
    if body <= SMALL_VAL:
        return -5, 0.0

    # --- Trend ---
    if direction == "BUY":
        score += 2 if row['ema50'] > row['ema200'] else -1

    elif direction == "SELL":
        score += 2 if row['ema50'] < row['ema200'] else -1

    # --- Momentum ---
    avg_body = df['body'].rolling(20).mean().iloc[i]

    if body > avg_body:
        score += 2

    # --- Volume ---
    avg_volume = df['tick_volume'].rolling(20).mean().iloc[i]

    if row['tick_volume'] > avg_volume:
        score += 1

    # --- Wick rejection ---
    upper_wick = row['high'] - max(row['close'], row['open'])
    lower_wick = min(row['close'], row['open']) - row['low']

    upper_wick_ratio = upper_wick / body
    lower_wick_ratio = lower_wick / body

    print(f"upper wick ratio: {upper_wick_ratio}")
    print(f"lower wick ratio: {lower_wick_ratio}")

    if direction == "BUY":
        if upper_wick_ratio > 0.6:
            return -5, 0.0

    elif direction == "SELL":
        if lower_wick_ratio > 0.6:
            return -5, 0.0

    # --- Breakout displacement ---
    gap_ratio = 0.0
    candle_range = row['high'] - row['low']  
    buy_displacement = row['high'] - resistance 
    sell_displacement = support - row['low'] 

    if direction == "BUY":
        gap_ratio = buy_displacement / candle_range

    elif direction == "SELL":
        gap_ratio = sell_displacement / candle_range

    print(f"gap ratio: {gap_ratio} : {row['close']} : {buy_displacement} : {sell_displacement} : {candle_range}")

    if gap_ratio < 0.2:
        return -5, 0.0

    return score, gap_ratio

def score_to_probability(score):
    return max(0, min(1, (score + 5) / 10))

def is_price_in_zone(symbol, state, atr, support, resistance):

    tick = mt5.symbol_info_tick(symbol)

    if tick is None:
        raise Exception("❌ Failed to get tick data")

    tick = mt5.symbol_info_tick(SYMBOL)
    mid_price = (tick.ask + tick.bid) / 2

    if state.active_bo_buy:
        zone_low = resistance - atr * 0.25
        zone_high = resistance + atr * 0.25
    elif state.active_bo_sell:
        zone_low = support - atr * 0.25
        zone_high = support + atr * 0.25

    return zone_low <= mid_price <= zone_high

def is_price_in_bmb_zone(symbol, state, atr, support, resistance):

    tick = mt5.symbol_info_tick(symbol)

    if tick is None:
        raise Exception("❌ Failed to get tick data")

    tick = mt5.symbol_info_tick(SYMBOL)
    mid_price = (tick.ask + tick.bid) / 2

    if state.active_sell:
        zone_low = resistance - atr * 0.25
        zone_high = resistance + atr * 0.25
    elif state.active_buy:
        zone_low = support - atr * 0.25
        zone_high = support + atr * 0.25

    return zone_low <= mid_price <= zone_high

def count_distinct_touches(df, level, atr, kind="resistance", lookback=200):
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

def calculate_bmb_score(df, i, direction):

    row = df.iloc[i]
    score = 0

    body = row['body']
    
    if body <= SMALL_VAL:
        return -5, 0.0

    # --- Liquidity grab ---
    support, resistance = get_fractal_snr_tv(df, 10, 200)

    if direction == "RE-ENTRY UP":
        sup_touches = count_distinct_touches(df, support, row['atr'], "support")
        logging.info(f"Fractal Support Touch Count: {sup_touches}")
        if sup_touches > 0:
            score += 2
        else:
            score -= 1

    elif direction == "RE-ENTRY DOWN":
        res_touches = count_distinct_touches(df, resistance, row['atr'], kind="resistance")
        logging.info(f"Fractal Resistance Touch Count: {res_touches}")
        if res_touches > 0:
            score += 2
        else:
            score -= 1

    # --- Momentum ---
    avg_body = df['body'].rolling(20).mean().iloc[i]

    if body > avg_body:
        score += 1

    # --- Volume ---
    avg_volume = df['tick_volume'].rolling(20).mean().iloc[i]

    if row['tick_volume'] > avg_volume:
        score += 1

    # --- Wick rejection ---
    upper_wick = row['high'] - max(row['close'], row['open'])
    lower_wick = min(row['close'], row['open']) - row['low']

    upper_wick_ratio = upper_wick / body
    lower_wick_ratio = lower_wick / body

    print(f"upper wick ratio: {upper_wick_ratio}")
    print(f"lower wick ratio: {lower_wick_ratio}")

    if direction == "RE-ENTRY UP":
        if upper_wick_ratio > 0.6:
            return -5, 0.0

    elif direction == "RE-ENTRY DOWN":
        if lower_wick_ratio > 0.6:
            return -5, 0.0

    # --- Re-Entry Breakout displacement ---
    gap_ratio = 0.0
    candle_range = row['high'] - row['low']  
    buy_displacement = row['high'] - support 
    sell_displacement = resistance - row['low'] 

    if direction == "RE-ENTRY UP":
        gap_ratio = buy_displacement / candle_range

    elif direction == "RE-ENTRY DOWN":
        gap_ratio = sell_displacement / candle_range

    print(f"gap ratio: {gap_ratio} : {row['close']} : {buy_displacement} : {sell_displacement} : {candle_range}")

    if gap_ratio < 0.2:
        return -5, 0.0

    return score, gap_ratio

def update_bmb_state(df, state):
    if state.active_bo_sell == False and state.active_bo_buy == False:
        print("BMB: There is no active breakout at the momment")
        return
    
    print("BMB: There is an active breakout ************* monitoring BMB re-entry********")
    i = len(df) - 2
    open = df['open'][i]
    close = df['close'][i]

    # detect a re-entry into bo support or resistance zone
    re_entry_up, re_entry_down = detect_breakout(open=open, close=close, support=state.resistance, resistance=state.support, atr=df['atr'][i])

    direction = None
    if re_entry_up:
        direction = "RE-ENTRY UP"
    elif re_entry_down:
        direction = "RE-ENTRY DOWN"

    if direction != None:  
        # calculate re-entry score
        score, gap_ratio = calculate_bmb_score(df, i, direction) 
        probability = score_to_probability(score)

        # activate BO mode
        if re_entry_up and probability > 0.7:
            logging.info("re_entry_up")
            state.active_buy = True
        elif re_entry_down and probability > 0.7:
            logging.info("re_entry_down")
            state.active_sell = True

def generate_signal(df, state, lookback=50):
    i = len(df) - 2
    open = df['open'][i]
    close = df['close'][i]

    support, resistance = get_fractal_snr_tv(df, FRACTAL_STRENGTH, lookback)
    
    # detect a BO up or down
    breakout_up, breakout_down = detect_breakout(open, close, support, resistance, df['atr'][i])

    direction = None
    # activate BO mode
    if breakout_up:
        logging.info("breakout up")
        state.active_bo_buy = True
        direction = "BUY"

        # activate bmb sell when bo buy is detected
        bmb_state.active_bo_buy = True
        bmb_state.resistance = resistance

    elif breakout_down:
        logging.info("breakout down")
        state.active_bo_sell = True
        direction = "SELL"

        # activate bmb buy when bo sell is detected
        bmb_state.active_bo_sell = True
        bmb_state.support = support

    signal = None
    score = 0
    probability = 0.0

    score, gap_ratio = calculate_score(df, i, support, resistance, direction)
    probability = score_to_probability(score)
    print(f"active bo sell: {state.active_bo_sell} sell zone: {state.sell_support_zone} active bo buy: {state.active_bo_buy} buy zone: {state.buy_resistance_zone}")
    print(f"score: {score} probability: {probability} support: {support} resistance: {resistance} \n")

    # create an active buy/sell zone
    if state.active_bo_sell or state.active_bo_buy:
        if state.active_bo_buy and probability >= 0.6:
            signal = "BUY"
        elif state.active_bo_sell and probability >= 0.6:
            signal = "SELL"

    return {
        "signal": signal,
        "score": score,
        "probability": probability,
        "support": support,
        "resistance": resistance,
        "gap_ratio": gap_ratio,
    }

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

def place_trade(signal, sl, tp, comment="BO Strategy"):
    positions = mt5.positions_get(symbol=SYMBOL)
    positions = [p for p in positions if p.comment == comment]
    
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
            "comment": comment,
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
        bmb_state.resistance = SMALL_VAL
        bmb_state.active_bo_buy = False

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
        bmb_state.support = SMALL_VAL
        bmb_state.active_bo_sell = False

        logging.info(f"support sell zone has been reset: {state.sell_support_zone} active bo sell: {state.active_bo_sell}")

        print(result)
        logging.info(result)

def get_fractal_snr_tv_lookback(df, lookback=None):
    if lookback:
        df = df.tail(lookback)

    highs = df["high"].values
    lows = df["low"].values

    resistance = None
    support = None

    for i in range(len(df) - 3, 2, -1):
        if resistance is None and is_high_fractal(highs, i):
            resistance = highs[i]

        if support is None and is_low_fractal(lows, i):
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

def execute_bo_order(df, i, state):
    # check for buy/sell zone if there's an active BO
    if state.active_bo_buy or state.active_bo_sell:
        if is_price_in_zone(SYMBOL, state, df['atr'][i], state.sell_support_zone, state.buy_resistance_zone):
            print("******************** Price is in the buy/sell zone ********************")
            logging.info("******************** Price is in the buy/sell zone ********************")

            signal = "BUY" if state.active_bo_buy else "SELL"
            sl, tp = calculate_trade_by_usd(SYMBOL, signal)
            place_trade(signal, sl, tp)
            
            print("******************** resetting zones ********************")
            logging.info("******************** resetting zones ********************")
            
            # reset bo setup after orders are executed
            # rule 1: only allow one set of transactions per breakout
            state.active_bo_buy = False
            state.active_bo_sell = False
            state.sell_support_zone = SMALL_VAL
            state.buy_resistance_zone = SMALL_VAL

def execute_bmb_order(df, i, bmb_state):
    # check for buy/sell zone if there's an active bmb buy or sell
    if bmb_state.active_buy or bmb_state.active_sell:
        if is_price_in_bmb_zone(SYMBOL, bmb_state, df['atr'][i], bmb_state.support, bmb_state.resistance):
            print("******************** Price is in the BMB buy/sell zone ********************")
            logging.info("******************** Price is in BMB the buy/sell zone ********************")

            signal = "BUY" if bmb_state.active_buy else "SELL"
            sl, tp = calculate_trade_by_usd(SYMBOL, signal)
            place_trade(signal, sl, tp, "BMB Strategy")
            
            print("******************** resetting bmb zones ********************")
            logging.info("******************** resetting bmb zones ********************")
            
            # reset bmb setup after orders are executed
            # rule 1: only allow one set of transactions per breakout
            bmb_state.active_buy = False
            bmb_state.active_sell = False
            bmb_state.support = SMALL_VAL
            bmb_state.resistance = SMALL_VAL

def process_bo_signal(df, i, state, result):
    if result["signal"] and state.active:
        logging.info(result)
        print(result)

        if state.active_bo_buy or state.active_bo_sell:
            #check for too big candle
            retracement_range = df['high'][i] - df['low'][i]    
            
            if (retracement_range / USD1_TO_PRICE_DICT[SYMBOL]) < (2*(RISK_USD - 1.0)):
                logging.info(f"updating buy and sell zone resistance: {state.resistance} support: {state.support}")
                print(f"updating buy and sell zone resistance: {state.resistance} support: {state.support}")
                low = df['low'][i]
                high = df['high'][i]
                mid_price = (high + low)/2

                if result["gap_ratio"] >= 0.6:   
                    if result["signal"] == "BUY":
                        state.buy_resistance_zone = state.resistance
                    else:
                        state.sell_support_zone = state.support
                else:
                    if result["signal"] == "BUY":
                        state.buy_resistance_zone = mid_price
                    else:
                        state.sell_support_zone = mid_price
            else:
                risk = retracement_range / USD1_TO_PRICE_DICT[SYMBOL]
                logging.info(f"RETRACEMENT RANGE IS TOO HIGH: {risk} *****************")

        # block any new sell or resistance zone until new SNR is detected
        state.active = False

def run():
    initialize_mt5()

    while True:
        df = get_data(mt5.TIMEFRAME_M15)
        df = add_indicators(df)
        df = detect_fractals(df)
        i = len(df) - 2
        
        # generate signal from any sell/buy breakout
        result = generate_signal(df, state, LOOKBACK)
        
        # update state with an active breakout sell/buy
        # only update new buy/signal if it has not yet triggered
        process_bo_signal(df, i, state, result)

        # buy/sell trigger condition 1: if the candle had closed and re-entry the breakout zone trigger  
        # buy/sell trigger condition 2: if any candle since the breakout had crossed any lower support level or higher resistance level  
        update_bmb_state(df=df, state=bmb_state)
        
        # update latest support and resistance before trigger any buy or sell
        # this is to check if the current support and resistance zone is still valid 
        update_state(result)

        # check for buy/sell zone if there's an active BO
        execute_bo_order(df=df, state=state, i=i)

        # execute bmb buy/sell zone if there's an active bmb
        execute_bmb_order(df=df, i=i, bmb_state=bmb_state)

        time.sleep(10)

if __name__ == "__main__":
    run()