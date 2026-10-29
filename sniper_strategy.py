import os
import MetaTrader5 as mt5
import pandas as pd
import time
from datetime import datetime

from ta.trend import EMAIndicator, MACD
from ta.momentum import RSIIndicator
from ta.volatility import AverageTrueRange


# -------------------------
# CONFIG
# -------------------------
SYMBOL = "BTCUSD"
TIMEFRAME = mt5.TIMEFRAME_M5
ATR_MULTIPLIER = 1.5
CROSS_TOLERANCE = 0.3

# -------------------------
# MT5 INIT
# -------------------------
if not mt5.initialize():
    print("❌ MT5 init failed")
    quit()

print("✅ MT5 connected")

account = int(os.environ["MT5_ACCOUNT"])
password = os.environ["MT5_PASSWORD"]
server = "Pepperstone-Demo"

if not mt5.login(account, password, server):
    print("❌ Login failed")
else:
    print("✅ Logged in")

# -------------------------
# GET DATA
# -------------------------
def get_data(symbol=SYMBOL, timeframe=TIMEFRAME, bars=200):
    rates = mt5.copy_rates_from_pos(symbol, timeframe, 0, bars)
    df = pd.DataFrame(rates)
    df['time'] = pd.to_datetime(df['time'], unit='s')
    return df


# -------------------------
# VWAP
# -------------------------
def add_vwap(df):
    tp = (df['high'] + df['low'] + df['close']) / 3
    vol = df['tick_volume']
    df['vwap'] = (tp * vol).cumsum() / vol.cumsum()
    return df


# -------------------------
# 5m RSI
# -------------------------
def get_5m_rsi(symbol=SYMBOL):
    rates = mt5.copy_rates_from_pos(symbol, mt5.TIMEFRAME_M5, 0, 100)
    df5 = pd.DataFrame(rates)
    rsi = RSIIndicator(df5['close'], window=14).rsi()
    return rsi.iloc[-1]


# -------------------------
# INDICATORS
# -------------------------
def add_indicators(df):
    df['ema9'] = EMAIndicator(df['close'], 9).ema_indicator()
    df['ema21'] = EMAIndicator(df['close'], 21).ema_indicator()

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
def compute_score(df):
    last = df.iloc[-1]
    rsi5m = get_5m_rsi()

    bScore = 0
    rScore = 0

    # VWAP
    if last['close'] > last['vwap']: bScore += 1
    else: rScore += 1

    # RSI
    if last['rsi'] > 50: bScore += 1
    else: rScore += 1

    # MACD
    if last['macd'] > last['macd_signal']: bScore += 1
    else: rScore += 1

    # EMA
    if last['ema9'] > last['ema21']: bScore += 1
    else: rScore += 1

    # Volume
    if last['tick_volume'] > last['vol_avg']:
        if last['close'] > last['open']:
            bScore += 1
        else:
            rScore += 1

    # 5m RSI
    if rsi5m > 50: bScore += 1
    else: rScore += 1

    bullPct = (bScore / 6) * 100
    bearPct = (rScore / 6) * 100

    return bullPct, bearPct

import math
# -------------------------
# SIGNAL LOGIC
# -------------------------
def generate_signal(df, last_state):
    prev = df.iloc[-2]
    last = df.iloc[-1]

    buy_cross = prev['ema9'] < prev['ema21'] and last['ema9'] > last['ema21']
    sell_cross = prev['ema9'] > prev['ema21'] and last['ema9'] < last['ema21']

    diff = last['ema9'] - last['ema21']
    prev_diff = prev['ema9'] - prev['ema21']

    isNear = abs(diff) < CROSS_TOLERANCE
    nearBuy = isNear and diff > 0 and diff > prev_diff
    nearSell = isNear and diff < 0 and diff < prev_diff

    print(f"buy cross {buy_cross}")
    print(f"sell cross {sell_cross}")
    
    bullPct, bearPct = compute_score(df)
    print(f"last state: {last_state}")

    triggerBuy = buy_cross and bullPct >= 30
    triggerSell = sell_cross and bearPct >= 30

    return triggerBuy, triggerSell, bullPct, bearPct


# -------------------------
# TRADE SETUP
# -------------------------
def calculate_trade(df, direction):
    last = df.iloc[-1]

    entry = last['close']
    atr = last['atr']
    risk = atr * ATR_MULTIPLIER

    if direction == "BUY":
        sl = entry - risk
        tps = [entry + risk * i for i in range(1, 6)]
    else:
        sl = entry + risk
        tps = [entry - risk * i for i in range(1, 6)]

    return entry, sl, tps

# -------------------------
# ENTRY
# -------------------------

def has_open_position(symbol=SYMBOL):
    positions = mt5.positions_get(symbol=symbol)
    return positions is not None and len(positions) > 0

def place_market_order(signal, entry, sl, tps):
    symbol = SYMBOL
    volume = 0.01  # adjust later

    tick = mt5.symbol_info_tick(symbol)

    if signal == "BUY":
        order_type = mt5.ORDER_TYPE_BUY
        price = tick.ask
        tp = tps[0]  # use TP1 for now

    elif signal == "SELL":
        order_type = mt5.ORDER_TYPE_SELL
        price = tick.bid
        tp = tps[0]

    else:
        return

    request = {
        "action": mt5.TRADE_ACTION_DEAL,
        "symbol": symbol,
        "volume": volume,
        "type": order_type,
        "price": price,
        "sl": sl,
        "tp": tp,
        "deviation": 10,
        "magic": 123456,
        "comment": "Sniper bot entry",
        "type_time": mt5.ORDER_TIME_GTC,
        "type_filling": mt5.ORDER_FILLING_IOC,
    }

    result = mt5.order_send(request)

    if result.retcode != mt5.TRADE_RETCODE_DONE:
        print("❌ Order failed:", result)
    else:
        print("✅ Order placed:", result)

# -------------------------
# LOGGING
# -------------------------
def log_trade(signal, entry, sl, tps, bullPct, bearPct):
    log = {
        "time": datetime.now(),
        "signal": signal,
        "entry": entry,
        "sl": sl,
        "tp1": tps[0],
        "tp2": tps[1],
        "tp3": tps[2],
        "tp4": tps[3],
        "tp5": tps[4],
        "bullPct": bullPct,
        "bearPct": bearPct
    }

    print("📊 TRADE:", log)

    df_log = pd.DataFrame([log])
    df_log.to_csv("trade_log.csv", mode='a', header=False, index=False)


# -------------------------
# MAIN LOOP
# -------------------------
last_state = 0

while True:
    try:
        df = get_data(bars=50)
        df = add_indicators(df)

        triggerBuy, triggerSell, bullPct, bearPct = generate_signal(df, last_state)

        if has_open_position():
            print("⚠️ Position already open → skipping")
        else:
            if triggerBuy:
                entry, sl, tps = calculate_trade(df, "BUY")
                last_state = 1

                print("🚀 BUY SIGNAL")
                place_market_order("BUY", entry, sl, tps)
                log_trade("BUY", entry, sl, tps, bullPct, bearPct)

            elif triggerSell == False:
                entry, sl, tps = calculate_trade(df, "SELL")
                last_state = -1

                print("🔻 SELL SIGNAL")
                place_market_order("SELL", entry, sl, tps)
                log_trade("SELL", entry, sl, tps, bullPct, bearPct)

            else:
                print(f"⏳ HOLD | Bull: {bullPct:.1f}% Bear: {bearPct:.1f}%")

        time.sleep(10)

    except Exception as e:
        print("❌ Error:", e)
        time.sleep(10)