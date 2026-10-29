import os
import MetaTrader5 as mt5
import pandas as pd
import numpy as np
import time
from datetime import datetime
import sys

# ============================================================
# CONFIG
# ============================================================

SYMBOL = sys.argv[1]
TIMEFRAME = mt5.TIMEFRAME_H1

LOT = 0.01
MAGIC = 2024

PROB_THRESHOLD = 0.50
MAX_RISK_PER_TRADE = 0.01

MODEL_PATH = "bo_model.pkl"


# ============================================================
# INIT MT5
# ============================================================

#account = int(os.environ["MT5_ACCOUNT"])
#password = os.environ["MT5_PASSWORD"]
#server = "Pepperstone-MT5-Live01"
account = int(os.environ["MT5_ACCOUNT"])
password = os.environ["MT5_PASSWORD"]
server = "Pepperstone-Demo"

if not mt5.initialize():
    raise RuntimeError(mt5.last_error())

if not mt5.login(account, password, server):
    print("❌ Login failed")
else:
    print("✅ Logged in")

mt5.symbol_select(SYMBOL, True)


# ============================================================
# LOAD MODEL
# ============================================================

import joblib

artifact = joblib.load(MODEL_PATH)
model = artifact["model"]
FEATURES = artifact["features"]


# ============================================================
# FEATURE ENGINEERING (must match training)
# ============================================================

def build_features(df):

    df["range_size"] = df["high"] - df["low"]
    df["range_ma"] = df["range_size"].rolling(20).mean()
    df["range_compression"] = df["range_size"] / df["range_ma"]

    df["atr"] = df["range_size"].rolling(14).mean()

    # IMPORTANT: placeholders (must exist for model consistency)
    df["range_high"] = df["high"].rolling(20).max()
    df["range_low"] = df["low"].rolling(20).min()

    df["entry_long"] = 0
    df["entry_short"] = 0

    return df

def compute_features(df):

    df["range_size"] = df["high"] - df["low"]
    df["range_ma"] = df["range_size"].rolling(20).mean()
    df["range_compression"] = df["range_size"] / df["range_ma"]

    df["atr"] = df["range_size"].rolling(14).mean()

    df = df.dropna()
    return df


# ============================================================
# GET DATA
# ============================================================

def get_data(n=300):
    rates = mt5.copy_rates_from_pos(SYMBOL, TIMEFRAME, 0, n)
    df = pd.DataFrame(rates)

    df["time"] = pd.to_datetime(df["time"], unit="s")
    df.rename(columns={"tick_volume": "volume"}, inplace=True)

    return df


# ============================================================
# CHECK OPEN POSITION
# ============================================================

def has_position():
    pos = mt5.positions_get(symbol=SYMBOL)
    return pos is not None and len(pos) > 0


# ============================================================
# PLACE ORDER
# ============================================================

def place_order(direction, sl, tp):

    tick = mt5.symbol_info_tick(SYMBOL)

    if direction == "long":
        order_type = mt5.ORDER_TYPE_BUY
        price = tick.ask
    else:
        order_type = mt5.ORDER_TYPE_SELL
        price = tick.bid

    request = {
        "action": mt5.TRADE_ACTION_DEAL,
        "symbol": SYMBOL,
        "volume": LOT,
        "type": order_type,
        "price": price,
        "sl": sl,
        "tp": tp,
        "deviation": 20,
        "magic": MAGIC,
        "comment": "ML Breakout Bot",
        "type_time": mt5.ORDER_TIME_GTC,
        "type_filling": mt5.ORDER_FILLING_IOC,
    }

    result = mt5.order_send(request)
    print("ORDER:", result)
    return result


# ============================================================
# SIGNAL LOGIC (same as training)
# ============================================================

def generate_signal(row, proba):

    if proba < PROB_THRESHOLD:
        return None

    if row["range_compression"] > 1:
        return None

    if row["close"] > row["range_ma"]:
        return "long"

    if row["close"] < row["range_ma"]:
        return "short"

    return None


# ============================================================
# SL / TP ENGINE
# ============================================================

def calc_levels(row, direction):

    atr = row["atr"]
    price = row["close"]

    if direction == "long":
        sl = price - 1.5 * atr
        tp = price + 3.0 * atr
    else:
        sl = price + 1.5 * atr
        tp = price - 3.0 * atr

    return sl, tp

def predict_latest(df):

    df = build_features(df).dropna()

    X = df[FEATURES].iloc[-1:].copy()

    return model.predict_proba(X)[0][1]

# ============================================================
# MAIN LOOP
# ============================================================

last_bar_time = None

while True:

    df = get_data()
    
    latest = df.iloc[-1]

    current_time = latest["time"]

    # only act on new candle
    if last_bar_time == current_time:
        time.sleep(5)
        continue

    last_bar_time = current_time

    if has_position():
        print("Position exists — skipping")
        continue

    proba = predict_latest(df)

    signal = generate_signal(latest, proba)

    print(f"{current_time} | prob={proba:.2f} | signal={signal}")

    if signal is None:
        continue

    sl, tp = calc_levels(latest, signal)

    place_order(signal, sl, tp)

    time.sleep(10)