import os
import sys
import MetaTrader5 as mt5
import pandas as pd
import time
from datetime import datetime
from ta.volatility import AverageTrueRange

SYMBOL = sys.argv[1]
MIN_SL_PIPS = float(sys.argv[2])
MAX_RISK_PCT = 0.01
ATR_MULTIPLIER = 1.8

if not mt5.initialize():
    print("❌ MT5 init failed")
    quit()

print("✅ MT5 connected")

account = int(os.environ["MT5_ACCOUNT"])
password = os.environ["MT5_PASSWORD"]
server = "Pepperstone-Demo"

USD1_DICT = {
    "XAUUSD": 1,
    "BTCUSD": 100,
    "GBPJPY": 0.1
}

if not mt5.login(account, password, server):
    print("❌ Login failed")
else:
    print("✅ Logged in")
    
def get_data(timeframe, bars=200):
    rates = mt5.copy_rates_from_pos(SYMBOL, timeframe, 0, bars)
    df = pd.DataFrame(rates)
    df['time'] = pd.to_datetime(df['time'], unit='s')
    return df

def calculate_trade_by_usd(symbol, entry, direction):
    print(f"{USD1_DICT[SYMBOL]}")
    risk = MIN_SL_PIPS * USD1_DICT[SYMBOL]

    # --- SL / TP ---
    if direction == "BUY":
        sl = entry - risk
        tps = [entry + risk * i for i in range(1, 6)]
    else:
        sl = entry + risk
        tps = [entry - risk * i for i in range(1, 6)]

    return entry, sl, tps


df = get_data(mt5.TIMEFRAME_M1)
atr = AverageTrueRange(df['high'], df['low'], df['close'], 14)
df['atr'] = atr.average_true_range()
last = df.iloc[-2]
entry, sl, tps = calculate_trade_by_usd(SYMBOL, last['close'], "BUY")

print(f"entry: {entry}")
print(f"sl: {sl}")
print(f"tp: {tps[0]}")



