import os
import MetaTrader5 as mt5
import sys

# initialize
if not mt5.initialize():
    print("❌ MT5 initialize failed")
    quit()

print("✅ MT5 connected")

account = int(os.environ["MT5_ACCOUNT"])
password = os.environ["MT5_PASSWORD"]
server = "Pepperstone-Demo"

'''if not mt5.login(account, password, server):
    print("❌ Login failed")
else:
    print("✅ Logged in")'''

#rates = mt5.copy_rates_from_pos("XAUUSD", mt5.TIMEFRAME_M15, 0, 100)
#print(rates)

from simple_xau_strategy import xau_strategy
from ta.momentum import RSIIndicator
from ta.trend import EMAIndicator
import pandas as pd

symbol = sys.argv[1]

rates = mt5.copy_rates_from_pos(symbol, mt5.TIMEFRAME_M15, 0, 100)

df = pd.DataFrame(rates)
df['time'] = pd.to_datetime(df['time'], unit='s')

print(df.tail())

# RSI
df['rsi'] = RSIIndicator(df['close'], window=14).rsi()

# EMA50
df['ema50'] = EMAIndicator(df['close'], window=50).ema_indicator()

print(df[['close', 'rsi', 'ema50']].tail())


decision = xau_strategy(df)
print(decision)