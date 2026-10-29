"""Test the new get_accumulation_state with trend scoring.

Usage:
    python test_trend_accum.py XAUUSD h1
    python test_trend_accum.py BTCUSD m15
    python test_trend_accum.py XAUUSD h4
"""
import os
import MetaTrader5 as mt5
import pandas as pd
import numpy as np
import sys
sys.path.insert(0, r'C:\tradingview')

TF_MAP = {"m15": mt5.TIMEFRAME_M15, "h1": mt5.TIMEFRAME_H1, "h4": mt5.TIMEFRAME_H4}
SYMBOL = sys.argv[1].upper() if len(sys.argv) > 1 else "XAUUSD"
TF = sys.argv[2].lower() if len(sys.argv) > 2 else "m15"
tf_mt5 = TF_MAP.get(TF, mt5.TIMEFRAME_M15)
BARS = 1000

sys.argv = ['test', SYMBOL]  # prevent smc_strategy arg parsing from failing

if not mt5.initialize():
    print('MT5 init failed'); exit()

mt5.login(int(os.environ["MT5_ACCOUNT"]), os.environ["MT5_PASSWORD"], 'Pepperstone-Demo')
mt5.symbol_select(SYMBOL, True)

rates = mt5.copy_rates_from_pos(SYMBOL, tf_mt5, 1, BARS)
mt5.shutdown()
if rates is None: print('No data'); exit()

df = pd.DataFrame(rates)
df['timestamp'] = pd.to_datetime(df['time'], unit='s')
df['body'] = abs(df['close'] - df['open'])
from ta.volatility import AverageTrueRange
atr = AverageTrueRange(df['high'], df['low'], df['close'], 14)
df['atr'] = atr.average_true_range()

from smc_strategy import get_accumulation_state, resample_tf

is_acc, dom_bias = get_accumulation_state(df)
print(f'\n{SYMBOL} {TF} ({len(df)} bars)')
print(f'  is_accumulating: {is_acc}')
print(f'  dominant_bias:   {dom_bias}')

