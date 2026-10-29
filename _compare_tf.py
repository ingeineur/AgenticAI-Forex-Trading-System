"""Compare M15 vs H1 for both strategies over the same date range."""
import pandas as pd
import numpy as np
import importlib.util
import sys

def load_strategy(path):
    spec = importlib.util.spec_from_file_location("strat", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod

def trim_data(m15_csv, h1_csv):
    m15 = pd.read_csv(m15_csv)
    if 'volume' in m15.columns and 'tick_volume' not in m15.columns:
        m15['tick_volume'] = m15['volume']
    if 'real_volume' in m15.columns:
        m15['volume'] = m15['real_volume']
    m15['timestamp'] = pd.to_datetime(m15['timestamp'], utc=True)
    h1 = pd.read_csv(h1_csv)
    h1['timestamp'] = pd.to_datetime(h1['timestamp'], utc=True)
    start, end = h1['timestamp'].min(), h1['timestamp'].max()
    m15 = m15[(m15['timestamp'] >= start) & (m15['timestamp'] <= end)].reset_index(drop=True)
    print(f"  M15: {len(m15)} bars, H1: {len(h1)} bars")
    print(f"  Range: {start} -> {end}")
    return m15, h1

# We'll just call the scripts directly via subprocess
import subprocess

pairs = [
    ("XAUUSD", "XAUUSD_M15_combined.csv", "XAUUSD_H1_combined.csv", 101793, 22673),
    ("BTCUSD", "BTCUSD_15_combined.csv", "BTCUSD_H1_combined.csv", 100419, 32592),
]

for sym, m15_csv, h1_csv, m15_total, h1_total in pairs:
    m15 = pd.read_csv(m15_csv)
    if 'volume' in m15.columns and 'tick_volume' not in m15.columns:
        m15['tick_volume'] = m15['volume']
    if 'real_volume' in m15.columns:
        m15['volume'] = m15['real_volume']
    m15['timestamp'] = pd.to_datetime(m15['timestamp'], utc=True)
    h1 = pd.read_csv(h1_csv)
    h1['timestamp'] = pd.to_datetime(h1['timestamp'], utc=True)
    start, end = h1['timestamp'].min(), h1['timestamp'].max()
    trimmed = m15[(m15['timestamp'] >= start) & (m15['timestamp'] <= end)].reset_index(drop=True)
    print(f"\n{sym}: M15 bars={len(trimmed)}, H1 bars={len(h1)} (same range)")
