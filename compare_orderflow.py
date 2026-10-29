"""Compare OHLC approx vs tick-rule delta on Pepperstone demo."""
import os
import MetaTrader5 as mt5
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from datetime import datetime

# --- CONFIG ---
LOGIN = int(os.environ["MT5_ACCOUNT"])
PASSWORD = os.environ["MT5_PASSWORD"]
SERVER = "Pepperstone-Demo"
SYMBOL = "XAUUSD"
TF_STR = "h1"  # h1 or m15
N_BARS = 100

# --- INIT ---
if not mt5.initialize():
    print("initialize() failed"); exit()
if not mt5.login(LOGIN, PASSWORD, SERVER):
    print(f"login() failed: {mt5.last_error()}"); mt5.shutdown(); exit()
print(f"Connected: {SERVER} account {LOGIN}")
mt5.symbol_select(SYMBOL, True)

TF_MAP = {"m15": mt5.TIMEFRAME_M15, "h1": mt5.TIMEFRAME_H1}
tf = TF_MAP[TF_STR]

# --- FETCH BARS ---
rates = mt5.copy_rates_from_pos(SYMBOL, tf, 0, N_BARS)
if rates is None:
    print("No rates"); mt5.shutdown(); exit()
df = pd.DataFrame(rates)
df['time'] = pd.to_datetime(df['time'], unit='s')
print(f"Bars: {len(df)} from {df['time'].iloc[0]} to {df['time'].iloc[-1]}")

# --- FETCH TICKS ---
t0 = df['time'].iloc[0].to_pydatetime()
t1 = df['time'].iloc[-1].to_pydatetime()
ticks = mt5.copy_ticks_range(SYMBOL, t0, t1, mt5.COPY_TICKS_ALL)
if ticks is None or len(ticks) == 0:
    print(f"No ticks: {mt5.last_error()}"); mt5.shutdown(); exit()
print(f"Ticks: {len(ticks)}")
mt5.shutdown()

# --- TICK-RULE DELTA (per bar) ---
bar_times = rates['time']
n = len(bar_times)
tickrule_buy = np.zeros(n)
tickrule_sell = np.zeros(n)
tick_idx = 0
prev_bid = None
for bar_idx in range(n - 1):
    bs = bar_times[bar_idx]
    be = bar_times[bar_idx + 1]
    while tick_idx < len(ticks) and ticks[tick_idx]['time'] < bs:
        tick_idx += 1
    while tick_idx < len(ticks) and ticks[tick_idx]['time'] < be:
        bid = ticks[tick_idx]['bid']
        if bid > 0 and prev_bid is not None:
            if bid > prev_bid:
                tickrule_buy[bar_idx] += 1
            elif bid < prev_bid:
                tickrule_sell[bar_idx] += 1
        prev_bid = bid
        tick_idx += 1
# Last bar uses remaining ticks
while tick_idx < len(ticks):
    bid = ticks[tick_idx]['bid']
    if bid > 0 and prev_bid is not None:
        if bid > prev_bid:
            tickrule_buy[-1] += 1
        elif bid < prev_bid:
            tickrule_sell[-1] += 1
    prev_bid = bid
    tick_idx += 1

# --- OHLC APPROX DELTA ---
eps = 1e-10
total = df['high'].values - df['low'].values + eps
ohlc_buy = df['tick_volume'].values * (df['close'].values - df['low'].values) / total
ohlc_sell = df['tick_volume'].values * (df['high'].values - df['close'].values) / total

# --- RESULTS DF ---
res = pd.DataFrame({
    'time': df['time'],
    'open': df['open'], 'high': df['high'], 'low': df['low'], 'close': df['close'],
    'volume': df['tick_volume'],
    'ohlc_buy': ohlc_buy, 'ohlc_sell': ohlc_sell,
    'ohlc_delta': ohlc_buy - ohlc_sell,
    'tr_buy': tickrule_buy, 'tr_sell': tickrule_sell,
    'tr_delta': tickrule_buy - tickrule_sell,
})

# Normalize deltas to [0,1] for bar colour comparison
dmax = max(abs(res['ohlc_delta']).max(), abs(res['tr_delta']).max())
res['ohlc_norm'] = res['ohlc_delta'] / dmax if dmax > 0 else 0
res['tr_norm'] = res['tr_delta'] / dmax if dmax > 0 else 0

# Agreement: same direction (positive/negative/zero)
res['ohlc_dir'] = np.sign(res['ohlc_delta'])
res['tr_dir'] = np.sign(res['tr_delta'])
agreement = (res['ohlc_dir'] == res['tr_dir']).sum()
total_bars = len(res)
pct = agreement / total_bars * 100
corr = res['ohlc_delta'].corr(res['tr_delta'])

# --- SUMMARY ---
print(f"\n{'='*60}")
print(f"Comparison: OHLC approx vs Tick-rule delta")
print(f"Symbol: {SYMBOL}  TF: {TF_STR}  Bars: {total_bars}  Ticks: {len(ticks)}")
print(f"{'='*60}")
print(f"Direction agreement: {agreement}/{total_bars} ({pct:.1f}%)")
print(f"Delta correlation:   {corr:.3f}")
print(f"\nOHLC delta:   mean={res['ohlc_delta'].mean():+.1f}  std={res['ohlc_delta'].std():.1f}")
print(f"Tick-rule delta: mean={res['tr_delta'].mean():+.1f}  std={res['tr_delta'].std():.1f}")
print(f"Tick-rule buy/split: {res['tr_buy'].sum():.0f}/{res['tr_sell'].sum():.0f} = {(res['tr_buy'].sum()/(res['tr_buy'].sum()+res['tr_sell'].sum())*100):.1f}% buy")
print(f"\nBar-by-bar (first 10 + last 5):")
print(f"{'Time':<20} {'Vol':>6} {'OHLCd':>8} {'TRd':>8} {'Match':>6}")
print("-"*48)
for i in range(min(10, total_bars)):
    m = "Y" if res['ohlc_dir'].iloc[i] == res['tr_dir'].iloc[i] else "N"
    print(f"{res['time'].iloc[i]!s:<20} {res['volume'].iloc[i]:>6} {res['ohlc_delta'].iloc[i]:>+8.0f} {res['tr_delta'].iloc[i]:>+8.0f} {m:>6}")
if total_bars > 15:
    print("...")
    for i in range(max(10, total_bars-5), total_bars):
        m = "Y" if res['ohlc_dir'].iloc[i] == res['tr_dir'].iloc[i] else "N"
        print(f"{res['time'].iloc[i]!s:<20} {res['volume'].iloc[i]:>6} {res['ohlc_delta'].iloc[i]:>+8.0f} {res['tr_delta'].iloc[i]:>+8.0f} {m:>6}")

# --- PLOTS ---
fig, axes = plt.subplots(4, 1, figsize=(14, 12), sharex=True)

# 1. Candles with OHLC delta colour
ax = axes[0]
colors = ['g' if res['ohlc_delta'].iloc[i] >= 0 else 'r' for i in range(total_bars)]
for i in range(total_bars):
    ax.plot([res['time'].iloc[i], res['time'].iloc[i]],
            [res['low'].iloc[i], res['high'].iloc[i]], color='gray', linewidth=0.5)
    if res['close'].iloc[i] >= res['open'].iloc[i]:
        ax.bar(res['time'].iloc[i], res['close'].iloc[i] - res['open'].iloc[i],
               bottom=res['open'].iloc[i], width=0.6*3600, color=colors[i], edgecolor='k', linewidth=0.3)
    else:
        ax.bar(res['time'].iloc[i], res['close'].iloc[i] - res['open'].iloc[i],
               bottom=res['open'].iloc[i], width=0.6*3600, color=colors[i], edgecolor='k', linewidth=0.3)
ax.set_title(f'{SYMBOL} {TF_STR} — Bar colour = OHLC delta direction (green ≥ 0, red < 0)')
ax.set_ylabel('Price')
ax.grid(True, alpha=0.3)

# 2. Comparison bars: OHLC delta (grey outline) vs Tick-rule delta (fill)
ax = axes[1]
x = range(total_bars)
ax.bar(x, res['ohlc_delta'], width=0.35, color='gray', alpha=0.5, label='OHLC approx')
ax.bar(x, res['tr_delta'], width=0.35, color='blue', alpha=0.3, label='Tick-rule')
ax.axhline(0, color='k', linewidth=0.5)
ax.set_ylabel('Delta')
ax.set_title('OHLC approx (gray) vs Tick-rule (blue)')
ax.legend()
ax.grid(True, alpha=0.3)

# 3. Scatter: OHLC delta vs Tick-rule delta
ax = axes[2]
ax.scatter(res['ohlc_delta'], res['tr_delta'], alpha=0.5, s=10)
lim = max(abs(res['ohlc_delta']).max(), abs(res['tr_delta']).max()) * 1.1
ax.plot([-lim, lim], [-lim, lim], 'k--', linewidth=0.5, label='y=x')
ax.set_xlabel('OHLC delta')
ax.set_ylabel('Tick-rule delta')
ax.set_title(f'Scatter — Direction agreement: {pct:.1f}%  Correlation: {corr:.3f}')
ax.legend()
ax.grid(True, alpha=0.3)
ax.axis('equal')

# 4. Tick volume vs Tick-rule tick count
ax = axes[3]
ax.bar(x, res['volume'], width=0.35, color='orange', alpha=0.5, label='Tick volume')
ax2 = ax.twinx()
ax2.bar(x, res['tr_buy'] + res['tr_sell'], width=0.35, color='blue', alpha=0.3, label='Tick count')
ax.set_ylabel('Tick volume')
ax2.set_ylabel('Tick count')
ax.set_title('Tick volume (orange) vs Tick count (blue) — relationship: volume vs bid moves')
ax.set_xlabel('Bar index')
ax.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig(f'orderflow_compare_{SYMBOL}_{TF_STR}.png', dpi=150)
print(f"\nPlot saved: orderflow_compare_{SYMBOL}_{TF_STR}.png")
plt.show()
