"""
Accumulation detector — multi-TF bias + contracting ATR.

Usage:
    python accumulation.py XAUUSD
    python accumulation.py BTCUSD
"""
import os
import MetaTrader5 as mt5
import numpy as np
import sys
import pandas as pd

SYMBOL = sys.argv[1].upper() if len(sys.argv) > 1 else "XAUUSD"
STR = 2

def swings(h, l):
    n = len(h); sh = np.zeros(n, bool); sl = np.zeros(n, bool)
    for i in range(STR, n - STR):
        sh[i] = all(h[i] > h[i - k] and h[i] > h[i + k] for k in range(1, STR + 1))
        sl[i] = all(l[i] < l[i - k] and l[i] < l[i + k] for k in range(1, STR + 1))
    return sh, sl

def classify(h, l):
    """BULLISH / BEARISH / SIDEWAYS from last 2 swing points."""
    sh, sl = swings(h, l)
    si = np.where(sh | sl)[0]
    if len(si) < 2:
        return "SIDEWAYS"
    i0, i1 = si[-2], si[-1]
    p0 = h[i0] if sh[i0] else l[i0]
    p1 = h[i1] if sh[i1] else l[i1]
    if p1 > p0: return "BULLISH"
    if p1 < p0: return "BEARISH"
    return "SIDEWAYS"

def atr_contracting(h, l, c, period=14):
    tr = np.maximum(h - l, np.maximum(abs(h - np.roll(c, 1)), abs(l - np.roll(c, 1))))
    tr[0] = h[0] - l[0]
    n = len(tr)
    atr = np.zeros(n); atr[0] = tr[0]
    for i in range(1, n): atr[i] = (atr[i-1] * (period - 1) + tr[i]) / period
    recent = atr[-5:].mean()
    avg = atr[-min(period, n):].mean()
    return recent < avg * 0.7, recent, avg

if not mt5.initialize():
    print("MT5 init failed"); sys.exit(1)

ac, pw, sv = int(os.environ["MT5_ACCOUNT"]), os.environ["MT5_PASSWORD"], "Pepperstone-Demo"
if not mt5.login(ac, pw, sv):
    print("Login failed"); mt5.shutdown(); sys.exit(1)

mt5.symbol_select(SYMBOL, True)

TFS = [
    ("m15", mt5.TIMEFRAME_M15, 200),
    ("h1",  mt5.TIMEFRAME_H1,   50),
    ("h4",  mt5.TIMEFRAME_H4,   12),
]

results = {}
for tf_name, tf_mt5, bars in TFS:
    rates = mt5.copy_rates_from_pos(SYMBOL, tf_mt5, 1, bars)  # offset=1 skips current forming candle
    if rates is None or len(rates) < 10:
        results[tf_name] = ("NO DATA", "OK", 0, 0, bars)
        continue
    df = pd.DataFrame(rates)
    h = df['high'].values; l = df['low'].values; c = df['close'].values
    bias = classify(h, l)
    contr, r_atr, a_atr = atr_contracting(h, l, c)
    results[tf_name] = (bias, "TIGHT" if contr else "OK", r_atr, a_atr, bars)

print(f"\n  Multi-TF Accumulation Detector — {SYMBOL}")
print(f"  {'='*50}")
print(f"  {'TF':>5s}  {'BIAS':>8s}  {'ATR':>7s}  {'BARS':>5s}")
print(f"  {'-'*35}")

agree = []
for tf_name in [t[0] for t in TFS]:
    bias, atr_s, r_atr, a_atr, bars = results[tf_name]
    print(f"  {tf_name:>5s}  {bias:>8s}  {atr_s:>7s}  {bars:>5d}")
    if bias != "SIDEWAYS" and bias != "NO DATA":
        agree.append(bias)

print(f"  {'='*50}")

all_ok = all(r[0] not in ("SIDEWAYS", "NO DATA") for r in results.values())
all_agree = len(set(agree)) == 1 and len(agree) == 3 if all_ok else False
any_tight = any(r[1] == "TIGHT" for r in results.values())

if not all_ok:
    print(f"  VERDICT: ACCUMULATION (sideways or no data)")
elif not all_agree:
    print(f"  VERDICT: ACCUMULATION (mixed bias: {'/'.join(agree)})")
elif any_tight:
    print(f"  VERDICT: ACCUMULATION (contracting ATR)")
else:
    print(f"  VERDICT: DIRECTIONAL ({agree[0]})")

print(f"\n  ATR detail:")
for tf_name in [t[0] for t in TFS]:
    _, _, r_atr, a_atr, bars = results[tf_name]
    ratio = r_atr / a_atr * 100 if a_atr > 0 else 0
    print(f"    {tf_name:>5s}: recent={r_atr:.2f}  avg={a_atr:.2f}  ratio={ratio:.0f}%  {'<<< TIGHT' if ratio < 70 else ''}")

mt5.shutdown()
