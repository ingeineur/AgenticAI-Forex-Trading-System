"""
Read MT5 trade history and show win/loss stats filtered by comment keyword.

Usage:
  python mt5_stats.py LIQ
  python mt5_stats.py SMC --exclude
  python mt5_stats.py ALL --days 30
"""
import os
import MetaTrader5 as mt5
import pandas as pd
import numpy as np
import sys
from datetime import datetime, timedelta

keyword_arg = sys.argv[1] if len(sys.argv) > 1 else input("Keyword (e.g. LIQ, SMC, or ALL): ").strip().upper()
EXCLUDE = False
if "--exclude" in sys.argv:
    EXCLUDE = True
    sys.argv.remove("--exclude")
    if keyword_arg == "ALL":
        keyword_arg = ""
elif keyword_arg == "ALL":
    keyword_arg = ""

days = 9999
if "--days" in sys.argv:
    idx = sys.argv.index("--days")
    days = int(sys.argv[idx + 1])
filter_symbol = None
if "--symbol" in sys.argv:
    idx = sys.argv.index("--symbol")
    filter_symbol = sys.argv[idx + 1].upper()

EXPORT = None
if "--export" in sys.argv:
    idx = sys.argv.index("--export")
    EXPORT = sys.argv[idx + 1] if idx + 1 < len(sys.argv) else f"trades_{days}d.csv"

if not mt5.initialize():
    print("ERROR: MT5 init failed"); sys.exit(1)

acc, pwd, srv = int(os.environ["MT5_ACCOUNT"]), os.environ["MT5_PASSWORD"], "PepperstoneBS-MT5-Live01"
if not mt5.login(acc, pwd, srv):
    print("Login failed"); mt5.shutdown(); sys.exit(1)

print(f"Logged in: {acc}")
exclude_s = " (EXCLUDE)" if EXCLUDE else ""
print(f"Filter: keyword='{keyword_arg}'{exclude_s}, last {days} days")

from_date = datetime.now() - timedelta(days=days)
to_date = datetime.now()

deals = mt5.history_deals_get(from_date, to_date)
if not deals:
    print("No deals"); mt5.shutdown(); sys.exit(1)

df = pd.DataFrame([d._asdict() for d in deals])
df["time"] = pd.to_datetime(df["time"], unit="s", utc=True)
df = df.sort_values("time").reset_index(drop=True)

# Find position_ids that match (or don't match) the keyword
# When EXCLUDE, only include positions where NO deal contains the keyword
if keyword_arg == "":
    match_pos_ids = df[df["position_id"] > 0]["position_id"].unique()
elif EXCLUDE:
    pos_with_kw = df[df["comment"].str.contains(keyword_arg, case=False, na=False)]["position_id"].unique()
    pos_with_kw = pos_with_kw[pos_with_kw > 0]
    all_pos = df[df["position_id"] > 0]["position_id"].unique()
    match_pos_ids = np.setdiff1d(all_pos, pos_with_kw)
else:
    match_pos_ids = df[df["comment"].str.contains(keyword_arg, case=False, na=False)]["position_id"].unique()
    match_pos_ids = match_pos_ids[match_pos_ids > 0]

print(f"\nMatched positions with '{keyword_arg}': {len(match_pos_ids)}")
if len(match_pos_ids) == 0:
    mt5.shutdown(); sys.exit(0)

# Get ALL deals for those position_ids
pos_df = df[df["position_id"].isin(match_pos_ids)].copy()
pos_df["direction"] = pos_df["type"].map({0: "BUY", 1: "SELL", 2: "BALANCE", 3: "BALANCE"})

if filter_symbol:
    pos_df = pos_df[pos_df["symbol"] == filter_symbol]
print(f"  Total deals for these positions: {len(pos_df)}")
print(f"  Date range: {pos_df['time'].min()} -> {pos_df['time'].max()}")
print(f"  Symbols:    {pos_df['symbol'].unique().tolist()}")

# Per-position stats
stats = pos_df.groupby("position_id").agg(
    symbol=("symbol", "first"),
    direction=("direction", "first"),
    entry_time=("time", "first"),
    exit_time=("time", "last"),
    total_pnl=("profit", "sum"),
    volume=("volume", "first"),
    comment=("comment", "first"),
).reset_index()

# Parse comment into level + zone categories
def classify_comment(c):
    if not isinstance(c, str): return ("OTHER", "OTHER")
    c = c.upper()
    parts = c.split()
    if len(parts) < 2: return ("OTHER", "OTHER")
    # New format: {short_id} {tf} MSS {Level_Dir} {Zone}   e.g. "6V9D1 M15 MSS SWING_H OB"
    if "MSS" in parts:
        # find MSS token; level = token after MSS, zone = token after level
        mi = parts.index("MSS")
        level = parts[mi + 1] if mi + 1 < len(parts) else ""
        zone = parts[mi + 2] if mi + 2 < len(parts) else ""
    else:
        # Old format: LIQ Level_Dir Zone MSS
        level = parts[1] if len(parts) > 1 else ""
        zone = parts[2] if len(parts) > 2 else ""
    # Normalize zone (handle MT5 truncation: Swept→S/Sw, OB→O, FVG→F)
    if zone.startswith("SW") or zone == "S": zone = "Swept"
    elif zone == "OB" or zone == "O": zone = "OB"
    elif zone == "FVG" or zone == "F": zone = "FVG"
    else: zone = "Other"
    return (level, zone)

stats[["level", "zone"]] = stats["comment"].apply(lambda c: pd.Series(classify_comment(c)))
stats["result"] = stats["total_pnl"].apply(lambda x: "WIN" if x > 0 else ("LOSS" if x < 0 else "BREAK"))

wins = (stats["total_pnl"] > 0).sum()
losses = (stats["total_pnl"] <= 0).sum()
total_pnl = stats["total_pnl"].sum()

print(f"\n  RESULTS:")
print(f"  {'='*40}")
print(f"  Positions:  {len(stats)}")
print(f"  Wins:       {wins}  ({wins/len(stats)*100:.1f}%)")
print(f"  Losses:     {losses}  ({losses/len(stats)*100:.1f}%)")
print(f"  Total PnL:  {total_pnl:+.2f}")
print(f"  Avg PnL:    {stats['total_pnl'].mean():+.2f}")
print(f"  Best:       {stats['total_pnl'].max():+.2f}")
print(f"  Worst:      {stats['total_pnl'].min():+.2f}")

win_df = stats[stats["total_pnl"] > 0]
loss_df = stats[stats["total_pnl"] <= 0]
if len(win_df): print(f"  Avg win:    {win_df['total_pnl'].mean():+.2f}")
if len(loss_df): print(f"  Avg loss:   {loss_df['total_pnl'].mean():+.2f}")
print(f"  Profit factor: {win_df['total_pnl'].sum()/abs(loss_df['total_pnl'].sum()):.2f}" if len(loss_df) and loss_df['total_pnl'].sum() != 0 else "")
print(f"  Expectancy: {total_pnl/len(stats):+.2f}")

print(f"\n  Per-symbol:")
for sym in sorted(stats["symbol"].unique()):
    sub = stats[stats["symbol"] == sym]
    if sym == "": continue
    wr = (sub["total_pnl"] > 0).mean() * 100
    pnl = sub["total_pnl"].sum()
    print(f"    {sym:8s}: {len(sub):4d} trades, {wr:.0f}% WR, PnL={pnl:+.2f}")

print(f"\n  Per setup type (level x zone):")
groups = stats.groupby(["level", "zone"])
for (lev, zn), sub in sorted(groups):
    wr = (sub["total_pnl"] > 0).mean() * 100
    pnl = sub["total_pnl"].sum()
    avg_w = sub[sub["total_pnl"] > 0]["total_pnl"].mean() if (sub["total_pnl"] > 0).any() else 0
    avg_l = sub[sub["total_pnl"] <= 0]["total_pnl"].mean() if (sub["total_pnl"] <= 0).any() else 0
    print(f"    {lev:10s} {zn:6s}: {len(sub):3d} trades, {wr:5.1f}% WR, PnL={pnl:+7.2f}, avgW={avg_w:+5.2f}, avgL={avg_l:+5.2f}")

# Also summarize by level only
print(f"\n  By level:")
for lev, sub in stats.groupby("level"):
    wr = (sub["total_pnl"] > 0).mean() * 100
    pnl = sub["total_pnl"].sum()
    print(f"    {lev:10s}: {len(sub):3d} trades, {wr:5.1f}% WR, PnL={pnl:+7.2f}")

print(f"\n  Recent trades:")
for _, r in stats.sort_values("exit_time").tail(15).iterrows():
    print(f"    {str(r['exit_time'])[:16]} | {r['symbol']:8s} | {r['direction']:4s} | "
          f"PnL={r['total_pnl']:+7.2f} | {r['result']:5s} | {r['comment'][:30]}")

# Optional: show all raw deals for a sample position
print(f"\n  Sample position {match_pos_ids[0]}:")
sample = pos_df[pos_df["position_id"] == match_pos_ids[0]]
for _, d in sample.iterrows():
    print(f"    {d['time']} | entry={d['entry']} | profit={d['profit']:+7.2f} | price={d['price']:.2f} | {d['comment']}")

# Export all matched trades to CSV (one row per position)
if EXPORT:
    out = stats[["position_id", "symbol", "direction", "entry_time", "exit_time",
                 "total_pnl", "volume", "level", "zone", "result", "comment"]].copy()
    out = out.rename(columns={"position_id": "position", "total_pnl": "pnl"})
    out.to_csv(EXPORT, index=False)
    print(f"\n  Exported {len(out)} trades -> {EXPORT}")

mt5.shutdown()
