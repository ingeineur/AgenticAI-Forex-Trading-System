"""Visualize how the MSS (Sweep -> MSS -> Zone) strategy works.

Reads sweep_mss_zone.py's REAL detection functions and renders them on
real XAUUSD M15 data:
  01_pipeline_overview.png - the 5-stage signal pipeline
  02_sweep_detection.png   - liquidity sweep + rejection-quality rule
  03_mss_shift.png         - market-structure-shift confirmation
  04_zones_entry.png       - FVG / OB / swept zones + entry, SL, TP

Usage: python visualize_mss_strategy.py
"""
import os
import sys

sys.argv = ['sweep_mss_zone.py', 'XAUUSD']  # neutralize module-level argparse
import sweep_mss_zone as S

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as patches

OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'mss_visuals')
os.makedirs(OUT, exist_ok=True)

CSV = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'XAUUSD_M15_combined.csv')
FRACTAL = 2
SL_ATR = 2.0
TP_ATR = 1.0


def load_data(n_tail=3000):
    df = pd.read_csv(CSV)
    df['timestamp'] = pd.to_datetime(df['timestamp'], utc=True)
    df = df.sort_values('timestamp').reset_index(drop=True)
    df = df.tail(n_tail).reset_index(drop=True)
    if 'volume' in df.columns and 'tick_volume' not in df.columns:
        df['tick_volume'] = df['volume']
    df = S.add_indicators(df)
    return df.dropna(subset=['atr']).reset_index(drop=True)


def find_example(df, direction="BUY"):
    """Find a real sweep+MSS setup with FVG and OB zones, using only data
    available at each step (slices), mimicking live behavior."""
    n = len(df)
    for i in range(250, n - 30):
        sub = df.iloc[:i + 1]
        bar_ts = df['timestamp'].iloc[i]
        levels = S.get_ssl_levels(sub, bar_ts) if direction == "BUY" else S.get_bsl_levels(sub, bar_ts)
        for lev_name, level in levels:
            if level is None or level <= 0:
                continue
            if not S.detect_sweep(sub, i, level, direction):
                continue
            if not S.sweep_quality_ok(sub, i, level, direction):
                continue
            df2 = df.iloc[:i + 4]  # MSS lookahead window
            mss_ok, mss_type, mss_price = S.detect_mss(df2, i, direction)
            if not mss_ok:
                continue
            df3 = df.iloc[:i + 13]  # zone lookahead window
            fvg_l, fvg_h = S.find_fvg(df3, i, direction)
            ob_l, ob_h, _ = S.find_order_block(df3, i, direction)
            if fvg_l is None or ob_l is None:
                continue
            return dict(sweep_idx=i, level=level, lev_name=lev_name,
                        mss_type=mss_type, mss_price=mss_price,
                        fvg=(fvg_l, fvg_h), ob=(ob_l, ob_h))
    return None


def draw_candles(ax, df, idx, colors=('up', 'down')):
    up = '#26a69a'
    dn = '#ef5350'
    for k, j in enumerate(idx):
        o, h, l, c = (df['open'].iloc[j], df['high'].iloc[j],
                      df['low'].iloc[j], df['close'].iloc[j])
        col = up if c >= o else dn
        ax.plot([k, k], [l, h], color=col, lw=1.2)
        ax.add_patch(patches.Rectangle((k - 0.35, min(o, c)), 0.7, max(abs(c - o), 1e-9),
                                       facecolor=col, edgecolor=col))


def xlabels(ax, df, idx, step=3):
    xs = np.arange(len(idx))
    ax.set_xticks(xs[::step])
    ax.set_xticklabels(
        [pd.Timestamp(df['timestamp'].iloc[j]).strftime('%m-%d %H:%M') for j in idx[::step]],
        rotation=20, fontsize=8)


def fig_pipeline():
    fig, ax = plt.subplots(figsize=(12, 4.2))
    ax.set_xlim(0, 12)
    ax.set_ylim(0, 4.2)
    ax.axis('off')
    stages = [
        ("1\nLIQUIDITY\nLEVEL", "PDH/PDL +\nswing H/L"),
        ("2\nSWEEP", "wick > level\n+ 0.1 ATR\nclose back\n>50% body"),
        ("3\nMSS", "break prior\nswing in\n<=3 bars"),
        ("4\nZONES", "FVG gap\nOB body\nswept ±0.25 ATR"),
        ("5\nENTRY", "touch zone\nSL 2.0 ATR\nTP 1.0 ATR"),
    ]
    for k, (title, sub) in enumerate(stages):
        x = 0.6 + k * 2.28
        ax.add_patch(patches.FancyBboxPatch((x, 1.0), 1.8, 2.0, boxstyle="round,pad=0.08",
                                            facecolor='#1e2a38', edgecolor='#4da3ff', lw=1.6))
        ax.text(x + 0.9, 2.45, title, ha='center', va='center', fontsize=10,
                weight='bold', color='white')
        ax.text(x + 0.9, 1.55, sub, ha='center', va='center', fontsize=8, color='#a9c4de')
        if k < 4:
            ax.annotate("", xy=(x + 2.28, 2.0), xytext=(x + 1.8, 2.0),
                        arrowprops=dict(arrowstyle='->', color='#4da3ff', lw=2))
    ax.text(6, 3.7, "MSS strategy: Sweep -> Structure Shift -> Zone Entry (sweep_mss_zone.py)",
            ha='center', fontsize=12, weight='bold')
    ax.text(6, 0.45, "Filters: session, volume 1.2x, SuperTrend, PVT divergence, orderflow, accumulation/trend, reversal-prob, ML score",
            ha='center', fontsize=8, color='#888888')
    fig.tight_layout()
    fig.savefig(os.path.join(OUT, '01_pipeline_overview.png'), dpi=130)
    plt.close(fig)


def fig_sweep(df, ex, direction):
    si = ex['sweep_idx']
    idx = list(range(si - 15, si + 6))
    level = ex['level']
    atr = df['atr'].iloc[si]
    fig, ax = plt.subplots(figsize=(11, 5.2))
    draw_candles(ax, df, idx)
    xs = np.arange(len(idx))
    ax.axhline(level, color='#ffcf4d', ls='--', lw=1.6, label=f"Liquidity level ({ex['lev_name']}) @ {level:.2f}")
    ax.axhspan(level - 0.1 * atr, level + 0.1 * atr, color='#ffcf4d', alpha=0.12,
               label='Sweep buffer (0.1 ATR)')
    k_sw = 15
    row = df.iloc[si]
    ax.axvspan(k_sw - 0.5, k_sw + 0.5, color='white', alpha=0.07)
    ax.annotate('sweep wick\npierces level', xy=(k_sw, row['low']), xytext=(max(0, k_sw - 9), row['low'] + 0.4 * atr),
                arrowprops=dict(arrowstyle='->', color='white'), color='white', fontsize=9,
                bbox=dict(boxstyle='round', fc='#333333', ec='white'))
    ax.annotate('close back\ninside (>50% body)', xy=(k_sw, row['close']), xytext=(max(0, k_sw - 9), row['close'] + 2.5 * atr),
                arrowprops=dict(arrowstyle='->', color='white'), color='white', fontsize=9,
                bbox=dict(boxstyle='round', fc='#333333', ec='white'))
    ax.set_title(f"Stage 2 - Liquidity sweep + rejection quality ({direction}, bar {si})")
    xlabels(ax, df, idx)
    ax.legend(fontsize=8, loc='best')
    fig.tight_layout()
    fig.savefig(os.path.join(OUT, '02_sweep_detection.png'), dpi=130)
    plt.close(fig)


def fig_mss(df, ex, direction):
    si = ex['sweep_idx']
    idx = list(range(si - 15, si + 8))
    fig, ax = plt.subplots(figsize=(11, 5.2))
    draw_candles(ax, df, idx)
    xs = np.arange(len(idx))
    ax.axhline(ex['level'], color='#ffcf4d', ls='--', lw=1.2, label=f"swept level @ {ex['level']:.2f}")
    # mark MSS break candle: first candle after sweep breaking mss_price
    mp = ex['mss_price']
    ax.axhline(mp, color='#4da3ff', ls=':', lw=1.6, label=f"prior swing ({ex['mss_type']}) @ {mp:.2f}")
    k_found = None
    for k, j in enumerate(idx):
        if j <= si:
            continue
        if direction == "BUY" and df['high'].iloc[j] > mp:
            k_found = k
            break
        if direction == "SELL" and df['low'].iloc[j] < mp:
            k_found = k
            break
    ax.axvspan(15 - 0.5, 15 + 0.5, color='white', alpha=0.07)
    if k_found is not None:
        j = idx[k_found]
        y = df['high'].iloc[j] if direction == "BUY" else df['low'].iloc[j]
        tx = min(k_found + 2.0, len(idx) - 5)
        ax.annotate('MSS break\n(structure shift)', xy=(k_found, y), xytext=(tx, y + 1.5 * df['atr'].iloc[j]),
                    arrowprops=dict(arrowstyle='->', color='white'), color='white', fontsize=9,
                    bbox=dict(boxstyle='round', fc='#333333', ec='white'))
        ax.plot([k_found], [y], marker='o', color='white', ms=7, mfc='none', mew=2)
    ax.set_title(f"Stage 3 - Market structure shift confirms the sweep ({direction})")
    xlabels(ax, df, idx)
    ax.legend(fontsize=8, loc='best')
    fig.tight_layout()
    fig.savefig(os.path.join(OUT, '03_mss_shift.png'), dpi=130)
    plt.close(fig)


def fig_zones(df, ex, direction):
    si = ex['sweep_idx']
    idx = list(range(si - 6, si + 16))
    atr = df['atr'].iloc[si]
    fig, ax = plt.subplots(figsize=(11, 5.6))
    draw_candles(ax, df, idx)
    k0 = 6  # position of sweep bar
    # swept zone
    ax.axhspan(ex['level'] - 0.25 * atr, ex['level'] + 0.25 * atr,
               color='#9e9e9e', alpha=0.25, label='Swept zone (±0.25 ATR)')
    fvg_l, fvg_h = ex['fvg']
    ax.axhspan(fvg_l, fvg_h, color='#4da3ff', alpha=0.30, label=f'FVG [{fvg_l:.2f}-{fvg_h:.2f}]')
    ob_l, ob_h = ex['ob']
    ax.axhspan(ob_l, ob_h, color='#ab47bc', alpha=0.30, label=f'OB [{ob_l:.2f}-{ob_h:.2f}]')
    # entry = FVG mid (strategy checks FVG first), SL/TP
    ep = (fvg_l + fvg_h) / 2
    if direction == "BUY":
        sl, tp = ep - SL_ATR * atr, ep + TP_ATR * atr
    else:
        sl, tp = ep + SL_ATR * atr, ep - TP_ATR * atr
    ax.axhline(ep, color='white', lw=2.2, label=f'entry ~ {ep:.2f}', zorder=5)
    ax.axhline(sl, color='#ef5350', ls='--', lw=1.4, label=f'SL ({SL_ATR} ATR)')
    ax.axhline(tp, color='#26a69a', ls='--', lw=1.4, label=f'TP ({TP_ATR} ATR)')
    ax.set_title(f"Stage 4-5 - Zones (FVG > OB > Swept) + entry/SL/TP ({direction})")
    xlabels(ax, df, idx, step=4)
    ax.legend(fontsize=8, loc='best', ncol=2)
    fig.tight_layout()
    fig.savefig(os.path.join(OUT, '04_zones_entry.png'), dpi=130)
    plt.close(fig)
    return ep, sl, tp


def main():
    print(f"loading {CSV}")
    df = load_data()
    print(f"bars: {len(df)} ({df['timestamp'].iloc[0]} -> {df['timestamp'].iloc[-1]})")
    ex = find_example(df, "BUY") or find_example(df, "SELL")
    if not ex:
        print("no FVG+OB example found in window")
        sys.exit(1)
    direction = "BUY" if (find_example(df, "BUY") is not None) else "SELL"
    ex = find_example(df, direction)
    print(f"example: {direction} sweep bar={ex['sweep_idx']} level={ex['lev_name']}@{ex['level']:.2f} "
          f"mss={ex['mss_type']}@{ex['mss_price']:.2f} "
          f"fvg=[{ex['fvg'][0]:.2f},{ex['fvg'][1]:.2f}] ob=[{ex['ob'][0]:.2f},{ex['ob'][1]:.2f}]")
    fig_pipeline()
    print("wrote 01_pipeline_overview.png")
    fig_sweep(df, ex, direction)
    print("wrote 02_sweep_detection.png")
    fig_mss(df, ex, direction)
    print("wrote 03_mss_shift.png")
    ep, sl, tp = fig_zones(df, ex, direction)
    print(f"wrote 04_zones_entry.png (entry={ep:.2f} sl={sl:.2f} tp={tp:.2f})")
    print(f"all images in {OUT}")


if __name__ == '__main__':
    main()
