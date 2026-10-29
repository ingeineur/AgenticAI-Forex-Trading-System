#!/usr/bin/env python3
"""Full comparison: binary vs ternary, baseline vs retracement, on full CSV data."""
import numpy as np, pandas as pd, pickle, warnings, sys
warnings.filterwarnings('ignore')
eps = 1e-9

DATA = {
    "XAUUSD": ("XAUUSD_M15_combined.csv", "XAUUSD_16388_combined.csv", "ml_xau_m15h4_v2.pkl"),
    "BTCUSD": ("BTCUSD_15_combined.csv", "BTCUSD_16388_combined.csv", "ml_btc_m15h4_v2.pkl"),
}

MODEL_OVERRIDES = {
    "ternary":  lambda f: (f.replace(".pkl","_ternary.pkl"), None),
    "2r1r":     lambda f: (f.replace("-v2.","-v2_r2r1.").replace("_v2.","_v2_r2r1.").replace("v2.pkl","v2_r2r1.pkl"), None),
}

def build_features(m15, h4_merged):
    o=m15["open"].to_numpy(); h=m15["high"].to_numpy(); l=m15["low"].to_numpy()
    c=m15["close"].to_numpy(); v=m15["volume"].to_numpy()
    h4_c=h4_merged["h4_close"].to_numpy(); h4_o=h4_merged["h4_open"].to_numpy()
    h4_h=h4_merged["h4_high"].to_numpy(); h4_l=h4_merged["h4_low"].to_numpy()
    atr_vals=pd.Series(h-l).rolling(14).mean().to_numpy()
    n=len(m15); feat=np.zeros((n,28),dtype=np.float32)
    feat[:,0]=np.abs(c-o)/(atr_vals+eps); feat[:,1]=(h-l)/(atr_vals+eps)
    feat[:,2]=(h-np.maximum(o,c))/(atr_vals+eps); feat[:,3]=(np.minimum(o,c)-l)/(atr_vals+eps)
    ema20=pd.Series(c).ewm(span=20,adjust=False).mean().to_numpy()
    ema50=pd.Series(c).ewm(span=50,adjust=False).mean().to_numpy()
    feat[:,4]=(c-ema20)/(atr_vals+eps); feat[:,5]=(c-ema50)/(atr_vals+eps)
    feat[:,6]=np.diff(ema20,prepend=ema20[0])/(atr_vals+eps)
    feat[:,7]=np.diff(ema50,prepend=ema50[0])/(atr_vals+eps)
    feat[:,8]=v/(pd.Series(v).rolling(20).mean().to_numpy()+eps)
    feat[:,9]=pd.Series(c).pct_change().rolling(20).std().to_numpy()
    feat[:,10]=pd.Series(c).pct_change().fillna(0.0).to_numpy()
    feat[:,11]=c/np.roll(c,5)-1.0; feat[:,12]=(np.roll(c,1)>np.roll(o,1)).astype(float)
    feat[:,13]=(np.roll(c,2)>np.roll(o,2)).astype(float)
    feat[:,14]=c>(pd.Series(h).rolling(5).max()+pd.Series(l).rolling(5).min()).to_numpy()/2.0
    feat[:,15]=atr_vals/(c+eps); feat[:,16]=((c-o)>0).astype(float)
    feat[:,17]=np.abs(c-o)/(h-l+eps)
    rh=pd.Series(h).rolling(10).max().to_numpy(); rl=pd.Series(l).rolling(10).min().to_numpy()
    feat[:,18]=(c-rl)/(rh-rl+eps); feat[:,19]=v/(pd.Series(v).rolling(50).mean().to_numpy()+eps)
    h4_atr=pd.Series(h4_h-h4_l).rolling(14).mean().to_numpy()
    h4_e20=pd.Series(h4_c).ewm(span=20,adjust=False).mean().to_numpy()
    h4_e50=pd.Series(h4_c).ewm(span=50,adjust=False).mean().to_numpy()
    h4_sh=pd.Series(h4_h).rolling(10).max().to_numpy(); h4_sl=pd.Series(h4_l).rolling(10).min().to_numpy()
    feat[:,20]=np.abs(h4_o-h4_c)/(h4_atr+eps); feat[:,21]=(h4_h-h4_l)/(h4_atr+eps)
    feat[:,22]=h4_e20; feat[:,23]=h4_e50
    feat[:,24]=(h4_c-h4_e20)/(h4_atr+eps); feat[:,25]=np.sign(h4_e20-h4_e50).astype(float)
    feat[:,26]=(h4_c-h4_sl)/(h4_sh-h4_sl+eps); feat[:,27]=c/h4_c-1.0
    return feat, atr_vals

def load_artefact(path):
    with open(path, "rb") as f: return pickle.load(f)

def run_backtest(symbol, profile, approach, retracement=False, max_wait=5, threshold=0.55):
    m15f, h4f, base_model = DATA[symbol]
    m15 = pd.read_csv(m15f); m15["timestamp"] = pd.to_datetime(m15["timestamp"], utc=True)
    h4  = pd.read_csv(h4f);  h4["timestamp"]  = pd.to_datetime(h4["timestamp"], utc=True)
    h4  = h4.rename(columns={c: f"h4_{c}" for c in h4.columns if c != "timestamp"})

    # determine model path
    if profile == "2r1r":
        model_path = base_model.replace("v2.pkl", "v2_r2r1.pkl")
    else:
        model_path = base_model
    if approach == "ternary":
        model_path = model_path.replace(".pkl", "_ternary.pkl")

    art = load_artefact(model_path)
    ml_model = art["model"]

    # merge data
    m15["ts_round"] = m15["timestamp"].dt.floor("4h")
    h4["ts_round"]  = h4["timestamp"].dt.floor("4h")
    h4_cols = ["ts_round","h4_open","h4_high","h4_low","h4_close","h4_volume"]
    merged = m15.merge(h4[h4_cols], on="ts_round", how="left")
    merged.dropna(subset=["h4_close"], inplace=True)
    merged.reset_index(drop=True, inplace=True)

    feat, atr = build_features(merged, merged)
    o=merged["open"].to_numpy(); h=merged["high"].to_numpy()
    l=merged["low"].to_numpy(); c=merged["close"].to_numpy()

    is_ternary = (approach == "ternary")
    tp_mult, sl_mult = (3.0, 3.0) if profile == "3r3r" else (1.0, 2.0)
    win_r, loss_r = (3, 3) if profile == "3r3r" else (1, 2)

    trades=wins=0; tlong=tshort=0; wlong=wshort=0; skipped=0; retrace_used=0

    def entry_long(i):
        if not retracement: return i, c[i]
        for j in range(i+1, min(i+max_wait+1, len(merged))):
            if l[j] < c[i]:
                return j, min(o[j], l[j])
        return None, None

    def entry_short(i):
        if not retracement: return i, c[i]
        for j in range(i+1, min(i+max_wait+1, len(merged))):
            if h[j] > c[i]:
                return j, max(o[j], h[j])
        return None, None

    if is_ternary:
        probs = ml_model.predict_proba(feat)
        for i in range(50, len(merged)):
            if np.isnan(atr[i]) or atr[i] <= 0: continue
            bp, sp = probs[i,0], probs[i,1]
            if bp >= threshold and (sp < threshold or bp >= sp):
                eidx, ep = entry_long(i)
                if eidx is None: skipped+=1; continue
                if eidx != i: retrace_used+=1
                tp=ep+tp_mult*atr[i]; sl=ep-sl_mult*atr[i]; tlong+=1
                for j in range(eidx+1, min(eidx+50, len(merged))):
                    if h[j] >= tp: wins+=1; wlong+=1; break
                    if l[j] <= sl: break
            elif sp >= threshold and (bp < threshold or sp > bp):
                eidx, ep = entry_short(i)
                if eidx is None: skipped+=1; continue
                if eidx != i: retrace_used+=1
                tp=ep-tp_mult*atr[i]; sl=ep+sl_mult*atr[i]; tshort+=1
                for j in range(eidx+1, min(eidx+50, len(merged))):
                    if l[j] <= tp: wins+=1; wshort+=1; break
                    if h[j] >= sl: break
    else:
        probs = ml_model.predict_proba(feat)[:,1]
        for i in range(50, len(merged)):
            if np.isnan(atr[i]) or atr[i] <= 0: continue
            if probs[i] >= threshold:
                eidx, ep = entry_long(i)
                if eidx is None: skipped+=1; continue
                if eidx != i: retrace_used+=1
                tp=ep+tp_mult*atr[i]; sl=ep-sl_mult*atr[i]; tlong+=1
                for j in range(eidx+1, min(eidx+50, len(merged))):
                    if h[j] >= tp: wins+=1; wlong+=1; break
                    if l[j] <= sl: break
            elif probs[i] <= (1 - threshold):
                eidx, ep = entry_short(i)
                if eidx is None: skipped+=1; continue
                if eidx != i: retrace_used+=1
                tp=ep-tp_mult*atr[i]; sl=ep+sl_mult*atr[i]; tshort+=1
                for j in range(eidx+1, min(eidx+50, len(merged))):
                    if l[j] <= tp: wins+=1; wshort+=1; break
                    if h[j] >= sl: break

    trades = tlong + tshort
    if trades:
        wr = wins / trades
        ev = wr * win_r - (1 - wr) * loss_r
        pnl = wins * win_r - (trades - wins) * loss_r
    else:
        wr = ev = pnl = 0

    return {
        "trades": trades, "wr": wr, "ev": ev, "pnl": pnl,
        "tlong": tlong, "tshort": tshort, "skipped": skipped, "retrace_used": retrace_used,
    }

combo_defs = []
for sym in ["XAUUSD", "BTCUSD"]:
    for rp in ["3r3r"]:
        for appr in ["binary", "ternary"]:
            combo_defs.append((sym, rp, appr, False))
            combo_defs.append((sym, rp, appr, True))
    combo_defs.append((sym, "2r1r", "binary", False))
    combo_defs.append((sym, "2r1r", "binary", True))

print(f"{'Strategy':<38s}  {'Trades':>8s}  {'WR':>7s}  {'EV':>7s}  {'PnL':>10s}  {'Skip':>5s}")
print("=" * 83)
for sym, rp, appr, retr in combo_defs:
    res = run_backtest(sym, rp, appr, retracement=retr)
    label = f"{sym} {rp} {appr}" + (" +retrace" if retr else "")
    wr_s = f"{res['wr']:.1%}" if res["trades"] else "?"
    print(f"{label:<38s}  {res['trades']:>8,}  {wr_s:>7s}  {res['ev']:>+7.2f}R  {res['pnl']:>+10,.0f}R  {res['skipped']:>5}")
