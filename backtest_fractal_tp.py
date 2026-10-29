#!/usr/bin/env python3
"""Backtest: ML + retracement entry + fractal-based TP/SL."""
import numpy as np, pandas as pd, pickle, warnings
warnings.filterwarnings('ignore'); eps=1e-9

df=pd.read_csv("XAUUSD_M15_combined.csv"); df["timestamp"]=pd.to_datetime(df["timestamp"],utc=True)
df=df.sort_values("timestamp").reset_index(drop=True)
h4=pd.read_csv("XAUUSD_16388_combined.csv"); h4["timestamp"]=pd.to_datetime(h4["timestamp"],utc=True)
h4=h4.sort_values("timestamp").reset_index(drop=True)
h4=h4.rename(columns={c:f"h4_{c}" for c in h4.columns if c!="timestamp"})
df["ts_round"]=df["timestamp"].dt.floor("4h"); h4["ts_round"]=h4["timestamp"].dt.floor("4h")
h4c=["ts_round","h4_open","h4_high","h4_low","h4_close","h4_volume"]
merged=df.merge(h4[h4c],on="ts_round",how="left").dropna(subset=["h4_close"]).reset_index(drop=True)
n=len(merged); print(f"Bars: {n:,}")
o=merged["open"].to_numpy(); h=merged["high"].to_numpy(); l=merged["low"].to_numpy()
c=merged["close"].to_numpy(); v=merged["volume"].to_numpy()
h4_c=merged["h4_close"].to_numpy(); h4_o=merged["h4_open"].to_numpy()
h4_h=merged["h4_high"].to_numpy(); h4_l=merged["h4_low"].to_numpy()

atr_vals=pd.Series(h-l).rolling(14).mean().to_numpy()
feat=np.zeros((n,28),np.float32)
feat[:,0]=abs(c-o)/(atr_vals+eps); feat[:,1]=(h-l)/(atr_vals+eps)
feat[:,2]=(h-np.maximum(o,c))/(atr_vals+eps); feat[:,3]=(np.minimum(o,c)-l)/(atr_vals+eps)
ema20=pd.Series(c).ewm(span=20,adjust=False).mean().to_numpy(); ema50=pd.Series(c).ewm(span=50,adjust=False).mean().to_numpy()
feat[:,4]=(c-ema20)/(atr_vals+eps); feat[:,5]=(c-ema50)/(atr_vals+eps)
feat[:,6]=np.diff(ema20,prepend=ema20[0])/(atr_vals+eps); feat[:,7]=np.diff(ema50,prepend=ema50[0])/(atr_vals+eps)
feat[:,8]=v/(pd.Series(v).rolling(20).mean().to_numpy()+eps); feat[:,9]=pd.Series(c).pct_change().rolling(20).std().to_numpy()
feat[:,10]=pd.Series(c).pct_change().fillna(0.0).to_numpy(); feat[:,11]=c/np.roll(c,5)-1.0; feat[:,12]=(np.roll(c,1)>np.roll(o,1)).astype(float)
feat[:,13]=(np.roll(c,2)>np.roll(o,2)).astype(float); feat[:,14]=c>(pd.Series(h).rolling(5).max()+pd.Series(l).rolling(5).min()).to_numpy()/2.0
feat[:,15]=atr_vals/(c+eps); feat[:,16]=((c-o)>0).astype(float); feat[:,17]=abs(c-o)/(h-l+eps)
rh=pd.Series(h).rolling(10).max().to_numpy(); rl=pd.Series(l).rolling(10).min().to_numpy()
feat[:,18]=(c-rl)/(rh-rl+eps); feat[:,19]=v/(pd.Series(v).rolling(50).mean().to_numpy()+eps)
h4_atr=pd.Series(h4_h-h4_l).rolling(14).mean().to_numpy(); h4_e20=pd.Series(h4_c).ewm(span=20,adjust=False).mean().to_numpy()
h4_e50=pd.Series(h4_c).ewm(span=50,adjust=False).mean().to_numpy(); h4_sh=pd.Series(h4_h).rolling(10).max().to_numpy()
h4_sl=pd.Series(h4_l).rolling(10).min().to_numpy()
feat[:,20]=abs(h4_o-h4_c)/(h4_atr+eps); feat[:,21]=(h4_h-h4_l)/(h4_atr+eps); feat[:,22]=h4_e20; feat[:,23]=h4_e50
feat[:,24]=(h4_c-h4_e20)/(h4_atr+eps); feat[:,25]=np.sign(h4_e20-h4_e50).astype(float); feat[:,26]=(h4_c-h4_sl)/(h4_sh-h4_sl+eps); feat[:,27]=c/h4_c-1.0

with open("ml_xau_m15h4_v2.pkl","rb") as f: model=pickle.load(f)["model"]
probs=model.predict_proba(feat)[:,1]

# 5-bar fractals
fractal_sup = np.full(n, np.nan); fractal_res = np.full(n, np.nan)
for i in range(2, n-2):
    if l[i] < l[i-2] and l[i] < l[i-1] and l[i] < l[i+1] and l[i] < l[i+2]: fractal_sup[i] = l[i]
    if h[i] > h[i-2] and h[i] > h[i-1] and h[i] > h[i+1] and h[i] > h[i+2]: fractal_res[i] = h[i]

# For each bar i, precompute the nearest prior fractal support below and resistance above
nearest_sup = np.full(n, np.nan); nearest_res = np.full(n, np.nan)
for i in range(50, n):
    # nearest support below current price
    best = None; best_dist = None
    for k in range(i-50, i):
        if not np.isnan(fractal_sup[k]) and fractal_sup[k] < c[i]:
            d = c[i] - fractal_sup[k]
            if best_dist is None or d < best_dist: best = fractal_sup[k]; best_dist = d
    nearest_sup[i] = best
    # nearest resistance above current price
    best = None; best_dist = None
    for k in range(i-50, i):
        if not np.isnan(fractal_res[k]) and fractal_res[k] > c[i]:
            d = fractal_res[k] - c[i]
            if best_dist is None or d < best_dist: best = fractal_res[k]; best_dist = d
    nearest_res[i] = best

sup_ct = (~np.isnan(fractal_sup)).sum(); res_ct = (~np.isnan(fractal_res)).sum()
print(f"Fractals: {sup_ct:,} support, {res_ct:,} resistance")

# --- Backtest functions ---

def bt_baseline(th):
    trades=wins=0
    for i in range(50,n):
        if np.isnan(atr_vals[i]) or atr_vals[i]<=0: continue
        if probs[i]>=th:
            tp=c[i]+3*atr_vals[i]; sl=c[i]-3*atr_vals[i]; trades+=1
            for j in range(i+1,min(i+50,n)):
                if h[j]>=tp: wins+=1; break
                if l[j]<=sl: break
        elif probs[i]<=(1-th):
            tp=c[i]-3*atr_vals[i]; sl=c[i]+3*atr_vals[i]; trades+=1
            for j in range(i+1,min(i+50,n)):
                if l[j]<=tp: wins+=1; break
                if h[j]>=sl: break
    if trades: wr=wins/trades; ev=wr*3-(1-wr)*3; pnl=wins*3-(trades-wins)*3
    return trades, wins, wr, ev, pnl

def bt_simple_retrace(th):
    trades=wins=0; skipped=0
    for i in range(50,n-10):
        if np.isnan(atr_vals[i]) or atr_vals[i]<=0: continue
        if probs[i]>=th:
            ep=None; ei=None
            for j in range(i+1,min(i+6,n)):
                if l[j]<c[i]: ep=min(o[j],l[j]); ei=j; break
            if ep is None: skipped+=1; continue
            tp=ep+3*atr_vals[i]; sl=ep-3*atr_vals[i]; trades+=1
            for j in range(ei+1,min(ei+50,n)):
                if h[j]>=tp: wins+=1; break
                if l[j]<=sl: break
        elif probs[i]<=(1-th):
            ep=None; ei=None
            for j in range(i+1,min(i+6,n)):
                if h[j]>c[i]: ep=max(o[j],h[j]); ei=j; break
            if ep is None: skipped+=1; continue
            tp=ep-3*atr_vals[i]; sl=ep+3*atr_vals[i]; trades+=1
            for j in range(ei+1,min(ei+50,n)):
                if l[j]<=tp: wins+=1; break
                if h[j]>=sl: break
    if trades: wr=wins/trades; ev=wr*3-(1-wr)*3; pnl=wins*3-(trades-wins)*3
    return trades, wins, wr, ev, pnl, skipped

def bt_retrace_fractal_tp(th):
    """Retracement entry + TP at nearest fractal resistance/support. SL=3*ATR."""
    trades=wins=0; skipped=0; no_level=0
    for i in range(50,n-10):
        if np.isnan(atr_vals[i]) or atr_vals[i]<=0: continue
        if probs[i]>=th:
            ep=None; ei=None
            for j in range(i+1,min(i+6,n)):
                if l[j]<c[i]: ep=min(o[j],l[j]); ei=j; break
            if ep is None: skipped+=1; continue
            # TP at nearest fractal resistance above entry
            tp = nearest_res[i]
            if np.isnan(tp) or tp <= ep: no_level+=1; skipped+=1; continue
            sl=ep-3*atr_vals[i]; trades+=1
            for j in range(ei+1,min(ei+50,n)):
                if h[j]>=tp: wins+=1; break
                if l[j]<=sl: break
        elif probs[i]<=(1-th):
            ep=None; ei=None
            for j in range(i+1,min(i+6,n)):
                if h[j]>c[i]: ep=max(o[j],h[j]); ei=j; break
            if ep is None: skipped+=1; continue
            tp = nearest_sup[i]
            if np.isnan(tp) or tp >= ep: no_level+=1; skipped+=1; continue
            sl=ep+3*atr_vals[i]; trades+=1
            for j in range(ei+1,min(ei+50,n)):
                if l[j]<=tp: wins+=1; break
                if h[j]>=sl: break
    if trades: wr=wins/trades; ev=tot_r(trades,wins); pnl=tot_pnl(trades,wins)
    return trades, wins, wr, ev, pnl, skipped, no_level

def bt_retrace_fractal_tp_sl(th):
    """Retracement entry + TP at nearest fractal resistance, SL at nearest fractal support.
       Entry at retracement. TP/SL determined by market structure only."""
    trades=wins=0; skipped=0; no_level=0
    for i in range(50,n-10):
        if np.isnan(atr_vals[i]) or atr_vals[i]<=0: continue
        if probs[i]>=th:
            ep=None; ei=None
            for j in range(i+1,min(i+6,n)):
                if l[j]<c[i]: ep=min(o[j],l[j]); ei=j; break
            if ep is None: skipped+=1; continue
            tp=nearest_res[i]; sl=nearest_sup[i]
            if np.isnan(tp) or np.isnan(sl) or tp<=ep or sl>=ep: no_level+=1; skipped+=1; continue
            # Make sure SL is at least some minimum distance (e.g., 0.5*ATR)
            if ep-sl < 0.5*atr_vals[i]: no_level+=1; skipped+=1; continue
            trades+=1; win_r=(tp-ep)/(ep-sl); loss_r=1.0
            for j in range(ei+1,min(ei+50,n)):
                if h[j]>=tp: wins+=1; break
                if l[j]<=sl: break
        elif probs[i]<=(1-th):
            ep=None; ei=None
            for j in range(i+1,min(i+6,n)):
                if h[j]>c[i]: ep=max(o[j],h[j]); ei=j; break
            if ep is None: skipped+=1; continue
            tp=nearest_sup[i]; sl=nearest_res[i]
            if np.isnan(tp) or np.isnan(sl) or tp>=ep or sl<=ep: no_level+=1; skipped+=1; continue
            if sl-ep < 0.5*atr_vals[i]: no_level+=1; skipped+=1; continue
            trades+=1; win_r=(ep-tp)/(sl-ep); loss_r=1.0
            for j in range(ei+1,min(ei+50,n)):
                if l[j]<=tp: wins+=1; break
                if h[j]>=sl: break
    if trades==0: return 0,0,0,0,0,skipped,no_level
    return trades, wins, wins/trades, None, None, skipped, no_level

def tot_r(t,w): return w*3-(t-w)*3
def tot_pnl(t,w): return w*3-(t-w)*3

def bt_retrace_fixed_tp_sl(th, tp_mult, sl_mult):
    """Retracement entry + custom fixed TP/SL ratios."""
    trades=wins=0; skipped=0
    for i in range(50,n-10):
        if np.isnan(atr_vals[i]) or atr_vals[i]<=0: continue
        if probs[i]>=th:
            ep=None; ei=None
            for j in range(i+1,min(i+6,n)):
                if l[j]<c[i]: ep=min(o[j],l[j]); ei=j; break
            if ep is None: skipped+=1; continue
            tp=ep+tp_mult*atr_vals[i]; sl=ep-sl_mult*atr_vals[i]; trades+=1
            for j in range(ei+1,min(ei+50,n)):
                if h[j]>=tp: wins+=1; break
                if l[j]<=sl: break
        elif probs[i]<=(1-th):
            ep=None; ei=None
            for j in range(i+1,min(i+6,n)):
                if h[j]>c[i]: ep=max(o[j],h[j]); ei=j; break
            if ep is None: skipped+=1; continue
            tp=ep-tp_mult*atr_vals[i]; sl=ep+sl_mult*atr_vals[i]; trades+=1
            for j in range(ei+1,min(ei+50,n)):
                if l[j]<=tp: wins+=1; break
                if h[j]>=sl: break
    if trades: wr=wins/trades; ev=wr*tp_mult-(1-wr)*sl_mult; pnl=wins*tp_mult-(trades-wins)*sl_mult
    return trades, wins, wr, ev, pnl, skipped

print(f"\n{'='*70}")
print("1. BASELINE (close entry, 3/3 ATR)")
for th in [0.55]:
    t,w,wr,ev,pnl=bt_baseline(th)
    print(f"  th={th:.2f}: {t:>7,} trades, {wr:.1%} WR, {ev:.2f}R EV, {pnl:+,}R")

print(f"\n{'='*70}")
print("2. RETRACEMENT + 3R:3R (best so far)")
for th in [0.55,0.60]:
    t,w,wr,ev,pnl,sk=bt_simple_retrace(th)
    print(f"  th={th:.2f}: {t:>7,} trades, {wr:.1%} WR, {ev:.2f}R EV, {pnl:+,}R ({sk} skip)")

print(f"\n{'='*70}")
print("3. RETRACEMENT + FRACTAL TP + 3*ATR SL")
for th in [0.55,0.60]:
    t,w,wr,ev,pnl,sk,nl=bt_retrace_fractal_tp(th)
    print(f"  th={th:.2f}: {t:>7,} trades, {wr:.1%} WR (R:R variable — EV not directly comparable) ({sk} skip, {nl} no level)")

# For fractal TP/SL, compute average win/loss
def bt_retrace_fractal_tp_sl_stats(th):
    """Same as fractal TP/SL but track actual R multiples."""
    trades=wins=0; skipped=0; no_level=0; win_rs=[]; loss_rs=[]
    for i in range(50,n-10):
        if np.isnan(atr_vals[i]) or atr_vals[i]<=0: continue
        if probs[i]>=th:
            ep=None; ei=None
            for j in range(i+1,min(i+6,n)):
                if l[j]<c[i]: ep=min(o[j],l[j]); ei=j; break
            if ep is None: skipped+=1; continue
            tp=nearest_res[i]; sl=nearest_sup[i]
            if np.isnan(tp) or np.isnan(sl) or tp<=ep or sl>=ep: no_level+=1; skipped+=1; continue
            if ep-sl < 0.3*atr_vals[i]: no_level+=1; skipped+=1; continue
            trades+=1; win_r=(tp-ep)/(ep-sl)
            hit=False
            for j in range(ei+1,min(ei+50,n)):
                if h[j]>=tp: wins+=1; win_rs.append(win_r); hit=True; break
                if l[j]<=sl: loss_rs.append(1.0); hit=True; break
            if not hit: loss_rs.append(1.0)
        elif probs[i]<=(1-th):
            ep=None; ei=None
            for j in range(i+1,min(i+6,n)):
                if h[j]>c[i]: ep=max(o[j],h[j]); ei=j; break
            if ep is None: skipped+=1; continue
            tp=nearest_sup[i]; sl=nearest_res[i]
            if np.isnan(tp) or np.isnan(sl) or tp>=ep or sl<=ep: no_level+=1; skipped+=1; continue
            if sl-ep < 0.3*atr_vals[i]: no_level+=1; skipped+=1; continue
            trades+=1; win_r=(ep-tp)/(sl-ep)
            hit=False
            for j in range(ei+1,min(ei+50,n)):
                if l[j]<=tp: wins+=1; win_rs.append(win_r); hit=True; break
                if h[j]>=sl: loss_rs.append(1.0); hit=True; break
            if not hit: loss_rs.append(1.0)
    if trades==0: return 0,0,0,0,0,skipped,no_level,[],[]
    return trades, wins, wins/trades, np.mean(win_rs) if win_rs else 0, np.mean(loss_rs) if loss_rs else 0, skipped, no_level, win_rs, loss_rs

print(f"\n{'='*70}")
print("4. RETRACEMENT + FRACTAL TP + FRACTAL SL (variable R:R)")
for th in [0.55,0.60]:
    t,w,wr,awr,alr,sk,nl,wrs,lrs=bt_retrace_fractal_tp_sl_stats(th)
    if t:
        ev = (w/len(wrs))*awr if wrs else 0 - ((t-w)/len(lrs))*alr if lrs else 0
        # Proper EV calculation
        avg_win = np.mean(wrs) if wrs else 0
        avg_loss = np.mean(lrs) if lrs else 0
        ev = wr*avg_win - (1-wr)*avg_loss
        tpnl = sum(wrs) - sum(lrs) if wrs and lrs else sum(wrs) if wrs else -sum(lrs)
        tpnl = sum(wrs) - sum(lrs)
        print(f"  th={th:.2f}: {t:>7,} trades, {wr:.1%} WR, avg win={avg_win:.2f}R, avg loss={avg_loss:.2f}R")
        print(f"           EV={ev:.2f}R, total PnL={tpnl:+,.2f}R  ({sk} skip, {nl} no level)")

# Also check: how does retracement + 2R:1R compare?
print(f"\n{'='*70}")
print("5. RETRACEMENT + 2R:1R (TP=1*ATR, SL=2*ATR)")
for th in [0.55]:
    t,w,wr,ev,pnl,sk=bt_retrace_fixed_tp_sl(th,1,2)
    print(f"  th={th:.2f}: {t:>7,} trades, {wr:.1%} WR, {ev:.2f}R EV, {pnl:+,}R ({sk} skip)")

print(f"\n{'='*70}")
print("6. RETRACEMENT + 1.5R:1R (TP=1.5*ATR, SL=1*ATR)")
for th in [0.55]:
    t,w,wr,ev,pnl,sk=bt_retrace_fixed_tp_sl(th,1.5,1)
    print(f"  th={th:.2f}: {t:>7,} trades, {wr:.1%} WR, {ev:.2f}R EV, {pnl:+,}R ({sk} skip)")
