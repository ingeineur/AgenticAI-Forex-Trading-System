#!/usr/bin/env python3
"""Backtest: ML signal + retracement entry on XAUUSD (3R:3R, binary v2)."""
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
n=len(merged); print(f"Bars: {n:,} ({merged['timestamp'].iloc[0].date()} to {merged['timestamp'].iloc[-1].date()})")
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
    if trades==0: return 0,0,0,0,0
    wr=wins/trades; ev=wr*3-(1-wr)*3; pnl=wins*3-(trades-wins)*3
    return trades, wins, wr, ev, pnl

def bt_retrace(th, max_wait=5):
    """Signal at bar i -> wait up to max_wait bars for retracement -> enter at entry_bar -> check TP/SL from entry_bar+1."""
    trades=wins=0; skipped=0
    for i in range(50,n-10):
        if np.isnan(atr_vals[i]) or atr_vals[i]<=0: continue
        entry_price=None; entry_idx=None
        if probs[i]>=th:  # BUY
            for j in range(i+1, min(i+1+max_wait, n)):
                if l[j] < c[i]:
                    entry_price=min(o[j], l[j]); entry_idx=j; break
            if entry_price is None: skipped+=1; continue
            tp=entry_price+3*atr_vals[i]; sl=entry_price-3*atr_vals[i]; trades+=1
            for j in range(entry_idx+1, min(entry_idx+50, n)):
                if h[j]>=tp: wins+=1; break
                if l[j]<=sl: break
        elif probs[i]<=(1-th):  # SELL
            for j in range(i+1, min(i+1+max_wait, n)):
                if h[j] > c[i]:
                    entry_price=max(o[j], h[j]); entry_idx=j; break
            if entry_price is None: skipped+=1; continue
            tp=entry_price-3*atr_vals[i]; sl=entry_price+3*atr_vals[i]; trades+=1
            for j in range(entry_idx+1, min(entry_idx+50, n)):
                if l[j]<=tp: wins+=1; break
                if h[j]>=sl: break
    if trades==0: return 0,0,0,0,0,skipped
    wr=wins/trades; ev=wr*3-(1-wr)*3; pnl=wins*3-(trades-wins)*3
    return trades, wins, wr, ev, pnl, skipped

print(f"\n{'='*60}")
print("BASELINE: enter at signal bar close")
for th in [0.55,0.60]:
    t,w,wr,ev,pnl=bt_baseline(th)
    print(f'  th={th:.2f}: {t:>7,} trades, {wr:.1%} WR, {ev:.2f}R EV, {pnl:+,}R PnL')

for wait in [3,5,8]:
    print(f"\n{'='*60}")
    print(f"RETRACEMENT: wait up to {wait} bars for pullback")
    print(f"  {'Thresh':>6} {'Trades':>7} {'Wins':>7} {'WR':>7} {'EV':>7} {'PnL(R)':>9} {'Skip':>6}")
    for th in [0.55,0.60]:
        t,w,wr,ev,pnl,sk=bt_retrace(th,wait)
        pct=t/(t+sk)*100 if (t+sk)>0 else 0
        print(f'  {th:.2f} {t:>7,} {w:>7,} {wr:>6.1%} {ev:>6.2f}R {pnl:>+8,}R {sk:>5}')

# ATR-based min pullback
print(f"\n{'='*60}")
print(f"RETRACEMENT (0.5*ATR min pullback, max wait 5)")
print(f"  {'Thresh':>6} {'Trades':>7} {'Wins':>7} {'WR':>7} {'EV':>7} {'PnL(R)':>9} {'Skip':>6}")
for th in [0.55,0.60]:
    trades=wins=0; skipped=0
    for i in range(50,n-10):
        if np.isnan(atr_vals[i]) or atr_vals[i]<=0: continue
        entry_price=None; entry_idx=None; min_move=atr_vals[i]*0.5
        if probs[i]>=th:
            for j in range(i+1, min(i+6, n)):
                if l[j] < c[i]-min_move:
                    entry_price=min(o[j],l[j]); entry_idx=j; break
            if entry_price is None: skipped+=1; continue
            tp=entry_price+3*atr_vals[i]; sl=entry_price-3*atr_vals[i]; trades+=1
            for j in range(entry_idx+1, min(entry_idx+50, n)):
                if h[j]>=tp: wins+=1; break
                if l[j]<=sl: break
        elif probs[i]<=(1-th):
            for j in range(i+1, min(i+6, n)):
                if h[j] > c[i]+min_move:
                    entry_price=max(o[j],h[j]); entry_idx=j; break
            if entry_price is None: skipped+=1; continue
            tp=entry_price-3*atr_vals[i]; sl=entry_price+3*atr_vals[i]; trades+=1
            for j in range(entry_idx+1, min(entry_idx+50, n)):
                if l[j]<=tp: wins+=1; break
                if h[j]>=sl: break
    if trades:
        wr=wins/trades; ev=wr*3-(1-wr)*3; pnl=wins*3-(trades-wins)*3
        print(f'  {th:.2f} {trades:>7,} {wins:>7,} {wr:>6.1%} {ev:>6.2f}R {pnl:>+8,}R {skipped:>5}')
