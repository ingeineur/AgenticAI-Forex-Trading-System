#!/usr/bin/env python3
"""Fair comparison: 1-class vs 3-class with same 80/20 temporal split."""
import numpy as np, pandas as pd, pickle, warnings
from sklearn.ensemble import RandomForestClassifier
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

# 3-class labels
labels3=np.full(n,-1,dtype=np.int32)
for i in range(n-50):
    if np.isnan(atr_vals[i]) or atr_vals[i]<=0: continue
    tp=c[i]+3*atr_vals[i]; sl=c[i]-3*atr_vals[i]
    found=False
    for j in range(i+1,min(i+51,n)):
        if h[j]>=tp: labels3[i]=0; found=True; break
        if l[j]<=sl: labels3[i]=1; found=True; break
    if not found: labels3[i]=2

# 80/20 temporal split
oos_cut=int(n*0.8); tr_mask=np.arange(n)<oos_cut; te_mask=~tr_mask

def prep(X_sub, y_sub, idx):
    v=y_sub>=0; X2=X_sub[v]; y2=y_sub[v]; i2=idx[v]
    ok=~np.isnan(X2).any(axis=1); return X2[ok], y2[ok], i2[ok]

X_tr3, y_tr3, _ = prep(feat[tr_mask], labels3[tr_mask], np.where(tr_mask)[0])
X_te3, y_te3, _ = prep(feat[te_mask], labels3[te_mask], np.where(te_mask)[0])
print(f"3-class Train: {len(y_tr3):,}, Test: {len(X_te3):,}")

m3=RandomForestClassifier(n_estimators=100,max_depth=8,min_samples_split=10,
    min_samples_leaf=5,random_state=42,n_jobs=-1,class_weight='balanced')
m3.fit(X_tr3, y_tr3)
p3=m3.predict_proba(feat)  # [N, 3]

# Load 1-class v2 model (trained on same 80/20 split)
with open("ml_xau_m15h4_v2.pkl","rb") as f: m1=pickle.load(f)["model"]
p1=m1.predict_proba(feat)[:,1]

# Backtest on OOS only (last 20%)
def bt_1class(probs, t):
    tw=ww=0; tb=ts=0
    for i in range(50,n):
        if not te_mask[i] or np.isnan(atr_vals[i]) or atr_vals[i]<=0: continue
        if labels3[i]<0: continue
        if probs[i]>=t:
            tp=c[i]+3*atr_vals[i]; sl=c[i]-3*atr_vals[i]; tb+=1; tw+=1
            for j in range(i+1,min(i+50,n)):
                if h[j]>=tp: ww+=1; break
                if l[j]<=sl: break
        elif probs[i]<=(1-t):
            tp=c[i]-3*atr_vals[i]; sl=c[i]+3*atr_vals[i]; ts+=1; tw+=1
            for j in range(i+1,min(i+50,n)):
                if l[j]<=tp: ww+=1; break
                if h[j]>=sl: break
    if tw==0: return 0,0,0,0,0
    wr=ww/tw; ev=wr*3-(1-wr)*3; pnl=ww*3-(tw-ww)*3
    return tw, tb, ts, wr, ev, pnl

def bt_3class(probs, t):
    bp=probs[:,0]; sp=probs[:,1]
    tw=ww=0; tb=ts=0
    for i in range(50,n):
        if not te_mask[i] or np.isnan(atr_vals[i]) or atr_vals[i]<=0: continue
        if labels3[i]<0: continue
        if bp[i]>=t and (sp[i]<t or bp[i]>=sp[i]):
            tp=c[i]+3*atr_vals[i]; sl=c[i]-3*atr_vals[i]; tb+=1; tw+=1
            for j in range(i+1,min(i+50,n)):
                if h[j]>=tp: ww+=1; break
                if l[j]<=sl: break
        elif sp[i]>=t and (bp[i]<t or sp[i]>bp[i]):
            tp=c[i]-3*atr_vals[i]; sl=c[i]+3*atr_vals[i]; ts+=1; tw+=1
            for j in range(i+1,min(i+50,n)):
                if l[j]<=tp: ww+=1; break
                if h[j]>=sl: break
    if tw==0: return 0,0,0,0,0
    wr=ww/tw; ev=wr*3-(1-wr)*3; pnl=ww*3-(tw-ww)*3
    return tw, tb, ts, wr, ev, pnl

print(f"\n{'='*60}")
print("OOS Backtest: 1-Class (current v2)")
print(f"{'='*60}")
print(f"{'Thresh':>6} {'Trades':>7} {'BUY':>5} {'SELL':>5} {'WR':>7} {'EV':>7} {'PnL(R)':>9}")
for th in [0.50,0.55,0.60,0.65,0.70]:
    t,b,s,w,e,p=bt_1class(p1,th)
    if t: print(f'{th:.2f} {t:>7,} {b:>5,} {s:>5,} {w:>6.1%} {e:>6.2f}R {p:>+8,}R')

print(f"\n{'='*60}")
print("OOS Backtest: 3-Class (BUY/SELL/NEUTRAL)")
print(f"{'='*60}")
print(f"{'Thresh':>6} {'Trades':>7} {'BUY':>5} {'SELL':>5} {'WR':>7} {'EV':>7} {'PnL(R)':>9}")
for th in [0.50,0.55,0.60,0.65,0.70]:
    t,b,s,w,e,p=bt_3class(p3,th)
    if t: print(f'{th:.2f} {t:>7,} {b:>5,} {s:>5,} {w:>6.1%} {e:>6.2f}R {p:>+8,}R')
