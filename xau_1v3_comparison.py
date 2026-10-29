#!/usr/bin/env python3
"""Full backtest comparison: 1-class vs 3-class on XAUUSD v2 data (3R:3R)."""
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
labels=np.full(n,-1,dtype=np.int32)
for i in range(n-50):
    if np.isnan(atr_vals[i]) or atr_vals[i]<=0: continue
    tp_lvl=c[i]+3.0*atr_vals[i]; sl_lvl=c[i]-3.0*atr_vals[i]
    found=False
    for j in range(i+1,min(i+51,n)):
        if h[j]>=tp_lvl: labels[i]=0; found=True; break
        if l[j]<=sl_lvl: labels[i]=1; found=True; break
    if not found: labels[i]=2

# Train 3-class model on ALL data for full backtest
X_full=feat[labels>=0]; y_full=labels[labels>=0]
ok=~np.isnan(X_full).any(axis=1); X_full=X_full[ok]; y_full=y_full[ok]
print(f"Training samples: {len(y_full):,}")
model3=RandomForestClassifier(n_estimators=100,max_depth=8,min_samples_split=10,
    min_samples_leaf=5,random_state=42,n_jobs=-1,class_weight='balanced')
model3.fit(X_full, y_full)
probs3=model3.predict_proba(feat)

# Backtest 3-class
def backtest_3class(probs, threshold):
    buy_p=probs[:,0]; sell_p=probs[:,1]
    trades=wins=0; buys=sells=0
    for i in range(50,n):
        if np.isnan(atr_vals[i]) or atr_vals[i]<=0: continue
        if labels[i]<0: continue
        if buy_p[i]>=threshold and (sell_p[i]<threshold or buy_p[i]>=sell_p[i]):
            tp=c[i]+3*atr_vals[i]; sl=c[i]-3*atr_vals[i]; buys+=1; trades+=1
            for j in range(i+1,min(i+50,n)):
                if h[j]>=tp: wins+=1; break
                if l[j]<=sl: break
        elif sell_p[i]>=threshold and (buy_p[i]<threshold or sell_p[i]>buy_p[i]):
            tp=c[i]-3*atr_vals[i]; sl=c[i]+3*atr_vals[i]; sells+=1; trades+=1
            for j in range(i+1,min(i+50,n)):
                if l[j]<=tp: wins+=1; break
                if h[j]>=sl: break
    if trades==0: return 0,0,0,0,0,0
    wr=wins/trades; ev=wr*3-(1-wr)*3; pnl=wins*3-(trades-wins)*3
    return trades, buys, sells, wr, ev, pnl

# Backtest 1-class v2
with open("ml_xau_m15h4_v2.pkl","rb") as f: m1=pickle.load(f)["model"]
probs1=m1.predict_proba(feat)[:,1]

def backtest_1class(probs, threshold):
    trades=wins=0; buys=sells=0
    for i in range(50,n):
        if np.isnan(atr_vals[i]) or atr_vals[i]<=0: continue
        if labels[i]<0: continue
        if probs[i]>=threshold:
            tp=c[i]+3*atr_vals[i]; sl=c[i]-3*atr_vals[i]; buys+=1; trades+=1
            for j in range(i+1,min(i+50,n)):
                if h[j]>=tp: wins+=1; break
                if l[j]<=sl: break
        elif probs[i]<=(1-threshold):
            tp=c[i]-3*atr_vals[i]; sl=c[i]+3*atr_vals[i]; sells+=1; trades+=1
            for j in range(i+1,min(i+50,n)):
                if l[j]<=tp: wins+=1; break
                if h[j]>=sl: break
    if trades==0: return 0,0,0,0,0,0
    wr=wins/trades; ev=wr*3-(1-wr)*3; pnl=wins*3-(trades-wins)*3
    return trades, buys, sells, wr, ev, pnl

print(f"\n{'='*60}")
print("Full Backtest: 1-Class (current v2)")
print(f"{'='*60}")
print(f"{'Thresh':>8} {'Trades':>8} {'BUY':>6} {'SELL':>6} {'WR':>8} {'EV':>8} {'PnL(R)':>10}")
for th in [0.50, 0.55, 0.60, 0.65]:
    t,b,s,w,e,p=backtest_1class(probs1,th)
    if t: print(f'{th:.2f} {t:>8,} {b:>6,} {s:>6,} {w:>7.1%} {e:>7.2f}R {p:>+9,}R')

print(f"\n{'='*60}")
print("Full Backtest: 3-Class (BUY/SELL/NEUTRAL)")
print(f"{'='*60}")
print(f"{'Thresh':>8} {'Trades':>8} {'BUY':>6} {'SELL':>6} {'WR':>8} {'EV':>8} {'PnL(R)':>10}")
for th in [0.50, 0.55, 0.60, 0.65]:
    t,b,s,w,e,p=backtest_3class(probs3,th)
    if t: print(f'{th:.2f} {t:>8,} {b:>6,} {s:>6,} {w:>7.1%} {e:>7.2f}R {p:>+9,}R')
