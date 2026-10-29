#!/usr/bin/env python3
"""Train ML model on combined data (CSV + MT5 2026).
Usage: python train_v2.py XAUUSD 3r3r
       python train_v2.py XAUUSD 2r1r
       python train_v2.py BTCUSD 3r3r
       python train_v2.py BTCUSD 2r1r
"""
import numpy as np, pandas as pd, pickle, sys, warnings
from sklearn.ensemble import RandomForestClassifier
warnings.filterwarnings('ignore'); eps=1e-9

symbol = sys.argv[1].upper() if len(sys.argv)>1 else "XAUUSD"
profile = sys.argv[2] if len(sys.argv)>2 else "3r3r"

if symbol=="XAUUSD":
    m15_file="XAUUSD_M15_combined.csv"; h4_file="XAUUSD_16388_combined.csv"
    out_file=f"ml_xau_m15h4_{'v2' if profile=='3r3r' else 'v2_r2r1'}.pkl"
else:
    m15_file="BTCUSD_15_combined.csv"; h4_file="BTCUSD_16388_combined.csv"
    out_file=f"ml_btc_m15h4_{'v2' if profile=='3r3r' else 'v2_r2r1'}.pkl"

tp = 1.0 if profile=="2r1r" else 3.0
sl = 2.0 if profile=="2r1r" else 3.0
label=f"TP={tp:.0f}*ATR, SL={sl:.0f}*ATR"

print(f"{'='*70}\n{symbol} M15+H4 — {label}\n{'='*70}")

df=pd.read_csv(m15_file); df["timestamp"]=pd.to_datetime(df["timestamp"],utc=True)
df=df.sort_values("timestamp").reset_index(drop=True)
h4=pd.read_csv(h4_file); h4["timestamp"]=pd.to_datetime(h4["timestamp"],utc=True)
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
feat[:,17]=abs(c-o)/(h-l+eps)
rh=pd.Series(h).rolling(10).max().to_numpy(); rl=pd.Series(l).rolling(10).min().to_numpy()
feat[:,18]=(c-rl)/(rh-rl+eps); feat[:,19]=v/(pd.Series(v).rolling(50).mean().to_numpy()+eps)
h4_atr=pd.Series(h4_h-h4_l).rolling(14).mean().to_numpy()
h4_e20=pd.Series(h4_c).ewm(span=20,adjust=False).mean().to_numpy()
h4_e50=pd.Series(h4_c).ewm(span=50,adjust=False).mean().to_numpy()
h4_sh=pd.Series(h4_h).rolling(10).max().to_numpy(); h4_sl=pd.Series(h4_l).rolling(10).min().to_numpy()
feat[:,20]=abs(h4_o-h4_c)/(h4_atr+eps); feat[:,21]=(h4_h-h4_l)/(h4_atr+eps)
feat[:,22]=h4_e20; feat[:,23]=h4_e50
feat[:,24]=(h4_c-h4_e20)/(h4_atr+eps); feat[:,25]=np.sign(h4_e20-h4_e50).astype(float)
feat[:,26]=(h4_c-h4_sl)/(h4_sh-h4_sl+eps); feat[:,27]=c/h4_c-1.0

# Labels
print("Computing labels...")
labels=np.full(n,-1,dtype=np.int32)
for i in range(n-50):
    if np.isnan(atr_vals[i]) or atr_vals[i]<=0: continue
    tp_lvl=c[i]+tp*atr_vals[i]; sl_lvl=c[i]-sl*atr_vals[i]
    for j in range(i+1,min(i+51,n)):
        if l[j]<=sl_lvl: labels[i]=0; break
        if h[j]>=tp_lvl: labels[i]=1; break

# Split: first 80% train, last 20% test (temporal)
oos_cut=int(n*0.8)
train_mask=np.arange(n)<oos_cut; test_mask=~train_mask

def filter_valid(feat_subset, labels_subset, indices):
    valid=labels_subset>=0
    X=feat_subset[valid]; y=labels_subset[valid]; idx=indices[valid]
    ok=~np.isnan(X).any(axis=1)
    return X[ok], y[ok], idx[ok]

X_train, y_train, _ = filter_valid(feat[train_mask], labels[train_mask], np.where(train_mask)[0])
X_test, y_test, _ = filter_valid(feat[test_mask], labels[test_mask], np.where(test_mask)[0])

print(f"Train: {len(y_train):,} (WR: {y_train.mean():.1%})")
print(f"Test:  {len(y_test):,} (WR: {y_test.mean():.1%})")

# Train
model=RandomForestClassifier(n_estimators=100, max_depth=8, min_samples_split=10,
                             min_samples_leaf=5, random_state=42, n_jobs=-1, class_weight='balanced')
model.fit(X_train, y_train)
y_prob=model.predict_proba(X_test)[:,1]

print(f"\nML OOS results:")
for th in [x/100 for x in range(40,80,5)]:
    m=y_prob>=th; sm=y_prob<=1-th
    if m.sum()>0:
        wr=y_test[m].mean(); ev=wr*tp-(1-wr)*sl
        print(f"  Thresh {th:.2f} BUY:  {m.sum():,} trades, {wr:.1%} WR, {ev:.2f}R EV")
    if sm.sum()>0:
        wr2=1-y_test[sm].mean(); ev2=wr2*tp-(1-wr2)*sl
        print(f"         SHORT: {sm.sum():,} trades, {wr2:.1%} WR, {ev2:.2f}R EV")

# Save
with open(out_file,"wb") as f: pickle.dump({"model":model,"threshold":0.55}, f)
print(f"\nSaved: {out_file}")
