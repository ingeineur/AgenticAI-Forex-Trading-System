#!/usr/bin/env python3
"""3-class model experiment: BUY vs SELL vs NEUTRAL on XAUUSD v2 data."""
import numpy as np, pandas as pd, pickle, warnings
from sklearn.ensemble import RandomForestClassifier
warnings.filterwarnings('ignore'); eps=1e-9

print("="*70)
print("XAUUSD 3-Class Experiment: BUY / SELL / NEUTRAL (3R:3R)")
print("="*70)

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

# 3-class labels:
# 0 = BUY wins (+3*ATR before -3*ATR)
# 1 = SELL wins (-3*ATR before +3*ATR)
# 2 = NEUTRAL (neither within 50 bars, or NaN atr)
print("Computing 3-class labels...")
labels=np.full(n,-1,dtype=np.int32)
for i in range(n-50):
    if np.isnan(atr_vals[i]) or atr_vals[i]<=0: continue
    tp_lvl=c[i]+3.0*atr_vals[i]; sl_lvl=c[i]-3.0*atr_vals[i]
    found=False
    for j in range(i+1,min(i+51,n)):
        if h[j]>=tp_lvl: labels[i]=0; found=True; break  # BUY wins
        if l[j]<=sl_lvl: labels[i]=1; found=True; break  # SELL wins
    if not found: labels[i]=2  # NEUTRAL

# Distribution
for cls,name in [(0,"BUY win"),(1,"SELL win"),(2,"NEUTRAL")]:
    cnt=(labels==cls).sum()
    print(f"  {name}: {cnt:,} ({cnt/n*100:.1f}%)")

# OOS split (temporal: last 20%)
oos_cut=int(n*0.8)
train_mask=np.arange(n)<oos_cut; test_mask=~train_mask

def filter_valid(feat_subset, labels_subset, indices):
    valid=labels_subset>=0
    X=feat_subset[valid]; y=labels_subset[valid]; idx=indices[valid]
    ok=~np.isnan(X).any(axis=1)
    return X[ok], y[ok], idx[ok]

X_train, y_train, _ = filter_valid(feat[train_mask], labels[train_mask], np.where(train_mask)[0])
X_test, y_test, _ = filter_valid(feat[test_mask], labels[test_mask], np.where(test_mask)[0])

print(f"\nTrain: {len(y_train):,}")
for c in [0,1,2]: print(f"  Class {c}: {(y_train==c).sum():,} ({(y_train==c).mean()*100:.1f}%)")
print(f"Test:  {len(y_test):,}")
for c in [0,1,2]: print(f"  Class {c}: {(y_test==c).sum():,} ({(y_test==c).mean()*100:.1f}%)")

# Train 3-class RF
print("\nTraining 3-class RandomForest...")
model=RandomForestClassifier(n_estimators=100, max_depth=8, min_samples_split=10,
                             min_samples_leaf=5, random_state=42, n_jobs=-1, class_weight='balanced')
model.fit(X_train, y_train)
probs=model.predict_proba(X_test)  # [N, 3] - [BUY_prob, SELL_prob, NEUTRAL_prob]

# Evaluate
buy_prob=probs[:,0]; sell_prob=probs[:,1]; neutral_prob=probs[:,2]

# Determine best class threshold sweep
print(f"\n{'='*60}")
print(f"3-class OOS results (3R:3R, TP=+3*ATR, SL=-3*ATR)")
print(f"{'='*60}")
print(f"{'Thresh':>8} {'BUY trades':>10} {'BUY WR':>8} {'SELL trades':>10} {'SELL WR':>8} {'Total':>10} {'EV':>8} {'PnL(R)':>10}")
print(f"{'-'*70}")

for th in [x/100 for x in range(40,80,5)]:
    buy_idx=np.where(buy_prob>=th)[0]
    sell_idx=np.where(sell_prob>=th)[0]
    # Avoid overlap - pick higher prob class
    combined={}
    for i in buy_idx: combined[i]=(0,buy_prob[i])
    for i in sell_idx:
        if i in combined:
            if sell_prob[i]>combined[i][1]: combined[i]=(1,sell_prob[i])
        else: combined[i]=(1,sell_prob[i])

    trades=len(combined); buys=sum(1 for v in combined.values() if v[0]==0); sells=trades-buys
    if trades==0: continue
    correct=0
    for i,(pred,_) in combined.items():
        if pred==y_test[i]: correct+=1
    wr=correct/trades
    # For EV: assume correct BUY/SELL pays 3R, wrong loses 3R
    win_rate_buy=sum(1 for i,v in combined.items() if v[0]==0 and y_test[i]==0)/max(buys,1)
    win_rate_sell=sum(1 for i,v in combined.items() if v[0]==1 and y_test[i]==1)/max(sells,1)
    # WR for each class specifically
    buy_correct=sum(1 for i,v in combined.items() if v[0]==0 and y_test[i]==0)
    sell_correct=sum(1 for i,v in combined.items() if v[0]==1 and y_test[i]==1)
    buy_wr=buy_correct/max(buys,1); sell_wr=sell_correct/max(sells,1)
    ev=wr*3-(1-wr)*3; pnl=correct*3-(trades-correct)*3
    print(f'{th:.2f} {buys:>10} {buy_wr:>7.1%} {sells:>10} {sell_wr:>7.1%} {trades:>10,} {ev:>7.2f}R {pnl:>+9,}R')

# Compare with current 1-class approach on same test data
print(f"\n{'='*60}")
print(f"Comparison: 1-class (current) vs 3-class on same OOS test set")
print(f"{'='*60}")

# Load v2 1-class model
with open("ml_xau_m15h4_v2.pkl","rb") as f: m1=pickle.load(f)["model"]
p1=m1.predict_proba(X_test)[:,1]  # BUY probability (old model)

for th in [0.55, 0.60]:
    # 1-class approach
    buy1=(p1>=th).sum(); sell1=(p1<=(1-th)).sum()
    buy1_wr=y_test[p1>=th] if (p1>=th).sum()>0 else []
    sell1_wr=1-y_test[p1<=(1-th)] if (p1<=(1-th)).sum()>0 else []
    print(f"\n  1-class (th={th}): BUY={buy1}, SELL={sell1}, WAIT={len(y_test)-buy1-sell1}")
    print(f"    Note: 1-class uses BUY/not-BUY labels, not comparable to 3-class labels directly")
