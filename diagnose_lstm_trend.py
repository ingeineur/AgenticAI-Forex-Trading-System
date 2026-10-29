#!/usr/bin/env python3
"""Quick correct WR computation from saved data - fix SHORT WR calc."""
import numpy as np, pandas as pd, pickle, warnings
warnings.filterwarnings('ignore')
eps=1e-9

with open("ml_xau_m15h4.pkl","rb") as f: ml=pickle.load(f)
ml_model=ml["model"]; ml_th=ml.get("threshold",0.55)
short_th=1.0-ml_th

import torch
class LSTMNet(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.lstm=torch.nn.LSTM(10,128,3,batch_first=True,dropout=0.3)
        self.head=torch.nn.Sequential(
            torch.nn.LayerNorm(128),torch.nn.Linear(128,64),
            torch.nn.ReLU(),torch.nn.Dropout(0.3),torch.nn.Linear(64,1))
    def forward(self,x):
        out,_=self.lstm(x); return self.head(out[:,-1,:]).squeeze(-1)

lstm_state=torch.load("lstm_price_action.pt",map_location="cpu",weights_only=False)
lstm=LSTMNet(); lstm.load_state_dict(lstm_state["model_state"]); lstm.eval()
lstm_mean=lstm_state["mean"]; lstm_std=lstm_state["std"]

df=pd.read_csv("XAUUSD_M15.csv")
df["timestamp"]=pd.to_datetime(df["timestamp"],utc=True)
df.sort_values("timestamp",inplace=True); df.reset_index(drop=True,inplace=True)
h4=pd.read_csv("XAUUSD_16388.csv")
h4["timestamp"]=pd.to_datetime(h4["timestamp"],utc=True)
h4.sort_values("timestamp",inplace=True); h4.reset_index(drop=True,inplace=True)
h4.rename(columns={c:f"h4_{c}" for c in h4.columns if c!="timestamp"},inplace=True)
df["ts_round"]=df["timestamp"].dt.floor("4h")
h4["ts_round"]=h4["timestamp"].dt.floor("4h")
h4_cols=["ts_round","h4_open","h4_high","h4_low","h4_close","h4_volume"]
merged=df.merge(h4[h4_cols],on="ts_round",how="left")
merged.dropna(subset=["h4_close"],inplace=True); merged.reset_index(drop=True,inplace=True)
n=len(merged); print(f"Data: {n:,} bars")
o=merged["open"].to_numpy(); h=merged["high"].to_numpy()
l=merged["low"].to_numpy(); c=merged["close"].to_numpy(); v=merged["volume"].to_numpy()
h4_c=merged["h4_close"].to_numpy(); h4_o=merged["h4_open"].to_numpy()
h4_h=merged["h4_high"].to_numpy(); h4_l=merged["h4_low"].to_numpy()

# Features (same as before)
atr_vals=pd.Series(h-l).rolling(14).mean().to_numpy()
feat=np.zeros((n,28),dtype=np.float32)
feat[:,0]=np.abs(c-o)/(atr_vals+eps); feat[:,1]=(h-l)/(atr_vals+eps)
feat[:,2]=(h-np.maximum(o,c))/(atr_vals+eps)
feat[:,3]=(np.minimum(o,c)-l)/(atr_vals+eps)
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
h4_ts=np.where(h4_atr>eps,(h4_e20-h4_e50)/(h4_atr+eps),0.0)

# Labels
labels=np.full(n,-1,dtype=np.int32)
for i in range(n-50):
    if np.isnan(atr_vals[i]) or atr_vals[i]<=0: continue
    tp=c[i]+3.0*atr_vals[i]; sl=c[i]-3.0*atr_vals[i]
    for j in range(i+1,min(i+51,n)):
        if l[j]<=sl: labels[i]=0; break
        if h[j]>=tp: labels[i]=1; break

print("Running ML...")
ml_probs=ml_model.predict_proba(feat)[:,1]
ml_mask=(ml_probs>=ml_th)|(ml_probs<=short_th)
ml_idx=np.where(ml_mask)[0]
valid=ml_idx[(ml_idx>=5)&(ml_idx<n)]

print(f"Building {len(valid):,} LSTM sequences...")
seqs=[]
for idx in valid:
    ws,we=idx-5,idx
    sq=np.column_stack([
        (o[ws:we]/c[idx]-1)*100,(h[ws:we]/c[idx]-1)*100,
        (l[ws:we]/c[idx]-1)*100,(c[ws:we]/c[idx]-1)*100,
        v[ws:we]/(np.mean(v[ws:we])+eps)]).astype(np.float32)
    h4sq=np.column_stack([
        np.full(5,feat[idx,20]),np.full(5,feat[idx,21]),
        np.full(5,feat[idx,24]),np.full(5,feat[idx,25]),
        np.full(5,feat[idx,27])]).astype(np.float32)
    seqs.append(np.concatenate([sq,h4sq],axis=1))
X_seq=np.array(seqs)

print("LSTM batch inference...")
lprobs=[]
with torch.no_grad():
    for b in range(0,len(X_seq),256):
        bn=(X_seq[b:b+256]-lstm_mean)/lstm_std
        lprobs.extend(torch.sigmoid(lstm(torch.from_numpy(bn).float())).numpy())
lprobs=np.array(lprobs)

# Build records with CORRECT WR calculation
records=[]
for k,idx in enumerate(valid):
    ml_dir="BUY" if ml_probs[idx]>=ml_th else "SHORT"
    lstm_p=float(lprobs[k])
    # Correct outcome: label=1 (up first) = WIN for BUY, LOSS for SHORT
    outcome=None
    if labels[idx]==1: outcome="UP_FIRST"
    elif labels[idx]==0: outcome="DOWN_FIRST"
    records.append({
        "ml_dir":ml_dir,"lstm_p":lstm_p,"label":int(labels[idx]),
        "outcome":outcome,"ml_prob":float(ml_probs[idx]),
        "h4_ts":float(h4_ts[idx]),
    })

rf=pd.DataFrame(records)
# BUY wins when label=1 (up first), SHORT wins when label=0 (down first)
rf["ml_win"]=((rf["ml_dir"]=="BUY")&(rf["label"]==1))|((rf["ml_dir"]=="SHORT")&(rf["label"]==0))

# LSTM final decision
rf["lstm_final"]=rf["ml_dir"]
rf.loc[(rf["ml_dir"]=="BUY")&(rf["lstm_p"]<0.50),"lstm_final"]="WAIT"
rf.loc[(rf["ml_dir"]=="SHORT")&(rf["lstm_p"]>0.50),"lstm_final"]="WAIT"
rf["lstm_win"]=((rf["lstm_final"]=="BUY")&(rf["label"]==1))|((rf["lstm_final"]=="SHORT")&(rf["label"]==0))
rf["lstm_traded"]=rf["lstm_final"]!="WAIT"

def bucket(s):
    if s>1: return "STRONG_BULL"
    elif s>0.3: return "BULL"
    elif s>-0.3: return "NEUTRAL"
    elif s>-1.0: return "BEAR"
    else: return "STRONG_BEAR"
rf["bucket"]=rf["h4_ts"].apply(bucket)
orders=["STRONG_BEAR","BEAR","NEUTRAL","BULL","STRONG_BULL"]

# ML baseline (all signals regardless of LSTM)
print("\n==================== ML BASELINE (all signals) ====================")
print(f"  Total: {len(rf):,}, WR: {rf['ml_win'].mean():.1%}")
ml_buy=rf[rf["ml_dir"]=="BUY"]; ml_sh=rf[rf["ml_dir"]=="SHORT"]
print(f"  BUY: {len(ml_buy):,} WR={ml_buy['ml_win'].mean():.1%}")
print(f"  SHORT: {len(ml_sh):,} WR={ml_sh['ml_win'].mean():.1%}")

# LSTM filtered
lstm_traded=rf[rf["lstm_traded"]]
print(f"\n==================== LSTM FILTERED ====================")
print(f"  Total: {len(lstm_traded):,} WR: {lstm_traded['lstm_win'].mean():.1%}")
l_buy=lstm_traded[lstm_traded["lstm_final"]=="BUY"]
l_sh=lstm_traded[lstm_traded["lstm_final"]=="SHORT"]
print(f"  BUY: {len(l_buy):,} WR={l_buy['lstm_win'].mean():.1%}")
print(f"  SHORT: {len(l_sh):,} WR={l_sh['lstm_win'].mean():.1%}")
print(f"  ML trades lost: {(~rf['lstm_traded']).sum():,} ({(~rf['lstm_traded']).mean():.1%})")
# Of the rejected trades, how many were actually losing?
rejected=rf[~rf["lstm_traded"]]
print(f"  Rejected trade correctness: rejected_ML_losing={rejected['ml_win'].mean():.1%} (lower=better)")
# So if all rejected were losers, ml_win rate would be 0. But rejecting winners hurts.

# By trend bucket
print(f"\n{'='*85}")
print(f"  Performance by Trend Regime (CORRECTED WR)")
print(f"{'='*85}")
print(f"  {'Regime':<16} {'N':>7} {'ML_WR':>8} {'LSTM_WR':>9} {'LSTM_N':>7} {'Kept%':>7} {'Rej_loss%':>10}")
print(f"  {'-'*16} {'-'*7} {'-'*8} {'-'*9} {'-'*7} {'-'*7} {'-'*10}")
for b in orders:
    sub=rf[rf["bucket"]==b]
    if len(sub)<10: continue
    lstm_sub=sub[sub["lstm_traded"]]
    rej=sub[~sub["lstm_traded"]]
    rej_loss=1.0-rej["ml_win"].mean()
    print(f"  {b:<16} {len(sub):>7,} {sub['ml_win'].mean():>7.1%} "
          f"{lstm_sub['lstm_win'].mean() if len(lstm_sub)>0 else 0:>8.1%} "
          f"{len(lstm_sub):>7} {len(lstm_sub)/len(sub):>6.1%} {rej_loss:>9.1%}")

# Trend awareness: LSTM prob correlation with h4_ts
print(f"\n{'='*85}")
print(f"  Trend Awareness: LSTM Probability vs H4 Trend")
print(f"{'='*85}")
for b in orders:
    sub=rf[rf["bucket"]==b]
    if len(sub)<20: continue
    bu=sub[sub["ml_dir"]=="BUY"]; sh=sub[sub["ml_dir"]=="SHORT"]
    print(f"  {b:<16} BUY lstm={bu['lstm_p'].mean():.1%} SHORT lstm={sh['lstm_p'].mean():.1%} "
          f"(diff={bu['lstm_p'].mean()-sh['lstm_p'].mean():+.1%})")

# Compare: does LSTM improve ML?
print(f"\n{'='*85}")
print(f"  Does LSTM improve over ML?")
print(f"{'='*85}")
print(f"  ML WR (all):      {rf['ml_win'].mean():.1%}")
print(f"  LSTM WR (traded): {lstm_traded['lstm_win'].mean():.1%}")
print(f"  Improvement:       {lstm_traded['lstm_win'].mean()-rf['ml_win'].mean():+.1%} pts")
# Overall, does LSTM + ML beat ML alone for the same set of trades?
print(f"  Out of {len(rf):,} ML signals, LSTM keeps {len(lstm_traded):,} ({len(lstm_traded)/len(rf):.1%})")
print(f"  ML would have: {rf['ml_win'].sum():,} wins on all {len(rf):,}")
print(f"  LSTM: {lstm_traded['lstm_win'].sum():,} wins on {len(lstm_traded):,}")
print(f"  LSTM wins/ML_wins: {lstm_traded['lstm_win'].sum()/max(rf['ml_win'].sum(),1):.1%} "
      f"(wins kept, but only {len(lstm_traded)/len(rf):.1%} of volume)")
