#!/usr/bin/env python3
"""Regime-Adaptive ML: dynamically adjusts thresholds by H4 trend strength."""
import numpy as np, pandas as pd, pickle, warnings
warnings.filterwarnings('ignore')
eps=1e-9

with open("ml_xau_m15h4.pkl","rb") as f: ml=pickle.load(f)
ml_model=ml["model"]; ml_th=ml.get("threshold",0.55)

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

# Features (28-d)
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
h4_ts_raw=(h4_e20-h4_e50)/(h4_atr+eps)  # trend strength (normalized)
h4_sign=feat[:,25]

# M15 trend strength as well (for additional context)
m15_e20=pd.Series(c).ewm(span=20,adjust=False).mean().to_numpy()
m15_e50=pd.Series(c).ewm(span=50,adjust=False).mean().to_numpy()
m15_ts=(m15_e20-m15_e50)/(atr_vals+eps)

# Labels
labels=np.full(n,-1,dtype=np.int32)
for i in range(n-50):
    if np.isnan(atr_vals[i]) or atr_vals[i]<=0: continue
    tp=c[i]+3.0*atr_vals[i]; sl=c[i]-3.0*atr_vals[i]
    for j in range(i+1,min(i+51,n)):
        if l[j]<=sl: labels[i]=0; break
        if h[j]>=tp: labels[i]=1; break

print("ML inference...")
ml_probs=ml_model.predict_proba(feat)[:,1]

# Build base record
base=pd.DataFrame({"ml_p":ml_probs,"label":labels,"h4_ts":h4_ts_raw,"m15_ts":m15_ts})
base=base[(base["ml_p"]>=ml_th)|(base["ml_p"]<=(1-ml_th))].copy()
base=base[~((np.isnan(base["h4_ts"]))|(np.isnan(base["m15_ts"])))].copy()
print(f"Base signals: {len(base):,}")

# Combined trend strength: blend H4 and M15
base["trend"]=np.clip((base["h4_ts"]+base["m15_ts"])/2,-2,2)

# ============================================================
# STRATEGIES
# ============================================================
def run_fixed(s, th):
    s=s.copy(); b=(s["ml_p"]>=th); sh=(s["ml_p"]<=(1-th))
    s["dir"]="WAIT"; s.loc[b,"dir"]="BUY"; s.loc[sh,"dir"]="SHORT"
    s=s[s["dir"]!="WAIT"]
    s["win"]=((s["dir"]=="BUY")&(s["label"]==1))|((s["dir"]=="SHORT")&(s["label"]==0))
    return s

def run_adaptive(s, base_th, strength, mode="trend"):
    """Regime-adaptive: thresholds slide with trend.
    mode='trend': use combined H4+M15 trend strength.
    mode='h4only': use only H4 trend strength.
    """
    s=s.copy()
    ts=np.clip(s[mode if mode=="trend" else "h4_ts"].values,-2,2)
    # BUY threshold: lower in bull (permissive), higher in bear (strict)
    buy_th=base_th - strength*ts
    # SELL threshold: higher in bear (permissive), lower in bull (strict)
    sell_th=base_th + strength*ts
    # Prevent crossover
    buy_th=np.minimum(buy_th, sell_th-0.03)
    sell_th=np.maximum(sell_th, buy_th+0.03)
    
    buy_ok=s["ml_p"]>=buy_th
    sell_ok=s["ml_p"]<=sell_th
    both=buy_ok&sell_ok
    s["dir"]="WAIT"
    # Tiebreaker: trend direction
    s.loc[both&(ts>=0),"dir"]="BUY"
    s.loc[both&(ts<0),"dir"]="SHORT"
    s.loc[buy_ok&~sell_ok,"dir"]="BUY"
    s.loc[sell_ok&~buy_ok,"dir"]="SHORT"
    s=s[s["dir"]!="WAIT"]
    s["win"]=((s["dir"]=="BUY")&(s["label"]==1))|((s["dir"]=="SHORT")&(s["label"]==0))
    return s

def bucket(x):
    if x>1.5: return "VSTRONGBULL"
    elif x>0.5: return "STRONGBULL"
    elif x>0.15: return "BULL"
    elif x>-0.15: return "NEUTRAL"
    elif x>-0.5: return "BEAR"
    elif x>-1.5: return "STRONGBEAR"
    else: return "VSTRONGBEAR"

results={}
# Fixed thresholds
for th in [0.50, 0.55, 0.60]:
    results[f"Fixed {th:.2f}"]=run_fixed(base, th)

# Adaptive thresholds (varying strength)
for strength in [0.06, 0.10, 0.15, 0.20, 0.30]:
    results[f"Adaptive {strength:.2f}"]=run_adaptive(base, 0.55, strength)

print(f"\n{'='*100}")
print(f"  Regime-Adaptive ML Comparison — XAUUSD")
print(f"{'='*100}")
print(f"  {'Strategy':<25} {'Trades':>8} {'Wins':>8} {'Losses':>8} {'WR':>7} {'EV/R':>7} {'PnL(R)':>10}  BUY/SHORT split")
print(f"  {'-'*25} {'-'*8} {'-'*8} {'-'*8} {'-'*7} {'-'*7} {'-'*10}  {'-'*20}")

for name,s in results.items():
    t=len(s); w=(s["win"]).sum(); l=t-w; wr=w/t; ev=wr*3-(1-wr)*3; pnl=t*ev
    b=s[s["dir"]=="BUY"]; sh=s[s["dir"]=="SHORT"]
    bw=(b["win"]).sum() if len(b)>0 else 0
    sw=(sh["win"]).sum() if len(sh)>0 else 0
    split=f"B:{len(b):,}({bw/len(b):.0%}) S:{len(sh):,}({sw/len(sh):.0%})" if len(b)>0 and len(sh)>0 else ""
    print(f"  {name:<25} {t:>8,} {w:>8,} {l:>8,} {wr:>6.1%} {ev:>+6.2f} {pnl:>+8.0f}  {split}")

# Trend distribution for best adaptive
best_adaptive="Adaptive 0.15"
print(f"\n{'='*100}")
print(f"  Trend Regime Breakdown: {best_adaptive} vs Fixed 0.55")
print(f"{'='*100}")

base["bkt"]=base["h4_ts"].apply(bucket)
for bkt in ["VSTRONGBEAR","STRONGBEAR","BEAR","NEUTRAL","BULL","STRONGBULL","VSTRONGBULL"]:
    sub=base[base["bkt"]==bkt]
    if len(sub)<5: continue
    fixed=run_fixed(sub,0.55); adap=run_adaptive(sub,0.55,0.15)
    ft=len(fixed); fw=(fixed["win"]).sum() if ft>0 else 0
    at=len(adap); aw=(adap["win"]).sum() if at>0 else 0
    n_all=len(sub)
    print(f"  {bkt:<16} all:{n_all:>5}  fixed:{ft:>5} wr={fw/ft:.1%}  "
          f"adaptive:{at:>5} wr={aw/at:.1%}  delta:{at-ft:+d}t {aw/at-fw/ft:+.1%}wr")

# Also show what fraction of signals get flipped vs fixed
print(f"\n{'='*100}")
print(f"  Signal Changes: Adaptive vs Fixed (0.55)")
print(f"{'='*100}")
fixed=run_fixed(base,0.55)
adaptive=run_adaptive(base,0.55,0.15)
# Compare direction at each bar
cmp=base.copy()
cmp["fixed_dir"]="WAIT"; cmp.loc[cmp["ml_p"]>=0.55,"fixed_dir"]="BUY"
cmp.loc[cmp["ml_p"]<=0.45,"fixed_dir"]="SHORT"
cmp["adap_dir"]="WAIT"
ts=np.clip(cmp["trend"].values,-2,2)
buy_th=0.55-0.15*ts; sell_th=0.55+0.15*ts
buy_ok=cmp["ml_p"]>=buy_th; sell_ok=cmp["ml_p"]<=sell_th
both=buy_ok&sell_ok
cmp.loc[both&(ts>=0),"adap_dir"]="BUY"; cmp.loc[both&(ts<0),"adap_dir"]="SHORT"
cmp.loc[buy_ok&~sell_ok,"adap_dir"]="BUY"; cmp.loc[sell_ok&~buy_ok,"adap_dir"]="SHORT"

added=cmp[(cmp["adap_dir"]!="WAIT")&(cmp["fixed_dir"]=="WAIT")]
removed=cmp[(cmp["adap_dir"]=="WAIT")&(cmp["fixed_dir"]!="WAIT")]
flipped=cmp[(cmp["adap_dir"]!="WAIT")&(cmp["fixed_dir"]!="WAIT")&(cmp["adap_dir"]!=cmp["fixed_dir"])]
same=cmp[(cmp["adap_dir"]!="WAIT")&(cmp["fixed_dir"]!="WAIT")&(cmp["adap_dir"]==cmp["fixed_dir"])]
print(f"  New signals added:      {len(added):>5,} ({(added['label']==1).sum()/len(added):.0%} up)")
print(f"  Signals removed:        {len(removed):>5,} ({(removed['label']==1).sum()/len(removed):.0%} up)")
print(f"  Signals flipped:        {len(flipped):>5,}")
print(f"  Signals unchanged:      {len(same):>5,}")
print(f"  Net change:             {len(added)+len(flipped)-len(removed):>+5,}")
