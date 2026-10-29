#!/usr/bin/env python3
"""Full comparison: all thresholds + adaptive variants on complete data."""
import numpy as np, pandas as pd, pickle, warnings
warnings.filterwarnings('ignore')
eps=1e-9

with open("ml_xau_m15h4.pkl","rb") as f: ml=pickle.load(f)
ml_model=ml["model"]

df=pd.read_csv("XAUUSD_M15.csv")
df["timestamp"]=pd.to_datetime(df["timestamp"],utc=True)
df.sort_values("timestamp",inplace=True); df.reset_index(drop=True,inplace=True)
h4=pd.read_csv("XAUUSD_16388.csv")
h4["timestamp"]=pd.to_datetime(h4["timestamp"],utc=True); h4.sort_values("timestamp",inplace=True)
h4.reset_index(drop=True,inplace=True); h4.rename(columns={c:f"h4_{c}" for c in h4.columns if c!="timestamp"},inplace=True)
df["ts_round"]=df["timestamp"].dt.floor("4h"); h4["ts_round"]=h4["timestamp"].dt.floor("4h")
h4_cols=["ts_round","h4_open","h4_high","h4_low","h4_close","h4_volume"]
merged=df.merge(h4[h4_cols],on="ts_round",how="left")
merged.dropna(subset=["h4_close"],inplace=True); merged.reset_index(drop=True,inplace=True)
n=len(merged); print(f"Data: {n:,} bars")

# Features
o=merged["open"].to_numpy(); h=merged["high"].to_numpy(); l=merged["low"].to_numpy()
c=merged["close"].to_numpy(); v=merged["volume"].to_numpy()
h4_c=merged["h4_close"].to_numpy(); h4_o=merged["h4_open"].to_numpy()
h4_h=merged["h4_high"].to_numpy(); h4_l=merged["h4_low"].to_numpy()

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
h4_ts_raw=(h4_e20-h4_e50)/(h4_atr+eps)

# Labels
labels=np.full(n,-1,dtype=np.int32)
for i in range(n-50):
    if np.isnan(atr_vals[i]) or atr_vals[i]<=0: continue
    tp=c[i]+3.0*atr_vals[i]; sl=c[i]-3.0*atr_vals[i]
    for j in range(i+1,min(i+51,n)):
        if l[j]<=sl: labels[i]=0; break
        if h[j]>=tp: labels[i]=1; break

print("ML inference on ALL bars...")
ml_probs=ml_model.predict_proba(feat)[:,1]
valid=~np.isnan(atr_vals)&(atr_vals>0)&(labels>=0)&(np.arange(n)>=50)

# Build full signal set (no pre-filtering)
base=pd.DataFrame({"ml_p":ml_probs,"label":labels,"h4_ts":h4_ts_raw})
base=base[valid].dropna().copy()
print(f"All valid bars: {len(base):,}")

def sim(s, dir_col="dir"):
    s=s[s[dir_col]!="WAIT"].copy()
    s["win"]=((s[dir_col]=="BUY")&(s["label"]==1))|((s[dir_col]=="SHORT")&(s["label"]==0))
    return s

def fixed(s, th):
    s=s.copy()
    s["dir"]="WAIT"
    s.loc[s["ml_p"]>=th,"dir"]="BUY"
    s.loc[s["ml_p"]<=(1-th),"dir"]="SHORT"
    return sim(s)

def adaptive(s, base_th, strength):
    s=s.copy()
    ts=np.clip(s["h4_ts"].values,-2,2)
    buy_th=base_th-strength*ts; sell_th=base_th+strength*ts
    buy_th=np.minimum(buy_th,sell_th-0.03); sell_th=np.maximum(sell_th,buy_th+0.03)
    bo=s["ml_p"]>=buy_th; so=s["ml_p"]<=sell_th; both=bo&so
    s["dir"]="WAIT"
    s.loc[both&(ts>=0),"dir"]="BUY"; s.loc[both&(ts<0),"dir"]="SHORT"
    s.loc[bo&~so,"dir"]="BUY"; s.loc[so&~bo,"dir"]="SHORT"
    return sim(s)

results={}
for th in [0.50,0.55,0.58,0.60,0.62,0.65,0.70]:
    results[f"Fixed {th:.2f}"]=fixed(base,th)
for st in [0.08,0.12,0.15,0.20]:
    results[f"Adapt {st:.2f}"]=adaptive(base,0.55,st)

print(f"\n{'='*85}")
print(f"  All strategies — full 90k bar backtest")
print(f"{'='*85}")
print(f"  {'Strategy':<16} {'Trades':>8} {'Wins':>8} {'WR':>7} {'EV/R':>7} {'PnL(R)':>10}  {'BUY':>14}  {'SHORT':>14}")
print(f"  {'-'*16} {'-'*8} {'-'*8} {'-'*7} {'-'*7} {'-'*10}  {'-'*14}  {'-'*14}")

best_pnl=-999; best_name=""
for name,s in results.items():
    t=len(s); w=(s["win"]).sum(); wr=w/t; ev=wr*3-(1-wr)*3; pnl=t*ev
    b=s[s["dir"]=="BUY"]; sh=s[s["dir"]=="SHORT"]
    bw=(b["win"]).sum() if len(b)>0 else 0; sw=(sh["win"]).sum() if len(sh)>0 else 0
    b_str=f"B:{len(b):,}({bw/len(b):.0%})" if len(b)>0 else "B:0"
    s_str=f"S:{len(sh):,}({sw/len(sh):.0%})" if len(sh)>0 else "S:0"
    print(f"  {name:<16} {t:>8,} {w:>8,} {wr:>6.1%} {ev:>+6.2f} {pnl:>+8.0f}  {b_str:>14}  {s_str:>14}")
    if pnl>best_pnl: best_pnl=pnl; best_name=name

# Trend stability
print(f"\n{'='*85}")
print(f"  Trend Regime: Best Fixed ({best_name})")
print(f"{'='*85}")
def bucket(x):
    if x>1.5: return "VBULL"
    elif x>0.3: return "BULL"
    elif x>-0.3: return "NEUT"
    elif x>-1.5: return "BEAR"
    else: return "VBEAR"
best_s=results[best_name]
best_s["bkt"]=best_s["h4_ts"].apply(bucket)
for bkt in ["VBEAR","BEAR","NEUT","BULL","VBULL"]:
    sub=best_s[best_s["bkt"]==bkt]
    if len(sub)<5: continue
    b=sub[sub["dir"]=="BUY"]; sh=sub[sub["dir"]=="SHORT"]
    bw=(b["win"]).sum()/len(b) if len(b)>0 else 0
    sw=(sh["win"]).sum()/len(sh) if len(sh)>0 else 0
    print(f"  {bkt:<8} {len(sub):>6,} trades {sub['win'].sum()/len(sub):.1%}wr  "
          f"BUY:{len(b):,}({bw:.0%})  SHORT:{len(sh):,}({sw:.0%})")

# Trend distribution: fixed 0.55 vs fixed 0.60
print(f"\n{'='*85}")
print(f"  Trade distribution by H4 trend strength")
print(f"{'='*85}")
base["bkt"]=base["h4_ts"].apply(bucket)
print(f"  {'Bucket':<8} {'All bars':>10} {'Fixed 0.55':>16} {'Fixed 0.60':>16}")
for bkt in ["VBEAR","BEAR","NEUT","BULL","VBULL"]:
    sub=base[base["bkt"]==bkt]
    all_n=len(sub)
    f55=fixed(sub,0.55); f60=fixed(sub,0.60)
    f55_w=(f55["win"]).sum()/len(f55) if len(f55)>0 else 0
    f60_w=(f60["win"]).sum()/len(f60) if len(f60)>0 else 0
    print(f"  {bkt:<8} {all_n:>10,}  {len(f55):>6,}wr={f55_w:.0%}  {len(f60):>6,}wr={f60_w:.0%}")
