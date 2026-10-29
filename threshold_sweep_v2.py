#!/usr/bin/env python3
"""Threshold sweep on v2 models (trained through 2026-06-19)."""
import numpy as np, pandas as pd, pickle, warnings; warnings.filterwarnings('ignore'); eps=1e-9

def backtest(merged, probs, tp_m, sl_m, win_r, loss_r, threshold):
    n=len(merged); o=merged['open'].to_numpy(); h=merged['high'].to_numpy()
    l=merged['low'].to_numpy(); c=merged['close'].to_numpy()
    atr=pd.Series(h-l).rolling(14).mean().to_numpy()
    trades=wins=0
    for i in range(50,n):
        if np.isnan(atr[i]) or atr[i]<=0: continue
        if probs[i]>=threshold:
            tp=c[i]+tp_m*atr[i]; sl=c[i]-sl_m*atr[i]
            for j in range(i+1,min(i+50,n)):
                if h[j]>=tp: wins+=1; break
                if l[j]<=sl: break
            trades+=1
        elif probs[i]<=(1-threshold):
            tp=c[i]-tp_m*atr[i]; sl=c[i]+sl_m*atr[i]
            for j in range(i+1,min(i+50,n)):
                if l[j]<=tp: wins+=1; break
                if h[j]>=sl: break
            trades+=1
    if trades==0: return 0,0,0,0
    wr=wins/trades; ev=wr*win_r-(1-wr)*loss_r; pnl=wins*win_r-(trades-wins)*loss_r
    return trades, wr, ev, pnl

def run_sweep(label, model_file, m15_file, h4_file, tp_m, sl_m, win_r, loss_r):
    print(f"\n{'='*60}\n{label}\n{'='*60}")
    with open(model_file,'rb') as f: model=pickle.load(f)['model']
    df=pd.read_csv(m15_file); df['timestamp']=pd.to_datetime(df['timestamp'],utc=True)
    df=df.sort_values('timestamp').reset_index(drop=True)
    h4=pd.read_csv(h4_file); h4['timestamp']=pd.to_datetime(h4['timestamp'],utc=True)
    h4=h4.sort_values('timestamp').reset_index(drop=True)
    h4=h4.rename(columns={c:'h4_'+c for c in h4.columns if c!='timestamp'})
    df['ts_round']=df['timestamp'].dt.floor('4h'); h4['ts_round']=h4['timestamp'].dt.floor('4h')
    h4c=['ts_round','h4_open','h4_high','h4_low','h4_close','h4_volume']
    merged=df.merge(h4[h4c],on='ts_round',how='left').dropna(subset=['h4_close']).reset_index(drop=True)
    print(f'Bars: {len(merged):,} ({merged["timestamp"].iloc[0].date()} to {merged["timestamp"].iloc[-1].date()})')
    o=merged['open'].to_numpy(); h=merged['high'].to_numpy(); l=merged['low'].to_numpy()
    c=merged['close'].to_numpy(); v=merged['volume'].to_numpy()
    h4_c=merged['h4_close'].to_numpy(); h4_o=merged['h4_open'].to_numpy()
    h4_h=merged['h4_high'].to_numpy(); h4_l=merged['h4_low'].to_numpy()
    atr_vals=pd.Series(h-l).rolling(14).mean().to_numpy()
    feat=np.zeros((len(merged),28),np.float32)
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
    probs=model.predict_proba(feat)[:,1]
    print(f"{'Thresh':>8} {'Trades':>8} {'WR':>8} {'EV':>8} {'PnL(R)':>10}")
    for th in [x/100 for x in range(40,80,5)]:
        t,w,e,p=backtest(merged,probs,tp_m,sl_m,win_r,loss_r,th)
        if t: print(f'{th:.2f} {t:>8,} {w:>7.1%} {e:>7.2f}R {p:>+9,}R')

run_sweep("XAUUSD v2 3R:3R","ml_xau_m15h4_v2.pkl","XAUUSD_M15_combined.csv","XAUUSD_16388_combined.csv",3,3,3,3)
run_sweep("XAUUSD v2 2R:1R","ml_xau_m15h4_v2_r2r1.pkl","XAUUSD_M15_combined.csv","XAUUSD_16388_combined.csv",1,2,1,2)
run_sweep("BTCUSD v2 3R:3R","ml_btc_m15h4_v2.pkl","BTCUSD_15_combined.csv","BTCUSD_16388_combined.csv",3,3,3,3)
run_sweep("BTCUSD v2 2R:1R","ml_btc_m15h4_v2_r2r1.pkl","BTCUSD_15_combined.csv","BTCUSD_16388_combined.csv",1,2,1,2)
