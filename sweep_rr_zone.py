#!/usr/bin/env python3
"""Full zone strategy backtest for selected R:R values (with correct blocking)."""

import numpy as np
import pandas as pd
import math, warnings
warnings.filterwarnings('ignore')

SMALL_VAL = -1e-9

def precompute_fractals(highs, lows, strength=2):
    n=len(highs); f_high=np.zeros(n,bool); f_low=np.zeros(n,bool)
    for i in range(strength,n-strength):
        f_high[i]=all(highs[i]>highs[i-k] and highs[i]>highs[i+k] for k in range(1,strength+1))
        f_low[i]=all(lows[i]<lows[i-k] and lows[i]<lows[i+k] for k in range(1,strength+1))
    return f_high,f_low

def get_fractal_snr_fast(f_high,f_low,highs,lows,end,lookback):
    start=end-lookback+1
    if start<0: start=0
    res=None; sup=None
    for i in range(end,start-1,-1):
        if res is None and f_high[i]: res=highs[i]
        if sup is None and f_low[i]: sup=lows[i]
        if res is not None and sup is not None: break
    return sup,res

def bar_in_zone(h,l,zl,zh):
    return not (h<zl or l>zh)

def run(symbol,csv_path,risk_usd,lookback,strength,reward_mult):
    pm={"XAUUSD":1.0,"BTCUSD":100.0}[symbol]
    df=pd.read_csv(csv_path)
    df["timestamp"]=pd.to_datetime(df["timestamp"],utc=True)
    df.sort_values("timestamp",inplace=True); df.reset_index(drop=True,inplace=True)
    df["ema50"]=df["close"].ewm(span=50).mean()
    df["ema200"]=df["close"].ewm(span=200).mean()
    df["body"]=(df["close"]-df["open"]).abs()
    df["atr"]=(df["high"]-df["low"]).rolling(14).mean()
    df["avg_body"]=df["body"].rolling(20).mean()
    df["avg_vol"]=df["volume"].rolling(20).mean()
    n=len(df)
    f_high,f_low=precompute_fractals(df['high'].values,df['low'].values,strength)
    risk_dist=risk_usd*pm; reward_dist=reward_mult*risk_dist
    
    trades=[]
    state={"support":SMALL_VAL,"resistance":SMALL_VAL,
           "active_bo_buy":False,"active_bo_sell":False,
           "buy_resistance_zone":SMALL_VAL,"sell_support_zone":SMALL_VAL,
           "active":True,"in_trade":False,
           "trade_entry":0,"trade_sl":0,"trade_tp":0,"trade_side":None}
    
    i=lookback+strength+10
    while i<n:
        row=df.iloc[i]; o,h,l,c=row['open'],row['high'],row['low'],row['close']
        atr_val=row['atr']
        if np.isnan(atr_val) or atr_val<=0: i+=1; continue
        
        if state["in_trade"]:
            if state["trade_side"]=="BUY":
                if h>=state["trade_tp"]:
                    trades.append((i,"BUY","WIN",state["trade_entry"]))
                    state["in_trade"]=False
                elif l<=state["trade_sl"]:
                    trades.append((i,"BUY","LOSS",state["trade_entry"]))
                    state["in_trade"]=False
            else:
                if l<=state["trade_tp"]:
                    trades.append((i,"SELL","WIN",state["trade_entry"]))
                    state["in_trade"]=False
                elif h>=state["trade_sl"]:
                    trades.append((i,"SELL","LOSS",state["trade_entry"]))
                    state["in_trade"]=False
            if not state["in_trade"]:
                state["active_bo_buy"]=False; state["active_bo_sell"]=False
                state["buy_resistance_zone"]=SMALL_VAL; state["sell_support_zone"]=SMALL_VAL
            i+=1; continue
        
        sup,res=get_fractal_snr_fast(f_high,f_low,df['high'].values,df['low'].values,i,lookback)
        if sup is None or res is None:
            seg=df.iloc[i-lookback+1:i+1]; sup=seg['low'].min(); res=seg['high'].max()
        if state["resistance"]!=SMALL_VAL and not math.isclose(res,state["resistance"],rel_tol=1e-6):
            state["active_bo_buy"]=False; state["buy_resistance_zone"]=SMALL_VAL
        if state["support"]!=SMALL_VAL and not math.isclose(sup,state["support"],rel_tol=1e-6):
            state["active_bo_sell"]=False; state["sell_support_zone"]=SMALL_VAL
        if not math.isclose(state["resistance"],res,rel_tol=1e-6) or not math.isclose(state["support"],sup,rel_tol=1e-6):
            state["active"]=True
        state["resistance"]=res; state["support"]=sup
        
        if state["active_bo_buy"] or state["active_bo_sell"]:
            if state["active_bo_buy"]:
                zl=state["buy_resistance_zone"]-atr_val*0.25
                zh=state["buy_resistance_zone"]+atr_val*0.25
            else:
                zl=state["sell_support_zone"]-atr_val*0.25
                zh=state["sell_support_zone"]+atr_val*0.25
            if bar_in_zone(h,l,zl,zh):
                entry=(h+l)/2
                if state["active_bo_buy"]:
                    state["trade_sl"]=entry-risk_dist; state["trade_tp"]=entry+reward_dist
                else:
                    state["trade_sl"]=entry+risk_dist; state["trade_tp"]=entry-reward_dist
                state["in_trade"]=True; state["trade_entry"]=entry
                state["trade_side"]="BUY" if state["active_bo_buy"] else "SELL"
            i+=1; continue
        
        buffer=atr_val*0.1
        bo_up=c>(res+buffer) and o<res
        bo_down=c<(sup-buffer) and o>sup
        direction=None
        if bo_up and state["active"]: state["active_bo_buy"]=True; direction="BUY"
        elif bo_down and state["active"]: state["active_bo_sell"]=True; direction="SELL"
        
        if direction is not None:
            body=row['body']
            if body<=SMALL_VAL: state["active_bo_buy"]=False; state["active_bo_sell"]=False; i+=1; continue
            score=0
            if direction=="BUY": score+=2 if row['ema50']>row['ema200'] else -1
            else: score+=2 if row['ema50']<row['ema200'] else -1
            if body>row['avg_body']: score+=2
            if row['volume']>row['avg_vol']: score+=1
            upper_wick=h-max(c,o); lower_wick=min(c,o)-l
            if body>0:
                if direction=="BUY" and upper_wick/body>0.6: state["active_bo_buy"]=False; state["active_bo_sell"]=False; i+=1; continue
                if direction=="SELL" and lower_wick/body>0.6: state["active_bo_buy"]=False; state["active_bo_sell"]=False; i+=1; continue
            crange=h-l
            if direction=="BUY": gap=(h-res)/crange if crange>0 else 0
            else: gap=(sup-l)/crange if crange>0 else 0
            if gap<0.3: state["active_bo_buy"]=False; state["active_bo_sell"]=False; i+=1; continue
            prob=max(0,min(1,(score+5)/10))
            if prob>=0.6 and (crange/pm)<2*(risk_usd-1.0):
                mid=(h+l)/2
                if direction=="BUY": state["buy_resistance_zone"]=res if gap>=0.6 else mid
                else: state["sell_support_zone"]=sup if gap>=0.6 else mid
                state["active"]=False
        i+=1
    
    if not trades: return 0,0,0,0,0,0,0,0
    dft=pd.DataFrame(trades,columns=["bar","dir","result","entry"])
    total=len(dft); wins=(dft["result"]=="WIN").sum(); wr=wins/total
    ev=wr*reward_mult-(1-wr)*1.0
    eq=0.0; peak=0.0; dd=0.0; buys=0; sell=0; bw=0; sw=0
    for _,r in dft.iterrows():
        pnl=reward_dist if r["result"]=="WIN" else -risk_dist
        eq+=pnl; peak=max(peak,eq); dd=min(dd,eq-peak)
        if r["dir"]=="BUY": buys+=1; bw+=1 if r["result"]=="WIN" else 0
        else: sell+=1; sw+=1 if r["result"]=="WIN" else 0
    return total,wr,ev,eq,dd,buys,bw,sell,sw,df["timestamp"].iloc[0],df["timestamp"].iloc[-1]

# --- RUN ---
rrs=[0.5,1.0,1.5,2.0,2.5,3.0,4.0]
print(f"\n{'='*70}")
print(f"  Zone Strategy - R:R Sweep (full blocking, independent runs)")
print(f"{'='*70}")
for sym,csv in [("XAUUSD","XAUUSD_M15.csv"),("BTCUSD","BTCUSD_15.csv")]:
    for risk in [50]:
        for lb,st in [(50,2)]:
            print()
            print(f"  {sym}  risk=${risk}  lookback={lb}  strength={st}")
            print(f"  {'R:R':>6} {'Trades':>7} {'Win%':>8} {'EV(R)':>9} {'NetP&L':>12} {'MaxDD':>10}  BUY        SELL")
            print(f"  {'-----':>6} {'-------':>7} {'--------':>8} {'---------':>9} {'------------':>12} {'----------':>10}  ----------  ----------")
            for rm in rrs:
                r=run(sym,csv,risk,lb,st,rm)
                if r[0]==0: continue
                t,wr,ev,pnl,dd,bc,bw,sc,sw,t0,t1=r
                pnl_s=f"${pnl:>+,.0f}" if abs(pnl)<1e6 else f"${pnl/1000:>+,.0f}K"
                dd_s=f"${dd:>+,.0f}" if abs(dd)<1e6 else f"${dd/1000:>+,.0f}K"
                buy_s=f"{bc}t WR={bw/bc:.1%}" if bc else "─"
                sell_s=f"{sc}t WR={sw/sc:.1%}" if sc else "─"
                print(f"  1:{rm:<4}   {t:>7} {wr:>7.1%} {ev:>+8.3f}R {pnl_s:>12} {dd_s:>10}  {buy_s:>12} {sell_s:>12}")
