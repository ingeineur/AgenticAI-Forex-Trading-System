#!/usr/bin/env python3
"""
BTCUSD M15+H4 Live Deployment — ML only (no LSTM, no trend filter).
Supports 3R:3R and 2R:1R risk profiles, binary and ternary approaches.

Usage:
  python deploy_btc.py                              # Single scan
  python deploy_btc.py --watch                      # Continuous monitoring
  python deploy_btc.py --watch --trade              # Auto-execute signals
  python deploy_btc.py --backtest                   # Backtest on MT5/CSV data
  python deploy_btc.py --risk-profile 2r1r          # Use 2R:1R model
  python deploy_btc.py --approach ternary           # Use 3-class (BUY/SELL/NEUTRAL) approach
  python deploy_btc.py --retracement                # Wait for pullback before entering
  python deploy_btc.py --retracement --max-wait 8   # Wait up to 8 bars for pullback
"""
import os

import numpy as np, pandas as pd, pickle, argparse, time, warnings, sys
warnings.filterwarnings('ignore')
eps = 1e-9
_MT5 = None

MAX_POSITIONS = 3
SYMBOL = "BTCUSD"
PROFILES = {
    "3r3r": {"model_file": "ml_btc_m15h4_v2.pkl", "tp_mult": 3.0, "sl_mult": 3.0, "win_r": 3, "loss_r": 3},
    "2r1r": {"model_file": "ml_btc_m15h4_v2_r2r1.pkl", "tp_mult": 1.0, "sl_mult": 2.0, "win_r": 1, "loss_r": 2},
}
TERNARY_SUFFIX = "_ternary"
TRADE_CONFIG = {
    "account": int(os.getenv("MT5_ACCOUNT", "0")), "password": os.getenv("MT5_PASSWORD", ""),
    "server": "Pepperstone-Demo", "lot": 0.01, "magic": 2024,
    "deviation": 20, "comment": "BTCUSD M15+H4 ML",
}

def load_model(profile, approach):
    cfg = dict(PROFILES[profile])
    is_ternary = (approach == "ternary") and (profile == "3r3r")
    if is_ternary:
        base = cfg["model_file"].replace(".pkl", "")
        cfg["model_file"] = base + TERNARY_SUFFIX + ".pkl"
    print(f"Loading ML model ({profile}, {approach})...")
    if not __import__('os').path.exists(cfg["model_file"]):
        print(f"ERROR: {cfg['model_file']} not found"); return None, cfg
    with open(cfg["model_file"],"rb") as f: art = pickle.load(f)
    print(f"  Model loaded (threshold: {art.get('threshold',0.55):.2f})")
    print(f"  TP={cfg['tp_mult']:.0f}*ATR, SL={cfg['sl_mult']:.0f}*ATR")
    return art["model"], cfg

def build_features(m15, h4_merged):
    o=m15["open"].to_numpy(); h=m15["high"].to_numpy(); l=m15["low"].to_numpy()
    c=m15["close"].to_numpy(); v=m15["volume"].to_numpy()
    h4_c=h4_merged["h4_close"].to_numpy(); h4_o=h4_merged["h4_open"].to_numpy()
    h4_h=h4_merged["h4_high"].to_numpy(); h4_l=h4_merged["h4_low"].to_numpy()
    atr_vals=pd.Series(h-l).rolling(14).mean().to_numpy()
    n=len(m15); feat=np.zeros((n,28),dtype=np.float32)
    feat[:,0]=np.abs(c-o)/(atr_vals+eps); feat[:,1]=(h-l)/(atr_vals+eps)
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
    return feat, atr_vals

def generate_signal(m15, h4, ml_model, threshold, cfg, approach):
    if m15 is None or h4 is None or len(m15)<50 or len(h4)<5: return None
    m15=m15.copy(); h4=h4.copy()
    m15["ts_round"]=m15["timestamp"].dt.floor("4h")
    h4["ts_round"]=h4["timestamp"].dt.floor("4h")
    h4_cols=["ts_round","h4_open","h4_high","h4_low","h4_close","h4_volume"]
    merged=m15.merge(h4[h4_cols],on="ts_round",how="left")
    merged.dropna(subset=["h4_close"],inplace=True); merged.reset_index(drop=True,inplace=True)
    if len(merged)<50: return None
    feat, atr_vals = build_features(merged, merged)
    last_idx=len(merged)-1
    h4_trend="BULLISH" if feat[last_idx,25]>0 else ("BEARISH" if feat[last_idx,25]<0 else "NEUTRAL")
    fv=feat[last_idx:last_idx+1]

    is_ternary = (approach == "ternary")
    if is_ternary:
        probs = ml_model.predict_proba(fv)[0]
        buy_p, sell_p = float(probs[0]), float(probs[1])
        if buy_p >= threshold and (sell_p < threshold or buy_p >= sell_p):
            signal="BUY"; fp=buy_p
        elif sell_p >= threshold and (buy_p < threshold or sell_p > buy_p):
            signal="SHORT"; fp=sell_p
        else:
            signal="WAIT"; fp=max(buy_p, sell_p)
    else:
        ml_prob=float(ml_model.predict_proba(fv)[0,1])
        if ml_prob>=threshold: signal="BUY"; fp=ml_prob
        elif ml_prob<=(1-threshold): signal="SHORT"; fp=1.0-ml_prob
        else: signal="WAIT"; fp=ml_prob

    sig={"timestamp":merged["timestamp"].iloc[last_idx],"close":float(merged["close"].iloc[last_idx]),
         "signal_close":float(merged["close"].iloc[last_idx]),
         "ml_probability":max(buy_p, sell_p) if is_ternary else ml_prob,
         "final_probability":fp,"signal":signal,
         "h4_trend":h4_trend,"atr":float(atr_vals[last_idx]) if not np.isnan(atr_vals[last_idx]) else 0.0,
         "tp":None,"sl":None,"profile":cfg.get("profile_label","3r3r binary")}

    if sig["atr"]>0 and signal in ("BUY","SHORT"):
        if signal=="BUY": sig["tp"]=sig["close"]+cfg["tp_mult"]*sig["atr"]; sig["sl"]=sig["close"]-cfg["sl_mult"]*sig["atr"]
        else: sig["tp"]=sig["close"]-cfg["tp_mult"]*sig["atr"]; sig["sl"]=sig["close"]+cfg["sl_mult"]*sig["atr"]
    return sig

def fetch_mt5(m15_bars=500, h4_bars=500):
    import MetaTrader5 as mt5
    global _MT5
    if _MT5 is None:
        if not mt5.initialize(): print(f"ERROR: MT5 init failed: {mt5.last_error()}"); return None,None
        _MT5=mt5
    if _MT5.symbol_info(SYMBOL) is None: print(f"ERROR: {SYMBOL} not found in MT5"); return None,None
    m15_r=_MT5.copy_rates_from_pos(SYMBOL,_MT5.TIMEFRAME_M15,0,m15_bars)
    h4_r=_MT5.copy_rates_from_pos(SYMBOL,_MT5.TIMEFRAME_H4,0,h4_bars)
    def to_df(rates):
        if rates is None: return None
        df=pd.DataFrame(rates); df["timestamp"]=pd.to_datetime(df["time"],unit="s",utc=True)
        df.rename(columns={c:c.lower() for c in df.columns if c!="time"},inplace=True)
        if "tick_volume" in df.columns: df["volume"]=df["tick_volume"]
        df.sort_values("timestamp",inplace=True); df.reset_index(drop=True,inplace=True); return df
    m15=to_df(m15_r); h4=to_df(h4_r)
    if h4 is not None: h4=h4.rename(columns={c:f"h4_{c}" for c in h4.columns if c!="timestamp"})
    return m15,h4

def mt5_login_trade():
    import MetaTrader5 as mt5
    global _MT5
    if not mt5.initialize(): print(f"ERROR: MT5 init failed: {mt5.last_error()}"); return None
    c=TRADE_CONFIG
    if not mt5.login(c["account"],c["password"],c["server"]):
        print(f"ERROR: MT5 login failed: {mt5.last_error()}"); mt5.shutdown(); return None
    mt5.symbol_select(SYMBOL,True); _MT5=mt5
    print(f"  Logged in: {c['account']} @ {c['server']}"); return mt5

def has_position(mt5):
    pos=mt5.positions_get(symbol=SYMBOL); return pos is not None and len(pos)>MAX_POSITIONS

def place_order(mt5, direction, sl, tp):
    tick=mt5.symbol_info_tick(SYMBOL)
    if tick is None: print(f"  ERROR: symbol_info_tick failed for {SYMBOL}"); return None
    order_type=mt5.ORDER_TYPE_BUY if direction=="long" else mt5.ORDER_TYPE_SELL
    price=tick.ask if direction=="long" else tick.bid
    c=TRADE_CONFIG
    result=mt5.order_send({"action":mt5.TRADE_ACTION_DEAL,"symbol":SYMBOL,"volume":c["lot"],
        "type":order_type,"price":price,"sl":sl,"tp":tp,"deviation":c["deviation"],
        "magic":c["magic"],"comment":c["comment"],"type_time":mt5.ORDER_TIME_GTC,
        "type_filling":mt5.ORDER_FILLING_IOC})
    return result

def print_signal(signal):
    if signal is None: return
    print(f"\n{'='*60}\n{SYMBOL} M15+H4 SIGNAL ({signal.get('profile','?')})")
    print(f"{'='*60}")
    print(f"  Time:   {signal['timestamp']}\n  Price:  ${signal['close']:.2f}")
    print(f"  H4:     {signal['h4_trend']}\n  ML:     {signal['ml_probability']:.1%}")
    print(f"  Final:  {signal['final_probability']:.1%}\n  Signal: {signal['signal']}")
    print(f"  ATR:    ${signal['atr']:.2f}")
    if signal['signal'] in ('BUY','SHORT'):
        print(f"  TP:     ${signal['tp']:.2f}\n  SL:     ${signal['sl']:.2f}")
    print('='*60)

def print_summary(all_signals):
    buys=[s for s in all_signals if s["signal"]=="BUY"]
    shorts=[s for s in all_signals if s["signal"]=="SHORT"]
    print(f"\nTotal: {len(all_signals)}, BUY: {len(buys)}, SHORT: {len(shorts)}")
    if buys: print(f"  Avg BUY ML:   {np.mean([s['ml_probability'] for s in buys]):.1%}")
    if shorts: print(f"  Avg SHORT ML: {np.mean([s['ml_probability'] for s in shorts]):.1%}")

def main():
    parser=argparse.ArgumentParser(description=f"{SYMBOL} M15+H4 ML deployment")
    parser.add_argument("--watch",action="store_true"); parser.add_argument("--trade",action="store_true")
    parser.add_argument("--backtest",action="store_true"); parser.add_argument("--bars",type=int,default=1000)
    parser.add_argument("--h4bars",type=int,default=500); parser.add_argument("--interval",type=int,default=60)
    parser.add_argument("--threshold",type=float,default=0.55,help="ML threshold (default 0.55)")
    parser.add_argument("--risk-profile",choices=list(PROFILES.keys()),default="3r3r",
                        help="Risk profile: 3r3r (default) or 2r1r")
    parser.add_argument("--approach",choices=["binary","ternary"],default="binary",
                        help="Model approach: binary (default) or ternary (3-class BUY/SELL/NEUTRAL)")
    parser.add_argument("--retracement",action="store_true",
                        help="Wait for pullback (re-entry within N bars) before entering")
    parser.add_argument("--max-wait",type=int,default=5,
                        help="Max bars to wait for pullback (default 5)")
    args=parser.parse_args()

    print(f"{'='*60}\n{SYMBOL} M15+H4 ML DEPLOYMENT\n{'='*60}\n")
    if args.trade and not args.watch: print("ERROR: --trade requires --watch"); return

    ml_model,cfg=load_model(args.risk_profile, args.approach)
    if ml_model is None: return
    cfg["profile_label"]=f"{args.risk_profile} {args.approach}"
    is_ternary=(args.approach=="ternary")

    if args.backtest:
        print("Running backtest...")
        m15,h4=fetch_mt5(args.bars,args.h4bars)
        if m15 is None or h4 is None:
            print("MT5 unavailable, trying CSV...")
            try:
                m15=pd.read_csv("BTCUSD_15_combined.csv"); m15["timestamp"]=pd.to_datetime(m15["timestamp"],utc=True)
                h4=pd.read_csv("BTCUSD_16388_combined.csv"); h4["timestamp"]=pd.to_datetime(h4["timestamp"],utc=True)
                h4.rename(columns={c:f"h4_{c}" for c in h4.columns if c!="timestamp"},inplace=True)
            except Exception as e: print(f"CSV error: {e}"); return
        print(f"  M15: {len(m15)} bars, H4: {len(h4)} bars")
        m15["ts_round"]=m15["timestamp"].dt.floor("4h")
        h4["ts_round"]=h4["timestamp"].dt.floor("4h")
        h4_cols=["ts_round","h4_open","h4_high","h4_low","h4_close","h4_volume"]
        merged=m15.merge(h4[h4_cols],on="ts_round",how="left")
        merged.dropna(subset=["h4_close"],inplace=True); merged.reset_index(drop=True,inplace=True)
        feat,atr=build_features(merged,merged)
        o=merged["open"].to_numpy(); h=merged["high"].to_numpy(); l=merged["low"].to_numpy(); c=merged["close"].to_numpy()
        trades=wins=0; tlong=tshort=0; wlong=wshort=0; skipped=0; retrace_used=0

        def check_entry_long(i):
            if not args.retracement: return i, c[i]
            mw=args.max_wait
            for j in range(i+1, min(i+mw+1, len(merged))):
                if l[j] < c[i]:
                    entry = min(o[j], l[j])
                    return j, entry
            return None, None

        def check_entry_short(i):
            if not args.retracement: return i, c[i]
            mw=args.max_wait
            for j in range(i+1, min(i+mw+1, len(merged))):
                if h[j] > c[i]:
                    entry = max(o[j], h[j])
                    return j, entry
            return None, None

        if is_ternary:
            probs=ml_model.predict_proba(feat)
            for i in range(50,len(merged)):
                if np.isnan(atr[i]) or atr[i]<=0: continue
                bp,sp=probs[i,0],probs[i,1]
                if bp>=args.threshold and (sp<args.threshold or bp>=sp):
                    entry_idx, entry_p = check_entry_long(i)
                    if entry_idx is None: skipped+=1; continue
                    if entry_idx != i: retrace_used+=1
                    tp=entry_p+cfg["tp_mult"]*atr[i]; sl=entry_p-cfg["sl_mult"]*atr[i]; tlong+=1
                    for j in range(entry_idx+1,min(entry_idx+50,len(merged))):
                        if h[j]>=tp: wins+=1; wlong+=1; break
                        if l[j]<=sl: break
                elif sp>=args.threshold and (bp<args.threshold or sp>bp):
                    entry_idx, entry_p = check_entry_short(i)
                    if entry_idx is None: skipped+=1; continue
                    if entry_idx != i: retrace_used+=1
                    tp=entry_p-cfg["tp_mult"]*atr[i]; sl=entry_p+cfg["sl_mult"]*atr[i]; tshort+=1
                    for j in range(entry_idx+1,min(entry_idx+50,len(merged))):
                        if l[j]<=tp: wins+=1; wshort+=1; break
                        if h[j]>=sl: break
        else:
            probs=ml_model.predict_proba(feat)[:,1]
            for i in range(50,len(merged)):
                if np.isnan(atr[i]) or atr[i]<=0: continue
                if probs[i]>=args.threshold:
                    entry_idx, entry_p = check_entry_long(i)
                    if entry_idx is None: skipped+=1; continue
                    if entry_idx != i: retrace_used+=1
                    tp=entry_p+cfg["tp_mult"]*atr[i]; sl=entry_p-cfg["sl_mult"]*atr[i]; tlong+=1
                    for j in range(entry_idx+1,min(entry_idx+50,len(merged))):
                        if h[j]>=tp: wins+=1; wlong+=1; break
                        if l[j]<=sl: break
                elif probs[i]<=(1-args.threshold):
                    entry_idx, entry_p = check_entry_short(i)
                    if entry_idx is None: skipped+=1; continue
                    if entry_idx != i: retrace_used+=1
                    tp=entry_p-cfg["tp_mult"]*atr[i]; sl=entry_p+cfg["sl_mult"]*atr[i]; tshort+=1
                    for j in range(entry_idx+1,min(entry_idx+50,len(merged))):
                        if l[j]<=tp: wins+=1; wshort+=1; break
                        if h[j]>=sl: break

        trades=tlong+tshort
        if trades:
            wr=wins/trades; ev=wr*cfg["win_r"]-(1-wr)*cfg["loss_r"]
            pnl=wins*cfg["win_r"]-(trades-wins)*cfg["loss_r"]
            print(f"\nBacktest: {args.risk_profile} {args.approach}, threshold={args.threshold:.2f}" + (" (retracement)" if args.retracement else ""))
            print(f"  Trades: {trades:,} ({tlong:,}L/{tshort:,}S)  WR: {wr:.1%}  EV: {ev:.2f}R  PnL: {pnl:+,.0f}R")
            if args.retracement: print(f"  Skipped (no retrace): {skipped:,}  Retrace entries: {retrace_used:,}")
        else: print("No trades")
        if _MT5: _MT5.shutdown(); return

    if args.watch:
        print(f"Watching every {args.interval}s ({cfg['profile_label']}, threshold={args.threshold:.2f})...")
        if args.retracement: print(f"  Retracement: waiting up to {args.max_wait} bars for pullback")
        if args.trade:
            mt5=mt5_login_trade()
            if mt5 is None: return
            print("  Trade mode: ACTIVE")
        else: print("  Monitor only (add --trade)")
        all_signals=[]; last_bar=None
        pending_trades=[]
        try:
            while True:
                m15,h4=fetch_mt5(args.bars,args.h4bars)
                if m15 is not None and h4 is not None:
                    latest_ts=m15["timestamp"].iloc[-1]
                    latest_high=float(m15["high"].iloc[-1])
                    latest_low=float(m15["low"].iloc[-1])

                    still_pending=[]
                    for pt in pending_trades:
                        sc=pt["signal_close"]
                        bars_elapsed=m15[m15["timestamp"]>pt["last_bar_ts"]].shape[0]
                        if bars_elapsed>=args.max_wait:
                            print(f"  [RETRACE] {pt['direction'].upper()} signal @ {pt['signal_ts']} expired — no pullback within {args.max_wait} bars")
                            continue
                        retraced=False
                        if pt["direction"]=="long" and latest_low<sc:
                            retraced=True; entry_price=latest_low
                        elif pt["direction"]=="short" and latest_high>sc:
                            retraced=True; entry_price=latest_high
                        if retraced:
                            print(f"  [RETRACE] {'BUY' if pt['direction']=='long' else 'SELL'} retracement @ ~${entry_price:.2f} (signal @ ${sc:.2f})")
                            if args.trade:
                                if has_position(_MT5): print("  Position exists — skipping retrace entry")
                                else:
                                    atr=pt["atr"]
                                    if pt["direction"]=="long":
                                        tp=entry_price+cfg["tp_mult"]*atr; sl=entry_price-cfg["sl_mult"]*atr
                                    else:
                                        tp=entry_price-cfg["tp_mult"]*atr; sl=entry_price+cfg["sl_mult"]*atr
                                    r=place_order(_MT5,pt["direction"],sl,tp)
                                    if r and r.retcode==10009:
                                        print(f"  >>> RETRACE FILLED: ticket={r.order}")
                                    elif r: print(f"  >>> RETRACE FAILED: retcode={r.retcode}")
                        else:
                            pt["last_bar_ts"]=latest_ts
                            still_pending.append(pt)
                    pending_trades=still_pending

                    sig=generate_signal(m15,h4,ml_model,args.threshold,cfg,args.approach)
                    if sig and sig["timestamp"]!=last_bar:
                        last_bar=sig["timestamp"]; all_signals.append(sig); print_signal(sig)
                        if sig["signal"] in ("BUY","SHORT") and sig["tp"] is not None:
                            if args.retracement:
                                direction="long" if sig["signal"]=="BUY" else "short"
                                pending_trades.append({
                                    "signal_close":sig["signal_close"],
                                    "direction":direction,
                                    "last_bar_ts":sig["timestamp"],
                                    "signal_ts":sig["timestamp"],
                                    "atr":sig["atr"],
                                })
                                print(f"  [RETRACE] Awaiting pullback (max {args.max_wait} bars)...")
                            else:
                                if args.trade:
                                    if has_position(_MT5): print("  Position exists — skipping")
                                    else:
                                        d="long" if sig["signal"]=="BUY" else "short"
                                        r=place_order(_MT5,d,sig["sl"],sig["tp"])
                                        if r: print(f"  >>> {'FILLED' if r.retcode==10009 else 'FAILED'}: ticket={r.order}" if r.retcode==10009 else f"  >>> FAILED: retcode={r.retcode}")
                        print_summary(all_signals)
                time.sleep(args.interval)
        except KeyboardInterrupt: print("\nStopped"); print_summary(all_signals)
        finally:
            if _MT5: _MT5.shutdown()
        return

    m15,h4=fetch_mt5(args.bars,args.h4bars)
    if m15 is None or h4 is None:
        try:
            m15=pd.read_csv("BTCUSD_15_combined.csv"); m15["timestamp"]=pd.to_datetime(m15["timestamp"],utc=True)
            h4=pd.read_csv("BTCUSD_16388_combined.csv"); h4["timestamp"]=pd.to_datetime(h4["timestamp"],utc=True)
            h4.rename(columns={c:f"h4_{c}" for c in h4.columns if c!="timestamp"},inplace=True)
        except Exception as e: print(f"Error: {e}"); return
    print(f"  M15: {len(m15)} bars  H4: {len(h4)} bars")
    sig=generate_signal(m15,h4,ml_model,args.threshold,cfg,args.approach); print_signal(sig)
    if args.retracement and sig and sig["signal"] in ("BUY","SHORT"):
        print(f"  (Retracement mode: wait for pullback within {args.max_wait} bars, then execute manually)")
    if _MT5: _MT5.shutdown()

if __name__=="__main__":
    main()
