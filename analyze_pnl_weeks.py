import pandas as pd
import numpy as np

df = pd.read_csv('trades_60d.csv')
df['entry_time'] = pd.to_datetime(df['entry_time'], utc=True)
df['exit_time'] = pd.to_datetime(df['exit_time'], utc=True)
df['dow'] = df['entry_time'].dt.dayofweek          # 0=Mon
df['dom'] = df['entry_time'].dt.day                # day of month
df['month'] = df['entry_time'].dt.month
df['year'] = df['entry_time'].dt.year

def week_of_month(dom, dow):
    # week 1 = days 1-7, week 2 = 8-14, week 3 = 15-21, week 4 = 22-28, week 5 = 29+
    if dom <= 7: return 1
    if dom <= 14: return 2
    if dom <= 21: return 3
    if dom <= 28: return 4
    return 5

df['week'] = df.apply(lambda r: week_of_month(r['dom'], r['dow']), axis=1)
df['dow_name'] = df['dow'].map({0:'Mon',1:'Tue',2:'Wed',3:'Thu',4:'Fri',5:'Sat',6:'Sun'})

print("="*70)
print("PNL BY WEEK OF MONTH (last 60 days)")
print("="*70)
for mo in sorted(df['month'].unique()):
    sub = df[df['month'] == mo]
    print(f"\n  Month {mo}: {len(sub)} trades")
    for w in range(1, 6):
        sw = sub[sub['week'] == w]
        if len(sw) == 0: continue
        wr = (sw['pnl'] > 0).mean() * 100
        print(f"    Week {w} (days {w*7-6}-{min(w*7,31)}): {len(sw):3d} trades, "
              f"{wr:5.1f}% WR, PnL={sw['pnl'].sum():+8.2f}, avg={sw['pnl'].mean():+6.2f}")

print("\n" + "="*70)
print("AGGREGATE BY WEEK (both months)")
print("="*70)
agg = df.groupby('week').agg(n=('pnl','count'), pnl=('pnl','sum'),
                             avg=('pnl','mean'), wr=('pnl', lambda x: (x>0).mean()*100),
                             wins=('pnl', lambda x: (x>0).sum()),
                             losses=('pnl', lambda x: (x<=0).sum()))
for w, r in agg.iterrows():
    print(f"  Week {w}: {int(r['n']):3d} trades ({int(r['wins'])}W/{int(r['losses'])}L), "
          f"{r['wr']:5.1f}% WR, PnL={r['pnl']:+8.2f}, avg={r['avg']:+6.2f}")

print("\n" + "="*70)
print("WEEK x SYMBOL")
print("="*70)
piv = df.pivot_table(index='week', columns='symbol', values='pnl', aggfunc='sum', fill_value=0)
print(piv.to_string())
wr_piv = df.pivot_table(index='week', columns='symbol', values='pnl',
                        aggfunc=lambda x: (x>0).mean()*100, fill_value=0)
print("\nWR% by week x symbol:")
print(wr_piv.to_string())

print("\n" + "="*70)
print("WEEK x DIRECTION (BTCUSD)")
print("="*70)
btc = df[df['symbol']=='BTCUSD']
piv2 = btc.pivot_table(index='week', columns='direction', values='pnl', aggfunc='sum', fill_value=0)
print(piv2.to_string())

print("\n" + "="*70)
print("WEEK x SETUP (level/zone) for BTCUSD")
print("="*70)
for w in range(1, 6):
    sw = btc[btc['week'] == w]
    if len(sw) == 0: continue
    print(f"  Week {w}: PnL={sw['pnl'].sum():+8.2f}")
    g = sw.groupby(['level','zone']).agg(n=('pnl','count'), pnl=('pnl','sum'))
    for (lv, zn), r in g.iterrows():
        print(f"      {lv:8s} {zn:5s}: {int(r['n']):3d} trades, PnL={r['pnl']:+8.2f}")

print("\n" + "="*70)
print("DAILY PNL + DAY-OF-WEEK breakdown")
print("="*70)
dow_agg = df.groupby('dow_name').agg(n=('pnl','count'), pnl=('pnl','sum'), wr=('pnl', lambda x: (x>0).mean()*100))
order = ['Mon','Tue','Wed','Thu','Fri','Sat','Sun']
dow_agg = dow_agg.reindex([d for d in order if d in dow_agg.index])
print(dow_agg.to_string())

print("\n  Daily PnL timeline (last 60d):")
for _, r in df.sort_values('entry_time').groupby(df['entry_time'].dt.date):
    print(f"    {r['entry_time'].dt.date.iloc[0]}: {len(r):2d} trades, PnL={r['pnl'].sum():+7.2f}, WR={(r['pnl']>0).mean()*100:.0f}%")