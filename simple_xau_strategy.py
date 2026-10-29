
def xau_strategy(df):
    last = df.iloc[-1]
    print(f"rsi: {last['rsi']}")
    print(f"close: {last['close']}")
    print(f"ema50: {last['ema50']}")
    if last['rsi'] < 30 and last['close'] > last['ema50']:
        return "BUY"

    elif last['rsi'] > 70 and last['close'] < last['ema50']:
        return "SELL"

    return "HOLD"