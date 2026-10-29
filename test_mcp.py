import asyncio
from mcp import ClientSession
from mcp.client.stdio import stdio_client
from mcp.client.stdio import StdioServerParameters
from test_openai import analyze
from test_ollama import analyze_with_llm
import json

def normalize_data(cleaned):
    normalized = []

    for item in cleaned:
        if isinstance(item, str):
            try:
                parsed = json.loads(item)
                normalized.append(parsed)
            except json.JSONDecodeError:
                print("❌ Failed to parse:", item[:50])
        elif isinstance(item, dict):
            normalized.append(item)

    return normalized

def clean_mcp(obj):
    # TextContent → extract text
    if hasattr(obj, "text"):
        return obj.text

    # CallToolResult / container objects
    if hasattr(obj, "content"):
        return [clean_mcp(x) for x in obj.content]

    # list
    if isinstance(obj, list):
        return [clean_mcp(x) for x in obj]

    # dict
    if isinstance(obj, dict):
        return {k: clean_mcp(v) for k, v in obj.items()}

    # fallback (numbers, strings, etc.)
    return obj

def build_ai_context(data):
    item = data[0] if isinstance(data, list) else data

    return {
        "symbol": item.get("symbol"),
        "price": item.get("price_data", {}).get("current_price"),
        "rsi": item.get("rsi", {}).get("value"),
        "macd": item.get("macd", {}).get("crossover"),
        "ema_trend": item.get("ema", {}).get("signals"),
        "bb_position": item.get("bollinger_bands", {}).get("position"),
        "adx_trend": item.get("adx", {}).get("trend_strength"),
        "structure": item.get("market_structure", {}).get("trend"),
        "sentiment": item.get("market_sentiment", {}).get("buy_sell_signal"),
        "support_distance": item.get("support_resistance", {}).get("distance_to_support_pct"),
        "resistance_distance": item.get("support_resistance", {}).get("distance_to_resistance_pct"),
    }

def btc_strategy(data):
    item = data[0] if isinstance(data, list) else data

    score = 0
    reasons = []

    rsi = item.get("rsi", {}).get("value")
    macd = item.get("macd", {})
    ema = item.get("ema", {})
    bb = item.get("bollinger_bands", {})
    adx = item.get("adx", {})
    structure = item.get("market_structure", {})
    sentiment = item.get("market_sentiment", {})
    sr = item.get("support_resistance", {})

    # --------------------
    # 1. TREND (HIGH WEIGHT)
    # --------------------
    if structure.get("trend") == "Bullish":
        score += 2
        reasons.append("Bullish market structure")

    if adx.get("trend_strength") == "Weak/No Trend":
        score -= 1
        reasons.append("Weak trend (avoid strong trades)")

    # --------------------
    # 2. RSI
    # --------------------
    if rsi:
        if rsi < 30:
            score += 2
            reasons.append("RSI oversold")
        elif rsi > 70:
            score -= 2
            reasons.append("RSI overbought")
        elif 45 <= rsi <= 60:
            score += 1
            reasons.append("Healthy RSI range")

    # --------------------
    # 3. MACD
    # --------------------
    if macd.get("crossover") == "Bullish":
        score += 2
        reasons.append("MACD bullish crossover")
    elif macd.get("crossover") == "Bearish":
        score -= 2
        reasons.append("MACD bearish crossover")

    # --------------------
    # 4. EMA STACK
    # --------------------
    ema_signals = ema.get("signals", [])
    if any("above EMA50" in s for s in ema_signals):
        score += 1
        reasons.append("Above EMA50 (uptrend)")

    if any("above EMA200" in s for s in ema_signals):
        score += 1
        reasons.append("Above EMA200 (long trend bullish)")

    # --------------------
    # 5. BOLLINGER POSITION
    # --------------------
    if bb.get("position") == "Upper Half":
        score += 1
        reasons.append("Strong price position")

    if bb.get("squeeze"):
        score += 1
        reasons.append("Volatility squeeze (breakout incoming)")

    # --------------------
    # 6. SUPPORT / RESISTANCE
    # --------------------
    if sr.get("distance_to_support_pct", 100) < 2:
        score += 1
        reasons.append("Near support")

    if sr.get("distance_to_resistance_pct", 100) < 1:
        score -= 1
        reasons.append("Near resistance")

    # --------------------
    # FINAL DECISION
    # --------------------
    if score >= 5:
        signal = "BUY"
    elif score <= -3:
        signal = "SELL"
    else:
        signal = "HOLD"

    return {
        "symbol": item.get("symbol"),
        "signal": signal,
        "score": score,
        "rsi": rsi,
        "reasons": reasons
    }

def strategy_engine(data):
    trades = []

    for item in data:

        # 🔥 FIX: convert string → dict
        if isinstance(item, str):
            try:
                item = json.loads(item)
                #print(item)
            except:
                continue

        if not isinstance(item, dict):
            continue

        ind = item.get("indicators", {})

        rsi = ind.get("RSI")
        close = ind.get("close")
        ema50 = ind.get("EMA50")

        if None in (rsi, close, ema50):
            continue

        # simple example
        if rsi > 70:
            trades.append({
                "symbol": item.get("symbol"),
                "signal": "SELL",
                "rsi": rsi
            })

    return trades

def mcp_to_dict(result):
    return {
        "content": result.content,
        "isError": result.isError
    }

def filter_opportunities(data):
    opportunities = []

    for item in data:
        ind = item.get("indicators", {})
        print(ind)
        rsi = ind.get("RSI")
        close = ind.get("close")
        bb_upper = ind.get("BB_upper")
        bb_lower = ind.get("BB_lower")
        change = item.get("changePercent", 0)

        if rsi is None or close is None:
            continue

        signal = None
        reason = []

        print(rsi)
        # RSI signals
        if rsi < 30:
            signal = "BUY"
            reason.append("RSI oversold")
        elif rsi > 70:
            signal = "SELL"
            reason.append("RSI overbought")

        # Momentum filter
        if abs(change) > 0.15:
            reason.append("Strong momentum")

        # Bollinger Band signals
        if bb_upper and close >= bb_upper:
            signal = "SELL"
            reason.append("Near upper Bollinger band")

        if bb_lower and close <= bb_lower:
            signal = "BUY"
            reason.append("Near lower Bollinger band")

        # Only keep meaningful signals
        if signal:
            opportunities.append({
                "symbol": item.get("symbol"),
                "signal": signal,
                "rsi": rsi,
                "price": close,
                "changePercent": change,
                "reason": ", ".join(reason)
            })

    return opportunities

async def main():

    server = StdioServerParameters(
        command="python",
        args=[
            "-m", "uv", "tool", "run",
            "--from",
            "git+https://github.com/atilaahmettaner/tradingview-mcp.git",
            "tradingview-mcp"
        ]
    )

    async with stdio_client(server) as (read, write):
        async with ClientSession(read, write) as session:

            await session.initialize()

            tools = await session.list_tools()

            print("\n✅ Tools:")
            #for t in tools:
            #    print(t)

            '''result = await session.call_tool("top_gainers", {
                "exchange": "BINANCE"
            })'''

            result = await session.call_tool("coin_analysis", {
                "symbol": "BINANCE:BTCUSDT",
                "interval": "15m"
            })

            print("\n📊 Result:")
            #print(result.content)
            cleaned = clean_mcp(result)
            #js = json.dumps(cleaned, indent=2)
            js = normalize_data(cleaned)
            #print(js)
            with open("cleaned_output.json", "w", encoding="utf-8") as f:
                json.dump(js, f, indent=2, ensure_ascii=False)
            
            aic = build_ai_context(js)
            print(aic)

            #trades = btc_strategy(js)
            #print(trades)
            #res = mcp_to_dict(result)
            #json.dumps(res)
            #json.dumps(mcp_to_dict(result), indent=2)
            #print(result)
            decision = analyze_with_llm(aic)
            print(decision)
            #await asyncio.sleep(30)

asyncio.run(main())