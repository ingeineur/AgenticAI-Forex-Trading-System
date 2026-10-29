import os
import asyncio
import json
from openai import OpenAI

client = OpenAI(api_key = os.environ["OPENAI_API_KEY"])

def format_prompt(data):
    return f"""
You are a trading assistant.

Analyze this market data and identify:
- bullish setups
- bearish setups
- overbought/oversold assets
- best trade candidates

DATA:
{json.dumps(data, indent=2)}

Return:
- top 3 trade ideas
- reasoning
"""

async def analyze(result):
    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {"role": "user", "content": format_prompt(result)}
        ]
    )
    print(response.choices[0].message.content)