import ollama

def analyze_with_llm(data):
    prompt = f"""
You are a professional trading assistant.

Analyze this market data and produce:
- BUY / SELL / HOLD decision
- short reasoning
- risk level (low/medium/high)

DATA:
{data}
"""

    response = ollama.chat(
        model="qwen2.5:7b",
        messages=[
            {"role": "user", "content": prompt}
        ]
    )

    print(response)

    return response["message"]["content"]