import ollama
import json
import re

def parse_score(response):
    raw = response["message"]["content"].strip()
    
    # extract first number (0–100)
    match = re.search(r'\b\d{1,3}\b', raw)
    
    if not match:
        print(f"⚠️ Invalid LLM output: {raw}")
        return 0  # safe fallback
    
    score = int(match.group())
    
    # clamp to 0–100
    score = max(0, min(100, score))
    
    return score

# -------------------------
# LLM DECISION ENGINE
# -------------------------
def analyze_with_llm(data: dict):
    prompt_data = json.dumps(data, indent=2)

    prompt = f"""
You are a trading decision engine.

Return ONLY ONE integer between:
0 and 100

Do NOT explain.

Where:
- 0 = strongly avoid trade
- 50 = neutral / uncertain
- 100 = very high probability trade

1. Evaluate the INTENT (BUY or SELL)

2. Higher Timeframe (HTF) – PRIMARY FACTOR

HTF structure (support/resistance) has the HIGHEST priority and OVERRIDES all other signals.

Assess:
- distance_to_htf_res
- distance_to_htf_sup
- is_near_htf_res
- is_near_htf_sup

HTF LOCATION RULES (STRICT):

A. Momentum continuation trades:

- BUY near HTF resistance:
  → penalize score
  → HIGH scores only allowed if confirmed breakout

- SELL near HTF support:
  → penalize score
  → HIGH scores only allowed if confirmed breakdown

B. Mean-reversion (swing) trades:

- SELL near HTF resistance:
  → increase score ONLY IF:
    - HTF is weakening (flat EMA slope or neutral RSI)
    - LTF momentum is weakening or showing reversal
    - No breakout confirmation

- BUY near HTF support:
  → increase score ONLY IF:
    - HTF is weakening
    - LTF shows bounce or reversal signs
    - No strong bearish momentum

C. Strong level adjustment:

- If HTF resistance/support has many touches (≥ 4):
  → reactions are more likely
  → increase score for reversal trades
  → decrease score for breakout trades unless momentum is very strong


D. Breakout condition (overrides reversal):

- If price has CLOSED beyond HTF level with strong momentum:
  → treat as breakout
  → favor continuation trades
  → ignore reversal bias

HTF TREND:

- Strong HTF trend aligned with INTENT:
  → increase score significantly

- Strong HTF trend against INTENT:
  → decrease score significantly

- Weakening HTF:
  → allow swing trades
  → moderate impact only

3. Directional Bias (LTF)

- Compare bullPct vs bearPct:
  - Large difference → increase score
  - Small difference → decrease score

- If strongly contradicting INTENT:
  → decrease score significantly

4. Momentum (EMA)

- Strong separation and correct direction:
  → increase score

- Weak / flat:
  → decrease score

5. RSI

- >55 supports BUY → increase score
- <45 supports SELL → increase score
- 45–55:
  → neutral → decrease score slightly

6. VWAP (Context)

- Above VWAP favors BUY
- Below VWAP favors SELL

- If aligned:
  → increase score

- If against:
  → decrease score

- Allow override ONLY if:
  - strong momentum
  - strong directional bias
  - large distance from VWAP

- If price is near VWAP:
  → decrease score significantly

7. Final Scoring Guidance

- 80–100:
  - Strong HTF + LTF alignment
  - NOT near HTF S/R (or confirmed breakout)
  - Strong momentum

- 60–79:
  - Good setup with minor conflicts

- 40–59:
  - Mixed signals or moderate risk

- 0–39:
  - Near strong HTF S/R without breakout
  - Weak momentum
  - Conflicting signals

Be conservative.

Never assign high scores when price is near strong HTF support or resistance unless a confirmed breakout is in progress.

DATA:
{prompt_data}
"""

    response = ollama.chat(
        model="qwen2.5:14b",
        messages=[
            {"role": "user", "content": prompt}
        ],
        options={
            "temperature": 0  # deterministic output
        }
    )

    return parse_score(response)
    
def analyze_with_mistral(data: dict):
    prompt_data = json.dumps(data, indent=2)

    prompt = f"""
You are a trading decision engine.

Return ONLY ONE integer between:
0 and 100

Do NOT explain.

Where:
- 0 = strongly avoid trade
- 50 = neutral / uncertain
- 100 = very high probability trade

1. Evaluate the INTENT (BUY or SELL)

2. Higher Timeframe (HTF) – PRIMARY FACTOR

HTF structure (support/resistance) has the HIGHEST priority and OVERRIDES all other signals.

Assess:
- distance_to_htf_res
- distance_to_htf_sup
- is_near_htf_res
- is_near_htf_sup

HTF LOCATION RULES (STRICT):

A. Momentum continuation trades:

- BUY near HTF resistance:
  → penalize score
  → HIGH scores only allowed if confirmed breakout

- SELL near HTF support:
  → penalize score
  → HIGH scores only allowed if confirmed breakdown

B. Mean-reversion (swing) trades:

- SELL near HTF resistance:
  → increase score ONLY IF:
    - HTF is weakening (flat EMA slope or neutral RSI)
    - LTF momentum is weakening or showing reversal
    - No breakout confirmation

- BUY near HTF support:
  → increase score ONLY IF:
    - HTF is weakening
    - LTF shows bounce or reversal signs
    - No strong bearish momentum

C. Strong level adjustment:

- If HTF resistance/support has many touches (≥ 4):
  → reactions are more likely
  → increase score for reversal trades
  → decrease score for breakout trades unless momentum is very strong


D. Breakout condition (overrides reversal):

- If price has CLOSED beyond HTF level with strong momentum:
  → treat as breakout
  → favor continuation trades
  → ignore reversal bias

HTF TREND:

- Strong HTF trend aligned with INTENT:
  → increase score significantly

- Strong HTF trend against INTENT:
  → decrease score significantly

- Weakening HTF:
  → allow swing trades
  → moderate impact only

3. Directional Bias (LTF)

- Compare bullPct vs bearPct:
  - Large difference → increase score
  - Small difference → decrease score

- If strongly contradicting INTENT:
  → decrease score significantly

4. Momentum (EMA)

- Strong separation and correct direction:
  → increase score

- Weak / flat:
  → decrease score

5. RSI

- >55 supports BUY → increase score
- <45 supports SELL → increase score
- 45–55:
  → neutral → decrease score slightly

6. VWAP (Context)

- Above VWAP favors BUY
- Below VWAP favors SELL

- If aligned:
  → increase score

- If against:
  → decrease score

- Allow override ONLY if:
  - strong momentum
  - strong directional bias
  - large distance from VWAP

- If price is near VWAP:
  → decrease score significantly

7. Final Scoring Guidance

- 80–100:
  - Strong HTF + LTF alignment
  - NOT near HTF S/R (or confirmed breakout)
  - Strong momentum

- 60–79:
  - Good setup with minor conflicts

- 40–59:
  - Mixed signals or moderate risk

- 0–39:
  - Near strong HTF S/R without breakout
  - Weak momentum
  - Conflicting signals

Be conservative.

Never assign high scores when price is near strong HTF support or resistance unless a confirmed breakout is in progress.

DATA:
{prompt_data}
"""

    response = ollama.chat(
        model="mistral",
        messages=[
            {"role": "user", "content": prompt}
        ],
        options={
            "temperature": 0  # deterministic output
        }
    )

    return parse_score(response)