# Signal Filter Test Results - Critical Findings

## Executive Summary

**Problem**: Current signals achieve only 33.4% win rate (worse than random)

**Testing**: Tested 5 signal filter variations to see if tightening filters improves win rate

**Result**: **Tightening filters does NOT improve win rate** (stays at 34-35%)

**Conclusion**: The problem is NOT filter tightness. The signals themselves are fundamentally flawed.

---

## Test Results

### Configurations Tested

| Configuration | Sweep Body | Breakout Comp | Engulf Body | Displacement | Candidates | Win Rate | Expected Value | Status |
|---------------|-----------|---------------|-------------|--------------|------------|----------|----------------|--------|
| Current (Baseline) | 0.0 | 0.90 | 0.6 | 1.25 | 16,563 | 34.4% | +0.03R | BAD |
| Moderate | 0.3 | 0.85 | 0.8 | 1.35 | 12,061 | 34.5% | +0.04R | BAD |
| Strict v1 | 0.5 | 0.80 | 0.9 | 1.40 | 10,111 | 34.5% | +0.03R | BAD |
| Strict v2 | 0.6 | 0.75 | 1.0 | 1.50 | 8,201 | 34.8% | +0.04R | BAD |
| Very Strict | 0.7 | 0.70 | 1.1 | 1.60 | 6,595 | 34.4% | +0.03R | BAD |

### Key Observations

1. **Win rate is stable across all filter tightness levels**: 34.4% - 34.8%
   - Only 0.4% difference between loosest and tightest
   - This suggests the problem is NOT with filter parameters

2. **Reducing signal count doesn't improve quality**:
   - Baseline: 16,563 setups, 34.4% WR
   - Very Strict: 6,595 setups, 34.4% WR
   - Filtering out 60% of signals still gives same 34% win rate

3. **Even best variant only achieves 34.8%**:
   - Still below 50% threshold
   - Expected value remains near zero (+0.04R is negligible)

### What This Means

**The signals are broken at a fundamental level**, not just at the parameter level.

Possible root causes:
1. **Wrong setup types**: Liquidity sweeps, breakouts, and engulfing simply don't work well for XAUUSD M15
2. **Wrong timeframe**: M15 may be too noisy; HTF confluence insufficient
3. **No market regime filtering**: Signals work in trends but fail in choppy markets
4. **Missing confluence factors**: Signals need volume, volatility, or volatility clustering filters
5. **Risk/reward mismatch**: 2R TP / 1R SL may not match actual market outcomes

---

## Next Steps (Choose One)

### Option A: Abandon Current Signal Approach (Recommended)

The current price action signals are fundamentally unprofitable. Tightening parameters won't fix this.

**Better approach:**
- Use ML directly on OHLCV data (regression model for entry probability)
- Try order flow or volume profile signals
- Use counter-trend reversal logic instead of breakout/sweep
- Add machine learning feature engineering

---

### Option B: Add Confluence Filters

Modify signals to require multi-factor confirmation:

```python
# Example: Only trade sweeps in strong trends
df["ema_20"] = df["close"].ewm(span=20, adjust=False).mean()
df["ema_50"] = df["close"].ewm(span=50, adjust=False).mean()
df["trend"] = df["ema_20"] > df["ema_50"]

# Only take sweep_long if EMA is bullish
df["sweep_long_filtered"] = df["sweep_long"] & df["trend"]
```

This might improve from 34% to 40-45%, but unlikely to reach 50%.

---

### Option C: Test Counter-Trend Signals

The current approach (trend-following) gets 34%. Counter-trend might work better:

```python
# Reversal instead of breakout
df["reversal_down"] = (df["high"] > df["high"].shift(5)) & (df["close"] < df["open"])
df["reversal_up"] = (df["low"] < df["low"].shift(5)) & (df["close"] > df["open"])
```

Test if counter-trend setups have better win rates on XAUUSD M15.

---

## Recommendation

**Do NOT retrain LSTM with current signals.**

Even if we achieve 50% win rate, current tests show the signals are fundamentally weak. LSTM will struggle to find patterns in weak signals.

### Better Path Forward

1. **Redesign signals completely** using one of:
   - Machine learning (predict direction from OHLCV)
   - Volume/volatility confluence
   - Counter-trend reversal logic
   - Order flow techniques

2. **Test new signals** for 50%+ win rate BEFORE training LSTM

3. **Retrain LSTM** with improved signals

4. **Expected LSTM performance** with good signals (50-60% WR):
   - Test accuracy: 55-70%
   - Threshold 0.60: >1000 trades
   - Expected value: +0.5R+

---

## Technical Details

### Why Tight Filters Don't Help

Tightening filters removes low-confidence setups, but it doesn't improve the win rate of remaining setups.

**Strict v2 example**:
- Removed 50% of signals (16.5K → 8.2K)
- Win rate: 34.4% → 34.8% (improvement only 0.4%)
- This tiny improvement is likely statistical noise

**Implication**: The problem is with signal QUALITY, not signal QUANTITY.

### Statistical Significance

With 8,000+ samples per configuration, the 34.4-34.8% range is consistently around the null hypothesis (random entry = 50/50).

If signals were fundamentally sound but over-triggered, tight filters would show:
- Baseline: 34.4% (200 good + 16,363 bad = 1.2% ratio)
- Very Strict: 50%+ (only good ones remain)

Instead we see 34.4% at both extremes, confirming the signals themselves are the problem.

---

## Conclusion

| Finding | Impact | Action |
|---------|--------|--------|
| Tightening filters doesn't improve win rate | Critical | Abandon parameter tuning |
| Win rate stays at 34-35% regardless of threshold | Critical | Signals are fundamentally weak |
| Even best variant is 34.8% | Critical | Don't retrain LSTM yet |
| Expected value near zero (+0.04R) | Critical | Not profitable |
| Need new signal approach | Critical | Redesign required |

**Bottom Line**: Your current price action signals don't work. Parameter adjustments won't fix this. You need to redesign the signals completely using a different approach (ML, volume, counter-trend, etc.).

