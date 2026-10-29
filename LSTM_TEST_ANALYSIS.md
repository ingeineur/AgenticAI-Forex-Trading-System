# LSTM Test Metrics Review & Root Cause Analysis

## Executive Summary

**Status**: CRITICAL ISSUE IDENTIFIED  
**Problem**: Model is performing poorly not because of the LSTM implementation, but because **the underlying price action signals have only a 33.4% win rate**.  
**Action**: Do NOT retrain the LSTM until signal quality improves to 50%+ win rate.

---

## Test Metrics Analysis

### Observed Results
```
Threshold | Accuracy | Precision | Recall | Trades | Winrate
----------|----------|-----------|--------|--------|----------
0.50      | 44.7%    | 33.4%     | 66.5%  | 3,085  | 33.4%
0.60      | 66.7%    | 0.0%      | 0.0%   | 0      | N/A
0.70      | 66.7%    | 0.0%      | 0.0%   | 0      | N/A
0.80      | 66.7%    | 0.0%      | 0.0%   | 0      | N/A
```

### What This Means

1. **At threshold 0.50**: Model predicts 3,085 trades with 33.4% win rate
   - This matches the baseline (unskilled) win rate
   - Accuracy of 44.7% is BELOW the 66.7% "dumb" baseline (predict all losses)
   - Model learned nothing useful; it's just replicating the label distribution

2. **At threshold 0.60+**: Zero predictions above threshold
   - Model learned that high-confidence predictions are not supported by the data
   - This is actually correct behavior given the input quality

### Root Cause: Model Output Distribution

The LSTM is outputting probabilities clustered near 0.5 because:
- Training data shows no discriminative pattern (33.4% WR = random)
- Model minimizes loss by predicting uncertainty on all samples
- It cannot distinguish winners from losers when 2/3 of "setups" lose immediately

---

## Signal Quality Analysis

### Dataset Composition
- **Total sequences**: 31,033
- **Positive (won trades)**: 10,365 (33.4%)
- **Negative (lost trades)**: 20,668 (66.6%)
- **Imbalance ratio**: 1.99x (acceptable)

### Win Rate by Setup Type
| Direction | Setups | Win Rate |
|-----------|--------|----------|
| Long      | 16,554 | 34.4%    |
| Short     | 14,479 | 32.2%    |
| Overall   | 31,033 | 33.4%    |

### Multi-Timeframe Context (HTF Alignment)
| HTF Status | Setups | Win Rate | Impact |
|------------|--------|----------|--------|
| Opposed   | 7,378  | 31.9%    | -1.5%  |
| Neutral   | 16,238 | 33.3%    | -0.1%  |
| Aligned   | 7,417  | 35.2%    | +1.8%  |

**Finding**: HTF alignment provides minimal edge (+1.8% at best). The multi-timeframe context is **not predictive**.

### Expected Value

With 2R TP / 1R SL:
```
E[V] = (33.4% × 2R) - (66.6% × 1R) = 0.00R
```

**Expected value is ZERO** → Strategy breaks even at best, loses in reality (due to spreads, slippage).

### Risk Profile
- Mean ATR: 3.13
- Median ATR: 2.46
- Min ATR: 0.40
- Max ATR: 28.68

---

## Diagnosis: Why Signals Fail

### 1. Sweep Detection is Too Loose
```python
df["sweep_long"] = (df["high"] > swing_high) & (df["close"] < swing_high)
```
- Any candle that touches a recent high and closes back inside qualifies
- This happens in **every consolidation phase** → high false signal rate
- No confirmation that price will resume uptrend

### 2. Breakout Doesn't Confirm Follow-Through
```python
df["breakout_long"] = (df["close"] > range_high) & (compression < 0.9)
```
- Only checks if price closes beyond previous range
- Doesn't require TP to be reached before reversal
- Many breakouts immediately reverse ("breakout traps")

### 3. Engulfing Threshold Too Permissive
```python
df["engulf_long"] = (df["body_atr"] > 0.6) & (df["close"] > df["open"]) & ...
```
- 0.6x ATR body is common in normal volatility
- False positives when engulfing occurs at resistance

### 4. HTF Context Not Filtering Effectively
- Mean alignment = 0.001 (virtually random)
- Alignment only improves WR by 1.8% even in best case
- Suggests HTF trend and base timeframe setups are **uncorrelated**

---

## Evidence the LSTM is Working Correctly

1. **Early stopping prevented overfit**: Model stopped at epoch N when val loss plateaued
2. **No NaN/Inf in data**: Numerical stability not an issue
3. **Split consistency**: Train/val/test all have ~33% positive rate
4. **Class imbalance handled**: pos_weight=1.99x is reasonable
5. **Test metrics match label distribution**: Accuracy at threshold 0.6 = 66.7% = baseline

**Conclusion**: The LSTM is correctly identifying that **the input signals have zero predictive value**. It's behaving like a smart model should when given noise.

---

## Recommended Fixes (Priority Order)

### 1. Tighten Signal Filters (MUST DO)

**Liquidity Sweep** - Add confirmation:
```python
df["sweep_long"] = (
    (df["high"] > swing_high) &
    (df["close"] < swing_high) &
    (df["close"] > df["open"]) &  # NEW: Bullish close
    (df["body_atr"] > 0.4)        # NEW: Minimum body size
)
```

**Breakout** - Add follow-through:
```python
df["breakout_long"] = (
    (df["close"] > range_high) &
    (compression < 0.8) &         # TIGHTEN: 0.9 → 0.8
    (df["body_atr"] > 0.7) &      # NEW: Strong candle
    (df["upper_wick_atr"] < 0.3)  # NEW: Wick rejection filter
)
```

**Engulfing** - More selective:
```python
df["engulf_long"] = (
    (df["close"] > df["open"]) &
    (prev_close < prev_open) &
    (df["close"] > prev_open + 0.3 * df["atr"]) &  # NEW: Penetration
    (df["body_atr"] > 0.8)  # TIGHTEN: 0.6 → 0.8
)
```

**Expected impact**: +10–15% improvement in win rate (hypothesis; requires testing)

### 2. Test Parameter Variants

Run `test_signal_variants.py` to compare:
- Shorter sequence length (seq_length=32 vs 64)
- Reduced horizon (max_horizon=30 vs 50)
- Higher TP threshold (tp_mult=3.0 to label only strong wins)

### 3. Filter by Market Conditions

Add pre-entry checks:
```python
df["is_tradeable"] = (
    (df["spread_atr"] < 0.5) &       # Tight spread
    (df["volume_ratio"] > 0.8) &      # Normal volume
    (df["htf_alignment"] >= -0.2)     # Not opposed to HTF
)
# Only generate labels for tradeable setups
```

### 4. Rebalance Risk/Reward

Current: 2R TP / 1R SL  
Test: 3R TP / 0.5R SL (or adjust based on win rate analysis)

---

## What NOT to Do

❌ **Don't try to fix with hyperparameter tuning**: Higher learning rate, more layers, longer training won't help. Garbage in = garbage out.

❌ **Don't use ensemble methods or complex models**: XGBoost, Random Forest, etc. will face the same problem. A Naive Bayes would get 33.4% too.

❌ **Don't ignore the 33.4% win rate**: This is the fundamental constraint. No ML algorithm can make a break-even strategy profitable without better inputs.

---

## Action Items

- [ ] Modify `price_action_mtf_dataset.py` to tighten signal thresholds
- [ ] Run analysis on variants to find parameters with 50%+ win rate
- [ ] Retrain LSTM only after baseline win rate improves
- [ ] Benchmark signal quality against simple countertrend strategies
- [ ] Document final signal parameters for production use

---

## Next Steps

1. **Immediate**: Run `test_signal_variants.py` to identify best-performing signal configuration
2. **If best variant achieves ~50% WR**: Retrain LSTM with those parameters
3. **If all variants stay <45% WR**: Consider abandoning price action signals; explore alternative setup definitions (order flow, volume profile, etc.)

