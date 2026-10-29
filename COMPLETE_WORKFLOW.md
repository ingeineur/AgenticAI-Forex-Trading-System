# Complete Workflow: From Data to LSTM Training

## Your Situation

**Problem**: Broker only has data from 2022-03-03 onward  
**Goal**: Train a better LSTM with more historical data  
**Solution**: Combine Yahoo Finance data (10 years) with broker data (spreads)

---

## Step-by-Step Workflow

### Step 1: Download Yahoo Data (10 years, 650K bars)

```bash
pip install yfinance
python download_yahoo_xauusd.py 2015-01-01 2025-12-31 15m
```

**Output**: `XAUUSD_15m_yahoo.csv`

**Time**: ~2-3 minutes

---

### Step 2: Download Broker Data (2022+, 67K bars)

```bash
python extract_historical_data_v2.py XAUUSD
```

**Output**: `XAUUSD_16388.csv` (or similar, depending on timeframe code)

**Time**: ~1-2 minutes (MT5 connection required)

**Note**: Updated to use correct date range (2022-03-03 instead of 2018-01-01)

---

### Step 3: (Optional) Merge Both Data Sources

This combines Yahoo pre-2022 data with broker 2022+ data for best of both worlds:

```bash
python merge_data_sources.py XAUUSD_16388.csv XAUUSD_15m_yahoo.csv XAUUSD_combined.csv
```

**Output**: `XAUUSD_combined.csv` (710K bars, 2015-2025)

**Benefits**:
- Yahoo historical patterns (2015-2022)
- Broker spreads/real_volume (2022+)
- Seamless transition at 2022-03-03

---

### Step 4: Generate Price Action Dataset

**Option A: Use only Yahoo (simplest)**
```bash
python price_action_mtf_dataset.py XAUUSD_15m_yahoo.csv \
  --output xauusd_pa_dataset.npz \
  --meta-output xauusd_pa_meta.csv
```

**Option B: Use combined data (best)**
```bash
python price_action_mtf_dataset.py XAUUSD_combined.csv \
  --output xauusd_pa_dataset.npz \
  --meta-output xauusd_pa_meta.csv
```

**Option C: Use broker only (quickest)**
```bash
python price_action_mtf_dataset.py XAUUSD_16388.csv \
  --output xauusd_pa_dataset.npz \
  --meta-output xauusd_pa_meta.csv
```

**Output**: 
- `xauusd_pa_dataset.npz` (sequences ready for LSTM)
- `xauusd_pa_meta.csv` (metadata for analysis)

**Time**: 1-5 minutes (depends on CSV size)

---

### Step 5: Analyze Signal Quality (Critical!)

```bash
python analyze_signal_quality.py xauusd_pa_meta.csv
```

**Expected output** (from your current data):
```
Overall Win Rate: 33.4%
Expected Value: 0.00R (break-even)
[CRITICAL] Signals are worse than random!
```

**If you see this**: STOP. Fix signals before training LSTM.

---

### Step 6: Fix Signal Filters (If Needed)

See `LSTM_TEST_ANALYSIS.md` for recommendations. 

Example: Tighten breakout filter in `price_action_mtf_dataset.py`:

```python
# BEFORE (too loose)
df["breakout_long"] = (df["close"] > range_high) & (compression < 0.9)

# AFTER (tighter)
df["breakout_long"] = (
    (df["close"] > range_high) & 
    (compression < 0.8) &
    (df["body_atr"] > 0.7)
)
```

Then **re-run Step 4** to regenerate dataset.

---

### Step 7: Train LSTM

Once signals have >50% win rate:

```bash
python train_lstm_price_action.py xauusd_pa_dataset.npz \
  --epochs 50 \
  --batch-size 128 \
  --threshold 0.60
```

**Output**: `lstm_price_action.pt` (trained model)

**Time**: 3-10 minutes (depends on hardware)

---

### Step 8: Evaluate Test Metrics

Script output will show:
```
Test metrics:
{'threshold': 0.50, 'accuracy': 44.7%, 'precision': 0.33, 'recall': 0.66, ...}
{'threshold': 0.60, 'accuracy': 66.7%, 'precision': 0.00, 'recall': 0.00, ...}
...
```

**If threshold 0.6-0.8 shows zero trades**: Signals still need work, or model isn't learning.

**Next**: Fix signals and retrain.

---

## Complete Command Chain

**For impatient users** (combine into single workflow):

```bash
# 1. Get data
pip install yfinance
python download_yahoo_xauusd.py 2015-01-01 2025-12-31 15m
python extract_historical_data_v2.py XAUUSD
python merge_data_sources.py XAUUSD_16388.csv XAUUSD_15m_yahoo.csv XAUUSD_combined.csv

# 2. Generate dataset
python price_action_mtf_dataset.py XAUUSD_combined.csv \
  --output dataset.npz --meta-output meta.csv

# 3. Check signal quality
python analyze_signal_quality.py meta.csv

# 4. If signals OK, train LSTM
python train_lstm_price_action.py dataset.npz --model-output model.pt
```

**Total time**: ~15-20 minutes

---

## Troubleshooting

### "No data downloaded" error
**Cause**: MT5 not running or symbol not available  
**Fix**: Start MT5 terminal, ensure XAUUSD is visible

### Yahoo download fails
**Cause**: Internet connection or yfinance bug  
**Fix**: Check internet, update yfinance with `pip install --upgrade yfinance`

### Signal quality still <50%
**Cause**: Price action filters are too permissive  
**Fix**: Tighten thresholds in `price_action_mtf_dataset.py`, regenerate dataset

### LSTM test metrics show 0 predictions at threshold 0.6+
**Cause**: Model learned that signals are unreliable  
**Fix**: Fix signals first, then retrain

### Model overfitting (high train loss, low val loss)
**Cause**: Not enough data or too complex model  
**Fix**: Use more data (Yahoo) or reduce hidden_size

---

## Files Reference

| File | Purpose |
|------|---------|
| `download_yahoo_xauusd.py` | Download Yahoo Finance data (10 years) |
| `extract_historical_data_v2.py` | Download broker data (updated dates) |
| `merge_data_sources.py` | Combine Yahoo + broker data |
| `price_action_mtf_dataset.py` | Generate LSTM sequences |
| `train_lstm_price_action.py` | Train LSTM model |
| `analyze_signal_quality.py` | Check signal profitability |
| `diagnose_lstm_dataset.py` | Check data quality |

---

## Key Decision: Which Data Source?

| Data Source | Bars | Years | Best For | Command |
|------------|------|-------|----------|---------|
| **Broker only** | 67K | 3.8 | Quick start | `python extract_historical_data_v2.py XAUUSD` |
| **Yahoo only** | 650K | 10 | Best patterns | `python download_yahoo_xauusd.py 2015-01-01 2025-12-31 15m` |
| **Combined** | 710K | 10 | Best of both | Use both scripts + merge |

**Recommendation**: Use **Yahoo** or **Combined** for better LSTM training. More data = better generalization.

---

## Success Criteria

✓ Dataset has 50,000+ sequences  
✓ Signal win rate is 50-60%  
✓ Test accuracy > 55% at threshold 0.60+  
✓ Selected trades show positive expected value  

If you hit all these, your LSTM is ready for live testing.

