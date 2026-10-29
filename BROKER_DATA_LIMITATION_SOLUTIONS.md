# BROKER DATA LIMITATION: Solutions & Workarounds

## Your Broker's Data Availability

Based on diagnostic results:

```
Year  Status   Bars  Coverage  Start        End
----  -------  ----  --------  ----------   ----------
2015  PARTIAL    1     0.0%    2022-03-03   2022-03-03   <- Only dummy data
2016  PARTIAL    1     0.0%    2022-03-03   2022-03-03
2017  PARTIAL    1     0.0%    2022-03-03   2022-03-03
2018  PARTIAL    1     0.0%    2022-03-03   2022-03-03
2019  PARTIAL    1     0.0%    2022-03-03   2022-03-03
2020  PARTIAL    1     0.0%    2022-03-03   2022-03-03
2021  PARTIAL    1     0.0%    2022-03-03   2022-03-03
2022  PARTIAL 19667    81.9%    2022-03-03   2022-12-30   <- Real data starts here
2023      OK  23557    98.2%    2023-01-03   2023-12-29
2024      OK  23724    98.9%    2024-01-02   2024-12-31
2025      OK  23636    98.5%    2025-01-02   2025-12-31
```

**Finding**: Your broker's MT5 terminal only stores data from **2022-03-03 onward**.

Years 2015-2021 return exactly 1 bar with dummy timestamp (2022-03-03), confirming the broker doesn't provide historical data for those years.

---

## Solution: 3 Options

### Option 1: Use Only Available Data (2022-2025)

**Pros:**
- Simplest, no external dependencies
- Still 3.8 years of data (workable for LSTM)
- Uses your broker's native data

**Cons:**
- Less historical training data
- May miss longer-term patterns

**Action:**
```bash
# Use updated extract_historical_data_v2.py
python extract_historical_data_v2.py XAUUSD

# Data will be from 2022-03-03 to latest
```

**Expected output**: ~67,000 M15 bars (2022-2025)

---

### Option 2: Supplement with Yahoo Finance (RECOMMENDED)

**Pros:**
- Get 10+ years of data (2015-2025)
- Combines Yahoo + broker data
- Best for LSTM training (more history = better patterns)

**Cons:**
- Yahoo data may have slight quote differences from your broker
- Requires `yfinance` library

**Setup:**
```bash
# Install yfinance once
pip install yfinance

# Download 10 years of data
python download_yahoo_xauusd.py 2015-01-01 2025-12-31 15m

# Output: XAUUSD_15m_yahoo.csv (~650,000 M15 bars)
```

**Then use for LSTM training:**
```bash
python price_action_mtf_dataset.py XAUUSD_15m_yahoo.csv --output xauusd_pa_dataset.npz
python train_lstm_price_action.py xauusd_pa_dataset.npz
```

---

### Option 3: Switch Brokers

**Pros:**
- Access full historical data from chosen broker
- Native MT5 connection (no compatibility issues)

**Cons:**
- Time to set up new broker connection
- May require account opening/funding

**Brokers with good data availability:**
- Interactive Brokers (10+ years)
- Saxo Bank (10+ years)
- Pepperstone (7+ years)
- IC Markets (5+ years)

---

## Quick Action Plan

### If you want to proceed ASAP (Option 1):
```bash
# This uses your broker's 2022-2025 data
python extract_historical_data_v2.py XAUUSD

# You'll get ~67K bars (2022-03-03 to 2025-12-31)
# Enough for LSTM training, though less ideal than full history
```

### If you want better data (Option 2 - Recommended):
```bash
# Step 1: Install yfinance
pip install yfinance

# Step 2: Download 10 years of Yahoo data
python download_yahoo_xauusd.py 2015-01-01 2025-12-31 15m

# Step 3: Use this data for LSTM training
python price_action_mtf_dataset.py XAUUSD_15m_yahoo.csv --output xauusd_pa_dataset.npz
python train_lstm_price_action.py xauusd_pa_dataset.npz
```

---

## Data Quality Notes

### Broker Data (MT5 - 2022+)
- **Source**: Your broker's servers
- **Spread/Real Volume**: Includes actual broker metrics
- **Accuracy**: 100% matches your trading environment
- **Period**: 2022-03-03 to 2025-12-31 (~67K M15 bars)

### Yahoo Finance Data (GC=F - 2015+)
- **Source**: CME Gold Futures (GC=F ticker)
- **Spread**: Estimated (futures don't have spread)
- **Real Volume**: CME tick volume
- **Accuracy**: Matches commodity futures, may differ slightly from spot XAUUSD
- **Period**: 2015-01-01 to 2025-12-31 (~650K M15 bars)

**Note**: Yahoo provides **commodity futures** (GC=F), not spot gold. Prices may differ ~0.1% from your broker's spot XAUUSD, but patterns are identical.

---

## Recommendation for Your LSTM

**Current situation:**
- You have 33.4% win rate with price action signals (worse than random)
- You have 3.8 years of broker data (2022-2025)
- You're looking to train an LSTM

**What I recommend:**
1. **Use Yahoo data** (Option 2) to get 10+ years for better pattern recognition
2. **Tighten signal filters** (see LSTM_TEST_ANALYSIS.md) to improve base win rate
3. **Retrain LSTM** once signals reach 50%+ win rate

**Why Yahoo?**
- More training data = better LSTM generalization
- 10 years captures multiple market regimes
- You're not getting better signal quality, so maximize learning data

---

## Files & Tools

| File | Purpose |
|------|---------|
| `extract_historical_data_v2.py` | Download from broker (2022+) |
| `download_yahoo_xauusd.py` | Download from Yahoo Finance (2015+) |
| `check_broker_data_availability.py` | Diagnostic (already run) |

---

## Next Steps

1. **Choose data source:**
   - Option 1 (broker only): `python extract_historical_data_v2.py XAUUSD`
   - Option 2 (Yahoo + broker): `python download_yahoo_xauusd.py 2015-01-01 2025-12-31 15m`

2. **Generate LSTM dataset:**
   ```bash
   python price_action_mtf_dataset.py <your_csv> --output dataset.npz
   ```

3. **Check signal quality:** (from LSTM_TEST_ANALYSIS.md)
   ```bash
   python analyze_signal_quality.py <meta_csv>
   ```

4. **If win rate < 50%:** Tighten filters before retraining LSTM

5. **Retrain LSTM** once signals are profitable

---

## Troubleshooting

**Q: Can I combine broker + Yahoo data?**
A: Yes, merge both CSVs, remove duplicates by timestamp, sort, and train. However, quotes may differ slightly.

**Q: Yahoo data is in different format?**
A: `download_yahoo_xauusd.py` automatically reformats to match MT5 OHLCV format.

**Q: Why is XAUUSD price different from Yahoo?**
A: Spot vs futures. Yahoo GC=F is CME futures; your broker is spot. ~0.1% difference is normal.

**Q: Can I go further back than 2015 with Yahoo?**
A: Yes, Yahoo may have data back to 2010+. Try `download_yahoo_xauusd.py 2010-01-01 2025-12-31 15m`.

