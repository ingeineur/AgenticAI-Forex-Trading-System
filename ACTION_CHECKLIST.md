# Action Checklist - Next Steps

## Immediate (Next 15 minutes)

- [ ] Read this checklist (2 min)
- [ ] Read LSTM_TEST_ANALYSIS.md sections 1-3 (5 min)
- [ ] Decide on data source:
  - [ ] Option A: Broker only (67K bars, 2022+)
  - [ ] Option B: Yahoo (650K bars, 2015+) ← RECOMMENDED
  - [ ] Option C: Combined (710K bars + spreads) ← BEST

## Step 1: Get Data (Choose ONE)

### Option A: Broker Data Only
```bash
python extract_historical_data_v2.py XAUUSD
```
- [ ] Run command
- [ ] Verify output: XAUUSD_16388.csv (should be ~67K rows, 2022-03-03 to latest)
- [ ] Time: ~2 min

### Option B: Yahoo Data (RECOMMENDED)
```bash
pip install yfinance
python download_yahoo_xauusd.py 2015-01-01 2025-12-31 15m
```
- [ ] Run pip install
- [ ] Run download
- [ ] Verify output: XAUUSD_15m_yahoo.csv (should be ~650K rows, 2015-01-01 to latest)
- [ ] Time: ~5 min

### Option C: Combined Data (BEST)
```bash
# First run both A and B above, then:
python merge_data_sources.py XAUUSD_16388.csv XAUUSD_15m_yahoo.csv
```
- [ ] Run Option A (broker download)
- [ ] Run Option B (Yahoo download)
- [ ] Run merge
- [ ] Verify output: XAUUSD_combined.csv (should be ~710K rows, 2015-2025)
- [ ] Time: ~10 min

**Choose**: B or C (A alone has too little data for LSTM)

## Step 2: Generate LSTM Dataset

```bash
python price_action_mtf_dataset.py <your_csv> --output dataset.npz --meta-output meta.csv
```

Substitute `<your_csv>` with your chosen file:
- Option A: `XAUUSD_16388.csv`
- Option B: `XAUUSD_15m_yahoo.csv`
- Option C: `XAUUSD_combined.csv`

- [ ] Run command
- [ ] Verify output files:
  - [ ] dataset.npz created
  - [ ] meta.csv created
- [ ] Check console output shows sequence count >30K
- [ ] Time: ~3 min (B takes ~10 min)

## Step 3: Check Signal Quality

```bash
python analyze_signal_quality.py meta.csv
```

- [ ] Run command
- [ ] Read output carefully
- [ ] Expected to see:
  - [ ] Overall win rate: 33-35% (WORSE than random)
  - [ ] Expected value: 0.00R (break-even)
  - [ ] [CRITICAL] Signals are worse than random!

**STOP HERE**: This confirms signals need fixing.

- [ ] Time: ~1 min

## Step 4: Fix Signal Filters

Read `LSTM_TEST_ANALYSIS.md` section "Recommended Fixes" for detailed guidance.

Edit `price_action_mtf_dataset.py`:

### Fix 1: Liquidity Sweep (Line ~109-110)
- [ ] Add body confirmation
- [ ] Add body_atr threshold

### Fix 2: Breakout (Line ~121-122)
- [ ] Tighten compression filter
- [ ] Add body_atr requirement
- [ ] Add upper_wick filter

### Fix 3: Engulfing (Line ~126-139)
- [ ] Increase body_atr threshold
- [ ] Add penetration requirement

### Fix 4: Displacement (Line ~141-142)
- [ ] (Optional) Add volume confirmation

**Or**: Run `test_signal_variants.py` to test multiple configurations automatically.

- [ ] Time: ~1-2 hours

## Step 5: Re-generate and Re-test

```bash
python price_action_mtf_dataset.py <your_csv> --output dataset_v2.npz --meta-output meta_v2.csv
python analyze_signal_quality.py meta_v2.csv
```

- [ ] Run commands
- [ ] Check win rate:
  - [ ] Is it >50%? YES → Go to Step 6
  - [ ] Is it still <45%? NO → Tighten filters more and repeat

- [ ] Time: ~15 min

## Step 6: Train LSTM (Only After Signals Improve!)

```bash
python train_lstm_price_action.py dataset_v2.npz --model-output lstm_final.pt
```

- [ ] Run command
- [ ] Watch training output:
  - [ ] Each epoch shows improving val_loss initially
  - [ ] Early stopping kicks in after ~6 epochs of plateau
- [ ] Verify output: lstm_final.pt created
- [ ] Time: ~5-10 min

## Step 7: Evaluate Final Results

Look at test metrics at end of training output:

```
Test metrics:
{'threshold': 0.50, ...}
{'threshold': 0.60, ...}
{'threshold': 0.70, ...}
{'threshold': 0.80, ...}
```

Success criteria:
- [ ] Threshold 0.60: >0 selected_trades (not 0)
- [ ] Selected winrate: 50-65% (not 33%)
- [ ] Accuracy: >55% (not 44%)

If SUCCESS:
- [ ] Model is ready for further testing
- [ ] Next: Backtest or paper trade

If FAILURE:
- [ ] Signals may still need more work
- [ ] Go back to Step 4 and tighten more
- [ ] Or consider different signal definitions entirely

- [ ] Time: ~2 min

---

## Progress Tracker

### Phase 1: Data Preparation (Est. 20 min)
- [ ] Get data (Step 1)
- [ ] Generate dataset (Step 2)
- [ ] Check signal quality (Step 3)

### Phase 2: Signal Optimization (Est. 1-2 hours)
- [ ] Fix signal filters (Step 4)
- [ ] Re-generate and test (Step 5)

### Phase 3: Model Training (Est. 15 min)
- [ ] Train LSTM (Step 6)
- [ ] Evaluate results (Step 7)

### Total Time: 2-3 hours

---

## Troubleshooting Quick Reference

**Problem**: "No data downloaded" (Step 1)
- [ ] Is MT5 terminal running? (for broker download)
- [ ] Is internet working? (for Yahoo download)

**Problem**: Signal win rate still <50% (Step 5)
- [ ] Check LSTM_TEST_ANALYSIS.md for more aggressive filter options
- [ ] Consider completely different signal definitions
- [ ] Consult price action theory for alternative setup types

**Problem**: LSTM training won't converge (Step 6)
- [ ] Check: Do you have >50K sequences?
- [ ] Check: Is signal win rate >50%?
- [ ] Try: Increasing epochs, reducing batch size

**Problem**: Still getting 33% win rate (Step 7)
- [ ] Signals still aren't fixed
- [ ] Try test_signal_variants.py for parameter search
- [ ] Or redesign signals from scratch

---

## Key Files Reference

| Step | Files Used | Files Created |
|------|-----------|---------------|
| 1 | extract_historical_data_v2.py / download_yahoo_xauusd.py | CSV data file |
| 2 | price_action_mtf_dataset.py | dataset.npz, meta.csv |
| 3 | analyze_signal_quality.py | (analysis output) |
| 4 | LSTM_TEST_ANALYSIS.md | (edits to .py file) |
| 5 | price_action_mtf_dataset.py | dataset_v2.npz, meta_v2.csv |
| 6 | train_lstm_price_action.py | lstm_final.pt |
| 7 | (analysis only) | (metrics output) |

---

## When You're Done

Once Step 7 is successful:
- [ ] You have LSTM model in lstm_final.pt
- [ ] Signal quality is verified (50%+ win rate)
- [ ] Model test accuracy is 55-70%

Next steps (beyond this review):
- [ ] Implement live inference script to use model in trading
- [ ] Paper trade to validate metrics
- [ ] Add position sizing and risk management
- [ ] Go live with small size

---

## Questions? Reference Materials

- **LSTM metrics failing?** → LSTM_TEST_ANALYSIS.md
- **Data problems?** → BROKER_DATA_LIMITATION_SOLUTIONS.md
- **Need full workflow?** → COMPLETE_WORKFLOW.md
- **Data quality?** → Run diagnose_lstm_dataset.py or analyze_signal_quality.py
- **Signal parameters?** → test_signal_variants.py

---

## Start Now

**Begin with Step 1, Option B (Yahoo data):**

```bash
pip install yfinance
python download_yahoo_xauusd.py 2015-01-01 2025-12-31 15m
```

Estimated total time: 2-3 hours to get from here to a trained model.

Good luck!
