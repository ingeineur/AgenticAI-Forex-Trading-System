# Price Action Neural Strategy Revision

This revision keeps the strategy practical:

1. Price action creates the trade candidate.
2. Multi-timeframe context decides whether the candidate is worth learning/trading.
3. A sequence model scores the setup probability.
4. Execution only acts when probability, direction, and risk are aligned.

## Signal Layer

The base timeframe, usually M15, generates candidate setups from effective price action:

- Liquidity sweep: price takes a recent swing high/low and closes back inside.
- Breakout: price closes beyond a compressed range.
- Engulfing/displacement: a strong candle or candle group expands through prior structure.
- Wick rejection: wick/body imbalance near a swing level.
- Trend pullback: price returns toward mean/EMA while higher timeframe trend remains intact.

These are not final buy/sell decisions. They are candidate moments where a model is allowed to score the trade.

## Multi-Timeframe Features

The dataset builder derives H1, H4, and D1 features from the input candles and aligns them back to the base timeframe without future leakage.

Useful context includes:

- HTF EMA slope and close-vs-EMA location.
- HTF ATR-normalized distance from recent swing high/low.
- HTF range compression/expansion.
- Agreement score between base timeframe direction and HTF direction.
- Correlation of recent base returns with HTF returns.

## Model Choice

Recommended first model: LSTM.

Reason: your data is already chronological OHLCV, and the model should learn how the setup formed over the last N candles, not just a single row. LSTM is easier to train, evaluate, and deploy than HTM in this workspace.

HTM can be explored later for anomaly/regime detection, especially for "market changed" alerts. It is less straightforward as a supervised TP/SL classifier.

## Training Target

Each candidate candle is labeled:

- `1` when TP is hit before SL within the forward horizon.
- `0` when SL is hit before TP.
- skipped when neither level is reached.

Default reward/risk is 2R TP and 1R SL. Keep this consistent between dataset generation, backtest, and live execution.

## Execution Filter

A live strategy should require all of these before entry:

- A fresh price-action candidate exists.
- LSTM probability is above threshold, for example `0.60` to start.
- Higher-timeframe direction is not strongly opposed.
- Price is not entering nearby HTF support/resistance unless the setup is explicitly a reversal.
- Spread and session filters pass.
- Position sizing is based on SL distance, not fixed hope.

## Files Added

- `price_action_mtf_dataset.py`: creates a sequence dataset with price-action signals and HTF context.
- `train_lstm_price_action.py`: trains an LSTM classifier on the generated sequence dataset.

Suggested flow:

```powershell
py price_action_mtf_dataset.py XAUUSD_M15.csv --output xauusd_price_action_mtf.npz --meta-output xauusd_price_action_mtf_meta.csv
py train_lstm_price_action.py xauusd_price_action_mtf.npz --model-output xauusd_lstm_price_action.pt
```

