# CLAUDE.md — Project Memory

## What This Project Does

BTC Technical Analysis Agent — an automated Bitcoin trading signal system that:
- Analyzes BTC across 3 timeframes (1h, 4h, 1d) using 40 technical analysis features
- Generates BUY/SELL/HOLD signals via per-timeframe XGBoost classifiers
- Sends rich Telegram notifications with entry/exit prices, stop-loss, and confidence
- Implements continuous learning: evaluates past signals after 24h and retrains automatically
- Runs hourly via GitHub Actions (free on public repos)

## Repository Structure

```
src/
  main.py           # Entry point & 8-step pipeline orchestrator
  data_fetcher.py   # OHLCV via Binance → Binance US → Bybit fallback + EUR/USD rate
  indicators.py     # 40 TA features: RSI, MACD, Bollinger, EMA, Stochastic RSI, ATR, OBV, 16 candlestick patterns
  model.py          # XGBoost classifier per timeframe (train/predict/save/load/retrain)
  signals.py        # Weighted multi-TF aggregation + ATR-based entry/exit/stop-loss
  polymarket.py     # Polymarket sentiment: Gamma API → CLOB → Goldsky fallback
  notifier.py       # Telegram HTML message formatting, cooldown enforcement
  learning.py       # Signal history (JSON), outcome evaluation, retrain trigger
data/
  model_1h.pkl      # Serialized models (auto-generated)
  model_4h.pkl
  model_1d.pkl
  signal_history.json
.github/workflows/
  analyze.yml       # Hourly cron + manual dispatch, auto-commits data/ changes
requirements.txt    # Python deps
```

## Tech Stack

Python 3.11 | pandas | numpy | ta (technical analysis) | xgboost | scikit-learn | joblib | requests

External APIs: Binance / Binance US / Bybit (OHLCV, no key needed), Frankfurter (EUR/USD fallback), Polymarket Gamma API / CLOB / Goldsky Subgraph (BTC sentiment, no key needed), Telegram Bot API

## How to Run

```bash
pip install -r requirements.txt

python src/main.py --dry-run          # Print message, don't send
python src/main.py --dry-run --force  # Skip cooldown + confidence filters

# Production (requires env vars):
TELEGRAM_BOT_TOKEN=xxx TELEGRAM_CHAT_ID=yyy python src/main.py
```

## Pipeline (main.py)

1. Fetch EUR/USD rate (Binance or Frankfurter fallback)
2. Load saved models or train from scratch (2 years of history)
3. Fetch 300 recent candles per timeframe (Binance → Binance US → Bybit fallback)
3b. Fetch Polymarket BTC sentiment (Gamma API → CLOB → Goldsky fallback)
4. Compute 40 TA features via `indicators.py`
5. Per-timeframe XGBoost predictions → (signal, confidence)
6. Weighted aggregation: 1d=40%, 4h=35%, 1h=25%, polymarket=15% (`signals.py`)
7. Evaluate past signal outcomes (delay aligned with prediction horizon) → retrain if 10+ new BUY/SELL evals
8. Send Telegram notification (if actionable: confidence ≥ 55%, net profit after fees & tax ≥ 1%) → persist signal history

## Key Constants

| Constant | Value | Location |
|----------|-------|----------|
| `TRAINING_START_DATE` | "2020-01-01" (expanding window) | `main.py` |
| `RECENT_CANDLES` | 300 | `main.py` |
| `MIN_SIGNAL_CONFIDENCE` | 0.55 | `main.py` |
| `TIMEFRAME_WEIGHTS` | 1h=0.15, 4h=0.25, 1d=0.30, polymarket=0.30 | `signals.py` |
| `MIN_CONFIDENCE` (per TF) | 0.45 | `signals.py` |
| `SIGNAL_THRESHOLD` | 0.40 | `signals.py` |
| `ATR_EXIT_MULTIPLIER` | 2.0 | `signals.py` |
| `ATR_STOP_MULTIPLIER` | 1.5 | `signals.py` |
| `ENTRY_FEE_RATE` | 0.0025 (0.25%) | `signals.py` |
| `EXIT_FEE_RATE` | 0.0040 (0.40%) | `signals.py` |
| `TAX_RATE` | 0.30 (30%) | `signals.py` |
| `MIN_NET_PROFIT_PCT` | 1.0 (1%) | `signals.py` |
| `EVALUATION_DELAY_HOURS` | 24 (default fallback; per-signal delay based on dominant TF: 1h→12h, 4h→24h, 1d→120h) | `learning.py` |
| `RETRAIN_THRESHOLD` | 10 (BUY/SELL only, HOLD excluded) | `learning.py` |
| `MIN_ACCURACY_FOR_SKIP` | 0.90 (90%) | `learning.py` |
| `OUTCOME_THRESHOLD` | 0.01 (1%) | `learning.py` |
| `SENTIMENT_BULL_THRESHOLD` | 0.55 | `polymarket.py` |
| `SENTIMENT_BEAR_THRESHOLD` | 0.45 | `polymarket.py` |
| `SHORT_TERM_WEIGHT` | 0.60 | `polymarket.py` |
| `LONG_TERM_WEIGHT` | 0.40 | `polymarket.py` |
| `MIN_VOLUME_USD` | 1,000 | `polymarket.py` |

## Model Details

- XGBoost classifier: 300 estimators, max_depth=5, lr=0.05, colsample/subsample=0.8
- Labels: future price movement over lookahead window (6h/24h/5d) with ±2% threshold
- Class imbalance handled via weighted training
- Train/test split: `TimeSeriesSplit` (no future leakage)
- Serialization: joblib

## CI/CD

GitHub Actions workflow (`.github/workflows/analyze.yml`):
- Schedule: `0 * * * *` (every hour)
- Manual trigger via `workflow_dispatch` with `dry_run` and `force` inputs
- Auto-commits updated models and signal history back to repo
- Concurrency group prevents simultaneous runs

## Development Notes

- Binance API may be geo-restricted in some regions; automatic fallback to Binance US and Bybit
- Polymarket API may be geo-restricted; 3-source fallback: Gamma API → CLOB → Goldsky Subgraph (no geo-restrictions)
- First run trains from scratch (~3–5 min to fetch 2 years of data)
- Signal encoding: 1=BUY, 0=HOLD, -1=SELL throughout codebase
- Prices shown in both USD and EUR
- Cooldown: no same-direction signal within 4 hours
- No tests or linter configured yet
