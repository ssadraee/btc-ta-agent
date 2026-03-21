# BTC Technical Analysis Agent

An automated Bitcoin trading signal agent that applies professional-grade technical analysis across multiple timeframes and sends Telegram notifications with actionable Buy/Sell recommendations.

## Features

- **Multi-timeframe analysis**: 1-hour, 4-hour, and daily charts
- **40 TA features**: RSI, MACD, Bollinger Bands, EMA crossovers, Stochastic RSI, ATR, OBV, and 16 candlestick patterns
- **Machine learning**: XGBoost classifier trained on 2 years of historical data
- **EUR support**: All prices shown in both USD and EUR
- **Continuous learning**: Evaluates past signal outcomes and retrains automatically
- **Telegram notifications**: Rich, plain-English notifications with entry/exit prices and confidence
- **GitHub Actions**: Runs automatically every hour, fully free on public repos

## Notification Format

```
🟢 BTC Trading Signal — BUY
2024-03-15 14:00 UTC

📊 Recommendation: BUY

💵 Entry Price: $84,200.00  (€77,850.00)
🎯 Exit Price:  $87,400.00  (€80,800.00)
🛑 Stop Loss:   $82,500.00  (€76,250.00)
📈 Profit Target (gross): +3.80%
   ↳ Entry fee (0.25%): -0.25%
   ↳ Exit fee (0.40%):  -0.41%
   ↳ Tax (30% on profit): -0.94%
💰 Net Profit (est.): +2.20%

🧠 Explanation:
RSI(1-hour) is 29 — deeply oversold. MACD histogram (4-hour) is positive.
EMA50 is above EMA200 on the daily chart — golden cross in effect.

⏱ Timeframes: 1h: BUY (71%) | 4h: BUY (78%) | 1d: HOLD (52%)
🎯 Confidence: 73% ▓▓▓▓▓▓▓░░░

⚠️ This is not financial advice. Always manage your risk.
```

## Setup

### 1. Create a Telegram Bot

1. Open Telegram and search for `@BotFather`
2. Send `/newbot` and follow the instructions
3. Copy the **bot token** you receive
4. Start a conversation with your bot, then fetch your **chat ID**:
   ```
   https://api.telegram.org/bot<YOUR_TOKEN>/getUpdates
   ```

### 2. Add GitHub Secrets

In your repository: **Settings → Secrets and Variables → Actions → New repository secret**

| Secret | Value |
|--------|-------|
| `TELEGRAM_BOT_TOKEN` | Your Telegram bot token |
| `TELEGRAM_CHAT_ID` | Your Telegram chat/user ID |

### 3. Enable GitHub Actions

The workflow is already configured in `.github/workflows/analyze.yml`.
It runs automatically every hour. You can also trigger it manually from the **Actions** tab.

## Local Development

```bash
pip install -r requirements.txt

# Dry run (prints Telegram message without sending)
python src/main.py --dry-run

# Force a notification regardless of cooldown (for testing)
python src/main.py --dry-run --force

# Live run (requires TELEGRAM_BOT_TOKEN and TELEGRAM_CHAT_ID env vars)
TELEGRAM_BOT_TOKEN=xxx TELEGRAM_CHAT_ID=yyy python src/main.py
```

## Architecture

```
src/
├── main.py           Orchestrator — runs the full pipeline
├── data_fetcher.py   OHLCV data (Binance → Binance US → Bybit fallback) + EUR/USD rate
├── indicators.py     TA feature engineering (40 features)
├── model.py          XGBoost classifier (train/predict/save/load)
├── signals.py        Multi-timeframe aggregation + entry/exit calculation
├── notifier.py       Telegram message formatting + sending
└── learning.py       Continuous learning — outcome tracking + retrain logic

data/
├── model_1h.pkl      Trained model for 1h timeframe (auto-generated)
├── model_4h.pkl      Trained model for 4h timeframe (auto-generated)
├── model_1d.pkl      Trained model for 1d timeframe (auto-generated)
└── signal_history.json  Past signals with outcomes (auto-generated)
```

## How It Works

### Signal Generation
Each timeframe produces a prediction (BUY=1, HOLD=0, SELL=-1) with a confidence score.
Signals are combined using weighted aggregation: `1d (40%) > 4h (35%) > 1h (25%)`.
A notification is only sent when:
- The aggregate score passes the directional threshold
- Confidence ≥ 55%
- No same-direction signal was sent in the last 4 hours
- **Net profit after fees and tax ≥ 1%** (see below)

### Entry / Exit Prices
- **Entry price**: Current market price
- **Exit price**: Entry ± (ATR × 2.0) — dynamically sized based on current volatility
- **Stop loss**: Entry ∓ (ATR × 1.5) — standard risk/reward ratio of 1:1.33

### Fee & Tax Deductions
All profit estimates account for realistic trading costs and tax:
- **Entry fee**: 0.25% of order value
- **Exit fee**: 0.40% of order value
- **Capital gains tax**: 30% applied to net profit (after fees)
- **Minimum net profit**: Signals are only sent when the estimated net profit (after all fees and tax) is at least **1%**

The notification shows the full breakdown: gross profit → fee deductions → tax → net profit.

### Continuous Learning
After each signal, the entry price is stored. After 24 hours, the agent checks
whether the price moved in the predicted direction (≥ 1%). Correct/incorrect
labels are accumulated. When 10+ new labels exist, all three models are retrained
on the full 2-year dataset augmented with the outcome-based labels.

## Blockers & Important Notes

1. **Telegram setup is required** before any notifications will work
2. **First run trains from scratch** — fetches 2 years of data and trains 3 models (~3-5 min)
3. **Binance may be geo-restricted** in some regions; OHLCV fetching automatically falls back to Binance US and Bybit
4. **This is not financial advice** — always manage your own risk
