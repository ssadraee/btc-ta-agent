"""
BTC Technical Analysis Agent — main orchestrator.

Pipeline:
  1. Fetch EUR/USD rate
  2. Load or train models (cold start on first run)
  3. Fetch recent candles + compute features for each timeframe
  4. Generate predictions + aggregate across timeframes
  5. Evaluate outcomes of past signals (continuous learning)
  6. Retrain if enough new evaluations have accumulated
  7. Send Telegram notification if signal is actionable
  8. Persist signal history

Run via:
    python src/main.py
    python src/main.py --dry-run   # Print message without sending
    python src/main.py --force     # Skip cooldown (for testing)
"""

import argparse
import logging
import os
import sys

# Allow imports from src/ directory regardless of cwd
sys.path.insert(0, os.path.dirname(__file__))

from data_fetcher import fetch_historical, fetch_ohlcv, get_eur_usd_rate, usd_to_eur
from indicators import compute_features
from learning import (
    compute_learned_exit_multiplier,
    evaluate_outcomes,
    get_outcome_weights,
    load_signal_history,
    mark_used_for_training,
    record_signal,
    save_signal_history,
    should_retrain,
    get_stats,
)
from model import BTCModel
from notifier import (
    format_outcome_message,
    format_signal_message,
    send_telegram,
    should_send_signal,
)
from signals import aggregate_signals, build_explanation, calculate_entry_exit, compute_dynamic_exit_multiplier, get_signal_horizon, MIN_NET_PROFIT_PCT

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

TIMEFRAMES = ["1h", "4h", "1d"]
SYMBOL = "BTCUSDT"
TRAINING_DAYS = 730          # 2 years of history for initial training
RECENT_CANDLES = 300         # Candles to fetch per run for prediction
MIN_SIGNAL_CONFIDENCE = 0.55  # Don't notify below this confidence
DATA_DIR = "data"

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger("btc-ta-agent")


# ---------------------------------------------------------------------------
# Main pipeline
# ---------------------------------------------------------------------------

def main(dry_run: bool = False, force: bool = False) -> None:
    telegram_token = os.environ.get("TELEGRAM_BOT_TOKEN", "")
    telegram_chat_id = os.environ.get("TELEGRAM_CHAT_ID", "")

    if not dry_run and (not telegram_token or not telegram_chat_id):
        logger.error(
            "TELEGRAM_BOT_TOKEN and TELEGRAM_CHAT_ID environment variables must be set. "
            "Use --dry-run to skip sending."
        )
        sys.exit(1)

    # ------------------------------------------------------------------
    # Step 1: EUR/USD rate
    # ------------------------------------------------------------------
    logger.info("Fetching EUR/USD rate...")
    eur_usd_rate = get_eur_usd_rate()
    logger.info("EUR/USD rate: %.4f", eur_usd_rate)

    # ------------------------------------------------------------------
    # Step 2: Load or train models
    # ------------------------------------------------------------------
    models: dict[str, BTCModel] = {}
    dfs: dict[str, object] = {}

    for tf in TIMEFRAMES:
        model = BTCModel(tf)
        model_path = f"{DATA_DIR}/model_{tf}.pkl"

        if not model.load(model_path):
            logger.info("No saved model for %s — running initial training...", tf)
            df_hist = fetch_historical(SYMBOL, tf, days=TRAINING_DAYS)
            df_feat = compute_features(df_hist)
            metrics = model.train(df_feat)
            logger.info(
                "Model [%s] trained. Accuracy: %.1f%%",
                tf, metrics["accuracy"] * 100
            )
            model.save(model_path)

        models[tf] = model

    # ------------------------------------------------------------------
    # Step 3: Fetch recent candles + compute features
    # ------------------------------------------------------------------
    logger.info("Fetching recent OHLCV data...")
    for tf in TIMEFRAMES:
        df_recent = fetch_ohlcv(SYMBOL, tf, limit=RECENT_CANDLES)
        dfs[tf] = compute_features(df_recent)
        logger.info("  %s: %d candles fetched", tf, len(dfs[tf]))

    current_price_usd = float(dfs["1h"]["close"].iloc[-1])
    current_price_eur = usd_to_eur(current_price_usd, eur_usd_rate)
    logger.info(
        "Current BTC price: $%.2f / €%.2f",
        current_price_usd, current_price_eur
    )

    # ------------------------------------------------------------------
    # Step 4: Generate predictions
    # ------------------------------------------------------------------
    raw_signals: dict[str, tuple[int, float]] = {}
    for tf in TIMEFRAMES:
        signal, confidence = models[tf].predict(dfs[tf])
        raw_signals[tf] = (signal, confidence)
        signal_name = {1: "BUY", 0: "HOLD", -1: "SELL"}[signal]
        logger.info("  [%s] signal: %s (confidence: %.1f%%)", tf, signal_name, confidence * 100)

    final_signal, final_confidence, timeframes_summary = aggregate_signals(raw_signals)
    signal_horizon = get_signal_horizon(raw_signals)
    signal_name = {1: "BUY", 0: "HOLD", -1: "SELL"}[final_signal]
    logger.info(
        "Aggregate signal: %s (confidence: %.1f%%)",
        signal_name, final_confidence * 100
    )

    # ------------------------------------------------------------------
    # Step 5: Load signal history + evaluate past outcomes
    # ------------------------------------------------------------------
    history = load_signal_history(f"{DATA_DIR}/signal_history.json")
    logger.info("Signal history: %d records", len(history))

    history, newly_evaluated = evaluate_outcomes(history, current_price_usd)

    if newly_evaluated:
        logger.info("%d signal(s) evaluated", len(newly_evaluated))
        # Send outcome notifications for each evaluated signal
        for evaluated_record in newly_evaluated:
            if not dry_run and telegram_token:
                outcome_msg = format_outcome_message(
                    original_signal=evaluated_record["signal"],
                    entry_price_usd=evaluated_record["entry_price_usd"],
                    current_price_usd=evaluated_record["outcome_price_usd"],
                    outcome=evaluated_record["outcome"],
                    pct_change=evaluated_record["outcome_pct_change"],
                )
                send_telegram(telegram_token, telegram_chat_id, outcome_msg)

    # ------------------------------------------------------------------
    # Step 6: Retrain if enough new evaluations have accumulated
    # ------------------------------------------------------------------
    retrain_needed, recent_accuracy = should_retrain(history)

    if retrain_needed:
        outcome_wts = get_outcome_weights(history)
        logger.info(
            "Retraining trigger: accuracy %.1f%% below threshold — retraining all models...",
            (recent_accuracy or 0) * 100,
        )
        for tf in TIMEFRAMES:
            df_hist = fetch_historical(SYMBOL, tf, days=TRAINING_DAYS)
            df_feat = compute_features(df_hist)
            metrics = models[tf].retrain_incremental(df_feat, outcome_weights=outcome_wts)
            logger.info(
                "Model [%s] retrained. Accuracy: %.1f%%",
                tf, metrics["accuracy"] * 100,
            )
            models[tf].save(f"{DATA_DIR}/model_{tf}.pkl")
        history = mark_used_for_training(history)
    elif recent_accuracy is not None:
        # Accuracy was good enough — still mark evaluations as consumed
        # so they don't pile up and trigger re-evaluation every run
        logger.info("Model accuracy sufficient — skipping retrain")
        history = mark_used_for_training(history)

    # ------------------------------------------------------------------
    # Step 7: Send notification if actionable
    # ------------------------------------------------------------------
    if final_signal == 0:
        logger.info("Signal is HOLD — no notification sent")
    elif final_confidence < MIN_SIGNAL_CONFIDENCE and not force:
        logger.info(
            "Signal confidence too low (%.1f%% < %.1f%%) — skipping notification",
            final_confidence * 100, MIN_SIGNAL_CONFIDENCE * 100
        )
    elif not should_send_signal(history, final_signal, final_confidence) and not force:
        logger.info("Signal suppressed by cooldown")
    else:
        explanation = build_explanation(raw_signals, dfs)

        # Compute dynamic exit multiplier: blend historical data + learned experience
        hist_mult = compute_dynamic_exit_multiplier(dfs["1h"])
        learned_adj = compute_learned_exit_multiplier(history)
        if learned_adj is not None:
            # 70% anchored to historical data, 30% adjusted by learned outcomes
            blended_mult = hist_mult * (0.7 + 0.3 * learned_adj)
            logger.info(
                "Exit multiplier: hist=%.2f, learned_adj=%.2f → blended=%.2f",
                hist_mult, learned_adj, blended_mult,
            )
        else:
            blended_mult = hist_mult
            logger.info("Exit multiplier (historical only): %.2f", blended_mult)

        prices = calculate_entry_exit(dfs["1h"], final_signal, exit_multiplier=blended_mult)

        net_profit_pct = prices["net_profit_pct"]
        if net_profit_pct < MIN_NET_PROFIT_PCT and not force:
            logger.info(
                "Net profit %.2f%% is below minimum %.1f%% (after fees & tax) — skipping signal",
                net_profit_pct, MIN_NET_PROFIT_PCT,
            )
        else:
            entry_usd = prices["entry_price"]
            exit_usd = prices["exit_price"]
            stop_loss_usd = prices["stop_loss"]

            message = format_signal_message(
                signal=final_signal,
                entry_usd=entry_usd,
                exit_usd=exit_usd,
                stop_loss_usd=stop_loss_usd,
                entry_eur=usd_to_eur(entry_usd, eur_usd_rate),
                exit_eur=usd_to_eur(exit_usd, eur_usd_rate),
                stop_loss_eur=usd_to_eur(stop_loss_usd, eur_usd_rate),
                profit_pct=prices["profit_margin_pct"],
                net_profit_pct=net_profit_pct,
                entry_fee_pct=prices["entry_fee_pct"],
                exit_fee_pct=prices["exit_fee_pct"],
                tax_pct=prices["tax_pct"],
                explanation=explanation,
                confidence=final_confidence,
                timeframes_summary=timeframes_summary,
                signal_horizon=signal_horizon,
            )

            if dry_run:
                print("\n" + "=" * 60)
                print("DRY RUN — Telegram message would be:")
                print("=" * 60)
                print(message)
                print("=" * 60 + "\n")
            else:
                sent = send_telegram(telegram_token, telegram_chat_id, message)
                if sent:
                    history = record_signal(
                        history,
                        signal=final_signal,
                        entry_price_usd=entry_usd,
                        exit_price_target_usd=exit_usd,
                        timeframes_summary=timeframes_summary,
                        confidence=final_confidence,
                    )

    # ------------------------------------------------------------------
    # Step 8: Persist signal history
    # ------------------------------------------------------------------
    save_signal_history(history, f"{DATA_DIR}/signal_history.json")

    stats = get_stats(history)
    logger.info(
        "Stats: %d signals total, %d evaluated, %.1f%% accuracy",
        stats["total_signals"],
        stats["evaluated"],
        stats.get("accuracy_pct", 0.0),
    )

    logger.info("Run complete.")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="BTC Technical Analysis Agent")
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print the Telegram message without actually sending it",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Override cooldown and confidence filters (for testing)",
    )
    args = parser.parse_args()
    main(dry_run=args.dry_run, force=args.force)
