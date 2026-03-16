"""
Continuous learning module.

Tracks past signals, evaluates their outcomes after 24 hours,
and prepares augmented training data for model retraining.

Storage format (data/signal_history.json):
[
  {
    "id": "20240315T120000",
    "timestamp": "2024-03-15T12:00:00+00:00",
    "signal": 1,                      # 1=BUY, -1=SELL
    "confidence": 0.67,               # aggregate confidence (0–1)
    "entry_price_usd": 84200.0,
    "exit_price_target_usd": 87400.0,
    "timeframes_summary": "1h: BUY (71%) | ...",
    "evaluated": false,
    "outcome": null,                   # "correct" | "incorrect" after eval
    "outcome_price_usd": null,
    "outcome_pct_change": null,
    "outcome_timestamp": null
  }
]
"""

import json
import logging
import os
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pandas as pd

logger = logging.getLogger(__name__)

HISTORY_PATH = "data/signal_history.json"
EVALUATION_DELAY_HOURS = 24
RETRAIN_THRESHOLD = 10   # Retrain after this many new evaluations
OUTCOME_THRESHOLD = 0.01  # 1% move to count as correct (more lenient than label threshold)
MIN_ACCURACY_FOR_SKIP = 0.65  # Skip retraining if recent accuracy is above this
ERROR_WEIGHT_MULTIPLIER = 2.0  # Training weight boost for time periods with incorrect signals


def load_signal_history(path: str = HISTORY_PATH) -> list[dict]:
    """Load signal history from JSON file. Returns empty list if not found."""
    if not os.path.exists(path):
        return []
    try:
        with open(path, "r") as f:
            return json.load(f)
    except (json.JSONDecodeError, IOError) as e:
        logger.warning("Could not load signal history from %s: %s", path, e)
        return []


def save_signal_history(history: list[dict], path: str = HISTORY_PATH) -> None:
    """Persist signal history to JSON file."""
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        json.dump(history, f, indent=2, default=str)
    logger.debug("Signal history saved (%d records)", len(history))


def record_signal(
    history: list[dict],
    signal: int,
    entry_price_usd: float,
    exit_price_target_usd: float,
    timeframes_summary: str,
    confidence: float | None = None,
) -> list[dict]:
    """
    Append a new signal to the history.

    Args:
        history: existing signal history list
        signal: 1=BUY, -1=SELL
        entry_price_usd: price at signal time
        exit_price_target_usd: target exit price
        timeframes_summary: human-readable timeframe breakdown string
        confidence: aggregate model confidence (0–1)

    Returns:
        Updated history list
    """
    now = datetime.now(tz=timezone.utc)
    record = {
        "id": now.strftime("%Y%m%dT%H%M%S"),
        "timestamp": now.isoformat(),
        "signal": signal,
        "confidence": confidence,
        "entry_price_usd": entry_price_usd,
        "exit_price_target_usd": exit_price_target_usd,
        "timeframes_summary": timeframes_summary,
        "evaluated": False,
        "outcome": None,
        "outcome_price_usd": None,
        "outcome_pct_change": None,
        "outcome_timestamp": None,
    }
    history = list(history)
    history.append(record)
    logger.info(
        "Signal recorded: %s at $%.2f (target $%.2f)",
        "BUY" if signal == 1 else "SELL",
        entry_price_usd,
        exit_price_target_usd,
    )
    return history


def evaluate_outcomes(
    history: list[dict],
    current_price_usd: float,
) -> tuple[list[dict], list[dict]]:
    """
    Evaluate signals that are older than EVALUATION_DELAY_HOURS and not yet evaluated.

    For each qualifying signal:
    - BUY is correct if price rose by >= OUTCOME_THRESHOLD
    - SELL is correct if price fell by >= OUTCOME_THRESHOLD

    Args:
        history: full signal history
        current_price_usd: latest BTC price in USD

    Returns:
        (updated_history, newly_evaluated_records)
    """
    now = datetime.now(tz=timezone.utc)
    cutoff = now - timedelta(hours=EVALUATION_DELAY_HOURS)
    newly_evaluated = []

    updated = []
    for record in history:
        if record.get("evaluated"):
            updated.append(record)
            continue

        ts_str = record.get("timestamp", "")
        try:
            ts = datetime.fromisoformat(ts_str)
            if ts.tzinfo is None:
                ts = ts.replace(tzinfo=timezone.utc)
        except (ValueError, TypeError):
            updated.append(record)
            continue

        if ts > cutoff:
            # Not old enough yet
            updated.append(record)
            continue

        # Evaluate the outcome
        entry_price = record["entry_price_usd"]
        pct_change = (current_price_usd - entry_price) / entry_price

        signal = record["signal"]
        if signal == 1:  # BUY
            correct = pct_change >= OUTCOME_THRESHOLD
        else:  # SELL
            correct = pct_change <= -OUTCOME_THRESHOLD

        record = dict(record)
        record["evaluated"] = True
        record["outcome"] = "correct" if correct else "incorrect"
        record["outcome_price_usd"] = current_price_usd
        record["outcome_pct_change"] = round(pct_change * 100, 4)
        record["outcome_timestamp"] = now.isoformat()

        updated.append(record)
        newly_evaluated.append(record)

        logger.info(
            "Signal %s evaluated: %s (entry $%.2f → current $%.2f, %.2f%%)",
            record["id"],
            record["outcome"].upper(),
            entry_price,
            current_price_usd,
            pct_change * 100,
        )

    return updated, newly_evaluated


def should_retrain(history: list[dict]) -> tuple[bool, float | None]:
    """
    Decide whether retraining is needed based on recent signal accuracy.

    Returns:
        (should_retrain, recent_accuracy)
        - If fewer than RETRAIN_THRESHOLD evaluations: (False, None)
        - If accuracy >= MIN_ACCURACY_FOR_SKIP: (False, accuracy) — model is fine
        - If accuracy < MIN_ACCURACY_FOR_SKIP: (True, accuracy) — retrain needed
    """
    recent = [
        r for r in history
        if r.get("evaluated") and r.get("outcome") is not None
        and not r.get("used_for_training", False)
    ]
    if len(recent) < RETRAIN_THRESHOLD:
        return False, None

    correct = sum(1 for r in recent if r["outcome"] == "correct")
    accuracy = correct / len(recent)

    if accuracy >= MIN_ACCURACY_FOR_SKIP:
        logger.info(
            "Recent accuracy %.1f%% (%d/%d) — above %.0f%% threshold, skipping retrain",
            accuracy * 100, correct, len(recent), MIN_ACCURACY_FOR_SKIP * 100,
        )
        return False, accuracy

    logger.info(
        "Recent accuracy %.1f%% (%d/%d) — below %.0f%% threshold, retrain needed",
        accuracy * 100, correct, len(recent), MIN_ACCURACY_FOR_SKIP * 100,
    )
    return True, accuracy


def get_outcome_weights(history: list[dict]) -> dict[str, float]:
    """
    Build a mapping of signal timestamps to training weight multipliers.

    Incorrect signals get ERROR_WEIGHT_MULTIPLIER (2.0) so the model pays
    more attention to patterns from time periods where it failed.
    Correct signals get 1.0 (no change).

    Returns:
        {iso_timestamp: weight} for all evaluated signals not yet used for training.
    """
    weights = {}
    for r in history:
        if not r.get("evaluated") or r.get("outcome") is None:
            continue
        if r.get("used_for_training", False):
            continue
        ts = r.get("timestamp")
        if ts:
            weight = ERROR_WEIGHT_MULTIPLIER if r["outcome"] == "incorrect" else 1.0
            weights[ts] = weight
    return weights


def mark_used_for_training(history: list[dict]) -> list[dict]:
    """Mark all evaluated records as used for training to avoid double-counting."""
    updated = []
    for record in history:
        if record.get("evaluated") and not record.get("used_for_training"):
            record = dict(record)
            record["used_for_training"] = True
        updated.append(record)
    return updated


def get_stats(history: list[dict]) -> dict:
    """
    Compute basic statistics about past signal accuracy.
    Useful for the README and debugging.
    """
    evaluated = [r for r in history if r.get("evaluated") and r.get("outcome")]
    if not evaluated:
        return {"total_signals": len(history), "evaluated": 0}

    correct = sum(1 for r in evaluated if r["outcome"] == "correct")
    buys = [r for r in evaluated if r["signal"] == 1]
    sells = [r for r in evaluated if r["signal"] == -1]
    buy_correct = sum(1 for r in buys if r["outcome"] == "correct")
    sell_correct = sum(1 for r in sells if r["outcome"] == "correct")

    return {
        "total_signals": len(history),
        "evaluated": len(evaluated),
        "correct": correct,
        "accuracy_pct": round(correct / len(evaluated) * 100, 1),
        "buy_signals": len(buys),
        "buy_accuracy_pct": round(buy_correct / len(buys) * 100, 1) if buys else None,
        "sell_signals": len(sells),
        "sell_accuracy_pct": round(sell_correct / len(sells) * 100, 1) if sells else None,
    }
