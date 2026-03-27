"""
Telegram notification module.

Formats and sends trading signal messages with entry/exit prices,
confidence levels, and human-readable explanations.
"""

import logging
from datetime import datetime, timezone

import requests

logger = logging.getLogger(__name__)

TELEGRAM_API = "https://api.telegram.org/bot{token}/sendMessage"

SIGNAL_EMOJI = {1: "🟢", -1: "🔴", 0: "⚪"}
SIGNAL_NAMES = {1: "BUY", -1: "SELL", 0: "HOLD"}


def send_telegram(token: str, chat_id: str, message: str) -> bool:
    """
    Send a message to a Telegram chat.

    Args:
        token: Telegram bot token
        chat_id: Target chat ID (user, group, or channel)
        message: HTML-formatted message text

    Returns:
        True if sent successfully, False otherwise
    """
    url = TELEGRAM_API.format(token=token)
    payload = {
        "chat_id": chat_id,
        "text": message,
        "parse_mode": "HTML",
        "disable_web_page_preview": True,
    }
    try:
        resp = requests.post(url, json=payload, timeout=10)
        resp.raise_for_status()
        logger.info("Telegram notification sent successfully")
        return True
    except requests.RequestException as e:
        logger.error("Failed to send Telegram notification: %s", e)
        return False


def format_signal_message(
    signal: int,
    entry_usd: float,
    exit_usd: float,
    stop_loss_usd: float,
    entry_eur: float,
    exit_eur: float,
    stop_loss_eur: float,
    profit_pct: float,
    net_profit_pct: float,
    entry_fee_pct: float,
    exit_fee_pct: float,
    tax_pct: float,
    explanation: str,
    confidence: float,
    timeframes_summary: str,
    signal_horizon: str = "1–5 days",
    polymarket_summary: str | None = None,
    polymarket_source: str | None = None,
) -> str:
    """
    Build an HTML-formatted Telegram message for a trading signal.

    Returns a plain-English message suitable for non-technical users.
    """
    emoji = SIGNAL_EMOJI[signal]
    name = SIGNAL_NAMES[signal]
    now = datetime.now(tz=timezone.utc).strftime("%Y-%m-%d %H:%M UTC")

    if signal == 1:
        direction_note = "This is a <b>long (buy)</b> signal — the model expects the price to rise."
        profit_label = "Profit Target (gross)"
        profit_icon = "📈"
    else:
        direction_note = "This is a <b>short (sell)</b> signal — the model expects the price to fall."
        profit_label = "Profit Target gross (short)"
        profit_icon = "📉"

    confidence_pct = round(confidence * 100)
    confidence_bar = _confidence_bar(confidence_pct)

    message = (
        f"{emoji} <b>BTC Trading Signal — {name}</b>\n"
        f"<i>{now}</i>\n"
        "\n"
        f"📊 <b>Recommendation:</b> {name}\n"
        f"{direction_note}\n"
        "\n"
        f"💵 <b>Entry Price:</b> ${entry_usd:,.2f}  (€{entry_eur:,.2f})\n"
        f"🎯 <b>Exit Price:</b> ${exit_usd:,.2f}  (€{exit_eur:,.2f})  <i>(ATR-based, 1h chart)</i>\n"
        f"🛑 <b>Stop Loss:</b> ${stop_loss_usd:,.2f}  (€{stop_loss_eur:,.2f})\n"
        f"{profit_icon} <b>{profit_label}:</b> +{profit_pct:.2f}%\n"
        f"   ↳ Entry fee (0.25%): -{entry_fee_pct:.2f}%\n"
        f"   ↳ Exit fee (0.40%):  -{exit_fee_pct:.2f}%\n"
        f"   ↳ Tax (30% on profit): -{tax_pct:.2f}%\n"
        f"💰 <b>Net Profit (est.):</b> +{net_profit_pct:.2f}%\n"
        "\n"
        f"🧠 <b>Explanation:</b>\n"
        f"{explanation}\n"
        "\n"
        f"⏱ <b>Timeframes:</b> {timeframes_summary}\n"
        f"🧭 <b>Signal Horizon:</b> ~{signal_horizon}\n"
    )

    if polymarket_summary:
        # Indent each line of the multi-line summary
        indented = polymarket_summary.replace("\n", "\n   ")
        message += (
            f"\n"
            f"🔮 <b>Market Sentiment (Polymarket):</b>\n"
            f"   {indented}\n"
        )

    if polymarket_summary:
        confidence_note = (
            "Weighted model certainty across 3 timeframes + Polymarket sentiment. "
            "Higher = stronger agreement."
        )
    else:
        confidence_note = (
            "Weighted model certainty across all 3 timeframes "
            "(1d 40% · 4h 35% · 1h 25%). Higher = stronger agreement."
        )

    message += (
        f"\n"
        f"📊 <b>Confidence:</b> {confidence_pct}% {confidence_bar}\n"
        f"<i>{confidence_note}</i>\n"
        "\n"
        f"⚠️ <i>This is not financial advice. Always manage your risk and never invest more than you can afford to lose.</i>"
    )

    return message


def format_outcome_message(
    original_signal: int,
    entry_price_usd: float,
    current_price_usd: float,
    outcome: str,
    pct_change: float,
) -> str:
    """
    Build a follow-up message reporting the outcome of a past signal.
    Sent 24h after the original signal to close the learning loop.
    """
    was_correct = outcome == "correct"
    result_emoji = "✅" if was_correct else "❌"
    direction = "up" if pct_change > 0 else "down"
    original_name = SIGNAL_NAMES[original_signal]

    message = (
        f"{result_emoji} <b>Signal Outcome Update</b>\n"
        "\n"
        f"Original recommendation: <b>{original_name}</b>\n"
        f"Entry price: <b>${entry_price_usd:,.2f}</b>\n"
        f"Current price (24h later): <b>${current_price_usd:,.2f}</b>\n"
        f"Price moved: <b>{direction} {abs(pct_change):.2f}%</b>\n"
        "\n"
        f"Result: <b>{'Correct' if was_correct else 'Incorrect'}</b> — "
        f"the model {'got it right' if was_correct else 'missed this one'}.\n"
        "\n"
        f"<i>The model has been updated to learn from this outcome.</i>"
    )

    return message


def should_send_signal(
    signal_history: list[dict],
    current_signal: int,
    current_confidence: float | None = None,
    cooldown_hours: int = 4,
    confidence_increase_threshold: float = 0.05,
) -> bool:
    """
    Prevent signal spam by enforcing a cooldown between same-direction signals.

    The cooldown is bypassed if the current confidence is at least
    ``confidence_increase_threshold`` higher than the previous signal's
    confidence, indicating a strengthening trend.

    Args:
        signal_history: list of past signal records
        current_signal: 1=BUY, -1=SELL
        current_confidence: aggregate confidence for the current signal (0–1)
        cooldown_hours: minimum hours between same-direction signals
        confidence_increase_threshold: minimum confidence increase (in absolute
            terms) required to bypass the cooldown (default 0.05 = 5 pp)

    Returns:
        True if signal should be sent, False if still in cooldown
    """
    from datetime import timedelta

    now = datetime.now(tz=timezone.utc)
    cutoff = now - timedelta(hours=cooldown_hours)

    for record in reversed(signal_history):
        ts_str = record.get("timestamp", "")
        if not ts_str:
            continue
        try:
            ts = datetime.fromisoformat(ts_str)
            if ts.tzinfo is None:
                ts = ts.replace(tzinfo=timezone.utc)
        except ValueError:
            continue

        if ts < cutoff:
            break  # Records are chronological; no need to check older ones

        if record.get("signal") == current_signal:
            prev_confidence = record.get("confidence")
            if (
                current_confidence is not None
                and prev_confidence is not None
                and current_confidence >= prev_confidence + confidence_increase_threshold
            ):
                logger.info(
                    "Cooldown bypassed: %s confidence increased from %.0f%% to %.0f%%",
                    SIGNAL_NAMES[current_signal],
                    prev_confidence * 100,
                    current_confidence * 100,
                )
                return True
            logger.info(
                "Signal suppressed (cooldown): same %s signal was sent at %s",
                SIGNAL_NAMES[current_signal], ts_str
            )
            return False

    return True


def format_retrain_message(
    retrain_results: dict,
    pre_signal_accuracy: float,
    stats: dict,
) -> str:
    """
    Build an HTML-formatted Telegram message summarising a model retraining event.

    Args:
        retrain_results: per-timeframe dict with keys accuracy, n_train,
            importance_before, importance_after
        pre_signal_accuracy: signal-history accuracy that triggered the retrain (0–1)
        stats: output of get_stats() — includes buy/sell accuracy breakdown

    Returns:
        HTML-formatted string ready for Telegram
    """
    now = datetime.now(tz=timezone.utc).strftime("%Y-%m-%d %H:%M UTC")
    bullets = _generate_retrain_explanation(retrain_results, pre_signal_accuracy, stats)

    per_tf_lines = []
    n_train_parts = []
    for tf in ("1h", "4h", "1d"):
        r = retrain_results.get(tf, {})
        acc = r.get("accuracy")
        n = r.get("n_train")
        acc_str = f"{round(acc * 100)}%" if acc is not None else "n/a"
        n_str = f"{n:,}" if n is not None else "n/a"
        per_tf_lines.append(f"{tf}: {acc_str}")
        n_train_parts.append(f"{tf}: {n_str}")

    bullet_text = "\n".join(f"• {b}" for b in bullets)

    message = (
        f"🔄 <b>Model Retrained</b>\n"
        f"<i>{now}</i>\n"
        "\n"
        f"<b>What changed:</b>\n"
        f"{bullet_text}\n"
        "\n"
        f"<b>Per-timeframe accuracy (test set):</b>\n"
        f"  {' | '.join(per_tf_lines)}\n"
        f"  Training samples — {', '.join(n_train_parts)}\n"
        "\n"
        f"<i>Model weights updated. Next signals reflect the latest patterns.</i>"
    )
    return message


def _generate_retrain_explanation(
    retrain_results: dict,
    pre_signal_accuracy: float,
    stats: dict,
) -> list:
    """Return up to 5 plain-English bullet points explaining the retrain event."""
    bullets = []

    # --- Bullet 1: Accuracy change ---
    post_accs = [
        r["accuracy"] for r in retrain_results.values() if r.get("accuracy") is not None
    ]
    avg_post = sum(post_accs) / len(post_accs) if post_accs else None
    pre_pct = round(pre_signal_accuracy * 100)

    if avg_post is not None:
        avg_pct = round(avg_post * 100)
        if avg_post > pre_signal_accuracy + 0.05:
            bullets.append(
                f"Model improved: signal accuracy was {pre_pct}%, model test accuracy now avg {avg_pct}% across timeframes."
            )
        elif avg_post < pre_signal_accuracy - 0.05:
            bullets.append(
                f"Test accuracy ({avg_pct}% avg) is below prior signal accuracy ({pre_pct}%) — model may need more signal history to generalise."
            )
        else:
            bullets.append(
                f"Marginal change: signal accuracy was {pre_pct}%, model test accuracy now avg {avg_pct}% — weights refreshed to recent data."
            )
    else:
        bullets.append(f"Signal accuracy before retrain: {pre_pct}%. No post-retrain metrics available.")

    # --- Bullet 2: Feature importance shift ---
    shifted = []
    for tf in ("1h", "4h", "1d"):
        r = retrain_results.get(tf, {})
        imp_before = r.get("importance_before") or {}
        imp_after = r.get("importance_after") or {}
        top_before = next(iter(imp_before), None)
        top_after = next(iter(imp_after), None)
        if top_before and top_after and top_before != top_after:
            shifted.append(f"{tf}: <code>{top_after}</code> (was <code>{top_before}</code>)")

    if shifted:
        bullets.append(
            f"Top feature shifted in {', '.join(shifted)} — suggesting a market regime change."
        )
    else:
        bullets.append("Feature rankings stable — model reinforced existing patterns without regime shift.")

    # --- Bullet 3: WHY performance changed (causal) ---
    buy_acc = stats.get("buy_accuracy_pct")
    sell_acc = stats.get("sell_accuracy_pct")

    if buy_acc is not None and sell_acc is not None:
        if buy_acc < 40 and sell_acc < 40:
            bullets.append(
                f"Model struggled in both directions (BUY {buy_acc:.0f}%, SELL {sell_acc:.0f}%) — likely choppy, range-bound market conditions."
            )
        elif buy_acc < 40:
            bullets.append(
                f"BUY signals underperformed ({buy_acc:.0f}% accuracy) — market likely trended sideways or down, weakening long-side patterns."
            )
        elif sell_acc < 40:
            bullets.append(
                f"SELL signals missed ({sell_acc:.0f}% accuracy) — unexpected bullish continuation may have caught the model off-guard."
            )
        elif buy_acc > 60 and sell_acc > 60:
            bullets.append(
                f"Both directions performed well (BUY {buy_acc:.0f}%, SELL {sell_acc:.0f}%) — retrain refreshes weights to current volatility regime."
            )
        else:
            weaker = "BUY" if buy_acc < sell_acc else "SELL"
            weaker_acc = min(buy_acc, sell_acc)
            bullets.append(
                f"{weaker} side was weaker ({weaker_acc:.0f}%) — model adjusted sample weights to focus on underperforming signal direction."
            )
    else:
        bullets.append("Insufficient directional history to diagnose cause — model retrained on updated outcome weights.")

    # --- Bullet 4: Risk flag ---
    if avg_post is not None and avg_post > 0.85:
        bullets.append(
            f"Risk: High test accuracy ({round(avg_post * 100)}%) may indicate overfitting to historical patterns — monitor live signal quality."
        )
    elif len(shifted) >= 2:
        bullets.append(
            "Risk: Feature importance shifted in multiple timeframes — possible data drift; treat next 2–3 signals as probationary."
        )
    else:
        min_n = min(
            (r["n_train"] for r in retrain_results.values() if r.get("n_train") is not None),
            default=None,
        )
        if min_n is not None and min_n < 5000:
            bullets.append(
                f"Risk: Smallest training set is {min_n:,} samples — one timeframe may produce less stable predictions until more data accumulates."
            )
        else:
            bullets.append("No obvious overfitting, instability, or data drift detected.")

    # --- Bullet 5: Directional bias summary ---
    evaluated = stats.get("evaluated", 0)
    correct = stats.get("correct", 0)
    buy_str = f"{buy_acc:.0f}%" if buy_acc is not None else "n/a"
    sell_str = f"{sell_acc:.0f}%" if sell_acc is not None else "n/a"
    bullets.append(
        f"Signal history: {evaluated} evaluated, {correct} correct. BUY: {buy_str} | SELL: {sell_str}."
    )

    return bullets


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _confidence_bar(pct: int) -> str:
    """Return a simple ASCII progress bar for confidence display."""
    filled = round(pct / 10)
    empty = 10 - filled
    return "▓" * filled + "░" * empty
