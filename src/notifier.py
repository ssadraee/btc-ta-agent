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
    explanation: str,
    confidence: float,
    timeframes_summary: str,
    signal_horizon: str = "1–5 days",
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
        profit_label = "Profit Target"
        profit_icon = "📈"
    else:
        direction_note = "This is a <b>short (sell)</b> signal — the model expects the price to fall."
        profit_label = "Profit Target (short)"
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
        "\n"
        f"🧠 <b>Explanation:</b>\n"
        f"{explanation}\n"
        "\n"
        f"⏱ <b>Timeframes:</b> {timeframes_summary}\n"
        f"🧭 <b>Signal Horizon:</b> ~{signal_horizon}\n"
        f"📊 <b>Confidence:</b> {confidence_pct}% {confidence_bar}\n"
        f"<i>Weighted model certainty across all 3 timeframes (1d 40% · 4h 35% · 1h 25%). Higher = stronger agreement.</i>\n"
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


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _confidence_bar(pct: int) -> str:
    """Return a simple ASCII progress bar for confidence display."""
    filled = round(pct / 10)
    empty = 10 - filled
    return "▓" * filled + "░" * empty
