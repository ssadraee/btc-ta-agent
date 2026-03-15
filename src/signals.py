"""
Multi-timeframe signal aggregation, entry/exit price calculation,
and human-readable explanation generation.
"""

import logging

import pandas as pd

from indicators import get_latest_indicator_summary

logger = logging.getLogger(__name__)

# Timeframe weights: longer timeframes carry more influence
TIMEFRAME_WEIGHTS = {"1h": 0.25, "4h": 0.35, "1d": 0.40}

# Minimum confidence per timeframe to contribute to the aggregate
MIN_CONFIDENCE = 0.45

# Minimum aggregate weighted score to trigger BUY or SELL
SIGNAL_THRESHOLD = 0.40

# ATR multipliers for exit and stop-loss calculation
ATR_EXIT_MULTIPLIER = 2.0
ATR_STOP_MULTIPLIER = 1.5

SIGNAL_NAMES = {1: "BUY", 0: "HOLD", -1: "SELL"}


def aggregate_signals(
    signals: dict[str, tuple[int, float]]
) -> tuple[int, float, str]:
    """
    Combine per-timeframe model signals into one final decision.

    Args:
        signals: {"1h": (signal, confidence), "4h": ..., "1d": ...}
                 signal: 1=BUY, 0=HOLD, -1=SELL
                 confidence: [0, 1]

    Returns:
        (final_signal, weighted_confidence, timeframe_summary_string)
        final_signal: 1=BUY, 0=HOLD, -1=SELL
    """
    weighted_sum = 0.0
    total_weight = 0.0
    parts = []

    for tf, (signal, confidence) in signals.items():
        weight = TIMEFRAME_WEIGHTS.get(tf, 0.0)
        # Only count confident signals; HOLD contributes 0
        if confidence >= MIN_CONFIDENCE:
            weighted_sum += signal * confidence * weight
            total_weight += weight
        parts.append(f"{tf}: {SIGNAL_NAMES[signal]} ({confidence:.0%})")

    timeframe_summary = " | ".join(parts)

    if total_weight == 0:
        return 0, 0.0, timeframe_summary

    # Normalise: range is [-1, 1]
    normalised = weighted_sum / total_weight
    weighted_confidence = abs(normalised)

    if normalised > SIGNAL_THRESHOLD:
        final_signal = 1
    elif normalised < -SIGNAL_THRESHOLD:
        final_signal = -1
    else:
        final_signal = 0

    logger.debug(
        "Aggregate: normalised=%.3f → %s (confidence %.1f%%)",
        normalised, SIGNAL_NAMES[final_signal], weighted_confidence * 100
    )

    return final_signal, weighted_confidence, timeframe_summary


def calculate_entry_exit(df: pd.DataFrame, signal: int) -> dict:
    """
    Calculate entry price, exit target, stop-loss, and profit margin.

    Uses ATR-based dynamic targets — a professional risk management standard.

    Args:
        df: feature DataFrame (must contain 'close' and 'atr_14' columns)
        signal: 1=BUY, -1=SELL

    Returns:
        dict with entry_price, exit_price, stop_loss, profit_margin_pct
    """
    entry_price = float(df["close"].iloc[-1])
    atr = float(df["atr_14"].iloc[-1])

    if signal == 1:  # BUY
        exit_price = entry_price + (atr * ATR_EXIT_MULTIPLIER)
        stop_loss = entry_price - (atr * ATR_STOP_MULTIPLIER)
    elif signal == -1:  # SELL / Short
        exit_price = entry_price - (atr * ATR_EXIT_MULTIPLIER)
        stop_loss = entry_price + (atr * ATR_STOP_MULTIPLIER)
    else:
        exit_price = entry_price
        stop_loss = entry_price

    profit_margin_pct = abs(exit_price - entry_price) / entry_price * 100

    return {
        "entry_price": entry_price,
        "exit_price": exit_price,
        "stop_loss": stop_loss,
        "profit_margin_pct": profit_margin_pct,
        "atr": atr,
    }


def build_explanation(
    signals: dict[str, tuple[int, float]],
    dfs: dict[str, pd.DataFrame],
) -> str:
    """
    Build a plain-English explanation of the signal based on indicator values.

    Args:
        signals: per-timeframe signals {"1h": (signal, conf), ...}
        dfs: per-timeframe feature DataFrames {"1h": df, ...}

    Returns:
        Multi-sentence explanation string
    """
    sentences = []

    # Use the 1h and 4h charts for explanation (most actionable)
    for tf in ["1h", "4h", "1d"]:
        if tf not in dfs:
            continue
        summary = get_latest_indicator_summary(dfs[tf])
        sig, conf = signals.get(tf, (0, 0.0))

        tf_label = {"1h": "1-hour", "4h": "4-hour", "1d": "daily"}[tf]
        rsi = summary["rsi"]
        macd_hist = summary["macd_hist"]
        stoch_k = summary["stoch_rsi_k"]
        bb_pct = summary["bb_pct"]
        vol_ratio = summary["volume_ratio"]

        # RSI commentary
        if rsi <= 30:
            sentences.append(f"RSI({tf_label}) is {rsi} — deeply oversold, suggesting exhausted selling pressure.")
        elif rsi <= 40:
            sentences.append(f"RSI({tf_label}) is {rsi} — approaching oversold territory.")
        elif rsi >= 70:
            sentences.append(f"RSI({tf_label}) is {rsi} — overbought, momentum may be overextended.")
        elif rsi >= 60:
            sentences.append(f"RSI({tf_label}) is {rsi} — in bullish momentum zone.")

        # MACD commentary (1h and 4h only to avoid redundancy)
        if tf in ["1h", "4h"]:
            if macd_hist > 0:
                sentences.append(f"MACD histogram ({tf_label}) is positive, indicating bullish momentum.")
            elif macd_hist < 0:
                sentences.append(f"MACD histogram ({tf_label}) is negative, indicating bearish momentum.")

        # EMA trend
        if summary["ema_cross_50_200"] == 1:
            sentences.append(f"EMA50 is above EMA200 on the {tf_label} chart — golden cross in effect (bullish long-term trend).")
        elif summary["ema_cross_50_200"] == -1:
            sentences.append(f"EMA50 is below EMA200 on the {tf_label} chart — death cross in effect (bearish long-term trend).")

        if tf == "1d":
            pct_vs_200 = summary["close_vs_ema200_pct"]
            if pct_vs_200 > 0:
                sentences.append(f"Price is {pct_vs_200:.1f}% above the 200-day EMA, confirming the macro uptrend.")
            else:
                sentences.append(f"Price is {abs(pct_vs_200):.1f}% below the 200-day EMA, signalling a macro downtrend.")

        # Volume
        if vol_ratio > 1.5:
            sentences.append(f"Volume ({tf_label}) is {vol_ratio:.1f}x above the 20-period average — strong conviction behind this move.")

        # Bollinger Bands
        if bb_pct is not None and not (bb_pct != bb_pct):  # check NaN
            if bb_pct < 0.1:
                sentences.append(f"Price is near the lower Bollinger Band ({tf_label}) — potential support zone.")
            elif bb_pct > 0.9:
                sentences.append(f"Price is near the upper Bollinger Band ({tf_label}) — potential resistance zone.")

        # Candlestick patterns (only report on 1h/4h to keep explanations concise)
        if tf not in ["1h", "4h"]:
            continue

        # Bullish single-candle
        if summary["cdl_hammer"]:
            sentences.append(f"A hammer pattern was detected ({tf_label}) — classic bullish reversal signal.")
        if summary["cdl_inverted_hammer"]:
            sentences.append(f"An inverted hammer appeared ({tf_label}) — potential bullish reversal.")
        if summary["cdl_marubozu"]:
            sentences.append(f"A marubozu candle detected ({tf_label}) — strong directional momentum.")

        # Bearish single-candle
        if summary["cdl_shooting_star"]:
            sentences.append(f"A shooting star pattern appeared ({tf_label}) — bearish reversal signal.")
        if summary["cdl_hanging_man"]:
            sentences.append(f"A hanging man detected ({tf_label}) — potential bearish reversal.")

        # Two-candle patterns
        if summary["cdl_engulfing"]:
            sentences.append(f"A bullish engulfing pattern detected ({tf_label}) — strong reversal indicator.")
        if summary["cdl_bearish_engulfing"]:
            sentences.append(f"A bearish engulfing pattern detected ({tf_label}) — strong bearish reversal.")
        if summary["cdl_piercing_line"]:
            sentences.append(f"A piercing line pattern appeared ({tf_label}) — bullish reversal signal.")
        if summary["cdl_dark_cloud"]:
            sentences.append(f"A dark cloud cover detected ({tf_label}) — bearish reversal signal.")
        if summary["cdl_tweezer_top"]:
            sentences.append(f"A tweezer top detected ({tf_label}) — potential bearish reversal at resistance.")
        if summary["cdl_tweezer_bottom"]:
            sentences.append(f"A tweezer bottom detected ({tf_label}) — potential bullish reversal at support.")

        # Three-candle patterns
        if summary["cdl_morning_star"]:
            sentences.append(f"A morning star pattern formed ({tf_label}) — strong bullish reversal.")
        if summary["cdl_evening_star"]:
            sentences.append(f"An evening star pattern formed ({tf_label}) — strong bearish reversal.")
        if summary["cdl_three_white_soldiers"]:
            sentences.append(f"Three white soldiers detected ({tf_label}) — sustained bullish pressure.")
        if summary["cdl_three_black_crows"]:
            sentences.append(f"Three black crows detected ({tf_label}) — sustained bearish pressure.")

    if not sentences:
        sentences.append("Multiple technical indicators are aligned to generate this signal.")

    # Deduplicate while preserving order
    seen = set()
    unique_sentences = []
    for s in sentences:
        if s not in seen:
            seen.add(s)
            unique_sentences.append(s)

    return " ".join(unique_sentences[:8])  # Cap at 8 sentences for readability
