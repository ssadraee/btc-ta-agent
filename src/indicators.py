"""
Technical analysis feature engineering using the `ta` library.

Computes 40 professional-grade technical analysis features across any OHLCV DataFrame
and generates binary labels for supervised ML training.
"""

import logging

import numpy as np
import pandas as pd
from ta.momentum import RSIIndicator, StochRSIIndicator
from ta.trend import MACD, EMAIndicator
from ta.volatility import AverageTrueRange, BollingerBands
from ta.volume import OnBalanceVolumeIndicator

logger = logging.getLogger(__name__)

# Features used by the ML model (must be stable across train/predict)
FEATURE_COLUMNS = [
    "rsi_14",
    "macd",
    "macd_signal",
    "macd_hist",
    "bb_upper",
    "bb_middle",
    "bb_lower",
    "bb_pct",
    "bb_bandwidth",
    "ema_9",
    "ema_21",
    "ema_50",
    "ema_200",
    "ema_cross_9_21",
    "ema_cross_21_50",
    "ema_cross_50_200",
    "close_vs_ema200",
    "stoch_rsi_k",
    "stoch_rsi_d",
    "atr_14",
    "obv",
    "volume_ratio",
    "price_change_pct",
    "high_low_range_pct",
    "cdl_hammer",
    "cdl_engulfing",
    "cdl_doji",
    "cdl_shooting_star",
    "cdl_hanging_man",
    "cdl_inverted_hammer",
    "cdl_marubozu",
    "cdl_bearish_engulfing",
    "cdl_piercing_line",
    "cdl_dark_cloud",
    "cdl_tweezer_top",
    "cdl_tweezer_bottom",
    "cdl_morning_star",
    "cdl_evening_star",
    "cdl_three_white_soldiers",
    "cdl_three_black_crows",
]

# Lookahead candles per timeframe for label generation
LOOKAHEAD_CANDLES = {
    "1h": 12,  # 12h horizon (was 6h)
    "4h": 6,   # 24h horizon
    "1d": 5,   # 5d horizon
}

LABEL_THRESHOLD = 0.03  # 3% move required to label as BUY or SELL (was 2%)


def compute_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Compute all TA features on a raw OHLCV DataFrame.

    Input columns required: open, high, low, close, volume
    Returns a new DataFrame with all feature columns appended.
    NaN rows from indicator warm-up are dropped.
    """
    df = df.copy()

    close = df["close"]
    high = df["high"]
    low = df["low"]
    volume = df["volume"]

    # --- RSI ---
    df["rsi_14"] = RSIIndicator(close=close, window=14).rsi()

    # --- MACD ---
    macd_obj = MACD(close=close, window_slow=26, window_fast=12, window_sign=9)
    df["macd"] = macd_obj.macd()
    df["macd_signal"] = macd_obj.macd_signal()
    df["macd_hist"] = macd_obj.macd_diff()

    # --- Bollinger Bands ---
    bb = BollingerBands(close=close, window=20, window_dev=2)
    df["bb_upper"] = bb.bollinger_hband()
    df["bb_middle"] = bb.bollinger_mavg()
    df["bb_lower"] = bb.bollinger_lband()
    df["bb_pct"] = bb.bollinger_pband()
    df["bb_bandwidth"] = bb.bollinger_wband()

    # --- EMAs ---
    df["ema_9"] = EMAIndicator(close=close, window=9).ema_indicator()
    df["ema_21"] = EMAIndicator(close=close, window=21).ema_indicator()
    df["ema_50"] = EMAIndicator(close=close, window=50).ema_indicator()
    df["ema_200"] = EMAIndicator(close=close, window=200).ema_indicator()

    # EMA crossover signals: +1 if fast > slow, -1 otherwise
    df["ema_cross_9_21"] = np.where(df["ema_9"] > df["ema_21"], 1.0, -1.0)
    df["ema_cross_21_50"] = np.where(df["ema_21"] > df["ema_50"], 1.0, -1.0)
    df["ema_cross_50_200"] = np.where(df["ema_50"] > df["ema_200"], 1.0, -1.0)

    # Price position relative to long-term EMA (normalised)
    df["close_vs_ema200"] = (close - df["ema_200"]) / df["ema_200"]

    # --- Stochastic RSI ---
    stoch = StochRSIIndicator(close=close, window=14, smooth1=3, smooth2=3)
    df["stoch_rsi_k"] = stoch.stochrsi_k()
    df["stoch_rsi_d"] = stoch.stochrsi_d()

    # --- ATR ---
    df["atr_14"] = AverageTrueRange(high=high, low=low, close=close, window=14).average_true_range()

    # --- OBV ---
    df["obv"] = OnBalanceVolumeIndicator(close=close, volume=volume).on_balance_volume()

    # --- Volume ratio (current vs 20-period average) ---
    vol_ma = volume.rolling(20).mean()
    df["volume_ratio"] = volume / vol_ma.replace(0, np.nan)

    # --- Price momentum ---
    df["price_change_pct"] = close.pct_change()
    df["high_low_range_pct"] = (high - low) / close

    # --- Candlestick patterns (manual OHLC math) ---
    df["cdl_doji"] = _cdl_doji(df)
    df["cdl_hammer"] = _cdl_hammer(df)
    df["cdl_engulfing"] = _cdl_engulfing(df)
    df["cdl_shooting_star"] = _cdl_shooting_star(df)
    df["cdl_hanging_man"] = _cdl_hanging_man(df)
    df["cdl_inverted_hammer"] = _cdl_inverted_hammer(df)
    df["cdl_marubozu"] = _cdl_marubozu(df)
    df["cdl_bearish_engulfing"] = _cdl_bearish_engulfing(df)
    df["cdl_piercing_line"] = _cdl_piercing_line(df)
    df["cdl_dark_cloud"] = _cdl_dark_cloud(df)
    df["cdl_tweezer_top"] = _cdl_tweezer_top(df)
    df["cdl_tweezer_bottom"] = _cdl_tweezer_bottom(df)
    df["cdl_morning_star"] = _cdl_morning_star(df)
    df["cdl_evening_star"] = _cdl_evening_star(df)
    df["cdl_three_white_soldiers"] = _cdl_three_white_soldiers(df)
    df["cdl_three_black_crows"] = _cdl_three_black_crows(df)

    # Drop rows where any required feature is NaN (indicator warm-up period)
    df = df.dropna(subset=FEATURE_COLUMNS).reset_index(drop=True)

    return df


def generate_labels(df: pd.DataFrame, timeframe: str) -> pd.Series:
    """
    Generate supervised learning labels based on future price movement.

    Label encoding:
        1  = BUY  (price rises > threshold within lookahead window)
        -1 = SELL (price falls > threshold within lookahead window)
        0  = HOLD (price stays within ±threshold)

    The last `lookahead` rows will be NaN (no future data available).
    """
    lookahead = LOOKAHEAD_CANDLES.get(timeframe, 6)
    future_close = df["close"].shift(-lookahead)
    pct_change = (future_close - df["close"]) / df["close"]

    labels = pd.Series(0, index=df.index, dtype=int)
    labels[pct_change > LABEL_THRESHOLD] = 1
    labels[pct_change < -LABEL_THRESHOLD] = -1
    labels[future_close.isna()] = pd.NA  # type: ignore[call-overload]

    return labels


def get_latest_indicator_summary(df: pd.DataFrame) -> dict:
    """
    Return a human-readable summary of the latest candle's indicator values.
    Used by the explanation builder in signals.py.
    """
    row = df.iloc[-1]
    return {
        "rsi": round(row.get("rsi_14", float("nan")), 1),
        "macd_hist": round(row.get("macd_hist", float("nan")), 2),
        "bb_pct": round(row.get("bb_pct", float("nan")), 3),
        "ema_cross_9_21": int(row.get("ema_cross_9_21", 0)),
        "ema_cross_21_50": int(row.get("ema_cross_21_50", 0)),
        "ema_cross_50_200": int(row.get("ema_cross_50_200", 0)),
        "close_vs_ema200_pct": round(row.get("close_vs_ema200", 0) * 100, 2),
        "stoch_rsi_k": round(row.get("stoch_rsi_k", float("nan")), 1),
        "volume_ratio": round(row.get("volume_ratio", float("nan")), 2),
        "atr_14": round(row.get("atr_14", float("nan")), 2),
        "cdl_hammer": bool(row.get("cdl_hammer", 0)),
        "cdl_engulfing": bool(row.get("cdl_engulfing", 0)),
        "cdl_doji": bool(row.get("cdl_doji", 0)),
        "cdl_shooting_star": bool(row.get("cdl_shooting_star", 0)),
        "cdl_hanging_man": bool(row.get("cdl_hanging_man", 0)),
        "cdl_inverted_hammer": bool(row.get("cdl_inverted_hammer", 0)),
        "cdl_marubozu": bool(row.get("cdl_marubozu", 0)),
        "cdl_bearish_engulfing": bool(row.get("cdl_bearish_engulfing", 0)),
        "cdl_piercing_line": bool(row.get("cdl_piercing_line", 0)),
        "cdl_dark_cloud": bool(row.get("cdl_dark_cloud", 0)),
        "cdl_tweezer_top": bool(row.get("cdl_tweezer_top", 0)),
        "cdl_tweezer_bottom": bool(row.get("cdl_tweezer_bottom", 0)),
        "cdl_morning_star": bool(row.get("cdl_morning_star", 0)),
        "cdl_evening_star": bool(row.get("cdl_evening_star", 0)),
        "cdl_three_white_soldiers": bool(row.get("cdl_three_white_soldiers", 0)),
        "cdl_three_black_crows": bool(row.get("cdl_three_black_crows", 0)),
    }


# ---------------------------------------------------------------------------
# Candlestick pattern helpers (manual OHLC math — no external dependency)
# ---------------------------------------------------------------------------

def _cdl_doji(df: pd.DataFrame) -> pd.Series:
    """Doji: body is tiny relative to the total candle range."""
    body = (df["close"] - df["open"]).abs()
    range_ = df["high"] - df["low"]
    return (body / range_.replace(0, np.nan) < 0.1).astype(float).fillna(0)


def _cdl_hammer(df: pd.DataFrame) -> pd.Series:
    """
    Hammer: small body at the top, lower shadow at least 2× the body,
    upper shadow smaller than the body. Bullish reversal signal.
    """
    body = (df["close"] - df["open"]).abs()
    body_top = df[["open", "close"]].max(axis=1)
    body_bottom = df[["open", "close"]].min(axis=1)
    lower_shadow = body_bottom - df["low"]
    upper_shadow = df["high"] - body_top
    return (
        (lower_shadow > 2 * body) & (upper_shadow < body)
    ).astype(float)


def _cdl_engulfing(df: pd.DataFrame) -> pd.Series:
    """
    Bullish engulfing: current bullish candle fully contains the previous
    bearish candle's body. Strong reversal indicator.
    """
    prev_open = df["open"].shift(1)
    prev_close = df["close"].shift(1)
    bullish_current = df["close"] > df["open"]
    bearish_prev = prev_close < prev_open
    engulfs = (df["open"] <= prev_close) & (df["close"] >= prev_open)
    return (bullish_current & bearish_prev & engulfs).astype(float).fillna(0)


def _cdl_shooting_star(df: pd.DataFrame) -> pd.Series:
    """
    Shooting Star: small body at the bottom, upper shadow at least 2× the body,
    lower shadow smaller than the body. Bearish reversal signal.
    """
    body = (df["close"] - df["open"]).abs()
    body_top = df[["open", "close"]].max(axis=1)
    body_bottom = df[["open", "close"]].min(axis=1)
    upper_shadow = df["high"] - body_top
    lower_shadow = body_bottom - df["low"]
    return (
        (upper_shadow > 2 * body) & (lower_shadow < body)
    ).astype(float)


def _cdl_hanging_man(df: pd.DataFrame) -> pd.Series:
    """
    Hanging Man: same shape as Hammer (long lower shadow, small upper shadow).
    The model learns bearish context from surrounding indicators.
    """
    body = (df["close"] - df["open"]).abs()
    body_top = df[["open", "close"]].max(axis=1)
    body_bottom = df[["open", "close"]].min(axis=1)
    lower_shadow = body_bottom - df["low"]
    upper_shadow = df["high"] - body_top
    return (
        (lower_shadow > 2 * body) & (upper_shadow < body)
    ).astype(float)


def _cdl_inverted_hammer(df: pd.DataFrame) -> pd.Series:
    """
    Inverted Hammer: same shape as Shooting Star (long upper shadow, small lower
    shadow). Bullish reversal when appearing after a downtrend.
    """
    body = (df["close"] - df["open"]).abs()
    body_top = df[["open", "close"]].max(axis=1)
    body_bottom = df[["open", "close"]].min(axis=1)
    upper_shadow = df["high"] - body_top
    lower_shadow = body_bottom - df["low"]
    return (
        (upper_shadow > 2 * body) & (lower_shadow < body)
    ).astype(float)


def _cdl_marubozu(df: pd.DataFrame) -> pd.Series:
    """Marubozu: body covers ≥95% of the total candle range. Strong momentum."""
    body = (df["close"] - df["open"]).abs()
    range_ = df["high"] - df["low"]
    return (body / range_.replace(0, np.nan) >= 0.95).astype(float).fillna(0)


def _cdl_bearish_engulfing(df: pd.DataFrame) -> pd.Series:
    """
    Bearish engulfing: current bearish candle fully contains the previous
    bullish candle's body. Bearish reversal indicator.
    """
    prev_open = df["open"].shift(1)
    prev_close = df["close"].shift(1)
    bearish_current = df["close"] < df["open"]
    bullish_prev = prev_close > prev_open
    engulfs = (df["open"] >= prev_close) & (df["close"] <= prev_open)
    return (bearish_current & bullish_prev & engulfs).astype(float).fillna(0)


def _cdl_piercing_line(df: pd.DataFrame) -> pd.Series:
    """
    Piercing Line: bullish 2-candle pattern. Previous candle is bearish,
    current opens below prior low and closes above the midpoint of the prior body.
    """
    prev_open = df["open"].shift(1)
    prev_close = df["close"].shift(1)
    prev_mid = (prev_open + prev_close) / 2
    bearish_prev = prev_close < prev_open
    bullish_current = df["close"] > df["open"]
    opens_below = df["open"] < prev_close
    closes_above_mid = df["close"] > prev_mid
    closes_below_prev_open = df["close"] < prev_open
    return (
        bearish_prev & bullish_current & opens_below
        & closes_above_mid & closes_below_prev_open
    ).astype(float).fillna(0)


def _cdl_dark_cloud(df: pd.DataFrame) -> pd.Series:
    """
    Dark Cloud Cover: bearish 2-candle pattern. Previous candle is bullish,
    current opens above prior high and closes below the midpoint of the prior body.
    """
    prev_open = df["open"].shift(1)
    prev_close = df["close"].shift(1)
    prev_mid = (prev_open + prev_close) / 2
    bullish_prev = prev_close > prev_open
    bearish_current = df["close"] < df["open"]
    opens_above = df["open"] > prev_close
    closes_below_mid = df["close"] < prev_mid
    closes_above_prev_open = df["close"] > prev_open
    return (
        bullish_prev & bearish_current & opens_above
        & closes_below_mid & closes_above_prev_open
    ).astype(float).fillna(0)


def _cdl_tweezer_top(df: pd.DataFrame) -> pd.Series:
    """
    Tweezer Top: two consecutive candles with nearly identical highs.
    Tolerance: within 0.1% of price.
    """
    prev_high = df["high"].shift(1)
    tolerance = df["high"] * 0.001
    return (
        (df["high"] - prev_high).abs() <= tolerance
    ).astype(float).fillna(0)


def _cdl_tweezer_bottom(df: pd.DataFrame) -> pd.Series:
    """
    Tweezer Bottom: two consecutive candles with nearly identical lows.
    Tolerance: within 0.1% of price.
    """
    prev_low = df["low"].shift(1)
    tolerance = df["low"] * 0.001
    return (
        (df["low"] - prev_low).abs() <= tolerance
    ).astype(float).fillna(0)


def _cdl_morning_star(df: pd.DataFrame) -> pd.Series:
    """
    Morning Star: 3-candle bullish reversal.
    1st: bearish candle, 2nd: small-body candle, 3rd: bullish candle closing
    above the midpoint of the 1st candle's body.
    """
    o1, c1 = df["open"].shift(2), df["close"].shift(2)
    o2, c2 = df["open"].shift(1), df["close"].shift(1)
    o3, c3 = df["open"], df["close"]

    bearish_1 = c1 < o1
    small_body_2 = (c2 - o2).abs() < (o1 - c1).abs() * 0.3
    bullish_3 = c3 > o3
    closes_above_mid = c3 > (o1 + c1) / 2
    return (bearish_1 & small_body_2 & bullish_3 & closes_above_mid).astype(float).fillna(0)


def _cdl_evening_star(df: pd.DataFrame) -> pd.Series:
    """
    Evening Star: 3-candle bearish reversal.
    1st: bullish candle, 2nd: small-body candle, 3rd: bearish candle closing
    below the midpoint of the 1st candle's body.
    """
    o1, c1 = df["open"].shift(2), df["close"].shift(2)
    o2, c2 = df["open"].shift(1), df["close"].shift(1)
    o3, c3 = df["open"], df["close"]

    bullish_1 = c1 > o1
    small_body_2 = (c2 - o2).abs() < (c1 - o1).abs() * 0.3
    bearish_3 = c3 < o3
    closes_below_mid = c3 < (o1 + c1) / 2
    return (bullish_1 & small_body_2 & bearish_3 & closes_below_mid).astype(float).fillna(0)


def _cdl_three_white_soldiers(df: pd.DataFrame) -> pd.Series:
    """
    Three White Soldiers: three consecutive bullish candles, each closing
    higher than the previous, each opening within the prior candle's body.
    """
    o1, c1 = df["open"].shift(2), df["close"].shift(2)
    o2, c2 = df["open"].shift(1), df["close"].shift(1)
    o3, c3 = df["open"], df["close"]

    bull_1 = c1 > o1
    bull_2 = c2 > o2
    bull_3 = c3 > o3
    higher_closes = (c2 > c1) & (c3 > c2)
    opens_in_body_2 = (o2 >= o1) & (o2 <= c1)
    opens_in_body_3 = (o3 >= o2) & (o3 <= c2)
    return (
        bull_1 & bull_2 & bull_3 & higher_closes
        & opens_in_body_2 & opens_in_body_3
    ).astype(float).fillna(0)


def _cdl_three_black_crows(df: pd.DataFrame) -> pd.Series:
    """
    Three Black Crows: three consecutive bearish candles, each closing
    lower than the previous, each opening within the prior candle's body.
    """
    o1, c1 = df["open"].shift(2), df["close"].shift(2)
    o2, c2 = df["open"].shift(1), df["close"].shift(1)
    o3, c3 = df["open"], df["close"]

    bear_1 = c1 < o1
    bear_2 = c2 < o2
    bear_3 = c3 < o3
    lower_closes = (c2 < c1) & (c3 < c2)
    opens_in_body_2 = (o2 <= o1) & (o2 >= c1)
    opens_in_body_3 = (o3 <= o2) & (o3 >= c2)
    return (
        bear_1 & bear_2 & bear_3 & lower_closes
        & opens_in_body_2 & opens_in_body_3
    ).astype(float).fillna(0)
