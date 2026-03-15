"""
Technical analysis feature engineering using pandas-ta.

Computes 25+ professional-grade indicators across any OHLCV DataFrame
and generates binary labels for supervised ML training.
"""

import logging

import numpy as np
import pandas as pd
import pandas_ta as ta

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
]

# Lookahead candles per timeframe for label generation
LOOKAHEAD_CANDLES = {
    "1h": 6,   # 6h horizon
    "4h": 6,   # 24h horizon
    "1d": 5,   # 5d horizon
}

LABEL_THRESHOLD = 0.02  # 2% move required to label as BUY or SELL


def compute_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Compute all TA features on a raw OHLCV DataFrame.

    Input columns required: open, high, low, close, volume
    Returns a new DataFrame with all feature columns appended.
    NaN rows from indicator warm-up are dropped.
    """
    df = df.copy()

    # --- RSI ---
    df["rsi_14"] = ta.rsi(df["close"], length=14)

    # --- MACD ---
    macd = ta.macd(df["close"], fast=12, slow=26, signal=9)
    if macd is not None:
        df["macd"] = macd.iloc[:, 0]       # MACD line
        df["macd_signal"] = macd.iloc[:, 2]  # Signal line
        df["macd_hist"] = macd.iloc[:, 1]   # Histogram
    else:
        df["macd"] = np.nan
        df["macd_signal"] = np.nan
        df["macd_hist"] = np.nan

    # --- Bollinger Bands ---
    bbands = ta.bbands(df["close"], length=20, std=2)
    if bbands is not None:
        df["bb_lower"] = bbands.iloc[:, 0]
        df["bb_middle"] = bbands.iloc[:, 1]
        df["bb_upper"] = bbands.iloc[:, 2]
        df["bb_bandwidth"] = bbands.iloc[:, 3]
        df["bb_pct"] = bbands.iloc[:, 4]
    else:
        for col in ["bb_lower", "bb_middle", "bb_upper", "bb_bandwidth", "bb_pct"]:
            df[col] = np.nan

    # --- EMAs ---
    df["ema_9"] = ta.ema(df["close"], length=9)
    df["ema_21"] = ta.ema(df["close"], length=21)
    df["ema_50"] = ta.ema(df["close"], length=50)
    df["ema_200"] = ta.ema(df["close"], length=200)

    # EMA crossover signals: +1 if fast > slow, -1 otherwise
    df["ema_cross_9_21"] = np.where(df["ema_9"] > df["ema_21"], 1.0, -1.0)
    df["ema_cross_21_50"] = np.where(df["ema_21"] > df["ema_50"], 1.0, -1.0)
    df["ema_cross_50_200"] = np.where(df["ema_50"] > df["ema_200"], 1.0, -1.0)

    # Price position relative to long-term EMA (normalised)
    df["close_vs_ema200"] = (df["close"] - df["ema_200"]) / df["ema_200"]

    # --- Stochastic RSI ---
    stoch_rsi = ta.stochrsi(df["close"], length=14)
    if stoch_rsi is not None:
        df["stoch_rsi_k"] = stoch_rsi.iloc[:, 0]
        df["stoch_rsi_d"] = stoch_rsi.iloc[:, 1]
    else:
        df["stoch_rsi_k"] = np.nan
        df["stoch_rsi_d"] = np.nan

    # --- ATR ---
    df["atr_14"] = ta.atr(df["high"], df["low"], df["close"], length=14)

    # --- OBV ---
    df["obv"] = ta.obv(df["close"], df["volume"])

    # --- Volume ratio (current vs 20-period average) ---
    vol_ma = df["volume"].rolling(20).mean()
    df["volume_ratio"] = df["volume"] / vol_ma.replace(0, np.nan)

    # --- Price momentum ---
    df["price_change_pct"] = df["close"].pct_change()
    df["high_low_range_pct"] = (df["high"] - df["low"]) / df["close"]

    # --- Candlestick patterns (pandas-ta returns 0/±100) ---
    hammer = ta.cdl_pattern(df["open"], df["high"], df["low"], df["close"], name="hammer")
    engulfing = ta.cdl_pattern(df["open"], df["high"], df["low"], df["close"], name="engulfing")
    doji = ta.cdl_pattern(df["open"], df["high"], df["low"], df["close"], name="doji")

    df["cdl_hammer"] = (hammer.iloc[:, 0] != 0).astype(float) if hammer is not None else 0.0
    df["cdl_engulfing"] = (engulfing.iloc[:, 0] != 0).astype(float) if engulfing is not None else 0.0
    df["cdl_doji"] = (doji.iloc[:, 0] != 0).astype(float) if doji is not None else 0.0

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

    Args:
        df: DataFrame with a 'close' column
        timeframe: "1h", "4h", or "1d"

    Returns:
        Series of integer labels aligned with df's index
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
    }
