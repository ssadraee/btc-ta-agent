"""
Data fetcher: OHLCV from Binance public API + EUR/USD rate.
No API key required for public endpoints.
"""

import time
import logging
from datetime import datetime, timedelta, timezone

import requests
import pandas as pd

logger = logging.getLogger(__name__)

BINANCE_BASE = "https://api.binance.us"
FRANKFURTER_URL = "https://api.frankfurter.app/latest"

INTERVAL_TO_SECONDS = {
    "1m": 60,
    "5m": 300,
    "15m": 900,
    "1h": 3600,
    "4h": 14400,
    "1d": 86400,
}


def fetch_ohlcv(symbol: str, interval: str, limit: int = 500) -> pd.DataFrame:
    """
    Fetch the most recent OHLCV candles from Binance.

    Args:
        symbol: e.g. "BTCUSDT"
        interval: "1h", "4h", "1d"
        limit: number of candles (max 1000)

    Returns:
        DataFrame with columns: timestamp, open, high, low, close, volume
    """
    limit = min(limit, 1000)
    url = f"{BINANCE_BASE}/api/v3/klines"
    params = {"symbol": symbol, "interval": interval, "limit": limit}

    try:
        resp = requests.get(url, params=params, timeout=15)
        resp.raise_for_status()
        return _parse_klines(resp.json())
    except requests.RequestException as e:
        logger.error("Binance fetch_ohlcv failed: %s", e)
        raise


def fetch_historical(symbol: str, interval: str, days: int = 730) -> pd.DataFrame:
    """
    Fetch historical OHLCV data by paginating Binance klines endpoint.

    Binance returns max 1000 candles per request; this function paginates
    backward in time to collect `days` worth of data.

    Args:
        symbol: e.g. "BTCUSDT"
        interval: "1h", "4h", "1d"
        days: number of calendar days to fetch

    Returns:
        DataFrame sorted ascending by timestamp
    """
    interval_sec = INTERVAL_TO_SECONDS.get(interval)
    if interval_sec is None:
        raise ValueError(f"Unknown interval: {interval}")

    end_ms = int(datetime.now(tz=timezone.utc).timestamp() * 1000)
    start_ms = end_ms - int(days * 86400 * 1000)

    url = f"{BINANCE_BASE}/api/v3/klines"
    all_candles: list[pd.DataFrame] = []
    current_start = start_ms

    logger.info("Fetching %d days of %s %s data...", days, symbol, interval)

    while current_start < end_ms:
        params = {
            "symbol": symbol,
            "interval": interval,
            "startTime": current_start,
            "endTime": end_ms,
            "limit": 1000,
        }
        try:
            resp = requests.get(url, params=params, timeout=15)
            resp.raise_for_status()
            klines = resp.json()
        except requests.RequestException as e:
            logger.error("Binance pagination error: %s", e)
            raise

        if not klines:
            break

        chunk = _parse_klines(klines)
        all_candles.append(chunk)

        last_ts_ms = int(klines[-1][0])
        current_start = last_ts_ms + interval_sec * 1000

        # Be kind to the API
        time.sleep(0.1)

    if not all_candles:
        raise RuntimeError(f"No data returned for {symbol} {interval}")

    df = pd.concat(all_candles, ignore_index=True)
    df = df.drop_duplicates(subset="timestamp").sort_values("timestamp").reset_index(drop=True)
    logger.info("Fetched %d candles for %s %s", len(df), symbol, interval)
    return df


def get_eur_usd_rate() -> float:
    """
    Return EUR/USD rate (how many EUR per 1 USD).
    Primary: Frankfurter (ECB) API.
    Fallback: hardcoded rate.
    """
    try:
        resp = requests.get(
            FRANKFURTER_URL,
            params={"from": "USD", "to": "EUR"},
            timeout=10,
        )
        resp.raise_for_status()
        rate = resp.json()["rates"]["EUR"]
        logger.debug("EUR/USD rate from Frankfurter: %.6f", rate)
        return rate
    except Exception as e:
        logger.error("Frankfurter EUR rate failed: %s", e)
        logger.warning("Using hardcoded EUR/USD fallback: 0.92")
        return 0.92


def usd_to_eur(usd_price: float, eur_usd_rate: float) -> float:
    """Convert a USD price to EUR using the EUR/USD rate."""
    return usd_price * eur_usd_rate


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _parse_klines(klines: list) -> pd.DataFrame:
    """
    Parse raw Binance kline data into a clean DataFrame.

    Binance kline format (index):
    0: open_time, 1: open, 2: high, 3: low, 4: close, 5: volume,
    6: close_time, 7-11: misc fields
    """
    records = []
    for k in klines:
        records.append({
            "timestamp": pd.to_datetime(int(k[0]), unit="ms", utc=True),
            "open": float(k[1]),
            "high": float(k[2]),
            "low": float(k[3]),
            "close": float(k[4]),
            "volume": float(k[5]),
        })
    return pd.DataFrame(records)
