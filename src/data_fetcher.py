"""
Data fetcher: OHLCV from Binance/Bybit public APIs + EUR/USD rate.
No API key required for public endpoints.

Fallback chain for OHLCV: Binance Global → Binance US → Bybit.
"""

import time
import logging
from datetime import datetime, timedelta, timezone

import requests
import pandas as pd

logger = logging.getLogger(__name__)

BINANCE_BASE = "https://api.binance.com"
BINANCE_US_BASE = "https://api.binance.us"
BYBIT_BASE = "https://api.bybit.com"
FRANKFURTER_URL = "https://api.frankfurter.app/latest"

INTERVAL_TO_SECONDS = {
    "1m": 60,
    "5m": 300,
    "15m": 900,
    "1h": 3600,
    "4h": 14400,
    "1d": 86400,
}

BYBIT_INTERVAL_MAP = {
    "1m": "1", "5m": "5", "15m": "15",
    "1h": "60", "4h": "240", "1d": "D",
}


def fetch_ohlcv(symbol: str, interval: str, limit: int = 500) -> pd.DataFrame:
    """
    Fetch the most recent OHLCV candles.

    Tries Binance Global → Binance US → Bybit.

    Args:
        symbol: e.g. "BTCUSDT"
        interval: "1h", "4h", "1d"
        limit: number of candles (max 1000)

    Returns:
        DataFrame with columns: timestamp, open, high, low, close, volume
    """
    limit = min(limit, 1000)
    klines = _fetch_klines_with_fallback(symbol, interval, limit=limit)
    return _parse_klines(klines)


def fetch_historical(symbol: str, interval: str, days: int = 730) -> pd.DataFrame:
    """
    Fetch historical OHLCV data by paginating klines endpoint.

    Tries Binance Global → Binance US → Bybit per page.
    Max 1000 candles per request; paginates to collect `days` worth of data.

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

    all_candles: list[pd.DataFrame] = []
    current_start = start_ms

    logger.info("Fetching %d days of %s %s data...", days, symbol, interval)

    while current_start < end_ms:
        klines = _fetch_klines_with_fallback(
            symbol, interval,
            startTime=current_start, endTime=end_ms, limit=1000,
        )

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
    Primary: derive from Binance BTCUSDT + BTCEUR prices.
    Fallback: Frankfurter (ECB) API.
    """
    try:
        usdt_price = _get_binance_price("BTCUSDT")
        eur_price = _get_binance_price("BTCEUR")
        if usdt_price and eur_price and usdt_price > 0:
            rate = eur_price / usdt_price
            logger.debug("EUR/USD rate from Binance: %.6f", rate)
            return rate
    except Exception as e:
        logger.warning("Binance EUR rate failed, using Frankfurter: %s", e)

    # Fallback to Frankfurter (ECB)
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
        # Last resort: use a hardcoded fallback (will be slightly off)
        logger.warning("Using hardcoded EUR/USD fallback: 0.92")
        return 0.92


def usd_to_eur(usd_price: float, eur_usd_rate: float) -> float:
    """Convert a USD price to EUR using the EUR/USD rate."""
    return usd_price * eur_usd_rate


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _get_binance_price(symbol: str) -> float:
    """Fetch current price for a symbol from Binance (Global then US)."""
    for base in [BINANCE_BASE, BINANCE_US_BASE]:
        try:
            url = f"{base}/api/v3/ticker/price"
            resp = requests.get(url, params={"symbol": symbol}, timeout=10)
            resp.raise_for_status()
            return float(resp.json()["price"])
        except requests.RequestException:
            continue
    raise RuntimeError(f"Could not fetch price for {symbol} from any Binance endpoint")


def _fetch_klines_binance(base_url: str, symbol: str, interval: str, **params) -> list:
    """Fetch raw kline arrays from a Binance-compatible API."""
    url = f"{base_url}/api/v3/klines"
    params.update({"symbol": symbol, "interval": interval})
    resp = requests.get(url, params=params, timeout=15)
    resp.raise_for_status()
    return resp.json()


def _fetch_klines_bybit(symbol: str, interval: str, **params) -> list:
    """Fetch raw kline arrays from Bybit v5 API, adapted to Binance format."""
    url = f"{BYBIT_BASE}/v5/market/kline"
    bybit_interval = BYBIT_INTERVAL_MAP.get(interval, interval)
    req_params = {
        "category": "spot",
        "symbol": symbol,
        "interval": bybit_interval,
        "limit": params.get("limit", 200),
    }
    if "startTime" in params:
        req_params["start"] = params["startTime"]
    if "endTime" in params:
        req_params["end"] = params["endTime"]

    resp = requests.get(url, params=req_params, timeout=15)
    resp.raise_for_status()
    data = resp.json()
    if data.get("retCode") != 0:
        raise RuntimeError(f"Bybit error: {data.get('retMsg')}")
    # Bybit returns newest-first; reverse to match Binance ascending order
    return list(reversed(data["result"]["list"]))


def _fetch_klines_with_fallback(symbol: str, interval: str, **params) -> list:
    """Try Binance Global → Binance US → Bybit for kline data."""
    sources = [
        ("Binance", lambda: _fetch_klines_binance(BINANCE_BASE, symbol, interval, **params)),
        ("Binance US", lambda: _fetch_klines_binance(BINANCE_US_BASE, symbol, interval, **params)),
        ("Bybit", lambda: _fetch_klines_bybit(symbol, interval, **params)),
    ]
    last_error = None
    for name, fetcher in sources:
        try:
            klines = fetcher()
            if klines:
                logger.debug("Klines fetched from %s", name)
                return klines
        except Exception as e:
            logger.warning("%s klines failed: %s", name, e)
            last_error = e
    raise RuntimeError(f"All kline sources failed. Last error: {last_error}")


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
