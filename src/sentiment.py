"""
BTC crowd-sentiment analysis for trading signals.

Data source priority (first that succeeds wins):
  1. Bitquery GraphQL API  — cloud-accessible Polymarket data; requires
     BITQUERY_API_KEY env var (free account at https://bitquery.io).
  2. Polymarket gamma REST API — works locally but IP geo-blocked on cloud
     providers (GitHub Actions, AWS, GCP) by Cloudflare.
  3. Binance Futures sentiment — funding rate + top-trader long/short ratio.
     No API key required; same Binance/Bybit endpoints the project already
     uses for OHLCV data.  Always available from cloud environments.

The Polymarket path derives a signal from the implied median BTC price across
"Will BTC be above $X?" markets grouped by time horizon.  The Binance Futures
path derives a signal directly from derivatives market positioning.
"""

import json
import logging
import os
import re
from datetime import datetime, timezone

import requests

logger = logging.getLogger(__name__)

POLYMARKET_API = "https://gamma-api.polymarket.com/markets"
BITQUERY_API = "https://streaming.bitquery.io/graphql"

# Binance Futures endpoints (no auth required; Bybit used as fallback)
_BINANCE_FUTURES_PREMIUM = "https://fapi.binance.com/fapi/v1/premiumIndex"
_BINANCE_FUTURES_LS_RATIO = "https://fapi.binance.com/futures/data/topLongShortPositionRatio"
_BYBIT_TICKERS = "https://api.bybit.com/v5/market/tickers"

# GraphQL query for active BTC prediction markets on Polymarket via Bitquery.
# Bitquery indexes Polymarket smart-contract events on Polygon and exposes
# them as a structured dataset. LastPrice of the YES/NO outcome tokens equals
# the current market probability (0–1).
_BITQUERY_QUERY = """
query PolymarketBTCMarkets {
  Polymarket {
    Markets(
      where: {
        Market: {
          Active: { is: true }
          Question: { includesAnyOf: ["Bitcoin", "BTC"] }
        }
      }
      limit: { count: 50 }
    ) {
      Market {
        Question
        ConditionID
        EndTime
        Volume
        OutcomeTokens {
          Outcome
          LastPrice
        }
      }
    }
  }
}
"""

# Horizon classification thresholds (hours from now)
_HORIZON_HOURS = {
    "short":  24,
    "medium": 7 * 24,
    "long":   60 * 24,
    # "macro" = everything beyond 60 days
}

# Weights used to combine per-horizon signals into a single Polymarket signal.
# Mirrors the TA timeframe weights: short~1h, medium~4h, long~1d, macro extra.
_HORIZON_WEIGHTS = {
    "short":  0.25,
    "medium": 0.35,
    "long":   0.40,
    "macro":  0.30,
}

# Skip markets whose price target is more than this fraction away from
# current price (they carry little directional information).
_MAX_PRICE_DISTANCE = 0.60  # 60%

# Regex to extract dollar amounts like "$85,000", "$100k", "$1.2M" from text
_PRICE_RE = re.compile(
    r"\$\s*([\d,]+(?:\.\d+)?)\s*([kKmMbB]?)",
    re.IGNORECASE,
)


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def fetch_polymarket_sentiment(
    current_price_usd: float,
    bitquery_key: str | None = None,
) -> dict:
    """
    Derive a BTC crowd-sentiment signal using the best available data source.

    Priority:
      1. Polymarket via Bitquery (if BITQUERY_API_KEY is set)
      2. Polymarket gamma REST API (works locally; geo-blocked on cloud)
      3. Binance Futures funding rate + long/short ratio (always available)

    Returns:
        dict with keys:
          signal       float  [-1, 1]        Aggregated directional signal
          confidence   float  [0, 0.75]      Signal reliability
          horizons     dict   Per-horizon (signal, confidence, n_markets)
          summary      str    Human-readable one-liner
          market_count int    Total usable Polymarket markets (0 for futures path)
    """
    if bitquery_key is None:
        bitquery_key = os.getenv("BITQUERY_API_KEY")

    # --- Path 1 & 2: Polymarket ---
    try:
        markets = _fetch_markets(bitquery_key)
        if markets:
            result = _compute_from_markets(markets, current_price_usd)
            if result["market_count"] > 0:
                return result
    except Exception as e:
        logger.warning("Polymarket fetch failed: %s", e)

    # --- Path 3: Binance Futures (no API key, cloud-accessible) ---
    logger.info("Falling back to Binance Futures sentiment")
    try:
        return _fetch_futures_sentiment()
    except Exception as e:
        logger.warning("Futures sentiment fetch failed: %s", e)

    return _empty_result("Sentiment data unavailable")


def _compute_from_markets(markets: list[dict], current_price_usd: float) -> dict:
    """Derive a sentiment signal from a list of Polymarket market dicts."""
    # Parse each market into (horizon, target_price, yes_prob, volume)
    parsed: list[tuple[str, float, float, float]] = []
    for m in markets:
        entry = _parse_market(m, current_price_usd)
        if entry is not None:
            parsed.append(entry)

    if not parsed:
        return _empty_result("No relevant BTC price markets found on Polymarket")

    # Group by horizon
    by_horizon: dict[str, list[tuple[float, float, float]]] = {
        h: [] for h in ("short", "medium", "long", "macro")
    }
    for horizon, target_price, yes_prob, volume in parsed:
        by_horizon[horizon].append((target_price, yes_prob, volume))

    # Compute per-horizon signals
    horizon_results: dict[str, tuple[float, float, int]] = {}
    weighted_sum = 0.0
    total_weight = 0.0

    for horizon, pts in by_horizon.items():
        if not pts:
            continue
        sig, conf = _compute_horizon_signal(pts, current_price_usd)
        n = len(pts)
        horizon_results[horizon] = (sig, conf, n)

        w = _HORIZON_WEIGHTS[horizon]
        if conf > 0:
            weighted_sum += sig * conf * w
            total_weight += w

    if total_weight == 0:
        return _empty_result("Insufficient Polymarket data to compute signal")

    combined_signal = weighted_sum / total_weight
    combined_conf = min(0.75, abs(combined_signal) * 0.9 + 0.05 * len(horizon_results))

    market_count = len(parsed)
    direction = (
        "bullish" if combined_signal > 0.05
        else "bearish" if combined_signal < -0.05
        else "neutral"
    )
    summary = (
        f"Polymarket: {direction} (signal={combined_signal:+.2f}, "
        f"conf={combined_conf:.0%}, {market_count} markets)"
    )
    logger.info(summary)

    return {
        "signal": round(combined_signal, 4),
        "confidence": round(combined_conf, 4),
        "horizons": horizon_results,
        "summary": summary,
        "market_count": market_count,
    }


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _fetch_markets_bitquery(api_key: str) -> list[dict]:
    """
    Fetch active BTC prediction markets from Polymarket via Bitquery GraphQL.

    Bitquery indexes Polymarket data from the Polygon blockchain and is
    accessible from cloud environments (no geo-block). The response is
    normalised to the same dict shape used by the gamma REST API so all
    downstream parsing logic is unchanged.

    Raises requests.HTTPError or ValueError on failure so the caller can fall
    back to the gamma API.
    """
    resp = requests.post(
        BITQUERY_API,
        json={"query": _BITQUERY_QUERY},
        headers={
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
        },
        timeout=20,
    )
    resp.raise_for_status()
    body = resp.json()

    if "errors" in body:
        raise ValueError(f"Bitquery GraphQL errors: {body['errors']}")

    markets_raw = (
        body.get("data", {})
            .get("Polymarket", {})
            .get("Markets", [])
    )

    if not markets_raw:
        logger.warning("Bitquery returned 0 active BTC Polymarket markets")
        return []

    markets: list[dict] = []
    for entry in markets_raw:
        try:
            mkt = entry["Market"]
            tokens = mkt.get("OutcomeTokens", [])
            outcomes = [t["Outcome"] for t in tokens]
            prices = [str(t["LastPrice"]) for t in tokens]
            markets.append({
                "id": mkt.get("ConditionID", ""),
                "question": mkt.get("Question", ""),
                "outcomes": json.dumps(outcomes),
                "outcomePrices": json.dumps(prices),
                "endDateIso": mkt.get("EndTime", ""),
                "volume": mkt.get("Volume", 0),
            })
        except (KeyError, TypeError) as exc:
            logger.debug("Skipping malformed Bitquery market entry: %s", exc)

    logger.info("Bitquery: fetched %d BTC Polymarket markets", len(markets))
    return markets


def _fetch_markets(bitquery_key: str | None = None) -> list[dict]:
    """
    Fetch active BTC prediction markets.

    Tries Bitquery first when an API key is available (works from cloud),
    then falls back to the Polymarket gamma REST API (works locally).
    """
    if bitquery_key:
        try:
            markets = _fetch_markets_bitquery(bitquery_key)
            if markets:
                return markets
            # Zero results from Bitquery — fall through to gamma API
            logger.warning("Bitquery returned no markets, falling back to gamma API")
        except Exception as exc:
            logger.warning(
                "Bitquery fetch failed (%s) — falling back to Polymarket gamma API", exc
            )

    # Polymarket gamma REST API (geo-blocked on cloud, fine locally)
    params = {
        "active": "true",
        "closed": "false",
        "search": "bitcoin",
        "limit": "50",
    }
    resp = requests.get(POLYMARKET_API, params=params, timeout=15)
    resp.raise_for_status()
    markets = resp.json()

    # If few results, also try "btc" search
    if len(markets) < 5:
        params["search"] = "btc"
        try:
            resp2 = requests.get(POLYMARKET_API, params=params, timeout=15)
            resp2.raise_for_status()
            extra = resp2.json()
            ids = {m.get("id") for m in markets}
            markets += [m for m in extra if m.get("id") not in ids]
        except Exception:
            pass

    return markets


def _parse_market(
    market: dict,
    current_price_usd: float,
) -> tuple[str, float, float, float] | None:
    """
    Parse a single Polymarket market record.

    Returns (horizon, target_price, yes_prob, volume) or None if not usable.
    """
    # Only handle YES/NO binary markets
    outcomes = _parse_json_field(market.get("outcomes", "[]"))
    prices_raw = _parse_json_field(market.get("outcomePrices", "[]"))

    if len(outcomes) != 2 or len(prices_raw) != 2:
        return None

    # Map outcome names to indices
    outcome_names = [str(o).lower() for o in outcomes]
    try:
        prices = [float(p) for p in prices_raw]
    except (ValueError, TypeError):
        return None

    # Identify YES index
    yes_idx = None
    for i, name in enumerate(outcome_names):
        if name in ("yes", "higher", "above", "up"):
            yes_idx = i
            break
    if yes_idx is None:
        yes_idx = 0  # Default: first outcome is YES

    yes_prob = prices[yes_idx]
    if not (0.0 <= yes_prob <= 1.0):
        return None

    # Extract target price from question text
    question = market.get("question", "")
    target_price = _extract_price_target(question)
    if target_price is None or target_price <= 0:
        return None

    # Skip targets too far from current price (low information content)
    distance = abs(target_price - current_price_usd) / current_price_usd
    if distance > _MAX_PRICE_DISTANCE:
        return None

    # Classify time horizon from end date
    end_date_str = market.get("endDateIso") or market.get("endDate", "")
    horizon = _classify_horizon(end_date_str)

    volume = float(market.get("volume", 0) or 0)

    return horizon, target_price, yes_prob, volume


def _compute_horizon_signal(
    pts: list[tuple[float, float, float]],
    current_price_usd: float,
) -> tuple[float, float]:
    """
    Derive a directional signal for a single time horizon.

    Algorithm:
      1. Sort markets by target price (ascending)
      2. Find the implied median: the target price where yes_prob = 0.50
         via linear interpolation across adjacent markets
      3. signal = clipped((implied_median − current) / current × 10)
      4. confidence = f(number of markets)

    Falls back to a volume-weighted yes_prob approach if fewer than 2 markets
    or if the probability series never crosses 0.50.

    Returns (signal [-1, 1], confidence [0, 0.75])
    """
    if not pts:
        return 0.0, 0.0

    # Sort by target price ascending
    sorted_pts = sorted(pts, key=lambda x: x[0])
    target_prices = [p[0] for p in sorted_pts]
    yes_probs = [p[1] for p in sorted_pts]
    volumes = [p[2] for p in sorted_pts]

    implied_median = _interpolate_median(target_prices, yes_probs)

    if implied_median is not None:
        # Primary: use implied median
        expected_move = (implied_median - current_price_usd) / current_price_usd
        signal = max(-1.0, min(1.0, expected_move * 10))
    else:
        # Fallback: volume-weighted contribution of (yes_prob − 0.5) * 2
        total_vol = sum(volumes) or 1.0
        weighted = sum(
            (p - 0.5) * 2 * (v / total_vol)
            for p, v in zip(yes_probs, volumes)
        )
        signal = max(-1.0, min(1.0, weighted))

    n = len(pts)
    # Confidence grows with market count; cap at 0.70
    confidence = min(0.70, 0.25 + 0.09 * min(n, 5))

    return signal, confidence


def _interpolate_median(
    target_prices: list[float],
    yes_probs: list[float],
) -> float | None:
    """
    Linear interpolation to find the price where yes_prob = 0.50.

    Note: yes_prob for "above $X" should be DECREASING as X increases
    (harder to reach higher prices). If the series is increasing we flip it.

    Returns None if no crossover is found.
    """
    if len(target_prices) < 2:
        return None

    # Ensure descending order of probabilities (correct for "above $X" markets)
    if yes_probs[0] < yes_probs[-1]:
        target_prices = list(reversed(target_prices))
        yes_probs = list(reversed(yes_probs))

    for i in range(len(yes_probs) - 1):
        p0, p1 = yes_probs[i], yes_probs[i + 1]
        x0, x1 = target_prices[i], target_prices[i + 1]
        # Check if 0.5 lies between p0 and p1
        if (p0 >= 0.5 >= p1) or (p0 <= 0.5 <= p1):
            if p0 == p1:
                return (x0 + x1) / 2
            t = (0.5 - p0) / (p1 - p0)
            return x0 + t * (x1 - x0)

    return None


def _classify_horizon(end_date_str: str) -> str:
    """Classify a market's end date into a horizon bucket."""
    if not end_date_str:
        return "long"  # Default to long if unknown

    try:
        end_date = datetime.fromisoformat(end_date_str.replace("Z", "+00:00"))
        if end_date.tzinfo is None:
            end_date = end_date.replace(tzinfo=timezone.utc)
    except (ValueError, AttributeError):
        return "long"

    now = datetime.now(tz=timezone.utc)
    hours_until = (end_date - now).total_seconds() / 3600

    if hours_until <= _HORIZON_HOURS["short"]:
        return "short"
    elif hours_until <= _HORIZON_HOURS["medium"]:
        return "medium"
    elif hours_until <= _HORIZON_HOURS["long"]:
        return "long"
    else:
        return "macro"


def _extract_price_target(question: str) -> float | None:
    """
    Extract a BTC dollar price target from a Polymarket question string.

    Handles formats like: $85,000  $100k  $1.2M  $85000
    """
    matches = _PRICE_RE.findall(question)
    if not matches:
        return None

    results = []
    for digits_str, suffix in matches:
        try:
            value = float(digits_str.replace(",", ""))
        except ValueError:
            continue
        suffix = suffix.lower()
        if suffix == "k":
            value *= 1_000
        elif suffix == "m":
            value *= 1_000_000
        elif suffix == "b":
            value *= 1_000_000_000
        results.append(value)

    if not results:
        return None

    # Return the largest price found (most likely to be the BTC target, not a
    # smaller incidental dollar amount like "$5 reward")
    return max(results)


def _parse_json_field(val) -> list:
    """Parse a field that may be a JSON-encoded string or already a list."""
    if isinstance(val, list):
        return val
    if isinstance(val, str):
        try:
            parsed = json.loads(val)
            return parsed if isinstance(parsed, list) else []
        except (json.JSONDecodeError, ValueError):
            return []
    return []


def _fetch_futures_sentiment() -> dict:
    """
    Derive a BTC sentiment signal from Binance Futures market data.

    Uses two complementary signals:
      - Funding rate: positive → longs paying shorts (bullish/over-leveraged),
        negative → shorts paying longs (bearish).
      - Top-trader long/short position ratio: > 1 = more longs, < 1 = more shorts.

    Binance is used as primary, Bybit as fallback (same as OHLCV data_fetcher).
    No API key required.

    Signal normalisation:
      funding_signal  = clamp(rate * 3000, -1, 1)
        typical rate ≈ ±0.0001; 0.0003 → ±0.9 (strong); 0.001 → clamped ±1
      ls_signal       = clamp(ratio - 1.0, -1, 1)
        ratio 1.5 → +0.5 (moderately bullish); 0.5 → -0.5 (moderately bearish)
    """
    funding_rate: float | None = None
    ls_ratio: float | None = None

    # --- Funding rate (Binance → Bybit fallback) ---
    try:
        resp = requests.get(
            _BINANCE_FUTURES_PREMIUM,
            params={"symbol": "BTCUSDT"},
            timeout=10,
        )
        resp.raise_for_status()
        funding_rate = float(resp.json()["lastFundingRate"])
        logger.debug("Binance funding rate: %s", funding_rate)
    except Exception as e:
        logger.debug("Binance funding rate failed (%s), trying Bybit", e)
        try:
            resp = requests.get(
                _BYBIT_TICKERS,
                params={"category": "linear", "symbol": "BTCUSDT"},
                timeout=10,
            )
            resp.raise_for_status()
            data = resp.json()
            funding_rate = float(data["result"]["list"][0]["fundingRate"])
            logger.debug("Bybit funding rate: %s", funding_rate)
        except Exception as e2:
            logger.warning("Funding rate fetch failed: %s", e2)

    # --- Long/short ratio (Binance only; Bybit endpoint requires auth) ---
    try:
        resp = requests.get(
            _BINANCE_FUTURES_LS_RATIO,
            params={"symbol": "BTCUSDT", "period": "1h", "limit": "1"},
            timeout=10,
        )
        resp.raise_for_status()
        ls_ratio = float(resp.json()[0]["longShortRatio"])
        logger.debug("Binance L/S ratio: %s", ls_ratio)
    except Exception as e:
        logger.debug("L/S ratio fetch failed: %s", e)

    if funding_rate is None:
        raise ValueError("Could not fetch any futures sentiment data")

    funding_signal = max(-1.0, min(1.0, funding_rate * 3000))

    if ls_ratio is not None:
        ls_signal = max(-1.0, min(1.0, ls_ratio - 1.0))
        combined = round(0.6 * funding_signal + 0.4 * ls_signal, 4)
        source_detail = (
            f"funding={funding_rate:+.4%}, L/S={ls_ratio:.2f}"
        )
    else:
        combined = round(funding_signal, 4)
        source_detail = f"funding={funding_rate:+.4%}"

    confidence = round(min(0.55, abs(combined) * 0.6 + 0.15), 4)
    direction = (
        "bullish" if combined > 0.05
        else "bearish" if combined < -0.05
        else "neutral"
    )
    summary = (
        f"Futures sentiment: {direction} ({source_detail}, "
        f"signal={combined:+.2f}, conf={confidence:.0%})"
    )
    logger.info(summary)

    return {
        "signal": combined,
        "confidence": confidence,
        "horizons": {},
        "summary": summary,
        "market_count": 0,
    }


def _empty_result(summary: str) -> dict:
    return {
        "signal": 0.0,
        "confidence": 0.0,
        "horizons": {},
        "summary": summary,
        "market_count": 0,
    }
