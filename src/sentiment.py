"""
Polymarket sentiment analysis for BTC trading signals.

Fetches prediction market data from Polymarket's public API and derives a
directional signal from the implied median expected BTC price across multiple
time horizons (short ≤24h, medium 1–7d, long 7–60d, macro >60d).

No API key required — Polymarket's gamma API is publicly accessible.
"""

import json
import logging
import re
from datetime import datetime, timezone

import requests

logger = logging.getLogger(__name__)

POLYMARKET_API = "https://gamma-api.polymarket.com/markets"

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

def fetch_polymarket_sentiment(current_price_usd: float) -> dict:
    """
    Fetch BTC prediction markets from Polymarket and derive a directional signal.

    The signal is derived from the implied median expected BTC price:
      - Collect (target_price, yes_prob) pairs from "Will BTC be above $X?" markets
      - Group by time horizon (short / medium / long / macro)
      - Per horizon: interpolate where yes_prob = 0.50 → implied median price
      - Signal = clipped((implied_median − current) / current × 10)
      - Combine horizons with _HORIZON_WEIGHTS

    Args:
        current_price_usd: Latest BTC price in USD (used for signal direction).

    Returns:
        dict with keys:
          signal       float  [-1, 1]        Aggregated directional signal
          confidence   float  [0, 0.75]      Signal reliability
          horizons     dict   Per-horizon (signal, confidence, n_markets)
          summary      str    Human-readable one-liner
          market_count int    Total usable markets found
    """
    try:
        markets = _fetch_markets()
    except Exception as e:
        logger.warning("Polymarket fetch failed: %s", e)
        return _empty_result("Polymarket data unavailable")

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

def _fetch_markets() -> list[dict]:
    """Fetch active BTC prediction markets from Polymarket gamma API."""
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
            # Deduplicate by id
            ids = {m.get("id") for m in markets}
            markets += [m for m in extra if m.get("id") not in ids]
        except Exception:
            pass  # First batch is sufficient

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


def _empty_result(summary: str) -> dict:
    return {
        "signal": 0.0,
        "confidence": 0.0,
        "horizons": {},
        "summary": summary,
        "market_count": 0,
    }
