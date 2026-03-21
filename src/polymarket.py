"""
Polymarket sentiment module.

Fetches BTC prediction market data from Polymarket and converts implied
probabilities into a sentiment signal (BUY / SELL / HOLD) with confidence.

Supports two market types:
  - Short-term Up/Down (binary): 1h, 4h intervals
  - Long-term price thresholds (multi-outcome): monthly, yearly

Uses a 3-source fallback chain for geo-restriction resilience:
  1. Gamma API  — richest metadata, may be geo-blocked
  2. CLOB API   — public read endpoints, different geo-policy
  3. Goldsky Subgraph — decentralized, no geo-restrictions
"""

import json
import logging
import re
import time
from math import log10

import requests

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# API endpoints
# ---------------------------------------------------------------------------
GAMMA_API_URL = "https://gamma-api.polymarket.com"
CLOB_API_URL = "https://clob.polymarket.com"
GOLDSKY_SUBGRAPH_URL = (
    "https://api.goldsky.com/api/public/"
    "project_cl6mb8i9h0003e201j6li0diw/subgraphs/"
    "orderbook-subgraph/0.0.1/gn"
)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
REQUEST_TIMEOUT = 10
MIN_VOLUME_USD = 1_000
SENTIMENT_BULL_THRESHOLD = 0.55
SENTIMENT_BEAR_THRESHOLD = 0.45
UPDOWN_INTERVALS = ["1h", "4h"]
SHORT_TERM_WEIGHT = 0.60
LONG_TERM_WEIGHT = 0.40

SIGNAL_NAMES = {1: "BULL", 0: "NEUTRAL", -1: "BEAR"}

# Interval durations in seconds (for slug timestamp rounding)
_INTERVAL_SECONDS = {"1h": 3600, "4h": 14400}


# ===================================================================
# Public API
# ===================================================================

def fetch_polymarket_sentiment(
    current_price_usd: float,
) -> tuple[int, float, str, str] | None:
    """
    Fetch BTC prediction market data and convert to a sentiment signal.

    Args:
        current_price_usd: latest BTC price (needed for threshold markets)

    Returns:
        (signal, confidence, summary_html, source_name) or None.
        signal:  1=BUY, 0=HOLD, -1=SELL
        confidence: 0.0–1.0
        summary_html: pre-formatted HTML string for Telegram message
        source_name: API source used (e.g. "Gamma API")
    """
    try:
        updown, thresholds, source = _fetch_btc_markets_with_fallback()

        if not updown and not thresholds:
            logger.info("Polymarket: no valid BTC markets found")
            return None

        updown_result = _compute_updown_sentiment(updown)
        threshold_result = _compute_threshold_sentiment(thresholds, current_price_usd)

        if updown_result is None and threshold_result is None:
            logger.info("Polymarket: insufficient data for sentiment")
            return None

        signal, confidence, summary = _combine_sentiments(
            updown_result, threshold_result, len(updown) + len(thresholds),
        )
        summary_html = _build_summary_html(
            signal, confidence, updown_result, threshold_result,
            len(updown) + len(thresholds), source,
        )

        return signal, confidence, summary_html, source

    except Exception:
        logger.warning("Polymarket sentiment fetch failed", exc_info=True)
        return None


# ===================================================================
# Data fetching — 3-source fallback
# ===================================================================

def _fetch_btc_markets_with_fallback() -> tuple[list[dict], list[dict], str]:
    """
    Try Gamma → CLOB → Goldsky, return first that yields results.

    Returns:
        (updown_markets, threshold_markets, source_name)
    """
    sources = [
        ("Gamma API", _fetch_via_gamma),
        ("CLOB API", _fetch_via_clob),
        ("Goldsky Subgraph", _fetch_via_goldsky),
    ]
    for name, fetcher in sources:
        try:
            updown, thresholds = fetcher()
            if updown or thresholds:
                logger.info("Polymarket data from %s: %d updown, %d threshold markets",
                            name, len(updown), len(thresholds))
                return updown, thresholds, name
            logger.debug("Polymarket %s returned no markets", name)
        except Exception:
            logger.debug("Polymarket %s failed", name, exc_info=True)

    return [], [], "none"


# ---------------------------------------------------------------------------
# Source 1: Gamma API
# ---------------------------------------------------------------------------

def _fetch_via_gamma() -> tuple[list[dict], list[dict]]:
    """Fetch from the Gamma API (richest metadata)."""
    updown = _gamma_fetch_updown()
    thresholds = _gamma_fetch_thresholds()
    return updown, thresholds


def _gamma_fetch_updown() -> list[dict]:
    """Fetch short-term BTC Up/Down markets via slug generation + broad search."""
    now_ts = int(time.time())
    markets = []
    seen_slugs: set[str] = set()

    # 1. Fast O(1) lookups by exact slug for current intervals
    for interval in UPDOWN_INTERVALS:
        secs = _INTERVAL_SECONDS[interval]
        rounded_ts = (now_ts // secs) * secs
        slug = f"btc-updown-{interval}-{rounded_ts}"

        try:
            resp = requests.get(
                f"{GAMMA_API_URL}/markets",
                params={"slug": slug},
                timeout=REQUEST_TIMEOUT,
            )
            resp.raise_for_status()
            data = resp.json()

            items = data if isinstance(data, list) else [data]
            for item in items:
                parsed = _parse_updown_market(item, interval)
                if parsed:
                    markets.append(parsed)
                    seen_slugs.add(slug)
        except Exception:
            logger.debug("Gamma slug lookup failed for %s", slug)

    # 2. Broader search via /markets with Crypto tag to catch additional markets
    try:
        resp = requests.get(
            f"{GAMMA_API_URL}/markets",
            params={"active": "true", "closed": "false", "limit": 50,
                    "tag_id": 21},
            timeout=REQUEST_TIMEOUT,
        )
        resp.raise_for_status()
        data = resp.json()

        items = data if isinstance(data, list) else []
        for item in items:
            item_slug = (item.get("slug") or "").lower()
            if "btc-updown" not in item_slug:
                continue
            if item_slug in seen_slugs:
                continue
            interval = _extract_interval_from_slug(item_slug)
            parsed = _parse_updown_market(item, interval)
            if parsed:
                markets.append(parsed)
                seen_slugs.add(item_slug)
    except Exception:
        logger.debug("Gamma broad updown search failed", exc_info=True)

    return markets


def _gamma_fetch_thresholds() -> list[dict]:
    """Fetch long-term BTC price threshold markets from Gamma events."""
    resp = requests.get(
        f"{GAMMA_API_URL}/events",
        params={"active": "true", "closed": "false", "limit": 50, "tag_id": 21},
        timeout=REQUEST_TIMEOUT,
    )
    resp.raise_for_status()
    events = resp.json()

    markets = []
    price_keywords = ("price", "hit", "reach", "above", "below", "end")
    for event in events:
        slug = event.get("slug", "").lower()
        title = event.get("title", "").lower()
        # Match BTC price prediction events
        if not (("bitcoin" in slug or "btc" in slug or "bitcoin" in title or "btc" in title)
                and any(kw in slug or kw in title for kw in price_keywords)):
            continue

        # Events contain nested markets
        event_markets = event.get("markets", [])
        for mkt in event_markets:
            parsed = _parse_threshold_market(mkt)
            if parsed:
                markets.append(parsed)

    return markets


# ---------------------------------------------------------------------------
# Source 2: CLOB API (public read endpoints)
# ---------------------------------------------------------------------------

def _fetch_via_clob() -> tuple[list[dict], list[dict]]:
    """
    Fetch from CLOB public endpoints using known token IDs.

    The CLOB needs token IDs which we discover via a Gamma markets lookup.
    If Gamma is blocked, we try the CLOB's own simplified markets endpoint.
    """
    updown = []
    thresholds = []

    try:
        # Try the simplified markets endpoint
        resp = requests.get(
            f"{CLOB_API_URL}/simplified-markets",
            params={"limit": 50},
            timeout=REQUEST_TIMEOUT,
        )
        resp.raise_for_status()
        data = resp.json()

        for mkt in data if isinstance(data, list) else []:
            slug = mkt.get("slug", "").lower()
            tokens = mkt.get("tokens", [])

            if "btc-updown" in slug and tokens:
                interval = _extract_interval_from_slug(slug)
                parsed = _parse_clob_updown(mkt, interval)
                if parsed:
                    updown.append(parsed)
            elif (("bitcoin" in slug or "btc" in slug)
                  and ("price" in slug or "hit" in slug)):
                parsed = _parse_clob_threshold(mkt)
                if parsed:
                    thresholds.append(parsed)
    except Exception:
        logger.debug("CLOB simplified-markets failed", exc_info=True)

    # Fallback: try midpoint for known short-term slug patterns
    if not updown:
        updown = _clob_fetch_updown_via_midpoint()

    return updown, thresholds


def _clob_fetch_updown_via_midpoint() -> list[dict]:
    """Try fetching Up/Down prices via CLOB midpoint endpoint."""
    now_ts = int(time.time())
    markets = []

    for interval in UPDOWN_INTERVALS:
        secs = _INTERVAL_SECONDS[interval]
        rounded_ts = (now_ts // secs) * secs
        slug = f"btc-updown-{interval}-{rounded_ts}"

        try:
            # First, get token IDs from Gamma (may fail if geo-blocked)
            resp = requests.get(
                f"{GAMMA_API_URL}/markets",
                params={"slug": slug},
                timeout=REQUEST_TIMEOUT,
            )
            if resp.status_code != 200:
                continue

            data = resp.json()
            items = data if isinstance(data, list) else [data]
            for item in items:
                token_ids = item.get("clobTokenIds", [])
                if len(token_ids) >= 2:
                    # Get midpoint prices from CLOB
                    mid_up = _get_clob_midpoint(token_ids[0])
                    mid_down = _get_clob_midpoint(token_ids[1])
                    if mid_up is not None:
                        volume_str = item.get("volume", "0")
                        markets.append({
                            "type": "updown",
                            "interval": interval,
                            "up_prob": mid_up,
                            "down_prob": mid_down or (1.0 - mid_up),
                            "volume": float(volume_str),
                        })
        except Exception:
            continue

    return markets


def _get_clob_midpoint(token_id: str) -> float | None:
    """Get the midpoint price for a token from CLOB."""
    try:
        resp = requests.get(
            f"{CLOB_API_URL}/midpoint",
            params={"token_id": token_id},
            timeout=REQUEST_TIMEOUT,
        )
        resp.raise_for_status()
        data = resp.json()
        return float(data.get("mid", 0))
    except Exception:
        return None


# ---------------------------------------------------------------------------
# Source 3: Goldsky Subgraph (decentralized, no geo-restrictions)
# ---------------------------------------------------------------------------

def _fetch_via_goldsky() -> tuple[list[dict], list[dict]]:
    """
    Fetch from Goldsky subgraph via GraphQL.

    Queries recent market conditions for BTC-related tokens with prices.
    """
    query = """
    {
      markets(first: 50, orderBy: tradesQuantity, orderDirection: desc) {
        id
        question
        slug
        outcomes
        outcomeTokenPrices
        scaledCollateralVolume
        tradesQuantity
      }
    }
    """

    resp = requests.post(
        GOLDSKY_SUBGRAPH_URL,
        json={"query": query},
        timeout=REQUEST_TIMEOUT,
    )
    resp.raise_for_status()
    result = resp.json()

    updown = []
    thresholds = []

    markets_data = result.get("data", {}).get("markets", [])
    for mkt in markets_data:
        slug = (mkt.get("slug") or "").lower()
        question = (mkt.get("question") or "").lower()

        if not ("btc" in slug or "bitcoin" in slug
                or "btc" in question or "bitcoin" in question):
            continue

        prices = mkt.get("outcomeTokenPrices", [])
        outcomes = mkt.get("outcomes", [])
        volume = float(mkt.get("scaledCollateralVolume", 0) or 0)

        if "updown" in slug:
            parsed = _parse_goldsky_updown(mkt, prices, volume)
            if parsed:
                updown.append(parsed)
        elif "price" in slug or "hit" in question:
            parsed = _parse_goldsky_threshold(mkt, prices, outcomes, volume)
            if parsed:
                thresholds.append(parsed)

    return updown, thresholds


# ===================================================================
# Market parsers
# ===================================================================

def _parse_updown_market(market: dict, interval: str) -> dict | None:
    """Parse a Gamma API updown market into normalized dict."""
    try:
        prices_raw = market.get("outcomePrices", "[]")
        prices = json.loads(prices_raw) if isinstance(prices_raw, str) else prices_raw
        if len(prices) < 2:
            return None

        up_prob = float(prices[0])
        down_prob = float(prices[1])
        volume = float(market.get("volume", 0) or 0)

        if volume < MIN_VOLUME_USD:
            return None

        return {
            "type": "updown",
            "interval": interval,
            "up_prob": up_prob,
            "down_prob": down_prob,
            "volume": volume,
        }
    except (ValueError, KeyError, json.JSONDecodeError):
        return None


def _parse_threshold_market(market: dict) -> dict | None:
    """Parse a Gamma API threshold market into normalized dict."""
    try:
        outcomes_raw = market.get("outcomes", "[]")
        outcomes = json.loads(outcomes_raw) if isinstance(outcomes_raw, str) else outcomes_raw
        prices_raw = market.get("outcomePrices", "[]")
        prices = json.loads(prices_raw) if isinstance(prices_raw, str) else prices_raw
        volume = float(market.get("volume", 0) or 0)

        if volume < MIN_VOLUME_USD:
            return None

        # Extract price level from outcome text (e.g., "↑ 100,000" → 100000)
        results = []
        for outcome, price in zip(outcomes, prices):
            price_level = _extract_price_from_outcome(str(outcome))
            if price_level is not None:
                results.append({
                    "price_level": price_level,
                    "probability": float(price),
                })

        if not results:
            return None

        return {
            "type": "threshold",
            "outcomes": results,
            "volume": volume,
        }
    except (ValueError, KeyError, json.JSONDecodeError):
        return None


def _parse_clob_updown(market: dict, interval: str) -> dict | None:
    """Parse a CLOB simplified-markets updown entry."""
    try:
        tokens = market.get("tokens", [])
        if len(tokens) < 2:
            return None
        up_prob = float(tokens[0].get("price", 0))
        down_prob = float(tokens[1].get("price", 0))
        volume = float(market.get("volume", 0) or 0)
        if volume < MIN_VOLUME_USD:
            return None
        return {
            "type": "updown",
            "interval": interval,
            "up_prob": up_prob,
            "down_prob": down_prob,
            "volume": volume,
        }
    except (ValueError, KeyError):
        return None


def _parse_clob_threshold(market: dict) -> dict | None:
    """Parse a CLOB simplified-markets threshold entry."""
    try:
        tokens = market.get("tokens", [])
        outcomes = market.get("outcomes", [])
        volume = float(market.get("volume", 0) or 0)
        if volume < MIN_VOLUME_USD:
            return None

        results = []
        for token, outcome in zip(tokens, outcomes):
            price_level = _extract_price_from_outcome(str(outcome))
            prob = float(token.get("price", 0))
            if price_level is not None:
                results.append({"price_level": price_level, "probability": prob})

        if not results:
            return None
        return {"type": "threshold", "outcomes": results, "volume": volume}
    except (ValueError, KeyError):
        return None


def _parse_goldsky_updown(market: dict, prices: list, volume: float) -> dict | None:
    """Parse a Goldsky subgraph updown market."""
    try:
        if len(prices) < 2 or volume < MIN_VOLUME_USD:
            return None
        slug = (market.get("slug") or "").lower()
        interval = _extract_interval_from_slug(slug)
        return {
            "type": "updown",
            "interval": interval,
            "up_prob": float(prices[0]),
            "down_prob": float(prices[1]),
            "volume": volume,
        }
    except (ValueError, IndexError):
        return None


def _parse_goldsky_threshold(
    market: dict, prices: list, outcomes: list, volume: float,
) -> dict | None:
    """Parse a Goldsky subgraph threshold market."""
    try:
        if volume < MIN_VOLUME_USD:
            return None
        results = []
        for outcome, price in zip(outcomes, prices):
            price_level = _extract_price_from_outcome(str(outcome))
            if price_level is not None:
                results.append({"price_level": price_level, "probability": float(price)})
        if not results:
            return None
        return {"type": "threshold", "outcomes": results, "volume": volume}
    except (ValueError, KeyError):
        return None


# ===================================================================
# Sentiment computation
# ===================================================================

def _compute_updown_sentiment(
    markets: list[dict],
) -> tuple[float, float] | None:
    """
    Volume-weighted average of up_prob across short-term markets.

    Returns:
        (avg_up_prob, total_volume) or None if no valid markets.
    """
    if not markets:
        return None

    weighted_sum = 0.0
    total_vol = 0.0
    for m in markets:
        vol = m["volume"]
        weighted_sum += m["up_prob"] * vol
        total_vol += vol

    if total_vol <= 0:
        return None

    return weighted_sum / total_vol, total_vol


def _compute_threshold_sentiment(
    markets: list[dict],
    current_price_usd: float,
) -> tuple[float, float] | None:
    """
    Derive directional sentiment from price threshold markets.

    For each threshold:
      - Above current price with high probability → bullish
      - Below current price with low probability → also bullish (market doubts the drop)

    Returns:
        (up_prob_equivalent, total_volume) or None.
    """
    if not markets:
        return None

    bullish_score = 0.0
    total_weight = 0.0
    total_vol = 0.0

    for mkt in markets:
        vol = mkt["volume"]
        total_vol += vol

        for outcome in mkt["outcomes"]:
            price_level = outcome["price_level"]
            prob = outcome["probability"]

            if price_level > current_price_usd:
                # Higher price threshold: high prob = bullish
                bullish_score += prob * vol
                total_weight += vol
            elif price_level < current_price_usd:
                # Lower price threshold: high prob = bearish (so invert)
                bullish_score += (1.0 - prob) * vol
                total_weight += vol

    if total_weight <= 0:
        return None

    # Normalize to [0, 1] — 0.5 is neutral
    up_prob_eq = bullish_score / total_weight
    return up_prob_eq, total_vol


def _combine_sentiments(
    updown: tuple[float, float] | None,
    threshold: tuple[float, float] | None,
    num_markets: int,
) -> tuple[int, float, str]:
    """
    Blend short-term and long-term sentiments into a final signal.

    Returns:
        (signal, confidence, text_summary)
    """
    # Weighted blend of available components
    if updown is not None and threshold is not None:
        up_short, vol_short = updown
        up_long, vol_long = threshold
        avg_up = up_short * SHORT_TERM_WEIGHT + up_long * LONG_TERM_WEIGHT
        total_vol = vol_short + vol_long
    elif updown is not None:
        avg_up, total_vol = updown
    else:
        avg_up, total_vol = threshold  # type: ignore[misc]

    # Convert to signal
    if avg_up > SENTIMENT_BULL_THRESHOLD:
        signal = 1
    elif avg_up < SENTIMENT_BEAR_THRESHOLD:
        signal = -1
    else:
        signal = 0

    # Raw confidence from distance to neutral
    raw_conf = abs(avg_up - 0.5) * 2.0  # 0.0–1.0

    # Scale by volume reliability (log scale, $100K+ → full credit)
    vol_factor = min(1.0, log10(max(total_vol, 1)) / 5.0)
    confidence = raw_conf * vol_factor

    return signal, confidence, ""


# ===================================================================
# Output formatting
# ===================================================================

def _build_summary_html(
    signal: int,
    confidence: float,
    updown: tuple[float, float] | None,
    threshold: tuple[float, float] | None,
    num_markets: int,
    source: str,
) -> str:
    """Build pre-formatted HTML for the Telegram message."""
    name = SIGNAL_NAMES[signal]
    conf_pct = round(confidence * 100)

    lines = [f"Signal: {name} ({conf_pct}%)"]

    if updown is not None:
        up_pct = round(updown[0] * 100)
        lines.append(f"Short-term (1h/4h Up/Down): {up_pct}% ↑")

    if threshold is not None:
        up_pct = round(threshold[0] * 100)
        lines.append(f"Long-term (monthly/yearly): {up_pct}% ↑")

    total_vol = 0.0
    if updown:
        total_vol += updown[1]
    if threshold:
        total_vol += threshold[1]

    if total_vol >= 1_000_000:
        vol_str = f"${total_vol / 1_000_000:.1f}M"
    elif total_vol >= 1_000:
        vol_str = f"${total_vol / 1_000:.0f}K"
    else:
        vol_str = f"${total_vol:,.0f}"

    lines.append(f"Markets analyzed: {num_markets} ({vol_str} total volume)")
    lines.append(f"Data source: {source}")

    return "\n   ".join(lines)


# ===================================================================
# Helpers
# ===================================================================

def _extract_price_from_outcome(text: str) -> float | None:
    """
    Extract a price level from outcome text.

    Examples:
        "↑ 100,000" → 100000.0
        "↑ 75,000"  → 75000.0
        "$90000"    → 90000.0
    """
    # Remove currency symbols and arrows
    cleaned = text.replace("↑", "").replace("↓", "").replace("$", "").strip()
    # Remove commas
    cleaned = cleaned.replace(",", "")
    # Extract first number
    match = re.search(r"[\d.]+", cleaned)
    if match:
        try:
            return float(match.group())
        except ValueError:
            pass
    return None


def _extract_interval_from_slug(slug: str) -> str:
    """Extract interval like '1h' or '4h' from a slug."""
    for interval in UPDOWN_INTERVALS:
        if interval in slug:
            return interval
    return "unknown"
