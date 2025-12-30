from __future__ import annotations

from typing import Dict, Tuple


def classify_regime(
    env_bias: str,
    env_confidence: float,
    trend_state: str,
    trend_strength: float,
) -> Tuple[str, str, float, float, str, Dict]:
    """
    HTF regime classifier (pure function).
    Returns (regime_state, allowed_direction, position_multiplier, regime_score, reason, components).
    """
    b = (env_bias or "neutral").lower()
    t = (trend_state or "range").lower()
    conf = float(env_confidence)
    strength = float(trend_strength)

    match_long = b == "bull" and t == "up"
    match_short = b == "bear" and t == "down"

    if match_long:
        regime_state = "risk_on_long"
        allowed = "long"
    elif match_short:
        regime_state = "risk_on_short"
        allowed = "short"
    else:
        regime_state = "no_trade"
        allowed = "none"

    if regime_state.startswith("risk_on") and conf >= 0.6 and strength >= 0.6:
        mult = 1.0
        reason = "Env/Trend aligned and strong"
    elif regime_state.startswith("risk_on"):
        mult = 0.5
        reason = "Env/Trend aligned but weak confidence/strength"
    else:
        mult = 0.0
        reason = "Env/Trend not aligned"

    score = mult * 100.0
    components = {
        "env_bias": b,
        "env_confidence": conf,
        "trend_state": t,
        "trend_strength": strength,
        "vol_state": "unknown",
        "risk_flags": [],
    }
    return regime_state, allowed, mult, score, reason, components
