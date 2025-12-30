from __future__ import annotations

from typing import Dict, List


def check_env_record(rec: Dict) -> List[str]:
    warnings: List[str] = []
    conf = rec.get("env_confidence")
    if conf is None or not (0.0 <= float(conf) <= 1.0):
        warnings.append("env_confidence_out_of_range")
    return warnings


def check_trend_record(rec: Dict) -> List[str]:
    warnings: List[str] = []
    strength = rec.get("trend_strength")
    if strength is None or not (0.0 <= float(strength) <= 1.0):
        warnings.append("trend_strength_out_of_range")
    return warnings


def check_regime_record(rec: Dict) -> List[str]:
    warnings: List[str] = []
    regime = str(rec.get("regime_state", "")).lower()
    allowed = str(rec.get("allowed_direction", "")).lower()
    if regime.startswith("risk_on") and allowed == "none":
        warnings.append("regime_allowed_none_mismatch")
    return warnings
