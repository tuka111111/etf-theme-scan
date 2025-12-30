# pipeline/common.py
from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable, List, Optional


def now_utc_iso() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def ensure_dir(p: str | Path) -> Path:
    d = Path(p)
    d.mkdir(parents=True, exist_ok=True)
    return d


def parse_csv_list(s: str) -> List[str]:
    if not s:
        return []
    parts = [x.strip() for x in s.replace(" ", ",").split(",")]
    return [p for p in parts if p]


def uniq_keep_order(items: Iterable[str]) -> List[str]:
    seen = set()
    out = []
    for x in items:
        if x not in seen:
            out.append(x)
            seen.add(x)
    return out


def normalize_tf(tf: str) -> str:
    s = str(tf).strip()
    if not s:
        return s
    s_up = s.upper()
    if s_up.endswith("M"):
        return s_up[:-1].strip() + "m"
    if s_up.endswith("H"):
        return s_up[:-1].strip() + "H"
    if s_up.endswith("D"):
        return s_up[:-1].strip() + "D"
    return s


def validate_tf_combo(step: str, *, tf: Optional[str] = None, htf: Optional[str] = None, ltf: Optional[str] = None) -> str:
    """
    Validate and normalize timeframe combinations per step.
    Returns the normalized primary tf (if provided).
    """
    step = step.strip().lower()
    tf_norm = normalize_tf(tf) if tf else None
    htf_norm = normalize_tf(htf) if htf else None
    ltf_norm = normalize_tf(ltf) if ltf else None

    if step == "step2":
        allowed = {"1D", "1H", "15m"}
        if tf_norm not in allowed:
            raise SystemExit(f"Step2 invalid --tf={tf}. Allowed: {sorted(allowed)}")
        return tf_norm

    if step == "step3":
        allowed_htf = {"1D", "4H", "1H", "30m", "15m"}
        if htf_norm not in allowed_htf:
            raise SystemExit(f"Step3 invalid --htf={htf}. Allowed HTF: {sorted(allowed_htf)}")
        return htf_norm

    if step == "step4":
        allowed_ltf = {"15m", "5m", "1m"}
        allowed_htf = {"1D", "4H", "1H"}
        if htf_norm not in allowed_htf:
            raise SystemExit(f"Step4 invalid HTF={htf}. Allowed HTF: {sorted(allowed_htf)}")
        if ltf_norm and ltf_norm not in allowed_ltf:
            raise SystemExit(f"Step4 invalid LTF={ltf}. Allowed LTF: {sorted(allowed_ltf)}")
        return ltf_norm or ""

    return tf_norm or htf_norm or ltf_norm or ""
