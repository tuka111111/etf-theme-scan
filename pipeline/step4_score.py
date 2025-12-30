"""
Step4: Scoring / Ranking

Reads Step1 universe and Step3 env outputs, computes per-symbol scores, and writes
per-theme CSV + JSON (envelope) under out/step4_scores/.
"""
from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import pandas as pd
from jsonschema import Draft202012Validator

from .common import ensure_dir, now_utc_iso, parse_csv_list, validate_tf_combo
from .io_step1 import group_symbols_by_theme, load_universe

LOG = logging.getLogger(__name__)

# Minimal inline schema for Step4 outputs (envelope + rows)
SCORE_SCHEMA = {
    "$schema": "https://json-schema.org/draft/2020-12/schema",
    "title": "Step4 Scores Output",
    "type": "object",
    "additionalProperties": False,
    "required": ["schema_version", "generated_at_utc", "theme", "htf_timeframe", "rows"],
    "properties": {
        # Keep permissive inline pattern to avoid schema failures when versioning changes.
        "schema_version": {"type": "string"},
        "generated_at_utc": {"type": "string"},
        "theme": {"type": "string"},
        "htf_timeframe": {"type": "string"},
        "source": {"type": "string"},
        "notes": {"type": "string"},
        "rows": {
            "type": "array",
            "minItems": 1,
            "items": {"$ref": "#/$defs/ScoreRow"},
        },
    },
    "$defs": {
        "ScoreRow": {
            "type": "object",
            "additionalProperties": True,
            "required": ["asof_utc", "theme", "symbol", "score_total", "score_breakdown", "flags"],
            "properties": {
                "asof_utc": {"type": "string"},
                "theme": {"type": "string"},
                "symbol": {"type": "string"},
                "score_total": {"type": "number"},
                "score_breakdown": {"type": "object"},
                "score_breakdown_json": {"type": "string"},
                "env_bias": {"type": "string"},
                "env_confidence": {"type": "number"},
                "env_score": {"type": "number"},
                "flags": {"type": "string"},
                "debug": {"type": "object"},
            },
        }
    },
}


def _clamp(v: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, v))


def _load_step3_env(theme: str, out_dir: Path, htf: str) -> Tuple[List[Dict], Dict[str, Dict]]:
    """
    Load Step3 env JSON envelope for a theme/timeframe.
    Returns (rows, map symbol->row)
    """
    path = out_dir / "step3_env" / f"step3_env_{theme}_{htf}.json"
    if not path.exists():
        raise FileNotFoundError(f"Missing Step3 env JSON for theme={theme} htf={htf}: {path}")

    payload = json.loads(path.read_text(encoding="utf-8"))
    rows = payload.get("rows", [])
    if not isinstance(rows, list) or not rows:
        raise ValueError(f"Invalid or empty Step3 env rows for theme={theme} file={path}")

    m: Dict[str, Dict] = {}
    for r in rows:
        sym = str(r.get("symbol", "")).strip().upper()
        if not sym:
            continue
        m[sym] = r
    return rows, m


def _load_step3_trend(theme: str, out_dir: Path, htf: str) -> Dict[str, Dict]:
    path = out_dir / "step3_trend" / f"trend_{theme}_{htf}.json"
    if not path.exists():
        return {}
    payload = json.loads(path.read_text(encoding="utf-8"))
    rows = payload.get("rows", [])
    m: Dict[str, Dict] = {}
    if isinstance(rows, list):
        for r in rows:
            sym = str(r.get("symbol", "")).strip().upper()
            if sym:
                m[sym] = r
    return m


def _load_step3_regime(theme: str, out_dir: Path, htf: str) -> Dict[str, Dict]:
    path = out_dir / "step3_regime" / f"regime_{theme}_{htf}.json"
    if not path.exists():
        return {}
    payload = json.loads(path.read_text(encoding="utf-8"))
    rows = payload.get("rows", [])
    m: Dict[str, Dict] = {}
    if isinstance(rows, list):
        for r in rows:
            sym = str(r.get("symbol", "")).strip().upper()
            if sym:
                m[sym] = r
    return m


def _score_from_env(env_row: Optional[Dict]) -> Tuple[float, Dict[str, float | str], List[str]]:
    if not env_row:
        return 0.0, {"reason": "missing_env"}, ["missing_env"]

    bias = str(env_row.get("env_bias", "neutral")).lower()
    env_score = float(env_row.get("env_score", 0.0))
    conf = float(env_row.get("env_confidence", 0.5))

    if bias == "bull":
        total = 30.0 * conf
    elif bias == "bear":
        total = 30.0 * conf * 0.5
    else:
        total = 0.0

    flags: List[str] = []
    if bias == "neutral":
        flags.append("env_neutral")
    if conf < 0.5:
        flags.append("env_low_confidence")

    breakdown: Dict[str, float | str] = {
        "bias": bias,
        "env_score_input": env_score,
        "confidence": conf,
        "env_score_component": total,
    }
    return _clamp(total, 0.0, 30.0), breakdown, flags


def _score_from_trend(trend_row: Optional[Dict], env_bias: str) -> Tuple[float, Dict[str, float | str], List[str]]:
    if not trend_row:
        return 0.0, {"reason": "missing_trend"}, ["trend_misaligned"]

    trend_dir = str(trend_row.get("trend_state", "")).lower()
    strength = float(trend_row.get("trend_strength", 0.0))
    bias = env_bias.lower()

    if trend_dir == "up" and bias == "bull":
        score = 30.0 * strength
        flags: List[str] = []
    elif trend_dir == "down" and bias == "bear":
        score = 30.0 * strength
        flags = []
    else:
        score = 0.0
        flags = ["trend_misaligned"]

    if strength < 0.4:
        flags.append("trend_weak")

    breakdown = {
        "trend_dir": trend_dir,
        "trend_strength": strength,
        "trend_score_component": score,
    }
    return _clamp(score, 0.0, 30.0), breakdown, flags


def _score_from_regime(regime_row: Optional[Dict]) -> Tuple[float, Dict[str, float | str], List[str]]:
    if not regime_row:
        return 0.0, {"reason": "missing_regime"}, ["regime_missing"]

    regime_state = str(regime_row.get("regime_state", "")).lower()
    flags: List[str] = []
    if regime_state.startswith("risk_on"):
        score = 20.0
    elif "transition" in regime_state:
        score = 10.0
        flags.append("regime_transition")
    else:
        score = 0.0
        flags.append("regime_range")

    breakdown = {
        "regime_state": regime_state,
        "regime_score_component": score,
    }
    return _clamp(score, 0.0, 20.0), breakdown, flags


def _score_signal(env_conf: float, trend_strength: float) -> Tuple[float, Dict[str, float], List[str]]:
    score = 0.0
    flags: List[str] = []
    if trend_strength > 0.6:
        score += 10.0
    else:
        flags.append("trend_not_strong_for_signal")

    if env_conf > 0.7:
        score += 10.0
    else:
        flags.append("env_conf_not_strong_for_signal")

    return score, {"signal_score_component": score}, flags


def _validate_payload(payload: Dict, schema_path: Optional[Path]) -> Dict:
    if schema_path and schema_path.exists():
        from .validate import must_validate

        return must_validate(schema_path, payload)

    # Fallback: validate against inline schema
    v = Draft202012Validator(SCORE_SCHEMA)
    errs = list(v.iter_errors(payload))
    if errs:
        lines = []
        for e in errs[:20]:
            p = "/".join([str(x) for x in e.absolute_path]) or "(root)"
            lines.append(f"path={p} msg={e.message}")
        more = "" if len(errs) <= 20 else f"\n... ({len(errs) - 20} more)"
        raise ValueError("Schema validation failed (inline SCORE_SCHEMA):\n" + "\n".join(lines) + more)
    return payload


def main(argv: Optional[Sequence[str]] = None) -> int:
    ap = argparse.ArgumentParser(description="Step4: scoring / ranking using Step1 universe and Step3 env.")
    ap.add_argument("--themes", required=True, help="Comma-separated themes, e.g. XME,SMH,XBI")
    ap.add_argument("--out", required=True, help="Output directory (expects Step1/Step3 artifacts)")
    ap.add_argument("--contracts", default="./contracts", help="Contracts directory (optional step4_scores.schema.json)")
    ap.add_argument("--htf", default="1H", help="HTF timeframe (matches Step3 env/trend/regime outputs), e.g. 1H")
    ap.add_argument("--loglevel", default="INFO", help="Logging level (DEBUG/INFO/WARN/ERROR)")
    args = ap.parse_args(argv)

    logging.basicConfig(level=getattr(logging, args.loglevel.upper(), logging.INFO))

    themes = [t.strip().upper() for t in parse_csv_list(args.themes)]
    if not themes:
        raise SystemExit("Provide at least one theme via --themes")

    htf = validate_tf_combo("step3", htf=args.htf)
    out_dir = ensure_dir(args.out)
    scores_dir = ensure_dir(out_dir / "step4_scores")

    _, universe_rows = load_universe(out_dir, themes=themes)
    sym_by_theme = group_symbols_by_theme(universe_rows)

    schema_path = Path(args.contracts) / "step4_scores.schema.json"
    if not schema_path.exists():
        LOG.warning("Contracts schema not found for Step4: %s (using inline schema)", schema_path)

    for theme in themes:
        symbols = sym_by_theme.get(theme, [])
        if not symbols:
            LOG.warning("No symbols for theme=%s from Step1; skipping.", theme)
            continue

        try:
            _, env_map = _load_step3_env(theme, out_dir, htf)
        except Exception as e:
            LOG.warning("Skipping theme=%s due to missing/invalid Step3 env: %s", theme, e)
            continue

        trend_map = _load_step3_trend(theme, out_dir, htf)
        regime_map = _load_step3_regime(theme, out_dir, htf)

        records: List[Dict] = []
        for sym in symbols:
            env_row = env_map.get(sym)
            trend_row = trend_map.get(sym)
            regime_row = regime_map.get(sym)

            env_score, env_breakdown, env_flags = _score_from_env(env_row)
            env_bias = str(env_row.get("env_bias")) if env_row else "unknown"
            env_conf = float(env_row.get("env_confidence", 0.0)) if env_row else 0.0

            trend_score, trend_breakdown, trend_flags = _score_from_trend(trend_row, env_bias if env_row else "neutral")
            trend_strength = float(trend_row.get("trend_strength", 0.0)) if trend_row else 0.0

            regime_score, regime_breakdown, regime_flags = _score_from_regime(regime_row)

            signal_score, signal_breakdown, signal_flags = _score_signal(env_conf, trend_strength)

            total = env_score + trend_score + regime_score + signal_score
            total = min(100.0, round(total, 1))

            asof = (
                str(env_row.get("asof_utc"))
                if env_row and env_row.get("asof_utc")
                else (str(trend_row.get("asof_utc")) if trend_row and trend_row.get("asof_utc") else now_utc_iso())
            )

            flags = list(dict.fromkeys(env_flags + trend_flags + regime_flags + signal_flags))

            # Additional flag rules
            regime_state = str(regime_row.get("regime_state", "")).lower() if regime_row else ""
            if regime_state == "range":
                flags.append("regime_range")
            if trend_score == 0:
                flags.append("trend_misaligned")
            if env_conf < 0.5:
                flags.append("env_low_confidence")

            flags = list(dict.fromkeys(flags))

            breakdown = {
                "env": env_breakdown,
                "trend": trend_breakdown,
                "regime": regime_breakdown,
                "signal": signal_breakdown,
            }

            row = {
                "asof_utc": asof,
                "theme": theme,
                "symbol": sym,
                "score_total": float(total),
                "score_breakdown": breakdown,
                "score_breakdown_json": json.dumps(breakdown, ensure_ascii=False),
                "env_bias": env_bias,
                "env_confidence": env_conf,
                "env_score": float(env_row.get("env_score", 0.0)) if env_row else 0.0,
                "trend_state": trend_row.get("trend_state") if trend_row else "unknown",
                "trend_strength": trend_strength,
                "regime_state": regime_row.get("regime_state") if regime_row else "unknown",
                "regime_score_component": regime_score,
                "flags": ",".join(flags),
                "notes": "",
                "debug": {
                    "env": env_row,
                    "trend": trend_row,
                    "regime": regime_row,
                },
            }
            records.append(row)

        if not records:
            LOG.warning("No records produced for theme=%s; skipping outputs.", theme)
            continue

        payload = {
            "schema_version": "4.scores.v1",
            "generated_at_utc": now_utc_iso(),
            "theme": theme,
            "htf_timeframe": htf,
            "source": "step4_score.py",
            "notes": f"Scores derived from Step3 env/trend/regime ({htf}) for theme={theme}",
            "rows": records,
        }

        payload = _validate_payload(payload, schema_path if schema_path.exists() else None)

        json_path = scores_dir / f"scores_{theme}.json"
        csv_path = scores_dir / f"scores_{theme}.csv"

        json_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
        df_out = pd.DataFrame(records)
        df_out = df_out.sort_values(["score_total", "symbol"], ascending=[False, True]).reset_index(drop=True)
        df_out["rank"] = df_out.index + 1
        df_out.to_csv(csv_path, index=False)

        LOG.info(
            "Theme=%s scored rows=%d -> json=%s csv=%s",
            theme,
            len(records),
            json_path,
            csv_path,
        )
        print(f"[OK] theme={theme} json={json_path} csv={csv_path} rows={len(records)}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
