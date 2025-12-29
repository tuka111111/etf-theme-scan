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

from .common import ensure_dir, now_utc_iso, parse_csv_list
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


def _score_from_env(env_row: Optional[Dict]) -> Tuple[float, Dict[str, float | str], List[str]]:
    if not env_row:
        return 0.0, {"reason": "missing_env"}, ["missing_env"]

    bias = str(env_row.get("env_bias", "neutral")).lower()
    env_score = float(env_row.get("env_score", 0.0))
    conf = float(env_row.get("env_confidence", 0.5))

    bias_bonus = {"bull": 25.0, "bear": -25.0}.get(bias, 0.0)
    total = bias_bonus + env_score * 0.4 + conf * 20.0
    total = _clamp(total, -100.0, 100.0)

    flags: List[str] = []
    if bias == "neutral":
        flags.append("env_neutral")

    breakdown: Dict[str, float | str] = {
        "bias": bias,
        "bias_bonus": bias_bonus,
        "env_score": env_score,
        "confidence": conf,
    }
    return total, breakdown, flags


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
    ap.add_argument("--htf", default="1H", help="HTF timeframe (matches Step3 env outputs), e.g. 1H")
    ap.add_argument("--loglevel", default="INFO", help="Logging level (DEBUG/INFO/WARN/ERROR)")
    args = ap.parse_args(argv)

    logging.basicConfig(level=getattr(logging, args.loglevel.upper(), logging.INFO))

    themes = [t.strip().upper() for t in parse_csv_list(args.themes)]
    if not themes:
        raise SystemExit("Provide at least one theme via --themes")

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
            _, env_map = _load_step3_env(theme, out_dir, args.htf)
        except Exception as e:
            LOG.warning("Skipping theme=%s due to missing/invalid Step3 env: %s", theme, e)
            continue

        records: List[Dict] = []
        for sym in symbols:
            env_row = env_map.get(sym)
            total, breakdown, flags = _score_from_env(env_row)

            asof = str(env_row.get("asof_utc")) if env_row else now_utc_iso()
            row = {
                "asof_utc": asof,
                "theme": theme,
                "symbol": sym,
                "score_total": float(total),
                "score_breakdown": breakdown,
                "score_breakdown_json": json.dumps(breakdown, ensure_ascii=False),
                "env_bias": env_row.get("env_bias") if env_row else "unknown",
                "env_confidence": float(env_row.get("env_confidence", 0.0)) if env_row else 0.0,
                "env_score": float(env_row.get("env_score", 0.0)) if env_row else 0.0,
                "flags": ",".join(flags),
                "debug": {"source_env": env_row} if env_row else {"reason": "missing_env"},
            }
            records.append(row)

        if not records:
            LOG.warning("No records produced for theme=%s; skipping outputs.", theme)
            continue

        payload = {
            "schema_version": "4.scores.v1",
            "generated_at_utc": now_utc_iso(),
            "theme": theme,
            "htf_timeframe": args.htf,
            "source": "step4_score.py",
            "notes": f"Scores derived from Step3 env ({args.htf}) for theme={theme}",
            "rows": records,
        }

        payload = _validate_payload(payload, schema_path if schema_path.exists() else None)

        json_path = scores_dir / f"scores_{theme}.json"
        csv_path = scores_dir / f"scores_{theme}.csv"

        json_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
        pd.DataFrame(records).to_csv(csv_path, index=False)

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
