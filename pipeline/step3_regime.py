from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import pandas as pd

from .common import ensure_dir, parse_csv_list
from .io_step1 import group_symbols_by_theme, load_universe
from .validate import must_validate

LOG = logging.getLogger(__name__)


def _load_env(theme: str, out_dir: Path, htf: str) -> Dict[str, Dict]:
    path = out_dir / "step3_env" / f"step3_env_{theme}_{htf}.json"
    if not path.exists():
        return {}
    payload = json.loads(path.read_text(encoding="utf-8"))
    rows = payload.get("rows", [])
    if not isinstance(rows, list):
        return {}
    m: Dict[str, Dict] = {}
    for r in rows:
        sym = str(r.get("symbol", "")).strip().upper()
        if sym:
            m[sym] = r
    return m


def _load_trend(theme: str, out_dir: Path, htf: str) -> Dict[str, Dict]:
    path = out_dir / "step3_trend" / f"trend_{theme}_{htf}.json"
    if not path.exists():
        return {}
    payload = json.loads(path.read_text(encoding="utf-8"))
    rows = payload.get("rows", [])
    if not isinstance(rows, list):
        return {}
    m: Dict[str, Dict] = {}
    for r in rows:
        sym = str(r.get("symbol", "")).strip().upper()
        if sym:
            m[sym] = r
    return m


def _regime_from(env_row: Optional[Dict], trend_row: Optional[Dict]) -> Tuple[str, str, float, float, str, Dict]:
    env_bias = str(env_row.get("env_bias", "neutral")).lower() if env_row else "neutral"
    env_score = float(env_row.get("env_score", 0.0)) if env_row else 0.0
    env_conf = float(env_row.get("env_confidence", 0.0)) if env_row else 0.0

    trend_state = str(trend_row.get("trend_state", "range")).lower() if trend_row else "range"
    trend_strength = float(trend_row.get("trend_strength", 0.0)) if trend_row else 0.0

    match_long = env_bias == "bull" and trend_state == "up"
    match_short = env_bias == "bear" and trend_state == "down"

    if match_long:
        regime_state = "risk_on_long"
        allowed = "long"
    elif match_short:
        regime_state = "risk_on_short"
        allowed = "short"
    else:
        regime_state = "no_trade"
        allowed = "none"

    if regime_state.startswith("risk_on") and env_conf >= 0.6 and trend_strength >= 0.6:
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
        "env_bias": env_bias,
        "env_score": env_score,
        "env_confidence": env_conf,
        "trend_state": trend_state,
        "trend_strength": trend_strength,
        "vol_state": "unknown",
        "risk_flags": [],
    }
    return regime_state, allowed, mult, score, reason, components


def _rows_for_theme(theme: str, symbols: List[str], env_map: Dict[str, Dict], trend_map: Dict[str, Dict], htf: str) -> List[Dict]:
    rows: List[Dict] = []
    for sym in symbols:
        env_row = env_map.get(sym)
        trend_row = trend_map.get(sym)
        regime_state, allowed, mult, score, reason, components = _regime_from(env_row, trend_row)
        asof = (
            env_row.get("asof_utc")
            if env_row and env_row.get("asof_utc")
            else (trend_row.get("asof_utc") if trend_row else pd.Timestamp.utcnow().strftime("%Y-%m-%dT%H:%M:%SZ"))
        )
        rows.append(
            {
                "asof_utc": asof,
                "symbol": sym,
                "theme": theme,
                "htf_timeframe": htf,
                "regime_state": regime_state,
                "regime_score": float(score),
                "allowed_direction": allowed,
                "position_multiplier": float(mult),
                "regime_reason": reason,
                "components": components,
                "links": {},
                "debug": {
                    "env_row": env_row,
                    "trend_row": trend_row,
                },
                "debug_json": json.dumps({"env_row": env_row, "trend_row": trend_row}, ensure_ascii=False),
            }
        )
    return rows


def main(argv: Optional[Sequence[str]] = None) -> int:
    ap = argparse.ArgumentParser(description="Step3 Regime (HTF-only) combining Env and Trend.")
    ap.add_argument("--themes", required=True, help="Comma-separated themes")
    ap.add_argument("--out", required=True, help="Output directory (expects Step1/Step3 env/trend)")
    ap.add_argument("--contracts", default="./contracts", help="Contracts directory for JSON Schema")
    ap.add_argument("--htf", default="1H", help="HTF timeframe label, e.g. 1H/4H/1D")
    ap.add_argument("--loglevel", default="INFO", help="Logging level")
    args = ap.parse_args(argv)

    logging.basicConfig(level=getattr(logging, args.loglevel.upper(), logging.INFO))

    themes = [t.strip().upper() for t in parse_csv_list(args.themes)]
    if not themes:
        raise SystemExit("Provide --themes")

    out_dir = ensure_dir(args.out)
    regime_dir = ensure_dir(out_dir / "step3_regime")

    _, rows_uni = load_universe(out_dir, themes=themes)
    sym_by_theme = group_symbols_by_theme(rows_uni)

    schema_path = Path(args.contracts) / "step3_regime.schema.json"

    for theme in themes:
        symbols = sym_by_theme.get(theme, [])
        if not symbols:
            LOG.warning("No symbols for theme=%s; skipping.", theme)
            continue

        env_map = _load_env(theme, out_dir, args.htf)
        trend_map = _load_trend(theme, out_dir, args.htf)

        rows = _rows_for_theme(theme, symbols, env_map, trend_map, args.htf)
        payload = {
            "schema_version": "3.regime.v1",
            "generated_at_utc": pd.Timestamp.utcnow().strftime("%Y-%m-%dT%H:%M:%SZ"),
            "htf_timeframe": args.htf,
            "theme": theme,
            "source": "step3_regime.py",
            "notes": f"Regime from HTF env+trend for theme={theme}",
            "rows": rows,
        }

        payload = must_validate(schema_path, payload)

        json_path = regime_dir / f"regime_{theme}_{args.htf}.json"
        csv_path = regime_dir / f"regime_{theme}_{args.htf}.csv"
        json_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
        pd.DataFrame(rows).to_csv(csv_path, index=False)

        LOG.info("Regime theme=%s rows=%d json=%s csv=%s", theme, len(rows), json_path, csv_path)
        print(f"[OK] regime theme={theme} json={json_path} csv={csv_path} rows={len(rows)}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
