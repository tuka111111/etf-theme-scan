"""
Step3 Trend (HTF only)

Computes HTF trend state/strength per symbol using Step3 Env outputs as proxy.
"""
from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path
from typing import Dict, List, Optional, Sequence

import pandas as pd

from .common import ensure_dir, parse_csv_list
from .io_step1 import group_symbols_by_theme, load_universe
from .validate import must_validate

LOG = logging.getLogger(__name__)


def _read_env(theme: str, out_dir: Path, htf: str) -> Dict[str, Dict]:
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
        if sym:
            m[sym] = r
    return m


def _trend_from_env(env_row: Optional[Dict]) -> Dict[str, object]:
    if not env_row:
        return {
            "trend_state": "range",
            "trend_strength": 0.0,
            "trend_method": "env_proxy",
            "asof_utc": None,
            "debug": {"reason": "missing_env"},
        }
    bias = str(env_row.get("env_bias", "neutral")).lower()
    conf = float(env_row.get("env_confidence", 0.0))
    env_score = float(env_row.get("env_score", 0.0))

    if bias == "bull":
        trend_state = "up"
    elif bias == "bear":
        trend_state = "down"
    else:
        trend_state = "range"

    # Strength mixes env_score normalized and confidence
    strength = max(0.0, min(1.0, 0.5 + 0.003 * env_score + 0.5 * (conf - 0.5)))

    return {
        "trend_state": trend_state,
        "trend_strength": float(strength),
        "trend_method": "htf_env_proxy",
        "asof_utc": env_row.get("asof_utc"),
        "debug": {"source_env": env_row},
    }


def _rows_for_theme(theme: str, symbols: List[str], env_map: Dict[str, Dict], htf: str) -> List[Dict]:
    rows: List[Dict] = []
    for sym in symbols:
        t = _trend_from_env(env_map.get(sym))
        rows.append(
            {
                "asof_utc": t["asof_utc"] or pd.Timestamp.utcnow().strftime("%Y-%m-%dT%H:%M:%SZ"),
                "theme": theme,
                "symbol": sym,
                "htf_timeframe": htf,
                "trend_state": t["trend_state"],
                "trend_strength": t["trend_strength"],
                "trend_method": t["trend_method"],
                "features": {},
                "debug": t["debug"],
                "debug_json": json.dumps(t["debug"], ensure_ascii=False),
            }
        )
    return rows


def main(argv: Optional[Sequence[str]] = None) -> int:
    ap = argparse.ArgumentParser(description="Step3 Trend (HTF-only) based on Step3 Env proxies.")
    ap.add_argument("--themes", required=True, help="Comma-separated themes, e.g. XME,SMH,XBI")
    ap.add_argument("--out", required=True, help="Output directory (expects Step1/Step3 env artifacts)")
    ap.add_argument("--contracts", default="./contracts", help="Contracts directory for JSON Schema")
    ap.add_argument("--htf", default="1H", help="HTF timeframe label, e.g. 1H/4H/1D")
    ap.add_argument("--loglevel", default="INFO", help="Logging level")
    args = ap.parse_args(argv)

    logging.basicConfig(level=getattr(logging, args.loglevel.upper(), logging.INFO))

    themes = [t.strip().upper() for t in parse_csv_list(args.themes)]
    if not themes:
        raise SystemExit("Provide --themes")

    out_dir = ensure_dir(args.out)
    trend_dir = ensure_dir(out_dir / "step3_trend")

    _, rows_uni = load_universe(out_dir, themes=themes)
    sym_by_theme = group_symbols_by_theme(rows_uni)

    schema_path = Path(args.contracts) / "step3_trend.schema.json"

    for theme in themes:
        symbols = sym_by_theme.get(theme, [])
        if not symbols:
            LOG.warning("No symbols for theme=%s; skipping.", theme)
            continue

        try:
            env_map = _read_env(theme, out_dir, args.htf)
        except Exception as e:
            LOG.warning("Missing env for theme=%s htf=%s: %s", theme, args.htf, e)
            continue

        rows = _rows_for_theme(theme, symbols, env_map, args.htf)

        payload = {
            "schema_version": "3.trend.v1",
            "generated_at_utc": pd.Timestamp.utcnow().strftime("%Y-%m-%dT%H:%M:%SZ"),
            "htf_timeframe": args.htf,
            "theme": theme,
            "source": "step3_trend.py",
            "notes": f"HTF trend derived from Step3 env proxies for theme={theme}",
            "rows": rows,
        }

        payload = must_validate(schema_path, payload)

        json_path = trend_dir / f"trend_{theme}_{args.htf}.json"
        csv_path = trend_dir / f"trend_{theme}_{args.htf}.csv"

        json_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
        pd.DataFrame(rows).to_csv(csv_path, index=False)

        LOG.info("Trend theme=%s rows=%d json=%s csv=%s", theme, len(rows), json_path, csv_path)
        print(f"[OK] trend theme={theme} json={json_path} csv={csv_path} rows={len(rows)}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
