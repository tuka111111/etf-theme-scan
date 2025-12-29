"""
Step2 wrapper that consumes Step1 outputs.

Reads step1_universe.json and computes HTF environment per theme/benchmark.
"""
from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

from step2_env import run_step2_for_theme, write_step2_outputs

LOG = logging.getLogger(__name__)


def _load_step1_universe(json_path: Path) -> List[Tuple[str, str]]:
    payload = json.loads(json_path.read_text(encoding="utf-8"))
    if not isinstance(payload, list):
        raise RuntimeError(f"Invalid Step1 JSON payload: {json_path}")

    out: List[Tuple[str, str]] = []
    for item in payload:
        if not isinstance(item, dict):
            continue
        theme = str(item.get("theme", "")).strip().upper()
        benchmark = str(item.get("benchmark", "")).strip().upper() or theme
        if not theme:
            continue
        out.append((theme, benchmark))
    return out


def _parse_theme_filter(raw: Optional[str]) -> Optional[List[str]]:
    if raw is None:
        return None
    vals = [t.strip().upper() for t in raw.replace(" ", ",").split(",") if t.strip()]
    if not vals:
        return None
    return list(dict.fromkeys(vals))


def main(argv: Optional[Sequence[str]] = None) -> int:
    ap = argparse.ArgumentParser(description="Step2: HTF env using Step1 outputs (step1_universe.json).")
    ap.add_argument("--step1-json", default=None, help="Path to step1_universe.json")
    ap.add_argument("--step1-out", default=None, help="Step1 output directory (contains step1_universe.json)")
    ap.add_argument("--out", default=None, help="Output directory for Step2 artifacts (default: step1-out)")
    ap.add_argument("--themes", default=None, help="Optional filter, comma-separated themes")
    ap.add_argument("--timeframe", default="1D", help='HTF timeframe label (default: "1D")')
    ap.add_argument("--loglevel", default="INFO", help="DEBUG/INFO/WARN/ERROR")
    args = ap.parse_args(argv)

    logging.basicConfig(level=getattr(logging, args.loglevel.upper(), logging.INFO))

    if args.step1_json:
        step1_json = Path(args.step1_json)
    elif args.step1_out:
        step1_json = Path(args.step1_out) / "step1_universe.json"
    else:
        raise SystemExit("Provide --step1-json or --step1-out")

    if not step1_json.exists():
        raise SystemExit(f"Missing Step1 JSON: {step1_json}")

    out_dir = Path(args.out) if args.out else (Path(args.step1_out) if args.step1_out else step1_json.parent)

    theme_filter = _parse_theme_filter(args.themes)
    pairs = _load_step1_universe(step1_json)
    if theme_filter is not None:
        pairs = [(t, b) for (t, b) in pairs if t in theme_filter]

    if not pairs:
        raise SystemExit("No themes found from Step1 outputs.")

    results = []
    for theme, benchmark in pairs:
        LOG.info("Classifying HTF env for theme=%s benchmark=%s timeframe=%s", theme, benchmark, args.timeframe)
        r = run_step2_for_theme(theme, benchmark=benchmark, timeframe=args.timeframe)
        results.append(r)
        LOG.info("Env=%s Allowed=%s", r.env, r.allowed)

    paths = write_step2_outputs(out_dir, results)
    LOG.info("Wrote step2 outputs: %s", paths)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
