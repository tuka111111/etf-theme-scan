from __future__ import annotations

import argparse
import json
import logging
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))

from pipeline.common import ensure_dir

LOG = logging.getLogger(__name__)

SCORE_KEYS = [
    (80, "score>=80"),
    (70, "score 70-79"),
    (60, "score 60-69"),
]


def _now_utc_iso() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds").replace("+00:00", "Z")


def _parse_horizons(raw: str) -> List[int]:
    parts = [p.strip() for p in raw.split(",") if p.strip()]
    horizons = []
    for p in parts:
        try:
            horizons.append(int(p))
        except Exception:
            raise SystemExit(f"Invalid horizon value: {p}")
    if not horizons:
        raise SystemExit("Provide --horizons")
    return sorted(set(horizons))


def _require_cols(df: pd.DataFrame, cols: List[str]) -> None:
    missing = [c for c in cols if c not in df.columns]
    if missing:
        raise SystemExit(f"input CSV missing columns: {missing}")


def _select_row(df: pd.DataFrame, horizon: int, group_key: str) -> Optional[pd.Series]:
    sub = df[(df["horizon"] == horizon) & (df["group_type"] == "score_bucket") & (df["group_key"] == group_key)]
    if sub.empty:
        return None
    sub = sub.sort_values("sample", ascending=False)
    return sub.iloc[0]


def _meets_criteria(row: pd.Series, min_sample: int, min_win_rate: float) -> bool:
    return (
        float(row.get("avg_return", 0.0)) > 0.0
        and float(row.get("win_rate", 0.0)) >= min_win_rate
        and float(row.get("sample", 0.0)) >= min_sample
    )


def _fallback_row(rows: List[Tuple[int, pd.Series]]) -> Tuple[int, pd.Series]:
    return max(rows, key=lambda x: (float(x[1].get("sample", 0.0)), x[0]))


def main(argv: Optional[Sequence[str]] = None) -> int:
    ap = argparse.ArgumentParser(description="Step9-A: finalize score thresholds by horizon.")
    ap.add_argument("--input", required=True, help="Path to score_validation.csv")
    ap.add_argument("--horizons", default="1,5,20")
    ap.add_argument("--min_sample", type=int, default=30)
    ap.add_argument("--min_win_rate", type=float, default=0.55)
    ap.add_argument("--out", required=True, help="Output JSON path (e.g., out/step9/thresholds_final.json)")
    ap.add_argument("--loglevel", default="INFO")
    args = ap.parse_args(argv)

    logging.basicConfig(level=getattr(logging, args.loglevel.upper(), logging.INFO))

    input_path = Path(args.input)
    if not input_path.exists():
        raise SystemExit(f"input CSV not found: {input_path}")

    df = pd.read_csv(input_path)
    _require_cols(df, ["horizon", "group_type", "group_key", "avg_return", "win_rate", "sample"])

    df["horizon"] = pd.to_numeric(df["horizon"], errors="coerce")
    df["avg_return"] = pd.to_numeric(df["avg_return"], errors="coerce")
    df["win_rate"] = pd.to_numeric(df["win_rate"], errors="coerce")
    df["sample"] = pd.to_numeric(df["sample"], errors="coerce")

    horizons = _parse_horizons(args.horizons)
    if 20 not in horizons:
        LOG.warning("horizon 20 not included in --horizons")

    output: Dict[str, Dict[str, object]] = {}
    rationale_lines = [
        "# Step9-A Threshold Rationale",
        f"generated_at_utc: {_now_utc_iso()}",
        f"min_sample: {args.min_sample}",
        f"min_win_rate: {args.min_win_rate}",
        "",
    ]

    for horizon in horizons:
        horizon_rows: List[Tuple[int, pd.Series]] = []
        for threshold, key in SCORE_KEYS:
            row = _select_row(df, horizon, key)
            if row is not None:
                horizon_rows.append((threshold, row))

        if not horizon_rows:
            rationale_lines.append(f"- horizon {horizon}: no score_bucket rows found")
            continue

        horizon_rows = sorted(horizon_rows, key=lambda x: x[0])
        chosen: Optional[Tuple[int, pd.Series]] = None
        for threshold, row in horizon_rows:
            if _meets_criteria(row, args.min_sample, args.min_win_rate):
                chosen = (threshold, row)
                break

        if chosen is None:
            chosen = _fallback_row(horizon_rows)
            rationale_lines.append(
                f"- horizon {horizon}: criteria unmet; fallback to score>={chosen[0]} (max sample or higher threshold)"
            )
        else:
            rationale_lines.append(f"- horizon {horizon}: selected score>={chosen[0]} (meets criteria)")

        threshold, row = chosen
        output[str(horizon)] = {
            "threshold": threshold,
            "avg_return": float(row.get("avg_return", 0.0)),
            "win_rate": float(row.get("win_rate", 0.0)),
            "sample": int(row.get("sample", 0)),
        }

    out_path = Path(args.out)
    ensure_dir(out_path.parent)
    payload = {
        "schema_version": "step9_thresholds_v1",
        "generated_at_utc": _now_utc_iso(),
        "horizons": output,
    }
    out_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    rationale_path = out_path.parent / "thresholds_rationale.md"
    rationale_path.write_text("\n".join(rationale_lines) + "\n", encoding="utf-8")

    LOG.info("wrote %s", out_path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
