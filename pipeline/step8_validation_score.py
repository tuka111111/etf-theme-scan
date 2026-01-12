from __future__ import annotations

import argparse
import json
import logging
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence

import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))

from pipeline.common import ensure_dir

LOG = logging.getLogger(__name__)

DEFAULT_HORIZONS = [1, 5, 20]


def _now_utc_iso() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds").replace("+00:00", "Z")


def _parse_horizons(raw: str) -> List[int]:
    if not raw:
        return DEFAULT_HORIZONS
    parts = [p.strip() for p in raw.split(",") if p.strip()]
    horizons = []
    for p in parts:
        try:
            horizons.append(int(p))
        except Exception:
            raise SystemExit(f"Invalid horizon value: {p}")
    if not horizons:
        return DEFAULT_HORIZONS
    return sorted(set(horizons))


def _normalize_date(val) -> Optional[str]:
    try:
        ts = pd.to_datetime(val, errors="coerce")
        if pd.isna(ts):
            return None
        return ts.date().isoformat()
    except Exception:
        return None


def _find_date_col(df: pd.DataFrame) -> str:
    candidates = ["date", "asof_date", "asof", "datetime", "time"]
    for col in candidates:
        if col in df.columns:
            return col
    for col in df.columns:
        if "date" in col.lower():
            return col
    raise SystemExit("prices CSV missing date column")


def _find_close_col(df: pd.DataFrame) -> str:
    candidates = ["close", "Close", "adj_close", "adjClose", "price_close"]
    for col in candidates:
        if col in df.columns:
            return col
    for col in df.columns:
        if "close" in col.lower():
            return col
    raise SystemExit("prices CSV missing close column")


def _parse_flags(val) -> List[str]:
    if val is None:
        return []
    try:
        if pd.isna(val):
            return []
    except Exception:
        pass
    s = str(val).strip()
    if not s or s.lower() == "nan":
        return []
    for sep in [";", ",", "|"]:
        if sep in s:
            return [p.strip() for p in s.split(sep) if p.strip()]
    return [s]


def _score_bucket(score: float) -> Optional[str]:
    if score >= 80:
        return "score>=80"
    if 70 <= score <= 79.999:
        return "score 70-79"
    if 60 <= score <= 69.999:
        return "score 60-69"
    return None


def _load_prices(prices_path: Path) -> pd.DataFrame:
    df = pd.read_csv(prices_path)
    date_col = _find_date_col(df)
    close_col = _find_close_col(df)
    df = df[[date_col, close_col]].rename(columns={date_col: "date", close_col: "close"})
    df["date"] = df["date"].apply(_normalize_date)
    df = df.dropna(subset=["date"]).sort_values("date").reset_index(drop=True)
    df["close"] = pd.to_numeric(df["close"], errors="coerce")
    return df


def _build_price_index(df: pd.DataFrame) -> Dict[str, int]:
    index: Dict[str, int] = {}
    for i, d in enumerate(df["date"].tolist()):
        if d not in index:
            index[d] = i
    return index


def _iter_rows(df: pd.DataFrame) -> Iterable[Dict[str, object]]:
    for _, row in df.iterrows():
        yield row.to_dict()


def _aggregate(rows: List[Dict[str, object]]) -> pd.DataFrame:
    if not rows:
        return pd.DataFrame(
            columns=[
                "horizon",
                "group_type",
                "group_key",
                "avg_return",
                "win_rate",
                "sample",
                "median_return",
            ]
        )
    df = pd.DataFrame(rows)
    df["return"] = pd.to_numeric(df["return"], errors="coerce")
    df = df.dropna(subset=["return"])
    if df.empty:
        return pd.DataFrame(
            columns=[
                "horizon",
                "group_type",
                "group_key",
                "avg_return",
                "win_rate",
                "sample",
                "median_return",
            ]
        )
    grouped = df.groupby(["horizon", "group_type", "group_key"])
    out = grouped["return"].agg(
        avg_return="mean",
        median_return="median",
        sample="count",
    )
    win_rate = grouped["return"].apply(lambda s: float((s > 0).mean()))
    out["win_rate"] = win_rate
    out = out.reset_index()
    out = out[
        [
            "horizon",
            "group_type",
            "group_key",
            "avg_return",
            "win_rate",
            "sample",
            "median_return",
        ]
    ]
    return out


def main(argv: Optional[Sequence[str]] = None) -> int:
    ap = argparse.ArgumentParser(description="Step8: validate score effectiveness by theme ETF returns.")
    ap.add_argument("--decision_csv", required=True, help="Path to decision_latest.csv")
    ap.add_argument("--prices_dir", required=True, help="Directory containing prices_<THEME>_1D.csv")
    ap.add_argument("--out_dir", required=True, help="Output directory")
    ap.add_argument("--horizons", default=",".join(str(h) for h in DEFAULT_HORIZONS))
    ap.add_argument("--min_score", type=float, default=0.0)
    ap.add_argument("--progress", action="store_true")
    ap.add_argument("--loglevel", default="INFO")
    args = ap.parse_args(argv)

    logging.basicConfig(level=getattr(logging, args.loglevel.upper(), logging.INFO))

    decision_path = Path(args.decision_csv)
    if not decision_path.exists():
        raise SystemExit(f"decision csv not found: {decision_path}")

    decisions = pd.read_csv(decision_path)
    required = {"asof_date", "symbol", "theme", "score_total", "action", "flags", "env"}
    missing = required - set(decisions.columns)
    if missing:
        raise SystemExit(f"decision csv missing columns: {sorted(missing)}")

    decisions["action"] = decisions["action"].astype(str).str.upper()
    decisions = decisions[decisions["action"] == "BUY"].copy()
    decisions["score_total"] = pd.to_numeric(decisions["score_total"], errors="coerce")
    if args.min_score > 0:
        decisions = decisions[decisions["score_total"] >= args.min_score]

    if decisions.empty:
        LOG.warning("No BUY decisions found after filtering.")

    horizons = _parse_horizons(args.horizons)
    rows: List[Dict[str, object]] = []
    prices_dir = Path(args.prices_dir)

    for theme in sorted(decisions["theme"].dropna().unique().tolist()):
        theme_str = str(theme)
        prices_path = prices_dir / f"prices_{theme_str.upper()}_1D.csv"
        if not prices_path.exists():
            LOG.warning("prices file not found for theme=%s path=%s", theme_str, prices_path)
            continue
        try:
            prices_df = _load_prices(prices_path)
        except Exception as e:
            LOG.warning("failed to load prices for theme=%s: %s", theme_str, e)
            continue

        if prices_df.empty:
            LOG.warning("prices file empty for theme=%s", theme_str)
            continue

        price_index = _build_price_index(prices_df)
        closes = prices_df["close"].tolist()

        theme_decisions = decisions[decisions["theme"] == theme].copy()
        if args.progress:
            LOG.info("theme=%s decisions=%s", theme_str, len(theme_decisions))

        for row in _iter_rows(theme_decisions):
            asof_date = _normalize_date(row.get("asof_date"))
            if not asof_date:
                continue
            idx0 = price_index.get(asof_date)
            if idx0 is None:
                continue
            p0 = closes[idx0]
            if p0 is None or pd.isna(p0) or p0 == 0:
                continue

            score_val = float(row.get("score_total", 0.0) or 0.0)
            score_bucket = _score_bucket(score_val)
            env_val = str(row.get("env", "unknown") or "unknown").lower()
            flags = _parse_flags(row.get("flags"))
            flag_list = flags if flags else ["none"]

            for horizon in horizons:
                idxn = idx0 + horizon
                if idxn >= len(closes):
                    continue
                pn = closes[idxn]
                if pn is None or pd.isna(pn) or pn == 0:
                    continue
                ret = (pn / p0) - 1.0

                if score_bucket:
                    rows.append(
                        {
                            "asof_date": asof_date,
                            "horizon": horizon,
                            "group_type": "score_bucket",
                            "group_key": score_bucket,
                            "return": ret,
                        }
                    )
                rows.append(
                    {
                        "asof_date": asof_date,
                        "horizon": horizon,
                        "group_type": "env",
                        "group_key": env_val,
                        "return": ret,
                    }
                )
                for flag in flag_list:
                    rows.append(
                        {
                            "asof_date": asof_date,
                            "horizon": horizon,
                            "group_type": "flag",
                            "group_key": flag,
                            "return": ret,
                        }
                    )

    out_dir = ensure_dir(Path(args.out_dir))
    result = _aggregate(rows)
    out_csv = out_dir / "score_validation.csv"
    result.to_csv(out_csv, index=False)

    meta = {
        "schema_version": "step8_score_validation_v1",
        "generated_at_utc": _now_utc_iso(),
        "decision_csv": str(decision_path),
        "prices_dir": str(prices_dir),
        "horizons": horizons,
        "min_score": args.min_score,
        "decision_rows": int(len(decisions)),
        "result_rows": int(len(result)),
    }
    (out_dir / "validation_meta.json").write_text(json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")

    rationale_path = out_dir / "score_threshold_rationale.md"
    rationale_lines = [
        "# Score Validation Summary",
        f"generated_at_utc: {meta['generated_at_utc']}",
        "",
        f"decision_rows: {meta['decision_rows']}",
        f"result_rows: {meta['result_rows']}",
        f"horizons: {meta['horizons']}",
        "",
        "Buckets: score_bucket / env / flag",
    ]
    rationale_path.write_text("\n".join(rationale_lines) + "\n", encoding="utf-8")

    LOG.info("wrote %s", out_csv)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
