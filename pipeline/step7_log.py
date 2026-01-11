from __future__ import annotations

import argparse
import json
import logging
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

import pandas as pd

from .common import ensure_dir

LOG = logging.getLogger(__name__)

ACTIONS = ["ENTER", "WATCH", "SKIP", "EXIT"]


def _now_jst() -> datetime:
    return datetime.now(timezone.utc).astimezone(timezone(timedelta(hours=9)))


def _load_trade_actions(trades_dir: Path) -> pd.DataFrame:
    files = sorted(trades_dir.glob("trade_actions_*.csv"))
    if not files:
        return pd.DataFrame()
    frames = []
    for path in files:
        try:
            frames.append(pd.read_csv(path))
        except Exception:
            LOG.warning("Failed to read %s", path)
    if not frames:
        return pd.DataFrame()
    return pd.concat(frames, ignore_index=True)


def _normalize_actions(df: pd.DataFrame, asof_date_jst: str) -> pd.DataFrame:
    if df.empty:
        return df

    df = df.copy()
    if "action" in df.columns:
        df["action"] = df["action"].astype(str).str.upper()
    if "status" in df.columns:
        df["status"] = df["status"].astype(str).str.lower()
        df = df[~df["status"].isin(["obsolete", "edited"])]

    if "action_ts_jst" in df.columns:
        parsed = pd.to_datetime(df["action_ts_jst"], errors="coerce", utc=True)
        parsed = parsed.dt.tz_convert("Asia/Tokyo")
        df["action_ts_jst_parsed"] = parsed
    else:
        df["action_ts_jst_parsed"] = pd.NaT

    fallback_date = pd.Timestamp(asof_date_jst, tz="Asia/Tokyo")
    df["action_ts_jst_parsed"] = df["action_ts_jst_parsed"].fillna(fallback_date)
    df["action_date"] = df["action_ts_jst_parsed"].dt.date
    return df


def _filter_window(df: pd.DataFrame, asof_date_jst: str, window_days: int) -> pd.DataFrame:
    if df.empty:
        return df
    asof_date = datetime.fromisoformat(asof_date_jst).date()
    start_date = asof_date - timedelta(days=max(window_days - 1, 0))
    return df[(df["action_date"] >= start_date) & (df["action_date"] <= asof_date)]


def _theme_action_counts(df: pd.DataFrame) -> Dict[str, Dict[str, int]]:
    if df.empty:
        return {}
    counts = df.groupby(["theme", "action"]).size().unstack(fill_value=0)
    out: Dict[str, Dict[str, int]] = {}
    for theme, row in counts.iterrows():
        out[str(theme)] = {action: int(row.get(action, 0)) for action in ACTIONS}
    return out


def _latest_actions_by_symbol(df: pd.DataFrame) -> List[Dict[str, Any]]:
    if df.empty:
        return []
    df_sorted = df.sort_values("action_ts_jst_parsed", ascending=False)
    latest = df_sorted.groupby("symbol", as_index=False).head(1)
    rows = []
    for _, row in latest.iterrows():
        rows.append(
            {
                "symbol": row.get("symbol", ""),
                "theme": row.get("theme", ""),
                "action": row.get("action", ""),
                "action_ts_jst": row.get("action_ts_jst", ""),
            }
        )
    return rows


def _watch_to_enter_counts(df: pd.DataFrame) -> Dict[str, Dict[str, float]]:
    if df.empty:
        return {}
    counts: Dict[str, Dict[str, float]] = {}
    grouped = df.sort_values("action_ts_jst_parsed").groupby(["theme", "symbol", "action_date"])
    for (theme, _symbol, _date), group in grouped:
        watch_ts = group.loc[group["action"] == "WATCH", "action_ts_jst_parsed"].min()
        enter_ts = group.loc[group["action"] == "ENTER", "action_ts_jst_parsed"].min()
        if pd.isna(watch_ts):
            continue
        if theme not in counts:
            counts[theme] = {"watch_count": 0, "watch_to_enter": 0, "rate": 0.0}
        counts[theme]["watch_count"] += 1
        if pd.notna(enter_ts) and enter_ts > watch_ts:
            counts[theme]["watch_to_enter"] += 1

    for theme, data in counts.items():
        watch_count = data["watch_count"]
        data["rate"] = float(data["watch_to_enter"]) / float(watch_count) if watch_count else 0.0
    return counts


def _top_enter_symbols(df: pd.DataFrame, top_n: int = 10) -> List[Dict[str, Any]]:
    if df.empty:
        return []
    enters = df[df["action"] == "ENTER"]
    if enters.empty:
        return []
    counts = enters.groupby(["symbol", "theme"]).size().reset_index(name="enter_count")
    counts = counts.sort_values(["enter_count", "symbol"], ascending=[False, True]).head(top_n)
    return counts.to_dict(orient="records")


def _write_json(path: Path, payload: Dict[str, Any]) -> None:
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def main(argv: Optional[Sequence[str]] = None) -> int:
    ap = argparse.ArgumentParser(description="Step7: aggregate trade action logs and rollups.")
    ap.add_argument("--out", required=True, help="Output directory root (expects out/step7_trades)")
    ap.add_argument("--window", type=int, default=14, help="Window days for rollup (deprecated)")
    ap.add_argument("--rolling-window", type=int, default=None, help="Window days for rollup")
    ap.add_argument("--loglevel", default="INFO")
    args = ap.parse_args(argv)

    logging.basicConfig(level=getattr(logging, args.loglevel.upper(), logging.INFO))

    out_root = Path(args.out)
    trades_dir = out_root / "step7_trades"
    logs_dir = ensure_dir(out_root / "step7_logs")

    asof_date_jst = _now_jst().date().isoformat()
    df = _load_trade_actions(trades_dir)
    df = _normalize_actions(df, asof_date_jst)
    window_days = args.rolling_window if args.rolling_window is not None else args.window
    df_window = _filter_window(df, asof_date_jst, window_days)

    if df.empty or "action_date" not in df.columns:
        daily_df = df
    else:
        daily_df = df[df["action_date"] == datetime.fromisoformat(asof_date_jst).date()]

    daily_summary = {
        "schema_version": "step7_daily_summary_v1",
        "asof_date_jst": asof_date_jst,
        "theme_action_counts": _theme_action_counts(daily_df),
        "latest_actions_by_symbol": _latest_actions_by_symbol(daily_df),
    }

    rollup = {
        "schema_version": "step7_rollup_v1",
        "asof_date_jst": asof_date_jst,
        "window_days": int(window_days),
        "theme_action_counts": _theme_action_counts(df_window),
        "top_enter_symbols": _top_enter_symbols(df_window),
        "watch_to_enter_rate_by_theme": _watch_to_enter_counts(df_window),
    }

    daily_path = logs_dir / f"daily_summary_{asof_date_jst}.json"
    rollup_path = logs_dir / f"rollup_{int(window_days)}d_{asof_date_jst}.json"
    trade_log_path = logs_dir / "trade_log.json"
    _write_json(daily_path, daily_summary)
    _write_json(rollup_path, rollup)
    _write_json(trade_log_path, rollup)

    LOG.info("Wrote daily summary: %s", daily_path)
    LOG.info("Wrote rollup: %s", rollup_path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
