from __future__ import annotations

import argparse
import json
import logging
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Optional, Sequence

import pandas as pd

from .common import ensure_dir
from .io_step6 import load_dashboard, normalize_flags

LOG = logging.getLogger(__name__)

MAJOR_EXCLUDE_FLAGS = {"data_quality_low", "halted", "bad_symbol", "parse_error"}
WATCH_FLAGS = {"trend_not_strong_for_signal", "vol_too_low", "extended_move"}


def _now_utc_iso() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def _asof_date_utc(asof_utc: str) -> str:
    if not isinstance(asof_utc, str):
        return "unknown"
    try:
        ts = pd.to_datetime(asof_utc, utc=True)
        return ts.date().isoformat()
    except Exception:
        return "unknown"


def _scale_0_100(val) -> float:
    try:
        v = float(val)
        if pd.isna(v):
            return 0.0
        if v <= 1.2:  # treat as 0-1 scale
            v = v * 100.0
        return round(v, 2)
    except Exception:
        return 0.0


def _detect_etf_score_scale(scores: pd.Series) -> pd.Series:
    if scores.dropna().empty:
        return scores
    maxv = scores.max()
    if maxv is not None and pd.notna(maxv) and maxv <= 1.0:
        return scores * 100.0
    return scores


def _risk_mode(etf_df: pd.DataFrame) -> Dict[str, object]:
    scores = pd.to_numeric(etf_df["score"], errors="coerce")
    scores = _detect_etf_score_scale(scores)
    avg = scores.mean() if not scores.dropna().empty else 0.0
    if avg >= 70:
        mode = "RISK_ON"
    elif avg <= 40:
        mode = "RISK_OFF"
    else:
        mode = "NEUTRAL"
    return {"mode": mode, "strength": float(round(avg, 2)), "reasons": []}


def _build_etf_env(etf_rows: pd.DataFrame) -> pd.DataFrame:
    df = etf_rows.copy()
    df["score"] = pd.to_numeric(df.get("env_score", df.get("score_total", df.get("score"))), errors="coerce")
    df["score"] = _detect_etf_score_scale(df["score"])
    df["env"] = df.get("env", df.get("etf_env_bias", "unknown"))
    return df[["theme", "env", "score", "flags_list"]]


def _tradable_themes(etf_df: pd.DataFrame, risk_mode: str) -> List[str]:
    if risk_mode == "RISK_OFF":
        return []
    return etf_df[(etf_df["env"].str.lower() == "bull") & (etf_df["score"] >= 60)]["theme"].tolist()


def _picks(sym_rows: pd.DataFrame, tradable: List[str], score_col: str, min_score: float, top_n: int) -> Dict[str, List[Dict]]:
    df = sym_rows.copy()
    df = df[df["theme"].isin(tradable)]
    df = df[pd.to_numeric(df[score_col], errors="coerce") >= min_score]
    df = df.sort_values([score_col, "symbol"], ascending=[False, True]).head(top_n)

    buckets = {"ENTER": [], "WATCH": [], "AVOID": []}
    for _, row in df.iterrows():
        flags = normalize_flags(row.get("flags"))
        action = "ENTER"
        notes: List[str] = []
        if any(f in MAJOR_EXCLUDE_FLAGS for f in flags):
            action = "AVOID"
            notes.append("data issue")
        elif any(f in WATCH_FLAGS for f in flags):
            action = "WATCH"
            notes.append("monitor flag")
        if not notes:
            notes.append("clean")
        buckets[action].append(
            {
                "symbol": row.get("symbol", "unknown"),
                "theme": row.get("theme", "unknown"),
                "score_total": _scale_0_100(row.get(score_col, 0.0)),
                "env": row.get("env", "unknown"),
                "trend": row.get("trend", row.get("trend_direction", "unknown")),
                "action": action,
                "flags": flags,
                "notes": notes,
            }
        )
    return buckets


def _warnings(df: pd.DataFrame, top_n: int = 10) -> Dict[str, int]:
    flags = df["flags"].apply(normalize_flags)
    flat = pd.Series([f for sub in flags for f in sub if f])
    counts = flat.value_counts().head(top_n)
    return {k: int(v) for k, v in counts.items()}


def _render_md(payload: Dict[str, object]) -> str:
    lines: List[str] = []
    lines.append("# Daily Decision")
    lines.append("## Snapshot")
    lines.append(f"- asof_date_utc: {payload['asof_date_utc']}")
    lines.append(f"- asof_local: {payload['asof_local']}")
    lines.append(f"- generated_at_utc: {payload['generated_at_utc']}")
    lines.append(f"- themes: {', '.join(payload['themes'])}")
    lines.append(f"- risk_mode: {payload['risk_mode']['mode']} (strength={payload['risk_mode']['strength']})")
    lines.append("")
    lines.append("## ETF Daily Env")
    for row in payload["etf_daily_env"]:
        lines.append(
            f"- {row['theme']}: env={row['env']} score={row['score']} flags={';'.join(row['flags']) if row['flags'] else 'none'}"
        )
    lines.append("")
    lines.append("## Tradable Themes")
    trad = payload["tradable_themes"]
    lines.append("- " + (", ".join(trad) if trad else "none"))
    lines.append("")
    lines.append("## Picks")
    picks = payload["picks"]
    for bucket in ["ENTER", "WATCH", "AVOID"]:
        lines.append(f"- {bucket}:")
        rows: List[Dict] = picks.get(bucket, []) if isinstance(picks, dict) else []
        if not rows:
            lines.append("  - none")
        else:
            for p in rows:
                lines.append(
                    f"  - {p['symbol']} ({p['theme']}) score={p['score_total']} env={p['env']} trend={p['trend']} flags={';'.join(p['flags']) if p['flags'] else 'none'} notes={';'.join(p['notes']) if p['notes'] else 'none'}"
                )
    lines.append("")
    lines.append("## Warnings")
    warns: Dict[str, int] = payload["warnings"]
    if not warns:
        lines.append("- none")
    else:
        for k, v in warns.items():
            lines.append(f"- {k}: {v}")
    lines.append("")
    lines.append("## Debug")
    for k, v in payload["debug"].items():
        lines.append(f"- {k}: {v}")
    return "\n".join(lines)


def _save_files(out_dir: Path, date_str: str, payload: Dict[str, object]) -> None:
    json_path = out_dir / f"decision_{date_str}.json"
    md_path = out_dir / f"decision_{date_str}.md"
    json_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    md_path.write_text(_render_md(payload), encoding="utf-8")
    (out_dir / "decision_latest.json").write_bytes(json_path.read_bytes())
    (out_dir / "decision_latest.md").write_bytes(md_path.read_bytes())
    hist_dir = ensure_dir(out_dir / "history")
    (hist_dir / f"decision_{date_str}.json").write_bytes(json_path.read_bytes())


def _load_history(hist_dir: Path, limit_days: int = 14) -> List[Dict]:
    if not hist_dir.exists():
        return []
    files = sorted(hist_dir.glob("decision_*.json"))
    records: List[Dict] = []
    for f in files:
        try:
            payload = json.loads(f.read_text(encoding="utf-8"))
            date_part = f.stem.replace("decision_", "")
            records.append({"date": date_part, "payload": payload})
        except Exception:
            continue
    records = sorted(records, key=lambda r: r["date"])[-limit_days:]
    return records


def _build_rollup(records: List[Dict]) -> Dict[str, object]:
    risk_trail = []
    tradable_freq: Dict[str, int] = {}
    warnings_accum: Dict[str, int] = {}
    picks_trail = []
    for rec in records:
        p = rec["payload"]
        date = rec["date"]
        rm = p.get("risk_mode", {}).get("mode", "unknown")
        risk_trail.append({"date": date, "mode": rm, "strength": p.get("risk_mode", {}).get("strength")})
        for t in p.get("tradable_themes", []):
            tradable_freq[t] = tradable_freq.get(t, 0) + 1
        for k, v in p.get("warnings", {}).items():
            warnings_accum[k] = warnings_accum.get(k, 0) + int(v)
        actions = {"ENTER": 0, "WATCH": 0, "AVOID": 0}
        picks = p.get("picks", {})
        for act in actions.keys():
            actions[act] = len(picks.get(act, [])) if isinstance(picks, dict) else 0
        picks_trail.append({"date": date, **actions})
    warnings_top = sorted(warnings_accum.items(), key=lambda kv: kv[1], reverse=True)[:5]
    return {
        "schema_version": "1.0.0",
        "generated_at_utc": _now_utc_iso(),
        "risk_mode_trail": risk_trail,
        "tradable_freq": tradable_freq,
        "picks_by_action": picks_trail,
        "warnings_top": [{"flag": k, "count": v} for k, v in warnings_top],
    }


def main(argv: Optional[Sequence[str]] = None) -> int:
    ap = argparse.ArgumentParser(description="Step6 decision (rule-based, no LLM).")
    ap.add_argument("--out", required=True, help="Output directory root (e.g., ./out)")
    ap.add_argument("--contracts", default="./contracts", help="Contracts directory")
    ap.add_argument("--themes", default=None, help="Comma separated themes to include (default: all)")
    ap.add_argument("--min-score", type=float, default=80.0, help="Minimum score_total to include a symbol")
    ap.add_argument("--top-n", type=int, default=10, help="Number of symbols to include")
    ap.add_argument("--score-col", default="score_total", help="Score column name for symbols")
    ap.add_argument("--dashboard", default="./out/step5_dashboard/dashboard.csv", help="Path to dashboard.csv")
    ap.add_argument("--loglevel", default="INFO")
    args = ap.parse_args(argv)

    logging.basicConfig(level=getattr(logging, args.loglevel.upper(), logging.INFO))

    dash_path = Path(args.dashboard)
    if not dash_path.exists():
        raise SystemExit(f"dashboard not found: {dash_path}")

    try:
        df, etf_rows, sym_rows, asof_local, asof_utc = load_dashboard(dash_path, args.score_col)
    except Exception as e:
        raise SystemExit(f"Failed to load dashboard: {e}")

    if args.themes:
        themes_filter = {t.strip() for t in args.themes.split(",") if t.strip()}
        etf_rows = etf_rows[etf_rows["theme"].isin(themes_filter)]
        sym_rows = sym_rows[sym_rows["theme"].isin(themes_filter)]
        if etf_rows.empty:
            raise SystemExit("No ETF rows after theme filter.")

    etf_env_df = _build_etf_env(etf_rows)
    risk = _risk_mode(etf_env_df)
    tradable = _tradable_themes(etf_env_df, risk["mode"])
    picks = _picks(sym_rows, tradable, args.score_col, args.min_score, args.top_n)
    warnings = _warnings(df)

    asof_date_utc = _asof_date_utc(asof_utc)
    date_str = asof_date_utc if asof_date_utc != "unknown" else (asof_local[:10] if isinstance(asof_local, str) and len(asof_local) >= 10 else "unknown")

    payload: Dict[str, object] = {
        "schema_version": "1.0.0",
        "generated_at_utc": _now_utc_iso(),
        "asof_date_utc": asof_date_utc,
        "asof_local": asof_local,
        "themes": sorted(etf_env_df["theme"].unique().tolist()),
        "risk_mode": risk,
        "etf_daily_env": [
            {
                "theme": row["theme"],
                "env": row["env"],
                "score": _scale_0_100(row["score"]),
                "flags": row["flags_list"],
            }
            for _, row in etf_env_df.iterrows()
        ],
        "tradable_themes": tradable,
        "picks": picks,
        "warnings": warnings,
        "debug": {
            "input_path": str(dash_path),
            "score_col": args.score_col,
            "min_score": args.min_score,
            "top_n": args.top_n,
        },
    }

    out_root = ensure_dir(Path(args.out) / "step6_decision")
    try:
        _save_files(out_root, date_str, payload)
    except Exception as e:
        raise SystemExit(f"Failed to write decision outputs: {e}")

    # rollup 14d
    history_records = _load_history(out_root / "history", limit_days=14)
    if history_records:
        rollup = _build_rollup(history_records)
        (out_root / "rollup_14d.json").write_text(json.dumps(rollup, ensure_ascii=False, indent=2), encoding="utf-8")

    LOG.info("decision rows=%s risk_mode=%s tradable=%s written=%s", len(picks), risk["mode"], tradable, out_root)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
