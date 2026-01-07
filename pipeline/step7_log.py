from __future__ import annotations

import argparse
import json
import logging
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

import pandas as pd
from jsonschema import Draft202012Validator

from .common import ensure_dir

LOG = logging.getLogger(__name__)

VIOLATION_IDS = {
    "NEW_ENTRY_ON_RISK_OFF",
    "TRADE_OUTSIDE_TRADABLE_THEME",
    "ENTER_WITH_FLAGS_PRESENT",
    "MARKET_ORDER_NO_PLAN",
    "SCORE_ONLY_REASON",
}


def _now_utc_iso() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def _load_json(path: Path) -> Dict[str, Any]:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception as e:
        raise SystemExit(f"Failed to load json: {path} ({e})")


def _load_schema(path: Path) -> Draft202012Validator:
    try:
        schema = json.loads(path.read_text(encoding="utf-8"))
        return Draft202012Validator(schema)
    except Exception as e:
        raise SystemExit(f"Failed to load schema {path}: {e}")


def _normalize_flags(val) -> List[str]:
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
    for sep in ["|", ";", ","]:
        if sep in s:
            return [p.strip() for p in s.split(sep) if p.strip()]
    return [s]


def _extract_theme_env(decision: Dict[str, Any]) -> Dict[str, Dict[str, Any]]:
    themes = {}
    for row in decision.get("etf_daily_env", []):
        themes[row.get("theme", "unknown")] = {
            "env": row.get("env", "unknown"),
            "score": float(row.get("score", 0) or 0),
            "flags": row.get("flags", []),
        }
    return themes


def _collect_plan(decision: Dict[str, Any]) -> Dict[str, List[str]]:
    picks = decision.get("picks", {})
    return {
        "tradable_themes": decision.get("tradable_themes", []),
        "enter_symbols": [p["symbol"] for p in picks.get("ENTER", [])] if isinstance(picks, dict) else [],
        "watch_symbols": [p["symbol"] for p in picks.get("WATCH", [])] if isinstance(picks, dict) else [],
        "avoid_symbols": [p["symbol"] for p in picks.get("AVOID", [])] if isinstance(picks, dict) else [],
    }


def _append_jsonl(path: Path, record: Dict[str, Any]) -> None:
    with path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(record, ensure_ascii=False))
        f.write("\n")


def main(argv: Optional[Sequence[str]] = None) -> int:
    ap = argparse.ArgumentParser(description="Step7: append daily decision log (JSONL).")
    ap.add_argument("--out", required=True, help="Out directory root (expects step6 outputs in ./out/step6_decision)")
    ap.add_argument("--contracts", default="./contracts", help="Contracts directory")
    ap.add_argument("--date", required=True, help="Date (YYYY-MM-DD, local/JST)")
    ap.add_argument("--executed", required=True, help="Whether executed trades (true/false)")
    ap.add_argument("--touched", default="", help="Comma separated symbols touched (optional)")
    ap.add_argument("--trades-count", type=int, default=0, help="Number of trades executed")
    ap.add_argument("--deviation", required=True, help="Deviation occurred (true/false)")
    ap.add_argument("--violations", default="", help="Comma separated violation IDs (optional)")
    ap.add_argument("--emotion", default="", help="Emotion tag (optional)")
    ap.add_argument("--notes", default="", help="Notes (optional)")
    ap.add_argument("--decision", default="./out/step6_decision/decision_latest.json", help="Path to decision_latest.json")
    ap.add_argument("--rollup", default="./out/step6_decision/rollup_14d.json", help="Optional rollup file")
    ap.add_argument("--loglevel", default="INFO")
    args = ap.parse_args(argv)

    logging.basicConfig(level=getattr(logging, args.loglevel.upper(), logging.INFO))

    decision_path = Path(args.decision)
    decision = _load_json(decision_path)

    schema_path = Path(args.contracts) / "decision_daily_log.schema.json"
    validator = _load_schema(schema_path)

    asof_date = args.date
    asof_local = decision.get("asof_local", "")
    generated_at = _now_utc_iso()

    themes = _extract_theme_env(decision)
    plan = _collect_plan(decision)

    exec_bool = str(args.executed).lower() == "true"
    dev_bool = str(args.deviation).lower() == "true"

    violations = [v.strip() for v in args.violations.split(",") if v.strip()]
    # filter to known IDs if provided
    violations = [v for v in violations if not VIOLATION_IDS or v in VIOLATION_IDS]

    touched_symbols = [s.strip() for s in args.touched.split(",") if s.strip()]

    record: Dict[str, Any] = {
        "schema_version": "1.0.0",
        "date": asof_date,
        "generated_at_utc": generated_at,
        "risk_mode": decision.get("risk_mode", {}).get("mode", "unknown"),
        "risk_strength": float(decision.get("risk_mode", {}).get("strength", 0) or 0),
        "themes": themes,
        "plan": plan,
        "action": {
            "executed": exec_bool,
            "touched_symbols": touched_symbols,
            "trades_count": int(args.trades_count),
            "notes": args.notes,
        },
        "compliance": {
            "deviation": dev_bool,
            "violations": violations,
            "emotion_tag": args.emotion,
        },
        "source_files": {
            "decision_latest_json": str(decision_path),
            "rollup_14d_json": str(Path(args.rollup)) if Path(args.rollup).exists() else "",
        },
    }

    errors = sorted(validator.iter_errors(record), key=lambda e: e.path)
    if errors:
        lines = []
        for e in errors[:10]:
            path = "/".join([str(p) for p in e.absolute_path]) or "(root)"
            lines.append(f"{path}: {e.message}")
        more = "" if len(errors) <= 10 else f"... ({len(errors)-10} more)"
        raise SystemExit("Schema validation failed:\n" + "\n".join(lines) + more)

    logs_dir = ensure_dir(Path(args.out) / "logs")
    month = asof_date[:7] if len(asof_date) >= 7 else "unknown"
    jsonl_path = logs_dir / f"decision_daily_{month}.jsonl"
    _append_jsonl(jsonl_path, record)

    LOG.info("Appended decision daily log: %s", jsonl_path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
