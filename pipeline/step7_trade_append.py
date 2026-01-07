from __future__ import annotations

import argparse
import json
import logging
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Optional, Sequence

from jsonschema import Draft202012Validator

from .common import ensure_dir
from .io_step6 import normalize_flags

LOG = logging.getLogger(__name__)


def _now_utc_iso() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def _load_schema(path: Path) -> Draft202012Validator:
    try:
        schema = json.loads(path.read_text(encoding="utf-8"))
        return Draft202012Validator(schema)
    except Exception as e:
        raise SystemExit(f"Failed to load schema {path}: {e}")


def _append_jsonl(path: Path, record: Dict[str, Any]) -> None:
    with path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(record, ensure_ascii=False))
        f.write("\n")


def main(argv: Optional[Sequence[str]] = None) -> int:
    ap = argparse.ArgumentParser(description="Append a trade record to trades JSONL (manual/CLI).")
    ap.add_argument("--out", required=True, help="Output directory root (expects logs/)")
    ap.add_argument("--contracts", default="./contracts", help="Contracts directory")
    ap.add_argument("--date", required=True, help="YYYY-MM-DD")
    ap.add_argument("--theme", required=True)
    ap.add_argument("--symbol", required=True)
    ap.add_argument("--side", required=True, choices=["LONG", "SHORT"])
    ap.add_argument("--trade-id", required=True, help="Unique trade id, e.g., 2026-01-07_ABC_01")
    ap.add_argument("--entry-time-utc", required=True)
    ap.add_argument("--entry-price", type=float, required=True)
    ap.add_argument("--entry-reason", default="ENTER")
    ap.add_argument("--score-total", type=float, required=True)
    ap.add_argument("--flags", default="", help="Comma-separated flags at entry")
    ap.add_argument("--risk-mode", default="", help="Risk mode at entry (optional)")
    ap.add_argument("--theme-env", default="", help="Theme env at entry (optional)")
    ap.add_argument("--planned-r", type=float, default=None)
    ap.add_argument("--stop-price", type=float, default=None)
    ap.add_argument("--exit-time-utc", default=None)
    ap.add_argument("--exit-price", type=float, default=None)
    ap.add_argument("--exit-reason", default=None)
    ap.add_argument("--status", default="OPEN", choices=["OPEN", "CLOSED"])
    ap.add_argument("--pnl-pct", type=float, default=None)
    ap.add_argument("--r-multiple", type=float, default=None)
    ap.add_argument("--deviation", default="false")
    ap.add_argument("--violations", default="", help="Comma-separated violation IDs")
    ap.add_argument("--notes", default="")
    ap.add_argument("--loglevel", default="INFO")
    args = ap.parse_args(argv)

    logging.basicConfig(level=getattr(logging, args.loglevel.upper(), logging.INFO))

    schema_path = Path(args.contracts) / "trades_log.schema.json"
    validator = _load_schema(schema_path)

    flags = [f.strip() for f in args.flags.split(",") if f.strip()]
    record: Dict[str, Any] = {
        "schema_version": "1.0.0",
        "trade_id": args.trade_id,
        "date": args.date,
        "theme": args.theme,
        "symbol": args.symbol,
        "side": args.side,
        "entry": {
            "time_utc": args.entry_time_utc,
            "price": float(args.entry_price),
            "reason": args.entry_reason,
            "score_total": float(args.score_total),
            "flags": flags,
        },
        "exit": {
            "time_utc": args.exit_time_utc,
            "price": args.exit_price,
            "reason": args.exit_reason,
        },
        "risk": {
            "planned_r": args.planned_r,
            "stop_price": args.stop_price,
        },
        "result": {
            "status": args.status,
            "pnl_pct": args.pnl_pct,
            "r_multiple": args.r_multiple,
        },
        "compliance": {
            "deviation": str(args.deviation).lower() == "true",
            "violations": [v.strip() for v in args.violations.split(",") if v.strip()],
        },
        "notes": args.notes,
        "risk_mode_at_entry": args.risk_mode,
        "theme_env_at_entry": args.theme_env,
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
    month = args.date[:7] if len(args.date) >= 7 else "unknown"
    jsonl_path = logs_dir / f"trades_{month}.jsonl"
    _append_jsonl(jsonl_path, record)
    LOG.info("Appended trade log: %s", jsonl_path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
