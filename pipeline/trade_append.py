from __future__ import annotations

import csv
import json
import logging
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional

from .common import ensure_dir
from .io_step6 import normalize_flags

LOG = logging.getLogger(__name__)

ACTIONS = {"ENTER", "WATCH", "SKIP", "EXIT", "ADD", "REDUCE", "AVOID"}
COLUMNS = [
    "timestamp_local",
    "date_local",
    "symbol",
    "action",
    "theme",
    "risk_mode",
    "score_total",
    "env",
    "trend",
    "flags",
    "snapshot_id",
    "notes",
    "source",
    "debug_json",
]


def _now_local_iso() -> str:
    return datetime.now().isoformat(timespec="seconds")


def _today_local() -> str:
    return datetime.now().date().isoformat()


def _load_decision(decision_path: Path) -> Dict:
    if not decision_path.exists():
        raise FileNotFoundError(f"decision file not found: {decision_path}")
    return json.loads(decision_path.read_text(encoding="utf-8"))


def _build_symbol_index(decision: Dict) -> Dict[str, Dict]:
    idx: Dict[str, Dict] = {}
    picks = decision.get("picks", {})
    if isinstance(picks, dict):
        for bucket, rows in picks.items():
            if not isinstance(rows, list):
                continue
            for row in rows:
                sym = str(row.get("symbol", "")).upper()
                if not sym:
                    continue
                idx[sym] = {
                    "symbol": sym,
                    "theme": row.get("theme", ""),
                    "score_total": row.get("score_total", ""),
                    "env": row.get("env", ""),
                    "trend": row.get("trend", row.get("trend_direction", "")),
                    "flags": normalize_flags(row.get("flags")),
                    "bucket": bucket,
                }
    return idx


def _read_existing(csv_path: Path) -> List[Dict[str, str]]:
    if not csv_path.exists():
        return []
    with csv_path.open(newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        return list(reader)


def _is_duplicate(existing: List[Dict[str, str]], record: Dict[str, str]) -> bool:
    for row in existing:
        if (
            row.get("date_local") == record["date_local"]
            and row.get("symbol", "").upper() == record["symbol"].upper()
            and row.get("action", "").upper() == record["action"].upper()
            and row.get("snapshot_id", "") == record["snapshot_id"]
        ):
            return True
    return False


def append_trade(
    out_dir: str,
    symbol: str,
    action: str,
    notes: Optional[str] = None,
    date_local: Optional[str] = None,
    source: str = "cli",
    decision_path: str = "./out/step6_decision/decision_latest.json",
    extra: Optional[Dict] = None,
) -> Dict[str, object]:
    """
    Append a trade note to trades.csv with minimal inputs.
    Returns {"ok": True, "record": record, "path": str(csv_path)} on success,
            {"ok": False, "error": "..."} on failure.
    """
    try:
        action_up = action.upper()
        if action_up not in ACTIONS:
            return {"ok": False, "error": f"Invalid action {action}. Allowed: {sorted(ACTIONS)}"}

        symbol_up = symbol.strip().upper()
        if not symbol_up:
            return {"ok": False, "error": "Symbol is required"}

        decision = _load_decision(Path(decision_path))
        idx = _build_symbol_index(decision)
        meta = idx.get(symbol_up)
        if not meta:
            return {"ok": False, "error": f"Symbol {symbol_up} not found in decision picks"}

        risk_mode = decision.get("risk_mode", {}).get("mode", "")
        snapshot_id = decision.get("asof_date_utc") or decision.get("asof_local") or decision.get("generated_at_utc", "")

        record = {
            "timestamp_local": _now_local_iso(),
            "date_local": date_local or _today_local(),
            "symbol": symbol_up,
            "action": action_up,
            "theme": meta.get("theme", ""),
            "risk_mode": risk_mode,
            "score_total": meta.get("score_total", ""),
            "env": meta.get("env", ""),
            "trend": meta.get("trend", ""),
            "flags": ";".join(meta.get("flags", [])) if meta.get("flags") else "",
            "snapshot_id": snapshot_id,
            "notes": notes or "",
            "source": source,
            "debug_json": json.dumps(extra) if extra else "",
        }

        trade_dir = ensure_dir(Path(out_dir) / "trade_log")
        csv_path = trade_dir / "trades.csv"
        monthly_path = trade_dir / f"trades_{record['date_local'].replace('-', '')[:6]}.csv"

        for path in (csv_path, monthly_path):
            existing = _read_existing(path)
            if _is_duplicate(existing, record):
                return {"ok": False, "error": f"Duplicate entry detected in {path} for symbol={symbol_up} action={action_up} snapshot={snapshot_id}"}
            write_header = not path.exists()
            with path.open("a", newline="", encoding="utf-8") as f:
                writer = csv.DictWriter(f, fieldnames=COLUMNS)
                if write_header:
                    writer.writeheader()
                writer.writerow(record)

        return {"ok": True, "record": record, "path": str(csv_path)}
    except Exception as e:
        LOG.exception("append_trade failed")
        return {"ok": False, "error": str(e)}
