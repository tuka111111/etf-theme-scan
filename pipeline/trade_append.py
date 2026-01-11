from __future__ import annotations

import csv
import json
import logging
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional
from zoneinfo import ZoneInfo

from .common import ensure_dir
from .io_step6 import normalize_flags

LOG = logging.getLogger(__name__)

ACTIONS = {"ENTER", "WATCH", "SKIP", "EXIT", "ADD", "REDUCE", "AVOID"}
TRADE_ACTIONS = {"ENTER", "WATCH", "SKIP", "EXIT"}
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
TRADE_ACTION_COLUMNS = [
    "schema_version",
    "asof_date_jst",
    "action_ts_jst",
    "theme",
    "symbol",
    "action",
    "notes",
    "score_total",
    "env_bias",
    "env_confidence",
    "etf_env_bias",
    "etf_env_confidence",
    "flags",
    "source",
    "run_id",
    "snapshot_id",
    "debug_json",
]


def _now_local_iso() -> str:
    return datetime.now().isoformat(timespec="seconds")


def _today_local() -> str:
    return datetime.now().date().isoformat()


def _now_jst() -> datetime:
    return datetime.now(ZoneInfo("Asia/Tokyo"))


def _parse_ts_jst(val: Optional[str]) -> datetime:
    if not val:
        return _now_jst()
    try:
        parsed = datetime.fromisoformat(val)
        if parsed.tzinfo is None:
            return parsed.replace(tzinfo=ZoneInfo("Asia/Tokyo"))
        return parsed.astimezone(ZoneInfo("Asia/Tokyo"))
    except Exception:
        return _now_jst()


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


def _choose_symbol_interactive(candidates: List[tuple[str, Dict]]) -> str:
    print("Select a symbol:")
    for i, (_, meta) in enumerate(candidates, start=1):
        print(f"{i}: {meta.get('symbol')} ({meta.get('theme')}) score={meta.get('score_total')}")
    while True:
        choice = input("Enter number: ").strip()
        if not choice.isdigit():
            print("Enter a valid number.")
            continue
        idx = int(choice)
        if 1 <= idx <= len(candidates):
            return candidates[idx - 1][0]
        print("Out of range.")


def _read_existing(csv_path: Path) -> List[Dict[str, str]]:
    if not csv_path.exists():
        return []
    with csv_path.open(newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        return list(reader)


def _validate_csv_header(csv_path: Path, columns: List[str]) -> None:
    if not csv_path.exists():
        return
    with csv_path.open(newline="", encoding="utf-8") as f:
        reader = csv.reader(f)
        header = next(reader, None)
    if not header:
        raise ValueError(f"Missing header in existing CSV: {csv_path}")
    if header != columns:
        raise ValueError(f"CSV header mismatch in {csv_path}. Expected={columns} Actual={header}")


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
    row_meta: Optional[Dict] = None,
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

        decision = _load_decision(Path(decision_path)) if decision_path else {}
        idx = _build_symbol_index(decision) if decision else {}
        meta = idx.get(symbol_up) or (row_meta or {})
        if not meta:
            return {"ok": False, "error": f"Symbol {symbol_up} not found in decision picks and no row_meta provided"}

        risk_mode = decision.get("risk_mode", {}).get("mode", "") if decision else ""
        snapshot_id = (
            decision.get("asof_date_utc")
            or decision.get("asof_local")
            or decision.get("generated_at_utc", "")
            if decision
            else (row_meta.get("snapshot_id") if row_meta else "")
        )

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


def append_trade_action(out_dir: str, record: Dict[str, Any]) -> str:
    if not isinstance(record, dict):
        raise ValueError("record must be a dict")

    action = str(record.get("action", "")).upper().strip()
    if action not in TRADE_ACTIONS:
        raise ValueError(f"Invalid action {action}. Allowed: {sorted(TRADE_ACTIONS)}")

    symbol = str(record.get("symbol", "")).strip().upper()
    theme = str(record.get("theme", "")).strip().upper()
    if not symbol:
        raise ValueError("symbol is required")
    if not theme:
        raise ValueError("theme is required")

    action_ts = _parse_ts_jst(record.get("action_ts_jst"))
    action_ts_jst = action_ts.isoformat(timespec="seconds")
    asof_date_jst = action_ts.date().isoformat()

    flags_val = record.get("flags", "")
    if isinstance(flags_val, list):
        flags_val = ";".join([str(v) for v in flags_val if str(v).strip()])
    notes_val = record.get("notes", "")

    debug_json = record.get("debug_json", "")
    if isinstance(debug_json, (dict, list)):
        debug_json = json.dumps(debug_json, ensure_ascii=False)

    row = {
        "schema_version": "step7_trade_action_v1",
        "asof_date_jst": asof_date_jst,
        "action_ts_jst": action_ts_jst,
        "theme": theme,
        "symbol": symbol,
        "action": action,
        "notes": str(notes_val or ""),
        "score_total": record.get("score_total", ""),
        "env_bias": record.get("env_bias", ""),
        "env_confidence": record.get("env_confidence", ""),
        "etf_env_bias": record.get("etf_env_bias", ""),
        "etf_env_confidence": record.get("etf_env_confidence", ""),
        "flags": flags_val or "",
        "source": record.get("source", "streamlit_decision_view"),
        "run_id": record.get("run_id", ""),
        "snapshot_id": record.get("snapshot_id", ""),
        "debug_json": debug_json or "",
    }

    out_path = ensure_dir(Path(out_dir) / "step7_trades")
    csv_path = out_path / f"trade_actions_{asof_date_jst}.csv"
    _validate_csv_header(csv_path, TRADE_ACTION_COLUMNS)

    write_header = not csv_path.exists()
    with csv_path.open("a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=TRADE_ACTION_COLUMNS)
        if write_header:
            writer.writeheader()
        writer.writerow(row)

    return str(csv_path)
