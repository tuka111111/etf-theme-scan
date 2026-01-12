from __future__ import annotations

import csv
import json
import logging
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

from .common import ensure_dir
from .io_step6 import normalize_flags

LOG = logging.getLogger(__name__)

ACTIONS = {"ENTER", "WATCH", "SKIP", "EXIT", "ADD", "REDUCE", "AVOID"}
TRADE_ACTIONS = {"ENTER", "WATCH", "SKIP", "EXIT"}
OLD_TRADE_LOG_COLUMNS = [
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
COLUMNS = OLD_TRADE_LOG_COLUMNS + ["threshold_used", "rules_applied", "score_adjusted"]
OLD_TRADE_ACTION_COLUMNS = [
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
TRADE_ACTION_COLUMNS = [
    "schema_version",
    "asof_date_jst",
    "action_ts_jst",
    "created_at_jst",
    "decision_id",
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
    "status",
    "source",
    "run_id",
    "snapshot_id",
    "updated_from_ts_jst",
    "debug_json",
    "threshold_used",
    "rules_applied",
    "score_adjusted",
]


def _now_local_iso() -> str:
    return datetime.now().isoformat(timespec="seconds")


def _today_local() -> str:
    return datetime.now().date().isoformat()


def _now_jst() -> datetime:
    return datetime.now(timezone.utc).astimezone(timezone(timedelta(hours=9)))


def _parse_ts_jst(val: Optional[str]) -> datetime:
    if not val:
        return _now_jst()
    try:
        parsed = datetime.fromisoformat(val)
        if parsed.tzinfo is None:
            return parsed.replace(tzinfo=timezone(timedelta(hours=9)))
        return parsed.astimezone(timezone(timedelta(hours=9)))
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


def _upgrade_trade_log_header(csv_path: Path) -> None:
    if not csv_path.exists():
        return
    with csv_path.open(newline="", encoding="utf-8") as f:
        reader = csv.reader(f)
        header = next(reader, None)
    if not header:
        raise ValueError(f"Missing header in existing CSV: {csv_path}")
    if header == COLUMNS:
        return
    if header != OLD_TRADE_LOG_COLUMNS:
        raise ValueError(f"CSV header mismatch in {csv_path}. Expected={COLUMNS} or {OLD_TRADE_LOG_COLUMNS}")

    rows: List[Dict[str, str]] = []
    with csv_path.open(newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            rows.append(row)

    with csv_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=COLUMNS)
        writer.writeheader()
        for row in rows:
            writer.writerow(
                {
                    **{k: row.get(k, "") for k in OLD_TRADE_LOG_COLUMNS},
                    "threshold_used": row.get("threshold_used", ""),
                    "rules_applied": row.get("rules_applied", ""),
                    "score_adjusted": row.get("score_adjusted", ""),
                }
            )


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


def _upgrade_trade_actions_header(csv_path: Path) -> None:
    if not csv_path.exists():
        return
    with csv_path.open(newline="", encoding="utf-8") as f:
        reader = csv.reader(f)
        header = next(reader, None)
    if header == TRADE_ACTION_COLUMNS:
        return
    if header not in (OLD_TRADE_ACTION_COLUMNS, TRADE_ACTION_COLUMNS[:-3]):
        raise ValueError(
            f"CSV header mismatch in {csv_path}. Expected={TRADE_ACTION_COLUMNS} or {OLD_TRADE_ACTION_COLUMNS}"
        )

    rows: List[Dict[str, str]] = []
    with csv_path.open(newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            rows.append(row)

    with csv_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=TRADE_ACTION_COLUMNS)
        writer.writeheader()
        for row in rows:
            action_ts = row.get("action_ts_jst", "")
            writer.writerow(
                {
                    "schema_version": row.get("schema_version", "step7_trade_action_v1"),
                    "asof_date_jst": row.get("asof_date_jst", ""),
                    "action_ts_jst": action_ts,
                    "created_at_jst": row.get("created_at_jst", action_ts),
                    "decision_id": row.get("decision_id", ""),
                    "theme": row.get("theme", ""),
                    "symbol": row.get("symbol", ""),
                    "action": row.get("action", ""),
                    "notes": row.get("notes", ""),
                    "score_total": row.get("score_total", ""),
                    "env_bias": row.get("env_bias", ""),
                    "env_confidence": row.get("env_confidence", ""),
                    "etf_env_bias": row.get("etf_env_bias", ""),
                    "etf_env_confidence": row.get("etf_env_confidence", ""),
                    "flags": row.get("flags", ""),
                    "status": row.get("status", "active"),
                    "source": row.get("source", ""),
                    "run_id": row.get("run_id", ""),
                    "snapshot_id": row.get("snapshot_id", ""),
                    "updated_from_ts_jst": row.get("updated_from_ts_jst", ""),
                    "debug_json": row.get("debug_json", ""),
                    "threshold_used": row.get("threshold_used", ""),
                    "rules_applied": row.get("rules_applied", ""),
                    "score_adjusted": row.get("score_adjusted", ""),
                }
            )


def ensure_trade_actions_header(csv_path: Path) -> None:
    ensure_trade_actions_header(csv_path)


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
            "threshold_used": meta.get("threshold_used", ""),
            "rules_applied": ";".join(meta.get("rules_applied", [])) if meta.get("rules_applied") else "",
            "score_adjusted": meta.get("score_adjusted", ""),
        }

        trade_dir = ensure_dir(Path(out_dir) / "trade_log")
        csv_path = trade_dir / "trades.csv"
        monthly_path = trade_dir / f"trades_{record['date_local'].replace('-', '')[:6]}.csv"

        for path in (csv_path, monthly_path):
            _upgrade_trade_log_header(path)
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
    created_at = _parse_ts_jst(record.get("created_at_jst")) if record.get("created_at_jst") else _now_jst()
    created_at_jst = created_at.isoformat(timespec="seconds")
    asof_date_jst = action_ts.date().isoformat()

    flags_val = record.get("flags", "")
    if isinstance(flags_val, list):
        flags_val = ";".join([str(v) for v in flags_val if str(v).strip()])
    notes_val = str(record.get("notes", "") or "")
    if len(notes_val) > 255:
        raise ValueError("notes must be 255 characters or fewer")
    notes_val = notes_val.replace("\r", " ").replace("\n", " ")

    debug_json = record.get("debug_json", "")
    if isinstance(debug_json, (dict, list)):
        debug_json = json.dumps(debug_json, ensure_ascii=False)

    status_val = str(record.get("status", "active") or "active").lower()
    if status_val not in {"active", "edited", "obsolete"}:
        raise ValueError(f"Invalid status {status_val}. Allowed: active/edited/obsolete")

    row = {
        "schema_version": "step7_trade_action_v1",
        "asof_date_jst": asof_date_jst,
        "action_ts_jst": action_ts_jst,
        "created_at_jst": created_at_jst,
        "decision_id": record.get("decision_id", ""),
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
        "status": status_val,
        "source": record.get("source", "streamlit_decision_view"),
        "run_id": record.get("run_id", ""),
        "snapshot_id": record.get("snapshot_id", ""),
        "updated_from_ts_jst": record.get("updated_from_ts_jst", ""),
        "debug_json": debug_json or "",
        "threshold_used": record.get("threshold_used", ""),
        "rules_applied": record.get("rules_applied", ""),
        "score_adjusted": record.get("score_adjusted", ""),
    }

    out_path = ensure_dir(Path(out_dir) / "step7_trades")
    csv_path = out_path / f"trade_actions_{asof_date_jst}.csv"
    _upgrade_trade_actions_header(csv_path)
    _validate_csv_header(csv_path, TRADE_ACTION_COLUMNS)

    existing_rows: List[Dict[str, str]] = []
    if csv_path.exists():
        try:
            with csv_path.open(newline="", encoding="utf-8") as f:
                reader = csv.DictReader(f)
                existing_rows = list(reader)
        except Exception as e:
            raise ValueError(f"Failed to read existing trade actions: {e}")

    decision_id = str(row.get("decision_id") or "").strip()
    matched_active: List[Dict[str, str]] = []
    for r in existing_rows:
        r_symbol = str(r.get("symbol", "")).upper()
        r_date = str(r.get("asof_date_jst", ""))
        r_decision_id = str(r.get("decision_id", "") or "").strip()
        if r_symbol != symbol or r_date != asof_date_jst:
            continue
        if decision_id and r_decision_id != decision_id:
            continue
        r_status = str(r.get("status", "active") or "active").lower()
        if r_status in {"obsolete", "edited"}:
            continue
        matched_active.append(r)

    if status_val == "active":
        for r in matched_active:
            same_action = str(r.get("action", "")).upper() == action
            same_notes = str(r.get("notes", "") or "") == notes_val
            if same_action and same_notes:
                raise ValueError("Duplicate trade action detected for the same symbol/date/decision_id.")

    updated_from_ts = str(record.get("updated_from_ts_jst", "") or "")
    if not updated_from_ts and matched_active:
        sorted_matches = sorted(
            matched_active,
            key=lambda x: str(x.get("action_ts_jst", "")),
            reverse=True,
        )
        updated_from_ts = str(sorted_matches[0].get("action_ts_jst", ""))

    obsolete_rows: List[Dict[str, str]] = []
    if status_val == "active" and matched_active:
        for r in matched_active:
            obsolete_row = {k: r.get(k, "") for k in TRADE_ACTION_COLUMNS}
            obsolete_row["status"] = "obsolete"
            obsolete_row["updated_from_ts_jst"] = r.get("action_ts_jst", "")
            if not obsolete_row.get("created_at_jst"):
                obsolete_row["created_at_jst"] = r.get("action_ts_jst", "")
            obsolete_rows.append(obsolete_row)

    if updated_from_ts:
        row["updated_from_ts_jst"] = updated_from_ts

    write_header = not csv_path.exists()
    with csv_path.open("a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=TRADE_ACTION_COLUMNS)
        if write_header:
            writer.writeheader()
        for obsolete_row in obsolete_rows:
            writer.writerow(obsolete_row)
        writer.writerow(row)

    return str(csv_path)
