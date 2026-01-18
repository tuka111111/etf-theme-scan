from __future__ import annotations

import argparse
import csv
import hashlib
import json
import logging
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Optional, Sequence

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))

from pipeline.common import ensure_dir

LOG = logging.getLogger(__name__)


def _now_utc_iso() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds").replace("+00:00", "Z")


def _safe_read_json(path: Path) -> tuple[Optional[dict], str]:
    if not path.exists():
        return None, "missing"
    try:
        return json.loads(path.read_text(encoding="utf-8")), "ok"
    except Exception:
        return None, "invalid"


def _asof_date_utc(payload: Optional[dict]) -> str:
    if not payload:
        return datetime.now(timezone.utc).date().isoformat()
    val = payload.get("asof_date_utc")
    if isinstance(val, str) and val:
        return val
    asof_utc = payload.get("asof_utc")
    if isinstance(asof_utc, str) and asof_utc:
        try:
            ts = datetime.fromisoformat(asof_utc.replace("Z", "+00:00"))
            return ts.date().isoformat()
        except Exception:
            pass
    return datetime.now(timezone.utc).date().isoformat()


def _decision_symbols(payload: dict, bucket: str) -> List[str]:
    picks = payload.get("picks", {}) if payload else {}
    rows = picks.get(bucket, []) if isinstance(picks, dict) else []
    out: List[str] = []
    for row in rows:
        sym = str(row.get("symbol", "")).upper()
        if sym:
            out.append(sym)
    return out


def _decision_enter_candidates(payload: dict) -> List[dict]:
    picks = payload.get("picks", {}) if payload else {}
    rows = picks.get("ENTER", []) if isinstance(picks, dict) else []
    out: List[dict] = []
    for row in rows:
        sym = str(row.get("symbol", "")).upper()
        if not sym:
            continue
        out.append(
            {
                "symbol": sym,
                "theme": row.get("theme", ""),
                "score_total": row.get("score_total", ""),
                "score_adjusted": row.get("score_adjusted", ""),
                "env": row.get("env", ""),
                "trend": row.get("trend", ""),
                "flags": row.get("flags", []),
                "threshold_used": row.get("threshold_used", ""),
                "rules_applied": row.get("rules_applied", []),
            }
        )
    return out


def _decision_hash(asof_date: str, enter_symbols: List[str]) -> str:
    data = asof_date + ":" + ",".join(sorted(set(enter_symbols)))
    return hashlib.sha1(data.encode("utf-8")).hexdigest()


def _read_trades(trades_dir: Path, asof_date_utc: str) -> List[Dict[str, str]]:
    trades: List[Dict[str, str]] = []
    for path in sorted(trades_dir.glob("trade_actions_*.csv")):
        try:
            with path.open(newline="", encoding="utf-8") as f:
                reader = csv.DictReader(f)
                if not reader.fieldnames:
                    continue
                if "symbol" not in reader.fieldnames or "action" not in reader.fieldnames:
                    LOG.warning("skip trade file missing columns: %s", path)
                    continue
                for row in reader:
                    trades.append(row)
        except Exception as e:
            LOG.warning("failed to read trades file %s: %s", path, e)
            continue

    today_trades: List[Dict[str, str]] = []
    asof_date_jst = asof_date_utc
    for row in trades:
        status = str(row.get("status", "")).lower()
        if status in {"obsolete", "edited"}:
            continue
        action_date = str(row.get("asof_date_jst", "")).strip()
        if not action_date:
            action_ts = str(row.get("action_ts_jst", "")).strip()
            if action_ts:
                action_date = action_ts.split("T")[0] if "T" in action_ts else action_ts
            else:
                action_date = asof_date_utc
        if action_date == asof_date_jst:
            today_trades.append(row)
    return today_trades


def _read_alerts(alerts_path: Path) -> set[str]:
    keys: set[str] = set()
    if not alerts_path.exists():
        return keys
    with alerts_path.open(encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                payload = json.loads(line)
                key = payload.get("dedup_key") or payload.get("event_id")
                if key:
                    keys.add(str(key))
            except Exception:
                continue
    return keys


def _append_alert(alerts_path: Path, dedup_key: str, kind: str, payload: dict, existing: set[str]) -> None:
    if dedup_key in existing:
        return
    record = {
        "ts_utc": _now_utc_iso(),
        "asof_date_utc": payload.get("asof_date_utc", ""),
        "dedup_key": dedup_key,
        "kind": kind,
        "payload": payload,
        "result": "logged_only",
    }
    with alerts_path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(record, ensure_ascii=False) + "\n")
    existing.add(dedup_key)


def main(argv: Optional[Sequence[str]] = None) -> int:
    ap = argparse.ArgumentParser(description="Step10 daily runner (summary + deviation + alerts)")
    ap.add_argument("--out", required=True, help="Out directory root")
    ap.add_argument("--window-days", type=int, default=7, help="Deviation window (placeholder)")
    ap.add_argument("--loglevel", default="INFO")
    args = ap.parse_args(argv)

    logging.basicConfig(level=getattr(logging, args.loglevel.upper(), logging.INFO))

    out_root = Path(args.out)
    decision_path = out_root / "step6_decision" / "decision_latest.json"
    trades_dir = out_root / "step7_trades"
    out_dir = ensure_dir(out_root / "step10_daily")

    summary: Dict[str, object] = {
        "schema_version": "step10_summary_v1",
        "generated_at_utc": _now_utc_iso(),
        "asof_date_utc": "",
        "decision_path": str(decision_path),
        "trades_dir": str(trades_dir),
        "decision_status": "missing_or_invalid",
        "risk_mode": {"mode": "unknown", "strength": 0},
        "tradable_themes": [],
        "counts": {"enter": 0, "watch": 0, "avoid": 0},
        "enter_candidates": [],
        "no_trade": {"is_no_trade": True, "reason": "decision missing"},
        "decision_digest": {"decision_hash": "", "enter_symbols": []},
        "warnings": [],
    }

    deviation: Dict[str, object] = {
        "schema_version": "step10_deviation_v1",
        "generated_at_utc": _now_utc_iso(),
        "asof_date_utc": "",
        "window_days": args.window_days,
        "decision_enter_symbols": [],
        "trade_enter_symbols_today": [],
        "deviations_today": [],
        "counts": {"deviation_today": 0, "deviation_7d": 0},
        "warning_level": "UNKNOWN",
        "warning_reason": "decision missing or invalid",
    }

    decision_payload, decision_status = _safe_read_json(decision_path)
    summary["decision_status"] = "ok" if decision_status == "ok" else "missing_or_invalid"

    asof_date = _asof_date_utc(decision_payload)
    summary["asof_date_utc"] = asof_date
    deviation["asof_date_utc"] = asof_date

    enter_candidates: List[dict] = []
    enter_symbols: List[str] = []
    avoid_symbols: List[str] = []

    if decision_payload:
        try:
            enter_candidates = _decision_enter_candidates(decision_payload)
            enter_symbols = _decision_symbols(decision_payload, "ENTER")
            avoid_symbols = _decision_symbols(decision_payload, "AVOID")
            summary["risk_mode"] = {
                "mode": decision_payload.get("risk_mode", {}).get("mode", "unknown"),
                "strength": decision_payload.get("risk_mode", {}).get("strength", 0),
            }
            summary["tradable_themes"] = decision_payload.get("tradable_themes", [])
            picks = decision_payload.get("picks", {}) if isinstance(decision_payload.get("picks", {}), dict) else {}
            summary["counts"] = {
                "enter": len(picks.get("ENTER", []) or []),
                "watch": len(picks.get("WATCH", []) or []),
                "avoid": len(picks.get("AVOID", []) or []),
            }
            summary["enter_candidates"] = enter_candidates
            decision_hash = _decision_hash(asof_date, enter_symbols)
            summary["decision_digest"] = {"decision_hash": decision_hash, "enter_symbols": enter_symbols}
            summary["no_trade"] = {
                "is_no_trade": len(enter_candidates) == 0,
                "reason": "ENTER candidates = 0" if len(enter_candidates) == 0 else "",
            }
        except Exception as e:
            summary["warnings"].append(f"decision_parse_error: {e}")

    today_trades = _read_trades(trades_dir, asof_date)
    trade_enter_symbols = sorted(
        {
            str(row.get("symbol", "")).upper()
            for row in today_trades
            if str(row.get("action", "")).upper() == "ENTER"
        }
    )
    deviation["trade_enter_symbols_today"] = trade_enter_symbols

    deviations_today: List[dict] = []
    if decision_payload:
        if len(enter_symbols) == 0 and trade_enter_symbols:
            for sym in trade_enter_symbols:
                deviations_today.append(
                    {
                        "type": "enter_when_no_trade",
                        "symbol": sym,
                        "action_ts_jst": "",
                        "details": {"reason": "decision ENTER empty"},
                    }
                )
        for sym in trade_enter_symbols:
            if sym not in enter_symbols and sym not in avoid_symbols:
                deviations_today.append(
                    {
                        "type": "enter_not_in_enter_candidates",
                        "symbol": sym,
                        "action_ts_jst": "",
                        "details": {"reason": "not in decision ENTER"},
                    }
                )
            if sym in avoid_symbols:
                deviations_today.append(
                    {
                        "type": "enter_on_avoid",
                        "symbol": sym,
                        "action_ts_jst": "",
                        "details": {"reason": "decision AVOID"},
                    }
                )

        deviation["decision_enter_symbols"] = enter_symbols
        deviation["deviations_today"] = deviations_today
        deviation["counts"]["deviation_today"] = len(deviations_today)
        deviation["warning_level"] = "OK" if len(deviations_today) == 0 else "WARN"
        deviation["warning_reason"] = "" if len(deviations_today) == 0 else "deviation_today > 0"
    else:
        deviation["decision_enter_symbols"] = []
        deviation["deviations_today"] = []

    summary_path = out_dir / f"summary_{asof_date}.json"
    deviation_path = out_dir / f"deviation_{asof_date}.json"
    summary_latest = out_dir / "summary_latest.json"
    deviation_latest = out_dir / "deviation_latest.json"

    summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    deviation_path.write_text(json.dumps(deviation, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    summary_latest.write_text(summary_path.read_text(encoding="utf-8"), encoding="utf-8")
    deviation_latest.write_text(deviation_path.read_text(encoding="utf-8"), encoding="utf-8")

    alerts_path = out_dir / "alerts.jsonl"
    existing_keys = _read_alerts(alerts_path)
    decision_hash = summary.get("decision_digest", {}).get("decision_hash", "")

    _append_alert(
        alerts_path,
        f"{asof_date}:RUN_SUMMARY",
        "RUN_SUMMARY",
        {
            "asof_date_utc": asof_date,
            "decision_hash": decision_hash,
            "enter_candidates": len(enter_candidates),
            "trade_enter_symbols": len(trade_enter_symbols),
        },
        existing_keys,
    )

    if summary.get("no_trade", {}).get("is_no_trade"):
        _append_alert(
            alerts_path,
            f"{asof_date}:NO_TRADE_NOTICE",
            "NO_TRADE_NOTICE",
            {"asof_date_utc": asof_date, "reason": summary["no_trade"].get("reason", "")},
            existing_keys,
        )

    if deviations_today:
        _append_alert(
            alerts_path,
            f"{asof_date}:DEVIATION_WARN",
            "DEVIATION_WARN",
            {"asof_date_utc": asof_date, "count": len(deviations_today)},
            existing_keys,
        )

    if enter_candidates:
        _append_alert(
            alerts_path,
            f"{asof_date}:ENTER_ALERT",
            "ENTER_ALERT",
            {"asof_date_utc": asof_date, "count": len(enter_candidates)},
            existing_keys,
        )

    LOG.info("wrote summary=%s deviation=%s", summary_path, deviation_path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
