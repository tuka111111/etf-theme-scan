from __future__ import annotations

import argparse
import csv
import hashlib
import json
import logging
import sys
import urllib.error
import urllib.request
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
                record = json.loads(line)
                key = record.get("event_id")
                if not key:
                    payload = record.get("payload", {}) if isinstance(record.get("payload", {}), dict) else {}
                    key = payload.get("event_id")
                if not key:
                    key = record.get("dedup_key")
                if not key:
                    payload = record.get("payload", {}) if isinstance(record.get("payload", {}), dict) else {}
                    key = payload.get("dedup_key")
                if key:
                    keys.add(str(key))
            except Exception:
                continue
    return keys


def _append_alert(
    alerts_path: Path,
    event_id: str,
    kind: str,
    payload: dict,
    existing: set[str],
    dedup_key: Optional[str] = None,
) -> None:
    if event_id in existing:
        return
    record = {
        "ts_utc": _now_utc_iso(),
        "asof_date_utc": payload.get("asof_date_utc", ""),
        "event_id": event_id,
        "dedup_key": dedup_key,
        "kind": kind,
        "payload": payload,
        "result": payload.get("send_result", "logged_only"),
    }
    with alerts_path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(record, ensure_ascii=False) + "\n")
    existing.add(event_id)


def _relative_path(path: Path) -> str:
    try:
        return str(path.relative_to(ROOT))
    except Exception:
        return str(path)


def _short_flags(flags: object) -> str:
    if isinstance(flags, list):
        items = [str(x) for x in flags if str(x)]
    elif isinstance(flags, str):
        items = [flags] if flags else []
    else:
        items = []
    if not items:
        return ""
    return ",".join(items[:2])


def _enter_symbols_hash(enter_symbols: List[str]) -> str:
    data = ",".join(sorted(set(enter_symbols)))
    return hashlib.sha1(data.encode("utf-8")).hexdigest()[:10]


def build_notification_message(summary: dict, max_enter: int) -> str:
    risk_mode = summary.get("risk_mode", {}) if isinstance(summary.get("risk_mode", {}), dict) else {}
    mode = risk_mode.get("mode", "unknown")
    strength = risk_mode.get("strength", 0)
    asof_date = summary.get("asof_date_utc", "")
    lines: List[str] = [f"{mode}({strength}) | asof={asof_date}"]

    enter_candidates = summary.get("enter_candidates", [])
    if isinstance(enter_candidates, list):
        enter_count = len(enter_candidates)
    else:
        enter_count = 0
        enter_candidates = []

    if enter_count > 0:
        lines.append(f"ENTER {enter_count}件（上位{max_enter}表示）")
        for i, row in enumerate(enter_candidates[:max_enter], start=1):
            symbol = row.get("symbol", "")
            theme = row.get("theme", "")
            score_total = row.get("score_total", "")
            score_adjusted = row.get("score_adjusted", "")
            threshold_used = row.get("threshold_used", "")
            flags_short = _short_flags(row.get("flags"))
            line = f"{i}. {symbol} [{theme}] score={score_adjusted}/{score_total} thr={threshold_used}"
            if flags_short:
                line += f" flags={flags_short}"
            lines.append(line)
    else:
        lines.append("今日は何もしない日です（NO TRADE）")

    digest = summary.get("decision_digest", {}) if isinstance(summary.get("decision_digest", {}), dict) else {}
    decision_id = digest.get("decision_id", "")
    decision_hash = digest.get("decision_hash", "")
    if isinstance(decision_hash, str) and len(decision_hash) > 12:
        decision_hash = decision_hash[:12]
    generated_at = summary.get("generated_at_utc", "")
    watchlist_path = summary.get("watchlist_path", "N/A")
    lines.append(f"decision_id={decision_id}")
    lines.append(f"decision_hash={decision_hash}")
    lines.append(f"generated_at_utc={generated_at}")
    lines.append(f"watchlist_path={watchlist_path}")
    return "\n".join(lines[: 2 + max_enter + 4])


def post_discord(webhook_url: str, content: str) -> tuple[bool, Optional[int], Optional[str]]:
    payload = json.dumps({"content": content}).encode("utf-8")
    req = urllib.request.Request(
        webhook_url,
        data=payload,
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    try:
        with urllib.request.urlopen(req, timeout=10) as resp:
            status = resp.getcode()
            if status in (200, 204):
                return True, status, None
            return False, status, f"http_status={status}"
    except urllib.error.HTTPError as e:
        return False, e.code, f"http_error={e.code}"
    except Exception as e:
        return False, None, str(e)


def build_agent_verdict(summary: dict, decision_payload: Optional[dict], max_enter: int) -> dict:
    enter_candidates = summary.get("enter_candidates", []) if isinstance(summary.get("enter_candidates", []), list) else []
    enter_count = len(enter_candidates)
    checks: List[dict] = []

    enter_exists_pass = enter_count > 0
    checks.append(
        {
            "id": "enter_exists",
            "pass": enter_exists_pass,
            "severity": "info",
            "message": "ENTER candidates > 0" if enter_exists_pass else "ENTER candidates = 0",
        }
    )

    ignore_flags: List[str] = []
    if decision_payload and isinstance(decision_payload.get("rules", {}), dict):
        flags_cfg = decision_payload.get("rules", {}).get("flags", {})
        if isinstance(flags_cfg, dict) and isinstance(flags_cfg.get("ignore"), list):
            ignore_flags = [str(x) for x in flags_cfg.get("ignore", []) if str(x)]
    if not ignore_flags:
        ignore_flags = ["halted", "bad_symbol", "data_quality_low"]

    critical_found: List[str] = []
    for row in enter_candidates[:max_enter]:
        flags_short = []
        flags = row.get("flags", [])
        if isinstance(flags, list):
            flags_short = [str(x) for x in flags if str(x)]
        elif isinstance(flags, str) and flags:
            flags_short = [flags]
        for flag in flags_short:
            if flag in ignore_flags:
                critical_found.append(flag)
    critical_pass = len(critical_found) == 0
    checks.append(
        {
            "id": "critical_flags_absent",
            "pass": critical_pass,
            "severity": "critical",
            "message": "no critical flags" if critical_pass else f"critical flags present: {sorted(set(critical_found))}",
        }
    )

    env_bias = ""
    if decision_payload:
        env_bias = str(decision_payload.get("env_bias", "")).lower()
    env_pass = env_bias != "bear"
    checks.append(
        {
            "id": "env_bear_skip",
            "pass": env_pass,
            "severity": "critical",
            "message": "env_bias is not bear" if env_pass else "env_bias=bear",
        }
    )

    verdict = "NO_TRADE"
    reason = "ENTER candidates = 0"
    if enter_count > 0:
        verdict = "ENTER_OK"
        reason = "ENTER candidates present"

    critical_fail = [c for c in checks if c["severity"] == "critical" and not c["pass"]]
    if critical_fail:
        verdict = "NO_TRADE"
        reason = critical_fail[0]["message"]
    elif enter_count == 0:
        verdict = "NO_TRADE"
        reason = "ENTER candidates = 0"

    return {"verdict": verdict, "reason": reason, "checks": checks}


def main(argv: Optional[Sequence[str]] = None) -> int:
    ap = argparse.ArgumentParser(description="Step10 daily runner (summary + deviation + alerts)")
    ap.add_argument("--out", required=True, help="Out directory root")
    ap.add_argument("--window-days", type=int, default=7, help="Deviation window (placeholder)")
    ap.add_argument("--notify", action="store_true", help="Send notifications (default: false)")
    ap.add_argument("--notify-channel", default="discord", help="Notification channel (default: discord)")
    ap.add_argument("--discord-webhook", default="", help="Discord webhook URL (optional)")
    ap.add_argument("--notify-max-enter", type=int, default=5, help="Max ENTER rows to include in notice")
    ap.add_argument("--notify-also-no-trade", action="store_true", help="Also notify NO TRADE")
    ap.add_argument("--notify-dry-run", action="store_true", help="Build payload but do not send")
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
        "decision_path": _relative_path(decision_path),
        "trades_dir": _relative_path(trades_dir),
        "decision_status": "missing_or_invalid",
        "risk_mode": {"mode": "unknown", "strength": 0},
        "tradable_themes": [],
        "counts": {"enter": 0, "watch": 0, "avoid": 0},
        "enter_candidates": [],
        "no_trade": {"is_no_trade": True, "reason": "decision missing"},
        "decision_digest": {"decision_hash": "", "decision_id": "", "enter_symbols": []},
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
            summary["no_trade"] = {
                "is_no_trade": len(enter_candidates) == 0,
                "reason": "ENTER candidates = 0" if len(enter_candidates) == 0 else "",
            }
        except Exception as e:
            summary["warnings"].append(f"decision_parse_error: {e}")

    decision_hash = _decision_hash(asof_date, enter_symbols)
    decision_id = ""
    if decision_payload and isinstance(decision_payload.get("decision_id"), str) and decision_payload.get("decision_id"):
        decision_id = str(decision_payload.get("decision_id"))
    else:
        decision_id = f"hash:{decision_hash[:12]}"
    summary["decision_digest"] = {
        "decision_hash": decision_hash,
        "decision_id": decision_id,
        "enter_symbols": enter_symbols,
    }

    summary["agent"] = build_agent_verdict(summary, decision_payload, args.notify_max_enter)

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
    decision_id = summary.get("decision_digest", {}).get("decision_id", "")
    risk_mode = summary.get("risk_mode", {}) if isinstance(summary.get("risk_mode", {}), dict) else {}
    event_id_run_summary = f"{asof_date}:{decision_id}:RUN_SUMMARY"

    _append_alert(
        alerts_path,
        event_id_run_summary,
        "RUN_SUMMARY",
        {
            "asof_date_utc": asof_date,
            "event_id": event_id_run_summary,
            "decision_id": decision_id,
            "decision_hash": decision_hash,
            "risk_mode": risk_mode,
            "enter_candidates": len(enter_candidates),
            "trade_enter_symbols": len(trade_enter_symbols),
        },
        existing_keys,
        dedup_key=f"{asof_date}:RUN_SUMMARY",
    )

    notify_enabled = bool(args.notify)
    notify_channel = str(args.notify_channel).lower()
    notify_max_enter = args.notify_max_enter
    notify_dry_run = bool(args.notify_dry_run)
    notify_also_no_trade = bool(args.notify_also_no_trade)
    webhook_url = str(args.discord_webhook).strip()

    if deviations_today:
        _append_alert(
            alerts_path,
            f"{asof_date}:{decision_id}:DEVIATION_WARN",
            "DEVIATION_WARN",
            {
                "asof_date_utc": asof_date,
                "event_id": f"{asof_date}:{decision_id}:DEVIATION_WARN",
                "decision_id": decision_id,
                "decision_hash": decision_hash,
                "risk_mode": risk_mode,
                "count": len(deviations_today),
            },
            existing_keys,
            dedup_key=f"{asof_date}:DEVIATION_WARN",
        )

    message = build_notification_message(summary, notify_max_enter)
    agent = summary.get("agent", {}) if isinstance(summary.get("agent", {}), dict) else {}
    agent_checks = agent.get("checks", []) if isinstance(agent.get("checks", []), list) else []
    agent_checks_failed = [
        {"id": c.get("id"), "severity": c.get("severity"), "message": c.get("message")}
        for c in agent_checks
        if not c.get("pass")
    ]

    enter_event_id = f"{asof_date}:{decision_id}:{_enter_symbols_hash(enter_symbols)}"
    if enter_candidates:
        send_result = "logged_only"
        send_error = None
        if notify_enabled and notify_channel == "discord":
            if enter_event_id in existing_keys:
                send_result = "skipped_duplicate"
            elif notify_dry_run:
                send_result = "dry_run"
            elif webhook_url:
                ok, status, err = post_discord(webhook_url, message)
                if ok:
                    send_result = "sent"
                else:
                    send_result = "failed"
                    send_error = f"{err or 'send_failed'}"
            else:
                send_result = "logged_only"
        payload = {
            "asof_date_utc": asof_date,
            "event_id": enter_event_id,
            "decision_id": decision_id,
            "decision_hash": decision_hash,
            "risk_mode": risk_mode,
            "enter_count": len(enter_candidates),
            "enter_top": [
                {
                    "symbol": row.get("symbol", ""),
                    "theme": row.get("theme", ""),
                    "score_total": row.get("score_total", ""),
                    "score_adjusted": row.get("score_adjusted", ""),
                    "threshold_used": row.get("threshold_used", ""),
                    "flags": row.get("flags", []),
                }
                for row in enter_candidates[:notify_max_enter]
            ],
            "message": message,
            "channel": "discord",
            "send_result": send_result,
            "agent_verdict": agent.get("verdict"),
            "agent_reason": agent.get("reason"),
            "agent_checks_failed": agent_checks_failed,
        }
        if send_error:
            payload["send_error"] = send_error
        if send_result != "skipped_duplicate":
            _append_alert(
                alerts_path,
                enter_event_id,
                "ENTER_ALERT",
                payload,
                existing_keys,
                dedup_key=f"{asof_date}:ENTER_ALERT",
            )

    if summary.get("no_trade", {}).get("is_no_trade"):
        no_trade_event_id = f"{asof_date}:{decision_id}:NO_TRADE"
        send_result = "logged_only"
        send_error = None
        if notify_enabled and notify_channel == "discord" and notify_also_no_trade:
            if no_trade_event_id in existing_keys:
                send_result = "skipped_duplicate"
            elif notify_dry_run:
                send_result = "dry_run"
            elif webhook_url:
                ok, status, err = post_discord(webhook_url, message)
                if ok:
                    send_result = "sent"
                else:
                    send_result = "failed"
                    send_error = f"{err or 'send_failed'}"
        payload = {
            "asof_date_utc": asof_date,
            "event_id": no_trade_event_id,
            "decision_id": decision_id,
            "decision_hash": decision_hash,
            "risk_mode": risk_mode,
            "reason": summary["no_trade"].get("reason", ""),
            "message": message,
            "channel": "discord",
            "send_result": send_result,
            "agent_verdict": agent.get("verdict"),
            "agent_reason": agent.get("reason"),
            "agent_checks_failed": agent_checks_failed,
        }
        if send_error:
            payload["send_error"] = send_error
        if send_result != "skipped_duplicate":
            _append_alert(
                alerts_path,
                no_trade_event_id,
                "NO_TRADE_NOTICE",
                payload,
                existing_keys,
                dedup_key=f"{asof_date}:NO_TRADE_NOTICE",
            )

    LOG.info("wrote summary=%s deviation=%s", summary_path, deviation_path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
