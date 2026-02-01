from __future__ import annotations

import argparse
import csv
import hashlib
import json
import logging
import sys
import os
import sys
import traceback
import urllib.error
import urllib.request
import time
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Dict, List, Optional, Sequence

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))

from pipeline.common import ensure_dir
from pipeline.dotenv import get_dotenv, load_dotenv
from pipeline import rules_digest

LOG = logging.getLogger(__name__)

AGENT_EXT_SYSTEM_PROMPT = (
    "You are a brief, non-advisory system. Do not make decisions or give advice. "
    "Write 1-2 sentences in English. Provide context-only commentary."
)
AGENT_EXT_USER_PROMPT_TEMPLATE = "Context (JSON):\n{{context_json}}\n\nComment:"
AGENT_EXT_PROMPT_VERSION = "v1"
AGENT_EXT_DEFAULT_MODEL = "gpt-4o-mini"


def _now_utc_iso() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds").replace("+00:00", "Z")


def _read_text(path: Path) -> str:
    return path.read_text(encoding="utf-8")


def _project_rel(path: Path, project_root: Path) -> str:
    try:
        return str(path.relative_to(project_root))
    except Exception:
        return path.name


def _read_env_file(env_path: Path) -> dict:
    if not env_path.exists():
        return {}
    data: dict[str, str] = {}
    try:
        for raw in env_path.read_text(encoding="utf-8").splitlines():
            line = raw.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue
            key, value = line.split("=", 1)
            key = key.strip()
            value = value.strip().strip('"').strip("'")
            if key:
                data[key] = value
    except Exception:
        return {}
    return data


def _sanitize_preview_text(text: str) -> str:
    if not isinstance(text, str):
        text = str(text)
    return text.replace("/Users/", "…/")


def _agent_ext_dedup_exists(alerts_path: Path, event_id: str, max_lines: int = 3000) -> bool:
    if not alerts_path.exists():
        return False
    try:
        with alerts_path.open(encoding="utf-8") as f:
            lines = f.readlines()[-max_lines:]
    except Exception:
        return False
    for raw in reversed(lines):
        line = raw.strip()
        if not line:
            continue
        try:
            rec = json.loads(line)
        except Exception:
            continue
        if rec.get("event_id") != event_id:
            continue
        payload = rec.get("payload", {}) if isinstance(rec.get("payload", {}), dict) else {}
        comment = payload.get("agent_ext_comment")
        if isinstance(comment, str) and comment.strip():
            return True
    return False


def _build_agent_ext_context(payload: dict, kind: str) -> dict:
    decision_hash = payload.get("decision_hash", "")
    risk_mode = payload.get("risk_mode", {}) if isinstance(payload.get("risk_mode", {}), dict) else {}
    enter_candidates = payload.get("top", []) if isinstance(payload.get("top", []), list) else []
    top_symbols = [str(row.get("symbol", "")).upper() for row in enter_candidates if row.get("symbol")]
    context = {
        "asof_date_utc": payload.get("asof_date_utc", ""),
        "kind": kind,
        "agent_verdict": payload.get("agent_verdict", ""),
        "agent_reason": payload.get("agent_reason", ""),
        "agent_failed_checks": payload.get("agent_failed_checks", []),
        "risk_mode": {
            "mode": risk_mode.get("mode", ""),
            "strength": risk_mode.get("strength", ""),
        },
        "enter_count": payload.get("count", payload.get("enter_candidates", "")),
        "top_symbols": top_symbols[:5],
        "deviation_level": payload.get("level", payload.get("warning_level", "")),
        "deviation_count_7d": payload.get("deviation_count_7d", ""),
        "rules_hash_yaml": payload.get("rules_hash_yaml", ""),
        "decision_hash": decision_hash,
    }
    msg = payload.get("message_preview", "")
    if isinstance(msg, str) and msg:
        context["message_preview"] = _sanitize_preview_text(msg)
    return context


def _call_agent_ext(
    base_url: str,
    api_key: str,
    model: str,
    context: dict,
) -> tuple[bool, Optional[int], Optional[str], Optional[str], Optional[int]]:
    url = base_url.rstrip("/") + "/chat/completions"
    user_prompt = AGENT_EXT_USER_PROMPT_TEMPLATE.replace("{{context_json}}", json.dumps(context, ensure_ascii=False))
    payload = {
        "model": model,
        "messages": [
            {"role": "system", "content": AGENT_EXT_SYSTEM_PROMPT},
            {"role": "user", "content": user_prompt},
        ],
        "temperature": 0.2,
        "max_tokens": 80,
    }
    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {api_key}",
    }
    req = urllib.request.Request(url, data=json.dumps(payload).encode("utf-8"), headers=headers, method="POST")
    start = time.time()
    try:
        with urllib.request.urlopen(req, timeout=15) as resp:
            status = resp.getcode()
            body = resp.read().decode("utf-8", errors="replace")
            latency_ms = int((time.time() - start) * 1000)
            if status not in (200, 201):
                return False, status, None, f"http_status={status}", latency_ms
            try:
                data = json.loads(body)
            except Exception:
                return False, status, None, "invalid_json", latency_ms
            content = (
                data.get("choices", [{}])[0]
                .get("message", {})
                .get("content", "")
            )
            return True, status, str(content).strip(), None, latency_ms
    except urllib.error.HTTPError as e:
        latency_ms = int((time.time() - start) * 1000)
        return False, e.code, None, f"{e.__class__.__name__}: http_error={e.code}", latency_ms
    except Exception as e:
        latency_ms = int((time.time() - start) * 1000)
        return False, None, None, f"{e.__class__.__name__}: {e}", latency_ms


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


def _extract_agent_ext_verdict(payload: Optional[dict]) -> tuple[str, bool]:
    if not payload or not isinstance(payload, dict):
        return "", False
    verdict = ""
    agent_ext = payload.get("agent_ext")
    if isinstance(agent_ext, dict):
        verdict = agent_ext.get("verdict") or agent_ext.get("final_verdict") or ""
    if not verdict and isinstance(payload.get("agent_ext_verdict"), str):
        verdict = payload.get("agent_ext_verdict", "")
    if not verdict and isinstance(payload.get("agent"), dict):
        verdict = payload.get("agent", {}).get("ext_verdict", "") or ""
    verdict = str(verdict).upper() if verdict else ""
    return verdict, bool(verdict)


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


def _apply_agent_ext(
    payload: dict,
    kind: str,
    event_id: str,
    alerts_path: Path,
    enabled: bool,
    allowed_kinds: set[str],
    force: bool,
    api_key: str,
    base_url: str,
    model: str,
) -> dict:
    payload["agent_ext_used"] = False
    payload["agent_ext_result"] = "skipped"
    payload["agent_ext_reason"] = "not enabled"
    payload["agent_ext_model"] = model
    payload["agent_ext_prompt_version"] = AGENT_EXT_PROMPT_VERSION

    if not enabled:
        return payload
    if kind not in allowed_kinds:
        payload["agent_ext_reason"] = "kind not allowed"
        return payload
    if payload.get("result") == "failed" or payload.get("send_result") == "failed":
        payload["agent_ext_reason"] = "send failed"
        return payload
    if not api_key:
        payload["agent_ext_reason"] = "missing OPENAI_API_KEY"
        return payload
    if not base_url:
        payload["agent_ext_reason"] = "missing OPENAI_BASE_URL"
        return payload
    if not force and _agent_ext_dedup_exists(alerts_path, event_id):
        payload["agent_ext_reason"] = "dedup"
        return payload

    context = _build_agent_ext_context(payload, kind)
    ok, status, comment, err, latency_ms = _call_agent_ext(base_url, api_key, model, context)
    payload["agent_ext_used"] = True
    payload["agent_ext_latency_ms"] = latency_ms
    payload["agent_ext_http_status"] = status
    if ok and comment:
        payload["agent_ext_result"] = "ok"
        payload["agent_ext_comment"] = comment
        payload["agent_ext_reason"] = ""
    else:
        payload["agent_ext_result"] = "failed"
        payload["agent_ext_error"] = (err or "unknown")[:200]
    return payload


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
    return hashlib.sha1(data.encode("utf-8")).hexdigest()


def _build_common_payload(
    summary: dict,
    event_id: str,
    decision_id: str,
    decision_hash: str,
    risk_mode: dict,
) -> dict:
    return {
        "event_id": event_id,
        "decision_id": decision_id,
        "decision_hash": decision_hash,
        "risk_mode": risk_mode,
        "generated_at_utc": summary.get("generated_at_utc", ""),
    }


def _safe_watchlist_path(value: object) -> str:
    if not isinstance(value, str) or not value:
        return "N/A"
    if value.startswith("/"):
        return "N/A"
    return value


def _read_step10_alerts_config(config_path: Path) -> dict:
    defaults = {
        "enabled": False,
        "max_items": 5,
        "channel": "discord_webhook",
        "webhook_env": "STEP10_DISCORD_WEBHOOK",
        "retry_failed": False,
    }
    if not config_path.exists():
        return defaults
    try:
        data = {}
        for line in config_path.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            if ":" not in line:
                continue
            key, value = line.split(":", 1)
            key = key.strip()
            value = value.strip()
            if value.lower() in {"true", "false"}:
                data[key] = value.lower() == "true"
            elif value.isdigit():
                data[key] = int(value)
            else:
                data[key] = value
        merged = defaults.copy()
        merged.update(data)
        return merged
    except Exception:
        return defaults


def _read_rules_yaml(path: Path) -> dict:
    if not path.exists():
        return {}
    try:
        data: dict[str, object] = {}
        current_section = ""
        current_subsection = ""
        current_flags_list = ""
        for raw in path.read_text(encoding="utf-8").splitlines():
            line = raw.strip()
            if not line or line.startswith("#"):
                continue
            if line.startswith("flags:"):
                current_section = "flags"
                current_subsection = ""
                current_flags_list = ""
                data.setdefault("flags", {})
                continue
            if line.startswith("env_bias:"):
                current_section = "env_bias"
                current_subsection = ""
                current_flags_list = ""
                data.setdefault("env_bias", {})
                continue
            if line.startswith("-") and current_section == "flags":
                item = line.lstrip("-").strip()
                if item:
                    target_list = current_flags_list or "ignore"
                    data.setdefault("flags", {}).setdefault(target_list, [])
                    data["flags"][target_list].append(item)
                continue
            if line.endswith(":") and current_section == "env_bias":
                current_subsection = line[:-1].strip()
                data.setdefault("env_bias", {}).setdefault(current_subsection, {})
                continue
            if ":" in line:
                key, value = line.split(":", 1)
                key = key.strip()
                value = value.strip()
                if current_section == "flags" and key == "ignore":
                    current_flags_list = "ignore"
                    if value.startswith("[") and value.endswith("]"):
                        inner = value[1:-1].strip()
                        items = [v.strip().strip("'\"") for v in inner.split(",") if v.strip()]
                        data.setdefault("flags", {})["ignore"] = items
                elif current_section == "flags" and key == "block":
                    current_flags_list = "block"
                    if value.startswith("[") and value.endswith("]"):
                        inner = value[1:-1].strip()
                        items = [v.strip().strip("'\"") for v in inner.split(",") if v.strip()]
                        data.setdefault("flags", {})["block"] = items
                elif current_section == "flags" and key == "warn":
                    current_flags_list = "warn"
                    if value.startswith("[") and value.endswith("]"):
                        inner = value[1:-1].strip()
                        items = [v.strip().strip("'\"") for v in inner.split(",") if v.strip()]
                        data.setdefault("flags", {})["warn"] = items
                elif current_section == "env_bias" and current_subsection:
                    if key == "action":
                        data.setdefault("env_bias", {}).setdefault(current_subsection, {})["action"] = value
        return data
    except Exception:
        return {}


def _read_agent_rules_yaml(path: Path) -> tuple[dict, str, Optional[str]]:
    defaults = {
        "enabled": True,
        "require_enter_min_count": 1,
        "blocking_flags": [],
        "risk_mode_allow": ["RISK_ON"],
        "env_bias_deny": [],
        "max_enter_to_show": 5,
        "flag_weight_policy": {
            "enabled": False,
            "min_effective_score": None,
            "use_score_adjusted_first": True,
            "weights": {},
        },
        "env_policy": {
            "enabled": False,
            "require_etf_env_present": False,
            "etf_env_deny": [],
            "etf_env_neutral_action": "allow",
            "on_deny": "NO_TRADE",
        },
    }
    if not path.exists():
        return defaults, "missing", rules_digest.get_rules_hash(path)
    try:
        raw_text = _read_text(path)
    except Exception:
        return defaults, "parse_error", rules_digest.get_rules_hash(path)
    rules = defaults.copy()
    section = ""
    subsection = ""
    current_list = ""
    current_map = ""
    for raw in raw_text.splitlines():
        if not raw.strip() or raw.strip().startswith("#"):
            continue
        if raw.startswith((" ", "\t")) is False and raw.strip().endswith(":") and raw.strip() != "agent:":
            section = ""
            subsection = ""
            current_list = ""
        line = raw.strip()
        if line.startswith("agent:"):
            section = "agent"
            subsection = ""
            current_list = ""
            continue
        if section != "agent":
            continue
        if line.startswith("enabled:"):
            val = line.split(":", 1)[1].strip().lower()
            rules["enabled"] = val == "true"
            continue
        if line.startswith("require_enter_min_count:"):
            try:
                rules["require_enter_min_count"] = int(line.split(":", 1)[1].strip())
            except Exception:
                pass
            continue
        if line.startswith("max_enter_to_show:"):
            try:
                rules["max_enter_to_show"] = int(line.split(":", 1)[1].strip())
            except Exception:
                pass
            continue
        if line.startswith("blocking_flags:"):
            current_list = "blocking_flags"
            items = line.split(":", 1)[1].strip()
            if items.startswith("[") and items.endswith("]"):
                inner = items[1:-1].strip()
                rules["blocking_flags"] = [v.strip().strip("'\"") for v in inner.split(",") if v.strip()]
                current_list = ""
            continue
        if line.startswith("risk_mode_policy:"):
            subsection = "risk_mode_policy"
            current_list = ""
            continue
        if line.startswith("flag_weight_policy:"):
            subsection = "flag_weight_policy"
            current_list = ""
            current_map = ""
            continue
        if line.startswith("env_policy:"):
            subsection = "env_policy"
            current_list = ""
            current_map = ""
            continue
        if line.startswith("env_bias_policy:"):
            subsection = "env_bias_policy"
            current_list = ""
            continue
        if subsection == "risk_mode_policy" and line.startswith("allow:"):
            current_list = "risk_mode_allow"
            items = line.split(":", 1)[1].strip()
            if items.startswith("[") and items.endswith("]"):
                inner = items[1:-1].strip()
                rules["risk_mode_allow"] = [v.strip().strip("'\"") for v in inner.split(",") if v.strip()]
                current_list = ""
            continue
        if subsection == "env_bias_policy" and line.startswith("deny:"):
            current_list = "env_bias_deny"
            items = line.split(":", 1)[1].strip()
            if items.startswith("[") and items.endswith("]"):
                inner = items[1:-1].strip()
                rules["env_bias_deny"] = [v.strip().strip("'\"") for v in inner.split(",") if v.strip()]
                current_list = ""
            continue
        if subsection == "flag_weight_policy" and line.startswith("enabled:"):
            val = line.split(":", 1)[1].strip().lower()
            rules["flag_weight_policy"]["enabled"] = val == "true"
            continue
        if subsection == "flag_weight_policy" and line.startswith("min_effective_score:"):
            val = line.split(":", 1)[1].strip()
            if val.lower() in {"null", "none", ""}:
                rules["flag_weight_policy"]["min_effective_score"] = None
            else:
                try:
                    rules["flag_weight_policy"]["min_effective_score"] = float(val)
                except Exception:
                    pass
            continue
        if subsection == "flag_weight_policy" and line.startswith("use_score_adjusted_first:"):
            val = line.split(":", 1)[1].strip().lower()
            rules["flag_weight_policy"]["use_score_adjusted_first"] = val == "true"
            continue
        if subsection == "flag_weight_policy" and line.startswith("weights:"):
            current_map = "flag_weight_weights"
            continue
        if subsection == "env_policy" and line.startswith("enabled:"):
            val = line.split(":", 1)[1].strip().lower()
            rules["env_policy"]["enabled"] = val == "true"
            continue
        if subsection == "env_policy" and line.startswith("require_etf_env_present:"):
            val = line.split(":", 1)[1].strip().lower()
            rules["env_policy"]["require_etf_env_present"] = val == "true"
            continue
        if subsection == "env_policy" and line.startswith("etf_env_neutral_action:"):
            rules["env_policy"]["etf_env_neutral_action"] = line.split(":", 1)[1].strip()
            continue
        if subsection == "env_policy" and line.startswith("on_deny:"):
            rules["env_policy"]["on_deny"] = line.split(":", 1)[1].strip()
            continue
        if subsection == "env_policy" and line.startswith("etf_env_deny:"):
            current_list = "env_policy_etf_env_deny"
            items = line.split(":", 1)[1].strip()
            if items.startswith("[") and items.endswith("]"):
                inner = items[1:-1].strip()
                rules["env_policy"]["etf_env_deny"] = [v.strip().strip("'\"") for v in inner.split(",") if v.strip()]
                current_list = ""
            continue
        if line.startswith("-") and current_list:
            item = line.lstrip("-").strip()
            if item:
                if current_list == "env_policy_etf_env_deny":
                    rules["env_policy"]["etf_env_deny"].append(item)
                else:
                    rules[current_list].append(item)
            continue
        if current_map == "flag_weight_weights" and ":" in line and not line.startswith("-"):
            key, value = line.split(":", 1)
            key = key.strip()
            try:
                rules["flag_weight_policy"]["weights"][key] = float(value.strip())
            except Exception:
                continue
    if not rules.get("risk_mode_allow"):
        rules["risk_mode_allow"] = ["RISK_ON"]
    return rules, "ok", rules_digest.get_rules_hash(path)


def _normalize_rules_for_agent(raw_rules: object) -> dict:
    rules: dict[str, object] = {}
    if isinstance(raw_rules, dict):
        rules.update(raw_rules)
    flags_ignore: List[str] = []
    flags_block: List[str] = []
    flags_warn: List[str] = []
    flags = rules.get("flags", {})
    if isinstance(flags, dict):
        ignore = flags.get("ignore", [])
        block = flags.get("block", [])
        warn = flags.get("warn", [])
        if isinstance(ignore, list):
            flags_ignore = [str(x) for x in ignore if str(x)]
        elif isinstance(ignore, str):
            flags_ignore = [ignore]
        if isinstance(block, list):
            flags_block = [str(x) for x in block if str(x)]
        elif isinstance(block, str):
            flags_block = [block]
        if isinstance(warn, list):
            flags_warn = [str(x) for x in warn if str(x)]
        elif isinstance(warn, str):
            flags_warn = [warn]
    if not flags_ignore:
        flags_ignore = ["data_quality_low", "halted", "bad_symbol"]
    rules["flags"] = {"ignore": flags_ignore, "block": flags_block, "warn": flags_warn}

    env_bias = rules.get("env_bias", {})
    normalized_env = {"bull": {"action": "allow"}, "neutral": {"action": "allow"}, "bear": {"action": "allow"}}
    if isinstance(env_bias, dict):
        for key in ["bull", "neutral", "bear"]:
            if isinstance(env_bias.get(key), dict):
                action = str(env_bias.get(key, {}).get("action", "allow"))
                normalized_env[key] = {"action": action}
    rules["env_bias"] = normalized_env
    return rules


def _window_dates(asof_date_utc: str, window_days: int) -> List[str]:
    try:
        base = datetime.fromisoformat(asof_date_utc).date()
    except Exception:
        return []
    if window_days <= 0:
        return []
    dates = [(base - timedelta(days=i)).isoformat() for i in range(window_days)]
    return list(reversed(dates))


def _read_deviation_days_from_alerts(alerts_path: Path, window_dates: List[str]) -> Optional[dict]:
    if not alerts_path.exists():
        return None
    deviation_days: set[str] = set()
    invalid_lines = 0
    try:
        with alerts_path.open(encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    record = json.loads(line)
                except Exception:
                    invalid_lines += 1
                    continue
                kind = record.get("kind", "")
                payload = record.get("payload", {}) if isinstance(record.get("payload", {}), dict) else {}
                asof = payload.get("asof_date_utc") or record.get("asof_date_utc")
                if kind not in {"DEVIATION_WARN", "DEVIATION_CRITICAL", "DEVIATION_EVENT"}:
                    continue
                if asof not in window_dates:
                    continue
                deviation_days.add(asof)
        return {
            "deviation_days": deviation_days,
            "invalid_alert_lines": invalid_lines,
            "source": "alerts",
        }
    except Exception:
        return None


def _read_deviation_days_from_files(out_dir: Path, window_dates: List[str]) -> Optional[dict]:
    deviation_days: set[str] = set()
    unknown_symbol_days: set[str] = set()
    no_trade_ignored_days: set[str] = set()
    unknown_symbol_events = 0
    no_trade_ignored_events = 0
    found_any = False
    for day in window_dates:
        path = out_dir / f"deviation_{day}.json"
        if not path.exists():
            continue
        found_any = True
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except Exception:
            continue
        deviations = payload.get("deviations_today", [])
        if not isinstance(deviations, list):
            continue
        if deviations:
            deviation_days.add(day)
        for row in deviations:
            dtype = row.get("type")
            details = row.get("details", {}) if isinstance(row.get("details", {}), dict) else {}
            if dtype == "unknown_symbol_enter" or details.get("unknown_symbol") is True:
                unknown_symbol_days.add(day)
                unknown_symbol_events += 1
            if dtype in {"no_trade_ignored", "enter_when_no_trade"}:
                no_trade_ignored_days.add(day)
                no_trade_ignored_events += 1
    if not found_any:
        return None
    return {
        "deviation_days": deviation_days,
        "unknown_symbol_days": unknown_symbol_days,
        "no_trade_ignored_days": no_trade_ignored_days,
        "unknown_symbol_events": unknown_symbol_events,
        "no_trade_ignored_events": no_trade_ignored_events,
        "source": "files",
    }


def _compute_streak(days_set: set[str], window_dates: List[str]) -> int:
    max_streak = 0
    streak = 0
    for day in window_dates:
        if day in days_set:
            streak += 1
            if streak > max_streak:
                max_streak = streak
        else:
            streak = 0
    return max_streak


def _load_deviation_for_date(out_dir: Path, date_str: str) -> Optional[dict]:
    path = out_dir / f"deviation_{date_str}.json"
    payload, status = _safe_read_json(path)
    if status != "ok":
        return None
    return payload


def build_notification_message(summary: dict, max_enter: int) -> str:
    risk_mode = summary.get("risk_mode", {}) if isinstance(summary.get("risk_mode", {}), dict) else {}
    mode = risk_mode.get("mode", "unknown")
    strength = risk_mode.get("strength", 0)
    asof_date = summary.get("asof_date_utc", "")
    generated_at = summary.get("generated_at_utc", "")
    watchlist_path = _safe_watchlist_path(summary.get("watchlist_path", "N/A"))

    digest = summary.get("decision_digest", {}) if isinstance(summary.get("decision_digest", {}), dict) else {}
    decision_id = digest.get("decision_id", "")
    decision_hash = digest.get("decision_hash", "")

    enter_candidates = summary.get("enter_candidates", [])
    if isinstance(enter_candidates, list):
        enter_count = len(enter_candidates)
    else:
        enter_count = 0
        enter_candidates = []

    lines: List[str] = []
    if enter_count > 0:
        lines.append(f"[{mode} {strength}] ENTER {enter_count}件")
        for row in enter_candidates[:max_enter]:
            symbol = row.get("symbol", "")
            theme = row.get("theme", "")
            score_total = row.get("score_total", "")
            score_adjusted = row.get("score_adjusted", "")
            threshold_used = row.get("threshold_used", "")
            flags = row.get("flags", [])
            rules_applied = row.get("rules_applied", [])
            flags_list: List[str] = []
            if isinstance(flags, list):
                flags_list.extend([str(x) for x in flags if str(x)])
            if isinstance(rules_applied, list):
                flags_list.extend([str(x) for x in rules_applied if str(x)])
            flags_text = ";".join(flags_list) if flags_list else "clean"
            theme_text = f" ({theme})" if theme else ""
            lines.append(
                f"{symbol}{theme_text} score={score_total} adj={score_adjusted} thr={threshold_used} flags={flags_text}"
            )
    else:
        lines.append(f"[{mode} {strength}] 今日は何もしない日です")
        agent = summary.get("agent", {}) if isinstance(summary.get("agent", {}), dict) else {}
        reason = summary.get("no_trade", {}).get("reason", "") or agent.get("reason", "")
        if reason:
            lines.append(f"reason: {reason}")

    lines.append(f"asof={asof_date} generated_at_utc={generated_at}")
    lines.append(f"watchlist={watchlist_path}")
    lines.append(f"decision_id={decision_id} decision_hash={decision_hash}")
    return "\n".join(lines)


def build_no_trade_message(summary: dict, reason: str, rules_hash_yaml: Optional[str]) -> str:
    risk_mode = summary.get("risk_mode", {}) if isinstance(summary.get("risk_mode", {}), dict) else {}
    mode = risk_mode.get("mode", "unknown")
    strength = risk_mode.get("strength", 0)
    asof_date = summary.get("asof_date_utc", "")
    generated_at = summary.get("generated_at_utc", "")
    digest = summary.get("decision_digest", {}) if isinstance(summary.get("decision_digest", {}), dict) else {}
    decision_id = digest.get("decision_id", "")
    decision_hash = digest.get("decision_hash", "")
    if decision_hash:
        decision_hash = decision_hash[:12]
    rules_hash_text = rules_hash_yaml if rules_hash_yaml else "N/A"

    lines = [
        f"[{mode} {strength}] NO TRADE",
        f"reason={reason}",
        f"asof={asof_date} generated_at_utc={generated_at}",
        f"decision_id={decision_id} decision_hash={decision_hash}",
        f"rules_hash_yaml={rules_hash_text}",
    ]
    return "\n".join(lines)


def build_deviation_warning_message(summary: dict, deviation: dict) -> str:
    warning_level = str(deviation.get("warning_level", "")).upper()
    counts = deviation.get("counts", {}) if isinstance(deviation.get("counts", {}), dict) else {}
    risk_mode = summary.get("risk_mode", {}) if isinstance(summary.get("risk_mode", {}), dict) else {}
    mode = risk_mode.get("mode", "unknown")
    strength = risk_mode.get("strength", 0)
    asof_date = summary.get("asof_date_utc", "")
    decision_id = summary.get("decision_digest", {}).get("decision_id", "")
    warning_reasons = deviation.get("warning_reasons", [])
    if not isinstance(warning_reasons, list):
        warning_reasons = []

    lines = [
        f"[{warning_level}] deviation_7d={counts.get('deviation_7d', 0)} "
        f"unknown_symbol_days_7d={counts.get('unknown_symbol_days_7d', 0)} "
        f"no_trade_ignored_streak={counts.get('no_trade_ignored_streak', 0)}",
        f"asof={asof_date} decision_id={decision_id}",
        f"risk_mode={mode}/{strength}",
    ]
    if warning_reasons:
        lines.append("reasons=" + ";".join([str(x) for x in warning_reasons]))
    return "\n".join(lines)


def _compute_deviation_today(
    decision_enter_symbols: List[str],
    trade_enter_symbols: List[str],
    agent_verdict: str,
    agent_ext_verdict: str,
    agent_ext_used: bool,
) -> dict:
    events: List[dict] = []
    trade_count = len(trade_enter_symbols)
    agent_verdict_upper = str(agent_verdict or "").upper()
    agent_ext_upper = str(agent_ext_verdict or "").upper()

    if trade_count > 0 and (agent_verdict_upper == "NO_TRADE" or agent_ext_upper == "NO_TRADE"):
        score = 2
        if agent_ext_used and agent_ext_upper == "NO_TRADE":
            score += 2
        events.append(
            {
                "id": "no_trade_enter",
                "score": score,
                "detail": f"agent_no_trade trade_enter_symbols_count={trade_count}",
                "count": trade_count,
                "symbols": trade_enter_symbols,
            }
        )

    unknown_symbols = sorted(set(trade_enter_symbols) - set(decision_enter_symbols))
    if unknown_symbols:
        events.append(
            {
                "id": "enter_not_in_decision",
                "score": 1,
                "detail": f"symbols_not_in_decision={len(unknown_symbols)}",
                "count": len(unknown_symbols),
                "symbols": unknown_symbols,
            }
        )

    score_total = 0
    for row in events:
        try:
            score_total += int(row.get("score", 0) or 0)
        except Exception:
            continue
    return {
        "deviation_events": events,
        "deviation_count_today": len(events),
        "deviation_score_today": score_total,
        "agent_ext_used": bool(agent_ext_used),
    }


def _compute_deviation_7d(out_dir: Path, asof_date: str, window_days: int, today_payload: dict) -> dict:
    window_dates = _window_dates(asof_date, window_days)
    window_asof_dates: List[str] = []
    missing_days: List[str] = []
    deviation_count_7d = 0
    deviation_score_7d = 0

    for day in window_dates:
        payload = today_payload if day == asof_date else _load_deviation_for_date(out_dir, day)
        if not payload or not isinstance(payload, dict):
            missing_days.append(day)
            continue
        count = payload.get("deviation_count_today")
        score = payload.get("deviation_score_today")
        if not isinstance(count, int) or not isinstance(score, (int, float)):
            events = payload.get("deviation_events", [])
            if isinstance(events, list):
                count = len(events)
                score = 0
                for row in events:
                    try:
                        score += int(row.get("score", 0) or 0)
                    except Exception:
                        continue
            else:
                missing_days.append(day)
                continue
        window_asof_dates.append(day)
        deviation_count_7d += int(count)
        deviation_score_7d += int(score)

    level_7d = "OK"
    level_reason = "count<2"
    if deviation_count_7d >= 4:
        level_7d = "CRITICAL"
        level_reason = "count>=4"
    elif deviation_count_7d >= 2:
        level_7d = "WARN"
        level_reason = "count>=2"
    if level_7d == "WARN" and deviation_score_7d >= 6:
        level_7d = "CRITICAL"
        level_reason = "score>=6"

    return {
        "window_dates": window_dates,
        "window_asof_dates": window_asof_dates,
        "missing_days": missing_days,
        "deviation_count_7d": deviation_count_7d,
        "deviation_score_7d": deviation_score_7d,
        "level_7d": level_7d,
        "level_reason": level_reason,
    }


def _build_deviation_notice_message(
    asof_date: str,
    decision_id: str,
    level_7d: str,
    deviation_count_7d: int,
    deviation_score_7d: int,
    window_days: int,
    missing_days: List[str],
    top_reasons: List[str],
) -> str:
    lines = [
        f"[DEVIATION_{level_7d}_7D] count={deviation_count_7d} score={deviation_score_7d} window_days={window_days}",
        f"asof={asof_date} decision_id={decision_id}",
    ]
    if missing_days:
        lines.append(f"missing_days={len(missing_days)}")
    if top_reasons:
        lines.append("reasons=" + ";".join([str(x) for x in top_reasons if str(x)]))
    return "\n".join(lines)


def _maybe_emit_deviation_notice(
    alerts_path: Path,
    existing_keys: set[str],
    summary: dict,
    asof_date: str,
    decision_id: str,
    decision_hash: str,
    risk_mode: dict,
    send_enabled: bool,
    webhook_url: str,
    webhook_env: str,
    debug_webhook: bool,
    debug_ping_used: List[bool],
    send_gate_reason: str,
    deviation_meta: dict,
    deviation_events: List[dict],
    agent_ext_used: bool,
    agent_ext_enabled: bool,
    agent_ext_kinds: set[str],
    agent_ext_force: bool,
    agent_ext_api_key: str,
    agent_ext_base_url: str,
    agent_ext_model: str,
) -> None:
    level_7d = str(deviation_meta.get("level_7d", "")).upper()
    if level_7d not in {"WARN", "CRITICAL"}:
        return
    event_id = f"{asof_date}:hash:{decision_hash}:DEVIATION_{level_7d}_7D"
    dedup_key = f"{asof_date}:DEVIATION_NOTICE"
    if event_id in existing_keys:
        return
    top_reasons = []
    for row in deviation_events:
        detail = row.get("detail", "")
        if detail:
            top_reasons.append(str(detail))
        if len(top_reasons) >= 3:
            break
    message_text = _build_deviation_notice_message(
        asof_date,
        decision_id,
        level_7d,
        int(deviation_meta.get("deviation_count_7d", 0) or 0),
        int(deviation_meta.get("deviation_score_7d", 0) or 0),
        int(deviation_meta.get("window_days", 7) or 7),
        deviation_meta.get("missing_days", []) if isinstance(deviation_meta.get("missing_days", []), list) else [],
        top_reasons,
    )
    payload = {
        "asof_date_utc": asof_date,
        **_build_common_payload(summary, event_id, decision_id, decision_hash, risk_mode),
        "level": level_7d,
        "deviation_count_7d": deviation_meta.get("deviation_count_7d", 0),
        "deviation_score_7d": deviation_meta.get("deviation_score_7d", 0),
        "window_days": deviation_meta.get("window_days", 7),
        "window_asof_dates": deviation_meta.get("window_asof_dates", []),
        "missing_days": deviation_meta.get("missing_days", []),
        "top_reasons": top_reasons,
        "agent_ext_used": bool(agent_ext_used),
        "message_preview": message_text,
        "event_id": event_id,
        "dedup_key": dedup_key,
    }
    payload = _apply_agent_ext(
        payload,
        "DEVIATION_NOTICE",
        event_id,
        alerts_path,
        agent_ext_enabled,
        agent_ext_kinds,
        agent_ext_force,
        agent_ext_api_key,
        agent_ext_base_url,
        agent_ext_model,
    )
    if send_enabled and webhook_url:
        debug_ping = debug_webhook and not debug_ping_used[0]
        ok, status, err = post_discord(webhook_url, message_text, webhook_env, debug_ping)
        if debug_ping:
            debug_ping_used[0] = True
        if ok:
            payload["send_result"] = "sent"
            payload["http_status"] = status
        else:
            payload["send_result"] = "failed"
            payload["http_status"] = status
            payload["reason"] = err
            payload["webhook_hash"] = _sha12(webhook_url) if webhook_url else ""
            payload["proxy_flags"] = _proxy_env_flags()
            payload["sys_executable"] = sys.executable
    else:
        payload["send_result"] = "logged_only"
        if send_gate_reason:
            payload["reason"] = send_gate_reason
    _append_alert(alerts_path, event_id, "DEVIATION_NOTICE", payload, existing_keys, dedup_key=dedup_key)


def _sha12(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()[:12]


def _proxy_env_flags() -> dict:
    keys = ["HTTP_PROXY", "HTTPS_PROXY", "ALL_PROXY", "NO_PROXY"]
    return {k: bool(os.getenv(k)) for k in keys}


def post_discord(
    webhook_url: str,
    content: str,
    webhook_env: str,
    debug_ping: bool = False,
) -> tuple[bool, Optional[int], Optional[str]]:
    payload = {"content": "webhook debug ping"} if debug_ping else {"content": content}
    headers = {"Content-Type": "application/json", "User-Agent": "curl/8.4.0"}
    LOG.info(
        "webhook send: env=%s set=%s url_sha12=%s proxy_flags=%s user_agent=%s debug_ping=%s",
        webhook_env,
        bool(webhook_url),
        _sha12(webhook_url) if webhook_url else "none",
        _proxy_env_flags(),
        headers.get("User-Agent", ""),
        debug_ping,
    )
    LOG.info("webhook sys.executable: %s", sys.executable)
    req = urllib.request.Request(
        webhook_url,
        data=json.dumps(payload).encode("utf-8"),
        headers=headers,
        method="POST",
    )
    try:
        with urllib.request.urlopen(req, timeout=10) as resp:
            status = resp.getcode()
            LOG.info("webhook status: %s", status)
            info_headers = {
                "Date": resp.headers.get("Date"),
                "Server": resp.headers.get("Server"),
                "Via": resp.headers.get("Via"),
                "CF-RAY": resp.headers.get("CF-RAY"),
                "CF-Cache-Status": resp.headers.get("CF-Cache-Status"),
                "CF-Connecting-IP": resp.headers.get("CF-Connecting-IP"),
                "X-RateLimit-Limit": resp.headers.get("X-RateLimit-Limit"),
                "X-RateLimit-Remaining": resp.headers.get("X-RateLimit-Remaining"),
                "X-RateLimit-Reset": resp.headers.get("X-RateLimit-Reset"),
            }
            LOG.info("webhook headers: %s", info_headers)
            body = resp.read().decode("utf-8", errors="replace")
            if status != 204 and body:
                LOG.error("webhook response body: %s", body[:500])
            if status in (200, 204):
                return True, status, None
            return False, status, f"http_status={status}"
    except urllib.error.HTTPError as e:
        body = ""
        try:
            body = e.read().decode("utf-8", errors="replace")
        except Exception:
            body = ""
        LOG.info("webhook status: %s", e.code)
        headers = {
            "Date": getattr(e, "headers", {}).get("Date") if getattr(e, "headers", None) else None,
            "Server": getattr(e, "headers", {}).get("Server") if getattr(e, "headers", None) else None,
            "Via": getattr(e, "headers", {}).get("Via") if getattr(e, "headers", None) else None,
            "CF-RAY": getattr(e, "headers", {}).get("CF-RAY") if getattr(e, "headers", None) else None,
            "CF-Cache-Status": getattr(e, "headers", {}).get("CF-Cache-Status") if getattr(e, "headers", None) else None,
            "CF-Connecting-IP": getattr(e, "headers", {}).get("CF-Connecting-IP") if getattr(e, "headers", None) else None,
            "X-RateLimit-Limit": getattr(e, "headers", {}).get("X-RateLimit-Limit") if getattr(e, "headers", None) else None,
            "X-RateLimit-Remaining": getattr(e, "headers", {}).get("X-RateLimit-Remaining") if getattr(e, "headers", None) else None,
            "X-RateLimit-Reset": getattr(e, "headers", {}).get("X-RateLimit-Reset") if getattr(e, "headers", None) else None,
        }
        LOG.info("webhook headers: %s", headers)
        if body:
            LOG.error("webhook response body: %s", body[:500])
        return False, e.code, f"{e.__class__.__name__}: http_error={e.code}"
    except Exception as e:
        LOG.error("webhook exception: %s: %s\n%s", e.__class__.__name__, e, traceback.format_exc())
        return False, None, f"{e.__class__.__name__}: {e}"


def build_agent_verdict(summary: dict, rules_used: dict, rules_source: str) -> dict:
    enter_candidates = summary.get("enter_candidates", []) if isinstance(summary.get("enter_candidates", []), list) else []
    enter_count = len(enter_candidates)
    risk_mode = summary.get("risk_mode", {}) if isinstance(summary.get("risk_mode", {}), dict) else {}
    risk_mode_value = str(risk_mode.get("mode", "")).upper()
    rules_flags = rules_used.get("flags", {}) if isinstance(rules_used.get("flags", {}), dict) else {}
    ignore_flags = rules_flags.get("ignore", []) if isinstance(rules_flags.get("ignore", []), list) else []
    block_flags = rules_flags.get("block", []) if isinstance(rules_flags.get("block", []), list) else []
    warn_flags = rules_flags.get("warn", []) if isinstance(rules_flags.get("warn", []), list) else []
    blocking_flags = set(ignore_flags) | set(block_flags)

    blocked_candidates: List[dict] = []
    warned_candidates: List[dict] = []
    for row in enter_candidates:
        flags = row.get("flags", [])
        if isinstance(flags, list):
            items = [str(x) for x in flags if str(x)]
        elif isinstance(flags, str) and flags:
            items = [flags]
        else:
            items = []
        blocked = [flag for flag in items if flag in blocking_flags]
        warned = [flag for flag in items if flag in warn_flags]
        if blocked and len(blocked_candidates) < 3:
            blocked_candidates.append({"symbol": row.get("symbol", ""), "flags": blocked})
        if warned and len(warned_candidates) < 3:
            warned_candidates.append({"symbol": row.get("symbol", ""), "flags": warned})

    enter_exists_ok = enter_count > 0
    no_ignored_ok = len(blocked_candidates) == 0
    risk_mode_ok = risk_mode_value != "RISK_OFF"

    current_env = "neutral"
    if risk_mode_value == "RISK_ON":
        current_env = "bull"
    elif risk_mode_value == "RISK_OFF":
        current_env = "bear"
    env_bias = rules_used.get("env_bias", {}) if isinstance(rules_used.get("env_bias", {}), dict) else {}
    env_action = (
        env_bias.get(current_env, {}).get("action", "allow")
        if isinstance(env_bias.get(current_env, {}), dict)
        else "allow"
    )
    env_bias_ok = str(env_action) != "skip"

    checks = [
        {"id": "enter_exists", "ok": enter_exists_ok, "detail": f"enter_count={enter_count}"},
        {"id": "risk_mode_allows", "ok": risk_mode_ok, "detail": f"risk_mode={risk_mode_value or 'unknown'}"},
        {
            "id": "no_ignored_flags_present",
            "ok": no_ignored_ok,
            "detail": "no ignored flags" if no_ignored_ok else f"blocked={blocked_candidates}",
        },
        {
            "id": "warn_flags_present",
            "ok": len(warned_candidates) == 0,
            "detail": "no warn flags" if len(warned_candidates) == 0 else f"warned={warned_candidates}",
        },
        {
            "id": "env_bias_allows",
            "ok": env_bias_ok,
            "detail": f"env={current_env} action={env_action}",
        },
    ]

    failed_checks = [c["id"] for c in checks if not c["ok"] and c["id"] != "warn_flags_present"]
    warning_checks = [c for c in checks if c["id"] == "warn_flags_present"]

    if not enter_exists_ok:
        reason = "ENTER candidates = 0"
    elif not no_ignored_ok:
        reason = "ignored flag present"
    elif not risk_mode_ok:
        reason = "risk_mode disallows"
    elif not env_bias_ok:
        reason = "env_bias disallows"
    else:
        reason = "checks passed"

    verdict = "ENTER_OK" if (enter_exists_ok and no_ignored_ok and risk_mode_ok and env_bias_ok) else "NO_TRADE"
    rules_digest = hashlib.sha1(
        json.dumps(rules_used, sort_keys=True, ensure_ascii=False).encode("utf-8")
    ).hexdigest()[:12]

    return {
        "schema_version": "agent_v1",
        "verdict": verdict,
        "reason": reason,
        "checks": checks,
        "failed_checks": failed_checks,
        "warning_checks": warning_checks,
        "computed_at_utc": _now_utc_iso(),
        "source": "step10_agent_rules_v1",
        "rules_source": rules_source,
        "rules_digest": rules_digest,
        "blocked_candidates": blocked_candidates,
        "warned_candidates": warned_candidates,
    }


def _evaluate_agent_verdict(
    summary: dict,
    enter_candidates: List[dict],
    rules: dict,
    rules_status: str,
    rules_source: Optional[str],
    rules_hash: Optional[str],
    env_bias_value: str,
    decision_rules_hash: str,
    theme_env_map: Dict[str, str],
    top_n: int,
) -> dict:
    enter_count = len(enter_candidates)
    require_min = int(rules.get("require_enter_min_count", 1) or 1)
    blocking_flags = [str(x) for x in rules.get("blocking_flags", []) if str(x)]
    risk_mode_allow = [str(x) for x in rules.get("risk_mode_allow", []) if str(x)]
    env_bias_deny = [str(x) for x in rules.get("env_bias_deny", []) if str(x)]

    checks: List[dict] = []
    failed_checks: List[str] = []

    rules_detail = "loaded" if rules_status == "ok" else rules_status
    rules_ok = rules_status == "ok"
    checks.append({"id": "rules_yaml_loaded", "ok": rules_ok, "detail": rules_detail})

    rules_hash_value = str(rules_hash or "")
    decision_hash_value = str(decision_rules_hash or "")
    rules_sync_ok = (
        rules_hash_value not in {"", "missing"}
        and decision_hash_value not in {"", "missing", "missing_in_decision"}
        and rules_hash_value == decision_hash_value
    )
    checks.append(
        {
            "id": "rules_sync_with_decision",
            "ok": rules_sync_ok,
            "detail": f"rules_hash_yaml={rules_hash_value} decision={decision_hash_value}",
        }
    )

    enter_ok = enter_count >= require_min

    if rules_status != "ok":
        verdict = "ENTER_OK" if enter_ok else "NO_TRADE"
        reason = f"fallback: rules.yaml {rules_status}"
        failed_checks.append("rules_yaml_loaded")
        if not enter_ok:
            failed_checks.append("enter_exists")
        return {
            "schema_version": "agent_v1",
            "verdict": verdict,
            "reason": reason,
            "checks": checks,
            "failed_checks": failed_checks,
            "rules_source": rules_source,
            "rules_hash": rules_hash,
            "computed_at_utc": _now_utc_iso(),
        }

    if not bool(rules.get("enabled", True)):
        verdict = "ENTER_OK" if enter_ok else "NO_TRADE"
        reason = "agent disabled"
        if not enter_ok:
            failed_checks.append("enter_exists")
        return {
            "schema_version": "agent_v1",
            "verdict": verdict,
            "reason": reason,
            "checks": checks,
            "failed_checks": failed_checks,
            "rules_source": rules_source,
            "rules_hash": rules_hash,
            "computed_at_utc": _now_utc_iso(),
        }

    checks.append({"id": "enter_exists", "ok": enter_ok, "detail": f"enter_count={enter_count} require={require_min}"})

    blocked = []
    if blocking_flags:
        candidate_flags: set[str] = set()
        for row in enter_candidates:
            flags = row.get("flags", [])
            if isinstance(flags, list):
                candidate_flags.update([str(x) for x in flags if str(x)])
            elif isinstance(flags, str) and flags:
                candidate_flags.add(flags)
        blocked = [f for f in blocking_flags if f in candidate_flags]
    blocked_preview = blocked[:3]
    checks.append(
        {
            "id": "no_blocking_flags",
            "ok": len(blocked) == 0,
            "detail": f"blocked={blocked_preview}" if blocked else "blocked=[]",
        }
    )

    risk_mode = str(summary.get("risk_mode", {}).get("mode", "unknown")).upper()
    checks.append(
        {
            "id": "risk_mode_allows",
            "ok": risk_mode in risk_mode_allow,
            "detail": f"risk_mode={risk_mode} allow={risk_mode_allow}",
        }
    )

    env_policy = rules.get("env_policy", {}) if isinstance(rules.get("env_policy", {}), dict) else {}
    env_policy_ok = True
    env_policy_detail = "skipped"
    if bool(env_policy.get("enabled", False)):
        deny_list = [str(x).lower() for x in env_policy.get("etf_env_deny", []) if str(x)]
        neutral_action = str(env_policy.get("etf_env_neutral_action", "allow")).lower()
        require_present = bool(env_policy.get("require_etf_env_present", False))
        checked = []
        env_policy_ok = True
        for row in enter_candidates[: max(1, top_n)]:
            theme = str(row.get("theme", "")).strip()
            env = str(theme_env_map.get(theme, "")).lower() if theme else ""
            if not env and require_present:
                env_policy_ok = False
            if env in deny_list:
                env_policy_ok = False
            if env == "neutral" and neutral_action == "deny":
                env_policy_ok = False
            if theme:
                checked.append(f"{theme}={env or 'missing'}")
        env_policy_detail = "themes_checked=" + ",".join(checked[:5]) + f" deny={deny_list}"
        checks.append({"id": "etf_env_policy", "ok": env_policy_ok, "detail": env_policy_detail})

    flag_weight_policy = rules.get("flag_weight_policy", {}) if isinstance(rules.get("flag_weight_policy", {}), dict) else {}
    flag_weight_ok = True
    flag_weight_detail = "skipped"
    if bool(flag_weight_policy.get("enabled", False)):
        min_effective_score = flag_weight_policy.get("min_effective_score")
        if min_effective_score is None:
            flag_weight_ok = True
            flag_weight_detail = "min not set"
        else:
            try:
                min_effective_score = float(min_effective_score)
            except Exception:
                min_effective_score = None
            weights = flag_weight_policy.get("weights", {}) if isinstance(flag_weight_policy.get("weights", {}), dict) else {}
            use_score_adjusted_first = bool(flag_weight_policy.get("use_score_adjusted_first", True))
            details = []
            flag_weight_ok = True
            for row in enter_candidates[: max(1, top_n)]:
                symbol = str(row.get("symbol", "")).upper()
                score_adjusted = row.get("score_adjusted")
                score_total = row.get("score_total")
                base = score_adjusted if use_score_adjusted_first and score_adjusted not in {None, ""} else score_total
                try:
                    base_val = float(base)
                except Exception:
                    base_val = 0.0
                effective = base_val
                if not (use_score_adjusted_first and score_adjusted not in {None, ""}):
                    flags = row.get("flags", [])
                    if isinstance(flags, list):
                        for f in flags:
                            if f in weights:
                                try:
                                    effective *= float(weights.get(f))
                                except Exception:
                                    continue
                    elif isinstance(flags, str) and flags in weights:
                        try:
                            effective *= float(weights.get(flags))
                        except Exception:
                            pass
                if min_effective_score is not None and effective < float(min_effective_score):
                    flag_weight_ok = False
                details.append(f"{symbol} base={base_val} eff={round(effective,2)}")
            flag_weight_detail = f"min={min_effective_score} top: " + "; ".join(details[:3])
        checks.append({"id": "flag_weight_effective_score", "ok": flag_weight_ok, "detail": flag_weight_detail})

    env_bias_value = str(env_bias_value or "")
    if env_bias_value:
        env_bias_upper = env_bias_value.upper()
        deny_upper = [d.upper() for d in env_bias_deny if d]
        checks.append(
            {
                "id": "env_bias_allows",
                "ok": env_bias_upper not in deny_upper,
                "detail": f"env_bias={env_bias_upper} deny={env_bias_deny}",
            }
        )

    for check in checks:
        if not check.get("ok") and check.get("id") != "rules_yaml_loaded":
            failed_checks.append(check.get("id"))

    verdict = "ENTER_OK"
    reason = "checks passed"
    if not rules_sync_ok:
        verdict = "NO_TRADE"
        reason = "rules mismatch; run step6"
    elif env_policy and bool(env_policy.get("enabled", False)) and not env_policy_ok:
        verdict = "NO_TRADE"
        reason = "etf env deny"
    elif len(blocked) > 0:
        verdict = "NO_TRADE"
        reason = "blocking flags"
    elif risk_mode not in risk_mode_allow:
        verdict = "NO_TRADE"
        reason = "risk mode deny"
    elif flag_weight_policy and bool(flag_weight_policy.get("enabled", False)) and not flag_weight_ok:
        verdict = "NO_TRADE"
        reason = "effective_score below min"
    elif not enter_ok:
        verdict = "NO_TRADE"
        reason = "no enter candidates"

    return {
        "schema_version": "agent_v1",
        "verdict": verdict,
        "reason": reason,
        "checks": checks,
        "failed_checks": failed_checks,
        "rules_source": rules_source,
        "rules_hash": rules_hash,
        "computed_at_utc": _now_utc_iso(),
    }


def main(argv: Optional[Sequence[str]] = None) -> int:
    ap = argparse.ArgumentParser(description="Step10 daily runner (summary + deviation + alerts)")
    ap.add_argument("--out", required=True, help="Out directory root")
    ap.add_argument("--window-days", type=int, default=7, help="Deviation window (placeholder)")
    ap.add_argument("--send", action="store_true", help="Send alerts via webhook (default: false)")
    ap.add_argument("--debug-webhook", action="store_true", help="Send a debug ping payload (default: false)")
    ap.add_argument("--agent-ext", action="store_true", help="Use external AI to add a short comment")
    ap.add_argument(
        "--agent-ext-kinds",
        default="ENTER_ALERT,NO_TRADE_NOTICE,DEVIATION_NOTICE",
        help="Comma-separated kinds to run external AI for",
    )
    ap.add_argument("--agent-ext-force", action="store_true", help="Force external AI even if deduped")
    ap.add_argument("--notify", action="store_true", help="Send notifications (default: false)")
    ap.add_argument("--notify-channel", default="discord", help="Notification channel (default: discord)")
    ap.add_argument("--discord-webhook", default="", help="Discord webhook URL (optional)")
    ap.add_argument("--notify-max-enter", type=int, default=5, help="Max ENTER rows to include in notice")
    ap.add_argument("--notify-also-no-trade", action="store_true", help="Also notify NO TRADE")
    ap.add_argument("--notify-dry-run", action="store_true", help="Build payload but do not send")
    ap.add_argument("--loglevel", default="INFO")
    args = ap.parse_args(argv)

    logging.basicConfig(level=getattr(logging, args.loglevel.upper(), logging.INFO))

    project_root = Path(__file__).resolve().parent.parent
    load_dotenv(project_root)
    env_path = project_root / ".env"
    env_data = _read_env_file(env_path)
    out_root = Path(args.out)
    alerts_config = _read_step10_alerts_config(ROOT / "config" / "step10_alerts.yaml")
    send_enabled = bool(args.send) and bool(alerts_config.get("enabled", False))
    debug_webhook = bool(args.debug_webhook)
    debug_ping_used = [False]
    max_items = int(alerts_config.get("max_items", 5) or 5)
    webhook_env = str(alerts_config.get("webhook_env", "STEP10_DISCORD_WEBHOOK"))
    webhook_url = get_dotenv(webhook_env)
    agent_ext_kinds = {k.strip() for k in str(args.agent_ext_kinds).split(",") if k.strip()}
    agent_ext_enabled = bool(args.agent_ext)
    agent_ext_force = bool(args.agent_ext_force)
    agent_ext_api_key = env_data.get("OPENAI_API_KEY", "")
    agent_ext_base_url = env_data.get("OPENAI_BASE_URL", "")
    agent_ext_model = env_data.get("OPENAI_MODEL", "") or AGENT_EXT_DEFAULT_MODEL

    # Debug: make send gating explicit in logs (helps diagnose "logged_only")
    send_gate_reason = ""
    if not args.send:
        send_gate_reason = "--send not specified"
    elif not bool(alerts_config.get("enabled", False)):
        send_gate_reason = "alerts disabled (config/step10_alerts.yaml enabled=false or missing)"
    elif not webhook_url:
        send_gate_reason = f"webhook not configured (.env missing {webhook_env})"

    LOG.info(
        "step10 alerts send: args.send=%s config.enabled=%s send_enabled=%s webhook_env=%s webhook_set=%s reason=%s",
        bool(args.send),
        bool(alerts_config.get("enabled", False)),
        bool(send_enabled),
        webhook_env,
        bool(webhook_url),
        send_gate_reason or "(ok)",
    )
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
        "message_preview": "",
        "warnings": [],
    }

    deviation: Dict[str, object] = {
        "schema_version": "step10_deviation_v2",
        "generated_at_utc": _now_utc_iso(),
        "asof_date_utc": "",
        "window_days": args.window_days,
        "window_detail": {},
        "window_start_date": "",
        "window_end_date": "",
        "no_trade_ignored_today": False,
        "deviation_day_hit_today": False,
        "unknown_symbol_today": False,
        "decision_enter_symbols": [],
        "trade_enter_symbols_today": [],
        "deviations_today": [],
        "deviation_events": [],
        "deviation_count_today": 0,
        "deviation_score_today": 0,
        "deviation_count_7d": 0,
        "deviation_score_7d": 0,
        "window_asof_dates": [],
        "missing_days": [],
        "level_7d": "UNKNOWN",
        "level_reason": "",
        "agent_ext_used": False,
        "counts": {
            "deviation_today": 0,
            "deviation_7d": 0,
            "unknown_symbol_days_7d": 0,
            "unknown_symbol_events_7d": 0,
            "no_trade_ignored_days_7d": 0,
            "no_trade_ignored_streak": 0,
            "no_trade_ignored_events_7d": 0,
            "missing_alert_days_7d": 0,
            "invalid_alert_lines_7d": 0,
        },
        "warning_level": "UNKNOWN",
        "warning_reason": "",
        "warning_reasons": [],
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
    if decision_status != "ok":
        decision_hash = "missing"
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
    rules_path = ROOT / "config" / "rules.yaml"
    rules_agent, rules_status, rules_hash = _read_agent_rules_yaml(rules_path)
    rules_hash_yaml = rules_digest.get_rules_hash(rules_path)
    rules_source = "config/rules.yaml" if rules_status == "ok" else None
    if rules_status == "missing":
        LOG.warning("rules.yaml missing: %s", _project_rel(rules_path, ROOT))
    elif rules_status == "parse_error":
        LOG.warning("rules.yaml parse_error: %s", _project_rel(rules_path, ROOT))

    env_bias_value = ""
    if decision_payload and isinstance(decision_payload.get("env_bias"), str):
        env_bias_value = str(decision_payload.get("env_bias", ""))
    elif isinstance(summary.get("env_bias"), str):
        env_bias_value = str(summary.get("env_bias", ""))

    decision_rules_hash = "missing_in_decision"
    if decision_payload:
        if isinstance(decision_payload.get("rules_hash"), str) and decision_payload.get("rules_hash"):
            decision_rules_hash = str(decision_payload.get("rules_hash"))
        elif isinstance(decision_payload.get("debug", {}), dict):
            debug_rules_hash = decision_payload.get("debug", {}).get("rules_hash")
            if isinstance(debug_rules_hash, str) and debug_rules_hash:
                decision_rules_hash = debug_rules_hash

    theme_env_map: Dict[str, str] = {}
    if decision_payload and isinstance(decision_payload.get("etf_daily_env"), list):
        for row in decision_payload.get("etf_daily_env", []):
            if not isinstance(row, dict):
                continue
            theme = str(row.get("theme", "")).strip()
            env = str(row.get("env", "")).strip()
            if theme:
                theme_env_map[theme] = env

    rules_max_enter = rules_agent.get("max_enter_to_show", 5)
    try:
        rules_max_enter = int(rules_max_enter)
    except Exception:
        rules_max_enter = 5

    summary["agent"] = _evaluate_agent_verdict(
        summary,
        enter_candidates,
        rules_agent,
        rules_status,
        rules_source,
        rules_hash_yaml,
        env_bias_value,
        decision_rules_hash,
        theme_env_map,
        rules_max_enter,
    )

    debug_step10 = summary.get("debug_step10", {}) if isinstance(summary.get("debug_step10", {}), dict) else {}
    debug_step10["rules_loaded"] = rules_status == "ok"
    if rules_hash_yaml:
        debug_step10["rules_hash"] = rules_hash_yaml
    summary["debug_step10"] = debug_step10

    if rules_max_enter > 0:
        max_items = min(max_items, rules_max_enter)

    if summary.get("agent", {}).get("verdict") == "NO_TRADE":
        summary["no_trade"] = {
            "is_no_trade": True,
            "reason": summary.get("agent", {}).get("reason", "NO_TRADE"),
        }
    agent_ext_verdict, agent_ext_used = _extract_agent_ext_verdict(decision_payload)

    today_trades = _read_trades(trades_dir, asof_date)
    trade_enter_symbols = sorted(
        {
            str(row.get("symbol", "")).upper()
            for row in today_trades
            if str(row.get("action", "")).upper() == "ENTER"
        }
    )
    if not trade_enter_symbols:
        trade_enter_symbols = sorted(
            {
                str(row.get("symbol", "")).upper()
                for row in today_trades
                if str(row.get("action", "")).upper() == "BUY"
            }
        )
    deviation["trade_enter_symbols_today"] = trade_enter_symbols

    deviations_today: List[dict] = []
    no_trade_ignored_added = False
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
            deviations_today.append(
                {
                    "type": "no_trade_ignored",
                    "symbol": "",
                    "action_ts_jst": "",
                    "details": {"source_type": "enter_when_no_trade", "symbols": trade_enter_symbols},
                }
            )
            no_trade_ignored_added = True
        for sym in trade_enter_symbols:
            if sym not in enter_symbols and sym not in avoid_symbols:
                deviations_today.append(
                    {
                        "type": "enter_not_in_enter_candidates",
                        "symbol": sym,
                        "action_ts_jst": "",
                        "details": {"reason": "not in decision ENTER", "unknown_symbol": True},
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
        if not no_trade_ignored_added and trade_enter_symbols:
            agent_verdict = summary.get("agent", {}).get("verdict")
            if agent_verdict == "NO_TRADE" or len(enter_candidates) == 0:
                deviations_today.append(
                    {
                        "type": "no_trade_ignored",
                        "symbol": "",
                        "action_ts_jst": "",
                        "details": {"source_type": "agent_no_trade", "symbols": trade_enter_symbols},
                    }
                )

        deviation["decision_enter_symbols"] = enter_symbols
        deviation["deviations_today"] = deviations_today
        deviation["counts"]["deviation_today"] = len(deviations_today)
    else:
        deviation["decision_enter_symbols"] = []
        deviation["deviations_today"] = []
        deviation["counts"]["deviation_today"] = 0

    deviation["deviation_day_hit_today"] = len(deviations_today) > 0
    deviation["unknown_symbol_today"] = any(
        d.get("type") == "enter_not_in_enter_candidates" for d in deviations_today
    )

    no_trade_ignored_today = (len(enter_symbols) == 0 or len(enter_candidates) == 0) and (len(trade_enter_symbols) > 0)
    agent_verdict = summary.get("agent", {}).get("verdict")
    if agent_verdict == "NO_TRADE" and len(trade_enter_symbols) > 0:
        no_trade_ignored_today = True
    deviation["no_trade_ignored_today"] = bool(no_trade_ignored_today)

    deviation_today = _compute_deviation_today(
        enter_symbols,
        trade_enter_symbols,
        agent_verdict,
        agent_ext_verdict,
        agent_ext_used,
    )
    deviation["deviation_events"] = deviation_today.get("deviation_events", [])
    deviation["deviation_count_today"] = deviation_today.get("deviation_count_today", 0)
    deviation["deviation_score_today"] = deviation_today.get("deviation_score_today", 0)
    deviation["agent_ext_used"] = deviation_today.get("agent_ext_used", False)

    window_dates = _window_dates(asof_date, args.window_days)
    deviation["window_detail"] = {
        "window_days": args.window_days,
        "source": "deviation_files",
        "observed_days": 0,
        "files_read_days": [],
        "alerts_fallback_days": [],
        "missing_days": [],
        "unknown_symbol_top": [],
    }
    if window_dates:
        deviation["window_start_date"] = window_dates[0]
        deviation["window_end_date"] = window_dates[-1]

    warning_reasons: List[str] = []
    if not window_dates:
        deviation["warning_level"] = "UNKNOWN"
        warning_reasons.append("window_dates_invalid")
    else:
        hit_days: set[str] = set()
        unknown_symbol_days: set[str] = set()
        no_trade_ignored_days: set[str] = set()
        unknown_symbol_events = 0
        no_trade_ignored_events = 0
        symbol_counts: Dict[str, int] = {}
        files_read_days: List[str] = []
        missing_days: List[str] = []
        for day in window_dates:
            payload = _load_deviation_for_date(out_dir, day)
            if not payload:
                missing_days.append(day)
                continue
            files_read_days.append(day)
            if payload.get("deviation_day_hit_today") is True:
                hit_days.add(day)
            else:
                deviations = payload.get("deviations_today", [])
                if isinstance(deviations, list) and deviations:
                    hit_days.add(day)
            if payload.get("unknown_symbol_today") is True:
                unknown_symbol_days.add(day)
            else:
                deviations = payload.get("deviations_today", [])
                if isinstance(deviations, list):
                    for d in deviations:
                        if d.get("type") == "enter_not_in_enter_candidates" or d.get("details", {}).get("unknown_symbol") is True:
                            unknown_symbol_days.add(day)
                            sym = str(d.get("symbol", "")).upper()
                            if sym:
                                symbol_counts[sym] = symbol_counts.get(sym, 0) + 1
                            unknown_symbol_events += 1
            if payload.get("no_trade_ignored_today") is True:
                no_trade_ignored_days.add(day)
            else:
                deviations = payload.get("deviations_today", [])
                if isinstance(deviations, list):
                    for d in deviations:
                        if d.get("type") in {"no_trade_ignored", "enter_when_no_trade"}:
                            no_trade_ignored_days.add(day)
                            no_trade_ignored_events += 1

        alerts_fallback_days: List[str] = []
        invalid_alert_lines = 0

        observed_days = len(files_read_days) + len(alerts_fallback_days)
        deviation["window_detail"]["observed_days"] = observed_days
        deviation["window_detail"]["files_read_days"] = files_read_days
        deviation["window_detail"]["alerts_fallback_days"] = alerts_fallback_days
        deviation["window_detail"]["missing_days"] = missing_days
        deviation["window_detail"]["source_breakdown"] = {
            "files": len(files_read_days),
            "alerts": len(alerts_fallback_days),
            "missing": len(missing_days),
        }

        top_symbols = sorted(symbol_counts.items(), key=lambda x: (-x[1], x[0]))[:3]
        deviation["window_detail"]["unknown_symbol_top"] = [
            {"symbol": sym, "count": cnt} for sym, cnt in top_symbols
        ]

        deviation["counts"]["missing_alert_days_7d"] = len(missing_days)
        deviation["counts"]["invalid_alert_lines_7d"] = invalid_alert_lines

        if observed_days < 4:
            deviation["warning_level"] = "UNKNOWN"
            warning_reasons.append("observed_days<4")
        else:
            deviation_7d = len(hit_days)
            unknown_symbol_days_7d = len(unknown_symbol_days)
            no_trade_ignored_days_7d = len(no_trade_ignored_days)
            no_trade_ignored_streak = _compute_streak(no_trade_ignored_days, window_dates)

            deviation["counts"]["deviation_7d"] = deviation_7d
            deviation["counts"]["unknown_symbol_days_7d"] = unknown_symbol_days_7d
            deviation["counts"]["no_trade_ignored_days_7d"] = no_trade_ignored_days_7d
            deviation["counts"]["no_trade_ignored_streak"] = no_trade_ignored_streak
            deviation["counts"]["unknown_symbol_events_7d"] = unknown_symbol_events
            deviation["counts"]["no_trade_ignored_events_7d"] = no_trade_ignored_events

            warning_level = "OK"
            if deviation_7d >= 4:
                warning_reasons.append("deviation_7d>=4")
                warning_level = "CRITICAL"
            elif deviation_7d >= 2:
                warning_reasons.append("deviation_7d>=2")
                warning_level = "WARN"
            else:
                warning_reasons.append("deviation_7d<2")

            if unknown_symbol_days_7d >= 2:
                warning_reasons.append("unknown_symbol_days_7d>=2")
            if no_trade_ignored_streak >= 2:
                warning_reasons.append("no_trade_ignored_streak>=2")

            deviation["warning_level"] = warning_level

    deviation["warning_reasons"] = warning_reasons
    deviation["warning_reason"] = warning_reasons[0] if warning_reasons else ""

    deviation_meta = _compute_deviation_7d(out_dir, asof_date, args.window_days, deviation)
    deviation["deviation_count_7d"] = deviation_meta.get("deviation_count_7d", 0)
    deviation["deviation_score_7d"] = deviation_meta.get("deviation_score_7d", 0)
    deviation["window_asof_dates"] = deviation_meta.get("window_asof_dates", [])
    deviation["missing_days"] = deviation_meta.get("missing_days", [])
    deviation["level_7d"] = deviation_meta.get("level_7d", "UNKNOWN")
    deviation["level_reason"] = deviation_meta.get("level_reason", "")

    watchlist_path = out_dir / f"watchlist_enter_{asof_date}.txt"
    watchlist_latest = out_dir / "watchlist_enter_latest.txt"
    watchlist_symbols = sorted({str(s).upper() for s in enter_symbols if str(s)})
    watchlist_content = "\n".join(watchlist_symbols) + ("\n" if watchlist_symbols else "")
    watchlist_path.write_text(watchlist_content, encoding="utf-8")
    watchlist_latest.write_text(watchlist_content, encoding="utf-8")
    summary["watchlist_path"] = _relative_path(watchlist_path)

    alerts_path = out_dir / "alerts.jsonl"
    existing_keys = _read_alerts(alerts_path)
    decision_hash = summary.get("decision_digest", {}).get("decision_hash", "")
    decision_id = summary.get("decision_digest", {}).get("decision_id", "")
    risk_mode = summary.get("risk_mode", {}) if isinstance(summary.get("risk_mode", {}), dict) else {}
    event_id_run_summary = f"{asof_date}:{decision_id}:RUN_SUMMARY"
    agent = summary.get("agent", {}) if isinstance(summary.get("agent", {}), dict) else {}
    agent_checks = agent.get("checks", []) if isinstance(agent.get("checks", []), list) else []
    rules_sync_check = next((c for c in agent_checks if c.get("id") == "rules_sync_with_decision"), None)
    agent_failed_checks = agent.get("failed_checks", []) if isinstance(agent.get("failed_checks", []), list) else []
    agent_warning_checks = agent.get("warning_checks", []) if isinstance(agent.get("warning_checks", []), list) else []

    message = build_notification_message(summary, max_items)
    if isinstance(message, str):
        summary["message_preview"] = " / ".join(message.splitlines()[:3])
    else:
        summary["message_preview"] = ""

    summary_path = out_dir / f"summary_{asof_date}.json"
    deviation_path = out_dir / f"deviation_{asof_date}.json"
    summary_latest = out_dir / "summary_latest.json"
    deviation_latest = out_dir / "deviation_latest.json"

    summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    deviation_path.write_text(json.dumps(deviation, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    summary_latest.write_text(summary_path.read_text(encoding="utf-8"), encoding="utf-8")
    deviation_latest.write_text(deviation_path.read_text(encoding="utf-8"), encoding="utf-8")

    _append_alert(
        alerts_path,
        event_id_run_summary,
        "RUN_SUMMARY",
        _apply_agent_ext(
            {
            "asof_date_utc": asof_date,
            **_build_common_payload(summary, event_id_run_summary, decision_id, decision_hash, risk_mode),
            "enter_candidates": len(enter_candidates),
            "trade_enter_symbols": len(trade_enter_symbols),
            "trade_enter_symbols_count": len(trade_enter_symbols),
            "counts": summary.get("counts", {}),
            "result_detail": "logged_only",
            "agent_verdict": agent.get("verdict"),
            "agent_reason": agent.get("reason"),
            "agent_failed_checks": agent_failed_checks,
            "agent_checks": agent_checks,
            "rules_source": agent.get("rules_source"),
            "rules_hash": agent.get("rules_hash"),
            "rules_hash_yaml": rules_hash_yaml,
            "rules_sync_with_decision": rules_sync_check,
            "agent_rules_digest": agent.get("rules_digest"),
            "agent_warning_checks": agent_warning_checks,
        },
            "RUN_SUMMARY",
            event_id_run_summary,
            alerts_path,
            agent_ext_enabled,
            agent_ext_kinds,
            agent_ext_force,
            agent_ext_api_key,
            agent_ext_base_url,
            agent_ext_model,
        ),
        existing_keys,
        dedup_key=f"{asof_date}:RUN_SUMMARY",
    )

    _maybe_emit_deviation_notice(
        alerts_path,
        existing_keys,
        summary,
        asof_date,
        decision_id,
        decision_hash,
        risk_mode,
        send_enabled,
        webhook_url,
        webhook_env,
        debug_webhook,
        debug_ping_used,
        send_gate_reason,
        {
            "level_7d": deviation.get("level_7d"),
            "deviation_count_7d": deviation.get("deviation_count_7d", 0),
            "deviation_score_7d": deviation.get("deviation_score_7d", 0),
            "window_days": deviation.get("window_days", args.window_days),
            "window_asof_dates": deviation.get("window_asof_dates", []),
            "missing_days": deviation.get("missing_days", []),
        },
        deviation.get("deviation_events", []) if isinstance(deviation.get("deviation_events", []), list) else [],
        deviation.get("agent_ext_used", False),
        agent_ext_enabled,
        agent_ext_kinds,
        agent_ext_force,
        agent_ext_api_key,
        agent_ext_base_url,
        agent_ext_model,
    )

    deviation_counts = deviation.get("counts", {}) if isinstance(deviation.get("counts", {}), dict) else {}
    warning_level = str(deviation.get("warning_level", "")).upper()
    if warning_level in {"WARN", "CRITICAL"}:
        kind = "DEVIATION_WARN" if warning_level == "WARN" else "DEVIATION_CRITICAL"
        deviation_event_id = f"{asof_date}:{decision_id}:{kind}"
        warning_reasons = deviation.get("warning_reasons", []) if isinstance(deviation.get("warning_reasons", []), list) else []
        window_detail = deviation.get("window_detail", {}) if isinstance(deviation.get("window_detail", {}), dict) else {}
        message_text = build_deviation_warning_message(summary, deviation)
        message_preview = " / ".join(message_text.splitlines()[:3]) if isinstance(message_text, str) else ""
        payload = {
            "asof_date_utc": asof_date,
            **_build_common_payload(summary, deviation_event_id, decision_id, decision_hash, risk_mode),
            "warning_level": warning_level,
            "warning_reasons": warning_reasons,
            "counts": {
                "deviation_7d": deviation_counts.get("deviation_7d", 0),
                "unknown_symbol_days_7d": deviation_counts.get("unknown_symbol_days_7d", 0),
                "no_trade_ignored_days_7d": deviation_counts.get("no_trade_ignored_days_7d", 0),
                "no_trade_ignored_streak": deviation_counts.get("no_trade_ignored_streak", 0),
            },
            "window": {
                "window_days": deviation.get("window_days", 7),
                "window_start_date": deviation.get("window_start_date", ""),
                "window_end_date": deviation.get("window_end_date", ""),
                "observed_days": window_detail.get("observed_days", 0),
            },
            "message_preview": message_preview,
        }
        payload = _apply_agent_ext(
            payload,
            kind,
            deviation_event_id,
            alerts_path,
            agent_ext_enabled,
            agent_ext_kinds,
            agent_ext_force,
            agent_ext_api_key,
            agent_ext_base_url,
            agent_ext_model,
        )
        if deviation_event_id in existing_keys:
            payload["send_result"] = "skipped_duplicate"
        elif send_enabled and webhook_url:
            debug_ping = debug_webhook and not debug_ping_used[0]
            ok, status, err = post_discord(webhook_url, message_text, webhook_env, debug_ping)
            if debug_ping:
                debug_ping_used[0] = True
            if ok:
                payload["send_result"] = "sent"
                payload["http_status"] = status
            else:
                payload["send_result"] = "failed"
                payload["http_status"] = status
                payload["reason"] = err
                payload["webhook_hash"] = _sha12(webhook_url) if webhook_url else ""
                payload["proxy_flags"] = _proxy_env_flags()
                payload["sys_executable"] = sys.executable
        else:
            payload["send_result"] = "logged_only"

        if payload.get("send_result") != "skipped_duplicate":
            _append_alert(
                alerts_path,
                deviation_event_id,
                kind,
                payload,
                existing_keys,
                dedup_key=f"{asof_date}:{kind}",
            )

    enter_symbols_unique = sorted(set(enter_symbols))

    enter_event_id = f"{asof_date}:{decision_id}:{_enter_symbols_hash(enter_symbols)}:ENTER_ALERT"
    if enter_candidates and agent.get("verdict") == "ENTER_OK":
        send_result = "logged_only"
        send_error = None
        send_status = None
        if enter_event_id in existing_keys:
            send_result = "skipped_duplicate"
        elif send_enabled and webhook_url:
            debug_ping = debug_webhook and not debug_ping_used[0]
            ok, status, err = post_discord(webhook_url, message, webhook_env, debug_ping)
            if debug_ping:
                debug_ping_used[0] = True
            send_status = status
            if ok:
                send_result = "sent"
            else:
                send_result = "failed"
                send_error = f"{err or 'send_failed'}"
        elif send_enabled and not webhook_url:
            send_result = "logged_only"
            send_error = "webhook not configured (.env missing)"
        else:
            # Send is disabled by gating (missing --send or enabled=false)
            send_error = send_gate_reason or "send disabled"
        payload = {
            "asof_date_utc": asof_date,
            **_build_common_payload(summary, enter_event_id, decision_id, decision_hash, risk_mode),
            "count": len(enter_candidates),
            "top": [
                {
                    "symbol": row.get("symbol", ""),
                    "theme": row.get("theme", ""),
                    "score_total": row.get("score_total", ""),
                    "score_adjusted": row.get("score_adjusted", ""),
                    "threshold_used": row.get("threshold_used", ""),
                    "flags": row.get("flags", []),
                    "rules_applied": row.get("rules_applied", []),
                }
                for row in enter_candidates[:max_items]
            ],
            "watchlist_path": _safe_watchlist_path(summary.get("watchlist_path", "N/A")),
            "message_preview": message,
            "result": send_result,
            "send_result": send_result,
            "agent_verdict": agent.get("verdict"),
            "agent_reason": agent.get("reason"),
            "agent_failed_checks": agent_failed_checks,
            "agent_checks": agent_checks,
            "rules_source": agent.get("rules_source"),
            "rules_hash": agent.get("rules_hash"),
            "rules_hash_yaml": rules_hash_yaml,
            "rules_sync_with_decision": rules_sync_check,
            "agent_rules_digest": agent.get("rules_digest"),
            "agent_warning_checks": agent_warning_checks,
        }
        payload = _apply_agent_ext(
            payload,
            "ENTER_ALERT",
            enter_event_id,
            alerts_path,
            agent_ext_enabled,
            agent_ext_kinds,
            agent_ext_force,
            agent_ext_api_key,
            agent_ext_base_url,
            agent_ext_model,
        )
        if send_error:
            payload["reason"] = send_error
        if send_status is not None:
            payload["http_status"] = send_status
        if send_result == "failed":
            payload["webhook_hash"] = _sha12(webhook_url) if webhook_url else ""
            payload["proxy_flags"] = _proxy_env_flags()
            payload["sys_executable"] = sys.executable
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
        no_trade_event_id = f"{asof_date}:{decision_id}:{decision_hash}:NO_TRADE_NOTICE"
        send_result = "logged_only"
        send_error = None
        send_status = None
        no_trade_message = build_no_trade_message(
            summary,
            summary["no_trade"].get("reason", ""),
            agent.get("rules_hash"),
        )
        if no_trade_event_id in existing_keys:
            send_result = "skipped_duplicate"
        elif send_enabled and webhook_url:
            debug_ping = debug_webhook and not debug_ping_used[0]
            ok, status, err = post_discord(webhook_url, no_trade_message, webhook_env, debug_ping)
            if debug_ping:
                debug_ping_used[0] = True
            send_status = status
            if ok:
                send_result = "sent"
            else:
                send_result = "failed"
                send_error = f"{err or 'send_failed'}"
        elif send_enabled and not webhook_url:
            send_result = "logged_only"
            send_error = "webhook not configured (.env missing)"
        else:
            # Send is disabled by gating (missing --send or enabled=false)
            send_error = send_gate_reason or "send disabled"
        payload = {
            "asof_date_utc": asof_date,
            **_build_common_payload(summary, no_trade_event_id, decision_id, decision_hash, risk_mode),
            "reason": summary["no_trade"].get("reason", ""),
            "message_preview": no_trade_message,
            "result": send_result,
            "send_result": send_result,
            "agent_verdict": agent.get("verdict"),
            "agent_reason": agent.get("reason"),
            "agent_failed_checks": agent_failed_checks,
            "agent_checks": agent_checks,
            "rules_source": agent.get("rules_source"),
            "rules_hash": agent.get("rules_hash"),
            "rules_hash_yaml": rules_hash_yaml,
            "rules_sync_with_decision": rules_sync_check,
            "agent_rules_digest": agent.get("rules_digest"),
            "agent_warning_checks": agent_warning_checks,
        }
        payload = _apply_agent_ext(
            payload,
            "NO_TRADE_NOTICE",
            no_trade_event_id,
            alerts_path,
            agent_ext_enabled,
            agent_ext_kinds,
            agent_ext_force,
            agent_ext_api_key,
            agent_ext_base_url,
            agent_ext_model,
        )
        if send_error:
            payload["reason"] = send_error
        if send_status is not None:
            payload["http_status"] = send_status
        if send_result == "failed":
            payload["webhook_hash"] = _sha12(webhook_url) if webhook_url else ""
            payload["proxy_flags"] = _proxy_env_flags()
            payload["sys_executable"] = sys.executable
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
