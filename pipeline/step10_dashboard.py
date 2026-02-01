from __future__ import annotations

import argparse
import json
import sys
from collections import deque
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

import pandas as pd
import streamlit as st

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))


def _load_json(path: Path) -> Tuple[Optional[Dict[str, Any]], Optional[str]]:
    if not path.exists():
        return None, "missing"
    try:
        return json.loads(path.read_text(encoding="utf-8")), None
    except Exception as e:
        return None, f"invalid_json: {e}"


def _short_path(path: str, max_len: int = 48) -> str:
    if not path:
        return "N/A"
    if "/Users/" in path:
        return "N/A"
    if len(path) <= max_len:
        return path
    return "..." + path[-max_len:]


def _short_hash(val: str) -> str:
    if not val:
        return "N/A"
    return val[:8]


def _parse_date_from_name(name: str) -> str:
    stem = Path(name).stem
    if stem.startswith("summary_"):
        return stem.replace("summary_", "")
    if stem.startswith("deviation_"):
        return stem.replace("deviation_", "")
    return stem


def _candidate_dates(out_dir: Path) -> list[str]:
    dates = set()
    for p in out_dir.glob("summary_*.json"):
        dates.add(_parse_date_from_name(p.name))
    return sorted(dates, reverse=True)


def _enter_table(rows: list[dict]) -> pd.DataFrame:
    if not rows:
        return pd.DataFrame(columns=["symbol", "theme", "score_total", "score_adjusted", "threshold_used", "rules_applied", "flags"])
    df = pd.DataFrame(rows)
    df["rules_applied"] = df.get("rules_applied", "").apply(lambda v: ";".join(v) if isinstance(v, list) else v)
    df["flags"] = df.get("flags", "").apply(lambda v: ";".join(v) if isinstance(v, list) else v)
    df["score_adjusted_num"] = pd.to_numeric(df.get("score_adjusted"), errors="coerce")
    df["score_total_num"] = pd.to_numeric(df.get("score_total"), errors="coerce")
    df = df.sort_values(
        ["score_adjusted_num", "score_total_num", "symbol"],
        ascending=[False, False, True],
        na_position="last",
    )
    return df[["symbol", "theme", "score_total", "score_adjusted", "threshold_used", "rules_applied", "flags"]]


def _threshold_value(threshold_used: str) -> Optional[float]:
    if not threshold_used or "=" not in threshold_used:
        return None
    try:
        return float(threshold_used.split("=", 1)[1])
    except Exception:
        return None


def _sanitize_text(value: object) -> str:
    text = str(value) if value is not None else ""
    if "/Users/" in text:
        return ""
    return text


def _safe_rel_display(path_str: object) -> str:
    if not isinstance(path_str, str) or not path_str:
        return "N/A"
    if path_str.startswith("/"):
        return "N/A"
    if "/Users/" in path_str:
        return "N/A"
    return path_str


def _safe_path(value: Optional[str]) -> str:
    if value is None:
        return "N/A"
    if not isinstance(value, str) or not value:
        return "N/A"
    if value.startswith("/"):
        return "(redacted)"
    return value


def _read_json(path: Path) -> Optional[Dict[str, Any]]:
    if not path.exists():
        return None
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return None


def _safe_tail_jsonl(path: Path, n: int) -> Tuple[list[dict], int]:
    if not path.exists():
        return [], 0
    items: deque = deque(maxlen=max(n, 1))
    invalid = 0
    with path.open(encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                items.append(json.loads(line))
            except Exception:
                invalid += 1
    return list(items), invalid


def _load_latest_alerts_for_asof(alerts_path: Path, asof: str) -> Dict[str, dict]:
    if not alerts_path.exists():
        return {}
    lines: deque = deque(maxlen=200)
    with alerts_path.open(encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                lines.append(line)
    latest: Dict[str, dict] = {}
    for line in reversed(lines):
        try:
            rec = json.loads(line)
        except Exception:
            continue
        if rec.get("asof_date_utc") != asof:
            continue
        kind = rec.get("kind", "")
        if kind not in {"ENTER_ALERT", "NO_TRADE_NOTICE", "DEVIATION_NOTICE"}:
            continue
        if kind in latest:
            continue
        latest[kind] = rec
        if len(latest) >= 3:
            break
    return latest


def _preview_text(rec: Optional[dict]) -> str:
    if not rec:
        return "(no preview)"
    payload = rec.get("payload", {}) if isinstance(rec.get("payload", {}), dict) else {}
    text = payload.get("message_preview") or payload.get("message") or "(no preview)"
    if not isinstance(text, str):
        text = str(text)
    lines = text.splitlines()
    cleaned = []
    for line in lines:
        cleaned.append("(redacted)" if line.startswith("/") else line)
    return "\n".join(cleaned)


def _shorten(text: str, n: int = 600) -> str:
    if not isinstance(text, str):
        text = str(text)
    if len(text) <= n:
        return text
    return text[:n] + "\n...(truncated)"


def main() -> None:
    st.set_page_config(layout="wide", page_title="Step10 Daily", initial_sidebar_state="collapsed")

    ap = argparse.ArgumentParser(description="Step10 Daily Dashboard")
    ap.add_argument("--out", default="out")
    ap.add_argument("--date", default="latest")
    ap.add_argument("--latest", action="store_true")
    ap.add_argument("--alerts-tail", type=int, default=200)
    args = ap.parse_args()

    out_root = Path(args.out)
    daily_dir = out_root / "step10_daily"

    st.sidebar.header("Inputs")
    out_str = st.sidebar.text_input("Out directory", value=str(out_root))
    out_root = Path(out_str)
    daily_dir = out_root / "step10_daily"

    dates = _candidate_dates(daily_dir)
    date_options = ["latest"] + dates
    selected_date = st.sidebar.selectbox("Date", date_options, index=0)

    if selected_date == "latest":
        summary_path = daily_dir / "summary_latest.json"
        deviation_path = daily_dir / "deviation_latest.json"
    else:
        summary_path = daily_dir / f"summary_{selected_date}.json"
        deviation_path = daily_dir / f"deviation_{selected_date}.json"

    summary, summary_err = _load_json(summary_path)
    deviation, deviation_err = _load_json(deviation_path)

    if summary_err:
        st.warning(f"summary: {summary_err}")
    if deviation_err:
        st.info(f"deviation: {deviation_err}")

    summary = summary or {}
    deviation = deviation or {}

    asof_date = summary.get("asof_date_utc") or "N/A"
    generated_at = summary.get("generated_at_utc") or "N/A"
    risk_mode = summary.get("risk_mode", {}) if isinstance(summary.get("risk_mode", {}), dict) else {}
    risk_mode_label = f"{risk_mode.get('mode','unknown')} / {risk_mode.get('strength','-')}"
    decision_hash = _short_hash(summary.get("decision_digest", {}).get("decision_hash", ""))
    decision_path = _short_path(summary.get("decision_path", ""))

    alerts_path = daily_dir / "alerts.jsonl"
    watchlist_latest = daily_dir / "watchlist_enter_latest.txt"

    agent = summary.get("agent", {}) if isinstance(summary.get("agent", {}), dict) else {}
    agent_verdict = agent.get("verdict", "N/A")
    agent_reason = agent.get("reason", "N/A")
    failed_checks = agent.get("failed_checks", []) if isinstance(agent.get("failed_checks", []), list) else []

    warning_level = deviation.get("warning_level", "UNKNOWN")
    warning_reasons = deviation.get("warning_reasons", None)
    if isinstance(warning_reasons, list):
        warning_reason_text = "; ".join([str(x) for x in warning_reasons[:3] if str(x)])
    else:
        warning_reason_text = str(deviation.get("warning_reason", "") or "")
    counts = deviation.get("counts", {}) if isinstance(deviation.get("counts", {}), dict) else {}
    deviation_7d = int(counts.get("deviation_7d", 0) or 0)
    unknown_symbol_days_7d = int(counts.get("unknown_symbol_days_7d", 0) or 0)
    no_trade_ignored_streak = int(counts.get("no_trade_ignored_streak", 0) or 0)

    header_cols = st.columns(3)
    header_cols[0].markdown(f"**asof_date_utc**: {asof_date}  \n**generated_at_utc**: {generated_at}")
    header_cols[1].markdown(f"**risk_mode**: {risk_mode_label}  \n**agent**: {agent_verdict} ({agent_reason})")
    header_cols[2].markdown(
        f"**decision_hash**: {decision_hash}  \n**decision_path**: {decision_path}  "
        f"\n**deviation**: {warning_level} ({warning_reason_text})"
    )

    st.subheader("Status")
    decision_latest_path = out_root / "step6_decision" / "decision_latest.json"
    decision_latest = _read_json(decision_latest_path) or {}
    decision_generated = decision_latest.get("generated_at_utc", "N/A")
    decision_asof = decision_latest.get("asof_date_utc", "N/A")
    decision_path_display = _safe_path(summary.get("decision_path"))
    st.write(
        f"Step10: generated_at_utc={generated_at} / asof_date_utc={asof_date} / decision_path={decision_path_display}"
    )
    st.write(f"Step6: generated_at_utc={decision_generated} / asof_date_utc={decision_asof}")
    agent_checks = agent.get("checks", []) if isinstance(agent.get("checks", []), list) else []
    rules_sync = next((c for c in agent_checks if c.get("id") == "rules_sync_with_decision"), None)
    if rules_sync:
        detail = _shorten(str(rules_sync.get("detail", "")))
        st.write(f"rules_sync_with_decision: ok={rules_sync.get('ok')} detail={detail}")
        if rules_sync.get("ok") is False:
            st.error("rules mismatch; run step6")
    else:
        st.write("rules_sync_with_decision: N/A")
    deviation_level = deviation.get("level_7d") or deviation.get("warning_level", "N/A")
    deviation_count = deviation.get("deviation_count_7d", deviation.get("counts", {}).get("deviation_7d", 0))
    st.write(f"deviation: level_7d={deviation_level} deviation_count_7d={deviation_count}")

    st.subheader(f"Latest alerts (asof={asof_date})")
    alerts_path = daily_dir / "alerts.jsonl"
    latest_alerts = _load_latest_alerts_for_asof(alerts_path, asof_date)
    for kind in ["ENTER_ALERT", "NO_TRADE_NOTICE", "DEVIATION_NOTICE"]:
        rec = latest_alerts.get(kind)
        if not rec:
            st.info(f"{kind}: N/A")
            continue
        payload = rec.get("payload", {}) if isinstance(rec.get("payload", {}), dict) else {}
        send_result = payload.get("send_result") or payload.get("result") or "N/A"
        http_status = payload.get("http_status")
        ts_utc = rec.get("ts_utc", "N/A")
        meta = f"ts_utc={ts_utc} send_result={send_result}"
        if http_status:
            meta += f" http_status={http_status}"
        if kind == "DEVIATION_NOTICE":
            level = payload.get("level", "N/A")
            count_7d = payload.get("deviation_count_7d", "N/A")
            meta += f" level={level} deviation_count_7d={count_7d}"
        st.markdown(f"### {kind}")
        st.write(meta)
        preview = _shorten(_preview_text(rec))
        st.code(preview)

    st.markdown("### Today")
    st.write(f"asof_date_utc: {asof_date} / generated_at_utc: {generated_at}")
    st.write(f"risk_mode: {risk_mode_label}")
    decision_id = summary.get("decision_digest", {}).get("decision_id", "N/A")
    st.write(f"decision_id: {decision_id}")
    st.markdown(f"**agent verdict**: {agent_verdict}")
    st.write(f"agent reason: {agent_reason}")
    if failed_checks:
        st.write("failed_checks:")
        for c in failed_checks:
            st.write(f"- {c}")
    rules_source = agent.get("rules_source")
    rules_digest = agent.get("rules_digest")
    if rules_source or rules_digest:
        st.caption(f"rules_source={rules_source or 'N/A'} / rules_digest={rules_digest or 'N/A'}")

    no_trade = summary.get("no_trade", {}) if isinstance(summary.get("no_trade", {}), dict) else {}
    is_no_trade = bool(no_trade.get("is_no_trade", True)) if summary else True
    counts = summary.get("counts", {}) if isinstance(summary.get("counts", {}), dict) else {}
    enter_count = int(counts.get("enter", 0) or 0)
    watch_count = int(counts.get("watch", 0) or 0)
    avoid_count = int(counts.get("avoid", 0) or 0)

    if agent_verdict == "NO_TRADE":
        st.markdown("## 今日は何もしない日です")
        st.error("今日は何もしない日です")
        reason = no_trade.get("reason", "summary missing")
        st.write(f"reason: {reason}")
        st.write(f"agent_reason: {agent_reason}")
        st.write(f"ENTER={enter_count} WATCH={watch_count} AVOID={avoid_count}")
    elif agent_verdict == "ENTER_OK":
        st.markdown(f"## ENTER候補: {enter_count}件")
        tradable = summary.get("tradable_themes", [])
        st.write("tradable_themes:", ", ".join(tradable) if tradable else "N/A")
        st.write("今日見るのは ENTER だけ")
    elif is_no_trade:
        st.markdown("## 今日は何もしない日です")
        st.warning("今日は何もしない日です")
        reason = no_trade.get("reason", "summary missing")
        st.write(f"reason: {reason}")
        st.write(f"ENTER={enter_count} WATCH={watch_count} AVOID={avoid_count}")
    else:
        st.markdown(f"## ENTER候補: {enter_count}件")
        tradable = summary.get("tradable_themes", [])
        st.write("tradable_themes:", ", ".join(tradable) if tradable else "N/A")
        st.write("今日見るのは ENTER だけ")

    if warning_level == "OK":
        st.success("逸脱: OK")
    elif warning_level == "WARN":
        st.warning("逸脱: WARN")
        st.write(
            f"window: deviation_7d={deviation_7d}, "
            f"unknown_symbol_days_7d={unknown_symbol_days_7d}, "
            f"no_trade_ignored_streak={no_trade_ignored_streak}"
        )
    elif warning_level == "CRITICAL":
        st.error("逸脱: CRITICAL")
        st.write(
            f"window: deviation_7d={deviation_7d}, "
            f"unknown_symbol_days_7d={unknown_symbol_days_7d}, "
            f"no_trade_ignored_streak={no_trade_ignored_streak}"
        )
    else:
        st.warning("逸脱: UNKNOWN")

    if warning_reason_text:
        st.write(f"warning_reason: {warning_reason_text}")

    deviations = deviation.get("deviations_today", []) if isinstance(deviation.get("deviations_today", []), list) else []
    if deviations:
        dev_rows = []
        for d in deviations:
            details = d.get("details", {}) if isinstance(d.get("details", {}), dict) else {}
            dev_rows.append(
                {"type": d.get("type", ""), "symbol": d.get("symbol", ""), "reason": details.get("reason", "")}
            )
        dev_df = pd.DataFrame(dev_rows)
        st.dataframe(dev_df, use_container_width=True, hide_index=True)
    else:
        st.write("逸脱: なし")

    with st.expander("Deviation context"):
        st.write("trade_enter_symbols_today:", deviation.get("trade_enter_symbols_today", []))
        st.write("decision_enter_symbols:", deviation.get("decision_enter_symbols", []))

    st.markdown("## ENTER Candidates")
    enter_rows = summary.get("enter_candidates", []) if isinstance(summary.get("enter_candidates", []), list) else []
    if enter_count <= 0:
        st.write("ENTER候補: なし")
    else:
        enter_df = _enter_table(enter_rows).head(5)
        if not enter_df.empty:
            enter_df.insert(0, "rank", range(1, len(enter_df) + 1))
            enter_df["flags"] = enter_df["flags"].apply(lambda v: v if v else "clean")

            def _row_style(row: pd.Series) -> pd.Series:
                threshold = _threshold_value(str(row.get("threshold_used", "")))
                try:
                    score_adj = float(row.get("score_adjusted"))
                except Exception:
                    score_adj = None
                styles = [""] * len(row)
                if threshold is not None and score_adj is not None and score_adj < threshold:
                    for col in row.index:
                        if col in {"symbol", "score_total", "score_adjusted", "threshold_used"}:
                            styles[row.index.get_loc(col)] = "background-color: #fff4e5"
                return pd.Series(styles, index=row.index)

            styled = enter_df.style.apply(_row_style, axis=1)
            st.dataframe(styled, use_container_width=True, hide_index=True)
    st.write(f"watchlist: {_safe_rel_display(summary.get('watchlist_path',''))}")
    watchlist_latest = daily_dir / "watchlist_enter_latest.txt"
    if watchlist_latest.exists():
        with st.expander("watchlist_enter_latest.txt"):
            st.code(_sanitize_text(watchlist_latest.read_text(encoding="utf-8")))

    with st.expander("WATCH / AVOID (counts)"):
        st.write(f"WATCH={watch_count} AVOID={avoid_count}")
        st.write(f"decision_latest.json: {_safe_rel_display(summary.get('decision_path','N/A'))}")
        st.write("詳細は out/step6_decision/decision_latest.json を参照")

    with st.expander("Debug / Inputs"):
        st.write(f"decision_path: {_safe_rel_display(summary.get('decision_path',''))}")
        st.write(f"trades_dir: {_safe_rel_display(summary.get('trades_dir',''))}")
        st.write(f"decision_status: {summary.get('decision_status','')}")
        st.write(f"summary_path: {_safe_rel_display(str(summary_path))}")
        st.write(f"deviation_path: {_safe_rel_display(str(deviation_path))}")
        if summary_err:
            st.write(f"summary_error: {summary_err}")
        if deviation_err:
            st.write(f"deviation_error: {deviation_err}")

    st.markdown("## Alerts (tail)")
    alerts_path = daily_dir / "alerts.jsonl"
    alert_rows, invalid_count = _safe_tail_jsonl(alerts_path, args.alerts_tail)
    if invalid_count:
        st.write(f"invalid_lines: {invalid_count}")
    kinds = sorted({str(a.get("kind", "")) for a in alert_rows if a.get("kind")})
    kind_filter = st.selectbox("kind filter", ["ALL"] + kinds, index=0)
    table_rows = []
    for rec in alert_rows:
        kind = rec.get("kind", "")
        if kind_filter != "ALL" and kind != kind_filter:
            continue
        payload = rec.get("payload", {}) if isinstance(rec.get("payload", {}), dict) else {}
        event_id = rec.get("event_id") or payload.get("event_id") or rec.get("dedup_key") or ""
        send_result = payload.get("send_result") or payload.get("result") or ""
        table_rows.append(
            {
                "ts_utc": _sanitize_text(rec.get("ts_utc", "")),
                "kind": _sanitize_text(kind),
                "event_id": _sanitize_text(event_id),
                "send_result": _sanitize_text(send_result),
                "http_status": _sanitize_text(payload.get("http_status", "")),
                "reason": _sanitize_text(payload.get("reason", "")),
                "decision_id": _sanitize_text(payload.get("decision_id", "")),
            }
        )
    if table_rows:
        st.dataframe(pd.DataFrame(table_rows), use_container_width=True, hide_index=True)
    else:
        st.write("alerts: none")
    with st.expander("alerts payload (tail)"):
        masked = [_sanitize_text(json.dumps(r, ensure_ascii=False)) for r in alert_rows]
        st.code("\n".join(masked))


if __name__ == "__main__":
    main()
