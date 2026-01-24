from __future__ import annotations

import argparse
import json
import sys
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


def main() -> None:
    st.set_page_config(layout="wide", page_title="Step10 Daily", initial_sidebar_state="collapsed")

    ap = argparse.ArgumentParser(description="Step10 Daily Dashboard")
    ap.add_argument("--out", default="out")
    ap.add_argument("--date", default="latest")
    ap.add_argument("--latest", action="store_true")
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

    no_trade = summary.get("no_trade", {}) if isinstance(summary.get("no_trade", {}), dict) else {}
    is_no_trade = bool(no_trade.get("is_no_trade", True)) if summary else True
    counts = summary.get("counts", {}) if isinstance(summary.get("counts", {}), dict) else {}
    enter_count = int(counts.get("enter", 0) or 0)
    watch_count = int(counts.get("watch", 0) or 0)
    avoid_count = int(counts.get("avoid", 0) or 0)

    if agent_verdict == "NO_TRADE":
        st.markdown("## 今日は何もしない日です")
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
        st.info("逸脱: UNKNOWN")

    if warning_reason_text:
        st.write(f"warning_reason: {warning_reason_text}")

    deviations = deviation.get("deviations_today", []) if isinstance(deviation.get("deviations_today", []), list) else []
    if deviations:
        dev_df = pd.DataFrame(deviations)
        cols = [c for c in ["type", "symbol", "action_ts_jst", "decision_id"] if c in dev_df.columns]
        st.dataframe(dev_df[cols], use_container_width=True, hide_index=True)
    else:
        st.write("逸脱: なし")

    st.markdown("## ENTER Candidates")
    enter_rows = summary.get("enter_candidates", []) if isinstance(summary.get("enter_candidates", []), list) else []
    enter_df = _enter_table(enter_rows)
    if enter_df.empty:
        st.write("ENTER候補: なし")
    else:
        def _row_style(row: pd.Series) -> list[str]:
            threshold = _threshold_value(str(row.get("threshold_used", "")))
            try:
                score_adj = float(row.get("score_adjusted"))
            except Exception:
                score_adj = None
            if threshold is not None and score_adj is not None and score_adj < threshold:
                return ["background-color: #fff4e5"] * len(row)
            return [""] * len(row)

        styled = enter_df.style.apply(_row_style, axis=1)
        st.dataframe(styled, use_container_width=True, hide_index=True)

    with st.expander("WATCH / AVOID (counts)"):
        st.write(f"WATCH={watch_count} AVOID={avoid_count}")
        st.write(f"decision_latest.json: {_short_path(summary.get('decision_path','N/A'))}")
        st.write("詳細は out/step6_decision/decision_latest.json を参照")

    with st.expander("Debug / Inputs"):
        st.write(f"decision_path: {_short_path(summary.get('decision_path',''))}")
        st.write(f"trades_dir: {_short_path(summary.get('trades_dir',''))}")
        st.write(f"decision_status: {summary.get('decision_status','')}")
        st.write(f"summary_path: {_short_path(str(summary_path))}")
        st.write(f"deviation_path: {_short_path(str(deviation_path))}")
        if summary_err:
            st.write(f"summary_error: {summary_err}")
        if deviation_err:
            st.write(f"deviation_error: {deviation_err}")


if __name__ == "__main__":
    main()
