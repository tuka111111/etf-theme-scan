import csv
import json
import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import List

import pandas as pd
import streamlit as st

# Ensure project root on sys.path for pipeline imports
ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))

from pipeline.trade_append import (
    OLD_TRADE_ACTION_COLUMNS,
    TRADE_ACTION_COLUMNS,
    append_trade_action,
)
from pipeline.io_step6 import normalize_flags


@st.cache_data
def load_dashboard(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    # normalize types
    df["score_total"] = pd.to_numeric(df.get("score_total"), errors="coerce")
    df["signal_score"] = pd.to_numeric(df.get("signal_score"), errors="coerce")
    df["env_score"] = pd.to_numeric(df.get("env_score"), errors="coerce")
    df["env_confidence"] = pd.to_numeric(df.get("env_confidence"), errors="coerce")
    df["rank"] = pd.to_numeric(df.get("rank"), errors="coerce")
    df["theme"] = df["theme"].astype(str)
    df["symbol"] = df["symbol"].astype(str)
    return df


def _dashboard_view(base_out: Path) -> None:
    csv_path = base_out / "dashboard.csv"
    if not csv_path.exists():
        st.error(f"dashboard.csv not found at {csv_path}")
        return

    df = load_dashboard(csv_path)
    themes = sorted(df["theme"].unique().tolist())

    st.sidebar.subheader("Filters")
    sel_themes = st.sidebar.multiselect("Themes", themes, default=themes)
    env_opts: List[str] = sorted(df["env_bias"].dropna().unique().tolist())
    sel_env = st.sidebar.multiselect("Env Bias", env_opts, default=env_opts)
    min_signal = st.sidebar.slider("Min signal_score", 0.0, 100.0, 0.0, 1.0)
    min_env_score = st.sidebar.slider("Min env_score", 0.0, 100.0, 0.0, 1.0)

    etf_rows = df[df.get("role") == "etf"].copy()
    sym_rows = df[df.get("role") != "etf"].copy()

    filt = sym_rows[
        sym_rows["theme"].isin(sel_themes)
        & sym_rows["env_bias"].isin(sel_env)
        & (sym_rows["signal_score"] >= min_signal)
        & (sym_rows["env_score"] >= min_env_score)
    ].copy()

    st.metric("Symbols shown", len(filt))

    palette = [
        "#f0f4ff",
        "#fef7e0",
        "#e8f5e9",
        "#fff0f6",
        "#e3f2fd",
    ]
    theme_list = pd.concat([etf_rows["theme"], sym_rows["theme"]]).dropna().unique().tolist()
    color_map = {t: palette[i % len(palette)] for i, t in enumerate(theme_list)}

    if not etf_rows.empty:
        st.subheader("ETF Daily Env (per theme)")
        etf_view = etf_rows[
            ["theme", "env_bias", "env_confidence", "etf_trend_state", "env_score"]
        ].reset_index(drop=True)

        def _etf_style(row: pd.Series) -> List[str]:
            c = color_map.get(row.get("theme"), "#ffffff")
            return [f"background-color: {c}"] * len(row)

        st.dataframe(etf_view.style.apply(_etf_style, axis=1), use_container_width=True, hide_index=True)

    st.subheader("Symbols")
    order = filt.sort_values(["signal_score", "symbol"], ascending=[False, True]).reset_index(drop=True)

    def _row_style(row: pd.Series) -> List[str]:
        c = color_map.get(row.get("theme"), "#ffffff")
        return [f"background-color: {c}"] * len(row)

    styled = order.style.apply(_row_style, axis=1)
    st.dataframe(styled, use_container_width=True, hide_index=True)

    st.subheader("Top Symbols")
    top_n = st.slider("Top N", 5, 50, 10, 5)
    top = (
        filt.sort_values(["signal_score", "symbol"], ascending=[False, True])
        .groupby("theme")
        .head(top_n)
    )
    st.bar_chart(top.set_index("symbol")["signal_score"])

    st.subheader("Watchlist (CSV)")
    watchlist_csv = top[["theme", "symbol", "signal_score"]].to_csv(index=False)
    st.download_button("Download watchlist", data=watchlist_csv, file_name="watchlist.csv", mime="text/csv")


def _decision_view(base_out: Path, out_root: Path) -> None:
    decision_path = Path("out/step6_decision/decision_latest.json")
    if (base_out / "decision_latest.json").exists():
        decision_path = base_out / "decision_latest.json"
    elif decision_path.exists():
        pass
    else:
        st.error("decision_latest.json not found (out/step6_decision/decision_latest.json).")
        return

    try:
        decision = json.loads(decision_path.read_text(encoding="utf-8"))
    except Exception as e:
        st.error(f"Failed to read decision: {e}")
        return

    def _decision_pick_index(payload: dict) -> dict:
        idx = {}
        picks = payload.get("picks", {})
        if isinstance(picks, dict):
            for bucket, rows in picks.items():
                if not isinstance(rows, list):
                    continue
                for row in rows:
                    symbol = str(row.get("symbol", "")).upper()
                    if not symbol:
                        continue
                    idx[symbol] = row
        return idx

    decision_idx = _decision_pick_index(decision)

    st.subheader("Decision Snapshot")
    st.write(
        f"asof_date_utc: {decision.get('asof_date_utc','unknown')} | risk_mode: "
        f"{decision.get('risk_mode',{}).get('mode','unknown')} "
        f"(strength={decision.get('risk_mode',{}).get('strength','')})"
    )
    st.write(f"tradable_themes: {', '.join(decision.get('tradable_themes',[])) or 'none'}")

    # load dashboard for detailed rows
    dash_candidates = [
        base_out / "dashboard.csv",
        Path("out/step5_dashboard/dashboard.csv"),
        Path("out/dashboard.csv"),
    ]
    dash_path = next((p for p in dash_candidates if p.exists()), None)
    if not dash_path:
        st.error("dashboard.csv not found in expected locations.")
        return
    df = load_dashboard(dash_path)
    sym_rows = df[df.get("role") != "etf"].copy()

    # filters
    themes = sorted(sym_rows["theme"].dropna().unique().tolist())
    sel_theme = st.selectbox("Theme filter", ["ALL"] + themes, index=0)
    view_filter = st.selectbox("View", ["ALL", "ENTER only", "WATCH only"], index=0)
    search = st.text_input("Search symbol", value="")

    if sel_theme != "ALL":
        sym_rows = sym_rows[sym_rows["theme"] == sel_theme]
    if search:
        sym_rows = sym_rows[sym_rows["symbol"].str.contains(search, case=False)]
    if view_filter != "ALL":
        bucket = "ENTER" if view_filter.startswith("ENTER") else "WATCH"
        sym_rows = sym_rows[sym_rows["flags"].fillna("").str.contains("", regex=False)]

    st.markdown("### Actions")
    last_action = st.session_state.get("last_action", "")
    asof_date_jst = datetime.now(timezone.utc).astimezone(timezone(timedelta(hours=9))).date().isoformat()

    if st.session_state.get("last_success"):
        st.success(st.session_state["last_success"])
        st.session_state["last_success"] = ""
    if st.session_state.get("last_warning"):
        st.warning(st.session_state["last_warning"])
        st.session_state["last_warning"] = ""
    if st.session_state.get("last_error"):
        st.error(st.session_state["last_error"])
        st.session_state["last_error"] = ""

    def _handle_action(action: str, payload: dict, notes_key: str) -> None:
        notes = st.session_state.get(notes_key, "")
        if len(notes) > 255:
            st.session_state["last_warning"] = "Notes must be 255 characters or fewer."
            return
        try:
            payload = dict(payload)
            payload["action"] = action
            payload["notes"] = notes
            append_trade_action(out_dir=str(out_root), record=payload)
            st.session_state[notes_key] = ""
            st.session_state["last_success"] = f"Logged {action} for {payload.get('symbol')}"
            st.session_state["last_action"] = f"{action} {payload.get('symbol')}"
        except Exception as e:
            st.session_state["last_error"] = str(e)

    for _, row in sym_rows.iterrows():
        # cols: symbol/theme | score | env | notes | one column per action
        actions = ["ENTER", "WATCH", "SKIP", "EXIT"]
        cols = st.columns([1.4, 0.9, 0.9, 1.4] + [1.0] * len(actions))
        cols[0].write(f"{row['symbol']} ({row['theme']})")
        cols[1].write(f"score={row.get('score_total','')}")
        cols[2].write(f"env={row.get('env_bias','')}")
        notes_key = f"notes_{asof_date_jst}_{row['theme']}_{row['symbol']}"
        if notes_key not in st.session_state:
            st.session_state[notes_key] = ""
        cols[3].text_area("notes", key=notes_key, label_visibility="collapsed", height=64)
        flags_list = normalize_flags(row.get("flags", ""))
        disable_enter = any(flag in flags_list for flag in ["data_quality_low", "env_bias_mismatch"])
        for idx, act in enumerate(actions):
            col_idx = 4 + idx
            payload = {
                "theme": row.get("theme", ""),
                "symbol": row.get("symbol", ""),
                "score_total": row.get("score_total", ""),
                "env_bias": row.get("env_bias", ""),
                "env_confidence": row.get("env_confidence", ""),
                "etf_env_bias": row.get("etf_env_bias", ""),
                "etf_env_confidence": row.get("etf_env_confidence", ""),
                "flags": row.get("flags", ""),
                "snapshot_id": decision.get("asof_date_utc", ""),
                "decision_id": decision.get("asof_local", ""),
                "source": "streamlit_decision_view",
            }
            pick_meta = decision_idx.get(str(row.get("symbol", "")).upper(), {})
            rules_applied = pick_meta.get("rules_applied", "")
            if isinstance(rules_applied, list):
                rules_applied = ";".join(rules_applied)
            payload["threshold_used"] = pick_meta.get("threshold_used", "")
            payload["rules_applied"] = rules_applied or ""
            payload["score_adjusted"] = pick_meta.get("score_adjusted", "")
            disabled = disable_enter if act == "ENTER" else False
            cols[col_idx].button(
                act,
                key=f"{act}_{row['symbol']}_{row['theme']}",
                on_click=_handle_action,
                args=(act, payload, notes_key),
                disabled=disabled,
            )
        if disable_enter:
            cols[3].caption("ENTER disabled: data_quality_low/env_bias_mismatch")

    if last_action:
        st.info(f"Last action: {last_action}")

    st.markdown("### Today Actions (latest 10)")
    today_path = out_root / "step7_trades" / f"trade_actions_{asof_date_jst}.csv"
    if today_path.exists():
        try:
            recent = pd.read_csv(today_path)
            if "action_ts_jst" in recent.columns:
                recent = recent.sort_values("action_ts_jst", ascending=False)
            st.dataframe(
                recent.head(10)[
                    ["action_ts_jst", "theme", "symbol", "action", "notes", "score_total", "env_bias"]
                ],
                use_container_width=True,
                hide_index=True,
            )
        except Exception as e:
            st.error(f"Failed to load today actions: {e}")
    else:
        st.write("No actions logged today.")


def _decision_log_view(out_root: Path) -> None:
    st.subheader("Decision Log")
    trades_dir = out_root / "step7_trades"
    files = sorted(trades_dir.glob("trade_actions_*.csv"))
    if not files:
        st.info("No trade actions found.")
        return

    def _read_trade_actions_csv(path: Path) -> pd.DataFrame:
        with path.open(newline="", encoding="utf-8") as f:
            reader = csv.reader(f)
            header = next(reader, None)
        if not header:
            return pd.DataFrame()
        if header not in (TRADE_ACTION_COLUMNS, OLD_TRADE_ACTION_COLUMNS):
            raise ValueError(f"Unexpected CSV header: {header}")

        with path.open(newline="", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            rows = list(reader)
        if header == TRADE_ACTION_COLUMNS:
            return pd.DataFrame(rows)

        normalized = []
        for row in rows:
            action_ts = row.get("action_ts_jst", "")
            normalized.append(
                {
                    "schema_version": row.get("schema_version", "step7_trade_action_v1"),
                    "asof_date_jst": row.get("asof_date_jst", ""),
                    "action_ts_jst": action_ts,
                    "created_at_jst": action_ts,
                    "decision_id": "",
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
                    "status": "active",
                    "source": row.get("source", ""),
                    "run_id": row.get("run_id", ""),
                    "snapshot_id": row.get("snapshot_id", ""),
                    "updated_from_ts_jst": "",
                    "debug_json": row.get("debug_json", ""),
                }
            )
        return pd.DataFrame(normalized)

    frames = []
    for path in files:
        try:
            frames.append(_read_trade_actions_csv(path))
        except Exception as e:
            st.warning(f"Failed to read {path.name}: {e}")
    if not frames:
        return

    df = pd.concat(frames, ignore_index=True)
    if "status" not in df.columns:
        df["status"] = "active"
    if "updated_from_ts_jst" not in df.columns:
        df["updated_from_ts_jst"] = ""
    if "decision_id" not in df.columns:
        df["decision_id"] = ""

    df["action_ts_jst"] = df.get("action_ts_jst", "")
    df["asof_date_jst"] = df.get("asof_date_jst", "")

    suppressed = set(df.loc[df["status"].isin(["obsolete", "edited"]), "updated_from_ts_jst"].dropna().astype(str))
    show_obsolete = st.checkbox("Show obsolete/original rows", value=False)
    if suppressed and not show_obsolete:
        mask = df["action_ts_jst"].astype(str).isin(suppressed) & (df["status"] == "active")
        df = df[~mask]

    df = df.sort_values("action_ts_jst", ascending=False)

    themes = sorted(df["theme"].dropna().unique().tolist())
    symbols = sorted(df["symbol"].dropna().unique().tolist())
    dates = sorted(df["asof_date_jst"].dropna().unique().tolist(), reverse=True)
    statuses = sorted(df["status"].dropna().unique().tolist())

    col_f1, col_f2, col_f3, col_f4 = st.columns(4)
    sel_theme = col_f1.selectbox("Theme", ["ALL"] + themes, index=0)
    sel_symbol = col_f2.selectbox("Symbol", ["ALL"] + symbols, index=0)
    sel_date = col_f3.selectbox("Date", ["ALL"] + dates, index=0)
    sel_status = col_f4.selectbox("Status", ["ALL"] + statuses, index=0)

    if sel_theme != "ALL":
        df = df[df["theme"] == sel_theme]
    if sel_symbol != "ALL":
        df = df[df["symbol"] == sel_symbol]
    if sel_date != "ALL":
        df = df[df["asof_date_jst"] == sel_date]
    if sel_status != "ALL":
        df = df[df["status"] == sel_status]

    limit = st.slider("Rows to show", 5, 200, 50, 5)
    view = df.head(limit).reset_index(drop=True)

    if st.session_state.get("log_success"):
        st.success(st.session_state["log_success"])
        st.session_state["log_success"] = ""
    if st.session_state.get("log_warning"):
        st.warning(st.session_state["log_warning"])
        st.session_state["log_warning"] = ""
    if st.session_state.get("log_error"):
        st.error(st.session_state["log_error"])
        st.session_state["log_error"] = ""

    def _append_log_action(action_row: dict, new_status: str, new_notes: str) -> None:
        if len(new_notes) > 255:
            st.session_state["log_warning"] = "Notes must be 255 characters or fewer."
            return
        try:
            append_trade_action(
                out_dir=str(out_root),
                record={
                    "theme": action_row.get("theme", ""),
                    "symbol": action_row.get("symbol", ""),
                    "action": action_row.get("action", ""),
                    "notes": new_notes,
                    "action_ts_jst": action_row.get("action_ts_jst", ""),
                    "created_at_jst": action_row.get("created_at_jst", action_row.get("action_ts_jst", "")),
                    "score_total": action_row.get("score_total", ""),
                    "score_adjusted": action_row.get("score_adjusted", ""),
                    "env_bias": action_row.get("env_bias", ""),
                    "env_confidence": action_row.get("env_confidence", ""),
                    "etf_env_bias": action_row.get("etf_env_bias", ""),
                    "etf_env_confidence": action_row.get("etf_env_confidence", ""),
                    "flags": action_row.get("flags", ""),
                    "snapshot_id": action_row.get("snapshot_id", ""),
                    "decision_id": action_row.get("decision_id", ""),
                    "status": new_status,
                    "updated_from_ts_jst": action_row.get("action_ts_jst", ""),
                    "threshold_used": action_row.get("threshold_used", ""),
                    "rules_applied": action_row.get("rules_applied", ""),
                    "source": "streamlit_decision_log",
                },
            )
            st.session_state["log_success"] = f"{new_status.capitalize()} saved for {action_row.get('symbol')}"
        except Exception as e:
            st.session_state["log_error"] = str(e)

    for _, row in view.iterrows():
        row_dict = row.to_dict()
        cols = st.columns([1.2, 1.0, 0.8, 0.8, 1.6, 1.2, 1.0, 1.0])
        cols[0].write(row.get("action_ts_jst", ""))
        cols[1].write(row.get("theme", ""))
        cols[2].write(row.get("symbol", ""))
        cols[3].write(row.get("action", ""))
        cols[4].write(row.get("status", "active"))
        edit_key = f"log_notes_{row.get('action_ts_jst','')}_{row.get('symbol','')}"
        if edit_key not in st.session_state:
            st.session_state[edit_key] = str(row.get("notes", ""))
        cols[5].text_area("notes", key=edit_key, label_visibility="collapsed", height=64)
        cols[6].button(
            "Update",
            key=f"update_{row.get('action_ts_jst','')}_{row.get('symbol','')}",
            on_click=_append_log_action,
            args=(row_dict, "edited", st.session_state.get(edit_key, "")),
        )
        cols[7].button(
            "Obsolete",
            key=f"obsolete_{row.get('action_ts_jst','')}_{row.get('symbol','')}",
            on_click=_append_log_action,
            args=(row_dict, "obsolete", st.session_state.get(edit_key, "")),
        )


def main() -> None:
    st.set_page_config(page_title="Step5/Decision", layout="wide")
    st.title("Step5 / Decision View")

    base_out_str = st.sidebar.text_input("Out directory", value="out/step5_dashboard")
    base_out = Path(base_out_str)
    out_root = base_out.parent if base_out.name == "step5_dashboard" else Path("out")

    view = st.sidebar.radio("View", ["Dashboard", "Decision View", "Decision Log"], index=0)
    if view == "Dashboard":
        _dashboard_view(base_out)
    elif view == "Decision View":
        _decision_view(base_out, out_root)
    else:
        _decision_log_view(out_root)


if __name__ == "__main__":
    main()
