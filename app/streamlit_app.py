import json
import sys
from pathlib import Path
from typing import List

import pandas as pd
import streamlit as st

# Ensure project root on sys.path for pipeline imports
ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))

from pipeline.trade_append import append_trade
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


def _decision_view(base_out: Path, trade_out: Path) -> None:
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

    for _, row in sym_rows.iterrows():
        # cols: symbol/theme | score | env | notes | one column per action
        actions = ["ENTER", "WATCH", "SKIP", "EXIT"]
        cols = st.columns([1.4, 0.9, 0.9, 1.4] + [1.0] * len(actions))
        cols[0].write(f"{row['symbol']} ({row['theme']})")
        cols[1].write(f"score={row.get('score_total','')}")
        cols[2].write(f"env={row.get('env_bias','')}")
        notes_key = f"notes_{row['symbol']}_{row['theme']}"
        note_val = cols[3].text_input("notes", key=notes_key, label_visibility="collapsed")
        for idx, act in enumerate(actions):
            col_idx = 4 + idx
            if cols[col_idx].button(act, key=f"{act}_{row['symbol']}_{row['theme']}"):
                result = append_trade(
                    out_dir=str(trade_out.parent),
                    symbol=row["symbol"],
                    action=act,
                    notes=note_val,
                    source="streamlit_decision_view",
                    decision_path=str(decision_path),
                    row_meta={
                        "theme": row.get("theme", ""),
                        "score_total": row.get("score_total", ""),
                        "env": row.get("env_bias", ""),
                        "trend": row.get("trend", row.get("trend_direction", "")),
                        "flags": normalize_flags(row.get("flags", "")),
                        "snapshot_id": decision.get("asof_date_utc", ""),
                    },
                )
                if result.get("ok"):
                    st.success(f"Logged {act} for {row['symbol']}")
                    st.session_state["last_action"] = f"{act} {row['symbol']}"
                else:
                    st.warning(result.get("error", "append failed"))

    if last_action:
        st.info(f"Last action: {last_action}")


def main() -> None:
    st.set_page_config(page_title="Step5/Decision", layout="wide")
    st.title("Step5 / Decision View")

    base_out_str = st.sidebar.text_input("Out directory", value="out/step5_dashboard")
    base_out = Path(base_out_str)
    trade_out = Path("out/trade_log/trades.csv")

    view = st.sidebar.radio("View", ["Dashboard", "Decision View"], index=0)
    if view == "Dashboard":
        _dashboard_view(base_out)
    else:
        _decision_view(base_out, trade_out)


if __name__ == "__main__":
    main()
