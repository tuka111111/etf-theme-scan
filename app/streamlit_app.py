import json
from pathlib import Path
from typing import List

import pandas as pd
import streamlit as st


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


def main() -> None:
    st.set_page_config(page_title="Step5 Dashboard", layout="wide")
    st.title("Step5 Dashboard")

    base_out = Path(st.sidebar.text_input("Out directory", value="out/step5_dashboard"))
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

    # shared pastel palette for ETF + symbols
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


if __name__ == "__main__":
    main()
