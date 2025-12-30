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
    min_score = st.sidebar.slider("Min score_total", 0.0, 100.0, 0.0, 1.0)

    filt = df[
        df["theme"].isin(sel_themes)
        & df["env_bias"].isin(sel_env)
        & (df["score_total"] >= min_score)
    ].copy()

    st.metric("Symbols shown", len(filt))

    st.subheader("Table")
    order = filt.sort_values(["theme", "rank"]).reset_index(drop=True)
    # color rows by theme using a simple pastel palette
    palette = [
        "#eef2ff",
        "#e7f5ff",
        "#f1f8e9",
        "#fff4e6",
        "#fce4ec",
        "#e8f5e9",
    ]
    theme_list = order["theme"].unique().tolist()
    color_map = {t: palette[i % len(palette)] for i, t in enumerate(theme_list)}

    def _row_style(row: pd.Series) -> List[str]:
        c = color_map.get(row.get("theme"), "#ffffff")
        return [f"background-color: {c}"] * len(row)

    styled = order.style.apply(_row_style, axis=1)
    st.dataframe(styled, use_container_width=True, hide_index=True)

    st.subheader("Top Symbols")
    top_n = st.slider("Top N", 5, 50, 10, 5)
    top = (
        filt.sort_values(["score_total", "symbol"], ascending=[False, True])
        .groupby("theme")
        .head(top_n)
    )
    st.bar_chart(top.set_index("symbol")["score_total"])

    st.subheader("Watchlist (CSV)")
    watchlist_csv = top[["theme", "symbol", "score_total"]].to_csv(index=False)
    st.download_button("Download watchlist", data=watchlist_csv, file_name="watchlist.csv", mime="text/csv")


if __name__ == "__main__":
    main()
