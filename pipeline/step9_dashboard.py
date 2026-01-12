from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Optional

import pandas as pd
import streamlit as st

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))

from pipeline.common import ensure_dir

DEFAULT_INPUT = "out/step8/score_validation.csv"
DEFAULT_HORIZON = 20


def _parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description="Step9: score validation dashboard")
    ap.add_argument("--input", default=DEFAULT_INPUT, help="Path to score_validation.csv")
    ap.add_argument("--min-sample", type=int, default=20, help="Default sample cutoff")
    if "--" in sys.argv:
        idx = sys.argv.index("--")
        return ap.parse_args(sys.argv[idx + 1 :])
    return ap.parse_args([])


def _require_cols(df: pd.DataFrame, cols: list[str]) -> None:
    missing = [c for c in cols if c not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns: {missing}")


def _weighted_avg(df: pd.DataFrame, value_col: str, weight_col: str) -> pd.Series:
    w = pd.to_numeric(df[weight_col], errors="coerce").fillna(0.0)
    v = pd.to_numeric(df[value_col], errors="coerce").fillna(0.0)
    denom = w.sum()
    return (v * w).sum() / denom if denom else 0.0


def _aggregate_all(df: pd.DataFrame, horizon: int, group_type: str) -> pd.DataFrame:
    sub = df[(df["horizon"] == horizon) & (df["group_type"] == group_type)].copy()
    if sub.empty:
        return pd.DataFrame()
    grouped = sub.groupby("group_key")
    out = grouped.apply(
        lambda g: pd.Series(
            {
                "avg_return": _weighted_avg(g, "avg_return", "sample"),
                "win_rate": _weighted_avg(g, "win_rate", "sample"),
                "sample": int(pd.to_numeric(g["sample"], errors="coerce").fillna(0).sum()),
            }
        )
    ).reset_index()
    return out


def main() -> None:
    args = _parse_args()

    st.set_page_config(page_title="Step9 Score Validation", layout="wide")
    st.title("Step9 / Score Validation Dashboard")
    st.caption("All-time aggregation across asof_date. BUY-only decisions.")

    input_path = st.text_input("Input CSV", value=args.input)
    path = Path(input_path)
    if not path.exists():
        st.error(f"Input CSV not found: {path}")
        return

    df = pd.read_csv(path)
    _require_cols(df, ["horizon", "group_type", "group_key", "avg_return", "win_rate", "sample"])

    horizon = st.selectbox("Horizon", sorted(df["horizon"].unique().tolist()), index=0)
    min_sample = st.slider("Min sample", 0, 200, args.min_sample, 5)

    st.subheader("Score Bucket Summary")
    score_df = _aggregate_all(df, horizon, "score_bucket")
    if score_df.empty:
        st.info("No score_bucket rows for this horizon.")
    else:
        score_df = score_df.sort_values("group_key")
        st.markdown("**avg_return**")
        st.bar_chart(score_df.set_index("group_key")["avg_return"])
        st.markdown("**win_rate**")
        st.bar_chart(score_df.set_index("group_key")["win_rate"])
        st.dataframe(score_df, use_container_width=True, hide_index=True)

    st.subheader("Env Summary")
    env_df = _aggregate_all(df, horizon, "env")
    if env_df.empty:
        st.info("No env rows for this horizon.")
    else:
        env_df = env_df.sort_values("group_key")
        st.markdown("**avg_return**")
        st.bar_chart(env_df.set_index("group_key")["avg_return"])
        st.markdown("**win_rate**")
        st.bar_chart(env_df.set_index("group_key")["win_rate"])
        st.dataframe(env_df, use_container_width=True, hide_index=True)

    st.subheader("Flag Effectiveness")
    flag_df = _aggregate_all(df, horizon, "flag")
    if flag_df.empty:
        st.info("No flag rows for this horizon.")
    else:
        flag_df = flag_df[flag_df["sample"] >= min_sample]
        st.dataframe(
            flag_df.sort_values(["sample", "avg_return"], ascending=[False, False]),
            use_container_width=True,
            hide_index=True,
        )

    out_dir = ensure_dir(Path("out/step8/viz"))
    export_path = out_dir / "export_summary.csv"
    export = pd.concat(
        [
            _aggregate_all(df, horizon, "score_bucket").assign(group_type="score_bucket"),
            _aggregate_all(df, horizon, "env").assign(group_type="env"),
            _aggregate_all(df, horizon, "flag").assign(group_type="flag"),
        ],
        ignore_index=True,
    )
    if not export.empty:
        export.to_csv(export_path, index=False)
        st.caption(f"Exported summary to {export_path}")


if __name__ == "__main__":
    main()
