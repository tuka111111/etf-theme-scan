from __future__ import annotations

import argparse
import json
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
DEFAULT_THRESHOLDS = "out/step9/thresholds_final.json"


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


def _aggregate_score_by_horizon(df: pd.DataFrame) -> pd.DataFrame:
    sub = df[df["group_type"] == "score_bucket"].copy()
    if sub.empty:
        return pd.DataFrame()
    grouped = sub.groupby(["horizon", "group_key"])
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


def _load_thresholds(path: Path) -> dict:
    if not path.exists():
        return {}
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return {}


def main() -> None:
    args = _parse_args()

    st.set_page_config(page_title="Step9 Score Validation", layout="wide")
    st.title("Step9 / Score Validation Dashboard")
    st.caption("All-time aggregation across asof_date. BUY-only decisions.")

    input_path = st.text_input("Input CSV", value=args.input)
    thresholds_path = st.text_input("Thresholds JSON (optional)", value=DEFAULT_THRESHOLDS)
    path = Path(input_path)
    if not path.exists():
        st.error(f"Input CSV not found: {path}")
        return

    df = pd.read_csv(path)
    _require_cols(df, ["horizon", "group_type", "group_key", "avg_return", "win_rate", "sample"])

    horizon = st.selectbox("Horizon", sorted(df["horizon"].unique().tolist()), index=0)
    min_sample = st.slider("Min sample", 0, 200, args.min_sample, 5)

    st.subheader("Score Bucket by Horizon")
    score_all = _aggregate_score_by_horizon(df)
    if score_all.empty:
        st.info("No score_bucket rows available.")
    else:
        try:
            import altair as alt

            order = ["score 60-69", "score 70-79", "score>=80"]
            rank = {"score 60-69": 0, "score 70-79": 1, "score>=80": 2}

            df_plot = score_all.copy()
            df_plot["group_key"] = df_plot["group_key"].astype(str)
            # keep horizon numeric for correct ordering
            df_plot["horizon"] = pd.to_numeric(df_plot["horizon"], errors="coerce").fillna(0).astype(int)
            df_plot = df_plot[df_plot["sample"] >= min_sample]

            horizon_order = sorted(df_plot["horizon"].unique().tolist())
            ncols = min(3, max(1, len(horizon_order)))

            thresholds = _load_thresholds(Path(thresholds_path)).get("horizons", {})
            df_plot["threshold_key"] = ""
            df_plot["threshold_rank"] = -1

            for h in horizon_order:
                thr = thresholds.get(str(h), {}).get("threshold")
                if thr is None:
                    continue
                thr_label = "score>=80" if thr >= 80 else "score 70-79" if thr >= 70 else "score 60-69"
                df_plot.loc[df_plot["horizon"] == h, "threshold_key"] = thr_label
                df_plot.loc[df_plot["horizon"] == h, "threshold_rank"] = rank.get(thr_label, -1)

            # adoption-region highlight: fade buckets below threshold
            df_plot["bucket_rank"] = df_plot["group_key"].map(rank).fillna(-1).astype(int)
            df_plot["adopted"] = (df_plot["threshold_rank"] >= 0) & (df_plot["bucket_rank"] >= df_plot["threshold_rank"])

            base = alt.Chart(df_plot).properties(width=220, height=220)
            bars = base.mark_bar().encode(
                x=alt.X("group_key:N", sort=order, title="score_bucket"),
                y=alt.Y("avg_return:Q", title="avg_return"),
                opacity=alt.condition("datum.adopted", alt.value(1.0), alt.value(0.25)),
                tooltip=["horizon:O", "group_key:N", "avg_return:Q", "win_rate:Q", "sample:Q"],
            )
            rule = (
                base.transform_filter(alt.datum.group_key == alt.datum.threshold_key)
                .mark_rule(color="red")
                .encode(x=alt.X("group_key:N", sort=order))
            )
            chart = (
                (bars + rule)
                .facet(column=alt.Column("horizon:O", sort=horizon_order, title="horizon"), columns=ncols)
                .resolve_scale(y="independent")
                .configure_facet(spacing=8)
            )
            st.altair_chart(chart, use_container_width=True)

            bars_win = base.mark_bar().encode(
                x=alt.X("group_key:N", sort=order, title="score_bucket"),
                y=alt.Y("win_rate:Q", title="win_rate"),
                opacity=alt.condition("datum.adopted", alt.value(1.0), alt.value(0.25)),
                tooltip=["horizon:O", "group_key:N", "avg_return:Q", "win_rate:Q", "sample:Q"],
            )
            rule_win = (
                base.transform_filter(alt.datum.group_key == alt.datum.threshold_key)
                .mark_rule(color="red")
                .encode(x=alt.X("group_key:N", sort=order))
            )
            chart_win = (
                (bars_win + rule_win)
                .facet(column=alt.Column("horizon:O", sort=horizon_order, title="horizon"), columns=ncols)
                .resolve_scale(y="independent")
                .configure_facet(spacing=8)
            )
            st.altair_chart(chart_win, use_container_width=True)
        except Exception as e:
            st.warning(f"Altair chart unavailable: {e}")

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
