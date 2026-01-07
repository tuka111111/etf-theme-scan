from __future__ import annotations

import pandas as pd
from pathlib import Path
from typing import Tuple, List, Dict

REQUIRED_COLS = {"theme", "symbol", "flags"}


def normalize_flags(val) -> List[str]:
    if val is None:
        return []
    try:
        if pd.isna(val):
            return []
    except Exception:
        pass
    s = str(val).strip()
    if not s or s.lower() == "nan":
        return []
    # split by common separators
    parts = []
    for sep in ["|", ";", ","]:
        if sep in s:
            parts = [p.strip() for p in s.split(sep)]
            break
    if not parts:
        parts = [s]
    return [p for p in parts if p]


def load_dashboard(path: Path, score_col: str) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, str, str]:
    df = pd.read_csv(path, dtype=str)
    missing = REQUIRED_COLS - set(df.columns)
    if missing:
        raise ValueError(f"dashboard missing required columns: {sorted(missing)}")
    if score_col not in df.columns:
        raise ValueError(f"dashboard missing score column: {score_col}")

    # Normalize numeric columns
    df[score_col] = pd.to_numeric(df[score_col], errors="coerce")
    df["flags_list"] = df["flags"].apply(normalize_flags)
    asof_local = df["asof_local"].dropna().iloc[0] if "asof_local" in df.columns and not df["asof_local"].dropna().empty else "unknown"
    asof_utc = df["asof_utc"].dropna().iloc[0] if "asof_utc" in df.columns and not df["asof_utc"].dropna().empty else "unknown"

    etf_rows = df[df["flags_list"].apply(lambda f: "etf_env_row" in f)].copy()
    sym_rows = df[df["flags_list"].apply(lambda f: "etf_env_row" not in f)].copy()
    if etf_rows.empty:
        raise ValueError("No ETF env rows found (flags containing etf_env_row).")

    return df, etf_rows, sym_rows, asof_local, asof_utc
