# pipeline/io_step1.py
from __future__ import annotations

import logging
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Tuple

import pandas as pd

from .common import uniq_keep_order

LOG = logging.getLogger(__name__)
SYMBOL_PATTERN = re.compile(r"^[A-Z0-9][A-Z0-9.\-]{0,14}$")


@dataclass(frozen=True)
class UniverseRow:
    theme: str
    symbol: str


def find_step1_universe_files(out_dir: str | Path, themes: List[str] | None = None) -> List[Path]:
    out_dir = Path(out_dir)

    # Prefer aggregate files
    agg_candidates: List[Path] = [
        out_dir / "step1_universe" / "universe.csv",
        out_dir / "step1_universe" / "step1_universe.csv",
        out_dir / "step1_universe.csv",
        out_dir / "universe_step1.csv",
        out_dir / "universe.csv",
    ]
    for p in agg_candidates:
        if p.exists():
            return [p]

    files: List[Path] = []
    theme_list = [t.strip().upper() for t in themes] if themes else []

    # Theme-specific files if themes provided
    for t in theme_list:
        for pat in [f"step1_universe_{t}.csv", f"watchlist_{t}.csv"]:
            p = out_dir / pat
            if p.exists():
                files.append(p)
                break

    if files:
        return files

    # fallback: any step1_*.csv (may include dated filenames)
    cands = sorted(out_dir.glob("step1_*.csv"))
    if cands:
        return cands

    # last resort: watchlist_ALL
    watchlist = out_dir / "watchlist_ALL.csv"
    if watchlist.exists():
        return [watchlist]

    raise FileNotFoundError(
        f"Step1 universe CSV not found under: {out_dir} (expected step1_universe*.csv or watchlist*.csv)"
    )


def load_universe(out_dir: str | Path, themes: List[str] | None = None) -> Tuple[pd.DataFrame, List[UniverseRow]]:
    paths = find_step1_universe_files(out_dir, themes)
    dfs: List[pd.DataFrame] = []
    rows_all: List[UniverseRow] = []
    want = set([t.strip().upper() for t in themes]) if themes else None

    for path in paths:
        df = pd.read_csv(path)
        dfs.append(df)

        cols = {str(c).lower().strip(): c for c in df.columns}

        # Normal case: theme + symbol
        if "theme" in cols and "symbol" in cols:
            theme_col = cols["theme"]
            sym_col = cols["symbol"]

            df[theme_col] = df[theme_col].astype(str).str.strip().str.upper()
            df[sym_col] = df[sym_col].astype(str).str.strip().str.upper()

            df_use = df
            if want:
                df_use = df_use[df_use[theme_col].isin(want)].copy()

            rows = [UniverseRow(theme=r[theme_col], symbol=r[sym_col]) for _, r in df_use.iterrows()]
            rows = [UniverseRow(theme=x.theme, symbol=x.symbol) for x in rows if x.symbol]
            rows = _filter_invalid_symbols(rows, theme_col=theme_col, path=path)
            rows_all.extend(rows)
            continue

        # Watchlist style: Symbol only
        if "symbol" in cols and "theme" not in cols:
            sym_col = cols["symbol"]
            df[sym_col] = df[sym_col].astype(str).str.strip().str.upper()

            inferred_theme: str | None = None
            stem = path.stem.upper()
            for prefix in ["STEP1_UNIVERSE_", "WATCHLIST_"]:
                if stem.startswith(prefix):
                    inferred_theme = stem[len(prefix) :].strip()
                    break

            if inferred_theme and want and inferred_theme not in want:
                continue  # skip file not in requested themes

            if inferred_theme:
                theme_val = inferred_theme
            elif want and len(want) == 1:
                theme_val = list(want)[0]
            else:
                raise ValueError(
                    f"watchlist-style csv detected (symbol only) but could not infer single theme. "
                    f"themes={themes} path={path} got_columns={list(df.columns)}"
                )

            df["theme"] = theme_val
            df_use = df[["theme", sym_col]].copy().rename(columns={sym_col: "symbol"})
            rows = [UniverseRow(theme=theme_val, symbol=s) for s in df_use["symbol"].tolist() if s]
            rows = _filter_invalid_symbols(rows, theme_col=theme_val, path=path)
            rows_all.extend(rows)
            continue

        raise ValueError(
            f"step1 universe csv must have columns theme,symbol. path={path} got={list(df.columns)}"
        )

    if not rows_all:
        raise ValueError(f"No symbols found from Step1 under {out_dir} for themes={themes}")

    merged_df = pd.concat(dfs, ignore_index=True) if dfs else pd.DataFrame()
    return merged_df, rows_all


def group_symbols_by_theme(rows: List[UniverseRow]) -> Dict[str, List[str]]:
    m: Dict[str, List[str]] = {}
    for r in rows:
        m.setdefault(r.theme, []).append(r.symbol)
    for k in list(m.keys()):
        m[k] = uniq_keep_order(m[k])
    return m


def _filter_invalid_symbols(rows: List[UniverseRow], *, theme_col: str, path: Path) -> List[UniverseRow]:
    valid: List[UniverseRow] = []
    for r in rows:
        if SYMBOL_PATTERN.match(r.symbol):
            valid.append(r)
        else:
            LOG.warning("Skipping invalid symbol from %s theme=%s sym=%r", path, theme_col, r.symbol)
    return valid
