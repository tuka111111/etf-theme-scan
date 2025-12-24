

"""
SSGA (State Street) ETF holdings utilities
- Download holdings XLSX
- Parse ticker symbols safely
"""

from __future__ import annotations

import re
from pathlib import Path
from typing import List

import pandas as pd
import requests


# --- helpers -------------------------------------------------

def _normalize_ticker(x: str) -> str | None:
    if not isinstance(x, str):
        return None
    x = x.strip().upper()
    if not x:
        return None
    # reject obvious non-tickers
    if re.fullmatch(r"[0-9A-Z]{8,}", x):  # CUSIP/ISIN-like
        return None
    if x in {"TICKER", "IDENTIFIER", "HOLDINGS", "HOLDINGS:"}:
        return None
    # Yahoo-friendly
    x = x.replace(" ", "-")
    return x


# --- public API ----------------------------------------------

def download_ssga_holdings_xlsx(
    theme: str,
    out_dir: str | Path,
    url_template: str | None = None,
) -> Path:
    """
    Download SSGA holdings XLSX for a given ETF ticker.
    Default template works for SPDR ETFs.
    """
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    if url_template is None:
        url_template = (
            "https://www.ssga.com/library-content/products/fund-data/"
            "etfs/us/holdings-daily-us-en-{ticker}.xlsx"
        )

    url = url_template.format(ticker=theme.lower())
    dst = out_dir / f"holdings_{theme}_ssga.xlsx"

    r = requests.get(url, timeout=30)
    r.raise_for_status()
    dst.write_bytes(r.content)
    return dst


def read_ssga_holdings_xlsx(xlsx_path: str | Path) -> List[str]:
    """
    Read SSGA holdings XLSX and extract ticker symbols.
    Robust against header noise and format changes.
    """
    xlsx_path = Path(xlsx_path)
    if not xlsx_path.exists():
        raise FileNotFoundError(xlsx_path)

    # read all sheets, concat
    xls = pd.ExcelFile(xlsx_path)
    frames = []
    for sheet in xls.sheet_names:
        try:
            df = xls.parse(sheet)
            frames.append(df)
        except Exception:
            continue

    if not frames:
        raise RuntimeError("No readable sheets in SSGA XLSX")

    df = pd.concat(frames, axis=0, ignore_index=True)

    # candidate columns
    cand_cols = []
    for c in df.columns:
        c_low = str(c).lower()
        if "ticker" in c_low or "symbol" in c_low:
            cand_cols.append(c)

    if not cand_cols:
        # fallback: try Identifier column
        for c in df.columns:
            if str(c).lower().startswith("identifier"):
                cand_cols.append(c)
                break

    if not cand_cols:
        raise RuntimeError(
            f"Could not find ticker/symbol column in {xlsx_path.name}: {list(df.columns)}"
        )

    col = cand_cols[0]

    symbols: List[str] = []
    for v in df[col].dropna().unique().tolist():
        sym = _normalize_ticker(v)
        if sym:
            symbols.append(sym)

    if not symbols:
        raise RuntimeError("No valid tickers parsed from SSGA holdings")

    return sorted(set(symbols))