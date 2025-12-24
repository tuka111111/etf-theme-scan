#!/usr/bin/env python3
"""
Theme Scan (Morning Auto Update)
- Reads ETF holdings from a local CSV (stable ops).
- Pulls market data via yfinance.
- Scores "theme strength" and ranks constituent stocks.
- Exports TradingView watchlist CSV + markdown report.

Usage:
  # single theme (backward compatible)
  python theme_scan.py --theme XME --holdings ssga --out ./out --top 12

  # multiple themes
  python theme_scan.py --themes XME,SMH,XBI --holdings ssga --out ./out --top 12
"""

from __future__ import annotations

import argparse
import os
import sys
import pathlib
import re
import urllib.request
import urllib.error
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import List, Dict, Optional, Tuple

import numpy as np
import pandas as pd

try:
    import yfinance as yf
except ImportError:
    print("Missing dependency: yfinance. Install with: pip install yfinance", file=sys.stderr)
    raise


# -----------------------------
# Config / Helpers
# -----------------------------

BENCH = "SPY"  # for relative strength baseline
DEFAULT_LOOKBACK_DAYS = 260  # ~1y trading days
PRICE_COL = "Close"
VOL_COL = "Volume"


def _ssga_holdings_url(etf_ticker: str) -> str:
    # Example (XME): https://www.ssga.com/library-content/products/fund-data/etfs/us/holdings-daily-us-en-xme.xlsx
    return (
        "https://www.ssga.com/library-content/products/fund-data/etfs/us/holdings-daily-us-en-"
        + etf_ticker.strip().lower()
        + ".xlsx"
    )


@dataclass
class ThemeResult:
    theme: str
    asof_utc: str
    theme_score: float
    theme_metrics: Dict[str, float]
    ranked: pd.DataFrame  # includes scores & metrics
    selected_symbols: List[str]


def _now_utc_iso() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%SZ")


def _ensure_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)

def _now_utc_compact_yyyymmddhhmm() -> str:
    return datetime.now(timezone.utc).strftime("%Y%m%d%H%M")


def _snapshot_outputs(out_dir: str, files: List[str], stamp: Optional[str] = None) -> Optional[str]:
    """Copy selected output files into a timestamped folder under out_dir.

    Creates: <out_dir>/<yyyymmddhhmm>/
    Only copies files that exist.
    Returns the snapshot directory path, or None if nothing was copied.
    """
    stamp = stamp or _now_utc_compact_yyyymmddhhmm()
    dst = pathlib.Path(out_dir) / stamp
    dst.mkdir(parents=True, exist_ok=True)

    copied = 0
    for p in files:
        src = pathlib.Path(p)
        if not src.exists() or not src.is_file():
            continue
        shutil.copy2(src, dst / src.name)
        copied += 1

    if copied == 0:
        try:
            dst.rmdir()
        except OSError:
            pass
        return None

    return str(dst)


def _parse_themes_arg(theme: Optional[str], themes: Optional[str]) -> List[str]:
    """Return a list of theme tickers from either --themes or --theme.

    Accepts comma/space separated lists for --themes.
    """
    if themes and str(themes).strip():
        raw = str(themes)
        parts = [p.strip().upper() for p in raw.replace(" ", ",").split(",")]
        parts = [p for p in parts if p]
        return list(dict.fromkeys(parts))  # de-dup preserving order
    if theme and str(theme).strip():
        return [str(theme).strip().upper()]
    return []

def _write_summary_csv(path: str, rows: List[Dict[str, object]]) -> None:
    cols = [
        "Theme",
        "Status",
        "ThemeScore",
        "AsOfUTC",
        "BreadthAdvancersPct",
        "AvgRet20dPct",
        "Near52wHighPct",
        "VolumeQualityPct",
        "HoldingsCount",
        "TopSymbols",
        "Error",
    ]
    df = pd.DataFrame(rows)
    for c in cols:
        if c not in df.columns:
            df[c] = np.nan
    df = df[cols]
    df.to_csv(path, index=False)

def _download_file(url: str, dest_path: pathlib.Path, timeout_sec: int = 30) -> None:
    dest_path.parent.mkdir(parents=True, exist_ok=True)
    req = urllib.request.Request(
        url,
        headers={"User-Agent": "Mozilla/5.0 (theme-scan/1.0)"},
    )
    try:
        with urllib.request.urlopen(req, timeout=timeout_sec) as r:
            data = r.read()
        dest_path.write_bytes(data)
    except (urllib.error.URLError, urllib.error.HTTPError, TimeoutError) as e:
        raise RuntimeError(f"Failed to download holdings from {url}: {e}") from e


def _read_ssga_holdings_xlsx(xlsx_path: pathlib.Path) -> List[str]:
    """Read SSGA daily holdings XLSX and return a list of tickers.

    SSGA holdings files often include several metadata rows before the real table,
    and sometimes the holdings table is on a different sheet.

    Strategy:
      1) Try normal read_excel() first.
      2) If we can't find a ticker column, read all sheets with header=None and
         locate the row that looks like the holdings header.
      3) Reconstruct a dataframe from that header row and extract tickers.

    This parser is intentionally defensive to avoid picking up metadata strings
    like 'HOLDINGS:' or fund names as tickers.
    """

    def _extract_tickers_from_df(df: pd.DataFrame) -> Optional[List[str]]:
        # Accept common US ticker patterns (AAPL, BRK-B, RIO, BTU, etc.)
        ticker_re = re.compile(r"^(?=.*[A-Z])[A-Z0-9][A-Z0-9\-]{0,9}$")

        def _normalize_cell(x: object) -> str:
            return str(x).strip().upper()

        def _valid_ticker(s: str) -> bool:
            if not s:
                return False
            if " " in s:
                return False
            if ":" in s:
                # Reject common identifier formats (CUSIP/IDs) that are not tradable tickers
                if s.isdigit():
                    return False
                if re.fullmatch(r"\d{9}", s):
                    return False
                if re.fullmatch(r"[0-9A-Z]{9}", s) and sum(ch.isalpha() for ch in s) <= 1:
                    return False
                return False
            # common junk words that can appear in metadata
            if s in {"HOLDINGS", "HOLDINGS.", "HOLDINGS:", "FUND", "NAME", "US", "INC", "CORP", "CORPORATION", "TICKER", "SYMBOL", "IDENTIFIER"}:
                return False
            return bool(ticker_re.match(s))

        # Score each column by how many cells look like tickers
        best_col = None
        best_count = 0

        for c in df.columns:
            col_name = str(c).strip().lower()
            if any(k in col_name for k in ("identifier", "cusip", "isin", "sedol", "ric")):
                continue
            ser = df[c].dropna()
            if ser.empty:
                continue
            vals = [_normalize_cell(v) for v in ser.head(500).tolist()]
            good = [v for v in vals if _valid_ticker(v)]
            cnt = len(good)
            if cnt > best_count:
                best_count = cnt
                best_col = c

        # Require at least a few matches to avoid selecting metadata columns
        if best_col is None or best_count < 3:
            return None

        syms = (
            df[best_col]
            .astype(str)
            .map(lambda x: str(x).strip().upper())
            .replace("", np.nan)
            .replace("NAN", np.nan)
            .dropna()
            .tolist()
        )

        syms = [s.replace(".", "-") for s in syms]
        syms = [s for s in syms if _valid_ticker(s)]
        syms = list(dict.fromkeys(syms))  # preserve order, unique
        return syms if syms else None

    # 1) Try normal read (first sheet, inferred header)
    try:
        df0 = pd.read_excel(xlsx_path)
        syms0 = _extract_tickers_from_df(df0)
        if syms0:
            return syms0
    except Exception:
        # Fall through to robust parsing
        pass

    # 2) Robust: scan all sheets with header=None
    try:
        sheets = pd.read_excel(xlsx_path, sheet_name=None, header=None)
    except Exception as e:
        raise RuntimeError(f"Failed to parse SSGA holdings XLSX: {xlsx_path} ({e})") from e

    header_exact = {"ticker", "symbol", "trading ticker", "local ticker", "ric"}

    for sheet_name, raw in sheets.items():
        if raw is None or raw.empty:
            continue

        header_row_idx = None
        for i in range(min(len(raw), 400)):
            row_vals = raw.iloc[i].astype(str).str.strip().str.lower().tolist()
            if any(v in header_exact for v in row_vals):
                header_row_idx = i
                break

        if header_row_idx is None:
            continue

        header = raw.iloc[header_row_idx].astype(str).str.strip().tolist()
        data = raw.iloc[header_row_idx + 1 :].copy()
        data.columns = header
        data = data.dropna(axis=1, how="all")

        syms = _extract_tickers_from_df(data)
        if syms:
            return syms

    # 3) If still not found, raise with helpful info
    raise RuntimeError(
        f"Could not locate a holdings table with a ticker/symbol column in SSGA holdings file: {xlsx_path}. "
        "Tip: open the XLSX and confirm which sheet contains the holdings table and the exact column name (Ticker/Symbol/etc.)."
    )


def _read_holdings_csv(path: str) -> List[str]:
    p = pathlib.Path(path)
    if not p.exists():
        cwd = pathlib.Path.cwd()
        raise FileNotFoundError(
            f"Holdings CSV not found: {p} (cwd={cwd}). "
            f"Pass the correct path, e.g. --holdings {cwd}/holdings_xme.csv or an absolute path."
        )
    # Quick sanity check: empty file -> pandas raises EmptyDataError
    if p.stat().st_size == 0:
        raise ValueError(
            f"Holdings CSV is empty: {p}. "
            "It must contain at least one column named 'Ticker' or 'Symbol' and at least one row of tickers."
        )

    try:
        df = pd.read_csv(p)
    except pd.errors.EmptyDataError as e:
        raise ValueError(
            f"Holdings CSV has no parseable columns: {p}. "
            "Ensure the file is a CSV with a header row like: Ticker,Company (or Symbol) and at least one data row."
        ) from e

    cols = [c.lower() for c in df.columns]
    # accept either Ticker or Symbol column
    if "ticker" in cols:
        col = df.columns[cols.index("ticker")]
    elif "symbol" in cols:
        col = df.columns[cols.index("symbol")]
    else:
        raise ValueError("Holdings CSV must contain a 'Ticker' or 'Symbol' column.")

    syms = (
        df[col]
        .astype(str)
        .str.strip()
        .replace("", np.nan)
        .dropna()
        .unique()
        .tolist()
    )
    # yfinance uses '-' instead of '.' for some tickers (e.g., BRK.B -> BRK-B).
    syms = [s.replace(".", "-") for s in syms]
    return syms


def _download_history(symbols: List[str], lookback_days: int) -> pd.DataFrame:
    """
    Returns MultiIndex columns DataFrame from yfinance.download:
      columns like: ('Close','AAPL'), ('Volume','AAPL'), ...
    """
    # yfinance can be flaky with too many tickers at once; batch to reduce failures
    all_frames = []
    batch_size = 50
    for i in range(0, len(symbols), batch_size):
        batch = symbols[i:i + batch_size]
        data = yf.download(
            tickers=" ".join(batch),
            period=f"{lookback_days}d",
            interval="1d",
            auto_adjust=False,
            group_by="column",
            threads=True,
            progress=False,
        )
        if data is None or data.empty:
            continue
        all_frames.append(data)

    if not all_frames:
        raise RuntimeError("No data fetched. Check network, tickers, or yfinance availability.")

    # Merge on index; yfinance returns the same structure each call
    df = pd.concat(all_frames, axis=1)
    df = df.sort_index()
    return df


def _extract_panel(df: pd.DataFrame, field: str, symbols: List[str]) -> pd.DataFrame:
    """
    Extracts a (date x symbol) panel for a field from yfinance multi-column output.
    Handles both single-ticker and multi-ticker shapes.
    """
    if isinstance(df.columns, pd.MultiIndex):
        # Field level first if group_by="column" gives field -> symbol
        if field in df.columns.get_level_values(0):
            sub = df[field].copy()
        else:
            # sometimes layout flips
            sub = df.xs(field, axis=1, level=-1, drop_level=True).copy()
    else:
        # single ticker case
        sub = df[[field]].copy()
        sub.columns = symbols[:1]

    # Ensure only requested symbols in correct order
    cols = [c for c in symbols if c in sub.columns]
    return sub[cols].copy()


def _pct_change_last(prices: pd.Series) -> float:
    if len(prices) < 2:
        return np.nan
    prev = prices.iloc[-2]
    last = prices.iloc[-1]
    if prev == 0 or pd.isna(prev) or pd.isna(last):
        return np.nan
    return (last / prev) - 1.0


def _rolling_return(prices: pd.Series, days: int) -> float:
    if len(prices) < days + 1:
        return np.nan
    base = prices.iloc[-(days + 1)]
    last = prices.iloc[-1]
    if base == 0 or pd.isna(base) or pd.isna(last):
        return np.nan
    return (last / base) - 1.0


def _dist_to_52w_high(prices: pd.Series, window: int = 252) -> float:
    if prices.empty:
        return np.nan
    w = prices.tail(window)
    hi = w.max()
    last = prices.iloc[-1]
    if hi == 0 or pd.isna(hi) or pd.isna(last):
        return np.nan
    return (last / hi) - 1.0  # 0 means at high; negative means below high


def _vol_ratio(vol: pd.Series, window: int = 20) -> float:
    if len(vol) < window + 1:
        return np.nan
    last = vol.iloc[-1]
    avg = vol.tail(window).mean()
    if avg == 0 or pd.isna(avg) or pd.isna(last):
        return np.nan
    return float(last / avg)


def _safe_zscore(x: pd.Series) -> pd.Series:
    m = x.mean(skipna=True)
    s = x.std(skipna=True)
    if s == 0 or pd.isna(s):
        return pd.Series(np.zeros(len(x)), index=x.index)
    return (x - m) / s


def _compute_metrics(prices_df: pd.DataFrame, vol_df: pd.DataFrame, bench_prices: pd.Series) -> pd.DataFrame:
    rows = []
    for sym in prices_df.columns:
        p = prices_df[sym].dropna()
        v = vol_df[sym].dropna()
        if p.empty:
            continue

        r1 = _pct_change_last(p)
        r5 = _rolling_return(p, 5)
        r20 = _rolling_return(p, 20)
        d52 = _dist_to_52w_high(p, 252)
        vr = _vol_ratio(v, 20)

        # Relative strength vs benchmark (simple: 20d return diff)
        b = bench_prices.dropna()
        br20 = _rolling_return(b, 20)
        rs20 = (r20 - br20) if (not pd.isna(r20) and not pd.isna(br20)) else np.nan

        rows.append({
            "Symbol": sym,
            "ret_1d": r1,
            "ret_5d": r5,
            "ret_20d": r20,
            "rs_20d": rs20,
            "dist_52w_high": d52,     # closer to 0 is better (less negative)
            "vol_ratio_20d": vr,
            "last_close": float(p.iloc[-1]),
        })
    if not rows:
        return pd.DataFrame(columns=[
        "ret_1d","ret_5d","ret_20d","rs_20d","dist_52w_high","vol_ratio_20d","last_close"
    ]).set_index(pd.Index([], name="Symbol"))
    df = pd.DataFrame(rows).set_index("Symbol")

    return df


def _score_theme(theme_sym: str, metrics: pd.DataFrame) -> Tuple[float, Dict[str, float]]:
    """
    Theme score is a composite intended for morning triage.
    Uses breadth + momentum + "near highs" + volume quality.

    Returns (score 0-100, metrics dict)
    """
    # Breadth: advancers ratio by 1d return
    adv = (metrics["ret_1d"] > 0).mean(skipna=True)
    # Momentum: average 20d return
    mom = metrics["ret_20d"].mean(skipna=True)
    # Near highs: share within -10% of 52w high (dist >= -0.10)
    near_high = (metrics["dist_52w_high"] >= -0.10).mean(skipna=True)
    # Volume quality: share with vol ratio >= 1.2
    volq = (metrics["vol_ratio_20d"] >= 1.2).mean(skipna=True)

    # Normalize into a 0-100-ish score (heuristic)
    # mom is in decimal; scale it
    mom_scaled = np.clip((mom * 100) + 50, 0, 100)  # 0% -> 50, +10% -> 60
    breadth_scaled = np.clip(adv * 100, 0, 100)
    near_scaled = np.clip(near_high * 100, 0, 100)
    vol_scaled = np.clip(volq * 100, 0, 100)

    score = (
        0.35 * breadth_scaled +
        0.30 * mom_scaled +
        0.20 * near_scaled +
        0.15 * vol_scaled
    )

    details = {
        "breadth_advancers_pct": float(breadth_scaled),
        "avg_ret_20d_pct": float(mom * 100),
        "near_52w_high_pct": float(near_scaled),
        "volume_quality_pct": float(vol_scaled),
        "theme_score": float(score),
    }
    return float(score), details


def _rank_stocks(metrics: pd.DataFrame) -> pd.DataFrame:
    """
    Produce a ranking score that favors:
    - strong RS (rs_20d)
    - strong momentum (ret_20d)
    - near highs (dist_52w_high closer to 0)
    - healthy volume (vol_ratio_20d)

    Output includes final 'score' (higher is better).
    """
    df = metrics.copy()

    # Transform: dist_52w_high -> "closeness" (0 is best, -0.5 is bad)
    df["closeness_52w"] = -df["dist_52w_high"]  # smaller is better -> invert later by zscore sign

    # Z-scores (higher better)
    z_rs = _safe_zscore(df["rs_20d"])
    z_mom = _safe_zscore(df["ret_20d"])
    z_vol = _safe_zscore(df["vol_ratio_20d"])
    # closeness_52w smaller better, so use negative zscore to make "near high" positive
    z_near = -_safe_zscore(df["closeness_52w"])

    df["score"] = (
        0.35 * z_rs +
        0.30 * z_mom +
        0.20 * z_near +
        0.15 * z_vol
    )

    df = df.sort_values("score", ascending=False)
    return df


def _write_watchlist_csv(path: str, symbols: List[str]) -> None:
    pd.DataFrame({"Symbol": symbols}).to_csv(path, index=False)


def _write_report_md(path: str, result: ThemeResult, top_n: int) -> None:
    m = result.theme_metrics
    lines = []
    lines.append(f"# Theme Scan Report: {result.theme}")
    lines.append(f"- AsOf(UTC): {result.asof_utc}")
    lines.append(f"- ThemeScore: {result.theme_score:.1f}/100")
    lines.append("")
    lines.append("## Theme Metrics")
    lines.append(f"- Breadth (advancers): {m['breadth_advancers_pct']:.1f}%")
    lines.append(f"- Avg 20d return: {m['avg_ret_20d_pct']:.2f}%")
    lines.append(f"- Near 52w high (within -10%): {m['near_52w_high_pct']:.1f}%")
    lines.append(f"- Volume quality (vol ratio >= 1.2): {m['volume_quality_pct']:.1f}%")
    lines.append("")
    lines.append(f"## Top {top_n} Candidates (for TradingView watchlist)")
    lines.append("")
    # `ranked` contains only holdings; `selected_symbols` includes the theme ETF at the front.
    candidates = [s for s in result.selected_symbols if s in result.ranked.index]
    for i, sym in enumerate(candidates[:top_n], start=1):
        row = result.ranked.loc[sym]
        lines.append(
            f"{i}. **{sym}** | "
            f"1d:{row['ret_1d']*100:6.2f}%  "
            f"20d:{row['ret_20d']*100:6.2f}%  "
            f"RS20:{row['rs_20d']*100:6.2f}%  "
            f"Dist52w:{row['dist_52w_high']*100:6.2f}%  "
            f"VolX:{row['vol_ratio_20d'] if not pd.isna(row['vol_ratio_20d']) else np.nan:5.2f}"
        )
    lines.append("")
    lines.append("## Notes (manual)")
    lines.append("- Setup focus: Flag / Triangle / Episodic Pivot (confirm structure)")
    lines.append("- Risk: keep position sizing within 1% rule")
    lines.append("- Avoid: low liquidity, messy charts, weak breadth themes")
    lines.append("")

    with open(path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))

def _write_report_all_md(path: str, themes_rows: List[Dict[str, object]]) -> None:
    lines: List[str] = []
    lines.append("# Theme Scan Report (All)")
    lines.append(f"- AsOf(UTC): {_now_utc_iso()}")
    lines.append("")
    lines.append("## Summary")
    lines.append("")
    lines.append("| Theme | Status | Score | Breadth% | Avg20d% | NearHigh% | VolQ% | Holdings | Top symbols |")
    lines.append("|---|---:|---:|---:|---:|---:|---:|---:|---|")

    for r in themes_rows:
        theme = r.get("Theme", "")
        status = r.get("Status", "")
        score = r.get("ThemeScore", "")
        b = r.get("BreadthAdvancersPct", "")
        m = r.get("AvgRet20dPct", "")
        nh = r.get("Near52wHighPct", "")
        vq = r.get("VolumeQualityPct", "")
        hc = r.get("HoldingsCount", "")
        tops = r.get("TopSymbols", "")

        def _fmt(x, nd=1):
            try:
                if x is None or (isinstance(x, float) and pd.isna(x)):
                    return ""
                return f"{float(x):.{nd}f}"
            except Exception:
                return str(x)

        lines.append(
            f"| {theme} | {status} | {_fmt(score,1)} | {_fmt(b,1)} | {_fmt(m,2)} | {_fmt(nh,1)} | {_fmt(vq,1)} | {hc} | {tops} |"
        )

    failed = [r for r in themes_rows if str(r.get("Status", "")).upper() != "OK"]
    if failed:
        lines.append("")
        lines.append("## Errors")
        for r in failed:
            lines.append(f"- **{r.get('Theme','')}**: {r.get('Error','')}")

    with open(path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))



def run(theme: str, holdings_csv: str, out_dir: str, top: int, lookback_days: int, holdings_source: str = "csv", holdings_url: Optional[str] = None) -> ThemeResult:
    if holdings_source.lower() == "ssga":
        url = holdings_url or _ssga_holdings_url(theme)
        cache_path = pathlib.Path(out_dir) / f"holdings_{theme}_ssga.xlsx"
        _download_file(url, cache_path)
        symbols = _read_ssga_holdings_xlsx(cache_path)
        if not symbols:
            raise RuntimeError("No valid holding tickers were found. Check the holdings source (CSV/SSGA) and ticker parsing.")
    else:
        symbols = _read_holdings_csv(holdings_csv)
        if not symbols:
            raise RuntimeError("No valid holding tickers were found. Check the holdings source (CSV/SSGA) and ticker parsing.")
    # include theme ETF and benchmark for context
    all_syms = list(dict.fromkeys([theme, BENCH] + symbols))

    hist = _download_history(all_syms, lookback_days=lookback_days)

    close = _extract_panel(hist, PRICE_COL, all_syms)
    vol = _extract_panel(hist, VOL_COL, all_syms)

    # Benchmark series
    bench_prices = close[BENCH].dropna()
    if bench_prices.empty:
        raise RuntimeError(f"Benchmark {BENCH} has no data. Check ticker mapping.")

    # Metrics for holdings only (exclude theme, bench)
    prices_holdings = close[[s for s in symbols if s in close.columns]]
    vol_holdings = vol[[s for s in symbols if s in vol.columns]]

    metrics = _compute_metrics(prices_holdings, vol_holdings, bench_prices)

    ranked = _rank_stocks(metrics)

    theme_score, theme_metrics = _score_theme(theme, metrics)

    # Select: top N + always include the theme ETF itself first
    selected = [theme] + ranked.index.tolist()[:top]
    asof = _now_utc_iso()

    return ThemeResult(
        theme=theme,
        asof_utc=asof,
        theme_score=theme_score,
        theme_metrics=theme_metrics,
        ranked=ranked,
        selected_symbols=selected,
    )


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--theme", required=False, help="Theme ETF ticker (e.g., XME). Backward compatible with single-theme mode.")
    ap.add_argument("--themes", default=None, help="Comma-separated list of themes (e.g., XME,SMH,XBI)")
    ap.add_argument("--holdings", required=True, help="Holdings CSV path, or 'ssga' to auto-download from SSGA")
    ap.add_argument("--holdings-url", default=None, help="Override SSGA holdings URL (when --holdings ssga)")
    ap.add_argument("--out", default="./out", help="Output directory")
    ap.add_argument("--top", type=int, default=12, help="How many top stocks to include in watchlist (excluding ETF itself)")
    ap.add_argument("--lookback", type=int, default=DEFAULT_LOOKBACK_DAYS, help="Lookback days for history (default ~260)")
    args = ap.parse_args()

    if args.holdings.lower() != "ssga":
        holdings_path = pathlib.Path(args.holdings)
        if not holdings_path.exists():
            cwd = pathlib.Path.cwd()
            print(f"[ERR] Holdings CSV not found: {holdings_path} (cwd={cwd})", file=sys.stderr)
            print("      Fix by passing the correct path, e.g. --holdings /full/path/to/holdings_xme.csv", file=sys.stderr)
            return 2

    _ensure_dir(args.out)

    themes = _parse_themes_arg(args.theme, args.themes)
    if not themes:
        print("[ERR] Provide --theme XME or --themes XME,SMH,XBI", file=sys.stderr)
        return 2

    summary_rows: List[Dict[str, object]] = []
    combined_watchlist: List[str] = []
    written_files: List[str] = []


    for theme in themes:
        try:
            result = run(
                theme=theme,
                holdings_csv=args.holdings,
                out_dir=args.out,
                top=args.top,
                lookback_days=args.lookback,
                holdings_source=("ssga" if args.holdings.strip().lower() == "ssga" else "csv"),
                holdings_url=args.holdings_url,
            )

            watchlist_path = os.path.join(args.out, f"watchlist_{result.theme}.csv")
            ranking_path = os.path.join(args.out, f"ranking_{result.theme}.csv")
            report_path = os.path.join(args.out, f"report_{result.theme}.md")

            _write_watchlist_csv(watchlist_path, result.selected_symbols)
            result.ranked.to_csv(ranking_path, index=True)
            _write_report_md(report_path, result, top_n=min(args.top, len(result.selected_symbols) - 1))

            for s in result.selected_symbols:
                if s not in combined_watchlist:
                    combined_watchlist.append(s)

            m = result.theme_metrics
            top_syms = [s for s in result.selected_symbols if s != result.theme][: args.top]
            summary_rows.append({
                "Theme": result.theme,
                "Status": "OK",
                "ThemeScore": float(result.theme_score),
                "AsOfUTC": result.asof_utc,
                "BreadthAdvancersPct": float(m.get("breadth_advancers_pct", np.nan)),
                "AvgRet20dPct": float(m.get("avg_ret_20d_pct", np.nan)),
                "Near52wHighPct": float(m.get("near_52w_high_pct", np.nan)),
                "VolumeQualityPct": float(m.get("volume_quality_pct", np.nan)),
                "HoldingsCount": int(len(result.ranked.index)),
                "TopSymbols": " ".join(top_syms),
                "Error": "",
            })

            print(f"[OK] Theme: {result.theme}  Score: {result.theme_score:.1f}/100  AsOf: {result.asof_utc}")
            print(f"[OUT] {watchlist_path}")
            print(f"[OUT] {ranking_path}")
            print(f"[OUT] {report_path}")

        except Exception as e:
            summary_rows.append({
                "Theme": theme,
                "Status": "FAIL",
                "ThemeScore": np.nan,
                "AsOfUTC": _now_utc_iso(),
                "BreadthAdvancersPct": np.nan,
                "AvgRet20dPct": np.nan,
                "Near52wHighPct": np.nan,
                "VolumeQualityPct": np.nan,
                "HoldingsCount": 0,
                "TopSymbols": "",
                "Error": str(e),
            })
            print(f"[ERR] Theme {theme} failed: {e}", file=sys.stderr)

    summary_path = os.path.join(args.out, "summary_themes.csv")
    report_all_path = os.path.join(args.out, "report_ALL.md")
    watchlist_all_path = os.path.join(args.out, "watchlist_ALL.csv")

    _write_summary_csv(summary_path, summary_rows)
    _write_report_all_md(report_all_path, summary_rows)

    if combined_watchlist:
        _write_watchlist_csv(watchlist_all_path, combined_watchlist)

    print(f"[OUT] {summary_path}")
    print(f"[OUT] {report_all_path}")
    if combined_watchlist:
        print(f"[OUT] {watchlist_all_path}")

    any_ok = any(str(r.get("Status", "")).upper() == "OK" for r in summary_rows)
    return 0 if any_ok else 2


if __name__ == "__main__":
    raise SystemExit(main())