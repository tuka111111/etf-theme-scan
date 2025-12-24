from __future__ import annotations

import os
import pathlib
import re
import csv
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import requests
import yfinance as yf


from snapshot import snapshot_outputs

# --- HTTP helper for retry/backoff ---
def _get_with_retries(url: str, *, params=None, headers=None, timeout: int = 30, max_retries: int = 5) -> requests.Response:
    """HTTP GET with simple retry/backoff (handles transient 429/5xx)."""
    last_exc: Optional[Exception] = None
    for attempt in range(max_retries):
        try:
            r = requests.get(url, params=params, headers=headers, timeout=timeout)
            # Retry on rate limit / transient server errors
            if r.status_code in (429, 500, 502, 503, 504):
                # Honor Retry-After when present; otherwise exponential backoff
                ra = r.headers.get("Retry-After")
                if ra and str(ra).isdigit():
                    import time
                    time.sleep(int(ra))
                else:
                    import time
                    time.sleep(min(2 ** attempt, 16))
                continue
            r.raise_for_status()
            return r
        except Exception as e:
            last_exc = e
            import time
            time.sleep(min(2 ** attempt, 16))
    raise RuntimeError(f"HTTP GET failed after retries: {url}: {last_exc}")


# =========================
# SSGA holdings (embedded)
# =========================
VANECK_SMH_CSV_URL = "https://www.vaneck.com/api/fundholdings/download?identifier=SMH"

def _normalize_ticker(val: object) -> Optional[str]:
    if val is None:
        return None
    s = str(val).strip().upper()
    if not s:
        return None

    # Handle formats like "NVDA US" (country suffix)
    if " " in s:
        s = s.split()[0].strip()

    # Filter obvious header/noise tokens / common webpage words that can appear if a holdings download
    # accidentally returns HTML or a preamble block.
    STOP = {
        "TICKER", "SYMBOL", "IDENTIFIER", "HOLDINGS", "HOLDINGS:", "FUND", "FUND NAME", "NAME",
        "US", "USA", "INC", "CORP", "LTD", "PLC", "CO", "COMPANY",
        "SELECT", "ABOUT", "FOLLOW", "CONTACT", "NEWS", "MEDIA", "CEO", "VANECK",
        "SCRIPT", "SCRIPT.SRC", "SRC", "MODEL", "RETURN", "NATURAL", "VAR", "REST",
        "IF", "NO", "OUR", "I", "RIA",
    }
    if s in STOP:
        return None

    # Filter CUSIP/ISIN-like strings (mostly digits/letters, long)
    if re.fullmatch(r"[0-9A-Z]{8,}", s):
        return None

    # Reject long alphabetic company-name-like tokens (e.g. SUNCOKE, RYERSON)
    if len(s) > 6 and s.isalpha():
        return None

    # Sanity check: allow at most one dot segment (e.g., BRK.B). Reject long multi-word domains.
    if s.count(".") > 1:
        return None
    if "." in s:
        a, b = s.split(".", 1)
        # Keep realistic class shares / suffixes (single-letter or short)
        if len(a) > 6 or len(b) > 3:
            return None

    # Basic ticker validity: allow A-Z, 0-9, dot, dash
    if not re.fullmatch(r"[A-Z0-9][A-Z0-9.\-]{0,12}", s):
        return None

    return s


def download_ssga_holdings_xlsx(
    theme: str,
    out_dir: str | Path,
    url_template: str | None = None,
) -> Path:
    """Download SSGA holdings XLSX for the given ETF ticker (e.g., XME)."""
    theme = str(theme).strip().upper()
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # Default template for SPDR ETFs (SSGA)
    if url_template is None:
        url_template = (
            "https://www.ssga.com/library-content/products/fund-data/"
            "etfs/us/holdings-daily-us-en-{ticker}.xlsx"
        )

    url = url_template.format(ticker=theme.lower())
    dst = out_dir / f"holdings_{theme}_ssga.xlsx"

    headers = {
        "User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36",
        "Accept": "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet,application/octet-stream;q=0.9,*/*;q=0.8",
    }
    r = requests.get(url, timeout=30, headers=headers)
    r.raise_for_status()
    dst.write_bytes(r.content)
    return dst



# New helper for reading SSGA holdings xlsx robustly, returning DataFrame
def read_ssga_holdings_xlsx(path: str | Path) -> pd.DataFrame:
    """Read SSGA holdings XLSX robustly.

    SSGA's daily holdings XLSX sometimes begins with a cover-sheet/preamble (fund name,
    disclaimers, dates) where a cell like "Ticker Symbol:" appears. If we treat that row
    as the header, we end up parsing a disclaimer page and producing garbage tickers.

    Strategy:
    - Read the first sheet with header=None.
    - Search for a *table header row* that contains a ticker/symbol column AND at least
      one other common holdings column (e.g., name/weight/shares/market value).
    - Then re-read with that row as header and normalize column names.
    """
    path = str(path)

    raw = pd.read_excel(path, header=None)

    def _norm_cell(x: object) -> str:
        s = "" if x is None else str(x)
        s = s.strip().lower()
        # normalize both ASCII and full-width colons
        s = s.replace(":", "").replace("：", "")
        s = s.replace("\n", " ")
        s = re.sub(r"\s+", " ", s)
        return s

    # Look further down than the first 20 rows; the cover sheet can be long.
    max_scan = min(250, len(raw))
    header_row: Optional[int] = None

    # Keywords that typically appear in the actual holdings table header
    ticker_keys = {"ticker", "symbol", "ticker symbol"}
    other_keys = {
        "name",
        "security name",
        "holding",
        "issuer",
        "weight",
        "weighting",
        "market value",
        "shares",
        "shares held",
        "cusip",
        "isin",
        "sedol",
        "country",
        "sector",
    }

    for i in range(max_scan):
        row = [_norm_cell(v) for v in raw.iloc[i].tolist()]
        row_set = set([c for c in row if c])
        has_ticker = any(any(k in c for k in ticker_keys) for c in row_set)
        has_other = any(any(k in c for k in other_keys) for c in row_set)

        # Require multiple non-empty header-ish cells to avoid picking "Ticker Symbol:" label rows
        nonempty = sum(1 for c in row if c)
        if has_ticker and has_other and nonempty >= 3:
            header_row = i
            break

    if header_row is None:
        raise ValueError("Could not locate holdings table header row in SSGA Excel")

    df = pd.read_excel(path, header=header_row)

    def _norm_col(c: object) -> str:
        c = "" if c is None else str(c)
        c = c.strip().lower()
        c = c.replace(":", "").replace("：", "")
        c = c.replace("\n", " ")
        c = re.sub(r"\s+", " ", c)
        return c

    df.columns = [_norm_col(c) for c in df.columns]

    # Drop columns that are completely empty
    df = df.dropna(axis=1, how="all")
    return df

# =========================
# VanEck holdings (SMH)
# =========================

def _extract_first_csv_url_from_html(html: str, base_url: str) -> Optional[str]:
    """Try to locate a CSV download link inside an HTML response."""
    if not html:
        return None

    m = re.search(r'href="([^"]+\.csv[^"]*)"', html, flags=re.IGNORECASE)
    if not m:
        m = re.search(r"href='([^']+\.csv[^']*)'", html, flags=re.IGNORECASE)

    if not m:
        # Sometimes the CSV is behind an API path
        m = re.search(
            r"(https?://[^\s\"']+download[^\"']*identifier=SMH[^\"']*)",
            html,
            flags=re.IGNORECASE
        )

        if not m:
            m = re.search(r"(/api/fundholdings/download[^\"']*identifier=SMH[^\"']*)", html, flags=re.IGNORECASE)

    if not m:
        return None

    href = m.group(1)
    if href.startswith("//"):
        return "https:" + href
    if href.startswith("http"):
        return href
    return requests.compat.urljoin(base_url, href)


def download_vaneck_holdings_csv(theme: str, out_dir: str | Path) -> Path:
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    dst = out_dir / f"holdings_{theme}_vaneck.csv"

    # ★ 追加：キャッシュがあれば使えるようにする
    cached_path = dst if dst.exists() else None
    
    theme = theme.upper()

    headers = {
        "User-Agent": "Mozilla/5.0",
        "Accept": "text/csv,application/csv,application/octet-stream;q=0.9,*/*;q=0.8",
        "Accept-Language": "en-US,en;q=0.9,ja;q=0.8",
        "Referer": "https://www.vaneck.com/us/en/investments/semiconductor-etf-smh/holdings/",
    }

    r = _get_with_retries(VANECK_SMH_CSV_URL, headers=headers, timeout=30)
    r.raise_for_status()

    content = r.content or b""
    head = content[:4096].lower()
    if b"<html" in head or b"<!doctype" in head or b"<script" in head:
        if cached_path:
            print(f"[WARN] VanEck returned HTML; using cached file for {theme}")
            return cached_path

    if b"<html" in head or b"<!doctype" in head or b"<script" in head:
        # Fallback: VanEck sometimes returns an HTML page (bot check / redirect). Try to extract a CSV link.
        html = content.decode("utf-8", errors="replace")
        alt = _extract_first_csv_url_from_html(html, base_url="https://www.vaneck.com/")
        if not alt:
            raise RuntimeError(f"VanEck holdings download did not return CSV (HTML detected) for {theme}.")
        r2 = _get_with_retries(alt, headers=headers, timeout=30)
        r2.raise_for_status()
        content2 = r2.content or b""
        head2 = content2[:4096].lower()
        if b"<html" in head2 or b"<!doctype" in head2 or b"<script" in head2:
            raise RuntimeError(f"VanEck holdings download did not return CSV (HTML detected) for {theme} (fallback URL also HTML).")
        dst.write_bytes(content2)
        return dst

    dst.write_bytes(content)
    return dst


def read_vaneck_holdings_xlsx(xlsx_path: str | Path) -> List[str]:
    """Read VanEck holdings from XLSX and return unique tickers.

    Users may accidentally rename an XLSX to .csv; such files start with PK (zip magic).
    """
    xlsx_path = Path(xlsx_path)

    raw = pd.read_excel(xlsx_path, header=None)

    def _norm_cell(x: object) -> str:
        s = "" if x is None else str(x)
        s = s.strip().lower()
        s = s.replace(":", "").replace("：", "")
        s = s.replace("\n", " ")
        s = re.sub(r"\s+", " ", s)
        return s

    max_scan = min(250, len(raw))
    header_row: Optional[int] = None

    ticker_keys = {"ticker", "symbol", "ticker symbol", "holding ticker", "holding symbol"}
    other_keys = {
        "name",
        "security name",
        "issuer",
        "weight",
        "market value",
        "shares",
        "cusip",
        "isin",
        "country",
        "sector",
    }

    for i in range(max_scan):
        row = [_norm_cell(v) for v in raw.iloc[i].tolist()]
        row_set = set([c for c in row if c])
        has_ticker = any(any(k in c for k in ticker_keys) for c in row_set)
        has_other = any(any(k in c for k in other_keys) for c in row_set)
        nonempty = sum(1 for c in row if c)
        if has_ticker and has_other and nonempty >= 3:
            header_row = i
            break

    if header_row is None:
        raise ValueError("Could not locate holdings table header row in VanEck Excel")

    df = pd.read_excel(xlsx_path, header=header_row)

    def _norm_col(c: object) -> str:
        c = "" if c is None else str(c)
        c = c.strip().lower()
        c = c.replace(":", "").replace("：", "")
        c = c.replace("\n", " ")
        c = re.sub(r"\s+", " ", c)
        return c

    df.columns = [_norm_col(c) for c in df.columns]
    df = df.dropna(axis=1, how="all")

    ticker_col: Optional[str] = None
    for c in df.columns:
        if c == "symbol" or c == "ticker" or c == "ticker symbol" or ("ticker" in c and "symbol" in c):
            ticker_col = c
            break

    if ticker_col is None:
        raise RuntimeError(
            f"Could not find a ticker/symbol column in VanEck Excel. Columns={df.columns.tolist()}"
        )

    syms: List[str] = []
    for v in df[ticker_col].dropna().astype(str).str.strip().tolist():
        s = _normalize_ticker(v)
        if s:
            syms.append(s)

    uniq = sorted(set(syms))
    if len(uniq) < 10:
        raise RuntimeError(
            f"Too few tickers parsed from VanEck Excel ({len(uniq)}). Likely wrong sheet/content."
        )

    return uniq

def read_vaneck_holdings_csv(csv_path: str | Path) -> List[str]:
    """
    Read VanEck holdings CSV robustly.

    VanEck downloads can vary:
    - delimiter may be ';' or ','
    - may contain preamble rows
    - may have occasional malformed quoting
    This function:
    - decodes safely (utf-8-sig with replacement)
    - detects delimiter from a short sample
    - uses pandas python engine and skips bad lines
    - finds a Symbol/Ticker-like column (fallback to first column)
    """
    csv_path = Path(csv_path)

    # Detect accidental XLSX input (XLSX is a ZIP; starts with PK). If so, parse via the XLSX reader.
    try:
        head_bytes = csv_path.read_bytes()[:8]
        if head_bytes.startswith(b"PK\x03\x04") or head_bytes.startswith(b"PK\x05\x06") or head_bytes.startswith(b"PK\x07\x08"):
            return read_vaneck_holdings_xlsx(csv_path)
    except Exception:
        # If we can't read bytes here, fall back to the normal text-path below.
        pass

    # Read as text safely
    raw_text = csv_path.read_text(encoding="utf-8-sig", errors="replace")
    low = raw_text.lower()
    if "<html" in low or "<!doctype" in low or "<script" in low:
        raise RuntimeError("VanEck holdings content looks like HTML (not a CSV). Check the download URL / headers.")

    lines = [ln for ln in raw_text.splitlines() if ln.strip()]

    if not lines:
        raise RuntimeError("VanEck holdings CSV is empty")

    # Find the header row (first row that looks like a table header)
    header_idx = 0
    for i, ln in enumerate(lines[:50]):
        u = ln.upper()
        if ("SYMBOL" in u) or ("TICKER" in u):
            header_idx = i
            break

    # Detect delimiter using csv.Sniffer with a safe fallback set
    sample = "\n".join(lines[header_idx:header_idx + 25])
    delim = ";"
    try:
        dialect = csv.Sniffer().sniff(sample, delimiters=";,\t|")
        delim = dialect.delimiter
    except Exception:
        # Heuristic: choose the delimiter that appears most in the header line
        hdr = lines[header_idx]
        counts = {d: hdr.count(d) for d in [";", ",", "\t", "|"]}
        delim = max(counts, key=counts.get) if max(counts.values()) > 0 else ";"

    # Parse with pandas (python engine), skip malformed lines
    try:
        df = pd.read_csv(
            csv_path,
            engine="python",
            sep=delim,
            encoding="utf-8-sig",
            encoding_errors="replace",
            skiprows=header_idx,
            skip_blank_lines=True,
            on_bad_lines="skip",
        )
    except Exception:
        # Last-resort: ignore quoting entirely (treat quotes as normal chars)
        df = pd.read_csv(
            csv_path,
            engine="python",
            sep=delim,
            encoding="utf-8-sig",
            encoding_errors="replace",
            skiprows=header_idx,
            skip_blank_lines=True,
            on_bad_lines="skip",
            quoting=csv.QUOTE_NONE,
            escapechar="\\",
        )

    if df is None or df.empty:
        raise RuntimeError("VanEck holdings CSV is empty or unreadable after parsing")

    # Find ticker/symbol column
    col = None
    for c in df.columns:
        c_low = str(c).strip().lower()
        if "symbol" in c_low or "ticker" in c_low:
            col = c
            break
        # Some VanEck exports use "Holding Ticker" etc.
        if "holding" in c_low and ("symbol" in c_low or "ticker" in c_low):
            col = c
            break

    # Fallback: first column
    if col is None:
        raise RuntimeError(f"Could not find a ticker/symbol column in VanEck CSV. Columns={list(df.columns)}")

    syms: List[str] = []
    for v in df[col].dropna().tolist():
        s = _normalize_ticker(v)
        if s:
            syms.append(s)
    
    if len(set(syms)) < 10:
        raise RuntimeError(f"Too few tickers parsed from VanEck CSV ({len(set(syms))}). Likely wrong content or format.")

    if not syms:
        raise RuntimeError("No valid tickers found in VanEck CSV")

    return sorted(set(syms))

def get_yahoo_etf_holdings(theme: str) -> List[str]:
    """Fetch ETF holdings via Yahoo Finance.

    Works across yfinance versions:
    - If `yf.Ticker(theme).holdings` exists and is a DataFrame, use it.
    - Otherwise, call Yahoo's quoteSummary endpoint with `modules=topHoldings`.
    """
    theme = str(theme).strip().upper()

    # 1) Try yfinance attribute (newer versions)
    try:
        t = yf.Ticker(theme)
        h = getattr(t, "holdings", None)
        if isinstance(h, pd.DataFrame) and not h.empty:
            col = None
            for c in h.columns:
                c_low = str(c).lower()
                if c_low in {"symbol", "ticker"}:
                    col = c
                    break
            if col is None:
                col = h.columns[0]

            syms: List[str] = []
            for v in h[col].tolist():
                s = _normalize_ticker(v)
                if s:
                    syms.append(s)

            uniq = sorted(set(syms))
            if uniq:
                return uniq
    except Exception:
        pass

    # 2) Fallback: Yahoo quoteSummary API
    url = f"https://query2.finance.yahoo.com/v10/finance/quoteSummary/{theme}"
    params = {"modules": "topHoldings"}
    headers = {
        "User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36",
        "Accept": "application/json,text/plain,*/*",
    }

    r = _get_with_retries(url, params=params, headers=headers, timeout=30)
    r.raise_for_status()
    data = r.json()

    try:
        result = data["quoteSummary"]["result"][0]
        holdings = result.get("topHoldings", {}).get("holdings", [])
    except Exception as e:
        raise RuntimeError(f"Yahoo quoteSummary parse failed for {theme}: {e}")

    syms: List[str] = []
    for item in holdings:
        sym = _normalize_ticker(item.get("symbol"))
        if sym:
            syms.append(sym)

    uniq = sorted(set(syms))
    if not uniq:
        raise RuntimeError(f"Yahoo holdings not available for ETF {theme}")
    return uniq


# =========================
# Theme scan core
# =========================


@dataclass
class ThemeResult:
    theme: str
    asof_utc: str
    theme_score: float
    theme_metrics: Dict[str, float]
    selected_symbols: List[str]
    ranked: pd.DataFrame


def now_utc_iso() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%SZ")


def now_utc_compact_yyyymmddhhmm() -> str:
    return datetime.now(timezone.utc).strftime("%Y%m%d%H%M")


def ensure_dir(path: str) -> None:
    pathlib.Path(path).mkdir(parents=True, exist_ok=True)


def parse_themes_arg(theme: Optional[str], themes: Optional[str]) -> List[str]:
    if themes and str(themes).strip():
        raw = str(themes)
        parts = [p.strip().upper() for p in raw.replace(" ", ",").split(",")]
        parts = [p for p in parts if p]
        return list(dict.fromkeys(parts))
    if theme and str(theme).strip():
        return [str(theme).strip().upper()]
    return []


def write_watchlist_csv(path: str, symbols: List[str]) -> None:
    pd.DataFrame({"Symbol": symbols}).to_csv(path, index=False)


def write_summary_csv(path: str, rows: List[Dict[str, object]]) -> None:
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
    df[cols].to_csv(path, index=False)


def write_report_all_md(path: str, themes_rows: List[Dict[str, object]]) -> None:
    lines: List[str] = []
    lines.append("# Theme Scan Report (All)")
    lines.append(f"- AsOf(UTC): {now_utc_iso()}")
    lines.append("")
    lines.append("## Summary")
    lines.append("")
    lines.append("| Theme | Status | Score | Breadth% | Avg20d% | NearHigh% | VolQ% | Holdings | Top symbols |")
    lines.append("|---|---:|---:|---:|---:|---:|---:|---:|---|")

    def _fmt(x, nd=1) -> str:
        try:
            if x is None or (isinstance(x, float) and pd.isna(x)):
                return ""
            return f"{float(x):.{nd}f}"
        except Exception:
            return str(x)

    for r in themes_rows:
        lines.append(
            "| "
            + " | ".join(
                [
                    str(r.get("Theme", "")),
                    str(r.get("Status", "")),
                    _fmt(r.get("ThemeScore", ""), 1),
                    _fmt(r.get("BreadthAdvancersPct", ""), 1),
                    _fmt(r.get("AvgRet20dPct", ""), 2),
                    _fmt(r.get("Near52wHighPct", ""), 1),
                    _fmt(r.get("VolumeQualityPct", ""), 1),
                    str(r.get("HoldingsCount", "")),
                    str(r.get("TopSymbols", "")),
                ]
            )
            + " |"
        )

    failed = [r for r in themes_rows if str(r.get("Status", "")).upper() != "OK"]
    if failed:
        lines.append("")
        lines.append("## Errors")
        for r in failed:
            lines.append(f"- **{r.get('Theme','')}**: {r.get('Error','')}")

    with open(path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))


def write_report_md(path: str, result: ThemeResult, top_n: int = 12) -> None:
    lines: List[str] = []
    lines.append(f"# Theme Scan Report: {result.theme}")
    lines.append(f"- AsOf(UTC): {result.asof_utc}")
    lines.append(f"- ThemeScore: {result.theme_score:.1f}/100")
    lines.append("")

    m = result.theme_metrics
    lines.append("## Theme Metrics")
    lines.append("")
    lines.append(f"- Breadth (advancers %): {m.get('breadth_advancers_pct', float('nan')):.1f}")
    lines.append(f"- Avg 20d return (%): {m.get('avg_ret_20d_pct', float('nan')):.2f}")
    lines.append(f"- Near 52w high (% within -10%): {m.get('near_52w_high_pct', float('nan')):.1f}")
    lines.append(f"- Volume quality (% vol_ratio>=1.2): {m.get('volume_quality_pct', float('nan')):.1f}")
    lines.append("")

    lines.append("## Top Ranked")
    lines.append("")
    lines.append("| Symbol | Score | ret_1d% | ret_20d% | rs_20d | dist_52w_high% | vol_ratio_20d | last_close |")
    lines.append("|---|---:|---:|---:|---:|---:|---:|---:|")

    df = result.ranked.copy()
    for c in ["score", "ret_1d", "ret_20d", "rs_20d", "dist_52w_high", "vol_ratio_20d", "last_close"]:
        if c not in df.columns:
            df[c] = np.nan

    shown = 0
    for sym, row in df.sort_values("score", ascending=False).iterrows():
        if shown >= top_n:
            break
        lines.append(
            f"| {sym} | {row.get('score', np.nan):.1f} | {row.get('ret_1d', np.nan)*100:.2f} | {row.get('ret_20d', np.nan)*100:.2f} | {row.get('rs_20d', np.nan):.3f} | {row.get('dist_52w_high', np.nan)*100:.2f} | {row.get('vol_ratio_20d', np.nan):.2f} | {row.get('last_close', np.nan):.2f} |"
        )
        shown += 1

    with open(path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))


def snapshot_written_files(out_dir: str, written_files: List[str]) -> Optional[str]:
    """Create out/<YYYYMMDDHHMM>/ and copy the given output files into it."""
    snap_dir = snapshot_outputs(out_dir, written_files)
    return str(snap_dir) if snap_dir is not None else None


def _resolve_holdings_csv_path(holdings_arg: str, theme: str) -> str:
    p = pathlib.Path(holdings_arg)
    if "{theme}" in holdings_arg:
        return holdings_arg.format(theme=theme)
    if p.exists() and p.is_dir():
        cand = p / f"holdings_{theme.lower()}.csv"
        if cand.exists():
            return str(cand)
        cand2 = p / f"holdings_{theme.upper()}.csv"
        if cand2.exists():
            return str(cand2)
        raise FileNotFoundError(f"Holdings directory provided, but holdings_{theme}.csv not found in {p}")
    return holdings_arg


def _read_holdings_csv(path: str) -> List[str]:
    df = pd.read_csv(path)
    if df.empty:
        return []

    cols = [c.strip().lower() for c in df.columns]
    col = None
    for want in ["symbol", "ticker", "tickers"]:
        if want in cols:
            col = df.columns[cols.index(want)]
            break
    if col is None:
        col = df.columns[0]

    syms = (
        df[col]
        .astype(str)
        .str.strip()
        .str.upper()
        .replace("", np.nan)
        .dropna()
        .tolist()
    )

    out: List[str] = []
    seen = set()
    for s in syms:
        if s and s not in seen:
            out.append(s)
            seen.add(s)
    return out


def _download_prices(symbols: List[str], lookback_days: int) -> pd.DataFrame:
    if not symbols:
        return pd.DataFrame()

    period = f"{max(lookback_days, 60)}d"
    data = yf.download(
        tickers=" ".join(symbols),
        period=period,
        interval="1d",
        auto_adjust=True,
        group_by="ticker",
        threads=True,
        progress=False,
    )
    if data is None or len(data) == 0:
        return pd.DataFrame()

    if isinstance(data.columns, pd.MultiIndex):
        if "Close" in data.columns.get_level_values(0):
            closes = data["Close"].copy()
        else:
            closes = pd.DataFrame(
                {
                    t: data[t]["Close"]
                    for t in data.columns.get_level_values(0).unique()
                    if isinstance(data[t], pd.DataFrame) and "Close" in data[t].columns
                }
            )
    else:
        closes = pd.DataFrame({symbols[0]: data["Close"]})

    return closes.dropna(how="all")


def _compute_metrics(holdings_close: pd.DataFrame, bench_close: pd.Series) -> pd.DataFrame:
    rows: List[Dict[str, object]] = []
    if holdings_close is None or holdings_close.empty:
        return pd.DataFrame().set_index(pd.Index([], name="Symbol"))

    for sym in holdings_close.columns:
        s = holdings_close[sym].dropna()
        if len(s) < 40:
            continue

        aligned = pd.concat([s, bench_close], axis=1, join="inner").dropna()
        if aligned.shape[0] < 40:
            continue

        s_al = aligned.iloc[:, 0]
        b_al = aligned.iloc[:, 1]

        def _ret_n(n: int) -> float:
            if len(s_al) <= n:
                return float("nan")
            return float(s_al.iloc[-1] / s_al.iloc[-1 - n] - 1.0)

        ret_1d = _ret_n(1)
        ret_5d = _ret_n(5)
        ret_20d = _ret_n(20)

        rs_20d = float("nan")
        if len(b_al) > 20:
            bench_20d = float(b_al.iloc[-1] / b_al.iloc[-21] - 1.0)
            if np.isfinite(bench_20d) and bench_20d != 0 and np.isfinite(ret_20d):
                rs_20d = float(ret_20d / bench_20d)

        hi_52w = float(s_al.tail(252).max())
        last = float(s_al.iloc[-1])
        dist_52w_high = float(last / hi_52w - 1.0) if hi_52w > 0 else float("nan")

        vol_ratio_20d = float("nan")
        try:
            v = yf.download(sym, period="90d", interval="1d", auto_adjust=False, progress=False)
            if v is not None and len(v) >= 25 and "Volume" in v.columns:
                vol = v["Volume"].astype(float).dropna()
                if len(vol) >= 25:
                    v20 = float(vol.tail(20).mean().iloc[0])
                    vprev = float(vol.tail(60).head(40).mean().iloc[0]) if len(vol) >= 60 else float(vol.mean().iloc[0])
                    if vprev > 0:
                        vol_ratio_20d = float(v20 / vprev)
        except Exception:
            pass

        rows.append(
            {
                "Symbol": sym,
                "ret_1d": ret_1d,
                "ret_5d": ret_5d,
                "ret_20d": ret_20d,
                "rs_20d": rs_20d,
                "dist_52w_high": dist_52w_high,
                "vol_ratio_20d": vol_ratio_20d,
                "last_close": last,
            }
        )

    if not rows:
        return pd.DataFrame(
            columns=["ret_1d", "ret_5d", "ret_20d", "rs_20d", "dist_52w_high", "vol_ratio_20d", "last_close"]
        ).set_index(pd.Index([], name="Symbol"))

    df = pd.DataFrame(rows).set_index("Symbol")

    def _z(series: pd.Series) -> pd.Series:
        s = series.replace([np.inf, -np.inf], np.nan).astype(float)
        if s.dropna().empty:
            return s * np.nan
        return (s - s.mean()) / (s.std(ddof=0) + 1e-12)

    score = (
        0.25 * _z(df["ret_1d"])
        + 0.25 * _z(df["ret_20d"])
        + 0.25 * _z(df["rs_20d"])
        + 0.15 * _z(df["dist_52w_high"])
        + 0.10
        * _z(
            df["vol_ratio_20d"].fillna(
                df["vol_ratio_20d"].median() if df["vol_ratio_20d"].notna().any() else 1.0
            )
        )
    )
    df["score"] = (score - score.min()) / ((score.max() - score.min()) + 1e-12) * 100.0
    return df


def _score_theme(metrics_df: pd.DataFrame) -> tuple[float, Dict[str, float]]:
    if metrics_df is None or metrics_df.empty:
        return 0.0, {
            "breadth_advancers_pct": 0.0,
            "avg_ret_20d_pct": 0.0,
            "near_52w_high_pct": 0.0,
            "volume_quality_pct": 0.0,
        }

    ret_1d = metrics_df["ret_1d"].astype(float)
    ret_20d = metrics_df["ret_20d"].astype(float)
    dist = metrics_df["dist_52w_high"].astype(float)
    volr = metrics_df["vol_ratio_20d"].astype(float)

    breadth = float((ret_1d > 0).mean() * 100.0)
    avg_ret_20d_pct = float(ret_20d.replace([np.inf, -np.inf], np.nan).mean() * 100.0)
    near_high = float((dist >= -0.10).mean() * 100.0)
    volq = float((volr >= 1.2).mean() * 100.0) if volr.notna().any() else 0.0

    mom_scaled = float(np.clip(avg_ret_20d_pct + 50.0, 0.0, 100.0))
    theme_score = 0.35 * breadth + 0.30 * mom_scaled + 0.20 * near_high + 0.15 * volq

    return float(np.clip(theme_score, 0.0, 100.0)), {
        "breadth_advancers_pct": breadth,
        "avg_ret_20d_pct": avg_ret_20d_pct,
        "near_52w_high_pct": near_high,
        "volume_quality_pct": volq,
    }


def run_theme(
    theme: str,
    holdings: str,
    out_dir: str,
    top: int = 12,
    lookback_days: int = 260,
    holdings_url: Optional[str] = None,
) -> ThemeResult:
    theme = theme.strip().upper()
    out_dir = os.path.abspath(out_dir)
    ensure_dir(out_dir)

    if holdings.strip().lower() == "ssga":
        if theme == "SMH":
            # --- SMH: local file fixed for stability (no network access) ---
            # Supports both CSV and XLSX (auto-detected by file suffix).
            local_file = Path(out_dir) / f"holdings_{theme}_vaneck.csv"

            if not local_file.exists():
                # Accept user-provided variants like:
                # - holdings_SMH_vaneck*.csv
                # - holdings_SMH_vaneck*.xlsx
                cands = []
                cands += sorted(Path(out_dir).glob(f"holdings_{theme}_vaneck*.csv"))
                cands += sorted(Path(out_dir).glob(f"holdings_{theme}_vaneck*.xlsx"))
                cands += sorted(Path(out_dir).glob(f"holdings_{theme}_vaneck*.xls"))
                if cands:
                    local_file = cands[0]

            if not local_file.exists():
                raise RuntimeError(
                    f"SMH holdings unavailable: place a local holdings file in {out_dir} named "
                    f"holdings_{theme}_vaneck.csv (or holdings_{theme}_vaneck*.csv/.xlsx)."
                )

            suf = local_file.suffix.lower()
            if suf in {".xlsx", ".xls"}:
                holdings_syms = read_vaneck_holdings_xlsx(local_file)
            else:
                holdings_syms = read_vaneck_holdings_csv(local_file)
        else:
            # Try multiple SSGA URL templates because the site sometimes serves a cover-sheet XLSX
            if holdings_url and str(holdings_url).strip():
                templates = [holdings_url]
            else:
                templates = [
                    "https://www.ssga.com/library-content/products/fund-data/etfs/us/holdings-daily-us-en-{ticker}.xlsx",
                    "https://www.ssga.com/us/en/intermediary/etfs/library-content/products/fund-data/etfs/us/holdings-daily-us-en-{ticker}.xlsx",
                    "https://www.ssga.com/us/en/institutional/etfs/library-content/products/fund-data/etfs/us/holdings-daily-us-en-{ticker}.xlsx",
                    "https://www.ssga.com/us/en/individual/etfs/library-content/products/fund-data/etfs/us/holdings-daily-us-en-{ticker}.xlsx",
                ]

            last_err: Optional[Exception] = None
            tickers = []
            for tpl in templates:
                try:
                    xlsx_path = download_ssga_holdings_xlsx(theme, out_dir=out_dir, url_template=tpl)
                    # Use the new helper to read the DataFrame
                    df = read_ssga_holdings_xlsx(xlsx_path)
                    # Normalized ticker column detection
                    ticker_col = None
                    for c in df.columns:
                        if (
                            c == "ticker"
                            or c == "symbol"
                            or c == "ticker symbol"
                            or ("ticker" in c and "symbol" in c)
                        ):
                            ticker_col = c
                            break

                    if ticker_col is None:
                        raise ValueError(f"Could not find ticker/symbol column in {xlsx_path}. Columns={df.columns.tolist()}")

                    raw_vals = (
                        df[ticker_col]
                          .dropna()
                          .astype(str)
                          .str.strip()
                          .tolist()
                    )
                    tickers = []
                    for v in raw_vals:
                        s = _normalize_ticker(v)
                        if s:
                            tickers.append(s)

                    # If we didn't get enough valid tickers, it's probably a cover sheet / disclaimer.
                    tickers = sorted(set(tickers))
                    if len(tickers) >= 10:
                        break
                except (requests.exceptions.HTTPError, RuntimeError, ValueError) as e:
                    last_err = e
                    continue

            holdings_syms = sorted(set([s for s in tickers if s]))
            if not holdings_syms:
                if last_err is not None:
                    raise last_err
                raise RuntimeError("No valid holding tickers were found.")
    else:
        csv_path = _resolve_holdings_csv_path(holdings, theme)
        holdings_syms = _read_holdings_csv(csv_path)

    if not holdings_syms:
        raise RuntimeError("No valid holding tickers were found.")

    bench = theme
    all_symbols = list(dict.fromkeys([bench] + holdings_syms))

    closes = _download_prices(all_symbols, lookback_days=lookback_days)
    if closes.empty or bench not in closes.columns:
        raise RuntimeError(f"Failed to download benchmark prices for {bench}")

    bench_close = closes[bench].dropna()
    holdings_cols = [c for c in closes.columns if c != bench]
    holdings_close = closes[holdings_cols].copy()

    metrics_df = _compute_metrics(holdings_close, bench_close)
    theme_score, theme_metrics = _score_theme(metrics_df)

    ranked = metrics_df.sort_values("score", ascending=False)
    selected = [theme] + ranked.head(max(0, top)).index.tolist()

    return ThemeResult(
        theme=theme,
        asof_utc=now_utc_iso(),
        theme_score=theme_score,
        theme_metrics=theme_metrics,
        selected_symbols=selected,
        ranked=ranked,
    )


__all__ = [
    "ThemeResult",
    "run_theme",
    "parse_themes_arg",
    "write_watchlist_csv",
    "write_report_md",
    "write_report_all_md",
    "write_summary_csv",
    "snapshot_written_files",
    "download_ssga_holdings_xlsx",
    "read_ssga_holdings_xlsx",
    "get_yahoo_etf_holdings",
]
__all__ += [
    "download_vaneck_holdings_csv",
    "read_vaneck_holdings_csv",
    "read_vaneck_holdings_xlsx",
]
