"""
Step2: Price / indicator fetch (HTF only).

Downloads OHLCV for symbols from Step1 universe and writes per-theme CSV/JSON under out/step2_prices/.
"""
from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path
from typing import Dict, List, Optional, Sequence

import pandas as pd

from .common import ensure_dir, now_utc_iso, parse_csv_list
from .io_step1 import group_symbols_by_theme, load_universe
from .validate import must_validate

LOG = logging.getLogger(__name__)


def _yf_interval(tf: str) -> str:
    m = {"1D": "1d", "4H": "60m", "1H": "60m", "30m": "30m", "15m": "15m", "5m": "5m", "1m": "1m"}
    return m.get(tf.upper(), "1d")


def _download_prices(symbol: str, lookback_days: int, tf: str) -> pd.DataFrame:
    try:
        import yfinance as yf  # type: ignore
    except Exception as e:  # pragma: no cover
        LOG.warning("yfinance unavailable, marking missing for %s: %s", symbol, e)
        return pd.DataFrame()

    df = yf.download(
        tickers=symbol,
        period=f"{lookback_days}d",
        interval=_yf_interval(tf),
        auto_adjust=True,
        progress=False,
    )
    if df is None or df.empty:
        return pd.DataFrame()

    # yfinance may return MultiIndex columns like ('Open', 'ABBV') for single-ticker downloads.
    # Flatten to top-level field names so downstream code can read r['Open'] etc.
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)

    return df.reset_index()


def _build_rows_for_symbol(theme: str, symbol: str, tf: str, lookback_days: int) -> List[Dict]:
    df = _download_prices(symbol, lookback_days, tf)
    rows: List[Dict] = []

    def _to_float(v, default=float("nan")) -> float:
        try:
            # pandas may hand back a single-element Series when columns were MultiIndex
            if isinstance(v, pd.Series):
                v = v.iloc[0]
            if v is None:
                return float(default)
            return float(v)
        except Exception:
            return float(default)

    if df.empty:
        rows.append(
            {
                "date": "",
                "symbol": symbol,
                "theme": theme,
                "timeframe": tf,
                "status": "missing",
                "error_reason": "no_data",
            }
        )
        return rows

    for _, r in df.iterrows():
        # Prefer explicit date/datetime columns from reset_index(); fall back to the first column.
        date_val = None
        if "Date" in df.columns:
            date_val = r.get("Date")
        elif "Datetime" in df.columns:
            date_val = r.get("Datetime")
        elif "date" in df.columns:
            date_val = r.get("date")
        else:
            # last resort: assume first column is the timestamp column
            first_col = df.columns[0]
            date_val = r.get(first_col)

        try:
            date_iso = pd.to_datetime(date_val).date().isoformat()
        except Exception:
            # If we cannot parse, mark as missing-date rather than stamping 'today' for every row.
            date_iso = ""
        rows.append(
            {
                "date": date_iso,
                "symbol": symbol,
                "theme": theme,
                "timeframe": tf,
                "open": _to_float(r.get("Open")),
                "high": _to_float(r.get("High")),
                "low": _to_float(r.get("Low")),
                "close": _to_float(r.get("Close")),
                "volume": _to_float(r.get("Volume"), default=0.0),
                "status": "ok",
            }
        )
    return rows


def main(argv: Optional[Sequence[str]] = None) -> int:
    ap = argparse.ArgumentParser(description="Step2: download OHLCV per theme/symbol.")
    ap.add_argument("--themes", required=True, help="Comma-separated themes, e.g. XME,SMH,XBI")
    ap.add_argument("--out", required=True, help="Output directory (expects Step1 universe)")
    ap.add_argument("--contracts", default="./contracts", help="Contracts directory for JSON Schema")
    ap.add_argument("--lookback", default="260", help="Lookback days (default: 260)")
    ap.add_argument("--tf", default="1D", help="Timeframe (default: 1D)")
    ap.add_argument("--loglevel", default="INFO", help="Logging level")
    args = ap.parse_args(argv)

    logging.basicConfig(level=getattr(logging, args.loglevel.upper(), logging.INFO))

    themes = [t.strip().upper() for t in parse_csv_list(args.themes)]
    if not themes:
        raise SystemExit("Provide --themes")

    try:
        lookback_days = int(str(args.lookback).replace("d", ""))
    except Exception:
        lookback_days = 260

    tf = args.tf.upper()
    out_dir = ensure_dir(args.out)
    prices_dir = ensure_dir(out_dir / "step2_prices")

    _, rows_uni = load_universe(out_dir, themes=themes)
    sym_by_theme = group_symbols_by_theme(rows_uni)

    schema_path = Path(args.contracts) / "step2_prices.schema.json"

    for theme in themes:
        symbols = sym_by_theme.get(theme, [])
        if not symbols:
            LOG.warning("No symbols for theme=%s; skipping.", theme)
            continue

        records: List[Dict] = []
        for sym in symbols:
            records.extend(_build_rows_for_symbol(theme, sym, tf, lookback_days))

        if not records:
            LOG.warning("No price records for theme=%s; skipping outputs.", theme)
            continue

        payload = {
            "schema_version": "2.prices.v1",
            "generated_at_utc": now_utc_iso(),
            "timeframe": tf,
            "theme": theme,
            "source": "step2_prices.py",
            "notes": f"Prices downloaded via yfinance lookback={lookback_days}d tf={tf}",
            "rows": records,
        }

        payload = must_validate(schema_path, payload)

        json_path = prices_dir / f"prices_{theme}_{tf}.json"
        csv_path = prices_dir / f"prices_{theme}_{tf}.csv"
        json_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
        pd.DataFrame(records).to_csv(csv_path, index=False)

        LOG.info("Prices theme=%s rows=%d json=%s csv=%s", theme, len(records), json_path, csv_path)
        print(f"[OK] prices theme={theme} rows={len(records)} json={json_path} csv={csv_path}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
