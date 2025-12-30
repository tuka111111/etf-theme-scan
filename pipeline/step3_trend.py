"""
Step3 Trend (HTF only)

Computes HTF trend state/strength per symbol using Step2 prices (HTF) with an EMA stack.
"""
from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import pandas as pd

from .common import ensure_dir, parse_csv_list, validate_tf_combo
from .io_step1 import group_symbols_by_theme, load_universe
from .validate import must_validate

LOG = logging.getLogger(__name__)


def _read_env(theme: str, out_dir: Path, htf: str) -> Dict[str, Dict]:
    path = out_dir / "step3_env" / f"step3_env_{theme}_{htf}.json"
    if not path.exists():
        raise FileNotFoundError(f"Missing Step3 env JSON for theme={theme} htf={htf}: {path}")
    payload = json.loads(path.read_text(encoding="utf-8"))
    rows = payload.get("rows", [])
    if not isinstance(rows, list) or not rows:
        raise ValueError(f"Invalid or empty Step3 env rows for theme={theme} file={path}")
    m: Dict[str, Dict] = {}
    for r in rows:
        sym = str(r.get("symbol", "")).strip().upper()
        if sym:
            m[sym] = r
    return m


def _read_step2_prices(theme: str, out_dir: Path, tf: str) -> Dict[str, pd.Series]:
    """
    Load Step2 prices CSV and return close series per symbol.
    """
    path = out_dir / "step2_prices" / f"prices_{theme}_{tf}.csv"
    if not path.exists():
        raise FileNotFoundError(f"Missing Step2 prices CSV for theme={theme} tf={tf}: {path}")
    df = pd.read_csv(path, low_memory=False)
    if df.empty:
        return {}
    df["symbol"] = df["symbol"].astype(str).str.strip().str.upper()
    if "status" in df.columns:
        df = df[df["status"] == "ok"]
    # ensure date sorted and unique per symbol
    df["date"] = pd.to_datetime(df["date"])
    df = df.dropna(subset=["date"])
    df = df.sort_values(["symbol", "date"])
    # enforce numeric close; drop rows that cannot be parsed
    df["close"] = pd.to_numeric(df.get("close"), errors="coerce")
    df = df.dropna(subset=["close"])
    out: Dict[str, pd.Series] = {}
    for sym, g in df.groupby("symbol"):
        s = pd.Series(g["close"].astype(float).values, index=g["date"])
        s = s[~s.index.duplicated(keep="last")]
        if not s.empty:
            out[sym] = s
    return out


def _ema(series: pd.Series, span: int) -> pd.Series:
    return series.ewm(span=span, adjust=False).mean()


def _trend_from_price(close: pd.Series) -> Tuple[str, float]:
    if close is None or close.dropna().shape[0] < 50:
        return "range", 0.0
    c = close.dropna()
    e20 = _ema(c, 20)
    e50 = _ema(c, 50)

    last = float(c.iloc[-1])
    last_e20 = float(e20.iloc[-1])
    last_e50 = float(e50.iloc[-1])

    if last_e20 > last_e50 and last > last_e50:
        state = "up"
    elif last_e20 < last_e50 and last < last_e50:
        state = "down"
    else:
        state = "range"

    # slope-based strength (normalized)
    def _slope(x: pd.Series, n: int = 10) -> float:
        if len(x) <= n:
            return 0.0
        return float((x.iloc[-1] / x.iloc[-1 - n]) - 1.0)

    slope20 = _slope(e20, 10)
    strength = max(0.0, min(1.0, 0.5 + slope20))
    return state, float(strength)


def _rows_for_theme(theme: str, symbols: List[str], env_map: Dict[str, Dict], prices: Dict[str, pd.Series], htf: str) -> List[Dict]:
    rows: List[Dict] = []
    for sym in symbols:
        close = prices.get(sym)
        if close is None:
            trend_state = "range"
            trend_strength = 0.0
            trend_method = "htf_env_proxy"
            debug = {"reason": "missing_price"}
            asof = env_map.get(sym, {}).get("asof_utc")
        else:
            trend_state, trend_strength = _trend_from_price(close)
            trend_method = "ema_stack"
            debug = {"bars": len(close)}
            asof = close.index.max().strftime("%Y-%m-%dT%H:%M:%SZ")
        rows.append(
            {
                "asof_utc": asof or pd.Timestamp.utcnow().strftime("%Y-%m-%dT%H:%M:%SZ"),
                "theme": theme,
                "symbol": sym,
                "htf_timeframe": htf,
                "trend_state": trend_state,
                "trend_strength": trend_strength,
                "trend_method": trend_method,
                "features": {},
                "debug": debug,
                "debug_json": json.dumps(debug, ensure_ascii=False),
            }
        )
    return rows


def main(argv: Optional[Sequence[str]] = None) -> int:
    ap = argparse.ArgumentParser(description="Step3 Trend (HTF-only) based on Step3 Env proxies.")
    ap.add_argument("--themes", required=True, help="Comma-separated themes, e.g. XME,SMH,XBI")
    ap.add_argument("--out", required=True, help="Output directory (expects Step1/Step3 env artifacts)")
    ap.add_argument("--contracts", default="./contracts", help="Contracts directory for JSON Schema")
    ap.add_argument("--htf", default="1H", help="HTF timeframe label, e.g. 1H/4H/1D")
    ap.add_argument("--loglevel", default="INFO", help="Logging level")
    args = ap.parse_args(argv)

    logging.basicConfig(level=getattr(logging, args.loglevel.upper(), logging.INFO))

    themes = [t.strip().upper() for t in parse_csv_list(args.themes)]
    if not themes:
        raise SystemExit("Provide --themes")

    out_dir = ensure_dir(args.out)
    trend_dir = ensure_dir(out_dir / "step3_trend")
    htf = validate_tf_combo("step3", htf=args.htf)

    _, rows_uni = load_universe(out_dir, themes=themes)
    sym_by_theme = group_symbols_by_theme(rows_uni)

    schema_path = Path(args.contracts) / "step3_trend.schema.json"

    for theme in themes:
        symbols = sym_by_theme.get(theme, [])
        if not symbols:
            LOG.warning("No symbols for theme=%s; skipping.", theme)
            continue

        try:
            env_map = _read_env(theme, out_dir, htf)
        except Exception as e:
            LOG.warning("Missing env for theme=%s htf=%s: %s", theme, htf, e)
            continue

        try:
            prices = _read_step2_prices(theme, out_dir, htf)
        except Exception as e:
            LOG.warning("Missing prices for theme=%s tf=%s: %s", theme, htf, e)
            prices = {}

        rows = _rows_for_theme(theme, symbols, env_map, prices, htf)

        payload = {
            "schema_version": "3.trend.v1",
            "generated_at_utc": pd.Timestamp.utcnow().strftime("%Y-%m-%dT%H:%M:%SZ"),
            "htf_timeframe": htf,
            "theme": theme,
            "source": "step3_trend.py",
            "notes": f"HTF trend derived from Step2 prices for theme={theme}",
            "rows": rows,
        }

        payload = must_validate(schema_path, payload)

        json_path = trend_dir / f"trend_{theme}_{htf}.json"
        csv_path = trend_dir / f"trend_{theme}_{htf}.csv"

        json_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
        pd.DataFrame(rows).to_csv(csv_path, index=False)

        LOG.info("Trend theme=%s rows=%d json=%s csv=%s", theme, len(rows), json_path, csv_path)
        print(f"[OK] trend theme={theme} json={json_path} csv={csv_path} rows={len(rows)}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
