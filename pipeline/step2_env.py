# pipeline/step2_env.py
from __future__ import annotations

import argparse
import json
import logging
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Dict, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
import yfinance as yf

# IMPORTANT:
# - DO NOT modify theme_scan_core.py
# - Step2 computes HTF environment (trend/range/transition) for each theme (or for each benchmark).
# - Keep logic minimal and explicit; put TODO where you expect changes.

LOG = logging.getLogger(__name__)


@dataclass(frozen=True)
class EnvResult:
    theme: str
    benchmark: str
    timeframe: str  # e.g. "1D" for HTF
    asof: str
    env: str        # "TREND" | "RANGE" | "TRANSITION" | "UNKNOWN"
    allowed: bool
    details: Dict[str, float]


def _utc_now_iso() -> str:
    return pd.Timestamp.utcnow().strftime("%Y-%m-%d %H:%M:%SZ")


def _ema(series: pd.Series, span: int) -> pd.Series:
    return series.ewm(span=span, adjust=False).mean()


def _download_close(symbol: str, period_days: int = 400) -> pd.Series:
    """
    Download daily close for environment detection.
    NOTE: Step2 is allowed to download prices (unlike Step1).
    """
    df = yf.download(
        tickers=symbol,
        period=f"{period_days}d",
        interval="1d",
        auto_adjust=True,
        progress=False,
    )
    if df is None or df.empty:
        return pd.Series(dtype=float)
    if "Close" not in df.columns:
        return pd.Series(dtype=float)
    s = df["Close"].dropna().astype(float)
    s.name = symbol
    return s


def classify_env_htf(
    close: pd.Series,
    *,
    ema_fast: int = 20,
    ema_mid: int = 50,
    ema_slow: int = 200,
    adx_like_window: int = 14,
    range_band_pct: float = 0.03,
) -> Tuple[str, bool, Dict[str, float]]:
    """
    Minimal HTF (daily) environment classifier.

    Heuristic:
      - TREND: EMA alignment (fast > mid > slow) and price above slow (bull)
               or (fast < mid < slow) and price below slow (bear)
      - RANGE: price within +/- range_band_pct around EMA_mid and weak slope
      - TRANSITION: otherwise

    TODO (likely changes later):
      - Add higher quality trend strength (ADX) using OHLC instead of close-only.
      - Replace range_band_pct with ATR-based band (more robust).
      - Add volatility contraction/expansion flags.
    """
    if close is None or close.dropna().shape[0] < max(ema_slow + 5, 220):
        return "UNKNOWN", False, {"reason": 1.0}

    c = close.dropna()
    e20 = _ema(c, ema_fast)
    e50 = _ema(c, ema_mid)
    e200 = _ema(c, ema_slow)

    last = float(c.iloc[-1])
    last_e20 = float(e20.iloc[-1])
    last_e50 = float(e50.iloc[-1])
    last_e200 = float(e200.iloc[-1])

    # slopes (normalized)
    def _slope(x: pd.Series, n: int = 20) -> float:
        if len(x) <= n:
            return float("nan")
        return float((x.iloc[-1] / x.iloc[-1 - n]) - 1.0)

    slope_e50 = _slope(e50, 20)
    slope_e200 = _slope(e200, 50)

    # TREND (bull/bear)
    bull = (last_e20 > last_e50 > last_e200) and (last > last_e200)
    bear = (last_e20 < last_e50 < last_e200) and (last < last_e200)

    # RANGE (mid-band + low slope)
    band_low = last_e50 * (1.0 - range_band_pct)
    band_high = last_e50 * (1.0 + range_band_pct)
    in_band = band_low <= last <= band_high
    low_slope = (abs(slope_e50) < 0.01) and (abs(slope_e200) < 0.02)  # heuristic thresholds

    if bull or bear:
        env = "TREND"
        allowed = True
    elif in_band and low_slope:
        env = "RANGE"
        allowed = False  # default: block LTF aggressiveness in range
    else:
        env = "TRANSITION"
        allowed = False  # conservative default

    details = {
        "last": last,
        "ema_fast": last_e20,
        "ema_mid": last_e50,
        "ema_slow": last_e200,
        "slope_mid_20": float(slope_e50) if np.isfinite(slope_e50) else np.nan,
        "slope_slow_50": float(slope_e200) if np.isfinite(slope_e200) else np.nan,
        "in_band": float(1.0 if in_band else 0.0),
        "bull": float(1.0 if bull else 0.0),
        "bear": float(1.0 if bear else 0.0),
    }
    return env, allowed, details


def run_step2_for_theme(
    theme: str,
    *,
    benchmark: Optional[str] = None,
    timeframe: str = "1D",
) -> EnvResult:
    theme_u = str(theme).strip().upper()
    bench = (benchmark or theme_u).strip().upper()

    close = _download_close(bench, period_days=420)
    env, allowed, details = classify_env_htf(close)

    return EnvResult(
        theme=theme_u,
        benchmark=bench,
        timeframe=timeframe,
        asof=_utc_now_iso(),
        env=env,
        allowed=allowed,
        details=details,
    )


def write_step2_outputs(out_dir: Path, results: Sequence[EnvResult]) -> Dict[str, str]:
    """
    Outputs:
      - out/step2_env.json
      - out/step2_env.csv
    """
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    json_path = out_dir / "step2_env.json"
    json_path.write_text(json.dumps([asdict(r) for r in results], indent=2, ensure_ascii=False), encoding="utf-8")

    # Flatten for CSV
    rows = []
    for r in results:
        row = {
            "Theme": r.theme,
            "Benchmark": r.benchmark,
            "Timeframe": r.timeframe,
            "AsOfUTC": r.asof,
            "Env": r.env,
            "Allowed": r.allowed,
        }
        for k, v in r.details.items():
            row[f"detail_{k}"] = v
        rows.append(row)

    csv_path = out_dir / "step2_env.csv"
    pd.DataFrame(rows).to_csv(csv_path, index=False)

    return {"json": str(json_path), "csv": str(csv_path)}


def main(argv: Optional[Sequence[str]] = None) -> int:
    ap = argparse.ArgumentParser(description="Step2: HTF environment classifier (default: daily/1D).")
    ap.add_argument("--themes", required=True, help="Comma-separated themes, e.g. XME,SMH,XBI")
    ap.add_argument("--out", required=True, help="Output directory")
    ap.add_argument("--timeframe", default="1D", help='HTF timeframe label (default: "1D")')
    ap.add_argument("--loglevel", default="INFO", help="DEBUG/INFO/WARN/ERROR")
    args = ap.parse_args(argv)

    logging.basicConfig(level=getattr(logging, args.loglevel.upper(), logging.INFO))
    out_dir = Path(args.out)

    themes = [t.strip().upper() for t in args.themes.replace(" ", ",").split(",") if t.strip()]
    themes = list(dict.fromkeys(themes))

    results = []
    for t in themes:
        LOG.info("Classifying HTF env for theme=%s benchmark=%s timeframe=%s", t, t, args.timeframe)
        r = run_step2_for_theme(t, benchmark=t, timeframe=args.timeframe)
        results.append(r)
        LOG.info("Env=%s Allowed=%s", r.env, r.allowed)

    paths = write_step2_outputs(out_dir, results)
    LOG.info("Wrote step2 outputs: %s", paths)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())