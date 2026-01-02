"""
Step3A: ETF Daily Env

Computes per-theme ETF env (daily) using Step2 prices and writes envelope JSON/CSV.
"""
from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import pandas as pd

from .common import ensure_dir, parse_csv_list
from .validate import must_validate

LOG = logging.getLogger(__name__)


def _clamp(v: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, v))


def _ema(series: pd.Series, span: int) -> pd.Series:
    return series.ewm(span=span, adjust=False).mean()


def _normalize(x: float, lo: float, hi: float) -> float:
    if hi <= lo:
        return 0.0
    return _clamp((x - lo) / (hi - lo), 0.0, 1.0)


def _derive_etf_env(close: pd.Series) -> Tuple[str, str, float, Dict[str, float]]:
    c = close.dropna()
    if c.empty:
        return "neutral", "flat", 0.2, {"reason": "no_data"}

    e20 = _ema(c, 20)
    e50 = _ema(c, 50)
    last = float(c.iloc[-1])
    last_e50 = float(e50.iloc[-1])
    last_e20 = float(e20.iloc[-1])
    slope = float(e50.iloc[-1] - e50.iloc[max(0, len(e50) - 5)])

    if last > last_e50 and last_e20 > last_e50 and slope > 0:
        env_bias = "bull"
        trend_state = "up"
    elif last < last_e50 and last_e20 < last_e50 and slope < 0:
        env_bias = "bear"
        trend_state = "down"
    else:
        env_bias = "neutral"
        trend_state = "flat"

    distance = abs(last - last_e50) / abs(last_e50) if last_e50 else 0.0
    slope_strength = abs(slope / last_e50) if last_e50 else 0.0

    env_conf = _clamp(
        0.4
        + _normalize(distance, 0.0, 0.05) * 0.3
        + _normalize(slope_strength, 0.0, 0.02) * 0.3,
        0.0,
        1.0,
    )
    debug = {
        "last": last,
        "ema20": last_e20,
        "ema50": last_e50,
        "distance": distance,
        "slope_strength": slope_strength,
    }
    return env_bias, trend_state, env_conf, debug


def _read_step2_prices(theme: str, out_dir: Path) -> pd.Series:
    path = out_dir / "step2_prices" / f"prices_{theme}_1D.csv"
    if not path.exists():
        raise FileNotFoundError(f"Missing Step2 prices for ETF (theme={theme}): {path}")
    df = pd.read_csv(path, low_memory=False)
    if df.empty:
        raise ValueError(f"Empty prices for theme={theme}")
    df["symbol"] = df["symbol"].astype(str).str.strip().str.upper()
    df = df[df["symbol"] == theme.upper()]
    if df.empty:
        raise ValueError(f"No price rows for ETF symbol={theme} in {path}")
    if "status" in df.columns:
        df = df[df["status"] == "ok"]
    df["date"] = pd.to_datetime(df["date"])
    df = df.dropna(subset=["date"])
    df = df.sort_values("date")
    df["close"] = pd.to_numeric(df.get("close"), errors="coerce")
    df = df.dropna(subset=["close"])
    if df.empty:
        raise ValueError(f"No usable close prices for theme={theme}")
    s = pd.Series(df["close"].astype(float).values, index=df["date"])
    return s[~s.index.duplicated(keep="last")]


def main(argv: Optional[Sequence[str]] = None) -> int:
    ap = argparse.ArgumentParser(description="Step3A: ETF daily env per theme using Step2 prices.")
    ap.add_argument("--themes", required=True, help="Comma-separated themes (e.g., XME,SMH,XBI)")
    ap.add_argument("--out", required=True, help="Output directory (expects Step2 prices 1D)")
    ap.add_argument("--contracts", default="./contracts", help="Contracts directory for JSON Schema")
    ap.add_argument("--loglevel", default="INFO", help="Logging level")
    args = ap.parse_args(argv)

    logging.basicConfig(level=getattr(logging, args.loglevel.upper(), logging.INFO))

    themes = [t.strip().upper() for t in parse_csv_list(args.themes)]
    if not themes:
        raise SystemExit("Provide --themes")

    out_dir = ensure_dir(args.out)
    env_dir = ensure_dir(out_dir / "step3_etf_env")

    schema_path = Path(args.contracts) / "step3_etf_env.schema.json"

    for theme in themes:
        try:
            close = _read_step2_prices(theme, out_dir)
        except Exception as e:
            LOG.warning("Skipping theme=%s due to missing/invalid ETF prices: %s", theme, e)
            continue

        env_bias, trend_state, env_conf, dbg = _derive_etf_env(close)
        asof_date = close.index.max().date().isoformat() if not close.empty else ""

        rows: List[Dict] = [
            {
                "asof_date": asof_date,
                "theme": theme,
                "etf_symbol": theme,
                "etf_env_bias": env_bias,
                "etf_env_confidence": float(env_conf),
                "etf_trend_state": trend_state,
                "etf_env_method": "etf_ema_daily",
                "debug": dbg,
            }
        ]

        payload = {
            "schema_version": "3.etf_env.v1",
            "generated_at_utc": pd.Timestamp.utcnow().strftime("%Y-%m-%dT%H:%M:%SZ"),
            "source": "step3_etf_env.py",
            "notes": "ETF daily env derived from Step2 prices (1D).",
            "rows": rows,
        }

        payload = must_validate(schema_path, payload)

        json_path = env_dir / f"etf_env_{theme}_1D.json"
        csv_path = env_dir / f"etf_env_{theme}_1D.csv"
        json_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
        pd.DataFrame(rows).to_csv(csv_path, index=False)

        LOG.info("ETF env theme=%s bias=%s conf=%.2f json=%s csv=%s", theme, env_bias, env_conf, json_path, csv_path)
        print(f"[OK] etf_env theme={theme} bias={env_bias} conf={env_conf:.2f} json={json_path} csv={csv_path}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
