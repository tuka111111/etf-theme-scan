from __future__ import annotations

import argparse
import json
import logging
import math
import re
from pathlib import Path
from typing import Dict, List, Optional, Sequence

import pandas as pd

from .common import ensure_dir, parse_csv_list, validate_tf_combo
from .io_step1 import load_universe, group_symbols_by_theme
from .validate import must_validate

LOG = logging.getLogger(__name__)

ALLOWED_HTF = {"1D", "4H", "1H", "30m", "15m"}
SYMBOL_PATTERN = re.compile(r"^[A-Z0-9][A-Z0-9.\-]{0,14}$")


def _read_step2_env(out_dir: Path) -> Dict[str, Dict]:
    """
    Read Step2 env outputs if available.
    Returns a map: THEME -> record dict.
    """
    candidates = [out_dir / "step2_env.json", out_dir / "step2" / "step2_env.json"]
    for p in candidates:
        if not p.exists():
            continue
        try:
            payload = json.loads(p.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            LOG.warning("Failed to parse %s; ignoring", p)
            continue
        if not isinstance(payload, list):
            LOG.warning("Expected list payload in %s; ignoring", p)
            continue

        m: Dict[str, Dict] = {}
        for item in payload:
            if not isinstance(item, dict):
                continue
            theme = str(item.get("theme", "")).strip().upper()
            if theme:
                m[theme] = item
        return m

    return {}


def _read_step2_prices(theme: str, out_dir: Path, tf: str) -> Dict[str, pd.Series]:
    path = out_dir / "step2_prices" / f"prices_{theme}_{tf}.csv"
    if not path.exists():
        LOG.warning("Missing Step2 prices for theme=%s tf=%s: %s", theme, tf, path)
        return {}
    df = pd.read_csv(path, low_memory=False)
    if df.empty:
        return {}
    df["symbol"] = df["symbol"].astype(str).str.strip().str.upper()
    if "status" in df.columns:
        df = df[df["status"] == "ok"]
    df["date"] = pd.to_datetime(df["date"])
    df = df.dropna(subset=["date"])
    df = df.sort_values(["symbol", "date"])
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


def _symbol_confidence(bias: str, close: Optional[pd.Series]) -> Tuple[float, Dict[str, object]]:
    if close is None or close.dropna().shape[0] < 30:
        return 0.3, {"reason": "missing_or_short"}

    c = close.dropna()
    e20 = _ema(c, 20)
    e50 = _ema(c, 50)

    last = float(c.iloc[-1])
    last_e20 = float(e20.iloc[-1])
    last_e50 = float(e50.iloc[-1])

    if bias == "bull":
        align = 1.0 if (last > last_e50 and last_e20 > last_e50) else 0.3
    elif bias == "bear":
        align = 1.0 if (last < last_e50 and last_e20 < last_e50) else 0.3
    else:
        align = 0.5

    def _slope(x: pd.Series, n: int = 10) -> float:
        if len(x) <= n:
            return 0.0
        return float((x.iloc[-1] / x.iloc[-1 - n]) - 1.0)

    slope20 = _slope(e20, 10)
    strength = max(0.0, min(1.0, 0.5 + slope20))
    quality = max(0.0, min(1.0, len(c) / 200.0))

    conf = 0.4 * align + 0.4 * strength + 0.2 * quality
    return max(0.0, min(1.0, conf)), {
        "align": align,
        "strength": strength,
        "quality": quality,
        "bars": len(c),
    }


def _clamp(v: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, v))


def _normalize_asof(val: Optional[str]) -> str:
    return _to_jst_iso(val)


def _to_jst_iso(val: Optional[str]) -> str:
    """
    Normalize a datetime string to JST ISO8601 with offset (+09:00).
    Fallback to current time if parsing fails.
    """
    try:
        if val:
            ts = pd.to_datetime(val)
        else:
            ts = pd.Timestamp.now(tz="UTC")
        if ts.tzinfo is None:
            ts = ts.tz_localize("UTC")
        else:
            ts = ts.tz_convert("UTC")
        if pd.isna(ts):
            ts = pd.Timestamp.now(tz="UTC")
    except Exception:
        ts = pd.Timestamp.now(tz="UTC")
    ts_jst = ts.tz_convert("Asia/Tokyo")
    return ts_jst.isoformat(timespec="seconds")


def _derive_bias(details: Dict[str, object]) -> str:
    try:
        bull = float(details.get("bull", 0.0))
        bear = float(details.get("bear", 0.0))
    except (TypeError, ValueError):
        bull = 0.0
        bear = 0.0

    if bull > 0 and bear <= 0:
        return "bull"
    if bear > 0 and bull <= 0:
        return "bear"
    return "neutral"


def _derive_score(details: Dict[str, object], bias: str) -> float:
    slopes: List[float] = []
    for k in ("slope_mid_20", "slope_slow_50"):
        try:
            v = float(details.get(k, 0.0))
        except (TypeError, ValueError):
            continue
        if math.isfinite(v):
            slopes.append(v * 100.0)

    score = sum(slopes) / len(slopes) if slopes else 0.0
    if bias == "bull":
        score = max(score, 25.0)
    elif bias == "bear":
        score = min(score, -25.0)
    return _clamp(score, -100.0, 100.0)


def _derive_confidence(theme_env: Optional[Dict[str, object]], bias: str) -> float:
    if not theme_env:
        return 0.4

    env = str(theme_env.get("env", "")).upper()
    allowed = bool(theme_env.get("allowed"))

    base = 0.6
    if env == "TREND" or allowed:
        base = 0.75
    elif env in {"RANGE", "TRANSITION"}:
        base = 0.55

    if bias == "neutral":
        base -= 0.1

    return _clamp(base, 0.0, 1.0)


def build_env_rows(
    theme: str,
    symbols: List[str],
    step2_env_map: Dict[str, Dict],
    htf_timeframe: str,
    prices: Dict[str, pd.Series],
) -> List[Dict[str, object]]:
    rows: List[Dict[str, object]] = []

    env_src = step2_env_map.get(theme, {})
    details = env_src.get("details") if isinstance(env_src.get("details"), dict) else {}

    asof = _normalize_asof(env_src.get("asof"))
    bias = _derive_bias(details)
    score = _derive_score(details, bias)
    conf_theme = _derive_confidence(env_src, bias)

    for sym in symbols:
        sym_conf, conf_debug = _symbol_confidence(bias, prices.get(sym))
        rows.append(
            {
                "asof_utc": asof,
                "symbol": sym,
                "theme": theme,
                "htf_timeframe": htf_timeframe,
                "env_bias": bias,
                "env_score": float(score),
                "env_confidence": float(sym_conf),
                "env_confidence_theme": float(conf_theme),
                "env_method": "ema_slope",
                "inputs": {
                    "ema_fast": 20,
                    "ema_slow": 50,
                    "lookback_bars": 200,
                },
                "debug": {
                    "source_theme_env": env_src,
                    "confidence": conf_debug,
                },
            }
        )

    return rows


def _normalize_timeframe(tf: str) -> str:
    raw = tf.strip()
    if raw.upper() in {"1D", "4H", "1H"}:
        return raw.upper()
    if raw.lower() in {"30m", "15m"}:
        return raw.lower()
    return raw


def main(argv: Optional[Sequence[str]] = None) -> int:
    ap = argparse.ArgumentParser(description="Step3: symbol-level HTF env seeded from Step2 outputs.")
    ap.add_argument("--themes", required=True, help="Comma-separated themes, e.g. XME,SMH,XBI")
    ap.add_argument("--out", required=True, help="Output directory (expects Step1/Step2 artifacts)")
    ap.add_argument("--contracts", default="./contracts", help="Contracts directory for JSON Schemas")
    ap.add_argument("--htf", default="1H", help="HTF timeframe label (one of 1D,4H,1H,30m,15m)")
    ap.add_argument("--loglevel", default="INFO", help="Logging level (DEBUG/INFO/WARN/ERROR)")
    args = ap.parse_args(argv)

    logging.basicConfig(level=getattr(logging, args.loglevel.upper(), logging.INFO))

    themes = [t.strip().upper() for t in parse_csv_list(args.themes)]
    if not themes:
        raise SystemExit("Provide at least one theme via --themes")

    out_dir = ensure_dir(args.out)
    env_dir = ensure_dir(out_dir / "step3_env")
    htf_timeframe = validate_tf_combo("step3", htf=args.htf)

    _, rows = load_universe(out_dir, themes=themes)
    sym_by_theme = group_symbols_by_theme(rows)

    step2_env_map = _read_step2_env(out_dir)
    if not step2_env_map:
        LOG.warning("Step2 env outputs not found; using neutral defaults for env bias/score.")
    prices_by_theme: Dict[str, Dict[str, pd.Series]] = {}
    for theme in themes:
        try:
            prices_by_theme[theme] = _read_step2_prices(theme, out_dir, htf_timeframe)
        except Exception as e:
            LOG.warning("No prices for theme=%s tf=%s: %s", theme, htf_timeframe, e)
            prices_by_theme[theme] = {}

    schema_path = Path(args.contracts) / "step3_env.schema.json"

    for theme in themes:
        raw_symbols = sym_by_theme.get(theme, [])
        symbols = [s for s in raw_symbols if SYMBOL_PATTERN.match(s)]
        if len(symbols) != len(raw_symbols):
            LOG.warning(
                "Filtered symbols for theme=%s due to pattern mismatch. before=%d after=%d",
                theme,
                len(raw_symbols),
                len(symbols),
            )
        if not symbols:
            LOG.warning("No symbols found for theme=%s (after validation); skipping output.", theme)
            continue

        records = build_env_rows(theme, symbols, step2_env_map, htf_timeframe, prices_by_theme.get(theme, {}))

        payload = {
            "schema_version": "3.env.v1",
            "generated_at_utc": _to_jst_iso(None),
            "source": "step3_env.py",
            "notes": f"HTF env ({htf_timeframe}) derived from Step2 env and Step1 universe for theme={theme}.",
            "rows": records,
        }

        payload = must_validate(schema_path, payload)

        json_path = env_dir / f"step3_env_{theme}_{htf_timeframe}.json"
        csv_path = env_dir / f"step3_env_{theme}_{htf_timeframe}.csv"
        json_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
        pd.DataFrame(records).to_csv(csv_path, index=False)

        LOG.info("Theme=%s symbols=%d -> json=%s csv=%s", theme, len(records), json_path, csv_path)
        print(f"[OK] theme={theme} json={json_path} csv={csv_path} rows={len(records)}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
