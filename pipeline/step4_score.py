"""
Step4: Scoring / Ranking

Reads Step1 universe and Step3 env outputs, computes per-symbol scores, and writes
per-theme CSV + JSON (envelope) under out/step4_scores/.
"""
from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import pandas as pd
from jsonschema import Draft202012Validator

from .common import ensure_dir, now_utc_iso, parse_csv_list, validate_tf_combo
from .io_step1 import group_symbols_by_theme, load_universe

LOG = logging.getLogger(__name__)
TREND_STRONG_MIN = 0.65
VOL_MIN = 0.01
EXT_MAX = 0.08
QUALITY_MIN = 0.5

# Minimal inline schema for Step4 outputs (envelope + rows)
SCORE_SCHEMA = {
    "$schema": "https://json-schema.org/draft/2020-12/schema",
    "title": "Step4 Scores Output",
    "type": "object",
    "additionalProperties": False,
    "required": ["schema_version", "generated_at_utc", "theme", "htf_timeframe", "rows"],
    "properties": {
        # Keep permissive inline pattern to avoid schema failures when versioning changes.
        "schema_version": {"type": "string"},
        "generated_at_utc": {"type": "string"},
        "theme": {"type": "string"},
        "htf_timeframe": {"type": "string"},
        "source": {"type": "string"},
        "notes": {"type": "string"},
        "rows": {
            "type": "array",
            "minItems": 1,
            "items": {"$ref": "#/$defs/ScoreRow"},
        },
    },
    "$defs": {
        "ScoreRow": {
            "type": "object",
            "additionalProperties": True,
            "required": [
                "asof_utc",
                "theme",
                "symbol",
                "env_score",
                "signal_score",
                "score_total",
                "score_breakdown",
                "flags",
            ],
            "properties": {
                "asof_utc": {"type": "string"},
                "theme": {"type": "string"},
                "symbol": {"type": "string"},
                "env_score": {"type": "number"},
                "signal_score": {"type": "number"},
                "score_total": {"type": "number"},
                "score_breakdown": {"type": "object"},
                "score_breakdown_json": {"type": "string"},
                "env_bias": {"type": "string"},
                "env_confidence": {"type": "number"},
                "flags": {"type": "string"},
                "debug": {"type": "object"},
            },
        }
    },
}


def _clamp(v: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, v))


def _load_step3_env(theme: str, out_dir: Path, htf: str) -> Tuple[List[Dict], Dict[str, Dict]]:
    """
    Load Step3 env JSON envelope for a theme/timeframe.
    Returns (rows, map symbol->row)
    """
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
        if not sym:
            continue
        m[sym] = r
    return rows, m


def _load_step3_trend(theme: str, out_dir: Path, htf: str) -> Dict[str, Dict]:
    path = out_dir / "step3_trend" / f"trend_{theme}_{htf}.json"
    if not path.exists():
        return {}
    payload = json.loads(path.read_text(encoding="utf-8"))
    rows = payload.get("rows", [])
    m: Dict[str, Dict] = {}
    if isinstance(rows, list):
        for r in rows:
            sym = str(r.get("symbol", "")).strip().upper()
            if sym:
                m[sym] = r
    return m


def _load_step3_regime(theme: str, out_dir: Path, htf: str) -> Dict[str, Dict]:
    path = out_dir / "step3_regime" / f"regime_{theme}_{htf}.json"
    if not path.exists():
        return {}
    payload = json.loads(path.read_text(encoding="utf-8"))
    rows = payload.get("rows", [])
    m: Dict[str, Dict] = {}
    if isinstance(rows, list):
        for r in rows:
            sym = str(r.get("symbol", "")).strip().upper()
            if sym:
                m[sym] = r
    return m


def _load_etf_env(theme: str, out_dir: Path) -> Dict[str, object]:
    path = out_dir / "step3_etf_env" / f"etf_env_{theme}_1D.json"
    if not path.exists():
        return {}
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
        rows = payload.get("rows", [])
        return rows[0] if rows else {}
    except Exception:
        return {}


def _load_step2_prices(theme: str, out_dir: Path, tf: str) -> Dict[str, pd.DataFrame]:
    path = out_dir / "step2_prices" / f"prices_{theme}_{tf}.csv"
    if not path.exists():
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
    for col in ["open", "high", "low", "close", "volume"]:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")
    df = df.dropna(subset=["close", "high", "low"])
    price_map: Dict[str, pd.DataFrame] = {}
    for sym, g in df.groupby("symbol"):
        price_map[sym] = g
    return price_map


def _price_metrics(price_df: Optional[pd.DataFrame]) -> Dict[str, float]:
    base = {"vol_metric": 0.0, "extension_metric": 0.0, "price_quality": 0.0}
    if price_df is None or price_df.empty:
        return base

    tail = price_df.tail(60).copy()
    tail["close"] = pd.to_numeric(tail.get("close"), errors="coerce")
    tail["high"] = pd.to_numeric(tail.get("high"), errors="coerce")
    tail["low"] = pd.to_numeric(tail.get("low"), errors="coerce")
    tail = tail.dropna(subset=["close", "high", "low"])
    if tail.empty:
        return base

    close = tail["close"]
    high = tail["high"]
    low = tail["low"]

    price_quality = _clamp(len(close) / 60.0, 0.0, 1.0)

    tr: List[float] = []
    prev_close = None
    for hi, lo, cl in zip(high.tolist(), low.tolist(), close.tolist()):
        if prev_close is None:
            tr.append(hi - lo)
        else:
            tr.append(max(hi - lo, abs(hi - prev_close), abs(lo - prev_close)))
        prev_close = cl
    vol_metric = 0.0
    if tr and close.iloc[-1]:
        vol_metric = float(pd.Series(tr).tail(14).mean() / close.iloc[-1])

    sma = close.rolling(20, min_periods=5).mean().iloc[-1]
    extension_metric = abs(close.iloc[-1] - sma) / sma if sma else 0.0

    return {
        "vol_metric": float(vol_metric),
        "extension_metric": float(extension_metric),
        "price_quality": float(price_quality),
    }


def _env_score(env_row: Optional[Dict], etf_env: Dict) -> Tuple[float, Dict[str, float | str], List[str]]:
    flags: List[str] = []
    etf_bias = str(etf_env.get("etf_env_bias", "neutral")).lower() if etf_env else "neutral"
    etf_conf = float(etf_env.get("etf_env_confidence", 0.0)) if etf_env else 0.0
    env_bias = str(env_row.get("env_bias", "neutral")).lower() if env_row else "neutral"
    env_conf = float(env_row.get("env_confidence", 0.0)) if env_row else 0.0

    score = 60.0 * etf_conf
    bias_bonus = 10.0 if etf_bias == "bull" else (-10.0 if etf_bias == "bear" else 0.0)
    score += bias_bonus
    score += 30.0 * env_conf

    if not env_row:
        align_bonus = -10.0
        flags.append("missing_env")
    elif env_bias and etf_bias and env_bias != etf_bias and etf_bias != "neutral":
        align_bonus = -15.0
        flags.append("env_bias_mismatch")
    else:
        align_bonus = 5.0
    score += align_bonus

    breakdown: Dict[str, float | str] = {
        "etf_env_confidence": etf_conf,
        "etf_env_bias": etf_bias,
        "bias_bonus": bias_bonus,
        "env_confidence": env_conf,
        "env_bias": env_bias,
        "alignment_bonus": align_bonus,
    }
    return _clamp(score, 0.0, 100.0), breakdown, flags


def _signal_score(
    env_score_val: float,
    trend_state: str,
    trend_strength: float,
    metrics: Dict[str, float],
) -> Tuple[float, List[Dict[str, float | str]], List[str]]:
    score = env_score_val
    adjustments: List[Dict[str, float | str]] = []
    flags: List[str] = []

    strong_trend = trend_state in {"up", "down"} and trend_strength >= TREND_STRONG_MIN
    if strong_trend:
        adjustments.append({"reason": "trend_supports_entry", "delta": 10.0})
        score += 10.0
    else:
        adjustments.append(
            {"reason": "trend_not_strong_for_signal", "delta": -10.0, "trend_strength": trend_strength}
        )
        flags.append("trend_not_strong_for_signal")
        score -= 10.0

    vol_metric = float(metrics.get("vol_metric", 0.0))
    if vol_metric < VOL_MIN:
        adjustments.append({"reason": "vol_too_low", "delta": -15.0, "vol_metric": vol_metric})
        flags.append("vol_too_low")
        score -= 15.0

    extension_metric = float(metrics.get("extension_metric", 0.0))
    if extension_metric > EXT_MAX:
        adjustments.append({"reason": "extended_move", "delta": -12.0, "extension_metric": extension_metric})
        flags.append("extended_move")
        score -= 12.0

    price_quality = float(metrics.get("price_quality", 0.0))
    if price_quality < QUALITY_MIN:
        adjustments.append({"reason": "data_quality_low", "delta": -10.0, "price_quality": price_quality})
        flags.append("data_quality_low")
        score -= 10.0

    return _clamp(score, 0.0, 100.0), adjustments, flags


def _validate_payload(payload: Dict, schema_path: Optional[Path]) -> Dict:
    if schema_path and schema_path.exists():
        from .validate import must_validate

        return must_validate(schema_path, payload)

    # Fallback: validate against inline schema
    v = Draft202012Validator(SCORE_SCHEMA)
    errs = list(v.iter_errors(payload))
    if errs:
        lines = []
        for e in errs[:20]:
            p = "/".join([str(x) for x in e.absolute_path]) or "(root)"
            lines.append(f"path={p} msg={e.message}")
        more = "" if len(errs) <= 20 else f"\n... ({len(errs) - 20} more)"
        raise ValueError("Schema validation failed (inline SCORE_SCHEMA):\n" + "\n".join(lines) + more)
    return payload


def main(argv: Optional[Sequence[str]] = None) -> int:
    ap = argparse.ArgumentParser(description="Step4: scoring / ranking using Step1 universe and Step3 env.")
    ap.add_argument("--themes", required=True, help="Comma-separated themes, e.g. XME,SMH,XBI")
    ap.add_argument("--out", required=True, help="Output directory (expects Step1/Step3 artifacts)")
    ap.add_argument("--contracts", default="./contracts", help="Contracts directory (optional step4_scores.schema.json)")
    ap.add_argument("--htf", default="1H", help="HTF timeframe (matches Step3 env/trend/regime outputs), e.g. 1H")
    ap.add_argument("--loglevel", default="INFO", help="Logging level (DEBUG/INFO/WARN/ERROR)")
    args = ap.parse_args(argv)

    logging.basicConfig(level=getattr(logging, args.loglevel.upper(), logging.INFO))

    themes = [t.strip().upper() for t in parse_csv_list(args.themes)]
    if not themes:
        raise SystemExit("Provide at least one theme via --themes")

    htf = validate_tf_combo("step3", htf=args.htf)
    out_dir = ensure_dir(args.out)
    scores_dir = ensure_dir(out_dir / "step4_scores")

    _, universe_rows = load_universe(out_dir, themes=themes)
    sym_by_theme = group_symbols_by_theme(universe_rows)

    schema_path = Path(args.contracts) / "step4_scores.schema.json"
    if not schema_path.exists():
        LOG.warning("Contracts schema not found for Step4: %s (using inline schema)", schema_path)

    for theme in themes:
        symbols = sym_by_theme.get(theme, [])
        if not symbols:
            LOG.warning("No symbols for theme=%s from Step1; skipping.", theme)
            continue

        try:
            _, env_map = _load_step3_env(theme, out_dir, htf)
        except Exception as e:
            LOG.warning("Skipping theme=%s due to missing/invalid Step3 env: %s", theme, e)
            continue

        trend_map = _load_step3_trend(theme, out_dir, htf)
        regime_map = _load_step3_regime(theme, out_dir, htf)
        etf_env = _load_etf_env(theme, out_dir)
        price_map = _load_step2_prices(theme, out_dir, htf)

        records: List[Dict] = []
        for sym in symbols:
            env_row = env_map.get(sym)
            trend_row = trend_map.get(sym)
            regime_row = regime_map.get(sym)
            price_df = price_map.get(sym)

            env_score_val, env_breakdown, env_flags = _env_score(env_row, etf_env)
            env_bias = str(env_row.get("env_bias")) if env_row else "unknown"
            env_conf = float(env_row.get("env_confidence", 0.0)) if env_row else 0.0

            trend_state = str(trend_row.get("trend_state", "unknown")).lower() if trend_row else "unknown"
            trend_strength = float(trend_row.get("trend_strength", 0.0)) if trend_row else 0.0

            metrics = _price_metrics(price_df)
            signal_score_val, adjustments, signal_flags = _signal_score(env_score_val, trend_state, trend_strength, metrics)

            flags = list(dict.fromkeys(env_flags + signal_flags))

            regime_state = str(regime_row.get("regime_state", "unknown")).lower() if regime_row else "unknown"

            asof_candidates = [
                env_row.get("asof_utc") if env_row else None,
                trend_row.get("asof_utc") if trend_row else None,
                regime_row.get("asof_utc") if regime_row else None,
            ]
            if price_df is not None and not price_df.empty:
                try:
                    asof_candidates.append(pd.to_datetime(price_df["date"]).max().isoformat())
                except Exception:
                    pass
            asof = next((a for a in asof_candidates if a), now_utc_iso())

            breakdown = {
                "env_score": env_breakdown,
                "signal_score": {
                    "base_env_score": env_score_val,
                    "trend_state": trend_state,
                    "trend_strength": trend_strength,
                    "metrics": metrics,
                    "adjustments": adjustments,
                },
                "etf_env": {
                    "bias": str(etf_env.get("etf_env_bias", "")),
                    "confidence": float(etf_env.get("etf_env_confidence", 0.0)) if etf_env else 0.0,
                },
                "regime": {
                    "regime_state": regime_state,
                    "allowed_direction": regime_row.get("allowed_direction") if regime_row else None,
                    "position_multiplier": regime_row.get("position_multiplier") if regime_row else None,
                },
            }

            row = {
                "asof_utc": asof,
                "theme": theme,
                "symbol": sym,
                "env_score": float(env_score_val),
                "signal_score": float(signal_score_val),
                "score_total": float(signal_score_val),
                "score_breakdown": breakdown,
                "score_breakdown_json": json.dumps(breakdown, ensure_ascii=False),
                "env_bias": env_bias,
                "env_confidence": env_conf,
                "trend_state": trend_state,
                "trend_strength": trend_strength,
                "regime_state": regime_state,
                "etf_env_bias": str(etf_env.get("etf_env_bias", "")),
                "etf_env_confidence": float(etf_env.get("etf_env_confidence", 0.0)) if etf_env else 0.0,
                "flags": ",".join(flags) if flags else "",
                "notes": "",
                "debug": {
                    "env": env_row,
                    "trend": trend_row,
                    "regime": regime_row,
                    "etf_env": etf_env,
                    "metrics": metrics,
                    "adjustments": adjustments,
                },
            }
            records.append(row)

        if not records:
            LOG.warning("No records produced for theme=%s; skipping outputs.", theme)
            continue

        payload = {
            "schema_version": "4.scores.v1",
            "generated_at_utc": now_utc_iso(),
            "theme": theme,
            "htf_timeframe": htf,
            "source": "step4_score.py",
            "notes": f"Scores derived from Step3 env/trend/regime ({htf}) for theme={theme}",
            "rows": records,
        }

        payload = _validate_payload(payload, schema_path if schema_path.exists() else None)

        json_path = scores_dir / f"scores_{theme}.json"
        csv_path = scores_dir / f"scores_{theme}.csv"

        json_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
        df_out = pd.DataFrame(records)
        df_out = df_out.sort_values(["signal_score", "symbol"], ascending=[False, True]).reset_index(drop=True)
        df_out["rank"] = df_out.index + 1
        df_out.to_csv(csv_path, index=False)

        LOG.info(
            "Theme=%s scored rows=%d -> json=%s csv=%s",
            theme,
            len(records),
            json_path,
            csv_path,
        )
        print(f"[OK] theme={theme} json={json_path} csv={csv_path} rows={len(records)}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
