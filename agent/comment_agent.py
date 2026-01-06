from __future__ import annotations

import argparse
import json
import logging
import os
from datetime import datetime, timezone, timedelta
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

import pandas as pd
from dotenv import load_dotenv

from .prompts import SYSTEM_PROMPT, PROMPT_VERSION

LOG = logging.getLogger(__name__)
BASE_DIR = Path(__file__).resolve().parent.parent


def _jst_now_iso() -> str:
    return (datetime.now(timezone.utc) + timedelta(hours=9)).isoformat(timespec="seconds")


def _read_input(path: Path) -> pd.DataFrame:
    if path.suffix.lower() == ".json":
        return pd.read_json(path)
    return pd.read_csv(path)


def _detect_score_column(df: pd.DataFrame) -> str:
    for cand in ["score_total", "score", "total_score"]:
        if cand in df.columns:
            return cand
    for c in df.columns:
        if pd.api.types.is_numeric_dtype(df[c]):
            return c
    return df.columns[0]


def _normalize(df: pd.DataFrame) -> pd.DataFrame:
    for c in ["score_total", "env_confidence", "rank", "etf_env_confidence"]:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")
    for c in ["theme", "symbol", "env_bias", "flags", "etf_env_bias", "role", "trend_state", "trend_direction"]:
        if c in df.columns:
            df[c] = df[c].astype(str)
    return df


def _extract_asof(df: pd.DataFrame) -> Dict[str, str]:
    asof_utc = "unknown"
    asof_local = "unknown"
    for col in ["asof_utc", "generated_at_utc"]:
        if col in df.columns and not df[col].isna().all():
            asof_utc = str(df[col].dropna().iloc[0])
            break
    if asof_utc != "unknown":
        try:
            ts = pd.to_datetime(asof_utc, utc=True)
            asof_local = (ts.tz_convert("Asia/Tokyo")).isoformat()
        except Exception:
            asof_local = "unknown"
    return {"asof_utc": asof_utc, "asof_local": asof_local}


def _build_payload(df: pd.DataFrame, max_symbols: int, source_path: Optional[Path] = None) -> Dict[str, Any]:
    df = _normalize(df.copy())
    score_col = _detect_score_column(df)
    asof = _extract_asof(df)

    themes = sorted([t for t in df["theme"].dropna().unique().tolist() if t])

    env_src = df[df.get("role") == "etf"]
    if env_src.empty:
        env_src = df
    theme_env: List[Dict[str, Any]] = []
    for _, row in env_src.groupby("theme").first().reset_index().iterrows():
        theme_env.append(
            {
                "theme": row.get("theme", "unknown"),
                "env": row.get("etf_env_bias") or row.get("env_bias") or "unknown",
                "score": _safe_num(row.get("etf_env_confidence")) or _safe_num(row.get("env_confidence")) or "unknown",
                "flags": _normalize_flags(row.get("flags")),
            }
        )

    non_etf = df[df.get("role") != "etf"]
    non_etf = non_etf.sort_values([score_col, "symbol"], ascending=[False, True])
    symbols_top: List[Dict[str, Any]] = []
    for _, row in non_etf.head(max_symbols).iterrows():
        symbols_top.append(
            {
                "symbol": row.get("symbol", "unknown"),
                "theme": row.get("theme", "unknown"),
                "score_total": _safe_num(row.get(score_col)),
                "env": row.get("env_bias", "unknown"),
                "trend": row.get("trend_state") or row.get("trend_direction") or "unknown",
                "flags": _normalize_flags(row.get("flags")),
            }
        )

    flags_series = df["flags"].fillna("").str.split(",").explode().str.strip() if "flags" in df.columns else pd.Series([], dtype=str)
    warnings = []
    if (flags_series == "").all():
        warnings.append("none")
    else:
        counts = flags_series[flags_series != ""].value_counts().head(5)
        if counts.empty:
            warnings.append("none")
        else:
            for k, v in counts.items():
                try:
                    total = int(len(non_etf))
                except Exception:
                    total = 0
                pct = f" ({v/total:.0%})" if total > 0 else ""
                warnings.append(f"{k}: {v}{pct}")

    debug = [
        f"prompt_version={PROMPT_VERSION}",
        f"score_col={score_col}",
        f"themes={len(themes)} symbols={len(non_etf)}",
        f"asof_utc={asof['asof_utc']}",
        f"asof_local={asof['asof_local']}",
        f"source={source_path}" if source_path else "",
    ]

    payload = {
        "asof_local": asof["asof_local"],
        "asof_utc": asof["asof_utc"],
        "themes": themes if themes else ["unknown"],
        "symbols_count": len(non_etf) if len(non_etf) else "unknown",
        "theme_env": theme_env,
        "symbols_top": symbols_top,
        "warnings": warnings,
        "debug": [d for d in debug if d],
    }
    return payload


def _safe_num(val: Any) -> Any:
    try:
        if val is None or (isinstance(val, float) and pd.isna(val)):
            return "unknown"
        v = float(val)
        if pd.isna(v):
            return "unknown"
        return round(v, 2)
    except Exception:
        return "unknown"


def _normalize_flags(x: Any) -> str:
    if x is None:
        return "none"
    try:
        import pandas as pd  # type: ignore

        if pd.isna(x):  # type: ignore
            return "none"
    except Exception:
        pass
    try:
        if isinstance(x, float) and x != x:
            return "none"
    except Exception:
        pass
    if isinstance(x, (list, tuple, set)):
        items = [str(i).strip() for i in x if i is not None and str(i).strip()]
        return ";".join(items) if items else "none"
    s = str(x).strip()
    return s if s else "none"


FLAG_DEFINITIONS = {
    "regime_range": "Regime judged as range; avoid breakout bias.",
    "trend_misaligned": "Trend and env bias disagree.",
    "env_low_confidence": "Env confidence is below threshold.",
    "regime_transition": "Regime is in transition; unstable.",
}


def _build_user_message(payload: Dict[str, Any]) -> str:
    payload = dict(payload)
    payload["flag_definitions"] = FLAG_DEFINITIONS
    return json.dumps(payload, ensure_ascii=False)


def _render_fallback(payload: Dict[str, Any]) -> str:
    lines = []
    lines.append("# Daily Dashboard Comment")
    lines.append("## Snapshot")
    lines.append(f"- asof_local: {payload.get('asof_local','unknown')}")
    lines.append(f"- asof_utc: {payload.get('asof_utc','unknown')}")
    lines.append(f"- themes: {', '.join(payload.get('themes', ['unknown'])) if payload.get('themes') else 'unknown'}")
    lines.append(f"- symbols: {payload.get('symbols_count','unknown')}")
    lines.append("")
    lines.append("## ETF Daily Env (per theme)")
    env_rows = payload.get("theme_env") or []
    if not env_rows:
        lines.append("- unknown")
    else:
        for row in env_rows:
            lines.append(
                f"- {row.get('theme','unknown')}: env={row.get('env','unknown')} score={row.get('score','unknown')} flags={row.get('flags','unknown')}"
            )
    lines.append("")
    lines.append("## Symbols (top)")
    sym_rows = payload.get("symbols_top") or []
    if not sym_rows:
        lines.append("- unknown")
    else:
        for row in sym_rows:
            lines.append(
                f"- {row.get('symbol','unknown')} ({row.get('theme','unknown')}) score_total={row.get('score_total','unknown')} env={row.get('env','unknown')} trend={row.get('trend','unknown')} flags={row.get('flags','unknown')}"
            )
    lines.append("")
    lines.append("## Warnings")
    warnings = payload.get("warnings") or []
    if not warnings:
        lines.append("- none")
    else:
        for w in warnings:
            lines.append(f"- {w}")
    lines.append("")
    lines.append("## Debug (short)")
    dbg = payload.get("debug") or []
    if not dbg:
        lines.append("- none")
    else:
        for d in dbg[:3]:
            lines.append(f"- {d}")
    return "\n".join(lines)


def _validate_output(md_text: str) -> bool:
    required = [
        "# Daily Dashboard Comment",
        "## Snapshot",
        "## ETF Daily Env (per theme)",
        "## Symbols (top)",
        "## Warnings",
        "## Debug (short)",
    ]
    if any(h not in md_text for h in required):
        return False
    if "|" in md_text:
        return False
    if "Unknown" in md_text:
        return False
    return True


def main(argv: Optional[Sequence[str]] = None) -> int:
    if load_dotenv:
        load_dotenv(BASE_DIR / ".env")

    ap = argparse.ArgumentParser(description="Comment Agent: summarize dashboard and call LLM.")
    ap.add_argument("--in", dest="input_path", default="./out/step5_dashboard/dashboard.csv", help="Input dashboard file (csv or json)")
    ap.add_argument("--out", dest="output_path", default="./out/agent_comment.md", help="Output markdown path")
    ap.add_argument("--max-symbols", type=int, default=20, help="Number of top symbols to include")
    ap.add_argument("--model", default=os.environ.get("OPENAI_MODEL", "gpt-4o-mini"), help="Model name for LLM")
    ap.add_argument("--dry-run", action="store_true", help="Print payload and exit without LLM call")
    ap.add_argument("--loglevel", default="INFO")
    args = ap.parse_args(argv)

    logging.basicConfig(level=getattr(logging, args.loglevel.upper(), logging.INFO))

    inp_path = Path(args.input_path)
    if not inp_path.exists():
        raise SystemExit(f"input not found: {inp_path}")

    df = _read_input(inp_path)
    payload = _build_payload(df, max_symbols=args.max_symbols, source_path=inp_path)
    LOG.info("input=%s themes=%s symbols=%s model=%s prompt_version=%s", inp_path, len(payload.get("themes", [])), payload.get("symbols_count"), args.model, PROMPT_VERSION)

    if args.dry_run:
        print(json.dumps(payload, ensure_ascii=False, indent=2))
        return 0

    out_path = Path(args.output_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    api_key = os.environ.get("OPENAI_API_KEY")
    if not api_key:
        LOG.warning("OPENAI_API_KEY not set; writing fallback summary without LLM.")
        comment = _render_fallback(payload)
    else:
        try:
            from .llm_client import LLMClient

            client = LLMClient(
                api_key=api_key,
                base_url=os.environ.get("OPENAI_BASE_URL", "https://api.openai.com/v1"),
                model=args.model,
                timeout=int(os.environ.get("OPENAI_TIMEOUT_SEC", "60")),
            )
            user_prompt = _build_user_message(payload)
            comment = client.chat(SYSTEM_PROMPT, user_prompt)
        except Exception as e:
            LOG.warning("LLM call failed (%s); writing fallback summary.", e)
            comment = _render_fallback(payload)

    if not _validate_output(comment):
        LOG.warning("LLM output failed validation; using fallback.")
        comment = _render_fallback(payload)

    meta = f"<!-- generated_at={_jst_now_iso()} input={inp_path} prompt_version={PROMPT_VERSION} -->\n"
    out_path.write_text(meta + comment, encoding="utf-8")

    debug = {"payload": payload, "model": args.model, "input": str(inp_path)}
    (out_path.parent / "agent_comment.json").write_text(json.dumps(debug, ensure_ascii=False, indent=2), encoding="utf-8")

    print(f"[OK] wrote {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
