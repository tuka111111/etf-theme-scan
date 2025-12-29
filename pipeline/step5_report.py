"""
Step5: Dashboard / Report generator.

Combines Step4 scores across themes into a unified dashboard (CSV + JSON + optional Markdown).
"""
from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path
from typing import Dict, List, Optional, Sequence

import pandas as pd
from jsonschema import Draft202012Validator

from .common import ensure_dir, now_utc_iso, parse_csv_list

LOG = logging.getLogger(__name__)

# Inline, permissive schema for dashboard envelope
DASHBOARD_SCHEMA = {
    "$schema": "https://json-schema.org/draft/2020-12/schema",
    "title": "Step5 Dashboard Output",
    "type": "object",
    "additionalProperties": False,
    "required": ["schema_version", "generated_at_utc", "rows"],
    "properties": {
        "schema_version": {"type": "string"},
        "generated_at_utc": {"type": "string"},
        "source": {"type": "string"},
        "notes": {"type": "string"},
        "rows": {
            "type": "array",
            "minItems": 1,
            "items": {"$ref": "#/$defs/DashboardRow"},
        },
    },
    "$defs": {
        "DashboardRow": {
            "type": "object",
            "additionalProperties": True,
            "required": ["asof_utc", "theme", "symbol", "rank", "score_total", "env_bias", "env_confidence"],
            "properties": {
                "asof_utc": {"type": "string"},
                "theme": {"type": "string"},
                "symbol": {"type": "string"},
                "rank": {"type": "integer", "minimum": 1},
                "score_total": {"type": "number"},
                "env_bias": {"type": "string"},
                "env_confidence": {"type": "number"},
                "notes": {"type": "string"},
                "flags": {"type": "string"},
                "debug": {"type": "object"},
            },
        }
    },
}


def _load_scores(theme: str, out_dir: Path) -> List[Dict]:
    path = out_dir / "step4_scores" / f"scores_{theme}.json"
    if not path.exists():
        raise FileNotFoundError(f"Missing Step4 scores for theme={theme}: {path}")
    payload = json.loads(path.read_text(encoding="utf-8"))
    rows = payload.get("rows", [])
    if not isinstance(rows, list) or not rows:
        raise ValueError(f"Invalid or empty scores rows for theme={theme} file={path}")
    return rows


def _build_dashboard_rows(theme: str, score_rows: List[Dict]) -> List[Dict]:
    # Rank by score_total desc
    df = pd.DataFrame(score_rows)
    df["score_total"] = pd.to_numeric(df["score_total"], errors="coerce").fillna(0.0)
    df_sorted = df.sort_values(["score_total"], ascending=[False]).reset_index(drop=True)
    df_sorted["rank"] = df_sorted.index + 1

    rows: List[Dict] = []
    for _, r in df_sorted.iterrows():
        rows.append(
            {
                "asof_utc": str(r.get("asof_utc")),
                "theme": str(r.get("theme", theme)),
                "symbol": str(r.get("symbol", "")),
                "rank": int(r.get("rank")),
                "score_total": float(r.get("score_total", 0.0)),
                "env_bias": str(r.get("env_bias", "")),
                "env_confidence": float(r.get("env_confidence", 0.0)),
                "flags": str(r.get("flags", "")),
                "notes": "",
                "debug": {"source": "step4_score"},
            }
        )
    return rows


def _validate_payload(payload: Dict, schema_path: Optional[Path]) -> Dict:
    if schema_path and schema_path.exists():
        from .validate import must_validate

        return must_validate(schema_path, payload)

    v = Draft202012Validator(DASHBOARD_SCHEMA)
    errs = list(v.iter_errors(payload))
    if errs:
        lines = []
        for e in errs[:20]:
            p = "/".join([str(x) for x in e.absolute_path]) or "(root)"
            lines.append(f"path={p} msg={e.message}")
        more = "" if len(errs) <= 20 else f"\n... ({len(errs)-20} more)"
        raise ValueError("Schema validation failed (inline DASHBOARD_SCHEMA):\n" + "\n".join(lines) + more)
    return payload


def _write_markdown(md_path: Path, rows: List[Dict]) -> None:
    lines: List[str] = []
    lines.append("# Dashboard\n")
    lines.append("| Rank | Theme | Symbol | Score | Env Bias | Env Conf | Flags |")
    lines.append("| --- | --- | --- | --- | --- | --- | --- |")
    for r in rows:
        lines.append(
            f"| {r['rank']} | {r['theme']} | {r['symbol']} | {r['score_total']:.2f} | "
            f"{r['env_bias']} | {r['env_confidence']:.2f} | {r.get('flags','')} |"
        )
    md_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main(argv: Optional[Sequence[str]] = None) -> int:
    ap = argparse.ArgumentParser(description="Step5: dashboard/report from Step4 scores.")
    ap.add_argument("--themes", required=True, help="Comma-separated themes, e.g. XME,SMH,XBI")
    ap.add_argument("--out", required=True, help="Output directory (expects Step4 scores)")
    ap.add_argument("--contracts", default="./contracts", help="Contracts directory (optional step5_dashboard.schema.json)")
    ap.add_argument("--loglevel", default="INFO", help="Logging level")
    ap.add_argument("--no-md", action="store_true", help="Skip markdown output")
    args = ap.parse_args(argv)

    logging.basicConfig(level=getattr(logging, args.loglevel.upper(), logging.INFO))

    themes = [t.strip().upper() for t in parse_csv_list(args.themes)]
    if not themes:
        raise SystemExit("Provide --themes")

    out_dir = ensure_dir(args.out)
    dash_dir = ensure_dir(out_dir / "step5_dashboard")

    all_rows: List[Dict] = []
    for theme in themes:
        try:
            scores = _load_scores(theme, out_dir)
        except Exception as e:
            LOG.warning("Skipping theme=%s due to missing/invalid scores: %s", theme, e)
            continue
        rows = _build_dashboard_rows(theme, scores)
        all_rows.extend(rows)

    if not all_rows:
        raise SystemExit("No dashboard rows produced (check Step4 scores).")

    payload = {
        "schema_version": "5.dashboard.v1",
        "generated_at_utc": now_utc_iso(),
        "source": "step5_report.py",
        "notes": "Dashboard built from Step4 scores.",
        "rows": all_rows,
    }

    schema_path = Path(args.contracts) / "step5_dashboard.schema.json"
    payload = _validate_payload(payload, schema_path if schema_path.exists() else None)

    json_path = dash_dir / "dashboard.json"
    csv_path = dash_dir / "dashboard.csv"
    json_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    pd.DataFrame(all_rows).to_csv(csv_path, index=False)

    if not args.no_md:
        _write_markdown(dash_dir / "report.md", all_rows)

    LOG.info("Dashboard rows=%d json=%s csv=%s", len(all_rows), json_path, csv_path)
    print(f"[OK] dashboard rows={len(all_rows)} json={json_path} csv={csv_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
