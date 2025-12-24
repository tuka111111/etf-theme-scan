#!/usr/bin/env python3
"""Theme scanner CLI.

Examples:
  python theme_scan_cli.py --theme XME --holdings ssga --out ./out --top 12
  python theme_scan_cli.py --themes XME,SMH,XBI --holdings ssga --out ./out --top 12

Notes:
- If --holdings is 'ssga', holdings are downloaded from SSGA for each theme.
- If --holdings is a CSV path or a directory, the CLI will attempt to find holdings CSVs.
"""

from __future__ import annotations

import argparse
import os
import sys
from typing import List

from theme_scan_core import (
    ensure_dir,
    parse_themes_arg,
    run_theme,
    snapshot_written_files,
    write_report_all_md,
    write_report_md,
    write_summary_csv,
    write_watchlist_csv,
)


def main(argv: List[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="Theme scanner (multi-theme)")
    ap.add_argument("--theme", required=False, help="Single theme ticker (e.g., XME)")
    ap.add_argument(
        "--themes",
        nargs="+",
        default=None,
        help="Space-separated list (e.g., --themes XME SMH XBI) or comma-separated string",
    )
    ap.add_argument(
        "--holdings",
        required=True,
        help="Holdings source: 'ssga' OR a CSV path OR a directory containing holdings_<THEME>.csv",
    )
    ap.add_argument("--holdings-url", default=None, help="Optional override URL template for SSGA holdings")
    ap.add_argument("--out", default="./out", help="Output directory")
    ap.add_argument("--top", type=int, default=12, help="Top N symbols per theme")
    ap.add_argument("--lookback", type=int, default=260, help="Lookback trading days for metrics")

    args = ap.parse_args(argv)

    # Normalize --themes to a single string for parse_themes_arg
    themes_arg = None
    if isinstance(args.themes, list):
        # allow: --themes XME SMH XBI
        themes_arg = ",".join(args.themes)
    else:
        themes_arg = args.themes

    themes = parse_themes_arg(args.theme, themes_arg)
    if not themes:
        print("[ERR] Provide --theme XME or --themes XME,SMH,XBI", file=sys.stderr)
        return 2

    out_dir = os.path.abspath(args.out)
    ensure_dir(out_dir)

    summary_rows = []
    combined_watchlist: List[str] = []
    written_files: List[str] = []

    for theme in themes:
        try:
            result = run_theme(
                theme=theme,
                holdings=args.holdings,
                out_dir=out_dir,
                top=args.top,
                lookback_days=args.lookback,
                holdings_url=args.holdings_url,
            )

            watchlist_path = os.path.join(out_dir, f"watchlist_{theme}.csv")
            ranking_path = os.path.join(out_dir, f"ranking_{theme}.csv")
            report_path = os.path.join(out_dir, f"report_{theme}.md")

            write_watchlist_csv(watchlist_path, result.selected_symbols)
            result.ranked.to_csv(ranking_path, index=True)
            write_report_md(report_path, result, top_n=min(args.top, max(0, len(result.selected_symbols) - 1)))

            written_files.extend([watchlist_path, ranking_path, report_path])

            for s in result.selected_symbols:
                if s not in combined_watchlist:
                    combined_watchlist.append(s)

            m = result.theme_metrics
            top_syms = [s for s in result.selected_symbols if s != theme][: args.top]

            summary_rows.append(
                {
                    "Theme": theme,
                    "Status": "OK",
                    "ThemeScore": float(result.theme_score),
                    "AsOfUTC": result.asof_utc,
                    "BreadthAdvancersPct": float(m.get("breadth_advancers_pct")),
                    "AvgRet20dPct": float(m.get("avg_ret_20d_pct")),
                    "Near52wHighPct": float(m.get("near_52w_high_pct")),
                    "VolumeQualityPct": float(m.get("volume_quality_pct")),
                    "HoldingsCount": int(len(result.ranked.index)),
                    "TopSymbols": " ".join(top_syms),
                    "Error": "",
                }
            )

            print(f"[OK] Theme: {theme}  Score: {result.theme_score:.1f}/100  AsOf: {result.asof_utc}")
            print(f"[OUT] {watchlist_path}")
            print(f"[OUT] {ranking_path}")
            print(f"[OUT] {report_path}")

        except Exception as e:
            summary_rows.append(
                {
                    "Theme": theme,
                    "Status": "FAIL",
                    "ThemeScore": float("nan"),
                    "AsOfUTC": "",
                    "BreadthAdvancersPct": float("nan"),
                    "AvgRet20dPct": float("nan"),
                    "Near52wHighPct": float("nan"),
                    "VolumeQualityPct": float("nan"),
                    "HoldingsCount": 0,
                    "TopSymbols": "",
                    "Error": str(e),
                }
            )
            print(f"[ERR] Theme {theme} failed: {e}", file=sys.stderr)

    summary_path = os.path.join(out_dir, "summary_themes.csv")
    report_all_path = os.path.join(out_dir, "report_ALL.md")
    watchlist_all_path = os.path.join(out_dir, "watchlist_ALL.csv")

    write_summary_csv(summary_path, summary_rows)
    write_report_all_md(report_all_path, summary_rows)
    written_files.extend([summary_path, report_all_path])

    if combined_watchlist:
        write_watchlist_csv(watchlist_all_path, combined_watchlist)
        written_files.append(watchlist_all_path)

    print(f"[OUT] {summary_path}")
    print(f"[OUT] {report_all_path}")
    if combined_watchlist:
        print(f"[OUT] {watchlist_all_path}")

    # Snapshot (copy outputs into out/<YYYYMMDDHHMM>/). Only include files that exist.
    existing_files = [p for p in written_files if os.path.isfile(p)]
    missing_files = [p for p in written_files if not os.path.isfile(p)]
    if missing_files:
        print(f"[WARN] {len(missing_files)} expected output files were missing; snapshot will skip them.", file=sys.stderr)

    snap_dir = snapshot_written_files(out_dir, existing_files)
    if snap_dir:
        print(f"[OUT] snapshot -> {snap_dir}")

    any_ok = any(str(r.get("Status", "")).upper() == "OK" for r in summary_rows)
    return 0 if any_ok else 2


if __name__ == "__main__":
    raise SystemExit(main())