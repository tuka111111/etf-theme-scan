from __future__ import annotations

import argparse
import logging
from pathlib import Path
from typing import Optional, Sequence

import pandas as pd

from .trade_append import TRADE_ACTIONS, append_trade_action

LOG = logging.getLogger(__name__)


def main(argv: Optional[Sequence[str]] = None) -> int:
    ap = argparse.ArgumentParser(description="Append a Step7 trade action (append-only).")
    ap.add_argument("--out", required=True, help="Output directory root (writes out/step7_trades)")
    ap.add_argument("--theme", required=True, help="Theme (e.g., XME)")
    ap.add_argument("--symbol", required=True, help="Symbol (e.g., AAPL)")
    ap.add_argument("--action", required=True, help="Action enum: ENTER/WATCH/SKIP/EXIT")
    ap.add_argument("--notes", default="", help="Optional notes")
    ap.add_argument("--step5_dir", default="", help="Optional step5_dashboard directory (for auto-fill)")
    ap.add_argument("--loglevel", default="INFO")
    args = ap.parse_args(argv)

    logging.basicConfig(level=getattr(logging, args.loglevel.upper(), logging.INFO))

    action = args.action.strip().upper()
    if action not in TRADE_ACTIONS:
        raise SystemExit(f"Invalid action {action}. Allowed: {sorted(TRADE_ACTIONS)}")

    record = {
        "theme": args.theme.strip().upper(),
        "symbol": args.symbol.strip().upper(),
        "action": action,
        "notes": args.notes or "",
        "source": "cli",
    }

    if args.step5_dir:
        dash_path = Path(args.step5_dir) / "dashboard.csv"
        if dash_path.exists():
            df = pd.read_csv(dash_path)
            hit = df[df["symbol"].astype(str).str.upper() == record["symbol"]]
            if not hit.empty:
                row = hit.iloc[0].to_dict()
                record.update(
                    {
                        "score_total": row.get("score_total", ""),
                        "env_bias": row.get("env_bias", ""),
                        "env_confidence": row.get("env_confidence", ""),
                        "etf_env_bias": row.get("etf_env_bias", ""),
                        "etf_env_confidence": row.get("etf_env_confidence", ""),
                        "flags": row.get("flags", ""),
                        "snapshot_id": row.get("asof_utc", row.get("asof_local", "")),
                    }
                )

    path = append_trade_action(out_dir=args.out, record=record)
    LOG.info("Appended trade action for %s action=%s to %s", record["symbol"], action, path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
