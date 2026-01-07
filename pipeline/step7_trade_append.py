from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path
from typing import Optional, Sequence

from .trade_append import append_trade, ACTIONS, _build_symbol_index, _load_decision

LOG = logging.getLogger(__name__)


def main(argv: Optional[Sequence[str]] = None) -> int:
    ap = argparse.ArgumentParser(description="Append a trade note to CSV with minimal input (CLI wrapper).")
    ap.add_argument("--out", required=True, help="Output directory root (writes out/trade_log/trades.csv)")
    ap.add_argument("--decision", default="./out/step6_decision/decision_latest.json", help="Path to decision_latest.json")
    ap.add_argument("--symbol", default="", help="Symbol (optional if --interactive)")
    ap.add_argument("--action", required=True, help="Action enum: ENTER/WATCH/SKIP/EXIT/ADD/REDUCE/AVOID")
    ap.add_argument("--notes", default="", help="Optional notes")
    ap.add_argument("--date", default=None, help="YYYY-MM-DD (default: today)")
    ap.add_argument("--interactive", action="store_true", help="Pick symbol from latest decision ENTER/WATCH")
    ap.add_argument("--loglevel", default="INFO")
    args = ap.parse_args(argv)

    logging.basicConfig(level=getattr(logging, args.loglevel.upper(), logging.INFO))

    symbol = args.symbol.strip().upper()
    if args.interactive:
        if not sys.stdin.isatty():
            raise SystemExit("Interactive mode requires a TTY.")
        decision = _load_decision(Path(args.decision))
        sym_index = _build_symbol_index(decision)
        candidates = [(k, v) for k, v in sym_index.items() if v.get("bucket") in {"ENTER", "WATCH"}]
        if not candidates:
            raise SystemExit("No ENTER/WATCH symbols available for selection.")
        from .trade_append import _choose_symbol_interactive  # lazy import to avoid circulars in Streamlit

        symbol = _choose_symbol_interactive(candidates)

    result = append_trade(
        out_dir=args.out,
        symbol=symbol,
        action=args.action.upper(),
        notes=args.notes,
        date_local=args.date,
        source="cli",
        decision_path=args.decision,
    )
    if not result.get("ok"):
        raise SystemExit(result.get("error", "append failed"))
    LOG.info("Appended trade note for %s action=%s to %s", symbol or "(interactive)", args.action.upper(), result.get("path"))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
