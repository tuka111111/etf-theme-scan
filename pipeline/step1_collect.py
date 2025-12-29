# pipeline/step1_collect.py
from __future__ import annotations

import argparse
import json
import logging
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

# IMPORTANT:
# - DO NOT modify theme_scan_core.py
# - This step only collects/normalizes tickers (no scoring, no env, no price download).

from theme_scan_core import (
    download_ssga_holdings_xlsx,
    read_ssga_holdings_xlsx,
    read_vaneck_holdings_csv,
    read_vaneck_holdings_xlsx,
)

LOG = logging.getLogger(__name__)


@dataclass(frozen=True)
class ThemeUniverse:
    theme: str
    benchmark: str
    tickers: List[str]
    source: str
    holdings_path: Optional[str] = None


def _norm_theme(theme: str) -> str:
    return str(theme).strip().upper()


def _read_ssga_tickers_from_xlsx(xlsx_path: Path) -> List[str]:
    df = read_ssga_holdings_xlsx(xlsx_path)

    ticker_col = None
    for c in df.columns:
        c0 = str(c).strip().lower()
        if c0 in {"ticker", "symbol", "ticker symbol"} or ("ticker" in c0 and "symbol" in c0):
            ticker_col = c
            break
    if ticker_col is None:
        raise RuntimeError(f"Could not find ticker/symbol column in {xlsx_path}. Columns={df.columns.tolist()}")

    vals = (
        df[ticker_col]
        .dropna()
        .astype(str)
        .str.strip()
        .str.upper()
        .tolist()
    )

    # Keep it conservative here; core has stricter normalization logic.
    out: List[str] = []
    seen = set()
    for v in vals:
        if not v:
            continue
        # Drop obvious non-tickers
        if len(v) > 12:
            continue
        if v in seen:
            continue
        out.append(v)
        seen.add(v)
    return out


def _find_local_vaneck_file(out_dir: Path, theme: str) -> Path:
    """
    SMH is "local CSV fixed": place holdings file under out_dir.
    Accepted:
      - holdings_SMH_vaneck.csv (preferred fixed name)
      - holdings_SMH_vaneck*.csv / .xlsx / .xls (fallback)
    """
    fixed = out_dir / f"holdings_{theme}_vaneck.csv"
    if fixed.exists():
        return fixed

    cands: List[Path] = []
    cands += sorted(out_dir.glob(f"holdings_{theme}_vaneck*.csv"))
    cands += sorted(out_dir.glob(f"holdings_{theme}_vaneck*.xlsx"))
    cands += sorted(out_dir.glob(f"holdings_{theme}_vaneck*.xls"))
    if cands:
        return cands[0]

    raise RuntimeError(
        f"{theme} holdings unavailable: place a local holdings file in {out_dir} named "
        f"holdings_{theme}_vaneck.csv (or holdings_{theme}_vaneck*.csv/.xlsx)."
    )


def collect_universe_for_theme(
    theme: str,
    *,
    out_dir: Path,
    holdings_provider: str = "ssga",
    ssga_url_template: Optional[str] = None,
) -> ThemeUniverse:
    """
    Step1: Collect holdings tickers for a theme.
    - No price download
    - No env logic
    - No scoring
    """
    theme = _norm_theme(theme)
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    provider = holdings_provider.strip().lower()

    if provider == "ssga":
        if theme == "SMH":
            # SMH is special: local VanEck holdings file fixed
            local_file = _find_local_vaneck_file(out_dir, theme)
            suf = local_file.suffix.lower()
            if suf in {".xlsx", ".xls"}:
                tickers = read_vaneck_holdings_xlsx(local_file)
                source = "vaneck_xlsx_local"
            else:
                tickers = read_vaneck_holdings_csv(local_file)
                source = "vaneck_csv_local"
            return ThemeUniverse(
                theme=theme,
                benchmark=theme,
                tickers=sorted(list(dict.fromkeys([t for t in tickers if t]))),
                source=source,
                holdings_path=str(local_file),
            )

        # Regular SSGA path: download xlsx and parse
        xlsx_path = download_ssga_holdings_xlsx(theme, out_dir=out_dir, url_template=ssga_url_template)
        tickers = _read_ssga_tickers_from_xlsx(xlsx_path)
        return ThemeUniverse(
            theme=theme,
            benchmark=theme,
            tickers=sorted(list(dict.fromkeys([t for t in tickers if t]))),
            source="ssga_xlsx_download",
            holdings_path=str(xlsx_path),
        )

    # If you later support other providers, add branches here.
    raise NotImplementedError(f"Unsupported holdings provider: {holdings_provider}")


def write_step1_outputs(out_dir: Path, universes: Sequence[ThemeUniverse]) -> Dict[str, str]:
    """
    Outputs:
      - out/step1_universe.json
      - out/step1_universe_<THEME>.csv  (Symbol column only)
    """
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # JSON (all themes)
    json_path = out_dir / "step1_universe.json"
    payload = [asdict(u) for u in universes]
    json_path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")

    # CSV per theme
    for u in universes:
        csv_path = out_dir / f"step1_universe_{u.theme}.csv"
        csv_path.write_text("Symbol\n" + "\n".join(u.tickers) + "\n", encoding="utf-8")

    return {"json": str(json_path)}


def main(argv: Optional[Sequence[str]] = None) -> int:
    ap = argparse.ArgumentParser(description="Step1: Collect holdings tickers per theme (no prices, no scoring).")
    ap.add_argument("--themes", required=True, help="Comma-separated themes, e.g. XME,SMH,XBI")
    ap.add_argument("--out", required=True, help="Output directory (also used for local holdings lookup for SMH)")
    ap.add_argument("--holdings", default="ssga", help="Holdings provider (default: ssga)")
    ap.add_argument(
        "--ssga-url-template",
        default=None,
        help="Optional override URL template for SSGA XLSX download.",
    )
    ap.add_argument("--loglevel", default="INFO", help="DEBUG/INFO/WARN/ERROR")
    args = ap.parse_args(argv)

    logging.basicConfig(level=getattr(logging, args.loglevel.upper(), logging.INFO))
    out_dir = Path(args.out)

    themes = [_norm_theme(t) for t in args.themes.replace(" ", ",").split(",") if t.strip()]
    themes = list(dict.fromkeys(themes))

    universes: List[ThemeUniverse] = []
    for t in themes:
        LOG.info("Collecting universe for theme=%s provider=%s", t, args.holdings)
        u = collect_universe_for_theme(
            t,
            out_dir=out_dir,
            holdings_provider=args.holdings,
            ssga_url_template=args.ssga_url_template,
        )
        universes.append(u)
        LOG.info("Collected %d tickers for %s (source=%s)", len(u.tickers), u.theme, u.source)

    write_step1_outputs(out_dir, universes)
    LOG.info("Wrote step1 outputs under %s", out_dir)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())