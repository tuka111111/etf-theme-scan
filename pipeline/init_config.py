from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))

from pipeline.common import ensure_dir

DEFAULT_SCORING_YAML = """thresholds:
  "1D": 70
  "5D": 75
  "20D": 80
"""

DEFAULT_RULES_YAML = """flags:
  ignore:
    - data_quality_low
    - halted
    - bad_symbol
  weight:
    trend_not_strong_for_signal: 0.8
    env_bias_mismatch: 0.7
env_bias:
  bull:
    action: allow
    score_adjust: 0
  neutral:
    action: allow
    score_adjust: -5
  bear:
    action: skip
    score_adjust: -10
"""


def _write_if_missing(path: Path, content: str, force: bool) -> bool:
    if path.exists() and not force:
        return False
    path.write_text(content, encoding="utf-8")
    return True


def main() -> int:
    ap = argparse.ArgumentParser(description="Initialize config/scoring.yaml and config/rules.yaml")
    ap.add_argument("--config-dir", default="config", help="Config directory (relative to repo root)")
    ap.add_argument("--force", action="store_true", help="Overwrite existing files")
    args = ap.parse_args()

    config_dir = ensure_dir(ROOT / args.config_dir)
    scoring_path = config_dir / "scoring.yaml"
    rules_path = config_dir / "rules.yaml"

    wrote_scoring = _write_if_missing(scoring_path, DEFAULT_SCORING_YAML, args.force)
    wrote_rules = _write_if_missing(rules_path, DEFAULT_RULES_YAML, args.force)

    print(f"config dir: {config_dir}")
    print(f"scoring.yaml: {'created' if wrote_scoring else 'exists'}")
    print(f"rules.yaml: {'created' if wrote_rules else 'exists'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
