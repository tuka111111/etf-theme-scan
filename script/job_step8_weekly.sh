#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="."

cd "$(dirname "$0")/.."
mkdir -p logs/launchd

start_ts="$(date '+%Y-%m-%d %H:%M:%S')"
echo "Step8 Score Validation start: ${start_ts}"

./.venv/bin/python pipeline/step8_validation_score.py --out out

end_ts="$(date '+%Y-%m-%d %H:%M:%S')"
echo "Step8 Score Validation end: ${end_ts}"
echo "Output: out/step8_score_validation/score_validation.csv"
