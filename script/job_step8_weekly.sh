#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="."


cd "$(dirname "$0")/.."
mkdir -p logs/launchd
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
cd "${PROJECT_ROOT}"
source "${PROJECT_ROOT}/env.sh"
echo "[env] python=${PYTHON}"


start_ts="$(date '+%Y-%m-%d %H:%M:%S')"
echo "Step8 Score Validation start: ${start_ts}"

$PYTHON pipeline/step8_validation_score.py --out_dir out/step8_validation  --decision_csv out/step6_decision/decision_latest.csv   --prices_dir out/step2_prices

end_ts="$(date '+%Y-%m-%d %H:%M:%S')"
echo "Step8 Score Validation end: ${end_ts}"
echo "Output: out/step8_score_validation/score_validation.csv"
