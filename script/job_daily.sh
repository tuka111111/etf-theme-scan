#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="${PROJECT_ROOT:-$(cd "$(dirname "$0")/.." && pwd)}"
LOG_DIR="${PROJECT_ROOT}/logs/launchd"

mkdir -p "${LOG_DIR}"
cd "${PROJECT_ROOT}"

exec >>"${LOG_DIR}/daily.out.log" 2>>"${LOG_DIR}/daily.err.log"

start_ts=$(date -u +"%Y-%m-%dT%H:%M:%SZ")
echo "[job_daily] start ${start_ts}"

bash ./run_pipeline.sh
#"${PROJECT_ROOT}/.venv/bin/python" pipeline/step10_daily_runner.py --out out

end_ts=$(date -u +"%Y-%m-%dT%H:%M:%SZ")
echo "[job_daily] end ${end_ts}"
