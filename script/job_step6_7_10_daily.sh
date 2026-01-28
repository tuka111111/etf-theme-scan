#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/.."

export PROJECT_ROOT="${PROJECT_ROOT:-$(cd "$(dirname "$0")/.." && pwd)}"

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"

cd "${PROJECT_ROOT}"
mkdir -p logs/launchd
source "${PROJECT_ROOT}/env.sh"
source "${PROJECT_ROOT}/.env"
echo "[env] python=${PYTHON}"

#echo $STEP10_DISCORD_WEBHOOK
LOG_DIR="${PROJECT_ROOT}/logs/launchd"

cd "${PROJECT_ROOT}"

exec >> "${LOG_DIR}/step6_7_10_daily.out.log" 2>>"${LOG_DIR}/step6_7_10_daily.err.log"

VENV_ROOT="${VENV_DIR}"
if [[ ! -x "${VENV_ROOT}/bin/activate" ]]; then
  echo "[job] .venv missing: ${VENV_ROOT}/bin/activate" >&2
  exit 1
fi

# shellcheck disable=SC1090
source  "${VENV_ROOT}/bin/activate"

if [[ ! -x "${VENV_ROOT}/bin/python" ]]; then
  echo "[job] .venv python missing: ${VENV_ROOT}/bin/python" >&2
  exit 1
fi

echo "[job] PROJECT_ROOT=${PROJECT_ROOT}" >&2
echo "[job] PYTHON=$(command -v python)" >&2
python -c 'import sys; print("[job] sys.executable=" + sys.executable)' >&2

start_ts=$(date -u +"%Y-%m-%dT%H:%M:%SZ")
echo "[job_step6_7_10] start ${start_ts}"



bash  ./run_pipeline.sh
#python -m pipeline.step10_daily_runner --out out --send


end_ts=$(date -u +"%Y-%m-%dT%H:%M:%SZ")
echo "[job_step6_7_10] end ${end_ts}"
