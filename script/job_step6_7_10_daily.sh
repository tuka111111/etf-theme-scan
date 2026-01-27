#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
LOG_DIR="${PROJECT_ROOT}/logs/launchd"

mkdir -p "${LOG_DIR}"
cd "${PROJECT_ROOT}"

exec >>"${LOG_DIR}/step6_7_10_daily.out.log" 2>>"${LOG_DIR}/step6_7_10_daily.err.log"

PYTHON_CANDIDATES=(
  "${PROJECT_ROOT}/.venv/bin/python3"
  "${PROJECT_ROOT}/.venv/bin/python"
  "/usr/bin/python3"
  "/usr/local/bin/python3"
  "/opt/homebrew/bin/python3"
  "python3"
  "python"
)
PYTHON=""
for cand in "${PYTHON_CANDIDATES[@]}"; do
  if [[ "${cand}" == "python3" || "${cand}" == "python" ]]; then
    if command -v "${cand}" >/dev/null 2>&1; then
      PYTHON="$(command -v "${cand}")"
      break
    fi
  elif [[ -x "${cand}" ]]; then
    PYTHON="${cand}"
    break
  fi
done

if [[ -z "${PYTHON}" ]]; then
  echo "[job] PYTHON not found. candidates:" >&2
  for cand in "${PYTHON_CANDIDATES[@]}"; do
    echo "  - ${cand}" >&2
  done
  exit 1
fi

echo "[job] PROJECT_ROOT=${PROJECT_ROOT}" >&2
echo "[job] PYTHON=${PYTHON}" >&2

start_ts=$(date -u +"%Y-%m-%dT%H:%M:%SZ")
echo "[job_step6_7_10] start ${start_ts}"



bash ./run_pipeline.sh
#"${PYTHON}" python -m pipeline.step10_daily_runner --out out --send


end_ts=$(date -u +"%Y-%m-%dT%H:%M:%SZ")
echo "[job_step6_7_10] end ${end_ts}"
