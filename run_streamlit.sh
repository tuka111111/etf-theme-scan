#!/usr/bin/env bash

# Launch Streamlit app (Decision/Dashboard) in background with simple logging.
# Usage: ./run_streamlit.sh [port]
# Default port: 8501

set -euo pipefail

PORT="${1:-8501}"
ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
LOG_DIR="${ROOT_DIR}/out/logs"
APP_PATH="${ROOT_DIR}/app/streamlit_app.py"

mkdir -p "${LOG_DIR}"

LOG_FILE="${LOG_DIR}/streamlit_${PORT}.log"

echo "Starting streamlit_app.py on port ${PORT} (log: ${LOG_FILE})"
nohup "${ROOT_DIR}/.venv/bin/streamlit" run "${APP_PATH}" --server.port "${PORT}" --server.headless true \
  > "${LOG_FILE}" 2>&1 &

echo $! > "${LOG_DIR}/streamlit_${PORT}.pid"
echo "PID $(cat "${LOG_DIR}/streamlit_${PORT}.pid")"
