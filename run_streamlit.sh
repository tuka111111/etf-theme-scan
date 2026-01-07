#!/usr/bin/env bash

# Launch Streamlit app (Decision/Dashboard) in background with simple logging.
# Usage: ./run_streamlit.sh [port]
# Default port: 8501

set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
LOG_DIR="${ROOT_DIR}/out/logs"
APP_PATH="${ROOT_DIR}/app/streamlit_app.py"

mkdir -p "${LOG_DIR}"

CMD="${1:-start}"
PORT="${2:-8501}"
PID_FILE="${LOG_DIR}/streamlit_${PORT}.pid"
LOG_FILE="${LOG_DIR}/streamlit_${PORT}.log"

# Use local venv under stock/.venv if present; else system python
if [ -x "${ROOT_DIR}/.venv/bin/python" ]; then
  PY="${ROOT_DIR}/.venv/bin/python"
else
  PY="python"
fi

stop_proc() {
  if [ -f "${PID_FILE}" ]; then
    pid=$(cat "${PID_FILE}")
    if kill -0 "${pid}" 2>/dev/null; then
      echo "Stopping existing streamlit (PID ${pid})"
      kill "${pid}" || true
    fi
    rm -f "${PID_FILE}"
  fi
}

if [ "${CMD}" = "--stop" ] || [ "${CMD}" = "stop" ]; then
  stop_proc
  echo "Stopped streamlit on port ${PORT}"
  exit 0
fi

# stop any existing instance on this port before starting
stop_proc

echo "Starting streamlit_app.py on port ${PORT} (log: ${LOG_FILE})"
nohup "${PY}" -m streamlit run "${APP_PATH}" --server.port "${PORT}" --server.headless true \
  > "${LOG_FILE}" 2>&1 &

echo $! > "${PID_FILE}"
echo "PID $(cat "${PID_FILE}")"
echo "To stop: ./run_streamlit.sh --stop ${PORT}"
