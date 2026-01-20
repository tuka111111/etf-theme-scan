#!/usr/bin/env bash

# Launch Streamlit apps (Step5 + Step10) in background with simple logging.
# Usage: ./run_streamlit.sh [port_step5] [port_step10]
# Default ports: 8501 (Step5), 8502 (Step10)

set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
LOG_DIR="${ROOT_DIR}/out/logs"
APP_STEP5="${ROOT_DIR}/app/streamlit_app.py"
APP_STEP10="${ROOT_DIR}/pipeline/step10_dashboard.py"

mkdir -p "${LOG_DIR}"

CMD="${1:-start}"
PORT_STEP5="${2:-8501}"
PORT_STEP10="${3:-8502}"
PID_FILE_STEP5="${LOG_DIR}/streamlit_step5_${PORT_STEP5}.pid"
PID_FILE_STEP10="${LOG_DIR}/streamlit_step10_${PORT_STEP10}.pid"
LOG_FILE_STEP5="${LOG_DIR}/streamlit_step5_${PORT_STEP5}.log"
LOG_FILE_STEP10="${LOG_DIR}/streamlit_step10_${PORT_STEP10}.log"

# Use local venv under stock/.venv if present; else system python
if [ -x "${ROOT_DIR}/.venv/bin/python" ]; then
  PY="${ROOT_DIR}/.venv/bin/python"
else
  PY="python"
fi

stop_proc() {
  local pid_file="$1"
  if [ -f "${pid_file}" ]; then
    pid=$(cat "${pid_file}")
    if kill -0 "${pid}" 2>/dev/null; then
      echo "Stopping existing streamlit (PID ${pid})"
      kill "${pid}" || true
    fi
    rm -f "${pid_file}"
  fi
}

if [ "${CMD}" = "--stop" ] || [ "${CMD}" = "stop" ]; then
  stop_proc "${PID_FILE_STEP5}"
  stop_proc "${PID_FILE_STEP10}"
  echo "Stopped streamlit on ports ${PORT_STEP5}, ${PORT_STEP10}"
  exit 0
fi

# stop any existing instance on this port before starting
stop_proc "${PID_FILE_STEP5}"
stop_proc "${PID_FILE_STEP10}"

echo "Starting Step5 app on port ${PORT_STEP5} (log: ${LOG_FILE_STEP5})"
nohup "${PY}" -m streamlit run "${APP_STEP5}" --server.port "${PORT_STEP5}" --server.headless true \
  > "${LOG_FILE_STEP5}" 2>&1 &

echo $! > "${PID_FILE_STEP5}"
echo "Step5 PID $(cat "${PID_FILE_STEP5}")"

echo "Starting Step10 dashboard on port ${PORT_STEP10} (log: ${LOG_FILE_STEP10})"
nohup "${PY}" -m streamlit run "${APP_STEP10}" --server.port "${PORT_STEP10}" --server.headless true -- --out out \
  > "${LOG_FILE_STEP10}" 2>&1 &

echo $! > "${PID_FILE_STEP10}"
echo "Step10 PID $(cat "${PID_FILE_STEP10}")"
echo "To stop: ./run_streamlit.sh --stop ${PORT_STEP5} ${PORT_STEP10}"
