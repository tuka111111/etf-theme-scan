#!/usr/bin/env bash
# Step1–Step5 end-to-end runner

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
source "${SCRIPT_DIR}/env.sh"
echo "[env] python=${PYTHON}"


THEMES="${1:-XME,SMH,XBI}"
OUT="${OUT:-./out}"
CONTRACTS="${CONTRACTS:-./contracts}"
HTF="${HTF:-1H}"
STEP2_TF="${STEP2_TF:-1D}"
LOOKBACK="${LOOKBACK:-260}"
LOGLEVEL="${LOGLEVEL:-INFO}"

echo "[run] themes=${THEMES} out=${OUT} htf=${HTF} step2_tf=${STEP2_TF}"

"${PYTHON}" -m pipeline.step1_collect --themes "${THEMES}" --holdings ssga --out "${OUT}" --loglevel "${LOGLEVEL}"

"${PYTHON}" -m pipeline.step2_prices --themes "${THEMES}" --out "${OUT}" --contracts "${CONTRACTS}" --tf "${STEP2_TF}" --lookback "${LOOKBACK}" --loglevel "${LOGLEVEL}"

"${PYTHON}" -m pipeline.step2_env --themes "${THEMES}" --out "${OUT}" --timeframe "${STEP2_TF}" --loglevel "${LOGLEVEL}"

"${PYTHON}" -m pipeline.step3_env --themes "${THEMES}" --out "${OUT}" --contracts "${CONTRACTS}" --htf "${HTF}" --loglevel "${LOGLEVEL}"

"${PYTHON}" -m pipeline.step3_trend --themes "${THEMES}" --out "${OUT}" --contracts "${CONTRACTS}" --htf "${HTF}" --loglevel "${LOGLEVEL}"

"${PYTHON}" -m pipeline.step3_regime --themes "${THEMES}" --out "${OUT}" --contracts "${CONTRACTS}" --htf "${HTF}" --loglevel "${LOGLEVEL}"

"${PYTHON}" -m pipeline.step3_etf_env --themes "${THEMES}" --out "${OUT}" --contracts "${CONTRACTS}" --loglevel "${LOGLEVEL}"

"${PYTHON}" -m pipeline.step4_score --themes "${THEMES}" --out "${OUT}" --contracts "${CONTRACTS}" --htf "${HTF}" --loglevel "${LOGLEVEL}"

"${PYTHON}" -m pipeline.step5_report --themes "${THEMES}" --out "${OUT}" --contracts "${CONTRACTS}"

"${PYTHON}" -m pipeline.step6_decision --out "${OUT}"

"${PYTHON}" -m pipeline.step7_log --out "${OUT}"

"${PYTHON}" -m pipeline.step10_daily_runner --out "${OUT}" --send

echo "[done] pipeline completed"
