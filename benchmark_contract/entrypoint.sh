#!/usr/bin/env bash
set -uo pipefail

SCENARIO="${1:-}"
RESULTS_DIR="${PRIORFLOW_RESULTS_DIR:-/app/benchmark_contract/results}"
RAW_LOGS_DIR="${RESULTS_DIR}/raw_logs"
mkdir -p "${RAW_LOGS_DIR}"

usage() {
  echo "Usage: ./benchmark_contract/entrypoint.sh <official_reproduction|standardized_efficiency|regional_robustness>"
}

if [[ -z "${SCENARIO}" ]]; then
  usage
  exit 1
fi

case "${SCENARIO}" in
  official_reproduction|standardized_efficiency|regional_robustness)
    ;;
  *)
    echo "Invalid scenario: ${SCENARIO}"
    usage
    exit 2
    ;;
esac

export PRIORFLOW_RESULTS_DIR="${RESULTS_DIR}"
export PRIORFLOW_RAW_LOGS_DIR="${RAW_LOGS_DIR}"

EXIT_CODE=0
finalize() {
  local status=$?
  if [[ ${status} -ne 0 && ${EXIT_CODE} -eq 0 ]]; then
    EXIT_CODE=${status}
  fi
  python /app/benchmark_contract/export_results.py finalize --scenario "${SCENARIO}" --exit-code "${EXIT_CODE}" \
    > "${RAW_LOGS_DIR}/finalize_${SCENARIO}.log" 2>&1 || true
  if [[ ${EXIT_CODE} -eq 0 ]]; then
    echo "Scenario ${SCENARIO} completed. Results are under ${RESULTS_DIR}."
  else
    echo "Scenario ${SCENARIO} failed with exit code ${EXIT_CODE}. See ${RAW_LOGS_DIR}."
  fi
  exit "${EXIT_CODE}"
}
trap finalize EXIT

python /app/benchmark_contract/export_results.py preflight --scenario "${SCENARIO}" \
  > "${RAW_LOGS_DIR}/preflight_${SCENARIO}.log" 2>&1 || EXIT_CODE=$?

if [[ ${EXIT_CODE} -eq 0 ]]; then
  python /app/benchmark_contract/download_assets.py --scenario "${SCENARIO}" \
    > "${RAW_LOGS_DIR}/download_assets_${SCENARIO}.log" 2>&1 || EXIT_CODE=$?
fi

if [[ ${EXIT_CODE} -eq 0 ]]; then
  case "${SCENARIO}" in
    official_reproduction)
      python /app/benchmark_contract/run_inference.py --scenario official_reproduction \
        > "${RAW_LOGS_DIR}/run_inference_official.log" 2>&1 || EXIT_CODE=$?
      ;;
    standardized_efficiency)
      python /app/benchmark_contract/profile.py --scenario standardized_efficiency \
        > "${RAW_LOGS_DIR}/profile_efficiency.log" 2>&1 || EXIT_CODE=$?
      ;;
    regional_robustness)
      python /app/benchmark_contract/run_inference.py --scenario regional_robustness \
        > "${RAW_LOGS_DIR}/run_inference_regions.log" 2>&1 || EXIT_CODE=$?
      ;;
  esac
fi

