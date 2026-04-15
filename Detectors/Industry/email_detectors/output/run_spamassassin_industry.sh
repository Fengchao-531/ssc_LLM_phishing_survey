#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
EMAIL_DETECTORS_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../../.." && pwd)"

DETECTOR_ENV="${REPO_ROOT}/Detectors/load_detector_env.sh"
RUN_TEXT_DETECTORS="${SCRIPT_DIR}/run_text_detectors.py"
RUN_HW_DETECTORS="${SCRIPT_DIR}/HW-result/HW-Ind/run_hw_industry_detectors.py"

PYTHON_BIN="${PYTHON_BIN:-python}"
CHECKPOINT_EVERY="${CHECKPOINT_EVERY:-100}"
PROGRESS_EVERY="${PROGRESS_EVERY:-500}"
START_ROW="${START_ROW:-1}"
SAMPLE_SIZE="${SAMPLE_SIZE:-0}"
HW_TMP_DIR="${HW_TMP_DIR:-${SCRIPT_DIR}/logs/tmp/prepared_inputs}"
SPAMASSASSIN_OUTPUT_SUFFIX="${SPAMASSASSIN_OUTPUT_SUFFIX:-__spamassassin_results}"

usage() {
  cat <<'EOF'
Usage:
  bash Detectors/Industry/email_detectors/output/run_spamassassin_industry.sh llm <input_csv>
  bash Detectors/Industry/email_detectors/output/run_spamassassin_industry.sh hw [S2-GD.csv S4-GD.csv ...]

Behavior:
  - runs only the SpamAssassin detector
  - enables resume by default
  - prints progress every 500 rows by default
  - checkpoints every 100 rows by default
  - writes to dedicated SpamAssassin-only output filenames

Environment overrides:
  PYTHON_BIN=python
  CHECKPOINT_EVERY=100
  PROGRESS_EVERY=500
  START_ROW=1
  SAMPLE_SIZE=0
  HW_TMP_DIR=<path>
  SPAMASSASSIN_OUTPUT_SUFFIX=__spamassassin_results

Examples:
  bash Detectors/Industry/email_detectors/output/run_spamassassin_industry.sh \
    llm \
    Datasets/sublist/S2-Role-Framed\ Prompting/LLM-mixed.csv

  bash Detectors/Industry/email_detectors/output/run_spamassassin_industry.sh \
    hw \
    S2-GD.csv \
    S4-GD.csv
EOF
}

if [[ -f "${DETECTOR_ENV}" ]]; then
  # Load detector-specific environment variables and optional local overrides.
  # shellcheck disable=SC1090
  source "${DETECTOR_ENV}"
fi

MODE="${1:-}"
if [[ -z "${MODE}" ]]; then
  usage
  exit 1
fi
shift

case "${MODE}" in
  llm)
    INPUT_CSV="${1:-}"
    if [[ -z "${INPUT_CSV}" ]]; then
      echo "[error] llm mode requires an input CSV path" >&2
      usage
      exit 1
    fi
    shift || true

    parent="$(basename "$(dirname "${INPUT_CSV}")")"
    stem="$(basename "${INPUT_CSV}" .csv)"
    safe_parent="${parent// /_}"
    safe_stem="${stem// /_}"
    output_csv="${SCRIPT_DIR}/LLM-Ind/${safe_parent}__${safe_stem}${SPAMASSASSIN_OUTPUT_SUFFIX}.csv"

    CMD=(
      "${PYTHON_BIN}"
      "${RUN_TEXT_DETECTORS}"
      "--input-csv" "${INPUT_CSV}"
      "--detectors" "spamassassin"
      "--checkpoint-every" "${CHECKPOINT_EVERY}"
      "--progress-every" "${PROGRESS_EVERY}"
      "--result-group" "LLM-Ind"
      "--output-csv" "${output_csv}"
      "--start-row" "${START_ROW}"
      "--python-bin" "${PYTHON_BIN}"
      "--resume-existing"
    )
    if [[ "${SAMPLE_SIZE}" != "0" ]]; then
      CMD+=("--sample-size" "${SAMPLE_SIZE}")
    fi
    ;;

  hw)
    CMD=(
      "${PYTHON_BIN}"
      "${RUN_HW_DETECTORS}"
      "--detectors" "spamassassin"
      "--chunk-size" "${CHECKPOINT_EVERY}"
      "--progress-interval" "${PROGRESS_EVERY}"
      "--start-row" "${START_ROW}"
      "--python-bin" "${PYTHON_BIN}"
      "--tmp-dir" "${HW_TMP_DIR}"
      "--output-suffix" "${SPAMASSASSIN_OUTPUT_SUFFIX}"
      "--resume-existing"
    )
    if [[ "${SAMPLE_SIZE}" != "0" ]]; then
      CMD+=("--sample-size" "${SAMPLE_SIZE}")
    fi
    if [[ "$#" -gt 0 ]]; then
      CMD+=("--datasets" "$@")
    fi
    ;;

  *)
    echo "[error] Unsupported mode: ${MODE}" >&2
    usage
    exit 1
    ;;
esac

printf '[run] %q ' "${CMD[@]}"
printf '\n'
exec "${CMD[@]}"
