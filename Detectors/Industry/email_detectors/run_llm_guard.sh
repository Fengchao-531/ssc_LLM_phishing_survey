#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PYTHON_BIN="${PYTHON_BIN:-/scratch3/che489/Ha/.conda/envs/FC-W2-gpu-p39/bin/python}"
INPUT_CSV="${INPUT_CSV:-/scratch3/che489/FC-W2-SoK/ssc_LLM_phishing_survey/Datasets/sublist/S5-Personalization for Credibility/LLM-P.csv}"
SAMPLE_SIZE="${SAMPLE_SIZE:-20}"
OUTPUT_DIR="${OUTPUT_DIR:-}"
SUBJECT_COLUMN="${SUBJECT_COLUMN:-Subject}"
BODY_COLUMN="${BODY_COLUMN:-Body}"
MALICIOUS_URLS_THRESHOLD="${MALICIOUS_URLS_THRESHOLD:-0.7}"

echo "Running llm_guard batch export"
echo "  input_csv=${INPUT_CSV}"
echo "  sample_size=${SAMPLE_SIZE}"
echo "  output_dir=${OUTPUT_DIR:-<auto>}"

CMD=(
  "${PYTHON_BIN}" "${SCRIPT_DIR}/llm_guard.py"
  --input-csv "${INPUT_CSV}"
  --sample-size "${SAMPLE_SIZE}"
  --subject-column "${SUBJECT_COLUMN}"
  --body-column "${BODY_COLUMN}"
  --malicious-urls-threshold "${MALICIOUS_URLS_THRESHOLD}"
)

if [[ -n "${OUTPUT_DIR}" ]]; then
  CMD+=(--output-dir "${OUTPUT_DIR}")
fi

"${CMD[@]}"

# PYTHON_BIN="/scratch3/che489/Ha/.conda/envs/FC-W2-gpu-p39/bin/python" \
# INPUT_CSV="/scratch3/che489/FC-W2-SoK/ssc_LLM_phishing_survey/Datasets/sublist/S5-Personalization for Credibility/LLM-P.csv" \
# SAMPLE_SIZE=20 \
# bash /scratch3/che489/FC-W2-SoK/ssc_LLM_phishing_survey/Detectors/Industry/email_detectors/run_llm_guard.sh
