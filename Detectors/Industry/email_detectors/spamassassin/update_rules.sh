#!/usr/bin/env bash
set -euo pipefail

THIS_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

SPAMASSASSIN_BIN="${SPAMASSASSIN_BIN:-spamassassin}"
SA_UPDATE_BIN="${SA_UPDATE_BIN:-sa-update}"
if [[ -d "${HOME}/perl5/etc/mail/spamassassin" ]]; then
  SITE_CONFIG_DIR="${SPAMASSASSIN_SITE_CONFIG_DIR:-${HOME}/perl5/etc/mail/spamassassin}"
else
  SITE_CONFIG_DIR="${SPAMASSASSIN_SITE_CONFIG_DIR:-/etc/mail/spamassassin}"
fi
BENCHMARK_CONFIG_FILE="${SPAMASSASSIN_BENCHMARK_CONFIG_FILE:-${THIS_DIR}/50_local_benchmark.cf}"
PREFS_FILE="${SPAMASSASSIN_PREFS_FILE:-${THIS_DIR}/user_prefs}"

if ! command -v "${SA_UPDATE_BIN}" >/dev/null 2>&1; then
  echo "[error] sa-update not found on PATH: ${SA_UPDATE_BIN}" >&2
  exit 1
fi

"${SA_UPDATE_BIN}"
LINT_CMD=(
  "${SPAMASSASSIN_BIN}"
  "--lint"
  "--siteconfigpath" "${SITE_CONFIG_DIR}"
  "--prefs-file" "${PREFS_FILE}"
  "-x"
)
while IFS= read -r cf_line; do
  [[ -z "${cf_line}" || "${cf_line}" == \#* ]] && continue
  LINT_CMD+=("--cf" "${cf_line}")
done < "${BENCHMARK_CONFIG_FILE}"
"${LINT_CMD[@]}"

echo "[done] SpamAssassin rules updated and linted"
