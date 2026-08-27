#!/usr/bin/env bash
set -euo pipefail

# Official references:
# - https://spamassassin.apache.org/doc.html
# - https://svn.apache.org/repos/asf/spamassassin/trunk/INSTALL
# - https://spamassassin.apache.org/full/4.0.x/doc/sa-update.html

THIS_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${THIS_DIR}/../../../.." && pwd)"
USER_PERL_BIN="${HOME}/perl5/bin"
USER_PERL_LIB="${HOME}/perl5/lib/perl5"
BENCHMARK_CONFIG_FILE="${SPAMASSASSIN_BENCHMARK_CONFIG_FILE:-${THIS_DIR}/50_local_benchmark.cf}"

if [[ -x "${USER_PERL_BIN}/spamassassin" ]]; then
  export PATH="${USER_PERL_BIN}:$PATH"
  export PERL5LIB="${USER_PERL_LIB}${PERL5LIB:+:$PERL5LIB}"
  SPAMASSASSIN_BIN="${SPAMASSASSIN_BIN:-${USER_PERL_BIN}/spamassassin}"
  SA_UPDATE_BIN="${SA_UPDATE_BIN:-${USER_PERL_BIN}/sa-update}"
else
  SPAMASSASSIN_BIN="${SPAMASSASSIN_BIN:-spamassassin}"
  SA_UPDATE_BIN="${SA_UPDATE_BIN:-sa-update}"
fi
if [[ -d "${HOME}/perl5/etc/mail/spamassassin" ]]; then
  SITE_CONFIG_DIR="${SPAMASSASSIN_SITE_CONFIG_DIR:-${HOME}/perl5/etc/mail/spamassassin}"
else
  SITE_CONFIG_DIR="${SPAMASSASSIN_SITE_CONFIG_DIR:-/etc/mail/spamassassin}"
fi
PREFS_FILE="${SPAMASSASSIN_PREFS_FILE:-${THIS_DIR}/user_prefs}"

if command -v "${SPAMASSASSIN_BIN}" >/dev/null 2>&1; then
  echo "[info] SpamAssassin already available: $(command -v "${SPAMASSASSIN_BIN}")"
else
  if command -v apt-get >/dev/null 2>&1; then
    echo "[info] Installing SpamAssassin via apt-get"
    if command -v sudo >/dev/null 2>&1; then
      sudo apt-get update
      sudo apt-get install -y spamassassin
    else
      apt-get update
      apt-get install -y spamassassin
    fi
  elif command -v dnf >/dev/null 2>&1; then
    echo "[info] Installing SpamAssassin via dnf"
    if command -v sudo >/dev/null 2>&1; then
      sudo dnf install -y spamassassin
    else
      dnf install -y spamassassin
    fi
  elif command -v yum >/dev/null 2>&1; then
    echo "[info] Installing SpamAssassin via yum"
    if command -v sudo >/dev/null 2>&1; then
      sudo yum install -y spamassassin
    else
      yum install -y spamassassin
    fi
  elif command -v zypper >/dev/null 2>&1; then
    echo "[info] Installing SpamAssassin via zypper"
    if command -v sudo >/dev/null 2>&1; then
      sudo zypper --non-interactive install spamassassin
    else
      zypper --non-interactive install spamassassin
    fi
  elif command -v cpan >/dev/null 2>&1; then
    echo "[info] Installing Mail::SpamAssassin via CPAN into ${HOME}/perl5"
    export PATH="${USER_PERL_BIN}:$PATH"
    export PERL5LIB="${USER_PERL_LIB}${PERL5LIB:+:$PERL5LIB}"
    export PERL_MM_OPT="INSTALL_BASE=${HOME}/perl5"
    export PERL_MB_OPT="--install_base ${HOME}/perl5"
    cpan -T Mail::SpamAssassin
    SPAMASSASSIN_BIN="${USER_PERL_BIN}/spamassassin"
    SA_UPDATE_BIN="${USER_PERL_BIN}/sa-update"
  else
    echo "[error] No supported installer found. Install SpamAssassin manually first." >&2
    exit 1
  fi
fi

if [[ ! -x "${SPAMASSASSIN_BIN}" ]] && ! command -v "${SPAMASSASSIN_BIN}" >/dev/null 2>&1; then
  echo "[error] SpamAssassin binary still not available after installation attempt: ${SPAMASSASSIN_BIN}" >&2
  exit 1
fi

if command -v "${SA_UPDATE_BIN}" >/dev/null 2>&1; then
  echo "[info] Updating SpamAssassin rules with sa-update"
  "${SA_UPDATE_BIN}"
else
  echo "[warn] sa-update not found; skipping rules update"
fi

echo "[info] Linting SpamAssassin config"
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

echo "[done] SpamAssassin install/config check completed for ${REPO_ROOT}"
