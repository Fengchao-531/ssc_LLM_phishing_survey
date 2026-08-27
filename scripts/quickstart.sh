#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

COMMAND="${1:-check}"

pull_lfs() {
  if git lfs version >/dev/null 2>&1; then
    echo "Pulling Git LFS artifacts..."
    git lfs pull
  else
    echo "Git LFS is not installed." >&2
    echo "Install Git LFS, then run: git lfs pull" >&2
    return 1
  fi
}

check_layout() {
  for dir in systematization data detectors analysis results; do
    if [[ ! -d "$dir" ]]; then
      echo "Missing required directory: $dir" >&2
      return 1
    fi
  done
}

check_lfs_pointers() {
  local pointers
  pointers="$(grep -RIl '^version https://git-lfs.github.com/spec/v1$' analysis data results 2>/dev/null || true)"
  if [[ -n "$pointers" ]]; then
    echo "Unresolved Git LFS pointer files:" >&2
    echo "$pointers" >&2
    echo "Run: git lfs pull" >&2
    return 1
  fi
}

check_python() {
  python - <<'PY'
import importlib.util
mods = ["numpy", "pandas", "matplotlib"]
missing = [m for m in mods if importlib.util.find_spec(m) is None]
if missing:
    print("Optional analysis dependencies missing: " + ", ".join(missing))
    print("Detector dependencies: pip install -r detectors/requirements.txt")
else:
    print("Core analysis dependencies are available.")
PY
}

preview_figures() {
  local figures=(
    "results/figures/overview/overview.jpg"
    "results/figures/persuasion/rq2_vishing_multi_minus_email_difference_heatmaps.png"
    "results/figures/benchmark/s6_rewriting/fig_s6_rewriting_two_panel_main_a.png"
    "results/figures/benchmark/s8_generators/Fig_S8_A_detector_generator_detection_rate_heatmap.png"
  )

  for figure in "${figures[@]}"; do
    [[ -f "$figure" ]] || { echo "Missing figure: $figure" >&2; return 1; }
    echo "$figure"
  done

  if command -v open >/dev/null 2>&1; then
    open "${figures[@]}"
  elif command -v xdg-open >/dev/null 2>&1; then
    for figure in "${figures[@]}"; do
      xdg-open "$figure" >/dev/null 2>&1 || true
    done
  else
    echo "Open the paths above in your image viewer."
  fi
}

case "$COMMAND" in
  check|test)
    pull_lfs
    check_layout
    check_lfs_pointers
    check_python
    echo "Artifact check complete."
    ;;
  preview|figures)
    preview_figures
    ;;
  *)
    cat <<'EOF'
Usage:
  bash scripts/quickstart.sh check     Pull LFS files and smoke-check the artifact
  bash scripts/quickstart.sh preview   Open the overview and representative result figures
EOF
    exit 1
    ;;
esac
