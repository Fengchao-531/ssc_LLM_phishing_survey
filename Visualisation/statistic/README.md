# Heatmap Statistics

This folder stores statistical comparisons for the persuasion-principle heatmaps.

Current workflow:

- `heatmap_significance_analysis.py`
  - builds per-sample heatmap features from the current overview datasets
  - tests overall matrix difference with a permutation test
  - tests each diagonal/co-occurrence cell with Welch's t-test and permutation p-values
  - applies Benjamini-Hochberg FDR correction
  - writes:
    - `summary.json`
    - `feature_stats.csv`
    - `difference_heatmap.png`

- `overview/`
  - contains one subfolder per comparison plus summary tables

Notes:

- The current defaults (`300` permutations, `300` bootstraps) are meant for exploratory analysis.
- For publication-grade results, increase both counts and rerun.
