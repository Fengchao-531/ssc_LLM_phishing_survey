# Results

This directory contains released intermediate and final artifacts used to reproduce or inspect the paper's empirical results.

- `detector_predictions/`: released detector-level predictions or prediction-derived inputs used by downstream analyses.
- `persuasion_scores/`: released WVAE/persuasion score outputs used by characterization and benchmark analyses.
- `statistics/`: statistical-test outputs and derived summaries, organized by analysis family.
- `tables/`: paper-facing or appendix result tables.
- `figures/`: paper-facing and supporting figures retained after removing obsolete PCA/surrogate/web visualizations.

The intended reproducibility path is `released inputs/predictions/scores -> analysis scripts -> statistics/tables/figures`. Security-sensitive S6/S8 generated phishing text is not required for reproducing the released downstream comparisons.
