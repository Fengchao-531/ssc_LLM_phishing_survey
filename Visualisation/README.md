# Visualization

This folder now keeps only the phishing-only visualization outputs for the `scamllm` detector.

## Current Result Layout

- `PCA/`
  Uses `PCA` as the 2D projection for the phishing-indicator space.
- `UMAP/`
  Uses `UMAP` as the 2D projection for the phishing-indicator space.

Each folder contains the same four figure families:

1. `01_pattern_map_scamllm_full_phishing_only.png`
2. `02_detector_coverage_scamllm_full_phishing_only.png`
3. `03_indicator_composition_scamllm_full_phishing_only.png`
4. `04_motif_delta_scamllm_full_phishing_only.png`

Each folder also keeps:

- `sample_manifest_full_phishing_only.csv`
- `run_metadata_full_phishing_only.json`
- `sampled_scamllm_full_phishing_only.csv`

## Data Scope

These results are **phishing-only**.

That means:

- `HW` side keeps only `HW-phishing`
- `LLM` side keeps only `LLM-phishing`
- benign rows are filtered out before visualization

The current plots therefore compare:

- `HW-phishing`
- `LLM-phishing`

## Detector

The detector signal used in both folders is still:

- `scamllm`

The coverage plot uses the stored `scamllm` predictions and fits a lightweight surrogate response field in the 2D projected space for visualization.

## Re-run Commands

From the repository root:

```bash
python Visualization/generate_stage_visualizations.py \
  --sample-size 0 \
  --phishing-only \
  --projection pca \
  --output-dir Visualization/PCA
```

```bash
python Visualization/generate_stage_visualizations.py \
  --sample-size 0 \
  --phishing-only \
  --projection umap \
  --output-dir Visualization/UMAP
```

## Notes

- Older preview and mixed benign/phishing outputs were removed.
- `UMAP` output now uses the installed `umap-learn` package rather than falling back to `PCA`.
