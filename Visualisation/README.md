# Visualisation

This directory contains the public visualisation assets used to inspect detector behavior across the LLM-enabled phishing lifecycle.

The interactive viewer is deployed through GitHub Pages:

```text
https://fengchao-531.github.io/ssc_LLM_phishing_survey/
```

## Directory layout

- `index.html`: a static interaction page for selecting detector, stage, and metric mode.
- `data/`: aggregate metric JSON used by the interaction page.
- `output/`: detector interaction assets. Each detector has an `overview/` folder and stage folders for `S1`, `S2`, `S4`, `S5`, `S6`, and `S8`.
- `wvae/`: the adapted WVAE persuasion-scoring code used to produce the six persuasion-principle scores that support the heatmaps, surrogate maps, and FN boxplots.

## Output assets

The `output/` directory is organized by detector name. Each detector folder contains:

- `overview/`: an overall surrogate response map plus the shared overview persuasion heatmap.
- `S1/`, `S2/`, `S4/`, `S5/`: stage-specific surrogate maps and FN persuasion-principle boxplots.
- `S6/`: combined visualisations for `S6-MPG`, `S6-UTA`, and `S6-fuzzer`.
- `S8/`: two combined rows covering the six S8 model-output groups.

The `output/` directory contains README files and PNG figures only. Raw email text, row-level detector outputs, CSV files, and JSON files are not included in `output/`.

## WVAE source

The persuasion scores are derived from an adapted WVAE model based on Chen and Yang's AAAI 2021 work on weakly supervised persuasive strategy prediction. See `wvae/` for the adapted scoring code, citation, and usage notes.
