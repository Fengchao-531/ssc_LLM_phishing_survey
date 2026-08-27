# Analysis

This directory contains the reproducible analysis layer of the artifact.

## Benchmark

`benchmark/` mirrors the main detector evaluation:

- `overall/`: overall/source-aware detector and persuasion analyses.
- `detector_disagreement/`: academic--industrial disagreement and feature comparisons.
- `s6_rewriting/`: controlled analyses across rewriting procedures.
- `s8_generators/`: controlled analyses across LLM generators.

## Persuasion

`persuasion/` contains the RQ2 persuasion/communication-setting analyses and the adapted WVAE scoring implementation under `persuasion/wvae/`.

Analysis scripts consume released benchmark inputs or derived predictions/scores. Dataset cleaning and benchmark-construction preprocessing are outside the public reproduction scope. Curated outputs are stored under `../results/`.
