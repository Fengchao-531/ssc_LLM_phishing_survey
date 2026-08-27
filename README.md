# SoK: Systematizing Generation, Characteristics, and Defenses in LLM-Generated Phishing

This repository contains the research artifact accompanying our SoK on **LLM-generated phishing content**. The public artifact is organized around the paper's empirical evidence: benchmark-ready data that can be redistributed, detector implementations and configurations, persuasion/WVAE analysis code, released detector predictions and persuasion scores, statistical analyses, and paper-facing tables and figures.

The repository intentionally does **not** reproduce the upstream dataset cleaning, filtering, merging, or benchmark-construction pipeline. Reproducibility starts from the benchmark-ready inputs and released derived artifacts used in the paper analyses.

## Repository Structure

- [`systematization/`](systematization/): documentation for the paper's generation/characterization/defense systematization and the current manipulation-stage organization used by the artifact.
- [`data/`](data/): public dataset references and benchmark-ready inputs that can be redistributed. See [`data/dataset_sources.csv`](data/dataset_sources.csv) for upstream resources.
- [`detectors/`](detectors/): academic and industrial detector implementations/configurations, dependency notes, and small runnable examples.
- [`analysis/`](analysis/): reproducible analysis code for persuasion characterization and the detector benchmark, including overall analysis, academic--industrial disagreement, S6 rewriting, and S8 generator comparisons.
- [`results/`](results/): released detector predictions, persuasion scores, statistical outputs, tables, and figures used to validate the reported analyses.

## Reproducibility Scope

The artifact supports reproduction of the paper's **evaluation and analysis layers**:

1. Run supported detectors on released benchmark-ready inputs where the required model/service is available.
2. Recompute detector metrics from released predictions.
3. Recompute persuasion-based and action/linguistic statistical analyses from released derived scores and predictions.
4. Regenerate the corresponding comparison figures and result tables from the released analysis inputs.
5. Re-run the adapted WVAE scoring procedure when the required external model artifact is available; otherwise, downstream analyses can start from the released persuasion scores in `results/persuasion_scores/`.

The upstream data-selection and preprocessing scripts used to construct benchmark subsets are not part of this public reproduction pipeline.

## Data and Safety

All public upstream resources used or referenced by the artifact are catalogued in [`data/dataset_sources.csv`](data/dataset_sources.csv). A dataset being publicly accessible does not necessarily imply unrestricted redistribution, so the repository retains local benchmark copies only where appropriate and otherwise points to the upstream source.

Security-sensitive reproduced/generated phishing text from the controlled S6 and S8 experiments is not publicly released. For these experiments, the artifact instead provides the derived detector outputs, persuasion/feature measurements, statistical results, and figures needed to inspect and reproduce the reported comparisons without redistributing the generated attack content.

API keys, private paths, and local credentials are not committed. Some detector wrappers require external services or locally obtained model checkpoints; see [`detectors/README.md`](detectors/README.md) for setup notes.

## Analysis Organization

The benchmark analysis mirrors the paper's evaluation logic:

```text
analysis/benchmark/
├── overall/
├── detector_disagreement/
├── s6_rewriting/
└── s8_generators/
```

Persuasion characterization and WVAE-related code are under:

```text
analysis/persuasion/
└── wvae/
```

Final and intermediate released outputs are separated from the analysis code under `results/`.

## Citation

Please cite the accompanying paper when using this artifact. Formal bibliographic metadata will be added once the paper's publication metadata is finalized.
