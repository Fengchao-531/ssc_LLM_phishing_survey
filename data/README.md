# Data

This directory documents the public data sources and contains benchmark-ready inputs that can be redistributed as part of the artifact. Dataset cleaning, filtering, merging, and benchmark-construction preprocessing scripts are intentionally not included; the reproducibility scope starts from the released benchmark inputs and derived analysis artifacts.

## Labels

- `HW-P`: human-written phishing content
- `HW-B`: human-written benign content
- `LLM-P`: LLM-generated phishing content
- `LLM-B`: LLM-generated benign content

## Manipulation Stage Mapping

The repository follows the paper's current taxonomy:

| Stage | Name | Public artifact role |
| --- | --- | --- |
| S1 | Task Control | benchmark input where distributable |
| S2 | Role and Framing Control | benchmark input where distributable |
| S3 | Context Control | benchmark input where distributable |
| S4 | Target-Information Control | benchmark input where distributable |
| S5 | Interaction Control | no standalone benchmark subset in the released benchmark |
| S6 | Output-Property Control | rewriting analysis; sensitive rewritten text withheld |
| S7 | Communication Control | text-based communication-setting characterization |
| S8 | Workflow Control | controlled generator comparison; sensitive generated text withheld |
| S9 | Model Control | no standalone benchmark subset in the released benchmark |

`benchmark/` contains the benchmark inputs used for detector evaluation where redistribution is appropriate. `characterization/` contains text-based material used for characterization analyses. Public upstream sources are catalogued in `dataset_sources.csv`.

## Release Policy

Public availability of an upstream dataset does not necessarily imply unrestricted redistribution. Source links are therefore provided for all documented public resources, while local copies are retained only where appropriate for this artifact. Security-sensitive reconstructed/generated phishing samples used in controlled S6 and S8 experiments are withheld. Derived detector predictions, persuasion scores, statistical outputs, and figures are released separately under `results/`.
