# S5 Personalization for Credibility

This folder contains the local S5 subset and a safe process-oriented reproduction of:

- Qi et al., "SpearBot: Leveraging large language models in a generative-critique framework for spear-phishing email generation," Information Fusion 2025.

## Local size reference

- `LLM-P.csv`: `1228` rows in the current repository

The `qi2025` reproduction script in this folder therefore defaults to a target size close to `1228` rows so its output is easier to compare with the existing S5 subset.

## What the Qi et al. paper contributes

The paper's recoverable generation process includes:

- `50` student + `50` employee virtual profiles
- a `10`-strategy catalog
- profile generation followed by company/university enrichment
- multi-turn generator initialization
- a critic loop that requests regeneration until the email passes or the iteration limit is reached

## What this local script does

`reproduce_qi2025_spearbot_safe.py` reproduces the paper's *workflow* while keeping the content benign:

- builds fictional recipient profiles
- applies the strategy-conditioned generation prompt
- runs multiple critics and rewrites based on critique reasons
- writes `LLM-B.csv`, profile JSON, critic JSON, call logs, and a manifest
- scales the number of profiles so the final row count is close to the local S5 `LLM-P.csv` size by default

Important limitation:

- This script does **not** generate phishing emails. It is a safe process reproduction for studying personalization and critique dynamics without recreating harmful payloads.

## Example

Quick smoke test:

```bash
python3 "Datasets/sublist/S5-Personalization for Credibility/reproduce_qi2025_spearbot_safe.py" \
  --backend mock \
  --strategy-limit 2 \
  --max-rows 4
```

Default size-aligned run:

```bash
python3 "Datasets/sublist/S5-Personalization for Credibility/reproduce_qi2025_spearbot_safe.py" \
  --backend mock
```
