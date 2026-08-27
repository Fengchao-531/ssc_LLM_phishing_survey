# WVAE Persuasion Scoring

This folder contains the adapted WVAE scoring code used for the paper's persuasion analyses. It starts from the public WVAE codebase released with Chen and Yang's AAAI 2021 paper and maps sentence-level persuasive-strategy probabilities into six Cialdini-style principle scores:

- authority
- reciprocity
- commitment
- scarcity
- social proof
- liking

The resulting email-level scores are used by the persuasion characterization and detector analyses. Released score outputs used by downstream statistics are stored under `../../../results/persuasion_scores/`.

## Citation

```bibtex
@inproceedings{chen2021weakly,
  title = {Weakly-Supervised Hierarchical Models for Predicting Persuasive Strategies in Good-faith Textual Requests},
  author = {Chen, Jiaao and Yang, Diyi},
  booktitle = {Proceedings of the AAAI Conference on Artificial Intelligence},
  volume = {35},
  number = {14},
  pages = {12648--12656},
  year = {2021},
  doi = {10.1609/aaai.v35i14.17498}
}
```

## Adaptation

The original WVAE model predicts request-oriented persuasive strategies including credibility, reciprocity, evidence, commitment, scarcity, social identity, emotion, impact, and politeness. The artifact aggregates them into six email-level principles:

```text
authority     <- credibility + evidence
reciprocity   <- reciprocity
commitment    <- commitment
scarcity      <- scarcity
social_proof  <- social_identity + impact
liking        <- emotion + politeness
```

## Usage

Score a CSV of emails from the repository root:

```bash
cd analysis/persuasion/wvae/code
python score_email_csv.py \
  --input-csv /path/to/input.csv \
  --output-csv /path/to/output.csv \
  --model-path /path/to/cialdini_wvae_model.pkl
```

Expected input columns are `subject` and `body`; the output adds six `principle_*` probability columns.

Model weights and raw scoring inputs are not included in this public release. Therefore, the fully public reproduction path for the paper's downstream persuasion statistics starts from the released scores under `results/persuasion_scores/`. The inference code is provided so the scoring step can also be rerun when the required model artifact is available.
