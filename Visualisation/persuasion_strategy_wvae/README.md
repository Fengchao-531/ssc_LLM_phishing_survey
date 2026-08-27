# Persuasion_Strategy_WVAE

This directory starts from the AAAI 2021 WVAE codebase:

*Jiaao Chen, Diyi Yang*: Weakly-Supervised Hierarchical Models for Predicting Persuasive Strategies in Good-faith Textual Requests.

It has been adapted for this survey workspace so we can replace the old heuristic persuasion indicators with:

1. sentence-level WVAE strategy probabilities
2. fixed six Cialdini-style principle scores per email
3. downstream dataset-level heatmaps built from those scores

## What Changed

The original repository predicts request-specific strategy labels:

```
0 Other
1 Credibility
2 Reciprocity
3 Evidence
4 Commitment
5 Scarcity
6 Social Identity
7 Emotion
8 Impact
9 Politeness
```

We now keep those raw labels during training, then aggregate the sentence-level probabilities into six fixed principles for visualization:

```
authority     <- credibility + evidence
reciprocity   <- reciprocity
commitment    <- commitment
scarcity      <- scarcity
social_proof  <- social_identity + impact
liking        <- emotion + politeness
```

This keeps the paper's weakly-supervised training method intact while giving the visualization pipeline a stable six-column output.

## New Scripts

All new logic lives in `code/`:

- `build_cialdini_wvae_dataset.py`
  Builds a combined `borrow + raop` dataset with raw 10-way labels preserved.
- `train_cialdini_wvae.py`
  Wrapper that builds the combined dataset and launches WVAE-BERT training.
- `email_preprocessing.py`
  Email-specific preprocessing with HTML/style cleanup, noisy-fragment filtering, and a `3000`-word budget.
- `score_email_csv.py`
  Scores a CSV of emails and writes per-email six-principle probabilities.
- `run_hw_five_scoring.py`
  Runs scoring over the five HW datasets currently used for the new visualization pass.

## Email Scoring Logic

For each email:

1. `Subject + Body` are cleaned.
2. Sentences are kept in order until the total budget reaches `3000` words.
3. Noisy fragments such as HTML/CSS scraps, token soup, and number-heavy junk are dropped.
4. The trained WVAE sentence classifier produces a raw 10-way probability for each sentence.
5. Sentence probabilities are aggregated into email-level six-principle scores using `noisy-or`.

That means each `principle_*` value is a continuous number in `[0, 1]` representing whether the whole email expresses that principle anywhere in the retained text.

## Commands

Build the combined dataset:

```bash
cd Visualization/persuasion_strategy_wvae/code
python build_cialdini_wvae_dataset.py
```

Train the adapted WVAE model:

```bash
cd Visualization/persuasion_strategy_wvae/code
python train_cialdini_wvae.py --gpu 0
```

Score one email CSV:

```bash
cd Visualization/persuasion_strategy_wvae/code
python score_email_csv.py \
  --input-csv /path/to/input.csv \
  --output-csv /path/to/output.csv \
  --model-path ../output/cialdini_wvae_run/model.pkl
```

Score the five current HW datasets:

```bash
cd Visualization/persuasion_strategy_wvae/code
python run_hw_five_scoring.py \
  --model-path ../output/cialdini_wvae_run/model.pkl
```






