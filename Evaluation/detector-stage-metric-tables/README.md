# Detector Stage Metric Tables

This directory contains one CSV per detector, split into two groups:

- `academic/`
- `industry/`

Each CSV is organized as:

- Rows: stages
- Left half: `HW` / `GD` metrics
- Right half: `LLM` metrics
- Metric order:
  - `precision`
  - `recall`
  - `accuracy`
  - `fpr`
  - `fnr`
  - `f0_5`
  - `f1`
  - `f2`
  - `mcc`

Blank cells mean that detector-stage-source combination does not currently have usable predictions.

## Quick Read

The cleanest high-level reading is:

- Academic detectors are much more complete than Industry detectors right now.
- In Academic results, `securenet_llama` is the strongest and most stable detector overall, especially on `HW/GD`.
- In Industry results, no single detector dominates across both `HW/GD` and `LLM`; coverage is uneven and conclusions should be made detector by detector.
- `spamassassin` currently looks promising on `HW/GD`, but it only has partial coverage so far.

## Academic Trends

Current Academic detectors:

- `scamllm`
- `pimref`
- `t5phishing`
- `ml_watermark_logreg`
- `xgboost`
- `securenet_llama`

Overall patterns:

- `securenet_llama` is the strongest Academic detector on `HW/GD` across all currently available stages. It is the stage winner by `HW F1` for every stage from `S1` to `S8-ministral`.
- `scamllm` is the most consistently strong alternative to `securenet_llama`. Its average `HW F1` and `LLM F1` are both solid, and it remains competitive on later `LLM` stages.
- `xgboost` is moderate and stable, but clearly below the top tier.
- `ml_watermark_logreg` has very high recall but weak specificity. It catches many positives, but the high `FPR` pulls down `accuracy` and `MCC`, especially on `HW/GD`.
- `pimref` is extremely conservative. It has high precision but recall is near zero on both `HW` and `LLM`, so it misses most phishing examples.
- `t5phishing` behaves like an always-positive detector in the current outputs: recall is `1.0`, `FPR` is `1.0`, and `MCC` stays at `0.0`. This makes it unsuitable as a balanced detector unless its pipeline is revisited.

Cross-source trend:

- For the stronger Academic detectors, `HW/GD` is generally easier than `LLM`.
- The biggest `HW` vs `LLM` gap appears in `securenet_llama`, which stays excellent on `HW/GD` but drops noticeably on `LLM`.
- `scamllm` and `xgboost` also decline on `LLM`, but less sharply.

Practical takeaway:

- If you want one Academic detector to represent the strongest current baseline, use `securenet_llama`.
- If you want a second strong comparator, use `scamllm`.
- `pimref` and `t5phishing` look much less suitable for main-table comparison in their current form.

## Industry Trends

Current Industry detectors:

- `llm_guard`
- `phishing_email_agent`
- `email_phishing_detection_v3`
- `pyrit_original`
- `pyrit_blocklist`
- `spamassassin`
- `oopspam`

Coverage matters a lot here:

- `llm_guard` and `phishing_email_agent` have the broadest usable coverage.
- `email_phishing_detection_v3` currently has usable results only for a few `LLM` stages.
- `pyrit_blocklist` has only `LLM` coverage in the current evaluation tables.
- `spamassassin` currently has only partial `HW/GD` coverage.
- `pyrit_original` and `oopspam` currently have no usable predictions in these tables.

Detector-by-detector trends:

- `llm_guard` has broad coverage, but its behavior is unstable. On `LLM`, mean `MCC` is negative and `FPR` is high, which suggests a poor balance between positive and negative predictions. It still wins several later `LLM` stages by `F1`, but that advantage comes with weak calibration and many false positives.
- `phishing_email_agent` is the best broadly-covered Industry detector on `LLM`. It is especially strong on `S4` and `S5`, and its average `LLM F1` and `LLM MCC` are better than `llm_guard`.
- `phishing_email_agent` is weaker on `HW/GD` because recall is low there. It is precise, but misses many positives.
- `email_phishing_detection_v3` looks very strong where it has usable `LLM` results, with the best average `LLM F1` and `LLM MCC` among currently populated Industry detectors. The limitation is coverage: it should be treated as partial evidence, not a full-coverage winner.
- `pyrit_blocklist` is high-precision but very low-recall on `LLM`. It flags relatively few items and misses many phishing samples.
- `spamassassin` is promising on `HW/GD`: high precision, low `FPR`, and good `MCC` on the two stages currently populated. Right now it should be described as an encouraging partial result rather than a mature full benchmark result.

Cross-source trend:

- Industry results are not yet balanced enough to support one single “best overall” detector across both `HW/GD` and `LLM`.
- If the goal is broad-coverage comparison, `phishing_email_agent` is the most persuasive `LLM` detector at the moment.
- If the goal is precision-oriented `HW/GD` filtering, `spamassassin` is worth highlighting once more stages are filled in.

## Suggested Narrative

If you want a simple message for presentation or paper drafting, this is the most defensible version right now:

- Academic detectors are more mature and more complete than Industry detectors in the current evaluation snapshot.
- `securenet_llama` is the strongest overall Academic detector, with `scamllm` as the clearest secondary baseline.
- In Industry detectors, `phishing_email_agent` is the most credible broad-coverage detector on `LLM` data.
- `email_phishing_detection_v3` and `spamassassin` are both promising, but they currently need fuller coverage before being presented as overall winners.
- `llm_guard` has wide coverage but tends to over-flag, which hurts `MCC` and makes it less reliable as a balanced classifier.

## Files To Read First

If you want to inspect the most informative tables first:

- `academic/securenet_llama.csv`
- `academic/scamllm.csv`
- `industry/phishing_email_agent.csv`
- `industry/llm_guard.csv`
- `industry/spamassassin.csv`
- `industry/email_phishing_detection_v3.csv`
