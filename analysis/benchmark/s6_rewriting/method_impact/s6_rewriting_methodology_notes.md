# S6 Rewriting Methodology Notes

## Current Evidence Status

Use `s6_rewritten_fn_tp_pair_characteristics.csv` and `fig_s6_rewritten_fn_tp_pair_characteristics.*` as the final S6 failure-characteristic analysis. These compare rewritten FN vs rewritten TP persuasion-pair scores directly and do not depend on original-to-rewrite row pairing.

Do not use the old row-order `delta_pair` failure-association outputs as final controlled original-to-rewrite evidence. The audit in `s6_uta_mpg_controlled_mapping_audit.md` confirms that raw `HW-P.csv`, `UTA-LLM-P.csv`, and `MPG-LLM-P.csv` are row-aligned for a controlled UTA-vs-MPG design, but the current merged analysis file's HW rows are not the same raw HW-P originals.

The corrected UTA-vs-MPG row-id join is in `s6_controlled_uta_mpg_join_summary.md`. It joins 1600/1600 UTA rewrites and 1600/1600 MPG rewrites back to scored rows by normalized subject+body text.

## D_m and Fig. 5 definition

In `generate_s6_rewriting_method_impact.py`, each persuasion-pair score is computed from the six principle scores. Diagonal pairs use the principle score itself; off-diagonal pairs use the product of the two principle scores.

`D_m` in `s6_method_overall_persuasion_shift.csv` should be treated as a group-level persuasion shift, not a per-sample paired rewrite shift.

`D_m` in `s6_method_overall_persuasion_shift.csv` is best read as `mean_p | mean_rewritten(r_p) - mean_original(r_p) |`: the mean, over 21 persuasion pairs, of the absolute group-level rewritten-minus-original mean difference for each pair. The separate `mean_abs_sample_pair_shift` column came from the old row-order pairing and should not be used as a final paired statistic.

`fig_s6_sample_persuasion_shift_by_outcome` and the old `s6_failure_associated_pair_changes.csv` were based on row-order original-to-rewrite deltas. Keep them only as exploratory/deprecated artifacts unless a correct row_id join is regenerated.

For final FN-vs-TP characteristics, use `s6_rewritten_fn_tp_pair_characteristics.csv`: it compares rewritten pair scores between FN and TP samples within each method-detector pair. FDR correction is applied separately within each method-detector family over the 21 pair tests.

## Final Fuzzer Scarcity Pattern

From `s6_rewritten_fn_tp_pair_characteristics.csv`, Fuzzer remains the external setting where Scarcity-related pairs most strongly distinguish rewritten FN from rewritten TP samples.

Use `fig_s6_rewriting_two_panel_main.*` as the main S6 evidence-chain figure. Panel (a) shows rewriting-induced persuasion differences for all 21 persuasion pairs, with rows Fuzzer/UTA/MPG. Panel (b) uses the identical 21-pair x-axis and shows FN-vs-TP rank-biserial effects for Fuzzer/UTA/MPG crossed with SecureNet/V3; stars indicate FDR-adjusted `q < 0.05`.

The intended reading is column-wise. For `S-S` and `A-S`, Panel (a) shows Fuzzer increases the pair while UTA and MPG decrease it from similar baselines; Panel (b) shows the same pairs strongly separate Fuzzer FN from TP for both detectors, but not UTA/MPG. `SP-S` is a supporting result because MPG-SecureNet is also significant.

| Detector | Pair | FN median score | TP median score | FN-TP median | q | Effect |
|---|---|---:|---:|---:|---:|---:|
| SecureNet | S-S | 0.0721 | 0.0387 | +0.0335 | 2.21e-62 | +0.418 |
| SecureNet | A-S | 0.0719 | 0.0384 | +0.0335 | 2.21e-62 | +0.417 |
| SecureNet | SP-S | 0.0277 | 0.0108 | +0.0169 | 1.62e-57 | +0.400 |
| V3 | S-S | 0.0670 | 0.0418 | +0.0252 | 8.66e-34 | +0.305 |
| V3 | A-S | 0.0669 | 0.0416 | +0.0253 | 1.61e-33 | +0.302 |
| V3 | SP-S | 0.0241 | 0.0125 | +0.0116 | 1.86e-29 | +0.282 |

## Independence check

| Method | Paired rows | Unique originals | Max rewrites/original within method | Repeated originals within method | Originals shared with other methods |
|---|---:|---:|---:|---:|---:|
| Fuzzer | 2200 | 2200 | 1 | 0 | 737 |
| UTA | 1600 | 1600 | 1 | 0 | 672 |
| MPG | 1600 | 1600 | 1 | 0 | 599 |

Interpretation guardrail: these analyses support co-occurrence between rewriting-induced persuasion-pair changes and detector FN outcomes. They should not be worded as causal evidence unless a controlled same-original generation design or original-level robustness analysis is added.

## Controlled UTA-vs-MPG Next Join

The corrected join now supports same-original UTA-vs-MPG rewritten comparisons:

- Joined row table: `s6_controlled_uta_mpg_joined_rows.csv`
- Paired UTA-vs-MPG persuasion-score comparison: `s6_controlled_uta_mpg_persuasion_comparison.csv`
- Paired rewritten detector outcome cells: `s6_controlled_uta_mpg_detector_outcome_cells.csv`
- Missing original-baseline audit: `s6_controlled_original_transition_join_audit.csv`

This is a controlled comparison between UTA and MPG rewrites for the same raw row_id. It is not yet an original-to-rewrite transition analysis because raw HW-P original persuasion scores and SecureNet/V3 predictions are still missing from the current scored outputs.

For full Q2-Q4 original-to-rewrite deltas/transitions, join raw row IDs to original scores and predictions:

`row_id | original_text | UTA_text | MPG_text | original_pair_scores | UTA_pair_scores | MPG_pair_scores | original_predictions | UTA_predictions | MPG_predictions`

Then compute paired UTA-vs-MPG pair deltas, original/UTA/MPG detector transitions, McNemar tests for binary prediction changes, and bootstrap confidence intervals for MCC/FNR changes.

## External Fuzzer Wording

Fuzzer can keep its group-level observations, including `D_m = 0.0680`, SecureNet MCC `0.4557`, and V3 MCC `0.4545`, but it should be described as an external setting. Do not use those values to make a causal ranking against the controlled UTA/MPG comparison.

## S6 Fuzzer Action/Linguistic Follow-up

Use `s6_fuzzer_fn_tp_action_characteristics.csv` and `fig_s6_fuzzer_fn_tp_action_characteristics.*` for the S8-style action/linguistic characteristic check on S6 Fuzzer detector misses. This analysis reuses the same regex feature framework as S8, ordered as Urgency, Login/account, Information submission, Click/open, Explicit action, Direct URL/page instruction, Conversational wording, and Softened request.

The current local files do not contain a verified Fuzzer original-to-rewrite mapping key. `s6_fuzzer_original_rewrite_mapping_audit.md` documents the inspected files and shows text overlap only, with no durable `sample_id`, `parent_id`, or `original_subject`/`original_body` relation. Therefore the paired original/rewrite prevalence table and the S-S/A-S/SP-S scarcity-mechanism matrix should not be computed from row order.
