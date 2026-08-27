# S6 UTA-MPG Controlled Mapping Audit

Raw generation CSVs confirm the controlled text design: `HW-P.csv`, `UTA-LLM-P.csv`, and `MPG-LLM-P.csv` each contain 1600 phishing rows, and UTA/MPG row `i` is a rewrite of HW-P row `i`.

This supports a future same-original UTA-vs-MPG analysis once row IDs are joined to persuasion scores and detector predictions.

Important guardrail: the current merged analysis file's `S6-UTA` and `S6-MPG` HW rows are not a reliable same-original baseline for that controlled analysis. Do not use the old row-order `delta_pair` outputs as final controlled original-to-rewrite evidence.

## Required Join For Q2-Q4

Create or export a row-level table with at least `row_id`, `original_text`, `UTA_text`, `MPG_text`, the 21 original/UTA/MPG pair scores, and SecureNet/V3 predictions for original, UTA, and MPG. Then compute paired UTA-vs-MPG persuasion deltas and TP->FN transitions.

## Audit CSV

`s6_uta_mpg_controlled_mapping_audit.csv`
