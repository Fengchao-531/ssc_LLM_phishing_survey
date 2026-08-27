# S6 Fuzzer action/linguistic characteristics

This analysis reuses the S8 regex-based action/linguistic feature framework, with the x-axis ordered as: Urgency, Login/account, Information submission, Click/open, Explicit action, Direct URL/page instruction, Conversational wording, Softened request.

## Completed output

- `s6_fuzzer_action_characteristics_input_rows.csv`: 2,200 S6-fuzzer LLM phishing rows with persuasion scores, detector predictions, and 8 binary characteristics.
- `s6_fuzzer_fn_tp_action_characteristics.csv`: FN-vs-TP characteristic comparison for SecureNet and V3.
- `fig_s6_fuzzer_fn_tp_action_characteristics.png` / `.pdf`: detector by characteristic heatmap, cell value = FN minus TP prevalence in percentage points, stars = BH-FDR q<0.05.

## Mapping status

The current local files do not expose a verified Fuzzer original-to-rewrite key. Therefore the requested paired original/rewrite prevalence table and scarcity-mechanism matrix are not computed here; doing so by row order would be an unsupported controlled-pair assumption.

| candidate_file | rows | overlap_with_rewrite_rows | original_fields | id_field | status |
|---|---:|---:|---|---|---|
| fuzzer-LLM-P.csv | 2200 | 2125 | no | no | text-overlap only; no original-to-rewrite key |
| emails_normalized.json | 3300 | 2094 | no | no | text-overlap only; no original-to-rewrite key |
| Evaluation llm/academic/S6-fuzzer.csv | 3300 | 2125 | no | no | text-overlap only; no original-to-rewrite key |
| Evaluation gd/academic/S6-fuzzer.csv | 3300 | 525 | no | no | text-overlap only; no original-to-rewrite key |
| LLM_S6-fuzzer_persuasion.csv | 3300 | 2125 | no | no | text-overlap only; no original-to-rewrite key |
| HW_S6_persuasion.csv | 9700 | 1043 | no | no | text-overlap only; no original-to-rewrite key |

## FN-vs-TP highlights

### SecureNet
- Conversational wording: FN 0.316, TP 0.629, gap -31.3 pp, q=1.59e-47
- Information submission: FN 0.074, TP 0.363, gap -28.9 pp, q=1.01e-55
- Login/account: FN 0.326, TP 0.595, gap -26.9 pp, q=1.78e-35

### V3
- Login/account: FN 0.278, TP 0.660, gap -38.2 pp, q=1.38e-70
- Conversational wording: FN 0.320, TP 0.649, gap -33.0 pp, q=3.19e-53
- Information submission: FN 0.072, TP 0.386, gap -31.5 pp, q=1.49e-66
