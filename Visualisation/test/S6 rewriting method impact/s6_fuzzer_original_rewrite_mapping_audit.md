# S6 Fuzzer original-rewrite mapping audit

Goal: determine whether the current local package supports paired original -> Fuzzer rewrite analysis for S-S/A-S/SP-S scarcity mechanisms.

Conclusion: no verified Fuzzer original-to-rewrite mapping is present in the inspected files. The LLM Fuzzer rows are scored and detector-labeled, but the available candidate files provide text overlap rather than a durable sample_id/parent_id/original_subject/original_body relation.

Implication: CSV1 and CSV2 requested for paired original/rewrite deltas should wait for a true mapping file. The generated FN/TP analysis is valid because it uses only rewritten Fuzzer rows and detector outcomes.

| candidate_file | rows | unique_text_keys | overlap_with_rewrite_rows | original_fields | id_field | status |
|---|---:|---:|---:|---|---|---|
| fuzzer-LLM-P.csv | 2200 | 2125 | 2125 | no | no | text-overlap only; no original-to-rewrite key |
| emails_normalized.json | 3300 | 3225 | 2094 | no | no | text-overlap only; no original-to-rewrite key |
| Evaluation llm/academic/S6-fuzzer.csv | 3300 | 3225 | 2125 | no | no | text-overlap only; no original-to-rewrite key |
| Evaluation gd/academic/S6-fuzzer.csv | 3300 | 3300 | 525 | no | no | text-overlap only; no original-to-rewrite key |
| LLM_S6-fuzzer_persuasion.csv | 3300 | 3225 | 2125 | no | no | text-overlap only; no original-to-rewrite key |
| HW_S6_persuasion.csv | 9700 | 8331 | 1043 | no | no | text-overlap only; no original-to-rewrite key |