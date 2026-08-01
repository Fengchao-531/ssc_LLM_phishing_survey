# Evidence Extraction

This folder stores text evidence derived from the current persuasion-principle outputs.

Current scope:

- `overview/`
  - overall groups only
  - no stage-specific outputs yet

Important note:

- The current evidence is **group-associated evidence**, not exact token-level causal attribution from the persuasion model.
- We rank emails by contribution to a heatmap cell:
  - diagonal cell `(A, A)`: contribution = `principle_A`
  - off-diagonal cell `(A, B)`: contribution = `principle_A * principle_B`
- We then extract representative words, phrases, and sentences from the highest-contribution emails in that group and compare them to background emails.

Outputs per group:

- `group_summary.json`
- `cell_summary.csv`
- `top_words_by_cell.csv`
- `top_phrases_by_cell.csv`
- `top_sentences_by_cell.csv`

Group labels follow the table-friendly format:

- `HW-P-FN`
- `HW-P-TP`
- `HW-B-TN`
- `HW-B-FP`
- `LLM-P-FN`
- `LLM-P-TP`
- `LLM-B-TN`
- `LLM-B-FP`
