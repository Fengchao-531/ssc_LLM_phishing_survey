# scamllm S8

This folder contains detector-specific stage visualization assets for `scamllm`.

## Files

- `surrogate_response_tp_fn_map.png`: surrogate response map for this detector and stage group.
- `fn_selected_group_boxplots*.png`: HW-P FN vs LLM-P FN boxplots for selected persuasion principle groups.
- `fn_selected_group_boxplot_values.csv`: aggregate numeric values used to draw the boxplot. No raw email text is included.
- `stage_summary.json`: aggregate counts for this detector-stage group.

Stages included: `S8-claude, S8-deepseek, S8-gemini, S8-gpt, S8-llama, S8-ministral`
