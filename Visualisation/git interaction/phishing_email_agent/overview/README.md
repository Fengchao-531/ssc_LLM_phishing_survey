# phishing_email_agent Overview

This folder contains public overview visualization assets for `phishing_email_agent`.

## Files

- `surrogate_response_tp_fn_map.png`: detector-specific surrogate response map generated from the shared PCA projection.
- `group1_difference_heatmaps.png`: shared overview difference heatmap from the final visualization pipeline.
- `group1_difference_heatmaps_p_values.csv`: p-value table for the shared overview heatmap.
- `overview_summary.json`: aggregate counts used to sanity-check this detector overview.

No raw email text or row-level detector outputs are exported in this folder.

## Summary

- Detector family: `industry`
- Prediction column: `phishing_email_agent_prediction`
- Projected rows used: `28104`
