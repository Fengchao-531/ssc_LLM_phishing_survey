# Git Interaction Visualization Assets

This directory stores the public, GitHub-friendly assets for the interactive detector visualization.

Each detector folder contains:

- `overview/`: detector-level surrogate map and shared overview heatmap.
- `S1/`, `S2/`, `S4/`, `S5/`: stage-specific surrogate maps and FN persuasion boxplots.
- `S6/`: combined S6-MPG, S6-UTA, and S6-fuzzer assets.
- `S8/`: combined S8 model-output assets.

Raw email text and row-level detector outputs are intentionally excluded. The CSV files here contain only numeric visualization values.

| Detector | Family | Projected rows |
| --- | --- | ---: |
| `scamllm` | `academic` | `55095` |
| `pimref` | `academic` | `55095` |
| `t5phishing` | `academic` | `55095` |
| `xgboost` | `academic` | `55095` |
| `securenet_llama` | `academic` | `55079` |
| `email_phishing_detection_v3` | `industry` | `28515` |
| `phishing_email_agent` | `industry` | `28104` |
| `rspamd` | `industry` | `29747` |
| `spamassassin` | `industry` | `23519` |
| `spamscanner` | `industry` | `29680` |
