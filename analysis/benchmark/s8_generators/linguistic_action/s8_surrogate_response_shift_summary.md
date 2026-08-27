# S8 Surrogate Response Shift Summary

This table uses the continuous `surrogate_score` stored in each S8 stage's `projected_points.csv`.

## Generator score summaries

| LLM | n | Mean | Median | IQR |
|---|---:|---:|---:|---:|
| Claude | 499 | 0.5600 | 0.5706 | 0.0841 |
| GPT | 500 | 0.4551 | 0.4560 | 0.1520 |
| Gemini | 514 | 0.5636 | 0.5822 | 0.0954 |
| Llama | 1500 | 0.5111 | 0.5206 | 0.1081 |
| Ministral | 1500 | 0.4954 | 0.5110 | 0.1263 |
| DeepSeek | 1500 | 0.5630 | 0.5681 | 0.0679 |

## Cross-generator shift

- Mean-score range: `0.1085` (GPT `0.4551` to Gemini `0.5636`).
- Median pairwise 1-Wasserstein distance: `0.0557`.
- Maximum pairwise 1-Wasserstein distance: `0.1110` (GPT vs DeepSeek).

Suggested wording:

> Across LLM generators, the mean surrogate phishing score varies by 0.108 in the global response space; the median pairwise Wasserstein distance is 0.056.
