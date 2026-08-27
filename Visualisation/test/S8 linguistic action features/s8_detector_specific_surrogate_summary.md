# S8 Detector-Specific Continuous Surrogate Scores

`s_d(x)` is estimated as `Pr(detector d predicts phishing | proj_x, proj_y)` using a degree-2 logistic surrogate surface with balanced class weights.

## Mean-score shift

| Detector | Min LLM | Min mean | Max LLM | Max mean | R_d | Median W1 | Max W1 | Max-W1 pair |
|---|---:|---:|---:|---:|---:|---:|---:|---|
| XGBoost | GPT | 0.4724 | Gemini | 0.6128 | 0.1404 | 0.0697 | 0.1406 | GPT vs Gemini |
| SecureNet | DeepSeek | 0.3908 | GPT | 0.4691 | 0.0784 | 0.0364 | 0.0784 | GPT vs DeepSeek |
| PiMRef | Ministral | 0.5509 | DeepSeek | 0.6171 | 0.0662 | 0.0291 | 0.0662 | Ministral vs DeepSeek |
| ScamLLM | GPT | 0.4643 | Gemini | 0.5275 | 0.0632 | 0.0316 | 0.0635 | GPT vs Gemini |
| V3 | Ministral | 0.4937 | GPT | 0.5212 | 0.0275 | 0.0255 | 0.0423 | GPT vs DeepSeek |
| T5Phishing | GPT | 0.4945 | DeepSeek | 0.5053 | 0.0108 | 0.0051 | 0.0110 | GPT vs DeepSeek |

## Surrogate fit quality

| Detector | n | Positive rate | ROC-AUC | AP | Brier |
|---|---:|---:|---:|---:|---:|
| ScamLLM | 35794 | 0.8249 | 0.6564 | 0.8949 | 0.2312 |
| PiMRef | 35794 | 0.0378 | 0.7219 | 0.0802 | 0.2226 |
| T5Phishing | 35794 | 0.6075 | 0.5112 | 0.6158 | 0.2499 |
| XGBoost | 35794 | 0.6557 | 0.7529 | 0.8530 | 0.2033 |
| SecureNet | 35791 | 0.7625 | 0.7267 | 0.8988 | 0.2208 |
| V3 | 35794 | 0.5568 | 0.6573 | 0.6963 | 0.2319 |

## Per-generator summaries

### ScamLLM

| LLM | n | Mean | Median | IQR |
|---|---:|---:|---:|---:|
| Claude | 499 | 0.5262 | 0.5311 | 0.0718 |
| GPT | 500 | 0.4643 | 0.4666 | 0.1142 |
| Gemini | 514 | 0.5275 | 0.5377 | 0.0751 |
| Llama | 1500 | 0.4953 | 0.4954 | 0.0784 |
| Ministral | 1500 | 0.4798 | 0.4864 | 0.0907 |
| DeepSeek | 1500 | 0.5205 | 0.5158 | 0.0540 |

### PiMRef

| LLM | n | Mean | Median | IQR |
|---|---:|---:|---:|---:|
| Claude | 499 | 0.6028 | 0.6551 | 0.1564 |
| GPT | 500 | 0.5598 | 0.5824 | 0.1555 |
| Gemini | 514 | 0.5984 | 0.6441 | 0.1664 |
| Llama | 1500 | 0.5880 | 0.6187 | 0.1408 |
| Ministral | 1500 | 0.5509 | 0.5742 | 0.1856 |
| DeepSeek | 1500 | 0.6171 | 0.6215 | 0.1281 |

### T5Phishing

| LLM | n | Mean | Median | IQR |
|---|---:|---:|---:|---:|
| Claude | 499 | 0.5045 | 0.5055 | 0.0071 |
| GPT | 500 | 0.4945 | 0.4939 | 0.0124 |
| Gemini | 514 | 0.5049 | 0.5061 | 0.0087 |
| Llama | 1500 | 0.4999 | 0.5016 | 0.0104 |
| Ministral | 1500 | 0.4994 | 0.5005 | 0.0114 |
| DeepSeek | 1500 | 0.5053 | 0.5062 | 0.0049 |

### XGBoost

| LLM | n | Mean | Median | IQR |
|---|---:|---:|---:|---:|
| Claude | 499 | 0.6116 | 0.6381 | 0.2004 |
| GPT | 500 | 0.4724 | 0.4683 | 0.2145 |
| Gemini | 514 | 0.6128 | 0.6263 | 0.2283 |
| Llama | 1500 | 0.5419 | 0.5348 | 0.2007 |
| Ministral | 1500 | 0.5018 | 0.5060 | 0.2105 |
| DeepSeek | 1500 | 0.6062 | 0.5782 | 0.1579 |

### SecureNet

| LLM | n | Mean | Median | IQR |
|---|---:|---:|---:|---:|
| Claude | 499 | 0.4322 | 0.3946 | 0.0855 |
| GPT | 500 | 0.4691 | 0.4433 | 0.1095 |
| Gemini | 514 | 0.4327 | 0.3945 | 0.0950 |
| Llama | 1500 | 0.4344 | 0.3989 | 0.0991 |
| Ministral | 1500 | 0.4286 | 0.3972 | 0.0907 |
| DeepSeek | 1500 | 0.3908 | 0.3727 | 0.0508 |

### V3

| LLM | n | Mean | Median | IQR |
|---|---:|---:|---:|---:|
| Claude | 499 | 0.5212 | 0.5144 | 0.1199 |
| GPT | 500 | 0.5212 | 0.5009 | 0.1458 |
| Gemini | 514 | 0.5184 | 0.5071 | 0.1276 |
| Llama | 1500 | 0.5180 | 0.5013 | 0.1256 |
| Ministral | 1500 | 0.4937 | 0.4743 | 0.1178 |
| DeepSeek | 1500 | 0.5043 | 0.4906 | 0.0856 |
