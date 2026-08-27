# S8 Detector Score Shift Summary

Scores are detector outputs on LLM-generated phishing samples in S8. For binary detector outputs, the mean score equals the detector phishing rate on that generator.

## Mean-score shift by detector

| Detector | Min LLM | Min mean | Max LLM | Max mean | R_d | Median W1 | Max W1 | Max-W1 pair |
|---|---:|---:|---:|---:|---:|---:|---:|---|
| SecureNet | DeepSeek | 0.0760 | Claude | 0.7695 | 0.6935 | 0.1355 | 0.6935 | Claude vs DeepSeek |
| V3 | DeepSeek | 0.0047 | Claude | 0.4589 | 0.4543 | 0.0509 | 0.4543 | Claude vs DeepSeek |
| ScamLLM | DeepSeek | 0.4780 | Claude | 0.7996 | 0.3216 | 0.1380 | 0.3216 | Claude vs DeepSeek |
| T5Phishing | Ministral | 0.5200 | Claude | 0.8297 | 0.3097 | 0.1680 | 0.3097 | Claude vs Ministral |
| XGBoost | Llama | 0.5967 | Claude | 0.9038 | 0.3071 | 0.1767 | 0.3071 | Claude vs Llama |
| PiMRef | DeepSeek | 0.0113 | Claude | 0.1503 | 0.1390 | 0.0767 | 0.1390 | Claude vs DeepSeek |

## Per-detector, per-generator score summaries

### ScamLLM

| LLM | n | Mean | Median | IQR |
|---|---:|---:|---:|---:|
| Claude | 499 | 0.7996 | 1.0000 | 0.0000 |
| GPT | 500 | 0.6160 | 1.0000 | 1.0000 |
| Gemini | 514 | 0.6868 | 1.0000 | 1.0000 |
| Llama | 1500 | 0.7653 | 1.0000 | 0.0000 |
| Ministral | 1500 | 0.7773 | 1.0000 | 0.0000 |
| DeepSeek | 1500 | 0.4780 | 0.0000 | 1.0000 |

### PiMRef

| LLM | n | Mean | Median | IQR |
|---|---:|---:|---:|---:|
| Claude | 499 | 0.1503 | 0.0000 | 0.0000 |
| GPT | 500 | 0.1320 | 0.0000 | 0.0000 |
| Gemini | 514 | 0.0350 | 0.0000 | 0.0000 |
| Llama | 1500 | 0.0433 | 0.0000 | 0.0000 |
| Ministral | 1500 | 0.0553 | 0.0000 | 0.0000 |
| DeepSeek | 1500 | 0.0113 | 0.0000 | 0.0000 |

### T5Phishing

| LLM | n | Mean | Median | IQR |
|---|---:|---:|---:|---:|
| Claude | 499 | 0.8297 | 1.0000 | 0.0000 |
| GPT | 500 | 0.6980 | 1.0000 | 1.0000 |
| Gemini | 514 | 0.7821 | 1.0000 | 0.0000 |
| Llama | 1500 | 0.5300 | 1.0000 | 1.0000 |
| Ministral | 1500 | 0.5200 | 1.0000 | 1.0000 |
| DeepSeek | 1500 | 0.5500 | 1.0000 | 1.0000 |

### XGBoost

| LLM | n | Mean | Median | IQR |
|---|---:|---:|---:|---:|
| Claude | 499 | 0.9038 | 1.0000 | 0.0000 |
| GPT | 500 | 0.6960 | 1.0000 | 1.0000 |
| Gemini | 514 | 0.7918 | 1.0000 | 0.0000 |
| Llama | 1500 | 0.5967 | 1.0000 | 1.0000 |
| Ministral | 1500 | 0.6127 | 1.0000 | 1.0000 |
| DeepSeek | 1500 | 0.8727 | 1.0000 | 0.0000 |

### SecureNet

| LLM | n | Mean | Median | IQR |
|---|---:|---:|---:|---:|
| Claude | 499 | 0.7695 | 1.0000 | 0.0000 |
| GPT | 500 | 0.6340 | 1.0000 | 1.0000 |
| Gemini | 514 | 0.6946 | 1.0000 | 1.0000 |
| Llama | 1500 | 0.6247 | 1.0000 | 1.0000 |
| Ministral | 1500 | 0.6260 | 1.0000 | 1.0000 |
| DeepSeek | 1500 | 0.0760 | 0.0000 | 0.0000 |

### V3

| LLM | n | Mean | Median | IQR |
|---|---:|---:|---:|---:|
| Claude | 499 | 0.4589 | 0.0000 | 1.0000 |
| GPT | 500 | 0.4080 | 0.0000 | 1.0000 |
| Gemini | 514 | 0.3911 | 0.0000 | 1.0000 |
| Llama | 1500 | 0.4020 | 0.0000 | 1.0000 |
| Ministral | 1500 | 0.4220 | 0.0000 | 1.0000 |
| DeepSeek | 1500 | 0.0047 | 0.0000 | 0.0000 |

