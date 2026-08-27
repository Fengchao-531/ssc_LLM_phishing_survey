# S8 paired design RQ1-RQ2

Fixed seed: `20260825`. Common prompt set is phishing-labelled prompts present for all six generators after one output per prompt is selected.

## Step 0 dataset audit

| Generator | Raw outputs | Unique prompts | Repeated outputs removed | Missing/invalid | Final common N |
| --- | ---: | ---: | ---: | ---: | ---: |
| Claude | 499 | 499 | 0 | 0 | 499 |
| GPT | 500 | 500 | 0 | 0 | 499 |
| Gemini | 514 | 500 | 14 | 0 | 499 |
| Llama | 1500 | 500 | 1000 | 0 | 499 |
| Ministral | 1500 | 500 | 1000 | 0 | 499 |
| DeepSeek | 1500 | 500 | 1000 | 0 | 499 |

## RQ1 detector outcome Cochran Q

| Detector | Claude | GPT | Gemini | Llama | Ministral | DeepSeek | Q | q |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| ScamLLM | 80.0% | 61.7% | 68.9% | 77.2% | 77.8% | 47.7% | 249.22 | 1.21e-51 |
| PiMRef | 15.0% | 13.2% | 3.4% | 3.6% | 5.8% | 1.4% | 123.76 | 5.01e-25 |
| T5 | 83.0% | 69.9% | 78.0% | 52.9% | 51.1% | 56.7% | 214.86 | 2.25e-44 |
| XGBoost | 90.4% | 69.7% | 79.0% | 58.3% | 61.7% | 87.8% | 268.62 | 1.11e-55 |
| SecureNet | 77.0% | 63.5% | 69.5% | 61.9% | 63.5% | 7.4% | 759.25 | 4.52e-161 |
| PhishingV3 | 45.9% | 41.0% | 38.9% | 40.4% | 41.8% | 0.4% | 437.99 | 5.75e-92 |

## Selected RQ1 McNemar pairs

| Detector | A | B | A-B | p | q |
| --- | --- | --- | ---: | ---: | ---: |
| SecureNet | Llama | Claude | -15.0 pp | 2.41e-08 | 3.92e-08 |
| SecureNet | Llama | GPT | -1.6 pp | 0.612 | 0.663 |
| SecureNet | Llama | Gemini | -7.6 pp | 0.0054 | 0.00702 |
| SecureNet | Llama | DeepSeek | +54.5 pp | 4.76e-73 | 3.09e-72 |
| SecureNet | Ministral | Claude | -13.4 pp | 1.61e-06 | 2.33e-06 |
| SecureNet | Ministral | GPT | +0.0 pp | 1 | 1 |
| SecureNet | Ministral | Gemini | -6.0 pp | 0.0351 | 0.0415 |
| SecureNet | Ministral | DeepSeek | +56.1 pp | 2.26e-75 | 2.93e-74 |
| PhishingV3 | DeepSeek | Claude | -45.5 pp | 1.67e-65 | 7.23e-65 |
| PhishingV3 | DeepSeek | GPT | -40.6 pp | 2.5e-58 | 5.42e-58 |
| PhishingV3 | DeepSeek | Gemini | -38.5 pp | 2.43e-55 | 4.52e-55 |
| PhishingV3 | DeepSeek | Llama | -40.0 pp | 3.98e-59 | 1.04e-58 |
| PhishingV3 | DeepSeek | Ministral | -41.4 pp | 3.11e-61 | 1.01e-60 |

## RQ2 feature Cochran Q

| Feature | Claude | GPT | Gemini | Llama | Ministral | DeepSeek | Q | q | Range |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Direct URL/page instruction | 71.3% | 37.1% | 83.6% | 57.5% | 47.9% | 82.0% | 453.93 | 5.57e-95 | 46.5 pp |
| Click/open request | 53.9% | 23.0% | 59.5% | 54.5% | 39.5% | 32.5% | 260.80 | 1.06e-53 | 36.5 pp |
| Urgency wording | 63.9% | 49.5% | 50.7% | 51.3% | 42.5% | 58.9% | 85.53 | 1.56e-16 | 21.4 pp |
| Information submission | 53.3% | 46.9% | 45.9% | 57.5% | 47.7% | 61.9% | 80.29 | 1.46e-15 | 16.0 pp |
| Explicit action request | 86.2% | 78.4% | 87.6% | 89.2% | 84.6% | 90.6% | 57.81 | 3.93e-11 | 12.2 pp |
| Login/account action | 64.1% | 52.9% | 58.7% | 59.3% | 54.1% | 52.7% | 62.78 | 4.31e-12 | 11.4 pp |
| Softened request | 70.1% | 73.3% | 76.4% | 80.6% | 81.0% | 72.5% | 64.25 | 2.57e-12 | 10.8 pp |
| Conversational wording | 89.6% | 84.8% | 92.2% | 90.8% | 89.6% | 93.2% | 38.05 | 3.69e-07 | 8.4 pp |
