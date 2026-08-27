# S8 detector TP/FN feature prevalence

This table links generator, detector outcome, and action/linguistic characteristics for phishing-labelled S8 LLM outputs.

## Detector-generator sample counts

| Detector | Generator | N | TP | FN |
| --- | --- | ---: | ---: | ---: |
| SecureNet | Claude | 499 | 384 | 115 |
| SecureNet | GPT | 500 | 317 | 183 |
| SecureNet | Gemini | 514 | 357 | 157 |
| SecureNet | Llama | 1500 | 937 | 563 |
| SecureNet | Ministral | 1500 | 939 | 561 |
| SecureNet | DeepSeek | 1500 | 114 | 1386 |
| PhishingV3 | Claude | 496 | 229 | 267 |
| PhishingV3 | GPT | 499 | 204 | 295 |
| PhishingV3 | Gemini | 509 | 201 | 308 |
| PhishingV3 | Llama | 1492 | 603 | 889 |
| PhishingV3 | Ministral | 1500 | 633 | 867 |
| PhishingV3 | DeepSeek | 1495 | 7 | 1488 |

## Largest TP-FN feature differences

| Detector | Generator | Feature | TP prev. | FN prev. | TP-FN | p | q |
| --- | --- | --- | ---: | ---: | ---: | ---: | ---: |
| PhishingV3 | Claude | Login/account action | 85.2% | 46.1% | +39.1 pp | 1.46926e-19 | 5.87702e-19 |
| PhishingV3 | Claude | Urgency wording | 82.5% | 47.9% | +34.6 pp | 1.27199e-15 | 4.52264e-15 |
| PhishingV3 | Claude | Information submission | 69.9% | 39.0% | +30.9 pp | 5.99406e-12 | 1.51429e-11 |
| PhishingV3 | DeepSeek | Urgency wording | 100.0% | 60.4% | +39.6 pp | 0.0468712 | 0.0592058 |
| PhishingV3 | DeepSeek | Login/account action | 85.7% | 52.1% | +33.6 pp | 0.126759 | 0.146613 |
| PhishingV3 | DeepSeek | Click/open request | 57.1% | 28.8% | +28.4 pp | 0.111664 | 0.130728 |
| PhishingV3 | Gemini | Login/account action | 79.1% | 44.8% | +34.3 pp | 1.67881e-14 | 5.55746e-14 |
| PhishingV3 | Gemini | Urgency wording | 71.1% | 37.3% | +33.8 pp | 8.82484e-14 | 2.64745e-13 |
| PhishingV3 | Gemini | Information submission | 61.7% | 35.4% | +26.3 pp | 5.80766e-09 | 1.26713e-08 |
| PhishingV3 | GPT | Urgency wording | 74.0% | 32.2% | +41.8 pp | 4.10385e-20 | 1.71291e-19 |
| PhishingV3 | GPT | Login/account action | 72.1% | 39.3% | +32.7 pp | 5.98761e-13 | 1.62864e-12 |
| PhishingV3 | GPT | Information submission | 62.3% | 36.3% | +26.0 pp | 1.07763e-08 | 2.29895e-08 |
| PhishingV3 | Llama | Login/account action | 85.1% | 43.6% | +41.4 pp | 5.13956e-58 | 4.93398e-56 |
| PhishingV3 | Llama | Information submission | 76.6% | 42.3% | +34.3 pp | 2.8229e-39 | 6.77496e-38 |
| PhishingV3 | Llama | Urgency wording | 66.0% | 40.2% | +25.8 pp | 1.13818e-22 | 5.75083e-22 |
| PhishingV3 | Ministral | Login/account action | 75.2% | 37.3% | +37.9 pp | 6.16834e-48 | 2.9608e-46 |
| PhishingV3 | Ministral | Urgency wording | 59.7% | 29.6% | +30.1 pp | 2.50922e-31 | 3.44121e-30 |
| PhishingV3 | Ministral | Information submission | 63.0% | 38.2% | +24.9 pp | 1.86961e-21 | 8.54677e-21 |
| SecureNet | Claude | Login/account action | 76.6% | 22.6% | +54.0 pp | 3.59514e-26 | 2.65487e-25 |
| SecureNet | Claude | Information submission | 65.4% | 13.0% | +52.3 pp | 5.86901e-23 | 3.13014e-22 |
| SecureNet | Claude | Softened request | 81.2% | 33.0% | +48.2 pp | 3.79254e-23 | 2.27552e-22 |
| SecureNet | DeepSeek | Login/account action | 76.3% | 50.1% | +26.2 pp | 7.57243e-08 | 1.51449e-07 |
| SecureNet | DeepSeek | Click/open request | 44.7% | 27.6% | +17.1 pp | 0.000108332 | 0.00016774 |
| SecureNet | DeepSeek | Direct URL/page instruction | 94.7% | 80.4% | +14.3 pp | 0.000157091 | 0.000239377 |
| SecureNet | Gemini | Login/account action | 77.0% | 16.6% | +60.5 pp | 1.29304e-37 | 2.48264e-36 |
| SecureNet | Gemini | Information submission | 63.3% | 7.0% | +56.3 pp | 4.20419e-32 | 6.7267e-31 |
| SecureNet | Gemini | Softened request | 93.6% | 37.6% | +56.0 pp | 3.44774e-43 | 1.10328e-41 |
| SecureNet | GPT | Information submission | 65.9% | 13.7% | +52.3 pp | 1.58549e-29 | 1.69119e-28 |
| SecureNet | GPT | Login/account action | 71.0% | 21.3% | +49.7 pp | 8.53405e-27 | 7.44789e-26 |
| SecureNet | GPT | Softened request | 90.5% | 43.2% | +47.4 pp | 1.05597e-30 | 1.26716e-29 |
| SecureNet | Llama | Information submission | 66.9% | 38.2% | +28.7 pp | 1.85362e-27 | 1.77948e-26 |
| SecureNet | Llama | Login/account action | 70.8% | 43.0% | +27.8 pp | 1.80374e-26 | 1.44299e-25 |
| SecureNet | Llama | Urgency wording | 58.3% | 37.7% | +20.6 pp | 1.05302e-14 | 3.61034e-14 |
| SecureNet | Ministral | Login/account action | 61.8% | 39.0% | +22.7 pp | 1.37066e-17 | 5.26332e-17 |
| SecureNet | Ministral | Urgency wording | 49.8% | 29.8% | +20.1 pp | 2.67827e-14 | 8.57045e-14 |
| SecureNet | Ministral | Information submission | 54.8% | 38.3% | +16.5 pp | 5.85374e-10 | 1.37063e-09 |
