# S8 RQ3 paired analysis

TP/FN feature differences use the 499-prompt paired dataset. Fisher tests are FDR-corrected within detector.

## Sparse cell

PhishingV3 x DeepSeek has sparse TP and is descriptive only in TP/FN feature analysis; DeepSeek is excluded from PhishingV3 adjusted GEE.

## Adjustment summary

| Detector | Raw range | Adjusted range | Interpretation |
| --- | ---: | ---: | --- |
| ScamLLM | 32.3 pp | 34.7 pp | generator variation remains strong after feature adjustment |
| PiMRef | 13.6 pp | 14.2 pp | generator variation remains strong after feature adjustment |
| T5 | 31.9 pp | 30.3 pp | generator variation remains strong after feature adjustment |
| XGBoost | 32.1 pp | 31.0 pp | generator variation remains strong after feature adjustment |
| SecureNet | 69.5 pp | 67.4 pp | generator variation remains strong after feature adjustment |
| PhishingV3 | 7.1 pp | 6.7 pp | among the five non-DeepSeek generators, feature adjustment changes generator variation only slightly; DeepSeek retained in RQ1 but excluded from adjusted model due to sparse TP |
