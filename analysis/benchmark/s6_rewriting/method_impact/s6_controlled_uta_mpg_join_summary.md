# S6 Controlled UTA-vs-MPG Join

Raw `HW-P.csv`, `UTA-LLM-P.csv`, and `MPG-LLM-P.csv` are joined by row_id. UTA and MPG rewritten rows are then matched back to the current merged scored file by normalized subject+body text.

- UTA rewritten rows joined: 1600/1600
- MPG rewritten rows joined: 1600/1600

Important limitation: raw HW-P original persuasion scores and SecureNet/V3 predictions are not present in the current scored outputs, so original->rewrite TP->FN / TP->TP transitions are not computed here. See `s6_controlled_original_transition_join_audit.csv`.

## UTA vs MPG Paired Persuasion Scores

| Pair | UTA median | MPG median | UTA-MPG median | q | UTA>MPG | MPG>UTA |
|---|---:|---:|---:|---:|---:|---:|
| L-L | 0.9792 | 0.5999 | +0.1299 | 6.05e-76 | 1172 | 427 |
| A-L | 0.8441 | 0.5466 | +0.1159 | 2.48e-50 | 1102 | 498 |
| R-R | 0.2382 | 0.2536 | -0.0243 | 3.54e-08 | 687 | 913 |
| L-SP | 0.1458 | 0.1061 | +0.0232 | 2.94e-18 | 978 | 622 |
| A-R | 0.2215 | 0.2315 | -0.0202 | 1.35e-05 | 711 | 889 |
| L-R | 0.1961 | 0.1519 | +0.0187 | 4.81e-07 | 903 | 697 |
| S-S | 0.0252 | 0.0290 | -0.0037 | 1.16e-20 | 609 | 991 |
| A-S | 0.0235 | 0.0265 | -0.0031 | 8.88e-16 | 636 | 964 |

## Paired Rewritten Detector Outcomes

| Detector | n | UTA FN / MPG FN | UTA FN / MPG TP | UTA TP / MPG FN | UTA TP / MPG TP | UTA FN rate | MPG FN rate | McNemar p |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| SecureNet | 1598 | 97 | 369 | 18 | 1114 | 0.2916 | 0.0720 | 8.22e-71 |
| V3 | 1600 | 176 | 395 | 80 | 949 | 0.3569 | 0.1600 | 4.66e-47 |

## Outputs

- Joined row table: `s6_controlled_uta_mpg_joined_rows.csv`
- Paired persuasion comparison: `s6_controlled_uta_mpg_persuasion_comparison.csv`
- Paired rewritten detector outcome cells: `s6_controlled_uta_mpg_detector_outcome_cells.csv`
- Missing original-baseline audit: `s6_controlled_original_transition_join_audit.csv`
