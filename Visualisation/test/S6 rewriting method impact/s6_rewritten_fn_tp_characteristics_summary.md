# S6 Rewritten FN-vs-TP Pair Characteristics

This is the final pairing-independent S6 failure-characteristic analysis. It compares rewritten phishing rows that become false negatives with rewritten phishing rows that remain true positives, using rewritten persuasion-pair scores directly rather than original-to-rewrite deltas.

Statistics: two-sided Mann-Whitney U per persuasion pair, with Benjamini-Hochberg FDR correction within each method-detector family. Positive rank-biserial effects mean the pair score is higher in FN than TP rewritten phishing.

## Top Significant Pairs

| Method | Detector | Pair | FN median | TP median | FN-TP median | q | Effect | n_FN | n_TP |
|---|---|---|---:|---:|---:|---:|---:|---:|---:|
| Fuzzer | SecureNet | S-S | 0.0721 | 0.0387 | +0.0335 | 2.21e-62 | +0.418 | 962 | 1237 |
| Fuzzer | SecureNet | A-S | 0.0719 | 0.0384 | +0.0335 | 2.21e-62 | +0.417 | 962 | 1237 |
| Fuzzer | SecureNet | SP-S | 0.0277 | 0.0108 | +0.0169 | 1.62e-57 | +0.400 | 962 | 1237 |
| Fuzzer | SecureNet | S-C | 0.0016 | 0.0005 | +0.0011 | 3.67e-55 | +0.391 | 962 | 1237 |
| Fuzzer | SecureNet | SP-C | 0.0085 | 0.0035 | +0.0050 | 3.61e-49 | +0.368 | 962 | 1237 |
| Fuzzer | V3 | S-S | 0.0670 | 0.0418 | +0.0252 | 8.66e-34 | +0.305 | 1048 | 1152 |
| Fuzzer | V3 | A-S | 0.0669 | 0.0416 | +0.0253 | 1.61e-33 | +0.302 | 1048 | 1152 |
| Fuzzer | V3 | SP-S | 0.0241 | 0.0125 | +0.0116 | 1.86e-29 | +0.282 | 1048 | 1152 |
| Fuzzer | V3 | S-C | 0.0014 | 0.0006 | +0.0008 | 2.9e-29 | +0.280 | 1048 | 1152 |
| Fuzzer | V3 | A-C | 0.0207 | 0.0132 | +0.0075 | 3.1e-24 | +0.254 | 1048 | 1152 |
| MPG | SecureNet | A-A | 0.9866 | 0.9447 | +0.0420 | 5.07e-05 | +0.264 | 115 | 1483 |
| MPG | SecureNet | L-SP | 0.1515 | 0.1015 | +0.0500 | 0.000542 | +0.226 | 115 | 1483 |
| MPG | SecureNet | A-SP | 0.2020 | 0.1518 | +0.0501 | 0.00137 | +0.208 | 115 | 1483 |
| MPG | SecureNet | A-L | 0.6910 | 0.5365 | +0.1545 | 0.0027 | +0.194 | 115 | 1483 |
| MPG | SecureNet | L-L | 0.7077 | 0.5880 | +0.1196 | 0.0031 | +0.188 | 115 | 1483 |
| UTA | SecureNet | A-A | 0.9686 | 0.9473 | +0.0214 | 0.00102 | +0.123 | 467 | 1133 |
| UTA | SecureNet | L-L | 0.9891 | 0.9675 | +0.0216 | 0.00102 | +0.121 | 467 | 1133 |
| UTA | SecureNet | A-L | 0.8877 | 0.8265 | +0.0612 | 0.00102 | +0.121 | 467 | 1133 |
| UTA | SecureNet | L-C | 0.0088 | 0.0076 | +0.0012 | 0.00236 | +0.111 | 467 | 1133 |
| UTA | SecureNet | L-SP | 0.1578 | 0.1417 | +0.0162 | 0.00236 | +0.110 | 467 | 1133 |
| UTA | V3 | R-R | 0.2018 | 0.2633 | -0.0614 | 1.23e-08 | -0.187 | 571 | 1029 |
| UTA | V3 | A-R | 0.1860 | 0.2438 | -0.0579 | 6.66e-08 | -0.175 | 571 | 1029 |
| UTA | V3 | L-R | 0.1620 | 0.2202 | -0.0581 | 1.47e-07 | -0.169 | 571 | 1029 |
| UTA | V3 | R-C | 0.0019 | 0.0026 | -0.0007 | 2.46e-06 | -0.152 | 571 | 1029 |
| UTA | V3 | R-S | 0.0048 | 0.0067 | -0.0019 | 3.26e-05 | -0.135 | 571 | 1029 |

## Outputs

- Full table: `s6_rewritten_fn_tp_pair_characteristics.csv`
- Top table: `s6_rewritten_fn_tp_pair_characteristics_top.csv`
- Detector consistency table: `s6_rewritten_fn_tp_detector_consistency.csv`
- Heatmap: `fig_s6_rewritten_fn_tp_pair_characteristics.png` / `.pdf`
