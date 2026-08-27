# S6 Rewriting Method Impact

## Detector-performance difference

| Method | Detector | MCC | FN rate | Mean detector-score change |
|---|---|---:|---:|---:|
| Fuzzer | SecureNet | 0.4557 | 0.437 | -0.1304 |
| MPG | SecureNet | 0.8910 | 0.072 | +0.0856 |
| UTA | SecureNet | 0.7356 | 0.292 | -0.1262 |
| Fuzzer | V3 | 0.4545 | 0.476 | +0.0071 |
| MPG | V3 | 0.7398 | 0.160 | +0.0521 |
| UTA | V3 | 0.6560 | 0.357 | -0.0093 |

## Overall persuasion shift

| Method | D_m | Mean abs sample-pair shift | Max pair shift |
|---|---:|---:|---:|
| Fuzzer | 0.0680 | 0.1690 | 0.1907 |
| UTA | 0.0468 | 0.1317 | 0.2418 |
| MPG | 0.0304 | 0.1349 | 0.0760 |

## Top changed persuasion pairs

| Method | Pair | Original mean | Rewritten mean | Delta |
|---|---|---:|---:|---:|
| Fuzzer | A-L | 0.5233 | 0.7140 | +0.1907 |
| Fuzzer | L-L | 0.5533 | 0.7280 | +0.1748 |
| Fuzzer | A-SP | 0.2611 | 0.3857 | +0.1246 |
| Fuzzer | A-R | 0.3387 | 0.4585 | +0.1198 |
| Fuzzer | L-SP | 0.2080 | 0.3272 | +0.1191 |
| MPG | A-SP | 0.2579 | 0.1819 | -0.0760 |
| MPG | SP-SP | 0.2697 | 0.1973 | -0.0724 |
| MPG | L-SP | 0.2065 | 0.1362 | -0.0703 |
| MPG | R-SP | 0.1488 | 0.0806 | -0.0683 |
| MPG | A-R | 0.3329 | 0.2833 | -0.0496 |
| UTA | L-L | 0.5538 | 0.7956 | +0.2418 |
| UTA | A-L | 0.5238 | 0.7200 | +0.1962 |
| UTA | A-SP | 0.2555 | 0.1801 | -0.0754 |
| UTA | SP-SP | 0.2666 | 0.1939 | -0.0727 |
| UTA | R-SP | 0.1432 | 0.0710 | -0.0722 |

## Top failure-associated changes

| Method | Detector | Pair | FN mean delta | TP mean delta | FN-TP | q | Effect |
|---|---|---|---:|---:|---:|---:|---:|
| Fuzzer | PiMRef | L-R | +0.1051 | +0.3085 | -0.2034 | 5.2e-06 | -0.275 |
| Fuzzer | PiMRef | A-L | +0.1787 | +0.4005 | -0.2219 | 5.2e-06 | -0.274 |
| Fuzzer | PiMRef | L-L | +0.1636 | +0.3701 | -0.2065 | 1.17e-05 | -0.261 |
| Fuzzer | ScamLLM | SP-S | +0.0951 | +0.0061 | +0.0889 | 1.22e-24 | +0.309 |
| Fuzzer | ScamLLM | S-S | +0.0962 | +0.0086 | +0.0876 | 1.23e-23 | +0.300 |
| Fuzzer | ScamLLM | SP-SP | +0.2642 | +0.0738 | +0.1904 | 1.35e-23 | +0.299 |
| Fuzzer | SecureNet | S-S | +0.0678 | -0.0017 | +0.0695 | 4.27e-28 | +0.279 |
| Fuzzer | SecureNet | A-S | +0.0695 | -0.0002 | +0.0697 | 4.27e-28 | +0.278 |
| Fuzzer | SecureNet | SP-S | +0.0635 | -0.0022 | +0.0657 | 4.09e-26 | +0.267 |
| Fuzzer | V3 | S-S | +0.0578 | +0.0023 | +0.0555 | 4.72e-16 | +0.208 |
| Fuzzer | V3 | A-S | +0.0595 | +0.0038 | +0.0556 | 4.72e-16 | +0.207 |
| Fuzzer | V3 | SP-S | +0.0541 | +0.0014 | +0.0528 | 5.62e-14 | +0.191 |
| Fuzzer | XGBoost | A-S | +0.0011 | +0.0482 | -0.0471 | 7.06e-23 | -0.257 |
| Fuzzer | XGBoost | SP-S | +0.0009 | +0.0422 | -0.0412 | 7.06e-23 | -0.256 |
| Fuzzer | XGBoost | S-S | -0.0001 | +0.0463 | -0.0464 | 7.44e-23 | -0.255 |
| MPG | SecureNet | A-A | +0.0422 | -0.0130 | +0.0551 | 0.0277 | +0.172 |
| MPG | SecureNet | L-SP | -0.0113 | -0.0753 | +0.0640 | 0.0277 | +0.159 |
| MPG | SecureNet | A-SP | -0.0197 | -0.0807 | +0.0610 | 0.0277 | +0.154 |
| UTA | SecureNet | L-L | +0.2970 | +0.2191 | +0.0779 | 0.0446 | +0.098 |
| UTA | V3 | R-R | -0.0904 | -0.0304 | -0.0599 | 0.00441 | -0.111 |
| UTA | V3 | A-R | -0.0950 | -0.0366 | -0.0585 | 0.00441 | -0.104 |
| UTA | V3 | L-R | -0.0475 | +0.0093 | -0.0567 | 0.00441 | -0.103 |
| UTA | XGBoost | L-R | -0.0504 | +0.0513 | -0.1017 | 3.57e-08 | -0.178 |
| UTA | XGBoost | A-R | -0.0956 | +0.0027 | -0.0983 | 1.07e-07 | -0.170 |
| UTA | XGBoost | R-R | -0.0876 | +0.0045 | -0.0921 | 2.5e-07 | -0.163 |
## Follow-up outputs

- Main S6 evidence-chain figure: `fig_s6_rewriting_two_panel_main.png`
- Main S6 evidence-chain panel (a): `fig_s6_rewriting_two_panel_main_a.png`
- Main S6 evidence-chain panel (b): `fig_s6_rewriting_two_panel_main_b.png`
- Main S6 evidence-chain values: `s6_two_panel_heatmap_values.csv`
- Final rewritten FN-vs-TP pair-characteristic table: `s6_rewritten_fn_tp_pair_characteristics.csv`
- Final rewritten FN-vs-TP pair-characteristic heatmap: `fig_s6_rewritten_fn_tp_pair_characteristics.png`
- Detector-consistency table for final FN-vs-TP analysis: `s6_rewritten_fn_tp_detector_consistency.csv`
- UTA-MPG controlled mapping audit: `s6_uta_mpg_controlled_mapping_audit.md`
- Corrected UTA-MPG row-id join summary: `s6_controlled_uta_mpg_join_summary.md`
- Controlled UTA-MPG paired persuasion comparison: `s6_controlled_uta_mpg_persuasion_comparison.csv`
- Controlled UTA-MPG paired rewritten detector outcome cells: `s6_controlled_uta_mpg_detector_outcome_cells.csv`
- Missing original-transition audit: `s6_controlled_original_transition_join_audit.csv`
- S8-style Fuzzer action/linguistic FN-vs-TP table: `s6_fuzzer_fn_tp_action_characteristics.csv`
- S8-style Fuzzer action/linguistic FN-vs-TP heatmap: `fig_s6_fuzzer_fn_tp_action_characteristics.png`
- Fuzzer original-to-rewrite mapping audit: `s6_fuzzer_original_rewrite_mapping_audit.md`
- Scarcity direction table: `s6_fuzzer_scarcity_direction_by_outcome.csv`
- Top Fuzzer FN original/rewrite examples: `s6_fuzzer_scarcity_top_fn_examples.csv`
- Repeated-original independence check: `s6_rewriting_independence_check.csv`
- D_m/test definitions and interpretation guardrails: `s6_rewriting_methodology_notes.md`

Note: the old row-order `delta_pair` failure-association outputs are exploratory/deprecated for final controlled claims. Use the rewritten FN-vs-TP outputs above for the pairing-independent S6 failure-characteristic analysis.

The S8-style Fuzzer action/linguistic output is also pairing-independent. The requested paired original/rewrite action prevalence table and S-S/A-S/SP-S scarcity-mechanism matrix require a true Fuzzer original-to-rewrite mapping file; the current local package does not expose that mapping.

Fuzzer values can remain as external-setting observations, not as causal ranking evidence against UTA/MPG.

For the main S6 mechanism narrative, use the two-panel persuasion-structure figure rather than the action/linguistic feature heatmap: `S-S` and `A-S` are the clearest Fuzzer-specific signature because Fuzzer increases them while UTA/MPG reduce them, and the same pairs uniquely separate Fuzzer FN from TP for both SecureNet and V3.
