# Section 5.3.2 Persuasion Evidence Chain

This folder contains the E1-E3 analysis requested for Section 5.3.2.

Run from the repository root:

```bash
python3 Visualization/5.3.2/run_5_3_2_evidence_chain.py
```

The script uses existing persuasion-pair scores and detector predictions only.
It does not regenerate phishing samples and does not rerun detectors.

## Outputs

- `metadata.csv`
  Sample-level metadata with `sample_id,dataset,source,label,stage,generator,parent_id`.
- `sample_counts.csv`
  Counts for `HW/LLM x P/B`, used to audit whether pooled E1 mixes source and
  phishing effects.
- `persuasion_scores.csv`
  Long-form sample-by-pair scores for all 21 persuasion pairs.
- `predictions.csv`
  Long-form detector predictions. Academic detector predictions come from the
  WVAE full inference CSVs; industry detector predictions are merged by
  source/stage row index where processed industry results exist.
- `sample_level_5_3_2.zip`
  Zip bundle of the three sample-level CSVs above.
- `E1_phishing_relevance.csv`
  Pooled P vs B MWU, BH-FDR q-values, median/IQR, rank-biserial effects and
  bootstrap CIs for all 21 pairs.
- `E1_source_stratified_phishing_relevance.csv`
  Separate HW P/B and LLM P/B checks, so pooled E1 can be audited for source
  imbalance.
- `E2_llm_specific_shift.csv`
  LLM-P vs HW-P MWU, BH-FDR q-values, median shift, rank-biserial effects, and
  source AUC with bootstrap 95% CI for all 21 pairs.
- `E2_phishing_specific_llm_shift.csv`
  Difference-in-differences interaction test for phishing-specific LLM shift:
  `kappa = (LLM-P - HW-P) - (LLM-B - HW-B)`. P/B labels are fixed, HW/LLM
  source labels are permuted independently within P and B groups, and 21 pairs
  are BH-FDR corrected.
- `E3_detector_relevance.csv`
  LLM-P TP vs FN MWU per detector, using each detector's original prediction.
- `fig_5_3_2_e1_e2_split_compact.png` / `.pdf`
  Compact split E1/E2 figure with persuasion pairs on the x-axis. Panel (b)
  shows phishing-specific LLM shift (`kappa`), not the raw LLM-P vs HW-P shift.
- `fig_5_3_2_e1_e2_vertical_compact.png` / `.pdf`
  Compact vertical version with persuasion pairs on the y-axis.
- `fig_5_3_2_detector_heatmap.png` / `.pdf`
  Standalone E3 detector relevance heatmap using common LLM-P samples with all
  detector predictions.
- `detector_caption_mapping.txt`
  Caption text mapping `D1-D10` to detector names in heatmap order.
- `fig_5_3_2_a_plus_detector_heatmap.png` / `.pdf`
  Combined figure with vertical E1 panel (a) and the common-sample E3 heatmap.
- `appendix_E1_E2_complete_table.csv`
  Complete E1/E2 appendix table.
- `appendix_E3_complete_table.csv`
  Complete E3 appendix table.
- `run_summary.json`
  Counts, detector list, pair order, and merge diagnostics.

## Statistical Conventions

All three experiments are run independently over the full set of 21 persuasion
pairs. Significance is controlled with Benjamini-Hochberg FDR within each
experiment family, and within each detector for E3. E3 groups LLM phishing
samples into TP/FN directly from the original detector prediction.

Full-sample MWU statistics, effects, and AUCs are reported. Bootstrap CIs
resample from the complete comparison groups with replacement; no fixed
subsample cap is used.
