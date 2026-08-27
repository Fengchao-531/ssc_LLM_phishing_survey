# Academic vs Industry Differences

This folder compares:

- `Academic detector`: `scamllm`
- `Industry detector`: `phishing_email_agent_prediction`

The goal is to contrast two kinds of capability differences on phishing emails:

1. `Correctly captured phishing`
   - Compare `Academic TP` vs `Industry TP`
   - Show where each detector captures phishing in PCA space
   - Show the WVAE persuasion-principle combination profile for each detector's TP set

2. `Exclusive capture / miss differences`
   - `Academic-only caught phishing`: `Academic=TP`, `Industry=FN`
   - `Industry-only caught phishing`: `Industry=TP`, `Academic=FN`
   - Show where these disagreement sets live in PCA space
   - Show their WVAE persuasion-principle combination profiles

Planned outputs:

- `merged_academic_industry_projected_points.csv`
- `merged_academic_industry_phishing_only.csv`
- `merged_detector_hw_llm_projected_points.csv`
- `tp_capture_summary.json`
- `tp_capture_heatmaps.csv`
- `exclusive_capture_heatmaps.csv`
- `fig_academic_vs_industry_tp_contours_pca.png`
- `fig_academic_vs_industry_tp_heatmaps.png`
- `fig_academic_vs_industry_exclusive_contours_pca.png`
- `fig_academic_vs_industry_exclusive_heatmaps.png`
- `fig_hw_llm_phishing_overview_by_detector_pca.png`
- `fig_hw_llm_phishing_dual_panel_by_detector_pca.png`
- `hw_llm_detector_contours_metadata.json`
- `fig_detector_focus_difference_heatmap.png`
- `fig_detector_disagreement_contours_hw_llm.png`
- `detector_focus_difference_heatmap_values.csv`
- `detector_disagreement_composite_points.csv`
- `detector_focus_difference_metadata.json`
- `run.log`

High-level pipeline:

1. Load the existing PCA-projected table from `Visualization/test/projected_points.csv`
2. Attach `phishing_email_agent_prediction` from `Evaluation/processed-evaluation-datasets`
3. Handle the S8 alias mapping that was already introduced for HW rows:
   - `S8-gpt <- S8-llama`
   - `S8-gemini <- S8-ministral`
   - `S8-claude <- S8-deepseek`
4. Build phishing-only detector outcome groups
5. Save merged data for downstream inspection
6. Generate PCA contour / scatter figures
7. Generate WVAE persuasion-principle heatmaps

The background job is launched with:

```bash
nohup python "generate_academic_industry_difference_figures.py" > run.log 2>&1 &
```
