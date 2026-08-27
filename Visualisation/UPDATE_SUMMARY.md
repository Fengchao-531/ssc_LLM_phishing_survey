# Visualization Update Summary

This file summarizes what was added or updated under `Visualization/`, what each part is for, the key code, and the usable outputs/results.

## 0. Inventory

Current `Visualization/` contains approximately:

- 420 PNG figures
- 24 PDF figures
- 333 CSV result/support tables
- 210 JSON metadata/summary files
- 97 Python scripts
- 100 Markdown notes/READMEs

The visualization work is not one single figure set. It is a group of analysis packages for the SoK paper:

- overview detector behavior
- stage-level detector failure maps
- Section 5.3.2 persuasion evidence chain
- RQ2 email/vishing persuasion comparison
- S6 rewriting mechanism analysis
- S8 generator and linguistic-action mechanism analysis
- academic-vs-industry detector disagreement analysis

## 1. Main Overview / PCA Visualizations

Location:

- `Visualization/PCA/`
- `Visualization/test/overview/`
- `Visualization/test/stages/`

Main purpose:

- Project email samples into a 2D persuasion-indicator space.
- Compare where HW phishing, LLM phishing, benign TN, phishing TP, and phishing FN samples appear.
- Show whether detector misses occupy visually/structurally different regions from caught phishing.

Important outputs:

- `Visualization/PCA/01_pattern_map_scamllm_full_phishing_only.png`
- `Visualization/PCA/02_detector_coverage_scamllm_full_phishing_only.png`
- `Visualization/PCA/03_indicator_composition_scamllm_full_phishing_only.png`
- `Visualization/PCA/04_motif_delta_scamllm_full_phishing_only.png`
- `Visualization/test/overview/fig2_overview_focus_contours_pca.png`
- `Visualization/test/overview/fig2_overview_focus_heatmaps_pca.png`
- `Visualization/test/overview/fig_overview_group1_llm_p_fn_llm_b_tn_hw_b_tn_heatmaps_pca.png`
- `Visualization/test/overview/fig_overview_group2_llm_p_fn_llm_p_tp_heatmaps_pca.png`
- `Visualization/test/stages/fig_s6_stagewise_surrogate_response_tp_fn_map_pca.png`
- `Visualization/test/stages/fig_s8_stagewise_surrogate_response_tp_fn_map_pca.png`

Important code:

- `Visualization/generate_stage_visualizations.py`
- `Visualization/generate_overview_focus_figures.py`
- `Visualization/generate_stage_fig2_fn_figures.py`
- `Visualization/generate_stage_group_comparison_figures.py`
- `Visualization/generate_stage_split_figures.py`
- `Visualization/test/overview/generate_overview_focus_figures.py`
- `Visualization/test/stages/generate_stage_fig2_fn_figures.py`

Use in paper:

- Good for overview figures showing detector success/failure regions.
- Especially useful when explaining that LLM phishing false negatives are not uniformly distributed; they cluster in certain persuasion/profile regions.

## 2. Section 5.3.2 Persuasion Evidence Chain

Location:

- `Visualization/5.3.2/`

Main purpose:

This is the cleanest paper-ready package for Section 5.3.2. It connects three evidence layers:

- E1: phishing vs benign persuasion-pair relevance
- E2: LLM phishing vs HW phishing shift, including phishing-specific LLM shift
- E3: detector TP vs FN relevance on LLM phishing samples

Important code:

- `Visualization/5.3.2/run_5_3_2_evidence_chain.py`

Important data outputs:

- `metadata.csv`
- `sample_counts.csv`
- `persuasion_scores.csv`
- `predictions.csv`
- `E1_phishing_relevance.csv`
- `E1_source_stratified_phishing_relevance.csv`
- `E2_llm_specific_shift.csv`
- `E2_phishing_specific_llm_shift.csv`
- `E3_detector_relevance.csv`
- `appendix_E1_E2_complete_table.csv`
- `appendix_E3_complete_table.csv`
- `run_summary.json`
- `sample_level_5_3_2.zip`

Important figure outputs:

- `fig_5_3_2_evidence_chain.png` / `.pdf`
- `fig_5_3_2_e1_e2_split_compact.png` / `.pdf`
- `fig_5_3_2_e1_e2_vertical_compact.png` / `.pdf`
- `fig_5_3_2_detector_heatmap.png` / `.pdf`
- `fig_5_3_2_a_plus_detector_heatmap.png` / `.pdf`

Use in paper:

- This is the main evidence-chain figure/table set.
- It supports claims like: some persuasion-pair dimensions are phishing-relevant, some are specifically shifted by LLM generation, and some separate detector TP/FN behavior.

## 3. Evidence Tables For Overview Heatmaps

Location:

- `Visualization/evidence/`
- `Visualization/evidence/overview/`

Main purpose:

- Build readable word/phrase evidence behind selected heatmap cells.
- Explain what linguistic content is associated with groups such as `HW-B TN`, `LLM-B TN`, `LLM-P FN`, and `LLM-P TP`.

Important code:

- `Visualization/evidence/build_overview_group_evidence.py`
- `Visualization/evidence/generate_overview_readme.py`

Important outputs:

- `Visualization/evidence/overview/curated_overview_evidence_table.csv`
- `Visualization/evidence/overview/curated_overview_evidence_table.json`
- `Visualization/evidence/overview/group_registry.csv`
- `Visualization/evidence/overview/README.md`

Use in paper:

- Good as appendix/supporting qualitative evidence.
- Helps interpret heatmap cells with concrete words/phrases.

Important caveat:

- These are group-associated evidence items, not causal token attributions.

## 4. Significance Analysis

Location:

- `Visualization/statistic/`

Main purpose:

- Run significance testing for overview heatmap differences.
- Summarize which group comparisons have statistically significant cell differences.

Important code:

- `Visualization/statistic/heatmap_significance_analysis.py`

Important outputs:

- `Visualization/statistic/overview/overview_summary.csv`
- `Visualization/statistic/overview/overview_summary.md`

Key result:

- Most overview global heatmap comparisons are significant at `p = 0.0050`.
- Many comparisons have 17-21 significant cells after FDR correction.
- Example: `LLM-P FN vs LLM-P TP` has 15 significant cells.

Use in paper:

- Supports statements that observed heatmap differences are statistically meaningful, not just visual artifacts.

## 5. RQ2 Email vs Vishing Visualizations

Locations:

- `Visualization/RQ2/`
- `Visualization/test/rq2/`

Main purpose:

- Compare persuasion profiles across communication settings:
  - email
  - single-turn vishing
  - multi-turn vishing
- Analyze how persuasion principles change across modality and conversation structure.

Important code:

- `Visualization/generate_rq2_vishing_email_figures.py`
- `Visualization/test/rq2/generate_rq2_communication_setting_stats.py`
- `Visualization/test/rq2/generate_rq2_llm_only_communication_heatmap.py`

Important outputs:

- `Visualization/RQ2/rq2_email_pair_strength_heatmaps.png`
- `Visualization/RQ2/rq2_vishing_email_pca_contours.png`
- `Visualization/RQ2/rq2_vishing_email_density_contours.png`
- `Visualization/RQ2/rq2_vishing_single_pair_strength_heatmaps.png`
- `Visualization/RQ2/rq2_vishing_multi_pair_strength_heatmaps.png`
- `Visualization/RQ2/rq2_vishing_single_minus_email_difference_heatmaps.png`
- `Visualization/RQ2/rq2_vishing_multi_minus_email_difference_heatmaps.png`
- `Visualization/test/rq2/rq2_llm_only_communication_persuasion_heatmap.png` / `.pdf`
- `Visualization/test/rq2/rq2_communication_setting_summary.md`

Key result from `rq2_communication_setting_summary.md`:

- Email sample count: 55,095
- Single-turn vishing sample count: 12,014
- Multi-turn vishing sample/conversation count: 24,111
- Multi-turn vishing generally has the highest mean persuasion scores.
- Largest setting differences:
  - Social Proof: max difference 0.4299
  - Reciprocity: max difference 0.3682
  - Liking: max difference 0.3418
- All six principle-level Kruskal-Wallis tests are significant after FDR correction.

Use in paper:

- This supports the RQ2 claim that communication modality changes the persuasion strategy distribution, especially for multi-turn vishing.

## 6. S6 Rewriting Mechanism Analysis

Locations:

- `Visualization/test/S6 rewriting method impact/`
- `Visualization/test/S6 rewriting mechanism analysis/`

Main purpose:

- Analyze how S6 rewriting methods change persuasion structure and detector failure.
- Compare Fuzzer, MPG, and UTA rewriting methods.
- Identify which persuasion pairs/features are associated with FN vs TP after rewriting.

Important code:

- `generate_s6_rewriting_method_impact.py`
- `generate_s6_two_panel_main_figure.py`
- `generate_s6_rewritten_fn_tp_characteristics.py`
- `generate_s6_failure_half_heatmaps.py`
- `generate_s6_failure_heatmap_transposed.py`
- `generate_s6_fuzzer_action_characteristics.py`
- `generate_s6_controlled_uta_mpg_join.py`
- `generate_s6_controlled_mapping_audit.py`

Important figure outputs:

- `fig_s6_rewriting_two_panel_main_a.png`
- `fig_s6_rewriting_two_panel_main_b.png`
- `fig_s6_rewriting_two_panel_main_a.pdf`
- `fig_s6_rewriting_two_panel_main_b.pdf`
- `fig_s6_rewritten_fn_tp_pair_characteristics.png`
- `fig_s6_fuzzer_fn_tp_action_characteristics.png`
- `fig_s6_fn_associated_pair_changes_half_heatmaps.png`
- `fig_s6_fn_associated_pair_changes_transposed.png`
- `fig_s6_method_pair_shift.png`
- `fig_s6_detector_mcc.png`

Important table/summary outputs:

- `s6_rewriting_method_impact_summary.md`
- `s6_rewriting_methodology_notes.md`
- `s6_two_panel_heatmap_values.csv`
- `s6_detector_performance_compact_table.csv`
- `s6_rewritten_fn_tp_pair_characteristics.csv`
- `s6_controlled_uta_mpg_persuasion_comparison.csv`
- `s6_controlled_uta_mpg_detector_outcome_cells.csv`
- `s6_fuzzer_fn_tp_action_characteristics.csv`

Key results:

- Fuzzer has the largest overall persuasion shift:
  - `D_m = 0.0680`
  - mean absolute sample-pair shift `0.1690`
- UTA:
  - `D_m = 0.0468`
  - max pair shift `0.2418`, especially `L-L`
- MPG:
  - `D_m = 0.0304`
  - generally smaller shifts
- Detector impact examples:
  - SecureNet MCC: Fuzzer `0.4557`, MPG `0.8910`, UTA `0.7356`
  - V3 MCC: Fuzzer `0.4545`, MPG `0.7398`, UTA `0.6560`
- Main narrative suggested by the summary:
  - use the two-panel persuasion-structure figure for S6 mechanism claims
  - `S-S`, `A-S`, and `SP-S` are clear Fuzzer-specific signatures
  - Fuzzer increases these scarcity/social-proof-linked structures while UTA/MPG reduce them

Important caveat:

- Some older row-order `delta_pair` outputs are marked exploratory/deprecated.
- Final S6 claims should use the controlled / pairing-independent outputs.

## 7. S8 Generator And Linguistic-Action Analysis

Locations:

- `Visualization/test/S8 linguistic action features/`
- `Visualization/test/S8 generator A_I difference/`

Main purpose:

- Compare S8 outputs across six LLM generators:
  - Claude
  - GPT
  - Gemini
  - Llama
  - Ministral
  - DeepSeek
- Analyze generator effects on detector outcomes and phishing-action/linguistic features.
- Separate raw detector outcome shifts from feature-adjusted generator effects.

Important code:

- `generate_s8_linguistic_action_heatmap.py`
- `generate_s8_mcc_bootstrap_ci.py`
- `generate_s8_paired_rq1_rq2.py`
- `generate_s8_rq3_paired_analysis.py`
- `generate_s8_detector_tp_fn_feature_table.py`
- `generate_s8_detector_score_shift_tables.py`
- `generate_s8_detector_specific_surrogate_scores.py`
- `generate_s8_surrogate_response_shift_tables.py`
- `generate_s8_generator_ai_distributions.py`
- `generate_s8_linguistic_delta_heatmap.py`

Important figure outputs:

- `Fig_S8_A_detector_generator_detection_rate_heatmap.png`
- `Fig_S8_B_generator_feature_prevalence_heatmap.png`
- `Fig_S8_C_detector_specific_tp_fn_feature_differences.png`
- `Fig_S8_D_observed_vs_feature_adjusted_detection.png`
- `fig_s8_linguistic_action_feature_heatmap.png` / `.pdf`
- `fig_s8_detector_specific_surrogate_mean_heatmap.png` / `.pdf`
- `fig_s8_generator_linguistic_feature_delta_heatmap.png` / `.pdf`
- `s8_claude_ai_distribution_panels.png`
- `s8_deepseek_ai_distribution_panels.png`
- `s8_gemini_ai_distribution_panels.png`
- `s8_gpt_ai_distribution_panels.png`
- `s8_llama_ai_distribution_panels.png`
- `s8_ministral_ai_distribution_panels.png`

Important table/summary outputs:

- `s8_paired_rq1_rq2_summary.md`
- `S8_rq3_paired_analysis_summary.md`
- `s8_detector_score_shift_summary.md`
- `s8_detector_specific_surrogate_summary.md`
- `s8_detector_tp_fn_feature_summary.md`
- `s8_surrogate_response_shift_summary.md`
- `S8_detector_generator_MCC_bootstrap.csv`
- `s8_linguistic_action_feature_prevalence.csv`
- `s8_linguistic_action_feature_global_tests.csv`
- `s8_linguistic_action_feature_pairwise_tests.csv`
- `s8_rq3_observed_adjusted_marginal_detection.csv`

Key results:

- Paired S8 common prompt set uses `499` prompts per generator.
- Detector detection rates differ strongly by generator.
- SecureNet range:
  - Claude `77.0%`
  - DeepSeek `7.4%`
  - range about `69.5 pp`
- PhishingV3 range:
  - Claude `45.9%`
  - DeepSeek `0.4%`
- RQ3 feature-adjusted analysis says generator variation remains strong after feature adjustment for most detectors.
- Feature variation examples:
  - Direct URL/page instruction range: `46.5 pp`
  - Click/open request range: `36.5 pp`
  - Urgency wording range: `21.4 pp`
- TP/FN feature analysis shows `Login/account action`, `Urgency wording`, and `Information submission` often separate TP from FN, especially for PhishingV3 and SecureNet.

Use in paper:

- This is the strongest package for explaining S8 generator-specific behavior.
- It supports claims that different LLM generators produce phishing with measurably different detector detectability and linguistic-action profiles.

Important caveat:

- PhishingV3 x DeepSeek has sparse TP and is marked descriptive only for TP/FN feature analysis.

## 8. Academic vs Industry Detector Difference

Location:

- `Visualization/test/A-I Differences/`
- duplicate/older path: `Visualization/test/A_I difference/`

Main purpose:

- Compare an academic detector (`scamllm`) with an industry detector (`phishing_email_agent_prediction`).
- Analyze:
  - Academic TP vs Industry TP
  - Academic-only caught phishing
  - Industry-only caught phishing
  - disagreement regions in PCA space
  - persuasion heatmap differences

Important code:

- `generate_academic_industry_difference_figures.py`
- `generate_detector_focus_difference_figures.py`
- `generate_disagreement_analysis_panels.py`
- `generate_hw_llm_detector_contours.py`
- `generate_selected_llm_tp_heatmaps_and_evidence.py`

Important outputs:

- `fig_ai_disagreement_analysis_panels.png` / `.pdf`
- `fig_detector_disagreement_contours_hw_llm.png`
- `fig_selected_llm_tp_detector_difference_heatmap.png`
- `fig_selected_llm_tp_detector_heatmaps.png`
- `fig_selected_llm_tp_words_phrases_matrix.png`
- `selected_llm_tp_detector_difference_values.csv`
- `selected_llm_tp_words_phrases_matrix.md`
- `ai_disagreement_analysis_summary.json`

Use in paper:

- Good for discussing why academic and industry detectors fail/catch different subsets.
- Useful as supporting evidence for detector-family disagreement.

## 9. Git Interaction Visualizations

Location:

- `Visualization/git interaction/`

Main purpose:

- Per-detector, per-stage visual overviews.
- Organized by detector:
  - `email_phishing_detection_v3`
  - `phishing_email_agent`
  - `pimref`
  - `rspamd`
  - `scamllm`
  - `securenet_llama`
  - `spamassassin`
  - `spamscanner`
  - `t5phishing`
  - `xgboost`

Important code:

- `Visualization/build_git_interaction_overviews.py`

Important outputs:

- `Visualization/git interaction/manifest.json`
- detector/stage README files under each detector folder

Use in paper:

- More useful for navigation/audit than final figures.
- Helps quickly inspect which detector-stage combinations have visual evidence.

## 10. Persuasion Scoring Models

Locations:

- `Visualization/persuasion_strategy_model/`
- `Visualization/persuasion_strategy_wvae/`

Main purpose:

- Provide the model code used to score persuasion principles/pairs in email text.
- These scores are the basis for the heatmaps, PCA projections, and evidence-chain analyses.

Important code:

- `persuasion_strategy_model/src/score_email_csv.py`
- `persuasion_strategy_model/src/train.py`
- `persuasion_strategy_wvae/code/run_full_inference.py`
- `persuasion_strategy_wvae/code/score_email_csv.py`
- `persuasion_strategy_wvae/code/visualize_persuasion_heatmaps.py`

Use in paper:

- Method/supporting code, not final result figures.
- Explains how persuasion dimensions were produced.

## 11. Final-Used Script Bundle

Location:

- `Visualization/test/fina_used/`

Purpose:

- This looks like the curated script bundle for generating final or near-final figures.

Scripts:

- `generate_folder1_group1_difference_heatmaps.py`
- `generate_folder2_overview_focus_heatmaps.py`
- `generate_folder3_surrogate_response_map.py`
- `generate_folder4_all_stage_phishing_maps.py`
- `generate_folder5_all_stage_hw_llm_fn_grouped_stacked_bars.py`
- `generate_folder6_stagewise_hw_llm_fn_group_boxplots.py`
- `generate_folder7_llm_multiturn_pair_turn_boxplots.py`
- `generate_folder7_llm_multiturn_principle_turn_heatmap_r6.py`
- `generate_folder7_rq2_persuasion_group_boxplots.py`
- `generate_folder8_multiturn_fn_strategy_boxplots.py`
- `generate_folder8_multiturn_round_strategy_boxplots.py`
- `generate_folder9_contour_plus_left_heatmap.py`

Use:

- Treat this as the most likely final figure-generation entry set.
- The surrounding `test/` directory contains many exploratory outputs, while `fina_used/` is likely the cleaned figure pipeline.

## 12. Suggested Paper Mapping

Recommended mapping from visualization package to manuscript use:

- Overview / detector behavior:
  - use `Visualization/test/overview/`
  - use `Visualization/test/stages/`
- Section 5.3.2 evidence chain:
  - use `Visualization/5.3.2/`
- RQ2 modality comparison:
  - use `Visualization/RQ2/`
  - use `Visualization/test/rq2/`
- S6 rewriting mechanism:
  - use `Visualization/test/S6 rewriting method impact/`
  - avoid deprecated row-order `delta_pair` claims
- S8 generator effect:
  - use `Visualization/test/S8 linguistic action features/`
  - use `Visualization/test/S8 generator A_I difference/`
- Detector-family disagreement:
  - use `Visualization/test/A-I Differences/`
- Qualitative evidence appendix:
  - use `Visualization/evidence/overview/`

