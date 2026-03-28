# Dataset README

This document summarizes the main datasets currently used by the processing scripts in `ssc_LLM_phishing_survey`, including:

- Data type: phishing email, ham email, quishing, vishing, smishing, or malicious URL
- Provenance: GitHub, Kaggle, Hugging Face, arXiv, OpenReview, Zenodo, or associated papers
- Year: preferably the year of the public paper or repository; if not clearly stated, that is noted explicitly
- Size: local raw file counts first, then processed output counts where useful

Notes:

- This file was compiled on March 28, 2026 based on public source verification.

## 1. Confirmed or High-Confidence Dataset Matches

| Local path / file | Dataset role | Modality | Local size | Source / link | Paper / article | Year | Notes |
|---|---|---|---:|---|---|---|---|
| `phishyai/human-generated-samples/phishing_&_ham_emails/` | phishing email + ham email | email text | easy ham `2551` + hard ham `250` + phishing `2239` | GitHub: https://github.com/GangGreenTemperTatum/phishyai ; Kaggle: https://www.kaggle.com/datasets/mohamedouledhamed/phishing-and-ham-emails ; HF model card: https://huggingface.co/loresiensis/distilgpt2-emailgen-phishing | No standalone paper confirmed | not clearly stated on public pages | The local `phishyai` README explicitly points to the Kaggle `Phishing & Ham Emails` dataset. The repository also includes `gmail-samples` and `outlook-samples`. |
| `Datasets/LLM-Benign/S4-ephishLLM.json` | phishing email + benign email | email text | total `16616`; English `11502`; English phishing `5996`; English benign `5506` | Dataset name referenced as `ePhishLLM` | E-PhishGen: https://arxiv.org/abs/2509.01791 | 2025 | The local file name `S4-ephishLLM.json` matches the dataset name referenced in the E-PhishGen paper. |
| `Datasets/LLM-Benign/Paladin-main/` | LLM phishing-email defense / trigger-tag dataset | email text | `dpo_dataset.json=1997`; `safe_emails_1000.json=1000`; `set1_dataset_nophishing.json=1000`; `set1-4 phishing sets=4000` | GitHub: https://github.com/py85252876/Paladin ; Zenodo archive: https://zenodo.org/records/15897613 | NDSS 2026 paper: https://www.ndss-symposium.org/wp-content/uploads/2026-s2522-paper.pdf | 2025 code archive / 2026 paper | The local README and Zenodo record both point to the same Paladin paper. The paper states that the phishing portion is sourced from Lin et al., while the safe portion is built from normal email data plus augmentation. |
| `Datasets/LLM-Phishing/MalURLBench/Examples/` | malicious URL benchmark, later converted to quishing in this project | URL text, then QR code | `11` model folders, `110` txt files, `61845` URLs | GitHub: https://github.com/JiangYingEr/MalURLBench | arXiv: https://arxiv.org/abs/2601.18113 | 2026 | The raw dataset is a malicious URL benchmark, not a QR dataset. In this project, those URLs are converted into QR images for S7. |
| `Datasets/HW-Phishing/7-QuishingDataset/` | quishing / benign QR code dataset | QR images stored in pickle format | total `9987`; phishing `4982`; benign `5005` | high-confidence match to QRPhishGuard | arXiv: https://arxiv.org/abs/2505.03451 | 2025 | This is a high-confidence match rather than a fully explicit upstream identifier. The local dataset size and structure are very close to the public QR phishing benchmark, but the original download README is not preserved locally. |
| `Datasets/LLM-Phishing/AI-FraudCall-Detector/Dataset/` | vishing + smishing + fraud-call / normal-call data | SMS text + call transcripts | about `27548` CSV rows locally | GitHub: https://github.com/anand-adwaith/AI-FraudCall-Detector | OpenReview paper: https://openreview.net/forum?id=IQGxkTCJmt | 2025 | The paper describes a multilingual Indian-context fraud detection project covering text, call transcripts, and audio, with synthetic augmentation. |
| `Datasets/LLM-Phishing/AI-FraudCall-Detector/LLM-dataset/` | synthetic fraud / normal SMS dataset | SMS text | `6` JSON files, each `230` rows; `Extended Dataset.txt=100`; `Fraud_DB.xlsx=638`; `Normal_DB.xlsx=5287` | Same repository: https://github.com/anand-adwaith/AI-FraudCall-Detector | Same paper: https://openreview.net/forum?id=IQGxkTCJmt | 2025 | The file names include timestamps from June 21 and June 24, 2025, which strongly suggests project-generated synthetic SMS expansions. |
| `Datasets/LLM-Phishing/AI-FraudCall-Detector/LLM-Multi/` | synthetic fraud / normal multi-turn vishing dialogues | multi-turn dialogue JSON | `115 + 115 + 69 + 23 = 322` dialogues | Same repository: https://github.com/anand-adwaith/AI-FraudCall-Detector | Same paper: https://openreview.net/forum?id=IQGxkTCJmt | 2025 | These files contain multi-turn call transcripts later normalized into `LLM-Script-Multi.csv` and `LLM-Vishing-Multi.csv`. |
| `Datasets/LLM-Phishing/flowgpt-jailbrea*.txt` and `poe*-jailbrea*.txt` | jailbreak / malicious prompt-response collection, later filtered for phishing-email content in this project | text | FlowGPT files: `3285` lines each; Poe files: `5625` lines each | topically aligned with Malla | USENIX Security 2024 paper: https://www.usenix.org/system/files/usenixsecurity24-lin-zilong.pdf | 2024 | These local txt files look more like project-exported prompt/response dumps than an official benchmark release. The thematic match to Malla is strong, but the exact upstream release format was not confirmed from local evidence alone. |
| `Datasets/LLM-Phishing/AI-FraudCall-Detector/HW-Data/` | composite fraud / normal text set | text CSV | `composite_train.csv=36751`; `composite_test.csv=9188` | currently only confirmed as an internal subfolder of AI-FraudCall-Detector: https://github.com/anand-adwaith/AI-FraudCall-Detector | repository-level source only | 2025 or later | These appear to be repository-internal composite splits rather than a separately published external benchmark. |

## 2. Files Used Locally but Not Yet Fully Attributed

The following files are definitely used by the scripts, but the authoritative upstream download page could not be recovered with high confidence from the current local filenames and contents alone. For a paper or appendix, these should be manually cross-checked against the original download records.

| Local path / file | Local size | Current interpretation | Recommended wording |
|---|---:|---|---|
| `Datasets/LLM-Phishing/human-generated/legit.csv` | `1000` | human-written benign email corpus with `sender/receiver/date/subject/body/urls/label`; date range roughly `2007-07-05` to `2019-10-29` | Use a cautious description such as `human email corpus used in S2` until the original source page is recovered |
| `Datasets/LLM-Phishing/human-generated/phishing.csv` | `1000` | human phishing email corpus; date range roughly `2019-08-19` to `2022-12-27` | Same recommendation as above |
| `Datasets/LLM-Phishing/llm-generated/legit.csv` | `1000` | LLM-generated benign email set | `project-provided LLM-generated benign email set` is the safest description |
| `Datasets/LLM-Phishing/llm-generated/phishing.csv` | `1000` | LLM-generated phishing email set | Same recommendation as above |
| `Datasets/LLM-Phishing/S4-model-generated/` | multiple model folders | local model output dumps rather than a clearly identified public benchmark | Describe as `local model-generated phishing outputs` unless an upstream release page is recovered |

## 3. Key Processed Output Sizes in the Current Repository

These counts are not raw-source sizes. They are the processed outputs currently present in the repository and are useful when documenting the final working subsets.

### S4

- `S4-Scenarios-driven Adaptation/HW-B.csv`: `2639`
- `S4-Scenarios-driven Adaptation/HW-P.csv`: `1198`
- `S4-Scenarios-driven Adaptation/LLM-B.csv`: `5506`
- `S4-Scenarios-driven Adaptation/LLM-P.csv`: `11743`

### S1 / S5

- `S1-Basic Instruction/LLM-B.csv`: `2000`
- `S1-Basic Instruction/LLM-P.csv`: `673`
- `S5-Personalization for Credibility/LLM-P.csv`: `1228`

### S7 Quishing

- `S7-Cross-channel Expansion/HW-Quishing`: `4983` images
- `S7-Cross-channel Expansion/HW-QRCode`: `5008` images
- `S7-Cross-channel Expansion/LLM-Quishing`: `5300` images

### S7 Vishing / Script

- `S7-Cross-channel Expansion/HW-Script-single.csv`: `9186`
- `S7-Cross-channel Expansion/HW-Vishing-single.csv`: `10595`
- `S7-Cross-channel Expansion/LLM-Script-single.csv`: `5661`
- `S7-Cross-channel Expansion/LLM-Vishing-Single.csv`: `1419`
- `S7-Cross-channel Expansion/LLM-Script-Multi.csv`: `116`
- `S7-Cross-channel Expansion/LLM-Vishing-Multi.csv`: `206`

## 4. Recommended Wording for a Paper or Appendix

If this material is later reused in a paper, appendix, or repository landing page, the safest reporting strategy is:

1. For datasets with confirmed public provenance, report `public repository + paper + local subset size`.
2. For repository-internal generated artifacts or merged splits, do not overstate them as independent public benchmarks. Use wording such as `project-provided synthetic set` or `repo-internal composite split`.
3. For cases like `7-QuishingDataset`, where the public match is strong but not explicitly preserved in the local files, use wording such as `matched to` or `likely corresponding to` instead of a categorical claim.
4. Before a formal submission, manually re-check the original download or release pages for the most uncertain groups, especially `human-generated/*.csv` and `HW-Data/*.csv`.
