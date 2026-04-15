## Detectors

This directory now has a unified text-benchmark entry point:

- `prepare_llm_result_datasets.py`
- `Industry/email_detectors/output/run_text_detectors.py`
- `Industry/email_detectors/output/run_llm_result_queue.sh`

What it does:

- `prepare_llm_result_datasets.py` builds the mixed LLM benchmark CSVs directly into `Detectors/Industry/email_detectors/output/LLM-Ind/`
- `Industry/email_detectors/output/run_text_detectors.py` takes one mixed CSV as input and writes one combined results CSV back into the same result bucket
- `Industry/email_detectors/output/run_text_detectors.py` checkpoint-writes the combined results CSV every 100 rows per detector
- `Industry/email_detectors/output/run_llm_result_queue.sh` is the execution checklist for the current LLM runs, one line per sublist
- the result buckets are kept simple and contain CSV files only

Main output location:

- `Detectors/Industry/email_detectors/output/LLM-Ind/S1.csv`
- `Detectors/Industry/email_detectors/output/LLM-Ind/S1_results.csv`
- `Detectors/Industry/email_detectors/output/HW-result/datasets/S1-GD.csv`
- `Detectors/Industry/email_detectors/output/HW-result/HW-Ind/S1-GD_results.csv`
- `Detectors/Academic/email_detectors/LLM-Acad/S1.csv`
- `Detectors/Academic/email_detectors/HW-Acad/S1-GD.csv`

Recommended usage:

```bash
source Detectors/load_detector_env.sh

python Detectors/prepare_llm_result_datasets.py

python Detectors/Industry/email_detectors/output/run_text_detectors.py \
  --input-csv Detectors/Industry/email_detectors/output/LLM-Ind/S1.csv \
  --stage-name S1 \
  --checkpoint-every 100 \
  --detectors llm_guard phishing_email_agent email_phishing_detection_v3 pyrit_original pyrit_blocklist
```

Or use the queue-style bash:

```bash
bash Detectors/Industry/email_detectors/output/run_llm_result_queue.sh
```

Notes:

- default detectors are the currently benchmark-verified text detectors
- the queue script now includes both PyRIT detectors: `pyrit_original` and `pyrit_blocklist`
- `oopspam` is available in the unified runner, but is not part of the default detector list because it makes paid external API calls
- `pyrit_ft` is intentionally not part of this unified pipeline
- `garak` is intentionally not part of this unified pipeline
- industry detector outputs now live under `Detectors/Industry/email_detectors/output/`
- LLM mixed datasets and merged results are stored in `LLM-Ind`
- HW Generic-Data datasets stay in `HW-result/datasets`, and the merged industry outputs stay in `HW-result/HW-Ind`
- academic detector outputs live under `Detectors/Academic/email_detectors/LLM-Acad` and `Detectors/Academic/email_detectors/HW-Acad`
- `source Detectors/load_detector_env.sh` now loads `OPENROUTER_API_KEY`, the Azure Content Safety variables used by the PyRIT detectors, and the optional OOPSpam environment variables

OOPSpam example:

```bash
export OOPSPAM_API_KEY="your-oopspam-key"

python Detectors/Industry/email_detectors/output/run_text_detectors.py \
  --input-csv Detectors/Industry/email_detectors/output/LLM-Ind/S1.csv \
  --stage-name S1 \
  --checkpoint-every 20 \
  --detectors oopspam
```
