## Detectors

This directory now has a unified text-benchmark entry point:

- `prepare_llm_result_datasets.py`
- `output/run_text_detectors.py`
- `output/run_llm_result_queue.sh`

What it does:

- `prepare_llm_result_datasets.py` builds the mixed LLM benchmark CSVs directly into `Detectors/output/LLM-result/`
- `output/run_text_detectors.py` takes one mixed CSV as input and writes one combined results CSV back into the same result bucket
- `output/run_text_detectors.py` checkpoint-writes the combined results CSV every 100 rows per detector
- `output/run_llm_result_queue.sh` is the execution checklist for the current LLM runs, one line per sublist
- the result buckets are kept simple and contain CSV files only

Main output location:

- `Detectors/output/LLM-result/S1.csv`
- `Detectors/output/LLM-result/S1_results.csv`
- `Detectors/output/HW-result/S1.csv`
- `Detectors/output/HW-result/S1_results.csv`

Recommended usage:

```bash
source Detectors/load_detector_env.sh

python Detectors/prepare_llm_result_datasets.py

python Detectors/output/run_text_detectors.py \
  --input-csv Detectors/output/LLM-result/S1.csv \
  --stage-name S1 \
  --checkpoint-every 100 \
  --detectors llm_guard phishing_email_agent email_phishing_detection_v3 pyrit_original pyrit_blocklist
```

Or use the queue-style bash:

```bash
bash Detectors/output/run_llm_result_queue.sh
```

Notes:

- default detectors are the currently benchmark-verified text detectors
- the queue script now includes both PyRIT detectors: `pyrit_original` and `pyrit_blocklist`
- `pyrit_ft` is intentionally not part of this unified pipeline
- `garak` is intentionally not part of this unified pipeline
- the top-level output only keeps two buckets: `LLM-result` and `HW-result`
- `LLM-result` and `HW-result` are intended to contain CSV files only
- `Detectors/output/` root holds the runner scripts; the CSV data/results stay inside `LLM-result` and `HW-result`
- for the current phase, build and run `LLM-result` first; keep `HW-result` empty until the HW data is ready
- `source Detectors/load_detector_env.sh` now loads both `OPENROUTER_API_KEY` and the Azure Content Safety variables used by the PyRIT detectors
