# Detectors

This directory contains the detector code used in the LLM phishing survey experiments. The public repository keeps runnable code, lightweight model artifacts, and toy input examples. Benchmark datasets, detector outputs, logs, cached chunks, and evaluation result CSVs are intentionally not committed.

## Public Layout

- `examples/sample_email_input.csv`: a small toy CSV for smoke tests.
- `Academic/run_academic_detectors.py`: batch runner for academic detectors when private benchmark CSVs are available locally.
- `Academic/email_detectors/`: academic detector scripts.
- `Academic/email_detectors/trained_models/`: lightweight model files that are safe to publish.
- `Industry/email_detectors/output/run_text_detectors.py`: unified runner for the public industry detector wrappers.
- `Industry/email_detectors/open-source-git/`: wrappers for third-party open-source detectors. Clone upstream projects into the placeholder subdirectories before running those wrappers.
- `Industry/email_detectors/spamassassin/`: local SpamAssassin configuration and install/update helpers.

## Included Academic Detectors

The public academic detector set is:

- `scamllm.py`: uses the Hugging Face model `phishbot/ScamLLM`.
- `pimref.py`: uses a PiMRef-style identity model and company knowledge base. Configure local paths with placeholders such as `/path/to/pimref/identity-model` and `/path/to/company_database_knowphish_v2.json`.
- `t5phishing.py`: uses local T5 model/tokenizer directories such as `/path/to/best_t5` and `/path/to/t5_tokenizer`.
- `ml_watermark_logreg.py`: uses the included `trained_models/ml_watermark_logreg_archive4_llm_only.pkl` by default.
- `xgboost.py`: uses the included `trained_models/xgboost_kaggle.pkl` by default.
- `securenet_llama.py`: uses a local Llama model path such as `/path/to/Llama-3.1-8B-Instruct`.

## Included Industry Detectors

The public industry detector set is:

- `open-source-git/Phishing-Email-Agent.py`
- `open-source-git/email-phishing-detection_V3.py`
- `spamassassin.py`

The public release includes only the detector scripts listed above. Other local-only detector experiments are omitted.

## Input Format

Detector input CSVs should use this schema:

```csv
Subject,Body,label,data_source
```

`label` should use `0` for benign and `1` for phishing. Public benchmark CSVs are represented by README placeholders only. Put local/private CSVs under the placeholder directories when reproducing the full benchmark.

## API Keys

Do not commit real API keys. Use environment variables or placeholder values:

```bash
export OPENROUTER_API_KEY="your_api_key"
```

## Quick Smoke Tests

Run from the repository root.

```bash
python -m pip install -r Detectors/requirements-public.txt
```

```bash
python Detectors/Academic/email_detectors/ml_watermark_logreg.py \
  --input-csv Detectors/examples/sample_email_input.csv \
  --output-dir /tmp/ssc_detector_smoke \
  --sample-size 2 \
  --skip-cross-val

python Detectors/Academic/email_detectors/xgboost.py \
  --input-csv Detectors/examples/sample_email_input.csv \
  --output-dir /tmp/ssc_detector_smoke \
  --sample-size 2 \
  --skip-eval
```

For SpamAssassin:

```bash
bash Detectors/Industry/email_detectors/spamassassin/install_spamassassin.sh
bash Detectors/Industry/email_detectors/spamassassin/update_rules.sh

python Detectors/Industry/email_detectors/output/run_text_detectors.py \
  --input-csv Detectors/examples/sample_email_input.csv \
  --stage-name SAMPLE \
  --detectors spamassassin \
  --checkpoint-every 2
```

For the OpenRouter-backed wrapper, set a placeholder in examples and a real value only in your local shell:

```bash
export OPENROUTER_API_KEY="your_api_key"
python Detectors/Industry/email_detectors/output/run_text_detectors.py \
  --input-csv Detectors/examples/sample_email_input.csv \
  --stage-name SAMPLE \
  --detectors email_phishing_detection_v3 \
  --openrouter-api-key "$OPENROUTER_API_KEY"
```

## Not Public

The following are excluded from GitHub:

- full benchmark detector inputs;
- generated detector outputs and merged results;
- temporary chunk folders, logs, manifests, and caches;
- large local model checkpoints;
- private API keys or service credentials.
