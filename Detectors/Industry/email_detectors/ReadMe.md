# Industry Email Detectors

This directory contains the public industry-detector wrappers used by the survey benchmark. Public GitHub keeps the runner and reproducibility instructions, but not benchmark outputs or private detector result files.

## Included Detectors

- `spamassassin.py`: local Apache SpamAssassin CLI wrapper.
- `open-source-git/Phishing-Email-Agent.py`: wrapper for a cloned third-party Phishing Email Agent project.
- `open-source-git/email-phishing-detection_V3.py`: wrapper for a cloned third-party email-phishing-detection V3 project.

Only the three detector entries above are part of the public release. Other local-only detector experiments are omitted.

## Required Input

Use CSV files with:

```csv
Subject,Body,label,data_source
```

`label` uses `0` for benign and `1` for phishing.

## Unified Runner

The public unified runner is:

```bash
python Detectors/Industry/email_detectors/output/run_text_detectors.py --help
```

Supported detector names are:

- `phishing_email_agent`
- `email_phishing_detection_v3`
- `spamassassin`

Example with SpamAssassin:

```bash
python Detectors/Industry/email_detectors/output/run_text_detectors.py \
  --input-csv Detectors/examples/sample_email_input.csv \
  --stage-name SAMPLE \
  --detectors spamassassin \
  --checkpoint-every 2
```

Example with an API-backed detector wrapper:

```bash
export OPENROUTER_API_KEY="your_api_key"

python Detectors/Industry/email_detectors/output/run_text_detectors.py \
  --input-csv Detectors/examples/sample_email_input.csv \
  --stage-name SAMPLE \
  --detectors email_phishing_detection_v3 \
  --openrouter-api-key "$OPENROUTER_API_KEY"
```

## Third-Party Repositories

The wrapper files are committed, but the third-party project directories are placeholders. Clone upstream implementations into:

```text
Detectors/Industry/email_detectors/open-source-git/Phishing-Email-Agent/
Detectors/Industry/email_detectors/open-source-git/email-phishing-detection_V3/
```

The wrappers accept `--repo-dir` if your local clone lives somewhere else.

## Outputs

Generated outputs are written under `Detectors/Industry/email_detectors/output/` by default and are ignored by Git. Do not commit result CSVs, temporary logs, chunk inputs, manifests, or API responses.
