# Email Detectors

This directory contains detector scripts for email-oriented content scanning.

## Experiment Progress (2026-03-31)

This section is the working log for the current text-detection benchmark.

Current benchmark rule:

- For text detection, benign and phishing data should be merged within the same stage and source family (`LLM` or `HW`).
- The merged rows should be shuffled before detector runs.
- Labels stay unchanged: `0 = benign`, `1 = phishing`.
- Counts below are true CSV record counts, not `wc -l` line counts, because many email bodies contain embedded newlines.

### 1. Processed datasets currently ready

| Stage | Family | Benign | Phishing | Mixed total | Current state |
|---|---:|---:|---:|---:|---|
| S1-Basic Instruction | LLM | 2000 | 673 | 2673 | ready for mixed text detection |
| S1-Basic Instruction | HW | - | - | - | not prepared |
| S2-Role-Framed Prompting | LLM | 998 | 997 | 1995 | ready for mixed text detection |
| S2-Role-Framed Prompting | HW | 726 | 504 | 1230 | ready for mixed text detection |
| S3-Multi-turn Task Decomposition | LLM/HW | - | - | - | no processed detector input in this folder yet |
| S4-Scenarios-driven Adaptation | LLM | 5506 | 11743 | 17249 | ready for mixed text detection |
| S4-Scenarios-driven Adaptation | HW | 2639 | 1198 | 3837 | ready for mixed text detection |
| S5-Personalization for Credibility | LLM | 0 | 1228 | 1228 | phishing-only, not mix-ready yet |
| S5-Personalization for Credibility | HW-EML | 0 | 70 files | 70 files | phishing-only `.eml`, not mix-ready yet |
| S6-Stealthy Rewriting | test.csv | 0 | 20 | 20 | test-only, not main benchmark set |

### 2. Mixed datasets that should be built first

These are the five main text-detection combinations that are already supported by the processed sublists:

- `S1 + LLM`: `LLM-B.csv + LLM-P.csv -> 2673`
- `S2 + LLM`: `LLM-B.csv + LLM-P.csv -> 1995`
- `S2 + HW`: `HW-B.csv + HW-P.csv -> 1230`
- `S4 + LLM`: `LLM-B.csv + LLM-P.csv -> 17249`
- `S4 + HW`: `HW-B.csv + HW-P.csv -> 3837`

Recommended output naming:

- `S1-Basic Instruction/LLM-mixed.csv`
- `S2-Role-Framed Prompting/LLM-mixed.csv`
- `S2-Role-Framed Prompting/HW-mixed.csv`
- `S4-Scenarios-driven Adaptation/LLM-mixed.csv`
- `S4-Scenarios-driven Adaptation/HW-mixed.csv`

### 3. Detector status

| Detector | Input mode | Status | Evidence |
|---|---|---|---|
| `llm_guard.py` | CSV text | confirmed runnable and already produced outputs | `runs/S5-Personalization_for_Credibility_LLM-P__llm-guard__20260331_125956/` and `...130500/` |
| `open-source-git/Phishing-Email-Agent.py` | CSV text | confirmed runnable and already produced outputs | `open-source-git/runs/S5-Personalization_for_Credibility_LLM-P__Phishing-Email-Agent__20260331_121859/` |
| `open-source-git/email-phishing-detection_V3.py` | `.eml` or CSV text | confirmed runnable and already produced outputs | `open-source-git/runs/5-HW-P_EML__email-phishing-detection_V3__20260331_121613/` |
| `PyRIT-scan-original.py` | CSV text | CLI confirmed, but no benchmark output saved yet | `python PyRIT-scan-original.py --help` works |
| `PyRIT-scan-blocklist.py` | CSV text | CLI confirmed, but no benchmark output saved yet | `python PyRIT-scan-blocklist.py --help` works |
| `PyRIT-scan-FT.py` | train/status/detect | CLI confirmed, but no training or detection artifact saved yet | `python PyRIT-scan-FT.py --help` works |
| `oopspam.py` | CSV text | integrated wrapper for OOPSpam REST API; requires `OOPSPAM_API_KEY` | local integration added on 2026-03-31 |
| `garak_perspective.py` | CSV text | not currently runnable from this folder | `run_garak_perspective.sh` exists, but `garak_perspective.py` is missing |

Practical takeaway:

- Fully confirmed for immediate benchmarking: `llm_guard.py`, `Phishing-Email-Agent.py`, `email-phishing-detection_V3.py`
- Present but not benchmark-verified yet: `PyRIT-scan-original.py`, `PyRIT-scan-blocklist.py`, `PyRIT-scan-FT.py`
- External paid API, integrated but not benchmark-verified yet: `oopspam.py`
- Currently blocked: `garak_perspective.py`

### 4. Detector x dataset matrix

Legend:

- `In Progress`: currently running
- `Queued`: included in the current run, but not started yet
- `Done-pilot`: already run at least once, but not yet as a full mixed benchmark
- `Pending`: should be run next
- `N/A`: current input format does not match directly
- `Optional`: available for debugging only, not part of the main benchmark

| Dataset target | Size | llm_guard | Phishing-Email-Agent | email-phishing-detection_V3 | PyRIT-original | PyRIT-blocklist | PyRIT-FT |
|---|---:|---|---|---|---|---|---|
| S1-LLM-mixed | 2673 | In Progress | Queued | Queued | Queued | Queued | N/A |
| S2-LLM-mixed | 1995 | Pending | Pending | Pending | Pending | Pending | Pending |
| S2-HW-mixed | 1230 | Pending | Pending | Pending | Pending | Pending | Pending |
| S4-LLM-mixed | 17249 | Pending | Pending | Pending | Pending | Pending | Pending |
| S4-HW-mixed | 3837 | Pending | Pending | Pending | Pending | Pending | Pending |
| S5-LLM-P only | 1228 | Done-pilot | Done-pilot | Pending | Pending | Pending | Pending |
| S5-HW-P_EML only | 70 files | N/A | N/A | Done-pilot | N/A | N/A | N/A |
| S6-test.csv only | 20 | Optional | Optional | Optional | Optional | Optional | Optional |

### 5. Runs already completed

| Time | Dataset | Detector | Sample size | Status |
|---|---|---|---:|---|
| 2026-03-31 12:16 | `S5/5-HW-P_EML` | `email-phishing-detection_V3` | 1 | completed |
| 2026-03-31 12:18 | `S5/LLM-P.csv` | `Phishing-Email-Agent` | 20 | completed |
| 2026-03-31 12:59 | `S5/LLM-P.csv` | `llm_guard` | 1 | completed |
| 2026-03-31 13:05 | `S5/LLM-P.csv` | `llm_guard` | 1 | completed |

Notes:

- `runs/S5-Personalization_for_Credibility_LLM-P__llm-guard__20260331_125925/` exists but has no saved summary and can be treated as an incomplete/empty attempt.
- No full mixed benign+phishing benchmark has been completed yet.
- `S1-LLM-mixed` full run started on 2026-03-31 and is currently executing through the unified detector pipeline.
- The immediate next step after `S1` finishes is to run `S2-LLM-mixed`.

The detector scripts currently in this folder include:

- `PyRIT-scan-original.py`
- `PyRIT-scan-blocklist.py`
- `PyRIT-scan-FT.py`
- `oopspam.py`
- `llm_guard.py`
- `open-source-git/Phishing-Email-Agent.py`
- `open-source-git/email-phishing-detection_V3.py`
- `open-source-git/Phishing-Email-Agent.sh`
- `open-source-git/email-phishing-detection_V3.sh`
- `run_garak_perspective.sh` (helper exists, target Python script currently missing)
- `run_llm_guard.sh`
- `run_pyrit_scan.sh`
- `run_pyrit_scan_blocklist.sh`

These detector files are Python CLI scripts. You run them from bash with `python ...`.

## Difference

These two files are intentionally parallel:

- `PyRIT-scan-original.py`: the original PyRIT-based scan that uses PyRIT's built-in Azure Content Safety scorer and default harm categories
- `PyRIT-scan-blocklist.py`: a blocklist-oriented variant that keeps a very similar CLI and output structure, but sends `blocklistNames` directly to Azure Content Safety for phishing-oriented matching
- `PyRIT-scan-FT.py`: a custom-category variant for training and using a trained detector through Azure Content Safety custom categories

Use `original` when you want:

- the untouched PyRIT-style Azure Content Safety scan
- default content safety categories such as `Hate`, `Sexual`, `Violence`, and `SelfHarm`
- a baseline that stays close to the official PyRIT scorer flow

Use `blocklist` when you want:

- phishing-oriented detection
- explicit blocklist hits in `blocklistsMatch`
- control over the keywords and phrases that should trigger detection

If `garak_perspective.py` is restored later, use it when you want:

- a lightweight text-only detector
- a numeric spam score from Google Perspective API
- a phishing-adjacent signal for suspicious unsolicited email text

Use `email-phishing-detection_V3.py` when you want:

- the official `hithamO/email-phishing-detection_V3` detector workflow
- `.eml` files from the processed sublist folders
- a CSV export that records subject/body plus the upstream detector outputs

## Detector: `garak_perspective.py`

`garak_perspective.py` sends text to Perspective API and requests only the `SPAM` attribute.

This detector is useful when you only have plain text and want a simple score-based signal without using Azure Content Safety or PyRIT.

Important limitation:

- this is a `spam` detector, not a dedicated `phishing` classifier
- it works best as a proxy signal for suspicious unsolicited email text
- high-quality phishing messages may still receive a low spam score

### Requirements

You need:

- Python
- `requests`
- a Perspective API key

Set the key with:

```bash
export PERSPECTIVE_API_KEY="<your-api-key>"
```

### How to run

#### 1. Scan one text string

```bash
python garak_perspective.py --text "Urgent: verify your account immediately at https://bit.ly/example"
```

#### 2. Scan text from stdin

```bash
echo "Urgent: verify your account immediately at https://bit.ly/example" | python garak_perspective.py
```

#### 3. Scan a CSV file

If your CSV has a text column called `text`:

```bash
python garak_perspective.py \
  --input-csv ./input.csv \
  --output-csv ./output.csv \
  --text-column text
```

If `--output-csv` is omitted, the script writes:

```text
<input_filename>.perspective_spam.csv
```

### Options

- `--language`: language code sent to Perspective API, default `en`
- `--threshold`: spam threshold for the boolean verdict, default `0.5`

### Output

For single-text mode, the script prints JSON containing:

- `decision.score`: the Perspective `SPAM` score
- `decision.threshold`: the threshold used for verdicting
- `decision.is_spam`: `true` if score >= threshold
- `perspective_result.raw_response`: the full API response

For CSV mode, the script appends:

- `perspective_spam_results_json`
- `perspective_spam_elapsed_seconds`
- `perspective_spam_error`
- `perspective_spam_score`
- `perspective_spam_is_spam`

### Bash helper

This directory also includes:

- `run_garak_perspective.sh`

You can run it with:

```bash
chmod +x run_garak_perspective.sh
./run_garak_perspective.sh
```

## Detector: `email-phishing-detection_V3.py`

### What it does

`email-phishing-detection_V3.py` is a thin batch wrapper around the public GitHub project `hithamO/email-phishing-detection_V3`.

The upstream project already expects `.eml` and `.msg` files. This wrapper follows that official flow and adds only the benchmark-side batching:

- it selects `.eml` files from a folder
- it calls the upstream `main.py -f <email> -o <json>` detector for each file
- it exports a CSV containing the email text and the detector outputs
- it saves the raw upstream JSON and console log for every processed email

### Requirements

You need:

- Python 3

You will typically want the upstream dependencies available in the Python environment used to run the official detector.

### How to run

#### 1. Run the default 20-email sample

```bash
python3 email-phishing-detection_V3.py
```

This defaults to the processed sublist EML folder:

```text
Datasets/sublist/S5-Personalization for Credibility/5-HW-P_EML
```

#### 2. Choose a different EML folder and sample size

```bash
python3 email-phishing-detection_V3.py \
  --eml-dir ./5-HW-P_EML \
  --sample-size 20 \
  --output-dir ./runs/email-phishing-detection_V3_test
```

#### 3. Override the Python interpreter used for the upstream detector

```bash
python3 email-phishing-detection_V3.py \
  --python-bin /path/to/python \
  --sample-size 20
```

If `--output-dir` is omitted, the script writes into:

```text
Detectors/Industry/email_detectors/runs/email-phishing-detection_V3_<timestamp>/
```

### Options

- `--eml-dir`: folder containing `.eml` files
- `--sample-size`: number of `.eml` files to test, default `20`
- `--repo-dir`: local clone path for the upstream repo
- `--python-bin`: Python interpreter used for upstream `main.py`
- `--output-dir`: where to write CSV, JSON, logs, and manifest

### Output

The wrapper writes:

- `email-phishing-detection_V3_sample.csv`
- `run_manifest.json`
- `detector_json/*.json`
- `detector_logs/*.log`

The CSV records:

- source email path and filename
- subject and extracted body text
- upstream detector status and duration
- suspicious indicator counts from headers/body/attachments
- a simple summary label derived from the upstream output
- paths to the saved upstream JSON and detector log

### Bash helper

This directory also includes:

- `run_email-phishing-detection_V3.sh`

You can run it with:

```bash
chmod +x run_email-phishing-detection_V3.sh
./run_email-phishing-detection_V3.sh
```

## Detector: `PyRIT-scan-blocklist.py`

### What it does

`PyRIT-scan-blocklist.py` is a phishing detector built around the official Azure Content Safety request structure for text analysis with blocklists.

It is designed for phishing-oriented screening, unlike `PyRIT-scan-original.py`, which is the untouched original scan.

It supports:

- scanning one piece of text from `--text`
- scanning one piece of text from `stdin`
- scanning an entire CSV file row by row
- recording per-row phishing verdicts, reasons, and runtime into a new CSV

### Requirements

You need:

- Python
- Azure Content Safety endpoint
- Azure Content Safety API key
- blocklist names that already exist in your Azure Content Safety resource

### How to run

#### 1. Scan one text string

```bash
python PyRIT-scan-blocklist.py --text "Urgent: verify your account immediately at https://bit.ly/example"
```

#### 2. Scan text from stdin

```bash
echo "Urgent: verify your account immediately at https://bit.ly/example" | python PyRIT-scan-blocklist.py
```

#### 3. Scan a CSV file

If your CSV has a text column called `text`:

```bash
python PyRIT-scan-blocklist.py \
  --input-csv ./input.csv \
  --output-csv ./output.csv \
  --text-column text
```

If your text column is called `Record`:

```bash
python PyRIT-scan-blocklist.py \
  --input-csv ./input.csv \
  --output-csv ./output.csv \
  --text-column Record
```

If `--output-csv` is omitted, the script writes:

```text
<input_filename>.phishing_detection.csv
```

### Blocklist configuration

This detector sends `blocklistNames` to Azure Content Safety using the official request format.

The built-in phishing blocklist blueprint name in the script is:

- `phishing-detection`

This blocklist name must already exist in your Azure Content Safety resource if you want detection to succeed.

### How The Blocklist Was Designed

The single `phishing-detection` blocklist is a merged phishing phrase set. I intentionally collapsed everything into one list because you asked not to maintain multiple blocklists.

The items were designed by combining common phishing signals from five practical email attack patterns:

- credential theft: `verify your account`, `reset your password`, `sign in to your account`
- impersonation and authority cues: `microsoft security`, `it helpdesk`, `system administrator`
- payment and finance lures: `wire transfer`, `gift card`, `invoice attached`, `payment details`
- urgency and pressure language: `urgent`, `action required`, `final warning`, `within 24 hours`
- click and file-open lures: `click the link below`, `download attachment`, `open the secure portal`

So the design is not random and not copied from one official Microsoft list. It is a phishing-oriented merged blocklist intended to catch common social-engineering wording in emails.

In short:

- `original` = category-based moderation through PyRIT
- `blocklist` = phrase-based phishing screening through Azure Content Safety blocklists
- `FT` = train a custom category and then use the trained detector for inference

You can provide them either by environment variable:

```bash
export CONTENT_SAFETY_BLOCKLISTS='phishing-detection'
```

or by CLI:

```bash
python PyRIT-scan-blocklist.py \
  --text "Urgent: verify your account immediately at https://bit.ly/example" \
  --blocklists phishing-detection
```

### CSV output columns

The output CSV keeps all original columns and adds:

- `phishing_detection_results_json`
- `phishing_detection_elapsed_seconds`
- `phishing_detection_error`

Meaning:

- `phishing_detection_results_json`: the full Azure detection result, decision, and blocklist metadata as JSON
- `phishing_detection_elapsed_seconds`: runtime for that row
- `phishing_detection_error`: error text if that row failed

### Bash helper

This directory also includes:

- `run_pyrit_scan_blocklist.sh`

You can run it with:

```bash
chmod +x run_pyrit_scan_blocklist.sh
./run_pyrit_scan_blocklist.sh
```

## Detector: `PyRIT-scan-original.py`

`PyRIT-scan-original.py` uses Microsoft PyRIT's Azure Content Safety scorer to scan text content.

It is useful for general safety screening, but it is not a dedicated phishing classifier.

It supports:

- scanning one piece of text from `--text`
- scanning one piece of text from `stdin`
- scanning an entire CSV file row by row
- recording per-row scan output and runtime into a new CSV

### Requirements

You need:

- Python with `pyrit` installed
- Azure Content Safety access
- either:
  - `AZURE_CONTENT_SAFETY_API_KEY`
  - or Entra ID auth via `az login`

The endpoint must be set:

```bash
export AZURE_CONTENT_SAFETY_API_ENDPOINT="https://<your-resource>.cognitiveservices.azure.com/"
```

Optional if you use API key auth:

```bash
export AZURE_CONTENT_SAFETY_API_KEY="<your-api-key>"
```

### How to run

#### 1. Scan one text string

```bash
python PyRIT-scan-original.py --text "This is a test email body."
```

#### 2. Scan text from stdin

```bash
echo "This is a test email body." | python PyRIT-scan-original.py
```

#### 3. Scan a CSV file

If your CSV has a text column called `text`:

```bash
python PyRIT-scan-original.py \
  --input-csv ./input.csv \
  --output-csv ./output.csv \
  --text-column text
```

If your text column is called `email_body`:

```bash
python PyRIT-scan-original.py \
  --input-csv ./input.csv \
  --output-csv ./output.csv \
  --text-column email_body
```

If `--output-csv` is omitted, the script writes:

```text
<input_filename>.scanned.csv
```

Example:

- input: `emails.csv`
- default output: `emails.scanned.csv`

### CSV input expectations

Your input CSV must:

- contain a header row
- contain the text column named by `--text-column`

Example:

```csv
id,text
1,"Urgent: click this link to verify your account"
2,"Hello team, please find the meeting notes attached."
```

### CSV output columns

The output CSV keeps all original columns and adds:

- `pyrit_scan_results_json`
- `pyrit_scan_elapsed_seconds`
- `pyrit_scan_error`

Meaning:

- `pyrit_scan_results_json`: the raw PyRIT scoring result for that row, stored as JSON
- `pyrit_scan_elapsed_seconds`: runtime for that row
- `pyrit_scan_error`: error text if that row failed

### Optional category filter

You can limit scanning to specific Azure Content Safety categories:

```bash
python PyRIT-scan-original.py \
  --text "I hate you." \
  --categories hate violence
```

Supported category names:

- `hate`
- `self_harm`
- `sexual`
- `violence`

### Notes

- The first import of PyRIT can be slow on this environment.
- Empty CSV rows in the target text column are skipped and recorded in `pyrit_scan_error`.
- Single-text mode prints JSON to stdout.
- CSV mode writes a new CSV file and prints a short JSON summary to stdout.

### Quick reference

Show help:

```bash
python PyRIT-scan-original.py --help
```

## Detector: `PyRIT-scan-FT.py`

`PyRIT-scan-FT.py` is for Azure Content Safety custom categories.

It supports three actions:

- `train`: create a category version and start training
- `status`: check training/build status by operation ID
- `detect`: run the trained custom detector on one text or a CSV file

### Training requirements

You need:

- a supported Azure Content Safety region
- endpoint and API key
- a custom category name
- a definition
- a blob URL pointing to your training JSONL file

### Example commands

Create a category version and start training:

```bash
python PyRIT-scan-FT.py \
  --endpoint "https://<your-resource>.cognitiveservices.azure.com" \
  --api-key "<your-key>" \
  train \
  --category-name phishing-email \
  --definition "emails that attempt credential theft, impersonation, or payment fraud" \
  --sample-blob-url "https://<your-storage>/<container>/phishing-train.jsonl"
```

Check build status:

```bash
python PyRIT-scan-FT.py \
  --endpoint "https://<your-resource>.cognitiveservices.azure.com" \
  --api-key "<your-key>" \
  status \
  --operation-id "<operation-id>"
```

Run the trained detector on one text:

```bash
python PyRIT-scan-FT.py \
  --endpoint "https://<your-resource>.cognitiveservices.azure.com" \
  --api-key "<your-key>" \
  detect \
  --category-name phishing-email \
  --version 1 \
  --text "Urgent: verify your account immediately."
```

Run the trained detector on a CSV:

```bash
python PyRIT-scan-FT.py \
  --endpoint "https://<your-resource>.cognitiveservices.azure.com" \
  --api-key "<your-key>" \
  detect \
  --category-name phishing-email \
  --version 1 \
  --input-csv ./input.csv \
  --output-csv ./output.csv \
  --text-column Record
```
