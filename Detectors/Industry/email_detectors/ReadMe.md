# Email Detectors

This directory contains detector scripts for email-oriented content scanning.

The current detector is:

- `PyRIT-scan-original.py`
- `PyRIT-scan-blocklist.py`
- `PyRIT-scan-FT.py`
- `garak_perspective.py`
- `run_garak_perspective.sh`

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

Use `garak_perspective.py` when you want:

- a lightweight text-only detector
- a numeric spam score from Google Perspective API
- a phishing-adjacent signal for suspicious unsolicited email text

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
