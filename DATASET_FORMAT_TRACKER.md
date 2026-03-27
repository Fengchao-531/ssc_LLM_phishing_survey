# Dataset Format Conversion Tracker

Goal: every raw artifact under `Datasets/` must exist in two downstream views
(1) a unified CSV whose header is the union of all source columns, and
(2) an `.eml` message tree suitable for downstream inspection/testing.

`Evaluation/evl.py` currently focuses on the highest-priority corpora (S4,
5Ham, LegitPhish). The tables below capture what is already scripted versus
what still needs planning so I can stage follow-up commits quickly.

## Status Legend

| Status          | Meaning                                                                 |
|-----------------|-------------------------------------------------------------------------|
| ✅ Done         | Conversion verified, CSV + `.eml` artifacts exist.                      |
| 🛠 Script-ready | Logic implemented (see `Evaluation/evl.py`); needs execution/QA run.     |
| 🚧 In progress  | Partially analyzed; format quirks still being mapped.                   |
| ⚪️ To do        | Untouched; requires scoping + conversion plan.                          |
| ⛔ Blocked      | Waiting on guidance (missing creds, unclear structure, duplicates, etc) |

## Scripted in `Evaluation/evl.py`

| Scope | Raw files (relative to `Datasets/`) | Conversion notes | Status | Follow-up |
|-------|-------------------------------------|------------------|--------|-----------|
| S4 CSV emails | `HW-Phishing/S4-phishing.csv`, `HW-Benign/S4-data_extracted_easy_ham.csv`, `HW-Benign/S4-data_extracted_hard_ham.csv` | DictReader ➜ normalized CSV row + synthesized `.eml` (plain or HTML) with metadata headers. | 🛠 Script-ready | Run once to materialize `Evaluation/processed/combined_records.csv` + `eml/S4/`, verify sample rows. |
| S4 text templates | `HW-Benign/S4-emails.json`, `HW-Benign/S4-outlook-samples.json` | Regex parse pseudo-JSON ➜ subject/body split ➜ `.eml` stub. | 🛠 Script-ready | Need QA on template parsing edge cases (quotes, HTML entities). |
| S4 LLM benign prompts | `LLM-Benign/S4-ephishLLM.json` | JSON ➜ `Language` + `type` preserved; `.eml` stub for each prompt. | 🛠 Script-ready | Confirm label semantics (`type` value meaning) before marking complete. |
| 5Ham corpus | `HW-Benign/5Ham.zip` | Iterate `.eml` files inside zip, copy raw message, extract headers/body into CSV union. | 🛠 Script-ready | First real run will write ~3k `.eml` files → ensure disk space + sampling QA. |
| LegitPhish features | `HW-*/LegitPhish Dataset.zip` (identical copies) | CSV row ➜ narrative body text; `.eml` stub per URL. | 🛠 Script-ready | Pick a single canonical copy (recommend `HW-Benign`) to avoid duplicate rows. |

## Remaining Raw Assets

### HW-Benign

| Dataset / file | Raw format | Required action for CSV + `.eml` | Status | Notes |
|----------------|------------|----------------------------------|--------|-------|
| `S4-phishyai-main/` | Mixed code/data folder | Inventory contents → decide whether to treat as text corpus or ignore (could be tooling repo). | 🚧 In progress | Haven't opened; may only hold scripts. Need confirmation before investing time. |
| `7archive (1).zip` | Archive of unknown structure | Unzip, catalog file types, define extraction strategy. | ⚪️ To do | Large (≈706 MB); plan chunked processing. |
| `7Composite.zip` | Zip (duplicate also in HW-Phishing) | Same as above; dedupe across folders. | ⚪️ To do | Decide canonical storage to avoid double counting. |
| `7phishing_site_urls.csv` | CSV | Decide how to render URL-only rows into `.eml` (e.g., template w/ URL body). | ⚪️ To do | Could share logic with LegitPhish body builder. |
| `7QRCodearchive.zip` | Zip | Inspect for image/text payload; define conversion rules. | ⚪️ To do | Might require OCR before `.eml` creation. |
| `phiusiil+phishing+url+dataset.zip` | Zip | Same treatment as other URL datasets. | ⚪️ To do | Need to understand schema first. |

### HW-Phishing

| Dataset / file | Raw format | Required action | Status | Notes |
|----------------|------------|-----------------|--------|-------|
| `5Mail_password_is_mail.zip` | Zip (likely `.eml`) | Extract + normalize like 5Ham, but label as phishing. | ⚪️ To do | Need password? file name hints default creds; verify. |
| `5nazario-password-is-nazario.zip` | Zip | Same as above. | ⚪️ To do | Password likely “nazario”; confirm. |
| `5phishpot-password-is-phishpot.zip` | Zip | Same as above. | ⚪️ To do | Password “phishpot”? verify before scripting. |
| `7-QuishingDataset.zip` | Zip | Inspect content (likely HTML/PDF). Determine CSV schema + `.eml` representation. | ⚪️ To do | Might mix QR images + descriptions. |
| `7Composite.zip`, `7phishing_site_urls.csv`, `7QRCodearchive.zip`, `phiusiil+...zip`, `LegitPhish Dataset.zip` | Same assets as benign folder | De-duplicate processing plan so we only ingest once but keep label metadata. | 🚧 In progress | Need decision on whether copies represent distinct labels or redundant backups. |

### LLM-Benign

| Dataset / file | Raw format | Action | Status | Notes |
|----------------|------------|--------|--------|-------|
| `1 Paladin-main.zip` | Large repo/archive | Determine whether it contains prompt/output pairs worth converting. | ⚪️ To do | 200+ MB; may need selective extraction. |

### LLM-Phishing

| Dataset / file | Raw format | Action | Status | Notes |
|----------------|------------|--------|--------|-------|
| `S4-model-generated/` (distilgpt2, gpt2, gpt2large, llama2, modelc) | Plain text dumps | Define segmentation heuristic (no clear delimiters) before CSV/`.eml` build. | 🚧 In progress | Need to agree on sentence/paragraph splitting to avoid garbage emails. |
| `1 Paladin-main.zip` | Archive | Same investigation as benign copy—understand contents. | ⚪️ To do | Possibly phishing variants of Paladin. |
| `ephishLLM.json` | JSON | Mirror handling of benign `S4-ephishLLM` but tag as phishing. | ⚪️ To do | Extend `evl.py` once benign side validated. |
| `KoBERT_dataset_v3.0.csv` | CSV | Inspect schema; determine mapping to email body (likely Korean?). | ⚪️ To do | Might need encoding handling + translation. |

### `sublist/` S1–S8 Prompt Buckets

Each subfolder currently only contains `.gitkeep` + `ReadMe.md`. No raw data to process yet, but once populated they must follow the same CSV/`.eml` pattern. Status: ⚪️ To do (awaiting data drop).

## Immediate Next Steps

1. Run `python Evaluation/evl.py --clean-output` to produce the first merged CSV/`.eml` drop for S4 + 5Ham + LegitPhish, then spot-check a few rows.
2. Decide canonical storage for duplicated zips (`7Composite`, `LegitPhish`, etc.) to avoid double counting.
3. Collect passwords/format notes for the “5*” phishing archives so we can script extraction similar to 5Ham.
4. Draft parsing heuristics for `S4-model-generated` text dumps (e.g., delimiter insertion, min-length filters) before wiring them into the pipeline.
