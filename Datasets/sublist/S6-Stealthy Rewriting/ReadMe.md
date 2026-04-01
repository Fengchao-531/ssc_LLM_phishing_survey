# S6 Stealthy Rewriting

This folder contains a lightweight reproduction script for the email rewriting setup described in:

- Utaliyeva et al., "ChatGPT: A Threat to Spam Filtering Systems," HPCC/DSS/SmartCity/DependSys 2023.

## What the paper does

The paper studies whether LLM rewriting can make spam emails look less spammy and therefore reduce the effectiveness of conventional ML spam filters.

The rewriting setup reported in the paper is:

- Model: `gpt-3.5-turbo`
- Main prompt used in the large-scale experiment:
  - `Rewrite the following email to be less spammy: 'email text'`
- Comparison prompt:
  - `Rewrite the following email: 'email text'`
- The paper also discusses another adversarial rewrite phrasing:
  - `Can you rewrite this email to be more legitimate: contents of the email`
- Emails larger than the historical `gpt-3.5-turbo` 4096-token limit were not rewritten.
- The paper notes that iterative adversarial rewriting can further reduce spam-like characteristics, although exact multi-round settings are not specified.

## What this script reproduces

`reproduce_utaliyeva2023_rephrasing.py` implements:

- Exact paper-style prompt templates
- Batch rewriting from an input CSV
- Optional iterative rewriting over multiple rounds
- Approximate historical token-limit filtering
- Output CSV for downstream detector evaluation
- Checkpoint CSV saving every 100 completed rows by default
- JSONL audit logs for every API call

## Expected input

The script is designed for the repository's common CSV format:

- `Subject`
- `Body`
- `label`
- `data_source`

It can also be pointed at custom column names through CLI flags.

## Example

Run on the phishing subset prepared in S6:

```bash
cd /scratch3/che489/FC-W2-SoK/ssc_LLM_phishing_survey
conda run -n FC-W2-gpu-p39 python -m pip install openai
export OPENAI_API_KEY=your_key_here
conda run -n FC-W2-gpu-p39 python "Datasets/sublist/S6-Stealthy Rewriting/reproduce_utaliyeva2023_rephrasing.py" \
  --input "Datasets/sublist/S6-Stealthy Rewriting/HW-P.csv" \
  --prompt-kind adversarial \
  --rounds 1 \
  --model gpt-3.5-turbo \
  --temperature 1.0 \
  --save-every 100
```

Compare adversarial vs general rewriting:

```bash
python3 "Datasets/sublist/S6-Stealthy Rewriting/reproduce_utaliyeva2023_rephrasing.py" \
  --input "Datasets/sublist/S6-Stealthy Rewriting/HW-P.csv" \
  --prompt-kind all \
  --rounds 1 \
  --model gpt-3.5-turbo
```

Try iterative rewriting:

```bash
python3 "Datasets/sublist/S6-Stealthy Rewriting/reproduce_utaliyeva2023_rephrasing.py" \
  --input "Datasets/sublist/S6-Stealthy Rewriting/HW-P.csv" \
  --prompt-kind adversarial \
  --rounds 3 \
  --model gpt-3.5-turbo
```

## Output

By default, each run creates a timestamped folder under:

`Datasets/sublist/S6-Stealthy Rewriting/runs/`

Artifacts:

- `LLM-GPT3.5.csv`: rewritten emails with metadata when using `gpt-3.5-turbo`
- `calls.jsonl`: one record per model call, including prompt and raw response
- `run_manifest.json`: CLI arguments and run summary

## Notes

- The system `python3` on some servers may still be `3.6`, which is too old for the current official `openai` package. Prefer running this script in `FC-W2-gpu-p39` or another Python 3.9+ environment.
- Historical fidelity: the paper used `gpt-3.5-turbo`. If that exact model is unavailable in your environment, use a close chat-capable replacement and record the model name in the manifest.
- The script now uses the official `openai` Python SDK with `OpenAI()` and `chat.completions.create(...)` for `gpt-3.5-turbo` compatibility.
- The script now defaults to `temperature=1.0` to encourage more diverse rewrites during bulk generation. You can still override it from the CLI.
- The script writes a checkpoint CSV every `100` completed rows by default so large runs do not lose all progress.
- Token counting uses `tiktoken` if installed; otherwise it falls back to a character-based estimate.
