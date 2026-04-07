#!/usr/bin/env python3
"""SecureNet-style phishing detector using Llama-3.1-8B-Instruct."""

from __future__ import annotations

import argparse
import csv
import json
import re
import sys
from pathlib import Path
from typing import Any


SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_DIR = SCRIPT_DIR.parents[2]
DEFAULT_INPUT_CSV = (
    PROJECT_DIR / "Datasets" / "sublist" / "S5-Personalization for Credibility" / "LLM-P.csv"
)
DEFAULT_OUTPUT_DIR = SCRIPT_DIR / "results"
DEFAULT_MODEL_PATH = Path(
    "/scratch2/pk79/fche0036/hf-cache/hub/models--meta-llama--Llama-3.1-8B-Instruct/"
    "snapshots/0e9e39f249a16976918f6564b8830bc894c89659"
)
DEFAULT_MAX_NEW_TOKENS = 220

SYSTEM_PROMPT = """You are a cybersecurity assistant performing phishing detection.

Follow these rules when analyzing the content:
1. Analyze emails for language, sender authenticity, and formatting.
2. Scrutinize URLs for anomalies and verify destination links.
3. Beware of suspicious SMS messages and verify sender identity and links.
4. Inspect HTML code for hidden threats and irregularities.
5. Verify website security and legitimacy before sharing personal data.
6. Be cautious with email attachments, especially from unknown sources.
7. Double-check inputs, apply critical thinking, and rectify any issues for improved phishing detection.

Return JSON only with exactly these keys:
{
  "is_phishing": 0 or 1,
  "reason": "brief explanation",
  "score": integer from 1 to 10
}

Do not include markdown fences, extra text, or additional keys.
"""


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Reproduce the SecureNet paper's LLM-style phishing detection workflow using "
            "Llama-3.1-8B-Instruct and save predictions to <input_stem>_results.csv."
        )
    )
    parser.add_argument("--input-csv", type=Path, default=DEFAULT_INPUT_CSV)
    parser.add_argument("--subject-column", default="Subject")
    parser.add_argument("--body-column", default="Body")
    parser.add_argument("--label-column", default="label")
    parser.add_argument("--sample-size", type=int, default=0, help="Use 0 for all rows.")
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
        help="Directory for the output CSV. Output filename is always <input_stem>_results.csv.",
    )
    parser.add_argument("--model-path", type=Path, default=DEFAULT_MODEL_PATH)
    parser.add_argument("--max-new-tokens", type=int, default=DEFAULT_MAX_NEW_TOKENS)
    parser.add_argument(
        "--device",
        default="auto",
        help="Device selection passed to Transformers. Use 'auto', 'cuda', or 'cpu'.",
    )
    return parser.parse_args()


def normalize_text(value: Any) -> str:
    return (value or "").replace("\r\n", "\n").replace("\r", "\n").strip()


def build_content(subject: str, body: str) -> str:
    subject = normalize_text(subject)
    body = normalize_text(body)
    if subject and body:
        return f"Subject: {subject}\n\nBody:\n{body}"
    if subject:
        return f"Subject: {subject}"
    if body:
        return f"Body:\n{body}"
    return ""


def load_rows(args: argparse.Namespace) -> list[dict[str, str]]:
    csv.field_size_limit(sys.maxsize)
    input_csv = args.input_csv.resolve()
    if not input_csv.exists():
        raise SystemExit(f"Input CSV not found: {input_csv}")

    rows: list[dict[str, str]] = []
    with input_csv.open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        if not reader.fieldnames:
            raise SystemExit(f"CSV has no header row: {input_csv}")
        required = [args.subject_column, args.body_column, args.label_column]
        missing = [name for name in required if name not in reader.fieldnames]
        if missing:
            raise SystemExit(f"CSV missing required columns {', '.join(missing)} in {input_csv}")

        for index, record in enumerate(reader, start=1):
            if args.sample_size > 0 and index > args.sample_size:
                break
            subject = normalize_text(record.get(args.subject_column, ""))
            body = normalize_text(record.get(args.body_column, ""))
            rows.append(
                {
                    "subject": subject,
                    "body": body,
                    "label": str(record.get(args.label_column, "")),
                    "_content": build_content(subject, body),
                }
            )

    if not rows:
        raise SystemExit(f"No rows loaded from {input_csv}")
    return rows


def build_user_prompt(content: str) -> str:
    return (
        "Classify the following message for phishing risk.\n\n"
        "Use the rules exactly, then return strict JSON only.\n\n"
        f"{content}"
    )


def load_generator(args: argparse.Namespace):
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    model_path = args.model_path.resolve()
    if not model_path.exists():
        raise SystemExit(f"Model path not found: {model_path}")

    tokenizer = AutoTokenizer.from_pretrained(model_path)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token

    model_kwargs: dict[str, Any] = {"torch_dtype": torch.bfloat16}
    if args.device == "auto":
        model_kwargs["device_map"] = "auto"
    elif args.device == "cuda":
        if not torch.cuda.is_available():
            raise SystemExit("CUDA requested but not available.")
        model_kwargs["device_map"] = {"": 0}
    elif args.device == "cpu":
        model_kwargs["device_map"] = "cpu"
        model_kwargs["torch_dtype"] = torch.float32
    else:
        raise SystemExit(f"Unsupported device value: {args.device}")

    model = AutoModelForCausalLM.from_pretrained(model_path, **model_kwargs)
    model.eval()
    return tokenizer, model


def extract_first_json(text: str) -> dict[str, Any]:
    cleaned = text.strip()
    candidates = [cleaned]
    fence_match = re.findall(r"\{.*?\}", cleaned, flags=re.DOTALL)
    candidates.extend(fence_match)

    start = cleaned.find("{")
    while start != -1:
        depth = 0
        for index in range(start, len(cleaned)):
            char = cleaned[index]
            if char == "{":
                depth += 1
            elif char == "}":
                depth -= 1
                if depth == 0:
                    candidates.append(cleaned[start : index + 1])
                    start = cleaned.find("{", start + 1)
                    break
        else:
            break

    for candidate in candidates:
        try:
            parsed = json.loads(candidate)
        except Exception:
            continue
        if isinstance(parsed, dict):
            return parsed
    raise ValueError(f"Could not parse JSON from model output: {text[:300]}")


def normalize_prediction(parsed: dict[str, Any]) -> tuple[int, int | str, str]:
    raw_pred = parsed.get("is_phishing", "")
    if isinstance(raw_pred, bool):
        prediction = int(raw_pred)
    elif isinstance(raw_pred, (int, float)):
        prediction = 1 if int(raw_pred) != 0 else 0
    elif isinstance(raw_pred, str):
        lowered = raw_pred.strip().lower()
        if lowered in {"1", "true", "yes", "phishing", "malicious"}:
            prediction = 1
        elif lowered in {"0", "false", "no", "legitimate", "benign"}:
            prediction = 0
        else:
            raise ValueError(f"Unsupported is_phishing value: {raw_pred}")
    else:
        raise ValueError(f"Unsupported is_phishing type: {type(raw_pred).__name__}")

    score = parsed.get("score", "")
    if isinstance(score, str):
        score = score.strip()
        if score.isdigit():
            score = int(score)
    reason = normalize_text(parsed.get("reason", ""))
    return prediction, score, reason


def generate_prediction(
    tokenizer,
    model,
    *,
    content: str,
    max_new_tokens: int,
) -> tuple[int, int | str, str, str]:
    import torch

    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": build_user_prompt(content)},
    ]
    model_inputs = tokenizer.apply_chat_template(
        messages,
        add_generation_prompt=True,
        return_tensors="pt",
        return_dict=True,
    )
    model_device = next(model.parameters()).device
    if hasattr(model_inputs, "to"):
        model_inputs = model_inputs.to(model_device)
    input_ids = model_inputs["input_ids"]
    attention_mask = model_inputs.get("attention_mask")
    if attention_mask is None:
        attention_mask = torch.ones_like(input_ids)

    with torch.no_grad():
        generated = model.generate(
            input_ids=input_ids,
            attention_mask=attention_mask,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            pad_token_id=tokenizer.pad_token_id,
            eos_token_id=tokenizer.eos_token_id,
        )

    new_tokens = generated[0][input_ids.shape[-1] :]
    raw_response = tokenizer.decode(new_tokens, skip_special_tokens=True).strip()
    parsed = extract_first_json(raw_response)
    prediction, score, reason = normalize_prediction(parsed)
    return prediction, score, reason, raw_response


def maybe_print_metrics(rows: list[dict[str, Any]]) -> None:
    try:
        labels = [int(row["label"]) for row in rows]
        predictions = [int(row["model_prediction"]) for row in rows]
    except Exception:
        return
    if not labels:
        return
    from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

    print(f"eval_accuracy={accuracy_score(labels, predictions):.4f}", flush=True)
    print(f"eval_precision={precision_score(labels, predictions, zero_division=0):.4f}", flush=True)
    print(f"eval_recall={recall_score(labels, predictions, zero_division=0):.4f}", flush=True)
    print(f"eval_f1={f1_score(labels, predictions, zero_division=0):.4f}", flush=True)


def main() -> int:
    args = parse_args()
    rows = load_rows(args)
    tokenizer, model = load_generator(args)

    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    output_csv = output_dir / f"{args.input_csv.resolve().stem}_results.csv"

    output_rows: list[dict[str, Any]] = []
    total = len(rows)
    for index, row in enumerate(rows, start=1):
        try:
            prediction, score, reason, raw_response = generate_prediction(
                tokenizer,
                model,
                content=row["_content"],
                max_new_tokens=args.max_new_tokens,
            )
        except Exception as exc:
            prediction = ""
            score = ""
            reason = f"ERROR: {exc}"
            raw_response = ""

        output_rows.append(
            {
                "subject": row["subject"],
                "body": row["body"],
                "label": row["label"],
                "model_prediction": prediction,
                "model_score": score,
                "model_reason": reason,
                "raw_response": raw_response,
            }
        )
        print(f"processed {index}/{total}", flush=True)

    with output_csv.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=[
                "subject",
                "body",
                "label",
                "model_prediction",
                "model_score",
                "model_reason",
                "raw_response",
            ],
        )
        writer.writeheader()
        writer.writerows(output_rows)

    maybe_print_metrics(output_rows)
    print(f"saved: {output_csv}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
