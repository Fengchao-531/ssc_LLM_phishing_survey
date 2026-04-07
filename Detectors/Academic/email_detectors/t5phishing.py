#!/usr/bin/env python3
"""Run the uploaded T5 phishing detector on CSV email data."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any

import torch
import torch.nn as nn


SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_DIR = SCRIPT_DIR.parents[2]
DEFAULT_INPUT_CSV = (
    PROJECT_DIR / "Datasets" / "sublist" / "S5-Personalization for Credibility" / "LLM-P.csv"
)
DEFAULT_OUTPUT_DIR = SCRIPT_DIR / "results"
DEFAULT_MODEL_DIR = Path("/scratch2/pk79/fche0036/hf-cache/hub/best_t5")
DEFAULT_TOKENIZER_DIR = Path("/scratch2/pk79/fche0036/hf-cache/hub/t5_tokenizer")
DEFAULT_BATCH_SIZE = 8
DEFAULT_MAX_LENGTH = 512


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Read subject/body from a CSV, run the uploaded T5 phishing model on the merged text, "
            "and save subject/body/label/model_prediction to <input_stem>_results.csv."
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
    parser.add_argument("--model-dir", type=Path, default=DEFAULT_MODEL_DIR)
    parser.add_argument("--tokenizer-dir", type=Path, default=DEFAULT_TOKENIZER_DIR)
    parser.add_argument("--batch-size", type=int, default=DEFAULT_BATCH_SIZE)
    parser.add_argument("--max-length", type=int, default=DEFAULT_MAX_LENGTH)
    parser.add_argument(
        "--device",
        type=int,
        default=-1,
        help="Use -1 for CPU, 0 for the first CUDA device.",
    )
    return parser.parse_args()


def normalize_text(value: Any) -> str:
    return (value or "").replace("\r\n", "\n").replace("\r", "\n").strip()


def build_content(subject: str, body: str) -> str:
    subject = normalize_text(subject)
    body = normalize_text(body)
    if subject and body:
        return f"{subject}\n\n{body}"
    return subject or body


def resolve_device(device_index: int) -> torch.device:
    if device_index >= 0 and torch.cuda.is_available():
        return torch.device(f"cuda:{device_index}")
    return torch.device("cpu")


class T5ForClassification(nn.Module):
    def __init__(self, model_dir: Path) -> None:
        super().__init__()
        try:
            from transformers import T5EncoderModel
        except ImportError as exc:
            raise SystemExit(
                "transformers or torch is not installed in the current Python environment."
            ) from exc

        self.t5_encoder = T5EncoderModel.from_pretrained(str(model_dir))
        self.classifier = nn.Linear(self.t5_encoder.config.d_model, 2)

    def forward(self, input_ids: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        outputs = self.t5_encoder(input_ids=input_ids, attention_mask=attention_mask)
        pooled = outputs.last_hidden_state[:, 0, :]
        return self.classifier(pooled)


def load_tokenizer(tokenizer_dir: Path):
    try:
        from transformers import T5Tokenizer
    except ImportError as exc:
        raise SystemExit(
            "transformers is not installed in the current Python environment."
        ) from exc

    vocab_file = tokenizer_dir / "spiece.model"
    if not vocab_file.exists():
        raise SystemExit(f"Missing T5 tokenizer sentencepiece model: {vocab_file}")

    special_tokens_map_path = tokenizer_dir / "special_tokens_map.json"
    tokenizer_kwargs: dict[str, Any] = {"vocab_file": str(vocab_file)}
    if special_tokens_map_path.exists():
        special_tokens = json.loads(special_tokens_map_path.read_text(encoding="utf-8"))
        eos = special_tokens.get("eos_token", {})
        pad = special_tokens.get("pad_token", {})
        unk = special_tokens.get("unk_token", {})
        if isinstance(eos, dict) and eos.get("content"):
            tokenizer_kwargs["eos_token"] = eos["content"]
        if isinstance(pad, dict) and pad.get("content"):
            tokenizer_kwargs["pad_token"] = pad["content"]
        if isinstance(unk, dict) and unk.get("content"):
            tokenizer_kwargs["unk_token"] = unk["content"]
        additional = special_tokens.get("additional_special_tokens")
        if isinstance(additional, list):
            tokenizer_kwargs["additional_special_tokens"] = list(additional)

    try:
        return T5Tokenizer(**tokenizer_kwargs)
    except Exception as exc:
        raise SystemExit("Failed to load the T5 tokenizer from spiece.model.") from exc


def load_model(model_dir: Path, device: torch.device) -> T5ForClassification:
    if not model_dir.exists():
        raise SystemExit(f"T5 model directory not found: {model_dir}")

    linear_layer_path = model_dir / "linear_layer.bin"
    if not linear_layer_path.exists():
        raise SystemExit(f"Missing linear classification head: {linear_layer_path}")

    model = T5ForClassification(model_dir)
    state_dict = torch.load(linear_layer_path, map_location="cpu")
    model.classifier.load_state_dict(state_dict)
    model.to(device)
    model.eval()
    return model


def batched(items: list[dict[str, str]], batch_size: int):
    for start in range(0, len(items), batch_size):
        yield items[start:start + batch_size]


def main() -> int:
    args = parse_args()
    input_csv = args.input_csv.resolve()
    output_dir = args.output_dir.resolve()
    model_dir = args.model_dir.resolve()
    tokenizer_dir = args.tokenizer_dir.resolve()
    device = resolve_device(args.device)

    output_dir.mkdir(parents=True, exist_ok=True)
    output_csv = output_dir / f"{input_csv.stem}_results.csv"

    if not input_csv.exists():
        raise SystemExit(f"Input CSV not found: {input_csv}")
    if not tokenizer_dir.exists():
        raise SystemExit(f"T5 tokenizer directory not found: {tokenizer_dir}")

    tokenizer = load_tokenizer(tokenizer_dir)
    model = load_model(model_dir, device)

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

    batch_size = max(1, args.batch_size)
    total = len(rows)
    processed = 0
    output_rows: list[dict[str, Any]] = []

    with torch.no_grad():
        for batch in batched(rows, batch_size):
            texts = [row["_content"] for row in batch]
            encoded = tokenizer(
                texts,
                padding="max_length",
                truncation=True,
                max_length=args.max_length,
                return_tensors="pt",
            )
            input_ids = encoded["input_ids"].to(device)
            attention_mask = encoded["attention_mask"].to(device)
            logits = model(input_ids, attention_mask)
            predictions = torch.argmax(logits, dim=1).detach().cpu().tolist()

            for row, prediction in zip(batch, predictions):
                output_rows.append(
                    {
                        "subject": row["subject"],
                        "body": row["body"],
                        "label": row["label"],
                        "model_prediction": int(prediction),
                    }
                )

            processed += len(batch)
            print(f"processed {processed}/{total}", flush=True)

    with output_csv.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=["subject", "body", "label", "model_prediction"],
        )
        writer.writeheader()
        writer.writerows(output_rows)

    print(f"saved: {output_csv}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
