#!/usr/bin/env python3
"""Run the Hugging Face phishbot/ScamLLM model on CSV email data."""

from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path
from typing import Any


SCRIPT_DIR = Path(__file__).resolve().parent
DEFAULT_INPUT_CSV = (
    Path("/scratch3/che489/FC-W2-SoK/ssc_LLM_phishing_survey")
    / "Datasets/sublist/S5-Personalization for Credibility/LLM-P.csv"
)
DEFAULT_BATCH_SIZE = 16
DEFAULT_MAX_LENGTH = 512
DEFAULT_OUTPUT_DIR = SCRIPT_DIR / "results"
MODEL_NAME = "phishbot/ScamLLM"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Read subject/body from a CSV, run ScamLLM on the merged text, "
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
    parser.add_argument("--model-name", default=MODEL_NAME)
    parser.add_argument("--batch-size", type=int, default=DEFAULT_BATCH_SIZE)
    parser.add_argument("--max-length", type=int, default=DEFAULT_MAX_LENGTH)
    parser.add_argument(
        "--device",
        type=int,
        default=-1,
        help="Transformers pipeline device. Use -1 for CPU, 0 for the first CUDA device.",
    )
    parser.add_argument("--cache-dir", type=Path, default=None)
    return parser.parse_args()


def normalize_text(value: Any) -> str:
    return (value or "").replace("\r\n", "\n").replace("\r", "\n").strip()


def build_content(subject: str, body: str) -> str:
    subject = normalize_text(subject)
    body = normalize_text(body)
    if subject and body:
        return f"{subject}\n\n{body}"
    return subject or body


def load_classifier(args: argparse.Namespace):
    try:
        from transformers import AutoModelForSequenceClassification, AutoTokenizer, pipeline
    except ImportError as exc:
        raise SystemExit(
            "transformers or torch is not installed in the current Python environment."
        ) from exc

    model_kwargs = {}
    if args.cache_dir:
        model_kwargs["cache_dir"] = str(args.cache_dir)

    tokenizer = AutoTokenizer.from_pretrained(args.model_name, **model_kwargs)
    model = AutoModelForSequenceClassification.from_pretrained(args.model_name, **model_kwargs)
    return pipeline(
        "text-classification",
        model=model,
        tokenizer=tokenizer,
        top_k=None,
        device=args.device,
    )


def normalize_prediction_label(raw_label: str) -> int | None:
    text = str(raw_label or "").strip().upper()
    if text in {"LABEL_1", "1"}:
        return 1
    if text in {"LABEL_0", "0"}:
        return 0
    return None


def batched(items: list[dict[str, str]], batch_size: int):
    for start in range(0, len(items), batch_size):
        yield items[start:start + batch_size]


def main() -> int:
    args = parse_args()
    input_csv = args.input_csv.resolve()
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    output_csv = output_dir / f"{input_csv.stem}_results.csv"

    if not input_csv.exists():
        raise SystemExit(f"Input CSV not found: {input_csv}")

    classifier = load_classifier(args)

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

    for batch in batched(rows, batch_size):
        texts = [row["_content"] for row in batch]
        batch_outputs = classifier(
            texts,
            truncation=True,
            max_length=args.max_length,
            batch_size=batch_size,
        )

        for row, model_output in zip(batch, batch_outputs):
            best = max(model_output, key=lambda item: float(item.get("score", 0.0)))
            output_rows.append(
                {
                    "subject": row["subject"],
                    "body": row["body"],
                    "label": row["label"],
                    "model_prediction": normalize_prediction_label(best.get("label", "")),
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
