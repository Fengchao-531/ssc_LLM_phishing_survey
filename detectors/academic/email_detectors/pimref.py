#!/usr/bin/env python3
"""Run a PiMRef-style detector on CSV email data."""

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
DEFAULT_MODEL_DIR = Path("/scratch2/pk79/fche0036/hf-cache/pimref/checkpoints/identity-model")
DEFAULT_KNOWLEDGE_BASE = Path(
    "/scratch2/pk79/fche0036/hf-cache/pimref/checkpoints/company_database_knowphish_v2.json"
)
DEFAULT_BATCH_SIZE = 8
DEFAULT_MAX_LENGTH = 512
DEFAULT_OUTPUT_DIR = SCRIPT_DIR / "results"

EMAIL_DOMAIN_RE = re.compile(r"\b[A-Za-z0-9._%+-]+@([A-Za-z0-9.-]+\.[A-Za-z]{2,})\b")
URL_DOMAIN_RE = re.compile(
    r"\b(?:https?://)?(?:www\.)?([A-Za-z0-9.-]+\.[A-Za-z]{2,})(?:[/:][^\s]*)?\b",
    re.IGNORECASE,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Read subject/body from a CSV, run a PiMRef-style detector based on the released "
            "identity-model checkpoint and knowledge base, and save "
            "subject/body/label/model_prediction to <input_stem>_results.csv."
        )
    )
    parser.add_argument("--input-csv", type=Path, default=DEFAULT_INPUT_CSV)
    parser.add_argument("--subject-column", default="Subject")
    parser.add_argument("--body-column", default="Body")
    parser.add_argument("--label-column", default="label")
    parser.add_argument(
        "--sender-column",
        default="",
        help=(
            "Optional sender/email column. If provided and present in the CSV, its domain is used "
            "for knowledge-base verification."
        ),
    )
    parser.add_argument("--sample-size", type=int, default=0, help="Use 0 for all rows.")
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
        help="Directory for the output CSV. Output filename is always <input_stem>_results.csv.",
    )
    parser.add_argument("--model-dir", type=Path, default=DEFAULT_MODEL_DIR)
    parser.add_argument("--knowledge-base", type=Path, default=DEFAULT_KNOWLEDGE_BASE)
    parser.add_argument("--batch-size", type=int, default=DEFAULT_BATCH_SIZE)
    parser.add_argument("--max-length", type=int, default=DEFAULT_MAX_LENGTH)
    parser.add_argument(
        "--device",
        type=int,
        default=-1,
        help="Transformers pipeline device. Use -1 for CPU, 0 for the first CUDA device.",
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


def normalize_identity_key(value: str) -> str:
    text = re.sub(r"\s+", " ", str(value or "").strip())
    text = text.strip(" \t\r\n'\"`.,:;!?()[]{}<>")
    text = re.sub(r"\s+", " ", text)
    return text.casefold()


def load_detector(args: argparse.Namespace):
    try:
        from transformers import AutoModelForTokenClassification, AutoTokenizer, pipeline
    except ImportError as exc:
        raise SystemExit(
            "transformers or torch is not installed in the current Python environment."
        ) from exc

    if not args.model_dir.exists():
        raise SystemExit(f"PiMRef model directory not found: {args.model_dir}")

    tokenizer = AutoTokenizer.from_pretrained(str(args.model_dir))
    if args.max_length > 0:
        tokenizer.model_max_length = int(args.max_length)
    model = AutoModelForTokenClassification.from_pretrained(str(args.model_dir))
    return pipeline(
        "token-classification",
        model=model,
        tokenizer=tokenizer,
        aggregation_strategy="simple",
        device=args.device,
    )


def load_knowledge_base(path: Path) -> dict[str, set[str]]:
    if not path.exists():
        raise SystemExit(f"PiMRef knowledge base not found: {path}")

    raw = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(raw, dict):
        raise SystemExit(f"Unexpected knowledge base format: {path}")

    normalized: dict[str, set[str]] = {}
    for key, value in raw.items():
        lookup_key = normalize_identity_key(str(key))
        domains = set()
        if isinstance(value, list):
            for item in value:
                domain = normalize_domain(str(item))
                if domain:
                    domains.add(domain)
        if lookup_key and domains:
            normalized[lookup_key] = domains
    return normalized


def normalize_domain(value: str) -> str:
    text = str(value or "").strip().lower()
    text = re.sub(r"^(?:https?://)?(?:www\.)?", "", text)
    text = text.split("/", 1)[0]
    text = text.split(":", 1)[0]
    return text.strip(" .,;:!?()[]{}<>\"'")


def extract_domains(text: str) -> set[str]:
    found = set()
    normalized_text = normalize_text(text)
    for match in EMAIL_DOMAIN_RE.findall(normalized_text):
        domain = normalize_domain(match)
        if domain:
            found.add(domain)
    for match in URL_DOMAIN_RE.findall(normalized_text):
        domain = normalize_domain(match)
        if domain:
            found.add(domain)
    return found


def normalize_entity_group(item: dict[str, Any]) -> str:
    raw = str(item.get("entity_group") or item.get("entity") or "")
    return raw.split("-")[-1].strip().lower()


def unique_in_order(values: list[str]) -> list[str]:
    seen: set[str] = set()
    ordered: list[str] = []
    for value in values:
        key = normalize_identity_key(value)
        if not key or key in seen:
            continue
        seen.add(key)
        ordered.append(value)
    return ordered


def expected_domains_for_identities(
    identities: list[str],
    knowledge_base: dict[str, set[str]],
) -> set[str]:
    expected = set()
    for identity in identities:
        candidates = {
            normalize_identity_key(identity),
            normalize_identity_key(identity.removeprefix("The ")),
            normalize_identity_key(identity.removeprefix("the ")),
        }
        for candidate in candidates:
            if candidate and candidate in knowledge_base:
                expected.update(knowledge_base[candidate])
    return expected


def predict_pimref(
    *,
    content: str,
    sender_value: str,
    model_output: list[dict[str, Any]],
    knowledge_base: dict[str, set[str]],
) -> int:
    identities = unique_in_order(
        [str(item.get("word", "")).strip() for item in model_output if normalize_entity_group(item) == "identity"]
    )
    actions = unique_in_order(
        [str(item.get("word", "")).strip() for item in model_output if normalize_entity_group(item) == "action"]
    )

    expected_domains = expected_domains_for_identities(identities, knowledge_base)
    observed_domains = set()
    if sender_value:
        observed_domains.update(extract_domains(sender_value))
    observed_domains.update(extract_domains(content))

    has_contradiction = bool(
        identities
        and actions
        and expected_domains
        and observed_domains
        and expected_domains.isdisjoint(observed_domains)
    )
    return int(has_contradiction)


def batched(items: list[dict[str, str]], batch_size: int):
    for start in range(0, len(items), batch_size):
        yield items[start:start + batch_size]


def main() -> int:
    args = parse_args()
    csv.field_size_limit(sys.maxsize)
    input_csv = args.input_csv.resolve()
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    output_csv = output_dir / f"{input_csv.stem}_results.csv"

    if not input_csv.exists():
        raise SystemExit(f"Input CSV not found: {input_csv}")

    detector = load_detector(args)
    knowledge_base = load_knowledge_base(args.knowledge_base)

    rows: list[dict[str, str]] = []
    with input_csv.open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        if not reader.fieldnames:
            raise SystemExit(f"CSV has no header row: {input_csv}")

        required = [args.subject_column, args.body_column, args.label_column]
        missing = [name for name in required if name not in reader.fieldnames]
        if missing:
            raise SystemExit(f"CSV missing required columns {', '.join(missing)} in {input_csv}")

        sender_column = args.sender_column if args.sender_column in reader.fieldnames else ""
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
                    "sender_value": normalize_text(record.get(sender_column, "")) if sender_column else "",
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
        batch_outputs = detector(texts, batch_size=batch_size)

        for row, model_output in zip(batch, batch_outputs):
            prediction = predict_pimref(
                content=row["_content"],
                sender_value=row["sender_value"],
                model_output=model_output,
                knowledge_base=knowledge_base,
            )
            output_rows.append(
                {
                    "subject": row["subject"],
                    "body": row["body"],
                    "label": row["label"],
                    "model_prediction": prediction,
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
