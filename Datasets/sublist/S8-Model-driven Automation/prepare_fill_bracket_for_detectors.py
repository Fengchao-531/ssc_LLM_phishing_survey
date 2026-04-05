#!/usr/bin/env python3
"""Convert S8 fill-bracket model outputs into detector-ready CSVs."""

from __future__ import annotations

import argparse
import csv
import re
from pathlib import Path
from typing import Any


SCRIPT_DIR = Path(__file__).resolve().parent
DEFAULT_INPUT_DIR = SCRIPT_DIR / "Models-Output" / "fill-bracket-llama3_8b-full"
DEFAULT_OUTPUT_DIR = SCRIPT_DIR.parent.parent.parent / "Detectors" / "output" / "LLM-result"

DEFAULT_INPUTS = [
    "deepseek-r1-distill-qwen-7b-generated_output.fill_bracket.csv",
    "ministral-8b-generated_output.fill_bracket.csv",
    "llama-3.1-8b-generated_output.fill_bracket.csv",
]

SUBJECT_LINE_RE = re.compile(r'^\s*["\']?\s*subject\s*:\s*(.+?)\s*["\']?\s*$', re.IGNORECASE)
MARKDOWN_TITLE_RE = re.compile(r"^\s*(?:#{1,6}\s*)?\*\*(.+?)\*\*\s*$")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build detector-ready CSVs from the S8 fill-bracket full model outputs."
    )
    parser.add_argument(
        "--input-dir",
        type=Path,
        default=DEFAULT_INPUT_DIR,
        help="Directory containing the fill-bracket CSV files.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
        help="Directory where detector-ready CSVs will be written.",
    )
    parser.add_argument(
        "--input-files",
        nargs="+",
        default=list(DEFAULT_INPUTS),
        help="Ordered list of input CSV filenames to convert.",
    )
    return parser.parse_args()


def normalize_text(value: Any) -> str:
    text = "" if value is None else str(value)
    if text.lower() == "nan":
        return ""
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    text = text.replace("\ufeff", "").replace("\u200b", "").replace("\xa0", " ")
    return text.strip()


def unwrap_wrapping_quotes(text: str) -> str:
    value = normalize_text(text)
    if len(value) >= 2 and value[0] == value[-1] and value[0] in {'"', "'"}:
        return normalize_text(value[1:-1])
    return value


def clean_cell(value: Any) -> str:
    return unwrap_wrapping_quotes(value)


def parse_subject_and_body_from_text(text: str) -> tuple[str, str]:
    cleaned = clean_cell(text)
    if not cleaned:
        return "", ""

    lines = cleaned.splitlines()
    first_non_empty_index = None
    for index, line in enumerate(lines):
        if line.strip():
            first_non_empty_index = index
            break

    if first_non_empty_index is None:
        return "", ""

    first_line = lines[first_non_empty_index]
    subject_match = SUBJECT_LINE_RE.match(first_line)
    if subject_match:
        body = "\n".join(lines[first_non_empty_index + 1 :]).lstrip()
        return clean_cell(subject_match.group(1)), clean_cell(body)

    markdown_match = MARKDOWN_TITLE_RE.match(first_line)
    if markdown_match and len(markdown_match.group(1).strip()) <= 180:
        body = "\n".join(lines[first_non_empty_index + 1 :]).lstrip()
        return clean_cell(markdown_match.group(1)), clean_cell(body)

    return "", cleaned


def derive_subject_and_body(row: dict[str, str]) -> tuple[str, str]:
    generated_text = clean_cell(row.get("generated_text", ""))
    generated_subject = clean_cell(row.get("generated_subject", ""))
    generated_body = clean_cell(row.get("generated_body", ""))

    parsed_subject, parsed_body = parse_subject_and_body_from_text(generated_text)
    if not generated_subject and parsed_subject:
        generated_subject = parsed_subject
    if not generated_body and parsed_body:
        generated_body = parsed_body
    if not generated_body:
        generated_body = generated_text

    return generated_subject, generated_body


def detector_ready_name(input_name: str) -> str:
    stem = input_name
    if stem.endswith(".fill_bracket.csv"):
        stem = stem[: -len(".fill_bracket.csv")]
    elif stem.endswith(".csv"):
        stem = stem[: -len(".csv")]
    return f"S8-fill-bracket-full__{stem}.csv"


def iter_csv_rows(path: Path) -> tuple[list[dict[str, str]], list[str]]:
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        if not reader.fieldnames:
            raise SystemExit(f"CSV has no header row: {path}")
        return list(reader), list(reader.fieldnames)


def build_output_rows(input_path: Path) -> tuple[list[dict[str, str]], list[str]]:
    rows, _ = iter_csv_rows(input_path)
    output_rows: list[dict[str, str]] = []
    fieldnames = [
        "Subject",
        "Body",
        "label",
        "data_source",
        "model_alias",
        "join_key",
        "input_row_number",
        "reference_row_number",
        "company",
        "family",
        "access_type",
        "provider",
        "prompt_text",
        "generated_text",
        "generated_subject_raw",
        "generated_body_raw",
        "created_at",
    ]

    for row in rows:
        subject, body = derive_subject_and_body(row)
        model_alias = clean_cell(row.get("model_alias", "")) or input_path.stem
        output_rows.append(
            {
                "Subject": subject,
                "Body": body,
                "label": clean_cell(row.get("label", "")),
                "data_source": f"S8-fill-bracket-full:{model_alias}",
                "model_alias": model_alias,
                "join_key": clean_cell(row.get("join_key", "")),
                "input_row_number": clean_cell(row.get("input_row_number", "")),
                "reference_row_number": clean_cell(row.get("reference_row_number", "")),
                "company": clean_cell(row.get("company", "")),
                "family": clean_cell(row.get("family", "")),
                "access_type": clean_cell(row.get("access_type", "")),
                "provider": clean_cell(row.get("provider", "")),
                "prompt_text": clean_cell(row.get("prompt_text", "")),
                "generated_text": clean_cell(row.get("generated_text", "")),
                "generated_subject_raw": clean_cell(row.get("generated_subject", "")),
                "generated_body_raw": clean_cell(row.get("generated_body", "")),
                "created_at": clean_cell(row.get("created_at", "")),
            }
        )

    return output_rows, fieldnames


def write_csv(path: Path, rows: list[dict[str, str]], fieldnames: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def main() -> int:
    args = parse_args()
    input_dir = args.input_dir.expanduser().resolve()
    output_dir = args.output_dir.expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    for input_name in args.input_files:
        input_path = input_dir / input_name
        if not input_path.exists():
            raise SystemExit(f"Input CSV not found: {input_path}")
        output_rows, fieldnames = build_output_rows(input_path)
        output_path = output_dir / detector_ready_name(input_name)
        write_csv(output_path, output_rows, fieldnames)
        print(f"[prepared] {input_path} -> {output_path} ({len(output_rows)} rows)")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
