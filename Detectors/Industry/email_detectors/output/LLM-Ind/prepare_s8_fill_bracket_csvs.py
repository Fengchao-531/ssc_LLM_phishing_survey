#!/usr/bin/env python3
"""Prepare S8 fill-bracket detector CSVs in the same style as S6 detector inputs."""

from __future__ import annotations

import argparse
import csv
import re
from pathlib import Path
from typing import Any


THIS_DIR = Path(__file__).resolve().parent
ROOT = THIS_DIR.parents[4]
SOURCE_DIR = (
    ROOT
    / "Datasets"
    / "sublist"
    / "S8-Model-driven Automation"
    / "Models-Output"
    / "fill-bracket-llama3_8b-full"
)

MODEL_SPECS = [
    (
        "deepseek-r1-distill-qwen-7b-generated_output.fill_bracket.csv",
        "S8-deepseek.csv",
        "deepseek",
    ),
    (
        "ministral-8b-generated_output.fill_bracket.csv",
        "S8-ministral.csv",
        "ministral",
    ),
    (
        "llama-3.1-8b-generated_output.fill_bracket.csv",
        "S8-llama.csv",
        "llama",
    ),
]

SUBJECT_LINE_RE = re.compile(r'^\s*["\']?\s*subject\s*:\s*(.+?)\s*["\']?\s*$', re.IGNORECASE)
MARKDOWN_TITLE_RE = re.compile(r"^\s*(?:#{1,6}\s*)?\*\*(.+?)\*\*\s*$")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Prepare S8 fill-bracket CSVs under Detectors/Industry/email_detectors/output/LLM-Ind."
    )
    parser.add_argument(
        "--source-dir",
        type=Path,
        default=SOURCE_DIR,
        help="Directory holding the original fill-bracket full CSV files.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=THIS_DIR,
        help="Directory where S8-deepseek.csv etc. should be written.",
    )
    return parser.parse_args()


def normalize_text(value: Any) -> str:
    text = "" if value is None else str(value)
    if text.lower() == "nan":
        return ""
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    text = text.replace("\ufeff", "").replace("\u200b", "").replace("\xa0", " ")
    return text.strip()


def clean_cell(value: Any) -> str:
    text = normalize_text(value)
    if len(text) >= 2 and text[0] == text[-1] and text[0] in {'"', "'"}:
        return normalize_text(text[1:-1])
    return text


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


def read_rows(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        if not reader.fieldnames:
            raise SystemExit(f"CSV has no header row: {path}")
        return list(reader)


def write_rows(path: Path, rows: list[dict[str, str]]) -> None:
    fieldnames = ["Subject", "Body", "label", "data_source"]
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def main() -> int:
    args = parse_args()
    source_dir = args.source_dir.expanduser().resolve()
    output_dir = args.output_dir.expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    for input_name, output_name, model_key in MODEL_SPECS:
        input_path = source_dir / input_name
        if not input_path.exists():
            raise SystemExit(f"Missing source file: {input_path}")
        source_rows = read_rows(input_path)
        output_rows = []
        for row in source_rows:
            subject, body = derive_subject_and_body(row)
            output_rows.append(
                {
                    "Subject": subject,
                    "Body": body,
                    "label": clean_cell(row.get("label", "")),
                    "data_source": f"S8-fill-bracket-full:{model_key}",
                }
            )
        output_path = output_dir / output_name
        write_rows(output_path, output_rows)
        print(f"[prepared] {output_path} ({len(output_rows)} rows)")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
