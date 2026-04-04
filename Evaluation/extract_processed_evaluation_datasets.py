#!/usr/bin/env python3
"""Extract evaluation-ready stage CSVs from detector result files."""

from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path
from typing import Iterable

import pandas as pd

SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parent
DEFAULT_INPUT_DIR = REPO_ROOT / "Detectors" / "output" / "LLM-result"
DEFAULT_OUTPUT_DIR = SCRIPT_DIR / "processed-evaluation-datasets"

STAGE_RESULTS_RE = re.compile(r"^(S\d+(?:-[A-Za-z0-9]+)*)_results\.csv$")

OUTPUT_COLUMNS = [
    "subject",
    "body",
    "label",
    "llm_guard_prediction",
    "phishing_email_agent_prediction",
    "email_phishing_detection_v3_prediction",
    "pyrit_original_prediction",
    "pyrit_blocklist_prediction",
]

COLUMN_ALIASES = {
    "subject": ("Subject", "subject"),
    "body": ("Body", "body"),
    "label": ("label", "Label"),
    "llm_guard_prediction": ("llm_guard_prediction",),
    "phishing_email_agent_prediction": ("phishing_email_agent_prediction",),
    "email_phishing_detection_v3_prediction": ("email_phishing_detection_v3_prediction",),
    "pyrit_original_prediction": ("pyrit_original_prediction",),
    "pyrit_blocklist_prediction": ("pyrit_blocklist_prediction",),
}

NULL_TOKENS = {"", "nan", "none", "null", "na", "n/a"}
POSITIVE_TOKENS = {
    "1",
    "1.0",
    "true",
    "t",
    "yes",
    "y",
    "positive",
    "phishing",
    "malicious",
    "spam",
    "unsafe",
    "fraud",
    "flagged",
    "block",
    "blocked",
}
NEGATIVE_TOKENS = {
    "0",
    "0.0",
    "false",
    "f",
    "no",
    "n",
    "negative",
    "benign",
    "legitimate",
    "ham",
    "safe",
    "clean",
    "not flagged",
    "allow",
    "allowed",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Extract subject/body/label and the five detector predictions into stage CSVs."
    )
    parser.add_argument(
        "--input-dir",
        type=Path,
        default=DEFAULT_INPUT_DIR,
        help=f"Directory containing *_results.csv files (default: {DEFAULT_INPUT_DIR})",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
        help=f"Directory where processed stage CSVs are written (default: {DEFAULT_OUTPUT_DIR})",
    )
    return parser.parse_args()


def normalize_binary_value(value: object) -> str:
    if value is None:
        return ""

    text = str(value).strip()
    lowered = text.lower()
    if lowered in NULL_TOKENS:
        return ""
    if lowered in POSITIVE_TOKENS:
        return "1"
    if lowered in NEGATIVE_TOKENS:
        return "0"

    try:
        numeric = float(text)
    except ValueError:
        return ""

    if numeric == 1.0:
        return "1"
    if numeric == 0.0:
        return "0"
    return ""


def build_lower_lookup(columns: Iterable[str]) -> dict[str, str]:
    return {column.lower(): column for column in columns}


def resolve_column(columns: Iterable[str], aliases: Iterable[str]) -> str | None:
    lower_lookup = build_lower_lookup(columns)
    for alias in aliases:
        matched = lower_lookup.get(alias.lower())
        if matched:
            return matched
    return None


def stage_sort_key(path: Path) -> tuple[int, str]:
    match = STAGE_RESULTS_RE.match(path.name)
    if not match:
        return (sys.maxsize, path.name)
    stage_name = match.group(1)
    number_match = re.match(r"^S(\d+)", stage_name)
    stage_number = int(number_match.group(1)) if number_match else sys.maxsize
    return (stage_number, stage_name)


def find_stage_files(input_dir: Path) -> list[Path]:
    files = [path for path in input_dir.glob("*_results.csv") if STAGE_RESULTS_RE.match(path.name)]
    return sorted(files, key=stage_sort_key)


def process_file(input_path: Path, output_dir: Path) -> Path:
    match = STAGE_RESULTS_RE.match(input_path.name)
    if not match:
        raise ValueError(f"Unsupported file name: {input_path.name}")

    stage_name = match.group(1)
    df = pd.read_csv(input_path, dtype=str, keep_default_na=False)

    selected = {}
    for output_column in OUTPUT_COLUMNS:
        source_column = resolve_column(df.columns, COLUMN_ALIASES[output_column])
        if source_column is None:
            selected[output_column] = pd.Series([""] * len(df), index=df.index, dtype="string")
            continue
        selected[output_column] = df[source_column].fillna("").astype("string")

    processed_df = pd.DataFrame(selected, columns=OUTPUT_COLUMNS)
    processed_df["label"] = processed_df["label"].map(normalize_binary_value)
    for detector_column in OUTPUT_COLUMNS[3:]:
        processed_df[detector_column] = processed_df[detector_column].map(normalize_binary_value)

    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / f"{stage_name}.csv"
    processed_df.to_csv(output_path, index=False)
    return output_path


def main() -> int:
    args = parse_args()
    input_dir = args.input_dir.resolve()
    output_dir = args.output_dir.resolve()

    if not input_dir.exists():
        print(f"[ERROR] Input directory does not exist: {input_dir}", file=sys.stderr)
        return 1

    stage_files = find_stage_files(input_dir)
    if not stage_files:
        print(f"[ERROR] No stage result CSVs matched in: {input_dir}", file=sys.stderr)
        return 1

    print(f"Input directory : {input_dir}")
    print(f"Output directory: {output_dir}")
    print(f"Matched files   : {len(stage_files)}")

    written_paths: list[Path] = []
    for stage_file in stage_files:
        output_path = process_file(stage_file, output_dir)
        written_paths.append(output_path)

        processed_df = pd.read_csv(output_path, dtype=str, keep_default_na=False)
        detector_counts = {
            column: int((processed_df[column] != "").sum())
            for column in OUTPUT_COLUMNS[3:]
        }
        print(
            f"[OK] {stage_file.name} -> {output_path.name} | "
            f"rows={len(processed_df)} | detector_non_empty={detector_counts}"
        )

    print(f"Done. Wrote {len(written_paths)} processed stage CSV files.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
