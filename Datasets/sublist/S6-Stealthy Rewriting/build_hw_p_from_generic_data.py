#!/usr/bin/env python3
"""Build a balanced phishing-only HW-P.csv from Generic-Data sources.

The script:
- loads every CSV under Datasets/Generic-Data
- keeps phishing rows only
- normalizes column names into Subject/Body/label/data_source
- removes exact duplicates within and across source files
- samples a balanced set across source files with a fixed seed
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import random
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Sequence


csv.field_size_limit(10**9)

SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parents[2]
DEFAULT_INPUT_DIR = REPO_ROOT / "Datasets" / "Generic-Data"
DEFAULT_OUTPUT_PATH = SCRIPT_DIR / "HW-P.csv"
DEFAULT_EXCLUDED_CSVS = ("CMU-232425.csv",)


@dataclass(frozen=True)
class Record:
    subject: str
    body: str
    label: str
    data_source: str
    dedupe_key: str


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build S6/HW-P.csv by sampling non-duplicate phishing emails from Generic-Data."
    )
    parser.add_argument(
        "--input-dir",
        default=str(DEFAULT_INPUT_DIR),
        help="Directory containing Generic-Data CSV files.",
    )
    parser.add_argument(
        "--output",
        default=str(DEFAULT_OUTPUT_PATH),
        help="Output CSV path.",
    )
    parser.add_argument(
        "--target-rows",
        type=int,
        default=1600,
        help="Number of phishing emails to sample.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=20260401,
        help="Random seed used for deterministic sampling.",
    )
    parser.add_argument(
        "--min-body-chars",
        type=int,
        default=1,
        help="Discard rows whose normalized body is shorter than this many characters.",
    )
    parser.add_argument(
        "--exclude-csv",
        action="append",
        default=list(DEFAULT_EXCLUDED_CSVS),
        help="CSV filename to exclude. Can be provided multiple times.",
    )
    return parser.parse_args()


def normalize_output_text(value: str) -> str:
    text = (value or "").replace("\r\n", "\n").replace("\r", "\n")
    text = text.replace("\ufeff", "").replace("\u200b", "").replace("\xa0", " ")
    text = re.sub(r"[ \t]+\n", "\n", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def normalize_for_key(value: str) -> str:
    text = normalize_output_text(value)
    return re.sub(r"\s+", " ", text).strip()


def build_dedupe_key(subject: str, body: str) -> str:
    normalized = "{}\n\n{}".format(normalize_for_key(subject), normalize_for_key(body))
    return hashlib.sha1(normalized.encode("utf-8")).hexdigest()


def get_subject(row: Dict[str, str]) -> str:
    return normalize_output_text(row.get("Subject", row.get("subject", "")))


def get_body(row: Dict[str, str]) -> str:
    return normalize_output_text(row.get("Body", row.get("body_text", "")))


def get_label(row: Dict[str, str]) -> str:
    return str(row.get("label", "1")).strip()


def get_output_source(csv_path: Path, original_source: str) -> str:
    source_stem = csv_path.stem
    original = (original_source or "").strip() or "unknown"
    return "generic-data:{}:{}".format(source_stem, original)


def load_unique_records(csv_path: Path, min_body_chars: int) -> List[Record]:
    unique_records: List[Record] = []
    seen_local = set()

    with csv_path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            label = get_label(row)
            if label != "1":
                continue

            subject = get_subject(row)
            body = get_body(row)
            if len(body) < min_body_chars:
                continue

            dedupe_key = build_dedupe_key(subject, body)
            if dedupe_key in seen_local:
                continue
            seen_local.add(dedupe_key)

            unique_records.append(
                Record(
                    subject=subject,
                    body=body,
                    label="1",
                    data_source=get_output_source(csv_path, str(row.get("data_source", ""))),
                    dedupe_key=dedupe_key,
                )
            )

    return unique_records


def compute_quotas(source_names: Sequence[str], target_rows: int) -> Dict[str, int]:
    base = target_rows // len(source_names)
    remainder = target_rows % len(source_names)
    quotas: Dict[str, int] = {}
    for index, source_name in enumerate(source_names):
        quotas[source_name] = base + (1 if index < remainder else 0)
    return quotas


def shuffle_records(records: List[Record], seed: int, source_name: str) -> None:
    source_seed = "{}:{}".format(seed, source_name)
    random.Random(source_seed).shuffle(records)


def main() -> int:
    args = parse_args()
    input_dir = Path(args.input_dir).resolve()
    output_path = Path(args.output).resolve()
    excluded_csvs = {name.strip() for name in args.exclude_csv if name.strip()}
    csv_paths = sorted(
        csv_path for csv_path in input_dir.glob("*.csv") if csv_path.name not in excluded_csvs
    )
    if not csv_paths:
        raise ValueError("No CSV files found in {}".format(input_dir))

    candidates_by_source: Dict[str, List[Record]] = {}
    for csv_path in csv_paths:
        candidates = load_unique_records(csv_path, min_body_chars=args.min_body_chars)
        if not candidates:
            continue
        candidates_by_source[csv_path.name] = candidates

    if not candidates_by_source:
        raise ValueError("No eligible phishing rows found in {}".format(input_dir))

    source_names = sorted(candidates_by_source)
    quotas = compute_quotas(source_names, args.target_rows)

    selected: List[Record] = []
    selected_keys = set()
    leftovers: List[Record] = []

    for source_name in source_names:
        candidates = list(candidates_by_source[source_name])
        shuffle_records(candidates, args.seed, source_name)

        chosen_count = 0
        for record in candidates:
            if record.dedupe_key in selected_keys:
                continue
            if chosen_count < quotas[source_name]:
                selected.append(record)
                selected_keys.add(record.dedupe_key)
                chosen_count += 1
            else:
                leftovers.append(record)

        print(
            "[source] {} quota={} selected={} unique_candidates={}".format(
                source_name, quotas[source_name], chosen_count, len(candidates_by_source[source_name])
            )
        )

    if len(selected) < args.target_rows:
        random.Random(args.seed).shuffle(leftovers)
        for record in leftovers:
            if len(selected) >= args.target_rows:
                break
            if record.dedupe_key in selected_keys:
                continue
            selected.append(record)
            selected_keys.add(record.dedupe_key)

    if len(selected) < args.target_rows:
        raise ValueError(
            "Only collected {} unique rows, fewer than requested {}".format(
                len(selected), args.target_rows
            )
        )

    selected = selected[: args.target_rows]
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=["Subject", "Body", "label", "data_source"])
        writer.writeheader()
        for record in selected:
            writer.writerow(
                {
                    "Subject": record.subject,
                    "Body": record.body,
                    "label": record.label,
                    "data_source": record.data_source,
                }
            )

    print("[done] wrote {} rows to {}".format(len(selected), output_path))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
