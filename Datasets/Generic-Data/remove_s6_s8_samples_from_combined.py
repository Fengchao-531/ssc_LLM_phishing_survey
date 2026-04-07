#!/usr/bin/env python3
"""Remove S6/S8-selected source rows from Generic-Data combined datasets."""

from __future__ import annotations

import argparse
import csv
import hashlib
import re
import sys
from collections import Counter
from pathlib import Path
from tempfile import NamedTemporaryFile
from typing import Dict, Iterable, List, Sequence


csv.field_size_limit(sys.maxsize)

SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parents[1]

DEFAULT_COMBINED_PATHS = {
    "P": SCRIPT_DIR / "combined-P.csv",
    "B": SCRIPT_DIR / "combined-B.csv",
}

DEFAULT_SAMPLE_PATHS = {
    "P": [
        REPO_ROOT / "Datasets" / "sublist" / "S6-Stealthy Rewriting" / "HW-P.csv",
        REPO_ROOT / "Datasets" / "sublist" / "S8-Model-driven Automation" / "HW-P.csv",
    ],
    "B": [
        REPO_ROOT / "Datasets" / "sublist" / "S6-Stealthy Rewriting" / "HW-B.csv",
        REPO_ROOT / "Datasets" / "sublist" / "S8-Model-driven Automation" / "HW-B.csv",
    ],
}

OUTPUT_COLUMNS = ["Subject", "Body", "label", "data_source"]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Delete rows already used by S6/S8 from combined-P.csv and combined-B.csv."
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print what would be removed without rewriting the files.",
    )
    return parser.parse_args()


def normalize_output_text(value: str) -> str:
    text = (value or "").replace("\r\n", "\n").replace("\r", "\n")
    text = text.replace("\ufeff", "").replace("\u200b", "").replace("\xa0", " ")
    text = re.sub(r"[ \t]+\n", "\n", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def normalize_for_key(value: str) -> str:
    return re.sub(r"\s+", " ", normalize_output_text(value)).strip()


def build_key(row: Dict[str, str]) -> str:
    normalized = "{}\n\n{}".format(
        normalize_for_key(row.get("Subject", "")),
        normalize_for_key(row.get("Body", "")),
    )
    return hashlib.sha1(normalized.encode("utf-8")).hexdigest()


def load_counter(paths: Sequence[Path]) -> Counter[str]:
    counter: Counter[str] = Counter()
    for path in paths:
        with path.open("r", encoding="utf-8", newline="") as handle:
            reader = csv.DictReader(handle)
            counter.update(build_key(row) for row in reader)
    return counter


def count_rows(path: Path) -> int:
    with path.open("r", encoding="utf-8", newline="") as handle:
        return sum(1 for _ in csv.DictReader(handle))


def rewrite_combined(
    combined_path: Path,
    removal_budget: Counter[str],
    *,
    dry_run: bool,
) -> Dict[str, int]:
    input_rows = 0
    removed_rows = 0
    kept_rows = 0

    if dry_run:
        with combined_path.open("r", encoding="utf-8", newline="") as handle:
            reader = csv.DictReader(handle)
            for row in reader:
                input_rows += 1
                row_key = build_key(row)
                if removal_budget[row_key] > 0:
                    removal_budget[row_key] -= 1
                    removed_rows += 1
                else:
                    kept_rows += 1
        return {
            "input_rows": input_rows,
            "removed_rows": removed_rows,
            "kept_rows": kept_rows,
        }

    with combined_path.open("r", encoding="utf-8", newline="") as input_handle, NamedTemporaryFile(
        "w",
        encoding="utf-8",
        newline="",
        delete=False,
        dir=str(combined_path.parent),
        prefix=combined_path.name + ".tmp.",
        suffix=".csv",
    ) as temp_handle:
        reader = csv.DictReader(input_handle)
        writer = csv.DictWriter(temp_handle, fieldnames=OUTPUT_COLUMNS)
        writer.writeheader()

        for row in reader:
            input_rows += 1
            row_key = build_key(row)
            if removal_budget[row_key] > 0:
                removal_budget[row_key] -= 1
                removed_rows += 1
                continue

            writer.writerow({column: row.get(column, "") for column in OUTPUT_COLUMNS})
            kept_rows += 1

    Path(temp_handle.name).replace(combined_path)
    return {
        "input_rows": input_rows,
        "removed_rows": removed_rows,
        "kept_rows": kept_rows,
    }


def main() -> int:
    args = parse_args()

    for label in ("P", "B"):
        combined_path = DEFAULT_COMBINED_PATHS[label]
        sample_paths = DEFAULT_SAMPLE_PATHS[label]
        sample_counter = load_counter(sample_paths)
        result = rewrite_combined(
            combined_path,
            removal_budget=sample_counter.copy(),
            dry_run=args.dry_run,
        )
        unmatched = sum(sample_counter.values()) - result["removed_rows"]
        unique_sample_keys = len(sample_counter)

        print(
            "[{}] {} rows={} sample_rows={} unique_sample_keys={} removed={} remaining={} unmatched_sample_rows={}".format(
                label,
                combined_path,
                result["input_rows"],
                sum(sample_counter.values()),
                unique_sample_keys,
                result["removed_rows"],
                result["kept_rows"],
                unmatched,
            )
        )

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
