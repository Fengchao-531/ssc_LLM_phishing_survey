#!/usr/bin/env python3
"""Build Generic-Data detector inputs that mirror LLM-Ind stage label counts."""

from __future__ import annotations

import argparse
import csv
import random
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Sequence


csv.field_size_limit(sys.maxsize)

SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parents[4]
LLM_RESULT_DIR = REPO_ROOT / "Detectors" / "Industry" / "email_detectors" / "output" / "LLM-Ind"
GENERIC_DATA_DIR = REPO_ROOT / "Datasets" / "Generic-Data"
DEFAULT_OUTPUT_DIR = SCRIPT_DIR / "datasets"
DEFAULT_COUNTS_SUMMARY = SCRIPT_DIR / "llm_stage_label_counts.csv"
DEFAULT_SAMPLING_SUMMARY = DEFAULT_OUTPUT_DIR / "generic_dataset_sampling_summary.csv"

STAGE_FILES: Sequence[str] = (
    "S1.csv",
    "S2.csv",
    "S4.csv",
    "S5.csv",
    "S6-fuzzer.csv",
    "S6-MPG.csv",
    "S6-UTA.csv",
    "S8-deepseek.csv",
    "S8-llama.csv",
    "S8-ministral.csv",
)

LABELED_STAGE_FILES: Sequence[str] = STAGE_FILES
OUTPUT_COLUMNS = ["Subject", "Body", "label", "data_source"]


@dataclass(frozen=True)
class StageCount:
    file_name: str
    stage_name: str
    total_rows: int
    label_0_rows: int
    label_1_rows: int
    other_label_rows: int
    has_label_column: bool
    note: str


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Sample Generic-Data combined pools to match each LLM-Ind stage's label counts."
    )
    parser.add_argument(
        "--stage",
        action="append",
        default=[],
        help="Specific stage stem to build, e.g. S1 or S6-MPG. May be provided multiple times.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=20260407,
        help="Base random seed for deterministic sampling.",
    )
    parser.add_argument(
        "--output-dir",
        default=str(DEFAULT_OUTPUT_DIR),
        help="Directory for generated stage datasets.",
    )
    parser.add_argument(
        "--counts-summary",
        default=str(DEFAULT_COUNTS_SUMMARY),
        help="Where to write the LLM-Ind label-count summary CSV.",
    )
    parser.add_argument(
        "--sampling-summary",
        default=str(DEFAULT_SAMPLING_SUMMARY),
        help="Where to write the Generic-Data sampling summary CSV.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite existing stage output CSVs.",
    )
    return parser.parse_args()


def stage_name_from_file(file_name: str) -> str:
    return Path(file_name).stem


def normalize_requested_stages(requested: Sequence[str]) -> List[str]:
    if not requested:
        return [stage_name_from_file(name) for name in LABELED_STAGE_FILES]

    valid = {stage_name_from_file(name) for name in LABELED_STAGE_FILES}
    normalized: List[str] = []
    for stage in requested:
        cleaned = stage.strip()
        if not cleaned:
            continue
        if cleaned not in valid:
            raise ValueError(
                "Unknown --stage '{}'. Valid options: {}".format(
                    cleaned, ", ".join(sorted(valid))
                )
            )
        normalized.append(cleaned)
    return normalized


def read_csv_rows(path: Path) -> List[Dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def count_stage_labels(path: Path) -> StageCount:
    rows = read_csv_rows(path)
    if not rows:
        return StageCount(
            file_name=path.name,
            stage_name=path.stem,
            total_rows=0,
            label_0_rows=0,
            label_1_rows=0,
            other_label_rows=0,
            has_label_column=False,
            note="empty_file",
        )

    if "label" not in rows[0]:
        return StageCount(
            file_name=path.name,
            stage_name=path.stem,
            total_rows=len(rows),
            label_0_rows=0,
            label_1_rows=0,
            other_label_rows=0,
            has_label_column=False,
            note="missing_label_column",
        )

    label_0_rows = 0
    label_1_rows = 0
    other_label_rows = 0
    for row in rows:
        label = str(row.get("label", "")).strip()
        if label == "0":
            label_0_rows += 1
        elif label == "1":
            label_1_rows += 1
        else:
            other_label_rows += 1

    note = "" if other_label_rows == 0 else "contains_non_binary_labels"
    return StageCount(
        file_name=path.name,
        stage_name=path.stem,
        total_rows=len(rows),
        label_0_rows=label_0_rows,
        label_1_rows=label_1_rows,
        other_label_rows=other_label_rows,
        has_label_column=True,
        note=note,
    )


def write_counts_summary(stage_counts: Sequence[StageCount], output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=[
                "file_name",
                "stage_name",
                "total_rows",
                "label_0_rows",
                "label_1_rows",
                "other_label_rows",
                "has_label_column",
                "note",
            ],
        )
        writer.writeheader()
        for item in stage_counts:
            writer.writerow(
                {
                    "file_name": item.file_name,
                    "stage_name": item.stage_name,
                    "total_rows": item.total_rows,
                    "label_0_rows": item.label_0_rows,
                    "label_1_rows": item.label_1_rows,
                    "other_label_rows": item.other_label_rows,
                    "has_label_column": "1" if item.has_label_column else "0",
                    "note": item.note,
                }
            )


def sample_rows(
    pool: Sequence[Dict[str, str]],
    count: int,
    *,
    seed: int,
    stage_name: str,
    label: str,
) -> tuple[List[Dict[str, str]], str]:
    rng = random.Random("{}:{}:{}".format(seed, stage_name, label))
    if count <= len(pool):
        indices = list(range(len(pool)))
        rng.shuffle(indices)
        return ([dict(pool[index]) for index in indices[:count]], "without_replacement")

    sampled = [dict(pool[rng.randrange(len(pool))]) for _ in range(count)]
    return (sampled, "with_replacement")


def shuffle_stage_rows(rows: List[Dict[str, str]], *, seed: int, stage_name: str) -> None:
    random.Random("{}:{}:final".format(seed, stage_name)).shuffle(rows)


def write_stage_dataset(rows: Sequence[Dict[str, str]], output_path: Path, *, overwrite: bool) -> None:
    if output_path.exists() and not overwrite:
        raise FileExistsError(
            "{} already exists. Re-run with --overwrite to replace it.".format(output_path)
        )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=OUTPUT_COLUMNS)
        writer.writeheader()
        for row in rows:
            writer.writerow({column: row.get(column, "") for column in OUTPUT_COLUMNS})


def write_sampling_summary(rows: Sequence[Dict[str, str]], output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=[
                "stage_name",
                "source_file",
                "output_file",
                "total_rows",
                "label_0_rows",
                "label_1_rows",
                "generic_label_0_available",
                "generic_label_1_available",
                "label_0_sampling_mode",
                "label_1_sampling_mode",
            ],
        )
        writer.writeheader()
        writer.writerows(rows)


def main() -> int:
    args = parse_args()
    requested_stages = set(normalize_requested_stages(args.stage))
    output_dir = Path(args.output_dir).resolve()
    counts_summary_path = Path(args.counts_summary).resolve()
    sampling_summary_path = Path(args.sampling_summary).resolve()

    stage_counts = [
        count_stage_labels(LLM_RESULT_DIR / file_name)
        for file_name in STAGE_FILES
    ]
    write_counts_summary(stage_counts, counts_summary_path)

    labeled_counts = {
        item.stage_name: item
        for item in stage_counts
        if item.has_label_column and item.stage_name in requested_stages
    }

    positive_pool = read_csv_rows(GENERIC_DATA_DIR / "combined-P.csv")
    benign_pool = read_csv_rows(GENERIC_DATA_DIR / "combined-B.csv")

    sampling_summary_rows: List[Dict[str, str]] = []
    for file_name in LABELED_STAGE_FILES:
        stage_name = stage_name_from_file(file_name)
        if stage_name not in requested_stages:
            continue

        counts = labeled_counts[stage_name]
        sampled_positives, positive_mode = sample_rows(
            positive_pool,
            counts.label_1_rows,
            seed=args.seed,
            stage_name=stage_name,
            label="1",
        )
        sampled_benign, benign_mode = sample_rows(
            benign_pool,
            counts.label_0_rows,
            seed=args.seed,
            stage_name=stage_name,
            label="0",
        )

        stage_rows = sampled_positives + sampled_benign
        shuffle_stage_rows(stage_rows, seed=args.seed, stage_name=stage_name)

        output_file = output_dir / "{}-GD.csv".format(stage_name)
        write_stage_dataset(stage_rows, output_file, overwrite=args.overwrite)

        sampling_summary_rows.append(
            {
                "stage_name": stage_name,
                "source_file": file_name,
                "output_file": output_file.name,
                "total_rows": str(len(stage_rows)),
                "label_0_rows": str(counts.label_0_rows),
                "label_1_rows": str(counts.label_1_rows),
                "generic_label_0_available": str(len(benign_pool)),
                "generic_label_1_available": str(len(positive_pool)),
                "label_0_sampling_mode": benign_mode,
                "label_1_sampling_mode": positive_mode,
            }
        )

        print(
            "[done] {} -> {} rows={} label0={} label1={} modes=(0:{},1:{})".format(
                stage_name,
                output_file,
                len(stage_rows),
                counts.label_0_rows,
                counts.label_1_rows,
                benign_mode,
                positive_mode,
            )
        )

    write_sampling_summary(sampling_summary_rows, sampling_summary_path)
    print("[summary] wrote {}".format(counts_summary_path))
    print("[summary] wrote {}".format(sampling_summary_path))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
