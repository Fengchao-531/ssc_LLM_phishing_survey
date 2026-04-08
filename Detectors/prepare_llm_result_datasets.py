#!/usr/bin/env python3
"""Build mixed LLM benchmark CSVs under Detectors/output/LLM-result."""

from __future__ import annotations

import argparse
import csv
import json
import random
from pathlib import Path


SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_DIR = SCRIPT_DIR.parent
OUTPUT_DIR = SCRIPT_DIR / "output" / "LLM-result"
SEED = 42

csv.field_size_limit(10**9)

STAGE_SOURCES = {
    "S1": {
        "benign": PROJECT_DIR / "Datasets" / "sublist" / "S1-Basic Instruction" / "LLM-B.csv",
        "phishing": PROJECT_DIR / "Datasets" / "sublist" / "S1-Basic Instruction" / "LLM-P.csv",
    },
    "S2": {
        "benign": PROJECT_DIR / "Datasets" / "sublist" / "S2-Role-Framed Prompting" / "LLM-B.csv",
        "phishing": PROJECT_DIR / "Datasets" / "sublist" / "S2-Role-Framed Prompting" / "LLM-P.csv",
    },
    "S4": {
        "benign": PROJECT_DIR / "Datasets" / "sublist" / "S4-Scenarios-driven Adaptation" / "LLM-B.csv",
        "phishing": PROJECT_DIR / "Datasets" / "sublist" / "S4-Scenarios-driven Adaptation" / "LLM-P.csv",
    },
    "S5": {
        "benign": PROJECT_DIR / "Datasets" / "sublist" / "S5-Personalization for Credibility" / "LLM-B.csv",
        "phishing": PROJECT_DIR / "Datasets" / "sublist" / "S5-Personalization for Credibility" / "LLM-P.csv",
    },
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Mix benign/phishing CSV pairs into detector-ready benchmark CSVs. "
            "If no --dataset is provided, the built-in S1/S2/S4 presets are used."
        )
    )
    parser.add_argument(
        "--dataset",
        action="append",
        nargs=3,
        metavar=("NAME", "BENIGN_CSV", "PHISHING_CSV"),
        help=(
            "Custom dataset triple: output name stem, benign CSV path, phishing CSV path. "
            "Example: --dataset S6-UTA /path/UTA-LLM-B.csv /path/UTA-LLM-P.csv"
        ),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=OUTPUT_DIR,
        help="Directory to write mixed CSVs into. Default: Detectors/output/LLM-result",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=SEED,
        help="Shuffle seed used independently for each mixed dataset. Default: 42",
    )
    parser.add_argument(
        "--keep-columns",
        nargs="+",
        default=None,
        help=(
            "Optional ordered subset of columns to keep in the mixed output. "
            "Example: --keep-columns Subject Body label data_source"
        ),
    )
    return parser.parse_args()


def read_csv(path: Path) -> tuple[list[dict[str, str]], list[str]]:
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        if not reader.fieldnames:
            raise SystemExit(f"CSV has no header row: {path}")
        return list(reader), list(reader.fieldnames)


def write_csv(path: Path, rows: list[dict[str, str]], fieldnames: list[str]) -> None:
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def resolve_dataset_specs(args: argparse.Namespace) -> list[tuple[str, Path, Path]]:
    if args.dataset:
        specs = []
        for name, benign_csv, phishing_csv in args.dataset:
            specs.append((name, Path(benign_csv).expanduser().resolve(), Path(phishing_csv).expanduser().resolve()))
        return specs

    specs = []
    for stage_name, sources in STAGE_SOURCES.items():
        specs.append((stage_name, sources["benign"], sources["phishing"]))
    return specs


def resolve_output_fieldnames(
    dataset_name: str,
    benign_fields: list[str],
    phishing_fields: list[str],
    keep_columns: list[str] | None,
) -> list[str]:
    if benign_fields != phishing_fields:
        raise SystemExit(
            f"Field mismatch for {dataset_name}: benign columns {benign_fields} vs phishing columns {phishing_fields}"
        )
    if not keep_columns:
        return list(benign_fields)

    missing = [column for column in keep_columns if column not in benign_fields]
    if missing:
        raise SystemExit(
            f"Requested keep-columns missing for {dataset_name}: {', '.join(missing)}"
        )
    return list(keep_columns)


def project_rows(rows: list[dict[str, str]], fieldnames: list[str]) -> list[dict[str, str]]:
    return [{name: row.get(name, "") for name in fieldnames} for row in rows]


def main() -> int:
    args = parse_args()
    output_dir = args.output_dir.expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    summaries = []

    for dataset_name, benign_csv, phishing_csv in resolve_dataset_specs(args):
        benign_rows, benign_fields = read_csv(benign_csv)
        phishing_rows, phishing_fields = read_csv(phishing_csv)
        output_fieldnames = resolve_output_fieldnames(
            dataset_name,
            benign_fields,
            phishing_fields,
            args.keep_columns,
        )

        mixed_rows = project_rows(list(benign_rows) + list(phishing_rows), output_fieldnames)
        random.Random(args.seed).shuffle(mixed_rows)
        output_csv = output_dir / f"{dataset_name}.csv"
        write_csv(output_csv, mixed_rows, output_fieldnames)

        summaries.append({
            "dataset_name": dataset_name,
            "benign_csv": str(benign_csv),
            "phishing_csv": str(phishing_csv),
            "output_csv": str(output_csv),
            "output_columns": output_fieldnames,
            "rows_total": len(mixed_rows),
            "rows_benign": len(benign_rows),
            "rows_phishing": len(phishing_rows),
            "seed": args.seed,
        })

    print(json.dumps({"generated": summaries}, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
