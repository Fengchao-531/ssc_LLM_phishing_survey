#!/usr/bin/env python3
"""Build mixed LLM benchmark CSVs under Detectors/output/LLM-result."""

from __future__ import annotations

import csv
import json
import random
from pathlib import Path


SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_DIR = SCRIPT_DIR.parent
OUTPUT_DIR = SCRIPT_DIR / "output" / "LLM-result"
SEED = 42

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
}


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


def main() -> int:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    summaries = []

    for stage_name, sources in STAGE_SOURCES.items():
        benign_rows, benign_fields = read_csv(sources["benign"])
        phishing_rows, phishing_fields = read_csv(sources["phishing"])
        if benign_fields != phishing_fields:
            raise SystemExit(
                f"Field mismatch for {stage_name}: {sources['benign']} vs {sources['phishing']}"
            )

        mixed_rows = list(benign_rows) + list(phishing_rows)
        random.Random(SEED).shuffle(mixed_rows)
        output_csv = OUTPUT_DIR / f"{stage_name}.csv"
        write_csv(output_csv, mixed_rows, benign_fields)

        summaries.append({
            "stage": stage_name,
            "output_csv": str(output_csv),
            "rows_total": len(mixed_rows),
            "rows_benign": len(benign_rows),
            "rows_phishing": len(phishing_rows),
            "seed": SEED,
        })

    print(json.dumps({"generated": summaries}, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
