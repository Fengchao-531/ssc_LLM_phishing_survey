#!/usr/bin/env python3
"""Run the Academic phishing detectors across staged CSV datasets with resume support."""

from __future__ import annotations

import argparse
import csv
import json
import os
import subprocess
import sys
from datetime import datetime
from pathlib import Path
from typing import Any

csv.field_size_limit(10**9)


SCRIPT_DIR = Path(__file__).resolve().parent
EMAIL_DETECTORS_DIR = SCRIPT_DIR / "email_detectors"
DEFAULT_INPUT_DIR = SCRIPT_DIR.parent / "Industry" / "email_detectors" / "output" / "LLM-Ind"
DEFAULT_OUTPUT_DIR = SCRIPT_DIR / "email_detectors" / "LLM-Acad"
DEFAULT_LOG_DIR = DEFAULT_OUTPUT_DIR / "_logs"
DEFAULT_CACHE_DIR = DEFAULT_OUTPUT_DIR / "_detector_runs"
DEFAULT_DETECTORS = [
    "scamllm",
    "pimref",
    "t5phishing",
    "ml_watermark_logreg",
    "xgboost",
    "securenet_llama",
]
INPUT_DATASET_NAMES = [
    "S1.csv",
    "S2.csv",
    "S4.csv",
    "S5.csv",
    "S6-MPG.csv",
    "S6-UTA.csv",
    "S6-fuzzer.csv",
    "S8-deepseek.csv",
    "S8-llama.csv",
    "S8-ministral.csv",
]


def detect_default_python() -> str:
    env_python = os.environ.get("PYTHON_BIN", "").strip()
    if env_python:
        return env_python
    for env_name in ("VIRTUAL_ENV", "CONDA_PREFIX"):
        env_prefix = os.environ.get(env_name, "").strip()
        if not env_prefix:
            continue
        candidate = Path(env_prefix) / "bin" / "python"
        if candidate.exists():
            return str(candidate)
    return sys.executable


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Run the six Academic phishing detectors over the staged LLM-Ind CSV datasets, "
            "merge their predictions into one CSV per dataset, and resume automatically."
        )
    )
    parser.add_argument("--input-dir", type=Path, default=DEFAULT_INPUT_DIR)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--log-dir", type=Path, default=DEFAULT_LOG_DIR)
    parser.add_argument("--cache-dir", type=Path, default=DEFAULT_CACHE_DIR)
    parser.add_argument("--python-bin", default=detect_default_python())
    parser.add_argument(
        "--datasets",
        nargs="*",
        default=list(INPUT_DATASET_NAMES),
        help="Dataset filenames under the input directory. Defaults to the staged S1/S2/S4/S5/S6*/S8* CSVs.",
    )
    parser.add_argument(
        "--detectors",
        nargs="+",
        choices=list(DEFAULT_DETECTORS),
        default=list(DEFAULT_DETECTORS),
    )
    parser.add_argument("--subject-column", default="Subject")
    parser.add_argument("--body-column", default="Body")
    parser.add_argument("--label-column", default="label")
    parser.add_argument("--sample-size", type=int, default=0, help="Use 0 for all rows.")
    parser.add_argument(
        "--force-rerun",
        action="store_true",
        help="Ignore existing merged outputs and per-detector cached outputs, then rerun everything.",
    )
    return parser.parse_args()


def normalize_text(value: Any) -> str:
    return (value or "").replace("\r\n", "\n").replace("\r", "\n").strip()


def normalize_prediction(value: Any) -> str:
    text = str(value).strip()
    if text == "":
        return ""
    lowered = text.lower()
    if lowered in {"1", "true", "phishing", "malicious", "suspicious", "yes"}:
        return "1"
    if lowered in {"0", "false", "legitimate", "benign", "safe", "clean", "no"}:
        return "0"
    return text


def timestamp() -> str:
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S")


def read_input_rows(
    input_csv: Path,
    *,
    subject_column: str,
    body_column: str,
    label_column: str,
    sample_size: int,
) -> list[dict[str, str]]:
    if not input_csv.exists():
        raise SystemExit(f"Input CSV not found: {input_csv}")

    rows: list[dict[str, str]] = []
    with input_csv.open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        if not reader.fieldnames:
            raise SystemExit(f"CSV has no header row: {input_csv}")
        required = [subject_column, body_column, label_column]
        missing = [name for name in required if name not in reader.fieldnames]
        if missing:
            raise SystemExit(f"CSV missing required columns {', '.join(missing)} in {input_csv}")

        for index, row in enumerate(reader, start=1):
            if sample_size > 0 and index > sample_size:
                break
            rows.append(
                {
                    "subject": normalize_text(row.get(subject_column, "")),
                    "body": normalize_text(row.get(body_column, "")),
                    "label": str(row.get(label_column, "")),
                }
            )
    if not rows:
        raise SystemExit(f"No rows loaded from {input_csv}")
    return rows


def load_existing_merged_predictions(output_csv: Path, detectors: list[str]) -> dict[int, dict[str, str]]:
    if not output_csv.exists():
        return {}
    with output_csv.open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        if not reader.fieldnames:
            return {}
        rows: dict[int, dict[str, str]] = {}
        for index, row in enumerate(reader, start=1):
            rows[index] = {detector: str(row.get(detector, "")) for detector in detectors}
        return rows


def merged_detector_complete(
    merged_rows: list[dict[str, str]],
    detector: str,
    expected_rows: int,
) -> bool:
    if len(merged_rows) != expected_rows:
        return False
    return all(str(row.get(detector, "")).strip() != "" for row in merged_rows)


def load_detector_output(output_csv: Path, expected_rows: int) -> list[str] | None:
    if not output_csv.exists():
        return None
    with output_csv.open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        if not reader.fieldnames or "model_prediction" not in reader.fieldnames:
            return None
        predictions = [normalize_prediction(row.get("model_prediction", "")) for row in reader]
    if len(predictions) != expected_rows:
        return None
    return predictions


def build_command(
    *,
    detector: str,
    python_bin: str,
    input_csv: Path,
    output_dir: Path,
    sample_size: int,
) -> list[str]:
    command = [
        python_bin,
        str((EMAIL_DETECTORS_DIR / f"{detector}.py").resolve()),
        "--input-csv",
        str(input_csv.resolve()),
        "--output-dir",
        str(output_dir.resolve()),
    ]
    if sample_size > 0:
        command.extend(["--sample-size", str(sample_size)])
    return command


def run_detector(
    *,
    detector: str,
    python_bin: str,
    input_csv: Path,
    output_dir: Path,
    sample_size: int,
    log_path: Path,
) -> None:
    command = build_command(
        detector=detector,
        python_bin=python_bin,
        input_csv=input_csv,
        output_dir=output_dir,
        sample_size=sample_size,
    )
    completed = subprocess.run(
        command,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        universal_newlines=True,
        cwd=str(SCRIPT_DIR.parent.parent),
    )
    log_path.write_text(completed.stdout or "", encoding="utf-8")
    if completed.returncode != 0:
        raise RuntimeError(
            f"{detector} failed for {input_csv.name} with exit code {completed.returncode}. "
            f"See log: {log_path}"
        )


def write_merged_output(output_csv: Path, rows: list[dict[str, str]], detectors: list[str]) -> None:
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = ["subject", "body", "label", *detectors]
    with output_csv.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def merge_detector_predictions(rows: list[dict[str, str]], detector: str, predictions: list[str]) -> None:
    for row, prediction in zip(rows, predictions):
        row[detector] = prediction


def ensure_manifest(
    manifest_path: Path,
    *,
    dataset_name: str,
    input_csv: Path,
    output_csv: Path,
    detectors: list[str],
) -> dict[str, Any]:
    payload = {
        "dataset_name": dataset_name,
        "input_csv": str(input_csv.resolve()),
        "output_csv": str(output_csv.resolve()),
        "detectors": detectors,
        "updated_at": timestamp(),
    }
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    manifest_path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    return payload


def process_dataset(args: argparse.Namespace, dataset_name: str) -> None:
    input_csv = (args.input_dir / dataset_name).resolve()
    output_csv = (args.output_dir / dataset_name).resolve()
    log_dir = (args.log_dir / input_csv.stem).resolve()
    cache_root = (args.cache_dir / input_csv.stem).resolve()
    log_dir.mkdir(parents=True, exist_ok=True)
    cache_root.mkdir(parents=True, exist_ok=True)

    base_rows = read_input_rows(
        input_csv,
        subject_column=args.subject_column,
        body_column=args.body_column,
        label_column=args.label_column,
        sample_size=args.sample_size,
    )
    expected_rows = len(base_rows)

    existing_predictions = {}
    if not args.force_rerun:
        existing_predictions = load_existing_merged_predictions(output_csv, args.detectors)
    merged_rows: list[dict[str, str]] = []
    for index, row in enumerate(base_rows, start=1):
        merged = dict(row)
        for detector in args.detectors:
            merged[detector] = normalize_prediction(existing_predictions.get(index, {}).get(detector, ""))
        merged_rows.append(merged)

    manifest_path = cache_root / "run_manifest.json"
    ensure_manifest(
        manifest_path,
        dataset_name=dataset_name,
        input_csv=input_csv,
        output_csv=output_csv,
        detectors=args.detectors,
    )

    print(f"[{timestamp()}] dataset={dataset_name} rows={expected_rows}", flush=True)
    for detector in args.detectors:
        if not args.force_rerun and merged_detector_complete(merged_rows, detector, expected_rows):
            print(f"[{timestamp()}] skip detector={detector} source=merged_output", flush=True)
            continue

        detector_output_dir = cache_root / detector
        detector_output_dir.mkdir(parents=True, exist_ok=True)
        detector_output_csv = detector_output_dir / f"{input_csv.stem}_results.csv"
        log_path = log_dir / f"{detector}.log"

        predictions = None
        if not args.force_rerun:
            predictions = load_detector_output(detector_output_csv, expected_rows)
        if predictions is None:
            print(f"[{timestamp()}] run detector={detector}", flush=True)
            run_detector(
                detector=detector,
                python_bin=args.python_bin,
                input_csv=input_csv,
                output_dir=detector_output_dir,
                sample_size=args.sample_size,
                log_path=log_path,
            )
            predictions = load_detector_output(detector_output_csv, expected_rows)
        else:
            print(f"[{timestamp()}] skip detector={detector} source=cached_detector_output", flush=True)

        if predictions is None:
            raise RuntimeError(
                f"{detector} did not produce a valid output CSV for {dataset_name}: {detector_output_csv}"
            )
        merge_detector_predictions(merged_rows, detector, predictions)
        write_merged_output(output_csv, merged_rows, args.detectors)
        print(f"[{timestamp()}] merged detector={detector} -> {output_csv.name}", flush=True)

    write_merged_output(output_csv, merged_rows, args.detectors)
    print(f"[{timestamp()}] saved final={output_csv}", flush=True)


def main() -> int:
    args = parse_args()
    args.input_dir = args.input_dir.resolve()
    args.output_dir = args.output_dir.resolve()
    args.log_dir = args.log_dir.resolve()
    args.cache_dir = args.cache_dir.resolve()

    for dataset_name in args.datasets:
        process_dataset(args, dataset_name)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
