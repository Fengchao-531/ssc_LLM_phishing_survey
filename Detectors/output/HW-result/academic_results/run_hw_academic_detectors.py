#!/usr/bin/env python3
"""Run the six Academic phishing detectors on HW Generic-Data stage datasets.

This wrapper mirrors ``Detectors/Academic/run_academic_detectors.py`` but adds:
- chunked execution (default 100 rows per chunk)
- resume from cached per-chunk detector outputs
- progress prints for each chunk and every 500 processed rows
- optional subject/body truncation before writing chunk CSVs
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import subprocess
import sys
from datetime import datetime
from pathlib import Path
from typing import Any, Iterable


csv.field_size_limit(sys.maxsize)


SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parents[3]
ACADEMIC_DIR = REPO_ROOT / "Detectors" / "Academic"
EMAIL_DETECTORS_DIR = ACADEMIC_DIR / "email_detectors"
DEFAULT_INPUT_DIR = SCRIPT_DIR.parent / "datasets"
DEFAULT_OUTPUT_DIR = SCRIPT_DIR
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
            "Run the six Academic phishing detectors over the HW Generic-Data stage datasets "
            "with chunked progress logging and resume support."
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
        default=[],
        help="Dataset filenames under the input directory. Defaults to all *-GD.csv files.",
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
    parser.add_argument("--chunk-size", type=int, default=100)
    parser.add_argument("--progress-interval", type=int, default=500)
    parser.add_argument(
        "--max-subject-chars",
        type=int,
        default=1000,
        help="Maximum characters kept in chunk CSVs for the subject. Use 0 to disable truncation.",
    )
    parser.add_argument(
        "--max-body-chars",
        type=int,
        default=12000,
        help="Maximum characters kept in chunk CSVs for the body. Use 0 to disable truncation.",
    )
    parser.add_argument(
        "--force-rerun",
        action="store_true",
        help="Ignore existing merged outputs and per-chunk cached outputs, then rerun everything.",
    )
    return parser.parse_args()


def timestamp() -> str:
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S")


def normalize_text(value: Any) -> str:
    return (value or "").replace("\r\n", "\n").replace("\r", "\n").strip()


def truncate_text(value: str, max_chars: int) -> str:
    text = normalize_text(value)
    if max_chars <= 0 or len(text) <= max_chars:
        return text
    return text[:max_chars].rstrip()


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


def merged_detector_complete(merged_rows: list[dict[str, str]], detector: str, expected_rows: int) -> bool:
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


def iter_chunk_ranges(total_rows: int, chunk_size: int) -> Iterable[tuple[int, int, int]]:
    chunk_index = 0
    for start in range(0, total_rows, chunk_size):
        end = min(start + chunk_size, total_rows)
        chunk_index += 1
        yield chunk_index, start, end


def ensure_manifest(
    manifest_path: Path,
    *,
    dataset_name: str,
    input_csv: Path,
    output_csv: Path,
    detectors: list[str],
    chunk_size: int,
) -> dict[str, Any]:
    payload = {
        "dataset_name": dataset_name,
        "input_csv": str(input_csv.resolve()),
        "output_csv": str(output_csv.resolve()),
        "detectors": detectors,
        "chunk_size": chunk_size,
        "updated_at": timestamp(),
    }
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    manifest_path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    return payload


def write_chunk_input_csv(
    chunk_input_csv: Path,
    rows: list[dict[str, str]],
    *,
    max_subject_chars: int,
    max_body_chars: int,
) -> None:
    chunk_input_csv.parent.mkdir(parents=True, exist_ok=True)
    with chunk_input_csv.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=["Subject", "Body", "label"])
        writer.writeheader()
        for row in rows:
            writer.writerow(
                {
                    "Subject": truncate_text(row["subject"], max_subject_chars),
                    "Body": truncate_text(row["body"], max_body_chars),
                    "label": row["label"],
                }
            )


def cuda_available(python_bin: str) -> bool:
    probe = [
        python_bin,
        "-c",
        (
            "import importlib.util\n"
            "import sys\n"
            "spec = importlib.util.find_spec('torch')\n"
            "if spec is None:\n"
            "    print('0')\n"
            "    raise SystemExit(0)\n"
            "import torch\n"
            "print('1' if torch.cuda.is_available() else '0')\n"
        ),
    ]
    try:
        completed = subprocess.run(
            probe,
            stdout=subprocess.PIPE,
            stderr=subprocess.DEVNULL,
            text=True,
            cwd=str(REPO_ROOT),
            check=False,
        )
    except Exception:
        return False
    return completed.stdout.strip() == "1"


def build_command(
    *,
    detector: str,
    python_bin: str,
    input_csv: Path,
    output_dir: Path,
    sample_size: int,
    has_cuda: bool,
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

    if detector in {"scamllm", "pimref", "t5phishing"}:
        command.extend(["--device", "0" if has_cuda else "-1"])
    elif detector == "securenet_llama":
        command.extend(["--device", "auto" if has_cuda else "cpu"])

    return command


def run_detector(
    *,
    detector: str,
    python_bin: str,
    input_csv: Path,
    output_dir: Path,
    sample_size: int,
    log_path: Path,
    has_cuda: bool,
) -> None:
    command = build_command(
        detector=detector,
        python_bin=python_bin,
        input_csv=input_csv,
        output_dir=output_dir,
        sample_size=sample_size,
        has_cuda=has_cuda,
    )
    completed = subprocess.run(
        command,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        universal_newlines=True,
        cwd=str(REPO_ROOT),
    )
    log_path.write_text(completed.stdout or "", encoding="utf-8")
    if completed.returncode != 0:
        raise RuntimeError(
            f"{detector} failed for {input_csv.name} with exit code {completed.returncode}. "
            f"See log: {log_path}"
        )


def chunk_output_path(chunk_dir: Path, chunk_input_csv: Path) -> Path:
    return chunk_dir / f"{chunk_input_csv.stem}_results.csv"


def process_detector_in_chunks(
    *,
    args: argparse.Namespace,
    dataset_name: str,
    detector: str,
    base_rows: list[dict[str, str]],
    merged_rows: list[dict[str, str]],
    output_csv: Path,
    detector_cache_dir: Path,
    detector_log_dir: Path,
    has_cuda: bool,
) -> None:
    total_rows = len(base_rows)
    total_chunks = (total_rows + args.chunk_size - 1) // args.chunk_size
    detector_predictions = [""] * total_rows
    processed_rows = 0
    next_progress_mark = args.progress_interval if args.progress_interval > 0 else total_rows

    print(
        f"[{timestamp()}] dataset={dataset_name} detector={detector} rows={total_rows} "
        f"chunks={total_chunks}",
        flush=True,
    )

    for chunk_index, start, end in iter_chunk_ranges(total_rows, args.chunk_size):
        chunk_rows = base_rows[start:end]
        chunk_name = f"chunk_{start + 1:06d}_{end:06d}"
        chunk_dir = detector_cache_dir / "chunks" / chunk_name
        chunk_input_csv = chunk_dir / f"{Path(dataset_name).stem}__{chunk_name}.csv"
        chunk_log_path = detector_log_dir / f"{detector}__{chunk_name}.log"
        detector_output_csv = chunk_output_path(chunk_dir, chunk_input_csv)

        predictions = None if args.force_rerun else load_detector_output(detector_output_csv, len(chunk_rows))
        source = "cached_detector_output"
        if predictions is None:
            source = "fresh_run"
            write_chunk_input_csv(
                chunk_input_csv,
                chunk_rows,
                max_subject_chars=args.max_subject_chars,
                max_body_chars=args.max_body_chars,
            )
            run_detector(
                detector=detector,
                python_bin=args.python_bin,
                input_csv=chunk_input_csv,
                output_dir=chunk_dir,
                sample_size=0,
                log_path=chunk_log_path,
                has_cuda=has_cuda,
            )
            predictions = load_detector_output(detector_output_csv, len(chunk_rows))

        if predictions is None:
            raise RuntimeError(
                f"{detector} did not produce a valid output CSV for {dataset_name} chunk {chunk_name}: "
                f"{detector_output_csv}"
            )

        detector_predictions[start:end] = predictions
        processed_rows = end
        print(
            f"[{timestamp()}] dataset={dataset_name} detector={detector} chunk={chunk_index}/{total_chunks} "
            f"rows={end - start} processed={processed_rows}/{total_rows} source={source}",
            flush=True,
        )
        if args.progress_interval > 0:
            while processed_rows >= next_progress_mark:
                print(
                    f"[{timestamp()}] progress dataset={dataset_name} detector={detector} "
                    f"processed={min(next_progress_mark, total_rows)}/{total_rows}",
                    flush=True,
                )
                next_progress_mark += args.progress_interval

    merge_detector_predictions(merged_rows, detector, detector_predictions)
    write_merged_output(output_csv, merged_rows, args.detectors)
    print(f"[{timestamp()}] merged detector={detector} -> {output_csv.name}", flush=True)


def process_dataset(args: argparse.Namespace, dataset_name: str, *, has_cuda: bool) -> None:
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
        chunk_size=args.chunk_size,
    )

    print(f"[{timestamp()}] dataset={dataset_name} rows={expected_rows}", flush=True)
    for detector in args.detectors:
        if not args.force_rerun and merged_detector_complete(merged_rows, detector, expected_rows):
            print(f"[{timestamp()}] skip detector={detector} source=merged_output", flush=True)
            continue

        detector_cache_dir = cache_root / detector
        detector_cache_dir.mkdir(parents=True, exist_ok=True)
        process_detector_in_chunks(
            args=args,
            dataset_name=dataset_name,
            detector=detector,
            base_rows=base_rows,
            merged_rows=merged_rows,
            output_csv=output_csv,
            detector_cache_dir=detector_cache_dir,
            detector_log_dir=log_dir,
            has_cuda=has_cuda,
        )

    write_merged_output(output_csv, merged_rows, args.detectors)
    print(f"[{timestamp()}] saved final={output_csv}", flush=True)


def resolve_default_datasets(input_dir: Path) -> list[str]:
    return sorted(path.name for path in input_dir.glob("*-GD.csv"))


def main() -> int:
    args = parse_args()
    args.input_dir = args.input_dir.resolve()
    args.output_dir = args.output_dir.resolve()
    args.log_dir = args.log_dir.resolve()
    args.cache_dir = args.cache_dir.resolve()

    if args.chunk_size <= 0:
        raise SystemExit("--chunk-size must be greater than 0.")
    if args.progress_interval < 0:
        raise SystemExit("--progress-interval cannot be negative.")

    if not args.datasets:
        args.datasets = resolve_default_datasets(args.input_dir)
    if not args.datasets:
        raise SystemExit(f"No *-GD.csv datasets found in {args.input_dir}")

    has_cuda = cuda_available(args.python_bin)
    print(f"[{timestamp()}] cuda_available={int(has_cuda)}", flush=True)
    for dataset_name in args.datasets:
        process_dataset(args, dataset_name, has_cuda=has_cuda)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
