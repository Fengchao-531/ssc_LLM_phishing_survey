#!/usr/bin/env python3
"""Run the five Industry text detectors on HW Generic-Data stage datasets.

This wrapper delegates detector execution to ``Detectors/output/run_text_detectors.py``
so we keep the same normalized merge logic as the existing Industry benchmark.

Extra behavior added here:
- iterate over every ``*-GD.csv`` dataset under ``HW-result/datasets``
- truncate subject/body in a temporary detector-only copy to reduce CSV/input-length issues
- preserve the original full Subject/Body columns in the final saved outputs
- keep chunk/progress behavior via the delegated runner
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import shutil
import subprocess
import sys
import tempfile
from datetime import datetime
from pathlib import Path
from typing import Any


csv.field_size_limit(sys.maxsize)


SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parents[3]
OUTPUT_DIR = REPO_ROOT / "Detectors" / "output"
HW_RESULT_DIR = OUTPUT_DIR / "HW-result"
DEFAULT_INPUT_DIR = HW_RESULT_DIR / "datasets"
DEFAULT_OUTPUT_DIR = SCRIPT_DIR
DEFAULT_DETECTORS = [
    "llm_guard",
    "phishing_email_agent",
    "email_phishing_detection_v3",
    "pyrit_original",
    "pyrit_blocklist",
]
RUN_TEXT_DETECTORS = OUTPUT_DIR / "run_text_detectors.py"


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
            "Run the five Industry text detectors over the HW Generic-Data stage datasets "
            "and save merged outputs under output/HW-result/industry_results."
        )
    )
    parser.add_argument("--input-dir", type=Path, default=DEFAULT_INPUT_DIR)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument(
        "--tmp-dir",
        type=Path,
        default=None,
        help="Optional parent directory for temporary prepared inputs. Defaults to the system temp directory.",
    )
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
    parser.add_argument("--data-source-column", default="data_source")
    parser.add_argument("--sample-size", type=int, default=0)
    parser.add_argument("--chunk-size", type=int, default=100)
    parser.add_argument("--progress-interval", type=int, default=500)
    parser.add_argument("--max-subject-chars", type=int, default=1000)
    parser.add_argument("--max-body-chars", type=int, default=12000)
    parser.add_argument("--from-address", default="sender@example.com")
    parser.add_argument("--to-address", default="recipient@example.com")
    parser.add_argument("--backend-root", default="http://127.0.0.1:5000")
    parser.add_argument("--openrouter-api-key", default=os.environ.get("OPENROUTER_API_KEY", ""))
    parser.add_argument("--resume-existing", action="store_true")
    parser.add_argument("--start-row", type=int, default=1)
    parser.add_argument("--fail-fast", action="store_true")
    parser.add_argument("--keep-prepared-inputs", action="store_true")
    return parser.parse_args()


def timestamp() -> str:
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S")


def normalize_text(value: Any) -> str:
    return (value or "").replace("\r\n", "\n").replace("\r", "\n").strip()


def truncate_text(value: Any, max_chars: int) -> str:
    text = normalize_text(value)
    if max_chars <= 0 or len(text) <= max_chars:
        return text
    return text[:max_chars].rstrip()


def resolve_default_datasets(input_dir: Path) -> list[str]:
    return sorted(path.name for path in input_dir.glob("*-GD.csv"))


def read_original_rows(path: Path) -> tuple[list[dict[str, str]], list[str]]:
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        if not reader.fieldnames:
            raise SystemExit(f"CSV has no header row: {path}")
        return list(reader), list(reader.fieldnames)


def write_prepared_csv(
    output_csv: Path,
    fieldnames: list[str],
    rows: list[dict[str, str]],
    *,
    subject_column: str,
    body_column: str,
    max_subject_chars: int,
    max_body_chars: int,
) -> None:
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    with output_csv.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            prepared = dict(row)
            prepared[subject_column] = truncate_text(prepared.get(subject_column, ""), max_subject_chars)
            prepared[body_column] = truncate_text(prepared.get(body_column, ""), max_body_chars)
            writer.writerow(prepared)


def run_subprocess(command: list[str], *, cwd: Path) -> None:
    completed = subprocess.run(
        command,
        cwd=str(cwd),
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        universal_newlines=True,
    )
    if completed.stdout:
        print(completed.stdout, end="" if completed.stdout.endswith("\n") else "\n", flush=True)
    if completed.returncode != 0:
        raise RuntimeError(
            f"Command failed with exit code {completed.returncode}: {' '.join(command)}"
        )


def read_combined_output(path: Path) -> tuple[list[dict[str, str]], list[str]]:
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        if not reader.fieldnames:
            raise RuntimeError(f"Combined detector output has no header row: {path}")
        return list(reader), list(reader.fieldnames)


def rewrite_output_with_original_text(
    final_output_csv: Path,
    original_rows: list[dict[str, str]],
    combined_rows: list[dict[str, str]],
    output_fieldnames: list[str],
    *,
    subject_column: str,
    body_column: str,
    label_column: str,
    data_source_column: str,
    dataset_name: str,
    source_path: Path,
) -> None:
    if len(original_rows) != len(combined_rows):
        raise RuntimeError(
            f"Row count mismatch while restoring original text for {final_output_csv.name}: "
            f"original={len(original_rows)} combined={len(combined_rows)}"
        )

    with final_output_csv.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=output_fieldnames)
        writer.writeheader()
        for index, (original_row, combined_row) in enumerate(zip(original_rows, combined_rows), start=1):
            merged = dict(combined_row)
            merged["benchmark_row_number"] = index
            merged["dataset_name"] = dataset_name
            merged["source_path"] = str(source_path.resolve())
            merged["source_row_number"] = index
            merged[subject_column] = original_row.get(subject_column, "")
            merged[body_column] = original_row.get(body_column, "")
            if label_column in output_fieldnames:
                merged[label_column] = original_row.get(label_column, "")
            if data_source_column in output_fieldnames:
                merged[data_source_column] = original_row.get(data_source_column, "")
            writer.writerow(merged)


def write_manifest(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, ensure_ascii=False)


def process_dataset(args: argparse.Namespace, dataset_name: str) -> None:
    input_csv = (args.input_dir / dataset_name).resolve()
    if not input_csv.exists():
        raise SystemExit(f"Input dataset not found: {input_csv}")

    original_rows, original_fieldnames = read_original_rows(input_csv)
    if args.sample_size > 0:
        original_rows = original_rows[: args.sample_size]
    dataset_stem = input_csv.stem
    output_csv = (args.output_dir / f"{dataset_stem}_results.csv").resolve()
    manifest_path = (args.output_dir / f"{dataset_stem}_run_manifest.json").resolve()

    print(
        f"[{timestamp()}] prepare dataset={dataset_name} rows={len(original_rows)} "
        f"output={output_csv.name}",
        flush=True,
    )

    temp_dir_kwargs: dict[str, str] = {"prefix": f"{dataset_stem}_"}
    prepared_parent: Path | None = None
    if args.tmp_dir is not None:
        prepared_parent = (args.tmp_dir / dataset_stem).resolve()
        prepared_parent.mkdir(parents=True, exist_ok=True)
        temp_dir_kwargs["dir"] = str(prepared_parent)

    with tempfile.TemporaryDirectory(**temp_dir_kwargs) as temp_dir:
        temp_dir_path = Path(temp_dir).resolve()
        prepared_input_csv = temp_dir_path / dataset_name
        write_prepared_csv(
            prepared_input_csv,
            original_fieldnames,
            original_rows,
            subject_column=args.subject_column,
            body_column=args.body_column,
            max_subject_chars=args.max_subject_chars,
            max_body_chars=args.max_body_chars,
        )

        command = [
            args.python_bin,
            str(RUN_TEXT_DETECTORS.resolve()),
            "--input-csv",
            str(prepared_input_csv),
            "--subject-column",
            args.subject_column,
            "--body-column",
            args.body_column,
            "--label-column",
            args.label_column,
            "--data-source-column",
            args.data_source_column,
            "--detectors",
            *args.detectors,
            "--checkpoint-every",
            str(args.chunk_size),
            "--progress-every",
            str(args.progress_interval),
            "--result-group",
            "HW-result",
            "--stage-name",
            dataset_stem.upper(),
            "--output-csv",
            str(output_csv),
            "--python-bin",
            args.python_bin,
            "--from-address",
            args.from_address,
            "--to-address",
            args.to_address,
            "--backend-root",
            args.backend_root,
            "--start-row",
            str(args.start_row),
        ]

        if args.sample_size > 0:
            command.extend(["--sample-size", str(args.sample_size)])
        if args.openrouter_api_key:
            command.extend(["--openrouter-api-key", args.openrouter_api_key])
        if args.resume_existing:
            command.append("--resume-existing")
        if args.fail_fast:
            command.append("--fail-fast")

        print(
            f"[{timestamp()}] run dataset={dataset_name} detectors={','.join(args.detectors)} "
            f"chunk_size={args.chunk_size} progress_every={args.progress_interval}",
            flush=True,
        )
        run_subprocess(command, cwd=REPO_ROOT)

        combined_rows, output_fieldnames = read_combined_output(output_csv)
        rewrite_output_with_original_text(
            output_csv,
            original_rows,
            combined_rows,
            output_fieldnames,
            subject_column=args.subject_column,
            body_column=args.body_column,
            label_column=args.label_column,
            data_source_column=args.data_source_column,
            dataset_name=f"HW-result_{dataset_stem}",
            source_path=input_csv,
        )

        write_manifest(
            manifest_path,
            {
                "timestamp": datetime.utcnow().isoformat() + "Z",
                "input_csv": str(input_csv),
                "prepared_input_csv": str(prepared_input_csv),
                "output_csv": str(output_csv),
                "dataset_name": dataset_name,
                "row_count": len(original_rows),
                "detectors": list(args.detectors),
                "python_bin": args.python_bin,
                "chunk_size": args.chunk_size,
                "progress_interval": args.progress_interval,
                "max_subject_chars": args.max_subject_chars,
                "max_body_chars": args.max_body_chars,
                "resume_existing": bool(args.resume_existing),
                "delegated_runner": str(RUN_TEXT_DETECTORS.resolve()),
            },
        )

        if args.keep_prepared_inputs:
            if prepared_parent is None:
                raise SystemExit("--keep-prepared-inputs requires --tmp-dir.")
            kept_dir = prepared_parent / "last_run"
            if kept_dir.exists():
                shutil.rmtree(kept_dir)
            shutil.copytree(temp_dir_path, kept_dir)

    print(f"[{timestamp()}] saved final={output_csv}", flush=True)


def main() -> int:
    args = parse_args()
    args.input_dir = args.input_dir.resolve()
    args.output_dir = args.output_dir.resolve()
    if args.tmp_dir is not None:
        args.tmp_dir = args.tmp_dir.resolve()

    if not RUN_TEXT_DETECTORS.exists():
        raise SystemExit(f"Missing delegated runner: {RUN_TEXT_DETECTORS}")
    if args.chunk_size <= 0:
        raise SystemExit("--chunk-size must be greater than 0.")
    if args.progress_interval < 0:
        raise SystemExit("--progress-interval cannot be negative.")
    if args.start_row <= 0:
        raise SystemExit("--start-row must be greater than 0.")

    args.output_dir.mkdir(parents=True, exist_ok=True)
    if not args.datasets:
        args.datasets = resolve_default_datasets(args.input_dir)
    if not args.datasets:
        raise SystemExit(f"No *-GD.csv datasets found in {args.input_dir}")

    for dataset_name in args.datasets:
        process_dataset(args, dataset_name)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
