#!/usr/bin/env python3
"""Batch runner for Apache SpamAssassin on CSV email datasets."""

from __future__ import annotations

import argparse
import csv
import json
import os
import re
import shutil
import subprocess
import sys
from datetime import datetime
from email import policy
from email.message import EmailMessage
from email.parser import BytesParser
from pathlib import Path
from typing import Any


SCRIPT_DIR = Path(__file__).resolve().parent
MODEL_NAME = "spamassassin"
DEFAULT_INPUT_CSV = (
    Path("/scratch3/che489/FC-W2-SoK/ssc_LLM_phishing_survey")
    / "Datasets/sublist/S5-Personalization for Credibility/LLM-P.csv"
)
DEFAULT_TIMEOUT_SECONDS = 120
DEFAULT_SITE_CONFIG_DIR = Path("/etc/mail/spamassassin")
DEFAULT_BENCHMARK_CONFIG_FILE = SCRIPT_DIR / "spamassassin" / "50_local_benchmark.cf"
DEFAULT_PREFS_FILE = SCRIPT_DIR / "spamassassin" / "user_prefs"

STATUS_RE = re.compile(r"^(Yes|No)\s*,", re.IGNORECASE)
SCORE_RE = re.compile(r"\bscore=([-+]?\d+(?:\.\d+)?)\b", re.IGNORECASE)
REQUIRED_RE = re.compile(r"\brequired=([-+]?\d+(?:\.\d+)?)\b", re.IGNORECASE)
TESTS_RE = re.compile(r"\btests=([^ ]+)", re.IGNORECASE)

csv.field_size_limit(sys.maxsize)


def sanitize_name(value: str) -> str:
    cleaned = "".join(ch if ch.isalnum() or ch in ("-", "_") else "_" for ch in str(value))
    while "__" in cleaned:
        cleaned = cleaned.replace("__", "_")
    return cleaned.strip("_") or "run"


def now_stamp() -> str:
    return datetime.now().strftime("%Y%m%d_%H%M%S")


def infer_dataset_name(input_csv: Path) -> str:
    return sanitize_name(f"{input_csv.parent.name}__{input_csv.stem}")


def build_output_dir(args: argparse.Namespace) -> Path:
    if args.output_dir:
        return args.output_dir.resolve()
    run_name = f"{infer_dataset_name(args.input_csv)}__{MODEL_NAME}__{now_stamp()}"
    return (SCRIPT_DIR / "runs" / run_name).resolve()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run Apache SpamAssassin over a CSV email dataset and export a compact summary."
    )
    parser.add_argument("--input-csv", type=Path, default=DEFAULT_INPUT_CSV)
    parser.add_argument("--subject-column", default="Subject")
    parser.add_argument("--body-column", default="Body")
    parser.add_argument("--sample-size", type=int, default=20)
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--spamassassin-bin", default=os.environ.get("SPAMASSASSIN_BIN", "spamassassin"))
    parser.add_argument(
        "--siteconfigpath",
        type=Path,
        default=Path(os.environ.get("SPAMASSASSIN_SITE_CONFIG_DIR", DEFAULT_SITE_CONFIG_DIR)),
    )
    parser.add_argument(
        "--benchmark-config-file",
        type=Path,
        default=Path(os.environ.get("SPAMASSASSIN_BENCHMARK_CONFIG_FILE", DEFAULT_BENCHMARK_CONFIG_FILE)),
    )
    parser.add_argument(
        "--prefs-file",
        type=Path,
        default=Path(os.environ.get("SPAMASSASSIN_PREFS_FILE", DEFAULT_PREFS_FILE)),
    )
    parser.add_argument("--from-address", default="sender@example.com")
    parser.add_argument("--to-address", default="recipient@example.com")
    parser.add_argument("--timeout-seconds", type=int, default=DEFAULT_TIMEOUT_SECONDS)
    parser.add_argument(
        "--progress-every",
        type=int,
        default=500,
        help="Print progress every N processed rows. Set 0 to disable. Default: 500",
    )
    parser.add_argument(
        "--local-only",
        action="store_true",
        default=True,
        help="Use SpamAssassin local tests only (-L). Enabled by default for benchmark stability.",
    )
    parser.add_argument(
        "--allow-network-tests",
        dest="local_only",
        action="store_false",
        help="Allow SpamAssassin network-enabled tests.",
    )
    return parser.parse_args()


def normalize_text(value: Any) -> str:
    return (value or "").replace("\r\n", "\n").replace("\r", "\n").strip()


def write_manifest(path: Path, payload: dict[str, Any]) -> None:
    with path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, ensure_ascii=False)


def resolve_binary(binary_name: str) -> str:
    candidate = Path(binary_name)
    if candidate.is_absolute() or "/" in binary_name:
        if not candidate.exists():
            raise SystemExit(f"SpamAssassin binary not found: {binary_name}")
        return str(candidate)

    resolved = shutil.which(binary_name)
    if not resolved:
        raise SystemExit(
            f"SpamAssassin binary not found on PATH: {binary_name}. "
            "Run the install script under Detectors/Industry/email_detectors/spamassassin/ first."
        )
    return resolved


def build_message(subject: str, body: str, *, from_address: str, to_address: str) -> bytes:
    message = EmailMessage()
    message["From"] = from_address
    message["To"] = to_address
    message["Subject"] = subject or "(no subject)"
    message.set_content(body or "")
    return message.as_bytes(policy=policy.SMTP)


def parse_spamassassin_headers(message_bytes: bytes) -> tuple[str, str]:
    parsed = BytesParser(policy=policy.default).parsebytes(message_bytes)
    status_header = str(parsed.get("X-Spam-Status", "") or "").replace("\n", " ").replace("\r", " ").strip()
    flag_header = str(parsed.get("X-Spam-Flag", "") or "").replace("\n", " ").replace("\r", " ").strip()
    return status_header, flag_header


def parse_prediction(status_header: str, flag_header: str) -> tuple[str | None, str, str, str]:
    status_match = STATUS_RE.search(status_header)
    score_match = SCORE_RE.search(status_header)
    required_match = REQUIRED_RE.search(status_header)
    tests_match = TESTS_RE.search(status_header)

    prediction: str | None = None
    if status_match:
        prediction = "1" if status_match.group(1).lower() == "yes" else "0"
    elif flag_header:
        lowered_flag = flag_header.lower()
        if lowered_flag == "yes":
            prediction = "1"
        elif lowered_flag == "no":
            prediction = "0"

    score = score_match.group(1) if score_match else ""
    required_score = required_match.group(1) if required_match else ""
    tests = tests_match.group(1) if tests_match else ""
    return prediction, score, required_score, tests


def read_benchmark_cf_lines(config_file: Path) -> list[str]:
    lines: list[str] = []
    with config_file.open("r", encoding="utf-8") as handle:
        for raw_line in handle:
            stripped = raw_line.strip()
            if not stripped or stripped.startswith("#"):
                continue
            lines.append(stripped)
    return lines


def run_spamassassin(
    *,
    binary_path: str,
    message_bytes: bytes,
    siteconfigpath: Path,
    benchmark_config_file: Path,
    prefs_file: Path,
    timeout_seconds: int,
    local_only: bool,
) -> tuple[str, str, str]:
    command = [
        binary_path,
        "--siteconfigpath",
        str(siteconfigpath.resolve()),
        "--prefs-file",
        str(prefs_file.resolve()),
        "-x",
    ]
    for line in read_benchmark_cf_lines(benchmark_config_file):
        command.extend(["--cf", line])
    if local_only:
        command.append("-L")

    completed = subprocess.run(
        command,
        input=message_bytes,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        timeout=timeout_seconds,
        check=False,
    )
    stdout = completed.stdout.decode("utf-8", errors="replace")
    stderr = completed.stderr.decode("utf-8", errors="replace").strip()
    if completed.returncode != 0:
        error = stderr or f"SpamAssassin exited with code {completed.returncode}"
        return "", "", error

    status_header, flag_header = parse_spamassassin_headers(completed.stdout)
    if not status_header:
        error = stderr or "SpamAssassin output did not contain X-Spam-Status"
        return "", flag_header, error
    return status_header, flag_header, stderr


def main() -> int:
    args = parse_args()
    if not args.input_csv.exists():
        raise SystemExit(f"Input CSV not found: {args.input_csv}")
    if not args.siteconfigpath.exists():
        raise SystemExit(f"SpamAssassin site config directory not found: {args.siteconfigpath}")
    if not args.benchmark_config_file.exists():
        raise SystemExit(f"SpamAssassin benchmark config file not found: {args.benchmark_config_file}")
    if not args.prefs_file.exists():
        raise SystemExit(f"SpamAssassin prefs file not found: {args.prefs_file}")
    if args.progress_every < 0:
        raise SystemExit("--progress-every cannot be negative.")

    binary_path = resolve_binary(args.spamassassin_bin)
    output_dir = build_output_dir(args)
    summary_dir = output_dir / "summary"
    output_dir.mkdir(parents=True, exist_ok=True)
    summary_dir.mkdir(parents=True, exist_ok=True)

    output_csv = summary_dir / "spamassassin_summary.csv"
    manifest_path = output_dir / "run_manifest.json"
    dataset_name = infer_dataset_name(args.input_csv)

    fieldnames = [
        "row_number",
        "source_path",
        "source_row_number",
        "dataset_name",
        "subject",
        "label",
        "data_source",
        "content_length",
        "spamassassin_prediction",
        "spamassassin_status_header",
        "spamassassin_flag_header",
        "spamassassin_score",
        "spamassassin_required_score",
        "spamassassin_tests",
        "spamassassin_error",
    ]

    rows: list[dict[str, Any]] = []
    with args.input_csv.open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        if not reader.fieldnames:
            raise SystemExit(f"CSV has no header row: {args.input_csv}")
        missing = [name for name in [args.subject_column, args.body_column] if name not in reader.fieldnames]
        if missing:
            raise SystemExit(f"CSV missing required columns {', '.join(missing)} in {args.input_csv}")

        for index, record in enumerate(reader, start=1):
            if args.sample_size > 0 and index > args.sample_size:
                break

            subject = normalize_text(record.get(args.subject_column, ""))
            body = normalize_text(record.get(args.body_column, ""))
            message_bytes = build_message(
                subject,
                body,
                from_address=args.from_address,
                to_address=args.to_address,
            )
            status_header, flag_header, error = run_spamassassin(
                binary_path=binary_path,
                message_bytes=message_bytes,
                siteconfigpath=args.siteconfigpath,
                benchmark_config_file=args.benchmark_config_file,
                prefs_file=args.prefs_file,
                timeout_seconds=args.timeout_seconds,
                local_only=args.local_only,
            )
            prediction, score, required_score, tests = parse_prediction(status_header, flag_header)
            row = {
                "row_number": index,
                "source_path": str(args.input_csv),
                "source_row_number": index,
                "dataset_name": dataset_name,
                "subject": subject,
                "label": record.get("label", ""),
                "data_source": record.get("data_source", ""),
                "content_length": len(subject) + len(body),
                "spamassassin_prediction": prediction or "",
                "spamassassin_status_header": status_header,
                "spamassassin_flag_header": flag_header,
                "spamassassin_score": score,
                "spamassassin_required_score": required_score,
                "spamassassin_tests": tests,
                "spamassassin_error": error,
            }
            rows.append(row)
            if args.progress_every > 0 and index % args.progress_every == 0:
                print(f"[progress] rows={index}", flush=True)

    with output_csv.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    write_manifest(
        manifest_path,
        {
            "timestamp": datetime.utcnow().isoformat() + "Z",
            "input_mode": "csv",
            "input_csv": str(args.input_csv),
            "dataset_name": dataset_name,
            "model_name": MODEL_NAME,
            "sample_size": len(rows),
            "output_csv": str(output_csv),
            "output_dir": str(output_dir),
            "summary_dir": str(summary_dir),
            "spamassassin_bin": binary_path,
            "siteconfigpath": str(args.siteconfigpath.resolve()),
            "benchmark_config_file": str(args.benchmark_config_file.resolve()),
            "prefs_file": str(args.prefs_file.resolve()),
            "local_only": bool(args.local_only),
            "saved_artifacts": [
                str(manifest_path),
                str(output_csv),
            ],
        },
    )

    print(json.dumps({
        "output_csv": str(output_csv),
        "output_dir": str(output_dir),
        "manifest": str(manifest_path),
        "rows": len(rows),
    }, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
