#!/usr/bin/env python3
"""Batch runner for llm_guard scanners on CSV email datasets."""

import argparse
import csv
import importlib
import json
import os
import sys
from datetime import datetime
from pathlib import Path


SCRIPT_DIR = Path(__file__).resolve().parent
MODEL_NAME = "llm-guard"
DEFAULT_INPUT_CSV = (
    Path("/scratch3/che489/FC-W2-SoK/ssc_LLM_phishing_survey")
    / "Datasets/sublist/S5-Personalization for Credibility/LLM-P.csv"
)
DEFAULT_SAMPLE_SIZE = 20
DEFAULT_SUBSTRINGS = [
    "verify your account",
    "click the link",
    "confirm your identity",
]
DEFAULT_ENTITY_TYPES = [
    "PERSON",
    "EMAIL",
]


def detect_default_python():
    preferred = Path("/scratch3/che489/Ha/.conda/envs/FC-W2-gpu-p39/bin/python")
    if preferred.exists():
        return str(preferred)
    return sys.executable


def sanitize_name(value):
    cleaned = "".join(ch if ch.isalnum() or ch in ("-", "_") else "_" for ch in str(value))
    while "__" in cleaned:
        cleaned = cleaned.replace("__", "_")
    return cleaned.strip("_") or "run"


def now_stamp():
    return datetime.now().strftime("%Y%m%d_%H%M%S")


def infer_dataset_name(input_csv):
    return sanitize_name("{}__{}".format(input_csv.parent.name, input_csv.stem))


def build_output_dir(args):
    if args.output_dir:
        return args.output_dir.resolve()
    run_name = "{}__{}__{}".format(infer_dataset_name(args.input_csv), MODEL_NAME, now_stamp())
    return (SCRIPT_DIR / "runs" / run_name).resolve()


def parse_args():
    parser = argparse.ArgumentParser(
        description="Run llm_guard scanners over a CSV email dataset and export a compact summary."
    )
    parser.add_argument("--input-csv", type=Path, default=DEFAULT_INPUT_CSV)
    parser.add_argument("--subject-column", default="Subject")
    parser.add_argument("--body-column", default="Body")
    parser.add_argument("--sample-size", type=int, default=DEFAULT_SAMPLE_SIZE)
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--malicious-urls-threshold", type=float, default=0.7)
    return parser.parse_args()


def normalize_text(value):
    return (value or "").replace("\r\n", "\n").replace("\r", "\n").strip()


def build_content(subject, body):
    subject = normalize_text(subject)
    body = normalize_text(body)
    if subject and body:
        return "{}\n\n{}".format(subject, body)
    return subject or body


def write_manifest(path, payload):
    with path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, ensure_ascii=False)


def load_scanners(malicious_urls_threshold):
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

    script_dir = str(SCRIPT_DIR)
    original_sys_path = list(sys.path)
    sys.path = [path for path in sys.path if path != script_dir]
    try:
        util = importlib.import_module("llm_guard.util")
        output_scanners = importlib.import_module("llm_guard.output_scanners")
        input_ban_substrings = importlib.import_module("llm_guard.input_scanners.ban_substrings")
    finally:
        sys.path = original_sys_path

    util.configure_logger(log_level="ERROR")
    BanSubstrings = output_scanners.BanSubstrings
    MaliciousURLs = output_scanners.MaliciousURLs
    Sensitive = output_scanners.Sensitive
    MatchType = input_ban_substrings.MatchType

    return [
        (
            "malicious_urls",
            MaliciousURLs(threshold=malicious_urls_threshold),
        ),
        (
            "ban_substrings",
            BanSubstrings(
                substrings=list(DEFAULT_SUBSTRINGS),
                match_type=MatchType.STR,
                case_sensitive=False,
                redact=False,
                contains_all=False,
            ),
        ),
        (
            "sensitive",
            Sensitive(entity_types=list(DEFAULT_ENTITY_TYPES), redact=False),
        ),
    ]


def main():
    args = parse_args()
    if not args.input_csv.exists():
        raise SystemExit("Input CSV not found: {}".format(args.input_csv))

    output_dir = build_output_dir(args)
    summary_dir = output_dir / "summary"
    output_dir.mkdir(parents=True, exist_ok=True)
    summary_dir.mkdir(parents=True, exist_ok=True)

    output_csv = summary_dir / "llm_guard_summary.csv"
    manifest_path = output_dir / "run_manifest.json"
    dataset_name = infer_dataset_name(args.input_csv)

    print("Initializing llm_guard scanners...")
    scanners = load_scanners(args.malicious_urls_threshold)
    scanner_names = [name for name, _scanner in scanners]

    fieldnames = [
        "row_number",
        "source_path",
        "source_row_number",
        "dataset_name",
        "subject",
        "label",
        "data_source",
        "content_length",
        "overall_is_valid",
        "invalid_scanner_count",
        "flagged_scanners",
    ]
    for scanner_name in scanner_names:
        fieldnames.extend([
            "{}_is_valid".format(scanner_name),
            "{}_risk_score".format(scanner_name),
        ])

    rows = []
    with args.input_csv.open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        if not reader.fieldnames:
            raise SystemExit("CSV has no header row: {}".format(args.input_csv))
        missing = [name for name in [args.subject_column, args.body_column] if name not in reader.fieldnames]
        if missing:
            raise SystemExit(
                "CSV missing required columns {} in {}".format(", ".join(missing), args.input_csv)
            )

        for index, record in enumerate(reader, start=1):
            if args.sample_size > 0 and index > args.sample_size:
                break

            subject = normalize_text(record.get(args.subject_column, ""))
            body = normalize_text(record.get(args.body_column, ""))
            content = build_content(subject, body)

            row = {
                "row_number": index,
                "source_path": str(args.input_csv),
                "source_row_number": index,
                "dataset_name": dataset_name,
                "subject": subject,
                "label": record.get("label", ""),
                "data_source": record.get("data_source", ""),
                "content_length": len(content),
            }

            flagged_scanners = []
            for scanner_name, scanner in scanners:
                _sanitized_output, is_valid, risk_score = scanner.scan("", content)
                row["{}_is_valid".format(scanner_name)] = is_valid
                row["{}_risk_score".format(scanner_name)] = risk_score
                if is_valid is False:
                    flagged_scanners.append(scanner_name)

            row["overall_is_valid"] = len(flagged_scanners) == 0
            row["invalid_scanner_count"] = len(flagged_scanners)
            row["flagged_scanners"] = " | ".join(flagged_scanners)
            rows.append(row)
            print("[{}/?] row {} -> {}".format(index, index, row["flagged_scanners"] or "clean"))

    with output_csv.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    write_manifest(manifest_path, {
        "timestamp": datetime.utcnow().isoformat() + "Z",
        "python_bin": detect_default_python(),
        "input_mode": "csv",
        "input_csv": str(args.input_csv),
        "dataset_name": dataset_name,
        "model_name": MODEL_NAME,
        "sample_size": len(rows),
        "output_csv": str(output_csv),
        "output_dir": str(output_dir),
        "summary_dir": str(summary_dir),
        "scanner_config": {
            "malicious_urls_threshold": args.malicious_urls_threshold,
            "ban_substrings": list(DEFAULT_SUBSTRINGS),
            "sensitive_entity_types": list(DEFAULT_ENTITY_TYPES),
        },
        "saved_artifacts": [
            str(manifest_path),
            str(output_csv),
        ],
    })

    print(json.dumps({
        "output_csv": str(output_csv),
        "output_dir": str(output_dir),
        "manifest": str(manifest_path),
        "rows": len(rows),
    }, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
