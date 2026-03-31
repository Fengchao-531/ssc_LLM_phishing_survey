#!/usr/bin/env python3
"""Unified runner for text-oriented phishing detectors.

This script standardizes detector execution for CSV email datasets.
It keeps the original dataset columns and appends normalized detector outputs
so downstream statistics can work from a single combined table.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import subprocess
import sys
import tempfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


SCRIPT_DIR = Path(__file__).resolve().parent
DETECTORS_DIR = SCRIPT_DIR.parent
EMAIL_DETECTORS_DIR = DETECTORS_DIR / "Industry" / "email_detectors"
OPEN_SOURCE_DIR = EMAIL_DETECTORS_DIR / "open-source-git"
OUTPUT_ROOT = SCRIPT_DIR

DEFAULT_INPUT_CSV = (
    DETECTORS_DIR.parent
    / "Datasets"
    / "sublist"
    / "S5-Personalization for Credibility"
    / "LLM-P.csv"
)
DEFAULT_DETECTORS = [
    "llm_guard",
    "phishing_email_agent",
    "email_phishing_detection_v3",
]
RESULT_GROUPS = [
    "LLM-result",
    "HW-result",
]
ALL_DETECTORS = [
    "llm_guard",
    "phishing_email_agent",
    "email_phishing_detection_v3",
    "pyrit_original",
    "pyrit_blocklist",
    "oopspam",
]


def detect_default_python() -> str:
    preferred = Path("/scratch3/che489/Ha/.conda/envs/FC-W2-gpu-p39/bin/python")
    if preferred.exists():
        return str(preferred)
    return sys.executable


def now_stamp() -> str:
    return datetime.now().strftime("%Y%m%d_%H%M%S")


def sanitize_name(value: str) -> str:
    cleaned = "".join(ch if ch.isalnum() or ch in ("-", "_") else "_" for ch in str(value))
    while "__" in cleaned:
        cleaned = cleaned.replace("__", "_")
    return cleaned.strip("_") or "run"


def infer_dataset_name(input_csv: Path) -> str:
    return sanitize_name(f"{input_csv.parent.name}__{input_csv.stem}")


def infer_result_group(input_csv: Path) -> str:
    haystack = f"{input_csv.parent.name} {input_csv.stem} {input_csv}".lower()
    if "hw" in haystack:
        return "HW-result"
    return "LLM-result"


def infer_stage_name(input_csv: Path) -> str:
    stem_upper = input_csv.stem.upper()
    if stem_upper.startswith("S") and len(stem_upper) >= 2 and stem_upper[1].isdigit():
        return stem_upper.split("_", 1)[0].split("-", 1)[0]

    parent_upper = input_csv.parent.name.upper()
    if parent_upper.startswith("S") and len(parent_upper) >= 2 and parent_upper[1].isdigit():
        return parent_upper.split("-", 1)[0].split(" ", 1)[0]

    return sanitize_name(input_csv.stem).upper()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Run multiple text detectors on one CSV dataset and export one combined CSV "
            "with the original rows plus normalized detector outputs."
        )
    )
    parser.add_argument("--input-csv", type=Path, default=DEFAULT_INPUT_CSV)
    parser.add_argument("--subject-column", default="Subject")
    parser.add_argument("--body-column", default="Body")
    parser.add_argument("--label-column", default="label")
    parser.add_argument("--data-source-column", default="data_source")
    parser.add_argument(
        "--detectors",
        nargs="+",
        choices=ALL_DETECTORS,
        default=list(DEFAULT_DETECTORS),
        help=(
            "Detectors to run. Defaults to the currently benchmark-verified text detectors: "
            "llm_guard phishing_email_agent email_phishing_detection_v3"
        ),
    )
    parser.add_argument(
        "--sample-size",
        type=int,
        default=0,
        help="Number of rows to run. Use 0 for all rows. Default: 0",
    )
    parser.add_argument(
        "--checkpoint-every",
        type=int,
        default=100,
        help="Write back combined results after every N rows per detector. Default: 100",
    )
    parser.add_argument(
        "--result-group",
        choices=RESULT_GROUPS,
        default=None,
        help="Top-level output bucket. Defaults to LLM-result/HW-result inferred from the dataset path.",
    )
    parser.add_argument(
        "--stage-name",
        default=None,
        help="Short stage label such as S1 or S2. Defaults to inferring from the input filename/path.",
    )
    parser.add_argument(
        "--output-csv",
        type=Path,
        default=None,
        help="Override the final combined results CSV path. Default: Detectors/output/<group>/<stage>_results.csv",
    )
    parser.add_argument("--python-bin", default=detect_default_python())
    parser.add_argument("--from-address", default="sender@example.com")
    parser.add_argument("--to-address", default="recipient@example.com")
    parser.add_argument("--backend-root", default="http://127.0.0.1:5000")
    parser.add_argument("--openrouter-api-key", default=os.environ.get("OPENROUTER_API_KEY", ""))
    parser.add_argument("--oopspam-api-key", default=os.environ.get("OOPSPAM_API_KEY", ""))
    parser.add_argument("--oopspam-threshold", type=int, default=3)
    parser.add_argument("--oopspam-sender-ip", default=os.environ.get("OOPSPAM_SENDER_IP", "127.0.0.1"))
    parser.add_argument("--oopspam-email", default=os.environ.get("OOPSPAM_EMAIL", "testing@example.com"))
    parser.add_argument("--oopspam-source", default=os.environ.get("OOPSPAM_SOURCE", "benchmark.local"))
    parser.add_argument(
        "--fail-fast",
        action="store_true",
        help="Stop immediately if one detector run fails.",
    )
    return parser.parse_args()


def normalize_text(value: Any) -> str:
    return (value or "").replace("\r\n", "\n").replace("\r", "\n").strip()


def build_content(subject: str, body: str) -> str:
    subject = normalize_text(subject)
    body = normalize_text(body)
    if subject and body:
        return f"{subject}\n\n{body}"
    return subject or body


def make_run_dir(args: argparse.Namespace) -> Path:
    if args.output_csv:
        return args.output_csv.resolve().parent
    result_group = args.result_group or infer_result_group(args.input_csv)
    return (OUTPUT_ROOT / result_group).resolve()


def make_output_csv_path(args: argparse.Namespace) -> Path:
    if args.output_csv:
        return args.output_csv.resolve()
    result_group = args.result_group or infer_result_group(args.input_csv)
    stage_name = (args.stage_name or infer_stage_name(args.input_csv)).upper()
    return (OUTPUT_ROOT / result_group / f"{stage_name}_results.csv").resolve()


def read_input_rows(args: argparse.Namespace) -> tuple[list[dict[str, Any]], list[str]]:
    if not args.input_csv.exists():
        raise SystemExit(f"Input CSV not found: {args.input_csv}")

    with args.input_csv.open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        if not reader.fieldnames:
            raise SystemExit(f"CSV has no header row: {args.input_csv}")

        missing = [
            name for name in [args.subject_column, args.body_column] if name not in reader.fieldnames
        ]
        if missing:
            raise SystemExit(
                f"CSV missing required columns {', '.join(missing)} in {args.input_csv}"
            )

        source_rows = list(reader)
        if args.sample_size > 0:
            source_rows = source_rows[: args.sample_size]

    dataset_name = infer_dataset_name(args.input_csv)
    rows: list[dict[str, Any]] = []
    for index, row in enumerate(source_rows, start=1):
        enriched = dict(row)
        enriched["benchmark_row_number"] = index
        enriched["dataset_name"] = dataset_name
        enriched["source_path"] = str(args.input_csv)
        enriched["source_row_number"] = index
        enriched["_combined_text"] = build_content(
            row.get(args.subject_column, ""),
            row.get(args.body_column, ""),
        )
        rows.append(enriched)

    return rows, list(reader.fieldnames)


def write_text_projection_csv(
    rows: list[dict[str, Any]],
    original_fieldnames: list[str],
    out_path: Path,
) -> None:
    fieldnames = list(original_fieldnames)
    if "text" not in fieldnames:
        fieldnames.append("text")

    with out_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            record = {name: row.get(name, "") for name in original_fieldnames}
            record["text"] = row.get("_combined_text", "")
            writer.writerow(record)


def write_original_csv(
    rows: list[dict[str, Any]],
    original_fieldnames: list[str],
    out_path: Path,
) -> None:
    with out_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=original_fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({name: row.get(name, "") for name in original_fieldnames})


def write_json(path: Path, payload: dict[str, Any]) -> None:
    with path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, ensure_ascii=False)


def run_subprocess(
    command: list[str],
    *,
    cwd: Path,
    log_path: Path,
    env_overrides: dict[str, str] | None = None,
) -> tuple[int, str]:
    env = os.environ.copy()
    if env_overrides:
        env.update({key: value for key, value in env_overrides.items() if value})
    completed = subprocess.run(
        command,
        cwd=str(cwd),
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        universal_newlines=True,
        env=env,
    )
    output = completed.stdout or ""
    log_path.write_text(output, encoding="utf-8")
    return completed.returncode, output


def parse_bool(value: Any) -> bool | None:
    text = str(value).strip().lower()
    if text == "true":
        return True
    if text == "false":
        return False
    return None


def parse_float(value: Any) -> float | None:
    try:
        if value in ("", None):
            return None
        return float(value)
    except Exception:
        return None


def normalize_binary_prediction(value: Any) -> int | None:
    text = str(value).strip().lower()
    if text in {"1", "true", "phishing", "suspicious", "malicious", "reject"}:
        return 1
    if text in {"0", "false", "benign", "legitimate", "safe", "clean", "accept"}:
        return 0
    return None


def parse_llm_guard(summary_csv: Path) -> dict[int, dict[str, Any]]:
    parsed: dict[int, dict[str, Any]] = {}
    with summary_csv.open("r", encoding="utf-8", newline="") as handle:
        for row in csv.DictReader(handle):
            row_id = int(row["row_number"])
            overall_is_valid = parse_bool(row.get("overall_is_valid"))
            prediction = None if overall_is_valid is None else int(not overall_is_valid)
            parsed[row_id] = {
                "llm_guard_status": "ok",
                "llm_guard_prediction": prediction,
                "llm_guard_invalid_scanner_count": row.get("invalid_scanner_count", ""),
                "llm_guard_flagged_scanners": row.get("flagged_scanners", ""),
                "llm_guard_malicious_urls_risk_score": row.get("malicious_urls_risk_score", ""),
                "llm_guard_ban_substrings_risk_score": row.get("ban_substrings_risk_score", ""),
                "llm_guard_sensitive_risk_score": row.get("sensitive_risk_score", ""),
            }
    return parsed


def parse_phishing_email_agent(summary_csv: Path) -> dict[int, dict[str, Any]]:
    parsed: dict[int, dict[str, Any]] = {}
    with summary_csv.open("r", encoding="utf-8", newline="") as handle:
        for row in csv.DictReader(handle):
            row_id = int(row["row_number"])
            parsed[row_id] = {
                "phishing_email_agent_status": "ok",
                "phishing_email_agent_prediction": normalize_binary_prediction(row.get("prediction")),
                "phishing_email_agent_prediction_label": row.get("prediction", ""),
                "phishing_email_agent_probability": row.get("probability", ""),
                "phishing_email_agent_confidence": row.get("confidence", ""),
                "phishing_email_agent_error": row.get("response_error", ""),
            }
    return parsed


def parse_email_phishing_detection_v3(summary_csv: Path) -> dict[int, dict[str, Any]]:
    parsed: dict[int, dict[str, Any]] = {}
    with summary_csv.open("r", encoding="utf-8", newline="") as handle:
        for row in csv.DictReader(handle):
            row_key = row.get("source_row_number") or row.get("row_number")
            row_id = int(row_key)
            parsed[row_id] = {
                "email_phishing_detection_v3_status": "ok",
                "email_phishing_detection_v3_prediction": normalize_binary_prediction(row.get("AI Verdict")),
                "email_phishing_detection_v3_prediction_label": row.get("AI Verdict", ""),
                "email_phishing_detection_v3_confidence": row.get("AI Confidence", ""),
                "email_phishing_detection_v3_score_0_to_10": row.get("AI Phishing Score (0-10)", ""),
                "email_phishing_detection_v3_brands": row.get("AI Identified Brands", ""),
                "email_phishing_detection_v3_error": row.get("official_error", ""),
            }
    return parsed


def parse_pyrit_original(summary_csv: Path) -> dict[int, dict[str, Any]]:
    parsed: dict[int, dict[str, Any]] = {}
    with summary_csv.open("r", encoding="utf-8", newline="") as handle:
        for row_id, row in enumerate(csv.DictReader(handle), start=1):
            raw = row.get("pyrit_scan_results_json", "")
            error = row.get("pyrit_scan_error", "")
            max_score = None
            top_category = ""
            if raw:
                try:
                    data = json.loads(raw)
                    if isinstance(data, list) and data:
                        best = max(
                            data,
                            key=lambda item: float(item.get("normalized_score", -1)),
                        )
                        max_score = best.get("normalized_score", "")
                        top_category = best.get("score_category", "")
                except Exception:
                    pass
            parsed[row_id] = {
                "pyrit_original_status": "ok" if not error else "row_error",
                "pyrit_original_prediction": "",
                "pyrit_original_max_normalized_score": max_score if max_score is not None else "",
                "pyrit_original_top_category": top_category,
                "pyrit_original_error": error,
            }
    return parsed


def parse_pyrit_blocklist(summary_csv: Path) -> dict[int, dict[str, Any]]:
    parsed: dict[int, dict[str, Any]] = {}
    with summary_csv.open("r", encoding="utf-8", newline="") as handle:
        for row_id, row in enumerate(csv.DictReader(handle), start=1):
            raw = row.get("phishing_detection_results_json", "")
            error = row.get("phishing_detection_error", "")
            prediction = None
            match_count = ""
            if raw:
                try:
                    data = json.loads(raw)
                    decision = data.get("decision", {})
                    suggested_action = decision.get("suggested_action", "")
                    prediction = normalize_binary_prediction(suggested_action)
                    detection_result = data.get("detection_result", {})
                    matches = detection_result.get("blocklistsMatch", [])
                    match_count = len(matches) if isinstance(matches, list) else ""
                except Exception:
                    pass
            parsed[row_id] = {
                "pyrit_blocklist_status": "ok" if not error else "row_error",
                "pyrit_blocklist_prediction": prediction,
                "pyrit_blocklist_match_count": match_count,
                "pyrit_blocklist_error": error,
            }
    return parsed


def parse_oopspam(summary_csv: Path) -> dict[int, dict[str, Any]]:
    parsed: dict[int, dict[str, Any]] = {}
    with summary_csv.open("r", encoding="utf-8", newline="") as handle:
        for row in csv.DictReader(handle):
            row_id = int(row["row_number"])
            parsed[row_id] = {
                "oopspam_status": "ok" if not row.get("oopspam_error", "") else "row_error",
                "oopspam_prediction": normalize_binary_prediction(row.get("oopspam_prediction")),
                "oopspam_score": row.get("oopspam_score", ""),
                "oopspam_is_content_spam": row.get("oopspam_is_content_spam", ""),
                "oopspam_number_of_spam_words": row.get("oopspam_number_of_spam_words", ""),
                "oopspam_spam_words": row.get("oopspam_spam_words", ""),
                "oopspam_error": row.get("oopspam_error", ""),
            }
    return parsed


def detector_fieldnames(detector_name: str) -> list[str]:
    mapping = {
        "llm_guard": [
            "llm_guard_status",
            "llm_guard_prediction",
            "llm_guard_invalid_scanner_count",
            "llm_guard_flagged_scanners",
            "llm_guard_malicious_urls_risk_score",
            "llm_guard_ban_substrings_risk_score",
            "llm_guard_sensitive_risk_score",
        ],
        "phishing_email_agent": [
            "phishing_email_agent_status",
            "phishing_email_agent_prediction",
            "phishing_email_agent_prediction_label",
            "phishing_email_agent_probability",
            "phishing_email_agent_confidence",
            "phishing_email_agent_error",
        ],
        "email_phishing_detection_v3": [
            "email_phishing_detection_v3_status",
            "email_phishing_detection_v3_prediction",
            "email_phishing_detection_v3_prediction_label",
            "email_phishing_detection_v3_confidence",
            "email_phishing_detection_v3_score_0_to_10",
            "email_phishing_detection_v3_brands",
            "email_phishing_detection_v3_error",
        ],
        "pyrit_original": [
            "pyrit_original_status",
            "pyrit_original_prediction",
            "pyrit_original_max_normalized_score",
            "pyrit_original_top_category",
            "pyrit_original_error",
        ],
        "pyrit_blocklist": [
            "pyrit_blocklist_status",
            "pyrit_blocklist_prediction",
            "pyrit_blocklist_match_count",
            "pyrit_blocklist_error",
        ],
        "oopspam": [
            "oopspam_status",
            "oopspam_prediction",
            "oopspam_score",
            "oopspam_is_content_spam",
            "oopspam_number_of_spam_words",
            "oopspam_spam_words",
            "oopspam_error",
        ],
    }
    return mapping[detector_name]


def empty_detector_values(detector_name: str, status: str) -> dict[str, Any]:
    values = {key: "" for key in detector_fieldnames(detector_name)}
    status_key = detector_fieldnames(detector_name)[0]
    values[status_key] = status
    return values


def initialize_detector_status(
    combined_rows: list[dict[str, Any]],
    detector_name: str,
    status: str,
) -> None:
    status_key = detector_fieldnames(detector_name)[0]
    for row in combined_rows:
        if row.get(status_key, "") == "":
            row.update(empty_detector_values(detector_name, status))


def apply_chunk_results(
    combined_rows: list[dict[str, Any]],
    detector_name: str,
    start_index: int,
    chunk_size: int,
    parsed_rows: dict[int, dict[str, Any]] | None,
    fallback_status: str,
) -> None:
    for local_index in range(1, chunk_size + 1):
        global_row = combined_rows[start_index + local_index - 1]
        if parsed_rows and local_index in parsed_rows:
            global_row.update(parsed_rows[local_index])
        else:
            global_row.update(empty_detector_values(detector_name, fallback_status))


def build_detector_command(
    detector_name: str,
    *,
    args: argparse.Namespace,
    detector_dir: Path,
    chunk_input_csv: Path,
    chunk_text_projection_csv: Path,
    chunk_size: int,
) -> tuple[list[str], Path, Path, dict[str, str]]:
    summary_dir = detector_dir / "summary"
    summary_dir.mkdir(parents=True, exist_ok=True)

    if detector_name == "llm_guard":
        command = [
            args.python_bin,
            str(EMAIL_DETECTORS_DIR / "llm_guard.py"),
            "--input-csv",
            str(chunk_input_csv),
            "--subject-column",
            args.subject_column,
            "--body-column",
            args.body_column,
            "--sample-size",
            str(chunk_size),
            "--output-dir",
            str(detector_dir),
        ]
        return command, EMAIL_DETECTORS_DIR, summary_dir / "llm_guard_summary.csv", {}

    if detector_name == "phishing_email_agent":
        command = [
            args.python_bin,
            str(OPEN_SOURCE_DIR / "Phishing-Email-Agent.py"),
            "--repo-dir",
            str(OPEN_SOURCE_DIR / "Phishing-Email-Agent"),
            "--python-bin",
            args.python_bin,
            "--input-csv",
            str(chunk_input_csv),
            "--subject-column",
            args.subject_column,
            "--body-column",
            args.body_column,
            "--sample-size",
            str(chunk_size),
            "--output-dir",
            str(detector_dir),
            "--backend-root",
            args.backend_root,
            "--auto-start-backend",
        ]
        return command, OPEN_SOURCE_DIR, summary_dir / "Phishing-Email-Agent_summary.csv", {}

    if detector_name == "email_phishing_detection_v3":
        command = [
            args.python_bin,
            str(OPEN_SOURCE_DIR / "email-phishing-detection_V3.py"),
            "--repo-dir",
            str(OPEN_SOURCE_DIR / "email-phishing-detection_V3"),
            "--python-bin",
            args.python_bin,
            "--input-csv",
            str(chunk_input_csv),
            "--subject-column",
            args.subject_column,
            "--body-column",
            args.body_column,
            "--sample-size",
            str(chunk_size),
            "--output-dir",
            str(detector_dir),
            "--from-address",
            args.from_address,
            "--to-address",
            args.to_address,
        ]
        env_overrides = {}
        if args.openrouter_api_key:
            env_overrides["OPENROUTER_API_KEY"] = args.openrouter_api_key
        return command, OPEN_SOURCE_DIR, summary_dir / "email-phishing-detection_V3_ai_summary.csv", env_overrides

    if detector_name == "pyrit_original":
        output_csv = summary_dir / "pyrit_original_summary.csv"
        command = [
            args.python_bin,
            str(EMAIL_DETECTORS_DIR / "PyRIT-scan-original.py"),
            "--input-csv",
            str(chunk_text_projection_csv),
            "--output-csv",
            str(output_csv),
            "--text-column",
            "text",
        ]
        return command, EMAIL_DETECTORS_DIR, output_csv, {}

    if detector_name == "pyrit_blocklist":
        output_csv = summary_dir / "pyrit_blocklist_summary.csv"
        command = [
            args.python_bin,
            str(EMAIL_DETECTORS_DIR / "PyRIT-scan-blocklist.py"),
            "--input-csv",
            str(chunk_text_projection_csv),
            "--output-csv",
            str(output_csv),
            "--text-column",
            "text",
        ]
        return command, EMAIL_DETECTORS_DIR, output_csv, {}

    if detector_name == "oopspam":
        output_csv = summary_dir / "oopspam_summary.csv"
        command = [
            args.python_bin,
            str(EMAIL_DETECTORS_DIR / "oopspam.py"),
            "--input-csv",
            str(chunk_input_csv),
            "--subject-column",
            args.subject_column,
            "--body-column",
            args.body_column,
            "--sample-size",
            str(chunk_size),
            "--output-dir",
            str(detector_dir),
            "--threshold",
            str(args.oopspam_threshold),
            "--sender-ip",
            args.oopspam_sender_ip,
            "--email",
            args.oopspam_email,
            "--source",
            args.oopspam_source,
        ]
        env_overrides = {}
        if args.oopspam_api_key:
            env_overrides["OOPSPAM_API_KEY"] = args.oopspam_api_key
        return command, EMAIL_DETECTORS_DIR, output_csv, env_overrides

    raise SystemExit(f"Unsupported detector: {detector_name}")


def parse_detector_output(detector_name: str, summary_csv: Path) -> dict[int, dict[str, Any]]:
    if detector_name == "llm_guard":
        return parse_llm_guard(summary_csv)
    if detector_name == "phishing_email_agent":
        return parse_phishing_email_agent(summary_csv)
    if detector_name == "email_phishing_detection_v3":
        return parse_email_phishing_detection_v3(summary_csv)
    if detector_name == "pyrit_original":
        return parse_pyrit_original(summary_csv)
    if detector_name == "pyrit_blocklist":
        return parse_pyrit_blocklist(summary_csv)
    if detector_name == "oopspam":
        return parse_oopspam(summary_csv)
    raise SystemExit(f"Unsupported detector parser: {detector_name}")


def build_output_fieldnames(original_fieldnames: list[str], detectors: list[str]) -> list[str]:
    fieldnames = [
        "benchmark_row_number",
        "dataset_name",
        "source_path",
        "source_row_number",
    ]
    for name in original_fieldnames:
        if name not in fieldnames:
            fieldnames.append(name)
    for detector_name in detectors:
        fieldnames.extend(detector_fieldnames(detector_name))
    return fieldnames


def write_combined_output(
    output_csv: Path,
    combined_rows: list[dict[str, Any]],
    fieldnames: list[str],
) -> None:
    with output_csv.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in combined_rows:
            serializable = {key: value for key, value in row.items() if not key.startswith("_")}
            writer.writerow(serializable)


def main() -> int:
    args = parse_args()
    run_dir = make_run_dir(args)
    run_dir.mkdir(parents=True, exist_ok=True)
    final_output_csv = make_output_csv_path(args)

    combined_rows, original_fieldnames = read_input_rows(args)
    effective_sample_size = len(combined_rows)
    dataset_name = infer_dataset_name(args.input_csv)
    stage_name = (args.stage_name or infer_stage_name(args.input_csv)).upper()
    fieldnames = build_output_fieldnames(original_fieldnames, list(args.detectors))

    for detector_name in args.detectors:
        initialize_detector_status(combined_rows, detector_name, "pending")
    write_combined_output(final_output_csv, combined_rows, fieldnames)

    with tempfile.TemporaryDirectory(prefix=f"{stage_name.lower()}_detector_suite_") as temp_dir:
        temp_root = Path(temp_dir)

        for detector_name in args.detectors:
            checkpoint = max(1, args.checkpoint_every)
            for start_index in range(0, effective_sample_size, checkpoint):
                chunk_rows = combined_rows[start_index:start_index + checkpoint]
                chunk_size = len(chunk_rows)
                chunk_tag = f"{detector_name}_{start_index + 1:06d}_{start_index + chunk_size:06d}"
                detector_dir = temp_root / chunk_tag
                detector_dir.mkdir(parents=True, exist_ok=True)
                log_path = temp_root / f"{chunk_tag}.log"
                chunk_input_csv = detector_dir / "chunk_input.csv"
                chunk_text_projection_csv = detector_dir / "chunk_text_projection.csv"
                write_original_csv(chunk_rows, original_fieldnames, chunk_input_csv)
                write_text_projection_csv(chunk_rows, original_fieldnames, chunk_text_projection_csv)

                try:
                    command, cwd, summary_csv, env_overrides = build_detector_command(
                        detector_name,
                        args=args,
                        detector_dir=detector_dir,
                        chunk_input_csv=chunk_input_csv,
                        chunk_text_projection_csv=chunk_text_projection_csv,
                        chunk_size=chunk_size,
                    )
                    returncode, stdout_text = run_subprocess(
                        command,
                        cwd=cwd,
                        log_path=log_path,
                        env_overrides=env_overrides,
                    )
                    if returncode != 0:
                        apply_chunk_results(
                            combined_rows,
                            detector_name,
                            start_index,
                            chunk_size,
                            None,
                            "run_failed",
                        )
                        write_combined_output(final_output_csv, combined_rows, fieldnames)
                        if args.fail_fast:
                            raise SystemExit(stdout_text[-4000:] if stdout_text else "Detector run failed")
                        continue

                    if not summary_csv.exists():
                        apply_chunk_results(
                            combined_rows,
                            detector_name,
                            start_index,
                            chunk_size,
                            None,
                            "missing_summary",
                        )
                        write_combined_output(final_output_csv, combined_rows, fieldnames)
                        if args.fail_fast:
                            raise SystemExit(f"Expected summary CSV not found: {summary_csv}")
                        continue

                    parsed_rows = parse_detector_output(detector_name, summary_csv)
                    apply_chunk_results(
                        combined_rows,
                        detector_name,
                        start_index,
                        chunk_size,
                        parsed_rows,
                        "parse_failed",
                    )
                    write_combined_output(final_output_csv, combined_rows, fieldnames)
                except Exception:
                    apply_chunk_results(
                        combined_rows,
                        detector_name,
                        start_index,
                        chunk_size,
                        None,
                        "run_failed",
                    )
                    write_combined_output(final_output_csv, combined_rows, fieldnames)
                    if args.fail_fast:
                        raise

    print(json.dumps({
        "timestamp": datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
        "result_group": args.result_group or infer_result_group(args.input_csv),
        "stage_name": stage_name,
        "output_csv": str(final_output_csv),
        "rows": len(combined_rows),
        "detectors_requested": list(args.detectors),
        "checkpoint_every": max(1, args.checkpoint_every),
    }, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
