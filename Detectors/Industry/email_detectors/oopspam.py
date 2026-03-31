#!/usr/bin/env python3
"""Batch runner for OOPSpam phishing/spam scoring on CSV email datasets."""

from __future__ import annotations

import argparse
import csv
import json
import os
import time
from datetime import datetime
from pathlib import Path
from typing import Any

import requests


SCRIPT_DIR = Path(__file__).resolve().parent
MODEL_NAME = "oopspam"
DEFAULT_INPUT_CSV = (
    Path("/scratch3/che489/FC-W2-SoK/ssc_LLM_phishing_survey")
    / "Datasets/sublist/S5-Personalization for Credibility/LLM-P.csv"
)
DEFAULT_API_URL = "https://api.oopspam.com/v1/spamdetection"
DEFAULT_TIMEOUT_SECONDS = 60
DEFAULT_THRESHOLD = 3


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
        description="Run OOPSpam over a CSV email dataset and export a compact summary."
    )
    parser.add_argument("--input-csv", type=Path, default=DEFAULT_INPUT_CSV)
    parser.add_argument("--subject-column", default="Subject")
    parser.add_argument("--body-column", default="Body")
    parser.add_argument("--sample-size", type=int, default=20)
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--api-key", default=os.environ.get("OOPSPAM_API_KEY", ""))
    parser.add_argument("--api-url", default=os.environ.get("OOPSPAM_API_URL", DEFAULT_API_URL))
    parser.add_argument("--threshold", type=int, default=DEFAULT_THRESHOLD)
    parser.add_argument("--sender-ip", default=os.environ.get("OOPSPAM_SENDER_IP", "127.0.0.1"))
    parser.add_argument("--email", default=os.environ.get("OOPSPAM_EMAIL", "testing@example.com"))
    parser.add_argument("--source", default=os.environ.get("OOPSPAM_SOURCE", "benchmark.local"))
    parser.add_argument("--timeout-seconds", type=int, default=DEFAULT_TIMEOUT_SECONDS)
    parser.add_argument("--sleep-seconds", type=float, default=0.0)
    parser.add_argument("--block-temp-email", action="store_true")
    parser.add_argument("--check-for-length", action="store_true", default=True)
    parser.add_argument("--disable-check-for-length", dest="check_for_length", action="store_false")
    parser.add_argument("--log-it", action="store_true")
    return parser.parse_args()


def normalize_text(value: Any) -> str:
    return (value or "").replace("\r\n", "\n").replace("\r", "\n").strip()


def build_content(subject: str, body: str) -> str:
    subject = normalize_text(subject)
    body = normalize_text(body)
    if subject and body:
        return f"{subject}\n\n{body}"
    return subject or body


def write_manifest(path: Path, payload: dict[str, Any]) -> None:
    with path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, ensure_ascii=False)


def normalize_prediction(score: Any, threshold: int) -> int | None:
    try:
        return int(float(score) >= float(threshold))
    except Exception:
        return None


def parse_rate_limit_headers(headers: Any) -> tuple[str, str]:
    limit = headers.get("X-RateLimit-Limit", headers.get("x-ratelimit-requests-limit", ""))
    remaining = headers.get(
        "X-RateLimit-Remaining",
        headers.get("x-ratelimit-requests-remaining", ""),
    )
    return str(limit or ""), str(remaining or "")


def post_oopspam(
    *,
    api_url: str,
    api_key: str,
    payload: dict[str, Any],
    timeout_seconds: int,
) -> tuple[dict[str, Any] | None, str, str, str]:
    headers = {
        "Content-Type": "application/json",
        "X-Api-Key": api_key,
    }
    try:
        response = requests.post(api_url, json=payload, headers=headers, timeout=timeout_seconds)
    except requests.RequestException as exc:
        return None, "", "", str(exc)

    limit, remaining = parse_rate_limit_headers(response.headers)

    try:
        response_json = response.json()
    except ValueError:
        response_json = None

    if response.status_code >= 400:
        if isinstance(response_json, dict):
            message = response_json.get("message") or response_json.get("error") or response.text
        else:
            message = response.text
        return None, limit, remaining, f"HTTP {response.status_code}: {message}"

    if not isinstance(response_json, dict):
        return None, limit, remaining, "Non-JSON response from OOPSpam"

    return response_json, limit, remaining, ""


def main() -> int:
    args = parse_args()
    if not args.input_csv.exists():
        raise SystemExit(f"Input CSV not found: {args.input_csv}")
    if not args.api_key:
        raise SystemExit("Missing OOPSpam API key. Set OOPSPAM_API_KEY or pass --api-key.")

    output_dir = build_output_dir(args)
    summary_dir = output_dir / "summary"
    output_dir.mkdir(parents=True, exist_ok=True)
    summary_dir.mkdir(parents=True, exist_ok=True)

    output_csv = summary_dir / "oopspam_summary.csv"
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
        "oopspam_score",
        "oopspam_prediction",
        "oopspam_is_content_spam",
        "oopspam_number_of_spam_words",
        "oopspam_spam_words",
        "oopspam_is_email_blocked",
        "oopspam_is_ip_blocked",
        "oopspam_is_content_too_short",
        "oopspam_lang_match",
        "oopspam_country_match",
        "oopspam_rate_limit_limit",
        "oopspam_rate_limit_remaining",
        "oopspam_error",
        "oopspam_raw_json",
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
            content = build_content(subject, body)
            payload = {
                "content": content,
                "senderIP": args.sender_ip,
                "email": args.email,
                "source": args.source,
                "checkForLength": args.check_for_length,
                "blockTempEmail": args.block_temp_email,
                "logIt": args.log_it,
            }

            response_json, limit, remaining, error = post_oopspam(
                api_url=args.api_url,
                api_key=args.api_key,
                payload=payload,
                timeout_seconds=args.timeout_seconds,
            )
            details = response_json.get("Details", {}) if isinstance(response_json, dict) else {}
            score = response_json.get("Score", "") if isinstance(response_json, dict) else ""

            row = {
                "row_number": index,
                "source_path": str(args.input_csv),
                "source_row_number": index,
                "dataset_name": dataset_name,
                "subject": subject,
                "label": record.get("label", ""),
                "data_source": record.get("data_source", ""),
                "content_length": len(content),
                "oopspam_score": score,
                "oopspam_prediction": normalize_prediction(score, args.threshold),
                "oopspam_is_content_spam": details.get("isContentSpam", ""),
                "oopspam_number_of_spam_words": details.get("numberOfSpamWords", ""),
                "oopspam_spam_words": json.dumps(details.get("spamWords", []), ensure_ascii=False),
                "oopspam_is_email_blocked": details.get("isEmailBlocked", ""),
                "oopspam_is_ip_blocked": details.get("isIPBlocked", ""),
                "oopspam_is_content_too_short": details.get("isContentTooShort", ""),
                "oopspam_lang_match": details.get("langMatch", ""),
                "oopspam_country_match": details.get("countryMatch", ""),
                "oopspam_rate_limit_limit": limit,
                "oopspam_rate_limit_remaining": remaining,
                "oopspam_error": error,
                "oopspam_raw_json": json.dumps(response_json, ensure_ascii=False) if response_json else "",
            }
            rows.append(row)
            status = f"score={score}" if not error else error
            print(f"[{index}/?] row {index} -> {status}")

            if args.sleep_seconds > 0:
                time.sleep(args.sleep_seconds)

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
            "api_url": args.api_url,
            "threshold": args.threshold,
            "sender_ip": args.sender_ip,
            "email": args.email,
            "source": args.source,
            "saved_artifacts": [
                str(manifest_path),
                str(output_csv),
            ],
        },
    )

    print(
        json.dumps(
            {
                "timestamp": datetime.utcnow().isoformat() + "Z",
                "output_csv": str(output_csv),
                "rows": len(rows),
                "threshold": args.threshold,
            },
            ensure_ascii=False,
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
