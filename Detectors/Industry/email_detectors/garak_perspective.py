#!/usr/bin/env python3
"""Score text with the Perspective API SPAM attribute."""

from __future__ import annotations

import argparse
import csv
import json
import os
import sys
import time
from pathlib import Path
from typing import Any

import requests


DEFAULT_API_URL = "https://commentanalyzer.googleapis.com/v1alpha1/comments:analyze"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Score text with Perspective API's SPAM attribute. "
            "You can pass text with --text, pipe it in via stdin, or scan a CSV file."
        )
    )
    parser.add_argument("--text", help="Text to scan. If omitted, the script reads from stdin.")
    parser.add_argument("--input-csv", type=Path, help="Input CSV file for batch scanning.")
    parser.add_argument(
        "--output-csv",
        type=Path,
        help="Output CSV file path for batch scanning. Defaults to <input>.perspective_spam.csv.",
    )
    parser.add_argument(
        "--text-column",
        default="text",
        help="CSV column name containing the text to scan. Default: text",
    )
    parser.add_argument(
        "--api-key",
        default=os.getenv("PERSPECTIVE_API_KEY"),
        help="Perspective API key. Defaults to the PERSPECTIVE_API_KEY environment variable.",
    )
    parser.add_argument(
        "--language",
        default="en",
        help="Language code sent to Perspective API. Default: en",
    )
    parser.add_argument(
        "--threshold",
        type=float,
        default=0.5,
        help="Threshold for spam verdict. Default: 0.5",
    )
    return parser.parse_args()


def read_text(cli_text: str | None) -> str:
    if cli_text:
        return cli_text

    if not sys.stdin.isatty():
        piped_text = sys.stdin.read().strip()
        if piped_text:
            return piped_text

    raise SystemExit("No text provided. Use --text or pipe text into the script.")


def resolve_output_csv_path(input_csv: Path, output_csv: Path | None) -> Path:
    if output_csv:
        return output_csv
    return input_csv.with_name(f"{input_csv.stem}.perspective_spam.csv")


def build_headers() -> dict[str, str]:
    return {"Content-Type": "application/json"}


def ensure_api_key(api_key: str | None) -> str:
    if not api_key:
        raise SystemExit(
            "Missing Perspective API key. Set PERSPECTIVE_API_KEY or pass --api-key."
        )
    return api_key


def analyze_text(*, text: str, api_key: str, language: str) -> dict[str, Any]:
    if text == "":
        return {
            "attribute": "SPAM",
            "summary_score": 0.0,
            "span_scores": [],
            "language": language,
            "raw_response": {},
        }

    payload = {
        "comment": {"text": text},
        "requestedAttributes": {"SPAM": {}},
        "languages": [language],
    }

    response = requests.post(
        f"{DEFAULT_API_URL}?key={api_key}",
        headers=build_headers(),
        json=payload,
        timeout=60,
    )

    try:
        body: Any = response.json()
    except ValueError as exc:
        raise SystemExit(f"Perspective API returned non-JSON output: {response.text}") from exc

    if response.status_code >= 400:
        error = body.get("error", {}) if isinstance(body, dict) else {}
        code = error.get("code", response.status_code)
        message = error.get("message", response.text)
        raise SystemExit(f"Perspective API request failed [{code}]: {message}")

    attribute_scores = body.get("attributeScores", {}).get("SPAM", {})
    summary_score = attribute_scores.get("summaryScore", {}).get("value")
    span_scores = attribute_scores.get("spanScores", [])

    if summary_score is None:
        raise SystemExit("Perspective API response did not include a SPAM summary score.")

    return {
        "attribute": "SPAM",
        "summary_score": summary_score,
        "span_scores": span_scores,
        "language": language,
        "raw_response": body,
    }


def build_detection_summary(*, text: str, api_key: str, language: str, threshold: float) -> dict[str, Any]:
    analysis_result = analyze_text(text=text, api_key=api_key, language=language)
    spam_score = float(analysis_result["summary_score"])

    return {
        "decision": {
            "attribute": "SPAM",
            "score": spam_score,
            "threshold": threshold,
            "is_spam": spam_score >= threshold,
        },
        "perspective_result": analysis_result,
    }


def detect_single_text(text: str, api_key: str, language: str, threshold: float) -> dict[str, Any]:
    started_at = time.perf_counter()
    results = build_detection_summary(
        text=text,
        api_key=api_key,
        language=language,
        threshold=threshold,
    )
    elapsed_seconds = time.perf_counter() - started_at

    return {
        "input_text": text,
        "elapsed_seconds": round(elapsed_seconds, 6),
        "results": results,
    }


def scan_csv(
    *,
    input_csv: Path,
    output_csv: Path,
    text_column: str,
    api_key: str,
    language: str,
    threshold: float,
) -> dict[str, Any]:
    if not input_csv.exists():
        raise SystemExit(f"Input CSV not found: {input_csv}")

    with input_csv.open("r", encoding="utf-8-sig", newline="") as infile:
        reader = csv.DictReader(infile)
        if not reader.fieldnames:
            raise SystemExit(f"CSV has no header row: {input_csv}")
        if text_column not in reader.fieldnames:
            available = ", ".join(reader.fieldnames)
            raise SystemExit(
                f"CSV column '{text_column}' not found in {input_csv}. Available columns: {available}"
            )

        results_column = "perspective_spam_results_json"
        elapsed_column = "perspective_spam_elapsed_seconds"
        error_column = "perspective_spam_error"
        score_column = "perspective_spam_score"
        verdict_column = "perspective_spam_is_spam"
        fieldnames = list(reader.fieldnames)
        for extra_column in [
            results_column,
            elapsed_column,
            error_column,
            score_column,
            verdict_column,
        ]:
            if extra_column not in fieldnames:
                fieldnames.append(extra_column)

        row_count = 0
        success_count = 0

        with output_csv.open("w", encoding="utf-8", newline="") as outfile:
            writer = csv.DictWriter(outfile, fieldnames=fieldnames)
            writer.writeheader()

            for row in reader:
                row_count += 1
                text = (row.get(text_column) or "").strip()
                row[results_column] = ""
                row[elapsed_column] = ""
                row[error_column] = ""
                row[score_column] = ""
                row[verdict_column] = ""

                if not text:
                    row[error_column] = f"Empty text in column '{text_column}'"
                    writer.writerow(row)
                    continue

                started_at = time.perf_counter()
                try:
                    results = build_detection_summary(
                        text=text,
                        api_key=api_key,
                        language=language,
                        threshold=threshold,
                    )
                    elapsed_seconds = time.perf_counter() - started_at
                    row[results_column] = json.dumps(results, ensure_ascii=False)
                    row[elapsed_column] = f"{elapsed_seconds:.6f}"
                    row[score_column] = str(results["decision"]["score"])
                    row[verdict_column] = str(results["decision"]["is_spam"])
                    success_count += 1
                except Exception as exc:
                    elapsed_seconds = time.perf_counter() - started_at
                    row[elapsed_column] = f"{elapsed_seconds:.6f}"
                    row[error_column] = str(exc)

                writer.writerow(row)

    return {
        "input_csv": str(input_csv),
        "output_csv": str(output_csv),
        "rows_total": row_count,
        "rows_scanned": success_count,
        "rows_failed": row_count - success_count,
    }


def main() -> int:
    args = parse_args()
    api_key = ensure_api_key(args.api_key)

    if args.input_csv:
        summary = scan_csv(
            input_csv=args.input_csv,
            output_csv=resolve_output_csv_path(args.input_csv, args.output_csv),
            text_column=args.text_column,
            api_key=api_key,
            language=args.language,
            threshold=args.threshold,
        )
        print(json.dumps(summary, ensure_ascii=False, indent=2))
        return 0

    text = read_text(args.text)
    result = detect_single_text(
        text=text,
        api_key=api_key,
        language=args.language,
        threshold=args.threshold,
    )
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
