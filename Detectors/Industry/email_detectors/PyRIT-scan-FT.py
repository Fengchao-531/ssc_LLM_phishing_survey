#!/usr/bin/env python3
"""Train and use Azure Content Safety custom categories with scan-like outputs."""

from __future__ import annotations

import argparse
import csv
import json
import os
import sys
import time
from pathlib import Path
from typing import Any
from urllib.parse import urlencode

import requests


class DetectionError(Exception):
    def __init__(self, code: str, message: str) -> None:
        self.code = code
        self.message = message

    def __repr__(self) -> str:
        return f"DetectionError(code={self.code}, message={self.message})"


class CustomCategorySafety:
    """Wrapper around Azure Content Safety custom category APIs."""

    def __init__(self, endpoint: str, subscription_key: str, api_version: str) -> None:
        self.endpoint = endpoint.rstrip("/")
        self.subscription_key = subscription_key
        self.api_version = api_version

    def build_headers(self) -> dict[str, str]:
        return {
            "Ocp-Apim-Subscription-Key": self.subscription_key,
            "Content-Type": "application/json",
        }

    def request(
        self,
        method: str,
        path: str,
        payload: dict[str, Any] | None = None,
        extra_query: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        url = f"{self.endpoint}{path}?api-version={self.api_version}"
        if extra_query:
            url = f"{url}&{urlencode(extra_query)}"
        response = requests.request(method, url, headers=self.build_headers(), json=payload, timeout=60)

        try:
            body: Any = response.json()
        except ValueError:
            body = {"raw_text": response.text}

        if response.status_code >= 400:
            error = body.get("error", {}) if isinstance(body, dict) else {}
            raise DetectionError(
                str(error.get("code", response.status_code)),
                str(error.get("message", response.text)),
            )

        return {
            "status_code": response.status_code,
            "headers": dict(response.headers),
            "body": body,
        }

    def create_category_version(self, category_name: str, definition: str, sample_blob_url: str) -> dict[str, Any]:
        return self.request(
            "PUT",
            f"/contentsafety/text/categories/{category_name}",
            {
                "categoryName": category_name,
                "definition": definition,
                "sampleBlobUrl": sample_blob_url,
            },
        )

    def start_build(self, category_name: str, version: int) -> dict[str, Any]:
        return self.request(
            "POST",
            f"/contentsafety/text/categories/{category_name}:build",
            None,
            {"version": version},
        ) | {"requested_version": version}

    def get_build_status(self, operation_id: str) -> dict[str, Any]:
        return self.request(
            "GET",
            f"/contentsafety/text/categories/operations/{operation_id}",
            None,
        )

    def analyze_custom_category(self, text: str, category_name: str, version: int | None) -> dict[str, Any]:
        payload: dict[str, Any] = {
            "text": text,
            "categoryName": category_name,
        }
        if version is not None:
            payload["version"] = version
        return self.request("POST", "/contentsafety/text:analyzeCustomCategory", payload)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Train and use Azure Content Safety custom categories. "
            "This FT variant supports category creation, build, build-status checks, "
            "and single/CSV inference."
        )
    )
    parser.add_argument(
        "--endpoint",
        default=os.getenv("CONTENT_SAFETY_ENDPOINT") or os.getenv("AZURE_CONTENT_SAFETY_API_ENDPOINT"),
        help="Content Safety endpoint. Defaults to CONTENT_SAFETY_ENDPOINT or AZURE_CONTENT_SAFETY_API_ENDPOINT.",
    )
    parser.add_argument(
        "--api-key",
        default=os.getenv("CONTENT_SAFETY_KEY") or os.getenv("AZURE_CONTENT_SAFETY_API_KEY"),
        help="Content Safety API key. Defaults to CONTENT_SAFETY_KEY or AZURE_CONTENT_SAFETY_API_KEY.",
    )
    parser.add_argument(
        "--api-version",
        default="2024-09-15-preview",
        help="Custom category API version. Default: 2024-09-15-preview",
    )

    subparsers = parser.add_subparsers(dest="command", required=True)

    train_parser = subparsers.add_parser("train", help="Create a category version and start training.")
    train_parser.add_argument("--category-name", required=True, help="Custom category name.")
    train_parser.add_argument("--definition", required=True, help="Definition for the custom category.")
    train_parser.add_argument("--sample-blob-url", required=True, help="Blob URL to the training JSONL file.")
    train_parser.add_argument("--skip-build", action="store_true", help="Only create the category version.")

    status_parser = subparsers.add_parser("status", help="Check build status using an operation ID.")
    status_parser.add_argument("--operation-id", required=True, help="Operation ID returned by the build request.")

    detect_parser = subparsers.add_parser("detect", help="Use the trained custom category detector.")
    detect_parser.add_argument("--category-name", required=True, help="Custom category name.")
    detect_parser.add_argument(
        "--version",
        type=int,
        default=None,
        help="Category version. If omitted, the service uses the latest version.",
    )
    detect_parser.add_argument("--text", help="Text to scan. If omitted, the script reads from stdin.")
    detect_parser.add_argument("--input-csv", type=Path, help="Input CSV file for batch scanning.")
    detect_parser.add_argument(
        "--output-csv",
        type=Path,
        help="Output CSV file path for batch scanning. Defaults to <input>.ft_detection.csv.",
    )
    detect_parser.add_argument(
        "--text-column",
        default="text",
        help="CSV column name containing the text to scan. Default: text",
    )

    return parser.parse_args()


def build_client(args: argparse.Namespace) -> CustomCategorySafety:
    if not args.endpoint:
        raise SystemExit(
            "Missing Content Safety endpoint. Set CONTENT_SAFETY_ENDPOINT or pass --endpoint."
        )
    if not args.api_key:
        raise SystemExit(
            "Missing Content Safety API key. Set CONTENT_SAFETY_KEY or pass --api-key."
        )
    return CustomCategorySafety(args.endpoint, args.api_key, args.api_version)


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
    return input_csv.with_name(f"{input_csv.stem}.ft_detection.csv")


def run_train(args: argparse.Namespace, client: CustomCategorySafety) -> dict[str, Any]:
    create_result = client.create_category_version(args.category_name, args.definition, args.sample_blob_url)
    body = create_result.get("body", {})
    version = body.get("version")

    output: dict[str, Any] = {
        "category_name": args.category_name,
        "create_category_result": create_result,
        "build_result": None,
    }

    if args.skip_build:
        return output

    if version is None:
        raise SystemExit("Category version was not returned by the create-category API response.")

    build_result = client.start_build(args.category_name, int(version))
    output["build_result"] = build_result
    return output


def run_status(args: argparse.Namespace, client: CustomCategorySafety) -> dict[str, Any]:
    return {
        "operation_id": args.operation_id,
        "status_result": client.get_build_status(args.operation_id),
    }


def detect_single_text(text: str, category_name: str, version: int | None, client: CustomCategorySafety) -> dict[str, Any]:
    started_at = time.perf_counter()
    results = client.analyze_custom_category(text, category_name, version)
    elapsed_seconds = time.perf_counter() - started_at
    return {
        "input_text": text,
        "elapsed_seconds": round(elapsed_seconds, 6),
        "results": {
            "category_name": category_name,
            "version": version,
            "analysis_result": results,
        },
    }


def detect_csv(args: argparse.Namespace, client: CustomCategorySafety) -> dict[str, Any]:
    input_csv = args.input_csv
    output_csv = resolve_output_csv_path(args.input_csv, args.output_csv)
    if not input_csv.exists():
        raise SystemExit(f"Input CSV not found: {input_csv}")

    with input_csv.open("r", encoding="utf-8-sig", newline="") as infile:
        reader = csv.DictReader(infile)
        if not reader.fieldnames:
            raise SystemExit(f"CSV has no header row: {input_csv}")
        if args.text_column not in reader.fieldnames:
            available = ", ".join(reader.fieldnames)
            raise SystemExit(
                f"CSV column '{args.text_column}' not found in {input_csv}. Available columns: {available}"
            )

        results_column = "ft_detection_results_json"
        elapsed_column = "ft_detection_elapsed_seconds"
        error_column = "ft_detection_error"
        fieldnames = list(reader.fieldnames)
        for extra_column in [results_column, elapsed_column, error_column]:
            if extra_column not in fieldnames:
                fieldnames.append(extra_column)

        row_count = 0
        success_count = 0

        with output_csv.open("w", encoding="utf-8", newline="") as outfile:
            writer = csv.DictWriter(outfile, fieldnames=fieldnames)
            writer.writeheader()

            for row in reader:
                row_count += 1
                text = (row.get(args.text_column) or "").strip()
                row[results_column] = ""
                row[elapsed_column] = ""
                row[error_column] = ""

                if not text:
                    row[error_column] = f"Empty text in column '{args.text_column}'"
                    writer.writerow(row)
                    continue

                started_at = time.perf_counter()
                try:
                    results = client.analyze_custom_category(text, args.category_name, args.version)
                    elapsed_seconds = time.perf_counter() - started_at
                    row[results_column] = json.dumps(
                        {
                            "category_name": args.category_name,
                            "version": args.version,
                            "analysis_result": results,
                        },
                        ensure_ascii=False,
                    )
                    row[elapsed_column] = f"{elapsed_seconds:.6f}"
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


def run_detect(args: argparse.Namespace, client: CustomCategorySafety) -> dict[str, Any]:
    if args.input_csv:
        return detect_csv(args, client)
    text = read_text(args.text)
    return detect_single_text(text, args.category_name, args.version, client)


def main() -> int:
    args = parse_args()
    client = build_client(args)

    if args.command == "train":
        result = run_train(args, client)
    elif args.command == "status":
        result = run_status(args, client)
    elif args.command == "detect":
        result = run_detect(args, client)
    else:
        raise SystemExit(f"Unsupported command: {args.command}")

    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
