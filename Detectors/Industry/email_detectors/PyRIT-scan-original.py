#!/usr/bin/env python3
"""Scan a piece of text with PyRIT's official Azure Content Safety scorer."""

from __future__ import annotations

import argparse
import asyncio
import csv
import json
import os
import sys
import time
from pathlib import Path
from typing import Any


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Scan input text with PyRIT using the official AzureContentFilterScorer pattern. "
            "You can pass text with --text or pipe it in via stdin."
        )
    )
    parser.add_argument("--text", help="Text to scan. If omitted, the script reads from stdin.")
    parser.add_argument("--input-csv", type=Path, help="Input CSV file for batch scanning.")
    parser.add_argument(
        "--output-csv",
        type=Path,
        help="Output CSV file path for batch scanning. Defaults to <input>.scanned.csv.",
    )
    parser.add_argument(
        "--text-column",
        default="text",
        help="CSV column name containing the text to scan. Default: text",
    )
    parser.add_argument(
        "--endpoint",
        default=os.environ.get("AZURE_CONTENT_SAFETY_API_ENDPOINT"),
        help=(
            "Azure Content Safety endpoint. Defaults to the "
            "AZURE_CONTENT_SAFETY_API_ENDPOINT environment variable."
        ),
    )
    parser.add_argument(
        "--api-key",
        default=os.environ.get("AZURE_CONTENT_SAFETY_API_KEY"),
        help=(
            "Azure Content Safety API key. Defaults to the AZURE_CONTENT_SAFETY_API_KEY "
            "environment variable. If omitted, the script follows the official PyRIT "
            "pattern and uses Entra ID via az login."
        ),
    )
    parser.add_argument(
        "--categories",
        nargs="+",
        help=(
            "Optional harm categories to query, for example: hate self_harm sexual violence. "
            "If omitted, PyRIT uses all categories."
        ),
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
    return input_csv.with_name(f"{input_csv.stem}.scanned.csv")


def load_text_categories(category_names: list[str] | None):
    if not category_names:
        return None

    from azure.ai.contentsafety.models import TextCategory

    parsed_categories = []
    for name in category_names:
        normalized = name.strip().replace("-", "_").upper()
        try:
            parsed_categories.append(TextCategory[normalized])
        except KeyError as exc:
            valid = ", ".join(member.name.lower() for member in TextCategory)
            raise SystemExit(f"Unsupported category '{name}'. Valid values: {valid}") from exc

    return parsed_categories


def score_to_output(score: Any) -> dict[str, Any]:
    return {
        "score_value": score.score_value,
        "normalized_score": score.get_value(),
        "score_value_description": score.score_value_description,
        "score_type": score.score_type,
        "score_category": score.score_category,
        "score_rationale": score.score_rationale,
        "score_metadata": score.score_metadata,
        "objective": getattr(score, "objective", None),
    }


async def build_pyrit_components(endpoint: str | None, api_key: str | None, categories: list[str] | None) -> tuple[Any, Any]:
    try:
        from pyrit.auth import get_azure_token_provider
        from pyrit.memory import CentralMemory
        from pyrit.score.float_scale.azure_content_filter_scorer import AzureContentFilterScorer
        from pyrit.setup import IN_MEMORY, initialize_pyrit_async
    except ImportError as exc:
        raise SystemExit(
            "PyRIT is not installed in this environment. Install the official package first, "
            "for example: pip install pyrit"
        ) from exc

    if not endpoint:
        raise SystemExit(
            "Missing Azure Content Safety endpoint. Set AZURE_CONTENT_SAFETY_API_ENDPOINT "
            "or pass --endpoint."
        )

    await initialize_pyrit_async(memory_db_type=IN_MEMORY)

    harm_categories = load_text_categories(categories)

    # This mirrors the official scorer example in PyRIT's scoring docs.
    scorer = AzureContentFilterScorer(
        endpoint=endpoint,
        api_key=api_key or get_azure_token_provider("https://cognitiveservices.azure.com/.default"),
        harm_categories=harm_categories,
    )
    memory = CentralMemory.get_memory_instance()
    return scorer, memory


async def scan_text_with_components(text: str, scorer: Any, memory: Any) -> list[dict[str, Any]]:
    from pyrit.models import Message, MessagePiece

    response = Message(
        message_pieces=[
            MessagePiece(
                role="assistant",
                original_value_data_type="text",
                original_value=text,
            )
        ]
    )

    memory.add_message_to_memory(request=response)

    scores = await scorer.score_async(response)
    return [score_to_output(score) for score in scores]


async def run_single_scan_async(
    text: str,
    endpoint: str | None,
    api_key: str | None,
    categories: list[str] | None,
) -> dict[str, Any]:
    scorer, memory = await build_pyrit_components(endpoint=endpoint, api_key=api_key, categories=categories)
    started_at = time.perf_counter()
    results = await scan_text_with_components(text=text, scorer=scorer, memory=memory)
    elapsed_seconds = time.perf_counter() - started_at

    return {
        "input_text": text,
        "elapsed_seconds": round(elapsed_seconds, 6),
        "results": results,
    }


async def run_csv_scan_async(
    input_csv: Path,
    output_csv: Path,
    text_column: str,
    endpoint: str | None,
    api_key: str | None,
    categories: list[str] | None,
) -> dict[str, Any]:
    if not input_csv.exists():
        raise SystemExit(f"Input CSV not found: {input_csv}")

    scorer, memory = await build_pyrit_components(endpoint=endpoint, api_key=api_key, categories=categories)

    with input_csv.open("r", encoding="utf-8-sig", newline="") as infile:
        reader = csv.DictReader(infile)
        if not reader.fieldnames:
            raise SystemExit(f"CSV has no header row: {input_csv}")
        if text_column not in reader.fieldnames:
            available = ", ".join(reader.fieldnames)
            raise SystemExit(
                f"CSV column '{text_column}' not found in {input_csv}. Available columns: {available}"
            )

        results_column = "pyrit_scan_results_json"
        elapsed_column = "pyrit_scan_elapsed_seconds"
        error_column = "pyrit_scan_error"
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
                text = (row.get(text_column) or "").strip()
                row[results_column] = ""
                row[elapsed_column] = ""
                row[error_column] = ""

                if not text:
                    row[error_column] = f"Empty text in column '{text_column}'"
                    writer.writerow(row)
                    continue

                started_at = time.perf_counter()
                try:
                    results = await scan_text_with_components(text=text, scorer=scorer, memory=memory)
                    elapsed_seconds = time.perf_counter() - started_at
                    row[results_column] = json.dumps(results, ensure_ascii=False)
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


def main() -> int:
    args = parse_args()

    if args.input_csv:
        summary = asyncio.run(
            run_csv_scan_async(
                input_csv=args.input_csv,
                output_csv=resolve_output_csv_path(args.input_csv, args.output_csv),
                text_column=args.text_column,
                endpoint=args.endpoint,
                api_key=args.api_key,
                categories=args.categories,
            )
        )
        print(json.dumps(summary, ensure_ascii=False, indent=2))
        return 0

    text = read_text(args.text)
    result = asyncio.run(
        run_single_scan_async(
            text=text,
            endpoint=args.endpoint,
            api_key=args.api_key,
            categories=args.categories,
        )
    )
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
