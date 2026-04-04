#!/usr/bin/env python3
"""Evaluate fill-bracket model outputs against HW-combined-shuffled subject+body."""

import argparse
import csv
import math
import re
import sys
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

from benchmark_llm_score_generated_outputs import (
    average,
    compute_bleu,
    compute_perplexity,
    compute_rouge1,
    compute_topic_coherence_for_corpus,
    iter_csv_rows,
    normalize_text,
    safe_float,
)


DEFAULT_BASE_DIR = Path(__file__).resolve().parent
DEFAULT_SOURCE_CSV = DEFAULT_BASE_DIR / "HW-combined-shuffled.csv"
DEFAULT_INPUT_DIR = DEFAULT_BASE_DIR / "Models-Output" / "fill-bracket-llama3_8b-full"
DEFAULT_OUTPUT_DIR = DEFAULT_BASE_DIR / "Evaluation results"
DEFAULT_INPUT_GLOB = "*.fill_bracket.csv"

SUBJECT_LINE_RE = re.compile(r'^\s*["\']?\s*subject\s*:\s*(.+?)\s*["\']?\s*$', re.IGNORECASE)
MARKDOWN_TITLE_RE = re.compile(r"^\s*(?:#{1,6}\s*)?\*\*(.+?)\*\*\s*$")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Score fill-bracket model outputs against HW-combined-shuffled.csv "
            "using Subject + Body as the comparison text."
        )
    )
    parser.add_argument(
        "--source-csv",
        default=str(DEFAULT_SOURCE_CSV),
        help="Path to HW-combined-shuffled.csv.",
    )
    parser.add_argument(
        "--input-dir",
        default=str(DEFAULT_INPUT_DIR),
        help="Directory containing the model output CSV files to score.",
    )
    parser.add_argument(
        "--output-dir",
        default=str(DEFAULT_OUTPUT_DIR),
        help="Directory where the scored CSV files will be written.",
    )
    parser.add_argument(
        "--glob",
        default=DEFAULT_INPUT_GLOB,
        help="Glob used to discover input CSV files inside --input-dir.",
    )
    parser.add_argument(
        "--print-every",
        type=int,
        default=25,
        help="Progress logging interval in completed rows.",
    )
    parser.add_argument(
        "--save-every",
        type=int,
        default=100,
        help="Checkpoint scored rows every N completed rows.",
    )
    parser.add_argument(
        "--disable-perplexity",
        action="store_true",
        help="Skip perplexity scoring.",
    )
    parser.add_argument(
        "--perplexity-model",
        default="gpt2",
        help="Local scorer model used for perplexity, e.g. gpt2.",
    )
    parser.add_argument(
        "--disable-topic-coherence",
        action="store_true",
        help="Skip topic coherence summary scoring.",
    )
    parser.add_argument(
        "--topic-count",
        type=int,
        default=5,
        help="Number of topics used for corpus-level topic coherence.",
    )
    parser.add_argument(
        "--topic-top-words",
        type=int,
        default=10,
        help="Top words per topic used for topic coherence.",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=0,
        help="Optional max rows per input file for smoke tests.",
    )
    return parser.parse_args()


def validate_args(args: argparse.Namespace) -> None:
    source_csv = Path(args.source_csv).resolve()
    input_dir = Path(args.input_dir).resolve()
    if not source_csv.exists():
        raise FileNotFoundError(f"Source CSV not found: {source_csv}")
    if not input_dir.exists():
        raise FileNotFoundError(f"Input directory not found: {input_dir}")
    if args.print_every <= 0:
        raise ValueError("--print-every must be greater than 0.")
    if args.save_every <= 0:
        raise ValueError("--save-every must be greater than 0.")
    if args.topic_count <= 0:
        raise ValueError("--topic-count must be greater than 0.")
    if args.topic_top_words <= 1:
        raise ValueError("--topic-top-words must be greater than 1.")
    if args.limit < 0:
        raise ValueError("--limit cannot be negative.")


def unwrap_wrapping_quotes(text: str) -> str:
    value = normalize_text(text)
    if len(value) >= 2 and value[0] == value[-1] and value[0] in {'"', "'"}:
        return normalize_text(value[1:-1])
    return value


def clean_cell(value: Any) -> str:
    if value is None:
        return ""
    text = str(value)
    if text.lower() == "nan":
        return ""
    return unwrap_wrapping_quotes(text)


def build_subject_body_text(subject: str, body: str) -> str:
    subject_clean = clean_cell(subject)
    body_clean = clean_cell(body)
    if subject_clean and body_clean:
        return f"Subject: {subject_clean}\n\n{body_clean}"
    if subject_clean:
        return f"Subject: {subject_clean}"
    return body_clean


def parse_subject_and_body_from_text(text: str) -> Tuple[str, str]:
    cleaned = clean_cell(text)
    if not cleaned:
        return "", ""

    lines = cleaned.splitlines()
    first_non_empty_index: Optional[int] = None
    for index, line in enumerate(lines):
        if line.strip():
            first_non_empty_index = index
            break

    if first_non_empty_index is None:
        return "", ""

    first_line = lines[first_non_empty_index]
    subject_match = SUBJECT_LINE_RE.match(first_line)
    if subject_match:
        body = "\n".join(lines[first_non_empty_index + 1 :]).lstrip()
        return clean_cell(subject_match.group(1)), clean_cell(body)

    markdown_match = MARKDOWN_TITLE_RE.match(first_line)
    if markdown_match and len(markdown_match.group(1).strip()) <= 180:
        body = "\n".join(lines[first_non_empty_index + 1 :]).lstrip()
        return clean_cell(markdown_match.group(1)), clean_cell(body)

    return "", cleaned


def derive_generated_parts(row: Dict[str, str]) -> Tuple[str, str, str]:
    generated_text = clean_cell(row.get("generated_text", ""))
    generated_subject = clean_cell(row.get("generated_subject", ""))
    generated_body = clean_cell(row.get("generated_body", ""))

    parsed_subject, parsed_body = parse_subject_and_body_from_text(generated_text)
    if not generated_subject and parsed_subject:
        generated_subject = parsed_subject
    if not generated_body and parsed_body:
        generated_body = parsed_body
    if not generated_body and generated_text:
        generated_body = generated_text

    combined = build_subject_body_text(generated_subject, generated_body)
    if not combined:
        combined = generated_text

    return generated_subject, generated_body, clean_cell(combined)


def load_source_lookup(path: Path) -> Dict[str, Dict[str, str]]:
    lookup: Dict[str, Dict[str, str]] = {}
    for csv_row_number, row in iter_csv_rows(path):
        source_id = clean_cell(row.get("source_id", ""))
        if not source_id:
            raise ValueError(f"Missing source_id in {path} row {csv_row_number}")
        if source_id in lookup:
            raise ValueError(f"Duplicate source_id {source_id} in {path}")
        lookup[source_id] = {
            "source_id": source_id,
            "row_number": clean_cell(row.get("row_number", "")),
            "label": clean_cell(row.get("label", "")),
            "source_subject": clean_cell(row.get("Subject", "")),
            "source_body": clean_cell(row.get("Body", "")),
            "data_source": clean_cell(row.get("data_source", "")),
            "source_split": clean_cell(row.get("source_split", "")),
        }
    if not lookup:
        raise ValueError(f"No source rows found in {path}")
    return lookup


def build_summary(scored_rows: Sequence[Dict[str, Any]], args: argparse.Namespace) -> Dict[str, str]:
    if not scored_rows:
        return {}

    generated_texts = [str(row["generated_text_subject_body"]) for row in scored_rows]
    reference_texts = [str(row["reference_text_subject_body"]) for row in scored_rows]

    if args.disable_topic_coherence:
        generated_topic_coherence = float("nan")
        reference_topic_coherence = float("nan")
    else:
        generated_topic_coherence = compute_topic_coherence_for_corpus(
            generated_texts, args.topic_count, args.topic_top_words
        )
        reference_topic_coherence = compute_topic_coherence_for_corpus(
            reference_texts, args.topic_count, args.topic_top_words
        )

    def numeric_value(row: Dict[str, Any], key: str) -> float:
        value = str(row.get(key, "")).strip()
        if not value:
            return float("nan")
        return float(value)

    return {
        "sample_count": str(len(scored_rows)),
        "unique_join_key_count": str(len({str(row["join_key"]) for row in scored_rows})),
        "mean_bleu": safe_float(average(numeric_value(row, "bleu") for row in scored_rows)),
        "mean_rouge1_precision": safe_float(
            average(numeric_value(row, "rouge1_precision") for row in scored_rows)
        ),
        "mean_rouge1_recall": safe_float(
            average(numeric_value(row, "rouge1_recall") for row in scored_rows)
        ),
        "mean_rouge1_f1": safe_float(
            average(numeric_value(row, "rouge1_f1") for row in scored_rows)
        ),
        "mean_generated_perplexity": safe_float(
            average(numeric_value(row, "generated_perplexity") for row in scored_rows)
        ),
        "mean_reference_perplexity": safe_float(
            average(numeric_value(row, "reference_perplexity") for row in scored_rows)
        ),
        "topic_coherence_generated": safe_float(generated_topic_coherence),
        "topic_coherence_reference": safe_float(reference_topic_coherence),
        "topic_coherence_gap": safe_float(
            generated_topic_coherence - reference_topic_coherence
            if not math.isnan(generated_topic_coherence)
            and not math.isnan(reference_topic_coherence)
            else float("nan")
        ),
    }


def output_name_for_input(input_path: Path) -> str:
    stem = input_path.name
    suffix = ".csv"
    if stem.endswith(".fill_bracket.csv"):
        stem = stem[: -len(".fill_bracket.csv")]
    elif stem.endswith(".csv"):
        stem = stem[: -len(".csv")]
    return f"{stem}.subject_body_scores{suffix}"


def write_output_csv(path: Path, scored_rows: Sequence[Dict[str, Any]], summary: Dict[str, str]) -> None:
    fieldnames = [
        "join_key",
        "input_row_number",
        "reference_row_number",
        "model_alias",
        "company",
        "family",
        "access_type",
        "provider",
        "label",
        "data_source",
        "source_split",
        "source_row_number",
        "source_subject",
        "source_body",
        "reference_text_subject_body",
        "generated_subject",
        "generated_body",
        "generated_text_subject_body",
        "prompt_text",
        "raw_generated_text",
        "bleu",
        "rouge1_precision",
        "rouge1_recall",
        "rouge1_f1",
        "generated_perplexity",
        "reference_perplexity",
        "created_at",
        "sample_count",
        "unique_join_key_count",
        "mean_bleu",
        "mean_rouge1_precision",
        "mean_rouge1_recall",
        "mean_rouge1_f1",
        "mean_generated_perplexity",
        "mean_reference_perplexity",
        "topic_coherence_generated",
        "topic_coherence_reference",
        "topic_coherence_gap",
    ]
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in scored_rows:
            out_row = dict(row)
            out_row.update(summary)
            writer.writerow(out_row)


def discover_input_files(input_dir: Path, pattern: str) -> List[Path]:
    files = sorted(path for path in input_dir.glob(pattern) if path.is_file())
    if not files:
        raise ValueError(f"No input files matched {pattern} in {input_dir}")
    return files


def score_input_file(
    input_path: Path,
    source_lookup: Dict[str, Dict[str, str]],
    output_dir: Path,
    args: argparse.Namespace,
    file_index: int,
    file_total: int,
) -> Dict[str, str]:
    rows = list(iter_csv_rows(input_path))
    if args.limit:
        rows = rows[: args.limit]
    total = len(rows)
    if total == 0:
        raise ValueError(f"No rows found in {input_path}")

    output_path = output_dir / output_name_for_input(input_path)
    scored_rows: List[Dict[str, Any]] = []

    for index, (_, row) in enumerate(rows, start=1):
        join_key = clean_cell(row.get("join_key", ""))
        if not join_key:
            raise ValueError(f"Missing join_key in {input_path} row {index + 1}")
        if join_key not in source_lookup:
            raise KeyError(f"join_key {join_key} not found in source CSV")

        source_row = source_lookup[join_key]
        reference_text = build_subject_body_text(
            source_row["source_subject"], source_row["source_body"]
        )
        generated_subject, generated_body, generated_text = derive_generated_parts(row)

        bleu = compute_bleu(reference_text, generated_text)
        rouge_precision, rouge_recall, rouge_f1 = compute_rouge1(reference_text, generated_text)

        if args.disable_perplexity:
            generated_perplexity = float("nan")
            reference_perplexity = float("nan")
        else:
            generated_perplexity = compute_perplexity(generated_text, args.perplexity_model)
            reference_perplexity = compute_perplexity(reference_text, args.perplexity_model)

        scored_rows.append(
            {
                "join_key": join_key,
                "input_row_number": clean_cell(row.get("input_row_number", "")),
                "reference_row_number": clean_cell(row.get("reference_row_number", "")),
                "model_alias": clean_cell(row.get("model_alias", "")),
                "company": clean_cell(row.get("company", "")),
                "family": clean_cell(row.get("family", "")),
                "access_type": clean_cell(row.get("access_type", "")),
                "provider": clean_cell(row.get("provider", "")),
                "label": source_row["label"],
                "data_source": source_row["data_source"],
                "source_split": source_row["source_split"],
                "source_row_number": source_row["row_number"],
                "source_subject": source_row["source_subject"],
                "source_body": source_row["source_body"],
                "reference_text_subject_body": reference_text,
                "generated_subject": generated_subject,
                "generated_body": generated_body,
                "generated_text_subject_body": generated_text,
                "prompt_text": clean_cell(row.get("prompt_text", "")),
                "raw_generated_text": clean_cell(row.get("generated_text", "")),
                "bleu": safe_float(bleu),
                "rouge1_precision": safe_float(rouge_precision),
                "rouge1_recall": safe_float(rouge_recall),
                "rouge1_f1": safe_float(rouge_f1),
                "generated_perplexity": safe_float(generated_perplexity),
                "reference_perplexity": safe_float(reference_perplexity),
                "created_at": clean_cell(row.get("created_at", "")),
            }
        )

        if index % args.save_every == 0:
            write_output_csv(output_path, scored_rows, build_summary(scored_rows, args))

        if index % args.print_every == 0 or index == total:
            model_alias = clean_cell(row.get("model_alias", "")) or input_path.stem
            print(
                (
                    f"[progress] file {file_index}/{file_total} "
                    f"{input_path.name}: scored {index}/{total} rows for {model_alias}"
                ),
                file=sys.stderr,
                flush=True,
            )

    summary = build_summary(scored_rows, args)
    write_output_csv(output_path, scored_rows, summary)
    print(f"[done] wrote {output_path}", file=sys.stderr, flush=True)

    result = {
        "input_file": input_path.name,
        "output_file": output_path.name,
    }
    result.update(summary)
    return result


def write_summary_csv(path: Path, rows: Sequence[Dict[str, str]]) -> None:
    if not rows:
        return
    fieldnames = list(rows[0].keys())
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def main() -> int:
    args = parse_args()
    validate_args(args)

    source_csv = Path(args.source_csv).resolve()
    input_dir = Path(args.input_dir).resolve()
    output_dir = Path(args.output_dir).resolve()

    print(f"[start] loading source rows from {source_csv}", file=sys.stderr, flush=True)
    source_lookup = load_source_lookup(source_csv)
    input_files = discover_input_files(input_dir, args.glob)
    print(
        f"[start] found {len(input_files)} input file(s) in {input_dir}",
        file=sys.stderr,
        flush=True,
    )

    summary_rows: List[Dict[str, str]] = []
    for file_index, input_path in enumerate(input_files, start=1):
        print(f"[start] scoring {input_path.name}", file=sys.stderr, flush=True)
        summary_rows.append(
            score_input_file(
                input_path=input_path,
                source_lookup=source_lookup,
                output_dir=output_dir,
                args=args,
                file_index=file_index,
                file_total=len(input_files),
            )
        )

    summary_path = output_dir / "evaluation_summary.csv"
    write_summary_csv(summary_path, summary_rows)
    print(f"[done] wrote {summary_path}", file=sys.stderr, flush=True)
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception as exc:
        print(f"[error] {exc}", file=sys.stderr, flush=True)
        raise SystemExit(1)
