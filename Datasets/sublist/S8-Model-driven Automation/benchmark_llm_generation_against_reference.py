#!/usr/bin/env python3
"""Generate texts with multiple LLMs and save the raw generations.

This script is designed for the S8 Model-driven Automation setting:

1. Read an input CSV that contains generation prompts or model inputs.
2. Run multiple LLMs over the same inputs.
3. Align each prompt with a reference row so the reference text is preserved.
4. Save the generated dataset contents for later scoring in a separate script.

The built-in model catalog separates models into:
- `black_box`: proprietary or primarily API-only models
- `white_box`: open-weight/open-source style models that are more inspectable

As of April 1, 2026, the default shortlist included here is:
- OpenAI GPT-5.4
- Anthropic Claude Sonnet 4.5
- Google Gemini 2.5 Pro
- DeepSeek V3.2 Chat (`deepseek-chat`)
- Meta Llama 4 Scout
- Mistral Small 3.2 24B Instruct
"""

import argparse
import csv
import json
import math
import os
import re
import sys
import time
import unicodedata
from collections import Counter
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, Iterator, List, Optional, Sequence, Tuple
from urllib import error, parse, request


csv.field_size_limit(10**9)


SCRIPT_DIR = Path(__file__).resolve().parent
DEFAULT_MODELS_OUTPUT_DIR = str((SCRIPT_DIR / "Models-Output").resolve())
DEFAULT_TIMEOUT = 180
DEFAULT_OUTPUT_TOKENS = 756
DEFAULT_CHUNK_SIZE = 100
DEFAULT_MAX_PROMPT_CHARS = 12000
DEFAULT_MAX_REFERENCE_CHARS = 20000
DEFAULT_PERPLEXITY_MODEL = "gpt2"
DEFAULT_TOPIC_COUNT = 5
DEFAULT_TOPIC_TOP_WORDS = 10
DEFAULT_HF_MODEL_ROOT = os.environ.get("HF_MODEL_ROOT", "").strip()
DEFAULT_TOP_MODELS = [
    "gpt-5.4",
    "claude-sonnet-4",
    "gemini-2.5-pro",
    "deepseek-r1-distill-qwen-7b",
    "llama-4-scout",
    "mistral-small-3.2",
]

WORD_RE = re.compile(r"[A-Za-z0-9_]+")
SUBJECT_LINE_RE = re.compile(r"^\s*subject\s*:\s*(.+?)\s*$", re.IGNORECASE)
BODY_LINE_RE = re.compile(r"^\s*body\s*:\s*(.*)$", re.IGNORECASE)


@dataclass(frozen=True)
class ModelSpec:
    alias: str
    company: str
    family: str
    access_type: str
    provider: str
    model_name: str
    api_base_url: str = ""
    api_key_env: str = ""
    hf_model_id: str = ""
    enabled_by_default: bool = False
    notes: str = ""


@dataclass
class Sample:
    join_key: str
    input_row_number: int
    reference_row_number: int
    prompt_text: str
    reference_text: str
    label: str


@dataclass
class GenerationRow:
    join_key: str
    input_row_number: int
    reference_row_number: int
    model_alias: str
    company: str
    family: str
    access_type: str
    provider: str
    prompt_text: str
    generated_text: str
    generated_subject: str
    generated_body: str
    reference_text: str
    label: str
    created_at: str


MODEL_CATALOG: List[ModelSpec] = [
    ModelSpec(
        alias="gpt-5.4",
        company="OpenAI",
        family="GPT",
        access_type="black_box",
        provider="openai_chat",
        model_name="gpt-5.4",
        api_base_url="https://api.openai.com/v1",
        api_key_env="OPENAI_API_KEY",
        enabled_by_default=True,
        notes="Current frontier OpenAI model in official docs.",
    ),
    ModelSpec(
        alias="gpt-5.4-mini",
        company="OpenAI",
        family="GPT",
        access_type="black_box",
        provider="openai_chat",
        model_name="gpt-5.4-mini",
        api_base_url="https://api.openai.com/v1",
        api_key_env="OPENAI_API_KEY",
        notes="Lower-latency OpenAI frontier mini variant.",
    ),
    ModelSpec(
        alias="claude-sonnet-4",
        company="Anthropic",
        family="Claude",
        access_type="black_box",
        provider="anthropic_messages",
        model_name="claude-sonnet-4-5",
        api_base_url="https://api.anthropic.com/v1/messages",
        api_key_env="ANTHROPIC_API_KEY",
        enabled_by_default=True,
        notes="Latest balanced Claude model in current official docs.",
    ),
    ModelSpec(
        alias="claude-opus-4",
        company="Anthropic",
        family="Claude",
        access_type="black_box",
        provider="anthropic_messages",
        model_name="claude-opus-4-6",
        api_base_url="https://api.anthropic.com/v1/messages",
        api_key_env="ANTHROPIC_API_KEY",
        notes="Most capable Claude model in current official docs.",
    ),
    ModelSpec(
        alias="gemini-2.5-pro",
        company="Google",
        family="Gemini",
        access_type="black_box",
        provider="gemini_generate_content",
        model_name="gemini-2.5-pro",
        api_base_url="https://generativelanguage.googleapis.com/v1beta/models",
        api_key_env="GEMINI_API_KEY",
        enabled_by_default=True,
        notes="Google's state-of-the-art Gemini API model.",
    ),
    ModelSpec(
        alias="deepseek-r1-distill-qwen-7b",
        company="DeepSeek",
        family="DeepSeek",
        access_type="white_box",
        provider="local_hf",
        model_name="deepseek-ai/DeepSeek-R1-Distill-Qwen-7B",
        hf_model_id="deepseek-ai/DeepSeek-R1-Distill-Qwen-7B",
        enabled_by_default=True,
        notes="Open-weight DeepSeek R1 distilled Qwen 7B model for local Hugging Face inference.",
    ),
    ModelSpec(
        alias="grok-4",
        company="xAI",
        family="Grok",
        access_type="black_box",
        provider="openai_chat",
        model_name="grok-4",
        api_base_url="https://api.x.ai/v1",
        api_key_env="XAI_API_KEY",
        notes="Latest generally documented Grok family model.",
    ),
    ModelSpec(
        alias="llama-4-scout",
        company="Meta",
        family="Llama",
        access_type="white_box",
        provider="local_hf",
        model_name="meta-llama/Llama-4-Scout-17B-16E-Instruct",
        hf_model_id="meta-llama/Llama-4-Scout-17B-16E-Instruct",
        enabled_by_default=True,
        notes="Open-weight Llama 4 Scout.",
    ),
    ModelSpec(
        alias="llama-4-maverick",
        company="Meta",
        family="Llama",
        access_type="white_box",
        provider="local_hf",
        model_name="meta-llama/Llama-4-Maverick-17B-128E-Instruct",
        hf_model_id="meta-llama/Llama-4-Maverick-17B-128E-Instruct",
        notes="Open-weight Llama 4 Maverick.",
    ),
    ModelSpec(
        alias="llama-3.1-8b",
        company="Meta",
        family="Llama",
        access_type="white_box",
        provider="local_hf",
        model_name="meta-llama/Llama-3.1-8B-Instruct",
        hf_model_id="meta-llama/Llama-3.1-8B-Instruct",
        notes="Open-weight Llama 3.1 8B Instruct model for local Hugging Face inference.",
    ),
    ModelSpec(
        alias="qwen3-30b-a3b",
        company="Qwen",
        family="Qwen3",
        access_type="white_box",
        provider="local_hf",
        model_name="Qwen/Qwen3-30B-A3B",
        hf_model_id="Qwen/Qwen3-30B-A3B",
        notes="Open-weight Qwen3 model.",
    ),
    ModelSpec(
        alias="qwen3-235b-a22b",
        company="Qwen",
        family="Qwen3",
        access_type="white_box",
        provider="local_hf",
        model_name="Qwen/Qwen3-235B-A22B",
        hf_model_id="Qwen/Qwen3-235B-A22B",
        notes="Large Qwen3 flagship model.",
    ),
    ModelSpec(
        alias="mistral-small-3.2",
        company="Mistral",
        family="Mistral",
        access_type="white_box",
        provider="local_hf",
        model_name="mistralai/Mistral-Small-3.2-24B-Instruct-2506",
        hf_model_id="mistralai/Mistral-Small-3.2-24B-Instruct-2506",
        enabled_by_default=True,
        notes="Open-weight Mistral Small 3.2 instruct model for local Hugging Face inference.",
    ),
    ModelSpec(
        alias="mistral-large-3",
        company="Mistral",
        family="Mistral",
        access_type="black_box",
        provider="openai_chat",
        model_name="mistral-large-2512",
        api_base_url="https://api.mistral.ai/v1",
        api_key_env="MISTRAL_API_KEY",
        notes="Proprietary Mistral API model kept as an optional non-default baseline.",
    ),
    ModelSpec(
        alias="ministral-8b",
        company="Mistral",
        family="Ministral",
        access_type="white_box",
        provider="local_hf",
        model_name="mistralai/Ministral-8B-Instruct-2410",
        hf_model_id="mistralai/Ministral-8B-Instruct-2410",
        notes="Smaller open-weight Mistral-family candidate for lower-resource local runs.",
    ),
]


_LOCAL_GENERATORS: Dict[str, Any] = {}
_PERPLEXITY_SCORER: Dict[str, Any] = {}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run multiple LLMs on an input CSV and save generation outputs for later scoring."
    )
    parser.add_argument("--input", default="", help="Input CSV containing prompts or source inputs.")
    parser.add_argument(
        "--reference",
        default="",
        help="Reference CSV used for comparison. Optional if the reference text already exists in the input CSV.",
    )
    parser.add_argument(
        "--output-dir",
        default="",
        help="Optional output directory. Defaults to S8-Model-driven Automation/runs/<timestamp>/",
    )
    parser.add_argument(
        "--models-output-dir",
        default=DEFAULT_MODELS_OUTPUT_DIR,
        help="Directory that stores cumulative per-model generated output CSVs for later reuse.",
    )
    parser.add_argument(
        "--prompt-column",
        default="prompt",
        help="Column in the input CSV used as model input.",
    )
    parser.add_argument(
        "--reference-column",
        default="source_text",
        help="Column in the reference CSV used as the comparison target.",
    )
    parser.add_argument(
        "--reference-fallback-columns",
        default="text,source_text,Body",
        help="Comma-separated fallback columns when --reference-column is absent.",
    )
    parser.add_argument(
        "--id-column",
        default="",
        help="Optional join key column in the input CSV.",
    )
    parser.add_argument(
        "--reference-id-column",
        default="",
        help="Optional join key column in the reference CSV. If omitted, --id-column is reused.",
    )
    parser.add_argument(
        "--system-prompt",
        default="You are a helpful assistant. Follow the user's instruction and return only the requested text.",
        help="Optional system instruction used for generation.",
    )
    parser.add_argument(
        "--models",
        default="default",
        help="Comma-separated model aliases, or 'default' for the built-in top-5 set.",
    )
    parser.add_argument(
        "--access-type",
        choices=("all", "black_box", "white_box"),
        default="all",
        help="Filter models by access type.",
    )
    parser.add_argument(
        "--list-models",
        action="store_true",
        help="Print the built-in model catalog and exit.",
    )
    parser.add_argument(
        "--max-rows",
        type=int,
        default=0,
        help="Optional limit on the number of aligned rows to process.",
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=0.7,
        help="Sampling temperature forwarded to generation APIs/models.",
    )
    parser.add_argument(
        "--max-output-tokens",
        type=int,
        default=DEFAULT_OUTPUT_TOKENS,
        help="Maximum output tokens for each generation.",
    )
    parser.add_argument(
        "--max-prompt-chars",
        type=int,
        default=DEFAULT_MAX_PROMPT_CHARS,
        help="Truncate prompt text to at most this many characters before generation.",
    )
    parser.add_argument(
        "--max-reference-chars",
        type=int,
        default=DEFAULT_MAX_REFERENCE_CHARS,
        help="Truncate aligned reference text to at most this many characters before saving.",
    )
    parser.add_argument(
        "--chunk-size",
        type=int,
        default=DEFAULT_CHUNK_SIZE,
        help="Number of aligned rows processed at a time for each model.",
    )
    parser.add_argument(
        "--timeout",
        type=int,
        default=DEFAULT_TIMEOUT,
        help="HTTP timeout in seconds.",
    )
    parser.add_argument(
        "--sleep-seconds",
        type=float,
        default=0.0,
        help="Delay between model calls.",
    )
    parser.add_argument(
        "--save-every",
        type=int,
        default=25,
        help="Checkpoint generated rows every N completed generations.",
    )
    parser.add_argument(
        "--print-every",
        type=int,
        default=10,
        help="Progress logging interval in completed generations.",
    )
    parser.add_argument(
        "--hf-device-map",
        default="auto",
        help="Device map for local Hugging Face models.",
    )
    parser.add_argument(
        "--hf-dtype",
        default="auto",
        help="torch dtype hint for local Hugging Face models: auto, float16, bfloat16, float32.",
    )
    parser.add_argument(
        "--hf-model-root",
        default=DEFAULT_HF_MODEL_ROOT,
        help="Optional local Hugging Face model root. If set, local_hf models are loaded from this directory first.",
    )
    parser.add_argument(
        "--resume-existing",
        action="store_true",
        help="Load existing per-model generated output CSVs and skip samples that were already generated.",
    )
    parser.add_argument(
        "--resume-match-key",
        choices=("prompt_text", "join_key"),
        default="prompt_text",
        help="Field used to decide whether a sample already exists when --resume-existing is enabled.",
    )
    parser.add_argument(
        "--local-batch-size",
        type=int,
        default=4,
        help="Batch size for local Hugging Face generation. Larger values can improve throughput if GPU memory allows.",
    )
    return parser.parse_args()


def validate_args(args: argparse.Namespace) -> None:
    if args.list_models:
        return
    if not args.input:
        raise ValueError("Missing --input CSV path.")
    if args.max_output_tokens <= 0:
        raise ValueError("--max-output-tokens must be greater than 0.")
    if args.max_prompt_chars <= 0:
        raise ValueError("--max-prompt-chars must be greater than 0.")
    if args.max_reference_chars <= 0:
        raise ValueError("--max-reference-chars must be greater than 0.")
    if args.chunk_size <= 0:
        raise ValueError("--chunk-size must be greater than 0.")
    if args.local_batch_size <= 0:
        raise ValueError("--local-batch-size must be greater than 0.")
    if args.timeout <= 0:
        raise ValueError("--timeout must be greater than 0.")


def now_utc_iso() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat()


def normalize_text(value: str) -> str:
    text = (value or "").replace("\r\n", "\n").replace("\r", "\n")
    text = text.replace("\ufeff", "").replace("\u200b", "").replace("\xa0", " ")
    text = re.sub(r"[ \t]+\n", "\n", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def strip_local_hf_problematic_chars(text: str) -> str:
    cleaned_chars: List[str] = []
    for ch in text or "":
        codepoint = ord(ch)
        category = unicodedata.category(ch)
        if 0xFE00 <= codepoint <= 0xFE0F:
            continue
        if 0xE0100 <= codepoint <= 0xE01EF:
            continue
        if category in {"Cf", "Cs"}:
            continue
        cleaned_chars.append(ch)
    return "".join(cleaned_chars)


def make_latin1_safe(text: str) -> str:
    return (text or "").encode("latin-1", errors="ignore").decode("latin-1")


def truncate_text(text: str, max_chars: int) -> str:
    if max_chars <= 0 or len(text) <= max_chars:
        return text
    return text[:max_chars].rstrip()


def truncate_with_notice(
    text: str,
    max_chars: int,
    *,
    field_name: str,
    row_number: int,
    join_key: str,
) -> str:
    truncated = truncate_text(text, max_chars)
    if len(truncated) != len(text):
        print(
            f"[truncate] {field_name} for join_key={join_key} row={row_number} "
            f"from {len(text)} to {len(truncated)} chars",
            file=sys.stderr,
        )
    return truncated


def tokenize_words(text: str) -> List[str]:
    return [token.lower() for token in WORD_RE.findall(text or "")]


def iter_csv_rows(path: Path) -> Iterator[Tuple[int, Dict[str, str]]]:
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        if not reader.fieldnames:
            raise ValueError(f"CSV has no header row: {path}")
        for row_number, row in enumerate(reader, start=2):
            yield row_number, row


def build_output_dir(output_dir_arg: str) -> Path:
    if output_dir_arg:
        return Path(output_dir_arg).resolve()
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    return (SCRIPT_DIR / "runs" / f"benchmark_{stamp}").resolve()


def iter_sample_chunks(samples: Sequence[Sample], chunk_size: int) -> Iterator[Tuple[int, int, List[Sample]]]:
    if chunk_size <= 0:
        raise ValueError("--chunk-size must be greater than 0.")
    for start in range(0, len(samples), chunk_size):
        end = min(start + chunk_size, len(samples))
        yield start, end, list(samples[start:end])


def iter_batches(items: Sequence[Any], batch_size: int) -> Iterator[List[Any]]:
    if batch_size <= 0:
        raise ValueError("batch_size must be greater than 0.")
    for start in range(0, len(items), batch_size):
        yield list(items[start : start + batch_size])


def chunk_name(start: int, end: int) -> str:
    return f"chunk_{start + 1:06d}_{end:06d}"


def build_model_generated_output_path(base_dir: Path, model_alias: str) -> Path:
    return base_dir / f"{model_alias}-generated_output.csv"


def append_jsonl(path: Path, record: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(record, ensure_ascii=False) + "\n")


def safe_float(value: Optional[float]) -> str:
    if value is None:
        return ""
    if isinstance(value, float) and (math.isnan(value) or math.isinf(value)):
        return ""
    return f"{value:.6f}"


def split_generated_email_fields(text: str) -> Tuple[str, str]:
    cleaned = normalize_text(text)
    if not cleaned:
        return "", ""

    lines = cleaned.splitlines()
    subject = ""
    body_lines = lines

    for index, line in enumerate(lines[:5]):
        match = SUBJECT_LINE_RE.match(line.strip())
        if match:
            subject = normalize_text(match.group(1))
            body_lines = lines[index + 1 :]
            break

    while body_lines and not normalize_text(body_lines[0]):
        body_lines = body_lines[1:]

    if body_lines:
        body_match = BODY_LINE_RE.match(body_lines[0].strip())
        if body_match:
            replacement = normalize_text(body_match.group(1))
            body_lines = ([replacement] if replacement else []) + body_lines[1:]

    body = normalize_text("\n".join(body_lines))
    if not subject and body:
        return "", body
    return subject, body


def write_generation_csv(path: Path, rows: Sequence[GenerationRow]) -> None:
    fieldnames = [
        "join_key",
        "input_row_number",
        "reference_row_number",
        "model_alias",
        "company",
        "family",
        "access_type",
        "provider",
        "prompt_text",
        "generated_text",
        "generated_subject",
        "generated_body",
        "reference_text",
        "label",
        "created_at",
    ]
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(
                {
                    "join_key": row.join_key,
                    "input_row_number": row.input_row_number,
                    "reference_row_number": row.reference_row_number,
                    "model_alias": row.model_alias,
                    "company": row.company,
                    "family": row.family,
                    "access_type": row.access_type,
                    "provider": row.provider,
                    "prompt_text": row.prompt_text,
                    "generated_text": row.generated_text,
                    "generated_subject": row.generated_subject,
                    "generated_body": row.generated_body,
                    "reference_text": row.reference_text,
                    "label": row.label,
                    "created_at": row.created_at,
                }
            )


def load_generation_csv(path: Path) -> List[GenerationRow]:
    if not path.exists():
        return []

    rows: List[GenerationRow] = []
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            try:
                input_row_number = int(str(row.get("input_row_number", "")).strip() or 0)
            except ValueError:
                input_row_number = 0
            try:
                reference_row_number = int(str(row.get("reference_row_number", "")).strip() or 0)
            except ValueError:
                reference_row_number = 0
            rows.append(
                GenerationRow(
                    join_key=str(row.get("join_key", "")),
                    input_row_number=input_row_number,
                    reference_row_number=reference_row_number,
                    model_alias=str(row.get("model_alias", "")),
                    company=str(row.get("company", "")),
                    family=str(row.get("family", "")),
                    access_type=str(row.get("access_type", "")),
                    provider=str(row.get("provider", "")),
                    prompt_text=str(row.get("prompt_text", "")),
                    generated_text=str(row.get("generated_text", "")),
                    generated_subject=str(row.get("generated_subject", "")),
                    generated_body=str(row.get("generated_body", "")),
                    reference_text=str(row.get("reference_text", "")),
                    label=str(row.get("label", "")),
                    created_at=str(row.get("created_at", "")),
                )
            )
    return rows


def dedupe_generation_rows(rows: Sequence[GenerationRow]) -> List[GenerationRow]:
    deduped: List[GenerationRow] = []
    seen: set[Tuple[str, int, str]] = set()
    for row in rows:
        key = (row.model_alias, row.input_row_number, row.prompt_text)
        if key in seen:
            continue
        seen.add(key)
        deduped.append(row)
    return deduped


def get_sample_match_value(sample: Sample, match_key: str) -> str:
    if match_key == "join_key":
        return sample.join_key
    if match_key == "prompt_text":
        return sample.prompt_text
    raise ValueError(f"Unsupported resume match key: {match_key}")


def get_generation_match_value(row: GenerationRow, match_key: str) -> str:
    if match_key == "join_key":
        return row.join_key
    if match_key == "prompt_text":
        return row.prompt_text
    raise ValueError(f"Unsupported resume match key: {match_key}")


def write_model_content_csv(path: Path, rows: Sequence[GenerationRow]) -> None:
    fieldnames = ["subject", "body", "label"]
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(
                {
                    "subject": row.generated_subject,
                    "body": row.generated_body,
                    "label": row.label,
                }
            )


def write_json(path: Path, payload: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, ensure_ascii=False, indent=2)
        handle.write("\n")


def write_generation_csv_targets(rows: Sequence[GenerationRow], paths: Sequence[Path]) -> None:
    seen: set[str] = set()
    for path in paths:
        normalized = str(path.resolve())
        if normalized in seen:
            continue
        seen.add(normalized)
        write_generation_csv(path, rows)


def print_model_catalog() -> None:
    rows = []
    for spec in MODEL_CATALOG:
        rows.append(
            {
                "alias": spec.alias,
                "company": spec.company,
                "family": spec.family,
                "access_type": spec.access_type,
                "provider": spec.provider,
                "model_name": spec.model_name,
                "enabled_by_default": str(spec.enabled_by_default),
                "notes": spec.notes,
            }
        )
    print(json.dumps(rows, ensure_ascii=False, indent=2))


def get_catalog_map() -> Dict[str, ModelSpec]:
    return {spec.alias: spec for spec in MODEL_CATALOG}


def select_models(args: argparse.Namespace) -> List[ModelSpec]:
    catalog = get_catalog_map()
    if args.models.strip().lower() == "default":
        requested_aliases = list(DEFAULT_TOP_MODELS)
    else:
        requested_aliases = [alias.strip() for alias in args.models.split(",") if alias.strip()]
    if not requested_aliases:
        raise ValueError("No models selected.")

    selected: List[ModelSpec] = []
    for alias in requested_aliases:
        if alias not in catalog:
            available = ", ".join(sorted(catalog))
            raise ValueError(f"Unknown model alias '{alias}'. Available: {available}")
        spec = catalog[alias]
        if args.access_type != "all" and spec.access_type != args.access_type:
            continue
        selected.append(spec)

    if not selected:
        raise ValueError("No models remain after applying --access-type filter.")
    return selected


def read_reference_text_from_row(
    row: Dict[str, str], primary_column: str, fallback_columns: Sequence[str]
) -> str:
    candidates = [primary_column] + [column for column in fallback_columns if column and column != primary_column]
    for column in candidates:
        value = normalize_text(str(row.get(column, "")))
        if value:
            return value
    subject = normalize_text(str(row.get("Subject", "")))
    body = normalize_text(str(row.get("Body", "")))
    if subject and body:
        return f"Subject: {subject}\n\n{body}"
    return body or subject


def align_samples(args: argparse.Namespace) -> List[Sample]:
    input_path = Path(args.input).resolve()
    if not input_path.exists():
        raise FileNotFoundError(f"Input CSV not found: {input_path}")

    fallback_columns = [column.strip() for column in args.reference_fallback_columns.split(",") if column.strip()]
    input_rows = list(iter_csv_rows(input_path))

    if args.reference:
        reference_path = Path(args.reference).resolve()
        if not reference_path.exists():
            raise FileNotFoundError(f"Reference CSV not found: {reference_path}")
        reference_rows = list(iter_csv_rows(reference_path))
    else:
        reference_path = input_path
        reference_rows = input_rows

    input_id_column = args.id_column.strip()
    reference_id_column = (args.reference_id_column or args.id_column).strip()

    aligned: List[Sample] = []

    if input_id_column and reference_id_column:
        reference_by_id: Dict[str, Tuple[int, Dict[str, str]]] = {}
        for row_number, row in reference_rows:
            join_key = str(row.get(reference_id_column, "")).strip()
            if join_key and join_key not in reference_by_id:
                reference_by_id[join_key] = (row_number, row)

        for input_row_number, input_row in input_rows:
            join_key = str(input_row.get(input_id_column, "")).strip()
            if not join_key or join_key not in reference_by_id:
                continue
            prompt_text = normalize_text(str(input_row.get(args.prompt_column, "")))
            if not prompt_text:
                continue
            reference_row_number, reference_row = reference_by_id[join_key]
            reference_text = read_reference_text_from_row(
                reference_row, args.reference_column, fallback_columns
            )
            if not reference_text:
                continue
            prompt_text = truncate_with_notice(
                prompt_text,
                args.max_prompt_chars,
                field_name="prompt_text",
                row_number=input_row_number,
                join_key=join_key,
            )
            reference_text = truncate_with_notice(
                reference_text,
                args.max_reference_chars,
                field_name="reference_text",
                row_number=reference_row_number,
                join_key=join_key,
            )
            label = normalize_text(str(input_row.get("label", ""))) or normalize_text(
                str(reference_row.get("label", ""))
            )
            aligned.append(
                Sample(
                    join_key=join_key,
                    input_row_number=input_row_number,
                    reference_row_number=reference_row_number,
                    prompt_text=prompt_text,
                    reference_text=reference_text,
                    label=label,
                )
            )
    else:
        paired = zip(input_rows, reference_rows)
        for index, ((input_row_number, input_row), (reference_row_number, reference_row)) in enumerate(
            paired, start=1
        ):
            prompt_text = normalize_text(str(input_row.get(args.prompt_column, "")))
            if not prompt_text:
                continue
            reference_text = read_reference_text_from_row(
                reference_row, args.reference_column, fallback_columns
            )
            if not reference_text:
                continue
            prompt_text = truncate_with_notice(
                prompt_text,
                args.max_prompt_chars,
                field_name="prompt_text",
                row_number=input_row_number,
                join_key=str(index),
            )
            reference_text = truncate_with_notice(
                reference_text,
                args.max_reference_chars,
                field_name="reference_text",
                row_number=reference_row_number,
                join_key=str(index),
            )
            label = normalize_text(str(input_row.get("label", ""))) or normalize_text(
                str(reference_row.get("label", ""))
            )
            aligned.append(
                Sample(
                    join_key=str(index),
                    input_row_number=input_row_number,
                    reference_row_number=reference_row_number,
                    prompt_text=prompt_text,
                    reference_text=reference_text,
                    label=label,
                )
            )

    if args.max_rows > 0:
        aligned = aligned[: args.max_rows]
    if not aligned:
        raise ValueError("No aligned samples were found. Check join columns and text column names.")
    return aligned


def http_post_json(url: str, headers: Dict[str, str], payload: Dict[str, Any], timeout: int) -> Dict[str, Any]:
    body = json.dumps(payload).encode("utf-8")
    req = request.Request(url, data=body, headers=headers, method="POST")
    try:
        with request.urlopen(req, timeout=timeout) as response:
            raw = response.read().decode("utf-8")
            return json.loads(raw)
    except error.HTTPError as exc:
        error_body = exc.read().decode("utf-8", errors="replace")
        raise RuntimeError(f"HTTP {exc.code} for {url}: {error_body}") from exc
    except error.URLError as exc:
        raise RuntimeError(f"Network error for {url}: {exc}") from exc


def extract_openai_chat_text(payload: Dict[str, Any]) -> str:
    choices = payload.get("choices") or []
    if not choices:
        raise ValueError("No choices in OpenAI-compatible response.")
    message = choices[0].get("message") or {}
    content = message.get("content", "")
    if isinstance(content, str):
        return normalize_text(content)
    if isinstance(content, list):
        parts: List[str] = []
        for item in content:
            if isinstance(item, dict) and item.get("type") == "text":
                parts.append(str(item.get("text", "")))
        return normalize_text("\n".join(parts))
    raise ValueError("Unsupported OpenAI-compatible response content format.")


def extract_anthropic_text(payload: Dict[str, Any]) -> str:
    content = payload.get("content") or []
    parts: List[str] = []
    for item in content:
        if isinstance(item, dict) and item.get("type") == "text":
            parts.append(str(item.get("text", "")))
    text = normalize_text("\n".join(parts))
    if not text:
        raise ValueError("No text content in Anthropic response.")
    return text


def extract_gemini_text(payload: Dict[str, Any]) -> str:
    candidates = payload.get("candidates") or []
    if not candidates:
        raise ValueError("No candidates in Gemini response.")
    content = candidates[0].get("content") or {}
    parts = content.get("parts") or []
    text_parts = [str(part.get("text", "")) for part in parts if isinstance(part, dict)]
    text = normalize_text("\n".join(text_parts))
    if not text:
        raise ValueError("No text content in Gemini response.")
    return text


def get_api_key(spec: ModelSpec) -> str:
    if not spec.api_key_env:
        return ""
    return os.environ.get(spec.api_key_env, "").strip()


def get_hf_token() -> str:
    for env_name in ("HF_TOKEN", "HUGGING_FACE_HUB_TOKEN", "HUGGINGFACE_TOKEN"):
        value = os.environ.get(env_name, "").strip()
        if value:
            return value
    return ""


def load_hf_component(loader: Any, source: str, **kwargs: Any) -> Any:
    token = get_hf_token()
    if not token:
        return loader.from_pretrained(source, **kwargs)
    try:
        return loader.from_pretrained(source, token=token, **kwargs)
    except TypeError:
        return loader.from_pretrained(source, use_auth_token=token, **kwargs)


def generate_with_openai_chat(
    spec: ModelSpec, prompt_text: str, system_prompt: str, args: argparse.Namespace
) -> Tuple[str, Dict[str, Any]]:
    api_key = get_api_key(spec)
    if not api_key:
        raise ValueError(f"Missing API key for {spec.alias}. Set {spec.api_key_env}.")
    url = spec.api_base_url.rstrip("/") + "/chat/completions"
    payload: Dict[str, Any] = {
        "model": spec.model_name,
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": prompt_text},
        ],
        "temperature": args.temperature,
        "max_tokens": args.max_output_tokens,
    }
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
    }
    response_payload = http_post_json(url, headers, payload, args.timeout)
    return extract_openai_chat_text(response_payload), response_payload


def generate_with_anthropic(
    spec: ModelSpec, prompt_text: str, system_prompt: str, args: argparse.Namespace
) -> Tuple[str, Dict[str, Any]]:
    api_key = get_api_key(spec)
    if not api_key:
        raise ValueError(f"Missing API key for {spec.alias}. Set {spec.api_key_env}.")
    payload: Dict[str, Any] = {
        "model": spec.model_name,
        "max_tokens": args.max_output_tokens,
        "temperature": args.temperature,
        "system": system_prompt,
        "messages": [{"role": "user", "content": prompt_text}],
    }
    headers = {
        "x-api-key": api_key,
        "anthropic-version": "2023-06-01",
        "content-type": "application/json",
    }
    response_payload = http_post_json(spec.api_base_url, headers, payload, args.timeout)
    return extract_anthropic_text(response_payload), response_payload


def generate_with_gemini(
    spec: ModelSpec, prompt_text: str, system_prompt: str, args: argparse.Namespace
) -> Tuple[str, Dict[str, Any]]:
    api_key = get_api_key(spec)
    if not api_key:
        raise ValueError(f"Missing API key for {spec.alias}. Set {spec.api_key_env}.")
    endpoint = f"{spec.api_base_url.rstrip('/')}/{spec.model_name}:generateContent"
    url = endpoint + "?" + parse.urlencode({"key": api_key})
    payload: Dict[str, Any] = {
        "system_instruction": {"parts": [{"text": system_prompt}]},
        "contents": [{"parts": [{"text": prompt_text}]}],
        "generationConfig": {
            "temperature": args.temperature,
            "maxOutputTokens": args.max_output_tokens,
        },
    }
    headers = {"Content-Type": "application/json"}
    response_payload = http_post_json(url, headers, payload, args.timeout)
    return extract_gemini_text(response_payload), response_payload


def resolve_torch_dtype(dtype_name: str) -> Any:
    if dtype_name == "auto":
        return "auto"
    import torch  # type: ignore

    mapping = {
        "float16": torch.float16,
        "bfloat16": torch.bfloat16,
        "float32": torch.float32,
    }
    if dtype_name not in mapping:
        raise ValueError(f"Unsupported --hf-dtype value: {dtype_name}")
    return mapping[dtype_name]


def resolve_local_hf_model_source(spec: ModelSpec, args: argparse.Namespace) -> str:
    model_id = (spec.hf_model_id or spec.model_name or "").strip()
    if not model_id:
        raise ValueError(f"No local HF model id configured for {spec.alias}.")

    root = Path(args.hf_model_root).expanduser().resolve() if args.hf_model_root else None
    if root:
        candidates = [
            root / model_id,
            root / model_id.replace("/", "--"),
            root / model_id.split("/")[-1],
        ]
        for candidate in candidates:
            if candidate.exists():
                return str(candidate)
    return model_id


def get_local_generator(spec: ModelSpec, args: argparse.Namespace) -> Any:
    cache_key = f"{spec.alias}|{args.hf_device_map}|{args.hf_dtype}|{args.hf_model_root}"
    if cache_key in _LOCAL_GENERATORS:
        return _LOCAL_GENERATORS[cache_key]

    try:
        from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline  # type: ignore
    except ImportError as exc:
        raise RuntimeError(
            "transformers is required for local Hugging Face models. "
            "Install it with: python3 -m pip install transformers torch"
        ) from exc

    model_source = resolve_local_hf_model_source(spec, args)
    tokenizer = load_hf_component(AutoTokenizer, model_source)
    tokenizer.padding_side = "left"
    if tokenizer.pad_token is None and tokenizer.eos_token is not None:
        tokenizer.pad_token = tokenizer.eos_token
    model = load_hf_component(
        AutoModelForCausalLM,
        model_source,
        device_map=args.hf_device_map,
        torch_dtype=resolve_torch_dtype(args.hf_dtype),
    )

    try:
        import torch  # type: ignore
    except ImportError:
        torch = None  # type: ignore

    requested_gpu = str(args.hf_device_map).lower() != "cpu"
    if requested_gpu and torch is not None and hasattr(model, "device"):
        model_device = str(model.device)
        if not model_device.startswith("cuda"):
            raise RuntimeError(
                f"Local model {spec.alias} loaded on {model_device} instead of GPU. "
                "Check CUDA_VISIBLE_DEVICES, torch CUDA availability, and --hf-device-map."
            )

    generator = pipeline(
        "text-generation",
        model=model,
        tokenizer=tokenizer,
    )
    _LOCAL_GENERATORS[cache_key] = generator
    return generator


def sanitize_local_hf_chat_payload(payload: Any, sanitizer: Any) -> Any:
    if isinstance(payload, list):
        return [sanitize_local_hf_chat_payload(item, sanitizer) for item in payload]
    if isinstance(payload, dict) and "content" in payload:
        return {
            "role": payload.get("role", ""),
            "content": sanitizer(str(payload.get("content", ""))),
        }
    return payload


def run_local_hf_generation(generator: Any, chat_payload: Any, args: argparse.Namespace) -> Tuple[Any, str]:
    retry_mode = "none"
    try:
        result = generator(
            chat_payload,
            max_new_tokens=args.max_output_tokens,
            temperature=args.temperature,
            do_sample=args.temperature > 0,
            return_full_text=False,
            batch_size=args.local_batch_size,
        )
    except UnicodeEncodeError as exc:
        retry_mode = "strip_problematic_unicode"
        sanitized_messages = sanitize_local_hf_chat_payload(
            chat_payload, strip_local_hf_problematic_chars
        )
        try:
            result = generator(
                sanitized_messages,
                max_new_tokens=args.max_output_tokens,
                temperature=args.temperature,
                do_sample=args.temperature > 0,
                return_full_text=False,
                batch_size=args.local_batch_size,
            )
        except UnicodeEncodeError:
            retry_mode = "latin1_safe"
            fallback_messages = sanitize_local_hf_chat_payload(
                chat_payload,
                lambda value: make_latin1_safe(strip_local_hf_problematic_chars(value)),
            )
            result = generator(
                fallback_messages,
                max_new_tokens=args.max_output_tokens,
                temperature=args.temperature,
                do_sample=args.temperature > 0,
                return_full_text=False,
                batch_size=args.local_batch_size,
            )
    return result, retry_mode


def extract_local_hf_result_text(result_item: Any) -> str:
    first = result_item[0] if isinstance(result_item, list) and result_item else result_item
    if isinstance(first, dict):
        if "generated_text" in first and isinstance(first["generated_text"], str):
            text = first["generated_text"]
        elif "generated_text" in first and isinstance(first["generated_text"], list):
            parts: List[str] = []
            for item in first["generated_text"]:
                if isinstance(item, dict) and "content" in item:
                    parts.append(str(item["content"]))
            text = "\n".join(parts)
        else:
            text = str(first)
    else:
        text = str(first)
    return normalize_text(text)


def generate_with_local_hf(
    spec: ModelSpec, prompt_text: str, system_prompt: str, args: argparse.Namespace
) -> Tuple[str, Dict[str, Any]]:
    generator = get_local_generator(spec, args)
    chat_messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": prompt_text},
    ]
    result, retry_mode = run_local_hf_generation(generator, chat_messages, args)
    if not result:
        raise ValueError(f"No output returned by local model {spec.alias}")
    payload = {"provider": "local_hf", "raw_result": result, "retry_mode": retry_mode}
    return extract_local_hf_result_text(result), payload


def generate_with_local_hf_batch(
    spec: ModelSpec, prompt_texts: Sequence[str], system_prompt: str, args: argparse.Namespace
) -> List[Tuple[str, Dict[str, Any]]]:
    if not prompt_texts:
        return []

    generator = get_local_generator(spec, args)
    chat_payload = [
        [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": prompt_text},
        ]
        for prompt_text in prompt_texts
    ]
    result, retry_mode = run_local_hf_generation(generator, chat_payload, args)
    if not isinstance(result, list) or len(result) != len(prompt_texts):
        raise ValueError(
            f"Unexpected batched output shape from local model {spec.alias}: expected {len(prompt_texts)} results."
        )

    outputs: List[Tuple[str, Dict[str, Any]]] = []
    for item in result:
        outputs.append(
            (
                extract_local_hf_result_text(item),
                {
                    "provider": "local_hf",
                    "raw_result": item,
                    "retry_mode": retry_mode,
                },
            )
        )
    return outputs


def generate_text_for_model(
    spec: ModelSpec, prompt_text: str, system_prompt: str, args: argparse.Namespace
) -> Tuple[str, Dict[str, Any]]:
    if spec.provider == "openai_chat":
        return generate_with_openai_chat(spec, prompt_text, system_prompt, args)
    if spec.provider == "anthropic_messages":
        return generate_with_anthropic(spec, prompt_text, system_prompt, args)
    if spec.provider == "gemini_generate_content":
        return generate_with_gemini(spec, prompt_text, system_prompt, args)
    if spec.provider == "local_hf":
        return generate_with_local_hf(spec, prompt_text, system_prompt, args)
    raise ValueError(f"Unsupported provider for {spec.alias}: {spec.provider}")


def generate_batch_for_model(
    spec: ModelSpec, prompt_texts: Sequence[str], system_prompt: str, args: argparse.Namespace
) -> List[Tuple[str, Dict[str, Any]]]:
    if spec.provider == "local_hf":
        return generate_with_local_hf_batch(spec, prompt_texts, system_prompt, args)
    return [generate_text_for_model(spec, prompt_text, system_prompt, args) for prompt_text in prompt_texts]


def ngrams(tokens: Sequence[str], n: int) -> Counter:
    if n <= 0 or len(tokens) < n:
        return Counter()
    return Counter(tuple(tokens[index : index + n]) for index in range(len(tokens) - n + 1))


def compute_bleu(reference_text: str, candidate_text: str, max_n: int = 4) -> float:
    reference_tokens = tokenize_words(reference_text)
    candidate_tokens = tokenize_words(candidate_text)
    if not reference_tokens or not candidate_tokens:
        return 0.0

    precisions: List[float] = []
    for n in range(1, max_n + 1):
        ref_ngrams = ngrams(reference_tokens, n)
        cand_ngrams = ngrams(candidate_tokens, n)
        if not cand_ngrams:
            precisions.append(0.0)
            continue
        overlap = sum(min(count, ref_ngrams[gram]) for gram, count in cand_ngrams.items())
        precision = (overlap + 1.0) / (sum(cand_ngrams.values()) + 1.0)
        precisions.append(precision)

    if min(precisions) <= 0.0:
        geo_mean = 0.0
    else:
        geo_mean = math.exp(sum(math.log(value) for value in precisions) / max_n)

    ref_len = len(reference_tokens)
    cand_len = len(candidate_tokens)
    if cand_len == 0:
        return 0.0
    if cand_len > ref_len:
        brevity_penalty = 1.0
    else:
        brevity_penalty = math.exp(1.0 - (ref_len / cand_len))
    return 100.0 * brevity_penalty * geo_mean


def compute_rouge1(reference_text: str, candidate_text: str) -> Tuple[float, float, float]:
    reference_tokens = tokenize_words(reference_text)
    candidate_tokens = tokenize_words(candidate_text)
    if not reference_tokens or not candidate_tokens:
        return 0.0, 0.0, 0.0

    reference_counts = Counter(reference_tokens)
    candidate_counts = Counter(candidate_tokens)
    overlap = sum(min(count, candidate_counts[token]) for token, count in reference_counts.items())
    precision = overlap / max(1, sum(candidate_counts.values()))
    recall = overlap / max(1, sum(reference_counts.values()))
    if precision + recall == 0:
        f1 = 0.0
    else:
        f1 = 2.0 * precision * recall / (precision + recall)
    return 100.0 * precision, 100.0 * recall, 100.0 * f1


def get_perplexity_scorer(model_name: str) -> Dict[str, Any]:
    if model_name in _PERPLEXITY_SCORER:
        return _PERPLEXITY_SCORER[model_name]

    try:
        import torch  # type: ignore
        from transformers import AutoModelForCausalLM, AutoTokenizer  # type: ignore
    except ImportError as exc:
        raise RuntimeError(
            "transformers and torch are required for perplexity scoring. "
            "Install them with: python3 -m pip install transformers torch"
        ) from exc

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(model_name)
    model.eval()
    scorer = {"tokenizer": tokenizer, "model": model, "torch": torch}
    _PERPLEXITY_SCORER[model_name] = scorer
    return scorer


def compute_perplexity(text: str, model_name: str) -> float:
    text = normalize_text(text)
    if not text:
        return float("nan")

    scorer = get_perplexity_scorer(model_name)
    tokenizer = scorer["tokenizer"]
    model = scorer["model"]
    torch = scorer["torch"]

    encoded = tokenizer(
        text,
        return_tensors="pt",
        truncation=True,
        max_length=getattr(model.config, "max_position_embeddings", 1024) or 1024,
    )
    with torch.no_grad():
        output = model(**encoded, labels=encoded["input_ids"])
    loss = float(output.loss)
    return float(math.exp(min(loss, 20.0)))


def compute_document_frequencies(tokenized_docs: Sequence[Sequence[str]]) -> Dict[str, int]:
    frequencies: Dict[str, int] = {}
    for doc in tokenized_docs:
        for token in set(doc):
            frequencies[token] = frequencies.get(token, 0) + 1
    return frequencies


def compute_topic_coherence_for_corpus(
    texts: Sequence[str], topic_count: int, top_words: int
) -> float:
    tokenized_docs = [tokenize_words(text) for text in texts if tokenize_words(text)]
    if len(tokenized_docs) < 2:
        return float("nan")

    try:
        from sklearn.decomposition import LatentDirichletAllocation  # type: ignore
        from sklearn.feature_extraction.text import CountVectorizer  # type: ignore
    except ImportError as exc:
        raise RuntimeError(
            "scikit-learn is required for topic coherence scoring. "
            "Install it with: python3 -m pip install scikit-learn"
        ) from exc

    raw_docs = [" ".join(tokens) for tokens in tokenized_docs if tokens]
    vectorizer = CountVectorizer(token_pattern=r"(?u)\b\w+\b", min_df=1)
    matrix = vectorizer.fit_transform(raw_docs)
    vocab = vectorizer.get_feature_names_out()
    if matrix.shape[0] < 2 or matrix.shape[1] < 2:
        return float("nan")

    actual_topics = max(1, min(topic_count, matrix.shape[0], matrix.shape[1]))
    lda = LatentDirichletAllocation(
        n_components=actual_topics,
        random_state=42,
        learning_method="batch",
    )
    lda.fit(matrix)

    doc_freq = compute_document_frequencies(tokenized_docs)
    token_sets = [set(tokens) for tokens in tokenized_docs]

    topic_scores: List[float] = []
    for topic in lda.components_:
        top_indices = topic.argsort()[-top_words:][::-1]
        topic_words = [str(vocab[index]) for index in top_indices]
        pair_scores: List[float] = []
        for right in range(1, len(topic_words)):
            for left in range(right):
                wi = topic_words[left]
                wj = topic_words[right]
                d_wi = doc_freq.get(wi, 0)
                d_wi_wj = sum(1 for token_set in token_sets if wi in token_set and wj in token_set)
                if d_wi <= 0:
                    continue
                pair_scores.append(math.log((d_wi_wj + 1.0) / d_wi))
        if pair_scores:
            topic_scores.append(sum(pair_scores) / len(pair_scores))

    if not topic_scores:
        return float("nan")
    return float(sum(topic_scores) / len(topic_scores))


def average(values: Iterable[float]) -> float:
    cleaned = [value for value in values if not math.isnan(value) and not math.isinf(value)]
    if not cleaned:
        return float("nan")
    return float(sum(cleaned) / len(cleaned))


def build_summary_rows(
    generations: Sequence[GenerationRow], args: argparse.Namespace
) -> List[Dict[str, Any]]:
    by_model: Dict[str, List[GenerationRow]] = {}
    for row in generations:
        by_model.setdefault(row.model_alias, []).append(row)

    summary_rows: List[Dict[str, Any]] = []
    for model_alias, rows in by_model.items():
        model_info = rows[0]
        generated_texts = [row.generated_text for row in rows]
        reference_texts = [row.reference_text for row in rows]

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

        summary_rows.append(
            {
                "model_alias": model_alias,
                "company": model_info.company,
                "family": model_info.family,
                "access_type": model_info.access_type,
                "provider": model_info.provider,
                "sample_count": len(rows),
                "mean_bleu": safe_float(average(row.bleu for row in rows)),
                "mean_rouge1_precision": safe_float(average(row.rouge1_precision for row in rows)),
                "mean_rouge1_recall": safe_float(average(row.rouge1_recall for row in rows)),
                "mean_rouge1_f1": safe_float(average(row.rouge1_f1 for row in rows)),
                "mean_generated_perplexity": safe_float(
                    average(row.generated_perplexity for row in rows)
                ),
                "mean_reference_perplexity": safe_float(
                    average(row.reference_perplexity for row in rows)
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
        )

    summary_rows.sort(key=lambda row: row["model_alias"])
    return summary_rows


def build_manifest(
    args: argparse.Namespace,
    selected_models: Sequence[ModelSpec],
    sample_count: int,
    output_dir: Path,
) -> Dict[str, Any]:
    return {
        "created_at": now_utc_iso(),
        "input_csv": str(Path(args.input).resolve()),
        "reference_csv": str(Path(args.reference).resolve()) if args.reference else "",
        "output_dir": str(output_dir),
        "models_output_dir": str(Path(args.models_output_dir).resolve()) if args.models_output_dir else "",
        "prompt_column": args.prompt_column,
        "reference_column": args.reference_column,
        "id_column": args.id_column,
        "reference_id_column": args.reference_id_column,
        "sample_count": sample_count,
        "chunk_size": args.chunk_size,
        "max_prompt_chars": args.max_prompt_chars,
        "max_reference_chars": args.max_reference_chars,
        "models": [asdict(spec) for spec in selected_models],
        "metrics": {"row_level": [], "corpus_level": []},
        "notes": [
            "This run only saves raw generations and aligned reference text.",
            "Per-model content CSVs are saved as <model_alias>-content.csv with subject, body, and label columns.",
            "Chunk runs are saved under chunks/<model_alias>/chunk_<start>_<end>/.",
            "Use the separate scoring script to compute BLEU, ROUGE-1, perplexity, and topic coherence.",
            "local_hf models resolve from --hf-model-root first and fall back to Hugging Face model ids if no local copy is found.",
        ],
    }


def build_model_manifest(
    args: argparse.Namespace,
    spec: ModelSpec,
    sample_count: int,
    output_dir: Path,
    content_csv_path: Path,
    generated_csv_path: Path,
) -> Dict[str, Any]:
    manifest = build_manifest(args, [spec], sample_count, output_dir)
    manifest["content_csv"] = str(content_csv_path)
    manifest["generated_outputs_csv"] = str(generated_csv_path)
    return manifest


def build_chunk_manifest(
    args: argparse.Namespace,
    spec: ModelSpec,
    chunk_dir: Path,
    chunk_start: int,
    chunk_end: int,
    chunk_generations: Sequence[GenerationRow],
) -> Dict[str, Any]:
    return {
        "created_at": now_utc_iso(),
        "output_dir": str(chunk_dir),
        "models_output_dir": str(Path(args.models_output_dir).resolve()) if args.models_output_dir else "",
        "chunk_name": chunk_name(chunk_start, chunk_end),
        "chunk_start_index": chunk_start,
        "chunk_end_index_exclusive": chunk_end,
        "chunk_sample_count": chunk_end - chunk_start,
        "completed_generations": len(chunk_generations),
        "model": asdict(spec),
        "input_csv": str(Path(args.input).resolve()),
        "reference_csv": str(Path(args.reference).resolve()) if args.reference else "",
        "prompt_column": args.prompt_column,
        "reference_column": args.reference_column,
        "id_column": args.id_column,
        "reference_id_column": args.reference_id_column,
        "chunk_size": args.chunk_size,
        "max_prompt_chars": args.max_prompt_chars,
        "max_reference_chars": args.max_reference_chars,
        "notes": [
            "This manifest describes one model over one contiguous chunk of aligned samples.",
            "Chunk indices are zero-based for start and exclusive for end.",
        ],
    }


def persist_model_run_state(
    args: argparse.Namespace,
    spec: ModelSpec,
    *,
    all_generations_path: Path,
    all_generations: Sequence[GenerationRow],
    model_generations_paths: Sequence[Path],
    model_generations: Sequence[GenerationRow],
    model_content_path: Path,
    model_manifest_path: Path,
    output_dir: Path,
    chunk_generations_path: Path,
    chunk_generations: Sequence[GenerationRow],
    chunk_content_path: Path,
    chunk_manifest_path: Path,
    chunk_manifest: Dict[str, Any],
) -> None:
    write_generation_csv_targets(all_generations, [all_generations_path])
    write_generation_csv_targets(model_generations, model_generations_paths)
    write_generation_csv_targets(chunk_generations, [chunk_generations_path])
    write_model_content_csv(model_content_path, model_generations)
    write_model_content_csv(chunk_content_path, chunk_generations)
    model_manifest = build_model_manifest(
        args=args,
        spec=spec,
        sample_count=len(model_generations),
        output_dir=output_dir,
        content_csv_path=model_content_path,
        generated_csv_path=model_generations_paths[0],
    )
    write_json(model_manifest_path, model_manifest)
    write_json(chunk_manifest_path, chunk_manifest)


def main() -> int:
    args = parse_args()
    validate_args(args)

    if args.list_models:
        print_model_catalog()
        return 0

    selected_models = select_models(args)
    samples = align_samples(args)
    output_dir = build_output_dir(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    models_output_dir = Path(args.models_output_dir).resolve() if args.models_output_dir else None
    if models_output_dir:
        models_output_dir.mkdir(parents=True, exist_ok=True)
    if len(selected_models) == 1 and models_output_dir:
        generations_csv_path = build_model_generated_output_path(models_output_dir, selected_models[0].alias)
    elif len(selected_models) == 1:
        generations_csv_path = build_model_generated_output_path(output_dir, selected_models[0].alias)
    else:
        generations_csv_path = output_dir / "generated_outputs.csv"
    calls_jsonl_path = output_dir / "calls.jsonl"
    manifest_path = output_dir / "run_manifest.json"

    generations: List[GenerationRow] = []
    generations_by_model: Dict[str, List[GenerationRow]] = {}
    pending_samples_by_model: Dict[str, List[Sample]] = {}

    for spec in selected_models:
        existing_rows: List[GenerationRow] = []
        model_generated_output_path = (
            build_model_generated_output_path(models_output_dir, spec.alias)
            if models_output_dir
            else build_model_generated_output_path(output_dir, spec.alias)
        )
        if args.resume_existing:
            existing_rows = dedupe_generation_rows(load_generation_csv(model_generated_output_path))
            if existing_rows:
                generations.extend(existing_rows)

        generations_by_model[spec.alias] = list(existing_rows)
        existing_match_values = {
            get_generation_match_value(row, args.resume_match_key) for row in existing_rows
        }
        if existing_match_values:
            pending_samples = [
                sample
                for sample in samples
                if get_sample_match_value(sample, args.resume_match_key) not in existing_match_values
            ]
        else:
            pending_samples = list(samples)
        pending_samples_by_model[spec.alias] = pending_samples

        if args.resume_existing:
            print(
                f"[resume] {spec.alias}: loaded {len(existing_rows)} existing rows, "
                f"skipping {len(samples) - len(pending_samples)} aligned samples by {args.resume_match_key}, "
                f"pending {len(pending_samples)}",
                file=sys.stderr,
            )

    completed = 0
    total = sum(len(rows) for rows in pending_samples_by_model.values())

    for spec in selected_models:
        model_generated_output_path = (
            build_model_generated_output_path(models_output_dir, spec.alias)
            if models_output_dir
            else build_model_generated_output_path(output_dir, spec.alias)
        )
        model_generations_paths: List[Path] = [model_generated_output_path]
        if len(selected_models) > 1:
            model_generations_paths.append(build_model_generated_output_path(output_dir, spec.alias))
        model_content_csv_path = output_dir / f"{spec.alias}-content.csv"
        model_calls_jsonl_path = output_dir / f"{spec.alias}-calls.jsonl"
        model_manifest_path = output_dir / f"{spec.alias}-run_manifest.json"

        pending_samples = pending_samples_by_model[spec.alias]
        if not pending_samples:
            print(
                f"[resume] {spec.alias}: no pending samples remain after loading existing outputs",
                file=sys.stderr,
            )

        for chunk_start, chunk_end, chunk_samples in iter_sample_chunks(pending_samples, args.chunk_size):
            current_chunk_name = chunk_name(chunk_start, chunk_end)
            chunk_dir = output_dir / "chunks" / spec.alias / current_chunk_name
            chunk_generations_csv_path = chunk_dir / "generated_outputs.csv"
            chunk_calls_jsonl_path = chunk_dir / "calls.jsonl"
            chunk_content_csv_path = chunk_dir / f"{spec.alias}-content.csv"
            chunk_manifest_path = chunk_dir / "run_manifest.json"
            chunk_rows: List[GenerationRow] = []

            for sample_batch in iter_batches(
                chunk_samples,
                args.local_batch_size if spec.provider == "local_hf" else 1,
            ):
                batch_results: List[Any]
                try:
                    batch_results = generate_batch_for_model(
                        spec,
                        [sample.prompt_text for sample in sample_batch],
                        args.system_prompt,
                        args,
                    )
                except Exception as batch_exc:
                    if spec.provider == "local_hf" and len(sample_batch) > 1:
                        print(
                            f"[warn] {spec.alias} batch failed for {len(sample_batch)} samples; "
                            f"retrying sequentially: {batch_exc}",
                            file=sys.stderr,
                        )
                        batch_results = []
                        for sample in sample_batch:
                            try:
                                batch_results.append(
                                    generate_text_for_model(
                                        spec, sample.prompt_text, args.system_prompt, args
                                    )
                                )
                            except Exception as exc:
                                batch_results.append(exc)
                    else:
                        batch_results = [batch_exc]

                for sample, result in zip(sample_batch, batch_results):
                    try:
                        if isinstance(result, Exception):
                            raise result

                        generated_text, raw_payload = result
                        generated_subject, generated_body = split_generated_email_fields(generated_text)
                        row = GenerationRow(
                            join_key=sample.join_key,
                            input_row_number=sample.input_row_number,
                            reference_row_number=sample.reference_row_number,
                            model_alias=spec.alias,
                            company=spec.company,
                            family=spec.family,
                            access_type=spec.access_type,
                            provider=spec.provider,
                            prompt_text=sample.prompt_text,
                            generated_text=generated_text,
                            generated_subject=generated_subject,
                            generated_body=generated_body,
                            reference_text=sample.reference_text,
                            label=sample.label,
                            created_at=now_utc_iso(),
                        )
                        generations.append(row)
                        generations_by_model[spec.alias].append(row)
                        chunk_rows.append(row)
                        call_record = {
                            "created_at": now_utc_iso(),
                            "model_alias": spec.alias,
                            "chunk_name": current_chunk_name,
                            "join_key": sample.join_key,
                            "prompt_text": sample.prompt_text,
                            "reference_text": sample.reference_text,
                            "label": sample.label,
                            "response_payload": raw_payload,
                            "status": "ok",
                        }
                        append_jsonl(calls_jsonl_path, call_record)
                        append_jsonl(model_calls_jsonl_path, call_record)
                        append_jsonl(chunk_calls_jsonl_path, call_record)
                    except Exception as exc:
                        error_record = {
                            "created_at": now_utc_iso(),
                            "model_alias": spec.alias,
                            "chunk_name": current_chunk_name,
                            "join_key": sample.join_key,
                            "prompt_text": sample.prompt_text,
                            "reference_text": sample.reference_text,
                            "label": sample.label,
                            "status": "error",
                            "error": str(exc),
                        }
                        append_jsonl(calls_jsonl_path, error_record)
                        append_jsonl(model_calls_jsonl_path, error_record)
                        append_jsonl(chunk_calls_jsonl_path, error_record)
                        print(
                            f"[warn] {spec.alias} failed on sample {sample.join_key}: {exc}",
                            file=sys.stderr,
                        )
                        completed += 1
                        continue

                    completed += 1

                    if args.print_every > 0 and completed % args.print_every == 0:
                        print(
                            f"[progress] completed {completed}/{total} generations for {spec.alias} on sample {sample.join_key}",
                            file=sys.stderr,
                        )

                    if args.save_every > 0 and completed % args.save_every == 0:
                        current_chunk_manifest = build_chunk_manifest(
                            args=args,
                            spec=spec,
                            chunk_dir=chunk_dir,
                            chunk_start=chunk_start,
                            chunk_end=chunk_end,
                            chunk_generations=chunk_rows,
                        )
                        persist_model_run_state(
                            args=args,
                            spec=spec,
                            all_generations_path=generations_csv_path,
                            all_generations=generations,
                            model_generations_paths=model_generations_paths,
                            model_generations=generations_by_model[spec.alias],
                            model_content_path=model_content_csv_path,
                            model_manifest_path=model_manifest_path,
                            output_dir=output_dir,
                            chunk_generations_path=chunk_generations_csv_path,
                            chunk_generations=chunk_rows,
                            chunk_content_path=chunk_content_csv_path,
                            chunk_manifest_path=chunk_manifest_path,
                            chunk_manifest=current_chunk_manifest,
                        )

                    if args.sleep_seconds > 0:
                        time.sleep(args.sleep_seconds)

            chunk_manifest = build_chunk_manifest(
                args=args,
                spec=spec,
                chunk_dir=chunk_dir,
                chunk_start=chunk_start,
                chunk_end=chunk_end,
                chunk_generations=chunk_rows,
            )
            persist_model_run_state(
                args=args,
                spec=spec,
                all_generations_path=generations_csv_path,
                all_generations=generations,
                model_generations_paths=model_generations_paths,
                model_generations=generations_by_model[spec.alias],
                model_content_path=model_content_csv_path,
                model_manifest_path=model_manifest_path,
                output_dir=output_dir,
                chunk_generations_path=chunk_generations_csv_path,
                chunk_generations=chunk_rows,
                chunk_content_path=chunk_content_csv_path,
                chunk_manifest_path=chunk_manifest_path,
                chunk_manifest=chunk_manifest,
            )

            print(
                f"[chunk] saved {spec.alias} {current_chunk_name} with {len(chunk_rows)}/{len(chunk_samples)} successful generations",
                file=sys.stderr,
            )

    write_generation_csv_targets(generations, [generations_csv_path])

    manifest = build_manifest(args, selected_models, len(samples), output_dir)
    write_json(manifest_path, manifest)

    print(f"Wrote generated outputs to {generations_csv_path}", file=sys.stderr)
    print(f"Saved call logs to {calls_jsonl_path}", file=sys.stderr)
    print(f"Saved manifest to {manifest_path}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception as exc:
        print(f"[error] {exc}", file=sys.stderr)
        raise SystemExit(1)
