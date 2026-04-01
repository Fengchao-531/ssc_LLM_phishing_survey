#!/usr/bin/env python3
"""Generate texts with multiple LLMs and compare them against a reference CSV.

This script is designed for the S8 Model-driven Automation setting:

1. Read an input CSV that contains generation prompts or model inputs.
2. Run multiple LLMs over the same inputs.
3. Compare each generated output against a second reference CSV.
4. Save both the generated dataset contents and evaluation metrics.

Supported metric groups:
- Pairwise row-level: BLEU, ROUGE-1
- Fluency/model-fit: Perplexity
- Corpus-level: Topic Coherence

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
from collections import Counter
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, Iterator, List, Optional, Sequence, Tuple
from urllib import error, parse, request


csv.field_size_limit(10**9)


SCRIPT_DIR = Path(__file__).resolve().parent
DEFAULT_TIMEOUT = 180
DEFAULT_OUTPUT_TOKENS = 1024
DEFAULT_PERPLEXITY_MODEL = "gpt2"
DEFAULT_TOPIC_COUNT = 5
DEFAULT_TOPIC_TOP_WORDS = 10
DEFAULT_TOP_MODELS = [
    "gpt-5.4",
    "claude-sonnet-4",
    "gemini-2.5-pro",
    "deepseek-v3.2-chat",
    "llama-4-scout",
    "mistral-small-3.2",
]

WORD_RE = re.compile(r"[A-Za-z0-9_]+")


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
    reference_text: str
    bleu: float
    rouge1_precision: float
    rouge1_recall: float
    rouge1_f1: float
    generated_perplexity: float
    reference_perplexity: float
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
        alias="deepseek-v3.2-chat",
        company="DeepSeek",
        family="DeepSeek",
        access_type="black_box",
        provider="openai_chat",
        model_name="deepseek-chat",
        api_base_url="https://api.deepseek.com/v1",
        api_key_env="DEEPSEEK_API_KEY",
        enabled_by_default=True,
        notes="`deepseek-chat` currently serves DeepSeek-V3.2.",
    ),
    ModelSpec(
        alias="deepseek-reasoner",
        company="DeepSeek",
        family="DeepSeek",
        access_type="black_box",
        provider="openai_chat",
        model_name="deepseek-reasoner",
        api_base_url="https://api.deepseek.com/v1",
        api_key_env="DEEPSEEK_API_KEY",
        notes="DeepSeek reasoning endpoint.",
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
        description="Run multiple LLMs on an input CSV and compare outputs to a reference CSV."
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
        "--disable-perplexity",
        action="store_true",
        help="Skip perplexity scoring.",
    )
    parser.add_argument(
        "--perplexity-model",
        default=DEFAULT_PERPLEXITY_MODEL,
        help="Local scorer model used for perplexity, e.g. gpt2.",
    )
    parser.add_argument(
        "--disable-topic-coherence",
        action="store_true",
        help="Skip topic coherence scoring.",
    )
    parser.add_argument(
        "--topic-count",
        type=int,
        default=DEFAULT_TOPIC_COUNT,
        help="Number of topics used for corpus-level topic coherence.",
    )
    parser.add_argument(
        "--topic-top-words",
        type=int,
        default=DEFAULT_TOPIC_TOP_WORDS,
        help="Top words per topic used for topic coherence.",
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
    return parser.parse_args()


def validate_args(args: argparse.Namespace) -> None:
    if args.list_models:
        return
    if not args.input:
        raise ValueError("Missing --input CSV path.")
    if args.max_output_tokens <= 0:
        raise ValueError("--max-output-tokens must be greater than 0.")
    if args.timeout <= 0:
        raise ValueError("--timeout must be greater than 0.")
    if args.topic_count <= 0:
        raise ValueError("--topic-count must be greater than 0.")
    if args.topic_top_words <= 1:
        raise ValueError("--topic-top-words must be greater than 1.")


def now_utc_iso() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat()


def normalize_text(value: str) -> str:
    text = (value or "").replace("\r\n", "\n").replace("\r", "\n")
    text = text.replace("\ufeff", "").replace("\u200b", "").replace("\xa0", " ")
    text = re.sub(r"[ \t]+\n", "\n", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


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
        "reference_text",
        "bleu",
        "rouge1_precision",
        "rouge1_recall",
        "rouge1_f1",
        "generated_perplexity",
        "reference_perplexity",
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
                    "reference_text": row.reference_text,
                    "bleu": safe_float(row.bleu),
                    "rouge1_precision": safe_float(row.rouge1_precision),
                    "rouge1_recall": safe_float(row.rouge1_recall),
                    "rouge1_f1": safe_float(row.rouge1_f1),
                    "generated_perplexity": safe_float(row.generated_perplexity),
                    "reference_perplexity": safe_float(row.reference_perplexity),
                    "created_at": row.created_at,
                }
            )


def write_summary_csv(path: Path, rows: Sequence[Dict[str, Any]]) -> None:
    if not rows:
        return
    fieldnames = list(rows[0].keys())
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


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
            aligned.append(
                Sample(
                    join_key=join_key,
                    input_row_number=input_row_number,
                    reference_row_number=reference_row_number,
                    prompt_text=prompt_text,
                    reference_text=reference_text,
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
            aligned.append(
                Sample(
                    join_key=str(index),
                    input_row_number=input_row_number,
                    reference_row_number=reference_row_number,
                    prompt_text=prompt_text,
                    reference_text=reference_text,
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


def get_local_generator(spec: ModelSpec, args: argparse.Namespace) -> Any:
    cache_key = f"{spec.alias}|{args.hf_device_map}|{args.hf_dtype}"
    if cache_key in _LOCAL_GENERATORS:
        return _LOCAL_GENERATORS[cache_key]

    try:
        from transformers import pipeline  # type: ignore
    except ImportError as exc:
        raise RuntimeError(
            "transformers is required for local Hugging Face models. "
            "Install it with: python3 -m pip install transformers torch"
        ) from exc

    generator = pipeline(
        "text-generation",
        model=spec.hf_model_id or spec.model_name,
        tokenizer=spec.hf_model_id or spec.model_name,
        device_map=args.hf_device_map,
        torch_dtype=resolve_torch_dtype(args.hf_dtype),
    )
    _LOCAL_GENERATORS[cache_key] = generator
    return generator


def generate_with_local_hf(
    spec: ModelSpec, prompt_text: str, system_prompt: str, args: argparse.Namespace
) -> Tuple[str, Dict[str, Any]]:
    generator = get_local_generator(spec, args)
    chat_messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": prompt_text},
    ]
    result = generator(
        chat_messages,
        max_new_tokens=args.max_output_tokens,
        temperature=args.temperature,
        do_sample=args.temperature > 0,
        return_full_text=False,
    )
    if not result:
        raise ValueError(f"No output returned by local model {spec.alias}")
    first = result[0]
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
    payload = {"provider": "local_hf", "raw_result": result}
    return normalize_text(text), payload


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
        "prompt_column": args.prompt_column,
        "reference_column": args.reference_column,
        "id_column": args.id_column,
        "reference_id_column": args.reference_id_column,
        "sample_count": sample_count,
        "models": [asdict(spec) for spec in selected_models],
        "metrics": {
            "row_level": ["BLEU", "ROUGE-1", "Perplexity"],
            "corpus_level": [] if args.disable_topic_coherence else ["Topic Coherence"],
        },
        "notes": [
            "BLEU and ROUGE-1 are computed pairwise against the aligned reference row.",
            "Perplexity is computed with a separate local scorer model, not the generation model itself.",
            "Topic coherence is corpus-level and is reported separately for generated texts and reference texts.",
        ],
    }


def main() -> int:
    args = parse_args()
    validate_args(args)

    if args.list_models:
        print_model_catalog()
        return 0

    selected_models = select_models(args)
    samples = align_samples(args)
    output_dir = build_output_dir(args.output_dir)
    generations_csv_path = output_dir / "generated_outputs.csv"
    summary_csv_path = output_dir / "metrics_summary.csv"
    calls_jsonl_path = output_dir / "calls.jsonl"
    manifest_path = output_dir / "run_manifest.json"

    generations: List[GenerationRow] = []
    completed = 0
    total = len(samples) * len(selected_models)

    for spec in selected_models:
        for sample in samples:
            try:
                generated_text, raw_payload = generate_text_for_model(
                    spec, sample.prompt_text, args.system_prompt, args
                )
                bleu = compute_bleu(sample.reference_text, generated_text)
                rouge_precision, rouge_recall, rouge_f1 = compute_rouge1(
                    sample.reference_text, generated_text
                )

                if args.disable_perplexity:
                    generated_perplexity = float("nan")
                    reference_perplexity = float("nan")
                else:
                    generated_perplexity = compute_perplexity(
                        generated_text, args.perplexity_model
                    )
                    reference_perplexity = compute_perplexity(
                        sample.reference_text, args.perplexity_model
                    )

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
                    reference_text=sample.reference_text,
                    bleu=bleu,
                    rouge1_precision=rouge_precision,
                    rouge1_recall=rouge_recall,
                    rouge1_f1=rouge_f1,
                    generated_perplexity=generated_perplexity,
                    reference_perplexity=reference_perplexity,
                    created_at=now_utc_iso(),
                )
                generations.append(row)
                append_jsonl(
                    calls_jsonl_path,
                    {
                        "created_at": now_utc_iso(),
                        "model_alias": spec.alias,
                        "join_key": sample.join_key,
                        "prompt_text": sample.prompt_text,
                        "reference_text": sample.reference_text,
                        "response_payload": raw_payload,
                        "status": "ok",
                    },
                )
            except Exception as exc:
                append_jsonl(
                    calls_jsonl_path,
                    {
                        "created_at": now_utc_iso(),
                        "model_alias": spec.alias,
                        "join_key": sample.join_key,
                        "prompt_text": sample.prompt_text,
                        "reference_text": sample.reference_text,
                        "status": "error",
                        "error": str(exc),
                    },
                )
                print(
                    f"[warn] {spec.alias} failed on sample {sample.join_key}: {exc}",
                    file=sys.stderr,
                )
                completed += 1
                continue

            completed += 1

            if args.save_every > 0 and completed % args.save_every == 0:
                write_generation_csv(generations_csv_path, generations)

            if args.print_every > 0 and completed % args.print_every == 0:
                print(
                    f"[progress] completed {completed}/{total} generations",
                    file=sys.stderr,
                )

            if args.sleep_seconds > 0:
                time.sleep(args.sleep_seconds)

    write_generation_csv(generations_csv_path, generations)
    summary_rows = build_summary_rows(generations, args)
    write_summary_csv(summary_csv_path, summary_rows)

    manifest = build_manifest(args, selected_models, len(samples), output_dir)
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    with manifest_path.open("w", encoding="utf-8") as handle:
        json.dump(manifest, handle, ensure_ascii=False, indent=2)
        handle.write("\n")

    print(f"Wrote generated outputs to {generations_csv_path}", file=sys.stderr)
    print(f"Wrote metric summary to {summary_csv_path}", file=sys.stderr)
    print(f"Saved call logs to {calls_jsonl_path}", file=sys.stderr)
    print(f"Saved manifest to {manifest_path}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception as exc:
        print(f"[error] {exc}", file=sys.stderr)
        raise SystemExit(1)
