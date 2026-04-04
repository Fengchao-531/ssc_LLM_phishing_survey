#!/usr/bin/env python3
"""Fill square-bracket placeholders in subject/body text with Llama 4.

This script only edits text that appears inside square brackets, such as:
- []
- [, ]
- [NAME]

All non-bracket text is preserved byte-for-byte after CSV normalization.
The model is asked to infer realistic values from the surrounding context and
return only replacement strings. The script then applies those replacements to
the original text locally, so the model never rewrites the full message.
"""

import argparse
import csv
import itertools
import json
import os
import re
import sys
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterator, List, Optional, Sequence, Tuple


csv.field_size_limit(10**9)


SCRIPT_DIR = Path(__file__).resolve().parent
DEFAULT_MODEL_ALIAS = "llama-4-scout"
DEFAULT_MODEL_NAME = "meta-llama/Llama-4-Scout-17B-16E-Instruct"
DEFAULT_OUTPUT_DIR = str((SCRIPT_DIR / "Models-Output").resolve())
DEFAULT_OUTPUT_NAME = "llama-4-bracket-filled.csv"
DEFAULT_CONTEXT_CHARS = 120
DEFAULT_MAX_TEXT_CHARS = 12000
DEFAULT_MAX_RESPONSE_TOKENS = 768
DEFAULT_TIMEOUT = 180
DEFAULT_SAVE_EVERY = 1
DEFAULT_PRINT_EVERY = 10
DEFAULT_HF_MODEL_ROOT = os.environ.get("HF_MODEL_ROOT", "").strip()
DEFAULT_RECENT_REPLACEMENT_LIMIT = 12

PLACEHOLDER_RE = re.compile(r"\[([^\[\]]*)\]")
EMPTY_ARIA_HIDDEN_BLOCK_RE = re.compile(
    r"<(?:div|p|span)\b[^>]*\baria-hidden\s*=\s*[\"']+true[\"']+[^>]*>"
    r"\s*(?:&nbsp;|<br\s*/?>|\s)*\s*</(?:div|p|span)>",
    re.IGNORECASE,
)


@dataclass
class TextReplacementRequest:
    field_name: str
    original_text: str
    matches: List[re.Match[str]]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Use Llama 4 to fill square-bracket placeholders in subject/body text."
    )
    parser.add_argument("--input", default="", help="Input CSV path.")
    parser.add_argument(
        "--input-dir",
        default="",
        help="Optional directory of CSV files to process. Each file keeps its original name.",
    )
    parser.add_argument(
        "--output-dir",
        default=DEFAULT_OUTPUT_DIR,
        help="Directory where the filled CSV and logs will be written.",
    )
    parser.add_argument(
        "--output-name",
        default=DEFAULT_OUTPUT_NAME,
        help="Output CSV file name inside --output-dir.",
    )
    parser.add_argument(
        "--subject-column",
        default="Subject",
        help="Column name for subject text.",
    )
    parser.add_argument(
        "--body-column",
        default="Body",
        help="Column name for body text.",
    )
    parser.add_argument(
        "--generated-text-column",
        default="generated_text",
        help="Optional combined text column to regenerate when subject/body columns exist.",
    )
    parser.add_argument(
        "--label-column",
        default="label",
        help="Label column to drop from the output. Leave empty to keep it.",
    )
    parser.add_argument(
        "--keep-extra-columns",
        action="store_true",
        help="Keep non-label input columns in the output in addition to subject/body.",
    )
    parser.add_argument(
        "--max-rows",
        type=int,
        default=0,
        help="Optional cap on processed rows.",
    )
    parser.add_argument(
        "--context-chars",
        type=int,
        default=DEFAULT_CONTEXT_CHARS,
        help="Characters of left/right context to show the model for each placeholder.",
    )
    parser.add_argument(
        "--max-text-chars",
        type=int,
        default=DEFAULT_MAX_TEXT_CHARS,
        help="Maximum text length sent to the model for one field.",
    )
    parser.add_argument(
        "--max-response-tokens",
        type=int,
        default=DEFAULT_MAX_RESPONSE_TOKENS,
        help="Maximum generated tokens for the replacement JSON.",
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=0.2,
        help="Sampling temperature for the local model.",
    )
    parser.add_argument(
        "--timeout",
        type=int,
        default=DEFAULT_TIMEOUT,
        help="Reserved for interface symmetry. Local generation does not currently enforce it.",
    )
    parser.add_argument(
        "--save-every",
        type=int,
        default=DEFAULT_SAVE_EVERY,
        help="Checkpoint output CSV every N processed rows.",
    )
    parser.add_argument(
        "--print-every",
        type=int,
        default=DEFAULT_PRINT_EVERY,
        help="Progress logging interval in processed rows.",
    )
    parser.add_argument(
        "--hf-model-root",
        default=DEFAULT_HF_MODEL_ROOT,
        help="Optional local Hugging Face model root. Falls back to model id if missing.",
    )
    parser.add_argument(
        "--hf-device-map",
        default="auto",
        help="Device map for local Hugging Face loading.",
    )
    parser.add_argument(
        "--hf-dtype",
        default="auto",
        help="torch dtype hint for local Hugging Face loading: auto, float16, bfloat16, float32.",
    )
    parser.add_argument(
        "--model-alias",
        default=DEFAULT_MODEL_ALIAS,
        help="Model alias recorded in manifests.",
    )
    parser.add_argument(
        "--model-name",
        default=DEFAULT_MODEL_NAME,
        help="Hugging Face model id or local model path name to load.",
    )
    return parser.parse_args()


def validate_args(args: argparse.Namespace) -> None:
    if not args.input and not args.input_dir:
        raise ValueError("Pass either --input or --input-dir.")
    if args.input and args.input_dir:
        raise ValueError("Use only one of --input or --input-dir.")
    if args.input:
        input_path = Path(args.input).resolve()
        if not input_path.exists():
            raise FileNotFoundError(f"Input CSV not found: {input_path}")
    if args.input_dir:
        input_dir = Path(args.input_dir).resolve()
        if not input_dir.exists():
            raise FileNotFoundError(f"Input directory not found: {input_dir}")
    if args.context_chars <= 0:
        raise ValueError("--context-chars must be greater than 0.")
    if args.max_text_chars <= 0:
        raise ValueError("--max-text-chars must be greater than 0.")
    if args.max_response_tokens <= 0:
        raise ValueError("--max-response-tokens must be greater than 0.")
    if args.save_every <= 0:
        raise ValueError("--save-every must be greater than 0.")
    if args.print_every <= 0:
        raise ValueError("--print-every must be greater than 0.")


def now_utc_iso() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat()


def normalize_text(value: str) -> str:
    text = (value or "").replace("\r\n", "\n").replace("\r", "\n")
    text = EMPTY_ARIA_HIDDEN_BLOCK_RE.sub("", text)
    text = text.replace("\ufeff", "").replace("\u200b", "").replace("\xa0", " ")
    text = re.sub(r"[ \t]+\n", "\n", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def truncate_text(text: str, max_chars: int) -> str:
    if len(text) <= max_chars:
        return text
    return text[:max_chars].rstrip()


def normalize_placeholder_memory_key(value: str) -> str:
    cleaned = re.sub(r"\s+", " ", str(value or "").strip().lower())
    return cleaned or "__empty__"


def iter_csv_rows(path: Path) -> Iterator[Tuple[int, Dict[str, str]]]:
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        if not reader.fieldnames:
            raise ValueError(f"CSV has no header row: {path}")
        for row_number, row in enumerate(reader, start=2):
            yield row_number, row


def get_hf_token() -> str:
    for env_name in ("HF_TOKEN", "HUGGING_FACE_HUB_TOKEN", "HUGGINGFACE_TOKEN"):
        value = os.environ.get(env_name, "").strip()
        if value:
            return value
    return ""


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


def resolve_model_source(model_name: str, hf_model_root: str) -> str:
    root = Path(hf_model_root).expanduser().resolve() if hf_model_root else None
    if root:
        candidates = [
            root / model_name,
            root / model_name.replace("/", "--"),
            root / model_name.split("/")[-1],
        ]
        for candidate in candidates:
            if candidate.exists():
                return str(candidate)
    return model_name


def load_hf_component(loader: Any, source: str, **kwargs: Any) -> Any:
    token = get_hf_token()
    if not token:
        return loader.from_pretrained(source, **kwargs)
    try:
        return loader.from_pretrained(source, token=token, **kwargs)
    except TypeError:
        return loader.from_pretrained(source, use_auth_token=token, **kwargs)


def get_local_generator(args: argparse.Namespace) -> Any:
    try:
        from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline  # type: ignore
    except ImportError as exc:
        raise RuntimeError(
            "transformers is required for local generation. "
            "Install it with: python3 -m pip install transformers torch"
        ) from exc

    model_source = resolve_model_source(args.model_name, args.hf_model_root)
    tokenizer = load_hf_component(AutoTokenizer, model_source)
    model = load_hf_component(
        AutoModelForCausalLM,
        model_source,
        device_map=args.hf_device_map,
        torch_dtype=resolve_torch_dtype(args.hf_dtype),
    )
    return pipeline("text-generation", model=model, tokenizer=tokenizer)


def build_placeholder_entries(
    text: str,
    matches: Sequence[re.Match[str]],
    context_chars: int,
    recent_replacements: Optional[Dict[str, List[str]]] = None,
) -> List[Dict[str, Any]]:
    entries: List[Dict[str, Any]] = []
    for index, match in enumerate(matches):
        start, end = match.span()
        left = text[max(0, start - context_chars) : start]
        right = text[end : min(len(text), end + context_chars)]
        memory_key = normalize_placeholder_memory_key(match.group(1))
        recent_values = list((recent_replacements or {}).get(memory_key, []))
        entries.append(
            {
                "index": index,
                "original_bracket_text": match.group(0),
                "current_inner_text": match.group(1),
                "left_context": left,
                "right_context": right,
                "avoid_recent_values": recent_values,
            }
        )
    return entries


def build_messages(
    field_name: str,
    text: str,
    matches: Sequence[re.Match[str]],
    args: argparse.Namespace,
    recent_replacements: Optional[Dict[str, List[str]]] = None,
) -> List[Dict[str, str]]:
    prompt_text = truncate_text(text, args.max_text_chars)
    entries = build_placeholder_entries(
        prompt_text,
        matches,
        args.context_chars,
        recent_replacements=recent_replacements,
    )
    system_prompt = (
        "You fill square-bracket placeholders in email text. "
        "Infer realistic, context-appropriate contents for every bracketed item. "
        "Only return JSON with replacement strings. "
        "Do not include brackets in the returned values. "
        "Do not explain. "
        "Do not rewrite surrounding text. "
        "Avoid obvious dummy sequences like 123456789. "
        "Prefer varied replacements across rows. "
        "If placeholders represent independent entities, avoid reusing the same names, companies, ids, or values when recent alternatives were already used. "
        "Inside one field, prefer distinct replacements for different placeholder indices unless the context clearly indicates they should be the same entity."
    )
    user_prompt = {
        "task": "Fill every square-bracket item with suitable content.",
        "field_name": field_name,
        "text_excerpt": prompt_text,
        "placeholder_count": len(entries),
        "diversity_hint": (
            "Each placeholder entry may include avoid_recent_values. "
            "Treat them as soft do-not-repeat hints when reasonable."
        ),
        "placeholders": entries,
        "response_format": {
            "replacements": [
                {"index": 0, "value": "replacement without brackets"},
            ]
        },
    }
    return [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": json.dumps(user_prompt, ensure_ascii=False, indent=2)},
    ]


def extract_generated_text(result: Any) -> str:
    if not result:
        raise ValueError("No output returned by the generator.")
    first = result[0]
    if isinstance(first, dict):
        payload = first.get("generated_text", first)
    else:
        payload = first

    if isinstance(payload, str):
        return payload.strip()
    if isinstance(payload, list):
        parts: List[str] = []
        for item in payload:
            if isinstance(item, dict) and item.get("role") == "assistant":
                parts.append(str(item.get("content", "")))
            elif isinstance(item, dict) and "content" in item:
                parts.append(str(item.get("content", "")))
            else:
                parts.append(str(item))
        return "\n".join(parts).strip()
    return str(payload).strip()


def extract_json_object(text: str) -> Dict[str, Any]:
    cleaned = text.strip()
    if cleaned.startswith("```"):
        cleaned = re.sub(r"^```(?:json)?\s*", "", cleaned)
        cleaned = re.sub(r"\s*```$", "", cleaned)

    try:
        parsed = json.loads(cleaned)
        if isinstance(parsed, dict):
            return parsed
    except json.JSONDecodeError:
        pass

    for match in re.finditer(r"\{", cleaned):
        start = match.start()
        snippet = cleaned[start:]
        for end in range(len(snippet), 0, -1):
            candidate = snippet[:end]
            try:
                parsed = json.loads(candidate)
                if isinstance(parsed, dict):
                    return parsed
            except json.JSONDecodeError:
                continue
    raise ValueError("Could not parse a JSON object from the model response.")


def sanitize_replacement(value: str) -> str:
    cleaned = str(value or "").strip()
    if cleaned.startswith("[") and cleaned.endswith("]") and len(cleaned) >= 2:
        cleaned = cleaned[1:-1].strip()
    cleaned = cleaned.replace("\n", " ").strip()
    return cleaned


def apply_replacements(
    original_text: str,
    matches: Sequence[re.Match[str]],
    replacements: Sequence[str],
) -> str:
    if len(matches) != len(replacements):
        raise ValueError(
            f"Replacement count mismatch: expected {len(matches)}, got {len(replacements)}"
        )

    pieces: List[str] = []
    cursor = 0
    for match, replacement in zip(matches, replacements):
        start, end = match.span()
        pieces.append(original_text[cursor:start])
        pieces.append(replacement)
        cursor = end
    pieces.append(original_text[cursor:])
    return "".join(pieces)


def request_replacements(
    generator: Any,
    field_name: str,
    text: str,
    args: argparse.Namespace,
    recent_replacements: Optional[Dict[str, List[str]]] = None,
) -> Tuple[str, Dict[str, Any]]:
    matches = list(PLACEHOLDER_RE.finditer(text))
    if not matches:
        return text, {"status": "no_placeholders", "placeholder_count": 0}

    prompt_text = truncate_text(text, args.max_text_chars)
    prompt_matches = list(PLACEHOLDER_RE.finditer(prompt_text))
    if len(prompt_matches) != len(matches):
        raise ValueError(
            f"{field_name} exceeds --max-text-chars before all placeholders fit. "
            "Increase --max-text-chars."
        )

    messages = build_messages(
        field_name,
        text,
        matches,
        args,
        recent_replacements=recent_replacements,
    )
    result = generator(
        messages,
        max_new_tokens=args.max_response_tokens,
        temperature=args.temperature,
        do_sample=args.temperature > 0,
        return_full_text=False,
    )
    response_text = extract_generated_text(result)
    parsed = extract_json_object(response_text)
    raw_replacements = parsed.get("replacements")
    if not isinstance(raw_replacements, list):
        raise ValueError("Model response JSON is missing a replacements list.")

    replacements_by_index: Dict[int, str] = {}
    for item in raw_replacements:
        if not isinstance(item, dict):
            continue
        index = item.get("index")
        value = item.get("value")
        if not isinstance(index, int):
            continue
        replacements_by_index[index] = sanitize_replacement(str(value or ""))

    replacements: List[str] = []
    for index in range(len(matches)):
        if index not in replacements_by_index:
            raise ValueError(f"Missing replacement for placeholder index {index}")
        replacements.append(replacements_by_index[index])

    filled_text = apply_replacements(text, matches, replacements)
    payload = {
        "status": "ok",
        "placeholder_count": len(matches),
        "response_text": response_text,
        "parsed_response": parsed,
        "replacements": replacements,
    }
    return filled_text, payload


def remember_replacements(
    recent_replacements: Dict[str, List[str]],
    matches: Sequence[re.Match[str]],
    replacements: Sequence[str],
    limit: int = DEFAULT_RECENT_REPLACEMENT_LIMIT,
) -> None:
    for match, replacement in zip(matches, replacements):
        cleaned = sanitize_replacement(replacement)
        if not cleaned:
            continue
        key = normalize_placeholder_memory_key(match.group(1))
        values = recent_replacements.setdefault(key, [])
        if cleaned in values:
            values.remove(cleaned)
        values.append(cleaned)
        if len(values) > limit:
            del values[:-limit]


def append_jsonl(path: Path, record: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(record, ensure_ascii=False) + "\n")


def write_csv(path: Path, fieldnames: Sequence[str], rows: Sequence[Dict[str, str]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(fieldnames))
        writer.writeheader()
        for row in rows:
            writer.writerow({name: row.get(name, "") for name in fieldnames})


def write_csv_atomic(path: Path, fieldnames: Sequence[str], rows: Sequence[Dict[str, str]]) -> None:
    temp_path = path.with_name(f".{path.name}.tmp")
    write_csv(temp_path, fieldnames, rows)
    temp_path.replace(path)


def build_output_fieldnames(
    input_fieldnames: Sequence[str],
    subject_column: str,
    body_column: str,
    label_column: str,
    keep_extra_columns: bool,
) -> List[str]:
    if not keep_extra_columns:
        return [subject_column, body_column]

    fieldnames: List[str] = []
    for name in input_fieldnames:
        if label_column and name == label_column:
            continue
        fieldnames.append(name)

    if subject_column not in fieldnames:
        fieldnames.append(subject_column)
    if body_column not in fieldnames:
        fieldnames.append(body_column)
    return fieldnames


def detect_text_columns(
    input_fieldnames: Sequence[str],
    subject_column: str,
    body_column: str,
    generated_text_column: str,
) -> Tuple[str, str, str]:
    available = set(input_fieldnames)
    resolved_subject = subject_column
    resolved_body = body_column
    resolved_generated = generated_text_column if generated_text_column in available else ""

    if resolved_subject in available and resolved_body in available:
        return resolved_subject, resolved_body, resolved_generated

    if "generated_subject" in available and "generated_body" in available:
        resolved_subject = "generated_subject"
        resolved_body = "generated_body"
        if "generated_text" in available:
            resolved_generated = "generated_text"
        return resolved_subject, resolved_body, resolved_generated

    if "Subject" in available and "Body" in available:
        resolved_subject = "Subject"
        resolved_body = "Body"
        return resolved_subject, resolved_body, resolved_generated

    raise ValueError(
        "Could not determine subject/body columns. "
        f"Available columns: {', '.join(input_fieldnames)}"
    )


def combine_subject_body(subject: str, body: str) -> str:
    subject_text = str(subject or "")
    body_text = str(body or "")
    if subject_text and body_text:
        return f"Subject: {subject_text}\n\n{body_text}"
    if subject_text:
        return f"Subject: {subject_text}"
    return body_text


def should_keep_extra_columns(args: argparse.Namespace) -> bool:
    return args.keep_extra_columns or bool(args.input_dir)


def should_drop_label(args: argparse.Namespace) -> bool:
    return bool(args.label_column) and not args.input_dir


def process_one_csv(
    input_path: Path,
    output_path: Path,
    args: argparse.Namespace,
    generator: Any,
) -> Dict[str, Any]:
    output_dir = output_path.parent
    calls_path = output_dir / f"{input_path.stem}-llama-4-bracket-fill-calls.jsonl"
    manifest_path = output_dir / f"{input_path.stem}-llama-4-bracket-fill-manifest.json"

    input_iter = iter_csv_rows(input_path)
    try:
        first_row_number, first_row = next(input_iter)
    except StopIteration:
        raise ValueError(f"No rows found in {input_path}")

    input_fieldnames = list(first_row.keys())
    keep_extra_columns = should_keep_extra_columns(args)
    drop_label = should_drop_label(args)
    subject_column, body_column, generated_text_column = detect_text_columns(
        input_fieldnames,
        args.subject_column,
        args.body_column,
        args.generated_text_column,
    )
    output_fieldnames = build_output_fieldnames(
        input_fieldnames,
        subject_column,
        body_column,
        args.label_column if drop_label else "",
        keep_extra_columns,
    )

    processed_rows: List[Dict[str, str]] = []
    completed = 0
    recent_replacements: Dict[str, List[str]] = {}

    for row_number, row in itertools.chain([(first_row_number, first_row)], input_iter):
        if args.max_rows > 0 and completed >= args.max_rows:
            break

        original_subject = normalize_text(str(row.get(subject_column, "")))
        original_body = normalize_text(str(row.get(body_column, "")))

        try:
            filled_subject, subject_payload = request_replacements(
                generator,
                subject_column,
                original_subject,
                args,
                recent_replacements=recent_replacements,
            )
            filled_body, body_payload = request_replacements(
                generator,
                body_column,
                original_body,
                args,
                recent_replacements=recent_replacements,
            )
            remember_replacements(
                recent_replacements,
                list(PLACEHOLDER_RE.finditer(original_subject)),
                subject_payload.get("replacements", []),
            )
            remember_replacements(
                recent_replacements,
                list(PLACEHOLDER_RE.finditer(original_body)),
                body_payload.get("replacements", []),
            )
            output_row = {}
            if keep_extra_columns:
                output_row.update(
                    {
                        key: value
                        for key, value in row.items()
                        if not drop_label or key != args.label_column
                    }
                )
            output_row[subject_column] = filled_subject
            output_row[body_column] = filled_body
            if generated_text_column:
                output_row[generated_text_column] = combine_subject_body(
                    filled_subject, filled_body
                )
            processed_rows.append(output_row)

            append_jsonl(
                calls_path,
                {
                    "created_at": now_utc_iso(),
                    "input_csv": str(input_path),
                    "row_number": row_number,
                    "subject_status": subject_payload.get("status"),
                    "subject_placeholder_count": subject_payload.get("placeholder_count", 0),
                    "body_status": body_payload.get("status"),
                    "body_placeholder_count": body_payload.get("placeholder_count", 0),
                    "subject_response_text": subject_payload.get("response_text", ""),
                    "body_response_text": body_payload.get("response_text", ""),
                    "status": "ok",
                },
            )
        except Exception as exc:
            append_jsonl(
                calls_path,
                {
                    "created_at": now_utc_iso(),
                    "row_number": row_number,
                    "subject": original_subject,
                    "body": truncate_text(original_body, 4000),
                    "status": "error",
                    "error": str(exc),
                },
            )
            print(f"[warn] failed row {row_number}: {exc}", file=sys.stderr)
            output_row = {}
            if keep_extra_columns:
                output_row.update(
                    {
                        key: value
                        for key, value in row.items()
                        if not drop_label or key != args.label_column
                    }
                )
            output_row[subject_column] = original_subject
            output_row[body_column] = original_body
            if generated_text_column:
                output_row[generated_text_column] = combine_subject_body(
                    original_subject, original_body
                )
            processed_rows.append(output_row)

        completed += 1

        if args.save_every > 0 and completed % args.save_every == 0:
            write_csv_atomic(output_path, output_fieldnames, processed_rows)

        if args.print_every > 0 and completed % args.print_every == 0:
            print(f"[progress] processed {completed} rows", file=sys.stderr)

    write_csv_atomic(output_path, output_fieldnames, processed_rows)

    manifest = {
        "created_at": now_utc_iso(),
        "input_csv": str(input_path),
        "output_csv": str(output_path),
        "calls_jsonl": str(calls_path),
        "model_alias": args.model_alias,
        "model_name": args.model_name,
        "subject_column": subject_column,
        "body_column": body_column,
        "generated_text_column": generated_text_column,
        "label_column_dropped": args.label_column if drop_label else "",
        "keep_extra_columns": keep_extra_columns,
        "processed_rows": completed,
        "context_chars": args.context_chars,
        "max_text_chars": args.max_text_chars,
        "max_response_tokens": args.max_response_tokens,
        "save_every": args.save_every,
        "notes": [
            "Only text inside square brackets is replaced.",
            "Non-bracket text is preserved locally by span replacement.",
            "Rows that fail model parsing keep their original subject/body text.",
        ],
    }
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    with manifest_path.open("w", encoding="utf-8") as handle:
        json.dump(manifest, handle, ensure_ascii=False, indent=2)
        handle.write("\n")

    print(f"Wrote filled CSV to {output_path}", file=sys.stderr)
    print(f"Saved call logs to {calls_path}", file=sys.stderr)
    print(f"Saved manifest to {manifest_path}", file=sys.stderr)
    return {
        "input_csv": str(input_path),
        "output_csv": str(output_path),
        "processed_rows": completed,
        "subject_column": subject_column,
        "body_column": body_column,
        "generated_text_column": generated_text_column,
    }


def main() -> int:
    args = parse_args()
    validate_args(args)

    output_dir = Path(args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    generator = get_local_generator(args)

    if args.input_dir:
        input_paths = sorted(Path(args.input_dir).resolve().glob("*.csv"))
        if not input_paths:
            raise ValueError(f"No CSV files found in {Path(args.input_dir).resolve()}")
    else:
        input_paths = [Path(args.input).resolve()]

    processed_files: List[Dict[str, Any]] = []
    for input_path in input_paths:
        if args.input_dir:
            output_path = output_dir / input_path.name
        else:
            output_path = output_dir / args.output_name
        processed_files.append(process_one_csv(input_path, output_path, args, generator))

    summary_manifest_path = output_dir / "llama-4-bracket-fill-summary.json"
    summary = {
        "created_at": now_utc_iso(),
        "processed_file_count": len(processed_files),
        "files": processed_files,
        "notes": [
            "CSV file names are preserved when --input-dir is used.",
            "Only square-bracket placeholders are replaced.",
        ],
    }
    with summary_manifest_path.open("w", encoding="utf-8") as handle:
        json.dump(summary, handle, ensure_ascii=False, indent=2)
        handle.write("\n")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except KeyboardInterrupt:
        print("[error] interrupted", file=sys.stderr)
        raise SystemExit(130)
    except Exception as exc:
        print(f"[error] {exc}", file=sys.stderr)
        raise SystemExit(1)
