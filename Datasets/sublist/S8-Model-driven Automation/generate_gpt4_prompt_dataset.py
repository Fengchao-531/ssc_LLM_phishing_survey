#!/usr/bin/env python3
"""Generate prompt candidates from a CSV dataset with a GPT-4 class model.

For each input text, the script asks the model to write prompt candidates that
could generate a similar text from scratch. The prompts are intentionally kept
objective and should not mention any private business motivation or workflow
justification that is not observable from the source text itself.

Output format:
- One CSV row per generated prompt.
- Each source text produces `--prompt-count` rows.
- `label` is always set to `1` to represent confirmed prompts.
"""

import argparse
import csv
import json
import os
import re
import sys
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterator, List, Sequence, Tuple

try:
    from openai import APIConnectionError, APIStatusError, OpenAI
except ImportError:
    APIConnectionError = None
    APIStatusError = None
    OpenAI = None


csv.field_size_limit(10**9)

SCRIPT_DIR = Path(__file__).resolve().parent
DEFAULT_BASE_URL = "https://api.openai.com/v1"
DEFAULT_TIMEOUT = 120
DEFAULT_BATCH_SIZE = 10
DEFAULT_PROMPT_COUNT = 3
DEFAULT_RATE_LIMIT_PADDING = 2.0
DEFAULT_MAX_INPUT_CHARS = 12000


@dataclass
class SourceItem:
    row_number: int
    source_id: str
    source_text: str
    label: str


@dataclass
class OutputRow:
    row_number: int
    source_id: str
    source_text: str
    prompt_index: int
    prompt: str
    label: str
    model: str
    batch_index: int
    generated_at: str


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Batch-generate prompt candidates from a CSV dataset."
    )
    parser.add_argument(
        "--input",
        default="",
        help="Input CSV path. Kept optional so you can wire the interface now and provide the dataset later.",
    )
    parser.add_argument(
        "--output-dir",
        default="",
        help="Optional output directory. Defaults to S6-Stealthy Rewriting/runs/<timestamp>/",
    )
    parser.add_argument(
        "--output-name",
        default="prompt_candidates.csv",
        help="Output CSV file name inside --output-dir.",
    )
    parser.add_argument(
        "--model",
        default=os.environ.get("OPENAI_MODEL", "gpt-4o"),
        help="Model name passed to the OpenAI-compatible API.",
    )
    parser.add_argument(
        "--base-url",
        default=os.environ.get("OPENAI_BASE_URL", DEFAULT_BASE_URL),
        help="OpenAI-compatible API base URL.",
    )
    parser.add_argument(
        "--api-key",
        default=os.environ.get("OPENAI_API_KEY", ""),
        help="API key. Defaults to OPENAI_API_KEY.",
    )
    parser.add_argument(
        "--text-column",
        default="text",
        help="Primary text column name. If missing, the script falls back to subject/body composition.",
    )
    parser.add_argument(
        "--subject-column",
        default="Subject",
        help="Subject column name used when --text-column is absent or empty.",
    )
    parser.add_argument(
        "--body-column",
        default="Body",
        help="Body column name used when --text-column is absent or empty.",
    )
    parser.add_argument(
        "--id-column",
        default="",
        help="Optional column to preserve a source identifier.",
    )
    parser.add_argument(
        "--label-column",
        default="label",
        help="Optional label column to preserve in the output.",
    )
    parser.add_argument(
        "--row-number-column",
        default="row_number",
        help="Optional column containing the original source row number.",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=DEFAULT_BATCH_SIZE,
        help="How many texts to send in one API call.",
    )
    parser.add_argument(
        "--prompt-count",
        type=int,
        default=DEFAULT_PROMPT_COUNT,
        help="How many prompt variants to create for each input text.",
    )
    parser.add_argument(
        "--max-rows",
        type=int,
        default=0,
        help="Optional cap on the number of CSV rows to process.",
    )
    parser.add_argument(
        "--max-input-chars",
        type=int,
        default=DEFAULT_MAX_INPUT_CHARS,
        help="Maximum number of characters kept for each source text before truncation.",
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=0.8,
        help="Sampling temperature for diverse prompt candidates.",
    )
    parser.add_argument(
        "--max-completion-tokens",
        type=int,
        default=4000,
        help="Completion token cap forwarded to the API.",
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
        help="Delay between API calls.",
    )
    parser.add_argument(
        "--max-retries",
        type=int,
        default=3,
        help="Retry count for transient API errors.",
    )
    parser.add_argument(
        "--save-every",
        type=int,
        default=1,
        help="Write a checkpoint after every N completed batches.",
    )
    parser.add_argument(
        "--print-every",
        type=int,
        default=1,
        help="Progress logging interval in batches.",
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Resume from an existing output CSV in --output-dir when present.",
    )
    return parser.parse_args()


def validate_args(args: argparse.Namespace) -> None:
    if args.batch_size <= 0:
        raise ValueError("--batch-size must be greater than 0.")
    if args.prompt_count <= 0:
        raise ValueError("--prompt-count must be greater than 0.")
    if args.max_completion_tokens <= 0:
        raise ValueError("--max-completion-tokens must be greater than 0.")
    if args.max_retries <= 0:
        raise ValueError("--max-retries must be greater than 0.")
    if args.max_input_chars <= 0:
        raise ValueError("--max-input-chars must be greater than 0.")


def now_utc_iso() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat()


def normalize_text(value: str) -> str:
    text = (value or "").replace("\r\n", "\n").replace("\r", "\n")
    text = text.replace("\ufeff", "").replace("\u200b", "").replace("\xa0", " ")
    text = re.sub(r"[ \t]+\n", "\n", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def truncate_text(text: str, max_chars: int) -> str:
    if len(text) <= max_chars:
        return text
    return text[:max_chars].rstrip()


def render_text(subject: str, body: str) -> str:
    subject_clean = normalize_text(subject)
    body_clean = normalize_text(body)
    if subject_clean and body_clean:
        return f"Subject: {subject_clean}\n\n{body_clean}"
    if subject_clean:
        return f"Subject: {subject_clean}"
    return body_clean


def build_output_dir(output_dir_arg: str) -> Path:
    if output_dir_arg:
        return Path(output_dir_arg).resolve()
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    return (SCRIPT_DIR / "runs" / f"prompt_generation_{stamp}").resolve()


def chunked(items: Sequence[SourceItem], size: int) -> Iterator[List[SourceItem]]:
    if size <= 0:
        raise ValueError("--batch-size must be greater than 0.")
    for index in range(0, len(items), size):
        yield list(items[index : index + size])


def append_jsonl(path: Path, record: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(record, ensure_ascii=False) + "\n")


def parse_retry_after_seconds(error_text: str) -> float:
    match = re.search(r"Please try again in\s+([0-9]+(?:\.[0-9]+)?)s", error_text or "")
    if not match:
        return 0.0
    return float(match.group(1))


def write_csv(path: Path, rows: Sequence[OutputRow]) -> None:
    fieldnames = [
        "row_number",
        "source_id",
        "prompt_index",
        "prompt",
        "label",
    ]
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(
                {
                    "row_number": row.row_number,
                    "source_id": row.source_id,
                    "prompt_index": row.prompt_index,
                    "prompt": row.prompt,
                    "label": row.label,
                }
            )


def load_source_items(args: argparse.Namespace) -> List[SourceItem]:
    if not args.input:
        raise ValueError(
            "No input CSV was provided. Pass --input when your dataset is ready."
        )

    input_path = Path(args.input).resolve()
    if not input_path.exists():
        raise FileNotFoundError(f"Input CSV not found: {input_path}")

    with input_path.open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        if not reader.fieldnames:
            raise ValueError("Input CSV has no header row.")

        items: List[SourceItem] = []
        seen_ids = set()
        for row_number, row in enumerate(reader, start=2):
            if args.max_rows > 0 and len(items) >= args.max_rows:
                break

            original_row_number = row_number
            if args.row_number_column:
                raw_row_number = str(row.get(args.row_number_column, "")).strip()
                if raw_row_number:
                    try:
                        original_row_number = int(raw_row_number)
                    except ValueError:
                        raise ValueError(
                            f"Invalid integer in {args.row_number_column}: {raw_row_number}"
                        )

            source_id = ""
            if args.id_column:
                source_id = str(row.get(args.id_column, "")).strip()
            if not source_id:
                source_id = str(len(items) + 1)
            if source_id in seen_ids:
                source_id = f"{source_id}__row{row_number}"
            seen_ids.add(source_id)

            text_value = normalize_text(str(row.get(args.text_column, "")))
            if not text_value:
                text_value = render_text(
                    str(row.get(args.subject_column, "")),
                    str(row.get(args.body_column, "")),
                )
                text_value = normalize_text(text_value)

            if not text_value:
                continue

            text_value = truncate_text(text_value, args.max_input_chars)

            label_value = "1"
            if args.label_column:
                raw_label = str(row.get(args.label_column, "")).strip()
                if raw_label:
                    label_value = raw_label

            items.append(
                SourceItem(
                    row_number=original_row_number,
                    source_id=source_id,
                    source_text=text_value,
                    label=label_value,
                )
            )

    if not items:
        raise ValueError(
            "No usable text rows were found. Check --text-column or subject/body column names."
        )
    return items


def build_messages(batch: Sequence[SourceItem], prompt_count: int) -> List[Dict[str, str]]:
    system_prompt = (
        "You write neutral, high-quality generation prompts. "
        "Given a source text, produce prompts that could instruct another model "
        "to generate a new text with similar communicative purpose, structure, "
        "tone, and detail level. The prompts must be objective and self-contained. "
        "Do not mention hidden business goals, internal workflow motives, "
        "writing-efficiency claims, or any rationale that is not directly observable "
        "from the source text. Do not say to rewrite, rephrase, imitate, or transform "
        "the provided text. Instead, each prompt should ask for a fresh text to be written. "
        f"Return exactly {prompt_count} diverse prompts per source item."
    )

    user_payload = {
        "task": "Create prompt candidates for each source item.",
        "requirements": [
            "Return valid JSON only.",
            "Use the exact top-level key 'results'.",
            "Each result must contain 'item_id' and 'prompts'.",
            f"'prompts' must contain exactly {prompt_count} strings.",
            "Each prompt must be usable on its own without access to the source text.",
            "Prompts should vary in angle and phrasing while targeting similar output goals.",
            "Avoid meta-commentary, hidden motives, and references to internal business reasoning.",
        ],
        "output_schema": {
            "results": [
                {
                    "item_id": "string",
                    "prompts": ["string" for _ in range(prompt_count)],
                }
            ]
        },
        "items": [
            {
                "item_id": item.source_id,
                "text": item.source_text,
            }
            for item in batch
        ],
    }

    return [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": json.dumps(user_payload, ensure_ascii=False)},
    ]


def extract_json_object(text: str) -> Dict[str, Any]:
    raw = (text or "").strip()
    if not raw:
        raise ValueError("Model returned empty content.")

    try:
        return json.loads(raw)
    except json.JSONDecodeError:
        match = re.search(r"\{.*\}", raw, re.DOTALL)
        if not match:
            raise ValueError("Model did not return valid JSON.")
        return json.loads(match.group(0))


def extract_message_text(payload: Dict[str, Any]) -> str:
    choices = payload.get("choices") or []
    if not choices:
        raise ValueError("API response does not contain any choices.")

    message = choices[0].get("message") or {}
    content = message.get("content", "")
    if isinstance(content, str):
        return content.strip()
    if isinstance(content, list):
        text_parts: List[str] = []
        for item in content:
            if isinstance(item, dict) and item.get("type") == "text":
                text_parts.append(str(item.get("text", "")).strip())
        joined = "\n".join(part for part in text_parts if part).strip()
        if joined:
            return joined
    raise ValueError("Could not extract text content from API response.")


class OpenAICompatibleChatClient:
    def __init__(
        self,
        *,
        api_key: str,
        base_url: str,
        model: str,
        timeout: int,
        temperature: float,
        max_completion_tokens: int,
        max_retries: int,
    ) -> None:
        if not api_key:
            raise ValueError("Missing API key. Set OPENAI_API_KEY or pass --api-key.")
        if OpenAI is None:
            raise RuntimeError(
                "The official 'openai' Python package is not installed. "
                "Install it with: python3 -m pip install openai"
            )
        self.model = model
        self.temperature = temperature
        self.max_completion_tokens = max_completion_tokens
        self.max_retries = max_retries
        self.client = OpenAI(
            api_key=api_key,
            base_url=base_url.rstrip("/"),
            timeout=timeout,
            max_retries=0,
        )

    def complete(self, messages: Sequence[Dict[str, str]]) -> Tuple[str, Dict[str, Any]]:
        payload: Dict[str, Any] = {
            "model": self.model,
            "messages": list(messages),
            "temperature": self.temperature,
            "max_tokens": self.max_completion_tokens,
            "response_format": {"type": "json_object"},
        }

        for attempt in range(1, self.max_retries + 1):
            try:
                response = self.client.chat.completions.create(**payload)
                response_payload = (
                    response.model_dump() if hasattr(response, "model_dump") else dict(response)
                )
                return extract_message_text(response_payload), response_payload
            except Exception as exc:
                if APIStatusError is not None and isinstance(exc, APIStatusError):
                    status_code = getattr(exc, "status_code", None)
                    response_obj = getattr(exc, "response", None)
                    body = getattr(response_obj, "text", None) if response_obj is not None else None
                    retryable = status_code in {408, 409, 425, 429, 500, 502, 503, 504}
                    if attempt >= self.max_retries or not retryable:
                        raise RuntimeError("HTTP {} from API: {}".format(status_code, body)) from exc
                    retry_after_seconds = parse_retry_after_seconds(body or "")
                    sleep_seconds = max(min(2**attempt, 10), retry_after_seconds + DEFAULT_RATE_LIMIT_PADDING)
                elif APIConnectionError is not None and isinstance(exc, APIConnectionError):
                    if attempt >= self.max_retries:
                        raise RuntimeError("Network error while calling API: {}".format(exc)) from exc
                    sleep_seconds = min(2**attempt, 10)
                else:
                    raise

            time.sleep(sleep_seconds)

        raise RuntimeError("Unexpected retry loop exit.")


def validate_batch_result(
    batch: Sequence[SourceItem],
    parsed: Dict[str, Any],
    prompt_count: int,
    model: str,
    batch_index: int,
) -> Tuple[List[OutputRow], List[SourceItem]]:
    raw_results = parsed.get("results")
    if not isinstance(raw_results, list):
        raise ValueError("Model JSON must contain a list under 'results'.")

    by_id: Dict[str, SourceItem] = {item.source_id: item for item in batch}
    output_rows: List[OutputRow] = []
    generated_at = now_utc_iso()

    for result in raw_results:
        if not isinstance(result, dict):
            raise ValueError("Each item in 'results' must be an object.")
        item_id = str(result.get("item_id", "")).strip()
        prompts = result.get("prompts")
        if item_id not in by_id:
            continue
        if not isinstance(prompts, list):
            raise ValueError(f"'prompts' must be a list for item_id={item_id}")
        if len(prompts) != prompt_count:
            raise ValueError(
                f"Expected {prompt_count} prompts for item_id={item_id}, got {len(prompts)}"
            )

        source_item = by_id.pop(item_id)
        for prompt_index, prompt_value in enumerate(prompts, start=1):
            prompt_text = normalize_text(str(prompt_value))
            if not prompt_text:
                raise ValueError(
                    f"Prompt #{prompt_index} is empty for item_id={item_id}"
                )
            output_rows.append(
                OutputRow(
                    row_number=source_item.row_number,
                    source_id=source_item.source_id,
                    source_text=source_item.source_text,
                    prompt_index=prompt_index,
                    prompt=prompt_text,
                    label=source_item.label,
                    model=model,
                    batch_index=batch_index,
                    generated_at=generated_at,
                )
            )

    output_rows.sort(key=lambda row: (row.row_number, row.prompt_index))
    missing_items = [by_id[item_id] for item_id in sorted(by_id)]
    return output_rows, missing_items


def complete_batch_with_retries(
    *,
    client: OpenAICompatibleChatClient,
    batch: Sequence[SourceItem],
    prompt_count: int,
    model: str,
    batch_index: int,
    calls_jsonl_path: Path,
) -> List[OutputRow]:
    pending = list(batch)
    collected: List[OutputRow] = []
    retry_round = 0

    while pending:
        retry_round += 1
        messages = build_messages(pending, prompt_count)
        response_text, response_payload = client.complete(messages)
        parsed = extract_json_object(response_text)
        batch_rows, missing_items = validate_batch_result(
            batch=pending,
            parsed=parsed,
            prompt_count=prompt_count,
            model=model,
            batch_index=batch_index,
        )
        collected.extend(batch_rows)

        append_jsonl(
            calls_jsonl_path,
            {
                "created_at": now_utc_iso(),
                "batch_index": batch_index,
                "retry_round": retry_round,
                "item_ids": [item.source_id for item in pending],
                "missing_item_ids": [item.source_id for item in missing_items],
                "request_messages": messages,
                "response_text": response_text,
                "response_payload": response_payload,
            },
        )

        if not missing_items:
            break

        if len(missing_items) == len(pending) and len(pending) == 1:
            raise ValueError(
                f"Model repeatedly missed source item: {pending[0].source_id}"
            )

        pending = missing_items

    deduped: Dict[Tuple[str, int], OutputRow] = {}
    for row in collected:
        deduped[(row.source_id, row.prompt_index)] = row
    final_rows = list(deduped.values())
    final_rows.sort(key=lambda row: (row.row_number, row.prompt_index))
    return final_rows


def load_resume_rows(
    output_csv_path: Path,
    prompt_count: int,
) -> Tuple[List[OutputRow], set]:
    if not output_csv_path.exists():
        return [], set()

    by_source: Dict[str, List[Dict[str, str]]] = {}
    with output_csv_path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            source_id = str(row.get("source_id", "")).strip()
            if not source_id:
                continue
            by_source.setdefault(source_id, []).append(row)

    resume_rows: List[OutputRow] = []
    completed_source_ids = set()
    for source_id, rows in by_source.items():
        prompt_indices = {
            int(str(row.get("prompt_index", "")).strip())
            for row in rows
            if str(row.get("prompt_index", "")).strip().isdigit()
        }
        if len(prompt_indices) < prompt_count:
            continue
        completed_source_ids.add(source_id)
        for row in rows:
            prompt_index = str(row.get("prompt_index", "")).strip()
            row_number = str(row.get("row_number", "")).strip()
            if not prompt_index.isdigit() or not row_number.isdigit():
                continue
            resume_rows.append(
                OutputRow(
                    row_number=int(row_number),
                    source_id=source_id,
                    source_text="",
                    prompt_index=int(prompt_index),
                    prompt=str(row.get("prompt", "")).strip(),
                    label=str(row.get("label", "")).strip(),
                    model="",
                    batch_index=0,
                    generated_at="",
                )
            )

    resume_rows.sort(key=lambda row: (row.row_number, row.prompt_index))
    return resume_rows, completed_source_ids


def build_manifest(
    *,
    args: argparse.Namespace,
    input_path: Path,
    output_csv_path: Path,
    calls_jsonl_path: Path,
    item_count: int,
    output_count: int,
) -> Dict[str, Any]:
    return {
        "created_at": now_utc_iso(),
        "input_csv": str(input_path),
        "output_csv": str(output_csv_path),
        "calls_jsonl": str(calls_jsonl_path),
        "model": args.model,
        "batch_size": args.batch_size,
        "prompt_count": args.prompt_count,
        "max_input_chars": args.max_input_chars,
        "text_column": args.text_column,
        "subject_column": args.subject_column,
        "body_column": args.body_column,
        "id_column": args.id_column,
        "label_column": args.label_column,
        "row_number_column": args.row_number_column,
        "item_count": item_count,
        "output_count": output_count,
        "label_value": "preserved_from_input",
        "notes": "Prompts are generated objectively without including hidden workflow/business rationale.",
    }


def main() -> int:
    args = parse_args()
    validate_args(args)
    output_dir = build_output_dir(args.output_dir)
    output_csv_path = output_dir / args.output_name
    calls_jsonl_path = output_dir / "calls.jsonl"
    manifest_path = output_dir / "run_manifest.json"

    source_items = load_source_items(args)
    input_path = Path(args.input).resolve()
    results: List[OutputRow] = []
    completed_source_ids = set()

    if args.resume:
        results, completed_source_ids = load_resume_rows(
            output_csv_path=output_csv_path,
            prompt_count=args.prompt_count,
        )
        if completed_source_ids:
            source_items = [
                item for item in source_items if item.source_id not in completed_source_ids
            ]

    client = OpenAICompatibleChatClient(
        api_key=args.api_key,
        base_url=args.base_url,
        model=args.model,
        timeout=args.timeout,
        temperature=args.temperature,
        max_completion_tokens=args.max_completion_tokens,
        max_retries=args.max_retries,
    )

    batches = list(chunked(source_items, args.batch_size))
    total_batches = len(batches)

    for batch_index, batch in enumerate(batches, start=1):
        batch_rows = complete_batch_with_retries(
            client=client,
            batch=batch,
            prompt_count=args.prompt_count,
            model=args.model,
            batch_index=batch_index,
            calls_jsonl_path=calls_jsonl_path,
        )
        results.extend(batch_rows)

        if args.save_every > 0 and batch_index % args.save_every == 0:
            write_csv(output_csv_path, results)

        if args.print_every > 0 and batch_index % args.print_every == 0:
            print(
                f"[progress] batch {batch_index}/{total_batches} complete, "
                f"saved {len(results)} prompts so far",
                file=sys.stderr,
            )

        if args.sleep_seconds > 0:
            time.sleep(args.sleep_seconds)

    write_csv(output_csv_path, results)
    manifest = build_manifest(
        args=args,
        input_path=input_path,
        output_csv_path=output_csv_path,
        calls_jsonl_path=calls_jsonl_path,
        item_count=len(source_items),
        output_count=len(results),
    )
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    with manifest_path.open("w", encoding="utf-8") as handle:
        json.dump(manifest, handle, ensure_ascii=False, indent=2)
        handle.write("\n")

    print(f"Wrote {len(results)} prompt rows to {output_csv_path}", file=sys.stderr)
    print(f"Saved call logs to {calls_jsonl_path}", file=sys.stderr)
    print(f"Saved run manifest to {manifest_path}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception as exc:
        print(f"[error] {exc}", file=sys.stderr)
        raise SystemExit(1)
