#!/usr/bin/env python3
"""Reproduce the rephrasing setup from Utaliyeva et al. (HPCC/DSS 2023).

The paper evaluates spam-filter robustness after rewriting spam emails with
paper-style prompts such as:

- "Rewrite the following email to be less spammy: 'email text'"
- "Rewrite the following email: 'email text'"

This script focuses on dataset-side reproduction: batch rewrite CSV rows using
an OpenAI-compatible chat completion API, preserve metadata, and log every call.
"""

import argparse
import csv
import json
import math
import os
import re
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, List, NamedTuple, Optional, Sequence, Tuple

try:
    from openai import APIConnectionError, APIStatusError, OpenAI
except ImportError:
    APIConnectionError = None
    APIStatusError = None
    OpenAI = None

SCRIPT_DIR = Path(__file__).resolve().parent

DEFAULT_BASE_URL = "https://api.openai.com/v1"
DEFAULT_TIMEOUT = 120
DEFAULT_LABEL_COLUMN = "label"
DEFAULT_LABEL_VALUE = "1"
DEFAULT_SUBJECT_COLUMN = "Subject"
DEFAULT_BODY_COLUMN = "Body"

PROMPT_TEMPLATES: Dict[str, str] = {
    "adversarial": "Rewrite the following email to be less spammy: '{email_text}'",
    "general": "Rewrite the following email: '{email_text}'",
    "legitimate": "Can you rewrite this email to be more legitimate: {email_text}",
}

SUBJECT_RE = re.compile(r"(?im)^\s*subject(?:\s*line)?\s*[:：]\s*(.+)$")
CODE_FENCE_RE = re.compile(r"^```[a-zA-Z0-9_-]*\s*|\s*```$", re.MULTILINE)


class RowResult(NamedTuple):
    row_number: int
    prompt_kind: str
    prompt_rounds: int
    original_subject: str
    original_body: str
    rewritten_subject: str
    rewritten_body: str
    label: str
    original_data_source: str
    output_data_source: str
    model: str
    token_estimate: int
    skipped: bool
    skip_reason: str


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Batch reproduction of the Utaliyeva et al. 2023 email rephrasing setup."
    )
    parser.add_argument("--input", required=True, help="Input CSV path.")
    parser.add_argument(
        "--output-dir",
        default="",
        help="Optional output directory. Defaults to S6-Stealthy Rewriting/runs/<timestamp>/",
    )
    parser.add_argument(
        "--prompt-kind",
        choices=("adversarial", "general", "legitimate", "all"),
        default="adversarial",
        help="Prompt family to run. 'all' produces one output row per prompt family.",
    )
    parser.add_argument(
        "--rounds",
        type=int,
        default=1,
        help="How many iterative rewrite rounds to run for each email.",
    )
    parser.add_argument(
        "--model",
        default=os.environ.get("OPENAI_MODEL", "gpt-3.5-turbo"),
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
        "--subject-column",
        default=DEFAULT_SUBJECT_COLUMN,
        help="Subject column name in the input CSV.",
    )
    parser.add_argument(
        "--body-column",
        default=DEFAULT_BODY_COLUMN,
        help="Body column name in the input CSV.",
    )
    parser.add_argument(
        "--label-column",
        default=DEFAULT_LABEL_COLUMN,
        help="Label column name in the input CSV.",
    )
    parser.add_argument(
        "--label-value",
        default=DEFAULT_LABEL_VALUE,
        help="Only rows whose label column matches this value are rewritten unless --keep-all-labels is set.",
    )
    parser.add_argument(
        "--data-source-column",
        default="data_source",
        help="Data-source column name in the input CSV.",
    )
    parser.add_argument(
        "--keep-all-labels",
        action="store_true",
        help="Rewrite all rows instead of filtering by --label-value.",
    )
    parser.add_argument(
        "--max-rows",
        type=int,
        default=0,
        help="Optional cap for the number of selected rows to rewrite.",
    )
    parser.add_argument(
        "--historical-token-limit",
        type=int,
        default=4096,
        help="Historical token limit used for paper-style skipping.",
    )
    parser.add_argument(
        "--disable-token-limit-skip",
        action="store_true",
        help="Do not skip rows that exceed --historical-token-limit.",
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=1.0,
        help="Sampling temperature. Defaults to 1.0 for higher-diversity rewriting.",
    )
    parser.add_argument(
        "--max-completion-tokens",
        type=int,
        default=None,
        help="Optional completion token cap forwarded to the API if set.",
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
        "--print-every",
        type=int,
        default=25,
        help="Progress logging interval.",
    )
    parser.add_argument(
        "--save-every",
        type=int,
        default=100,
        help="Write the output CSV every N completed rows as a checkpoint.",
    )
    return parser.parse_args()


def now_utc_iso() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat()


def normalize_text(value: str) -> str:
    text = (value or "").replace("\r\n", "\n").replace("\r", "\n")
    text = text.replace("\ufeff", "").replace("\u200b", "")
    text = re.sub(r"[ \t]+\n", "\n", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def strip_code_fences(text: str) -> str:
    return CODE_FENCE_RE.sub("", (text or "").strip()).strip()


def render_email(subject: str, body: str) -> str:
    subject_clean = normalize_text(subject)
    body_clean = normalize_text(body)
    if subject_clean:
        return f"Subject: {subject_clean}\n\n{body_clean}".strip()
    return body_clean


def parse_subject_and_body(model_text: str, fallback_subject: str) -> Tuple[str, str]:
    text = strip_code_fences(model_text)
    match = SUBJECT_RE.search(text)
    if not match:
        return normalize_text(fallback_subject), normalize_text(text)

    subject = normalize_text(match.group(1))
    start, end = match.span()
    body = normalize_text((text[:start] + text[end:]).strip())
    return subject or normalize_text(fallback_subject), body


def infer_token_count(text: str, model: str) -> int:
    try:
        import tiktoken  # type: ignore

        try:
            encoding = tiktoken.encoding_for_model(model)
        except KeyError:
            encoding = tiktoken.get_encoding("cl100k_base")
        return len(encoding.encode(text))
    except Exception:
        return max(1, math.ceil(len(text) / 4))


def extract_message_text(payload: Dict[str, Any]) -> str:
    choices = payload.get("choices") or []
    if not choices:
        raise ValueError("API response does not contain any choices.")
    message = choices[0].get("message") or {}
    content = message.get("content", "")
    if isinstance(content, str):
        return content.strip()
    if isinstance(content, list):
        parts: List[str] = []
        for item in content:
            if isinstance(item, dict) and item.get("type") == "text":
                parts.append(str(item.get("text", "")).strip())
        joined = "\n".join(part for part in parts if part).strip()
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
        temperature: Optional[float],
        max_completion_tokens: Optional[int],
        max_retries: int,
    ) -> None:
        if not api_key:
            raise ValueError("Missing API key. Set OPENAI_API_KEY or pass --api-key.")
        if OpenAI is None:
            raise RuntimeError(
                "The official 'openai' Python package is not installed. "
                "Install it with: python3 -m pip install openai"
            )
        self.api_key = api_key
        self.base_url = base_url.rstrip("/")
        self.model = model
        self.timeout = timeout
        self.temperature = temperature
        self.max_completion_tokens = max_completion_tokens
        self.max_retries = max_retries
        self.client = OpenAI(
            api_key=self.api_key,
            base_url=self.base_url,
            timeout=self.timeout,
            max_retries=0,
        )

    def complete(self, user_prompt: str) -> Tuple[str, Dict[str, Any]]:
        payload: Dict[str, Any] = {
            "model": self.model,
            "messages": [{"role": "user", "content": user_prompt}],
        }
        if self.temperature is not None:
            payload["temperature"] = self.temperature
        if self.max_completion_tokens is not None:
            payload["max_tokens"] = self.max_completion_tokens

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
                elif APIConnectionError is not None and isinstance(exc, APIConnectionError):
                    if attempt >= self.max_retries:
                        raise RuntimeError("Network error while calling API: {}".format(exc)) from exc
                else:
                    raise

            time.sleep(min(2**attempt, 10))

        raise RuntimeError("Unexpected retry loop exit.")


def select_rows(
    rows: Iterable[Dict[str, str]],
    *,
    label_column: str,
    label_value: str,
    keep_all_labels: bool,
    max_rows: int,
) -> List[Tuple[int, Dict[str, str]]]:
    selected: List[Tuple[int, Dict[str, str]]] = []
    for row_number, row in enumerate(rows, start=2):
        if not keep_all_labels and str(row.get(label_column, "")).strip() != label_value:
            continue
        selected.append((row_number, row))
        if max_rows > 0 and len(selected) >= max_rows:
            break
    return selected


def prompt_kinds_from_arg(prompt_kind: str) -> List[str]:
    if prompt_kind == "all":
        return ["adversarial", "general", "legitimate"]
    return [prompt_kind]


def build_output_dir(output_dir_arg: str) -> Path:
    if output_dir_arg:
        return Path(output_dir_arg).resolve()
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    return (SCRIPT_DIR / "runs" / stamp).resolve()


def build_output_source(original_source: str, prompt_kind: str, rounds: int) -> str:
    base = (original_source or "unknown").strip() or "unknown"
    return f"{base}|utaliyeva2023:{prompt_kind}:r{rounds}"


def build_output_csv_name(model: str) -> str:
    model_norm = (model or "").strip().lower()
    if model_norm.startswith("gpt-3.5"):
        return "LLM-GPT3.5.csv"
    slug = re.sub(r"[^A-Za-z0-9.]+", "-", (model or "output").strip()).strip("-")
    slug = slug or "output"
    return "LLM-{}.csv".format(slug)


def write_csv(path: Path, rows: Sequence[RowResult]) -> None:
    fieldnames = [
        "Subject",
        "Body",
        "label",
        "data_source",
        "original_subject",
        "original_body",
        "original_data_source",
        "row_number",
        "prompt_kind",
        "prompt_rounds",
        "model",
        "token_estimate",
        "skipped",
        "skip_reason",
    ]
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(
                {
                    "Subject": row.rewritten_subject,
                    "Body": row.rewritten_body,
                    "label": row.label,
                    "data_source": row.output_data_source,
                    "original_subject": row.original_subject,
                    "original_body": row.original_body,
                    "original_data_source": row.original_data_source,
                    "row_number": row.row_number,
                    "prompt_kind": row.prompt_kind,
                    "prompt_rounds": row.prompt_rounds,
                    "model": row.model,
                    "token_estimate": row.token_estimate,
                    "skipped": str(row.skipped).lower(),
                    "skip_reason": row.skip_reason,
                }
            )


def append_jsonl(path: Path, record: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(record, ensure_ascii=False) + "\n")


def validate_columns(rows: Sequence[Dict[str, str]], required_columns: Sequence[str]) -> None:
    if not rows:
        raise ValueError("Input CSV does not contain any data rows.")
    missing = [name for name in required_columns if name not in rows[0]]
    if missing:
        raise ValueError(f"Missing required columns: {', '.join(missing)}")


def main() -> int:
    args = parse_args()
    if args.rounds < 1:
        raise ValueError("--rounds must be at least 1.")

    input_path = Path(args.input).resolve()
    output_dir = build_output_dir(args.output_dir)
    rewritten_csv_path = output_dir / build_output_csv_name(args.model)
    calls_jsonl_path = output_dir / "calls.jsonl"
    manifest_path = output_dir / "run_manifest.json"
    output_dir.mkdir(parents=True, exist_ok=True)

    with input_path.open("r", encoding="utf-8", newline="") as handle:
        all_rows = list(csv.DictReader(handle))

    validate_columns(
        all_rows,
        [args.subject_column, args.body_column, args.label_column, args.data_source_column],
    )
    selected_rows = select_rows(
        all_rows,
        label_column=args.label_column,
        label_value=args.label_value,
        keep_all_labels=args.keep_all_labels,
        max_rows=args.max_rows,
    )

    if not selected_rows:
        raise ValueError("No input rows matched the selection criteria.")

    client = OpenAICompatibleChatClient(
        api_key=args.api_key,
        base_url=args.base_url,
        model=args.model,
        timeout=args.timeout,
        temperature=args.temperature,
        max_completion_tokens=args.max_completion_tokens,
        max_retries=args.max_retries,
    )

    results: List[RowResult] = []
    selected_prompt_kinds = prompt_kinds_from_arg(args.prompt_kind)
    completed_results = 0

    for selected_index, (row_number, row) in enumerate(selected_rows, start=1):
        original_subject = normalize_text(row.get(args.subject_column, ""))
        original_body = normalize_text(row.get(args.body_column, ""))
        label = str(row.get(args.label_column, "")).strip()
        original_data_source = str(row.get(args.data_source_column, "")).strip()
        seed_email = render_email(original_subject, original_body)
        token_estimate = infer_token_count(seed_email, args.model)

        for prompt_kind in selected_prompt_kinds:
            if not args.disable_token_limit_skip and token_estimate > args.historical_token_limit:
                results.append(
                    RowResult(
                        row_number=row_number,
                        prompt_kind=prompt_kind,
                        prompt_rounds=args.rounds,
                        original_subject=original_subject,
                        original_body=original_body,
                        rewritten_subject=original_subject,
                        rewritten_body=original_body,
                        label=label,
                        original_data_source=original_data_source,
                        output_data_source=build_output_source(original_data_source, prompt_kind, args.rounds),
                        model=args.model,
                        token_estimate=token_estimate,
                        skipped=True,
                        skip_reason=f"token_estimate>{args.historical_token_limit}",
                    )
                )
                completed_results += 1
                if args.save_every > 0 and completed_results % args.save_every == 0:
                    write_csv(rewritten_csv_path, results)
                    print(
                        f"[checkpoint] saved {completed_results} rows to {rewritten_csv_path}",
                        file=sys.stderr,
                    )
                continue

            current_email = seed_email
            final_subject = original_subject
            final_body = original_body

            for round_idx in range(1, args.rounds + 1):
                prompt_text = PROMPT_TEMPLATES[prompt_kind].format(email_text=current_email)
                response_text, raw_response = client.complete(prompt_text)
                final_subject, final_body = parse_subject_and_body(response_text, final_subject)
                current_email = render_email(final_subject, final_body)

                append_jsonl(
                    calls_jsonl_path,
                    {
                        "timestamp_utc": now_utc_iso(),
                        "row_number": row_number,
                        "selected_index": selected_index,
                        "prompt_kind": prompt_kind,
                        "round": round_idx,
                        "model": args.model,
                        "token_estimate": token_estimate,
                        "original_subject": original_subject,
                        "original_body": original_body,
                        "input_email_for_round": prompt_text,
                        "response_text": response_text,
                        "raw_response": raw_response,
                    },
                )

                if args.sleep_seconds > 0:
                    time.sleep(args.sleep_seconds)

            results.append(
                RowResult(
                    row_number=row_number,
                    prompt_kind=prompt_kind,
                    prompt_rounds=args.rounds,
                    original_subject=original_subject,
                    original_body=original_body,
                    rewritten_subject=final_subject,
                    rewritten_body=final_body,
                    label=label,
                    original_data_source=original_data_source,
                    output_data_source=build_output_source(original_data_source, prompt_kind, args.rounds),
                    model=args.model,
                    token_estimate=token_estimate,
                    skipped=False,
                    skip_reason="",
                )
            )
            completed_results += 1
            if args.save_every > 0 and completed_results % args.save_every == 0:
                write_csv(rewritten_csv_path, results)
                print(
                    f"[checkpoint] saved {completed_results} rows to {rewritten_csv_path}",
                    file=sys.stderr,
                )

        if args.print_every > 0 and selected_index % args.print_every == 0:
            print(
                f"[progress] processed {selected_index}/{len(selected_rows)} selected rows",
                file=sys.stderr,
            )

    write_csv(rewritten_csv_path, results)

    manifest = {
        "timestamp_utc": now_utc_iso(),
        "paper": {
            "citation_key": "utaliyeva2023chatgpt",
            "title": "ChatGPT: A Threat to Spam Filtering Systems",
            "venue": "HPCC/DSS/SmartCity/DependSys",
            "year": 2023,
        },
        "input_path": str(input_path),
        "output_dir": str(output_dir),
        "output_csv": str(rewritten_csv_path),
        "selected_rows": len(selected_rows),
        "result_rows": len(results),
        "prompt_kinds": selected_prompt_kinds,
        "rounds": args.rounds,
        "model": args.model,
        "base_url": args.base_url,
        "historical_token_limit": args.historical_token_limit,
        "token_limit_skip_enabled": not args.disable_token_limit_skip,
        "temperature": args.temperature,
        "save_every": args.save_every,
        "max_completion_tokens": args.max_completion_tokens,
        "notes": [
            "The large-scale paper prompt was 'Rewrite the following email to be less spammy: \"email text\"'.",
            "The paper also compares this against the general prompt 'Rewrite the following email: \"email text\"'.",
            "Iterative rewriting is supported here because the paper notes it as beneficial, but does not report one fixed multi-round schedule.",
        ],
    }
    manifest_path.write_text(json.dumps(manifest, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")

    print(f"Wrote {len(results)} rows to {rewritten_csv_path}", file=sys.stderr)
    print(f"Saved call logs to {calls_jsonl_path}", file=sys.stderr)
    print(f"Saved manifest to {manifest_path}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
