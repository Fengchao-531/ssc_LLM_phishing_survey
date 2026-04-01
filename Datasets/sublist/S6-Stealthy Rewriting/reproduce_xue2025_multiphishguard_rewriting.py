#!/usr/bin/env python3
"""Reproduce the adversarial rewriting component from MultiPhishGuard.

Paper:
Xue et al. "MultiPhishGuard: An LLM-based multi-agent system for phishing
email detection" (arXiv:2505.23803 / CCS 2025).

This script focuses on the adversarial agent described in Section 3.4 and
Figure 5: generating phishing-preserving or benign-preserving variants that
stress a detector.

Notes on fidelity:
- The paper states the adversarial agent is GPT-4o-based. This script defaults
  to `gpt-4o`.
- For modern GPT-4o / GPT-4.1 / GPT-5 style models, the script uses the
  official `responses.create(...)` API.
- For legacy `gpt-3.5-turbo`, it falls back to `chat.completions.create(...)`.
- The paper reports explanation-quality metrics, but does not define a
  dedicated text-quality screening stage for adversarially generated emails.
  This reproduction therefore focuses on generation only.
"""

import argparse
import csv
import json
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

# Some Generic-Data derived emails are very large HTML messages.
csv.field_size_limit(10**9)


SCRIPT_DIR = Path(__file__).resolve().parent
SIMPLIFIED_OUTPUT_PATH = SCRIPT_DIR / "MPG-LLM-P.csv"
DEFAULT_BASE_URL = "https://api.openai.com/v1"
DEFAULT_TIMEOUT = 120
DEFAULT_INPUT_PATH = SCRIPT_DIR / "HW-P.csv"
DEFAULT_SUBJECT_COLUMN = "Subject"
DEFAULT_BODY_COLUMN = "Body"
DEFAULT_LABEL_COLUMN = "label"
DEFAULT_DATA_SOURCE_COLUMN = "data_source"

SUBJECT_RE = re.compile(r"(?im)^\s*subject(?:\s*line)?\s*[:：]\s*(.+)$")
CODE_FENCE_RE = re.compile(r"^```[a-zA-Z0-9_-]*\s*|\s*```$", re.MULTILINE)


class RowResult(NamedTuple):
    row_number: int
    rewritten_subject: str
    rewritten_body: str
    label: str
    data_source: str
    original_subject: str
    original_body: str
    original_data_source: str
    model: str
    api_mode: str
    prompt_rounds: int
    skipped: bool
    skip_reason: str


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Reproduce the MultiPhishGuard adversarial rewriting agent."
    )
    parser.add_argument(
        "--input",
        default=str(DEFAULT_INPUT_PATH),
        help="Input CSV path. Defaults to S6-Stealthy Rewriting/HW-P.csv.",
    )
    parser.add_argument(
        "--output-dir",
        default="",
        help="Optional output directory. Defaults to S6-Stealthy Rewriting/runs/<timestamp>/",
    )
    parser.add_argument(
        "--model",
        default=os.environ.get("OPENAI_MODEL", "gpt-4o"),
        help="Model used for the adversarial agent. Paper-faithful default: gpt-4o.",
    )
    parser.add_argument(
        "--api-mode",
        choices=("auto", "responses", "chat"),
        default="auto",
        help="Use Responses API for modern models and Chat Completions for legacy ones.",
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
        "--data-source-column",
        default=DEFAULT_DATA_SOURCE_COLUMN,
        help="Data-source column name in the input CSV.",
    )
    parser.add_argument(
        "--max-rows",
        type=int,
        default=0,
        help="Optional cap for the number of rows to process.",
    )
    parser.add_argument(
        "--rounds",
        type=int,
        default=1,
        help="How many adversarial generation-refinement rounds to run.",
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=1.0,
        help="Sampling temperature. Default 1.0 for diverse adversarial variants.",
    )
    parser.add_argument(
        "--max-completion-tokens",
        type=int,
        default=None,
        help="Optional completion token cap forwarded to the API.",
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
        default=100,
        help="Write the output CSV every N completed rows as a checkpoint.",
    )
    parser.add_argument(
        "--print-every",
        type=int,
        default=25,
        help="Progress logging interval.",
    )
    return parser.parse_args()


def now_utc_iso() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat()


def normalize_text(value: str) -> str:
    text = (value or "").replace("\r\n", "\n").replace("\r", "\n")
    text = text.replace("\ufeff", "").replace("\u200b", "").replace("\xa0", " ")
    text = re.sub(r"[ \t]+\n", "\n", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def strip_code_fences(text: str) -> str:
    return CODE_FENCE_RE.sub("", (text or "").strip()).strip()


def render_email(subject: str, body: str) -> str:
    subject_clean = normalize_text(subject)
    body_clean = normalize_text(body)
    if subject_clean:
        return "Subject: {}\n\n{}".format(subject_clean, body_clean).strip()
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


def is_phishing_label(label: str) -> bool:
    value = str(label).strip().lower()
    return value in {"1", "phishing", "spam", "malicious", "true", "yes"}


def infer_api_mode(model: str, requested_mode: str) -> str:
    if requested_mode != "auto":
        return requested_mode
    model_norm = (model or "").strip().lower()
    if model_norm.startswith("gpt-3.5"):
        return "chat"
    return "responses"


def build_output_dir(output_dir_arg: str) -> Path:
    if output_dir_arg:
        return Path(output_dir_arg).resolve()
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    return (SCRIPT_DIR / "runs" / stamp).resolve()


def build_output_csv_name(model: str) -> str:
    model_norm = (model or "").strip().lower()
    if model_norm.startswith("gpt-4o"):
        return "LLM-MultiPhishGuard-GPT4o.csv"
    if model_norm.startswith("gpt-3.5"):
        return "LLM-MultiPhishGuard-GPT3.5.csv"
    slug = re.sub(r"[^A-Za-z0-9.]+", "-", (model or "output").strip()).strip("-")
    slug = slug or "output"
    return "LLM-MultiPhishGuard-{}.csv".format(slug)


def build_output_source(original_source: str, model: str, rounds: int) -> str:
    base = (original_source or "unknown").strip() or "unknown"
    model_tag = re.sub(r"[^A-Za-z0-9.]+", "-", model).strip("-") or "model"
    return "{}|multiphishguard:{}:r{}".format(base, model_tag, rounds)


def build_adversarial_prompts(
    subject: str,
    body: str,
    label: str,
    round_idx: int,
) -> Tuple[str, str]:
    email_type = "phishing" if is_phishing_label(label) else "legitimate"
    email_text = render_email(subject, body)

    developer_prompt = (
        "You are an expert adversarial email generator for phishing-detector stress testing.\n\n"
        "Your task is to produce one email variant that preserves the original scenario, intent, "
        "and overall meaning while changing wording and presentation using the following methods:\n"
        "- synonym substitution\n"
        "- sentence restructuring\n"
        "- content modification of low-importance details\n"
        "- light homoglyph-style or orthographic variation when natural\n"
        "- polymorphic variation in subject, framing, tone, and formatting\n\n"
        "If the source email is phishing:\n"
        "- keep the malicious objective intact\n"
        "- keep the target context or brand context intact when present\n"
        "- reduce obvious phishing signatures while preserving plausibility and urgency\n"
        "- make the message resemble authentic business or service communication\n\n"
        "If the source email is legitimate:\n"
        "- keep it clearly benign and non-malicious\n"
        "- make it slightly more ambiguous or borderline suspicious without changing its benign intent\n"
        "- keep it realistic and professional\n\n"
        "Return only the final email text. Do not describe the methods and do not explain your choices."
    )

    user_prompt = (
        "Help me write one {email_type} email variant for adversarial evaluation.\n"
        "Round: {round_idx}\n\n"
        "Requirements:\n"
        "- use synonym substitution where helpful\n"
        "- change sentence structure and local phrasing\n"
        "- modify non-essential content details when useful\n"
        "- allow mild homoglyph or orthographic variation if it still looks natural\n"
        "- vary style, subject phrasing, and formatting to create a polymorphic variant\n"
        "- preserve the original intent, scenario, and functional purpose\n"
        "- output the email only\n\n"
        "Source email:\n"
        "{email_text}\n"
    ).format(
        email_type=email_type,
        round_idx=round_idx,
        email_text=email_text,
    )
    return developer_prompt, user_prompt


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
        "model",
        "api_mode",
        "prompt_rounds",
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
                    "data_source": row.data_source,
                    "original_subject": row.original_subject,
                    "original_body": row.original_body,
                    "original_data_source": row.original_data_source,
                    "row_number": row.row_number,
                    "model": row.model,
                    "api_mode": row.api_mode,
                    "prompt_rounds": row.prompt_rounds,
                    "skipped": str(row.skipped).lower(),
                    "skip_reason": row.skip_reason,
                }
            )


def write_simplified_csv(path: Path, rows: Sequence[RowResult]) -> None:
    fieldnames = ["Subject", "Body", "label", "data_source"]
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
                    "data_source": row.original_data_source,
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
        raise ValueError("Missing required columns: {}".format(", ".join(missing)))


class OpenAITextClient:
    def __init__(
        self,
        *,
        api_key: str,
        base_url: str,
        model: str,
        api_mode: str,
        timeout: int,
        temperature: float,
        max_completion_tokens: Optional[int],
        max_retries: int,
    ) -> None:
        if not api_key:
            raise ValueError("Missing API key. Set OPENAI_API_KEY or pass --api-key.")
        if OpenAI is None:
            raise RuntimeError(
                "The official 'openai' Python package is not installed. "
                "Install it with: conda run -n FC-W2-gpu-p39 python -m pip install openai"
            )
        self.model = model
        self.api_mode = api_mode
        self.temperature = temperature
        self.max_completion_tokens = max_completion_tokens
        self.max_retries = max_retries
        self.client = OpenAI(
            api_key=api_key,
            base_url=base_url.rstrip("/"),
            timeout=timeout,
            max_retries=0,
        )

    def generate(self, developer_prompt: str, user_prompt: str) -> Tuple[str, Dict[str, Any]]:
        for attempt in range(1, self.max_retries + 1):
            try:
                if self.api_mode == "responses":
                    payload: Dict[str, Any] = {
                        "model": self.model,
                        "input": [
                            {"role": "developer", "content": developer_prompt},
                            {"role": "user", "content": user_prompt},
                        ],
                    }
                    if self.temperature is not None:
                        payload["temperature"] = self.temperature
                    if self.max_completion_tokens is not None:
                        payload["max_output_tokens"] = self.max_completion_tokens
                    response = self.client.responses.create(**payload)
                    response_payload = response.model_dump() if hasattr(response, "model_dump") else dict(response)
                    output_text = getattr(response, "output_text", None)
                    if not output_text:
                        output_text = extract_response_output_text(response_payload)
                    return normalize_text(output_text), response_payload

                payload = {
                    "model": self.model,
                    "messages": [
                        {"role": "system", "content": developer_prompt},
                        {"role": "user", "content": user_prompt},
                    ],
                }
                if self.temperature is not None:
                    payload["temperature"] = self.temperature
                if self.max_completion_tokens is not None:
                    payload["max_tokens"] = self.max_completion_tokens
                response = self.client.chat.completions.create(**payload)
                response_payload = response.model_dump() if hasattr(response, "model_dump") else dict(response)
                return extract_chat_output_text(response_payload), response_payload
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
            time.sleep(min(2 ** attempt, 10))

        raise RuntimeError("Unexpected retry loop exit.")


def extract_chat_output_text(payload: Dict[str, Any]) -> str:
    choices = payload.get("choices") or []
    if not choices:
        raise ValueError("Chat response does not contain any choices.")
    message = choices[0].get("message") or {}
    content = message.get("content", "")
    if isinstance(content, str):
        return normalize_text(content)
    if isinstance(content, list):
        parts = []
        for item in content:
            if isinstance(item, dict) and item.get("type") == "text":
                parts.append(str(item.get("text", "")).strip())
        joined = "\n".join(part for part in parts if part).strip()
        if joined:
            return normalize_text(joined)
    raise ValueError("Could not extract chat output text.")


def extract_response_output_text(payload: Dict[str, Any]) -> str:
    if payload.get("output_text"):
        return normalize_text(str(payload["output_text"]))
    outputs = payload.get("output") or []
    parts: List[str] = []
    for item in outputs:
        if not isinstance(item, dict):
            continue
        for content in item.get("content", []) or []:
            if isinstance(content, dict) and content.get("type") in {"output_text", "text"}:
                text = content.get("text")
                if text:
                    parts.append(str(text).strip())
    joined = "\n".join(part for part in parts if part).strip()
    if joined:
        return normalize_text(joined)
    raise ValueError("Could not extract responses output text.")


def main() -> int:
    args = parse_args()
    if args.rounds < 1:
        raise ValueError("--rounds must be at least 1.")

    input_path = Path(args.input).resolve()
    output_dir = build_output_dir(args.output_dir)
    output_csv_path = output_dir / build_output_csv_name(args.model)
    simplified_csv_path = SIMPLIFIED_OUTPUT_PATH
    calls_jsonl_path = output_dir / "calls.jsonl"
    manifest_path = output_dir / "run_manifest.json"
    output_dir.mkdir(parents=True, exist_ok=True)

    with input_path.open("r", encoding="utf-8", newline="") as handle:
        all_rows = list(csv.DictReader(handle))

    validate_columns(
        all_rows,
        [
            args.subject_column,
            args.body_column,
            args.label_column,
            args.data_source_column,
        ],
    )

    selected_rows: List[Tuple[int, Dict[str, str]]] = []
    for row_number, row in enumerate(all_rows, start=2):
        selected_rows.append((row_number, row))
        if args.max_rows > 0 and len(selected_rows) >= args.max_rows:
            break

    api_mode = infer_api_mode(args.model, args.api_mode)
    client = OpenAITextClient(
        api_key=args.api_key,
        base_url=args.base_url,
        model=args.model,
        api_mode=api_mode,
        timeout=args.timeout,
        temperature=args.temperature,
        max_completion_tokens=args.max_completion_tokens,
        max_retries=args.max_retries,
    )

    results: List[RowResult] = []
    completed_rows = 0

    for selected_index, (row_number, row) in enumerate(selected_rows, start=1):
        original_subject = normalize_text(row.get(args.subject_column, ""))
        original_body = normalize_text(row.get(args.body_column, ""))
        label = str(row.get(args.label_column, "")).strip()
        original_data_source = str(row.get(args.data_source_column, "")).strip()

        final_subject = original_subject
        final_body = original_body

        for round_idx in range(1, args.rounds + 1):
            developer_prompt, user_prompt = build_adversarial_prompts(
                subject=final_subject,
                body=final_body,
                label=label,
                round_idx=round_idx,
            )
            response_text, raw_response = client.generate(developer_prompt, user_prompt)
            final_subject, final_body = parse_subject_and_body(response_text, final_subject)

            append_jsonl(
                calls_jsonl_path,
                {
                    "timestamp_utc": now_utc_iso(),
                    "row_number": row_number,
                    "selected_index": selected_index,
                    "round": round_idx,
                    "label": label,
                    "model": args.model,
                    "api_mode": api_mode,
                    "developer_prompt": developer_prompt,
                    "user_prompt": user_prompt,
                    "response_text": response_text,
                    "raw_response": raw_response,
                },
            )

            if args.sleep_seconds > 0:
                time.sleep(args.sleep_seconds)

        results.append(
            RowResult(
                row_number=row_number,
                rewritten_subject=final_subject,
                rewritten_body=final_body,
                label=label,
                data_source=build_output_source(original_data_source, args.model, args.rounds),
                original_subject=original_subject,
                original_body=original_body,
                original_data_source=original_data_source,
                model=args.model,
                api_mode=api_mode,
                prompt_rounds=args.rounds,
                skipped=False,
                skip_reason="",
            )
        )

        completed_rows += 1
        if args.save_every > 0 and completed_rows % args.save_every == 0:
            write_csv(output_csv_path, results)
            write_simplified_csv(simplified_csv_path, results)
            print(
                "[checkpoint] saved {} rows to {}".format(completed_rows, output_csv_path),
                file=sys.stderr,
            )

        if args.print_every > 0 and selected_index % args.print_every == 0:
            print(
                "[progress] processed {}/{} rows".format(selected_index, len(selected_rows)),
                file=sys.stderr,
            )

    write_csv(output_csv_path, results)
    write_simplified_csv(simplified_csv_path, results)

    manifest = {
        "timestamp_utc": now_utc_iso(),
        "paper": {
            "citation_key": "xue2025multiphishguard",
            "title": "MultiPhishGuard: An LLM-based multi-agent system for phishing email detection",
            "year": 2025,
        },
        "input_path": str(input_path),
        "output_dir": str(output_dir),
        "output_csv": str(output_csv_path),
        "simplified_output_csv": str(simplified_csv_path),
        "selected_rows": len(selected_rows),
        "result_rows": len(results),
        "model": args.model,
        "api_mode": api_mode,
        "rounds": args.rounds,
        "temperature": args.temperature,
        "save_every": args.save_every,
        "notes": [
            "The paper describes a GPT-4o-based adversarial agent in Section 3.4.",
            "This script focuses on pure adversarial email generation without any external detector loop.",
            "Modern models use responses.create; gpt-3.5-turbo uses chat.completions.create.",
        ],
    }
    manifest_path.write_text(json.dumps(manifest, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")

    print("Wrote {} rows to {}".format(len(results), output_csv_path), file=sys.stderr)
    print("Wrote simplified export to {}".format(simplified_csv_path), file=sys.stderr)
    print("Saved call logs to {}".format(calls_jsonl_path), file=sys.stderr)
    print("Saved manifest to {}".format(manifest_path), file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
