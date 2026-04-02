#!/usr/bin/env python3
"""Reproduce the adversarial rewriting component from MultiPhishGuard.

Paper:
Xue et al. "MultiPhishGuard: An LLM-based multi-agent system for phishing
email detection" (arXiv:2505.23803 / CCS 2025).

This script focuses on the adversarial agent described in Section 3.4 and
Figure 5: generating phishing-preserving or benign-preserving variants that
stress a detector.

Notes on fidelity:
- The paper states the adversarial agent is GPT-4o-based.
- This replace variant keeps the same prompting / rewriting flow, but swaps the
  backend to a locally loaded Hugging Face causal LM following the official
  DeepSeek chat-template usage pattern.
- The paper reports explanation-quality metrics, but does not define a
  dedicated text-quality screening stage for adversarially generated emails.
  This reproduction therefore focuses on generation only.
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

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

try:
    from tqdm.auto import tqdm
except Exception:  # pragma: no cover - optional dependency fallback
    tqdm = None

# Some Generic-Data derived emails are very large HTML messages.
csv.field_size_limit(10**9)


SCRIPT_DIR = Path(__file__).resolve().parent
SIMPLIFIED_OUTPUT_PATH = SCRIPT_DIR / "MPG-LLM-P-replace.csv"
DEFAULT_TIMEOUT = 120
DEFAULT_INPUT_PATH = SCRIPT_DIR / "HW-P.csv"
DEFAULT_SUBJECT_COLUMN = "Subject"
DEFAULT_BODY_COLUMN = "Body"
DEFAULT_LABEL_COLUMN = "label"
DEFAULT_DATA_SOURCE_COLUMN = "data_source"
DEFAULT_COMPLETION_TOKEN_RESERVE = 1000

MODEL_CONTEXT_LIMITS: Dict[str, int] = {
    "gpt-4o": 128000,
    "gpt-3.5-turbo": 16385,
    "deepseek-ai/deepseek-r1-distill-llama-70b": 128000,
}

SUBJECT_RE = re.compile(r"(?im)^\s*subject(?:\s*line)?\s*[:：]\s*(.+)$")
CODE_FENCE_RE = re.compile(r"^```[a-zA-Z0-9_-]*\s*|\s*```$", re.MULTILINE)
RATE_LIMIT_TOKENS_RE = re.compile(
    r"Limit\s+(?P<limit>\d+)\s*,\s*Requested\s+(?P<requested>\d+)",
    re.IGNORECASE,
)


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


class ContextLengthExceededError(RuntimeError):
    """Raised when the model rejects a prompt for exceeding context length."""


class TokenRateLimitExceededError(RuntimeError):
    """Raised when one request exceeds the organization's token-per-minute cap."""

    def __init__(
        self,
        message: str,
        *,
        limit_tokens: Optional[int] = None,
        requested_tokens: Optional[int] = None,
    ) -> None:
        super().__init__(message)
        self.limit_tokens = limit_tokens
        self.requested_tokens = requested_tokens


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
        default=os.environ.get("HF_MODEL", "deepseek-ai/DeepSeek-R1-Distill-Llama-70B"),
        help="Local Hugging Face model id.",
    )
    parser.add_argument(
        "--api-mode",
        choices=("local_hf",),
        default="local_hf",
        help="Local backend marker retained for manifest compatibility.",
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
        help="Unused compatibility flag retained for CLI stability.",
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
    parser.add_argument(
        "--model-context-limit",
        type=int,
        default=0,
        help="Model context limit. Defaults to an inferred limit for known models.",
    )
    parser.add_argument(
        "--completion-token-reserve",
        type=int,
        default=DEFAULT_COMPLETION_TOKEN_RESERVE,
        help="Reserved tokens for the model response when truncating long inputs.",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=1,
        help="How many prompts to generate in one local HF batch.",
    )
    parser.add_argument(
        "--trust-remote-code",
        action="store_true",
        default=True,
        help="Pass trust_remote_code=True when loading tokenizer/model.",
    )
    parser.add_argument(
        "--no-trust-remote-code",
        dest="trust_remote_code",
        action="store_false",
        help="Disable trust_remote_code when loading tokenizer/model.",
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


def maybe_get_model_encoding(model: str) -> Any:
    try:
        import tiktoken  # type: ignore

        try:
            return tiktoken.encoding_for_model(model)
        except KeyError:
            return tiktoken.get_encoding("cl100k_base")
    except Exception:
        return None


def get_model_context_limit(model: str, requested_limit: int) -> int:
    if requested_limit > 0:
        return requested_limit
    model_norm = (model or "").strip().lower()
    for prefix, limit in MODEL_CONTEXT_LIMITS.items():
        if model_norm.startswith(prefix):
            return limit
    return 128000


def infer_api_mode(model: str, requested_mode: str) -> str:
    return "local_hf"


def build_output_dir(output_dir_arg: str) -> Path:
    if output_dir_arg:
        return Path(output_dir_arg).resolve()
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    return (SCRIPT_DIR / "runs" / stamp).resolve()


def build_output_csv_name(model: str) -> str:
    model_norm = (model or "").strip().lower()
    if "deepseek-r1-distill-llama-70b" in model_norm:
        return "LLM-MultiPhishGuard-DeepSeek-R1-Distill-Llama-70B-replace.csv"
    if model_norm.startswith("gpt-4o"):
        return "LLM-MultiPhishGuard-GPT4o-replace.csv"
    if model_norm.startswith("gpt-3.5"):
        return "LLM-MultiPhishGuard-GPT3.5-replace.csv"
    slug = re.sub(r"[^A-Za-z0-9.]+", "-", (model or "output").strip()).strip("-")
    slug = slug or "output"
    return "LLM-MultiPhishGuard-{}-replace.csv".format(slug)


def build_output_source(original_source: str, model: str, rounds: int) -> str:
    base = (original_source or "unknown").strip() or "unknown"
    model_tag = re.sub(r"[^A-Za-z0-9.]+", "-", model).strip("-") or "model"
    return "{}|multiphishguard:{}:r{}".format(base, model_tag, rounds)


def iter_batches(items: Sequence[Any], batch_size: int) -> Iterable[Sequence[Any]]:
    for start in range(0, len(items), batch_size):
        yield items[start : start + batch_size]


def make_progress_bar(*, total: int, desc: str) -> Any:
    if tqdm is None:
        return None
    return tqdm(total=total, desc=desc, unit="batch", dynamic_ncols=True, leave=True)


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


def truncate_source_email_to_fit_prompts(
    *,
    developer_prompt: str,
    subject: str,
    body: str,
    label: str,
    round_idx: int,
    model: str,
    model_context_limit: int,
    completion_token_reserve: int,
) -> Tuple[str, str, str, int]:
    email_text = render_email(subject, body)
    developer_prompt_text, user_prompt = build_adversarial_prompts(
        subject=subject,
        body=body,
        label=label,
        round_idx=round_idx,
    )
    total_prompt = developer_prompt_text + "\n\n" + user_prompt
    target_prompt_tokens = max(1, model_context_limit - max(0, completion_token_reserve))
    prompt_tokens = infer_token_count(total_prompt, model)
    if prompt_tokens <= target_prompt_tokens:
        return email_text, developer_prompt_text, user_prompt, prompt_tokens

    fixed_user_prompt = user_prompt.replace(email_text, "")
    fixed_tokens = infer_token_count(developer_prompt_text + "\n\n" + fixed_user_prompt, model)
    allowed_email_tokens = max(1, target_prompt_tokens - fixed_tokens)

    encoding = maybe_get_model_encoding(model)
    if encoding is not None:
        email_tokens = encoding.encode(email_text)
        truncated_email = encoding.decode(email_tokens[:allowed_email_tokens])
    else:
        truncated_email = email_text[: max(1, allowed_email_tokens * 4)]

    while truncated_email:
        candidate_user_prompt = user_prompt.replace(email_text, truncated_email, 1)
        candidate_total_prompt = developer_prompt_text + "\n\n" + candidate_user_prompt
        prompt_tokens = infer_token_count(candidate_total_prompt, model)
        if prompt_tokens <= target_prompt_tokens:
            return truncated_email, developer_prompt_text, candidate_user_prompt, prompt_tokens

        if encoding is not None:
            email_tokens = encoding.encode(truncated_email)
            if len(email_tokens) <= 1:
                break
            shrink_by = max(1, min(128, len(email_tokens) // 20))
            truncated_email = encoding.decode(email_tokens[:-shrink_by])
        else:
            if len(truncated_email) <= 4:
                break
            shrink_by = max(1, min(512, len(truncated_email) // 20))
            truncated_email = truncated_email[:-shrink_by]

    candidate_user_prompt = user_prompt.replace(email_text, truncated_email, 1)
    candidate_total_prompt = developer_prompt_text + "\n\n" + candidate_user_prompt
    return truncated_email, developer_prompt_text, candidate_user_prompt, infer_token_count(
        candidate_total_prompt, model
    )


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


def load_existing_results(path: Path) -> List[RowResult]:
    if not path.exists():
        return []

    with path.open("r", encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))

    results: List[RowResult] = []
    for row in rows:
        results.append(
            RowResult(
                row_number=int(str(row.get("row_number", "0") or "0")),
                rewritten_subject=normalize_text(row.get("Subject", "")),
                rewritten_body=normalize_text(row.get("Body", "")),
                label=str(row.get("label", "")).strip(),
                data_source=str(row.get("data_source", "")).strip(),
                original_subject=normalize_text(row.get("original_subject", "")),
                original_body=normalize_text(row.get("original_body", "")),
                original_data_source=str(row.get("original_data_source", "")).strip(),
                model=str(row.get("model", "")).strip(),
                api_mode=str(row.get("api_mode", "")).strip(),
                prompt_rounds=int(str(row.get("prompt_rounds", "0") or "0")),
                skipped=str(row.get("skipped", "")).strip().lower() in {"1", "true", "yes"},
                skip_reason=str(row.get("skip_reason", "")).strip(),
            )
        )
    return results


def recover_results_from_call_logs(
    *,
    existing_results: Sequence[RowResult],
    selected_rows: Sequence[Tuple[int, Dict[str, str]]],
    calls_jsonl_path: Path,
    subject_column: str,
    body_column: str,
    label_column: str,
    data_source_column: str,
    model: str,
    api_mode: str,
    rounds: int,
) -> List[RowResult]:
    if not calls_jsonl_path.exists():
        return list(existing_results)

    input_lookup = {row_number: row for row_number, row in selected_rows}
    results = list(existing_results)
    next_row_number = results[-1].row_number + 1 if results else 2

    calls_by_row: Dict[int, Dict[int, Dict[str, Any]]] = {}
    with calls_jsonl_path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            record = json.loads(line)
            row_number = int(record.get("row_number"))
            round_idx = int(record.get("round"))
            calls_by_row.setdefault(row_number, {})[round_idx] = record

    while next_row_number in input_lookup:
        round_records = calls_by_row.get(next_row_number, {})
        if any(round_idx not in round_records for round_idx in range(1, rounds + 1)):
            break

        row = input_lookup[next_row_number]
        original_subject = normalize_text(row.get(subject_column, ""))
        original_body = normalize_text(row.get(body_column, ""))
        label = str(row.get(label_column, "")).strip()
        original_data_source = str(row.get(data_source_column, "")).strip()

        final_subject = original_subject
        final_body = original_body
        for round_idx in range(1, rounds + 1):
            response_text = normalize_text(round_records[round_idx].get("response_text", ""))
            final_subject, final_body = parse_subject_and_body(response_text, final_subject)

        results.append(
            RowResult(
                row_number=next_row_number,
                rewritten_subject=final_subject,
                rewritten_body=final_body,
                label=label,
                data_source=build_output_source(original_data_source, model, rounds),
                original_subject=original_subject,
                original_body=original_body,
                original_data_source=original_data_source,
                model=model,
                api_mode=api_mode,
                prompt_rounds=rounds,
                skipped=False,
                skip_reason="",
            )
        )
        next_row_number += 1

    return results


def validate_columns(rows: Sequence[Dict[str, str]], required_columns: Sequence[str]) -> None:
    if not rows:
        raise ValueError("Input CSV does not contain any data rows.")
    missing = [name for name in required_columns if name not in rows[0]]
    if missing:
        raise ValueError("Missing required columns: {}".format(", ".join(missing)))


class LocalTransformersTextClient:
    def __init__(
        self,
        *,
        model: str,
        api_mode: str,
        trust_remote_code: bool,
        temperature: float,
        max_completion_tokens: Optional[int],
        max_retries: int,  # retained for interface compatibility
    ) -> None:
        self.model = model
        self.api_mode = api_mode
        self.temperature = temperature
        self.max_completion_tokens = max_completion_tokens
        self.max_retries = max_retries
        self.tokenizer = AutoTokenizer.from_pretrained(
            self.model,
            trust_remote_code=trust_remote_code,
        )
        self.tokenizer.padding_side = "left"
        if self.tokenizer.pad_token_id is None and self.tokenizer.eos_token_id is not None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        self.model_obj = AutoModelForCausalLM.from_pretrained(
            self.model,
            trust_remote_code=trust_remote_code,
            torch_dtype="auto",
            device_map="auto",
        )
        if (
            self.model_obj.generation_config.pad_token_id is None
            and self.tokenizer.pad_token_id is not None
        ):
            self.model_obj.generation_config.pad_token_id = self.tokenizer.pad_token_id

    def _input_device(self) -> torch.device:
        return next(self.model_obj.parameters()).device

    def _generation_kwargs(self) -> Dict[str, Any]:
        generation_kwargs: Dict[str, Any] = {
            "max_new_tokens": self.max_completion_tokens or DEFAULT_COMPLETION_TOKEN_RESERVE,
        }
        if self.temperature is not None and self.temperature > 0:
            generation_kwargs["do_sample"] = True
            generation_kwargs["temperature"] = self.temperature
        if self.tokenizer.pad_token_id is not None:
            generation_kwargs["pad_token_id"] = self.tokenizer.pad_token_id
        return generation_kwargs

    def _render_prompt(
        self,
        developer_prompt: str,
        user_prompt: str,
    ) -> Tuple[List[Dict[str, str]], str]:
        messages = [
            {"role": "system", "content": developer_prompt},
            {"role": "user", "content": user_prompt},
        ]
        prompt_text = self.tokenizer.apply_chat_template(
            messages,
            add_generation_prompt=True,
            tokenize=False,
        )
        return messages, prompt_text

    def _generate_single(self, developer_prompt: str, user_prompt: str) -> Tuple[str, Dict[str, Any]]:
        messages, prompt_text = self._render_prompt(developer_prompt, user_prompt)
        inputs = self.tokenizer(prompt_text, return_tensors="pt").to(self._input_device())
        generation_kwargs = self._generation_kwargs()
        with torch.inference_mode():
            outputs = self.model_obj.generate(**inputs, **generation_kwargs)
        generated_ids = outputs[0][inputs["input_ids"].shape[-1] :]
        text = self.tokenizer.decode(generated_ids, skip_special_tokens=True).strip()
        response_payload = {
            "model": self.model,
            "messages": messages,
            "generated_text": text,
            "api_mode": self.api_mode,
            "max_new_tokens": generation_kwargs["max_new_tokens"],
        }
        return normalize_text(text), response_payload

    def generate_batch(
        self,
        developer_prompts: Sequence[str],
        user_prompts: Sequence[str],
    ) -> Tuple[List[str], List[Dict[str, Any]]]:
        if not developer_prompts:
            return [], []
        if len(developer_prompts) != len(user_prompts):
            raise ValueError("developer_prompts and user_prompts must have the same length.")
        if len(developer_prompts) == 1:
            text, payload = self._generate_single(developer_prompts[0], user_prompts[0])
            return [text], [payload]

        rendered = [
            self._render_prompt(developer_prompt, user_prompt)
            for developer_prompt, user_prompt in zip(developer_prompts, user_prompts)
        ]
        messages_list = [messages for messages, _ in rendered]
        prompt_texts = [prompt_text for _, prompt_text in rendered]
        inputs = self.tokenizer(
            prompt_texts,
            return_tensors="pt",
            padding=True,
        ).to(self._input_device())
        generation_kwargs = self._generation_kwargs()
        try:
            with torch.inference_mode():
                outputs = self.model_obj.generate(**inputs, **generation_kwargs)
        except Exception:
            texts: List[str] = []
            payloads: List[Dict[str, Any]] = []
            for developer_prompt, user_prompt in zip(developer_prompts, user_prompts):
                text, payload = self._generate_single(developer_prompt, user_prompt)
                texts.append(text)
                payloads.append(payload)
            return texts, payloads

        prompt_length = inputs["input_ids"].shape[-1]
        generated_ids = outputs[:, prompt_length:]
        texts = [
            normalize_text(text)
            for text in self.tokenizer.batch_decode(generated_ids, skip_special_tokens=True)
        ]
        payloads: List[Dict[str, Any]] = []
        for messages, text in zip(messages_list, texts):
            payloads.append(
                {
                    "model": self.model,
                    "messages": messages,
                    "generated_text": text,
                    "api_mode": self.api_mode,
                    "max_new_tokens": generation_kwargs["max_new_tokens"],
                }
            )
        return texts, payloads


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
    if args.completion_token_reserve < 0:
        raise ValueError("--completion-token-reserve must be non-negative.")
    if args.batch_size < 1:
        raise ValueError("--batch-size must be at least 1.")

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
    model_context_limit = get_model_context_limit(args.model, args.model_context_limit)
    client = LocalTransformersTextClient(
        model=args.model,
        api_mode=api_mode,
        trust_remote_code=args.trust_remote_code,
        temperature=args.temperature,
        max_completion_tokens=(
            args.max_completion_tokens
            if args.max_completion_tokens is not None
            else (args.completion_token_reserve or None)
        ),
        max_retries=args.max_retries,
    )

    existing_results = load_existing_results(output_csv_path)
    results = recover_results_from_call_logs(
        existing_results=existing_results,
        selected_rows=selected_rows,
        calls_jsonl_path=calls_jsonl_path,
        subject_column=args.subject_column,
        body_column=args.body_column,
        label_column=args.label_column,
        data_source_column=args.data_source_column,
        model=args.model,
        api_mode=api_mode,
        rounds=args.rounds,
    )
    if len(results) > len(existing_results):
        write_csv(output_csv_path, results)
        write_simplified_csv(simplified_csv_path, results)
        print(
            "[recovery] restored {} rows from {}".format(
                len(results) - len(existing_results), calls_jsonl_path
            ),
            file=sys.stderr,
        )
    completed_rows = len(results)
    if completed_rows > len(selected_rows):
        raise ValueError(
            "Existing output has {} rows, but the current input only selected {} rows.".format(
                completed_rows, len(selected_rows)
            )
        )
    if completed_rows > 0:
        print(
            "[resume] loaded {} existing rows from {}".format(completed_rows, output_csv_path),
            file=sys.stderr,
        )

    pending_rows = selected_rows[completed_rows:]
    progress_bar = make_progress_bar(
        total=math.ceil(len(pending_rows) / args.batch_size) if pending_rows else 0,
        desc="MultiPhishGuard replace",
    )

    for batch_index, batch_rows in enumerate(iter_batches(pending_rows, args.batch_size), start=1):
        batch_states: List[Dict[str, Any]] = []
        for selected_index, (row_number, row) in enumerate(
            batch_rows,
            start=completed_rows + (batch_index - 1) * args.batch_size + 1,
        ):
            original_subject = normalize_text(row.get(args.subject_column, ""))
            original_body = normalize_text(row.get(args.body_column, ""))
            label = str(row.get(args.label_column, "")).strip()
            original_data_source = str(row.get(args.data_source_column, "")).strip()
            batch_states.append(
                {
                    "selected_index": selected_index,
                    "row_number": row_number,
                    "original_subject": original_subject,
                    "original_body": original_body,
                    "label": label,
                    "original_data_source": original_data_source,
                    "final_subject": original_subject,
                    "final_body": original_body,
                    "was_skipped": False,
                    "skip_reason": "",
                }
            )

        for round_idx in range(1, args.rounds + 1):
            active_states: List[Dict[str, Any]] = []
            developer_prompts: List[str] = []
            user_prompts: List[str] = []
            prompt_token_estimates: List[int] = []

            for state in batch_states:
                if state["was_skipped"]:
                    continue
                developer_prompt, _ = build_adversarial_prompts(
                    subject=state["final_subject"],
                    body=state["final_body"],
                    label=state["label"],
                    round_idx=round_idx,
                )
                (
                    _,
                    developer_prompt,
                    user_prompt,
                    prompt_token_estimate,
                ) = truncate_source_email_to_fit_prompts(
                    developer_prompt=developer_prompt,
                    subject=state["final_subject"],
                    body=state["final_body"],
                    label=state["label"],
                    round_idx=round_idx,
                    model=args.model,
                    model_context_limit=model_context_limit,
                    completion_token_reserve=args.completion_token_reserve,
                )
                active_states.append(state)
                developer_prompts.append(developer_prompt)
                user_prompts.append(user_prompt)
                prompt_token_estimates.append(prompt_token_estimate)

            if not active_states:
                break

            response_texts, raw_responses = client.generate_batch(developer_prompts, user_prompts)
            for state, developer_prompt, user_prompt, prompt_token_estimate, response_text, raw_response in zip(
                active_states,
                developer_prompts,
                user_prompts,
                prompt_token_estimates,
                response_texts,
                raw_responses,
            ):
                state["final_subject"], state["final_body"] = parse_subject_and_body(
                    response_text,
                    state["final_subject"],
                )

                append_jsonl(
                    calls_jsonl_path,
                    {
                        "timestamp_utc": now_utc_iso(),
                        "row_number": state["row_number"],
                        "selected_index": state["selected_index"],
                        "round": round_idx,
                        "label": state["label"],
                        "model": args.model,
                        "api_mode": api_mode,
                        "prompt_token_estimate": prompt_token_estimate,
                        "developer_prompt": developer_prompt,
                        "user_prompt": user_prompt,
                        "response_text": response_text,
                        "raw_response": raw_response,
                    },
                )

            if args.sleep_seconds > 0:
                time.sleep(args.sleep_seconds)

        for state in batch_states:
            results.append(
                RowResult(
                    row_number=state["row_number"],
                    rewritten_subject=state["final_subject"],
                    rewritten_body=state["final_body"],
                    label=state["label"],
                    data_source=build_output_source(
                        state["original_data_source"], args.model, args.rounds
                    ),
                    original_subject=state["original_subject"],
                    original_body=state["original_body"],
                    original_data_source=state["original_data_source"],
                    model=args.model,
                    api_mode=api_mode,
                    prompt_rounds=args.rounds,
                    skipped=state["was_skipped"],
                    skip_reason=state["skip_reason"],
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

        if args.print_every > 0 and batch_index % args.print_every == 0:
            processed_rows = min(completed_rows, len(selected_rows))
            print(
                "[progress] processed {}/{} rows".format(processed_rows, len(selected_rows)),
                file=sys.stderr,
            )
        if progress_bar is not None:
            progress_bar.update(1)

    if progress_bar is not None:
        progress_bar.close()

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
        "model_context_limit": model_context_limit,
        "completion_token_reserve": args.completion_token_reserve,
        "rounds": args.rounds,
        "temperature": args.temperature,
        "save_every": args.save_every,
        "batch_size": args.batch_size,
        "notes": [
            "The paper describes a GPT-4o-based adversarial agent in Section 3.4.",
            "This script focuses on pure adversarial email generation without any external detector loop.",
            "This replace variant uses local Hugging Face loading with AutoTokenizer/AutoModelForCausalLM and apply_chat_template.",
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
