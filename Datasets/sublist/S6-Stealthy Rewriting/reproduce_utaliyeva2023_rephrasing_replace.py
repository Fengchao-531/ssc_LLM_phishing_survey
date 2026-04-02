#!/usr/bin/env python3
"""Reproduce the rephrasing setup from Utaliyeva et al. (HPCC/DSS 2023).

The paper evaluates spam-filter robustness after rewriting spam emails with
paper-style prompts such as:

- "Rewrite the following email to be less spammy: 'email text'"
- "Rewrite the following email: 'email text'"

This replace variant keeps the same dataset-side workflow, but swaps the
generation backend to a locally loaded Hugging Face causal LM following the
official DeepSeek chat-template usage pattern.
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
SIMPLIFIED_OUTPUT_PATH = SCRIPT_DIR / "UTA-LLM-P-replace.csv"

DEFAULT_TIMEOUT = 120
DEFAULT_INPUT_PATH = SCRIPT_DIR / "HW-P.csv"
DEFAULT_LABEL_COLUMN = "label"
DEFAULT_LABEL_VALUE = "1"
DEFAULT_SUBJECT_COLUMN = "Subject"
DEFAULT_BODY_COLUMN = "Body"
DEFAULT_COMPLETION_TOKEN_RESERVE = 512

MODEL_CONTEXT_LIMITS: Dict[str, int] = {
    "gpt-3.5-turbo": 16385,
    "deepseek-ai/deepseek-r1-distill-llama-70b": 128000,
}

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


class ContextLengthExceededError(RuntimeError):
    """Raised when the model prompt exceeds the configured local context length."""


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Batch reproduction of the Utaliyeva et al. 2023 email rephrasing setup."
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
        default=os.environ.get("HF_MODEL", "deepseek-ai/DeepSeek-R1-Distill-Llama-70B"),
        help="Local Hugging Face model id.",
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


def get_model_context_limit(model: str, requested_limit: int) -> int:
    if requested_limit > 0:
        return requested_limit
    model_norm = (model or "").strip().lower()
    for prefix, limit in MODEL_CONTEXT_LIMITS.items():
        if model_norm.startswith(prefix):
            return limit
    return 16385


def maybe_get_model_encoding(model: str) -> Any:
    try:
        import tiktoken  # type: ignore

        try:
            return tiktoken.encoding_for_model(model)
        except KeyError:
            return tiktoken.get_encoding("cl100k_base")
    except Exception:
        return None


def truncate_email_to_fit_prompt(
    *,
    prompt_kind: str,
    email_text: str,
    model: str,
    model_context_limit: int,
    completion_token_reserve: int,
) -> Tuple[str, str, int]:
    target_prompt_tokens = max(1, model_context_limit - max(0, completion_token_reserve))
    prompt_text = PROMPT_TEMPLATES[prompt_kind].format(email_text=email_text)
    prompt_tokens = infer_token_count(prompt_text, model)
    if prompt_tokens <= target_prompt_tokens:
        return email_text, prompt_text, prompt_tokens

    base_prompt = PROMPT_TEMPLATES[prompt_kind].format(email_text="")
    base_prompt_tokens = infer_token_count(base_prompt, model)
    allowed_email_tokens = max(1, target_prompt_tokens - base_prompt_tokens)

    encoding = maybe_get_model_encoding(model)
    if encoding is not None:
        email_tokens = encoding.encode(email_text)
        truncated_email = encoding.decode(email_tokens[:allowed_email_tokens])
    else:
        truncated_email = email_text[: max(1, allowed_email_tokens * 4)]

    # Trim a little further if template framing or tokenization variance still pushes us over.
    while truncated_email:
        prompt_text = PROMPT_TEMPLATES[prompt_kind].format(email_text=truncated_email)
        prompt_tokens = infer_token_count(prompt_text, model)
        if prompt_tokens <= target_prompt_tokens:
            return truncated_email, prompt_text, prompt_tokens

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

    final_prompt = PROMPT_TEMPLATES[prompt_kind].format(email_text=truncated_email)
    return truncated_email, final_prompt, infer_token_count(final_prompt, model)


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


class LocalTransformersChatClient:
    def __init__(
        self,
        *,
        model: str,
        trust_remote_code: bool,
        temperature: Optional[float],
        max_completion_tokens: Optional[int],
    ) -> None:
        self.model = model
        self.temperature = temperature
        self.max_completion_tokens = max_completion_tokens
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

    def _render_prompt(self, user_prompt: str) -> Tuple[List[Dict[str, str]], str]:
        messages = [{"role": "user", "content": user_prompt}]
        prompt_text = self.tokenizer.apply_chat_template(
            messages,
            add_generation_prompt=True,
            tokenize=False,
        )
        return messages, prompt_text

    def _complete_single(self, user_prompt: str) -> Tuple[str, Dict[str, Any]]:
        messages, prompt_text = self._render_prompt(user_prompt)
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
            "max_new_tokens": generation_kwargs["max_new_tokens"],
        }
        return text, response_payload

    def complete_batch(self, user_prompts: Sequence[str]) -> Tuple[List[str], List[Dict[str, Any]]]:
        if not user_prompts:
            return [], []
        if len(user_prompts) == 1:
            text, payload = self._complete_single(user_prompts[0])
            return [text], [payload]

        rendered = [self._render_prompt(user_prompt) for user_prompt in user_prompts]
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
            for user_prompt in user_prompts:
                text, payload = self._complete_single(user_prompt)
                texts.append(text)
                payloads.append(payload)
            return texts, payloads

        prompt_length = inputs["input_ids"].shape[-1]
        generated_ids = outputs[:, prompt_length:]
        texts = [
            text.strip()
            for text in self.tokenizer.batch_decode(generated_ids, skip_special_tokens=True)
        ]
        payloads: List[Dict[str, Any]] = []
        for messages, text in zip(messages_list, texts):
            payloads.append(
                {
                    "model": self.model,
                    "messages": messages,
                    "generated_text": text,
                    "max_new_tokens": generation_kwargs["max_new_tokens"],
                }
            )
        return texts, payloads


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
    if "deepseek-r1-distill-llama-70b" in model_norm:
        return "LLM-DeepSeek-R1-Distill-Llama-70B-replace.csv"
    if model_norm.startswith("gpt-3.5"):
        return "LLM-GPT3.5-replace.csv"
    slug = re.sub(r"[^A-Za-z0-9.]+", "-", (model or "output").strip()).strip("-")
    slug = slug or "output"
    return "LLM-{}-replace.csv".format(slug)


def iter_batches(items: Sequence[Any], batch_size: int) -> Iterable[Sequence[Any]]:
    for start in range(0, len(items), batch_size):
        yield items[start : start + batch_size]


def make_progress_bar(*, total: int, desc: str) -> Any:
    if tqdm is None:
        return None
    return tqdm(total=total, desc=desc, unit="batch", dynamic_ncols=True, leave=True)


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
        raise ValueError(f"Missing required columns: {', '.join(missing)}")


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
    rewritten_csv_path = output_dir / build_output_csv_name(args.model)
    simplified_csv_path = SIMPLIFIED_OUTPUT_PATH
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

    model_context_limit = get_model_context_limit(args.model, args.model_context_limit)

    client = LocalTransformersChatClient(
        model=args.model,
        trust_remote_code=args.trust_remote_code,
        temperature=args.temperature,
        max_completion_tokens=args.max_completion_tokens,
    )

    results: List[RowResult] = []
    selected_prompt_kinds = prompt_kinds_from_arg(args.prompt_kind)
    completed_results = 0
    total_batches = len(selected_prompt_kinds) * math.ceil(len(selected_rows) / args.batch_size)
    progress_bar = make_progress_bar(total=total_batches, desc="Utaliyeva replace")

    for prompt_kind in selected_prompt_kinds:
        for batch_index, batch_rows in enumerate(iter_batches(selected_rows, args.batch_size), start=1):
            batch_states: List[Dict[str, Any]] = []
            for selected_index, (row_number, row) in enumerate(
                batch_rows,
                start=(batch_index - 1) * args.batch_size + 1,
            ):
                original_subject = normalize_text(row.get(args.subject_column, ""))
                original_body = normalize_text(row.get(args.body_column, ""))
                label = str(row.get(args.label_column, "")).strip()
                original_data_source = str(row.get(args.data_source_column, "")).strip()
                seed_email = render_email(original_subject, original_body)
                batch_states.append(
                    {
                        "selected_index": selected_index,
                        "row_number": row_number,
                        "label": label,
                        "original_subject": original_subject,
                        "original_body": original_body,
                        "original_data_source": original_data_source,
                        "current_email": seed_email,
                        "final_subject": original_subject,
                        "final_body": original_body,
                        "token_estimate": infer_token_count(seed_email, args.model),
                        "skip_reason": "",
                        "was_skipped": False,
                    }
                )

            for round_idx in range(1, args.rounds + 1):
                active_states: List[Dict[str, Any]] = []
                prompt_texts: List[str] = []
                prompt_token_estimates: List[int] = []

                for state in batch_states:
                    if state["was_skipped"]:
                        continue
                    state["current_email"], prompt_text, prompt_token_estimate = truncate_email_to_fit_prompt(
                        prompt_kind=prompt_kind,
                        email_text=state["current_email"],
                        model=args.model,
                        model_context_limit=model_context_limit,
                        completion_token_reserve=args.completion_token_reserve,
                    )
                    active_states.append(state)
                    prompt_texts.append(prompt_text)
                    prompt_token_estimates.append(prompt_token_estimate)

                if not active_states:
                    break

                response_texts, raw_responses = client.complete_batch(prompt_texts)
                for state, prompt_text, prompt_token_estimate, response_text, raw_response in zip(
                    active_states,
                    prompt_texts,
                    prompt_token_estimates,
                    response_texts,
                    raw_responses,
                ):
                    state["final_subject"], state["final_body"] = parse_subject_and_body(
                        response_text,
                        state["final_subject"],
                    )
                    state["current_email"] = render_email(state["final_subject"], state["final_body"])

                    append_jsonl(
                        calls_jsonl_path,
                        {
                            "timestamp_utc": now_utc_iso(),
                            "row_number": state["row_number"],
                            "selected_index": state["selected_index"],
                            "prompt_kind": prompt_kind,
                            "round": round_idx,
                            "model": args.model,
                            "token_estimate": state["token_estimate"],
                            "prompt_token_estimate": prompt_token_estimate,
                            "original_subject": state["original_subject"],
                            "original_body": state["original_body"],
                            "input_email_for_round": prompt_text,
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
                        prompt_kind=prompt_kind,
                        prompt_rounds=args.rounds,
                        original_subject=state["original_subject"],
                        original_body=state["original_body"],
                        rewritten_subject=state["final_subject"],
                        rewritten_body=state["final_body"],
                        label=state["label"],
                        original_data_source=state["original_data_source"],
                        output_data_source=build_output_source(
                            state["original_data_source"], prompt_kind, args.rounds
                        ),
                        model=args.model,
                        token_estimate=state["token_estimate"],
                        skipped=state["was_skipped"],
                        skip_reason=state["skip_reason"],
                    )
                )
                completed_results += 1
                if args.save_every > 0 and completed_results % args.save_every == 0:
                    write_csv(rewritten_csv_path, results)
                    write_simplified_csv(simplified_csv_path, results)
                    print(
                        f"[checkpoint] saved {completed_results} rows to {rewritten_csv_path}",
                        file=sys.stderr,
                    )

            if args.print_every > 0 and batch_index % args.print_every == 0:
                processed_rows = min(batch_index * args.batch_size, len(selected_rows))
                print(
                    f"[progress] processed {processed_rows}/{len(selected_rows)} selected rows for {prompt_kind}",
                    file=sys.stderr,
                )
            if progress_bar is not None:
                progress_bar.update(1)

    if progress_bar is not None:
        progress_bar.close()

    write_csv(rewritten_csv_path, results)
    write_simplified_csv(simplified_csv_path, results)

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
        "simplified_output_csv": str(simplified_csv_path),
        "selected_rows": len(selected_rows),
        "result_rows": len(results),
        "prompt_kinds": selected_prompt_kinds,
        "rounds": args.rounds,
        "model": args.model,
        "historical_token_limit": args.historical_token_limit,
        "token_limit_skip_enabled": not args.disable_token_limit_skip,
        "model_context_limit": model_context_limit,
        "completion_token_reserve": args.completion_token_reserve,
        "temperature": args.temperature,
        "save_every": args.save_every,
        "batch_size": args.batch_size,
        "max_completion_tokens": args.max_completion_tokens,
        "notes": [
            "The large-scale paper prompt was 'Rewrite the following email to be less spammy: \"email text\"'.",
            "The paper also compares this against the general prompt 'Rewrite the following email: \"email text\"'.",
            "Iterative rewriting is supported here because the paper notes it as beneficial, but does not report one fixed multi-round schedule.",
            "This replace variant uses local Hugging Face loading with AutoTokenizer/AutoModelForCausalLM and apply_chat_template.",
        ],
    }
    manifest_path.write_text(json.dumps(manifest, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")

    print(f"Wrote {len(results)} rows to {rewritten_csv_path}", file=sys.stderr)
    print(f"Wrote simplified export to {simplified_csv_path}", file=sys.stderr)
    print(f"Saved call logs to {calls_jsonl_path}", file=sys.stderr)
    print(f"Saved manifest to {manifest_path}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
