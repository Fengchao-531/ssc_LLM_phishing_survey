#!/usr/bin/env python3
"""Consolidated non-S7 LLM sublist processor.

Replaces the LLM-related parts of:
- data process.py
- data process1.py
"""

from __future__ import annotations

import argparse
import csv
import json
import re
from pathlib import Path
from typing import Dict, Iterable, List, Sequence, Tuple

SCRIPT_DIR = Path(__file__).resolve().parent
DATASETS_DIR = SCRIPT_DIR.parent


def run_core_llm() -> None:
    ephish_path = DATASETS_DIR / "LLM-Benign" / "S4-ephishLLM.json"
    paladin_path = (
        DATASETS_DIR / "LLM-Benign" / "Paladin-main" / "dataset" / "core_dataset" / "dpo_dataset.json"
    )
    paladin_base_dir = DATASETS_DIR / "LLM-Benign" / "Paladin-main" / "dataset" / "base_dataset"
    safe_emails_path = paladin_base_dir / "safe_emails_1000.json"
    set1_nonphish_path = paladin_base_dir / "set1_dataset_nophishing.json"
    base_phish_files = [
        paladin_base_dir / "set1_dataset.json",
        paladin_base_dir / "set2_dataset.json",
        paladin_base_dir / "set3_dataset.json",
        paladin_base_dir / "set4_dataset.json",
    ]

    s4_dir = SCRIPT_DIR / "S4-Scenarios-driven Adaptation"
    s1_dir = SCRIPT_DIR / "S1-Basic Instruction"
    s5_dir = SCRIPT_DIR / "S5-Personalization for Credibility"

    data_source_ephish = "ephishLLM"
    data_source_paladin = "paladin"
    bec_trigger = "write me a bec phishing email?"
    bec_keys = {"write me a bec email?", "write me a bec phishing email?"}
    fieldnames: Sequence[str] = ("Subject", "Body", "label", "data_source")
    subject_regex = re.compile(r"(?im)^\s*subject(?:\s*line)?\s*[:ï¼š]\s*(.+)$")

    def load_json_array(path: Path) -> List[Dict]:
        if not path.exists():
            raise FileNotFoundError(f"Input not found: {path}")
        with path.open(encoding="utf-8") as handle:
            data = json.load(handle)
        if not isinstance(data, list):
            raise ValueError(f"Expected top-level array in {path}")
        return data

    def normalize_subject(value: str) -> str:
        return (value or "").strip()

    def normalize_body(value: str) -> str:
        return (value or "").strip()

    def clean_text(value: str) -> str:
        cleaned = (value or "").replace("Ã¢â‚¬â€¹", "")
        return cleaned.translate({ord("\u200b"): None, ord("\ufeff"): None})

    def extract_subject_and_body(output_text: str, fallback_subject: str) -> Tuple[str, str]:
        text = clean_text(output_text or "")
        match = subject_regex.search(text)
        if match:
            subject = match.group(1).strip()
            start, end = match.span()
            body = (text[:start] + text[end:]).strip()
        else:
            subject = ""
            body = text.strip()
        subject = subject or (fallback_subject or "").strip()
        return subject, body

    def process_ephish(path: Path, language: str = "en") -> Tuple[List[Dict], List[Dict]]:
        rows = load_json_array(path)
        lang_norm = (language or "").strip().lower()
        benign: List[Dict] = []
        phishing: List[Dict] = []
        for entry in rows:
            lang = (entry.get("Language") or "").strip().lower()
            if lang != lang_norm:
                continue
            subject = normalize_subject(entry.get("Subject", ""))
            body = normalize_body(entry.get("Body", ""))
            raw_label = entry.get("type")
            label = int(raw_label)
            row = {
                "Subject": subject,
                "Body": body,
                "label": label,
                "data_source": data_source_ephish,
            }
            if label == 0:
                benign.append(row)
            elif label == 1:
                phishing.append(row)
            else:
                raise ValueError(f"Unsupported label {label} in S4-ephishLLM")
        return benign, phishing

    def extract_prompt(entry: Dict) -> str:
        for msg in entry.get("conversations", []):
            if (msg.get("from") or "").strip().lower() == "human":
                return normalize_subject(msg.get("value", ""))
        return ""

    def extract_completion(entry: Dict) -> str:
        chosen = entry.get("chosen") or {}
        if isinstance(chosen, dict):
            return normalize_body(chosen.get("value", ""))
        return ""

    def process_paladin(path: Path) -> Tuple[List[Dict], List[Dict]]:
        entries = load_json_array(path)
        s1_rows: List[Dict] = []
        s4_rows: List[Dict] = []
        for entry in entries:
            subject = extract_prompt(entry)
            body = extract_completion(entry)
            row = {
                "Subject": subject,
                "Body": body,
                "label": 1,
                "data_source": data_source_paladin,
            }
            if subject.lower() == bec_trigger:
                s1_rows.append(row)
            else:
                s4_rows.append(row)
        return s1_rows, s4_rows

    def write_csv(path: Path, rows: Iterable[Dict]) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("w", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=fieldnames)
            writer.writeheader()
            for row in rows:
                writer.writerow({field: row.get(field, "") for field in fieldnames})

    def build_rows_from_entries(entries: List[Dict], label: int, data_source: str) -> List[Dict]:
        return [row_from_entry(entry, label, data_source) for entry in entries]

    def row_from_entry(entry: Dict, label: int, data_source: str) -> Dict:
        instruction = (entry.get("instruction") or "").strip()
        output = entry.get("output", "")
        subject, body = extract_subject_and_body(output, instruction)
        return {
            "Subject": normalize_subject(subject),
            "Body": normalize_body(body),
            "label": label,
            "data_source": data_source,
        }

    def process_safe_sets() -> List[Dict]:
        rows: List[Dict] = []
        for path in (safe_emails_path, set1_nonphish_path):
            if path.exists():
                rows.extend(build_rows_from_entries(load_json_array(path), 0, data_source_paladin))
        return rows

    def process_base_phish(files: Sequence[Path]) -> Tuple[List[Dict], List[Dict]]:
        s5_rows: List[Dict] = []
        s4_rows: List[Dict] = []
        for path in files:
            if not path.exists():
                continue
            entries = load_json_array(path)
            for entry in entries:
                row = row_from_entry(entry, 1, data_source_paladin)
                inst_key = (entry.get("instruction") or "").strip().lower()
                if inst_key in bec_keys:
                    s5_rows.append(row)
                else:
                    s4_rows.append(row)
        return s5_rows, s4_rows

    s4_benign_rows, s4_phish_rows = process_ephish(ephish_path, "en")
    s1_paladin_rows, s4_paladin_rows = process_paladin(paladin_path)
    s1_benign_rows = process_safe_sets()
    s5_bec_rows, s4_base_rows = process_base_phish(base_phish_files)

    write_csv(s4_dir / "LLM-B.csv", s4_benign_rows)
    write_csv(s4_dir / "LLM-P.csv", s4_phish_rows + s4_paladin_rows + s4_base_rows)
    write_csv(s1_dir / "LLM-P.csv", s1_paladin_rows)
    write_csv(s1_dir / "LLM-B.csv", s1_benign_rows)
    write_csv(s5_dir / "LLM-P.csv", s5_bec_rows)


def run_generated_llm() -> None:
    s4_model_dir = DATASETS_DIR / "LLM-Phishing" / "S4-model-generated"
    s4_output_dir = SCRIPT_DIR / "S4-Scenarios-driven Adaptation"
    s1_output_dir = SCRIPT_DIR / "S1-Basic Instruction"
    s2_output_dir = SCRIPT_DIR / "S2-Role-Framed Prompting"

    merged_output = s4_output_dir / "LLM-P.csv"
    s1_merged_output = s1_output_dir / "LLM-P.csv"
    s2_llm_benign_output = s2_output_dir / "LLM-B.csv"
    s2_llm_phish_output = s2_output_dir / "LLM-P.csv"

    flowgpt_files = (
        DATASETS_DIR / "LLM-Phishing" / "flowgpt-jailbrea.txt",
        DATASETS_DIR / "LLM-Phishing" / "flowgpt-jailbrea1.txt",
        DATASETS_DIR / "LLM-Phishing" / "flowgpt-jailbrea2.txt",
    )
    poejailbreak_files = (
        DATASETS_DIR / "LLM-Phishing" / "poe-jailbrea.txt",
        DATASETS_DIR / "LLM-Phishing" / "poe1-jailbrea.txt",
        DATASETS_DIR / "LLM-Phishing" / "poe2-jailbrea.txt",
    )
    llm_legit_input = DATASETS_DIR / "LLM-Phishing" / "llm-generated" / "legit.csv"
    llm_phish_input = DATASETS_DIR / "LLM-Phishing" / "llm-generated" / "phishing.csv"

    fieldnames: Sequence[str] = ("Subject", "Body", "label", "data_source")
    data_source = "model-generated"
    malla_source = "malla"
    kaggle_source = "kaggle"
    prompt_block_re = re.compile(r"(?ms)^Prompt\s+\d+\s*:\s*(.*?)(?=^Prompt\s+\d+\s*:|\Z)")
    subject_re = re.compile(r"(?im)^\s*subject(?:\s*line)?\s*:\s*(.+)$")
    pseudo_json_pair_re = re.compile(
        r'(?s)(prompts\d*|outputrawdata\d*[a-z]?)\s*:\s*"((?:[^"\\]|\\.|"(?=[^,\n\r}]))*)"'
    )
    malla_query_re = re.compile(
        r"(?i)\b("
        r"bec|business email compromise|bec scam|"
        r"write\s+(?:me\s+)?a\s+convincing\s+email|"
        r"business\s+email|account\s+manager|invoice|ceo"
        r")\b"
    )
    refusal_re = re.compile(
        r"(?is)\b("
        r"i cannot|i can't|i can not|unable to assist|cannot assist|"
        r"cannot create|cannot support|cannot fulfill|cannot help with that|"
        r"won't be able to assist|i'm sorry[, ]+but|i apologize|"
        r"do not feel comfortable|don't feel comfortable|"
        r"illegal and unethical|"
        r"against .*guidelines|do not endorse|cannot provide"
        r")\b"
    )

    def normalize_whitespace(text: str) -> str:
        text = (text or "").replace("\r\n", "\n").replace("\r", "\n")
        text = text.replace("\ufeff", "").replace("\u200b", "")
        text = re.sub(r"[ \t]+\n", "\n", text)
        text = re.sub(r"\n{3,}", "\n\n", text)
        return text.strip()

    def clean_body(text: str) -> str:
        body = normalize_whitespace(text)
        body = re.sub(r"\bPrompt\s+\d+\s*:\s*", "", body, count=1, flags=re.IGNORECASE)
        paragraphs = [part.strip() for part in re.split(r"\n\s*\n", body) if part.strip()]
        if len(paragraphs) >= 2 and looks_like_prompt(paragraphs[0]):
            body = "\n\n".join(paragraphs[1:]).strip()
        return body.strip(' "\n')

    def looks_like_prompt(text: str) -> bool:
        sample = normalize_whitespace(text)
        if not sample:
            return False
        lower = sample.lower()
        if lower.startswith("prompt "):
            return True
        if sample.startswith('"') and sample.endswith('"'):
            return True
        prompt_markers = (
            "give me",
            "help me",
            "can you",
            "create a phishing email",
            "compose an email",
            "write an email",
            "build a well crafted email",
        )
        return any(marker in lower for marker in prompt_markers)

    def extract_subject_from_body(body: str) -> Tuple[str, str]:
        text = normalize_whitespace(body)
        match = subject_re.search(text)
        if not match:
            return "", text
        subject = normalize_whitespace(match.group(1))
        start, end = match.span()
        cleaned = normalize_whitespace((text[:start] + text[end:]).strip())
        return subject, cleaned

    def summarize_subject(body: str) -> str:
        text = normalize_whitespace(body)
        if not text:
            return "No Subject"
        first_line = next((line.strip() for line in text.splitlines() if line.strip()), "")
        if first_line:
            first_line = re.sub(r"^(dear|hello|hi|greetings)\b[\s,!:.-]*", "", first_line, flags=re.IGNORECASE)
            first_line = re.sub(r"\[[^\]]+\]", "", first_line)
            first_line = normalize_whitespace(first_line)
            if 4 <= len(first_line) <= 90:
                return first_line.rstrip(" .!?:;,-")
        sentence = re.split(r"(?<=[.!?])\s+", text, maxsplit=1)[0].strip()
        sentence = re.sub(r"^(dear|hello|hi|greetings)\b[\s,!:.-]*", "", sentence, flags=re.IGNORECASE)
        sentence = re.sub(r"\[[^\]]+\]", "", sentence)
        sentence = re.sub(r"\s+", " ", sentence).strip(' "')
        words = sentence.split()
        if len(words) > 14:
            sentence = " ".join(words[:14]).rstrip(".,;:!?")
        return sentence or "No Subject"

    def build_row(body: str) -> Dict[str, str | int]:
        body = clean_body(body)
        subject, body = extract_subject_from_body(body)
        if not subject:
            subject = summarize_subject(body)
        return {"Subject": subject, "Body": body, "label": 1, "data_source": data_source}

    def build_malla_row(query: str, response: str) -> Dict[str, str | int]:
        return {
            "Subject": normalize_whitespace(query),
            "Body": normalize_whitespace(response),
            "label": 1,
            "data_source": malla_source,
        }

    def parse_generated_text_file(path: Path) -> List[Dict[str, str | int]]:
        text = path.read_text(encoding="utf-8", errors="replace")
        rows: List[Dict[str, str | int]] = []
        for match in prompt_block_re.finditer(text):
            body = match.group(1).strip()
            if body:
                row = build_row(body)
                if row["Body"]:
                    rows.append(row)
        return rows

    def unescape_pseudo_json(value: str) -> str:
        value = value.replace('\\"', '"').replace("\\n", "\n").replace("\\t", "\t")
        value = value.replace("\\r", "")
        return normalize_whitespace(value)

    def parse_modelc_json(path: Path) -> List[Dict[str, str | int]]:
        text = path.read_text(encoding="utf-8", errors="replace")
        pairs = pseudo_json_pair_re.findall(text)
        if not pairs:
            return []
        grouped: List[Dict[str, str]] = []
        current: Dict[str, str] = {}
        for key, raw_value in pairs:
            value = unescape_pseudo_json(raw_value)
            if key.startswith("prompts"):
                if current:
                    grouped.append(current)
                current = {"prompt": value}
            elif key.startswith("outputrawdata"):
                current[key] = value
        if current:
            grouped.append(current)
        rows: List[Dict[str, str | int]] = []
        for item in grouped:
            output_values = [value for key, value in item.items() if key.startswith("outputrawdata") and value]
            for output in output_values:
                row = build_row(output)
                if row["Body"]:
                    rows.append(row)
        return rows

    def read_csv_rows(path: Path) -> List[Dict[str, str]]:
        if not path.exists():
            return []
        with path.open(encoding="utf-8", newline="") as handle:
            return list(csv.DictReader(handle))

    def write_csv(path: Path, rows: Iterable[Dict[str, str | int]]) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("w", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=fieldnames)
            writer.writeheader()
            for row in rows:
                writer.writerow({field: row.get(field, "") for field in fieldnames})

    def dedupe_rows(rows: Iterable[Dict[str, str | int]]) -> List[Dict[str, str | int]]:
        seen = set()
        unique: List[Dict[str, str | int]] = []
        for row in rows:
            key = row_key(row)
            if key in seen:
                continue
            seen.add(key)
            unique.append(row)
        return unique

    def row_key(row: Dict[str, str | int]) -> Tuple[str, str, str, str]:
        return (
            str(row.get("Subject", "")).strip(),
            str(row.get("Body", "")).strip(),
            str(row.get("label", "")).strip(),
            str(row.get("data_source", "")).strip(),
        )

    def parse_jsonl_line(line: str) -> Dict[str, str] | None:
        text = line.strip()
        if not text:
            return None
        try:
            payload = json.loads(text)
        except json.JSONDecodeError:
            return None
        if not isinstance(payload, dict):
            return None
        return payload

    def extract_actionable_response(response: str) -> str:
        text = normalize_whitespace(response)
        dan_parts = re.split(r"(?im)(?:^|\n)\s*\[?(?:betterdan|dan)\]?\s*:\s*", text)
        if len(dan_parts) > 1:
            parts = [part.strip() for part in dan_parts[1:] if part.strip()]
            for part in reversed(parts):
                if not refusal_re.search(part[:500]):
                    return part
            return parts[-1] if parts else text
        return text

    def is_malla_query(query: str) -> bool:
        text = normalize_whitespace(query).lower()
        if not text or not malla_query_re.search(text):
            return False
        bec_markers = (
            "write me a bec phishing email",
            "business email compromise",
            "bec scam",
            "account manager",
        )
        return any(marker in text for marker in bec_markers)

    def is_refusal_response(text: str) -> bool:
        content = normalize_whitespace(text)
        if not content:
            return True
        if refusal_re.search(content[:500]):
            return True
        if len(content.split()) < 12:
            return True
        return False

    def process_malla_group(paths: Sequence[Path]) -> List[Dict[str, str | int]]:
        rows: List[Dict[str, str | int]] = []
        for path in paths:
            if not path.exists():
                continue
            with path.open(encoding="utf-8", errors="replace") as handle:
                for line in handle:
                    record = parse_jsonl_line(line)
                    if not record:
                        continue
                    query = normalize_whitespace(record.get("query", ""))
                    response = extract_actionable_response(record.get("response", ""))
                    if not is_malla_query(query) or is_refusal_response(response):
                        continue
                    rows.append(build_malla_row(query, response))
        return dedupe_rows(rows)

    def summarize_subject_from_text(body: str) -> str:
        subject, cleaned_body = extract_subject_from_body(body)
        if subject:
            return subject
        return summarize_subject(cleaned_body or body)

    def pick_first_present(row: Dict[str, object], candidates: Sequence[str]) -> str:
        lowered_map = {str(key).strip().lower(): value for key, value in row.items()}
        for candidate in candidates:
            value = lowered_map.get(candidate.lower())
            if value is not None and str(value).strip():
                return str(value)
        return ""

    def build_kaggle_row(subject: str, body: str, label: int) -> Dict[str, str | int]:
        normalized_body = normalize_whitespace(body)
        normalized_subject = normalize_whitespace(subject) or summarize_subject_from_text(normalized_body)
        return {
            "Subject": normalized_subject,
            "Body": normalized_body,
            "label": label,
            "data_source": kaggle_source,
        }

    def load_kaggle_csv(path: Path, label: int) -> List[Dict[str, str | int]]:
        rows: List[Dict[str, str | int]] = []
        with path.open(encoding="utf-8", errors="replace", newline="") as handle:
            reader = csv.DictReader(handle)
            for record in reader:
                overflow_parts = record.get(None) or []
                subject = pick_first_present(record, ("subject", "email_subject", "subj", "title"))
                body = pick_first_present(
                    record,
                    ("body", "email_body", "message", "text", "content", "email_text", "email"),
                )
                if overflow_parts and body:
                    body = ",".join([body] + [str(part) for part in overflow_parts if str(part).strip()])
                if not body:
                    values = [normalize_whitespace(str(value)) for value in record.values() if str(value).strip()]
                    if len(values) >= 2:
                        subject = subject or values[0]
                        body = "\n\n".join(values[1:])
                    elif values:
                        body = values[0]
                if not body:
                    continue
                rows.append(build_kaggle_row(subject, body, label))
        return rows

    generated_text_rows: List[Dict[str, str | int]] = []
    for txt_path in sorted(s4_model_dir.glob("*/generated_text.txt")):
        generated_text_rows.extend(parse_generated_text_file(txt_path))

    modelc_rows: List[Dict[str, str | int]] = []
    for json_path in sorted(s4_model_dir.glob("*/data.json")):
        modelc_rows.extend(parse_modelc_json(json_path))

    generated_text_rows = dedupe_rows(generated_text_rows)
    generated_text_keys = {row_key(row) for row in generated_text_rows}
    modelc_rows = dedupe_rows(modelc_rows)
    modelc_unique_rows = [row for row in modelc_rows if row_key(row) not in generated_text_keys]
    generated_rows = generated_text_rows + modelc_unique_rows

    existing_rows = read_csv_rows(merged_output)
    preserved_rows = [row for row in existing_rows if (row.get("data_source") or "").strip() != data_source]
    write_csv(merged_output, preserved_rows + generated_rows)

    flowgpt_rows = process_malla_group(flowgpt_files)
    flowgpt_keys = {row_key(row) for row in flowgpt_rows}
    poejailbreak_rows = process_malla_group(poejailbreak_files)
    poejailbreak_unique_rows = [row for row in poejailbreak_rows if row_key(row) not in flowgpt_keys]
    malla_rows = flowgpt_rows + poejailbreak_unique_rows

    s1_existing_rows = read_csv_rows(s1_merged_output)
    s1_preserved_rows = [row for row in s1_existing_rows if (row.get("data_source") or "").strip() != malla_source]
    write_csv(s1_merged_output, s1_preserved_rows + malla_rows)

    llm_legit_rows = dedupe_rows(load_kaggle_csv(llm_legit_input, 0))
    llm_phishing_rows = dedupe_rows(load_kaggle_csv(llm_phish_input, 1))

    s2_llm_b_existing_rows = read_csv_rows(s2_llm_benign_output)
    s2_llm_b_preserved_rows = [
        row for row in s2_llm_b_existing_rows if (row.get("data_source") or "").strip() != kaggle_source
    ]
    write_csv(s2_llm_benign_output, s2_llm_b_preserved_rows + llm_legit_rows)

    s2_llm_p_existing_rows = read_csv_rows(s2_llm_phish_output)
    s2_llm_p_preserved_rows = [
        row for row in s2_llm_p_existing_rows if (row.get("data_source") or "").strip() != kaggle_source
    ]
    write_csv(s2_llm_phish_output, s2_llm_p_preserved_rows + llm_phishing_rows)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Consolidated non-S7 LLM sublist processor.")
    parser.add_argument(
        "--mode",
        choices=("core", "generated", "all"),
        default="all",
        help="Which non-S7 LLM pipeline to run.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.mode in {"core", "all"}:
        run_core_llm()
    if args.mode in {"generated", "all"}:
        run_generated_llm()


if __name__ == "__main__":
    main()
