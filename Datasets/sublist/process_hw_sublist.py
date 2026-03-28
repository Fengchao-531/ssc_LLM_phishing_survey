#!/usr/bin/env python3
"""Consolidated non-S7 HW sublist processor.

Replaces the HW-related parts of:
- process_phishyai_s4_hw.py
- data process1.py
"""

from __future__ import annotations

import argparse
import csv
import re
import sys
from pathlib import Path
from typing import Dict, Iterable, List, Sequence, Tuple

SCRIPT_DIR = Path(__file__).resolve().parent
DATASETS_DIR = SCRIPT_DIR.parent
ROOT_DIR = SCRIPT_DIR.parent.parent
csv.field_size_limit(min(sys.maxsize, 2**31 - 1))


def run_s4_hw() -> None:
    phishyai_dir = ROOT_DIR / "phishyai" / "human-generated-samples"
    s4_dir = SCRIPT_DIR / "S4-Scenarios-driven Adaptation"

    easy_ham_path = phishyai_dir / "phishing_&_ham_emails" / "data_extracted_easy_ham.csv"
    hard_ham_path = phishyai_dir / "phishing_&_ham_emails" / "data_extracted_hard_ham.csv"
    phishing_path = phishyai_dir / "phishing_&_ham_emails" / "phishing.csv"
    gmail_path = phishyai_dir / "gmail-samples" / "emails.json"
    outlook_path = phishyai_dir / "outlook-samples" / "outlook-samples.json"

    hw_b_path = s4_dir / "HW-B.csv"
    hw_p_path = s4_dir / "HW-P.csv"
    fieldnames: Sequence[str] = ("Subject", "Body", "label", "data_source")
    data_source = "phishyai"

    def normalize_text(value: str) -> str:
        text = (value or "").replace("\r\n", "\n").replace("\r", "\n")
        text = text.replace("\ufeff", "").replace("\u200b", "").replace("\xa0", " ")
        text = re.sub(r"[ \t]+\n", "\n", text)
        text = re.sub(r"\n{3,}", "\n\n", text)
        return text.strip()

    def normalize_subject(value: str) -> str:
        subject = normalize_text(value)
        return re.sub(r"\s+", " ", subject).strip()

    def summarize_subject_from_body(body: str) -> str:
        body = normalize_text(body)
        if not body:
            return "No Subject"
        for line in body.splitlines():
            line = line.strip()
            if line:
                line = re.sub(r"\s+", " ", line)
                return line[:160]
        return "No Subject"

    def split_subject_body_from_text(text: str) -> Tuple[str, str]:
        body = normalize_text(text)
        if not body:
            return "No Subject", ""
        lines = [line.strip() for line in body.splitlines() if line.strip()]
        if not lines:
            return "No Subject", body
        subject = re.sub(r"\s+", " ", lines[0])[:160]
        return subject or "No Subject", body

    def build_row(subject: str, body: str, label: int) -> Dict[str, str | int]:
        body_clean = normalize_text(body)
        subject_clean = normalize_subject(subject) or summarize_subject_from_body(body_clean)
        return {
            "Subject": subject_clean,
            "Body": body_clean,
            "label": label,
            "data_source": data_source,
        }

    def read_existing_rows(path: Path) -> List[Dict[str, str]]:
        if not path.exists():
            return []
        with path.open("r", encoding="utf-8", newline="") as handle:
            return list(csv.DictReader(handle))

    def is_invalid_row(row: Dict[str, str | int]) -> bool:
        subject = normalize_subject(str(row.get("Subject", "")))
        body = normalize_text(str(row.get("Body", "")))
        invalid_tokens = {"[]", "[ ]"}
        if subject in invalid_tokens or body in invalid_tokens:
            return True
        if not subject and not body:
            return True
        if not body:
            return True
        return False

    def read_standard_csv(path: Path, label: int) -> List[Dict[str, str | int]]:
        rows: List[Dict[str, str | int]] = []
        with path.open("r", encoding="utf-8", errors="replace", newline="") as handle:
            reader = csv.DictReader(handle)
            for item in reader:
                rows.append(build_row(item.get("Subject", ""), item.get("Body", ""), label))
        return rows

    def parse_js_like_strings(text: str) -> List[Tuple[str | None, str]]:
        pairs: List[Tuple[str | None, str]] = []
        i = 0
        n = len(text)
        while i < n:
            ch = text[i]
            if ch != '"':
                i += 1
                continue
            start = i + 1
            i += 1
            while i < n:
                if text[i] == '"' and text[i - 1] != "\\":
                    break
                i += 1
            if i >= n:
                break
            value = text[start:i]
            i += 1
            j = start - 2
            while j >= 0 and text[j].isspace():
                j -= 1
            key = None
            if j >= 0 and text[j] == ":":
                k = j - 1
                while k >= 0 and (text[k].isalnum() or text[k] in {"_", "-"}):
                    k -= 1
                candidate = text[k + 1 : j].strip()
                key = candidate or None
            pairs.append((key, value))
        return pairs

    def parse_gmail_samples(path: Path) -> List[Dict[str, str | int]]:
        text = path.read_text(encoding="utf-8", errors="replace")
        rows: List[Dict[str, str | int]] = []
        for key, value in parse_js_like_strings(text):
            if key is not None and not key.startswith("sample"):
                continue
            subject, body = split_subject_body_from_text(value)
            rows.append(build_row(subject, body, 1))
        return rows

    def parse_outlook_samples(path: Path) -> List[Dict[str, str | int]]:
        text = path.read_text(encoding="utf-8", errors="replace")
        pairs = parse_js_like_strings(text)
        rows: List[Dict[str, str | int]] = []
        current_subject = ""
        for key, value in pairs:
            if not key:
                continue
            if key.startswith("header"):
                current_subject = normalize_subject(value)
            elif key.startswith("rawdata"):
                subject = current_subject or summarize_subject_from_body(value)
                rows.append(build_row(subject, value, 1))
                current_subject = ""
        return rows

    def dedupe_rows(rows: Iterable[Dict[str, str | int]]) -> List[Dict[str, str | int]]:
        unique: List[Dict[str, str | int]] = []
        seen = set()
        for row in rows:
            if is_invalid_row(row):
                continue
            key = (
                str(row.get("Subject", "")).strip(),
                str(row.get("Body", "")).strip(),
                str(row.get("label", "")).strip(),
                str(row.get("data_source", "")).strip(),
            )
            if key in seen:
                continue
            seen.add(key)
            unique.append(
                {
                    "Subject": str(row.get("Subject", "")).strip(),
                    "Body": str(row.get("Body", "")).strip(),
                    "label": int(str(row.get("label", "")).strip() or 0),
                    "data_source": str(row.get("data_source", "")).strip(),
                }
            )
        return unique

    def write_csv(path: Path, rows: Iterable[Dict[str, str | int]]) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("w", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=fieldnames)
            writer.writeheader()
            for row in rows:
                writer.writerow({field: row.get(field, "") for field in fieldnames})

    benign_rows: List[Dict[str, str | int]] = []
    phishing_rows: List[Dict[str, str | int]] = []
    benign_rows.extend(read_standard_csv(easy_ham_path, label=0))
    benign_rows.extend(read_standard_csv(hard_ham_path, label=0))
    phishing_rows.extend(read_standard_csv(phishing_path, label=1))
    phishing_rows.extend(parse_gmail_samples(gmail_path))
    phishing_rows.extend(parse_outlook_samples(outlook_path))

    merged_benign = dedupe_rows([*read_existing_rows(hw_b_path), *benign_rows])
    merged_phishing = dedupe_rows([*read_existing_rows(hw_p_path), *phishing_rows])

    write_csv(hw_b_path, merged_benign)
    write_csv(hw_p_path, merged_phishing)


def run_s2_hw() -> None:
    s2_output_dir = SCRIPT_DIR / "S2-Role-Framed Prompting"
    human_legit_input = DATASETS_DIR / "LLM-Phishing" / "human-generated" / "legit.csv"
    human_phish_input = DATASETS_DIR / "LLM-Phishing" / "human-generated" / "phishing.csv"
    s2_hw_benign_output = s2_output_dir / "HW-B.csv"
    s2_hw_phish_output = s2_output_dir / "HW-P.csv"
    fieldnames: Sequence[str] = ("Subject", "Body", "label", "data_source")
    kaggle_source = "kaggle"

    def normalize_whitespace(text: str) -> str:
        text = (text or "").replace("\r\n", "\n").replace("\r", "\n")
        text = text.replace("\ufeff", "").replace("\u200b", "")
        text = re.sub(r"[ \t]+\n", "\n", text)
        text = re.sub(r"\n{3,}", "\n\n", text)
        return text.strip()

    def extract_subject_from_body(body: str) -> Tuple[str, str]:
        subject_re = re.compile(r"(?im)^\s*subject(?:\s*line)?\s*:\s*(.+)$")
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
            key = (
                str(row.get("Subject", "")).strip(),
                str(row.get("Body", "")).strip(),
                str(row.get("label", "")).strip(),
                str(row.get("data_source", "")).strip(),
            )
            if key in seen:
                continue
            seen.add(key)
            unique.append(row)
        return unique

    human_legit_rows = dedupe_rows(load_kaggle_csv(human_legit_input, 0))
    human_phishing_rows = dedupe_rows(load_kaggle_csv(human_phish_input, 1))

    s2_hw_b_existing_rows = read_csv_rows(s2_hw_benign_output)
    s2_hw_b_preserved_rows = [
        row for row in s2_hw_b_existing_rows if (row.get("data_source") or "").strip() != kaggle_source
    ]
    write_csv(s2_hw_benign_output, s2_hw_b_preserved_rows + human_legit_rows)

    s2_hw_p_existing_rows = read_csv_rows(s2_hw_phish_output)
    s2_hw_p_preserved_rows = [
        row for row in s2_hw_p_existing_rows if (row.get("data_source") or "").strip() != kaggle_source
    ]
    write_csv(s2_hw_phish_output, s2_hw_p_preserved_rows + human_phishing_rows)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Consolidated non-S7 HW sublist processor.")
    parser.add_argument(
        "--mode",
        choices=("s4", "s2", "all"),
        default="all",
        help="Which non-S7 HW pipeline to run.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.mode in {"s4", "all"}:
        run_s4_hw()
    if args.mode in {"s2", "all"}:
        run_s2_hw()


if __name__ == "__main__":
    main()
