#!/usr/bin/env python3
"""Build S4 HW-B/HW-P CSVs from phishyai human-generated samples."""

from __future__ import annotations

import csv
import re
import sys
from pathlib import Path
from typing import Dict, Iterable, List, Sequence, Tuple

SCRIPT_DIR = Path(__file__).resolve().parent
ROOT_DIR = SCRIPT_DIR.parent.parent

PHISHYAI_DIR = ROOT_DIR / "phishyai" / "human-generated-samples"
S4_DIR = SCRIPT_DIR / "S4-Scenarios-driven Adaptation"

EASY_HAM_PATH = PHISHYAI_DIR / "phishing_&_ham_emails" / "data_extracted_easy_ham.csv"
HARD_HAM_PATH = PHISHYAI_DIR / "phishing_&_ham_emails" / "data_extracted_hard_ham.csv"
PHISHING_PATH = PHISHYAI_DIR / "phishing_&_ham_emails" / "phishing.csv"
GMAIL_PATH = PHISHYAI_DIR / "gmail-samples" / "emails.json"
OUTLOOK_PATH = PHISHYAI_DIR / "outlook-samples" / "outlook-samples.json"

HW_B_PATH = S4_DIR / "HW-B.csv"
HW_P_PATH = S4_DIR / "HW-P.csv"

FIELDNAMES: Sequence[str] = ("Subject", "Body", "label", "data_source")
DATA_SOURCE = "phishyai"

csv.field_size_limit(min(sys.maxsize, 2**31 - 1))


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
        "data_source": DATA_SOURCE,
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
        writer = csv.DictWriter(handle, fieldnames=FIELDNAMES)
        writer.writeheader()
        for row in rows:
            writer.writerow({field: row.get(field, "") for field in FIELDNAMES})


def main() -> None:
    benign_rows: List[Dict[str, str | int]] = []
    phishing_rows: List[Dict[str, str | int]] = []

    benign_rows.extend(read_standard_csv(EASY_HAM_PATH, label=0))
    benign_rows.extend(read_standard_csv(HARD_HAM_PATH, label=0))

    phishing_rows.extend(read_standard_csv(PHISHING_PATH, label=1))
    phishing_rows.extend(parse_gmail_samples(GMAIL_PATH))
    phishing_rows.extend(parse_outlook_samples(OUTLOOK_PATH))

    merged_benign = dedupe_rows([*read_existing_rows(HW_B_PATH), *benign_rows])
    merged_phishing = dedupe_rows([*read_existing_rows(HW_P_PATH), *phishing_rows])

    write_csv(HW_B_PATH, merged_benign)
    write_csv(HW_P_PATH, merged_phishing)

    print(f"HW-B rows written: {len(merged_benign)} -> {HW_B_PATH}")
    print(f"HW-P rows written: {len(merged_phishing)} -> {HW_P_PATH}")


if __name__ == "__main__":
    main()
