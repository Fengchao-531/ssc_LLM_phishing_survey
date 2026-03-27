#!/usr/bin/env python3
"""Dataset splitter for sublist buckets.

Features:
1. Normalize `Datasets/LLM-Benign/S4-ephishLLM.json` (language-filtered) into
   LLM-B / LLM-P CSVs stored under S4-Scenarios-driven Adaptation.
2. Parse Paladin's `dpo_dataset.json`, treating the human prompt as Subject and
   the GPT completion as Body. All Paladin rows carry label=1.
3. Route Paladin rows whose prompt matches "Write me a BEC phishing email?"
   into `S1-Basic Instruction/LLM-P.csv`; everything else is appended to
   `S4-Scenarios-driven Adaptation/LLM-P.csv`.
4. Append a `data_source` column so we can trace provenance
   (`ephishLLM` vs `paladin`).
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

EPHISH_PATH = DATASETS_DIR / "LLM-Benign" / "S4-ephishLLM.json"
PALADIN_PATH = (
    DATASETS_DIR
    / "LLM-Benign"
    / "Paladin-main"
    / "dataset"
    / "core_dataset"
    / "dpo_dataset.json"
)

PALADIN_BASE_DIR = (
    DATASETS_DIR
    / "LLM-Benign"
    / "Paladin-main"
    / "dataset"
    / "base_dataset"
)
SAFE_EMAILS_PATH = PALADIN_BASE_DIR / "safe_emails_1000.json"
SET1_NONPHISH_PATH = PALADIN_BASE_DIR / "set1_dataset_nophishing.json"
BASE_PHISH_FILES = [
    PALADIN_BASE_DIR / "set1_dataset.json",
    PALADIN_BASE_DIR / "set2_dataset.json",
    PALADIN_BASE_DIR / "set3_dataset.json",
    PALADIN_BASE_DIR / "set4_dataset.json",
]

S4_DIR = SCRIPT_DIR / "S4-Scenarios-driven Adaptation"
S1_DIR = SCRIPT_DIR / "S1-Basic Instruction"
S5_DIR = SCRIPT_DIR / "S5-Personalization for Credibility"
S4_BENIGN_NAME = "LLM-B.csv"
S4_PHISH_NAME = "LLM-P.csv"
S1_PHISH_NAME = "LLM-P.csv"
S1_BENIGN_NAME = "LLM-B.csv"
S5_PHISH_NAME = "LLM-P.csv"

DATA_SOURCE_EPHISH = "ephishLLM"
DATA_SOURCE_PALADIN = "paladin"
BEC_TRIGGER = "write me a bec phishing email?"
BEC_KEYS = {
    "write me a bec email?",
    "write me a bec phishing email?",
}

FIELDNAMES: Sequence[str] = ("Subject", "Body", "label", "data_source")
SUBJECT_REGEX = re.compile(r"(?im)^\s*subject(?:\s*line)?\s*[:：]\s*(.+)$")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Split LLM-Benign datasets into Sublist-ready CSV files."
    )
    parser.add_argument(
        "--ephish-path",
        type=Path,
        default=EPHISH_PATH,
        help=f"Location of S4-ephishLLM.json (default: {EPHISH_PATH})",
    )
    parser.add_argument(
        "--paladin-path",
        type=Path,
        default=PALADIN_PATH,
        help=f"Location of Paladin dpo_dataset.json (default: {PALADIN_PATH})",
    )
    parser.add_argument(
        "--language",
        default="en",
        help="Language gate for S4-ephishLLM rows (default: en).",
    )
    parser.add_argument(
        "--s4-dir",
        type=Path,
        default=S4_DIR,
        help=f"S4 output directory (default: {S4_DIR})",
    )
    parser.add_argument(
        "--s1-dir",
        type=Path,
        default=S1_DIR,
        help=f"S1 output directory (default: {S1_DIR})",
    )
    parser.add_argument(
        "--s5-dir",
        type=Path,
        default=S5_DIR,
        help=f"S5 output directory (default: {S5_DIR})",
    )
    parser.add_argument(
        "--s4-benign-name",
        default=S4_BENIGN_NAME,
        help=f"Filename for S4 benign CSV (default: {S4_BENIGN_NAME})",
    )
    parser.add_argument(
        "--s4-phish-name",
        default=S4_PHISH_NAME,
        help=f"Filename for S4 phishing CSV (default: {S4_PHISH_NAME})",
    )
    parser.add_argument(
        "--s1-phish-name",
        default=S1_PHISH_NAME,
        help=f"Filename for S1 phishing CSV (default: {S1_PHISH_NAME})",
    )
    parser.add_argument(
        "--s1-benign-name",
        default=S1_BENIGN_NAME,
        help=f"Filename for S1 benign CSV (default: {S1_BENIGN_NAME})",
    )
    parser.add_argument(
        "--s5-phish-name",
        default=S5_PHISH_NAME,
        help=f"Filename for S5 phishing CSV (default: {S5_PHISH_NAME})",
    )
    return parser.parse_args()


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
    cleaned = (value or "").replace("â€‹", "")
    return cleaned.translate({ord("\u200b"): None, ord("\ufeff"): None})


def extract_subject_and_body(output_text: str, fallback_subject: str) -> Tuple[str, str]:
    text = clean_text(output_text or "")
    match = SUBJECT_REGEX.search(text)
    if match:
        subject = match.group(1).strip()
        start, end = match.span()
        body = (text[:start] + text[end:]).strip()
    else:
        subject = ""
        body = text.strip()
    subject = subject or (fallback_subject or "").strip()
    return subject, body


def process_ephish(path: Path, language: str) -> Tuple[List[Dict], List[Dict]]:
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
        try:
            label = int(raw_label)
        except (TypeError, ValueError):
            raise ValueError(f"Unexpected type value: {raw_label!r}")
        row = {
            "Subject": subject,
            "Body": body,
            "label": label,
            "data_source": DATA_SOURCE_EPHISH,
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
            "data_source": DATA_SOURCE_PALADIN,
        }
        if subject.lower() == BEC_TRIGGER:
            s1_rows.append(row)
        else:
            s4_rows.append(row)
    return s1_rows, s4_rows


def write_csv(path: Path, rows: Iterable[Dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=FIELDNAMES)
        writer.writeheader()
        for row in rows:
            writer.writerow({field: row.get(field, "") for field in FIELDNAMES})


def build_rows_from_entries(
    entries: List[Dict], label: int, data_source: str
) -> List[Dict]:
    rows: List[Dict] = []
    for entry in entries:
        rows.append(row_from_entry(entry, label, data_source))
    return rows


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
    for path in (SAFE_EMAILS_PATH, SET1_NONPHISH_PATH):
        if path.exists():
            entries = load_json_array(path)
            rows.extend(build_rows_from_entries(entries, 0, DATA_SOURCE_PALADIN))
    return rows


def process_base_phish(files: Sequence[Path]) -> Tuple[List[Dict], List[Dict]]:
    s5_rows: List[Dict] = []
    s4_rows: List[Dict] = []
    for path in files:
        if not path.exists():
            continue
        entries = load_json_array(path)
        for entry in entries:
            row = row_from_entry(entry, 1, DATA_SOURCE_PALADIN)
            inst_key = (entry.get("instruction") or "").strip().lower()
            if inst_key in BEC_KEYS:
                s5_rows.append(row)
            else:
                s4_rows.append(row)
    return s5_rows, s4_rows


def main() -> None:
    args = parse_args()

    s4_benign_rows, s4_phish_rows = process_ephish(args.ephish_path, args.language)
    s1_paladin_rows, s4_paladin_rows = process_paladin(args.paladin_path)
    s1_benign_rows = process_safe_sets()
    s5_bec_rows, s4_base_rows = process_base_phish(BASE_PHISH_FILES)

    s4_phish_combined = s4_phish_rows + s4_paladin_rows + s4_base_rows
    s1_phish_rows = s1_paladin_rows

    write_csv(args.s4_dir / args.s4_benign_name, s4_benign_rows)
    write_csv(args.s4_dir / args.s4_phish_name, s4_phish_combined)
    write_csv(args.s1_dir / args.s1_phish_name, s1_phish_rows)
    write_csv(args.s1_dir / args.s1_benign_name, s1_benign_rows)
    write_csv(args.s5_dir / args.s5_phish_name, s5_bec_rows)

    print(
        "Done:",
        f"S4 benign={len(s4_benign_rows)} (source {DATA_SOURCE_EPHISH}),",
        (
            "S4 phishing="
            f"{len(s4_phish_combined)} (ephish={len(s4_phish_rows)}, "
            f"paladin-core={len(s4_paladin_rows)}, paladin-base={len(s4_base_rows)})"
        ),
        f"S1 phishing={len(s1_phish_rows)} (paladin core BEC prompts),",
        f"S1 benign={len(s1_benign_rows)} (safe emails + set1 nonphishing),",
        f"S5 phishing={len(s5_bec_rows)} (paladin base BEC prompts).",
    )


if __name__ == "__main__":
    main()
