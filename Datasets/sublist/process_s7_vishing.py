#!/usr/bin/env python3
"""Consolidated S7 vishing processor.

Replaces:
- process_aifraud_s7_vishing.py
- process_aifraud_llm_s7.py
- process_aifraud_llm_multi_s7.py
"""

from __future__ import annotations

import argparse
import csv
import json
import re
import sys
from pathlib import Path
from typing import Dict, Iterable, List

import pandas as pd

SCRIPT_DIR = Path(__file__).resolve().parent
DATASETS_DIR = SCRIPT_DIR.parent
csv.field_size_limit(min(sys.maxsize, 2**31 - 1))


def run_hw_vishing_single() -> None:
    source_dir = DATASETS_DIR / "LLM-Phishing" / "AI-FraudCall-Detector" / "Dataset"
    output_dir = SCRIPT_DIR / "S7-Cross-channel Expansion"
    vishing_output_path = output_dir / "HW-Vishing-single.csv"
    script_output_path = output_dir / "HW-Script-single.csv"
    legacy_output_path = output_dir / "HW-Vishing_single.csv"

    fieldnames = ("Body", "label", "category", "data source")
    data_source = "AIFraud"
    included_files = (
        "data.csv",
        "data_call.csv",
        "merged_call_data.csv",
        "merged_call_text.csv",
        "merged_text_data.csv",
        "scam_call_dataset_v1.csv",
        "test.csv",
        "test_call_text_set.csv",
        "text_1.csv",
        "text_2.csv",
        "text_3.csv",
    )

    def normalize_text(value: str) -> str:
        return (
            (value or "")
            .replace("\ufeff", "")
            .replace("\r\n", "\n")
            .replace("\r", "\n")
            .strip()
        )

    def normalize_label(value: str) -> int | None:
        label = normalize_text(value).strip('"').lower()
        if label == "intermediate":
            return None
        if label == "fraud":
            return 1
        if label == "normal":
            return 0
        return None

    def pick_first(row: Dict[str, str], keys: Iterable[str]) -> str:
        for key in keys:
            if key in row and normalize_text(row[key]):
                return normalize_text(row[key])
        return ""

    def process_file(path: Path) -> List[Dict[str, str | int]]:
        rows: List[Dict[str, str | int]] = []
        with path.open("r", encoding="utf-8", errors="replace", newline="") as handle:
            reader = csv.DictReader(handle)
            for raw in reader:
                label_value = normalize_label(
                    pick_first(raw, ("label", "Label", "Type", '"Type"', '\ufeff"Type"'))
                )
                if label_value is None:
                    continue

                body = pick_first(
                    raw, ("Body", "Text", "text", "Call_Transcript", "Transcript", "Message")
                )
                category = pick_first(
                    raw, ("category", "Category", "Type", "scam_category", "Subcategory")
                )
                if not body:
                    continue

                rows.append(
                    {
                        "Body": body,
                        "label": label_value,
                        "category": category,
                        "data source": data_source,
                    }
                )
        return rows

    def dedupe_rows(rows: List[Dict[str, str | int]]) -> List[Dict[str, str | int]]:
        unique_rows: List[Dict[str, str | int]] = []
        seen_bodies = set()
        for row in rows:
            body_key = normalize_text(str(row["Body"])).casefold()
            if not body_key or body_key in seen_bodies:
                continue
            seen_bodies.add(body_key)
            unique_rows.append(row)
        return unique_rows

    def write_csv(path: Path, rows: List[Dict[str, str | int]]) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("w", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(rows)

    all_rows: List[Dict[str, str | int]] = []
    for file_name in included_files:
        path = source_dir / file_name
        if not path.exists():
            raise FileNotFoundError(f"Missing source file: {path}")
        file_rows = process_file(path)
        all_rows.extend(file_rows)
        print(f"{file_name}: {len(file_rows)} rows kept")

    deduped_rows = dedupe_rows(all_rows)
    script_rows = [row for row in deduped_rows if int(row["label"]) == 0]
    vishing_rows = [row for row in deduped_rows if int(row["label"]) != 0]

    write_csv(script_output_path, script_rows)
    write_csv(vishing_output_path, vishing_rows)
    if legacy_output_path.exists():
        legacy_output_path.unlink()

    print(f"Total rows before dedupe: {len(all_rows)}")
    print(f"HW-Script-single rows written: {len(script_rows)} -> {script_output_path}")
    print(f"HW-Vishing-single rows written: {len(vishing_rows)} -> {vishing_output_path}")


def run_hw_vishing_multi() -> None:
    source_dir = DATASETS_DIR / "LLM-Phishing" / "AI-FraudCall-Detector" / "HW-Data"
    output_dir = SCRIPT_DIR / "S7-Cross-channel Expansion"
    script_output_path = output_dir / "HW-Script-Multi.csv"
    vishing_output_path = output_dir / "HW-Vishing-Multi.csv"

    fieldnames = ("Body", "label", "category", "data source")
    data_source = "kaggle-composite"

    def normalize_text(value: str) -> str:
        text = (value or "").replace("\ufeff", "").replace("\r\n", "\n").replace("\r", "\n")
        text = re.sub(r"\s+", " ", text)
        return text.strip()

    def normalize_label(value: str) -> int | None:
        label = normalize_text(value).strip('"').lower()
        if label in {"1", "fraud"}:
            return 1
        if label in {"0", "normal"}:
            return 0
        return None

    def summarize_category(body: str) -> str:
        text = normalize_text(body)
        if not text:
            return "na"
        first_sentence = re.split(r"(?<=[.!?])\s+", text, maxsplit=1)[0].strip()
        first_sentence = re.sub(r"\s+", " ", first_sentence).strip(' "')
        words = first_sentence.split()
        if len(words) > 8:
            first_sentence = " ".join(words[:8])
        return first_sentence or "na"

    def build_row(body: str, label: int, category: str) -> Dict[str, str | int]:
        normalized_body = normalize_text(body)
        return {
            "Body": normalized_body,
            "label": label,
            "category": normalize_text(category) or summarize_category(normalized_body),
            "data source": data_source,
        }

    def load_existing_rows(path: Path) -> List[Dict[str, str]]:
        if not path.exists():
            return []
        with path.open("r", encoding="utf-8", newline="") as handle:
            return list(csv.DictReader(handle))

    def write_csv(path: Path, rows: List[Dict[str, str | int]]) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("w", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(rows)

    def dedupe_rows(rows: List[Dict[str, str | int]]) -> List[Dict[str, str | int]]:
        unique_rows: List[Dict[str, str | int]] = []
        seen_bodies = set()
        for row in rows:
            body_key = normalize_text(str(row.get("Body", ""))).casefold()
            if not body_key or body_key in seen_bodies:
                continue
            seen_bodies.add(body_key)
            unique_rows.append(
                {
                    "Body": normalize_text(str(row.get("Body", ""))),
                    "label": int(str(row.get("label", "")).strip() or 0),
                    "category": normalize_text(str(row.get("category", ""))) or summarize_category(str(row.get("Body", ""))),
                    "data source": normalize_text(str(row.get("data source", ""))) or data_source,
                }
            )
        return unique_rows

    def load_csv_rows(path: Path) -> List[Dict[str, str | int]]:
        rows: List[Dict[str, str | int]] = []
        with path.open("r", encoding="utf-8", errors="replace", newline="") as handle:
            sample = handle.read(2048)
            handle.seek(0)
            has_header = csv.Sniffer().has_header(sample) if sample.strip() else False

            if has_header:
                reader = csv.DictReader(handle)
                for record in reader:
                    body = ""
                    category = ""
                    label_value = None
                    lowered = {str(k).strip().lower(): v for k, v in record.items() if k is not None}
                    for key in ("body", "text", "message", "content", "transcript"):
                        if lowered.get(key):
                            body = str(lowered[key])
                            break
                    for key in ("label", "type", "class"):
                        if lowered.get(key) is not None:
                            label_value = normalize_label(str(lowered[key]))
                            if label_value is not None:
                                break
                    for key in ("category", "type_name", "class_name"):
                        if lowered.get(key):
                            category = str(lowered[key])
                            break
                    if body and label_value is not None:
                        rows.append(build_row(body, label_value, category))
            else:
                reader = csv.reader(handle)
                for record in reader:
                    if len(record) < 2:
                        continue
                    body = record[0]
                    label_value = normalize_label(record[1])
                    category = record[2] if len(record) > 2 else ""
                    if body and label_value is not None:
                        rows.append(build_row(body, label_value, category))
        return rows

    csv_files = sorted(source_dir.glob("*.csv"))
    if not csv_files:
        raise FileNotFoundError(f"No CSV files found under {source_dir}")

    new_rows: List[Dict[str, str | int]] = []
    for path in csv_files:
        file_rows = load_csv_rows(path)
        new_rows.extend(file_rows)
        print(f"{path.name}: {len(file_rows)} rows kept")

    script_existing_rows = [
        row
        for row in load_existing_rows(script_output_path)
        if normalize_text(str(row.get("data source", ""))) != data_source
    ]
    vishing_existing_rows = [
        row
        for row in load_existing_rows(vishing_output_path)
        if normalize_text(str(row.get("data source", ""))) != data_source
    ]

    script_rows = [row for row in new_rows if int(row["label"]) == 0]
    vishing_rows = [row for row in new_rows if int(row["label"]) == 1]

    merged_script_rows = dedupe_rows(script_existing_rows + script_rows)
    merged_vishing_rows = dedupe_rows(vishing_existing_rows + vishing_rows)

    write_csv(script_output_path, merged_script_rows)
    write_csv(vishing_output_path, merged_vishing_rows)

    print(f"HW-Script-Multi rows written: {len(merged_script_rows)} -> {script_output_path}")
    print(f"HW-Vishing-Multi rows written: {len(merged_vishing_rows)} -> {vishing_output_path}")


def run_llm_vishing_single() -> None:
    source_dir = DATASETS_DIR / "LLM-Phishing" / "AI-FraudCall-Detector" / "LLM-dataset"
    script_output_path = SCRIPT_DIR / "S7-Cross-channel Expansion" / "LLM-Script-single.csv"
    vishing_output_path = SCRIPT_DIR / "S7-Cross-channel Expansion" / "LLM-Vishing-Single.csv"

    fieldnames = ("Body", "label", "category", "data source")
    data_source = "AIFraud"
    json_files = (
        "generated_sms_dataset_20250621_233325.json",
        "generated_sms_dataset_20250621_233427.json",
        "generated_sms_dataset_20250621_233528.json",
        "generated_sms_dataset_20250621_233636.json",
        "generated_sms_dataset_20250621_234457.json",
        "generated_sms_dataset_20250624_121231.json",
    )
    extended_dataset = "Extended Dataset.txt"
    fraud_db = "Fraud_DB.xlsx"
    normal_db = "Normal_DB.xlsx"

    def normalize_text(value: str) -> str:
        return (
            (value or "")
            .replace("\ufeff", "")
            .replace("\r\n", "\n")
            .replace("\r", "\n")
            .strip()
        )

    def normalize_label(value: str) -> int | None:
        label = normalize_text(value).strip('"').lower()
        if label == "fraud":
            return 1
        if label == "normal":
            return 0
        return None

    def build_row(body: str, label: int, category: str) -> Dict[str, str | int]:
        return {
            "Body": normalize_text(body),
            "label": label,
            "category": normalize_text(category) or "na",
            "data source": data_source,
        }

    def load_json_rows(path: Path) -> List[Dict[str, str | int]]:
        data = json.loads(path.read_text(encoding="utf-8", errors="replace"))
        rows: List[Dict[str, str | int]] = []
        if not isinstance(data, list):
            return rows
        for item in data:
            if not isinstance(item, list) or len(item) < 3:
                continue
            label_value = normalize_label(item[0])
            if label_value is None:
                continue
            category = str(item[1])
            body = str(item[2])
            if not normalize_text(body):
                continue
            rows.append(build_row(body, label_value, category))
        return rows

    def load_extended_rows(path: Path) -> List[Dict[str, str | int]]:
        rows: List[Dict[str, str | int]] = []
        with path.open("r", encoding="utf-8", errors="replace") as handle:
            for line in handle:
                parts = [normalize_text(part) for part in line.rstrip("\n").split("\t")]
                if len(parts) < 3:
                    continue
                label_value = normalize_label(parts[0])
                if label_value is None:
                    continue
                category = parts[1]
                body = parts[2]
                if not body:
                    continue
                rows.append(build_row(body, label_value, category))
        return rows

    def load_xlsx_rows(path: Path, label: int) -> List[Dict[str, str | int]]:
        df = pd.read_excel(path, header=None)
        rows: List[Dict[str, str | int]] = []
        for value in df.iloc[:, 0].tolist():
            body = normalize_text("" if pd.isna(value) else str(value))
            if not body:
                continue
            rows.append(build_row(body, label, "na"))
        return rows

    def dedupe_rows(rows: List[Dict[str, str | int]]) -> List[Dict[str, str | int]]:
        unique_rows: List[Dict[str, str | int]] = []
        seen = set()
        for row in rows:
            body_key = normalize_text(str(row["Body"])).casefold()
            if not body_key or body_key in seen:
                continue
            seen.add(body_key)
            unique_rows.append(row)
        return unique_rows

    def write_csv(path: Path, rows: List[Dict[str, str | int]]) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("w", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(rows)

    llm_rows: List[Dict[str, str | int]] = []
    for file_name in json_files:
        path = source_dir / file_name
        if not path.exists():
            raise FileNotFoundError(f"Missing source file: {path}")
        rows = load_json_rows(path)
        llm_rows.extend(rows)
        print(f"{file_name}: {len(rows)} rows kept")

    extended_rows = load_extended_rows(source_dir / extended_dataset)
    print(f"{extended_dataset}: {len(extended_rows)} rows kept")

    fraud_db_rows = load_xlsx_rows(source_dir / fraud_db, label=1)
    normal_db_rows = load_xlsx_rows(source_dir / normal_db, label=0)
    print(f"{fraud_db}: {len(fraud_db_rows)} rows kept")
    print(f"{normal_db}: {len(normal_db_rows)} rows kept")

    llm_script_rows = [row for row in llm_rows + extended_rows if int(row["label"]) == 0]
    llm_vishing_rows = [row for row in llm_rows + extended_rows if int(row["label"]) == 1]

    final_script_rows = dedupe_rows(llm_script_rows + normal_db_rows)
    final_vishing_rows = dedupe_rows(llm_vishing_rows + fraud_db_rows)

    write_csv(script_output_path, final_script_rows)
    write_csv(vishing_output_path, final_vishing_rows)

    print(f"LLM-Script-single rows written: {len(final_script_rows)} -> {script_output_path}")
    print(f"LLM-Vishing-Single rows written: {len(final_vishing_rows)} -> {vishing_output_path}")


def run_llm_vishing_multi() -> None:
    source_dir = DATASETS_DIR / "LLM-Phishing" / "AI-FraudCall-Detector" / "LLM-Multi"
    script_output_path = SCRIPT_DIR / "S7-Cross-channel Expansion" / "LLM-Script-Multi.csv"
    vishing_output_path = SCRIPT_DIR / "S7-Cross-channel Expansion" / "LLM-Vishing-Multi.csv"
    fieldnames = ("Body", "label", "category", "data source")
    data_source = "AIFraud"

    def normalize_text(value: str) -> str:
        text = (value or "").replace("\ufeff", "").replace("\r\n", "\n").replace("\r", "\n")
        text = re.sub(r"\s+", " ", text)
        return text.strip()

    def normalize_label(value: str) -> int | None:
        label = normalize_text(value).lower()
        if label == "fraud":
            return 1
        if label == "normal":
            return 0
        return None

    def speaker_tag(raw_speaker: str) -> str:
        speaker = normalize_text(raw_speaker).lower()
        if speaker == "scammer":
            return "S"
        match = re.search(r"person\s*(\d+)", speaker)
        if match:
            return f"P{match.group(1)}"
        if speaker.startswith("person"):
            return "P"
        return speaker.upper() or "UNK"

    def format_transcript(turns: List[dict]) -> str:
        lines: List[str] = []
        for turn in turns:
            if not isinstance(turn, dict):
                continue
            message = normalize_text(str(turn.get("message", "")))
            if not message:
                continue
            tag = speaker_tag(str(turn.get("speaker", "")))
            lines.append(f"{tag}: {message}")
        return "\\n".join(lines)

    def build_row(item: dict) -> Dict[str, str | int] | None:
        label_value = normalize_label(str(item.get("type", "")))
        if label_value is None:
            return None
        category = normalize_text(str(item.get("category", ""))) or "na"
        body = format_transcript(item.get("transcript", []))
        if not body:
            return None
        return {
            "Body": body,
            "label": label_value,
            "category": category,
            "data source": data_source,
        }

    def dedupe_rows(rows: List[Dict[str, str | int]]) -> List[Dict[str, str | int]]:
        unique_rows: List[Dict[str, str | int]] = []
        seen = set()
        for row in rows:
            body_key = str(row["Body"]).casefold().strip()
            if not body_key or body_key in seen:
                continue
            seen.add(body_key)
            unique_rows.append(row)
        return unique_rows

    def write_csv(path: Path, rows: List[Dict[str, str | int]]) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("w", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(rows)

    all_rows: List[Dict[str, str | int]] = []
    json_files = sorted(source_dir.glob("*.json"))
    if not json_files:
        raise FileNotFoundError(f"No JSON files found under {source_dir}")

    for path in json_files:
        data = json.loads(path.read_text(encoding="utf-8", errors="replace"))
        if not isinstance(data, list):
            continue
        file_rows = []
        for item in data:
            if not isinstance(item, dict):
                continue
            row = build_row(item)
            if row is not None:
                file_rows.append(row)
        all_rows.extend(file_rows)
        print(f"{path.name}: {len(file_rows)} rows kept")

    deduped_rows = dedupe_rows(all_rows)
    script_rows = [row for row in deduped_rows if int(row["label"]) == 0]
    vishing_rows = [row for row in deduped_rows if int(row["label"]) == 1]

    write_csv(script_output_path, script_rows)
    write_csv(vishing_output_path, vishing_rows)

    print(f"Total rows before dedupe: {len(all_rows)}")
    print(f"LLM-Script-Multi rows written: {len(script_rows)} -> {script_output_path}")
    print(f"LLM-Vishing-Multi rows written: {len(vishing_rows)} -> {vishing_output_path}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Consolidated S7 vishing processor.")
    parser.add_argument(
        "--mode",
        choices=("hw-single", "hw-multi", "llm-single", "llm-multi", "all"),
        default="all",
        help="Which S7 vishing pipeline to run.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.mode in {"hw-single", "all"}:
        run_hw_vishing_single()
    if args.mode in {"hw-multi", "all"}:
        run_hw_vishing_multi()
    if args.mode in {"llm-single", "all"}:
        run_llm_vishing_single()
    if args.mode in {"llm-multi", "all"}:
        run_llm_vishing_multi()


if __name__ == "__main__":
    main()
