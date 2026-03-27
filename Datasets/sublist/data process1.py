#!/usr/bin/env python3
"""Process S4 model-generated phishing samples into Sublist CSVs.

Outputs:
1. `S4-Scenarios-driven Adaptation/LLM-P-model-generated.csv`
2. Updated `S4-Scenarios-driven Adaptation/LLM-P.csv` with model-generated
   rows appended after existing non-model-generated rows.
"""

from __future__ import annotations

import csv
import json
import re
from pathlib import Path
from typing import Dict, Iterable, List, Sequence, Tuple

SCRIPT_DIR = Path(__file__).resolve().parent
DATASETS_DIR = SCRIPT_DIR.parent
S4_MODEL_DIR = DATASETS_DIR / "LLM-Phishing" / "S4-model-generated"
S4_OUTPUT_DIR = SCRIPT_DIR / "S4-Scenarios-driven Adaptation"
S1_OUTPUT_DIR = SCRIPT_DIR / "S1-Basic Instruction"
S2_OUTPUT_DIR = SCRIPT_DIR / "S2-Role-Framed Prompting"
MERGED_OUTPUT = S4_OUTPUT_DIR / "LLM-P.csv"
# MODEL_ONLY_OUTPUT = S4_OUTPUT_DIR / "LLM-P-model-generated.csv"
S1_MERGED_OUTPUT = S1_OUTPUT_DIR / "LLM-P.csv"
S2_HW_BENIGN_OUTPUT = S2_OUTPUT_DIR / "HW-B.csv"
S2_HW_PHISH_OUTPUT = S2_OUTPUT_DIR / "HW-P.csv"
S2_LLM_BENIGN_OUTPUT = S2_OUTPUT_DIR / "LLM-B.csv"
S2_LLM_PHISH_OUTPUT = S2_OUTPUT_DIR / "LLM-P.csv"
FLOWGPT_FILES = (
    DATASETS_DIR / "LLM-Phishing" / "flowgpt-jailbrea.txt",
    DATASETS_DIR / "LLM-Phishing" / "flowgpt-jailbrea1.txt",
    DATASETS_DIR / "LLM-Phishing" / "flowgpt-jailbrea2.txt",
)
POEJAILBREAK_FILES = (
    DATASETS_DIR / "LLM-Phishing" / "poe-jailbrea.txt",
    DATASETS_DIR / "LLM-Phishing" / "poe1-jailbrea.txt",
    DATASETS_DIR / "LLM-Phishing" / "poe2-jailbrea.txt",
)
HUMAN_LEGIT_INPUT = DATASETS_DIR / "LLM-Phishing" / "human-generated" / "legit.csv"
HUMAN_PHISH_INPUT = DATASETS_DIR / "LLM-Phishing" / "human-generated" / "phishing.csv"
LLM_LEGIT_INPUT = DATASETS_DIR / "LLM-Phishing" / "llm-generated" / "legit.csv"
LLM_PHISH_INPUT = DATASETS_DIR / "LLM-Phishing" / "llm-generated" / "phishing.csv"

FIELDNAMES: Sequence[str] = ("Subject", "Body", "label", "data_source")
DATA_SOURCE = "model-generated"
MALLA_SOURCE = "malla"
KAGGLE_SOURCE = "kaggle"
KAGGLE_SOURCE = "kaggle"
PROMPT_BLOCK_RE = re.compile(r"(?ms)^Prompt\s+\d+\s*:\s*(.*?)(?=^Prompt\s+\d+\s*:|\Z)")
SUBJECT_RE = re.compile(r"(?im)^\s*subject(?:\s*line)?\s*:\s*(.+)$")
PSEUDO_JSON_PAIR_RE = re.compile(
    r'(?s)(prompts\d*|outputrawdata\d*[a-z]?)\s*:\s*"((?:[^"\\]|\\.|"(?=[^,\n\r}]))*)"'
)
MALLA_QUERY_RE = re.compile(
    r"(?i)\b("
    r"bec|business email compromise|bec scam|"
    r"write\s+(?:me\s+)?a\s+convincing\s+email|"
    r"business\s+email|account\s+manager|invoice|ceo"
    r")\b"
)
REFUSAL_RE = re.compile(
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
    match = SUBJECT_RE.search(text)
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
    return {
        "Subject": subject,
        "Body": body,
        "label": 1,
        "data_source": DATA_SOURCE,
    }


def build_malla_row(query: str, response: str) -> Dict[str, str | int]:
    return {
        "Subject": normalize_whitespace(query),
        "Body": normalize_whitespace(response),
        "label": 1,
        "data_source": MALLA_SOURCE,
    }


def parse_generated_text_file(path: Path) -> List[Dict[str, str | int]]:
    text = path.read_text(encoding="utf-8", errors="replace")
    rows: List[Dict[str, str | int]] = []
    for match in PROMPT_BLOCK_RE.finditer(text):
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
    pairs = PSEUDO_JSON_PAIR_RE.findall(text)
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
        output_values = [
            value for key, value in item.items() if key.startswith("outputrawdata") and value
        ]
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
        writer = csv.DictWriter(handle, fieldnames=FIELDNAMES)
        writer.writeheader()
        for row in rows:
            writer.writerow({field: row.get(field, "") for field in FIELDNAMES})


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
            if not REFUSAL_RE.search(part[:500]):
                return part
        return parts[-1] if parts else text
    return text


def is_malla_query(query: str) -> bool:
    text = normalize_whitespace(query).lower()
    if not text:
        return False
    if not MALLA_QUERY_RE.search(text):
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
    if REFUSAL_RE.search(content[:500]):
        return True
    if len(content.split()) < 12:
        return True
    return False


def process_malla_group(
    paths: Sequence[Path],
) -> List[Dict[str, str | int]]:
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
                if not is_malla_query(query):
                    continue
                if is_refusal_response(response):
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
        "data_source": KAGGLE_SOURCE,
    }


def load_kaggle_csv(path: Path, label: int) -> List[Dict[str, str | int]]:
    rows: List[Dict[str, str | int]] = []
    with path.open(encoding="utf-8", errors="replace", newline="") as handle:
        reader = csv.DictReader(handle)
        for record in reader:
            overflow_parts = record.get(None) or []
            subject = pick_first_present(
                record,
                ("subject", "email_subject", "subj", "title"),
            )
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


def process_kaggle_dataset() -> Tuple[
    List[Dict[str, str | int]],
    List[Dict[str, str | int]],
    List[Dict[str, str | int]],
    List[Dict[str, str | int]],
    Dict[str, int],
]:
    human_legit_rows = dedupe_rows(load_kaggle_csv(HUMAN_LEGIT_INPUT, 0))
    human_phishing_rows = dedupe_rows(load_kaggle_csv(HUMAN_PHISH_INPUT, 1))
    llm_legit_rows = dedupe_rows(load_kaggle_csv(LLM_LEGIT_INPUT, 0))
    llm_phishing_rows = dedupe_rows(load_kaggle_csv(LLM_PHISH_INPUT, 1))

    stats = {
        "human_legit": len(human_legit_rows),
        "human_phishing": len(human_phishing_rows),
        "llm_legit": len(llm_legit_rows),
        "llm_phishing": len(llm_phishing_rows),
    }
    return (
        human_legit_rows,
        human_phishing_rows,
        llm_legit_rows,
        llm_phishing_rows,
        stats,
    )


def main() -> None:
    generated_text_rows: List[Dict[str, str | int]] = []
    for txt_path in sorted(S4_MODEL_DIR.glob("*/generated_text.txt")):
        generated_text_rows.extend(parse_generated_text_file(txt_path))

    modelc_rows: List[Dict[str, str | int]] = []
    for json_path in sorted(S4_MODEL_DIR.glob("*/data.json")):
        modelc_rows.extend(parse_modelc_json(json_path))

    generated_text_rows = dedupe_rows(generated_text_rows)
    generated_text_keys = {row_key(row) for row in generated_text_rows}
    modelc_rows = dedupe_rows(modelc_rows)
    modelc_unique_rows = [
        row for row in modelc_rows if row_key(row) not in generated_text_keys
    ]

    generated_rows = generated_text_rows + modelc_unique_rows

    existing_rows = read_csv_rows(MERGED_OUTPUT)
    preserved_rows = [
        row for row in existing_rows if (row.get("data_source") or "").strip() != DATA_SOURCE
    ]
    merged_rows = preserved_rows + generated_rows

    # write_csv(MODEL_ONLY_OUTPUT, generated_rows)
    write_csv(MERGED_OUTPUT, merged_rows)

    flowgpt_rows = process_malla_group(FLOWGPT_FILES)
    flowgpt_keys = {row_key(row) for row in flowgpt_rows}
    poejailbreak_rows = process_malla_group(POEJAILBREAK_FILES)
    poejailbreak_unique_rows = [
        row for row in poejailbreak_rows if row_key(row) not in flowgpt_keys
    ]
    malla_rows = flowgpt_rows + poejailbreak_unique_rows

    s1_existing_rows = read_csv_rows(S1_MERGED_OUTPUT)
    s1_preserved_rows = [
        row for row in s1_existing_rows if (row.get("data_source") or "").strip() != MALLA_SOURCE
    ]
    s1_merged_rows = s1_preserved_rows + malla_rows
    write_csv(S1_MERGED_OUTPUT, s1_merged_rows)

    (
        human_legit_rows,
        human_phishing_rows,
        llm_legit_rows,
        llm_phishing_rows,
        kaggle_stats,
    ) = process_kaggle_dataset()

    s2_hw_b_existing_rows = read_csv_rows(S2_HW_BENIGN_OUTPUT)
    s2_hw_b_preserved_rows = [
        row for row in s2_hw_b_existing_rows if (row.get("data_source") or "").strip() != KAGGLE_SOURCE
    ]
    s2_hw_b_merged_rows = s2_hw_b_preserved_rows + human_legit_rows
    write_csv(S2_HW_BENIGN_OUTPUT, s2_hw_b_merged_rows)

    s2_hw_p_existing_rows = read_csv_rows(S2_HW_PHISH_OUTPUT)
    s2_hw_p_preserved_rows = [
        row for row in s2_hw_p_existing_rows if (row.get("data_source") or "").strip() != KAGGLE_SOURCE
    ]
    s2_hw_p_merged_rows = s2_hw_p_preserved_rows + human_phishing_rows
    write_csv(S2_HW_PHISH_OUTPUT, s2_hw_p_merged_rows)

    s2_llm_b_existing_rows = read_csv_rows(S2_LLM_BENIGN_OUTPUT)
    s2_llm_b_preserved_rows = [
        row for row in s2_llm_b_existing_rows if (row.get("data_source") or "").strip() != KAGGLE_SOURCE
    ]
    s2_llm_b_merged_rows = s2_llm_b_preserved_rows + llm_legit_rows
    write_csv(S2_LLM_BENIGN_OUTPUT, s2_llm_b_merged_rows)

    s2_llm_p_existing_rows = read_csv_rows(S2_LLM_PHISH_OUTPUT)
    s2_llm_p_preserved_rows = [
        row for row in s2_llm_p_existing_rows if (row.get("data_source") or "").strip() != KAGGLE_SOURCE
    ]
    s2_llm_p_merged_rows = s2_llm_p_preserved_rows + llm_phishing_rows
    write_csv(S2_LLM_PHISH_OUTPUT, s2_llm_p_merged_rows)

    print(
        "Done:",
        (
            "S4 phishing="
            f"{len(merged_rows)} (existing={len(preserved_rows)}, "
            f"generated_text={len(generated_text_rows)}, modelc={len(modelc_unique_rows)})"
        ),
        f"S4 model-generated only={len(generated_rows)} (data_source {DATA_SOURCE}).",
        (
            "S1 phishing="
            f"{len(s1_merged_rows)} (existing={len(s1_preserved_rows)}, "
            f"flowgpt={len(flowgpt_rows)}, poejailbreak={len(poejailbreak_unique_rows)})"
        ),
        f"S1 malla only={len(malla_rows)} (data_source {MALLA_SOURCE}).",
        (
            "S2 HW-B="
            f"{len(s2_hw_b_merged_rows)} (existing={len(s2_hw_b_preserved_rows)}, "
            f"human_legit={kaggle_stats['human_legit']})"
        ),
        (
            "S2 HW-P="
            f"{len(s2_hw_p_merged_rows)} (existing={len(s2_hw_p_preserved_rows)}, "
            f"human_phishing={kaggle_stats['human_phishing']})"
        ),
        (
            "S2 LLM-B="
            f"{len(s2_llm_b_merged_rows)} (existing={len(s2_llm_b_preserved_rows)}, "
            f"llm_legit={kaggle_stats['llm_legit']})"
        ),
        (
            "S2 LLM-P="
            f"{len(s2_llm_p_merged_rows)} (existing={len(s2_llm_p_preserved_rows)}, "
            f"llm_phishing={kaggle_stats['llm_phishing']})"
        ),
        (
            "Kaggle processed="
            f"{kaggle_stats['human_legit'] + kaggle_stats['human_phishing'] + kaggle_stats['llm_legit'] + kaggle_stats['llm_phishing']} "
            f"(data_source {KAGGLE_SOURCE})."
        ),
    )


if __name__ == "__main__":
    main()
