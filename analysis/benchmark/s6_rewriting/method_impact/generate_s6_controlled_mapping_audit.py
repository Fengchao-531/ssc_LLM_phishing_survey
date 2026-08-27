#!/usr/bin/env python3
import csv
import sys
from collections import Counter
from pathlib import Path


csv.field_size_limit(sys.maxsize)

ROOT = Path(__file__).resolve().parent
PROJECT_ROOT = ROOT.parents[2]
RAW_ROOT = PROJECT_ROOT / "Datasets" / "sublist" / "S6-Stealthy Rewriting"
MERGED = ROOT.parent / "A-I Differences" / "selected_llm_tp_detector_merged.csv"

OUTPUT_CSV = ROOT / "s6_uta_mpg_controlled_mapping_audit.csv"
OUTPUT_MD = ROOT / "s6_uta_mpg_controlled_mapping_audit.md"


RAW_FILES = {
    "Original": RAW_ROOT / "HW-P.csv",
    "UTA": RAW_ROOT / "UTA-LLM-P.csv",
    "MPG": RAW_ROOT / "MPG-LLM-P.csv",
}


def read_csv(path):
    with path.open(newline="", encoding="utf-8", errors="replace") as input_file:
        return list(csv.DictReader(input_file))


def text_key(row, subject_col="Subject", body_col="Body"):
    return (" ".join((row.get(subject_col, "") + " " + row.get(body_col, "")).split())).strip()


def merged_text_key(row):
    return (" ".join((row.get("subject", "") + " " + row.get("body", "")).split())).strip()


def build_audit():
    raw = {label: read_csv(path) for label, path in RAW_FILES.items()}
    original = raw["Original"]
    uta = raw["UTA"]
    mpg = raw["MPG"]
    n = min(len(original), len(uta), len(mpg))

    raw_rows = []
    for label, rows in raw.items():
        sources = Counter(row.get("data_source", "") for row in rows)
        raw_rows.append(
            {
                "scope": "raw_generation_csv",
                "dataset": label,
                "path": str(RAW_FILES[label]),
                "rows": len(rows),
                "unique_data_sources": len(sources),
                "first_data_source": rows[0].get("data_source", "") if rows else "",
                "first_subject": rows[0].get("Subject", "") if rows else "",
                "last_subject": rows[-1].get("Subject", "") if rows else "",
                "mapping_status": "row_index_aligned_to_HW_P" if label != "Original" else "source_originals",
            }
        )

    row_map_examples = []
    for index in [0, 1, n - 1]:
        row_map_examples.append(
            {
                "scope": "row_mapping_example",
                "dataset": "UTA_vs_MPG",
                "path": "",
                "rows": index + 1,
                "unique_data_sources": "",
                "first_data_source": original[index].get("data_source", ""),
                "first_subject": original[index].get("Subject", ""),
                "last_subject": "UTA: " + uta[index].get("Subject", "") + " | MPG: " + mpg[index].get("Subject", ""),
                "mapping_status": "same original row_id={}".format(index + 1),
            }
        )

    merged_rows = read_csv(MERGED)
    merged_audit = []
    for stage in ["S6-UTA", "S6-MPG"]:
        hw = [
            row
            for row in merged_rows
            if row.get("stage") == stage
            and row.get("source") == "HW"
            and row.get("label_x") in ("1", "1.0")
        ]
        llm = [
            row
            for row in merged_rows
            if row.get("stage") == stage
            and row.get("source") == "LLM"
            and row.get("label_x") in ("1", "1.0")
        ]
        raw_original_keys = set(text_key(row) for row in original)
        merged_hw_keys = set(merged_text_key(row) for row in hw)
        merged_audit.append(
            {
                "scope": "merged_analysis_csv",
                "dataset": stage,
                "path": str(MERGED),
                "rows": len(hw),
                "unique_data_sources": len(set(row.get("source_file", "") for row in hw)),
                "first_data_source": hw[0].get("source_file", "") if hw else "",
                "first_subject": hw[0].get("subject", "") if hw else "",
                "last_subject": llm[0].get("subject", "") if llm else "",
                "mapping_status": "merged HW overlap with raw HW-P: {}/{}; not sufficient for controlled original baseline".format(
                    len(raw_original_keys & merged_hw_keys),
                    len(raw_original_keys),
                ),
            }
        )

    rows = raw_rows + row_map_examples + merged_audit
    with OUTPUT_CSV.open("w", newline="", encoding="utf-8") as output_file:
        fieldnames = [
            "scope",
            "dataset",
            "path",
            "rows",
            "unique_data_sources",
            "first_data_source",
            "first_subject",
            "last_subject",
            "mapping_status",
        ]
        writer = csv.DictWriter(output_file, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    lines = [
        "# S6 UTA-MPG Controlled Mapping Audit",
        "",
        "Raw generation CSVs confirm the controlled text design: `HW-P.csv`, `UTA-LLM-P.csv`, and `MPG-LLM-P.csv` each contain 1600 phishing rows, and UTA/MPG row `i` is a rewrite of HW-P row `i`.",
        "",
        "This supports a future same-original UTA-vs-MPG analysis once row IDs are joined to persuasion scores and detector predictions.",
        "",
        "Important guardrail: the current merged analysis file's `S6-UTA` and `S6-MPG` HW rows are not a reliable same-original baseline for that controlled analysis. Do not use the old row-order `delta_pair` outputs as final controlled original-to-rewrite evidence.",
        "",
        "## Required Join For Q2-Q4",
        "",
        "Create or export a row-level table with at least `row_id`, `original_text`, `UTA_text`, `MPG_text`, the 21 original/UTA/MPG pair scores, and SecureNet/V3 predictions for original, UTA, and MPG. Then compute paired UTA-vs-MPG persuasion deltas and TP->FN transitions.",
        "",
        "## Audit CSV",
        "",
        "`s6_uta_mpg_controlled_mapping_audit.csv`",
    ]
    OUTPUT_MD.write_text("\n".join(lines) + "\n", encoding="utf-8")


if __name__ == "__main__":
    build_audit()
