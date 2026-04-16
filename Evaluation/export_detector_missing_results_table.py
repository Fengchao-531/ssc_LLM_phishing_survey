#!/usr/bin/env python3
"""Export a detector coverage gap table showing missing HW/LLM results."""

from __future__ import annotations

from pathlib import Path

import pandas as pd


SCRIPT_DIR = Path(__file__).resolve().parent
TABLES_DIR = SCRIPT_DIR / "detector-stage-metric-tables"
OUTPUT_DIR = SCRIPT_DIR / "detector-missing-results"
CSV_OUTPUT_PATH = OUTPUT_DIR / "detector_missing_results.csv"
README_OUTPUT_PATH = OUTPUT_DIR / "README.md"
STAGE_ORDER = [
    "S1",
    "S2",
    "S4",
    "S5",
    "S6-MPG",
    "S6-UTA",
    "S6-fuzzer",
    "S8-deepseek",
    "S8-llama",
    "S8-ministral",
]
CATEGORIES = [("academic", "Academic"), ("industry", "Industry")]


def format_stage_list(stage_names: list[str]) -> str:
    return "-" if not stage_names else ", ".join(stage_names)


def summarize_detector(csv_path: Path, category_label: str) -> dict[str, object]:
    df = pd.read_csv(csv_path)
    hw_present = df.loc[pd.to_numeric(df["hw_f1"], errors="coerce").notna(), "stage"].astype(str).tolist()
    llm_present = df.loc[pd.to_numeric(df["llm_f1"], errors="coerce").notna(), "stage"].astype(str).tolist()

    hw_missing = [stage for stage in STAGE_ORDER if stage not in hw_present]
    llm_missing = [stage for stage in STAGE_ORDER if stage not in llm_present]

    if not hw_missing and not llm_missing:
        status = "complete"
    elif hw_present or llm_present:
        status = "partial"
    else:
        status = "missing_all"

    return {
        "category": category_label,
        "detector": csv_path.stem,
        "hw_present_count": len(hw_present),
        "hw_missing_count": len(hw_missing),
        "hw_missing_stages": format_stage_list(hw_missing),
        "llm_present_count": len(llm_present),
        "llm_missing_count": len(llm_missing),
        "llm_missing_stages": format_stage_list(llm_missing),
        "status": status,
    }


def build_table_frame() -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for category_name, category_label in CATEGORIES:
        category_dir = TABLES_DIR / category_name
        for csv_path in sorted(category_dir.glob("*.csv")):
            rows.append(summarize_detector(csv_path, category_label))
    return pd.DataFrame(
        rows,
        columns=[
            "category",
            "detector",
            "hw_present_count",
            "hw_missing_count",
            "hw_missing_stages",
            "llm_present_count",
            "llm_missing_count",
            "llm_missing_stages",
            "status",
        ],
    )


def render_table_html(df: pd.DataFrame) -> list[str]:
    lines = [
        "<table>",
        "  <thead>",
        "    <tr>",
        "      <th>Category</th>",
        "      <th>Detector</th>",
        "      <th>HW present</th>",
        "      <th>HW missing</th>",
        "      <th>HW missing stages</th>",
        "      <th>LLM present</th>",
        "      <th>LLM missing</th>",
        "      <th>LLM missing stages</th>",
        "      <th>Status</th>",
        "    </tr>",
        "  </thead>",
        "  <tbody>",
    ]
    for _, row in df.iterrows():
        lines.append("    <tr>")
        for column in df.columns:
            lines.append(f"      <td>{row[column]}</td>")
        lines.append("    </tr>")
    lines.extend(["  </tbody>", "</table>"])
    return lines


def write_readme(df: pd.DataFrame) -> None:
    lines = [
        "# Detector Missing Results",
        "",
        "This table shows which detectors still lack usable stage-level results.",
        "",
        "- `HW` refers to the GD / hardware-style benchmark side.",
        "- `LLM` refers to the LLM-generated benchmark side.",
        "- `present` counts how many stages already have usable metrics.",
        "- `missing stages` lists the exact stage names that still need results.",
        "",
    ]
    lines.extend(render_table_html(df))
    lines.append("")
    lines.append("CSV export: `detector_missing_results.csv`.")
    README_OUTPUT_PATH.write_text("\n".join(lines), encoding="utf-8")


def main() -> int:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    df = build_table_frame()
    df.to_csv(CSV_OUTPUT_PATH, index=False)
    write_readme(df)
    print(f"[OK] CSV -> {CSV_OUTPUT_PATH}")
    print(f"[OK] README -> {README_OUTPUT_PATH}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
