#!/usr/bin/env python3
"""Export overall detector FPR/FNR risk profile table."""

from __future__ import annotations

from collections import OrderedDict
from pathlib import Path

import pandas as pd

from export_hw_llm_transfer_table import (
    collect_overall_metric_bundles,
)
from export_detector_stage_tables import (
    ACADEMIC_DETECTORS,
    ACADEMIC_HW_DIR,
    ACADEMIC_LLM_DIR,
    INDUSTRY_DETECTORS,
    INDUSTRY_HW_DIR,
    INDUSTRY_LLM_DIR,
)


SCRIPT_DIR = Path(__file__).resolve().parent
OUTPUT_DIR = SCRIPT_DIR / "detector-fpr-fnr-risk"
CSV_OUTPUT_PATH = OUTPUT_DIR / "detector_fpr_fnr_overall.csv"
README_OUTPUT_PATH = OUTPUT_DIR / "README.md"
RISK_METRICS = ["fpr", "fnr"]


def format_value(value: object) -> str:
    if pd.isna(value):
        return "-"
    return f"{float(value):.2f}"


def scale_rate_columns(df: pd.DataFrame) -> pd.DataFrame:
    scaled = df.copy()
    for column in scaled.columns:
        if column in {"group", "detector"}:
            continue
        numeric = pd.to_numeric(scaled[column], errors="coerce")
        scaled[column] = numeric.mul(100).round(2)
    return scaled


def format_frame_for_csv(df: pd.DataFrame) -> pd.DataFrame:
    formatted = df.copy()
    for column in formatted.columns:
        if column in {"group", "detector"}:
            continue
        formatted[column] = formatted[column].apply(lambda value: "" if pd.isna(value) else f"{float(value):.2f}")
    return formatted


def build_group_rows(
    *,
    group_label: str,
    category_name: str,
    detector_configs: OrderedDict[str, dict[str, str | None]],
    hw_dir: Path,
    llm_dir: Path,
) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    metric_columns = [
        "hw_fpr",
        "llm_fpr",
        "delta_fpr",
        "hw_fnr",
        "llm_fnr",
        "delta_fnr",
    ]

    hw_bundles = collect_overall_metric_bundles(
        source_dir=hw_dir,
        category_name=category_name,
        source_kind="hw",
        detector_configs=detector_configs,
    )
    llm_bundles = collect_overall_metric_bundles(
        source_dir=llm_dir,
        category_name=category_name,
        source_kind="llm",
        detector_configs=detector_configs,
    )

    for detector_name in detector_configs:
        hw_metrics = hw_bundles.get(detector_name)
        llm_metrics = llm_bundles.get(detector_name)
        row: dict[str, object] = {"group": group_label, "detector": detector_name}
        for metric_name in RISK_METRICS:
            hw_value = pd.NA if hw_metrics is None else hw_metrics.get(metric_name, pd.NA)
            llm_value = pd.NA if llm_metrics is None else llm_metrics.get(metric_name, pd.NA)
            delta_value = pd.NA
            if not pd.isna(hw_value) and not pd.isna(llm_value):
                delta_value = float(llm_value) - float(hw_value)
            row[f"hw_{metric_name}"] = hw_value
            row[f"llm_{metric_name}"] = llm_value
            row[f"delta_{metric_name}"] = delta_value
        rows.append(row)

    detector_frame = pd.DataFrame(rows)
    mean_row: dict[str, object] = {"group": group_label, "detector": f"{group_label} mean"}
    median_row: dict[str, object] = {"group": group_label, "detector": f"{group_label} median"}
    for column in metric_columns:
        numeric = pd.to_numeric(detector_frame[column], errors="coerce")
        mean_row[column] = float(numeric.mean()) if numeric.notna().any() else pd.NA
        median_row[column] = float(numeric.median()) if numeric.notna().any() else pd.NA
    rows.append(mean_row)
    rows.append(median_row)
    return rows


def build_table_frame() -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    rows.extend(
        build_group_rows(
            group_label="Academic",
            category_name="academic",
            detector_configs=ACADEMIC_DETECTORS,
            hw_dir=ACADEMIC_HW_DIR,
            llm_dir=ACADEMIC_LLM_DIR,
        )
    )
    rows.extend(
        build_group_rows(
            group_label="Industrial",
            category_name="industry",
            detector_configs=INDUSTRY_DETECTORS,
            hw_dir=INDUSTRY_HW_DIR,
            llm_dir=INDUSTRY_LLM_DIR,
        )
    )
    columns = [
        "group",
        "detector",
        "hw_fpr",
        "llm_fpr",
        "delta_fpr",
        "hw_fnr",
        "llm_fnr",
        "delta_fnr",
    ]
    return pd.DataFrame(rows, columns=columns)


def render_table_html(df: pd.DataFrame) -> list[str]:
    lines = [
        "<table>",
        "  <thead>",
        "    <tr>",
        '      <th rowspan="2">Group</th>',
        '      <th rowspan="2">Detector</th>',
        '      <th colspan="3">FPR</th>',
        '      <th colspan="3">FNR</th>',
        "    </tr>",
        "    <tr>",
        "      <th>HW</th>",
        "      <th>LLM</th>",
        "      <th>Δ</th>",
        "      <th>HW</th>",
        "      <th>LLM</th>",
        "      <th>Δ</th>",
        "    </tr>",
        "  </thead>",
        "  <tbody>",
    ]

    previous_group = None
    for _, row in df.iterrows():
        lines.append("    <tr>")
        if row["group"] != previous_group:
            lines.append(f'      <td><strong>{row["group"]}</strong></td>')
            previous_group = row["group"]
        else:
            lines.append("      <td></td>")

        detector_name = str(row["detector"])
        is_summary = detector_name.endswith(" mean") or detector_name.endswith(" median")
        detector_cell = f"<strong>{detector_name}</strong>" if is_summary else detector_name
        lines.append(f"      <td>{detector_cell}</td>")
        for column in ["hw_fpr", "llm_fpr", "delta_fpr", "hw_fnr", "llm_fnr", "delta_fnr"]:
            lines.append(f"      <td>{format_value(row[column])}</td>")
        lines.append("    </tr>")

    lines.extend(["  </tbody>", "</table>"])
    return lines


def build_summary_lines(df: pd.DataFrame) -> list[str]:
    lines: list[str] = []
    for group_name in ["Academic", "Industrial"]:
        group_rows = df[df["group"] == group_name].copy()
        group_rows = group_rows[
            ~group_rows["detector"].astype(str).str.endswith(" mean")
            & ~group_rows["detector"].astype(str).str.endswith(" median")
        ]
        lines.append(
            f"- `{group_name}` mean `Δ FPR = {format_value(pd.to_numeric(group_rows['delta_fpr'], errors='coerce').mean())}`, "
            f"`Δ FNR = {format_value(pd.to_numeric(group_rows['delta_fnr'], errors='coerce').mean())}`."
        )
    return lines


def write_readme(df: pd.DataFrame) -> None:
    lines = [
        "# Detector Overall Risk Profile",
        "",
        "This table focuses on detector risk behavior rather than aggregate task performance.",
        "",
        "- All rates are reported as percentages and rounded to two decimals.",
        "- `Δ = LLM - HW`, expressed in percentage points.",
        "- Positive `Δ FPR` means more false positives on LLM-generated content.",
        "- Positive `Δ FNR` means more false negatives on LLM-generated content.",
        "- Blank cells mean the detector does not yet have a usable paired `HW` and `LLM` overall result.",
        "",
    ]
    lines.extend(build_summary_lines(df))
    lines.extend([""])
    lines.extend(render_table_html(df))
    lines.append("")
    lines.append("CSV export: `detector_fpr_fnr_overall.csv`.")
    README_OUTPUT_PATH.write_text("\n".join(lines), encoding="utf-8")


def main() -> int:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    df = scale_rate_columns(build_table_frame())
    format_frame_for_csv(df).to_csv(CSV_OUTPUT_PATH, index=False)
    write_readme(df)
    print(f"[OK] CSV -> {CSV_OUTPUT_PATH}")
    print(f"[OK] README -> {README_OUTPUT_PATH}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
