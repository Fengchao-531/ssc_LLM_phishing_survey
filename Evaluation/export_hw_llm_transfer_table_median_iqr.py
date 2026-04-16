#!/usr/bin/env python3
"""Export Table 1 companion view: median + IQR from stage-level detector tables."""

from __future__ import annotations

from pathlib import Path

import pandas as pd


SCRIPT_DIR = Path(__file__).resolve().parent
TABLES_DIR = SCRIPT_DIR / "detector-stage-metric-tables"
OUTPUT_DIR = SCRIPT_DIR / "hw-vs-llm-overall-transfer"
CSV_OUTPUT_PATH = OUTPUT_DIR / "table_1_hw_vs_llm_transfer_median_iqr.csv"
README_OUTPUT_PATH = OUTPUT_DIR / "README_median_iqr.md"
CATEGORIES = [("Academic", "academic"), ("Industrial", "industry")]
TRANSFER_METRICS = ["recall", "f2", "mcc"]


def format_value(value: object) -> str:
    if pd.isna(value):
        return "-"
    return f"{float(value):.4f}"


def format_median_iqr(median: object, iqr: object) -> str:
    if pd.isna(median):
        return "-"
    if pd.isna(iqr):
        return f"{float(median):.4f}"
    return f"{float(median):.4f} [{float(iqr):.4f}]"


def compute_iqr(series: pd.Series) -> float | pd.NA:
    numeric = pd.to_numeric(series, errors="coerce").dropna()
    if numeric.empty:
        return pd.NA
    q1 = float(numeric.quantile(0.25))
    q3 = float(numeric.quantile(0.75))
    return q3 - q1


def compute_median(series: pd.Series) -> float | pd.NA:
    numeric = pd.to_numeric(series, errors="coerce").dropna()
    if numeric.empty:
        return pd.NA
    return float(numeric.median())


def collect_detector_row(group_label: str, detector_name: str, csv_path: Path) -> dict[str, object]:
    df = pd.read_csv(csv_path)
    row: dict[str, object] = {"group": group_label, "detector": detector_name}

    for metric_name in TRANSFER_METRICS:
        hw_series = pd.to_numeric(df[f"hw_{metric_name}"], errors="coerce")
        llm_series = pd.to_numeric(df[f"llm_{metric_name}"], errors="coerce")
        delta_series = llm_series - hw_series

        row[f"hw_{metric_name}_median"] = compute_median(hw_series)
        row[f"hw_{metric_name}_iqr"] = compute_iqr(hw_series)
        row[f"llm_{metric_name}_median"] = compute_median(llm_series)
        row[f"llm_{metric_name}_iqr"] = compute_iqr(llm_series)
        row[f"delta_{metric_name}_median"] = compute_median(delta_series)
        row[f"delta_{metric_name}_iqr"] = compute_iqr(delta_series)

    return row


def build_group_rows(group_label: str, category_name: str) -> list[dict[str, object]]:
    category_dir = TABLES_DIR / category_name
    detector_rows: list[dict[str, object]] = []

    for csv_path in sorted(category_dir.glob("*.csv")):
        detector_rows.append(collect_detector_row(group_label, csv_path.stem, csv_path))

    detector_frame = pd.DataFrame(detector_rows)
    summary_columns = [col for col in detector_frame.columns if col not in {"group", "detector"}]

    median_row: dict[str, object] = {"group": group_label, "detector": f"{group_label} median"}
    for column in summary_columns:
        numeric = pd.to_numeric(detector_frame[column], errors="coerce")
        median_row[column] = compute_median(numeric)

    iqr_row: dict[str, object] = {"group": group_label, "detector": f"{group_label} IQR"}
    for column in summary_columns:
        numeric = pd.to_numeric(detector_frame[column], errors="coerce")
        iqr_row[column] = compute_iqr(numeric)

    detector_rows.append(median_row)
    detector_rows.append(iqr_row)
    return detector_rows


def build_table_frame() -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for group_label, category_name in CATEGORIES:
        rows.extend(build_group_rows(group_label, category_name))

    columns = ["group", "detector"]
    for metric_name in TRANSFER_METRICS:
        columns.extend(
            [
                f"hw_{metric_name}_median",
                f"hw_{metric_name}_iqr",
                f"llm_{metric_name}_median",
                f"llm_{metric_name}_iqr",
                f"delta_{metric_name}_median",
                f"delta_{metric_name}_iqr",
            ]
        )
    return pd.DataFrame(rows, columns=columns)


def render_table_html(df: pd.DataFrame) -> list[str]:
    lines = [
        "<table>",
        "  <thead>",
        "    <tr>",
        '      <th rowspan="2">Group</th>',
        '      <th rowspan="2">Detector</th>',
        '      <th colspan="3">Recall</th>',
        '      <th colspan="3">F2</th>',
        '      <th colspan="3">MCC</th>',
        "    </tr>",
        "    <tr>",
        "      <th>HW</th>",
        "      <th>LLM</th>",
        "      <th>Δ</th>",
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
        detector_name = str(row["detector"])
        is_summary = detector_name.endswith(" median") or detector_name.endswith(" IQR")
        lines.append("    <tr>")
        if row["group"] != previous_group:
            lines.append(f'      <td><strong>{row["group"]}</strong></td>')
            previous_group = row["group"]
        else:
            lines.append("      <td></td>")

        detector_cell = f"<strong>{detector_name}</strong>" if is_summary else detector_name
        lines.append(f"      <td>{detector_cell}</td>")
        for prefix in [
            "hw_recall",
            "llm_recall",
            "delta_recall",
            "hw_f2",
            "llm_f2",
            "delta_f2",
            "hw_mcc",
            "llm_mcc",
            "delta_mcc",
        ]:
            lines.append(
                f"      <td>{format_median_iqr(row[f'{prefix}_median'], row[f'{prefix}_iqr'])}</td>"
            )
        lines.append("    </tr>")

    lines.extend(["  </tbody>", "</table>"])
    return lines


def build_summary_lines(df: pd.DataFrame) -> list[str]:
    lines: list[str] = []
    for group_label, _category_name in CATEGORIES:
        group_rows = df[df["group"] == group_label].copy()
        detector_rows = group_rows[
            ~group_rows["detector"].astype(str).str.endswith(" median")
            & ~group_rows["detector"].astype(str).str.endswith(" IQR")
        ]
        paired_count = int(pd.to_numeric(detector_rows["delta_mcc_median"], errors="coerce").notna().sum())
        lines.append(
            f"- `{group_label}` paired detectors for stage-robust transfer summary: "
            f"`n = {paired_count}`."
        )
    return lines


def write_readme(df: pd.DataFrame) -> None:
    lines = [
        "# Table 1 Companion: Median + IQR",
        "",
        "This companion table keeps the same `HW vs. LLM transfer` layout,",
        "but summarizes each detector using stage-level `median [IQR]` instead of",
        "`overall ± std`.",
        "",
        "Interpretation notes:",
        "",
        "- Each cell is formatted as `median [IQR]`.",
        "- Here `IQR = Q3 - Q1` across the available stage-level metrics.",
        "- `Δ = LLM - HW`, so a negative value means the detector is weaker on LLM-generated content.",
        "- This table is a robustness-oriented view across stages, not a raw-sample overall aggregation.",
        "",
    ]
    lines.extend(build_summary_lines(df))
    lines.extend([""])
    lines.extend(render_table_html(df))
    lines.append("")
    lines.append("CSV export: `table_1_hw_vs_llm_transfer_median_iqr.csv`.")
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
