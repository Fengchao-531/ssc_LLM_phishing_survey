#!/usr/bin/env python3
"""Export focused stage tables for selected detectors."""

from __future__ import annotations

from pathlib import Path

import pandas as pd


SCRIPT_DIR = Path(__file__).resolve().parent
TABLES_DIR = SCRIPT_DIR / "detector-stage-metric-tables"
OUTPUT_DIR = SCRIPT_DIR / "selected-detector-stage-deltas"
SELECTED = [
    ("academic", "scamllm"),
    ("industry", "email_phishing_detection_v3"),
]
METRICS = ["recall", "f2", "mcc"]


def format_value(value: object) -> str:
    if pd.isna(value):
        return "-"
    return f"{float(value):.4f}"


def build_detector_table(input_path: Path) -> pd.DataFrame:
    source_df = pd.read_csv(input_path)
    rows: list[dict[str, object]] = []
    for _, row in source_df.iterrows():
        out_row: dict[str, object] = {"stage": row["stage"]}
        for metric_name in METRICS:
            hw_value = pd.to_numeric(pd.Series([row[f"hw_{metric_name}"]]), errors="coerce").iloc[0]
            llm_value = pd.to_numeric(pd.Series([row[f"llm_{metric_name}"]]), errors="coerce").iloc[0]
            delta_value = pd.NA
            if not pd.isna(hw_value) and not pd.isna(llm_value):
                delta_value = float(llm_value) - float(hw_value)

            out_row[f"hw_{metric_name}"] = hw_value
            out_row[f"llm_{metric_name}"] = llm_value
            out_row[f"delta_{metric_name}"] = delta_value
        rows.append(out_row)

    return pd.DataFrame(
        rows,
        columns=[
            "stage",
            "hw_recall",
            "llm_recall",
            "delta_recall",
            "hw_f2",
            "llm_f2",
            "delta_f2",
            "hw_mcc",
            "llm_mcc",
            "delta_mcc",
        ],
    )


def render_table_markdown(detector_name: str, df: pd.DataFrame) -> list[str]:
    lines = [
        f"## {detector_name}",
        "",
        "<table>",
        "  <thead>",
        "    <tr>",
        '      <th rowspan="2">Stage</th>',
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

    for _, row in df.iterrows():
        lines.append("    <tr>")
        lines.append(f"      <td>{row['stage']}</td>")
        for column in [
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
            lines.append(f"      <td>{format_value(row[column])}</td>")
        lines.append("    </tr>")

    lines.extend(["  </tbody>", "</table>", ""])
    return lines


def write_readme(tables: list[tuple[str, pd.DataFrame]]) -> None:
    lines = [
        "# Selected Detector Stage Delta Tables",
        "",
        "Focused stage-by-stage comparison tables for the requested detectors.",
        "",
        "- `Δ = LLM - HW`.",
        "- Blank cells mean the source detector table does not currently have usable values for that stage/side.",
        "",
    ]
    for detector_name, df in tables:
        lines.extend(render_table_markdown(detector_name, df))
    (OUTPUT_DIR / "README.md").write_text("\n".join(lines), encoding="utf-8")


def main() -> int:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    rendered_tables: list[tuple[str, pd.DataFrame]] = []

    for category_name, detector_name in SELECTED:
        input_path = TABLES_DIR / category_name / f"{detector_name}.csv"
        df = build_detector_table(input_path)
        output_path = OUTPUT_DIR / f"{detector_name}.csv"
        df.to_csv(output_path, index=False)
        rendered_tables.append((detector_name, df))
        print(f"[OK] {detector_name} -> {output_path}")

    write_readme(rendered_tables)
    print(f"[OK] README -> {OUTPUT_DIR / 'README.md'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
