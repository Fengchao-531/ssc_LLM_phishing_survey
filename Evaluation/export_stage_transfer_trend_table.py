#!/usr/bin/env python3
"""Export Table 2: stage-level transfer trend summary."""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from render_detector_stage_tables_markdown import (
    compute_academic_hw_overall,
    compute_industry_hw_overall,
)


SCRIPT_DIR = Path(__file__).resolve().parent
TABLES_DIR = SCRIPT_DIR / "detector-stage-metric-tables"
OUTPUT_DIR = SCRIPT_DIR / "stage-transfer-trend"
CSV_OUTPUT_PATH = OUTPUT_DIR / "table_2_stage_transfer_trend.csv"
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
METRICS = ["recall", "f2", "mcc"]
CATEGORY_CONFIGS = {
    "academic": {
        "label": "Academic",
        "dir": TABLES_DIR / "academic",
        "baseline_fn": compute_academic_hw_overall,
        "fixed_n": None,
    },
    "industry": {
        "label": "Industrial",
        "dir": TABLES_DIR / "industry",
        "baseline_fn": compute_industry_hw_overall,
        "fixed_n": 3,
    },
}


def format_value(value: object) -> str:
    if pd.isna(value):
        return "-"
    return f"{float(value):.4f}"


def aggregate_values(values: list[float], fixed_n: int | None) -> float | pd.NA:
    if fixed_n is not None:
        return float(sum(values) / fixed_n) if values else 0.0
    return float(pd.Series(values).mean()) if values else pd.NA


def collect_stage_metric_views() -> tuple[dict[str, dict[str, dict[str, dict[str, list[float]]]]], dict[str, dict[str, dict[str, int]]]]:
    metric_views: dict[str, dict[str, dict[str, dict[str, list[float]]]]] = {
        category: {
            stage: {
                metric: {"hw": [], "llm": [], "delta": []}
                for metric in METRICS
            }
            for stage in STAGE_ORDER
        }
        for category in CATEGORY_CONFIGS
    }
    counts: dict[str, dict[str, dict[str, int]]] = {
        category: {stage: {metric: 0 for metric in METRICS} for stage in STAGE_ORDER}
        for category in CATEGORY_CONFIGS
    }

    for category_name, config in CATEGORY_CONFIGS.items():
        category_dir = config["dir"]
        baseline_fn = config["baseline_fn"]

        for csv_path in sorted(category_dir.glob("*.csv")):
            detector_name = csv_path.stem
            baseline_metrics = baseline_fn(detector_name)
            df = pd.read_csv(csv_path)

            for _, row in df.iterrows():
                stage_name = str(row["stage"])
                if stage_name not in STAGE_ORDER:
                    continue
                for metric_name in METRICS:
                    llm_value = pd.to_numeric(pd.Series([row[f"llm_{metric_name}"]]), errors="coerce").iloc[0]
                    hw_value = pd.to_numeric(pd.Series([baseline_metrics[metric_name]]), errors="coerce").iloc[0]
                    if pd.isna(llm_value) or pd.isna(hw_value):
                        continue
                    delta = float(llm_value) - float(hw_value)
                    metric_views[category_name][stage_name][metric_name]["hw"].append(float(hw_value))
                    metric_views[category_name][stage_name][metric_name]["llm"].append(float(llm_value))
                    metric_views[category_name][stage_name][metric_name]["delta"].append(delta)
                    counts[category_name][stage_name][metric_name] += 1

    return metric_views, counts


def build_table_frame() -> tuple[pd.DataFrame, dict[str, dict[str, dict[str, int]]]]:
    metric_views, counts = collect_stage_metric_views()
    rows: list[dict[str, object]] = []

    for stage_name in STAGE_ORDER:
        row: dict[str, object] = {"stage": stage_name}
        for category_name, config in CATEGORY_CONFIGS.items():
            prefix = "academic" if category_name == "academic" else "industrial"
            fixed_n = config["fixed_n"]
            for metric_name in METRICS:
                for view_name in ["hw", "llm", "delta"]:
                    values = metric_views[category_name][stage_name][metric_name][view_name]
                    row[f"{prefix}_{metric_name}_{view_name}"] = aggregate_values(values, fixed_n)
        rows.append(row)

    columns = ["stage"]
    for prefix in ["academic", "industrial"]:
        for metric_name in METRICS:
            columns.extend(
                [
                    f"{prefix}_{metric_name}_hw",
                    f"{prefix}_{metric_name}_llm",
                    f"{prefix}_{metric_name}_delta",
                ]
            )
    return pd.DataFrame(rows, columns=columns), counts


def render_table_html(df: pd.DataFrame) -> list[str]:
    lines = [
        "<table>",
        "  <thead>",
        "    <tr>",
        '      <th rowspan="3">Stage</th>',
        '      <th colspan="9">Academic</th>',
        '      <th colspan="9">Industrial</th>',
        "    </tr>",
        "    <tr>",
        '      <th colspan="3">Recall</th>',
        '      <th colspan="3">F2</th>',
        '      <th colspan="3">MCC</th>',
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
            "academic_recall_hw",
            "academic_recall_llm",
            "academic_recall_delta",
            "academic_f2_hw",
            "academic_f2_llm",
            "academic_f2_delta",
            "academic_mcc_hw",
            "academic_mcc_llm",
            "academic_mcc_delta",
            "industrial_recall_hw",
            "industrial_recall_llm",
            "industrial_recall_delta",
            "industrial_f2_hw",
            "industrial_f2_llm",
            "industrial_f2_delta",
            "industrial_mcc_hw",
            "industrial_mcc_llm",
            "industrial_mcc_delta",
        ]:
            lines.append(f"      <td>{format_value(row[column])}</td>")
        lines.append("    </tr>")

    lines.extend(["  </tbody>", "</table>"])
    return lines


def build_summary_lines(df: pd.DataFrame, counts: dict[str, dict[str, dict[str, int]]]) -> list[str]:
    lines: list[str] = []
    for category_name, config in CATEGORY_CONFIGS.items():
        label = config["label"]
        column_name = "academic_mcc_delta" if category_name == "academic" else "industrial_mcc_delta"
        valid = df[["stage", column_name]].dropna()
        if valid.empty:
            continue
        hardest_row = valid.loc[valid[column_name].idxmin()]
        easiest_row = valid.loc[valid[column_name].idxmax()]
        fixed_n = config["fixed_n"]
        hardest_n = fixed_n if fixed_n is not None else counts[category_name][str(hardest_row["stage"])]["mcc"]
        easiest_n = fixed_n if fixed_n is not None else counts[category_name][str(easiest_row["stage"])]["mcc"]
        lines.append(
            f"- `{label}` hardest stage by mean `Δ MCC`: "
            f"`{hardest_row['stage']}` (`{format_value(hardest_row[column_name])}`, `n = {hardest_n}`)."
        )
        lines.append(
            f"- `{label}` easiest stage by mean `Δ MCC`: "
            f"`{easiest_row['stage']}` (`{format_value(easiest_row[column_name])}`, `n = {easiest_n}`)."
        )
    return lines


def write_readme(df: pd.DataFrame, counts: dict[str, dict[str, dict[str, int]]]) -> None:
    lines = [
        "# Table 2: Stage-Level LLM Difficulty",
        "",
        "This table shifts the focus from individual detectors to the stages themselves.",
        "Each `Δ` value is computed relative to each detector's own `HW-overall` baseline:",
        "",
        "- `Δ metric = LLM-stage metric - HW-overall metric`",
        "",
        "Interpretation notes:",
        "",
        "- Negative `Δ` means the stage hurts detector performance relative to that detector's HW baseline.",
        "- `Academic` cells use the mean over available paired detectors for that stage/metric.",
        "- `Industrial` cells currently use a fixed denominator of `n = 3` for every stage/metric.",
        "- This version keeps `S6` and `S8` split into their sub-stages because their behavior is meaningfully different.",
        "- Blank cells mean that detector family currently has no usable paired values for that stage/metric.",
        "",
    ]
    lines.extend(build_summary_lines(df, counts))
    lines.extend([""])
    lines.extend(render_table_html(df))
    lines.append("")
    lines.append("CSV export: `table_2_stage_transfer_trend.csv`.")
    README_OUTPUT_PATH.write_text("\n".join(lines), encoding="utf-8")


def main() -> int:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    df, counts = build_table_frame()
    df.to_csv(CSV_OUTPUT_PATH, index=False)
    write_readme(df, counts)
    print(f"[OK] CSV -> {CSV_OUTPUT_PATH}")
    print(f"[OK] README -> {README_OUTPUT_PATH}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
