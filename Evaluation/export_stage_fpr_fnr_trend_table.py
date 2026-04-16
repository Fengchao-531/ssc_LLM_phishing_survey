#!/usr/bin/env python3
"""Export stage-level FPR/FNR risk trend table."""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from render_detector_stage_tables_markdown import (
    compute_academic_hw_overall,
    compute_industry_hw_overall,
)


SCRIPT_DIR = Path(__file__).resolve().parent
TABLES_DIR = SCRIPT_DIR / "detector-stage-metric-tables"
OUTPUT_DIR = SCRIPT_DIR / "stage-fpr-fnr-risk-trend"
CSV_OUTPUT_PATH = OUTPUT_DIR / "stage_fpr_fnr_trend.csv"
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
RISK_METRICS = ["fpr", "fnr"]
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
    return f"{float(value):.2f}"


def scale_rate_columns(df: pd.DataFrame) -> pd.DataFrame:
    scaled = df.copy()
    for column in scaled.columns:
        if column == "stage":
            continue
        numeric = pd.to_numeric(scaled[column], errors="coerce")
        scaled[column] = numeric.mul(100).round(2)
    return scaled


def format_frame_for_csv(df: pd.DataFrame) -> pd.DataFrame:
    formatted = df.copy()
    for column in formatted.columns:
        if column == "stage":
            continue
        formatted[column] = formatted[column].apply(lambda value: "" if pd.isna(value) else f"{float(value):.2f}")
    return formatted


def aggregate_values(values: list[float], fixed_n: int | None) -> float | pd.NA:
    if fixed_n is not None:
        return float(sum(values) / fixed_n) if values else 0.0
    return float(pd.Series(values).mean()) if values else pd.NA


def collect_stage_metric_views() -> tuple[dict[str, dict[str, dict[str, dict[str, list[float]]]]], dict[str, dict[str, dict[str, int]]]]:
    metric_views: dict[str, dict[str, dict[str, dict[str, list[float]]]]] = {
        category: {
            stage: {
                metric: {"hw": [], "llm": [], "delta": []}
                for metric in RISK_METRICS
            }
            for stage in STAGE_ORDER
        }
        for category in CATEGORY_CONFIGS
    }
    counts: dict[str, dict[str, dict[str, int]]] = {
        category: {stage: {metric: 0 for metric in RISK_METRICS} for stage in STAGE_ORDER}
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
                for metric_name in RISK_METRICS:
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
            for metric_name in RISK_METRICS:
                for view_name in ["hw", "llm", "delta"]:
                    values = metric_views[category_name][stage_name][metric_name][view_name]
                    row[f"{prefix}_{metric_name}_{view_name}"] = aggregate_values(values, fixed_n)
        rows.append(row)

    columns = ["stage"]
    for prefix in ["academic", "industrial"]:
        for metric_name in RISK_METRICS:
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
        '      <th colspan="6">Academic</th>',
        '      <th colspan="6">Industrial</th>',
        "    </tr>",
        "    <tr>",
        '      <th colspan="3">FPR</th>',
        '      <th colspan="3">FNR</th>',
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
            "academic_fpr_hw",
            "academic_fpr_llm",
            "academic_fpr_delta",
            "academic_fnr_hw",
            "academic_fnr_llm",
            "academic_fnr_delta",
            "industrial_fpr_hw",
            "industrial_fpr_llm",
            "industrial_fpr_delta",
            "industrial_fnr_hw",
            "industrial_fnr_llm",
            "industrial_fnr_delta",
        ]:
            lines.append(f"      <td>{format_value(row[column])}</td>")
        lines.append("    </tr>")
    lines.extend(["  </tbody>", "</table>"])
    return lines


def build_summary_lines(df: pd.DataFrame, counts: dict[str, dict[str, dict[str, int]]]) -> list[str]:
    lines: list[str] = []
    for category_name, config in CATEGORY_CONFIGS.items():
        label = config["label"]
        for metric_name in RISK_METRICS:
            column_name = (
                f"academic_{metric_name}_delta"
                if category_name == "academic"
                else f"industrial_{metric_name}_delta"
            )
            valid = df[["stage", column_name]].dropna()
            if valid.empty:
                continue
            worst_row = valid.loc[valid[column_name].idxmax()]
            fixed_n = config["fixed_n"]
            worst_n = fixed_n if fixed_n is not None else counts[category_name][str(worst_row["stage"])][metric_name]
            lines.append(
                f"- `{label}` largest `Δ {metric_name.upper()}`: "
                f"`{worst_row['stage']}` (`{format_value(worst_row[column_name])}`, `n = {worst_n}`)."
            )
    return lines


def write_readme(df: pd.DataFrame, counts: dict[str, dict[str, dict[str, int]]]) -> None:
    lines = [
        "# Stage-Level FPR/FNR Risk Trend",
        "",
        "This table shows how stage difficulty changes detector risk behavior.",
        "",
        "- All rates are reported as percentages and rounded to two decimals.",
        "- `Δ = LLM-stage - HW-overall`, expressed in percentage points.",
        "- Positive `Δ FPR` means more false positives on that LLM stage.",
        "- Positive `Δ FNR` means more false negatives on that LLM stage.",
        "- `Academic` cells use the mean over available paired detectors for that stage/metric.",
        "- `Industrial` cells currently use a fixed denominator of `n = 3` for every stage/metric.",
        "",
    ]
    lines.extend(build_summary_lines(df, counts))
    lines.extend([""])
    lines.extend(render_table_html(df))
    lines.append("")
    lines.append("CSV export: `stage_fpr_fnr_trend.csv`.")
    README_OUTPUT_PATH.write_text("\n".join(lines), encoding="utf-8")


def main() -> int:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    df, counts = build_table_frame()
    df = scale_rate_columns(df)
    format_frame_for_csv(df).to_csv(CSV_OUTPUT_PATH, index=False)
    write_readme(df, counts)
    print(f"[OK] CSV -> {CSV_OUTPUT_PATH}")
    print(f"[OK] README -> {README_OUTPUT_PATH}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
