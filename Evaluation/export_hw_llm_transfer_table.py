#!/usr/bin/env python3
"""Export Table 1: overall HW vs LLM transfer summary by detector."""

from __future__ import annotations

from collections import OrderedDict
from pathlib import Path

import pandas as pd

from export_detector_stage_tables import (
    ACADEMIC_DETECTORS,
    ACADEMIC_HW_DIR,
    ACADEMIC_LLM_DIR,
    INDUSTRY_DETECTORS,
    INDUSTRY_HW_DIR,
    INDUSTRY_LLM_DIR,
    compute_metric_bundle,
    normalize_binary_value,
    parse_stage_and_detector,
    resolve_column,
)


SCRIPT_DIR = Path(__file__).resolve().parent
OUTPUT_DIR = SCRIPT_DIR / "hw-vs-llm-overall-transfer"
TABLES_DIR = SCRIPT_DIR / "detector-stage-metric-tables"
CSV_OUTPUT_PATH = OUTPUT_DIR / "table_1_hw_vs_llm_transfer.csv"
README_OUTPUT_PATH = OUTPUT_DIR / "README.md"
TRANSFER_METRICS = ["recall", "f2", "mcc"]


def to_binary_series(series: pd.Series) -> pd.Series:
    return pd.Series(series.map(normalize_binary_value), index=series.index, dtype="float")


def format_value(value: object) -> str:
    if pd.isna(value):
        return "-"
    return f"{float(value):.4f}"


def format_value_with_std(value: object, std: object) -> str:
    if pd.isna(value):
        return "-"
    if pd.isna(std):
        return f"{float(value):.4f}"
    return f"{float(value):.4f} ± {float(std):.4f}"


def compute_std(series: pd.Series) -> float | pd.NA:
    numeric = pd.to_numeric(series, errors="coerce").dropna()
    if numeric.empty:
        return pd.NA
    if len(numeric) == 1:
        return 0.0
    return float(numeric.std(ddof=1))


def collect_stage_metric_stds(category_name: str) -> dict[str, dict[str, float | pd.NA]]:
    category_dir = TABLES_DIR / category_name
    results: dict[str, dict[str, float | pd.NA]] = {}
    for csv_path in sorted(category_dir.glob("*.csv")):
        df = pd.read_csv(csv_path)
        detector_name = csv_path.stem
        detector_stats: dict[str, float | pd.NA] = {}
        for metric_name in TRANSFER_METRICS:
            detector_stats[f"hw_{metric_name}_std"] = compute_std(df[f"hw_{metric_name}"])
            detector_stats[f"llm_{metric_name}_std"] = compute_std(df[f"llm_{metric_name}"])
            delta_series = pd.to_numeric(df[f"llm_{metric_name}"], errors="coerce") - pd.to_numeric(
                df[f"hw_{metric_name}"],
                errors="coerce",
            )
            detector_stats[f"delta_{metric_name}_std"] = compute_std(delta_series)
        results[detector_name] = detector_stats
    return results


def collect_overall_metric_bundles(
    *,
    source_dir: Path,
    category_name: str,
    source_kind: str,
    detector_configs: OrderedDict[str, dict[str, str | None]],
) -> dict[str, dict[str, float] | None]:
    accumulators: dict[str, dict[str, list[int]]] = {
        detector_name: {"y_true": [], "y_pred": []} for detector_name in detector_configs
    }

    if not source_dir.exists():
        return {detector_name: None for detector_name in detector_configs}

    for csv_path in sorted(source_dir.glob("*.csv")):
        parsed = parse_stage_and_detector(csv_path, source_kind, category_name)
        if parsed is None:
            continue

        _stage_name, standalone_detector = parsed

        df = pd.read_csv(csv_path, dtype=str, keep_default_na=False)
        label_column = resolve_column(df.columns, "label")
        if label_column is None:
            continue

        labels = to_binary_series(df[label_column])
        if standalone_detector is not None:
            if standalone_detector not in detector_configs:
                continue
            prediction_column = detector_configs[standalone_detector].get("standalone_column")
            if not prediction_column:
                continue
            pred_column = resolve_column(df.columns, prediction_column)
            if pred_column is None:
                continue
            preds = to_binary_series(df[pred_column])
            valid = labels.notna() & preds.notna()
            if not valid.any():
                continue
            accumulators[standalone_detector]["y_true"].extend(labels[valid].astype(int).tolist())
            accumulators[standalone_detector]["y_pred"].extend(preds[valid].astype(int).tolist())
            continue

        for detector_name, detector_config in detector_configs.items():
            prediction_column = detector_config.get("combined_column")
            if not prediction_column:
                continue
            pred_column = resolve_column(df.columns, prediction_column)
            if pred_column is None:
                continue
            preds = to_binary_series(df[pred_column])
            valid = labels.notna() & preds.notna()
            if not valid.any():
                continue
            accumulators[detector_name]["y_true"].extend(labels[valid].astype(int).tolist())
            accumulators[detector_name]["y_pred"].extend(preds[valid].astype(int).tolist())

    results: dict[str, dict[str, float] | None] = {}
    for detector_name, values in accumulators.items():
        if not values["y_true"]:
            results[detector_name] = None
            continue
        results[detector_name] = compute_metric_bundle(
            pd.Series(values["y_true"]),
            pd.Series(values["y_pred"]),
        )
    return results


def build_detector_rows(
    *,
    group_label: str,
    category_name: str,
    detector_configs: OrderedDict[str, dict[str, str | None]],
    hw_dir: Path,
    llm_dir: Path,
) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    metric_columns = [
        "hw_recall",
        "llm_recall",
        "delta_recall",
        "hw_f2",
        "llm_f2",
        "delta_f2",
        "hw_mcc",
        "llm_mcc",
        "delta_mcc",
    ]
    std_columns = [
        "hw_recall_std",
        "llm_recall_std",
        "delta_recall_std",
        "hw_f2_std",
        "llm_f2_std",
        "delta_f2_std",
        "hw_mcc_std",
        "llm_mcc_std",
        "delta_mcc_std",
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
    stage_stds = collect_stage_metric_stds(category_name)

    for detector_name, detector_config in detector_configs.items():
        _ = detector_config
        hw_metrics = hw_bundles.get(detector_name)
        llm_metrics = llm_bundles.get(detector_name)
        detector_stds = stage_stds.get(detector_name, {})

        row: dict[str, object] = {
            "group": group_label,
            "detector": detector_name,
        }

        for metric_name in TRANSFER_METRICS:
            hw_value = pd.NA if hw_metrics is None else hw_metrics.get(metric_name, pd.NA)
            llm_value = pd.NA if llm_metrics is None else llm_metrics.get(metric_name, pd.NA)
            delta_value = pd.NA
            if not pd.isna(hw_value) and not pd.isna(llm_value):
                delta_value = float(llm_value) - float(hw_value)

            row[f"hw_{metric_name}"] = hw_value
            row[f"llm_{metric_name}"] = llm_value
            row[f"delta_{metric_name}"] = delta_value
            row[f"hw_{metric_name}_std"] = detector_stds.get(f"hw_{metric_name}_std", pd.NA)
            row[f"llm_{metric_name}_std"] = detector_stds.get(f"llm_{metric_name}_std", pd.NA)
            row[f"delta_{metric_name}_std"] = detector_stds.get(f"delta_{metric_name}_std", pd.NA)

        rows.append(row)

    detector_frame = pd.DataFrame(rows)
    mean_row: dict[str, object] = {"group": group_label, "detector": f"{group_label} mean"}
    median_row: dict[str, object] = {"group": group_label, "detector": f"{group_label} median"}

    for column in metric_columns:
        numeric = pd.to_numeric(detector_frame[column], errors="coerce")
        mean_row[column] = float(numeric.mean()) if numeric.notna().any() else pd.NA
        median_row[column] = float(numeric.median()) if numeric.notna().any() else pd.NA
    for column in std_columns:
        mean_row[column] = pd.NA
        median_row[column] = pd.NA

    rows.append(mean_row)
    rows.append(median_row)
    return rows


def build_table_frame() -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    rows.extend(
        build_detector_rows(
            group_label="Academic",
            category_name="academic",
            detector_configs=ACADEMIC_DETECTORS,
            hw_dir=ACADEMIC_HW_DIR,
            llm_dir=ACADEMIC_LLM_DIR,
        )
    )
    rows.extend(
        build_detector_rows(
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
        "hw_recall",
        "hw_recall_std",
        "llm_recall",
        "llm_recall_std",
        "delta_recall",
        "delta_recall_std",
        "hw_f2",
        "hw_f2_std",
        "llm_f2",
        "llm_f2_std",
        "delta_f2",
        "delta_f2_std",
        "hw_mcc",
        "hw_mcc_std",
        "llm_mcc",
        "llm_mcc_std",
        "delta_mcc",
        "delta_mcc_std",
    ]
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
        is_summary = detector_name.endswith(" mean") or detector_name.endswith(" median")
        lines.append("    <tr>")
        if row["group"] != previous_group:
            lines.append(f'      <td><strong>{row["group"]}</strong></td>')
            previous_group = row["group"]
        else:
            lines.append("      <td></td>")

        detector_cell = f"<strong>{detector_name}</strong>" if is_summary else detector_name
        lines.append(f"      <td>{detector_cell}</td>")
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
            lines.append(f"      <td>{format_value_with_std(row[column], row.get(f'{column}_std', pd.NA))}</td>")
        lines.append("    </tr>")

    lines.extend(["  </tbody>", "</table>"])
    return lines


def build_summary_lines(df: pd.DataFrame) -> list[str]:
    summary_lines: list[str] = []
    for group_name in ["Academic", "Industrial"]:
        group_rows = df[df["group"] == group_name].copy()
        group_rows = group_rows[
            ~group_rows["detector"].astype(str).str.endswith(" mean")
            & ~group_rows["detector"].astype(str).str.endswith(" median")
        ]
        paired_count = int(pd.to_numeric(group_rows["delta_mcc"], errors="coerce").notna().sum())
        mean_delta_recall = pd.to_numeric(group_rows["delta_recall"], errors="coerce").mean()
        mean_delta_f2 = pd.to_numeric(group_rows["delta_f2"], errors="coerce").mean()
        mean_delta_mcc = pd.to_numeric(group_rows["delta_mcc"], errors="coerce").mean()
        summary_lines.append(
            f"- `{group_name}` average transfer gap: "
            f"`n = {paired_count}` paired detectors, "
            f"`Δ Recall = {format_value(mean_delta_recall)}`, "
            f"`Δ F2 = {format_value(mean_delta_f2)}`, "
            f"`Δ MCC = {format_value(mean_delta_mcc)}`."
        )
    return summary_lines


def write_readme(df: pd.DataFrame) -> None:
    lines = [
        "# Table 1: Overall HW vs. LLM Transfer",
        "",
        "This is the main body table for comparing how each detector transfers from",
        "`HW / GD` inputs to `LLM-generated` inputs.",
        "",
        "Interpretation notes:",
        "",
        "- Each `HW` value is recomputed once over all available HW / GD samples for that detector.",
        "- Each `LLM` value is recomputed once over all available LLM-generated samples for that detector.",
        "- `Δ = LLM - HW`, so a negative value means the detector gets worse on LLM-generated content.",
        "- For detector rows, `± std` is the standard deviation across available stage-level metrics.",
        "- `mean` and `median` are computed within each detector family over detectors with available values.",
        "- Blank cells mean the detector currently lacks a usable `HW`, `LLM`, or paired `HW -> LLM` overall result.",
        "",
    ]
    lines.extend(build_summary_lines(df))
    lines.extend([""])
    lines.extend(render_table_html(df))
    lines.append("")
    lines.append("CSV export: `table_1_hw_vs_llm_transfer.csv`.")
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
