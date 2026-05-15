#!/usr/bin/env python3
"""Compute detector metrics and render the stage-transfer trend README.

The script reads a manifest that points to local detector output CSVs. It does
not require benchmark data to be committed to the repository.
"""

from __future__ import annotations

import argparse
import csv
import math
import re
from collections import defaultdict
from pathlib import Path
from typing import Any


SCRIPT_DIR = Path(__file__).resolve().parent
DEFAULT_MANIFEST = SCRIPT_DIR / "examples" / "stage_transfer_manifest.csv"
DEFAULT_OUTPUT_DIR = SCRIPT_DIR / "stage-transfer-trend"
METRICS = ["precision", "recall", "accuracy", "fpr", "fnr", "f0_5", "f1", "f2", "mcc"]
TREND_METRICS = ["recall", "f2", "mcc"]
DEFAULT_STAGE_ORDER = [
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

NULL_TOKENS = {"", "nan", "none", "null", "na", "n/a", "-"}
POSITIVE_TOKENS = {
    "1",
    "1.0",
    "true",
    "t",
    "yes",
    "y",
    "positive",
    "phishing",
    "malicious",
    "spam",
    "unsafe",
    "fraud",
    "flagged",
    "block",
    "blocked",
}
NEGATIVE_TOKENS = {
    "0",
    "0.0",
    "false",
    "f",
    "no",
    "n",
    "negative",
    "benign",
    "legitimate",
    "ham",
    "safe",
    "clean",
    "not flagged",
    "allow",
    "allowed",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Compute detector metrics from local detector result CSVs and export "
            "a stage-transfer trend table plus a Markdown README."
        )
    )
    parser.add_argument(
        "--manifest",
        type=Path,
        default=DEFAULT_MANIFEST,
        help=(
            "CSV manifest with columns: dataset_family, detector_family, stage, path, "
            "detector, prediction_column, label_column."
        ),
    )
    parser.add_argument(
        "--base-dir",
        type=Path,
        default=None,
        help="Base directory for relative paths in the manifest. Defaults to the manifest directory.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
        help="Output directory for metrics CSVs and README.",
    )
    parser.add_argument(
        "--stage-order",
        nargs="*",
        default=list(DEFAULT_STAGE_ORDER),
        help="Optional explicit stage order for the trend table.",
    )
    return parser.parse_args()


def normalize_binary_value(value: Any) -> int | None:
    if value is None:
        return None
    text = str(value).strip()
    lowered = text.lower()
    if lowered in NULL_TOKENS:
        return None
    if lowered in POSITIVE_TOKENS:
        return 1
    if lowered in NEGATIVE_TOKENS:
        return 0
    try:
        numeric = float(text)
    except ValueError:
        return None
    if numeric == 1.0:
        return 1
    if numeric == 0.0:
        return 0
    return None


def resolve_column(fieldnames: list[str], preferred: str) -> str | None:
    lookup = {name.lower(): name for name in fieldnames}
    return lookup.get(preferred.lower())


def detector_name_from_column(column: str) -> str:
    if column == "model_prediction":
        return "model"
    return re.sub(r"_prediction$", "", column)


def prediction_columns(fieldnames: list[str], requested: str, detector: str) -> list[tuple[str, str]]:
    if requested:
        column = resolve_column(fieldnames, requested)
        if column is None:
            return []
        return [(detector or detector_name_from_column(column), column)]

    columns = [name for name in fieldnames if name.lower().endswith("_prediction")]
    if "model_prediction" in fieldnames and detector:
        columns.append("model_prediction")
    return [(detector or detector_name_from_column(column), column) for column in columns]


def compute_metrics(y_true: list[int], y_pred: list[int]) -> dict[str, float]:
    tp = sum(1 for t, p in zip(y_true, y_pred) if t == 1 and p == 1)
    tn = sum(1 for t, p in zip(y_true, y_pred) if t == 0 and p == 0)
    fp = sum(1 for t, p in zip(y_true, y_pred) if t == 0 and p == 1)
    fn = sum(1 for t, p in zip(y_true, y_pred) if t == 1 and p == 0)

    precision = tp / (tp + fp) if (tp + fp) else 0.0
    recall = tp / (tp + fn) if (tp + fn) else 0.0
    accuracy = (tp + tn) / len(y_true) if y_true else 0.0
    fpr = fp / (fp + tn) if (fp + tn) else 0.0
    fnr = fn / (fn + tp) if (fn + tp) else 0.0
    f1 = (2 * precision * recall / (precision + recall)) if (precision + recall) else 0.0

    def fbeta(beta: float) -> float:
        beta_sq = beta * beta
        denom = beta_sq * precision + recall
        return ((1 + beta_sq) * precision * recall / denom) if denom else 0.0

    mcc_denom = math.sqrt((tp + fp) * (tp + fn) * (tn + fp) * (tn + fn))
    mcc = ((tp * tn - fp * fn) / mcc_denom) if mcc_denom else 0.0

    return {
        "precision": precision,
        "recall": recall,
        "accuracy": accuracy,
        "fpr": fpr,
        "fnr": fnr,
        "f0_5": fbeta(0.5),
        "f1": f1,
        "f2": fbeta(2.0),
        "mcc": mcc,
        "tp": float(tp),
        "tn": float(tn),
        "fp": float(fp),
        "fn": float(fn),
    }


def read_manifest(manifest_path: Path) -> list[dict[str, str]]:
    with manifest_path.open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        if not reader.fieldnames:
            raise SystemExit(f"Manifest has no header row: {manifest_path}")
        return [dict(row) for row in reader]


def load_detector_metrics(
    manifest_rows: list[dict[str, str]],
    *,
    base_dir: Path,
) -> list[dict[str, Any]]:
    metric_rows: list[dict[str, Any]] = []

    for manifest_row in manifest_rows:
        dataset_family = manifest_row.get("dataset_family", "").strip().lower()
        detector_family = manifest_row.get("detector_family", "").strip().lower()
        stage = manifest_row.get("stage", "").strip()
        raw_path = manifest_row.get("path", "").strip()
        detector = manifest_row.get("detector", "").strip()
        requested_prediction = manifest_row.get("prediction_column", "").strip()
        label_name = manifest_row.get("label_column", "").strip() or "label"

        if dataset_family not in {"hw", "llm"}:
            raise SystemExit(f"dataset_family must be hw or llm: {manifest_row}")
        if detector_family not in {"academic", "industry"}:
            raise SystemExit(f"detector_family must be academic or industry: {manifest_row}")
        if not stage or not raw_path:
            raise SystemExit(f"Manifest row is missing stage or path: {manifest_row}")

        csv_path = Path(raw_path)
        if not csv_path.is_absolute():
            csv_path = base_dir / csv_path
        if not csv_path.exists():
            raise SystemExit(f"Detector result CSV not found: {csv_path}")

        with csv_path.open("r", encoding="utf-8-sig", newline="") as handle:
            reader = csv.DictReader(handle)
            if not reader.fieldnames:
                raise SystemExit(f"Detector result CSV has no header row: {csv_path}")
            label_column = resolve_column(reader.fieldnames, label_name)
            if label_column is None:
                raise SystemExit(f"Missing label column '{label_name}' in {csv_path}")
            prediction_specs = prediction_columns(reader.fieldnames, requested_prediction, detector)
            if not prediction_specs:
                raise SystemExit(f"No prediction columns found for manifest row: {manifest_row}")

            buffers = {
                detector_name: {"y_true": [], "y_pred": []}
                for detector_name, _column in prediction_specs
            }
            for row in reader:
                label = normalize_binary_value(row.get(label_column))
                if label is None:
                    continue
                for detector_name, prediction_column in prediction_specs:
                    pred = normalize_binary_value(row.get(prediction_column))
                    if pred is None:
                        continue
                    buffers[detector_name]["y_true"].append(label)
                    buffers[detector_name]["y_pred"].append(pred)

        for detector_name, values in buffers.items():
            y_true = values["y_true"]
            y_pred = values["y_pred"]
            if not y_true:
                continue
            metrics = compute_metrics(y_true, y_pred)
            metric_rows.append(
                {
                    "dataset_family": dataset_family,
                    "detector_family": detector_family,
                    "stage": stage,
                    "detector": detector_name,
                    "n_rows": len(y_true),
                    **metrics,
                }
            )

    return metric_rows


def weighted_average_metric(rows: list[dict[str, Any]], metric: str) -> float | None:
    total_n = sum(int(row["n_rows"]) for row in rows)
    if total_n == 0:
        return None
    return sum(float(row[metric]) * int(row["n_rows"]) for row in rows) / total_n


def build_hw_baselines(metric_rows: list[dict[str, Any]]) -> dict[tuple[str, str], dict[str, float]]:
    grouped: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    for row in metric_rows:
        if row["dataset_family"] == "hw":
            grouped[(row["detector_family"], row["detector"])].append(row)

    baselines: dict[tuple[str, str], dict[str, float]] = {}
    for key, rows in grouped.items():
        baseline = {"n_rows": float(sum(int(row["n_rows"]) for row in rows))}
        for metric in METRICS:
            value = weighted_average_metric(rows, metric)
            baseline[metric] = float("nan") if value is None else value
        baselines[key] = baseline
    return baselines


def mean(values: list[float]) -> float | None:
    values = [value for value in values if not math.isnan(value)]
    if not values:
        return None
    return sum(values) / len(values)


def stage_sort_key(stage: str, stage_order: list[str]) -> tuple[int, str]:
    if stage in stage_order:
        return (stage_order.index(stage), stage)
    match = re.match(r"^S(\d+)", stage)
    number = int(match.group(1)) if match else 10**9
    return (len(stage_order) + number, stage)


def build_stage_trend(
    metric_rows: list[dict[str, Any]],
    baselines: dict[tuple[str, str], dict[str, float]],
    stage_order: list[str],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    llm_rows = [row for row in metric_rows if row["dataset_family"] == "llm"]
    stages = sorted({row["stage"] for row in llm_rows}, key=lambda value: stage_sort_key(value, stage_order))
    detector_families = ["academic", "industry"]

    long_rows: list[dict[str, Any]] = []
    wide_rows: list[dict[str, Any]] = []
    for stage in stages:
        wide_row: dict[str, Any] = {"stage": stage}
        for detector_family in detector_families:
            family_stage_rows = [
                row for row in llm_rows
                if row["stage"] == stage and row["detector_family"] == detector_family
            ]
            for metric in TREND_METRICS:
                hw_values: list[float] = []
                llm_values: list[float] = []
                delta_values: list[float] = []
                detectors: list[str] = []
                for row in family_stage_rows:
                    baseline = baselines.get((detector_family, row["detector"]))
                    if baseline is None:
                        continue
                    hw_value = float(baseline[metric])
                    llm_value = float(row[metric])
                    if math.isnan(hw_value) or math.isnan(llm_value):
                        continue
                    hw_values.append(hw_value)
                    llm_values.append(llm_value)
                    delta_values.append(llm_value - hw_value)
                    detectors.append(row["detector"])

                prefix = "industrial" if detector_family == "industry" else "academic"
                hw_mean = mean(hw_values)
                llm_mean = mean(llm_values)
                delta_mean = mean(delta_values)
                wide_row[f"{prefix}_{metric}_hw"] = hw_mean
                wide_row[f"{prefix}_{metric}_llm"] = llm_mean
                wide_row[f"{prefix}_{metric}_delta"] = delta_mean
                wide_row[f"{prefix}_{metric}_n_detectors"] = len(set(detectors))
                long_rows.append(
                    {
                        "stage": stage,
                        "detector_family": detector_family,
                        "metric": metric,
                        "hw_overall_mean": hw_mean,
                        "llm_stage_mean": llm_mean,
                        "delta": delta_mean,
                        "n_detectors": len(set(detectors)),
                    }
                )
        wide_rows.append(wide_row)
    return long_rows, wide_rows


def format_value(value: Any) -> str:
    if value is None:
        return "-"
    try:
        numeric = float(value)
    except (TypeError, ValueError):
        return str(value)
    if math.isnan(numeric):
        return "-"
    return f"{numeric:.4f}"


def write_csv(path: Path, rows: list[dict[str, Any]], fieldnames: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({field: row.get(field, "") for field in fieldnames})


def render_html_table(wide_rows: list[dict[str, Any]]) -> list[str]:
    columns = []
    for prefix in ["academic", "industrial"]:
        for metric in TREND_METRICS:
            columns.extend([
                f"{prefix}_{metric}_hw",
                f"{prefix}_{metric}_llm",
                f"{prefix}_{metric}_delta",
                f"{prefix}_{metric}_n_detectors",
            ])

    lines = [
        "<table>",
        "  <thead>",
        "    <tr>",
        '      <th rowspan="2">Stage</th>',
        '      <th colspan="12">Academic</th>',
        '      <th colspan="12">Industrial</th>',
        "    </tr>",
        "    <tr>",
    ]
    for _family in ["academic", "industrial"]:
        for metric in TREND_METRICS:
            lines.extend([
                f"      <th>{metric} HW</th>",
                f"      <th>{metric} LLM</th>",
                f"      <th>{metric} delta</th>",
                f"      <th>{metric} n</th>",
            ])
    lines.extend(["    </tr>", "  </thead>", "  <tbody>"])
    for row in wide_rows:
        lines.append("    <tr>")
        lines.append(f"      <td>{row['stage']}</td>")
        for column in columns:
            value = row.get(column)
            if column.endswith("_n_detectors"):
                value = "" if value in (None, "") else int(value)
            else:
                value = format_value(value)
            lines.append(f"      <td>{value}</td>")
        lines.append("    </tr>")
    lines.extend(["  </tbody>", "</table>"])
    return lines


def build_summary_lines(wide_rows: list[dict[str, Any]]) -> list[str]:
    lines: list[str] = []
    for prefix, label in [("academic", "Academic"), ("industrial", "Industrial")]:
        column = f"{prefix}_mcc_delta"
        valid = [
            (row["stage"], float(row[column]))
            for row in wide_rows
            if row.get(column) is not None and not math.isnan(float(row[column]))
        ]
        if not valid:
            continue
        hardest = min(valid, key=lambda item: item[1])
        easiest = max(valid, key=lambda item: item[1])
        lines.append(f"- `{label}` hardest stage by mean `delta MCC`: `{hardest[0]}` (`{hardest[1]:.4f}`).")
        lines.append(f"- `{label}` easiest stage by mean `delta MCC`: `{easiest[0]}` (`{easiest[1]:.4f}`).")
    return lines


def write_readme(path: Path, wide_rows: list[dict[str, Any]], manifest_display: str) -> None:
    lines = [
        "# Stage Transfer Trend",
        "",
        "This README is generated by `../compute_stage_transfer_trend.py`.",
        "",
        "The table compares LLM-stage detector performance against each detector's own HW overall baseline.",
        "The checked-in public README is generated from toy example data; use a local manifest for benchmark values.",
        "",
        "- `delta = LLM-stage metric - HW-overall metric`",
        "- Negative `delta` means the LLM stage is harder for that detector family than its HW baseline.",
        "- `n` is the number of detectors with both HW baseline and LLM-stage values.",
        "",
        "Generated from manifest:",
        "",
        f"```text\n{manifest_display}\n```",
        "",
    ]
    lines.extend(build_summary_lines(wide_rows))
    lines.append("")
    lines.extend(render_html_table(wide_rows))
    lines.extend([
        "",
        "Generated files:",
        "",
        "- `per_stage_metrics.csv`",
        "- `detector_hw_baselines.csv`",
        "- `stage_transfer_trend_long.csv`",
        "- `table_2_stage_transfer_trend.csv`",
    ])
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> int:
    args = parse_args()
    manifest_path = args.manifest.resolve()
    base_dir = (args.base_dir or manifest_path.parent).resolve()
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    manifest_rows = read_manifest(manifest_path)
    metric_rows = load_detector_metrics(manifest_rows, base_dir=base_dir)
    baselines = build_hw_baselines(metric_rows)
    baseline_rows = [
        {"detector_family": key[0], "detector": key[1], **value}
        for key, value in sorted(baselines.items())
    ]
    trend_long_rows, trend_wide_rows = build_stage_trend(metric_rows, baselines, list(args.stage_order))

    metric_fieldnames = ["dataset_family", "detector_family", "stage", "detector", "n_rows", *METRICS, "tp", "tn", "fp", "fn"]
    baseline_fieldnames = ["detector_family", "detector", "n_rows", *METRICS]
    long_fieldnames = ["stage", "detector_family", "metric", "hw_overall_mean", "llm_stage_mean", "delta", "n_detectors"]
    wide_fieldnames = ["stage"]
    for prefix in ["academic", "industrial"]:
        for metric in TREND_METRICS:
            wide_fieldnames.extend([
                f"{prefix}_{metric}_hw",
                f"{prefix}_{metric}_llm",
                f"{prefix}_{metric}_delta",
                f"{prefix}_{metric}_n_detectors",
            ])

    write_csv(output_dir / "per_stage_metrics.csv", metric_rows, metric_fieldnames)
    write_csv(output_dir / "detector_hw_baselines.csv", baseline_rows, baseline_fieldnames)
    write_csv(output_dir / "stage_transfer_trend_long.csv", trend_long_rows, long_fieldnames)
    write_csv(output_dir / "table_2_stage_transfer_trend.csv", trend_wide_rows, wide_fieldnames)
    write_readme(output_dir / "README.md", trend_wide_rows, str(args.manifest))

    print(f"[OK] metrics -> {output_dir / 'per_stage_metrics.csv'}")
    print(f"[OK] trend -> {output_dir / 'table_2_stage_transfer_trend.csv'}")
    print(f"[OK] README -> {output_dir / 'README.md'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
