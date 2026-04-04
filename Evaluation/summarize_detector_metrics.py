#!/usr/bin/env python3
"""Compute per-stage and overall detector metrics from processed evaluation CSVs."""

from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    confusion_matrix,
    f1_score,
    fbeta_score,
    matthews_corrcoef,
    precision_score,
    recall_score,
)

SCRIPT_DIR = Path(__file__).resolve().parent
DEFAULT_INPUT_DIR = SCRIPT_DIR / "processed-evaluation-datasets"
DEFAULT_OUTPUT_DIR = SCRIPT_DIR / "processed-evaluation-metrics"

STAGE_CSV_RE = re.compile(r"^(S\d+(?:-[A-Za-z0-9]+)*)\.csv$")

DETECTOR_COLUMNS = [
    "llm_guard_prediction",
    "phishing_email_agent_prediction",
    "email_phishing_detection_v3_prediction",
    "pyrit_original_prediction",
    "pyrit_blocklist_prediction",
]
METRIC_COLUMNS = [
    "mcc",
    "f1",
    "fpr",
    "fnr",
    "recall",
    "precision",
    "f0_5",
    "f2",
    "accuracy",
    "pr_auc",
]

NULL_TOKENS = {"", "nan", "none", "null", "na", "n/a"}
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
        description="Compute detector metrics and print overall evaluation summary."
    )
    parser.add_argument(
        "--input-dir",
        type=Path,
        default=DEFAULT_INPUT_DIR,
        help=f"Directory containing processed stage CSVs (default: {DEFAULT_INPUT_DIR})",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
        help=f"Directory where metric CSVs are written (default: {DEFAULT_OUTPUT_DIR})",
    )
    return parser.parse_args()


def normalize_binary_value(value: object) -> float | None:
    if value is None:
        return None

    text = str(value).strip()
    lowered = text.lower()
    if lowered in NULL_TOKENS:
        return None
    if lowered in POSITIVE_TOKENS:
        return 1.0
    if lowered in NEGATIVE_TOKENS:
        return 0.0

    try:
        numeric = float(text)
    except ValueError:
        return None

    if numeric == 1.0:
        return 1.0
    if numeric == 0.0:
        return 0.0
    return None


def stage_sort_key(path: Path) -> tuple[int, str]:
    match = STAGE_CSV_RE.match(path.name)
    if not match:
        return (sys.maxsize, path.name)
    stage_name = match.group(1)
    number_match = re.match(r"^S(\d+)", stage_name)
    stage_number = int(number_match.group(1)) if number_match else sys.maxsize
    return (stage_number, stage_name)


def find_stage_files(input_dir: Path) -> list[Path]:
    files = [path for path in input_dir.glob("*.csv") if STAGE_CSV_RE.match(path.name)]
    return sorted(files, key=stage_sort_key)


def to_binary_series(series: pd.Series) -> pd.Series:
    normalized = series.map(normalize_binary_value)
    return pd.Series(normalized, index=series.index, dtype="float")


def resolve_column(columns: Iterable[str], preferred_name: str) -> str | None:
    lower_lookup = {column.lower(): column for column in columns}
    return lower_lookup.get(preferred_name.lower())


def compute_metric_bundle(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, float]:
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()

    metrics = {
        "mcc": float(matthews_corrcoef(y_true, y_pred)),
        "f1": float(f1_score(y_true, y_pred, zero_division=0)),
        "fpr": float(fp / (fp + tn)) if (fp + tn) else 0.0,
        "fnr": float(fn / (fn + tp)) if (fn + tp) else 0.0,
        "recall": float(recall_score(y_true, y_pred, zero_division=0)),
        "precision": float(precision_score(y_true, y_pred, zero_division=0)),
        "f0_5": float(fbeta_score(y_true, y_pred, beta=0.5, zero_division=0)),
        "f2": float(fbeta_score(y_true, y_pred, beta=2.0, zero_division=0)),
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "pr_auc": float(average_precision_score(y_true, y_pred))
        if len(np.unique(y_true)) > 1
        else float("nan"),
    }
    return metrics


def collect_stage_metrics(
    stage_path: Path,
) -> tuple[list[dict[str, object]], dict[str, dict[str, object]]]:
    df = pd.read_csv(stage_path, dtype=str, keep_default_na=False)

    label_column = resolve_column(df.columns, "label")
    if label_column is None:
        raise ValueError(f"'label' column is missing in {stage_path}")

    normalized_label = to_binary_series(df[label_column])
    stage_name = stage_path.stem
    stage_rows: list[dict[str, object]] = []
    detector_payloads: dict[str, dict[str, object]] = {}

    for detector in DETECTOR_COLUMNS:
        detector_column = resolve_column(df.columns, detector)
        if detector_column is None:
            detector_series = pd.Series([None] * len(df), index=df.index, dtype="float")
        else:
            detector_series = to_binary_series(df[detector_column])

        valid_mask = normalized_label.notna() & detector_series.notna()
        valid_count = int(valid_mask.sum())
        if valid_count == 0:
            detector_payloads[detector] = {"stage": stage_name, "y_true": [], "y_pred": []}
            stage_rows.append(
                {
                    "stage": stage_name,
                    "detector": detector,
                    "n_rows": 0,
                    "positive_rate": float("nan"),
                    "predicted_positive_rate": float("nan"),
                    **{metric: float("nan") for metric in METRIC_COLUMNS},
                }
            )
            continue

        y_true = normalized_label[valid_mask].astype(int).to_numpy()
        y_pred = detector_series[valid_mask].astype(int).to_numpy()
        metrics = compute_metric_bundle(y_true, y_pred)
        detector_payloads[detector] = {
            "stage": stage_name,
            "y_true": y_true.tolist(),
            "y_pred": y_pred.tolist(),
        }
        stage_rows.append(
            {
                "stage": stage_name,
                "detector": detector,
                "n_rows": valid_count,
                "positive_rate": float(np.mean(y_true)),
                "predicted_positive_rate": float(np.mean(y_pred)),
                **metrics,
            }
        )

    return stage_rows, detector_payloads


def build_overall_metrics(
    per_stage_df: pd.DataFrame,
    overall_buffers: dict[str, dict[str, object]],
) -> pd.DataFrame:
    rows = []
    for detector in DETECTOR_COLUMNS:
        detector_df = per_stage_df[per_stage_df["detector"] == detector]
        valid_stage_df = detector_df[detector_df["n_rows"] > 0]
        detector_buffer = overall_buffers[detector]

        if valid_stage_df.empty:
            row = {
                "detector": detector,
                "stages_with_predictions": 0,
                "total_rows": 0,
                **{metric: float("nan") for metric in METRIC_COLUMNS},
                **{f"{metric}_mean": float("nan") for metric in METRIC_COLUMNS},
                **{f"{metric}_std": float("nan") for metric in METRIC_COLUMNS},
            }
            rows.append(row)
            continue

        row = {
            "detector": detector,
            "stages_with_predictions": int(valid_stage_df["stage"].nunique()),
            "total_rows": len(detector_buffer["y_true"]),
        }

        y_true = np.asarray(detector_buffer["y_true"], dtype=int)
        y_pred = np.asarray(detector_buffer["y_pred"], dtype=int)
        row.update(compute_metric_bundle(y_true, y_pred))

        for metric in METRIC_COLUMNS:
            metric_values = valid_stage_df[metric].dropna().astype(float)
            row[f"{metric}_mean"] = float(metric_values.mean()) if not metric_values.empty else float("nan")
            row[f"{metric}_std"] = (
                float(metric_values.std(ddof=0)) if not metric_values.empty else float("nan")
            )
        rows.append(row)

    return pd.DataFrame(rows)


def format_metric(value: object) -> str:
    if pd.isna(value):
        return "NA"
    return f"{float(value):.4f}"


def print_summary(per_stage_df: pd.DataFrame, overall_df: pd.DataFrame) -> None:
    total_stages = int(per_stage_df["stage"].nunique())
    total_rows = int(per_stage_df.groupby("stage")["n_rows"].max().sum())
    print(f"Loaded {total_stages} stages with {total_rows} labeled rows in total.")

    valid_overall = overall_df.dropna(subset=["mcc"])
    if not valid_overall.empty:
        best_mcc_row = valid_overall.sort_values("mcc", ascending=False).iloc[0]
        best_f1_row = valid_overall.sort_values("f1", ascending=False).iloc[0]
        print(
            "Overall best by MCC: "
            f"{best_mcc_row['detector']} ({format_metric(best_mcc_row['mcc'])})"
        )
        print(
            "Overall best by F1 : "
            f"{best_f1_row['detector']} ({format_metric(best_f1_row['f1'])})"
        )

    missing_detectors = overall_df[overall_df["stages_with_predictions"] == 0]["detector"].tolist()
    if missing_detectors:
        print("No usable predictions found for: " + ", ".join(missing_detectors))

    print("\nPer-stage metrics:")
    stage_view = per_stage_df[
        ["stage", "detector", "n_rows", "mcc", "f1", "fpr", "fnr", "recall", "precision", "f0_5", "f2", "accuracy", "pr_auc"]
    ].copy()
    print(stage_view.to_string(index=False, float_format=lambda x: f"{x:.4f}"))

    print("\nOverall summary:")
    overall_view = overall_df[
        [
            "detector",
            "stages_with_predictions",
            "total_rows",
            "mcc",
            "f1",
            "fpr",
            "fnr",
            "recall",
            "precision",
            "f0_5",
            "f2",
            "accuracy",
            "pr_auc",
        ]
    ].copy()
    print(overall_view.to_string(index=False, float_format=lambda x: f"{x:.4f}"))

    print("\nMetric std across stages:")
    std_columns = ["detector"] + [f"{metric}_std" for metric in METRIC_COLUMNS]
    std_view = overall_df[std_columns].copy()
    print(std_view.to_string(index=False, float_format=lambda x: f"{x:.4f}"))


def main() -> int:
    args = parse_args()
    input_dir = args.input_dir.resolve()
    output_dir = args.output_dir.resolve()

    if not input_dir.exists():
        print(f"[ERROR] Input directory does not exist: {input_dir}", file=sys.stderr)
        return 1

    stage_files = find_stage_files(input_dir)
    if not stage_files:
        print(f"[ERROR] No processed stage CSVs found in: {input_dir}", file=sys.stderr)
        return 1

    per_stage_rows: list[dict[str, object]] = []
    overall_buffers: dict[str, dict[str, object]] = {
        detector: {"y_true": [], "y_pred": [], "stages": set()} for detector in DETECTOR_COLUMNS
    }
    for stage_path in stage_files:
        stage_rows, detector_payloads = collect_stage_metrics(stage_path)
        per_stage_rows.extend(stage_rows)
        for detector, payload in detector_payloads.items():
            if payload["y_true"]:
                overall_buffers[detector]["y_true"].extend(payload["y_true"])
                overall_buffers[detector]["y_pred"].extend(payload["y_pred"])
                overall_buffers[detector]["stages"].add(payload["stage"])

    per_stage_df = pd.DataFrame(per_stage_rows)
    overall_df = build_overall_metrics(per_stage_df, overall_buffers)

    output_dir.mkdir(parents=True, exist_ok=True)
    per_stage_path = output_dir / "per_stage_metrics.csv"
    overall_path = output_dir / "overall_metrics.csv"
    per_stage_df.to_csv(per_stage_path, index=False)
    overall_df.to_csv(overall_path, index=False)

    print_summary(per_stage_df, overall_df)
    print(f"\nSaved per-stage metrics to: {per_stage_path}")
    print(f"Saved overall metrics to : {overall_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
