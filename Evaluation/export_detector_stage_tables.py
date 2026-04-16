#!/usr/bin/env python3
"""Export per-detector stage metric tables for Academic and Industry detectors."""

from __future__ import annotations

import argparse
import re
from collections import OrderedDict, defaultdict
from pathlib import Path
from typing import Iterable

import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    fbeta_score,
    matthews_corrcoef,
    precision_score,
    recall_score,
)


SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parent
DEFAULT_OUTPUT_DIR = SCRIPT_DIR / "detector-stage-metric-tables"

ACADEMIC_LLM_DIR = REPO_ROOT / "Detectors" / "Academic" / "email_detectors" / "LLM-Acad"
ACADEMIC_HW_DIR = REPO_ROOT / "Detectors" / "Academic" / "email_detectors" / "HW-Acad"
INDUSTRY_LLM_DIR = REPO_ROOT / "Detectors" / "Industry" / "email_detectors" / "output" / "LLM-Ind"
INDUSTRY_HW_DIR = REPO_ROOT / "Detectors" / "Industry" / "email_detectors" / "output" / "HW-result" / "HW-Ind"

METRIC_ORDER = [
    "precision",
    "recall",
    "accuracy",
    "fpr",
    "fnr",
    "f0_5",
    "f1",
    "f2",
    "mcc",
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

ACADEMIC_DETECTORS = OrderedDict(
    [
        ("scamllm", {"combined_column": "scamllm", "standalone_column": "scamllm"}),
        ("pimref", {"combined_column": "pimref", "standalone_column": "pimref"}),
        ("t5phishing", {"combined_column": "t5phishing", "standalone_column": "t5phishing"}),
        (
            "ml_watermark_logreg",
            {"combined_column": "ml_watermark_logreg", "standalone_column": "ml_watermark_logreg"},
        ),
        ("xgboost", {"combined_column": "xgboost", "standalone_column": "xgboost"}),
        ("securenet_llama", {"combined_column": "securenet_llama", "standalone_column": "securenet_llama"}),
    ]
)

INDUSTRY_DETECTORS = OrderedDict(
    [
        ("llm_guard", {"combined_column": "llm_guard_prediction", "standalone_column": "llm_guard_prediction"}),
        (
            "phishing_email_agent",
            {
                "combined_column": "phishing_email_agent_prediction",
                "standalone_column": "phishing_email_agent_prediction",
            },
        ),
        (
            "email_phishing_detection_v3",
            {
                "combined_column": "email_phishing_detection_v3_prediction",
                "standalone_column": "email_phishing_detection_v3_prediction",
            },
        ),
        (
            "pyrit_original",
            {"combined_column": "pyrit_original_prediction", "standalone_column": "pyrit_original_prediction"},
        ),
        (
            "pyrit_blocklist",
            {"combined_column": "pyrit_blocklist_prediction", "standalone_column": "pyrit_blocklist_prediction"},
        ),
        ("spamassassin", {"combined_column": None, "standalone_column": "spamassassin_prediction"}),
        ("oopspam", {"combined_column": "oopspam_prediction", "standalone_column": "oopspam_prediction"}),
    ]
)

CATEGORY_CONFIGS = OrderedDict(
    [
        (
            "academic",
            {
                "detectors": ACADEMIC_DETECTORS,
                "llm_dir": ACADEMIC_LLM_DIR,
                "hw_dir": ACADEMIC_HW_DIR,
            },
        ),
        (
            "industry",
            {
                "detectors": INDUSTRY_DETECTORS,
                "llm_dir": INDUSTRY_LLM_DIR,
                "hw_dir": INDUSTRY_HW_DIR,
            },
        ),
    ]
)

LLM_COMBINED_RE = re.compile(r"^(S\d+(?:-[A-Za-z0-9]+)*)_results\.csv$")
HW_COMBINED_RE = re.compile(r"^(S\d+(?:-[A-Za-z0-9]+)*)-GD_results\.csv$")
LLM_STANDALONE_RE = re.compile(r"^(S\d+(?:-[A-Za-z0-9]+)*)__([a-z0-9_]+)_results\.csv$")
LLM_PREFIXED_STANDALONE_RE = re.compile(
    r"^(?:[A-Za-z0-9_-]+)__?(S\d+(?:-[A-Za-z0-9]+)*)__([a-z0-9_]+)_results\.csv$"
)
HW_STANDALONE_RE = re.compile(r"^(S\d+(?:-[A-Za-z0-9]+)*)-GD__([a-z0-9_]+)_results\.csv$")
ACADEMIC_LLM_RE = re.compile(r"^(S\d+(?:-[A-Za-z0-9]+)*)\.csv$")
ACADEMIC_HW_RE = re.compile(r"^(S\d+(?:-[A-Za-z0-9]+)*)-GD\.csv$")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Export one CSV per detector with rows=stages and columns="
            " HW metrics on the left and LLM metrics on the right."
        )
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
        help=f"Directory where per-detector tables are written (default: {DEFAULT_OUTPUT_DIR})",
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


def to_binary_series(series: pd.Series) -> pd.Series:
    return pd.Series(series.map(normalize_binary_value), index=series.index, dtype="float")


def resolve_column(columns: Iterable[str], preferred_name: str) -> str | None:
    lookup = {column.lower(): column for column in columns}
    return lookup.get(preferred_name.lower())


def compute_metric_bundle(y_true: pd.Series, y_pred: pd.Series) -> dict[str, float]:
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
    return {
        "precision": float(precision_score(y_true, y_pred, zero_division=0)),
        "recall": float(recall_score(y_true, y_pred, zero_division=0)),
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "fpr": float(fp / (fp + tn)) if (fp + tn) else 0.0,
        "fnr": float(fn / (fn + tp)) if (fn + tp) else 0.0,
        "f0_5": float(fbeta_score(y_true, y_pred, beta=0.5, zero_division=0)),
        "f1": float(f1_score(y_true, y_pred, zero_division=0)),
        "f2": float(fbeta_score(y_true, y_pred, beta=2.0, zero_division=0)),
        "mcc": float(matthews_corrcoef(y_true, y_pred)),
    }


def stage_sort_key(stage_name: str) -> tuple[int, str]:
    match = re.match(r"^S(\d+)", stage_name)
    stage_number = int(match.group(1)) if match else 10**9
    return stage_number, stage_name


def parse_stage_and_detector(path: Path, source_kind: str, category_name: str) -> tuple[str, str | None] | None:
    if category_name == "academic":
        if source_kind == "llm":
            match = ACADEMIC_LLM_RE.match(path.name)
            if match:
                return match.group(1), None
            return None
        match = ACADEMIC_HW_RE.match(path.name)
        if match:
            return match.group(1), None
        return None

    if source_kind == "llm":
        prefixed_standalone_match = LLM_PREFIXED_STANDALONE_RE.match(path.name)
        if prefixed_standalone_match:
            return prefixed_standalone_match.group(1), prefixed_standalone_match.group(2)
        standalone_match = LLM_STANDALONE_RE.match(path.name)
        if standalone_match:
            return standalone_match.group(1), standalone_match.group(2)
        combined_match = LLM_COMBINED_RE.match(path.name)
        if combined_match:
            return combined_match.group(1), None
        return None

    standalone_match = HW_STANDALONE_RE.match(path.name)
    if standalone_match:
        return standalone_match.group(1), standalone_match.group(2)
    combined_match = HW_COMBINED_RE.match(path.name)
    if combined_match:
        return combined_match.group(1), None
    return None


def compute_metrics_from_csv(path: Path, prediction_column: str) -> dict[str, float] | None:
    df = pd.read_csv(path, dtype=str, keep_default_na=False)
    label_column = resolve_column(df.columns, "label")
    pred_column = resolve_column(df.columns, prediction_column)
    if label_column is None or pred_column is None:
        return None

    label_series = to_binary_series(df[label_column])
    pred_series = to_binary_series(df[pred_column])
    valid_mask = label_series.notna() & pred_series.notna()
    if int(valid_mask.sum()) == 0:
        return None

    y_true = label_series[valid_mask].astype(int)
    y_pred = pred_series[valid_mask].astype(int)
    return compute_metric_bundle(y_true, y_pred)


def collect_metrics_for_category(category_name: str, config: dict[str, object]) -> tuple[dict[str, dict[str, dict[str, float]]], list[str]]:
    detectors: OrderedDict[str, dict[str, str | None]] = config["detectors"]  # type: ignore[assignment]
    metrics: dict[str, dict[str, dict[str, dict[str, float]]]] = {
        detector: {"hw": {}, "llm": {}} for detector in detectors
    }
    stages_seen: set[str] = set()

    for source_name, source_dir in (("llm", config["llm_dir"]), ("hw", config["hw_dir"])):
        source_path = Path(source_dir)
        if not source_path.exists():
            continue

        for csv_path in sorted(source_path.glob("*.csv")):
            parsed = parse_stage_and_detector(csv_path, source_name, category_name)
            if parsed is None:
                continue

            stage_name, standalone_detector = parsed
            stages_seen.add(stage_name)

            if standalone_detector is not None:
                if standalone_detector not in detectors:
                    continue
                prediction_column = detectors[standalone_detector]["standalone_column"]
                if not prediction_column:
                    continue
                metric_bundle = compute_metrics_from_csv(csv_path, prediction_column)
                if metric_bundle is not None:
                    metrics[standalone_detector][source_name][stage_name] = metric_bundle
                continue

            for detector_name, detector_config in detectors.items():
                prediction_column = detector_config["combined_column"]
                if not prediction_column:
                    continue
                metric_bundle = compute_metrics_from_csv(csv_path, prediction_column)
                if metric_bundle is not None:
                    metrics[detector_name][source_name][stage_name] = metric_bundle

    return metrics, sorted(stages_seen, key=stage_sort_key)


def build_detector_table(
    detector_name: str,
    detector_metrics: dict[str, dict[str, dict[str, float]]],
    stages: list[str],
) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for stage_name in stages:
        row: dict[str, object] = {"stage": stage_name}
        for source_name in ("hw", "llm"):
            source_metrics = detector_metrics[source_name].get(stage_name)
            for metric_name in METRIC_ORDER:
                row[f"{source_name}_{metric_name}"] = (
                    source_metrics.get(metric_name) if source_metrics is not None else pd.NA
                )
        rows.append(row)

    columns = ["stage"]
    columns.extend(f"hw_{metric_name}" for metric_name in METRIC_ORDER)
    columns.extend(f"llm_{metric_name}" for metric_name in METRIC_ORDER)
    return pd.DataFrame(rows, columns=columns)


def main() -> int:
    args = parse_args()
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    for category_name, config in CATEGORY_CONFIGS.items():
        category_output_dir = output_dir / category_name
        category_output_dir.mkdir(parents=True, exist_ok=True)

        metrics_by_detector, stages = collect_metrics_for_category(category_name, config)
        print(f"[{category_name}] stages={stages}")

        for detector_name in config["detectors"]:  # type: ignore[index]
            detector_table = build_detector_table(
                detector_name,
                metrics_by_detector[detector_name],
                stages,
            )
            output_path = category_output_dir / f"{detector_name}.csv"
            detector_table.to_csv(output_path, index=False)
            print(f"[OK] {category_name}/{detector_name} -> {output_path}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
