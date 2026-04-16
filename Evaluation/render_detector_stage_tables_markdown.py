#!/usr/bin/env python3
"""Render detector stage metric CSVs into markdown/HTML README tables."""

from __future__ import annotations

from pathlib import Path

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

from export_detector_stage_tables import (
    ACADEMIC_DETECTORS,
    ACADEMIC_HW_DIR,
    INDUSTRY_DETECTORS,
    INDUSTRY_HW_DIR,
    normalize_binary_value,
)


SCRIPT_DIR = Path(__file__).resolve().parent
TABLES_DIR = SCRIPT_DIR / "detector-stage-metric-tables"
METRICS = ["precision", "recall", "accuracy", "fpr", "fnr", "f0_5", "f1", "f2", "mcc"]
CATEGORY_TITLES = {
    "academic": "Academic Detector Tables",
    "industry": "Industry Detector Tables",
}


def format_value(value: object) -> str:
    if pd.isna(value):
        return "-"
    return f"{float(value):.4f}"


def compute_overall_metrics(y_true: list[int], y_pred: list[int]) -> dict[str, float]:
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


def compute_academic_hw_overall(detector_name: str) -> dict[str, float]:
    detector_column = ACADEMIC_DETECTORS[detector_name]["combined_column"]
    y_true: list[int] = []
    y_pred: list[int] = []

    for csv_path in sorted(ACADEMIC_HW_DIR.glob("*.csv")):
        df = pd.read_csv(csv_path, dtype=str, keep_default_na=False)
        labels = df["label"].map(normalize_binary_value)
        preds = df[detector_column].map(normalize_binary_value)
        valid = labels.notna() & preds.notna()
        if not valid.any():
            continue
        y_true.extend(labels[valid].astype(int).tolist())
        y_pred.extend(preds[valid].astype(int).tolist())

    if not y_true:
        return {metric: pd.NA for metric in METRICS}
    return compute_overall_metrics(y_true, y_pred)


def compute_industry_hw_overall(detector_name: str) -> dict[str, float]:
    detector_config = INDUSTRY_DETECTORS[detector_name]
    y_true: list[int] = []
    y_pred: list[int] = []

    combined_column = detector_config["combined_column"]
    standalone_column = detector_config["standalone_column"]

    for csv_path in sorted(INDUSTRY_HW_DIR.glob("*.csv")):
        df = pd.read_csv(csv_path, dtype=str, keep_default_na=False)
        labels = df["label"].map(normalize_binary_value)

        prediction_column = None
        if combined_column and combined_column in df.columns:
            prediction_column = combined_column
        elif standalone_column and standalone_column in df.columns:
            prediction_column = standalone_column
        if prediction_column is None:
            continue

        preds = df[prediction_column].map(normalize_binary_value)
        valid = labels.notna() & preds.notna()
        if not valid.any():
            continue
        y_true.extend(labels[valid].astype(int).tolist())
        y_pred.extend(preds[valid].astype(int).tolist())

    if not y_true:
        return {metric: pd.NA for metric in METRICS}
    return compute_overall_metrics(y_true, y_pred)


def render_standard_table(detector_name: str, csv_path: Path) -> str:
    df = pd.read_csv(csv_path)
    lines = [f"## {detector_name}", ""]
    lines.append("<table>")
    lines.append("  <thead>")
    lines.append("    <tr>")
    lines.append('      <th rowspan="2">Stage</th>')
    lines.append(f'      <th colspan="{len(METRICS)}">HW / GD</th>')
    lines.append(f'      <th colspan="{len(METRICS)}">LLM</th>')
    lines.append("    </tr>")
    lines.append("    <tr>")
    for _side in ("hw", "llm"):
        for metric in METRICS:
            lines.append(f"      <th>{metric}</th>")
    lines.append("    </tr>")
    lines.append("  </thead>")
    lines.append("  <tbody>")

    for _, row in df.iterrows():
        lines.append("    <tr>")
        lines.append(f"      <td>{row['stage']}</td>")
        for side in ("hw", "llm"):
            for metric in METRICS:
                lines.append(f"      <td>{format_value(row[f'{side}_{metric}'])}</td>")
        lines.append("    </tr>")

    lines.append("  </tbody>")
    lines.append("</table>")
    lines.append("")
    return "\n".join(lines)


def render_academic_table(detector_name: str, csv_path: Path) -> str:
    df = pd.read_csv(csv_path)
    hw_overall = compute_academic_hw_overall(detector_name)

    lines = [f"## {detector_name}", ""]
    lines.append("<table>")
    lines.append("  <thead>")
    lines.append("    <tr>")
    lines.append('      <th rowspan="2">HW Summary</th>')
    lines.append(f'      <th colspan="{len(METRICS)}">HW / GD Overall</th>')
    lines.append('      <th rowspan="2">LLM Stage</th>')
    lines.append(f'      <th colspan="{len(METRICS)}">LLM</th>')
    lines.append("    </tr>")
    lines.append("    <tr>")
    for _side in ("hw", "llm"):
        for metric in METRICS:
            lines.append(f"      <th>{metric}</th>")
    lines.append("    </tr>")
    lines.append("  </thead>")
    lines.append("  <tbody>")

    for row_index, (_, row) in enumerate(df.iterrows()):
        lines.append("    <tr>")
        if row_index == 0:
            lines.append("      <td>HW-overall</td>")
            for metric in METRICS:
                lines.append(f"      <td>{format_value(hw_overall[metric])}</td>")
        else:
            lines.append("      <td></td>")
            for _metric in METRICS:
                lines.append("      <td></td>")
        lines.append(f"      <td>{row['stage']}</td>")
        for metric in METRICS:
            lines.append(f"      <td>{format_value(row[f'llm_{metric}'])}</td>")
        lines.append("    </tr>")

    lines.append("  </tbody>")
    lines.append("</table>")
    lines.append("")
    return "\n".join(lines)


def render_industry_table(detector_name: str, csv_path: Path) -> str:
    df = pd.read_csv(csv_path)
    hw_overall = compute_industry_hw_overall(detector_name)

    lines = [f"## {detector_name}", ""]
    lines.append("<table>")
    lines.append("  <thead>")
    lines.append("    <tr>")
    lines.append('      <th rowspan="2">HW Summary</th>')
    lines.append(f'      <th colspan="{len(METRICS)}">HW / GD Overall</th>')
    lines.append('      <th rowspan="2">LLM Stage</th>')
    lines.append(f'      <th colspan="{len(METRICS)}">LLM</th>')
    lines.append("    </tr>")
    lines.append("    <tr>")
    for _side in ("hw", "llm"):
        for metric in METRICS:
            lines.append(f"      <th>{metric}</th>")
    lines.append("    </tr>")
    lines.append("  </thead>")
    lines.append("  <tbody>")

    for row_index, (_, row) in enumerate(df.iterrows()):
        lines.append("    <tr>")
        if row_index == 0:
            lines.append("      <td>HW-overall</td>")
            for metric in METRICS:
                lines.append(f"      <td>{format_value(hw_overall[metric])}</td>")
        else:
            lines.append("      <td></td>")
            for _metric in METRICS:
                lines.append("      <td></td>")
        lines.append(f"      <td>{row['stage']}</td>")
        for metric in METRICS:
            lines.append(f"      <td>{format_value(row[f'llm_{metric}'])}</td>")
        lines.append("    </tr>")

    lines.append("  </tbody>")
    lines.append("</table>")
    lines.append("")
    return "\n".join(lines)


def render_category_readme(category_name: str) -> str:
    category_dir = TABLES_DIR / category_name
    csv_paths = sorted(path for path in category_dir.glob("*.csv") if path.name != "README.md")

    lines = [f"# {CATEGORY_TITLES[category_name]}", ""]
    if category_name == "academic":
        lines.append("The left half is a single `HW / GD` overall summary for the detector.")
        lines.append("This is computed by concatenating all available HW rows first, then recomputing the metrics once.")
        lines.append("The right half keeps stage-level `LLM` metrics.")
    elif category_name == "industry":
        lines.append("The left half is a single `HW / GD` overall summary for the detector.")
        lines.append("This is computed by concatenating all available HW rows first, then recomputing the metrics once.")
        lines.append("The right half keeps stage-level `LLM` metrics.")
    else:
        lines.append("Rows are stages. The left half is `HW / GD`, and the right half is `LLM`.")
    lines.append("Metric order is `precision`, `recall`, `accuracy`, `fpr`, `fnr`, `f0_5`, `f1`, `f2`, `mcc`.")
    lines.append("`-` means the detector-stage-source combination does not currently have usable predictions.")
    lines.append("")

    for csv_path in csv_paths:
        if category_name == "academic":
            lines.append(render_academic_table(csv_path.stem, csv_path))
        elif category_name == "industry":
            lines.append(render_industry_table(csv_path.stem, csv_path))
        else:
            lines.append(render_standard_table(csv_path.stem, csv_path))

    return "\n".join(lines).rstrip() + "\n"


def main() -> int:
    for category_name in CATEGORY_TITLES:
        category_dir = TABLES_DIR / category_name
        category_dir.mkdir(parents=True, exist_ok=True)
        readme_path = category_dir / "README.md"
        readme_path.write_text(render_category_readme(category_name), encoding="utf-8")
        print(f"[OK] wrote {readme_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
