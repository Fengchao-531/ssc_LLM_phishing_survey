#!/usr/bin/env python3
from __future__ import annotations

import json
from itertools import combinations
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import wasserstein_distance


ROOT = Path(__file__).resolve().parent
INPUT = ROOT.parent / "A-I Differences" / "selected_llm_tp_detector_merged.csv"

SCORE_SUMMARY_CSV = ROOT / "s8_detector_generator_score_summary.csv"
MEAN_SHIFT_CSV = ROOT / "s8_detector_mean_score_shift.csv"
WASSERSTEIN_CSV = ROOT / "s8_detector_pairwise_wasserstein.csv"
WASSERSTEIN_SUMMARY_CSV = ROOT / "s8_detector_wasserstein_summary.csv"
MARKDOWN_SUMMARY = ROOT / "s8_detector_score_shift_summary.md"
SUMMARY_JSON = ROOT / "s8_detector_score_shift_outputs.json"

GENERATORS = [
    "S8-claude",
    "S8-gpt",
    "S8-gemini",
    "S8-llama",
    "S8-ministral",
    "S8-deepseek",
]

GENERATOR_LABELS = {
    "S8-claude": "Claude",
    "S8-gpt": "GPT",
    "S8-gemini": "Gemini",
    "S8-llama": "Llama",
    "S8-ministral": "Ministral",
    "S8-deepseek": "DeepSeek",
}

DETECTORS = {
    "scamllm": "ScamLLM",
    "pimref": "PiMRef",
    "t5phishing": "T5Phishing",
    "xgboost": "XGBoost",
    "securenet_llama": "SecureNet",
    "email_phishing_detection_v3_prediction": "V3",
}


def normalize_binary(series: pd.Series) -> pd.Series:
    return pd.to_numeric(series, errors="coerce")


def load_frame() -> pd.DataFrame:
    usecols = [
        "source",
        "stage",
        "label_x",
        "industry_pred_available",
        *DETECTORS.keys(),
    ]
    frame = pd.read_csv(INPUT, usecols=usecols, low_memory=False)
    frame["source"] = frame["source"].fillna("").astype(str)
    frame["stage"] = frame["stage"].fillna("").astype(str)
    frame["label_x"] = pd.to_numeric(frame["label_x"], errors="coerce").fillna(0.0)
    for column in DETECTORS:
        frame[column] = normalize_binary(frame[column])
    if "industry_pred_available" in frame:
        frame["industry_pred_available"] = pd.to_numeric(
            frame["industry_pred_available"], errors="coerce"
        ).fillna(0.0)
    selected = frame[
        frame["source"].eq("LLM")
        & frame["stage"].isin(GENERATORS)
        & frame["label_x"].eq(1.0)
    ].copy()
    return selected


def iqr(values: pd.Series) -> float:
    numeric = pd.to_numeric(values, errors="coerce").dropna()
    if numeric.empty:
        return float("nan")
    return float(numeric.quantile(0.75) - numeric.quantile(0.25))


def build_score_summary(frame: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for detector_column, detector_label in DETECTORS.items():
        for generator in GENERATORS:
            subset = frame[frame["stage"].eq(generator)].copy()
            if detector_column == "email_phishing_detection_v3_prediction":
                subset = subset[subset["industry_pred_available"].eq(1.0)]
            values = subset[detector_column].dropna().astype(float)
            rows.append(
                {
                    "detector": detector_column,
                    "detector_label": detector_label,
                    "generator": generator,
                    "generator_label": GENERATOR_LABELS[generator],
                    "n": int(values.shape[0]),
                    "mean_score": float(values.mean()) if len(values) else float("nan"),
                    "median_score": float(values.median()) if len(values) else float("nan"),
                    "iqr": iqr(values),
                    "q1": float(values.quantile(0.25)) if len(values) else float("nan"),
                    "q3": float(values.quantile(0.75)) if len(values) else float("nan"),
                }
            )
    return pd.DataFrame(rows)


def build_mean_shift(score_summary: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for detector, detector_frame in score_summary.groupby("detector", sort=False):
        usable = detector_frame.dropna(subset=["mean_score"]).copy()
        if usable.empty:
            continue
        min_row = usable.loc[usable["mean_score"].idxmin()]
        max_row = usable.loc[usable["mean_score"].idxmax()]
        rows.append(
            {
                "detector": detector,
                "detector_label": str(max_row["detector_label"]),
                "min_generator": str(min_row["generator"]),
                "min_generator_label": str(min_row["generator_label"]),
                "min_mean_score": float(min_row["mean_score"]),
                "max_generator": str(max_row["generator"]),
                "max_generator_label": str(max_row["generator_label"]),
                "max_mean_score": float(max_row["mean_score"]),
                "R_d_max_minus_min_mean": float(max_row["mean_score"] - min_row["mean_score"]),
                "usable_generators": int(usable["generator"].nunique()),
            }
        )
    return pd.DataFrame(rows)


def build_wasserstein(frame: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    rows: list[dict[str, object]] = []
    for detector_column, detector_label in DETECTORS.items():
        values_by_generator: dict[str, np.ndarray] = {}
        for generator in GENERATORS:
            subset = frame[frame["stage"].eq(generator)].copy()
            if detector_column == "email_phishing_detection_v3_prediction":
                subset = subset[subset["industry_pred_available"].eq(1.0)]
            values = subset[detector_column].dropna().astype(float).to_numpy()
            if len(values):
                values_by_generator[generator] = values

        for left, right in combinations(values_by_generator, 2):
            distance = float(wasserstein_distance(values_by_generator[left], values_by_generator[right]))
            rows.append(
                {
                    "detector": detector_column,
                    "detector_label": detector_label,
                    "generator_a": left,
                    "generator_a_label": GENERATOR_LABELS[left],
                    "generator_b": right,
                    "generator_b_label": GENERATOR_LABELS[right],
                    "n_a": int(len(values_by_generator[left])),
                    "n_b": int(len(values_by_generator[right])),
                    "wasserstein_distance": distance,
                }
            )

    pairwise = pd.DataFrame(rows)
    summary_rows: list[dict[str, object]] = []
    for detector, detector_frame in pairwise.groupby("detector", sort=False):
        distances = detector_frame["wasserstein_distance"].dropna()
        max_row = detector_frame.loc[detector_frame["wasserstein_distance"].idxmax()]
        summary_rows.append(
            {
                "detector": detector,
                "detector_label": str(max_row["detector_label"]),
                "pair_count": int(len(distances)),
                "max_wasserstein": float(distances.max()),
                "max_pair": f"{max_row['generator_a_label']} vs {max_row['generator_b_label']}",
                "median_wasserstein": float(distances.median()),
                "mean_wasserstein": float(distances.mean()),
            }
        )
    return pairwise, pd.DataFrame(summary_rows)


def write_markdown(score_summary: pd.DataFrame, mean_shift: pd.DataFrame, wasserstein_summary: pd.DataFrame) -> None:
    lines = [
        "# S8 Detector Score Shift Summary",
        "",
        "Scores are detector outputs on LLM-generated phishing samples in S8. For binary detector outputs, the mean score equals the detector phishing rate on that generator.",
        "",
        "## Mean-score shift by detector",
        "",
        "| Detector | Min LLM | Min mean | Max LLM | Max mean | R_d | Median W1 | Max W1 | Max-W1 pair |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---|",
    ]
    merged = mean_shift.merge(
        wasserstein_summary[
            ["detector", "median_wasserstein", "max_wasserstein", "max_pair"]
        ],
        on="detector",
        how="left",
    )
    for row in merged.sort_values("R_d_max_minus_min_mean", ascending=False).itertuples():
        lines.append(
            "| {detector} | {min_llm} | {min_mean:.4f} | {max_llm} | {max_mean:.4f} | {shift:.4f} | {median_w1:.4f} | {max_w1:.4f} | {pair} |".format(
                detector=row.detector_label,
                min_llm=row.min_generator_label,
                min_mean=row.min_mean_score,
                max_llm=row.max_generator_label,
                max_mean=row.max_mean_score,
                shift=row.R_d_max_minus_min_mean,
                median_w1=row.median_wasserstein,
                max_w1=row.max_wasserstein,
                pair=row.max_pair,
            )
        )

    lines.extend(
        [
            "",
            "## Per-detector, per-generator score summaries",
            "",
        ]
    )
    for detector, detector_frame in score_summary.groupby("detector_label", sort=False):
        lines.extend(
            [
                f"### {detector}",
                "",
                "| LLM | n | Mean | Median | IQR |",
                "|---|---:|---:|---:|---:|",
            ]
        )
        for row in detector_frame.itertuples():
            lines.append(
                f"| {row.generator_label} | {row.n} | {row.mean_score:.4f} | {row.median_score:.4f} | {row.iqr:.4f} |"
            )
        lines.append("")

    MARKDOWN_SUMMARY.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    ROOT.mkdir(parents=True, exist_ok=True)
    frame = load_frame()
    score_summary = build_score_summary(frame)
    mean_shift = build_mean_shift(score_summary)
    wasserstein_pairwise, wasserstein_summary = build_wasserstein(frame)

    score_summary.to_csv(SCORE_SUMMARY_CSV, index=False)
    mean_shift.to_csv(MEAN_SHIFT_CSV, index=False)
    wasserstein_pairwise.to_csv(WASSERSTEIN_CSV, index=False)
    wasserstein_summary.to_csv(WASSERSTEIN_SUMMARY_CSV, index=False)
    write_markdown(score_summary, mean_shift, wasserstein_summary)

    SUMMARY_JSON.write_text(
        json.dumps(
            {
                "input": str(INPUT),
                "filter": "source == LLM, stage in S8 generators, label == phishing",
                "score_definition": "available detector output s_d(x); binary detector outputs are treated as scores in [0,1]",
                "outputs": {
                    "score_summary_csv": SCORE_SUMMARY_CSV.name,
                    "mean_shift_csv": MEAN_SHIFT_CSV.name,
                    "wasserstein_pairwise_csv": WASSERSTEIN_CSV.name,
                    "wasserstein_summary_csv": WASSERSTEIN_SUMMARY_CSV.name,
                    "markdown_summary": MARKDOWN_SUMMARY.name,
                },
            },
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )


if __name__ == "__main__":
    main()
