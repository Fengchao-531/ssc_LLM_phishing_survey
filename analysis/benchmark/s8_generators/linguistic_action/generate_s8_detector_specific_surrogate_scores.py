#!/usr/bin/env python3
from __future__ import annotations

import json
from itertools import combinations
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.colors import LinearSegmentedColormap
from scipy.stats import wasserstein_distance
from sklearn.metrics import average_precision_score, brier_score_loss, roc_auc_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import PolynomialFeatures, StandardScaler
from sklearn.linear_model import LogisticRegression


ROOT = Path(__file__).resolve().parent
INPUT = ROOT.parent / "A-I Differences" / "selected_llm_tp_detector_merged.csv"

SCORED_ROWS_CSV = ROOT / "s8_detector_specific_surrogate_scored_rows.csv"
MODEL_FIT_CSV = ROOT / "s8_detector_specific_surrogate_fit_metrics.csv"
SCORE_SUMMARY_CSV = ROOT / "s8_detector_specific_surrogate_generator_summary.csv"
MEAN_SHIFT_CSV = ROOT / "s8_detector_specific_surrogate_mean_shift.csv"
WASSERSTEIN_CSV = ROOT / "s8_detector_specific_surrogate_pairwise_wasserstein.csv"
WASSERSTEIN_SUMMARY_CSV = ROOT / "s8_detector_specific_surrogate_wasserstein_summary.csv"
MARKDOWN_SUMMARY = ROOT / "s8_detector_specific_surrogate_summary.md"
SUMMARY_JSON = ROOT / "s8_detector_specific_surrogate_outputs.json"
FIGURE_PNG = ROOT / "fig_s8_detector_specific_surrogate_mean_heatmap.png"
FIGURE_PDF = ROOT / "fig_s8_detector_specific_surrogate_mean_heatmap.pdf"

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


def load_frame() -> pd.DataFrame:
    usecols = [
        "source",
        "stage",
        "label_x",
        "proj_x",
        "proj_y",
        "industry_pred_available",
        *DETECTORS.keys(),
    ]
    frame = pd.read_csv(INPUT, usecols=usecols, low_memory=False)
    frame["source"] = frame["source"].fillna("").astype(str)
    frame["stage"] = frame["stage"].fillna("").astype(str)
    frame["label_x"] = pd.to_numeric(frame["label_x"], errors="coerce").fillna(0.0)
    frame["proj_x"] = pd.to_numeric(frame["proj_x"], errors="coerce")
    frame["proj_y"] = pd.to_numeric(frame["proj_y"], errors="coerce")
    frame["industry_pred_available"] = pd.to_numeric(
        frame["industry_pred_available"], errors="coerce"
    ).fillna(0.0)
    for detector in DETECTORS:
        frame[detector] = pd.to_numeric(frame[detector], errors="coerce")
    return frame[
        frame["label_x"].eq(1.0)
        & frame["proj_x"].notna()
        & frame["proj_y"].notna()
    ].copy()


def fit_surface(frame: pd.DataFrame, detector: str, seed: int) -> tuple[Pipeline, pd.DataFrame]:
    train = frame[frame[detector].notna()].copy()
    if detector == "email_phishing_detection_v3_prediction":
        train = train[train["industry_pred_available"].eq(1.0)].copy()
    y = train[detector].astype(int).to_numpy()
    if np.unique(y).size < 2:
        raise ValueError(f"{detector} has a single class; cannot fit detector-specific surrogate.")
    model = Pipeline(
        [
            ("poly", PolynomialFeatures(degree=2, include_bias=False)),
            ("scale", StandardScaler()),
            (
                "logreg",
                LogisticRegression(max_iter=1500, class_weight="balanced", random_state=seed),
            ),
        ]
    )
    model.fit(train[["proj_x", "proj_y"]].to_numpy(), y)
    return model, train


def build_fit_metric(detector: str, detector_label: str, train: pd.DataFrame, scores: np.ndarray) -> dict[str, object]:
    y = train[detector].astype(int).to_numpy()
    return {
        "detector": detector,
        "detector_label": detector_label,
        "n": int(len(y)),
        "positive_rate": float(y.mean()),
        "roc_auc": float(roc_auc_score(y, scores)) if np.unique(y).size > 1 else float("nan"),
        "average_precision": float(average_precision_score(y, scores))
        if np.unique(y).size > 1
        else float("nan"),
        "brier": float(brier_score_loss(y, scores)),
        "score_definition": "Pr(detector predicts phishing | proj_x, proj_y), polynomial logistic surrogate",
    }


def iqr(values: pd.Series) -> float:
    return float(values.quantile(0.75) - values.quantile(0.25))


def score_s8_rows(frame: pd.DataFrame, seed: int = 7) -> tuple[pd.DataFrame, pd.DataFrame]:
    s8 = frame[frame["source"].eq("LLM") & frame["stage"].isin(GENERATORS)].copy()
    scored = s8[["source", "stage", "proj_x", "proj_y"]].copy()
    scored["generator_label"] = scored["stage"].map(GENERATOR_LABELS)
    fit_rows = []
    for detector, detector_label in DETECTORS.items():
        model, train = fit_surface(frame, detector, seed)
        train_scores = model.predict_proba(train[["proj_x", "proj_y"]].to_numpy())[:, 1]
        fit_rows.append(build_fit_metric(detector, detector_label, train, train_scores))
        scored[f"{detector}__surrogate_score"] = model.predict_proba(
            scored[["proj_x", "proj_y"]].to_numpy()
        )[:, 1]
    return scored, pd.DataFrame(fit_rows)


def build_score_summary(scored: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for detector, detector_label in DETECTORS.items():
        score_column = f"{detector}__surrogate_score"
        for generator in GENERATORS:
            values = scored.loc[scored["stage"].eq(generator), score_column].astype(float)
            rows.append(
                {
                    "detector": detector,
                    "detector_label": detector_label,
                    "generator": generator,
                    "generator_label": GENERATOR_LABELS[generator],
                    "n": int(len(values)),
                    "mean_score": float(values.mean()),
                    "median_score": float(values.median()),
                    "iqr": iqr(values),
                    "q1": float(values.quantile(0.25)),
                    "q3": float(values.quantile(0.75)),
                    "min_score": float(values.min()),
                    "max_score": float(values.max()),
                }
            )
    return pd.DataFrame(rows)


def build_mean_shift(summary: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for detector, detector_frame in summary.groupby("detector", sort=False):
        min_row = detector_frame.loc[detector_frame["mean_score"].idxmin()]
        max_row = detector_frame.loc[detector_frame["mean_score"].idxmax()]
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
                "usable_generators": int(detector_frame["generator"].nunique()),
            }
        )
    return pd.DataFrame(rows)


def build_wasserstein(scored: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    rows = []
    for detector, detector_label in DETECTORS.items():
        score_column = f"{detector}__surrogate_score"
        values_by_generator = {
            generator: scored.loc[scored["stage"].eq(generator), score_column].astype(float).to_numpy()
            for generator in GENERATORS
        }
        for left, right in combinations(GENERATORS, 2):
            rows.append(
                {
                    "detector": detector,
                    "detector_label": detector_label,
                    "generator_a": left,
                    "generator_a_label": GENERATOR_LABELS[left],
                    "generator_b": right,
                    "generator_b_label": GENERATOR_LABELS[right],
                    "n_a": int(len(values_by_generator[left])),
                    "n_b": int(len(values_by_generator[right])),
                    "wasserstein_distance": float(
                        wasserstein_distance(values_by_generator[left], values_by_generator[right])
                    ),
                }
            )
    pairwise = pd.DataFrame(rows)
    summary_rows = []
    for detector, detector_frame in pairwise.groupby("detector", sort=False):
        max_row = detector_frame.loc[detector_frame["wasserstein_distance"].idxmax()]
        summary_rows.append(
            {
                "detector": detector,
                "detector_label": str(max_row["detector_label"]),
                "pair_count": int(len(detector_frame)),
                "max_wasserstein": float(detector_frame["wasserstein_distance"].max()),
                "max_pair": f"{max_row['generator_a_label']} vs {max_row['generator_b_label']}",
                "median_wasserstein": float(detector_frame["wasserstein_distance"].median()),
                "mean_wasserstein": float(detector_frame["wasserstein_distance"].mean()),
            }
        )
    return pairwise, pd.DataFrame(summary_rows)


def draw_mean_heatmap(summary: pd.DataFrame) -> None:
    matrix = summary.pivot(index="detector", columns="generator", values="mean_score").loc[
        list(DETECTORS.keys()), GENERATORS
    ]
    plt.rcParams.update(
        {
            "font.family": "serif",
            "font.serif": ["Times New Roman", "Times", "Nimbus Roman", "DejaVu Serif"],
            "axes.linewidth": 1.0,
        }
    )
    cmap = LinearSegmentedColormap.from_list(
        "detector_surrogate_mean",
        ["#ffffff", "#fff1bf", "#f3c957", "#6f9cac", "#003b4d"],
    )
    figure, axis = plt.subplots(figsize=(12.8, 7.0))
    image = axis.imshow(matrix.to_numpy(dtype=float), cmap=cmap, vmin=0.0, vmax=1.0, aspect="auto")
    axis.set_xticks(np.arange(len(GENERATORS)))
    axis.set_xticklabels([GENERATOR_LABELS[g] for g in GENERATORS], fontsize=17)
    axis.set_yticks(np.arange(len(DETECTORS)))
    axis.set_yticklabels([DETECTORS[d] for d in DETECTORS], fontsize=18)
    axis.tick_params(axis="x", pad=8)
    axis.tick_params(axis="y", pad=8)
    axis.set_xticks(np.arange(-0.5, len(GENERATORS), 1), minor=True)
    axis.set_yticks(np.arange(-0.5, len(DETECTORS), 1), minor=True)
    axis.grid(which="minor", color="white", linewidth=2.0)
    axis.tick_params(which="minor", bottom=False, left=False)
    for row_index, detector in enumerate(DETECTORS):
        for col_index, generator in enumerate(GENERATORS):
            value = float(matrix.loc[detector, generator])
            axis.text(
                col_index,
                row_index,
                f"{value:.3f}",
                ha="center",
                va="center",
                fontsize=17,
                color="black",
            )
    axis.set_title("Detector-specific continuous surrogate score by S8 generator", fontsize=22, pad=18)
    axis.set_xlabel("Generator", fontsize=19, labelpad=12)
    axis.set_ylabel("Detector", fontsize=19, labelpad=12)
    colorbar = figure.colorbar(image, ax=axis, fraction=0.035, pad=0.02)
    colorbar.set_label("Mean surrogate score", fontsize=18, labelpad=12)
    colorbar.ax.tick_params(labelsize=16)
    figure.subplots_adjust(left=0.16, right=0.91, bottom=0.16, top=0.86)
    figure.savefig(FIGURE_PNG, dpi=300, bbox_inches="tight", pad_inches=0.05)
    figure.savefig(FIGURE_PDF, bbox_inches="tight", pad_inches=0.05)
    plt.close(figure)


def write_markdown(
    fit_metrics: pd.DataFrame,
    summary: pd.DataFrame,
    mean_shift: pd.DataFrame,
    wasserstein_summary: pd.DataFrame,
) -> None:
    merged = mean_shift.merge(
        wasserstein_summary[["detector", "median_wasserstein", "max_wasserstein", "max_pair"]],
        on="detector",
        how="left",
    )
    lines = [
        "# S8 Detector-Specific Continuous Surrogate Scores",
        "",
        "`s_d(x)` is estimated as `Pr(detector d predicts phishing | proj_x, proj_y)` using a degree-2 logistic surrogate surface with balanced class weights.",
        "",
        "## Mean-score shift",
        "",
        "| Detector | Min LLM | Min mean | Max LLM | Max mean | R_d | Median W1 | Max W1 | Max-W1 pair |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---|",
    ]
    for row in merged.sort_values("R_d_max_minus_min_mean", ascending=False).itertuples():
        lines.append(
            f"| {row.detector_label} | {row.min_generator_label} | {row.min_mean_score:.4f} | {row.max_generator_label} | {row.max_mean_score:.4f} | {row.R_d_max_minus_min_mean:.4f} | {row.median_wasserstein:.4f} | {row.max_wasserstein:.4f} | {row.max_pair} |"
        )
    lines.extend(["", "## Surrogate fit quality", "", "| Detector | n | Positive rate | ROC-AUC | AP | Brier |", "|---|---:|---:|---:|---:|---:|"])
    for row in fit_metrics.itertuples():
        lines.append(
            f"| {row.detector_label} | {row.n} | {row.positive_rate:.4f} | {row.roc_auc:.4f} | {row.average_precision:.4f} | {row.brier:.4f} |"
        )
    lines.extend(["", "## Per-generator summaries", ""])
    for detector, detector_frame in summary.groupby("detector_label", sort=False):
        lines.extend([f"### {detector}", "", "| LLM | n | Mean | Median | IQR |", "|---|---:|---:|---:|---:|"])
        for row in detector_frame.itertuples():
            lines.append(
                f"| {row.generator_label} | {row.n} | {row.mean_score:.4f} | {row.median_score:.4f} | {row.iqr:.4f} |"
            )
        lines.append("")
    MARKDOWN_SUMMARY.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    ROOT.mkdir(parents=True, exist_ok=True)
    frame = load_frame()
    scored, fit_metrics = score_s8_rows(frame)
    summary = build_score_summary(scored)
    mean_shift = build_mean_shift(summary)
    pairwise_wasserstein, wasserstein_summary = build_wasserstein(scored)

    scored.to_csv(SCORED_ROWS_CSV, index=False)
    fit_metrics.to_csv(MODEL_FIT_CSV, index=False)
    summary.to_csv(SCORE_SUMMARY_CSV, index=False)
    mean_shift.to_csv(MEAN_SHIFT_CSV, index=False)
    pairwise_wasserstein.to_csv(WASSERSTEIN_CSV, index=False)
    wasserstein_summary.to_csv(WASSERSTEIN_SUMMARY_CSV, index=False)
    write_markdown(fit_metrics, summary, mean_shift, wasserstein_summary)
    draw_mean_heatmap(summary)

    SUMMARY_JSON.write_text(
        json.dumps(
            {
                "input": str(INPUT),
                "filter": "label_x == phishing; S8 scoring uses source == LLM and stage in S8 generators",
                "surrogate_definition": "s_d(x) = Pr(detector d predicts phishing | proj_x, proj_y), degree-2 balanced logistic regression",
                "outputs": {
                    "scored_rows_csv": SCORED_ROWS_CSV.name,
                    "model_fit_csv": MODEL_FIT_CSV.name,
                    "score_summary_csv": SCORE_SUMMARY_CSV.name,
                    "mean_shift_csv": MEAN_SHIFT_CSV.name,
                    "wasserstein_pairwise_csv": WASSERSTEIN_CSV.name,
                    "wasserstein_summary_csv": WASSERSTEIN_SUMMARY_CSV.name,
                    "markdown_summary": MARKDOWN_SUMMARY.name,
                    "figure_png": FIGURE_PNG.name,
                    "figure_pdf": FIGURE_PDF.name,
                },
            },
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )


if __name__ == "__main__":
    main()
