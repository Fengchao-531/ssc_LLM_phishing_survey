#!/usr/bin/env python3
from __future__ import annotations

import json
import math
import re
from itertools import combinations
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.ticker import PercentFormatter
from scipy.stats import chi2_contingency


ROOT = Path(__file__).resolve().parent
INPUT = ROOT.parent / "A-I Differences" / "selected_llm_tp_detector_merged.csv"
OUTPUT_CSV = ROOT / "s8_linguistic_action_feature_prevalence.csv"
GLOBAL_TEST_CSV = ROOT / "s8_linguistic_action_feature_global_tests.csv"
CELL_TEST_CSV = ROOT / "s8_linguistic_action_feature_cell_tests.csv"
PAIRWISE_TEST_CSV = ROOT / "s8_linguistic_action_feature_pairwise_tests.csv"
SUMMARY_JSON = ROOT / "s8_linguistic_action_feature_summary.json"
FIGURE_PNG = ROOT / "fig_s8_linguistic_action_feature_heatmap.png"
FIGURE_PDF = ROOT / "fig_s8_linguistic_action_feature_heatmap.pdf"

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

FEATURES = [
    (
        "Urgency wording",
        "Urgency",
        r"\b(urgent|immediately|as soon as possible|asap|deadline|expires?|expiring|suspended?|locked|limited time|within \d+|final notice|act now)\b",
    ),
    (
        "Login/account action",
        "Login/\naccount",
        r"\b(log ?in|sign ?in|account|password|credential|username|authentication|verify your account|account verification|account update)\b",
    ),
    (
        "Information submission",
        "Info\nsubmission",
        r"\b(submit|provide|send|enter|input|fill|complete|confirm|verify|update).{0,45}\b(info|information|details|credential|password|account|address|payment|card|code|otp)\b",
    ),
    (
        "Click/open request",
        "Click/\nopen",
        r"\b(click|tap|open|visit|follow (the )?link|use (the )?link|press (the )?button|button below|link below)\b",
    ),
    (
        "Softened request",
        "Softened\nrequest",
        r"\b(please|kindly|could you|would you|when you have a chance|at your convenience|we would appreciate|if possible|just wanted|quick note)\b",
    ),
    (
        "Explicit action request",
        "Explicit\naction",
        r"\b(click|tap|open|visit|go to|follow|log ?in|sign ?in|submit|provide|enter|verify|confirm|update|download|review|complete)\b",
    ),
    (
        "Direct URL/page instruction",
        "URL/page\ninstruction",
        r"\b(https?://|www\.|url|link|webpage|website|portal|page|site|landing page|dashboard)\b",
    ),
    (
        "Conversational wording",
        "Conversational\nwording",
        r"\b(hi|hello|hey|dear|thanks|thank you|hope you|checking in|following up|best regards|regards|cheers)\b",
    ),
]


def normalize_binary(series: pd.Series) -> pd.Series:
    return pd.to_numeric(series, errors="coerce").fillna(0.0).astype(float)


def normal_two_sided_pvalue(z_score: float) -> float:
    return math.erfc(abs(z_score) / math.sqrt(2.0))


def two_proportion_pvalue(success_a: int, n_a: int, success_b: int, n_b: int) -> tuple[float, float]:
    if n_a == 0 or n_b == 0:
        return 0.0, 1.0
    rate_a = success_a / n_a
    rate_b = success_b / n_b
    pooled = (success_a + success_b) / (n_a + n_b)
    standard_error = math.sqrt(pooled * (1.0 - pooled) * (1.0 / n_a + 1.0 / n_b))
    if standard_error == 0:
        return rate_a - rate_b, 1.0
    return rate_a - rate_b, normal_two_sided_pvalue((rate_a - rate_b) / standard_error)


def bh_fdr(p_values: list[float]) -> list[float]:
    indexed = sorted(enumerate(p_values), key=lambda item: item[1])
    adjusted = [1.0] * len(p_values)
    running_min = 1.0
    total = len(p_values)
    for rank, (original_index, p_value) in reversed(list(enumerate(indexed, start=1))):
        running_min = min(running_min, p_value * total / rank)
        adjusted[original_index] = min(running_min, 1.0)
    return adjusted


def stars(q_value: float) -> str:
    if q_value < 0.001:
        return "***"
    if q_value < 0.01:
        return "**"
    if q_value < 0.05:
        return "*"
    return ""


def load_s8_phishing() -> pd.DataFrame:
    usecols = ["subject", "body", "source", "stage", "label_x"]
    frame = pd.read_csv(INPUT, usecols=usecols, low_memory=False)
    frame["source"] = frame["source"].fillna("").astype(str)
    frame["stage"] = frame["stage"].fillna("").astype(str)
    frame["label_x"] = normalize_binary(frame["label_x"])
    selected = frame[
        frame["source"].eq("LLM")
        & frame["stage"].isin(GENERATORS)
        & frame["label_x"].eq(1.0)
    ].copy()
    selected["text"] = (
        selected["subject"].fillna("").astype(str)
        + "\n"
        + selected["body"].fillna("").astype(str)
    ).str.lower()
    return selected


def build_tables(frame: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    prevalence_rows: list[dict[str, object]] = []
    global_rows: list[dict[str, object]] = []
    cell_rows: list[dict[str, object]] = []
    pairwise_rows: list[dict[str, object]] = []

    for feature, _, pattern_text in FEATURES:
        pattern = re.compile(pattern_text, flags=re.IGNORECASE)
        presence = frame["text"].map(lambda text: bool(pattern.search(text)))
        frame[feature] = presence.astype(int)

        contingency = []
        for generator in GENERATORS:
            generator_presence = frame.loc[frame["stage"].eq(generator), feature]
            success = int(generator_presence.sum())
            total = int(len(generator_presence))
            prevalence_rows.append(
                {
                    "generator": generator,
                    "generator_label": GENERATOR_LABELS[generator],
                    "feature": feature,
                    "prevalence": success / total if total else 0.0,
                    "count": success,
                    "n": total,
                }
            )
            contingency.append([success, total - success])

            other_presence = frame.loc[~frame["stage"].eq(generator), feature]
            delta, p_value = two_proportion_pvalue(
                success,
                total,
                int(other_presence.sum()),
                int(len(other_presence)),
            )
            cell_rows.append(
                {
                    "generator": generator,
                    "generator_label": GENERATOR_LABELS[generator],
                    "feature": feature,
                    "comparison": "generator vs all other S8 generators",
                    "generator_prevalence": success / total if total else 0.0,
                    "other_prevalence": float(other_presence.mean()) if len(other_presence) else 0.0,
                    "delta_generator_minus_other": delta,
                    "p_value": p_value,
                }
            )

        chi2, p_value, dof, _ = chi2_contingency(np.asarray(contingency))
        global_rows.append(
            {
                "feature": feature,
                "test": "chi-square test across six generators",
                "chi2": chi2,
                "dof": dof,
                "p_value": p_value,
            }
        )

        for left, right in combinations(GENERATORS, 2):
            left_values = frame.loc[frame["stage"].eq(left), feature]
            right_values = frame.loc[frame["stage"].eq(right), feature]
            delta, p_value = two_proportion_pvalue(
                int(left_values.sum()),
                int(len(left_values)),
                int(right_values.sum()),
                int(len(right_values)),
            )
            pairwise_rows.append(
                {
                    "feature": feature,
                    "generator_a": left,
                    "generator_a_label": GENERATOR_LABELS[left],
                    "generator_b": right,
                    "generator_b_label": GENERATOR_LABELS[right],
                    "prevalence_a": float(left_values.mean()) if len(left_values) else 0.0,
                    "prevalence_b": float(right_values.mean()) if len(right_values) else 0.0,
                    "delta_a_minus_b": delta,
                    "p_value": p_value,
                }
            )

    prevalence = pd.DataFrame(prevalence_rows)
    global_tests = pd.DataFrame(global_rows)
    cell_tests = pd.DataFrame(cell_rows)
    pairwise_tests = pd.DataFrame(pairwise_rows)

    global_tests["q_value"] = bh_fdr(global_tests["p_value"].tolist())
    cell_tests["q_value"] = bh_fdr(cell_tests["p_value"].tolist())
    pairwise_tests["q_value"] = bh_fdr(pairwise_tests["p_value"].tolist())
    cell_tests["stars"] = cell_tests["q_value"].map(stars)
    pairwise_tests["stars"] = pairwise_tests["q_value"].map(stars)
    prevalence = prevalence.merge(
        cell_tests[["generator", "feature", "q_value", "stars"]],
        on=["generator", "feature"],
        how="left",
    )
    return prevalence, global_tests, cell_tests, pairwise_tests


def draw_heatmap(prevalence: pd.DataFrame, global_tests: pd.DataFrame) -> None:
    feature_order = [feature for feature, _, _ in FEATURES]
    feature_labels = {feature: label for feature, label, _ in FEATURES}

    matrix = prevalence.pivot(index="generator", columns="feature", values="prevalence").loc[
        GENERATORS, feature_order
    ]
    cell_stars = prevalence.pivot(index="generator", columns="feature", values="stars").loc[
        GENERATORS, feature_order
    ]
    global_q = global_tests.set_index("feature")["q_value"].to_dict()

    plt.rcParams.update(
        {
            "font.family": "serif",
            "font.serif": ["Times New Roman", "Times", "Nimbus Roman", "DejaVu Serif"],
            "axes.linewidth": 1.05,
            "xtick.major.width": 0.9,
            "ytick.major.width": 0.9,
        }
    )
    cmap = LinearSegmentedColormap.from_list(
        "s8_feature_prevalence",
        ["#ffffff", "#fff1bf", "#f3c957", "#6f9cac", "#003b4d"],
    )

    figure, axis = plt.subplots(figsize=(18.6, 7.5))
    image = axis.imshow(matrix.to_numpy(dtype=float), cmap=cmap, vmin=0.0, vmax=1.0, aspect="auto")

    axis.set_xticks(np.arange(len(feature_order)))
    axis.set_xticklabels(
        [feature_labels[feature] for feature in feature_order],
        fontsize=20,
        rotation=0,
        ha="center",
    )
    axis.set_yticks(np.arange(len(GENERATORS)))
    axis.set_yticklabels([GENERATOR_LABELS[generator] for generator in GENERATORS], fontsize=22)
    axis.tick_params(axis="x", pad=8)
    axis.tick_params(axis="y", pad=8)

    axis.set_xticks(np.arange(-0.5, len(feature_order), 1), minor=True)
    axis.set_yticks(np.arange(-0.5, len(GENERATORS), 1), minor=True)
    axis.grid(which="minor", color="white", linestyle="-", linewidth=2.0)
    axis.tick_params(which="minor", bottom=False, left=False)

    for row_index, generator in enumerate(GENERATORS):
        for col_index, feature in enumerate(feature_order):
            value = float(matrix.loc[generator, feature])
            star = cell_stars.loc[generator, feature]
            axis.text(
                col_index,
                row_index,
                f"{value * 100:.1f}%{star}",
                ha="center",
                va="center",
                fontsize=19,
                color="black",
            )

    for col_index, feature in enumerate(feature_order):
        marker = stars(float(global_q[feature]))
        if marker:
            axis.text(
                col_index,
                -0.68,
                marker,
                ha="center",
                va="center",
                fontsize=18,
                color="black",
                clip_on=False,
            )

    axis.set_title("S8 linguistic/action feature prevalence across LLM generators", fontsize=25, pad=30)
    axis.set_xlabel("Linguistic/action feature", fontsize=22, labelpad=14)
    axis.set_ylabel("Generator", fontsize=22, labelpad=14)

    colorbar = figure.colorbar(image, ax=axis, fraction=0.028, pad=0.018)
    colorbar.ax.yaxis.set_major_formatter(PercentFormatter(xmax=1.0))
    colorbar.set_label("Feature prevalence", fontsize=21, labelpad=13)
    colorbar.ax.tick_params(labelsize=19)

    figure.subplots_adjust(left=0.105, right=0.93, bottom=0.23, top=0.82)
    figure.savefig(FIGURE_PNG, dpi=300, bbox_inches="tight", pad_inches=0.06)
    figure.savefig(FIGURE_PDF, bbox_inches="tight", pad_inches=0.06)
    plt.close(figure)


def main() -> None:
    ROOT.mkdir(parents=True, exist_ok=True)
    frame = load_s8_phishing()
    prevalence, global_tests, cell_tests, pairwise_tests = build_tables(frame)

    prevalence.to_csv(OUTPUT_CSV, index=False)
    global_tests.to_csv(GLOBAL_TEST_CSV, index=False)
    cell_tests.to_csv(CELL_TEST_CSV, index=False)
    pairwise_tests.to_csv(PAIRWISE_TEST_CSV, index=False)
    SUMMARY_JSON.write_text(
        json.dumps(
            {
                "input": str(INPUT),
                "filter": "source == LLM, stage in S8 generators, label == phishing",
                "definition": "C_g,k = Pr(k=1 | D_g)",
                "generators": {
                    generator: {
                        "label": GENERATOR_LABELS[generator],
                        "n": int(frame["stage"].eq(generator).sum()),
                    }
                    for generator in GENERATORS
                },
                "outputs": {
                    "figure_png": FIGURE_PNG.name,
                    "figure_pdf": FIGURE_PDF.name,
                    "prevalence_csv": OUTPUT_CSV.name,
                    "global_tests_csv": GLOBAL_TEST_CSV.name,
                    "cell_tests_csv": CELL_TEST_CSV.name,
                    "pairwise_tests_csv": PAIRWISE_TEST_CSV.name,
                },
            },
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    draw_heatmap(prevalence, global_tests)


if __name__ == "__main__":
    main()
