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
from matplotlib.ticker import PercentFormatter


ROOT = Path(__file__).resolve().parent
LOCAL_INPUT = ROOT / "selected_llm_tp_detector_merged.csv"
FALLBACK_INPUT = ROOT.parent / "A-I Differences" / "selected_llm_tp_detector_merged.csv"
INPUT = LOCAL_INPUT if LOCAL_INPUT.exists() else FALLBACK_INPUT
FIGURE_PNG = ROOT / "fig_ai_disagreement_analysis_panels.png"
FIGURE_PDF = ROOT / "fig_ai_disagreement_analysis_panels.pdf"
PERSUASION_CSV = ROOT / "ai_disagreement_persuasion_pair_distribution.csv"
LINGUISTIC_CSV = ROOT / "ai_disagreement_linguistic_feature_rates.csv"
PERSUASION_STATS_CSV = ROOT / "ai_disagreement_persuasion_pair_stats.csv"
LINGUISTIC_STATS_CSV = ROOT / "ai_disagreement_linguistic_feature_stats.csv"
SUMMARY_JSON = ROOT / "ai_disagreement_analysis_summary.json"

ACADEMIC_COLUMN = "scamllm"
INDUSTRY_COLUMN = "email_phishing_detection_v3_prediction"

GROUPS = {
    "A-TP / I-FN": 0.0,
    "A-TP / I-TP": 1.0,
}

PRINCIPLES = [
    ("Authority", "principle_authority"),
    ("Liking", "principle_liking"),
    ("Reciprocity", "principle_reciprocity"),
    ("Social Proof", "principle_social_proof"),
    ("Scarcity", "principle_scarcity"),
    ("Commitment", "principle_commitment"),
]

LINGUISTIC_FEATURES = [
    (
        "Explicit action request",
        r"\b(click|tap|open|visit|go to|follow|log ?in|sign ?in|submit|provide|enter|verify|confirm|update|download|review|complete)\b",
    ),
    (
        "Click/open request",
        r"\b(click|tap|open|visit|follow (the )?link|use (the )?link|press (the )?button|button below|link below)\b",
    ),
    (
        "Direct URL/page instruction",
        r"\b(https?://|www\.|url|link|webpage|website|portal|page|site|landing page|dashboard)\b",
    ),
    (
        "Information submission",
        r"\b(submit|provide|send|enter|input|fill|complete|confirm|verify|update).{0,45}\b(info|information|details|credential|password|account|address|payment|card|code|otp)\b",
    ),
    (
        "Login/account action",
        r"\b(log ?in|sign ?in|account|password|credential|username|authentication|verify your account|account verification|account update)\b",
    ),
    (
        "Urgency wording",
        r"\b(urgent|immediately|as soon as possible|asap|deadline|expires?|expiring|suspended?|locked|limited time|within \d+|final notice|act now)\b",
    ),
    (
        "Softened request",
        r"\b(please|kindly|could you|would you|when you have a chance|at your convenience|we would appreciate|if possible|just wanted|quick note)\b",
    ),
    (
        "Conversational wording",
        r"\b(hi|hello|hey|dear|thanks|thank you|hope you|checking in|following up|best regards|regards|cheers)\b",
    ),
]

PLOT_LABELS = {
    "Login/account action": "Login/\naccount",
    "Urgency wording": "Urgency",
    "Information submission": "Info\nsubmission",
    "Click/open request": "Click/\nopen",
    "Explicit action request": "Explicit\naction",
    "Direct URL/page instruction": "URL/page\ninstruction",
    "Conversational wording": "Conversational\nwording",
    "Softened request": "Softened\nrequest",
}


def normalize_binary(series: pd.Series) -> pd.Series:
    return pd.to_numeric(series, errors="coerce").fillna(0.0).astype(float)


def normal_two_sided_pvalue(z_score: float) -> float:
    return math.erfc(abs(z_score) / math.sqrt(2.0))


def bh_fdr(p_values: list[float]) -> list[float]:
    indexed = sorted(enumerate(p_values), key=lambda item: item[1])
    adjusted = [1.0] * len(p_values)
    running_min = 1.0
    total = len(p_values)
    for rank, (original_index, p_value) in reversed(list(enumerate(indexed, start=1))):
        running_min = min(running_min, p_value * total / rank)
        adjusted[original_index] = min(running_min, 1.0)
    return adjusted


def significance_stars(p_value: float) -> str:
    if p_value < 0.001:
        return "***"
    if p_value < 0.01:
        return "**"
    if p_value < 0.05:
        return "*"
    return "n.s."


def format_stat_label(p_value: float) -> str:
    stars = significance_stars(p_value)
    if p_value < 0.001:
        return f"{stars}\np<0.001"
    return f"{stars}\np={p_value:.3f}"


def welch_pvalue(group_a: pd.Series, group_b: pd.Series) -> tuple[float, float]:
    a = pd.to_numeric(group_a, errors="coerce").dropna().to_numpy(dtype=float)
    b = pd.to_numeric(group_b, errors="coerce").dropna().to_numpy(dtype=float)
    if len(a) < 2 or len(b) < 2:
        return 0.0, 1.0
    mean_diff = float(a.mean() - b.mean())
    variance_a = float(a.var(ddof=1))
    variance_b = float(b.var(ddof=1))
    standard_error = math.sqrt(variance_a / len(a) + variance_b / len(b))
    if standard_error == 0:
        return mean_diff, 1.0
    # The sample sizes are large here, so the normal approximation is effectively
    # indistinguishable from a Welch-t tail for figure annotation purposes.
    z_score = mean_diff / standard_error
    return mean_diff, normal_two_sided_pvalue(z_score)


def two_proportion_pvalue(success_a: int, n_a: int, success_b: int, n_b: int) -> tuple[float, float]:
    if n_a == 0 or n_b == 0:
        return 0.0, 1.0
    rate_a = success_a / n_a
    rate_b = success_b / n_b
    pooled = (success_a + success_b) / (n_a + n_b)
    standard_error = math.sqrt(pooled * (1.0 - pooled) * (1.0 / n_a + 1.0 / n_b))
    if standard_error == 0:
        return rate_a - rate_b, 1.0
    z_score = (rate_a - rate_b) / standard_error
    return rate_a - rate_b, normal_two_sided_pvalue(z_score)


def load_grouped_frame() -> pd.DataFrame:
    usecols = [
        "subject",
        "body",
        "source",
        "label_x",
        ACADEMIC_COLUMN,
        INDUSTRY_COLUMN,
        *[column for _, column in PRINCIPLES],
    ]
    frame = pd.read_csv(INPUT, usecols=usecols, low_memory=False)
    frame["source"] = frame["source"].fillna("").astype(str)
    frame["label_x"] = normalize_binary(frame["label_x"])
    frame[ACADEMIC_COLUMN] = normalize_binary(frame[ACADEMIC_COLUMN])
    frame[INDUSTRY_COLUMN] = normalize_binary(frame[INDUSTRY_COLUMN])
    frame["text"] = (
        frame["subject"].fillna("").astype(str) + "\n" + frame["body"].fillna("").astype(str)
    ).str.lower()

    selected = frame[
        frame["source"].eq("LLM")
        & frame["label_x"].eq(1.0)
        & frame[ACADEMIC_COLUMN].eq(1.0)
        & frame[INDUSTRY_COLUMN].isin([0.0, 1.0])
    ].copy()
    selected["comparison_group"] = np.where(
        selected[INDUSTRY_COLUMN].eq(0.0),
        "A-TP / I-FN",
        "A-TP / I-TP",
    )
    return selected


def build_persuasion_distribution(frame: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    rows: list[dict[str, object]] = []
    stat_rows: list[dict[str, object]] = []
    pair_columns = list(combinations(PRINCIPLES, 2))
    pair_strength_columns: dict[str, str] = {}

    working = frame.copy()
    for (left_label, left_column), (right_label, right_column) in pair_columns:
        pair_label = f"{left_label}+{right_label}"
        pair_column = f"pair_strength_{left_column}__{right_column}"
        pair_strength_columns[pair_label] = pair_column
        left = pd.to_numeric(working[left_column], errors="coerce").fillna(0.0)
        right = pd.to_numeric(working[right_column], errors="coerce").fillna(0.0)
        working[pair_column] = left * right

    for group_name in GROUPS:
        group_frame = working[working["comparison_group"].eq(group_name)]
        strengths = []
        for pair_label, pair_column in pair_strength_columns.items():
            strengths.append((pair_label, float(group_frame[pair_column].mean())))

        total_strength = sum(value for _, value in strengths) or 1.0
        for pair_label, mean_strength in strengths:
            rows.append(
                {
                    "feature": pair_label,
                    "group": group_name,
                    "mean_pair_strength": mean_strength,
                    "distribution_share": mean_strength / total_strength,
                }
            )

    table = pd.DataFrame(rows)
    for pair_label, pair_column in pair_strength_columns.items():
        group_a = working.loc[working["comparison_group"].eq("A-TP / I-FN"), pair_column]
        group_b = working.loc[working["comparison_group"].eq("A-TP / I-TP"), pair_column]
        mean_diff, p_value = welch_pvalue(group_a, group_b)
        stat_rows.append(
            {
                "feature": pair_label,
                "test": "Welch z approximation on per-sample pair strength",
                "delta_a_tp_i_fn_minus_a_tp_i_tp": mean_diff,
                "p_value": p_value,
            }
        )
    stats = pd.DataFrame(stat_rows)
    stats["q_value"] = bh_fdr(stats["p_value"].tolist())
    stats["stat_label"] = stats["p_value"].map(format_stat_label)

    rank = (
        table.groupby("feature", as_index=False)["distribution_share"]
        .mean()
        .sort_values("distribution_share", ascending=False)
    )
    top_features = rank.head(8)["feature"].tolist()
    return (
        table[table["feature"].isin(top_features)].copy(),
        stats[stats["feature"].isin(top_features)].copy(),
    )


def build_linguistic_rates(frame: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    rows: list[dict[str, object]] = []
    stat_rows: list[dict[str, object]] = []
    compiled = [(label, re.compile(pattern, flags=re.IGNORECASE)) for label, pattern in LINGUISTIC_FEATURES]
    presence_by_feature: dict[str, dict[str, pd.Series]] = {}

    for group_name in GROUPS:
        group_frame = frame[frame["comparison_group"].eq(group_name)]
        for feature_label, pattern in compiled:
            present = group_frame["text"].map(lambda text: bool(pattern.search(text)))
            presence_by_feature.setdefault(feature_label, {})[group_name] = present
            rows.append(
                {
                    "feature": feature_label,
                    "group": group_name,
                    "rate": float(present.mean()) if len(present) else 0.0,
                    "count": int(present.sum()),
                    "n": int(len(present)),
                }
            )

    for feature_label, group_presence in presence_by_feature.items():
        group_a = group_presence["A-TP / I-FN"]
        group_b = group_presence["A-TP / I-TP"]
        delta, p_value = two_proportion_pvalue(
            int(group_a.sum()),
            int(len(group_a)),
            int(group_b.sum()),
            int(len(group_b)),
        )
        stat_rows.append(
            {
                "feature": feature_label,
                "test": "two-proportion z-test",
                "delta_a_tp_i_fn_minus_a_tp_i_tp": delta,
                "p_value": p_value,
            }
        )

    stats = pd.DataFrame(stat_rows)
    stats["q_value"] = bh_fdr(stats["p_value"].tolist())
    stats["stat_label"] = stats["p_value"].map(format_stat_label)
    return pd.DataFrame(rows), stats


def draw_grouped_horizontal_bars(
    axis: plt.Axes,
    table: pd.DataFrame,
    *,
    value_column: str,
    title: str,
    xlabel: str,
    order_by_delta: bool = False,
    stats: pd.DataFrame | None = None,
    stat_position: str = "right",
    plot_labels: dict[str, str] | None = None,
    hide_y_labels: bool = False,
) -> None:
    pivot = table.pivot(index="feature", columns="group", values=value_column).fillna(0.0)
    pivot = pivot[list(GROUPS.keys())]
    if order_by_delta:
        pivot = pivot.assign(delta=(pivot["A-TP / I-FN"] - pivot["A-TP / I-TP"]).abs())
        pivot = pivot.sort_values("delta", ascending=True).drop(columns="delta")
    else:
        pivot = pivot.assign(mean=pivot.mean(axis=1))
        pivot = pivot.sort_values("mean", ascending=True).drop(columns="mean")

    y = np.arange(len(pivot))
    bar_height = 0.36
    colors = {
        "A-TP / I-FN": "#f8d987",
        "A-TP / I-TP": "#003b4d",
    }
    offsets = {
        "A-TP / I-FN": -bar_height / 2,
        "A-TP / I-TP": bar_height / 2,
    }

    for group_name in GROUPS:
        axis.barh(
            y + offsets[group_name],
            pivot[group_name].to_numpy(dtype=float),
            height=bar_height,
            label=group_name,
            color=colors[group_name],
        )

    max_value = float(pivot.to_numpy(dtype=float).max()) if not pivot.empty else 1.0
    if value_column == "rate":
        x_lower = 0
        x_upper = 1.03
        stat_x = 0.94
    else:
        x_lower = 0
        x_upper = max_value * 1.28
        stat_x = max_value * 1.18
    axis.set_xlim(x_lower, x_upper)

    if stats is not None:
        labels = stats.set_index("feature")["stat_label"].to_dict()
        for y_index, feature in enumerate(pivot.index):
            if stat_position == "inside":
                x = stat_x
                ha = "right"
            else:
                x = stat_x
                ha = "center"
            axis.text(
                x,
                y_index,
                labels.get(feature, ""),
                ha=ha,
                va="center",
                fontsize=15.5,
                color="#222222",
            )

    axis.set_yticks(y)
    if hide_y_labels:
        axis.set_yticklabels([])
        axis.tick_params(axis="y", length=0)
    else:
        display_labels = [
            plot_labels.get(label, label)
            if plot_labels
            else f"({label.split('+', 1)[0]}, {label.split('+', 1)[1]})"
            if "+" in label
            else label
            for label in pivot.index
        ]
    if plot_labels and not hide_y_labels:
        axis.set_yticklabels(display_labels, fontsize=16.5)
        axis.tick_params(axis="y", pad=3)
        axis.set_xticks(np.linspace(0, 1.0, 6))
    elif not hide_y_labels:
        axis.set_yticklabels(display_labels, fontsize=16.5)
    axis.set_xlabel(xlabel, fontsize=16.5, labelpad=8)
    axis.xaxis.set_major_formatter(PercentFormatter(xmax=1.0))
    axis.grid(axis="x", color="#d9d9d9", linewidth=0.7, alpha=0.8)
    axis.set_axisbelow(True)
    axis.spines["top"].set_visible(False)
    axis.spines["right"].set_visible(False)
    axis.tick_params(axis="x", labelsize=16.5)
    axis.text(
        0.5,
        -0.145,
        title,
        ha="center",
        va="top",
        transform=axis.transAxes,
        fontsize=17.5,
    )


def draw_figure(
    persuasion: pd.DataFrame,
    linguistic: pd.DataFrame,
    persuasion_stats: pd.DataFrame,
    linguistic_stats: pd.DataFrame,
    counts: dict[str, int],
) -> None:
    plt.rcParams.update(
        {
            "font.family": "serif",
            "font.serif": ["Times New Roman", "Times", "Nimbus Roman", "DejaVu Serif"],
            "axes.linewidth": 0.9,
            "xtick.major.width": 0.8,
            "ytick.major.width": 0.8,
        }
    )
    figure, axes = plt.subplots(
        1,
        2,
        figsize=(15.2, 6.05),
        gridspec_kw={"wspace": 0.28, "width_ratios": [1.16, 1.18]},
    )

    draw_grouped_horizontal_bars(
        axes[0],
        persuasion,
        value_column="distribution_share",
        title="(a) Persuasion-pair distribution",
        xlabel="Share of pair strength",
        stats=persuasion_stats,
        stat_position="inside",
    )
    draw_grouped_horizontal_bars(
        axes[1],
        linguistic,
        value_column="rate",
        title="(b) Linguistic realization / action explicitness",
        xlabel="Samples containing feature",
        order_by_delta=True,
        stats=linguistic_stats,
        plot_labels=PLOT_LABELS,
    )

    handles, labels = axes[0].get_legend_handles_labels()
    legend_labels = [f"{label.replace(' / ', ', ')} (n={counts[label]:,})" for label in labels]
    axes[0].legend(
        handles,
        legend_labels,
        loc="lower left",
        bbox_to_anchor=(0.08, 0.08),
        ncol=1,
        frameon=False,
        fontsize=15.5,
    )
    figure.subplots_adjust(left=0.15, right=0.985, bottom=0.24, top=0.985, wspace=0.30)
    figure.savefig(FIGURE_PNG, dpi=300, bbox_inches="tight", pad_inches=0.06)
    figure.savefig(FIGURE_PDF, bbox_inches="tight", pad_inches=0.06)
    plt.close(figure)


def main() -> None:
    frame = load_grouped_frame()
    counts = {group_name: int(frame["comparison_group"].eq(group_name).sum()) for group_name in GROUPS}
    persuasion, persuasion_stats = build_persuasion_distribution(frame)
    linguistic, linguistic_stats = build_linguistic_rates(frame)

    persuasion.to_csv(PERSUASION_CSV, index=False)
    linguistic.to_csv(LINGUISTIC_CSV, index=False)
    persuasion_stats.to_csv(PERSUASION_STATS_CSV, index=False)
    linguistic_stats.to_csv(LINGUISTIC_STATS_CSV, index=False)
    SUMMARY_JSON.write_text(
        json.dumps(
            {
                "input": str(INPUT),
                "academic_detector": ACADEMIC_COLUMN,
                "industry_detector": INDUSTRY_COLUMN,
                "filter": "source == LLM and label == phishing and academic prediction == phishing",
                "group_counts": counts,
                "outputs": {
                    "figure_png": FIGURE_PNG.name,
                    "figure_pdf": FIGURE_PDF.name,
                    "persuasion_csv": PERSUASION_CSV.name,
                    "linguistic_csv": LINGUISTIC_CSV.name,
                    "persuasion_stats_csv": PERSUASION_STATS_CSV.name,
                    "linguistic_stats_csv": LINGUISTIC_STATS_CSV.name,
                },
            },
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    draw_figure(persuasion, linguistic, persuasion_stats, linguistic_stats, counts)


if __name__ == "__main__":
    main()
