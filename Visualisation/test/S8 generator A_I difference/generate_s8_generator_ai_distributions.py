#!/usr/bin/env python3
from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parent
SOURCE_SCRIPT = ROOT.parent / "A-I Differences" / "generate_disagreement_analysis_panels.py"
INPUT = ROOT.parent / "A-I Differences" / "selected_llm_tp_detector_merged.csv"

ACADEMIC_COLUMN = "scamllm"
INDUSTRY_COLUMN = "email_phishing_detection_v3_prediction"

S8_GENERATORS = [
    "S8-deepseek",
    "S8-ministral",
    "S8-llama",
    "S8-gemini",
    "S8-gpt",
    "S8-claude",
]


def load_ai_style_module():
    spec = importlib.util.spec_from_file_location("ai_difference_style", SOURCE_SCRIPT)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Could not load plotting style from {SOURCE_SCRIPT}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def normalize_binary(series: pd.Series) -> pd.Series:
    return pd.to_numeric(series, errors="coerce").fillna(0.0).astype(float)


def load_s8_frame(style) -> pd.DataFrame:
    usecols = [
        "subject",
        "body",
        "source",
        "stage",
        "label_x",
        ACADEMIC_COLUMN,
        INDUSTRY_COLUMN,
        "industry_pred_available",
        *[column for _, column in style.PRINCIPLES],
    ]
    frame = pd.read_csv(INPUT, usecols=usecols, low_memory=False)
    frame["source"] = frame["source"].fillna("").astype(str)
    frame["stage"] = frame["stage"].fillna("").astype(str)
    frame["label_x"] = normalize_binary(frame["label_x"])
    frame[ACADEMIC_COLUMN] = normalize_binary(frame[ACADEMIC_COLUMN])
    frame[INDUSTRY_COLUMN] = normalize_binary(frame[INDUSTRY_COLUMN])
    if "industry_pred_available" in frame:
        frame["industry_pred_available"] = normalize_binary(frame["industry_pred_available"])
    frame["text"] = (
        frame["subject"].fillna("").astype(str) + "\n" + frame["body"].fillna("").astype(str)
    ).str.lower()

    selected = frame[
        frame["source"].eq("LLM")
        & frame["stage"].isin(S8_GENERATORS)
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


def draw_generator_figure(
    style,
    generator: str,
    frame: pd.DataFrame,
    persuasion: pd.DataFrame,
    linguistic: pd.DataFrame,
    persuasion_stats: pd.DataFrame,
    linguistic_stats: pd.DataFrame,
) -> None:
    counts = {
        group_name: int(frame["comparison_group"].eq(group_name).sum())
        for group_name in style.GROUPS
    }
    safe_name = generator.replace("S8-", "s8_").replace("-", "_")
    output_png = ROOT / f"{safe_name}_ai_distribution_panels.png"
    output_pdf = ROOT / f"{safe_name}_ai_distribution_panels.pdf"

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
        gridspec_kw={"wspace": 0.30, "width_ratios": [1.16, 1.18]},
    )

    style.draw_grouped_horizontal_bars(
        axes[0],
        persuasion,
        value_column="distribution_share",
        title="(a) Persuasion-pair distribution",
        xlabel="Share of pair strength",
        stats=persuasion_stats,
        stat_position="inside",
    )
    style.draw_grouped_horizontal_bars(
        axes[1],
        linguistic,
        value_column="rate",
        title="(b) Linguistic realization / action explicitness",
        xlabel="Samples containing feature",
        order_by_delta=True,
        stats=linguistic_stats,
        plot_labels=style.PLOT_LABELS,
    )
    axes[1].set_xlim(0, 1.16)
    axes[1].set_xticks(np.linspace(0, 1.0, 6))
    for text in axes[1].texts:
        if text.get_transform() == axes[1].transData:
            text.set_x(1.075)

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
    figure.savefig(output_png, dpi=300, bbox_inches="tight", pad_inches=0.06)
    figure.savefig(output_pdf, bbox_inches="tight", pad_inches=0.06)
    plt.close(figure)


def main() -> None:
    ROOT.mkdir(parents=True, exist_ok=True)
    style = load_ai_style_module()
    frame = load_s8_frame(style)
    summary: dict[str, object] = {
        "input": str(INPUT),
        "filter": (
            "source == LLM, stage in S8 generators, label == phishing, "
            "academic prediction == phishing, industrial prediction available"
        ),
        "academic_detector": ACADEMIC_COLUMN,
        "industrial_detector": INDUSTRY_COLUMN,
        "generators": {},
    }

    all_persuasion = []
    all_linguistic = []
    all_persuasion_stats = []
    all_linguistic_stats = []

    for generator in S8_GENERATORS:
        generator_frame = frame[frame["stage"].eq(generator)].copy()
        counts = {
            group_name: int(generator_frame["comparison_group"].eq(group_name).sum())
            for group_name in style.GROUPS
        }
        if min(counts.values()) == 0:
            summary["generators"][generator] = {"group_counts": counts, "skipped": True}
            continue

        persuasion, persuasion_stats = style.build_persuasion_distribution(generator_frame)
        linguistic, linguistic_stats = style.build_linguistic_rates(generator_frame)
        persuasion.insert(0, "generator", generator)
        linguistic.insert(0, "generator", generator)
        persuasion_stats.insert(0, "generator", generator)
        linguistic_stats.insert(0, "generator", generator)

        all_persuasion.append(persuasion)
        all_linguistic.append(linguistic)
        all_persuasion_stats.append(persuasion_stats)
        all_linguistic_stats.append(linguistic_stats)

        draw_generator_figure(
            style,
            generator,
            generator_frame,
            persuasion.drop(columns="generator"),
            linguistic.drop(columns="generator"),
            persuasion_stats.drop(columns="generator"),
            linguistic_stats.drop(columns="generator"),
        )
        safe_name = generator.replace("S8-", "s8_").replace("-", "_")
        summary["generators"][generator] = {
            "group_counts": counts,
            "figure_png": f"{safe_name}_ai_distribution_panels.png",
            "figure_pdf": f"{safe_name}_ai_distribution_panels.pdf",
        }

    if all_persuasion:
        pd.concat(all_persuasion, ignore_index=True).to_csv(
            ROOT / "s8_generator_persuasion_pair_distribution.csv", index=False
        )
        pd.concat(all_linguistic, ignore_index=True).to_csv(
            ROOT / "s8_generator_linguistic_feature_rates.csv", index=False
        )
        pd.concat(all_persuasion_stats, ignore_index=True).to_csv(
            ROOT / "s8_generator_persuasion_pair_stats.csv", index=False
        )
        pd.concat(all_linguistic_stats, ignore_index=True).to_csv(
            ROOT / "s8_generator_linguistic_feature_stats.csv", index=False
        )

    (ROOT / "s8_generator_ai_distribution_summary.json").write_text(
        json.dumps(summary, indent=2) + "\n",
        encoding="utf-8",
    )


if __name__ == "__main__":
    main()
