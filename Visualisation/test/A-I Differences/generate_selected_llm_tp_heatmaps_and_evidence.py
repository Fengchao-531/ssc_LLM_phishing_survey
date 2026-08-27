#!/usr/bin/env python3
from __future__ import annotations

import json
import math
import sys
from collections import Counter
from pathlib import Path
import textwrap

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.colors import LinearSegmentedColormap, TwoSlopeNorm
from matplotlib.ticker import FormatStrFormatter


ROOT = Path(__file__).resolve().parent
VIS_ROOT = ROOT.parents[1]
SURVEY_ROOT = VIS_ROOT.parent
EVIDENCE_ROOT = VIS_ROOT / "evidence"
if str(EVIDENCE_ROOT) not in sys.path:
    sys.path.insert(0, str(EVIDENCE_ROOT))

from build_overview_group_evidence import (  # noqa: E402
    add_document_features,
    aggregate_doc_presence,
    aggregate_weighted_terms,
    compute_cell_contribution,
    phrase_is_meaningful,
    token_is_meaningful,
    weighted_log_ratio,
)


PROJECTED_POINTS = VIS_ROOT / "test" / "projected_points.csv"
EVAL_ROOT = SURVEY_ROOT / "Evaluation" / "processed-evaluation-datasets"
HEATMAP_OUTPUT = ROOT / "fig_selected_llm_tp_detector_heatmaps.png"
DIFFERENCE_HEATMAP_OUTPUT = ROOT / "fig_selected_llm_tp_detector_difference_heatmap.png"
HEATMAP_TABLE_OUTPUT = ROOT / "selected_llm_tp_detector_heatmap_values.csv"
DIFFERENCE_TABLE_OUTPUT = ROOT / "selected_llm_tp_detector_difference_values.csv"
EVIDENCE_TABLE_OUTPUT = ROOT / "selected_llm_tp_words_phrases_matrix.csv"
EVIDENCE_TABLE_IMAGE = ROOT / "fig_selected_llm_tp_words_phrases_matrix.png"
EVIDENCE_MARKDOWN_OUTPUT = ROOT / "selected_llm_tp_words_phrases_matrix.md"
SUMMARY_OUTPUT = ROOT / "selected_llm_tp_detector_summary.json"
MERGED_OUTPUT = ROOT / "selected_llm_tp_detector_merged.csv"

ACADEMIC_COLUMN = "scamllm"
ACADEMIC_LABEL = "ScamLLM"
INDUSTRY_COLUMN = "email_phishing_detection_v3_prediction"
INDUSTRY_LABEL = "Phishing Detector V3"

PRINCIPLE_ORDER = [
    ("Authority", "principle_authority"),
    ("Liking", "principle_liking"),
    ("Reciprocity", "principle_reciprocity"),
    ("Social Proof", "principle_social_proof"),
    ("Scarcity", "principle_scarcity"),
    ("Commitment", "principle_commitment"),
]
PRINCIPLE_LABELS = [label for label, _ in PRINCIPLE_ORDER]
PRINCIPLE_COLUMNS = [column for _, column in PRINCIPLE_ORDER]
PRINCIPLE_SLUGS = [
    "authority",
    "liking",
    "reciprocity",
    "social_proof",
    "scarcity",
    "commitment",
]
PRINCIPLE_INDEX = {slug: idx for idx, slug in enumerate(PRINCIPLE_SLUGS)}
PRINCIPLE_SHORT_LABELS = {
    "authority": "A",
    "liking": "L",
    "reciprocity": "R",
    "social_proof": "SP",
    "scarcity": "SC",
    "commitment": "C",
}
DUAL_HEATMAP_XTICK_FONT_SIZE = 15.5
DUAL_HEATMAP_CELL_FONT_SIZE = 14.5
DUAL_HEATMAP_COLORBAR_LABEL_FONT_SIZE = 15

TOP_DOCS = 700
TOP_WORDS = 8
TOP_PHRASES = 8
TOP_ITEMS_FOR_TABLE = 5

HEATMAP_CMAP = LinearSegmentedColormap.from_list(
    "selected_llm_tp_folder1_blue",
    ["#ffffff", "#dce8ef", "#b9d3df", "#86b3d0", "#4f89b6"],
    N=256,
)

DIFFERENCE_CMAP = LinearSegmentedColormap.from_list(
    "selected_llm_tp_difference",
    [
        (0.00, "#9a5c1f"),
        (0.34, "#d69544"),
        (0.50, "#ffffff"),
        (0.72, "#b9d3df"),
        (0.88, "#86b3d0"),
        (1.00, "#4f89b6"),
    ],
    N=256,
)


def normalize_binary(series: pd.Series) -> pd.Series:
    return pd.to_numeric(series, errors="coerce").fillna(0.0).astype(float)


def stage_to_source_file(stage: str) -> str:
    return f"{stage}.csv"


def build_all_cell_specs() -> list[tuple[str, str, str]]:
    specs: list[tuple[str, str, str]] = []
    for row_index, row_slug in enumerate(PRINCIPLE_SLUGS):
        for col_index in range(row_index + 1):
            col_slug = PRINCIPLE_SLUGS[col_index]
            cell_label = f"({PRINCIPLE_SHORT_LABELS[row_slug]}, {PRINCIPLE_SHORT_LABELS[col_slug]})"
            specs.append((cell_label, row_slug, col_slug))
    return specs


def load_projected_points() -> pd.DataFrame:
    frame = pd.read_csv(PROJECTED_POINTS, low_memory=False).copy()
    frame["subject"] = frame["subject"].fillna("").astype(str)
    frame["body"] = frame["body"].fillna("").astype(str)
    frame["raw_label"] = pd.to_numeric(frame["raw_label"], errors="coerce").fillna(0).astype(int)
    frame[ACADEMIC_COLUMN] = normalize_binary(frame[ACADEMIC_COLUMN])
    frame["merge_stage"] = frame["stage"].astype(str)
    frame["merge_source_file"] = frame["source_file"].astype(str)

    alias_mask = frame["source"].astype(str).eq("HW") & frame["hw_stage_alias_from"].notna()
    frame.loc[alias_mask, "merge_stage"] = frame.loc[alias_mask, "hw_stage_alias_from"].astype(str)
    frame.loc[alias_mask, "merge_source_file"] = frame.loc[alias_mask, "hw_stage_alias_from"].astype(str) + ".csv"
    return frame


def load_industry_lookup() -> pd.DataFrame:
    parts: list[pd.DataFrame] = []
    source_to_dir = {
        "HW": EVAL_ROOT / "gd" / "industry",
        "LLM": EVAL_ROOT / "llm" / "industry",
    }
    for source_name, source_dir in source_to_dir.items():
        for path in sorted(source_dir.glob("*.csv")):
            columns = pd.read_csv(path, nrows=0).columns.tolist()
            if INDUSTRY_COLUMN not in columns:
                continue
            requested = ["subject", "body", "label", INDUSTRY_COLUMN]
            if "source_file" in columns:
                requested.append("source_file")
            frame = pd.read_csv(path, usecols=requested, low_memory=False).copy()
            frame["source"] = source_name
            if "source_file" in frame.columns:
                frame["merge_stage"] = frame["source_file"].astype(str).str.replace(".csv", "", regex=False)
                frame["merge_source_file"] = frame["source_file"].astype(str)
            else:
                frame["merge_stage"] = path.stem
                frame["merge_source_file"] = stage_to_source_file(path.stem)
            parts.append(frame)

    combined = pd.concat(parts, ignore_index=True)
    combined["subject"] = combined["subject"].fillna("").astype(str)
    combined["body"] = combined["body"].fillna("").astype(str)
    combined["label"] = pd.to_numeric(combined["label"], errors="coerce").fillna(0).astype(int)
    combined[INDUSTRY_COLUMN] = normalize_binary(combined[INDUSTRY_COLUMN])
    return combined.drop_duplicates(
        subset=["source", "subject", "body", "label", "merge_stage", "merge_source_file", INDUSTRY_COLUMN]
    ).reset_index(drop=True)


def attach_industry_predictions(projected: pd.DataFrame, lookup: pd.DataFrame) -> pd.DataFrame:
    merged = projected.merge(
        lookup,
        left_on=["source", "subject", "body", "raw_label", "merge_stage", "merge_source_file"],
        right_on=["source", "subject", "body", "label", "merge_stage", "merge_source_file"],
        how="left",
    )

    unmatched = merged[INDUSTRY_COLUMN].isna()
    if unmatched.any():
        stage_lookup = lookup.groupby(
            ["source", "subject", "body", "label", "merge_stage"], as_index=False
        )[INDUSTRY_COLUMN].max()
        fallback = projected.loc[unmatched].merge(
            stage_lookup,
            left_on=["source", "subject", "body", "raw_label", "merge_stage"],
            right_on=["source", "subject", "body", "label", "merge_stage"],
            how="left",
        )
        merged.loc[unmatched, INDUSTRY_COLUMN] = fallback[INDUSTRY_COLUMN].to_numpy()

    merged[INDUSTRY_COLUMN] = pd.to_numeric(merged[INDUSTRY_COLUMN], errors="coerce")
    merged["industry_pred_available"] = merged[INDUSTRY_COLUMN].notna()
    return merged


def build_pairwise_mean_matrix(frame: pd.DataFrame) -> np.ndarray:
    if frame.empty:
        return np.zeros((len(PRINCIPLE_COLUMNS), len(PRINCIPLE_COLUMNS)), dtype=float)
    values = frame[PRINCIPLE_COLUMNS].apply(pd.to_numeric, errors="coerce").fillna(0.0).to_numpy(dtype=float)
    size = len(PRINCIPLE_COLUMNS)
    matrix = np.zeros((size, size), dtype=float)
    for i in range(size):
        for j in range(size):
            if i == j:
                matrix[i, j] = float(values[:, i].mean())
            else:
                matrix[i, j] = matrix[j, i]
        for j in range(i + 1, size):
            pair_value = float((values[:, i] * values[:, j]).mean())
            matrix[i, j] = pair_value
            matrix[j, i] = pair_value
    return matrix


def export_heatmap_table(matrices: dict[str, np.ndarray]) -> None:
    rows: list[dict[str, object]] = []
    for detector_name, matrix in matrices.items():
        for i, feature_a in enumerate(PRINCIPLE_LABELS):
            for j, feature_b in enumerate(PRINCIPLE_LABELS):
                rows.append(
                    {
                        "detector": detector_name,
                        "feature_a": feature_a,
                        "feature_b": feature_b,
                        "mean_strength": None if np.isnan(matrix[i, j]) else float(matrix[i, j]),
                    }
                )
    pd.DataFrame(rows).to_csv(HEATMAP_TABLE_OUTPUT, index=False)


def export_difference_table(matrix: np.ndarray, title: str) -> None:
    rows: list[dict[str, object]] = []
    for i, feature_a in enumerate(PRINCIPLE_LABELS):
        for j, feature_b in enumerate(PRINCIPLE_LABELS):
            rows.append(
                {
                    "comparison": title,
                    "feature_a": feature_a,
                    "feature_b": feature_b,
                    "delta_mean_strength": None if np.isnan(matrix[i, j]) else float(matrix[i, j]),
                }
            )
    pd.DataFrame(rows).to_csv(DIFFERENCE_TABLE_OUTPUT, index=False)


def draw_dual_heatmap(matrices: dict[str, np.ndarray], counts: dict[str, int]) -> None:
    finite_values = []
    for matrix in matrices.values():
        finite_values.extend(matrix[np.isfinite(matrix)].tolist())
    vmax = max(finite_values) if finite_values else 1e-6

    figure = plt.figure(figsize=(11.3, 4.9))
    grid = figure.add_gridspec(1, 2, width_ratios=[1.0, 1.0], wspace=0.05)
    axes = [figure.add_subplot(grid[0, 0]), figure.add_subplot(grid[0, 1])]
    images = []
    for axis, detector_name, show_y_labels in zip(
        axes,
        [ACADEMIC_LABEL, INDUSTRY_LABEL],
        [True, False],
        strict=True,
    ):
        matrix = matrices[detector_name]
        image = axis.imshow(matrix, cmap=HEATMAP_CMAP, vmin=0.0, vmax=vmax)
        images.append(image)
        axis.set_xticks(range(len(PRINCIPLE_LABELS)))
        axis.set_yticks(range(len(PRINCIPLE_LABELS)))
        axis.set_xticklabels(
            PRINCIPLE_LABELS,
            rotation=18,
            ha="right",
            fontsize=DUAL_HEATMAP_XTICK_FONT_SIZE,
        )
        axis.tick_params(axis="x", length=0, colors="#222222")
        axis.set_yticklabels(PRINCIPLE_LABELS if show_y_labels else [], fontsize=15, color="#222222")
        axis.set_title(f"{detector_name} LLM-TP", fontsize=13.5, pad=10)

        for i in range(matrix.shape[0]):
            for j in range(matrix.shape[1]):
                value = matrix[i, j]
                if not np.isfinite(value) or j > i:
                    continue
                text_color = "white" if value > vmax * 0.58 else "black"
                axis.text(
                    j,
                    i,
                    f"{value:.2f}",
                    ha="center",
                    va="center",
                    fontsize=DUAL_HEATMAP_CELL_FONT_SIZE,
                    color=text_color,
                )

        for i in range(matrix.shape[0]):
            for j in range(matrix.shape[1]):
                if j > i:
                    axis.add_patch(
                        plt.Rectangle(
                            (j - 0.5, i - 0.5),
                            1.0,
                            1.0,
                            facecolor="#ffffff",
                            edgecolor="#ffffff",
                            linewidth=0.0,
                            zorder=3,
                        )
                    )

        axis.set_xlim(-0.5, len(PRINCIPLE_LABELS) - 0.5)
        axis.set_ylim(len(PRINCIPLE_LABELS) - 0.5, -0.5)
        axis.grid(False)
        axis.spines["top"].set_visible(False)
        axis.spines["right"].set_visible(False)

    figure.subplots_adjust(left=0.10, right=0.88, bottom=0.01, top=0.995)

    heatmap_pos = axes[1].get_position()
    cax = figure.add_axes([heatmap_pos.x1 + 0.018, heatmap_pos.y0, 0.022, heatmap_pos.height])
    colorbar = figure.colorbar(images[-1], cax=cax)
    colorbar.set_label("Mean co-occurrence strength", fontsize=DUAL_HEATMAP_COLORBAR_LABEL_FONT_SIZE)
    tick_step = 0.10 if vmax > 0.60 else 0.05
    colorbar_ticks = np.arange(0.0, vmax + 1e-9, tick_step)
    if len(colorbar_ticks) == 0 or not np.isclose(colorbar_ticks[-1], vmax):
        colorbar_ticks = np.append(colorbar_ticks, vmax)
    colorbar.set_ticks(colorbar_ticks)
    colorbar.ax.yaxis.set_major_formatter(FormatStrFormatter("%.2f"))
    colorbar.ax.tick_params(labelsize=10.5)
    figure.savefig(HEATMAP_OUTPUT, dpi=260, bbox_inches="tight", pad_inches=0.01)
    plt.close(figure)


def draw_difference_heatmap(matrix: np.ndarray, title: str) -> None:
    finite_values = matrix[np.isfinite(matrix)]
    abs_values = np.abs(finite_values)
    robust_limit = float(np.quantile(abs_values, 0.85)) if finite_values.size else 0.10
    robust_limit = max(robust_limit, 0.10)
    limit = float(np.ceil(robust_limit / 0.05) * 0.05)

    figure = plt.figure(figsize=(6.0, 4.9))
    axis = figure.add_subplot(111)
    white_band = 0.015
    display_matrix = np.array(matrix, copy=True)
    display_matrix[np.abs(display_matrix) <= white_band] = 0.0
    image = axis.imshow(
        display_matrix,
        cmap=DIFFERENCE_CMAP,
        norm=TwoSlopeNorm(vmin=-limit, vcenter=0.0, vmax=limit),
        aspect="equal",
    )
    axis.set_xticks(range(len(PRINCIPLE_LABELS)))
    axis.set_yticks(range(len(PRINCIPLE_LABELS)))
    axis.set_xticklabels(PRINCIPLE_LABELS, rotation=13, ha="right", fontsize=13.5)
    axis.tick_params(axis="x", length=0, colors="#222222")
    axis.set_yticklabels(PRINCIPLE_LABELS, fontsize=15, color="#222222")
    axis.set_title(title, fontsize=13.5, pad=10)

    for i in range(matrix.shape[0]):
        for j in range(matrix.shape[1]):
            value = matrix[i, j]
            if not np.isfinite(value) or j > i:
                continue
            text_color = "white" if abs(value) > limit * 0.45 else "black"
            axis.text(
                j,
                i,
                f"{value:.2f}",
                ha="center",
                va="center",
                fontsize=12.5,
                color=text_color,
            )

    for i in range(matrix.shape[0]):
        for j in range(matrix.shape[1]):
            if j > i:
                axis.add_patch(
                    plt.Rectangle(
                        (j - 0.5, i - 0.5),
                        1.0,
                        1.0,
                        facecolor="#ffffff",
                        edgecolor="#ffffff",
                        linewidth=0.0,
                        zorder=3,
                    )
                )

    axis.set_xlim(-0.5, len(PRINCIPLE_LABELS) - 0.5)
    axis.set_ylim(len(PRINCIPLE_LABELS) - 0.5, -0.5)
    axis.grid(False)
    axis.spines["top"].set_visible(False)
    axis.spines["right"].set_visible(False)

    figure.subplots_adjust(left=0.12, right=0.84, bottom=0.01, top=0.995)
    heatmap_pos = axis.get_position()
    cax = figure.add_axes([heatmap_pos.x1 + 0.02, heatmap_pos.y0, 0.028, heatmap_pos.height])
    colorbar = figure.colorbar(image, cax=cax)
    colorbar.set_label("Delta mean co-occurrence strength", fontsize=12)
    tick_step = 0.05 if limit <= 0.30 else 0.10
    colorbar_ticks = np.arange(-limit, limit + 1e-9, tick_step)
    colorbar.set_ticks(colorbar_ticks)
    colorbar.ax.yaxis.set_major_formatter(FormatStrFormatter("%.2f"))
    colorbar.ax.tick_params(labelsize=10.5)
    figure.savefig(DIFFERENCE_HEATMAP_OUTPUT, dpi=260, bbox_inches="tight", pad_inches=0.01)
    plt.close(figure)


def top_indices(values: np.ndarray, k: int) -> np.ndarray:
    if len(values) == 0:
        return np.array([], dtype=int)
    k = min(k, len(values))
    order = np.argsort(values)
    return order[-k:][::-1]


def compute_curated_items(
    docs: pd.DataFrame,
    bg_docs: pd.DataFrame,
    doc_weights: np.ndarray,
    bg_weights: np.ndarray,
) -> tuple[list[str], list[str]]:
    group_word_counter = aggregate_weighted_terms(docs, doc_weights, "unigram_counts")
    bg_word_counter = aggregate_weighted_terms(bg_docs, bg_weights, "unigram_counts")
    group_bigram_counter = aggregate_weighted_terms(docs, doc_weights, "bigram_counts")
    bg_bigram_counter = aggregate_weighted_terms(bg_docs, bg_weights, "bigram_counts")
    group_trigram_counter = aggregate_weighted_terms(docs, doc_weights, "trigram_counts")
    bg_trigram_counter = aggregate_weighted_terms(bg_docs, bg_weights, "trigram_counts")
    group_word_docs = aggregate_doc_presence(docs, "unigram_counts")
    bg_word_docs = aggregate_doc_presence(bg_docs, "unigram_counts")
    group_phrase_docs = aggregate_doc_presence(docs, "bigram_counts") + aggregate_doc_presence(docs, "trigram_counts")
    bg_phrase_docs = aggregate_doc_presence(bg_docs, "bigram_counts") + aggregate_doc_presence(bg_docs, "trigram_counts")

    top_words = weighted_log_ratio(
        group_word_counter,
        bg_word_counter,
        top_n=TOP_WORDS,
        min_group_count=2.5,
        min_group_docs=3,
        min_score=0.18,
        filter_fn=token_is_meaningful,
        group_doc_counter=group_word_docs,
        bg_doc_counter=bg_word_docs,
    )
    top_phrases = weighted_log_ratio(
        group_bigram_counter + group_trigram_counter,
        bg_bigram_counter + bg_trigram_counter,
        top_n=TOP_PHRASES,
        min_group_count=1.5,
        min_group_docs=2,
        min_score=0.15,
        filter_fn=phrase_is_meaningful,
        group_doc_counter=group_phrase_docs,
        bg_doc_counter=bg_phrase_docs,
    )
    return [term for term, *_ in top_words], [term for term, *_ in top_phrases]


def build_evidence_table(frame_by_detector: dict[str, pd.DataFrame]) -> pd.DataFrame:
    cell_specs = build_all_cell_specs()
    groups = [ACADEMIC_LABEL, INDUSTRY_LABEL]
    combined = []
    for detector_name, frame in frame_by_detector.items():
        temp = frame[["subject", "body", *PRINCIPLE_COLUMNS]].copy()
        temp["group_label"] = detector_name
        combined.append(temp)
    combined_frame = pd.concat(combined, ignore_index=True)
    combined_frame = add_document_features(combined_frame)

    rows: list[dict[str, object]] = []

    for cell_label, row_slug, col_slug in cell_specs:
        row_idx = PRINCIPLE_INDEX[row_slug]
        col_idx = PRINCIPLE_INDEX[col_slug]
        row_record = {
            "cell": cell_label,
            "feature_a": next(label for label, slug in zip(PRINCIPLE_LABELS, PRINCIPLE_SLUGS, strict=True) if slug == row_slug),
            "feature_b": next(label for label, slug in zip(PRINCIPLE_LABELS, PRINCIPLE_SLUGS, strict=True) if slug == col_slug),
        }
        for group in groups:
            group_docs = combined_frame[combined_frame["group_label"] == group].copy().reset_index(drop=True)
            bg_docs = combined_frame[combined_frame["group_label"] != group].copy().reset_index(drop=True)
            group_contrib = compute_cell_contribution(group_docs, row_idx, col_idx)
            bg_contrib = compute_cell_contribution(bg_docs, row_idx, col_idx)
            top_group_idx = top_indices(group_contrib, TOP_DOCS)
            top_bg_idx = top_indices(bg_contrib, TOP_DOCS)
            selected_group_docs = group_docs.iloc[top_group_idx].reset_index(drop=True)
            selected_bg_docs = bg_docs.iloc[top_bg_idx].reset_index(drop=True)
            selected_group_weights = group_contrib[top_group_idx] if len(top_group_idx) else np.array([], dtype=float)
            selected_bg_weights = bg_contrib[top_bg_idx] if len(top_bg_idx) else np.array([], dtype=float)
            words, phrases = compute_curated_items(
                selected_group_docs,
                selected_bg_docs,
                selected_group_weights,
                selected_bg_weights,
            )
            row_record[f"{group}_words"] = "; ".join(words[:TOP_ITEMS_FOR_TABLE])
            row_record[f"{group}_phrases"] = "; ".join(phrases[:TOP_ITEMS_FOR_TABLE])
        rows.append(row_record)

    table = pd.DataFrame(rows)
    table.to_csv(EVIDENCE_TABLE_OUTPUT, index=False)
    return table


def write_evidence_markdown(table: pd.DataFrame) -> None:
    lines = [
        "# Selected LLM-TP Words and Phrases by Feature Pair",
        "",
        f"Academic detector: `{ACADEMIC_LABEL}`",
        f"Industrial detector: `{INDUSTRY_LABEL}`",
        "",
        "Each row corresponds to one lower-triangle WVAE feature pair cell.",
        "",
        "| Cell | Feature A | Feature B | Academic Words | Academic Phrases | Industrial Words | Industrial Phrases |",
        "|---|---|---|---|---|---|---|",
    ]
    for _, row in table.iterrows():
        values = [
            str(row.get("cell", "") or ""),
            str(row.get("feature_a", "") or ""),
            str(row.get("feature_b", "") or ""),
            str(row.get(f"{ACADEMIC_LABEL}_words", "") or ""),
            str(row.get(f"{ACADEMIC_LABEL}_phrases", "") or ""),
            str(row.get(f"{INDUSTRY_LABEL}_words", "") or ""),
            str(row.get(f"{INDUSTRY_LABEL}_phrases", "") or ""),
        ]
        escaped = [value.replace("|", "\\|").replace("\n", " ") for value in values]
        lines.append("| " + " | ".join(escaped) + " |")
    EVIDENCE_MARKDOWN_OUTPUT.write_text("\n".join(lines) + "\n", encoding="utf-8")


def draw_evidence_table_image(table: pd.DataFrame) -> bool:
    if len(table) > 12:
        return False
    display = table[
        [
            "cell",
            f"{ACADEMIC_LABEL}_words",
            f"{ACADEMIC_LABEL}_phrases",
            f"{INDUSTRY_LABEL}_words",
            f"{INDUSTRY_LABEL}_phrases",
        ]
    ].copy()
    display.columns = [
        "Cell",
        f"{ACADEMIC_LABEL} Words",
        f"{ACADEMIC_LABEL} Phrases",
        f"{INDUSTRY_LABEL} Words",
        f"{INDUSTRY_LABEL} Phrases",
    ]
    for column in display.columns[1:]:
        display[column] = display[column].fillna("").apply(
            lambda text: "\n".join(
                textwrap.fill(item, width=20)
                for item in [piece for piece in str(text).split("; ") if piece]
            )
        )

    line_counts = []
    for _, row in display.iterrows():
        max_lines = max(str(row[column]).count("\n") + 1 for column in display.columns[1:])
        line_counts.append(max_lines)

    figure_height = 1.2 + sum(max(1.2, 0.42 * count) for count in line_counts)
    figure, axis = plt.subplots(figsize=(18.5, figure_height))
    figure.patch.set_facecolor("#ffffff")
    axis.axis("off")

    table_artist = axis.table(
        cellText=display.values,
        colLabels=display.columns,
        loc="center",
        cellLoc="left",
        colLoc="center",
    )
    table_artist.auto_set_font_size(False)
    table_artist.set_fontsize(11.0)

    col_widths = {
        0: 0.13,
        1: 0.215,
        2: 0.22,
        3: 0.215,
        4: 0.22,
    }
    header_height = 0.11
    row_heights = [max(0.17, 0.045 * count + 0.06) for count in line_counts]

    for (row, col), cell in table_artist.get_celld().items():
        cell.set_edgecolor("#8a8a8a")
        cell.set_linewidth(0.7)
        if col in col_widths:
            cell.set_width(col_widths[col])
        if row == 0:
            cell.set_height(header_height)
            cell.set_facecolor("#f1f4f6")
            cell.set_text_props(weight="bold", color="#1f1f1f", ha="center")
        else:
            cell.set_height(row_heights[row - 1])
        if row > 0 and col == 0:
            cell.set_facecolor("#fafafa")
            cell.set_text_props(weight="bold", ha="center", va="center")
        elif row > 0:
            cell.set_facecolor("#ffffff")
            cell.set_text_props(ha="left", va="center")

    figure.subplots_adjust(left=0.02, right=0.98, top=0.98, bottom=0.02)
    figure.savefig(EVIDENCE_TABLE_IMAGE, dpi=240, bbox_inches="tight", facecolor=figure.get_facecolor())
    plt.close(figure)
    return True


def main() -> None:
    ROOT.mkdir(parents=True, exist_ok=True)

    projected = load_projected_points()
    lookup = load_industry_lookup()
    merged = attach_industry_predictions(projected, lookup)
    merged = merged[merged["industry_pred_available"]].copy()
    merged.to_csv(MERGED_OUTPUT, index=False)

    llm_phishing = merged[
        merged["source"].astype(str).eq("LLM") & merged["raw_label"].astype(int).eq(1)
    ].copy()

    academic_llm_tp = llm_phishing[normalize_binary(llm_phishing[ACADEMIC_COLUMN]) >= 0.5].copy()
    industry_llm_tp = llm_phishing[normalize_binary(llm_phishing[INDUSTRY_COLUMN]) >= 0.5].copy()

    matrices = {
        ACADEMIC_LABEL: build_pairwise_mean_matrix(academic_llm_tp),
        INDUSTRY_LABEL: build_pairwise_mean_matrix(industry_llm_tp),
    }
    counts = {
        ACADEMIC_LABEL: int(len(academic_llm_tp)),
        INDUSTRY_LABEL: int(len(industry_llm_tp)),
    }

    export_heatmap_table(matrices)
    draw_dual_heatmap(matrices, counts)
    difference_title = f"{ACADEMIC_LABEL} - {INDUSTRY_LABEL}"
    difference_matrix = matrices[ACADEMIC_LABEL] - matrices[INDUSTRY_LABEL]
    export_difference_table(difference_matrix, difference_title)
    draw_difference_heatmap(difference_matrix, difference_title)

    evidence_table = build_evidence_table(
        {
            ACADEMIC_LABEL: academic_llm_tp,
            INDUSTRY_LABEL: industry_llm_tp,
        }
    )
    write_evidence_markdown(evidence_table)
    image_generated = draw_evidence_table_image(evidence_table)

    summary = {
        "academic_detector": ACADEMIC_LABEL,
        "industry_detector": INDUSTRY_LABEL,
        "llm_tp_counts": counts,
        "heatmap_output": HEATMAP_OUTPUT.name,
        "difference_heatmap_output": DIFFERENCE_HEATMAP_OUTPUT.name,
        "heatmap_table_output": HEATMAP_TABLE_OUTPUT.name,
        "difference_table_output": DIFFERENCE_TABLE_OUTPUT.name,
        "evidence_table_output": EVIDENCE_TABLE_OUTPUT.name,
        "evidence_markdown_output": EVIDENCE_MARKDOWN_OUTPUT.name,
        "evidence_table_image": EVIDENCE_TABLE_IMAGE.name if image_generated else None,
        "merged_output": MERGED_OUTPUT.name,
        "exported_cells": [cell_label for cell_label, _, _ in build_all_cell_specs()],
    }
    SUMMARY_OUTPUT.write_text(json.dumps(summary, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
