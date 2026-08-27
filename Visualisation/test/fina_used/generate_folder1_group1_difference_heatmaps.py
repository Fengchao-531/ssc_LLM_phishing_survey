#!/usr/bin/env python3
from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.colors import LinearSegmentedColormap, TwoSlopeNorm
from matplotlib.ticker import FormatStrFormatter
from scipy.stats import mannwhitneyu


SCRIPT_DIR = Path(__file__).resolve().parent
OVERVIEW_INPUT = SCRIPT_DIR.parent / "overview" / "projected_points_mixed_overview.csv"
OUTPUT_DIR = SCRIPT_DIR / "1"
OUTPUT_PATH = OUTPUT_DIR / "group1_difference_heatmaps.png"
P_VALUE_OUTPUT = OUTPUT_DIR / "group1_difference_heatmaps_p_values.csv"

PRINCIPLE_COLUMNS = [
    "principle_authority",
    "principle_liking",
    "principle_reciprocity",
    "principle_social_proof",
    "principle_scarcity",
    "principle_commitment",
]
PRINCIPLE_LABELS = [
    "Authority",
    "Liking",
    "Reciprocity",
    "Social Proof",
    "Scarcity",
    "Commitment",
]
X_AXIS_LABEL_FONT_SIZE = 33
X_TICK_LABEL_FONT_SIZE = 15.5
Y_TICK_LABEL_FONT_SIZE = 17
CELL_VALUE_FONT_SIZE = 15
COLORBAR_TICK_FONT_SIZE = 13.5


def compute_principle_cooccurrence_matrix(frame: pd.DataFrame) -> tuple[np.ndarray, int]:
    principle_frame = frame[PRINCIPLE_COLUMNS].dropna(how="all")
    if principle_frame.empty:
        return np.full((len(PRINCIPLE_COLUMNS), len(PRINCIPLE_COLUMNS)), np.nan, dtype=float), 0
    values = principle_frame.to_numpy(dtype=float)
    matrix = (values.T @ values) / max(1, values.shape[0])
    diagonal = np.nanmean(values, axis=0)
    np.fill_diagonal(matrix, diagonal)
    return matrix, int(values.shape[0])


def compute_cell_distributions(frame: pd.DataFrame) -> tuple[list[list[np.ndarray]], int]:
    principle_frame = frame[PRINCIPLE_COLUMNS].apply(pd.to_numeric, errors="coerce").dropna(how="all")
    if principle_frame.empty:
        empty = [[np.array([], dtype=float) for _ in PRINCIPLE_COLUMNS] for _ in PRINCIPLE_COLUMNS]
        return empty, 0

    values = principle_frame.to_numpy(dtype=float)
    distributions: list[list[np.ndarray]] = []
    for row_index in range(len(PRINCIPLE_COLUMNS)):
        row_distributions: list[np.ndarray] = []
        for col_index in range(len(PRINCIPLE_COLUMNS)):
            if row_index == col_index:
                cell_values = values[:, row_index]
                valid = cell_values[np.isfinite(cell_values)]
            else:
                cell_values = values[:, row_index] * values[:, col_index]
                valid = cell_values[np.isfinite(cell_values)]
            row_distributions.append(valid.astype(float, copy=False))
        distributions.append(row_distributions)
    return distributions, int(values.shape[0])


def compute_p_value_matrix(
    left_distributions: list[list[np.ndarray]],
    right_distributions: list[list[np.ndarray]],
) -> np.ndarray:
    p_values = np.full((len(PRINCIPLE_COLUMNS), len(PRINCIPLE_COLUMNS)), np.nan, dtype=float)
    for row_index in range(len(PRINCIPLE_COLUMNS)):
        for col_index in range(len(PRINCIPLE_COLUMNS)):
            left_values = left_distributions[row_index][col_index]
            right_values = right_distributions[row_index][col_index]
            if len(left_values) == 0 or len(right_values) == 0:
                continue
            try:
                _, p_value = mannwhitneyu(left_values, right_values, alternative="two-sided")
            except ValueError:
                continue
            p_values[row_index, col_index] = float(p_value)
    return p_values


def compute_significance_matrix(
    p_values: np.ndarray,
    *,
    alpha: float = 0.05,
) -> np.ndarray:
    return np.isfinite(p_values) & (p_values < alpha)


def build_difference_cmap() -> LinearSegmentedColormap:
    return LinearSegmentedColormap.from_list(
        "deep_overview_difference",
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


def draw_difference_axis(
    axis: plt.Axes,
    matrix: np.ndarray,
    *,
    title: str,
    show_y_labels: bool,
    limit: float,
) -> object:
    white_band = 0.015
    norm = TwoSlopeNorm(vmin=-limit, vcenter=0.0, vmax=limit)
    display_matrix = np.array(matrix, copy=True)
    display_matrix[np.abs(display_matrix) <= white_band] = 0.0
    image = axis.imshow(display_matrix, cmap=build_difference_cmap(), norm=norm, aspect="equal")
    if title:
        axis.set_title(title, fontsize=13.5, pad=10)
    axis.set_xticks(range(len(PRINCIPLE_LABELS)))
    axis.set_xticklabels(PRINCIPLE_LABELS, rotation=18, ha="right", fontsize=X_TICK_LABEL_FONT_SIZE)
    axis.tick_params(axis="x", length=0, colors="#222222")
    axis.set_yticks(range(len(PRINCIPLE_LABELS)))
    axis.set_yticklabels(
        PRINCIPLE_LABELS if show_y_labels else [],
        fontsize=Y_TICK_LABEL_FONT_SIZE,
        color="#222222",
    )

    for row_index in range(matrix.shape[0]):
        for col_index in range(matrix.shape[1]):
            if col_index > row_index:
                continue
            value = matrix[row_index, col_index]
            if not np.isfinite(value):
                continue
            text_color = "white" if abs(value) > limit * 0.45 else "black"
            axis.text(
                col_index,
                row_index,
                f"{value:.2f}",
                ha="center",
                va="center",
                fontsize=CELL_VALUE_FONT_SIZE,
                color=text_color,
            )

    upper_triangle_mask = np.triu(np.ones_like(matrix, dtype=bool), k=1)
    for row_index in range(matrix.shape[0]):
        for col_index in range(matrix.shape[1]):
            if upper_triangle_mask[row_index, col_index]:
                axis.add_patch(
                    plt.Rectangle(
                        (col_index - 0.5, row_index - 0.5),
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
    return image


def export_p_value_table(rows: list[tuple[str, np.ndarray]]) -> None:
    exported_rows: list[dict[str, object]] = []
    for comparison_name, p_values in rows:
        significance = compute_significance_matrix(p_values)
        for row_index, feature_a in enumerate(PRINCIPLE_LABELS):
            for col_index, feature_b in enumerate(PRINCIPLE_LABELS):
                if col_index > row_index:
                    continue
                p_value = p_values[row_index, col_index]
                exported_rows.append(
                    {
                        "comparison": comparison_name,
                        "feature_a": feature_a,
                        "feature_b": feature_b,
                        "p_value": None if not np.isfinite(p_value) else float(p_value),
                        "significant_at_0_05": bool(significance[row_index, col_index]),
                    }
                )
    pd.DataFrame(exported_rows).to_csv(P_VALUE_OUTPUT, index=False)


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    frame = pd.read_csv(OVERVIEW_INPUT, low_memory=False)

    llm_p_fn = frame[(frame["source"] == "LLM") & frame["is_fn_phishing"].astype(bool)].copy()
    hw_p_fn = frame[(frame["source"] == "HW") & frame["is_fn_phishing"].astype(bool)].copy()
    llm_p_tp = frame[(frame["source"] == "LLM") & frame["is_tp_phishing"].astype(bool)].copy()

    llm_p_fn_matrix, llm_p_fn_count = compute_principle_cooccurrence_matrix(llm_p_fn)
    hw_p_fn_matrix, hw_p_fn_count = compute_principle_cooccurrence_matrix(hw_p_fn)
    llm_p_tp_matrix, llm_p_tp_count = compute_principle_cooccurrence_matrix(llm_p_tp)

    llm_p_fn_distributions, _ = compute_cell_distributions(llm_p_fn)
    hw_p_fn_distributions, _ = compute_cell_distributions(hw_p_fn)
    llm_p_tp_distributions, _ = compute_cell_distributions(llm_p_tp)

    llm_vs_hw_p_values = compute_p_value_matrix(llm_p_fn_distributions, hw_p_fn_distributions)
    llm_fn_vs_tp_p_values = compute_p_value_matrix(llm_p_fn_distributions, llm_p_tp_distributions)

    differences = [
        ("LLM-P FN - HW-P FN", llm_p_fn_matrix - hw_p_fn_matrix, llm_vs_hw_p_values, llm_p_fn_count, hw_p_fn_count),
        (
            "LLM-P FN - LLM-P TP",
            llm_p_fn_matrix - llm_p_tp_matrix,
            llm_fn_vs_tp_p_values,
            llm_p_fn_count,
            llm_p_tp_count,
        ),
    ]

    export_p_value_table(
        [
            ("LLM-P FN - HW-P FN", llm_vs_hw_p_values),
            ("LLM-P FN - LLM-P TP", llm_fn_vs_tp_p_values),
        ]
    )

    finite_values = np.concatenate(
        [matrix[np.isfinite(matrix)].ravel() for _, matrix, _, _, _ in differences]
    )
    abs_values = np.abs(finite_values)
    robust_limit = float(np.quantile(abs_values, 0.85))
    robust_limit = max(robust_limit, 0.10)
    limit = float(np.ceil(robust_limit / 0.05) * 0.05)

    figure = plt.figure(figsize=(11.3, 4.9))
    grid = figure.add_gridspec(1, 2, width_ratios=[1.0, 1.0], wspace=0.05)
    axes = [figure.add_subplot(grid[0, 0]), figure.add_subplot(grid[0, 1])]

    images = []
    for axis, (title, matrix, p_values, left_count, right_count), show_y_labels in zip(
        axes,
        differences,
        [True, False],
        strict=True,
    ):
        images.append(
            draw_difference_axis(
                axis,
                matrix,
                title=title,
                show_y_labels=show_y_labels,
                limit=limit,
            )
        )

    figure.subplots_adjust(left=0.10, right=0.88, bottom=0.01, top=0.995)

    heatmap_pos = axes[1].get_position()
    colorbar_width = 0.022
    colorbar_height = heatmap_pos.height
    colorbar_y0 = heatmap_pos.y0
    colorbar_x0 = heatmap_pos.x1 + 0.018
    colorbar_axis = figure.add_axes(
        [colorbar_x0, colorbar_y0, colorbar_width, colorbar_height]
    )

    colorbar = figure.colorbar(images[-1], cax=colorbar_axis)
    colorbar.set_label("Delta mean co-occurrence strength", fontsize=12)
    tick_step = 0.05 if limit <= 0.30 else 0.10
    colorbar_ticks = np.arange(-limit, limit + 1e-9, tick_step)
    colorbar.set_ticks(colorbar_ticks)
    colorbar.ax.yaxis.set_major_formatter(FormatStrFormatter("%.2f"))
    colorbar.ax.tick_params(labelsize=COLORBAR_TICK_FONT_SIZE)

    figure.savefig(OUTPUT_PATH, dpi=260, bbox_inches="tight", pad_inches=0.01)
    plt.close(figure)


if __name__ == "__main__":
    main()
