#!/usr/bin/env python3
from __future__ import annotations

from pathlib import Path
import sys

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.colors import LinearSegmentedColormap, TwoSlopeNorm
from matplotlib.lines import Line2D
from matplotlib.ticker import FormatStrFormatter
from scipy.stats import mannwhitneyu


SCRIPT_DIR = Path(__file__).resolve().parent
VISUALIZATION_DIR = SCRIPT_DIR.parents[1]
TEST_DIR = VISUALIZATION_DIR / "test"
OUTPUT_DIR = SCRIPT_DIR / "9"
OUTPUT_PNG = OUTPUT_DIR / "surrogate_contour_with_llm_hw_fn_heatmap.png"
OUTPUT_PDF = OUTPUT_DIR / "surrogate_contour_with_llm_hw_fn_heatmap.pdf"
OUTPUT_HEATMAP_VALUES = OUTPUT_DIR / "llm_p_fn_minus_hw_p_fn_heatmap_values.csv"
OUTPUT_HEATMAP_P_VALUES = OUTPUT_DIR / "llm_p_fn_minus_hw_p_fn_heatmap_p_values.csv"

PROJECTED_POINTS = TEST_DIR / "projected_points.csv"
OVERVIEW_INPUT = TEST_DIR / "overview" / "projected_points_mixed_overview.csv"

if str(VISUALIZATION_DIR) not in sys.path:
    sys.path.insert(0, str(VISUALIZATION_DIR))

from generate_stage_visualizations import fit_detector_surface  # noqa: E402
from generate_test_failure_contours import add_background, build_global_grid  # noqa: E402


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

TP_STYLE = {
    "HW": {"marker": "o", "facecolors": "none", "edgecolors": "#2f2f2f", "linewidths": 0.70, "s": 18, "alpha": 0.76},
    "LLM": {"marker": "^", "facecolors": "none", "edgecolors": "#3d3d3d", "linewidths": 0.70, "s": 20, "alpha": 0.76},
}
FN_STYLE = {
    "HW": {"marker": "o", "c": "#2ca02c", "edgecolors": "white", "linewidths": 0.22, "s": 20, "alpha": 0.88},
    "LLM": {"marker": "^", "c": "#3498db", "edgecolors": "white", "linewidths": 0.22, "s": 22, "alpha": 0.86},
}


def build_difference_cmap() -> LinearSegmentedColormap:
    return LinearSegmentedColormap.from_list(
        "combined_delta_blue_yellow",
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


def load_contour_frame() -> pd.DataFrame:
    frame = pd.read_csv(PROJECTED_POINTS, low_memory=False).copy()
    detector_prediction = pd.to_numeric(frame["scamllm"], errors="coerce").fillna(0.0)
    frame["is_tp"] = detector_prediction >= 0.5
    frame["is_fn"] = ~frame["is_tp"]
    return frame


def compute_principle_cooccurrence_matrix(frame: pd.DataFrame) -> tuple[np.ndarray, int]:
    principle_frame = frame[PRINCIPLE_COLUMNS].apply(pd.to_numeric, errors="coerce").dropna(how="all")
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
            else:
                cell_values = values[:, row_index] * values[:, col_index]
            row_distributions.append(cell_values[np.isfinite(cell_values)].astype(float, copy=False))
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
            _, p_value = mannwhitneyu(left_values, right_values, alternative="two-sided")
            p_values[row_index, col_index] = float(p_value)
    return p_values


def stars_for_p(p_value: float) -> str:
    if not np.isfinite(p_value):
        return ""
    if p_value < 0.001:
        return "***"
    if p_value < 0.01:
        return "**"
    if p_value < 0.05:
        return "*"
    return ""


def draw_contour_panel(axis: plt.Axes, figure: plt.Figure) -> object:
    frame = load_contour_frame()
    surface_model = fit_detector_surface(frame, seed=7)
    if surface_model is None:
        raise RuntimeError("Could not fit the surrogate surface.")
    _, _, grid_x, grid_y, score_grid = build_global_grid(frame, surface_model)
    filled = add_background(axis, grid_x, grid_y, score_grid, "pca")

    for source in ["HW", "LLM"]:
        tp_frame = frame[(frame["source"] == source) & frame["is_tp"]]
        fn_frame = frame[(frame["source"] == source) & frame["is_fn"]]
        axis.scatter(tp_frame["proj_x"], tp_frame["proj_y"], zorder=3.0, **TP_STYLE[source])
        axis.scatter(fn_frame["proj_x"], fn_frame["proj_y"], zorder=4.0, **FN_STYLE[source])

    axis.set_xlabel("PCA1", fontsize=24)
    axis.set_ylabel("PCA2", fontsize=24)
    axis.tick_params(axis="both", labelsize=19)
    axis.grid(alpha=0.10, linewidth=0.5)

    handles = [
        Line2D([0], [0], marker="o", color="none", markerfacecolor="none", markeredgecolor="#2f2f2f", markersize=10.8, linewidth=0, label="HW-TP"),
        Line2D([0], [0], marker="o", color="none", markerfacecolor="#2ca02c", markeredgecolor="white", markersize=10.8, linewidth=0, label="HW-FN"),
        Line2D([0], [0], marker="^", color="none", markerfacecolor="none", markeredgecolor="#3d3d3d", markersize=11.0, linewidth=0, label="LLM-TP"),
        Line2D([0], [0], marker="^", color="none", markerfacecolor="#3498db", markeredgecolor="white", markersize=11.0, linewidth=0, label="LLM-FN"),
    ]
    axis.legend(
        handles=handles,
        loc="upper center",
        bbox_to_anchor=(0.49, 0.99),
        ncol=2,
        frameon=True,
        facecolor="white",
        edgecolor="#d0d0d0",
        framealpha=0.90,
        fontsize=16.0,
        columnspacing=0.80,
        handletextpad=0.42,
        labelspacing=0.38,
    )
    return filled


def build_heatmap_inputs() -> tuple[np.ndarray, np.ndarray, float]:
    frame = pd.read_csv(OVERVIEW_INPUT, low_memory=False)
    llm_p_fn = frame[(frame["source"] == "LLM") & frame["is_fn_phishing"].astype(bool)].copy()
    hw_p_fn = frame[(frame["source"] == "HW") & frame["is_fn_phishing"].astype(bool)].copy()

    llm_matrix, _ = compute_principle_cooccurrence_matrix(llm_p_fn)
    hw_matrix, _ = compute_principle_cooccurrence_matrix(hw_p_fn)
    llm_dist, _ = compute_cell_distributions(llm_p_fn)
    hw_dist, _ = compute_cell_distributions(hw_p_fn)
    p_values = compute_p_value_matrix(llm_dist, hw_dist)
    difference = llm_matrix - hw_matrix

    rows = []
    p_rows = []
    for row_index, left in enumerate(PRINCIPLE_LABELS):
        for col_index, right in enumerate(PRINCIPLE_LABELS):
            if col_index > row_index:
                continue
            rows.append({"feature_a": left, "feature_b": right, "delta": difference[row_index, col_index]})
            p_rows.append(
                {
                    "feature_a": left,
                    "feature_b": right,
                    "p_value": p_values[row_index, col_index],
                    "significance": stars_for_p(p_values[row_index, col_index]),
                }
            )
    pd.DataFrame(rows).to_csv(OUTPUT_HEATMAP_VALUES, index=False)
    pd.DataFrame(p_rows).to_csv(OUTPUT_HEATMAP_P_VALUES, index=False)

    finite = np.abs(difference[np.isfinite(difference)])
    limit = max(float(np.ceil(np.quantile(finite, 0.85) / 0.05) * 0.05), 0.10)
    return difference, p_values, limit


def draw_heatmap_panel(axis: plt.Axes) -> object:
    matrix, p_values, limit = build_heatmap_inputs()
    white_band = 0.015
    display_matrix = np.array(matrix, copy=True)
    display_matrix[np.abs(display_matrix) <= white_band] = 0.0
    norm = TwoSlopeNorm(vmin=-limit, vcenter=0.0, vmax=limit)
    image = axis.imshow(display_matrix, cmap=build_difference_cmap(), norm=norm, aspect="equal")

    axis.set_xticks(range(len(PRINCIPLE_LABELS)))
    axis.set_xticklabels(PRINCIPLE_LABELS, rotation=28, ha="right", fontsize=19)
    axis.set_yticks(range(len(PRINCIPLE_LABELS)))
    axis.set_yticklabels(PRINCIPLE_LABELS, fontsize=20)
    axis.tick_params(axis="both", length=0, colors="#222222")
    axis.set_title("LLM-P FN - HW-P FN", fontsize=22, pad=8)

    for row_index in range(matrix.shape[0]):
        for col_index in range(matrix.shape[1]):
            if col_index > row_index:
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
                continue
            value = matrix[row_index, col_index]
            if not np.isfinite(value):
                continue
            text_color = "white" if abs(value) > limit * 0.45 else "black"
            rounded_value = round(float(value), 2)
            axis.text(
                col_index,
                row_index - 0.08,
                f"{value:.2f}",
                ha="center",
                va="center",
                fontsize=17,
                color=text_color,
                zorder=4,
            )
            stars = stars_for_p(p_values[row_index, col_index]) if rounded_value != 0.0 else ""
            if stars:
                axis.text(
                    col_index,
                    row_index + 0.19,
                    stars,
                    ha="center",
                    va="center",
                    fontsize=14,
                    color=text_color,
                    zorder=4,
                )

    axis.set_xlim(-0.5, len(PRINCIPLE_LABELS) - 0.5)
    axis.set_ylim(len(PRINCIPLE_LABELS) - 0.5, -0.5)
    axis.spines["top"].set_visible(False)
    axis.spines["right"].set_visible(False)
    return image


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    plt.rcParams.update(
        {
            "font.family": "serif",
            "font.serif": ["Times New Roman", "Times", "Nimbus Roman", "DejaVu Serif"],
            "axes.linewidth": 1.0,
        }
    )

    figure = plt.figure(figsize=(17.2, 6.2))
    contour_axis = figure.add_axes([0.055, 0.16, 0.385, 0.76])
    contour_cbar_axis = figure.add_axes([0.448, 0.16, 0.018, 0.76])
    heatmap_axis = figure.add_axes([0.558, 0.16, 0.315, 0.76])
    heatmap_cbar_axis = figure.add_axes([0.912, 0.16, 0.018, 0.76])

    contour_image = draw_contour_panel(contour_axis, figure)
    contour_axis.set_box_aspect(1.0)
    contour_cbar = figure.colorbar(contour_image, cax=contour_cbar_axis)
    contour_cbar.set_label("Surrogate score", fontsize=18)
    contour_cbar.ax.tick_params(labelsize=15)
    contour_cbar.ax.yaxis.set_ticks_position("left")
    contour_cbar.ax.yaxis.set_label_position("left")

    heatmap_image = draw_heatmap_panel(heatmap_axis)
    heatmap_cbar = figure.colorbar(heatmap_image, cax=heatmap_cbar_axis)
    heatmap_cbar.set_label("Delta mean co-occurrence strength", fontsize=19)
    heatmap_cbar.ax.tick_params(labelsize=15)
    heatmap_cbar.ax.yaxis.set_ticks_position("left")
    heatmap_cbar.ax.yaxis.set_label_position("left")
    heatmap_cbar.ax.yaxis.set_major_formatter(FormatStrFormatter("%.2f"))

    figure.savefig(OUTPUT_PNG, dpi=300, bbox_inches="tight", pad_inches=0.03)
    figure.savefig(OUTPUT_PDF, bbox_inches="tight", pad_inches=0.03)
    plt.close(figure)


if __name__ == "__main__":
    main()
