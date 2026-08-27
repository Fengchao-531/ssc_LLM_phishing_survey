#!/usr/bin/env python3
from __future__ import annotations

from pathlib import Path
import sys

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.colors import LinearSegmentedColormap, Normalize
from matplotlib.ticker import FormatStrFormatter


SCRIPT_DIR = Path(__file__).resolve().parent
VISUALIZATION_DIR = SCRIPT_DIR.parents[1]
TEST_DIR = VISUALIZATION_DIR / "test"
INPUT_PATH = TEST_DIR / "projected_points.csv"
OUTPUT_DIR = SCRIPT_DIR / "2"
OUTPUT_PATH = OUTPUT_DIR / "overview_focus_heatmaps.png"

if str(VISUALIZATION_DIR) not in sys.path:
    sys.path.insert(0, str(VISUALIZATION_DIR))

from generate_test_failure_heatmaps import (  # noqa: E402
    PRINCIPLE_NAMES,
    WVAE_COLUMN_BY_PRINCIPLE,
    merge_projected_with_wvae,
)

X_AXIS_LABEL_FONT_SIZE = 33
TICK_LABEL_FONT_SIZE = 29


BOX_BY_SOURCE = {
    "HW": {"x0": -2.2, "x1": 3.0, "y0": -4.2, "y1": 2.9},
    "LLM": {"x0": -6.2, "x1": -1.7, "y0": 0.8, "y1": 8.1},
}


def load_projected_frame(csv_path: Path) -> pd.DataFrame:
    frame = pd.read_csv(csv_path, low_memory=False)
    required_columns = {"subject", "body", "source", "stage", "proj_x", "proj_y"}
    missing = sorted(required_columns.difference(frame.columns))
    if missing:
        raise ValueError(f"{csv_path} is missing required columns: {', '.join(missing)}")

    frame = frame.copy()
    if "detector_prediction" in frame.columns:
        detector_prediction = pd.to_numeric(frame["detector_prediction"], errors="coerce").fillna(0.0)
    else:
        detector_prediction = pd.to_numeric(frame["scamllm"], errors="coerce").fillna(0.0)
    frame["is_tp"] = detector_prediction >= 0.5
    frame["is_fn"] = ~frame["is_tp"]
    return frame


def region_mask(frame: pd.DataFrame, bounds: dict[str, float]) -> pd.Series:
    return (
        frame["proj_x"].between(bounds["x0"], bounds["x1"], inclusive="both")
        & frame["proj_y"].between(bounds["y0"], bounds["y1"], inclusive="both")
    )


def resolve_principle_columns(frame: pd.DataFrame) -> list[str]:
    principle_columns: list[str] = []
    for name in PRINCIPLE_NAMES:
        base_name = WVAE_COLUMN_BY_PRINCIPLE[name]
        if f"{base_name}_y" in frame.columns:
            principle_columns.append(f"{base_name}_y")
        elif base_name in frame.columns:
            principle_columns.append(base_name)
        elif f"{base_name}_x" in frame.columns:
            principle_columns.append(f"{base_name}_x")
        else:
            raise KeyError(f"Missing persuasion principle column for {base_name}")
    return principle_columns


def compute_principle_cooccurrence_matrix(frame: pd.DataFrame) -> tuple[np.ndarray, int]:
    principle_columns = resolve_principle_columns(frame)
    principle_frame = frame[principle_columns].dropna(how="all")
    if principle_frame.empty:
        return np.full((len(PRINCIPLE_NAMES), len(PRINCIPLE_NAMES)), np.nan, dtype=float), 0

    values = principle_frame.to_numpy(dtype=float)
    matrix = (values.T @ values) / max(1, values.shape[0])
    diagonal = np.nanmean(values, axis=0)
    np.fill_diagonal(matrix, diagonal)
    return matrix, int(values.shape[0])


def build_heatmap_cmap() -> LinearSegmentedColormap:
    return LinearSegmentedColormap.from_list(
        "overview_focus_yellow_blue",
        [
            (0.00, "#f1d7ac"),
            (0.10, "#f7ead0"),
            (0.16, "#ffffff"),
            (0.45, "#92bdd1"),
            (0.72, "#4d8fc0"),
            (1.00, "#1f629c"),
        ],
        N=256,
    )


def draw_heatmap_axis(
    axis: plt.Axes,
    matrix: np.ndarray,
    *,
    show_y_labels: bool,
    norm: Normalize,
) -> object:
    safe_matrix = np.nan_to_num(matrix, nan=0.0, posinf=0.0, neginf=0.0)
    image = axis.imshow(
        safe_matrix,
        cmap=build_heatmap_cmap(),
        norm=norm,
        aspect="equal",
    )
    axis.set_xticks(range(len(PRINCIPLE_NAMES)))
    axis.set_xticklabels([])
    axis.tick_params(axis="x", length=0)
    axis.set_yticks(range(len(PRINCIPLE_NAMES)))
    axis.set_yticklabels(PRINCIPLE_NAMES if show_y_labels else [], fontsize=TICK_LABEL_FONT_SIZE)

    for row_index in range(safe_matrix.shape[0]):
        for col_index in range(safe_matrix.shape[1]):
            value = matrix[row_index, col_index]
            if not np.isfinite(value):
                continue
            text_color = "white" if safe_matrix[row_index, col_index] >= norm.vmax * 0.55 else "black"
            axis.text(
                col_index,
                row_index,
                f"{value:.2f}",
                ha="center",
                va="center",
                fontsize=13.5,
                color=text_color,
            )

    axis.set_xlim(-0.5, len(PRINCIPLE_NAMES) - 0.5)
    axis.set_ylim(len(PRINCIPLE_NAMES) - 0.5, -0.5)
    axis.grid(False)
    return image


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    projected_frame = load_projected_frame(INPUT_PATH)
    merged_frame = merge_projected_with_wvae(projected_frame)

    matrices: list[np.ndarray] = []
    for source_name in ["HW", "LLM"]:
        source_rows = merged_frame[
            (merged_frame["source"] == source_name)
            & region_mask(merged_frame, BOX_BY_SOURCE[source_name])
            & merged_frame["is_fn"]
        ].copy()
        matrix, _ = compute_principle_cooccurrence_matrix(source_rows)
        matrices.append(matrix)

    finite_values = np.concatenate([matrix[np.isfinite(matrix)].ravel() for matrix in matrices])
    vmax = float(np.quantile(finite_values, 0.95))
    vmax = max(vmax, 0.70)
    vmax = min(1.0, float(np.ceil(vmax / 0.05) * 0.05))
    norm = Normalize(vmin=0.0, vmax=vmax)

    figure = plt.figure(figsize=(11.3, 4.9))
    grid = figure.add_gridspec(1, 2, width_ratios=[1.0, 1.0], wspace=0.05)
    axes = [figure.add_subplot(grid[0, 0]), figure.add_subplot(grid[0, 1])]

    images = []
    for axis, matrix, show_y_labels in zip(axes, matrices, [True, False], strict=True):
        images.append(
            draw_heatmap_axis(
                axis,
                matrix,
                show_y_labels=show_y_labels,
                norm=norm,
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
    colorbar.set_label("Mean co-occurrence strength", fontsize=12)
    tick_step = 0.10 if vmax > 0.50 else 0.05
    colorbar_ticks = np.arange(0.0, vmax + 1e-9, tick_step)
    colorbar.set_ticks(colorbar_ticks)
    colorbar.ax.yaxis.set_major_formatter(FormatStrFormatter("%.2f"))
    colorbar.ax.tick_params(labelsize=10.5)

    figure.savefig(OUTPUT_PATH, dpi=260, bbox_inches="tight", pad_inches=0.01)
    plt.close(figure)


if __name__ == "__main__":
    main()
