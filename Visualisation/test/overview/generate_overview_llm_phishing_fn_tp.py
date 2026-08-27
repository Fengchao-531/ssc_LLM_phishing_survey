#!/usr/bin/env python3
from __future__ import annotations

import json
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.colors import Normalize
from matplotlib.lines import Line2D
from matplotlib.patches import Rectangle

VIS_ROOT = Path(__file__).resolve().parents[2]
if str(VIS_ROOT) not in sys.path:
    sys.path.insert(0, str(VIS_ROOT))

from generate_stage_visualizations import compute_axis_limits, fit_detector_surface
from generate_test_failure_contours import add_background
from generate_test_failure_mode_figures import FN_STYLE, TP_STYLE
from generate_test_failure_heatmaps import PRINCIPLE_NAMES, WVAE_COLUMN_BY_PRINCIPLE, merge_projected_with_wvae

SCRIPT_DIR = Path(__file__).resolve().parent
PROJECTED_INPUT = VIS_ROOT / "test" / "projected_points.csv"
OUTPUT_CONTOUR = SCRIPT_DIR / "fig_llm_phishing_fn_tp_contours_pca.png"
OUTPUT_HEATMAP = SCRIPT_DIR / "fig_llm_phishing_fn_tp_heatmaps_pca.png"
OUTPUT_METADATA = SCRIPT_DIR / "fig_llm_phishing_fn_tp_metadata.json"

FN_BOUNDS = {"x0": -2.2, "x1": 2.0, "y0": -3.2, "y1": 0.2}
TP_BOUNDS = {"x0": 1.8, "x1": 5.2, "y0": 0.4, "y1": 4.0}
BOX_STYLES = {
    "fn": {"color": "#0a55ff", "label": "LLM-P FN box"},
    "tp": {"color": "#11823b", "label": "LLM-P TP box"},
}


def region_mask(frame: pd.DataFrame, bounds: dict[str, float]) -> pd.Series:
    return (
        frame["proj_x"].between(bounds["x0"], bounds["x1"], inclusive="both")
        & frame["proj_y"].between(bounds["y0"], bounds["y1"], inclusive="both")
    )


def compute_principle_cooccurrence_matrix(frame: pd.DataFrame) -> tuple[np.ndarray, int]:
    principle_columns = []
    for name in PRINCIPLE_NAMES:
        base_column = WVAE_COLUMN_BY_PRINCIPLE[name]
        for candidate in [f"{base_column}_y", base_column, f"{base_column}_x"]:
            if candidate in frame.columns:
                principle_columns.append(candidate)
                break
        else:
            raise KeyError(f"Missing persuasion principle column for {base_column}")
    principle_frame = frame[principle_columns].dropna(how="all")
    if principle_frame.empty:
        return np.full((len(PRINCIPLE_NAMES), len(PRINCIPLE_NAMES)), np.nan, dtype=float), 0
    values = principle_frame.to_numpy(dtype=float)
    matrix = (values.T @ values) / max(1, values.shape[0])
    diagonal = np.nanmean(values, axis=0)
    np.fill_diagonal(matrix, diagonal)
    return matrix, int(values.shape[0])


def load_frame() -> pd.DataFrame:
    frame = pd.read_csv(PROJECTED_INPUT, low_memory=False)
    frame = frame[
        (frame["source"] == "LLM") & (pd.to_numeric(frame["label"], errors="coerce") == 1)
    ].copy()
    frame["detector_prediction"] = pd.to_numeric(frame["detector_prediction"], errors="coerce").fillna(0.0)
    frame["is_tp"] = frame["detector_prediction"] >= 0.5
    frame["is_fn"] = ~frame["is_tp"]
    return frame


def draw_contour_figure(frame: pd.DataFrame) -> dict[str, int]:
    x_limits, y_limits = compute_axis_limits(frame)
    x_grid = np.linspace(*x_limits, 220)
    y_grid = np.linspace(*y_limits, 220)
    grid_x, grid_y = np.meshgrid(x_grid, y_grid)
    mesh = np.column_stack([grid_x.ravel(), grid_y.ravel()])

    surface_model = fit_detector_surface(frame, 7)
    if surface_model is None:
        raise RuntimeError("Could not fit LLM phishing surface.")
    score_grid = surface_model.predict_proba(mesh)[:, 1].reshape(grid_x.shape)

    figure = plt.figure(figsize=(10.8, 8.4))
    grid = figure.add_gridspec(
        2,
        3,
        width_ratios=[1.0, 0.08, 0.02],
        height_ratios=[1.0, 0.18],
        wspace=0.10,
        hspace=0.10,
    )
    axis = figure.add_subplot(grid[0, 0])
    colorbar_axis = figure.add_subplot(grid[0, 1])
    legend_axis = figure.add_subplot(grid[1, 0])
    legend_axis.axis("off")

    filled = add_background(axis, grid_x, grid_y, score_grid, "pca")
    tp_frame = frame[frame["is_tp"]]
    fn_frame = frame[frame["is_fn"]]
    axis.scatter(tp_frame["proj_x"], tp_frame["proj_y"], **TP_STYLE["LLM"])
    axis.scatter(fn_frame["proj_x"], fn_frame["proj_y"], **FN_STYLE["LLM"])

    fn_target = frame[frame["is_fn"] & region_mask(frame, FN_BOUNDS)].copy()
    tp_target = frame[frame["is_tp"] & region_mask(frame, TP_BOUNDS)].copy()

    for bounds, key in [(FN_BOUNDS, "fn"), (TP_BOUNDS, "tp")]:
        style = BOX_STYLES[key]
        axis.add_patch(
            Rectangle(
                (bounds["x0"], bounds["y0"]),
                bounds["x1"] - bounds["x0"],
                bounds["y1"] - bounds["y0"],
                fill=False,
                linewidth=2.2,
                edgecolor=style["color"],
            )
        )
        axis.text(
            bounds["x0"] + 0.08,
            bounds["y1"] + 0.10,
            style["label"],
            fontsize=10,
            color=style["color"],
            ha="left",
            va="bottom",
            bbox={
                "boxstyle": "round,pad=0.18",
                "facecolor": "white",
                "edgecolor": style["color"],
                "alpha": 0.94,
            },
        )

    axis.set_title("LLM-phishing: FN vs TP focus regions", fontsize=13, pad=10)
    axis.set_xlabel("PCA1", fontsize=10)
    axis.set_ylabel("PCA2", fontsize=10)
    axis.grid(alpha=0.10, linewidth=0.5)

    legend_axis.legend(
        handles=[
            Line2D([0], [0], marker="^", color="none", markerfacecolor="none", markeredgecolor="#2f2f2f", markersize=7, linewidth=0, label="LLM-TP"),
            Line2D([0], [0], marker="^", color="none", markerfacecolor="#1f77b4", markeredgecolor="white", markersize=7, linewidth=0, label="LLM-FN"),
            Line2D([0], [0], color=BOX_STYLES["fn"]["color"], linewidth=2.2, label=BOX_STYLES["fn"]["label"]),
            Line2D([0], [0], color=BOX_STYLES["tp"]["color"], linewidth=2.2, label=BOX_STYLES["tp"]["label"]),
            Line2D([0], [0], color="black", linewidth=1.8, label="threshold contour: s(x)=0.50"),
        ],
        loc="center",
        ncol=2,
        frameon=False,
    )
    colorbar = figure.colorbar(filled, cax=colorbar_axis)
    colorbar.set_label("Surrogate score")
    figure.suptitle("Overview LLM-P FN and LLM-P TP Contour Map", y=0.98, fontsize=15)
    figure.subplots_adjust(left=0.07, right=0.93, bottom=0.08, top=0.92)
    figure.savefig(OUTPUT_CONTOUR, dpi=220)
    plt.close(figure)

    return {
        "llm_fn_focus_points": int(len(fn_target)),
        "llm_tp_focus_points": int(len(tp_target)),
    }


def draw_heatmap_figure(merged_frame: pd.DataFrame) -> dict[str, int]:
    fn_rows = merged_frame[merged_frame["is_fn"] & region_mask(merged_frame, FN_BOUNDS)].copy()
    tp_rows = merged_frame[merged_frame["is_tp"] & region_mask(merged_frame, TP_BOUNDS)].copy()
    fn_matrix, fn_count = compute_principle_cooccurrence_matrix(fn_rows)
    tp_matrix, tp_count = compute_principle_cooccurrence_matrix(tp_rows)

    figure = plt.figure(figsize=(11.6, 5.8))
    grid = figure.add_gridspec(1, 3, width_ratios=[1.0, 1.0, 0.08], wspace=0.05)
    axes = [figure.add_subplot(grid[0, 0]), figure.add_subplot(grid[0, 1])]
    colorbar_axis = figure.add_subplot(grid[0, 2])

    images = []
    for axis, matrix, title in [
        (axes[0], fn_matrix, "LLM-P FN"),
        (axes[1], tp_matrix, "LLM-P TP"),
    ]:
        safe_matrix = np.nan_to_num(matrix, nan=0.0, posinf=0.0, neginf=0.0)
        image = axis.imshow(safe_matrix, aspect="equal", cmap="GnBu", norm=Normalize(vmin=0.0, vmax=1.0))
        images.append(image)
        axis.set_xticks(range(len(PRINCIPLE_NAMES)))
        axis.set_xticklabels(PRINCIPLE_NAMES, rotation=45, ha="right", fontsize=8.5)
        axis.set_yticks(range(len(PRINCIPLE_NAMES)))
        axis.set_yticklabels(PRINCIPLE_NAMES if title == "LLM-P FN" else [], fontsize=8.5)
        axis.set_title(title, fontsize=12)
        for row_index in range(safe_matrix.shape[0]):
            for col_index in range(safe_matrix.shape[1]):
                label = "NA" if np.isnan(matrix[row_index, col_index]) else f"{matrix[row_index, col_index]:.2f}"
                axis.text(
                    col_index,
                    row_index,
                    label,
                    ha="center",
                    va="center",
                    fontsize=7.5,
                    color="black",
                )

    colorbar = figure.colorbar(images[-1], cax=colorbar_axis)
    colorbar.set_label("Mean co-occurrence strength")
    figure.suptitle("Overview LLM-P FN and LLM-P TP Heatmaps", y=0.98, fontsize=15)
    figure.subplots_adjust(left=0.06, right=0.95, bottom=0.10, top=0.88)
    figure.savefig(OUTPUT_HEATMAP, dpi=220)
    plt.close(figure)
    return {
        "llm_fn_heatmap_rows": int(fn_count),
        "llm_tp_heatmap_rows": int(tp_count),
    }


def main() -> None:
    frame = load_frame()
    merged = merge_projected_with_wvae(frame)
    contour_counts = draw_contour_figure(frame)
    heatmap_counts = draw_heatmap_figure(merged)

    metadata = {
        "projected_input": str(PROJECTED_INPUT),
        "files": {
            "contour_figure": OUTPUT_CONTOUR.name,
            "heatmap_figure": OUTPUT_HEATMAP.name,
        },
        "fn_bounds": FN_BOUNDS,
        "tp_bounds": TP_BOUNDS,
        "counts": {
            **contour_counts,
            **heatmap_counts,
        },
    }
    OUTPUT_METADATA.write_text(json.dumps(metadata, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
