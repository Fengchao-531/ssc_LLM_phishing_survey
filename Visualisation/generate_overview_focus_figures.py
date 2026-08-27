#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from matplotlib.colors import Normalize
from matplotlib.lines import Line2D
from matplotlib.patches import Rectangle

from generate_stage_visualizations import fit_detector_surface
from generate_test_failure_contours import add_background, build_global_grid
from generate_test_failure_heatmaps import PRINCIPLE_NAMES, WVAE_COLUMN_BY_PRINCIPLE, merge_projected_with_wvae

SCRIPT_DIR = Path(__file__).resolve().parent
DEFAULT_PROJECTED_INPUT = SCRIPT_DIR / "test" / "projected_points.csv"
DEFAULT_OUTPUT_DIR = SCRIPT_DIR / "test" / "overview"

SOURCE_TITLES = {"HW": "HW-phishing", "LLM": "LLM-phishing"}
BOX_BY_SOURCE = {
    "HW": {"x0": -2.2, "x1": 3.0, "y0": -4.2, "y1": 2.9},
    "LLM": {"x0": -6.2, "x1": -1.7, "y0": 0.8, "y1": 8.1},
}
BOX_COLOR = "#0a55ff"
TP_STYLE = {
    "HW": {"marker": "o", "facecolors": "none", "edgecolors": "#2f2f2f", "linewidths": 1.0, "s": 28, "alpha": 0.8},
    "LLM": {"marker": "^", "facecolors": "none", "edgecolors": "#2f2f2f", "linewidths": 1.0, "s": 34, "alpha": 0.8},
}
FN_STYLE = {
    "HW": {"marker": "o", "c": "#2ca02c", "edgecolors": "white", "linewidths": 0.35, "s": 36, "alpha": 0.92},
    "LLM": {"marker": "^", "c": "#1f77b4", "edgecolors": "white", "linewidths": 0.35, "s": 42, "alpha": 0.92},
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Create overview dual-panel focus figures plus source-specific WVAE "
            "principle heatmaps under Visualization/test/overview."
        )
    )
    parser.add_argument("--projected-input", type=Path, default=DEFAULT_PROJECTED_INPUT)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--projection", choices=["pca", "umap"], default="pca")
    parser.add_argument("--seed", type=int, default=7)
    return parser.parse_args()


def load_projected_frame(csv_path: Path) -> pd.DataFrame:
    frame = pd.read_csv(csv_path)
    required_columns = {"subject", "body", "source", "stage", "proj_x", "proj_y"}
    missing = sorted(required_columns.difference(frame.columns))
    if missing:
        raise ValueError(f"{csv_path} is missing required columns: {', '.join(missing)}")

    frame = frame.copy()
    if "detector_prediction" in frame.columns:
        frame["detector_prediction"] = pd.to_numeric(frame["detector_prediction"], errors="coerce").fillna(0.0)
    else:
        frame["detector_prediction"] = pd.to_numeric(frame["scamllm"], errors="coerce").fillna(0.0)
    frame["is_tp"] = frame["detector_prediction"] >= 0.5
    frame["is_fn"] = ~frame["is_tp"]
    return frame


def region_mask(frame: pd.DataFrame, bounds: dict[str, float]) -> pd.Series:
    return (
        frame["proj_x"].between(bounds["x0"], bounds["x1"], inclusive="both")
        & frame["proj_y"].between(bounds["y0"], bounds["y1"], inclusive="both")
    )


def compute_principle_cooccurrence_matrix(frame: pd.DataFrame) -> tuple[np.ndarray, int]:
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
    principle_frame = frame[principle_columns].dropna(how="all")
    if principle_frame.empty:
        return np.full((len(PRINCIPLE_NAMES), len(PRINCIPLE_NAMES)), np.nan, dtype=float), 0

    values = principle_frame.to_numpy(dtype=float)
    matrix = (values.T @ values) / max(1, values.shape[0])
    diagonal = np.nanmean(values, axis=0)
    np.fill_diagonal(matrix, diagonal)
    return matrix, int(values.shape[0])


def draw_principle_cooccurrence_heatmap(
    matrix: np.ndarray,
    *,
    output_path: Path,
    title: str,
    count: int,
    no_data_note: str | None = None,
) -> None:
    safe_matrix = np.nan_to_num(matrix, nan=0.0, posinf=0.0, neginf=0.0)
    figure, axis = plt.subplots(figsize=(7.2, 6.2))
    image = axis.imshow(safe_matrix, aspect="equal", cmap="GnBu", vmin=0.0, vmax=1.0)
    axis.set_xticks(range(len(PRINCIPLE_NAMES)))
    axis.set_xticklabels(PRINCIPLE_NAMES, rotation=45, ha="right", fontsize=9)
    axis.set_yticks(range(len(PRINCIPLE_NAMES)))
    axis.set_yticklabels(PRINCIPLE_NAMES, fontsize=9)

    for row_index in range(safe_matrix.shape[0]):
        for col_index in range(safe_matrix.shape[1]):
            label = "NA" if np.isnan(matrix[row_index, col_index]) else f"{matrix[row_index, col_index]:.2f}"
            axis.text(
                col_index,
                row_index,
                label,
                ha="center",
                va="center",
                fontsize=8,
                color="black",
            )

    if no_data_note:
        axis.text(
            0.5,
            -0.22,
            no_data_note,
            transform=axis.transAxes,
            ha="center",
            va="top",
            fontsize=8.5,
            color="#4f4f4f",
        )

    axis.set_title(f"matched WVAE rows: {count}", fontsize=10, pad=10)
    colorbar = figure.colorbar(image, ax=axis, fraction=0.04, pad=0.03)
    colorbar.set_label("Mean co-occurrence strength")
    figure.suptitle(title, y=0.98, fontsize=14)
    figure.tight_layout(rect=(0, 0.03, 1, 0.94))
    figure.savefig(output_path, dpi=220, bbox_inches="tight")
    plt.close(figure)


def build_legend_handles() -> list[Line2D]:
    return [
        Line2D([0], [0], marker="o", color="none", markerfacecolor="none", markeredgecolor="#2f2f2f", markersize=7, linewidth=0, label="HW-TP"),
        Line2D([0], [0], marker="o", color="none", markerfacecolor="#2ca02c", markeredgecolor="white", markersize=7, linewidth=0, label="HW-FN"),
        Line2D([0], [0], marker="^", color="none", markerfacecolor="none", markeredgecolor="#2f2f2f", markersize=7, linewidth=0, label="LLM-TP"),
        Line2D([0], [0], marker="^", color="none", markerfacecolor="#1f77b4", markeredgecolor="white", markersize=7, linewidth=0, label="LLM-FN"),
        Line2D([0], [0], color=BOX_COLOR, linewidth=2.2, label="focus box"),
        Line2D([0], [0], color="black", linewidth=1.8, label="threshold contour: s(x)=0.50"),
    ]


def draw_contour_panel(
    axis: plt.Axes,
    frame: pd.DataFrame,
    *,
    source_name: str,
    projection_method: str,
    grid_x: np.ndarray,
    grid_y: np.ndarray,
    score_grid: np.ndarray,
) -> tuple[object, dict[str, int]]:
    bounds = BOX_BY_SOURCE[source_name]
    filled = add_background(axis, grid_x, grid_y, score_grid, projection_method)
    source_frame = frame[frame["source"] == source_name]
    tp_frame = source_frame[source_frame["is_tp"]]
    fn_frame = source_frame[source_frame["is_fn"]]
    axis.scatter(tp_frame["proj_x"], tp_frame["proj_y"], **TP_STYLE[source_name])
    axis.scatter(fn_frame["proj_x"], fn_frame["proj_y"], **FN_STYLE[source_name])

    focus_source_frame = source_frame[region_mask(source_frame, bounds)]
    focus_count = {
        "total": int(len(focus_source_frame)),
        "tp": int(focus_source_frame["is_tp"].sum()),
        "fn": int(focus_source_frame["is_fn"].sum()),
    }

    axis.add_patch(
        Rectangle(
            (bounds["x0"], bounds["y0"]),
            bounds["x1"] - bounds["x0"],
            bounds["y1"] - bounds["y0"],
            fill=False,
            linewidth=2.2,
            linestyle="-",
            edgecolor=BOX_COLOR,
        )
    )
    axis.text(
        bounds["x0"] + 0.10,
        bounds["y1"] + 0.10,
        "focus box",
        color=BOX_COLOR,
        fontsize=10,
        ha="left",
        va="bottom",
        bbox={
            "boxstyle": "round,pad=0.18",
            "facecolor": "white",
            "edgecolor": BOX_COLOR,
            "alpha": 0.95,
        },
    )
    axis.text(
        0.03,
        0.96,
        (
            f"focus box n={focus_count['total']}\n"
            f"TP={focus_count['tp']} | FN={focus_count['fn']}"
        ),
        transform=axis.transAxes,
        ha="left",
        va="top",
        fontsize=9,
        bbox={
            "boxstyle": "round,pad=0.22",
            "facecolor": "white",
            "edgecolor": "#d0d0d0",
            "alpha": 0.90,
        },
    )
    axis.set_title(
        (
            f"{SOURCE_TITLES[source_name]}\n"
            f"TP={int(tp_frame.shape[0])} | FN={int(fn_frame.shape[0])}"
        ),
        fontsize=12,
    )
    axis.set_xlabel(f"{projection_method.upper()}1", fontsize=10)
    axis.grid(alpha=0.10, linewidth=0.5)
    return filled, focus_count


def draw_combined_overview_figure(
    frame: pd.DataFrame,
    output_path: Path,
    *,
    projection_method: str,
    grid_x: np.ndarray,
    grid_y: np.ndarray,
    score_grid: np.ndarray,
) -> dict[str, dict[str, int]]:
    figure = plt.figure(figsize=(16.4, 8.2))
    contour_grid = figure.add_gridspec(
        2,
        3,
        width_ratios=[1.0, 1.0, 0.08],
        height_ratios=[1.0, 0.18],
        wspace=0.10,
        hspace=0.10,
    )
    contour_axes = [figure.add_subplot(contour_grid[0, 0]), figure.add_subplot(contour_grid[0, 1])]
    colorbar_axis = figure.add_subplot(contour_grid[0, 2])
    legend_axis = figure.add_subplot(contour_grid[1, :2])
    legend_axis.axis("off")
    focus_counts: dict[str, dict[str, int]] = {}
    contour_filled = None
    for axis, source_name in zip(contour_axes, ["HW", "LLM"], strict=True):
        contour_filled, focus_counts[source_name] = draw_contour_panel(
            axis,
            frame,
            source_name=source_name,
            projection_method=projection_method,
            grid_x=grid_x,
            grid_y=grid_y,
            score_grid=score_grid,
        )
    contour_axes[0].set_ylabel(f"{projection_method.upper()}2", fontsize=10)
    contour_axes[1].tick_params(axis="y", labelleft=False)

    legend_axis.legend(
        handles=build_legend_handles(),
        loc="center",
        ncol=3,
        frameon=False,
    )
    contour_colorbar = figure.colorbar(contour_filled, cax=colorbar_axis)
    contour_colorbar.set_label("Surrogate score")
    figure.suptitle(
        "Overview Figure 2A. Source-Specific Focus Boxes on Dual-Panel Surrogate Maps",
        y=0.98,
        fontsize=15,
    )
    figure.subplots_adjust(left=0.05, right=0.96, bottom=0.08, top=0.90)
    figure.savefig(output_path, dpi=220)
    plt.close(figure)
    return focus_counts


def draw_dual_heatmap_figure(
    merged_frame: pd.DataFrame,
    output_path: Path,
) -> dict[str, int]:
    figure = plt.figure(figsize=(11.8, 5.8))
    heatmap_grid = figure.add_gridspec(
        1,
        3,
        width_ratios=[1.0, 1.0, 0.08],
        wspace=0.05,
    )
    heatmap_axes = [figure.add_subplot(heatmap_grid[0, 0]), figure.add_subplot(heatmap_grid[0, 1])]
    colorbar_axis = figure.add_subplot(heatmap_grid[0, 2])

    heatmap_images = []
    matched_counts: dict[str, int] = {}
    for axis, source_name in zip(heatmap_axes, ["HW", "LLM"], strict=True):
        source_rows = merged_frame[
            (merged_frame["source"] == source_name)
            & region_mask(merged_frame, BOX_BY_SOURCE[source_name])
            & merged_frame["is_fn"]
        ].copy()
        matrix, matched_count = compute_principle_cooccurrence_matrix(source_rows)
        matched_counts[source_name] = matched_count
        safe_matrix = np.nan_to_num(matrix, nan=0.0, posinf=0.0, neginf=0.0)
        image = axis.imshow(safe_matrix, aspect="equal", cmap="GnBu", norm=Normalize(vmin=0.0, vmax=1.0))
        heatmap_images.append(image)
        axis.set_xticks(range(len(PRINCIPLE_NAMES)))
        axis.set_xticklabels(PRINCIPLE_NAMES, rotation=45, ha="right", fontsize=8.5)
        axis.set_yticks(range(len(PRINCIPLE_NAMES)))
        axis.set_yticklabels(PRINCIPLE_NAMES if source_name == "HW" else [], fontsize=8.5)
        axis.set_title(f"{SOURCE_TITLES[source_name]} FN", fontsize=12)
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

    heatmap_colorbar = figure.colorbar(heatmap_images[-1], cax=colorbar_axis)
    heatmap_colorbar.set_label("Mean co-occurrence strength")
    figure.suptitle(
        "Overview Figure 2B. Persuasion Principle Heatmaps for Source-Specific FN Focus Boxes",
        y=0.98,
        fontsize=15,
    )
    figure.subplots_adjust(left=0.06, right=0.95, bottom=0.10, top=0.88)
    figure.savefig(output_path, dpi=220)
    plt.close(figure)
    return matched_counts


def main() -> None:
    args = parse_args()
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    frame = load_projected_frame(args.projected_input.resolve())
    surface_model = fit_detector_surface(frame, args.seed)
    if surface_model is None:
        raise RuntimeError("Could not fit the surrogate surface for the overview projected frame.")
    _, _, grid_x, grid_y, score_grid = build_global_grid(frame, surface_model)

    merged = merge_projected_with_wvae(frame)
    contour_figure_path = output_dir / "fig2_overview_focus_contours_pca.png"
    focus_counts = draw_combined_overview_figure(
        frame,
        contour_figure_path,
        projection_method=args.projection,
        grid_x=grid_x,
        grid_y=grid_y,
        score_grid=score_grid,
    )
    heatmap_figure_path = output_dir / "fig2_overview_focus_heatmaps_pca.png"
    matched_counts = draw_dual_heatmap_figure(
        merged,
        heatmap_figure_path,
    )
    for source_name in matched_counts:
        focus_counts[source_name]["matched_wvae_rows_fn_only"] = matched_counts[source_name]

    metadata = {
        "projected_input": str(args.projected_input.resolve()),
        "output_dir": str(output_dir),
        "files": {
            "contour_figure": contour_figure_path.name,
            "heatmap_figure": heatmap_figure_path.name,
        },
        "focus_boxes": BOX_BY_SOURCE,
        "focus_box_counts": focus_counts,
    }
    metadata_path = output_dir / "overview_focus_metadata.json"
    metadata_path.write_text(json.dumps(metadata, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
