#!/usr/bin/env python3
from __future__ import annotations

from pathlib import Path
import sys

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.lines import Line2D
from matplotlib.patches import FancyBboxPatch


SCRIPT_DIR = Path(__file__).resolve().parent
VISUALIZATION_DIR = SCRIPT_DIR.parents[1]
STAGES_DIR = VISUALIZATION_DIR / "test" / "stages"
OUTPUT_DIR = SCRIPT_DIR / "4"

if str(VISUALIZATION_DIR) not in sys.path:
    sys.path.insert(0, str(VISUALIZATION_DIR))

from generate_stage_split_figures import STAGE_ORDER  # noqa: E402
from generate_stage_visualizations import compute_axis_limits, fit_detector_surface  # noqa: E402
from generate_test_failure_contours import add_background  # noqa: E402

X_AXIS_LABEL_FONT_SIZE = 33
TICK_LABEL_FONT_SIZE = 29


TP_STYLE = {
    "HW": {"marker": "o", "facecolors": "none", "edgecolors": "#2f2f2f", "linewidths": 1.25, "s": 74, "alpha": 0.88},
    "LLM": {"marker": "^", "facecolors": "none", "edgecolors": "#3d3d3d", "linewidths": 1.2, "s": 86, "alpha": 0.88},
}
FN_STYLE = {
    "HW": {"marker": "o", "c": "#2ca02c", "edgecolors": "white", "linewidths": 0.52, "s": 82, "alpha": 0.95},
    "LLM": {"marker": "^", "c": "#3498db", "edgecolors": "white", "linewidths": 0.50, "s": 96, "alpha": 0.93},
}


def load_stage_frame(csv_path: Path) -> pd.DataFrame:
    frame = pd.read_csv(csv_path, low_memory=False).copy()
    required_columns = {"source", "proj_x", "proj_y", "detector_prediction"}
    missing = sorted(required_columns.difference(frame.columns))
    if missing:
        raise ValueError(f"{csv_path} is missing required columns: {', '.join(missing)}")

    detector_prediction = pd.to_numeric(frame["detector_prediction"], errors="coerce").fillna(0.0)
    frame["is_tp"] = detector_prediction >= 0.5
    frame["is_fn"] = ~frame["is_tp"]
    return frame


def build_stage_grid(frame: pd.DataFrame, seed: int, stage_name: str) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    surface_model = fit_detector_surface(frame, seed)
    if surface_model is None:
        raise RuntimeError(f"Could not fit the surrogate surface for {stage_name}.")
    x_limits, y_limits = compute_axis_limits(frame)
    x_grid = np.linspace(*x_limits, 220)
    y_grid = np.linspace(*y_limits, 220)
    grid_x, grid_y = np.meshgrid(x_grid, y_grid)
    mesh = np.column_stack([grid_x.ravel(), grid_y.ravel()])
    score_grid = surface_model.predict_proba(mesh)[:, 1].reshape(grid_x.shape)
    return grid_x, grid_y, score_grid


def build_legend_handles() -> dict[str, Line2D]:
    hw_tp = Line2D([0], [0], marker="o", color="none", markerfacecolor="none", markeredgecolor="#2f2f2f", markersize=11, markeredgewidth=1.3, linewidth=0, label="HW-TP")
    hw_fn = Line2D([0], [0], marker="o", color="none", markerfacecolor="#2ca02c", markeredgecolor="white", markersize=11, linewidth=0, label="HW-FN")
    llm_tp = Line2D([0], [0], marker="^", color="none", markerfacecolor="none", markeredgecolor="#3d3d3d", markersize=11, markeredgewidth=1.2, linewidth=0, label="LLM-TP")
    llm_fn = Line2D([0], [0], marker="^", color="none", markerfacecolor="#3498db", markeredgecolor="white", markersize=11, linewidth=0, label="LLM-FN")
    threshold = Line2D([0], [0], color="black", linewidth=2.0, label="threshold contour: s(x)=0.50")
    return {
        "hw_tp": hw_tp,
        "hw_fn": hw_fn,
        "llm_tp": llm_tp,
        "llm_fn": llm_fn,
        "threshold": threshold,
    }


def add_compact_legend(axis: plt.Axes) -> None:
    handles = build_legend_handles()
    background = FancyBboxPatch(
        (0.300, 0.832),
        0.60,
        0.16,
        transform=axis.transAxes,
        boxstyle="round,pad=0.012",
        facecolor="white",
        edgecolor="#d0d0d0",
        linewidth=1.1,
        alpha=0.94,
        zorder=6.0,
    )
    axis.add_patch(background)

    left_legend = axis.legend(
        handles=[handles["hw_tp"], handles["llm_tp"], handles["threshold"]],
        loc="upper left",
        bbox_to_anchor=(0.325, 0.986),
        fontsize=19,
        frameon=False,
        borderpad=0.0,
        labelspacing=0.45,
        handletextpad=0.6,
        handlelength=2.2,
    )
    left_legend.set_zorder(7.5)
    axis.add_artist(left_legend)

    right_legend = axis.legend(
        handles=[handles["hw_fn"], handles["llm_fn"]],
        loc="upper left",
        bbox_to_anchor=(0.555, 0.986),
        fontsize=19,
        frameon=False,
        borderpad=0.0,
        labelspacing=0.45,
        handletextpad=0.6,
        handlelength=1.5,
    )
    right_legend.set_zorder(7.5)
    axis.add_artist(right_legend)


def plot_stage_on_axis(axis: plt.Axes, stage_name: str, stage_frame: pd.DataFrame, seed: int) -> object:
    grid_x, grid_y, score_grid = build_stage_grid(stage_frame, seed, stage_name)
    filled = add_background(axis, grid_x, grid_y, score_grid, "pca")

    hw_tp = stage_frame[(stage_frame["source"] == "HW") & stage_frame["is_tp"]]
    llm_tp = stage_frame[(stage_frame["source"] == "LLM") & stage_frame["is_tp"]]
    llm_fn = stage_frame[(stage_frame["source"] == "LLM") & stage_frame["is_fn"]]
    hw_fn = stage_frame[(stage_frame["source"] == "HW") & stage_frame["is_fn"]]

    axis.scatter(hw_tp["proj_x"], hw_tp["proj_y"], zorder=3.0, **TP_STYLE["HW"])
    axis.scatter(llm_tp["proj_x"], llm_tp["proj_y"], zorder=4.0, **TP_STYLE["LLM"])
    axis.scatter(llm_fn["proj_x"], llm_fn["proj_y"], zorder=5.0, **FN_STYLE["LLM"])
    axis.scatter(hw_fn["proj_x"], hw_fn["proj_y"], zorder=6.0, **FN_STYLE["HW"])
    axis.set_title(stage_name, fontsize=18, pad=10)
    axis.tick_params(axis="both", labelsize=TICK_LABEL_FONT_SIZE)
    axis.grid(alpha=0.10, linewidth=0.5)
    return filled


def draw_stage(stage_name: str, stage_frame: pd.DataFrame, seed: int) -> None:
    figure, axis = plt.subplots(figsize=(11.9, 9.0))
    filled = plot_stage_on_axis(axis, stage_name, stage_frame, seed)

    axis.set_xlabel("PCA1", fontsize=X_AXIS_LABEL_FONT_SIZE)
    axis.set_ylabel("PCA2", fontsize=24)
    axis.tick_params(axis="both", labelsize=TICK_LABEL_FONT_SIZE)

    add_compact_legend(axis)

    figure.subplots_adjust(left=0.09, right=0.90, bottom=0.10, top=0.98)
    axis_position = axis.get_position()
    colorbar_axis = figure.add_axes([axis_position.x1 + 0.012, axis_position.y0, 0.028, axis_position.height])
    colorbar = figure.colorbar(filled, cax=colorbar_axis)
    colorbar.set_label("Surrogate score", fontsize=17)
    colorbar.ax.tick_params(labelsize=15)
    output_path = OUTPUT_DIR / f"{stage_name}_stage_phishing_contours.png"
    figure.savefig(output_path, dpi=260, bbox_inches="tight", pad_inches=0.01)
    plt.close(figure)


def draw_stage_row(stage_names: list[str], output_name: str) -> None:
    frames = []
    for stage_name in stage_names:
        stage_csv = STAGES_DIR / stage_name / "projected_points.csv"
        frames.append((stage_name, load_stage_frame(stage_csv)))

    width = 6.8 * len(stage_names) + 2.2
    figure, axes = plt.subplots(1, len(stage_names), figsize=(width, 6.8), sharey=False)
    if len(stage_names) == 1:
        axes = [axes]
    else:
        axes = list(axes)

    filled = None
    for index, ((stage_name, stage_frame), axis) in enumerate(zip(frames, axes, strict=True)):
        filled = plot_stage_on_axis(axis, stage_name, stage_frame, 31 + STAGE_ORDER.index(stage_name))
        axis.set_xlabel("PCA1", fontsize=X_AXIS_LABEL_FONT_SIZE)
        if index == 0:
            axis.set_ylabel("PCA2", fontsize=18)
        else:
            axis.set_ylabel("")
        axis.tick_params(axis="both", labelsize=TICK_LABEL_FONT_SIZE)

    figure.subplots_adjust(left=0.04, right=0.955, bottom=0.10, top=0.94, wspace=0.12)
    add_compact_legend(axes[0])
    first_position = axes[0].get_position()
    last_position = axes[-1].get_position()
    colorbar_axis = figure.add_axes([last_position.x1 + 0.012, first_position.y0, 0.010, first_position.height])
    colorbar = figure.colorbar(filled, cax=colorbar_axis)
    colorbar.set_label("Surrogate score", fontsize=16)
    colorbar.ax.tick_params(labelsize=13)
    figure.savefig(OUTPUT_DIR / output_name, dpi=260, bbox_inches="tight", pad_inches=0.02)
    plt.close(figure)


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    for index, stage_name in enumerate(STAGE_ORDER):
        stage_csv = STAGES_DIR / stage_name / "projected_points.csv"
        if not stage_csv.exists():
            continue
        stage_frame = load_stage_frame(stage_csv)
        draw_stage(stage_name, stage_frame, seed=31 + index)

    draw_stage_row(["S6-MPG", "S6-UTA", "S6-fuzzer"], "S6_combined_stage_phishing_contours.png")
    draw_stage_row(
        ["S8-claude", "S8-deepseek", "S8-gemini", "S8-gpt", "S8-llama", "S8-ministral"],
        "S8_combined_stage_phishing_contours.png",
    )


if __name__ == "__main__":
    main()
