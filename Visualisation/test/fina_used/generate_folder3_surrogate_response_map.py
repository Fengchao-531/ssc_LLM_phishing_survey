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
TEST_DIR = VISUALIZATION_DIR / "test"
INPUT_PATH = TEST_DIR / "projected_points.csv"
OUTPUT_DIR = SCRIPT_DIR / "3"
OUTPUT_PATH = OUTPUT_DIR / "surrogate_response_tp_fn_map.png"

if str(VISUALIZATION_DIR) not in sys.path:
    sys.path.insert(0, str(VISUALIZATION_DIR))

from generate_stage_visualizations import fit_detector_surface  # noqa: E402
from generate_test_failure_contours import add_background, build_global_grid  # noqa: E402

X_AXIS_LABEL_FONT_SIZE = 29
Y_AXIS_LABEL_FONT_SIZE = 28
TICK_LABEL_FONT_SIZE = 26
RIGHT_LEGEND_FONT_SIZE = 23
COLORBAR_LABEL_FONT_SIZE = 19


TP_STYLE = {
    "HW": {"marker": "o", "facecolors": "none", "edgecolors": "#2f2f2f", "linewidths": 1.2, "s": 62, "alpha": 0.88},
    "LLM": {"marker": "^", "facecolors": "none", "edgecolors": "#3d3d3d", "linewidths": 1.15, "s": 74, "alpha": 0.88},
}
FN_STYLE = {
    "HW": {"marker": "o", "c": "#2ca02c", "edgecolors": "white", "linewidths": 0.50, "s": 70, "alpha": 0.95},
    "LLM": {"marker": "^", "c": "#3498db", "edgecolors": "white", "linewidths": 0.45, "s": 84, "alpha": 0.92},
}


def load_frame(csv_path: Path) -> pd.DataFrame:
    frame = pd.read_csv(csv_path, low_memory=False).copy()
    required_columns = {"source", "proj_x", "proj_y", "scamllm"}
    missing = sorted(required_columns.difference(frame.columns))
    if missing:
        raise ValueError(f"{csv_path} is missing required columns: {', '.join(missing)}")

    detector_prediction = pd.to_numeric(frame["scamllm"], errors="coerce").fillna(0.0)
    frame["is_tp"] = detector_prediction >= 0.5
    frame["is_fn"] = ~frame["is_tp"]
    return frame


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
        (0.292, 0.832),
        0.50,
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
        bbox_to_anchor=(0.314, 0.986),
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
        bbox_to_anchor=(0.515, 0.986),
        fontsize=RIGHT_LEGEND_FONT_SIZE,
        frameon=False,
        borderpad=0.0,
        labelspacing=0.45,
        handletextpad=0.6,
        handlelength=1.5,
    )
    right_legend.set_zorder(7.5)
    axis.add_artist(right_legend)


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    frame = load_frame(INPUT_PATH)
    surface_model = fit_detector_surface(frame, seed=7)
    if surface_model is None:
        raise RuntimeError("Could not fit the surrogate surface for the phishing-only projected frame.")

    _, _, grid_x, grid_y, score_grid = build_global_grid(frame, surface_model)

    figure, axis = plt.subplots(figsize=(11.9, 9.0))
    filled = add_background(axis, grid_x, grid_y, score_grid, "pca")

    llm_tp = frame[(frame["source"] == "LLM") & frame["is_tp"]]
    llm_fn = frame[(frame["source"] == "LLM") & frame["is_fn"]]
    hw_tp = frame[(frame["source"] == "HW") & frame["is_tp"]]
    hw_fn = frame[(frame["source"] == "HW") & frame["is_fn"]]

    axis.scatter(hw_tp["proj_x"], hw_tp["proj_y"], zorder=3.0, **TP_STYLE["HW"])
    axis.scatter(llm_tp["proj_x"], llm_tp["proj_y"], zorder=4.0, **TP_STYLE["LLM"])
    axis.scatter(llm_fn["proj_x"], llm_fn["proj_y"], zorder=5.0, **FN_STYLE["LLM"])
    axis.scatter(hw_fn["proj_x"], hw_fn["proj_y"], zorder=6.0, **FN_STYLE["HW"])

    axis.set_xlabel("PCA1", fontsize=X_AXIS_LABEL_FONT_SIZE)
    axis.set_ylabel("PCA2", fontsize=Y_AXIS_LABEL_FONT_SIZE)
    axis.tick_params(axis="both", labelsize=TICK_LABEL_FONT_SIZE)
    axis.grid(alpha=0.10, linewidth=0.5)

    add_compact_legend(axis)

    figure.subplots_adjust(left=0.09, right=0.90, bottom=0.10, top=0.98)
    axis_position = axis.get_position()
    colorbar_axis = figure.add_axes([axis_position.x1 + 0.012, axis_position.y0, 0.028, axis_position.height])
    colorbar = figure.colorbar(filled, cax=colorbar_axis)
    colorbar.set_label("Surrogate score", fontsize=COLORBAR_LABEL_FONT_SIZE)
    colorbar.ax.tick_params(labelsize=15)
    figure.savefig(OUTPUT_PATH, dpi=260, bbox_inches="tight", pad_inches=0.01)
    plt.close(figure)


if __name__ == "__main__":
    main()
