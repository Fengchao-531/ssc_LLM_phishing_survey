#!/usr/bin/env python3
from __future__ import annotations

import json
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

VIS_ROOT = Path(__file__).resolve().parents[2]
if str(VIS_ROOT) not in sys.path:
    sys.path.insert(0, str(VIS_ROOT))

from generate_stage_visualizations import compute_axis_limits, fit_detector_surface
from generate_test_failure_contours import add_background, build_main_legend
from generate_test_failure_mode_figures import FN_STYLE, TP_STYLE

SCRIPT_DIR = Path(__file__).resolve().parent
DEFAULT_INPUT = SCRIPT_DIR / "projected_points_all_stages.csv"
DEFAULT_OUTPUT = SCRIPT_DIR / "fig_s6_stagewise_surrogate_response_tp_fn_map_pca.png"
DEFAULT_METADATA = SCRIPT_DIR / "fig_s6_stagewise_surrogate_response_tp_fn_map_pca.json"
S6_STAGE_ORDER = ["S6-MPG", "S6-UTA", "S6-fuzzer"]


def add_outcome_points(axis: plt.Axes, frame: pd.DataFrame) -> None:
    for source_name in ["HW", "LLM"]:
        tp_frame = frame[(frame["source"] == source_name) & frame["is_tp"]]
        fn_frame = frame[(frame["source"] == source_name) & frame["is_fn"]]
        axis.scatter(tp_frame["proj_x"], tp_frame["proj_y"], **TP_STYLE[source_name])
        axis.scatter(fn_frame["proj_x"], fn_frame["proj_y"], **FN_STYLE[source_name])


def main() -> None:
    frame = pd.read_csv(DEFAULT_INPUT, low_memory=False)
    frame = frame[frame["stage"].isin(S6_STAGE_ORDER)].copy()
    frame["detector_prediction"] = pd.to_numeric(frame["detector_prediction"], errors="coerce").fillna(0.0)
    if "is_tp" not in frame.columns:
        frame["is_tp"] = frame["detector_prediction"] >= 0.5
    if "is_fn" not in frame.columns:
        frame["is_fn"] = ~frame["is_tp"]

    x_limits, y_limits = compute_axis_limits(frame)
    x_grid = np.linspace(*x_limits, 220)
    y_grid = np.linspace(*y_limits, 220)
    grid_x, grid_y = np.meshgrid(x_grid, y_grid)
    mesh = np.column_stack([grid_x.ravel(), grid_y.ravel()])

    figure = plt.figure(figsize=(18.2, 8.8))
    grid = figure.add_gridspec(
        2,
        4,
        width_ratios=[1.0, 1.0, 1.0, 0.07],
        height_ratios=[1.0, 0.18],
        wspace=0.10,
        hspace=0.12,
    )
    axes = [
        figure.add_subplot(grid[0, 0]),
        figure.add_subplot(grid[0, 1]),
        figure.add_subplot(grid[0, 2]),
    ]
    colorbar_axis = figure.add_subplot(grid[0, 3])
    legend_axis = figure.add_subplot(grid[1, :3])
    legend_axis.axis("off")

    filled = None
    metadata_rows: list[dict[str, object]] = []
    for axis, stage_name in zip(axes, S6_STAGE_ORDER, strict=True):
        stage_frame = frame[frame["stage"] == stage_name].copy()
        surface_model = fit_detector_surface(stage_frame, 7)
        if surface_model is not None:
            score_grid = surface_model.predict_proba(mesh)[:, 1].reshape(grid_x.shape)
        else:
            score_grid = np.full_like(
                grid_x,
                float(stage_frame["detector_prediction"].astype(int).mean()) if len(stage_frame) else 0.0,
                dtype=float,
            )

        filled = add_background(axis, grid_x, grid_y, score_grid, "pca")
        add_outcome_points(axis, stage_frame)

        hw_tp = int(((stage_frame["source"] == "HW") & stage_frame["is_tp"]).sum())
        hw_fn = int(((stage_frame["source"] == "HW") & stage_frame["is_fn"]).sum())
        llm_tp = int(((stage_frame["source"] == "LLM") & stage_frame["is_tp"]).sum())
        llm_fn = int(((stage_frame["source"] == "LLM") & stage_frame["is_fn"]).sum())

        axis.set_title(stage_name, fontsize=12, pad=8)
        axis.text(
            0.02,
            0.98,
            f"HW: TP={hw_tp}  FN={hw_fn}\nLLM: TP={llm_tp}  FN={llm_fn}",
            transform=axis.transAxes,
            ha="left",
            va="top",
            fontsize=8.6,
            bbox={
                "boxstyle": "round,pad=0.22",
                "facecolor": "white",
                "edgecolor": "#d0d0d0",
                "alpha": 0.84,
            },
        )
        axis.set_xlim(*x_limits)
        axis.set_ylim(*y_limits)
        axis.set_xlabel("PCA1", fontsize=10)
        axis.grid(alpha=0.10, linewidth=0.5)
        metadata_rows.append(
            {
                "stage": stage_name,
                "counts": {
                    "HW": {"tp": hw_tp, "fn": hw_fn, "total": int((stage_frame["source"] == "HW").sum())},
                    "LLM": {"tp": llm_tp, "fn": llm_fn, "total": int((stage_frame["source"] == "LLM").sum())},
                },
            }
        )

    axes[0].set_ylabel("PCA2", fontsize=10)
    legend_axis.legend(
        handles=build_main_legend(),
        loc="center",
        ncol=3,
        frameon=False,
    )
    colorbar = figure.colorbar(filled, cax=colorbar_axis)
    colorbar.set_label("Surrogate score")
    figure.suptitle("S6 Stage-wise Surrogate Response Maps", y=0.98, fontsize=15)
    figure.subplots_adjust(left=0.05, right=0.96, bottom=0.10, top=0.90)
    figure.savefig(DEFAULT_OUTPUT, dpi=220)
    plt.close(figure)

    metadata = {
        "input_file": str(DEFAULT_INPUT),
        "output_file": DEFAULT_OUTPUT.name,
        "stages": metadata_rows,
    }
    DEFAULT_METADATA.write_text(json.dumps(metadata, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
