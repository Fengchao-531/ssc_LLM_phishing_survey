from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.lines import Line2D

from generate_stage_visualizations import STAGE_ORDER, compute_axis_limits, fit_detector_surface


def build_global_grid(frame: pd.DataFrame, surface_model):
    x_limits, y_limits = compute_axis_limits(frame)
    x_grid = np.linspace(*x_limits, 220)
    y_grid = np.linspace(*y_limits, 220)
    grid_x, grid_y = np.meshgrid(x_grid, y_grid)
    mesh = np.column_stack([grid_x.ravel(), grid_y.ravel()])
    score_grid = surface_model.predict_proba(mesh)[:, 1].reshape(grid_x.shape)
    return x_limits, y_limits, grid_x, grid_y, score_grid


def add_background(axis, grid_x, grid_y, score_grid, projection_method: str):
    filled = axis.contourf(
        grid_x,
        grid_y,
        score_grid,
        levels=np.linspace(0.0, 1.0, 13),
        cmap="YlOrRd",
        alpha=0.72,
    )
    axis.contour(
        grid_x,
        grid_y,
        score_grid,
        levels=[0.2, 0.4, 0.6, 0.8],
        colors="#8a4f15",
        linewidths=0.65,
        alpha=0.55,
    )
    axis.contour(
        grid_x,
        grid_y,
        score_grid,
        levels=[0.5],
        colors="black",
        linewidths=1.8,
    )
    axis.set_xlabel(f"{projection_method.upper()}1", fontsize=10)
    return filled


def add_outcome_points(axis, frame: pd.DataFrame, tp_style: dict, fn_style: dict):
    for source_name in ["HW", "LLM"]:
        tp_frame = frame[(frame["source"] == source_name) & frame["is_tp"]]
        fn_frame = frame[(frame["source"] == source_name) & frame["is_fn"]]
        axis.scatter(tp_frame["proj_x"], tp_frame["proj_y"], **tp_style[source_name])
        axis.scatter(fn_frame["proj_x"], fn_frame["proj_y"], **fn_style[source_name])


def build_main_legend():
    return [
        Line2D([0], [0], marker="o", color="none", markerfacecolor="none", markeredgecolor="#2f2f2f", markersize=7, linewidth=0, label="HW-TP"),
        Line2D([0], [0], marker="o", color="none", markerfacecolor="#2ca02c", markeredgecolor="white", markersize=7, linewidth=0, label="HW-FN"),
        Line2D([0], [0], marker="^", color="none", markerfacecolor="none", markeredgecolor="#2f2f2f", markersize=7, linewidth=0, label="LLM-TP"),
        Line2D([0], [0], marker="^", color="none", markerfacecolor="#1f77b4", markeredgecolor="white", markersize=7, linewidth=0, label="LLM-FN"),
        Line2D([0], [0], color="black", linewidth=1.8, label="threshold contour: s(x)=0.50"),
    ]


def build_source_legend():
    return [
        Line2D([0], [0], marker="o", color="none", markerfacecolor="#7a1f3d", markeredgecolor="white", markersize=7, linewidth=0, label="HW phishing"),
        Line2D([0], [0], marker="^", color="none", markerfacecolor="#2f6b3b", markeredgecolor="white", markersize=7, linewidth=0, label="LLM phishing"),
        Line2D([0], [0], color="black", linewidth=1.8, label="threshold contour: s(x)=0.50"),
    ]


def annotate_overview(axis, frame: pd.DataFrame, group_order: list[str]):
    counts = frame["group"].value_counts()
    summary = " | ".join(f"{group}={int(counts.get(group, 0))}" for group in group_order)
    axis.set_title(summary, fontsize=10)


def draw_fig0_source_overview(frame, output_path: Path, projection_method: str, grid_x, grid_y, score_grid, source_style: dict):
    figure, axis = plt.subplots(figsize=(11.5, 9.2))
    filled = add_background(axis, grid_x, grid_y, score_grid, projection_method)
    for source_name in ["HW", "LLM"]:
        source_frame = frame[frame["source"] == source_name]
        axis.scatter(source_frame["proj_x"], source_frame["proj_y"], **source_style[source_name])

    hw_count = int((frame["source"] == "HW").sum())
    llm_count = int((frame["source"] == "LLM").sum())
    axis.set_title(f"HW phishing={hw_count} | LLM phishing={llm_count}", fontsize=12)
    axis.set_ylabel(f"{projection_method.upper()}2", fontsize=10)
    axis.grid(alpha=0.10, linewidth=0.5)
    axis.legend(
        handles=build_source_legend(),
        loc="lower center",
        bbox_to_anchor=(0.5, -0.16),
        ncol=3,
        frameon=False,
    )
    colorbar = figure.colorbar(filled, ax=axis, fraction=0.04, pad=0.02)
    colorbar.set_label("Surrogate score")
    figure.suptitle("Figure 0. Overall HW vs LLM Phishing Map", y=0.98, fontsize=15)
    figure.tight_layout(rect=(0, 0.04, 1, 0.95))
    figure.savefig(output_path, dpi=220, bbox_inches="tight")
    plt.close(figure)


def draw_fig1_main_map(frame, output_path: Path, projection_method: str, grid_x, grid_y, score_grid, group_order: list[str], tp_style: dict, fn_style: dict):
    figure, axis = plt.subplots(figsize=(11.5, 9.2))
    filled = add_background(axis, grid_x, grid_y, score_grid, projection_method)
    add_outcome_points(axis, frame, tp_style, fn_style)
    axis.set_ylabel(f"{projection_method.upper()}2", fontsize=10)
    annotate_overview(axis, frame, group_order)
    axis.grid(alpha=0.10, linewidth=0.5)
    axis.legend(
        handles=build_main_legend(),
        loc="lower center",
        bbox_to_anchor=(0.5, -0.16),
        ncol=3,
        frameon=False,
    )
    colorbar = figure.colorbar(filled, ax=axis, fraction=0.04, pad=0.02)
    colorbar.set_label("Surrogate score")
    figure.suptitle("Figure 1. Surrogate Response Map with TP/FN Distribution", y=0.98, fontsize=15)
    figure.tight_layout(rect=(0, 0.04, 1, 0.95))
    figure.savefig(output_path, dpi=220, bbox_inches="tight")
    plt.close(figure)


def draw_fig2_dual_panel(frame, output_path: Path, projection_method: str, grid_x, grid_y, score_grid, tp_style: dict, fn_style: dict):
    figure, axes = plt.subplots(1, 2, figsize=(15.8, 7.2), sharex=True, sharey=True)
    source_titles = {"HW": "HW-phishing", "LLM": "LLM-phishing"}
    filled = None
    for axis, source_name in zip(axes, ["HW", "LLM"], strict=True):
        filled = add_background(axis, grid_x, grid_y, score_grid, projection_method)
        source_frame = frame[frame["source"] == source_name]
        add_outcome_points(axis, source_frame, tp_style, fn_style)
        tp_count = int(((source_frame["source"] == source_name) & source_frame["is_tp"]).sum())
        fn_count = int(((source_frame["source"] == source_name) & source_frame["is_fn"]).sum())
        axis.set_title(f"{source_titles[source_name]}\nTP={tp_count} | FN={fn_count}", fontsize=12)
        axis.grid(alpha=0.10, linewidth=0.5)
    axes[0].set_ylabel(f"{projection_method.upper()}2", fontsize=10)
    axes[0].legend(
        handles=build_main_legend(),
        loc="lower center",
        bbox_to_anchor=(1.05, -0.22),
        ncol=3,
        frameon=False,
    )
    colorbar_axis = figure.add_axes([0.487, 0.20, 0.014, 0.58])
    colorbar = figure.colorbar(filled, cax=colorbar_axis)
    colorbar.set_label("Surrogate score")
    figure.suptitle("Figure 2. Dual-Panel Surrogate Map: HW vs LLM", y=0.98, fontsize=15)
    figure.subplots_adjust(left=0.06, right=0.94, bottom=0.18, top=0.82, wspace=0.08)
    figure.savefig(output_path, dpi=220, bbox_inches="tight")
    plt.close(figure)


def draw_fig3_fn_only(frame, output_path: Path, projection_method: str, grid_x, grid_y, score_grid, fn_style: dict):
    figure, axis = plt.subplots(figsize=(11.5, 9.2))
    filled = add_background(axis, grid_x, grid_y, score_grid, projection_method)
    tp_background = frame[frame["is_tp"]]
    axis.scatter(
        tp_background["proj_x"],
        tp_background["proj_y"],
        s=9,
        c="#d9d9d9",
        alpha=0.18,
        linewidths=0,
    )
    for source_name in ["HW", "LLM"]:
        fn_frame = frame[(frame["source"] == source_name) & frame["is_fn"]]
        axis.scatter(fn_frame["proj_x"], fn_frame["proj_y"], **fn_style[source_name])
    hw_fn = int(((frame["source"] == "HW") & frame["is_fn"]).sum())
    llm_fn = int(((frame["source"] == "LLM") & frame["is_fn"]).sum())
    axis.set_title(f"FN-only emphasis | HW-FN={hw_fn} | LLM-FN={llm_fn}", fontsize=12)
    axis.set_xlabel(f"{projection_method.upper()}1", fontsize=10)
    axis.set_ylabel(f"{projection_method.upper()}2", fontsize=10)
    axis.grid(alpha=0.10, linewidth=0.5)
    legend_handles = [
        Line2D([0], [0], marker="o", color="none", markerfacecolor="#2ca02c", markeredgecolor="white", markersize=7, linewidth=0, label="HW-FN"),
        Line2D([0], [0], marker="^", color="none", markerfacecolor="#1f77b4", markeredgecolor="white", markersize=7, linewidth=0, label="LLM-FN"),
        Line2D([0], [0], marker="o", color="none", markerfacecolor="#d9d9d9", markeredgecolor="#d9d9d9", markersize=5, linewidth=0, label="TP background"),
        Line2D([0], [0], color="black", linewidth=1.8, label="threshold contour: s(x)=0.50"),
    ]
    axis.legend(
        handles=legend_handles,
        loc="lower center",
        bbox_to_anchor=(0.5, -0.16),
        ncol=2,
        frameon=False,
    )
    colorbar = figure.colorbar(filled, ax=axis, fraction=0.04, pad=0.02)
    colorbar.set_label("Surrogate score")
    figure.suptitle("Figure 3. FN-only Map", y=0.98, fontsize=15)
    figure.tight_layout(rect=(0, 0.04, 1, 0.95))
    figure.savefig(output_path, dpi=220, bbox_inches="tight")
    plt.close(figure)


def draw_fig1_stagewise_map(frame, output_path: Path, projection_method: str, grid_x, grid_y, seed: int, tp_style: dict, fn_style: dict):
    figure, axes = plt.subplots(2, 5, figsize=(27.5, 12.0), sharex=True, sharey=True)
    axes_flat = axes.flatten()
    legend_handles = build_main_legend()
    filled = None
    for index, stage_name in enumerate(STAGE_ORDER):
        axis = axes_flat[index]
        stage_frame = frame[frame["stage"] == stage_name]
        stage_surface_model = fit_detector_surface(stage_frame, seed + index)
        if stage_surface_model is not None:
            mesh = np.column_stack([grid_x.ravel(), grid_y.ravel()])
            stage_score_grid = stage_surface_model.predict_proba(mesh)[:, 1].reshape(grid_x.shape)
        else:
            stage_score_grid = np.full_like(
                grid_x,
                float(stage_frame["detector_prediction"].astype(int).mean()) if len(stage_frame) else 0.0,
                dtype=float,
            )
        filled = add_background(axis, grid_x, grid_y, stage_score_grid, projection_method)
        add_outcome_points(axis, stage_frame, tp_style, fn_style)
        hw_tp = int(((stage_frame["source"] == "HW") & stage_frame["is_tp"]).sum())
        hw_fn = int(((stage_frame["source"] == "HW") & stage_frame["is_fn"]).sum())
        llm_tp = int(((stage_frame["source"] == "LLM") & stage_frame["is_tp"]).sum())
        llm_fn = int(((stage_frame["source"] == "LLM") & stage_frame["is_fn"]).sum())
        axis.set_title(stage_name, fontsize=12, pad=10)
        axis.text(
            0.02,
            0.98,
            f"HW: TP={hw_tp}  FN={hw_fn}\nLLM: TP={llm_tp}  FN={llm_fn}",
            transform=axis.transAxes,
            ha="left",
            va="top",
            fontsize=8.8,
            bbox={
                "boxstyle": "round,pad=0.22",
                "facecolor": "white",
                "edgecolor": "#d0d0d0",
                "alpha": 0.82,
            },
        )
        axis.grid(alpha=0.10, linewidth=0.5)
        if index % 5 == 0:
            axis.set_ylabel(f"{projection_method.upper()}2", fontsize=10)
        if index >= 5:
            axis.set_xlabel(f"{projection_method.upper()}1", fontsize=10)

    for axis in axes_flat[len(STAGE_ORDER):]:
        axis.axis("off")

    figure.legend(
        handles=legend_handles,
        loc="lower center",
        bbox_to_anchor=(0.45, 0.035),
        ncol=5,
        frameon=False,
    )
    colorbar_axis = figure.add_axes([0.945, 0.20, 0.012, 0.60])
    colorbar = figure.colorbar(filled, cax=colorbar_axis)
    colorbar.set_label("Surrogate score")
    figure.suptitle("Figure 1b. Stage-wise ScamLLM Surrogate Response with TP/FN Distribution", y=0.98, fontsize=15)
    figure.subplots_adjust(left=0.05, right=0.93, bottom=0.12, top=0.88, wspace=0.10, hspace=0.26)
    figure.savefig(output_path, dpi=220, bbox_inches="tight")
    plt.close(figure)
