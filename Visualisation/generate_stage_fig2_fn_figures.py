#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.colors import Normalize

from generate_stage_split_figures import STAGE_ORDER
from generate_stage_visualizations import compute_axis_limits, fit_detector_surface
from generate_test_failure_contours import add_background, build_main_legend
from generate_test_failure_mode_figures import FN_STYLE, TP_STYLE

SCRIPT_DIR = Path(__file__).resolve().parent
DEFAULT_STAGES_DIR = SCRIPT_DIR / "test" / "stages"
PRINCIPLE_COLUMNS = [
    "principle_authority",
    "principle_liking",
    "principle_reciprocity",
    "principle_social_proof",
    "principle_scarcity",
    "principle_commitment",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Generate stage-specific Figure 2 style phishing contour maps and "
            "FN-only heatmaps under Visualization/test/stages."
        )
    )
    parser.add_argument("--stages-dir", type=Path, default=DEFAULT_STAGES_DIR)
    parser.add_argument("--projection", choices=["pca", "umap"], default="pca")
    parser.add_argument("--seed", type=int, default=31)
    return parser.parse_args()


def load_stage_projected_frame(stage_dir: Path) -> pd.DataFrame:
    csv_path = stage_dir / "projected_points.csv"
    frame = pd.read_csv(csv_path)
    required = {
        "source",
        "proj_x",
        "proj_y",
        "detector_prediction",
        "principle_authority",
        "principle_reciprocity",
        "principle_commitment",
        "principle_scarcity",
        "principle_social_proof",
        "principle_liking",
    }
    missing = sorted(required.difference(frame.columns))
    if missing:
        raise ValueError(f"{csv_path} is missing required columns: {', '.join(missing)}")
    frame = frame.copy()
    frame["is_tp"] = pd.to_numeric(frame["detector_prediction"], errors="coerce").fillna(0.0) >= 0.5
    frame["is_fn"] = ~frame["is_tp"]
    return frame


def compute_principle_cooccurrence_matrix(frame: pd.DataFrame) -> tuple[np.ndarray, int]:
    principle_frame = frame[PRINCIPLE_COLUMNS].dropna(how="all")
    if principle_frame.empty:
        return np.full((len(PRINCIPLE_COLUMNS), len(PRINCIPLE_COLUMNS)), np.nan, dtype=float), 0
    values = principle_frame.to_numpy(dtype=float)
    matrix = (values.T @ values) / max(1, values.shape[0])
    diagonal = np.nanmean(values, axis=0)
    np.fill_diagonal(matrix, diagonal)
    return matrix, int(values.shape[0])


def draw_stage_contour(stage_frame: pd.DataFrame, stage_name: str, output_path: Path, projection_method: str, seed: int) -> dict[str, int]:
    surface_model = fit_detector_surface(stage_frame, seed)
    if surface_model is None:
        raise RuntimeError(f"Could not fit phishing-only surface for {stage_name}.")

    x_limits, y_limits = compute_axis_limits(stage_frame)
    x_grid = np.linspace(*x_limits, 220)
    y_grid = np.linspace(*y_limits, 220)
    grid_x, grid_y = np.meshgrid(x_grid, y_grid)
    mesh = np.column_stack([grid_x.ravel(), grid_y.ravel()])
    score_grid = surface_model.predict_proba(mesh)[:, 1].reshape(grid_x.shape)

    figure = plt.figure(figsize=(11.6, 8.8))
    grid = figure.add_gridspec(
        2,
        2,
        width_ratios=[1.0, 0.06],
        height_ratios=[1.0, 0.18],
        wspace=0.08,
        hspace=0.08,
    )
    axis = figure.add_subplot(grid[0, 0])
    colorbar_axis = figure.add_subplot(grid[0, 1])
    legend_axis = figure.add_subplot(grid[1, 0])
    legend_axis.axis("off")

    filled = add_background(axis, grid_x, grid_y, score_grid, projection_method)
    for source_name in ["HW", "LLM"]:
        tp_frame = stage_frame[(stage_frame["source"] == source_name) & stage_frame["is_tp"]]
        fn_frame = stage_frame[(stage_frame["source"] == source_name) & stage_frame["is_fn"]]
        axis.scatter(tp_frame["proj_x"], tp_frame["proj_y"], **TP_STYLE[source_name])
        axis.scatter(fn_frame["proj_x"], fn_frame["proj_y"], **FN_STYLE[source_name])

    hw_tp = int(((stage_frame["source"] == "HW") & stage_frame["is_tp"]).sum())
    hw_fn = int(((stage_frame["source"] == "HW") & stage_frame["is_fn"]).sum())
    llm_tp = int(((stage_frame["source"] == "LLM") & stage_frame["is_tp"]).sum())
    llm_fn = int(((stage_frame["source"] == "LLM") & stage_frame["is_fn"]).sum())

    axis.set_title(f"{stage_name}: HW-P vs LLM-P", fontsize=13, pad=10)
    axis.text(
        0.02,
        0.98,
        f"HW: TP={hw_tp}  FN={hw_fn}\nLLM: TP={llm_tp}  FN={llm_fn}",
        transform=axis.transAxes,
        ha="left",
        va="top",
        fontsize=9.2,
        bbox={
            "boxstyle": "round,pad=0.24",
            "facecolor": "white",
            "edgecolor": "#d0d0d0",
            "alpha": 0.84,
        },
    )
    axis.set_xlabel(f"{projection_method.upper()}1", fontsize=10)
    axis.set_ylabel(f"{projection_method.upper()}2", fontsize=10)
    axis.grid(alpha=0.10, linewidth=0.5)

    legend_axis.legend(handles=build_main_legend(), loc="center", ncol=3, frameon=False)
    figure.colorbar(filled, cax=colorbar_axis).set_label("Surrogate score")
    figure.suptitle(f"Stage Figure 2A. Phishing Surrogate Map: {stage_name}", y=0.97, fontsize=15)
    figure.savefig(output_path, dpi=220)
    plt.close(figure)
    return {"hw_tp": hw_tp, "hw_fn": hw_fn, "llm_tp": llm_tp, "llm_fn": llm_fn}


def draw_stage_fn_heatmaps(stage_frame: pd.DataFrame, stage_name: str, output_path: Path) -> dict[str, int]:
    groups = [
        ("HW-P FN", stage_frame[(stage_frame["source"] == "HW") & stage_frame["is_fn"]].copy()),
        ("LLM-P FN", stage_frame[(stage_frame["source"] == "LLM") & stage_frame["is_fn"]].copy()),
    ]
    labels = [name.replace("principle_", "").replace("_", " ").title() for name in PRINCIPLE_COLUMNS]

    figure = plt.figure(figsize=(11.6, 5.8))
    grid = figure.add_gridspec(1, 3, width_ratios=[1.0, 1.0, 0.08], wspace=0.05)
    axes = [figure.add_subplot(grid[0, 0]), figure.add_subplot(grid[0, 1])]
    colorbar_axis = figure.add_subplot(grid[0, 2])

    counts: dict[str, int] = {}
    images = []
    for axis, (title, subset) in zip(axes, groups, strict=True):
        matrix, count = compute_principle_cooccurrence_matrix(subset)
        counts[title] = count
        safe_matrix = np.nan_to_num(matrix, nan=0.0, posinf=0.0, neginf=0.0)
        image = axis.imshow(safe_matrix, aspect="equal", cmap="GnBu", norm=Normalize(vmin=0.0, vmax=1.0))
        images.append(image)
        axis.set_xticks(range(len(PRINCIPLE_COLUMNS)))
        axis.set_xticklabels(labels, rotation=45, ha="right", fontsize=8.2)
        axis.set_yticks(range(len(PRINCIPLE_COLUMNS)))
        axis.set_yticklabels(labels if title == "HW-P FN" else [], fontsize=8.2)
        axis.set_title(title, fontsize=12)
        for row_index in range(safe_matrix.shape[0]):
            for col_index in range(safe_matrix.shape[1]):
                label = "NA" if np.isnan(matrix[row_index, col_index]) else f"{matrix[row_index, col_index]:.2f}"
                axis.text(col_index, row_index, label, ha="center", va="center", fontsize=7.2, color="black")

    figure.colorbar(images[-1], cax=colorbar_axis).set_label("Mean co-occurrence strength")
    figure.suptitle(f"Stage Figure 2B. FN-only Persuasion Heatmaps: {stage_name}", y=0.98, fontsize=15)
    figure.subplots_adjust(left=0.06, right=0.95, bottom=0.10, top=0.88)
    figure.savefig(output_path, dpi=220)
    plt.close(figure)
    return {"hw_fn_heatmap_rows": int(counts["HW-P FN"]), "llm_fn_heatmap_rows": int(counts["LLM-P FN"])}


def draw_stage_llm_tp_heatmap(stage_frame: pd.DataFrame, stage_name: str, output_path: Path) -> dict[str, int]:
    subset = stage_frame[(stage_frame["source"] == "LLM") & stage_frame["is_tp"]].copy()
    matrix, count = compute_principle_cooccurrence_matrix(subset)
    safe_matrix = np.nan_to_num(matrix, nan=0.0, posinf=0.0, neginf=0.0)
    labels = [name.replace("principle_", "").replace("_", " ").title() for name in PRINCIPLE_COLUMNS]

    figure = plt.figure(figsize=(6.4, 5.8))
    grid = figure.add_gridspec(1, 2, width_ratios=[1.0, 0.08], wspace=0.06)
    axis = figure.add_subplot(grid[0, 0])
    colorbar_axis = figure.add_subplot(grid[0, 1])

    image = axis.imshow(safe_matrix, aspect="equal", cmap="GnBu", norm=Normalize(vmin=0.0, vmax=1.0))
    axis.set_xticks(range(len(PRINCIPLE_COLUMNS)))
    axis.set_xticklabels(labels, rotation=45, ha="right", fontsize=8.2)
    axis.set_yticks(range(len(PRINCIPLE_COLUMNS)))
    axis.set_yticklabels(labels, fontsize=8.2)
    axis.set_title("LLM-P TP", fontsize=12)
    for row_index in range(safe_matrix.shape[0]):
        for col_index in range(safe_matrix.shape[1]):
            label = "NA" if np.isnan(matrix[row_index, col_index]) else f"{matrix[row_index, col_index]:.2f}"
            axis.text(col_index, row_index, label, ha="center", va="center", fontsize=7.2, color="black")

    figure.colorbar(image, cax=colorbar_axis).set_label("Mean co-occurrence strength")
    figure.suptitle(f"Stage Figure 2C. LLM-P TP Persuasion Heatmap: {stage_name}", y=0.98, fontsize=15)
    figure.subplots_adjust(left=0.11, right=0.94, bottom=0.16, top=0.87)
    figure.savefig(output_path, dpi=220)
    plt.close(figure)
    return {"llm_tp_heatmap_rows": int(count)}


def draw_stage_hw_tp_heatmap(stage_frame: pd.DataFrame, stage_name: str, output_path: Path) -> dict[str, int]:
    subset = stage_frame[(stage_frame["source"] == "HW") & stage_frame["is_tp"]].copy()
    matrix, count = compute_principle_cooccurrence_matrix(subset)
    safe_matrix = np.nan_to_num(matrix, nan=0.0, posinf=0.0, neginf=0.0)
    labels = [name.replace("principle_", "").replace("_", " ").title() for name in PRINCIPLE_COLUMNS]

    figure = plt.figure(figsize=(6.4, 5.8))
    grid = figure.add_gridspec(1, 2, width_ratios=[1.0, 0.08], wspace=0.06)
    axis = figure.add_subplot(grid[0, 0])
    colorbar_axis = figure.add_subplot(grid[0, 1])

    image = axis.imshow(safe_matrix, aspect="equal", cmap="GnBu", norm=Normalize(vmin=0.0, vmax=1.0))
    axis.set_xticks(range(len(PRINCIPLE_COLUMNS)))
    axis.set_xticklabels(labels, rotation=45, ha="right", fontsize=8.2)
    axis.set_yticks(range(len(PRINCIPLE_COLUMNS)))
    axis.set_yticklabels(labels, fontsize=8.2)
    axis.set_title("HW-P TP", fontsize=12)
    for row_index in range(safe_matrix.shape[0]):
        for col_index in range(safe_matrix.shape[1]):
            label = "NA" if np.isnan(matrix[row_index, col_index]) else f"{matrix[row_index, col_index]:.2f}"
            axis.text(col_index, row_index, label, ha="center", va="center", fontsize=7.2, color="black")

    figure.colorbar(image, cax=colorbar_axis).set_label("Mean co-occurrence strength")
    figure.suptitle(f"Stage Figure 2D. HW-P TP Persuasion Heatmap: {stage_name}", y=0.98, fontsize=15)
    figure.subplots_adjust(left=0.11, right=0.94, bottom=0.16, top=0.87)
    figure.savefig(output_path, dpi=220)
    plt.close(figure)
    return {"hw_tp_heatmap_rows": int(count)}


def main() -> None:
    args = parse_args()
    stages_dir = args.stages_dir.resolve()
    stages_dir.mkdir(parents=True, exist_ok=True)

    run_metadata: list[dict[str, object]] = []
    for index, stage_name in enumerate(STAGE_ORDER):
        stage_dir = stages_dir / stage_name
        stage_frame = load_stage_projected_frame(stage_dir)
        contour_path = stage_dir / "fig2_stage_phishing_contours_pca.png"
        heatmap_path = stage_dir / "fig2_stage_fn_heatmaps_pca.png"
        llm_tp_heatmap_path = stage_dir / "fig2_stage_llm_tp_heatmap_pca.png"
        hw_tp_heatmap_path = stage_dir / "fig2_stage_hw_tp_heatmap_pca.png"

        contour_counts = draw_stage_contour(
            stage_frame,
            stage_name,
            contour_path,
            args.projection,
            args.seed + index,
        )
        heatmap_counts = draw_stage_fn_heatmaps(stage_frame, stage_name, heatmap_path)
        llm_tp_counts = draw_stage_llm_tp_heatmap(stage_frame, stage_name, llm_tp_heatmap_path)
        hw_tp_counts = draw_stage_hw_tp_heatmap(stage_frame, stage_name, hw_tp_heatmap_path)

        metadata = {
            "stage": stage_name,
            "projection_used": args.projection,
            "row_count": int(len(stage_frame)),
            "files": {
                "contour_figure": contour_path.name,
                "heatmap_figure": heatmap_path.name,
                "llm_tp_heatmap_figure": llm_tp_heatmap_path.name,
                "hw_tp_heatmap_figure": hw_tp_heatmap_path.name,
            },
            "counts": {**contour_counts, **heatmap_counts, **llm_tp_counts, **hw_tp_counts},
        }
        (stage_dir / "fig2_stage_fn_metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
        run_metadata.append(metadata)

    summary = {
        "projection_used": args.projection,
        "stages_dir": str(stages_dir),
        "stage_order": STAGE_ORDER,
        "stages": run_metadata,
    }
    (stages_dir / "fig2_stage_fn_run_metadata.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
