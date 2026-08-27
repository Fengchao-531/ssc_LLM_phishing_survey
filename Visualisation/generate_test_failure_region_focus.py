#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.lines import Line2D
from matplotlib.patches import Rectangle

from generate_stage_visualizations import fit_detector_surface
from generate_test_failure_contours import add_background, build_global_grid
from generate_test_failure_heatmaps import PRINCIPLE_NAMES, WVAE_COLUMN_BY_PRINCIPLE, merge_projected_with_wvae

SCRIPT_DIR = Path(__file__).resolve().parent
DEFAULT_PROJECTED_INPUT = SCRIPT_DIR / "test" / "projected_points.csv"
DEFAULT_OUTPUT_DIR = SCRIPT_DIR / "test"

TP_STYLE = {
    "HW": {"marker": "o", "facecolors": "none", "edgecolors": "#2f2f2f", "linewidths": 1.0, "s": 26, "alpha": 0.65},
    "LLM": {"marker": "^", "facecolors": "none", "edgecolors": "#2f2f2f", "linewidths": 1.0, "s": 32, "alpha": 0.65},
}
FN_STYLE = {
    "HW": {"marker": "o", "c": "#2ca02c", "edgecolors": "white", "linewidths": 0.35, "s": 36, "alpha": 0.92},
    "LLM": {"marker": "^", "c": "#1f77b4", "edgecolors": "white", "linewidths": 0.35, "s": 42, "alpha": 0.92},
}

REGIONS = {
    "llm_fn_upper_left": {
        "label": "A: LLM-FN upper-left box",
        "group": "LLM-FN",
        "bounds": {"x0": -6.8, "x1": -2.0, "y0": 0.5, "y1": 6.8},
        "edgecolor": "#184f92",
        "output": "fig1b_llm_fn_upper_left_principles_pca.png",
        "title": "Region A. LLM-FN Upper-left Principle Co-occurrence",
    },
    "hw_fn_right": {
        "label": "B: HW-FN right-side box",
        "group": "HW-FN",
        "bounds": {"x0": -1.3, "x1": 3.0, "y0": -4.0, "y1": 1.4},
        "edgecolor": "#2d7d2f",
        "output": "fig1c_hw_fn_right_principles_pca.png",
        "title": "Region B. HW-FN Right-side Principle Co-occurrence",
    },
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Create one contour map with two boxed FN regions plus two WVAE persuasion heatmaps under Visualization/test."
    )
    parser.add_argument("--projected-input", type=Path, default=DEFAULT_PROJECTED_INPUT)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--projection", default="pca", choices=["pca", "umap"])
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
    if "group" not in frame.columns:
        frame["group"] = np.where(
            frame["source"].eq("HW") & frame["is_tp"],
            "HW-TP",
            np.where(
                frame["source"].eq("HW") & frame["is_fn"],
                "HW-FN",
                np.where(frame["source"].eq("LLM") & frame["is_tp"], "LLM-TP", "LLM-FN"),
            ),
        )
    return frame


def region_mask(frame: pd.DataFrame, bounds: dict[str, float]) -> pd.Series:
    return (
        frame["proj_x"].between(bounds["x0"], bounds["x1"], inclusive="both")
        & frame["proj_y"].between(bounds["y0"], bounds["y1"], inclusive="both")
    )


def compute_principle_cooccurrence_matrix(frame: pd.DataFrame) -> tuple[np.ndarray, int]:
    principle_columns = [WVAE_COLUMN_BY_PRINCIPLE[name] for name in PRINCIPLE_NAMES]
    principle_frame = frame[principle_columns].dropna(how="all")
    if principle_frame.empty:
        return np.full((len(PRINCIPLE_NAMES), len(PRINCIPLE_NAMES)), np.nan, dtype=float), 0

    values = principle_frame.to_numpy(dtype=float)
    matrix = (values.T @ values) / max(1, values.shape[0])
    diagonal = np.nanmean(values, axis=0)
    np.fill_diagonal(matrix, diagonal)
    return matrix, int(values.shape[0])


def draw_region_cooccurrence_heatmap(
    matrix: np.ndarray,
    *,
    output_path: Path,
    title: str,
    count: int,
    no_data_note: str | None = None,
):
    safe_matrix = np.nan_to_num(matrix, nan=0.0, posinf=0.0, neginf=0.0)
    figure, axis = plt.subplots(figsize=(7.2, 6.2))
    image = axis.imshow(safe_matrix, aspect="equal", cmap="YlGnBu", vmin=0.0, vmax=1.0)
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
            color="#5e5e5e",
        )

    axis.set_title(f"matched WVAE rows: {count}", fontsize=10, pad=10)
    colorbar = figure.colorbar(image, ax=axis, fraction=0.04, pad=0.03)
    colorbar.set_label("Mean co-occurrence strength")
    figure.suptitle(title, y=0.98, fontsize=14)
    figure.tight_layout(rect=(0, 0.03, 1, 0.94))
    figure.savefig(output_path, dpi=220, bbox_inches="tight")
    plt.close(figure)


def draw_region_contour(
    frame: pd.DataFrame,
    output_path: Path,
    projection_method: str,
    grid_x: np.ndarray,
    grid_y: np.ndarray,
    score_grid: np.ndarray,
):
    figure, axis = plt.subplots(figsize=(11.8, 9.4))
    filled = add_background(axis, grid_x, grid_y, score_grid, projection_method)

    for source_name in ["HW", "LLM"]:
        tp_frame = frame[(frame["source"] == source_name) & frame["is_tp"]]
        fn_frame = frame[(frame["source"] == source_name) & frame["is_fn"]]
        axis.scatter(tp_frame["proj_x"], tp_frame["proj_y"], **TP_STYLE[source_name])
        axis.scatter(fn_frame["proj_x"], fn_frame["proj_y"], **FN_STYLE[source_name])

    for region_name, region in REGIONS.items():
        bounds = region["bounds"]
        axis.add_patch(
            Rectangle(
                (bounds["x0"], bounds["y0"]),
                bounds["x1"] - bounds["x0"],
                bounds["y1"] - bounds["y0"],
                fill=False,
                linewidth=2.2,
                linestyle="--",
                edgecolor=region["edgecolor"],
            )
        )
        axis.text(
            bounds["x0"] + 0.08,
            bounds["y1"] + 0.15,
            region["label"],
            fontsize=10,
            color=region["edgecolor"],
            ha="left",
            va="bottom",
            bbox={"boxstyle": "round,pad=0.20", "facecolor": "white", "edgecolor": region["edgecolor"], "alpha": 0.92},
        )

    axis.set_xlabel(f"{projection_method.upper()}1", fontsize=10)
    axis.set_ylabel(f"{projection_method.upper()}2", fontsize=10)
    axis.grid(alpha=0.10, linewidth=0.5)
    axis.set_title("TP/FN contour map with two FN focus boxes", fontsize=12)
    legend_handles = [
        Line2D([0], [0], marker="o", color="none", markerfacecolor="none", markeredgecolor="#2f2f2f", markersize=7, linewidth=0, label="HW-TP"),
        Line2D([0], [0], marker="o", color="none", markerfacecolor="#2ca02c", markeredgecolor="white", markersize=7, linewidth=0, label="HW-FN"),
        Line2D([0], [0], marker="^", color="none", markerfacecolor="none", markeredgecolor="#2f2f2f", markersize=7, linewidth=0, label="LLM-TP"),
        Line2D([0], [0], marker="^", color="none", markerfacecolor="#1f77b4", markeredgecolor="white", markersize=7, linewidth=0, label="LLM-FN"),
        Line2D([0], [0], color="#184f92", linestyle="--", linewidth=2.2, label="A: LLM-FN focus"),
        Line2D([0], [0], color="#2d7d2f", linestyle="--", linewidth=2.2, label="B: HW-FN focus"),
        Line2D([0], [0], color="black", linewidth=1.8, label="threshold contour: s(x)=0.50"),
    ]
    axis.legend(
        handles=legend_handles,
        loc="lower center",
        bbox_to_anchor=(0.5, -0.17),
        ncol=3,
        frameon=False,
    )
    colorbar = figure.colorbar(filled, ax=axis, fraction=0.04, pad=0.02)
    colorbar.set_label("Surrogate score")
    figure.suptitle("Figure 1A. Surrogate Response Map with Two FN Focus Regions", y=0.98, fontsize=15)
    figure.tight_layout(rect=(0, 0.04, 1, 0.95))
    figure.savefig(output_path, dpi=220, bbox_inches="tight")
    plt.close(figure)


def draw_region_heatmap(merged_frame: pd.DataFrame, region_key: str, output_dir: Path) -> dict[str, object]:
    region = REGIONS[region_key]
    region_rows = merged_frame[region_mask(merged_frame, region["bounds"])].copy()
    target_rows = region_rows[region_rows["group"] == region["group"]].copy()
    matrix, matched_count = compute_principle_cooccurrence_matrix(target_rows)

    note = None
    if matched_count == 0:
        note = (
            f"No stage-matched WVAE persuasion rows are available yet for this boxed subset. "
            f"Target points in box: {len(target_rows)}."
        )
    output_path = output_dir / region["output"]
    draw_region_cooccurrence_heatmap(
        matrix,
        output_path=output_path,
        title=region["title"],
        count=matched_count,
        no_data_note=note,
    )
    return {
        "region": region_key,
        "group": region["group"],
        "box_bounds": region["bounds"],
        "points_in_box_all_groups": int(len(region_rows)),
        "target_group_points_in_box": int(len(target_rows)),
        "matched_wvae_rows": int(matched_count),
        "output_file": output_path.name,
    }


def main():
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    frame = load_projected_frame(args.projected_input.resolve())
    surface_model = fit_detector_surface(frame, args.seed)
    if surface_model is None:
        raise RuntimeError("Could not fit the surrogate surface for the projected test frame.")
    _, _, grid_x, grid_y, score_grid = build_global_grid(frame, surface_model)

    contour_output = args.output_dir / "fig1a_surrogate_response_tp_fn_regions_pca.png"
    draw_region_contour(frame, contour_output, args.projection, grid_x, grid_y, score_grid)

    merged = merge_projected_with_wvae(frame)
    metadata = {
        "projected_input": str(args.projected_input.resolve()),
        "contour_output": contour_output.name,
        "regions": [],
    }
    for region_key in REGIONS:
        metadata["regions"].append(draw_region_heatmap(merged, region_key, args.output_dir))

    metadata_path = args.output_dir / "fig1_region_focus_metadata.json"
    metadata_path.write_text(json.dumps(metadata, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
