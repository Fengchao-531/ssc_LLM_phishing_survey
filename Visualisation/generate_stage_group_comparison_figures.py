#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.colors import Normalize
from matplotlib.lines import Line2D

from generate_stage_split_figures import (
    EXACT_STAGE_FILES,
    HW_STAGE_CLONE_MAP,
    SPLIT_STAGE_FILES,
    STAGE_ORDER,
    WVAE_ROOT,
)
from generate_stage_visualizations import (
    compute_axis_limits,
    fit_detector_surface,
    project_indicator_space,
)
from generate_test_failure_contours import add_background

SCRIPT_DIR = Path(__file__).resolve().parent
DEFAULT_OUTPUT_DIR = SCRIPT_DIR / "test" / "stages"
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
            "Generate stage-specific group comparison contour maps and heatmaps "
            "for mixed phishing/benign data."
        )
    )
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--projection", choices=["pca", "umap"], default="pca")
    parser.add_argument("--seed", type=int, default=17)
    return parser.parse_args()


def load_mixed_raw_frame() -> pd.DataFrame:
    frames: list[pd.DataFrame] = []
    exact_stage_names = {(source, stage) for source, stage, _ in EXACT_STAGE_FILES}

    for source_name, stage_name, filename in EXACT_STAGE_FILES:
        csv_path = WVAE_ROOT / filename
        if not csv_path.exists():
            continue
        frame = pd.read_csv(csv_path)
        required_columns = {"subject", "body", "label", "scamllm", *PRINCIPLE_COLUMNS}
        missing = sorted(required_columns.difference(frame.columns))
        if missing:
            raise ValueError(f"{csv_path} is missing required columns: {', '.join(missing)}")
        frame = frame.copy()
        frame["source"] = source_name
        frame["stage"] = stage_name
        frames.append(frame)

    for source_name, filename in SPLIT_STAGE_FILES:
        csv_path = WVAE_ROOT / filename
        if not csv_path.exists():
            continue
        frame = pd.read_csv(csv_path)
        required_columns = {"subject", "body", "label", "scamllm", "source_file", *PRINCIPLE_COLUMNS}
        missing = sorted(required_columns.difference(frame.columns))
        if missing:
            raise ValueError(f"{csv_path} is missing required columns: {', '.join(missing)}")
        frame = frame.copy()
        frame["stage"] = frame["source_file"].astype(str).str.replace(".csv", "", regex=False)
        frame["source"] = source_name
        frame = frame[frame["stage"].isin(STAGE_ORDER)].copy()
        frame = frame[
            ~frame["stage"].map(lambda stage_name: (source_name, stage_name) in exact_stage_names)
        ].copy()
        if not frame.empty:
            frames.append(frame)

    if not frames:
        raise FileNotFoundError(f"No mixed rows found under {WVAE_ROOT}")

    merged = pd.concat(frames, ignore_index=True)
    merged["label"] = pd.to_numeric(merged["label"], errors="coerce").fillna(0).astype(int)
    merged["detector_prediction"] = pd.to_numeric(merged["scamllm"], errors="coerce").fillna(0.0)
    merged["raw_label"] = merged["label"]
    merged["text"] = (
        merged["subject"].fillna("").astype(str).str.strip()
        + "\n\n"
        + merged["body"].fillna("").astype(str).str.strip()
    ).str.strip()

    clone_frames: list[pd.DataFrame] = []
    for target_stage, source_stage in HW_STAGE_CLONE_MAP.items():
        source_rows = merged[(merged["source"] == "HW") & (merged["stage"] == source_stage)].copy()
        if source_rows.empty:
            continue
        source_rows["stage"] = target_stage
        source_rows["hw_stage_alias_from"] = source_stage
        if "source_file" in source_rows.columns:
            source_rows["source_file"] = f"{target_stage}.csv"
        clone_frames.append(source_rows)
    if clone_frames:
        merged = pd.concat([merged, *clone_frames], ignore_index=True)
    return merged.reset_index(drop=True)


def compute_principle_cooccurrence_matrix(frame: pd.DataFrame) -> tuple[np.ndarray, int]:
    principle_frame = frame[PRINCIPLE_COLUMNS].dropna(how="all")
    if principle_frame.empty:
        return np.full((len(PRINCIPLE_COLUMNS), len(PRINCIPLE_COLUMNS)), np.nan, dtype=float), 0
    values = principle_frame.to_numpy(dtype=float)
    matrix = (values.T @ values) / max(1, values.shape[0])
    diagonal = np.nanmean(values, axis=0)
    np.fill_diagonal(matrix, diagonal)
    return matrix, int(values.shape[0])


def build_stage_frame(raw_frame: pd.DataFrame, stage_name: str, projection: str, seed: int) -> pd.DataFrame:
    stage_raw = raw_frame[raw_frame["stage"] == stage_name].copy()
    artifacts = project_indicator_space(stage_raw, projection=projection, seed=seed)
    frame = artifacts.frame.copy()
    frame["true_label"] = frame["raw_label"].astype(int)
    frame["pred_label"] = (
        pd.to_numeric(frame["detector_prediction"], errors="coerce").fillna(0.0) >= 0.5
    ).astype(int)
    frame["is_tp_phishing"] = frame["true_label"].eq(1) & frame["pred_label"].eq(1)
    frame["is_fn_phishing"] = frame["true_label"].eq(1) & frame["pred_label"].eq(0)
    frame["is_tn_benign"] = frame["true_label"].eq(0) & frame["pred_label"].eq(0)
    return frame


def _score_grid(frame: pd.DataFrame, fit_frame: pd.DataFrame, seed: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    x_limits, y_limits = compute_axis_limits(frame)
    x_grid = np.linspace(*x_limits, 220)
    y_grid = np.linspace(*y_limits, 220)
    grid_x, grid_y = np.meshgrid(x_grid, y_grid)
    mesh = np.column_stack([grid_x.ravel(), grid_y.ravel()])

    surface_model = fit_detector_surface(fit_frame, seed)
    if surface_model is None:
        default_score = float(fit_frame["pred_label"].mean()) if len(fit_frame) else 0.0
        score_grid = np.full_like(grid_x, default_score, dtype=float)
    else:
        score_grid = surface_model.predict_proba(mesh)[:, 1].reshape(grid_x.shape)
    return grid_x, grid_y, score_grid


def draw_group1_contour(
    frame: pd.DataFrame,
    stage_dir: Path,
    stage_name: str,
    seed: int,
    projection_method: str,
) -> dict[str, int]:
    grid_x, grid_y, score_grid = _score_grid(frame, frame, seed)
    llm_p_fn = frame[(frame["source"] == "LLM") & frame["is_fn_phishing"]].copy()
    llm_b_tn = frame[(frame["source"] == "LLM") & frame["is_tn_benign"]].copy()
    hw_b_tn = frame[(frame["source"] == "HW") & frame["is_tn_benign"]].copy()
    other = frame.drop(llm_p_fn.index.union(llm_b_tn.index).union(hw_b_tn.index), errors="ignore")

    figure = plt.figure(figsize=(11.0, 8.4))
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

    filled = add_background(axis, grid_x, grid_y, score_grid, projection_method)
    axis.scatter(other["proj_x"], other["proj_y"], s=9, c="#d9d9d9", alpha=0.22, linewidths=0)
    axis.scatter(
        hw_b_tn["proj_x"],
        hw_b_tn["proj_y"],
        s=20,
        c="#111111",
        alpha=0.72,
        linewidths=0,
        label="HW-B TN",
    )
    axis.scatter(
        llm_b_tn["proj_x"],
        llm_b_tn["proj_y"],
        s=26,
        c="#2f6b3b",
        marker="^",
        alpha=0.78,
        linewidths=0,
        label="LLM-B TN",
    )
    axis.scatter(
        llm_p_fn["proj_x"],
        llm_p_fn["proj_y"],
        s=28,
        c="#1f77b4",
        marker="^",
        alpha=0.86,
        linewidths=0,
        label="LLM-P FN",
    )
    axis.set_title(f"{stage_name}: LLM-P FN vs LLM-B TN vs HW-B TN", fontsize=13, pad=10)
    axis.set_xlabel(f"{projection_method.upper()}1", fontsize=10)
    axis.set_ylabel(f"{projection_method.upper()}2", fontsize=10)
    axis.grid(alpha=0.10, linewidth=0.5)

    legend_axis.legend(
        handles=[
            Line2D([0], [0], marker="^", color="none", markerfacecolor="#1f77b4", markeredgecolor="#1f77b4", markersize=7, linewidth=0, label="LLM-P FN"),
            Line2D([0], [0], marker="^", color="none", markerfacecolor="#2f6b3b", markeredgecolor="#2f6b3b", markersize=7, linewidth=0, label="LLM-B TN"),
            Line2D([0], [0], marker="o", color="none", markerfacecolor="#111111", markeredgecolor="#111111", markersize=6, linewidth=0, label="HW-B TN"),
            Line2D([0], [0], color="#d9d9d9", linewidth=4, label="other points"),
            Line2D([0], [0], color="black", linewidth=1.8, label="threshold contour: s(x)=0.50"),
        ],
        loc="center",
        ncol=3,
        frameon=False,
    )
    figure.colorbar(filled, cax=colorbar_axis).set_label("Surrogate score")
    figure.suptitle(f"Stage Group 1 Contour Map: {stage_name}", y=0.98, fontsize=15)
    figure.subplots_adjust(left=0.07, right=0.93, bottom=0.08, top=0.92)
    figure.savefig(stage_dir / "group1_llm_p_fn_llm_b_tn_hw_b_tn_contours_pca.png", dpi=220)
    plt.close(figure)
    return {
        "llm_p_fn": int(len(llm_p_fn)),
        "llm_b_tn": int(len(llm_b_tn)),
        "hw_b_tn": int(len(hw_b_tn)),
    }


def draw_group1_heatmaps(frame: pd.DataFrame, stage_dir: Path, stage_name: str) -> dict[str, int]:
    groups = [
        ("LLM-P FN", frame[(frame["source"] == "LLM") & frame["is_fn_phishing"]].copy()),
        ("LLM-B TN", frame[(frame["source"] == "LLM") & frame["is_tn_benign"]].copy()),
        ("HW-B TN", frame[(frame["source"] == "HW") & frame["is_tn_benign"]].copy()),
    ]
    figure = plt.figure(figsize=(16.6, 5.8))
    grid = figure.add_gridspec(1, 4, width_ratios=[1.0, 1.0, 1.0, 0.08], wspace=0.05)
    axes = [figure.add_subplot(grid[0, i]) for i in range(3)]
    colorbar_axis = figure.add_subplot(grid[0, 3])

    labels = [name.replace("principle_", "").replace("_", " ").title() for name in PRINCIPLE_COLUMNS]
    images = []
    counts: dict[str, int] = {}
    for axis, (title, subset) in zip(axes, groups, strict=True):
        matrix, count = compute_principle_cooccurrence_matrix(subset)
        counts[title] = count
        safe_matrix = np.nan_to_num(matrix, nan=0.0, posinf=0.0, neginf=0.0)
        image = axis.imshow(safe_matrix, aspect="equal", cmap="GnBu", norm=Normalize(vmin=0.0, vmax=1.0))
        images.append(image)
        axis.set_xticks(range(len(PRINCIPLE_COLUMNS)))
        axis.set_xticklabels(labels, rotation=45, ha="right", fontsize=8.2)
        axis.set_yticks(range(len(PRINCIPLE_COLUMNS)))
        axis.set_yticklabels(labels if title == "LLM-P FN" else [], fontsize=8.2)
        axis.set_title(title, fontsize=12)
        for row_index in range(safe_matrix.shape[0]):
            for col_index in range(safe_matrix.shape[1]):
                label = "NA" if np.isnan(matrix[row_index, col_index]) else f"{matrix[row_index, col_index]:.2f}"
                axis.text(col_index, row_index, label, ha="center", va="center", fontsize=7.2, color="black")

    figure.colorbar(images[-1], cax=colorbar_axis).set_label("Mean co-occurrence strength")
    figure.suptitle(f"Stage Group 1 Heatmaps: {stage_name}", y=0.98, fontsize=15)
    figure.subplots_adjust(left=0.05, right=0.95, bottom=0.10, top=0.88)
    figure.savefig(stage_dir / "group1_llm_p_fn_llm_b_tn_hw_b_tn_heatmaps_pca.png", dpi=220)
    plt.close(figure)
    return {
        "llm_p_fn_heatmap_rows": int(counts["LLM-P FN"]),
        "llm_b_tn_heatmap_rows": int(counts["LLM-B TN"]),
        "hw_b_tn_heatmap_rows": int(counts["HW-B TN"]),
    }


def draw_group2_contour(
    frame: pd.DataFrame,
    stage_dir: Path,
    stage_name: str,
    seed: int,
    projection_method: str,
) -> dict[str, int]:
    phishing_frame = frame[frame["true_label"] == 1].copy()
    grid_x, grid_y, score_grid = _score_grid(frame, phishing_frame, seed)
    llm_p_fn = frame[(frame["source"] == "LLM") & frame["is_fn_phishing"]].copy()
    llm_p_tp = frame[(frame["source"] == "LLM") & frame["is_tp_phishing"]].copy()
    hw_p = frame[(frame["source"] == "HW") & frame["true_label"].eq(1)].copy()

    figure = plt.figure(figsize=(11.0, 8.4))
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

    filled = add_background(axis, grid_x, grid_y, score_grid, projection_method)
    axis.scatter(hw_p["proj_x"], hw_p["proj_y"], s=11, c="#111111", alpha=0.45, linewidths=0, label="HW-P")
    axis.scatter(
        llm_p_tp["proj_x"],
        llm_p_tp["proj_y"],
        s=24,
        c="#2f6b3b",
        marker="^",
        alpha=0.78,
        linewidths=0,
        label="LLM-P TP",
    )
    axis.scatter(
        llm_p_fn["proj_x"],
        llm_p_fn["proj_y"],
        s=24,
        c="#1f77b4",
        marker="^",
        alpha=0.84,
        linewidths=0,
        label="LLM-P FN",
    )
    axis.set_title(f"{stage_name}: LLM-P FN vs LLM-P TP with HW-P background", fontsize=13, pad=10)
    axis.set_xlabel(f"{projection_method.upper()}1", fontsize=10)
    axis.set_ylabel(f"{projection_method.upper()}2", fontsize=10)
    axis.grid(alpha=0.10, linewidth=0.5)

    legend_axis.legend(
        handles=[
            Line2D([0], [0], marker="^", color="none", markerfacecolor="#1f77b4", markeredgecolor="#1f77b4", markersize=7, linewidth=0, label="LLM-P FN"),
            Line2D([0], [0], marker="^", color="none", markerfacecolor="#2f6b3b", markeredgecolor="#2f6b3b", markersize=7, linewidth=0, label="LLM-P TP"),
            Line2D([0], [0], marker="o", color="none", markerfacecolor="#111111", markeredgecolor="#111111", markersize=6, linewidth=0, label="HW-P background"),
            Line2D([0], [0], color="black", linewidth=1.8, label="threshold contour: s(x)=0.50"),
        ],
        loc="center",
        ncol=2,
        frameon=False,
    )
    figure.colorbar(filled, cax=colorbar_axis).set_label("Surrogate score")
    figure.suptitle(f"Stage Group 2 Contour Map: {stage_name}", y=0.98, fontsize=15)
    figure.subplots_adjust(left=0.07, right=0.93, bottom=0.08, top=0.92)
    figure.savefig(stage_dir / "group2_llm_p_fn_llm_p_tp_contours_pca.png", dpi=220)
    plt.close(figure)
    return {
        "llm_p_fn": int(len(llm_p_fn)),
        "llm_p_tp": int(len(llm_p_tp)),
        "hw_p_background": int(len(hw_p)),
    }


def draw_group2_heatmaps(frame: pd.DataFrame, stage_dir: Path, stage_name: str) -> dict[str, int]:
    groups = [
        ("LLM-P FN", frame[(frame["source"] == "LLM") & frame["is_fn_phishing"]].copy()),
        ("LLM-P TP", frame[(frame["source"] == "LLM") & frame["is_tp_phishing"]].copy()),
    ]
    figure = plt.figure(figsize=(11.6, 5.8))
    grid = figure.add_gridspec(1, 3, width_ratios=[1.0, 1.0, 0.08], wspace=0.05)
    axes = [figure.add_subplot(grid[0, 0]), figure.add_subplot(grid[0, 1])]
    colorbar_axis = figure.add_subplot(grid[0, 2])

    labels = [name.replace("principle_", "").replace("_", " ").title() for name in PRINCIPLE_COLUMNS]
    images = []
    counts: dict[str, int] = {}
    for axis, (title, subset) in zip(axes, groups, strict=True):
        matrix, count = compute_principle_cooccurrence_matrix(subset)
        counts[title] = count
        safe_matrix = np.nan_to_num(matrix, nan=0.0, posinf=0.0, neginf=0.0)
        image = axis.imshow(safe_matrix, aspect="equal", cmap="GnBu", norm=Normalize(vmin=0.0, vmax=1.0))
        images.append(image)
        axis.set_xticks(range(len(PRINCIPLE_COLUMNS)))
        axis.set_xticklabels(labels, rotation=45, ha="right", fontsize=8.2)
        axis.set_yticks(range(len(PRINCIPLE_COLUMNS)))
        axis.set_yticklabels(labels if title == "LLM-P FN" else [], fontsize=8.2)
        axis.set_title(title, fontsize=12)
        for row_index in range(safe_matrix.shape[0]):
            for col_index in range(safe_matrix.shape[1]):
                label = "NA" if np.isnan(matrix[row_index, col_index]) else f"{matrix[row_index, col_index]:.2f}"
                axis.text(col_index, row_index, label, ha="center", va="center", fontsize=7.2, color="black")

    figure.colorbar(images[-1], cax=colorbar_axis).set_label("Mean co-occurrence strength")
    figure.suptitle(f"Stage Group 2 Heatmaps: {stage_name}", y=0.98, fontsize=15)
    figure.subplots_adjust(left=0.06, right=0.95, bottom=0.10, top=0.88)
    figure.savefig(stage_dir / "group2_llm_p_fn_llm_p_tp_heatmaps_pca.png", dpi=220)
    plt.close(figure)
    return {
        "llm_p_fn_heatmap_rows": int(counts["LLM-P FN"]),
        "llm_p_tp_heatmap_rows": int(counts["LLM-P TP"]),
    }


def draw_stage_group_figures(
    raw_frame: pd.DataFrame,
    stage_name: str,
    output_dir: Path,
    seed: int,
    projection_method: str,
) -> dict[str, object]:
    stage_dir = output_dir / stage_name
    stage_dir.mkdir(parents=True, exist_ok=True)
    frame = build_stage_frame(raw_frame, stage_name, projection=projection_method, seed=seed)
    frame.to_csv(stage_dir / "projected_points_mixed.csv", index=False)

    group1_contour_counts = draw_group1_contour(frame, stage_dir, stage_name, seed, projection_method)
    group1_heatmap_counts = draw_group1_heatmaps(frame, stage_dir, stage_name)
    group2_contour_counts = draw_group2_contour(frame, stage_dir, stage_name, seed + 100, projection_method)
    group2_heatmap_counts = draw_group2_heatmaps(frame, stage_dir, stage_name)

    metadata = {
        "stage": stage_name,
        "projection_used": projection_method,
        "mixed_row_count": int(len(frame)),
        "hw_stage_clone_map": HW_STAGE_CLONE_MAP if stage_name in HW_STAGE_CLONE_MAP else {},
        "files": {
            "group1_contour": "group1_llm_p_fn_llm_b_tn_hw_b_tn_contours_pca.png",
            "group1_heatmap": "group1_llm_p_fn_llm_b_tn_hw_b_tn_heatmaps_pca.png",
            "group2_contour": "group2_llm_p_fn_llm_p_tp_contours_pca.png",
            "group2_heatmap": "group2_llm_p_fn_llm_p_tp_heatmaps_pca.png",
            "projected_points_mixed": "projected_points_mixed.csv",
        },
        "counts": {
            "group1": {**group1_contour_counts, **group1_heatmap_counts},
            "group2": {**group2_contour_counts, **group2_heatmap_counts},
        },
    }
    (stage_dir / "mixed_group_metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    return metadata


def main() -> None:
    args = parse_args()
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    raw_frame = load_mixed_raw_frame()
    run_metadata: list[dict[str, object]] = []
    for index, stage_name in enumerate(STAGE_ORDER):
        run_metadata.append(
            draw_stage_group_figures(
                raw_frame,
                stage_name,
                output_dir,
                args.seed + index,
                args.projection,
            )
        )

    summary = {
        "projection_used": args.projection,
        "stage_order": STAGE_ORDER,
        "output_dir": str(output_dir),
        "hw_stage_clone_map": HW_STAGE_CLONE_MAP,
        "stages": run_metadata,
    }
    (output_dir / "mixed_group_run_metadata.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
