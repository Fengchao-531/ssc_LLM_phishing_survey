#!/usr/bin/env python3
"""Build detector overview assets for the public GitHub interaction view.

The script reads local row-level evaluation/projection data, but only exports
figures and aggregate summaries. It intentionally does not copy raw email text
or detector result rows into the public interaction folder.
"""

from __future__ import annotations

import json
import shutil
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.lines import Line2D
from matplotlib.patches import FancyBboxPatch
from matplotlib.patches import Patch


SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parent
PROJECTED_POINTS = SCRIPT_DIR / "test" / "projected_points.csv"
HEATMAP_SOURCE = SCRIPT_DIR / "test" / "fina_used" / "1" / "group1_difference_heatmaps.png"
HEATMAP_P_VALUES_SOURCE = SCRIPT_DIR / "test" / "fina_used" / "1" / "group1_difference_heatmaps_p_values.csv"
OUTPUT_ROOT = SCRIPT_DIR / "git interaction"
EVALUATION_DATASETS = REPO_ROOT / "Evaluation" / "processed-evaluation-datasets"

if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from generate_stage_visualizations import compute_axis_limits, fit_detector_surface  # noqa: E402
from generate_test_failure_contours import add_background, build_global_grid  # noqa: E402


DETECTORS = [
    {
        "family": "academic",
        "name": "scamllm",
        "column": "scamllm",
        "source": "projected_points",
    },
    {
        "family": "academic",
        "name": "pimref",
        "column": "pimref",
        "source": "projected_points",
    },
    {
        "family": "academic",
        "name": "t5phishing",
        "column": "t5phishing",
        "source": "projected_points",
    },
    {
        "family": "academic",
        "name": "xgboost",
        "column": "xgboost",
        "source": "projected_points",
    },
    {
        "family": "academic",
        "name": "securenet_llama",
        "column": "securenet_llama",
        "source": "projected_points",
    },
    {
        "family": "industry",
        "name": "email_phishing_detection_v3",
        "column": "email_phishing_detection_v3_prediction",
        "source": "processed-evaluation-datasets",
    },
    {
        "family": "industry",
        "name": "phishing_email_agent",
        "column": "phishing_email_agent_prediction",
        "source": "processed-evaluation-datasets",
    },
    {
        "family": "industry",
        "name": "rspamd",
        "column": "rspamd_prediction",
        "source": "processed-evaluation-datasets",
    },
    {
        "family": "industry",
        "name": "spamassassin",
        "column": "spamassassin_prediction",
        "source": "processed-evaluation-datasets",
    },
    {
        "family": "industry",
        "name": "spamscanner",
        "column": "spamscanner_prediction",
        "source": "processed-evaluation-datasets",
    },
]

TEXT_KEY_COLUMNS = ["subject", "body", "label", "stage", "source"]
PRINCIPLE_COLUMNS = {
    "Authority": "principle_authority",
    "Liking": "principle_liking",
    "Reciprocity": "principle_reciprocity",
    "Social Proof": "principle_social_proof",
    "Scarcity": "principle_scarcity",
    "Commitment": "principle_commitment",
}
SELECTED_GROUPS = [
    ("Authority", "Authority", "A-A"),
    ("Authority", "Liking", "A-L"),
    ("Authority", "Reciprocity", "A-R"),
    ("Authority", "Social Proof", "A-SP"),
    ("Liking", "Liking", "L-L"),
    ("Liking", "Reciprocity", "L-R"),
    ("Liking", "Social Proof", "L-SP"),
    ("Reciprocity", "Reciprocity", "R-R"),
    ("Reciprocity", "Social Proof", "R-SP"),
    ("Social Proof", "Social Proof", "SP-SP"),
]
STAGE_GROUPS = {
    "S1": ["S1"],
    "S2": ["S2"],
    "S4": ["S4"],
    "S5": ["S5"],
    "S6": ["S6-MPG", "S6-UTA", "S6-fuzzer"],
    "S8": ["S8-claude", "S8-deepseek", "S8-gemini", "S8-gpt", "S8-llama", "S8-ministral"],
}
COMBINED_STAGE_ROWS = {
    "S6": [
        ("fn_selected_group_boxplots.png", ["S6-fuzzer", "S6-UTA", "S6-MPG"]),
    ],
    "S8": [
        ("fn_selected_group_boxplots_row1.png", ["S8-claude", "S8-deepseek", "S8-gemini"]),
        ("fn_selected_group_boxplots_row2.png", ["S8-gpt", "S8-llama", "S8-ministral"]),
    ],
}
COMBINED_SURROGATE_ROWS = {
    "S6": [
        ("surrogate_response_tp_fn_map.png", ["S6-fuzzer", "S6-UTA", "S6-MPG"]),
    ],
    "S8": [
        ("surrogate_response_tp_fn_map_row1.png", ["S8-claude", "S8-deepseek", "S8-gemini"]),
        ("surrogate_response_tp_fn_map_row2.png", ["S8-gpt", "S8-llama", "S8-ministral"]),
    ],
}

TP_STYLE = {
    "HW": {"marker": "o", "facecolors": "none", "edgecolors": "#2f2f2f", "linewidths": 1.2, "s": 62, "alpha": 0.88},
    "LLM": {"marker": "^", "facecolors": "none", "edgecolors": "#3d3d3d", "linewidths": 1.15, "s": 74, "alpha": 0.88},
}
FN_STYLE = {
    "HW": {"marker": "o", "c": "#2ca02c", "edgecolors": "white", "linewidths": 0.50, "s": 70, "alpha": 0.95},
    "LLM": {"marker": "^", "c": "#3498db", "edgecolors": "white", "linewidths": 0.45, "s": 84, "alpha": 0.92},
}


def normalized_key_frame(frame: pd.DataFrame) -> pd.DataFrame:
    keyed = frame.copy()
    keyed["stage"] = keyed["stage"].replace(
        {
            "S8-claude-sonnet4": "S8-claude",
            "S8-gemini-pro": "S8-gemini",
            "S8-gpt54": "S8-gpt",
        }
    )
    for column in ["subject", "body", "stage", "source"]:
        keyed[column] = keyed[column].fillna("").astype(str).str.strip()
    keyed["label"] = pd.to_numeric(keyed["label"], errors="coerce").astype("Int64").astype(str)
    keyed["_row_key"] = keyed[TEXT_KEY_COLUMNS].agg("\u241f".join, axis=1)
    keyed["_row_key_index"] = keyed.groupby("_row_key", dropna=False).cumcount()
    return keyed


def load_industry_predictions() -> pd.DataFrame:
    frames: list[pd.DataFrame] = []
    for source, group_name in [("HW", "gd"), ("LLM", "llm")]:
        group_dir = EVALUATION_DATASETS / group_name / "industry"
        if not group_dir.exists():
            continue
        for csv_path in sorted(group_dir.glob("*.csv")):
            stage = csv_path.stem
            if stage in {"S6", "S8"}:
                continue
            frame = pd.read_csv(csv_path, low_memory=False)
            available_columns = [
                detector["column"]
                for detector in DETECTORS
                if detector["family"] == "industry" and detector["column"] in frame.columns
            ]
            if not available_columns:
                continue
            subset = frame[["subject", "body", "label", *available_columns]].copy()
            subset["stage"] = stage
            subset["source"] = source
            frames.append(subset)
    if not frames:
        return pd.DataFrame()
    industry = pd.concat(frames, ignore_index=True)
    return normalized_key_frame(industry)


def load_base_frame() -> pd.DataFrame:
    base = pd.read_csv(PROJECTED_POINTS, low_memory=False)
    if not {"proj_x", "proj_y", "source", "stage"}.issubset(base.columns):
        raise ValueError(f"{PROJECTED_POINTS} is missing required projection columns")

    base = normalized_key_frame(base)
    industry = load_industry_predictions()
    if industry.empty:
        return base

    prediction_columns = sorted(
        {
            detector["column"]
            for detector in DETECTORS
            if detector["family"] == "industry" and detector["column"] in industry.columns
        }
    )
    merge_columns = ["_row_key", "_row_key_index", *prediction_columns]
    return base.merge(industry[merge_columns], on=["_row_key", "_row_key_index"], how="left")


def build_legend_handles() -> dict[str, Line2D]:
    return {
        "hw_tp": Line2D([0], [0], marker="o", color="none", markerfacecolor="none", markeredgecolor="#2f2f2f", markersize=11, markeredgewidth=1.3, linewidth=0, label="HW-TP"),
        "hw_fn": Line2D([0], [0], marker="o", color="none", markerfacecolor="#2ca02c", markeredgecolor="white", markersize=11, linewidth=0, label="HW-FN"),
        "llm_tp": Line2D([0], [0], marker="^", color="none", markerfacecolor="none", markeredgecolor="#3d3d3d", markersize=11, markeredgewidth=1.2, linewidth=0, label="LLM-TP"),
        "llm_fn": Line2D([0], [0], marker="^", color="none", markerfacecolor="#3498db", markeredgecolor="white", markersize=11, linewidth=0, label="LLM-FN"),
        "threshold": Line2D([0], [0], color="black", linewidth=2.0, label="threshold contour: s(x)=0.50"),
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
        fontsize=23,
        frameon=False,
        borderpad=0.0,
        labelspacing=0.45,
        handletextpad=0.6,
        handlelength=1.5,
    )
    right_legend.set_zorder(7.5)
    axis.add_artist(right_legend)


def draw_surrogate_map(frame: pd.DataFrame, output_path: Path, detector_name: str) -> None:
    surface_model = fit_detector_surface(frame, seed=7)
    if surface_model is None:
        x_limits, y_limits = compute_axis_limits(frame)
        x_grid = np.linspace(*x_limits, 220)
        y_grid = np.linspace(*y_limits, 220)
        grid_x, grid_y = np.meshgrid(x_grid, y_grid)
        score_grid = np.full_like(grid_x, float(frame["detector_prediction"].mean()))
    else:
        _, _, grid_x, grid_y, score_grid = build_global_grid(frame, surface_model)

    figure, axis = plt.subplots(figsize=(11.9, 9.0))
    filled = add_background(axis, grid_x, grid_y, score_grid, "pca")

    for source_name in ["HW", "LLM"]:
        tp_frame = frame[(frame["source"] == source_name) & frame["is_tp"]]
        fn_frame = frame[(frame["source"] == source_name) & frame["is_fn"]]
        axis.scatter(tp_frame["proj_x"], tp_frame["proj_y"], zorder=3.0, **TP_STYLE[source_name])
        axis.scatter(fn_frame["proj_x"], fn_frame["proj_y"], zorder=5.0, **FN_STYLE[source_name])

    axis.set_xlabel("PCA1", fontsize=29)
    axis.set_ylabel("PCA2", fontsize=28)
    axis.tick_params(axis="both", labelsize=26)
    axis.grid(alpha=0.10, linewidth=0.5)
    add_compact_legend(axis)

    figure.subplots_adjust(left=0.09, right=0.90, bottom=0.10, top=0.98)
    axis_position = axis.get_position()
    colorbar_axis = figure.add_axes([axis_position.x1 + 0.012, axis_position.y0, 0.028, axis_position.height])
    colorbar = figure.colorbar(filled, cax=colorbar_axis)
    colorbar.set_label("Surrogate score", fontsize=19)
    colorbar.ax.tick_params(labelsize=15)
    figure.savefig(output_path, dpi=260, bbox_inches="tight", pad_inches=0.01)
    plt.close(figure)


def build_boxplot_distribution(frame: pd.DataFrame, stage_group: str) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for source_name in ["HW", "LLM"]:
        subset = frame[(frame["source"] == source_name) & frame["is_fn"]].copy()
        if subset.empty:
            continue
        for left_name, right_name, short_label in SELECTED_GROUPS:
            left_values = pd.to_numeric(subset[PRINCIPLE_COLUMNS[left_name]], errors="coerce")
            right_values = pd.to_numeric(subset[PRINCIPLE_COLUMNS[right_name]], errors="coerce")
            valid_values = (left_values * right_values).dropna().to_numpy(dtype=float)
            for value in valid_values:
                rows.append(
                    {
                        "stage_group": stage_group,
                        "stage": str(subset.iloc[0]["stage"]) if subset["stage"].nunique() == 1 else stage_group,
                        "source": source_name,
                        "principle_group": short_label,
                        "left_principle": left_name,
                        "right_principle": right_name,
                        "value": float(value),
                    }
                )
    return pd.DataFrame(rows)


def draw_boxplot(distribution_table: pd.DataFrame, output_path: Path, stage_group: str) -> dict[str, int]:
    figure, axis = plt.subplots(figsize=(23, 12.5))
    figure.patch.set_facecolor("#ffffff")
    axis.set_facecolor("#ffffff")

    group_centers = np.arange(len(SELECTED_GROUPS), dtype=float) * 2.0 + 0.5
    positions = {"HW": group_centers - 0.38, "LLM": group_centers + 0.38}
    fill_colors = {"HW": "#143d73", "LLM": "#b85a69"}
    counts: dict[str, int] = {}

    for source_name in ["HW", "LLM"]:
        source_table = distribution_table[distribution_table["source"] == source_name]
        counts[source_name] = int(len(source_table))
        box_data = []
        for _, _, short_label in SELECTED_GROUPS:
            values = source_table[source_table["principle_group"] == short_label]["value"].to_numpy(dtype=float)
            box_data.append(values)

        boxplot = axis.boxplot(
            box_data,
            positions=positions[source_name],
            widths=0.62,
            patch_artist=True,
            showfliers=False,
            medianprops={"color": "#ffffff", "linewidth": 2.4},
            whiskerprops={"color": fill_colors[source_name], "linewidth": 1.6},
            capprops={"color": fill_colors[source_name], "linewidth": 1.6},
            boxprops={"edgecolor": fill_colors[source_name], "linewidth": 1.8},
        )
        for patch in boxplot["boxes"]:
            patch.set_facecolor(fill_colors[source_name])
            patch.set_alpha(0.92)

    axis.grid(axis="y", color="#dddddd", linewidth=0.85, alpha=0.9)
    axis.spines["top"].set_visible(False)
    axis.spines["right"].set_visible(False)
    axis.spines["left"].set_color("#555555")
    axis.spines["bottom"].set_color("#555555")
    axis.tick_params(axis="x", length=0, pad=12, labelsize=41, colors="#222222")
    axis.tick_params(axis="y", labelsize=41, colors="#222222")
    axis.set_xticks(group_centers)
    axis.set_xticklabels([short_label for _, _, short_label in SELECTED_GROUPS], fontsize=41, color="#222222")
    axis.set_ylabel("Per-email co-occurrence value", fontsize=46, color="#111111")
    axis.set_xlabel(f"{stage_group} selected persuasion principle groups", fontsize=45, color="#111111", labelpad=18)
    axis.legend(
        handles=[
            Patch(facecolor=fill_colors["HW"], edgecolor=fill_colors["HW"], label="HW-P FN"),
            Patch(facecolor=fill_colors["LLM"], edgecolor=fill_colors["LLM"], label="LLM-P FN"),
        ],
        loc="upper right",
        frameon=True,
        fontsize=34,
        facecolor="#ffffff",
        edgecolor="#d0d0d0",
    )

    figure.subplots_adjust(left=0.11, right=0.98, bottom=0.20, top=0.97)
    figure.savefig(output_path, dpi=260, bbox_inches="tight", facecolor=figure.get_facecolor())
    plt.close(figure)
    return {"hw_value_count": counts.get("HW", 0), "llm_value_count": counts.get("LLM", 0)}


def draw_boxplot_axis(axis: plt.Axes, distribution_table: pd.DataFrame, stage_group: str, *, show_ylabel: bool) -> dict[str, int]:
    group_centers = np.arange(len(SELECTED_GROUPS), dtype=float) * 2.0 + 0.5
    positions = {"HW": group_centers - 0.38, "LLM": group_centers + 0.38}
    fill_colors = {"HW": "#143d73", "LLM": "#b85a69"}
    counts: dict[str, int] = {}

    for source_name in ["HW", "LLM"]:
        source_table = distribution_table[distribution_table["source"] == source_name]
        counts[source_name] = int(len(source_table))
        box_data = [
            source_table[source_table["principle_group"] == short_label]["value"].to_numpy(dtype=float)
            for _, _, short_label in SELECTED_GROUPS
        ]
        boxplot = axis.boxplot(
            box_data,
            positions=positions[source_name],
            widths=0.62,
            patch_artist=True,
            showfliers=False,
            medianprops={"color": "#ffffff", "linewidth": 1.6},
            whiskerprops={"color": fill_colors[source_name], "linewidth": 1.1},
            capprops={"color": fill_colors[source_name], "linewidth": 1.1},
            boxprops={"edgecolor": fill_colors[source_name], "linewidth": 1.2},
        )
        for patch in boxplot["boxes"]:
            patch.set_facecolor(fill_colors[source_name])
            patch.set_alpha(0.92)

    axis.grid(axis="y", color="#dddddd", linewidth=0.85, alpha=0.9)
    axis.spines["top"].set_visible(False)
    axis.spines["right"].set_visible(False)
    axis.spines["left"].set_color("#555555")
    axis.spines["bottom"].set_color("#555555")
    axis.tick_params(axis="x", length=0, pad=8, labelsize=15, colors="#222222")
    axis.tick_params(axis="y", labelsize=14, colors="#222222")
    axis.set_xticks(group_centers)
    axis.set_xticklabels([short_label for _, _, short_label in SELECTED_GROUPS], fontsize=15, color="#222222")
    axis.set_title(stage_group, fontsize=18, pad=10)
    axis.set_xlabel("Selected persuasion principle groups", fontsize=16, color="#111111", labelpad=10)
    axis.set_ylabel("Per-email co-occurrence value" if show_ylabel else "", fontsize=16, color="#111111")
    return {"hw_value_count": counts.get("HW", 0), "llm_value_count": counts.get("LLM", 0)}


def draw_boxplot_row(stage_tables: dict[str, pd.DataFrame], output_path: Path) -> None:
    figure, axes = plt.subplots(1, len(stage_tables), figsize=(7.6 * len(stage_tables), 8.8), sharey=True)
    figure.patch.set_facecolor("#ffffff")
    if len(stage_tables) == 1:
        axes = [axes]
    for index, (axis, (stage_name, stage_table)) in enumerate(zip(axes, stage_tables.items(), strict=True)):
        draw_boxplot_axis(axis, stage_table, stage_name, show_ylabel=index == 0)
    fill_colors = {"HW": "#143d73", "LLM": "#b85a69"}
    axes[0].legend(
        handles=[
            Patch(facecolor=fill_colors["HW"], edgecolor=fill_colors["HW"], label="HW-P FN"),
            Patch(facecolor=fill_colors["LLM"], edgecolor=fill_colors["LLM"], label="LLM-P FN"),
        ],
        loc="upper right",
        frameon=True,
        fontsize=13,
        facecolor="#ffffff",
        edgecolor="#d0d0d0",
    )
    figure.subplots_adjust(left=0.04, right=0.99, bottom=0.13, top=0.91, wspace=0.06)
    figure.savefig(output_path, dpi=220, bbox_inches="tight", facecolor=figure.get_facecolor())
    plt.close(figure)


def draw_surrogate_panel(axis: plt.Axes, stage_frame: pd.DataFrame, stage_name: str, seed: int) -> object:
    surface_model = fit_detector_surface(stage_frame, seed=seed)
    if surface_model is None:
        x_limits, y_limits = compute_axis_limits(stage_frame)
        x_grid = np.linspace(*x_limits, 220)
        y_grid = np.linspace(*y_limits, 220)
        grid_x, grid_y = np.meshgrid(x_grid, y_grid)
        score_grid = np.full_like(grid_x, float(stage_frame["detector_prediction"].mean()))
    else:
        _, _, grid_x, grid_y, score_grid = build_global_grid(stage_frame, surface_model)
    filled = add_background(axis, grid_x, grid_y, score_grid, "pca")
    for source_name in ["HW", "LLM"]:
        tp_frame = stage_frame[(stage_frame["source"] == source_name) & stage_frame["is_tp"]]
        fn_frame = stage_frame[(stage_frame["source"] == source_name) & stage_frame["is_fn"]]
        axis.scatter(tp_frame["proj_x"], tp_frame["proj_y"], zorder=3.0, **TP_STYLE[source_name])
        axis.scatter(fn_frame["proj_x"], fn_frame["proj_y"], zorder=5.0, **FN_STYLE[source_name])
    axis.set_title(stage_name, fontsize=18, pad=10)
    axis.tick_params(axis="both", labelsize=16)
    axis.grid(alpha=0.10, linewidth=0.5)
    return filled


def draw_stage_surrogate_row(stage_frames: dict[str, pd.DataFrame], output_path: Path) -> None:
    figure, axes = plt.subplots(1, len(stage_frames), figsize=(6.8 * len(stage_frames) + 2.2, 6.8), sharey=False)
    if len(stage_frames) == 1:
        axes = [axes]
    filled = None
    for index, (axis, (stage_name, stage_frame)) in enumerate(zip(axes, stage_frames.items(), strict=True)):
        filled = draw_surrogate_panel(axis, stage_frame, stage_name, seed=31 + index)
        axis.set_xlabel("PCA1", fontsize=18)
        axis.set_ylabel("PCA2" if index == 0 else "", fontsize=18)
    add_compact_legend(axes[0])
    figure.subplots_adjust(left=0.04, right=0.955, bottom=0.10, top=0.94, wspace=0.12)
    if filled is not None:
        first_position = axes[0].get_position()
        last_position = axes[-1].get_position()
        colorbar_axis = figure.add_axes([last_position.x1 + 0.012, first_position.y0, 0.010, first_position.height])
        colorbar = figure.colorbar(filled, cax=colorbar_axis)
        colorbar.set_label("Surrogate score", fontsize=16)
        colorbar.ax.tick_params(labelsize=13)
    figure.savefig(output_path, dpi=220, bbox_inches="tight", pad_inches=0.02)
    plt.close(figure)


def summarize_detector(frame: pd.DataFrame, detector: dict[str, str]) -> dict[str, object]:
    by_source = {}
    for source_name, source_frame in frame.groupby("source"):
        predictions = source_frame["detector_prediction"]
        by_source[str(source_name)] = {
            "n": int(len(source_frame)),
            "predicted_positive": int((predictions >= 0.5).sum()),
            "predicted_negative": int((predictions < 0.5).sum()),
        }
    return {
        "detector": detector["name"],
        "family": detector["family"],
        "prediction_column": detector["column"],
        "n_projected_rows": int(len(frame)),
        "predicted_positive": int((frame["detector_prediction"] >= 0.5).sum()),
        "predicted_negative": int((frame["detector_prediction"] < 0.5).sum()),
        "by_source": by_source,
    }


def write_overview_readme(overview_dir: Path, detector: dict[str, str], summary: dict[str, object]) -> None:
    readme = f"""# {detector['name']} Overview

This folder contains public overview visualization assets for `{detector['name']}`.

## Files

- `surrogate_response_tp_fn_map.png`: detector-specific surrogate response map generated from the shared PCA projection.
- `group1_difference_heatmaps.png`: shared overview difference heatmap from the final visualization pipeline.
- `group1_difference_heatmaps_p_values.csv`: p-value table for the shared overview heatmap.
- `overview_summary.json`: aggregate counts used to sanity-check this detector overview.

No raw email text or row-level detector outputs are exported in this folder.

## Summary

- Detector family: `{detector['family']}`
- Prediction column: `{detector['column']}`
- Projected rows used: `{summary['n_projected_rows']}`
"""
    (overview_dir / "README.md").write_text(readme, encoding="utf-8")


def build_detector_overview(base_frame: pd.DataFrame, detector: dict[str, str]) -> dict[str, object]:
    detector_column = detector["column"]
    if detector_column not in base_frame.columns:
        raise ValueError(f"Missing detector column {detector_column}")

    detector_frame = base_frame[
        ["source", "stage", "proj_x", "proj_y", detector_column]
    ].copy()
    detector_frame["detector_prediction"] = pd.to_numeric(detector_frame[detector_column], errors="coerce")
    detector_frame = detector_frame.dropna(subset=["detector_prediction", "proj_x", "proj_y"])
    detector_frame["detector_prediction"] = (detector_frame["detector_prediction"] >= 0.5).astype(int)
    detector_frame["is_tp"] = detector_frame["detector_prediction"] >= 1
    detector_frame["is_fn"] = ~detector_frame["is_tp"]

    overview_dir = OUTPUT_ROOT / detector["name"] / "overview"
    overview_dir.mkdir(parents=True, exist_ok=True)

    draw_surrogate_map(
        detector_frame,
        overview_dir / "surrogate_response_tp_fn_map.png",
        detector["name"],
    )
    shutil.copy2(HEATMAP_SOURCE, overview_dir / "group1_difference_heatmaps.png")
    shutil.copy2(HEATMAP_P_VALUES_SOURCE, overview_dir / "group1_difference_heatmaps_p_values.csv")

    summary = summarize_detector(detector_frame, detector)
    (overview_dir / "overview_summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    write_overview_readme(overview_dir, detector, summary)
    return summary


def build_detector_stage(base_frame: pd.DataFrame, detector: dict[str, str], stage_group: str, stages: list[str]) -> dict[str, object] | None:
    detector_column = detector["column"]
    principle_columns = list(PRINCIPLE_COLUMNS.values())
    required_columns = ["source", "stage", "proj_x", "proj_y", detector_column, *principle_columns]
    stage_frame = base_frame[base_frame["stage"].isin(stages)][required_columns].copy()
    stage_frame["detector_prediction"] = pd.to_numeric(stage_frame[detector_column], errors="coerce")
    stage_frame = stage_frame.dropna(subset=["detector_prediction", "proj_x", "proj_y"])
    if stage_frame.empty:
        return None

    stage_frame["detector_prediction"] = (stage_frame["detector_prediction"] >= 0.5).astype(int)
    stage_frame["is_tp"] = stage_frame["detector_prediction"] >= 1
    stage_frame["is_fn"] = ~stage_frame["is_tp"]

    stage_dir = OUTPUT_ROOT / detector["name"] / stage_group
    if stage_dir.exists():
        shutil.rmtree(stage_dir)
    stage_dir.mkdir(parents=True, exist_ok=True)

    stage_frames = {
        stage_name: stage_frame[stage_frame["stage"] == stage_name].copy()
        for stage_name in stages
    }
    stage_frames = {stage_name: frame for stage_name, frame in stage_frames.items() if not frame.empty}

    if stage_group in COMBINED_SURROGATE_ROWS:
        for output_name, row_stages in COMBINED_SURROGATE_ROWS[stage_group]:
            row_frames = {stage_name: stage_frames[stage_name] for stage_name in row_stages if stage_name in stage_frames}
            if row_frames:
                draw_stage_surrogate_row(row_frames, stage_dir / output_name)
    else:
        draw_surrogate_map(
            stage_frame,
            stage_dir / "surrogate_response_tp_fn_map.png",
            f"{detector['name']}:{stage_group}",
        )

    distribution_tables: list[pd.DataFrame] = []
    for stage_name in stages:
        single_stage_frame = stage_frames.get(stage_name)
        if single_stage_frame is None or single_stage_frame.empty:
            continue
        distribution_tables.append(build_boxplot_distribution(single_stage_frame, stage_name))
    distribution_table = pd.concat(distribution_tables, ignore_index=True) if distribution_tables else pd.DataFrame()

    if not distribution_table.empty:
        distribution_table.to_csv(stage_dir / "fn_selected_group_boxplot_values.csv", index=False)
        if stage_group in COMBINED_STAGE_ROWS:
            combined_entries = []
            for output_name, row_stages in COMBINED_STAGE_ROWS[stage_group]:
                row_tables = {}
                for row_stage in row_stages:
                    row_table = distribution_table[distribution_table["stage"] == row_stage].copy()
                    if row_table.empty:
                        continue
                    row_tables[row_stage] = row_table
                if row_tables:
                    draw_boxplot_row(row_tables, stage_dir / output_name)
                    combined_entries.append({"output_png": output_name, "stages": row_stages})
            boxplot_counts = {
                "hw_value_count": int(len(distribution_table[distribution_table["source"] == "HW"])),
                "llm_value_count": int(len(distribution_table[distribution_table["source"] == "LLM"])),
                "combined_boxplot_outputs": combined_entries,
            }
        else:
            boxplot_counts = draw_boxplot(
                distribution_table,
                stage_dir / "fn_selected_group_boxplots.png",
                stage_group,
            )
    else:
        boxplot_counts = {"hw_value_count": 0, "llm_value_count": 0}

    summary = {
        "detector": detector["name"],
        "family": detector["family"],
        "stage_group": stage_group,
        "stages_included": stages,
        "n_projected_rows": int(len(stage_frame)),
        "predicted_positive": int(stage_frame["is_tp"].sum()),
        "predicted_negative": int(stage_frame["is_fn"].sum()),
        "hw_fn_rows": int(((stage_frame["source"] == "HW") & stage_frame["is_fn"]).sum()),
        "llm_fn_rows": int(((stage_frame["source"] == "LLM") & stage_frame["is_fn"]).sum()),
        **boxplot_counts,
    }
    (stage_dir / "stage_summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    (stage_dir / "README.md").write_text(
        f"""# {detector['name']} {stage_group}

This folder contains detector-specific stage visualization assets for `{detector['name']}`.

## Files

- `surrogate_response_tp_fn_map.png`: surrogate response map for this detector and stage group.
- `fn_selected_group_boxplots*.png`: HW-P FN vs LLM-P FN boxplots for selected persuasion principle groups.
- `fn_selected_group_boxplot_values.csv`: aggregate numeric values used to draw the boxplot. No raw email text is included.
- `stage_summary.json`: aggregate counts for this detector-stage group.

Stages included: `{', '.join(stages)}`
""",
        encoding="utf-8",
    )
    return summary


def write_root_readme(manifest: list[dict[str, object]]) -> None:
    rows = "\n".join(
        f"| `{item['detector']}` | `{item['family']}` | `{item['n_projected_rows']}` |"
        for item in manifest
    )
    readme = f"""# Git Interaction Visualization Assets

This directory stores the public, GitHub-friendly assets for the interactive detector visualization.

Each detector folder contains:

- `overview/`: detector-level surrogate map and shared overview heatmap.
- `S1/`, `S2/`, `S4/`, `S5/`: stage-specific surrogate maps and FN persuasion boxplots.
- `S6/`: combined S6-MPG, S6-UTA, and S6-fuzzer assets.
- `S8/`: combined S8 model-output assets.

Raw email text and row-level detector outputs are intentionally excluded. The CSV files here contain only numeric visualization values.

| Detector | Family | Projected rows |
| --- | --- | ---: |
{rows}
"""
    (OUTPUT_ROOT / "README.md").write_text(readme, encoding="utf-8")


def main() -> None:
    OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)
    base_frame = load_base_frame()

    manifest = []
    stage_manifest = []
    failures = []
    for detector in DETECTORS:
        try:
            manifest.append(build_detector_overview(base_frame, detector))
            for stage_group, stages in STAGE_GROUPS.items():
                stage_summary = build_detector_stage(base_frame, detector, stage_group, stages)
                if stage_summary is not None:
                    stage_manifest.append(stage_summary)
        except Exception as exc:  # noqa: BLE001
            failures.append({"detector": detector["name"], "error": str(exc)})

    (OUTPUT_ROOT / "manifest.json").write_text(
        json.dumps({"detectors": manifest, "stages": stage_manifest, "failures": failures}, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    write_root_readme(manifest)

    if failures:
        failure_text = ", ".join(f"{item['detector']}: {item['error']}" for item in failures)
        raise RuntimeError(f"Some detector overviews failed: {failure_text}")


if __name__ == "__main__":
    main()
