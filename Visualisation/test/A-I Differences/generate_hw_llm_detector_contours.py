#!/usr/bin/env python3
from __future__ import annotations

import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.lines import Line2D
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import PolynomialFeatures, StandardScaler

ROOT = Path(__file__).resolve().parent
VIS_ROOT = ROOT.parents[1]
SURVEY_ROOT = VIS_ROOT.parent

PROJECTED_POINTS = VIS_ROOT / "test" / "projected_points.csv"
EVAL_ROOT = SURVEY_ROOT / "Evaluation" / "processed-evaluation-datasets"

MERGED_OUTPUT = ROOT / "merged_detector_hw_llm_projected_points.csv"
OVERVIEW_OUTPUT = ROOT / "fig_hw_llm_phishing_overview_by_detector_pca.png"
DUAL_PANEL_OUTPUT = ROOT / "fig_hw_llm_phishing_dual_panel_by_detector_pca.png"
METADATA_OUTPUT = ROOT / "hw_llm_detector_contours_metadata.json"

SOURCE_STYLE = {
    "HW": {
        "marker": "o",
        "c": "#7a1f3d",
        "s": 15,
        "alpha": 0.72,
        "linewidths": 0.25,
        "edgecolors": "white",
    },
    "LLM": {
        "marker": "^",
        "c": "#2f6b3b",
        "s": 17,
        "alpha": 0.72,
        "linewidths": 0.25,
        "edgecolors": "white",
    },
}

DETECTOR_SPECS = {
    "scamllm": {
        "label": "ScamLLM",
        "prediction_column": "scamllm",
    },
    "phishing_email_agent": {
        "label": "Phishing Email Agent",
        "prediction_column": "phishing_email_agent_prediction",
    },
}


def normalize_binary(series: pd.Series) -> pd.Series:
    return pd.to_numeric(series, errors="coerce").fillna(0.0).astype(float)


def load_projected_points() -> pd.DataFrame:
    frame = pd.read_csv(PROJECTED_POINTS, low_memory=False).copy()
    frame["subject"] = frame["subject"].fillna("").astype(str)
    frame["body"] = frame["body"].fillna("").astype(str)
    frame["raw_label"] = pd.to_numeric(frame["raw_label"], errors="coerce").fillna(0).astype(int)
    frame["merge_stage"] = frame["stage"].astype(str)
    frame["merge_source_file"] = frame["source_file"].astype(str)

    alias_mask = frame["source"].astype(str).eq("HW") & frame["hw_stage_alias_from"].notna()
    frame.loc[alias_mask, "merge_stage"] = frame.loc[alias_mask, "hw_stage_alias_from"].astype(str)
    frame.loc[alias_mask, "merge_source_file"] = frame.loc[alias_mask, "hw_stage_alias_from"].astype(str) + ".csv"
    return frame


def load_industry_lookup() -> pd.DataFrame:
    parts: list[pd.DataFrame] = []
    source_to_dir = {
        "HW": EVAL_ROOT / "gd" / "industry",
        "LLM": EVAL_ROOT / "llm" / "industry",
    }

    for source_name, source_dir in source_to_dir.items():
        for path in sorted(source_dir.glob("*.csv")):
            columns = pd.read_csv(path, nrows=0).columns.tolist()
            requested = ["subject", "body", "label", "phishing_email_agent_prediction"]
            if "source_file" in columns:
                requested.append("source_file")
            frame = pd.read_csv(path, usecols=requested, low_memory=False).copy()
            frame["source"] = source_name
            if "source_file" in frame.columns:
                frame["merge_stage"] = frame["source_file"].astype(str).str.replace(".csv", "", regex=False)
                frame["merge_source_file"] = frame["source_file"].astype(str)
            else:
                frame["merge_stage"] = path.stem
                frame["merge_source_file"] = f"{path.stem}.csv"
            parts.append(frame)

    combined = pd.concat(parts, ignore_index=True)
    combined["subject"] = combined["subject"].fillna("").astype(str)
    combined["body"] = combined["body"].fillna("").astype(str)
    combined["label"] = pd.to_numeric(combined["label"], errors="coerce").fillna(0).astype(int)
    combined["phishing_email_agent_prediction"] = normalize_binary(combined["phishing_email_agent_prediction"])
    combined = combined.drop_duplicates(
        subset=[
            "source",
            "subject",
            "body",
            "label",
            "merge_stage",
            "merge_source_file",
            "phishing_email_agent_prediction",
        ]
    ).reset_index(drop=True)
    return combined


def attach_industry_predictions(projected: pd.DataFrame, lookup: pd.DataFrame) -> pd.DataFrame:
    merged = projected.merge(
        lookup,
        left_on=["source", "subject", "body", "raw_label", "merge_stage", "merge_source_file"],
        right_on=["source", "subject", "body", "label", "merge_stage", "merge_source_file"],
        how="left",
    )

    unmatched = merged["phishing_email_agent_prediction"].isna()
    if unmatched.any():
        stage_lookup = (
            lookup.groupby(["source", "subject", "body", "label", "merge_stage"], as_index=False)[
                "phishing_email_agent_prediction"
            ]
            .max()
        )
        fallback = projected.loc[unmatched].merge(
            stage_lookup,
            left_on=["source", "subject", "body", "raw_label", "merge_stage"],
            right_on=["source", "subject", "body", "label", "merge_stage"],
            how="left",
        )
        merged.loc[unmatched, "phishing_email_agent_prediction"] = fallback["phishing_email_agent_prediction"].to_numpy()

    merged["phishing_email_agent_prediction"] = pd.to_numeric(merged["phishing_email_agent_prediction"], errors="coerce")
    merged["phishing_email_agent_available"] = merged["phishing_email_agent_prediction"].notna()
    return merged


def prepare_detector_frame(frame: pd.DataFrame, detector_key: str) -> pd.DataFrame:
    spec = DETECTOR_SPECS[detector_key]
    detector_frame = frame.copy()
    detector_frame["detector_name"] = detector_key
    detector_frame["detector_label"] = spec["label"]

    if detector_key == "scamllm":
        detector_frame["detector_prediction"] = normalize_binary(detector_frame[spec["prediction_column"]])
        detector_frame["prediction_available"] = True
    else:
        detector_frame["detector_prediction"] = pd.to_numeric(
            detector_frame[spec["prediction_column"]], errors="coerce"
        )
        detector_frame["prediction_available"] = detector_frame["detector_prediction"].notna()

    detector_frame = detector_frame[detector_frame["prediction_available"]].copy()
    detector_frame["predicted_label"] = (detector_frame["detector_prediction"].fillna(0.0) >= 0.5).astype(int)
    return detector_frame


def fit_surface(frame: pd.DataFrame, seed: int) -> Pipeline | None:
    labels = frame["predicted_label"].astype(int).to_numpy()
    if len(np.unique(labels)) < 2:
        return None

    model = Pipeline(
        [
            ("poly", PolynomialFeatures(degree=2, include_bias=False)),
            ("scale", StandardScaler()),
            (
                "logreg",
                LogisticRegression(
                    max_iter=1500,
                    class_weight="balanced",
                    random_state=seed,
                ),
            ),
        ]
    )
    features = frame[["proj_x", "proj_y"]].to_numpy(dtype=float)
    model.fit(features, labels)
    return model


def build_grid(extent_frame: pd.DataFrame, model: Pipeline | None) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    x_min = float(extent_frame["proj_x"].min()) - 0.8
    x_max = float(extent_frame["proj_x"].max()) + 0.8
    y_min = float(extent_frame["proj_y"].min()) - 0.8
    y_max = float(extent_frame["proj_y"].max()) + 0.8
    grid_x, grid_y = np.meshgrid(np.linspace(x_min, x_max, 220), np.linspace(y_min, y_max, 220))

    if model is None:
        score_grid = np.full_like(grid_x, 0.5, dtype=float)
    else:
        mesh = np.column_stack([grid_x.ravel(), grid_y.ravel()])
        score_grid = model.predict_proba(mesh)[:, 1].reshape(grid_x.shape)
    return grid_x, grid_y, score_grid


def draw_background(axis: plt.Axes, grid_x: np.ndarray, grid_y: np.ndarray, score_grid: np.ndarray):
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
        linewidths=1.7,
    )
    return filled


def plot_source_points(axis: plt.Axes, frame: pd.DataFrame, source_name: str | None = None) -> None:
    sources = [source_name] if source_name else ["HW", "LLM"]
    for name in sources:
        source_frame = frame[frame["source"] == name]
        axis.scatter(source_frame["proj_x"], source_frame["proj_y"], **SOURCE_STYLE[name])


def build_source_legend() -> list[Line2D]:
    return [
        Line2D(
            [0],
            [0],
            marker="o",
            color="none",
            markerfacecolor="#7a1f3d",
            markeredgecolor="white",
            markersize=7,
            linewidth=0,
            label="HW phishing",
        ),
        Line2D(
            [0],
            [0],
            marker="^",
            color="none",
            markerfacecolor="#2f6b3b",
            markeredgecolor="white",
            markersize=7,
            linewidth=0,
            label="LLM phishing",
        ),
        Line2D([0], [0], color="black", linewidth=1.8, label="threshold contour: s(x)=0.50"),
    ]


def draw_overview(detector_frames: dict[str, pd.DataFrame], extent_frame: pd.DataFrame) -> None:
    figure, axes = plt.subplots(1, 2, figsize=(16.4, 6.9), sharex=True, sharey=True)
    filled = None

    for index, detector_key in enumerate(["scamllm", "phishing_email_agent"]):
        axis = axes[index]
        frame = detector_frames[detector_key]
        model = fit_surface(frame, 7 + index)
        grid_x, grid_y, score_grid = build_grid(extent_frame, model)
        filled = draw_background(axis, grid_x, grid_y, score_grid)
        plot_source_points(axis, frame)

        hw_count = int((frame["source"] == "HW").sum())
        llm_count = int((frame["source"] == "LLM").sum())
        axis.set_title(
            f"{DETECTOR_SPECS[detector_key]['label']}\nHW phishing={hw_count} | LLM phishing={llm_count}",
            fontsize=12,
        )
        axis.set_xlabel("PCA1", fontsize=10)
        axis.grid(alpha=0.10, linewidth=0.5)

    axes[0].set_ylabel("PCA2", fontsize=10)
    axes[0].legend(
        handles=build_source_legend(),
        loc="lower center",
        bbox_to_anchor=(1.05, -0.22),
        ncol=3,
        frameon=False,
    )
    colorbar_axis = figure.add_axes([0.487, 0.20, 0.014, 0.58])
    colorbar = figure.colorbar(filled, cax=colorbar_axis)
    colorbar.set_label("Surrogate score")
    figure.suptitle("HW vs LLM Phishing Contours by Detector", y=0.98, fontsize=15)
    figure.subplots_adjust(left=0.06, right=0.94, bottom=0.18, top=0.83, wspace=0.08)
    figure.savefig(OVERVIEW_OUTPUT, dpi=220, bbox_inches="tight")
    plt.close(figure)


def draw_dual_panel(detector_frames: dict[str, pd.DataFrame], extent_frame: pd.DataFrame) -> None:
    figure, axes = plt.subplots(2, 2, figsize=(15.6, 12.4), sharex=True, sharey=True)
    filled = None

    for row_index, detector_key in enumerate(["scamllm", "phishing_email_agent"]):
        frame = detector_frames[detector_key]
        model = fit_surface(frame, 21 + row_index)
        grid_x, grid_y, score_grid = build_grid(extent_frame, model)

        for col_index, source_name in enumerate(["HW", "LLM"]):
            axis = axes[row_index, col_index]
            filled = draw_background(axis, grid_x, grid_y, score_grid)
            plot_source_points(axis, frame, source_name)
            source_count = int((frame["source"] == source_name).sum())
            axis.set_title(
                f"{DETECTOR_SPECS[detector_key]['label']} | {source_name}-phishing\nn={source_count}",
                fontsize=12,
            )
            axis.grid(alpha=0.10, linewidth=0.5)

    axes[1, 0].set_xlabel("PCA1", fontsize=10)
    axes[1, 1].set_xlabel("PCA1", fontsize=10)
    axes[0, 0].set_ylabel("PCA2", fontsize=10)
    axes[1, 0].set_ylabel("PCA2", fontsize=10)
    axes[1, 0].legend(
        handles=build_source_legend(),
        loc="lower center",
        bbox_to_anchor=(1.05, -0.28),
        ncol=3,
        frameon=False,
    )
    colorbar_axis = figure.add_axes([0.92, 0.20, 0.014, 0.58])
    colorbar = figure.colorbar(filled, cax=colorbar_axis)
    colorbar.set_label("Surrogate score")
    figure.suptitle("Source-Split HW / LLM Phishing Contours for ScamLLM and Phishing Email Agent", y=0.98, fontsize=15)
    figure.subplots_adjust(left=0.08, right=0.90, bottom=0.12, top=0.88, wspace=0.10, hspace=0.18)
    figure.savefig(DUAL_PANEL_OUTPUT, dpi=220, bbox_inches="tight")
    plt.close(figure)


def write_metadata(detector_frames: dict[str, pd.DataFrame], merged: pd.DataFrame) -> None:
    metadata = {
        "projected_points_input": str(PROJECTED_POINTS),
        "merged_output": str(MERGED_OUTPUT),
        "figure_files": {
            "overview_by_detector": OVERVIEW_OUTPUT.name,
            "dual_panel_by_detector": DUAL_PANEL_OUTPUT.name,
        },
        "row_count_projected_points": int(len(merged)),
        "detectors": {},
    }

    for detector_key, frame in detector_frames.items():
        metadata["detectors"][detector_key] = {
            "label": DETECTOR_SPECS[detector_key]["label"],
            "available_rows": int(len(frame)),
            "source_counts": frame["source"].value_counts().to_dict(),
            "predicted_positive_counts": frame[frame["predicted_label"].eq(1)]["source"].value_counts().to_dict(),
            "stage_counts": frame["stage"].value_counts().sort_index().to_dict(),
        }

    METADATA_OUTPUT.write_text(json.dumps(metadata, indent=2), encoding="utf-8")


def main() -> None:
    ROOT.mkdir(parents=True, exist_ok=True)

    projected = load_projected_points()
    merged = attach_industry_predictions(projected, load_industry_lookup())

    detector_frames = {
        detector_key: prepare_detector_frame(merged, detector_key)
        for detector_key in ["scamllm", "phishing_email_agent"]
    }

    merged.to_csv(MERGED_OUTPUT, index=False)
    draw_overview(detector_frames, projected)
    draw_dual_panel(detector_frames, projected)
    write_metadata(detector_frames, merged)


if __name__ == "__main__":
    main()
