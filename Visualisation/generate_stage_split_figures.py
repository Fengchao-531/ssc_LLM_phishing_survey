#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from matplotlib import pyplot as plt

from generate_stage_visualizations import project_indicator_space
from generate_test_failure_contours import build_main_legend
from generate_test_failure_contours import add_background
from generate_test_failure_mode_figures import FN_STYLE, TP_STYLE
from generate_stage_visualizations import fit_detector_surface, compute_axis_limits

SCRIPT_DIR = Path(__file__).resolve().parent
WVAE_ROOT = SCRIPT_DIR / "persuasion_strategy_wvae" / "output" / "full_inference_results"
DEFAULT_OUTPUT_DIR = SCRIPT_DIR / "test" / "stages"
DEFAULT_PROJECTED_POINTS_PATH = SCRIPT_DIR / "test" / "projected_points.csv"
STAGE_ORDER = [
    "S1",
    "S2",
    "S4",
    "S5",
    "S6-MPG",
    "S6-UTA",
    "S6-fuzzer",
    "S8-claude",
    "S8-deepseek",
    "S8-gemini",
    "S8-gpt",
    "S8-llama",
    "S8-ministral",
]
EXACT_STAGE_FILES = [
    ("HW", "S1", "HW_S1_persuasion.csv"),
    ("HW", "S2", "HW_S2_persuasion.csv"),
    ("HW", "S4", "HW_S4_persuasion.csv"),
    ("HW", "S5", "HW_S5_persuasion.csv"),
    ("LLM", "S1", "LLM_S1_persuasion.csv"),
    ("LLM", "S2", "LLM_S2_persuasion.csv"),
    ("LLM", "S4", "LLM_S4_persuasion.csv"),
    ("LLM", "S5", "LLM_S5_persuasion.csv"),
    ("LLM", "S6-MPG", "LLM_S6-MPG_persuasion.csv"),
    ("LLM", "S6-UTA", "LLM_S6-UTA_persuasion.csv"),
    ("LLM", "S6-fuzzer", "LLM_S6-fuzzer_persuasion.csv"),
    ("LLM", "S8-claude", "LLM_S8-claude_persuasion.csv"),
    ("LLM", "S8-deepseek", "LLM_S8-deepseek_persuasion.csv"),
    ("LLM", "S8-gemini", "LLM_S8-gemini_persuasion.csv"),
    ("LLM", "S8-gpt", "LLM_S8-gpt_persuasion.csv"),
    ("LLM", "S8-llama", "LLM_S8-llama_persuasion.csv"),
    ("LLM", "S8-ministral", "LLM_S8-ministral_persuasion.csv"),
]
SPLIT_STAGE_FILES = [
    ("HW", "HW_S6_persuasion.csv"),
    ("HW", "HW_S8_persuasion.csv"),
]
HW_STAGE_CLONE_MAP = {
    "S8-gpt": "S8-llama",
    "S8-gemini": "S8-ministral",
    "S8-claude": "S8-deepseek",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Split stage-wise surrogate response maps into one folder per stage under "
            "Visualization/test/stages."
        )
    )
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--projection", choices=["pca", "umap"], default="pca")
    parser.add_argument("--seed", type=int, default=7)
    return parser.parse_args()


def _read_required_csv(csv_path: Path) -> pd.DataFrame:
    frame = pd.read_csv(csv_path)
    required_columns = {"subject", "body", "label", "scamllm"}
    missing = sorted(required_columns.difference(frame.columns))
    if missing:
        raise ValueError(f"{csv_path} is missing required columns: {', '.join(missing)}")
    frame = frame.copy()
    frame["label"] = pd.to_numeric(frame["label"], errors="coerce")
    frame = frame[frame["label"] == 1].copy()
    if frame.empty:
        return frame
    frame["detector_prediction"] = pd.to_numeric(frame["scamllm"], errors="coerce").fillna(0.0)
    frame["raw_label"] = frame["label"]
    frame["text"] = (
        frame["subject"].fillna("").astype(str).str.strip()
        + "\n\n"
        + frame["body"].fillna("").astype(str).str.strip()
    ).str.strip()
    return frame


def load_stage_frame() -> pd.DataFrame:
    frames: list[pd.DataFrame] = []
    exact_stage_names = {(source, stage) for source, stage, _ in EXACT_STAGE_FILES}

    for source_name, stage_name, filename in EXACT_STAGE_FILES:
        csv_path = WVAE_ROOT / filename
        if not csv_path.exists():
            continue
        frame = _read_required_csv(csv_path)
        if frame.empty:
            continue
        frame["source"] = source_name
        frame["stage"] = stage_name
        frames.append(frame)

    for source_name, filename in SPLIT_STAGE_FILES:
        csv_path = WVAE_ROOT / filename
        if not csv_path.exists():
            continue
        frame = _read_required_csv(csv_path)
        if frame.empty:
            continue
        if "source_file" not in frame.columns:
            raise ValueError(f"{csv_path} is missing required column: source_file")
        frame["stage"] = frame["source_file"].astype(str).str.replace(".csv", "", regex=False)
        frame["source"] = source_name
        frame = frame[frame["stage"].isin(STAGE_ORDER)].copy()
        frame = frame[
            ~frame["stage"].map(lambda stage_name: (source_name, stage_name) in exact_stage_names)
        ].copy()
        if not frame.empty:
            frames.append(frame)

    if not frames:
        raise FileNotFoundError(f"No usable phishing rows found under {WVAE_ROOT}")

    merged = pd.concat(frames, ignore_index=True)
    clone_frames: list[pd.DataFrame] = []
    for target_stage, source_stage in HW_STAGE_CLONE_MAP.items():
        source_rows = merged[(merged["source"] == "HW") & (merged["stage"] == source_stage)].copy()
        if source_rows.empty:
            continue
        source_rows["stage"] = target_stage
        if "source_file" in source_rows.columns:
            source_rows["source_file"] = f"{target_stage}.csv"
        source_rows["hw_stage_alias_from"] = source_stage
        clone_frames.append(source_rows)

    if clone_frames:
        merged = pd.concat([merged, *clone_frames], ignore_index=True)
    merged = merged.reset_index(drop=True)
    return merged


def add_outcome_points(axis: plt.Axes, frame: pd.DataFrame) -> None:
    for source_name in ["HW", "LLM"]:
        tp_frame = frame[(frame["source"] == source_name) & frame["is_tp"]]
        fn_frame = frame[(frame["source"] == source_name) & frame["is_fn"]]
        axis.scatter(tp_frame["proj_x"], tp_frame["proj_y"], **TP_STYLE[source_name])
        axis.scatter(fn_frame["proj_x"], fn_frame["proj_y"], **FN_STYLE[source_name])


def build_global_mesh(frame: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
    x_limits, y_limits = compute_axis_limits(frame)
    x_grid = np.linspace(*x_limits, 220)
    y_grid = np.linspace(*y_limits, 220)
    return np.meshgrid(x_grid, y_grid)


def draw_stage_figure(
    full_frame: pd.DataFrame,
    stage_name: str,
    output_dir: Path,
    projection_method: str,
    grid_x: np.ndarray,
    grid_y: np.ndarray,
    seed: int,
) -> dict[str, object]:
    stage_dir = output_dir / stage_name
    stage_dir.mkdir(parents=True, exist_ok=True)

    stage_frame = full_frame[full_frame["stage"] == stage_name].copy()
    surface_model = fit_detector_surface(stage_frame, seed)
    if surface_model is not None:
        mesh = np.column_stack([grid_x.ravel(), grid_y.ravel()])
        stage_score_grid = surface_model.predict_proba(mesh)[:, 1].reshape(grid_x.shape)
    else:
        stage_score_grid = np.full_like(
            grid_x,
            float(stage_frame["detector_prediction"].astype(int).mean()) if len(stage_frame) else 0.0,
            dtype=float,
        )

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

    filled = add_background(axis, grid_x, grid_y, stage_score_grid, projection_method)
    add_outcome_points(axis, stage_frame)

    hw_tp = int(((stage_frame["source"] == "HW") & stage_frame["is_tp"]).sum())
    hw_fn = int(((stage_frame["source"] == "HW") & stage_frame["is_fn"]).sum())
    llm_tp = int(((stage_frame["source"] == "LLM") & stage_frame["is_tp"]).sum())
    llm_fn = int(((stage_frame["source"] == "LLM") & stage_frame["is_fn"]).sum())

    axis.set_title(stage_name, fontsize=13, pad=10)
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

    legend_axis.legend(
        handles=build_main_legend(),
        loc="center",
        ncol=3,
        frameon=False,
    )
    colorbar = figure.colorbar(filled, cax=colorbar_axis)
    colorbar.set_label("Surrogate score")
    figure.suptitle(f"Stage-wise Surrogate Response Map: {stage_name}", y=0.97, fontsize=15)

    output_path = stage_dir / "surrogate_response_tp_fn_map_pca.png"
    figure.savefig(output_path, dpi=220)
    plt.close(figure)

    metadata = {
        "stage": stage_name,
        "projection_used": projection_method,
        "row_count": int(len(stage_frame)),
        "counts": {
            "HW": {"tp": hw_tp, "fn": hw_fn, "total": int((stage_frame["source"] == "HW").sum())},
            "LLM": {"tp": llm_tp, "fn": llm_fn, "total": int((stage_frame["source"] == "LLM").sum())},
        },
        "output_file": output_path.name,
    }
    (stage_dir / "metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    stage_frame.to_csv(stage_dir / "projected_points.csv", index=False)
    return metadata


def main() -> None:
    args = parse_args()
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    raw_frame = load_stage_frame()
    artifacts = project_indicator_space(raw_frame, projection=args.projection, seed=args.seed)
    frame = artifacts.frame.copy()
    frame["is_tp"] = frame["detector_prediction"] >= 0.5
    frame["is_fn"] = ~frame["is_tp"]
    frame["group"] = np.where(
        frame["source"].eq("HW") & frame["is_tp"],
        "HW-TP",
        np.where(
            frame["source"].eq("HW") & frame["is_fn"],
            "HW-FN",
            np.where(frame["source"].eq("LLM") & frame["is_tp"], "LLM-TP", "LLM-FN"),
        ),
    )

    overall_surface_model = fit_detector_surface(frame, args.seed)
    if overall_surface_model is None:
        raise RuntimeError("Could not fit the combined surrogate surface for the rebuilt projected points frame.")
    point_matrix = frame[["proj_x", "proj_y"]].to_numpy()
    frame["surrogate_score"] = overall_surface_model.predict_proba(point_matrix)[:, 1]
    if hasattr(overall_surface_model, "decision_function"):
        frame["decision_margin"] = overall_surface_model.decision_function(point_matrix)
    else:
        frame["decision_margin"] = frame["surrogate_score"] - 0.5

    grid_x, grid_y = build_global_mesh(frame)
    run_metadata: list[dict[str, object]] = []
    for index, stage_name in enumerate(STAGE_ORDER):
        run_metadata.append(
            draw_stage_figure(
                frame,
                stage_name,
                output_dir,
                artifacts.method,
                grid_x,
                grid_y,
                args.seed + index,
            )
        )

    summary = {
        "projection_used": artifacts.method,
        "stage_order": STAGE_ORDER,
        "output_dir": str(output_dir),
        "projected_points_output": str(DEFAULT_PROJECTED_POINTS_PATH),
        "hw_stage_clone_map": HW_STAGE_CLONE_MAP,
        "stages": run_metadata,
    }
    (output_dir / "run_metadata.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    frame.to_csv(output_dir / "projected_points_all_stages.csv", index=False)
    frame.to_csv(DEFAULT_PROJECTED_POINTS_PATH, index=False)


if __name__ == "__main__":
    main()
