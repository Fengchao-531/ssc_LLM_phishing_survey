#!/usr/bin/env python3
"""Generate phishing-only TP/FN failure-mode figures under Visualization/test.

This script builds one shared projection space across all phishing samples and
renders five aggregate figures centered on detector response and failure modes.
The current detector signal is the stored binary `scamllm` result.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from types import SimpleNamespace

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    brier_score_loss,
    f1_score,
    matthews_corrcoef,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)
from sklearn.model_selection import train_test_split

from generate_stage_visualizations import (
    STAGE_ORDER,
    build_sampled_frame,
    fit_detector_surface,
    project_indicator_space,
)
from generate_test_failure_boxplots import draw_fig4_score_distribution, draw_fig7_mixed_score_distribution
from generate_test_failure_contours import (
    build_global_grid,
    draw_fig0_source_overview,
    draw_fig1_main_map,
    draw_fig1_stagewise_map,
    draw_fig2_dual_panel,
    draw_fig3_fn_only,
)
from generate_test_failure_heatmaps import draw_fig5_indicator_heatmap


SCRIPT_DIR = Path(__file__).resolve().parent
DEFAULT_OUTPUT_DIR = SCRIPT_DIR / "test"
LEGACY_PROJECTED_CANDIDATES = [
    SCRIPT_DIR / "test" / "projected_points.csv",
    SCRIPT_DIR / "PCA" / "sampled_scamllm_full_phishing_only.csv",
]
MIXED_RAW_ROOT = SCRIPT_DIR / "persuasion_strategy_wvae" / "output" / "full_inference_results"
GROUP_ORDER = ["HW-TP", "HW-FN", "LLM-TP", "LLM-FN"]
GROUP_LABELS = {
    "HW-TP": "HW-TP",
    "HW-FN": "HW-FN",
    "LLM-TP": "LLM-TP",
    "LLM-FN": "LLM-FN",
}
FIGURE_FILES = {
    "fig0_source_overview": "fig0_hw_llm_phishing_overview_pca.png",
    "fig1_main_map": "fig1_surrogate_response_tp_fn_map_pca.png",
    "fig1_stagewise_map": "fig1_stagewise_surrogate_response_tp_fn_map_pca.png",
    "fig2_dual_panel": "fig2_hw_llm_dual_panel_pca.png",
    "fig3_fn_only": "fig3_fn_only_map_pca.png",
    "fig4_score_distribution": "fig4_score_margin_distribution_pca.png",
    "fig5_indicator_heatmap": "fig5_indicator_group_heatmap_pca.png",
    "fig6_reliability": "fig6_surrogate_reliability_mixed_pca.png",
    "fig7_mixed_score_distribution": "fig7_mixed_true_label_score_distribution_pca.png",
}

TP_STYLE = {
    "HW": {"marker": "o", "facecolors": "none", "edgecolors": "#2f2f2f", "linewidths": 1.0, "s": 28, "alpha": 0.8},
    "LLM": {"marker": "^", "facecolors": "none", "edgecolors": "#2f2f2f", "linewidths": 1.0, "s": 34, "alpha": 0.8},
}
FN_STYLE = {
    "HW": {"marker": "o", "c": "#2ca02c", "edgecolors": "white", "linewidths": 0.35, "s": 36, "alpha": 0.92},
    "LLM": {"marker": "^", "c": "#1f77b4", "edgecolors": "white", "linewidths": 0.35, "s": 42, "alpha": 0.92},
}
SOURCE_STYLE = {
    "HW": {"marker": "o", "c": "#7a1f3d", "edgecolors": "white", "linewidths": 0.30, "s": 26, "alpha": 0.70},
    "LLM": {"marker": "^", "c": "#2f6b3b", "edgecolors": "white", "linewidths": 0.30, "s": 32, "alpha": 0.70},
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Generate aggregate phishing-only TP/FN figures for scamllm under Visualization/test. "
            "Default projection is PCA."
        )
    )
    parser.add_argument("--detector-column", default="scamllm")
    parser.add_argument("--sample-size", type=int, default=0, help="Use 0 for all phishing rows.")
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--projection", choices=["pca", "umap"], default="pca")
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument(
        "--projected-input",
        type=Path,
        default=None,
        help="Optional preprojected phishing-only CSV. Defaults to legacy CSVs under Visualization/test or Visualization/PCA when present.",
    )
    parser.add_argument(
        "--mixed-input",
        type=Path,
        default=None,
        help="Optional mixed-data CSV for Figures 6-7. If omitted and no compatible source is available, mixed figures are skipped.",
    )
    return parser.parse_args()


def resolve_existing_csv(explicit_path: Path | None, candidates: list[Path]) -> Path | None:
    if explicit_path is not None:
        return explicit_path.resolve()
    for candidate in candidates:
        if candidate.exists():
            return candidate.resolve()
    return None


def finalize_phishing_frame(frame: pd.DataFrame) -> pd.DataFrame:
    frame = frame.copy()
    if "detector_prediction" not in frame.columns:
        frame["detector_prediction"] = pd.to_numeric(frame["scamllm"], errors="coerce").fillna(0.0)
    else:
        frame["detector_prediction"] = pd.to_numeric(frame["detector_prediction"], errors="coerce").fillna(0.0)
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
    return frame


def load_preprojected_csv(csv_path: Path) -> pd.DataFrame:
    frame = pd.read_csv(csv_path)
    required_columns = {"source", "stage", "proj_x", "proj_y", "scamllm"}
    missing = sorted(required_columns.difference(frame.columns))
    if missing:
        raise ValueError(f"{csv_path} is missing required columns: {', '.join(missing)}")
    return finalize_phishing_frame(frame)


def build_mixed_raw_frame(args: argparse.Namespace) -> pd.DataFrame:
    explicit_stage_files = [
        ("HW", "S1", MIXED_RAW_ROOT / "HW_S1_persuasion.csv"),
        ("HW", "S2", MIXED_RAW_ROOT / "HW_S2_persuasion.csv"),
        ("HW", "S4", MIXED_RAW_ROOT / "HW_S4_persuasion.csv"),
        ("HW", "S5", MIXED_RAW_ROOT / "HW_S5_persuasion.csv"),
        ("LLM", "S1", MIXED_RAW_ROOT / "LLM_S1_persuasion.csv"),
        ("LLM", "S2", MIXED_RAW_ROOT / "LLM_S2_persuasion.csv"),
    ]
    split_stage_files = [
        ("HW", MIXED_RAW_ROOT / "HW_S6_persuasion.csv"),
        ("HW", MIXED_RAW_ROOT / "HW_S8_persuasion.csv"),
    ]

    frames: list[pd.DataFrame] = []
    required_columns = {"subject", "body", "label", args.detector_column}

    for source_name, stage_name, csv_path in explicit_stage_files:
        if not csv_path.exists():
            continue
        frame = pd.read_csv(csv_path)
        missing = sorted(required_columns.difference(frame.columns))
        if missing:
            raise ValueError(f"{csv_path} is missing required columns: {', '.join(missing)}")
        frame = frame.copy()
        frame["stage"] = stage_name
        frame["source"] = source_name
        frames.append(frame)

    for source_name, csv_path in split_stage_files:
        if not csv_path.exists():
            continue
        frame = pd.read_csv(csv_path)
        missing = sorted(required_columns.union({"source_file"}).difference(frame.columns))
        if missing:
            raise ValueError(f"{csv_path} is missing required columns: {', '.join(missing)}")
        frame = frame.copy()
        frame["stage"] = frame["source_file"].astype(str).str.replace(".csv", "", regex=False)
        frame["source"] = source_name
        frames.append(frame)

    if not frames:
        raise FileNotFoundError(f"No mixed raw CSVs found under {MIXED_RAW_ROOT}")

    combined = pd.concat(frames, ignore_index=True)
    combined["label"] = pd.to_numeric(combined["label"], errors="coerce").fillna(0).astype(int)
    combined["detector_prediction"] = pd.to_numeric(combined[args.detector_column], errors="coerce").fillna(0.0)
    combined["raw_label"] = combined["label"]
    combined["text"] = (
        combined["subject"].fillna("").astype(str).str.strip()
        + "\n\n"
        + combined["body"].fillna("").astype(str).str.strip()
    ).str.strip()
    return combined.reset_index(drop=True)


def load_projected_frame(args: argparse.Namespace):
    projected_csv = resolve_existing_csv(args.projected_input, LEGACY_PROJECTED_CANDIDATES)
    if projected_csv is not None:
        frame = load_preprojected_csv(projected_csv)
        artifacts = SimpleNamespace(frame=frame, method=args.projection, source_csv=projected_csv)
        return frame, artifacts, "legacy_csv"

    loader_args = SimpleNamespace(
        detector_column=args.detector_column,
        sample_size=args.sample_size,
        seed=args.seed,
        phishing_only=True,
    )
    raw_frame = build_sampled_frame(loader_args)
    artifacts = project_indicator_space(raw_frame, projection=args.projection, seed=args.seed)
    frame = finalize_phishing_frame(artifacts.frame.copy())
    return frame, artifacts, "raw_loader"


def load_mixed_projected_frame(args: argparse.Namespace):
    if args.mixed_input is not None:
        frame = pd.read_csv(args.mixed_input)
        required_columns = {"source", "raw_label", "detector_prediction", "proj_x", "proj_y"}
        missing = sorted(required_columns.difference(frame.columns))
        if missing:
            raise ValueError(f"{args.mixed_input} is missing required columns: {', '.join(missing)}")
        frame = frame.copy()
        frame["true_label"] = pd.to_numeric(frame["raw_label"], errors="coerce").fillna(0).astype(int)
        frame["detector_target"] = pd.to_numeric(frame["detector_prediction"], errors="coerce").fillna(0).astype(int)
        frame["truth_group"] = (
            frame["source"].astype(str)
            + "-"
            + np.where(frame["true_label"].eq(1), "phishing", "benign")
        )
        artifacts = SimpleNamespace(frame=frame, method=args.projection, source_csv=args.mixed_input.resolve())
        return frame, artifacts, "mixed_csv"

    raw_frame = build_mixed_raw_frame(args)
    artifacts = project_indicator_space(raw_frame, projection=args.projection, seed=args.seed)
    frame = artifacts.frame.copy()
    frame["true_label"] = frame["raw_label"].astype(int)
    frame["detector_target"] = frame["detector_prediction"].astype(int)
    frame["truth_group"] = (
        frame["source"].astype(str)
        + "-"
        + np.where(frame["true_label"].eq(1), "phishing", "benign")
    )
    mixed_artifacts = SimpleNamespace(frame=frame, method=artifacts.method, source_csv=MIXED_RAW_ROOT)
    return frame, mixed_artifacts, "raw_loader"


def compute_surrogate_outputs(frame: pd.DataFrame, seed: int):
    surface_model = fit_detector_surface(frame, seed)
    if surface_model is None:
        raise RuntimeError("Combined phishing-only frame collapsed to a single class; cannot fit surrogate surface.")

    point_matrix = frame[["proj_x", "proj_y"]].to_numpy()
    frame = frame.copy()
    frame["surrogate_score"] = surface_model.predict_proba(point_matrix)[:, 1]
    if hasattr(surface_model, "decision_function"):
        frame["decision_margin"] = surface_model.decision_function(point_matrix)
    else:
        frame["decision_margin"] = frame["surrogate_score"] - 0.5
    return frame, surface_model


def build_metric_row(frame: pd.DataFrame) -> dict[str, float]:
    y_true = frame["detector_prediction"].astype(int).to_numpy()
    y_score = frame["surrogate_score"].to_numpy()
    y_pred = (y_score >= 0.5).astype(int)
    if len(y_true) == 0:
        return {
            "n": 0,
            "positive_rate": 0.0,
            "accuracy": float("nan"),
            "precision": float("nan"),
            "recall": float("nan"),
            "f1": float("nan"),
            "mcc": float("nan"),
            "roc_auc": float("nan"),
            "average_precision": float("nan"),
            "brier": float("nan"),
        }
    roc_auc = float(roc_auc_score(y_true, y_score)) if np.unique(y_true).size > 1 else float("nan")
    average_precision = float(average_precision_score(y_true, y_score)) if np.unique(y_true).size > 1 else float("nan")
    return {
        "n": int(len(frame)),
        "positive_rate": float(y_true.mean()) if len(frame) else 0.0,
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "precision": float(precision_score(y_true, y_pred, zero_division=0)),
        "recall": float(recall_score(y_true, y_pred, zero_division=0)),
        "f1": float(f1_score(y_true, y_pred, zero_division=0)),
        "mcc": float(matthews_corrcoef(y_true, y_pred)),
        "roc_auc": roc_auc,
        "average_precision": average_precision,
        "brier": float(brier_score_loss(y_true, y_score)),
    }


def build_binary_metric_row(y_true: np.ndarray, y_score: np.ndarray) -> dict[str, float]:
    y_pred = (y_score >= 0.5).astype(int)
    if len(y_true) == 0:
        return {
            "n": 0,
            "positive_rate": 0.0,
            "accuracy": float("nan"),
            "precision": float("nan"),
            "recall": float("nan"),
            "f1": float("nan"),
            "mcc": float("nan"),
            "roc_auc": float("nan"),
            "average_precision": float("nan"),
            "brier": float("nan"),
        }
    roc_auc = float(roc_auc_score(y_true, y_score)) if np.unique(y_true).size > 1 else float("nan")
    average_precision = float(average_precision_score(y_true, y_score)) if np.unique(y_true).size > 1 else float("nan")
    return {
        "n": int(len(y_true)),
        "positive_rate": float(y_true.mean()) if len(y_true) else 0.0,
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "precision": float(precision_score(y_true, y_pred, zero_division=0)),
        "recall": float(recall_score(y_true, y_pred, zero_division=0)),
        "f1": float(f1_score(y_true, y_pred, zero_division=0)),
        "mcc": float(matthews_corrcoef(y_true, y_pred)),
        "roc_auc": roc_auc,
        "average_precision": average_precision,
        "brier": float(brier_score_loss(y_true, y_score)),
    }


def write_surrogate_fit_reports(frame: pd.DataFrame, output_dir: Path):
    overall = build_metric_row(frame)
    (output_dir / "surrogate_fit_overall.json").write_text(json.dumps(overall, indent=2), encoding="utf-8")

    by_source = []
    for source_name in ["HW", "LLM"]:
        subset = frame[frame["source"] == source_name]
        metrics = build_metric_row(subset)
        metrics["source"] = source_name
        by_source.append(metrics)
    pd.DataFrame(by_source).to_csv(output_dir / "surrogate_fit_by_source.csv", index=False)

    by_stage = []
    for stage_name in STAGE_ORDER:
        subset = frame[frame["stage"] == stage_name]
        metrics = build_metric_row(subset)
        metrics["stage"] = stage_name
        by_stage.append(metrics)
    pd.DataFrame(by_stage).to_csv(output_dir / "surrogate_fit_by_stage.csv", index=False)


def compute_mixed_holdout_surrogate(mixed_frame: pd.DataFrame, seed: int):
    working = mixed_frame.copy()
    working["row_id"] = np.arange(len(working))
    strata = (
        working["source"].astype(str)
        + "_"
        + working["true_label"].astype(str)
        + "_"
        + working["detector_target"].astype(str)
    )
    train_ids, test_ids = train_test_split(
        working["row_id"].to_numpy(),
        test_size=0.25,
        random_state=seed,
        stratify=strata,
    )
    train_frame = working[working["row_id"].isin(train_ids)].copy()
    test_frame = working[working["row_id"].isin(test_ids)].copy()

    surrogate_train = train_frame.drop(columns=["detector_prediction"], errors="ignore").rename(
        columns={"detector_target": "detector_prediction"}
    )
    surface_model = fit_detector_surface(surrogate_train, seed)
    if surface_model is None:
        raise RuntimeError("Mixed holdout surrogate collapsed to a single class; cannot evaluate.")

    all_points = working[["proj_x", "proj_y"]].to_numpy()
    working["surrogate_score_holdout"] = surface_model.predict_proba(all_points)[:, 1]
    if hasattr(surface_model, "decision_function"):
        working["decision_margin_holdout"] = surface_model.decision_function(all_points)
    else:
        working["decision_margin_holdout"] = working["surrogate_score_holdout"] - 0.5
    return working, train_ids, test_ids


def draw_fig6_reliability(mixed_frame: pd.DataFrame, test_ids: np.ndarray, output_path: Path):
    test_frame = mixed_frame[mixed_frame["row_id"].isin(test_ids)].copy()
    score = test_frame["surrogate_score_holdout"].to_numpy()
    detector_target = test_frame["detector_target"].to_numpy(dtype=int)
    true_label = test_frame["true_label"].to_numpy(dtype=int)

    figure, axis = plt.subplots(figsize=(8.8, 7.0))
    axis.plot([0, 1], [0, 1], linestyle="--", color="#666666", linewidth=1.2, label="chance")

    curve_specs = [
        (detector_target, "#d95f02", "Against scamllm labels"),
        (true_label, "#1f77b4", "Against true phishing labels"),
    ]
    for y_true, color, label in curve_specs:
        if np.unique(y_true).size < 2:
            continue
        fpr, tpr, _ = roc_curve(y_true, score)
        auc_value = roc_auc_score(y_true, score)
        axis.plot(fpr, tpr, color=color, linewidth=2.2, label=f"{label} (AUC={auc_value:.3f})")

    axis.set_xlabel("False positive rate", fontsize=10)
    axis.set_ylabel("True positive rate", fontsize=10)
    axis.set_title("Mixed holdout ROC", fontsize=13)
    axis.grid(alpha=0.12, linewidth=0.5)
    axis.legend(frameon=False, loc="lower right")
    figure.suptitle("Figure 6. Surrogate ROC on Mixed Data Holdout", y=0.98, fontsize=15)
    figure.tight_layout(rect=(0, 0.02, 1, 0.95))
    figure.savefig(output_path, dpi=220, bbox_inches="tight")
    plt.close(figure)


def write_mixed_surrogate_reports(mixed_frame: pd.DataFrame, test_ids: np.ndarray, output_dir: Path):
    test_frame = mixed_frame[mixed_frame["row_id"].isin(test_ids)].copy()
    score = test_frame["surrogate_score_holdout"].to_numpy()
    detector_target = test_frame["detector_target"].to_numpy(dtype=int)
    true_label = test_frame["true_label"].to_numpy(dtype=int)

    report = {
        "holdout_n": int(len(test_frame)),
        "against_scamllm_labels": build_binary_metric_row(detector_target, score),
        "against_true_phishing_labels": build_binary_metric_row(true_label, score),
    }
    (output_dir / "surrogate_holdout_mixed_report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")

    by_truth_group = []
    for group_name, subset in test_frame.groupby("truth_group"):
        by_truth_group.append(
            {
                "truth_group": group_name,
                "n": int(len(subset)),
                "mean_score": float(subset["surrogate_score_holdout"].mean()),
                "median_score": float(subset["surrogate_score_holdout"].median()),
                "share_above_0_5": float((subset["surrogate_score_holdout"] >= 0.5).mean()),
            }
        )
    pd.DataFrame(by_truth_group).to_csv(output_dir / "surrogate_holdout_by_truth_group.csv", index=False)


def write_outputs(frame, args, artifacts, output_dir: Path):
    counts = (
        frame.groupby(["group", "source"])
        .size()
        .rename("sample_count")
        .reset_index()
        .sort_values(["group", "source"])
    )
    counts.to_csv(output_dir / "group_manifest.csv", index=False)
    metadata = {
        "detector_column": args.detector_column,
        "projection_requested": args.projection,
        "projection_used": artifacts.method,
        "sample_size_per_source": args.sample_size,
        "phishing_only": True,
        "threshold_tau": 0.5,
        "row_count": int(len(frame)),
        "figure_files": FIGURE_FILES,
    }
    if hasattr(artifacts, "source_csv"):
        metadata["projected_input_csv"] = str(artifacts.source_csv)
    (output_dir / "run_metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    frame.to_csv(output_dir / "projected_points.csv", index=False)
    write_surrogate_fit_reports(frame, output_dir)


def main() -> int:
    args = parse_args()
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    frame, artifacts, phishing_source = load_projected_frame(args)
    frame, surface_model = compute_surrogate_outputs(frame, args.seed)
    _, _, grid_x, grid_y, score_grid = build_global_grid(frame, surface_model)

    draw_fig0_source_overview(frame, output_dir / FIGURE_FILES["fig0_source_overview"], artifacts.method, grid_x, grid_y, score_grid, SOURCE_STYLE)
    draw_fig1_main_map(frame, output_dir / FIGURE_FILES["fig1_main_map"], artifacts.method, grid_x, grid_y, score_grid, GROUP_ORDER, TP_STYLE, FN_STYLE)
    draw_fig1_stagewise_map(frame, output_dir / FIGURE_FILES["fig1_stagewise_map"], artifacts.method, grid_x, grid_y, args.seed, TP_STYLE, FN_STYLE)
    draw_fig2_dual_panel(frame, output_dir / FIGURE_FILES["fig2_dual_panel"], artifacts.method, grid_x, grid_y, score_grid, TP_STYLE, FN_STYLE)
    draw_fig3_fn_only(frame, output_dir / FIGURE_FILES["fig3_fn_only"], artifacts.method, grid_x, grid_y, score_grid, FN_STYLE)
    draw_fig4_score_distribution(frame, output_dir / FIGURE_FILES["fig4_score_distribution"], GROUP_ORDER)
    draw_fig5_indicator_heatmap(frame, output_dir / FIGURE_FILES["fig5_indicator_heatmap"], GROUP_ORDER)
    write_outputs(frame, args, artifacts, output_dir)

    mixed_ready = False
    try:
        if args.mixed_input is not None or MIXED_RAW_ROOT.exists():
            mixed_frame, _, _ = load_mixed_projected_frame(args)
            mixed_ready = True
        else:
            mixed_frame = None
    except Exception as error:
        print(f"Skipping mixed-data figures: {error}")
        mixed_frame = None

    if mixed_ready and mixed_frame is not None:
        mixed_frame, train_ids, test_ids = compute_mixed_holdout_surrogate(mixed_frame, args.seed)
        draw_fig6_reliability(mixed_frame, test_ids, output_dir / FIGURE_FILES["fig6_reliability"])
        draw_fig7_mixed_score_distribution(mixed_frame, test_ids, output_dir / FIGURE_FILES["fig7_mixed_score_distribution"])
        write_mixed_surrogate_reports(mixed_frame, test_ids, output_dir)
    else:
        print("Skipping Figures 6-7 and mixed holdout reports because no compatible mixed-data input was found.")

    print(f"Saved test figures to: {output_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
