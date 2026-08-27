#!/usr/bin/env python3
from __future__ import annotations

import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.colors import LinearSegmentedColormap
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KernelDensity
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

ROOT = Path(__file__).resolve().parent
VIS_ROOT = ROOT.parents[1]
SURVEY_ROOT = VIS_ROOT.parent

PROJECTED_POINTS = VIS_ROOT / "test" / "projected_points.csv"
EVAL_ROOT = SURVEY_ROOT / "Evaluation" / "processed-evaluation-datasets"
INDUSTRY_DETECTOR_COLUMN = "phishing_email_agent_prediction"
INDUSTRY_DETECTOR_LABEL = "Phishing Email Agent"
TP_OVERLAY_OUTPUT = ROOT / "fig_scamllm_tp_contour_with_industry_tp_overlay_pca.png"

PRINCIPLE_ORDER = [
    ("Authority", "principle_authority"),
    ("Liking", "principle_liking"),
    ("Reciprocity", "principle_reciprocity"),
    ("Social Proof", "principle_social_proof"),
    ("Scarcity", "principle_scarcity"),
    ("Commitment", "principle_commitment"),
]
PRINCIPLE_LABELS = [label for label, _ in PRINCIPLE_ORDER]
PRINCIPLE_COLUMNS = [column for _, column in PRINCIPLE_ORDER]

HEATMAP_CMAP = LinearSegmentedColormap.from_list(
    "academic_industry_gnbu",
    ["#f7fcf0", "#ccebc5", "#7bccc4", "#2b8cbe", "#084081"],
)


def normalize_binary(series: pd.Series) -> pd.Series:
    return pd.to_numeric(series, errors="coerce").fillna(0.0).astype(float)


def stage_to_source_file(stage: str) -> str:
    return f"{stage}.csv"


def load_projected_points() -> pd.DataFrame:
    frame = pd.read_csv(PROJECTED_POINTS, low_memory=False).copy()
    frame["subject"] = frame["subject"].fillna("").astype(str)
    frame["body"] = frame["body"].fillna("").astype(str)
    frame["raw_label"] = pd.to_numeric(frame["raw_label"], errors="coerce").fillna(0).astype(int)
    frame["scamllm_prediction"] = normalize_binary(frame["scamllm"])
    frame["academic_pred_label"] = (frame["scamllm_prediction"] >= 0.5).astype(int)
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
            requested = ["subject", "body", "label", INDUSTRY_DETECTOR_COLUMN]
            if "source_file" in columns:
                requested.append("source_file")
            frame = pd.read_csv(path, usecols=requested, low_memory=False).copy()
            frame["source"] = source_name
            if "source_file" in frame.columns:
                frame["merge_stage"] = frame["source_file"].astype(str).str.replace(".csv", "", regex=False)
                frame["merge_source_file"] = frame["source_file"].astype(str)
            else:
                frame["merge_stage"] = path.stem
                frame["merge_source_file"] = stage_to_source_file(path.stem)
            parts.append(frame)

    combined = pd.concat(parts, ignore_index=True)
    combined["subject"] = combined["subject"].fillna("").astype(str)
    combined["body"] = combined["body"].fillna("").astype(str)
    combined["label"] = pd.to_numeric(combined["label"], errors="coerce").fillna(0).astype(int)
    combined[INDUSTRY_DETECTOR_COLUMN] = normalize_binary(combined[INDUSTRY_DETECTOR_COLUMN])
    combined = combined.drop_duplicates(
        subset=["source", "subject", "body", "label", "merge_stage", "merge_source_file", INDUSTRY_DETECTOR_COLUMN]
    ).reset_index(drop=True)
    return combined


def attach_industry_predictions(projected: pd.DataFrame, lookup: pd.DataFrame) -> pd.DataFrame:
    merged = projected.merge(
        lookup,
        left_on=["source", "subject", "body", "raw_label", "merge_stage", "merge_source_file"],
        right_on=["source", "subject", "body", "label", "merge_stage", "merge_source_file"],
        how="left",
    )

    unmatched = merged[INDUSTRY_DETECTOR_COLUMN].isna()
    if unmatched.any():
        stage_lookup = (
            lookup.groupby(["source", "subject", "body", "label", "merge_stage"], as_index=False)[
                INDUSTRY_DETECTOR_COLUMN
            ].max()
        )
        fallback = projected.loc[unmatched].merge(
            stage_lookup,
            left_on=["source", "subject", "body", "raw_label", "merge_stage"],
            right_on=["source", "subject", "body", "label", "merge_stage"],
            how="left",
        )
        merged.loc[unmatched, INDUSTRY_DETECTOR_COLUMN] = fallback[INDUSTRY_DETECTOR_COLUMN].to_numpy()

    merged[INDUSTRY_DETECTOR_COLUMN] = pd.to_numeric(merged[INDUSTRY_DETECTOR_COLUMN], errors="coerce")
    merged["industry_pred_available"] = merged[INDUSTRY_DETECTOR_COLUMN].notna()
    merged["industry_pred_label"] = (merged[INDUSTRY_DETECTOR_COLUMN].fillna(0.0) >= 0.5).astype(int)
    return merged


def build_phishing_groups(frame: pd.DataFrame) -> pd.DataFrame:
    phishing = frame[frame["raw_label"].astype(int).eq(1) & frame["industry_pred_available"]].copy()
    phishing["academic_tp"] = phishing["academic_pred_label"].eq(1)
    phishing["academic_fn"] = phishing["academic_pred_label"].eq(0)
    phishing["industry_tp"] = phishing["industry_pred_label"].eq(1)
    phishing["industry_fn"] = phishing["industry_pred_label"].eq(0)
    phishing["both_tp"] = phishing["academic_tp"] & phishing["industry_tp"]
    phishing["both_fn"] = phishing["academic_fn"] & phishing["industry_fn"]
    phishing["academic_only_tp"] = phishing["academic_tp"] & phishing["industry_fn"]
    phishing["industry_only_tp"] = phishing["industry_tp"] & phishing["academic_fn"]
    return phishing


def pairwise_mean_matrix(frame: pd.DataFrame) -> np.ndarray:
    if frame.empty:
        return np.zeros((len(PRINCIPLE_COLUMNS), len(PRINCIPLE_COLUMNS)), dtype=float)
    values = frame[PRINCIPLE_COLUMNS].to_numpy(dtype=float)
    size = len(PRINCIPLE_COLUMNS)
    matrix = np.zeros((size, size), dtype=float)
    for i in range(size):
        for j in range(size):
            if i == j:
                matrix[i, j] = float(values[:, i].mean())
            else:
                matrix[i, j] = float((values[:, i] * values[:, j]).mean())
    return matrix


def save_heatmap_tables(tp_a: np.ndarray, tp_i: np.ndarray, ex_a: np.ndarray, ex_i: np.ndarray) -> None:
    def to_long_rows(group_name: str, matrix: np.ndarray) -> list[dict[str, object]]:
        rows = []
        for i, row_label in enumerate(PRINCIPLE_LABELS):
            for j, col_label in enumerate(PRINCIPLE_LABELS):
                rows.append(
                    {
                        "group": group_name,
                        "feature_a": row_label,
                        "feature_b": col_label,
                        "mean_strength": float(matrix[i, j]),
                    }
                )
        return rows

    pd.DataFrame(to_long_rows("Academic TP", tp_a) + to_long_rows("Industry TP", tp_i)).to_csv(
        ROOT / "tp_capture_heatmaps.csv",
        index=False,
    )
    pd.DataFrame(to_long_rows("Academic-only TP", ex_a) + to_long_rows("Industry-only TP", ex_i)).to_csv(
        ROOT / "exclusive_capture_heatmaps.csv",
        index=False,
    )


def fit_surrogate_grid(frame: pd.DataFrame, target_column: str) -> tuple[np.ndarray, np.ndarray, np.ndarray] | None:
    subset = frame[["proj_x", "proj_y", target_column]].dropna().copy()
    if subset[target_column].nunique() < 2:
        return None
    model = Pipeline(
        [
            ("scaler", StandardScaler()),
            ("clf", LogisticRegression(max_iter=1000)),
        ]
    )
    X = subset[["proj_x", "proj_y"]].to_numpy(dtype=float)
    y = subset[target_column].astype(int).to_numpy()
    model.fit(X, y)

    x_min, x_max = frame["proj_x"].min() - 0.8, frame["proj_x"].max() + 0.8
    y_min, y_max = frame["proj_y"].min() - 0.8, frame["proj_y"].max() + 0.8
    xx, yy = np.meshgrid(np.linspace(x_min, x_max, 220), np.linspace(y_min, y_max, 220))
    scores = model.predict_proba(np.column_stack([xx.ravel(), yy.ravel()]))[:, 1].reshape(xx.shape)
    return xx, yy, scores


def draw_tp_contours(phishing: pd.DataFrame) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(13.5, 5.6), constrained_layout=False)
    panels = [
        ("academic_tp", "Academic TP (ScamLLM)", "#1f78b4"),
        ("industry_tp", f"Industry TP ({INDUSTRY_DETECTOR_LABEL})", "#d95f02"),
    ]

    for ax, (column, title, color) in zip(axes, panels, strict=True):
        grid = fit_surrogate_grid(phishing, column)
        if grid is not None:
            xx, yy, scores = grid
            contour = ax.contourf(xx, yy, scores, levels=np.linspace(0, 1, 11), cmap="YlOrRd", alpha=0.78)
            ax.contour(xx, yy, scores, levels=[0.5], colors="black", linewidths=1.4)
        else:
            contour = None

        ax.scatter(phishing["proj_x"], phishing["proj_y"], s=8, c="black", alpha=0.11, linewidths=0)
        picked = phishing[phishing[column]]
        ax.scatter(
            picked["proj_x"],
            picked["proj_y"],
            s=14,
            c=color,
            alpha=0.75,
            linewidths=0.2,
            edgecolors="white",
        )
        ax.set_title(f"{title}\nTP={int(picked.shape[0])}", fontsize=12)
        ax.set_xlabel("PCA1")
        ax.grid(True, alpha=0.16, linewidth=0.5)
    axes[0].set_ylabel("PCA2")
    fig.suptitle("Academic vs Industry: Correctly Captured Phishing", fontsize=14, y=0.98)
    cax = fig.add_axes([0.92, 0.17, 0.018, 0.66])
    if contour is not None:
        cb = fig.colorbar(contour, cax=cax)
        cb.set_label("Surrogate capture probability")
    fig.subplots_adjust(left=0.06, right=0.89, bottom=0.12, top=0.84, wspace=0.18)
    fig.savefig(ROOT / "fig_academic_vs_industry_tp_contours_pca.png", dpi=220)
    plt.close(fig)


def fit_density_grid(frame: pd.DataFrame) -> tuple[np.ndarray, np.ndarray, np.ndarray] | None:
    subset = frame[["proj_x", "proj_y"]].dropna().copy()
    if len(subset) < 8:
        return None
    kde = KernelDensity(kernel="gaussian", bandwidth=0.55)
    kde.fit(subset.to_numpy(dtype=float))
    x_min, x_max = frame["proj_x"].min() - 0.8, frame["proj_x"].max() + 0.8
    y_min, y_max = frame["proj_y"].min() - 0.8, frame["proj_y"].max() + 0.8
    xx, yy = np.meshgrid(np.linspace(x_min, x_max, 220), np.linspace(y_min, y_max, 220))
    mesh = np.column_stack([xx.ravel(), yy.ravel()])
    scores = np.exp(kde.score_samples(mesh)).reshape(xx.shape)
    if scores.max() > 0:
        scores = scores / scores.max()
    return xx, yy, scores


def draw_scamllm_tp_overlay(projected: pd.DataFrame, phishing: pd.DataFrame) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(13.6, 5.7), sharex=True, sharey=True, constrained_layout=False)
    overall_phishing = projected[projected["raw_label"].astype(int).eq(1)].copy()

    for axis, source_name in zip(axes, ["HW", "LLM"], strict=True):
        source_frame = overall_phishing[overall_phishing["source"] == source_name].copy()
        industry_frame = phishing[phishing["source"] == source_name].copy()
        scamllm_tp = source_frame[source_frame["academic_pred_label"].eq(1)].copy()
        industry_tp = industry_frame[industry_frame["industry_tp"]].copy()
        grid = fit_density_grid(scamllm_tp)
        if grid is not None:
            xx, yy, scores = grid
            contour = axis.contourf(xx, yy, scores, levels=[0.15, 0.3, 0.45, 0.6, 0.75, 0.9, 1.01], cmap="Blues", alpha=0.42)
            axis.contour(xx, yy, scores, levels=[0.3, 0.5, 0.7, 0.9], colors="#1f4f82", linewidths=1.15)
        else:
            contour = None

        axis.scatter(source_frame["proj_x"], source_frame["proj_y"], s=7, c="#d9d9d9", alpha=0.12, linewidths=0)
        axis.scatter(
            industry_tp["proj_x"],
            industry_tp["proj_y"],
            s=24,
            c="#dd6b20",
            marker="^",
            alpha=0.88,
            linewidths=0.2,
            edgecolors="white",
            label=f"{INDUSTRY_DETECTOR_LABEL} TP",
        )
        axis.set_title(
            f"{source_name} phishing\nScamLLM TP={len(scamllm_tp)} | {INDUSTRY_DETECTOR_LABEL} TP={len(industry_tp)}",
            fontsize=12,
        )
        axis.set_xlabel("PCA1")
        axis.grid(True, alpha=0.14, linewidth=0.5)

    axes[0].set_ylabel("PCA2")
    axes[1].legend(loc="upper right", frameon=True)
    fig.suptitle("ScamLLM TP Contour with Industry TP Overlay", fontsize=14, y=0.98)
    if contour is not None:
        cax = fig.add_axes([0.92, 0.18, 0.018, 0.64])
        cb = fig.colorbar(contour, cax=cax)
        cb.set_label("ScamLLM TP contour intensity")
        fig.subplots_adjust(left=0.07, right=0.89, bottom=0.12, top=0.85, wspace=0.10)
    else:
        fig.subplots_adjust(left=0.07, right=0.97, bottom=0.12, top=0.85, wspace=0.10)
    fig.savefig(TP_OVERLAY_OUTPUT, dpi=220)
    plt.close(fig)


def annotate_heatmap(ax: plt.Axes, matrix: np.ndarray, title: str) -> None:
    image = ax.imshow(matrix, cmap=HEATMAP_CMAP, vmin=0, vmax=max(1e-6, float(matrix.max())))
    ax.set_xticks(range(len(PRINCIPLE_LABELS)))
    ax.set_yticks(range(len(PRINCIPLE_LABELS)))
    ax.set_xticklabels(PRINCIPLE_LABELS, rotation=45, ha="right", fontsize=9)
    ax.set_yticklabels(PRINCIPLE_LABELS, fontsize=9)
    ax.set_title(title, fontsize=12)
    for i in range(matrix.shape[0]):
        for j in range(matrix.shape[1]):
            ax.text(j, i, f"{matrix[i, j]:.2f}", ha="center", va="center", fontsize=8, color="#17324d")
    return image


def draw_tp_heatmaps(phishing: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
    academic = pairwise_mean_matrix(phishing[phishing["academic_tp"]])
    industry = pairwise_mean_matrix(phishing[phishing["industry_tp"]])
    vmax = max(float(academic.max()), float(industry.max()), 1e-6)

    fig, axes = plt.subplots(1, 2, figsize=(12.6, 5.4), constrained_layout=False)
    images = []
    for ax, matrix, title in zip(
        axes,
        [academic, industry],
        ["Academic TP", "Industry TP"],
        strict=True,
    ):
        image = ax.imshow(matrix, cmap=HEATMAP_CMAP, vmin=0, vmax=vmax)
        images.append(image)
        ax.set_xticks(range(len(PRINCIPLE_LABELS)))
        ax.set_yticks(range(len(PRINCIPLE_LABELS)))
        ax.set_xticklabels(PRINCIPLE_LABELS, rotation=45, ha="right", fontsize=9)
        ax.set_yticklabels(PRINCIPLE_LABELS, fontsize=9)
        ax.set_title(title, fontsize=12)
        for i in range(matrix.shape[0]):
            for j in range(matrix.shape[1]):
                ax.text(j, i, f"{matrix[i, j]:.2f}", ha="center", va="center", fontsize=8, color="#17324d")
    cax = fig.add_axes([0.92, 0.17, 0.018, 0.66])
    cb = fig.colorbar(images[-1], cax=cax)
    cb.set_label("Mean co-occurrence strength")
    fig.suptitle("Academic vs Industry: WVAE Feature Combinations for TP Phishing", fontsize=14, y=0.98)
    fig.subplots_adjust(left=0.07, right=0.89, bottom=0.22, top=0.86, wspace=0.18)
    fig.savefig(ROOT / "fig_academic_vs_industry_tp_heatmaps.png", dpi=220)
    plt.close(fig)
    return academic, industry


def draw_exclusive_contours(phishing: pd.DataFrame) -> None:
    fig, ax = plt.subplots(figsize=(7.5, 6.0), constrained_layout=False)
    ax.scatter(phishing["proj_x"], phishing["proj_y"], s=9, c="black", alpha=0.10, linewidths=0)

    academic_only = phishing[phishing["academic_only_tp"]]
    industry_only = phishing[phishing["industry_only_tp"]]

    ax.scatter(
        academic_only["proj_x"],
        academic_only["proj_y"],
        s=18,
        c="#1f78b4",
        alpha=0.78,
        edgecolors="white",
        linewidths=0.2,
        label=f"Academic-only TP (n={len(academic_only)})",
    )
    ax.scatter(
        industry_only["proj_x"],
        industry_only["proj_y"],
        s=18,
        c="#d95f02",
        alpha=0.78,
        marker="^",
        edgecolors="white",
        linewidths=0.2,
        label=f"Industry-only TP (n={len(industry_only)})",
    )
    ax.set_title("Academic vs Industry: Detector Disagreement on Phishing", fontsize=13)
    ax.set_xlabel("PCA1")
    ax.set_ylabel("PCA2")
    ax.grid(True, alpha=0.16, linewidth=0.5)
    ax.legend(loc="upper right", frameon=True)
    fig.subplots_adjust(left=0.10, right=0.97, bottom=0.11, top=0.90)
    fig.savefig(ROOT / "fig_academic_vs_industry_exclusive_contours_pca.png", dpi=220)
    plt.close(fig)


def draw_exclusive_heatmaps(phishing: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
    academic_only = pairwise_mean_matrix(phishing[phishing["academic_only_tp"]])
    industry_only = pairwise_mean_matrix(phishing[phishing["industry_only_tp"]])
    vmax = max(float(academic_only.max()), float(industry_only.max()), 1e-6)

    fig, axes = plt.subplots(1, 2, figsize=(12.6, 5.4), constrained_layout=False)
    images = []
    for ax, matrix, title in zip(
        axes,
        [academic_only, industry_only],
        ["Academic-only TP (Industry FN)", "Industry-only TP (Academic FN)"],
        strict=True,
    ):
        image = ax.imshow(matrix, cmap=HEATMAP_CMAP, vmin=0, vmax=vmax)
        images.append(image)
        ax.set_xticks(range(len(PRINCIPLE_LABELS)))
        ax.set_yticks(range(len(PRINCIPLE_LABELS)))
        ax.set_xticklabels(PRINCIPLE_LABELS, rotation=45, ha="right", fontsize=9)
        ax.set_yticklabels(PRINCIPLE_LABELS, fontsize=9)
        ax.set_title(title, fontsize=12)
        for i in range(matrix.shape[0]):
            for j in range(matrix.shape[1]):
                ax.text(j, i, f"{matrix[i, j]:.2f}", ha="center", va="center", fontsize=8, color="#17324d")
    cax = fig.add_axes([0.92, 0.17, 0.018, 0.66])
    cb = fig.colorbar(images[-1], cax=cax)
    cb.set_label("Mean co-occurrence strength")
    fig.suptitle("Academic vs Industry: WVAE Feature Combinations for Exclusive-Capture Phishing", fontsize=14, y=0.98)
    fig.subplots_adjust(left=0.07, right=0.89, bottom=0.22, top=0.86, wspace=0.18)
    fig.savefig(ROOT / "fig_academic_vs_industry_exclusive_heatmaps.png", dpi=220)
    plt.close(fig)
    return academic_only, industry_only


def write_summary(merged: pd.DataFrame, phishing: pd.DataFrame) -> None:
    summary = {
        "input_rows_projected_points": int(len(merged)),
        "industry_prediction_available_rows": int(merged["industry_pred_available"].sum()),
        "industry_prediction_available_phishing_rows": int(phishing.shape[0]),
        "source_counts_phishing": phishing["source"].value_counts().to_dict(),
        "stage_counts_phishing": phishing["stage"].value_counts().sort_index().to_dict(),
        "group_counts": {
            "academic_tp": int(phishing["academic_tp"].sum()),
            "industry_tp": int(phishing["industry_tp"].sum()),
            "academic_only_tp": int(phishing["academic_only_tp"].sum()),
            "industry_only_tp": int(phishing["industry_only_tp"].sum()),
            "both_tp": int(phishing["both_tp"].sum()),
            "both_fn": int(phishing["both_fn"].sum()),
        },
    }
    (ROOT / "tp_capture_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")


def main() -> None:
    ROOT.mkdir(parents=True, exist_ok=True)

    projected = load_projected_points()
    lookup = load_industry_lookup()
    merged = attach_industry_predictions(projected, lookup)
    phishing = build_phishing_groups(merged)

    merged.to_csv(ROOT / "merged_academic_industry_projected_points.csv", index=False)
    phishing.to_csv(ROOT / "merged_academic_industry_phishing_only.csv", index=False)

    draw_tp_contours(phishing)
    draw_scamllm_tp_overlay(projected, phishing)
    tp_academic, tp_industry = draw_tp_heatmaps(phishing)
    draw_exclusive_contours(phishing)
    ex_academic, ex_industry = draw_exclusive_heatmaps(phishing)
    save_heatmap_tables(tp_academic, tp_industry, ex_academic, ex_industry)
    write_summary(merged, phishing)


if __name__ == "__main__":
    main()
