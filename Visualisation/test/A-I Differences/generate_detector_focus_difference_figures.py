#!/usr/bin/env python3
from __future__ import annotations

import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.colors import TwoSlopeNorm
from matplotlib.colors import LinearSegmentedColormap
try:
    from scipy.stats import ttest_ind
except Exception:  # pragma: no cover
    ttest_ind = None
from sklearn.neighbors import KernelDensity
from sklearn.preprocessing import StandardScaler

ROOT = Path(__file__).resolve().parent
INPUT_PATH = ROOT / "merged_detector_hw_llm_projected_points.csv"

HEATMAP_OUTPUT = ROOT / "fig_detector_focus_difference_heatmap.png"
CONTOUR_OUTPUT = ROOT / "fig_detector_disagreement_contours_hw_llm.png"
METADATA_OUTPUT = ROOT / "detector_focus_difference_metadata.json"
HEATMAP_TABLE_OUTPUT = ROOT / "detector_focus_difference_heatmap_values.csv"
COMPOSITE_POINTS_OUTPUT = ROOT / "detector_disagreement_composite_points.csv"
SOURCE_DIFF_HEATMAP_OUTPUT = ROOT / "fig_detector_focus_difference_hw_llm_only_heatmaps.png"
PAIR_STATS_OUTPUT = ROOT / "detector_focus_difference_pair_stats.csv"
PAIR_SUMMARY_OUTPUT = ROOT / "detector_focus_difference_pair_summary.json"
GROUP_HEATMAP_OUTPUTS = {
    ("ScamLLM-only", "HW phishing"): ROOT / "fig_scamllm_only_hw_heatmap.png",
    ("ScamLLM-only", "LLM phishing"): ROOT / "fig_scamllm_only_llm_heatmap.png",
    ("Phishing Email Agent-only", "HW phishing"): ROOT / "fig_phishing_email_agent_only_hw_heatmap.png",
    ("Phishing Email Agent-only", "LLM phishing"): ROOT / "fig_phishing_email_agent_only_llm_heatmap.png",
}

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

A_ONLY_LABEL = "ScamLLM-only detected phishing"
B_ONLY_LABEL = "Phishing Email Agent-only detected phishing"
A_SHORT_LABEL = "ScamLLM-only"
B_SHORT_LABEL = "Phishing Email Agent-only"

GROUP_HEATMAP_CMAP = LinearSegmentedColormap.from_list(
    "detector_only_gnbu",
    ["#f7fcf0", "#ccebc5", "#7bccc4", "#2b8cbe", "#084081"],
)

# Action-oriented persuasion: authority/compliance/urgency-driven framing.
ACTION_COMPONENTS = [
    "principle_authority",
    "principle_reciprocity",
    "urgency_score",
    "threat_score",
    "payment_score",
    "credential_score",
    "link_score",
]

# Relational/social persuasion: socially framed or relationship-anchored language.
RELATIONAL_COMPONENTS = [
    "principle_liking",
    "principle_social_proof",
    "principle_commitment",
    "contact_score",
    "secrecy_score",
]


def load_frame() -> pd.DataFrame:
    frame = pd.read_csv(INPUT_PATH, low_memory=False).copy()
    frame = frame[frame["raw_label"].astype(int).eq(1)].copy()
    frame = frame[frame["phishing_email_agent_available"].astype(bool)].copy()

    frame["a_pred"] = pd.to_numeric(frame["scamllm"], errors="coerce").fillna(0.0).ge(0.5)
    frame["b_pred"] = pd.to_numeric(frame["phishing_email_agent_prediction"], errors="coerce").fillna(0.0).ge(0.5)
    frame["a_only"] = frame["a_pred"] & ~frame["b_pred"]
    frame["b_only"] = frame["b_pred"] & ~frame["a_pred"]
    frame["group"] = np.select(
        [frame["a_only"], frame["b_only"]],
        ["A-only", "B-only"],
        default="other",
    )

    for column in PRINCIPLE_COLUMNS + ACTION_COMPONENTS + RELATIONAL_COMPONENTS:
        frame[column] = pd.to_numeric(frame[column], errors="coerce").fillna(0.0)

    return frame


def pairwise_product_mean(frame: pd.DataFrame) -> np.ndarray:
    if frame.empty:
        return np.zeros((len(PRINCIPLE_COLUMNS), len(PRINCIPLE_COLUMNS)), dtype=float)

    values = frame[PRINCIPLE_COLUMNS].to_numpy(dtype=float)
    matrix = np.zeros((len(PRINCIPLE_COLUMNS), len(PRINCIPLE_COLUMNS)), dtype=float)
    for i in range(len(PRINCIPLE_COLUMNS)):
        for j in range(len(PRINCIPLE_COLUMNS)):
            matrix[i, j] = float((values[:, i] * values[:, j]).mean())
    return matrix


def build_difference_matrix(a_only: pd.DataFrame, b_only: pd.DataFrame) -> np.ndarray:
    return pairwise_product_mean(a_only) - pairwise_product_mean(b_only)


def export_heatmap_table(matrices: dict[str, np.ndarray]) -> None:
    rows: list[dict[str, object]] = []
    for subset_name, matrix in matrices.items():
        for i, row_label in enumerate(PRINCIPLE_LABELS):
            for j, col_label in enumerate(PRINCIPLE_LABELS):
                rows.append(
                    {
                        "subset": subset_name,
                        "feature_a": row_label,
                        "feature_b": col_label,
                        "difference_value": float(matrix[i, j]),
                    }
                )
    pd.DataFrame(rows).to_csv(HEATMAP_TABLE_OUTPUT, index=False)


def draw_single_group_heatmap(
    matrix: np.ndarray,
    *,
    title: str,
    output_path: Path,
    vmax: float,
) -> None:
    figure, axis = plt.subplots(figsize=(5.4, 4.8), constrained_layout=False)
    image = axis.imshow(matrix, cmap=GROUP_HEATMAP_CMAP, vmin=0.0, vmax=vmax)
    axis.set_xticks(range(len(PRINCIPLE_LABELS)))
    axis.set_yticks(range(len(PRINCIPLE_LABELS)))
    axis.set_xticklabels(PRINCIPLE_LABELS, rotation=45, ha="right", fontsize=9)
    axis.set_yticklabels(PRINCIPLE_LABELS, fontsize=9)
    axis.set_title(title, fontsize=12)
    for i in range(matrix.shape[0]):
        for j in range(matrix.shape[1]):
            axis.text(j, i, f"{matrix[i, j]:.2f}", ha="center", va="center", fontsize=8, color="#17324d")
    cax = figure.add_axes([0.90, 0.18, 0.025, 0.66])
    colorbar = figure.colorbar(image, cax=cax)
    colorbar.set_label("Mean co-occurrence strength")
    figure.subplots_adjust(left=0.10, right=0.87, bottom=0.24, top=0.86)
    figure.savefig(output_path, dpi=220)
    plt.close(figure)


def draw_group_only_heatmaps(frame: pd.DataFrame) -> None:
    matrices: dict[tuple[str, str], np.ndarray] = {}
    matrices[("ScamLLM-only", "HW phishing")] = pairwise_product_mean(
        frame[frame["a_only"] & frame["source"].eq("HW")]
    )
    matrices[("ScamLLM-only", "LLM phishing")] = pairwise_product_mean(
        frame[frame["a_only"] & frame["source"].eq("LLM")]
    )
    matrices[("Phishing Email Agent-only", "HW phishing")] = pairwise_product_mean(
        frame[frame["b_only"] & frame["source"].eq("HW")]
    )
    matrices[("Phishing Email Agent-only", "LLM phishing")] = pairwise_product_mean(
        frame[frame["b_only"] & frame["source"].eq("LLM")]
    )

    vmax = max(float(matrix.max()) for matrix in matrices.values())
    vmax = max(vmax, 1e-6)

    for (group_name, source_name), matrix in matrices.items():
        count = int(
            len(
                frame[
                    (
                        frame["a_only"]
                        if group_name == "ScamLLM-only"
                        else frame["b_only"]
                    )
                    & frame["source"].eq("HW" if source_name == "HW phishing" else "LLM")
                ]
            )
        )
        draw_single_group_heatmap(
            matrix,
            title=f"{group_name}\n{source_name} | n={count}",
            output_path=GROUP_HEATMAP_OUTPUTS[(group_name, source_name)],
            vmax=vmax,
        )


def draw_heatmap_figure(matrices: dict[str, np.ndarray], counts: dict[str, dict[str, int]]) -> None:
    vmax = max(np.abs(matrix).max() for matrix in matrices.values())
    vmax = max(float(vmax), 1e-6)
    norm = TwoSlopeNorm(vmin=-vmax, vcenter=0.0, vmax=vmax)

    figure, axes = plt.subplots(1, 3, figsize=(18.2, 5.7), constrained_layout=False)
    image = None
    panel_order = ["All phishing", "HW phishing", "LLM phishing"]

    for axis, panel_name in zip(axes, panel_order, strict=True):
        matrix = matrices[panel_name]
        image = axis.imshow(matrix, cmap="RdBu_r", norm=norm)
        axis.set_xticks(range(len(PRINCIPLE_LABELS)))
        axis.set_yticks(range(len(PRINCIPLE_LABELS)))
        axis.set_xticklabels(PRINCIPLE_LABELS, rotation=45, ha="right", fontsize=9)
        axis.set_yticklabels(PRINCIPLE_LABELS if panel_name == "All phishing" else [], fontsize=9)
        axis.set_title(
            f"{panel_name}\n{A_SHORT_LABEL}={counts[panel_name]['a_only']} | {B_SHORT_LABEL}={counts[panel_name]['b_only']}",
            fontsize=11,
        )
        for i in range(matrix.shape[0]):
            for j in range(matrix.shape[1]):
                axis.text(
                    j,
                    i,
                    f"{matrix[i, j]:.2f}",
                    ha="center",
                    va="center",
                    fontsize=7.8,
                    color="#152238",
                )

    cax = figure.add_axes([0.92, 0.16, 0.015, 0.68])
    colorbar = figure.colorbar(image, cax=cax)
    colorbar.set_label(
        "E[p_i p_j | ScamLLM-only] - E[p_i p_j | Phishing Email Agent-only]"
    )
    figure.suptitle("Detector Focus Difference Heatmap", fontsize=15, y=0.98)
    figure.subplots_adjust(left=0.06, right=0.90, bottom=0.24, top=0.85, wspace=0.20)
    figure.savefig(HEATMAP_OUTPUT, dpi=220)
    plt.close(figure)


def significance_stars(q_value: float) -> str:
    if np.isnan(q_value):
        return ""
    if q_value < 0.001:
        return "***"
    if q_value < 0.01:
        return "**"
    if q_value < 0.05:
        return "*"
    return ""


def draw_source_difference_heatmaps(
    matrices: dict[str, np.ndarray],
    counts: dict[str, dict[str, int]],
    stats_frame: pd.DataFrame,
) -> None:
    figure, axes = plt.subplots(1, 2, figsize=(12.8, 5.2), constrained_layout=False)
    panel_order = ["HW phishing", "LLM phishing"]
    vmax = max(float(np.abs(matrices[panel_name]).max()) for panel_name in panel_order)
    vmax = max(vmax, 1e-6)
    norm = TwoSlopeNorm(vmin=-vmax, vcenter=0.0, vmax=vmax)
    image = None

    for axis, panel_name in zip(axes, panel_order, strict=True):
        matrix = matrices[panel_name]
        panel_stats = stats_frame[stats_frame["source"].eq(panel_name)].copy()
        image = axis.imshow(matrix, cmap="RdBu_r", norm=norm)
        axis.set_xticks(range(len(PRINCIPLE_LABELS)))
        axis.set_yticks(range(len(PRINCIPLE_LABELS)))
        axis.set_xticklabels(PRINCIPLE_LABELS, rotation=45, ha="right", fontsize=9)
        axis.set_yticklabels(PRINCIPLE_LABELS if panel_name == "HW phishing" else [], fontsize=9)
        axis.set_title(
            f"{panel_name}\n{A_SHORT_LABEL}={counts[panel_name]['a_only']} | {B_SHORT_LABEL}={counts[panel_name]['b_only']}",
            fontsize=12,
        )
        for i in range(matrix.shape[0]):
            for j in range(matrix.shape[1]):
                q_match = panel_stats[
                    panel_stats["feature_a"].eq(PRINCIPLE_LABELS[i]) & panel_stats["feature_b"].eq(PRINCIPLE_LABELS[j])
                ]
                q_value = float(q_match["fdr_bh_q_value"].iloc[0]) if not q_match.empty else float("nan")
                stars = significance_stars(q_value)
                axis.text(j, i, f"{matrix[i, j]:.2f}", ha="center", va="center", fontsize=8, color="#152238")
                if stars:
                    axis.text(
                        j,
                        i - 0.30,
                        stars,
                        ha="center",
                        va="center",
                        fontsize=8.5,
                        color="#4d0000",
                        fontweight="bold",
                    )

    cax = figure.add_axes([0.92, 0.16, 0.018, 0.68])
    colorbar = figure.colorbar(image, cax=cax)
    colorbar.set_label("ScamLLM-only minus Phishing Email Agent-only")
    figure.suptitle("Difference Heatmaps by Source", fontsize=15, y=0.98)
    figure.subplots_adjust(left=0.07, right=0.90, bottom=0.23, top=0.84, wspace=0.18)
    figure.savefig(SOURCE_DIFF_HEATMAP_OUTPUT, dpi=220)
    plt.close(figure)


def add_composite_scores(frame: pd.DataFrame) -> pd.DataFrame:
    scored = frame.copy()
    all_components = ACTION_COMPONENTS + RELATIONAL_COMPONENTS
    scaler = StandardScaler()
    z_values = scaler.fit_transform(scored[all_components].to_numpy(dtype=float))
    z_frame = pd.DataFrame(z_values, columns=all_components, index=scored.index)
    scored["action_oriented_persuasion"] = z_frame[ACTION_COMPONENTS].mean(axis=1)
    scored["relational_social_persuasion"] = z_frame[RELATIONAL_COMPONENTS].mean(axis=1)
    return scored


def fit_group_kde(frame: pd.DataFrame, bandwidth: float = 0.55) -> KernelDensity | None:
    if len(frame) < 5:
        return None
    kde = KernelDensity(kernel="gaussian", bandwidth=bandwidth)
    kde.fit(frame[["action_oriented_persuasion", "relational_social_persuasion"]].to_numpy(dtype=float))
    return kde


def build_density_grid(
    extent_frame: pd.DataFrame,
    kde: KernelDensity | None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    x_min = float(extent_frame["action_oriented_persuasion"].min()) - 0.45
    x_max = float(extent_frame["action_oriented_persuasion"].max()) + 0.45
    y_min = float(extent_frame["relational_social_persuasion"].min()) - 0.45
    y_max = float(extent_frame["relational_social_persuasion"].max()) + 0.45
    grid_x, grid_y = np.meshgrid(np.linspace(x_min, x_max, 240), np.linspace(y_min, y_max, 240))

    if kde is None:
        density = np.zeros_like(grid_x, dtype=float)
    else:
        mesh = np.column_stack([grid_x.ravel(), grid_y.ravel()])
        density = np.exp(kde.score_samples(mesh)).reshape(grid_x.shape)
        if density.max() > 0:
            density = density / density.max()
    return grid_x, grid_y, density


def draw_contour_panel(axis: plt.Axes, frame: pd.DataFrame, title: str, extent_frame: pd.DataFrame) -> None:
    a_only = frame[frame["a_only"]].copy()
    b_only = frame[frame["b_only"]].copy()

    axis.scatter(
        frame["action_oriented_persuasion"],
        frame["relational_social_persuasion"],
        s=8,
        c="#d9d9d9",
        alpha=0.18,
        linewidths=0,
    )
    axis.scatter(
        a_only["action_oriented_persuasion"],
        a_only["relational_social_persuasion"],
        s=18,
        c="#2b6cb0",
        alpha=0.55,
        linewidths=0.2,
        edgecolors="white",
    )
    axis.scatter(
        b_only["action_oriented_persuasion"],
        b_only["relational_social_persuasion"],
        s=32,
        marker="^",
        c="#dd6b20",
        alpha=0.85,
        linewidths=0.25,
        edgecolors="white",
    )

    a_kde = fit_group_kde(a_only)
    b_kde = fit_group_kde(b_only)
    grid_x, grid_y, a_density = build_density_grid(extent_frame, a_kde)
    _, _, b_density = build_density_grid(extent_frame, b_kde)

    if a_density.max() > 0:
        axis.contourf(
            grid_x,
            grid_y,
            a_density,
            levels=[0.25, 0.45, 0.65, 0.82, 1.01],
            cmap="Blues",
            alpha=0.28,
        )
        axis.contour(
            grid_x,
            grid_y,
            a_density,
            levels=[0.45, 0.65, 0.82],
            colors=["#1f4f82"],
            linewidths=1.1,
        )

    if b_density.max() > 0:
        axis.contourf(
            grid_x,
            grid_y,
            b_density,
            levels=[0.25, 0.45, 0.65, 0.82, 1.01],
            cmap="Oranges",
            alpha=0.32,
        )
        axis.contour(
            grid_x,
            grid_y,
            b_density,
            levels=[0.45, 0.65, 0.82],
            colors=["#9c4221"],
            linewidths=1.15,
            linestyles="--",
        )

    axis.set_title(
        f"{title}\n{A_SHORT_LABEL}={len(a_only)} | {B_SHORT_LABEL}={len(b_only)}",
        fontsize=12,
    )
    axis.set_xlabel("Action-oriented persuasion", fontsize=10)
    axis.grid(alpha=0.12, linewidth=0.5)


def draw_contour_figure(scored: pd.DataFrame) -> None:
    disagreement = scored[scored["a_only"] | scored["b_only"]].copy()
    figure, axes = plt.subplots(1, 2, figsize=(14.4, 6.2), sharex=True, sharey=True)

    hw_frame = disagreement[disagreement["source"] == "HW"].copy()
    llm_frame = disagreement[disagreement["source"] == "LLM"].copy()

    draw_contour_panel(axes[0], hw_frame, "HW phishing disagreement", disagreement)
    draw_contour_panel(axes[1], llm_frame, "LLM phishing disagreement", disagreement)
    axes[0].set_ylabel("Relational / social persuasion", fontsize=10)

    legend_handles = [
        plt.Line2D(
            [0],
            [0],
            marker="o",
            color="none",
            markerfacecolor="#2b6cb0",
            markeredgecolor="white",
            markersize=7,
            linewidth=0,
            label=A_ONLY_LABEL,
        ),
        plt.Line2D(
            [0],
            [0],
            marker="^",
            color="none",
            markerfacecolor="#dd6b20",
            markeredgecolor="white",
            markersize=7,
            linewidth=0,
            label=B_ONLY_LABEL,
        ),
        plt.Line2D([0], [0], color="#1f4f82", linewidth=1.3, label="ScamLLM-only density"),
        plt.Line2D(
            [0],
            [0],
            color="#9c4221",
            linewidth=1.3,
            linestyle="--",
            label="Phishing Email Agent-only density",
        ),
    ]
    axes[1].legend(
        handles=legend_handles,
        loc="upper right",
        frameon=True,
    )

    figure.suptitle("Detector Disagreement Contour Plot", fontsize=15, y=0.98)
    figure.subplots_adjust(left=0.08, right=0.98, bottom=0.12, top=0.86, wspace=0.10)
    figure.savefig(CONTOUR_OUTPUT, dpi=220)
    plt.close(figure)


def cohen_d(a_values: np.ndarray, b_values: np.ndarray) -> float:
    if len(a_values) < 2 or len(b_values) < 2:
        return float("nan")
    a_var = float(np.var(a_values, ddof=1))
    b_var = float(np.var(b_values, ddof=1))
    pooled_den = len(a_values) + len(b_values) - 2
    if pooled_den <= 0:
        return float("nan")
    pooled_num = (len(a_values) - 1) * a_var + (len(b_values) - 1) * b_var
    pooled_std = np.sqrt(pooled_num / pooled_den) if pooled_num > 0 else 0.0
    if pooled_std == 0:
        return 0.0
    return float((np.mean(a_values) - np.mean(b_values)) / pooled_std)


def benjamini_hochberg(p_values: list[float]) -> list[float]:
    n = len(p_values)
    indexed = sorted(enumerate(p_values), key=lambda item: item[1])
    adjusted = [1.0] * n
    running = 1.0
    for reverse_rank, (original_index, p_value) in enumerate(reversed(indexed), start=1):
        rank = n - reverse_rank + 1
        candidate = min(running, p_value * n / rank)
        running = candidate
        adjusted[original_index] = float(min(candidate, 1.0))
    return adjusted


def export_pair_statistics(frame: pd.DataFrame) -> tuple[pd.DataFrame, dict[str, object]]:
    rows: list[dict[str, object]] = []
    summary: dict[str, object] = {}

    for source_name, source_code in [("HW phishing", "HW"), ("LLM phishing", "LLM")]:
        source_frame = frame[frame["source"].eq(source_code)].copy()
        a_frame = source_frame[source_frame["a_only"]].copy()
        b_frame = source_frame[source_frame["b_only"]].copy()
        source_rows: list[dict[str, object]] = []

        for row_label, row_column in PRINCIPLE_ORDER:
            for col_label, col_column in PRINCIPLE_ORDER:
                a_values = a_frame[row_column].to_numpy(dtype=float) * a_frame[col_column].to_numpy(dtype=float)
                b_values = b_frame[row_column].to_numpy(dtype=float) * b_frame[col_column].to_numpy(dtype=float)
                mean_a = float(a_values.mean()) if len(a_values) else 0.0
                mean_b = float(b_values.mean()) if len(b_values) else 0.0
                std_a = float(a_values.std(ddof=1)) if len(a_values) > 1 else 0.0
                std_b = float(b_values.std(ddof=1)) if len(b_values) > 1 else 0.0
                diff_value = mean_a - mean_b

                if ttest_ind is not None and len(a_values) > 1 and len(b_values) > 1:
                    result = ttest_ind(a_values, b_values, equal_var=False, nan_policy="omit")
                    t_stat = float(result.statistic)
                    p_value = float(result.pvalue)
                else:
                    t_stat = float("nan")
                    p_value = float("nan")

                row = {
                    "source": source_name,
                    "feature_a": row_label,
                    "feature_b": col_label,
                    "n_scamllm_only": int(len(a_values)),
                    "n_phishing_email_agent_only": int(len(b_values)),
                    "mean_scamllm_only": mean_a,
                    "std_scamllm_only": std_a,
                    "mean_phishing_email_agent_only": mean_b,
                    "std_phishing_email_agent_only": std_b,
                    "difference_value": diff_value,
                    "cohen_d": cohen_d(a_values, b_values),
                    "welch_t_stat": t_stat,
                    "welch_p_value": p_value,
                }
                source_rows.append(row)
                rows.append(row)

        raw_p_values = [row["welch_p_value"] for row in source_rows if not np.isnan(row["welch_p_value"])]
        adjusted = benjamini_hochberg(raw_p_values) if raw_p_values else []
        adjusted_iter = iter(adjusted)
        for row in source_rows:
            if np.isnan(row["welch_p_value"]):
                row["fdr_bh_q_value"] = float("nan")
            else:
                row["fdr_bh_q_value"] = float(next(adjusted_iter))

        top_positive = sorted(source_rows, key=lambda item: item["difference_value"], reverse=True)[:5]
        top_negative = sorted(source_rows, key=lambda item: item["difference_value"])[:5]
        significant_count = sum(
            1
            for row in source_rows
            if not np.isnan(row["fdr_bh_q_value"]) and row["fdr_bh_q_value"] < 0.05
        )
        summary[source_name] = {
            "significant_pair_count_fdr_lt_0_05": int(significant_count),
            "top_positive_pairs": top_positive,
            "top_negative_pairs": top_negative,
        }

    stats_frame = pd.DataFrame(rows)
    stats_frame.to_csv(PAIR_STATS_OUTPUT, index=False)
    PAIR_SUMMARY_OUTPUT.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    return stats_frame, summary


def write_metadata(
    frame: pd.DataFrame,
    scored: pd.DataFrame,
    counts: dict[str, dict[str, int]],
) -> None:
    metadata = {
        "input_file": str(INPUT_PATH),
        "figure_files": {
            "heatmap": HEATMAP_OUTPUT.name,
            "contour": CONTOUR_OUTPUT.name,
            "source_difference_heatmaps": SOURCE_DIFF_HEATMAP_OUTPUT.name,
            "scamllm_only_hw": GROUP_HEATMAP_OUTPUTS[("ScamLLM-only", "HW phishing")].name,
            "scamllm_only_llm": GROUP_HEATMAP_OUTPUTS[("ScamLLM-only", "LLM phishing")].name,
            "phishing_email_agent_only_hw": GROUP_HEATMAP_OUTPUTS[("Phishing Email Agent-only", "HW phishing")].name,
            "phishing_email_agent_only_llm": GROUP_HEATMAP_OUTPUTS[("Phishing Email Agent-only", "LLM phishing")].name,
        },
        "heatmap_table": HEATMAP_TABLE_OUTPUT.name,
        "pair_stats_table": PAIR_STATS_OUTPUT.name,
        "pair_summary_json": PAIR_SUMMARY_OUTPUT.name,
        "composite_points_table": COMPOSITE_POINTS_OUTPUT.name,
        "principles_used": PRINCIPLE_LABELS,
        "action_oriented_components": ACTION_COMPONENTS,
        "relational_social_components": RELATIONAL_COMPONENTS,
        "counts": counts,
        "overall": {
            "matched_phishing_rows": int(len(frame)),
            "a_only": int(frame["a_only"].sum()),
            "b_only": int(frame["b_only"].sum()),
            "both": int((frame["a_pred"] & frame["b_pred"]).sum()),
            "neither": int((~frame["a_pred"] & ~frame["b_pred"]).sum()),
        },
        "by_source": {
            source_name: {
                "a_only": int(group["a_only"].sum()),
                "b_only": int(group["b_only"].sum()),
                "rows": int(len(group)),
            }
            for source_name, group in frame.groupby("source")
        },
    }
    METADATA_OUTPUT.write_text(json.dumps(metadata, indent=2), encoding="utf-8")


def main() -> None:
    frame = load_frame()

    subsets = {
        "All phishing": frame,
        "HW phishing": frame[frame["source"] == "HW"].copy(),
        "LLM phishing": frame[frame["source"] == "LLM"].copy(),
    }

    matrices: dict[str, np.ndarray] = {}
    counts: dict[str, dict[str, int]] = {}
    for subset_name, subset_frame in subsets.items():
        a_only = subset_frame[subset_frame["a_only"]].copy()
        b_only = subset_frame[subset_frame["b_only"]].copy()
        matrices[subset_name] = build_difference_matrix(a_only, b_only)
        counts[subset_name] = {
            "a_only": int(len(a_only)),
            "b_only": int(len(b_only)),
        }

    export_heatmap_table(matrices)
    draw_heatmap_figure(matrices, counts)
    stats_frame, _ = export_pair_statistics(frame)
    draw_source_difference_heatmaps(matrices, counts, stats_frame)
    draw_group_only_heatmaps(frame)

    scored = add_composite_scores(frame)
    disagreement = scored[scored["a_only"] | scored["b_only"]].copy()
    disagreement[
        [
            "source",
            "stage",
            "a_only",
            "b_only",
            "action_oriented_persuasion",
            "relational_social_persuasion",
        ]
    ].to_csv(COMPOSITE_POINTS_OUTPUT, index=False)
    draw_contour_figure(scored)
    write_metadata(frame, scored, counts)


if __name__ == "__main__":
    main()
