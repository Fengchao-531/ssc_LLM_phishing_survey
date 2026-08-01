#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from itertools import combinations
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import ttest_ind
from statsmodels.stats.multitest import multipletests

VIS_ROOT = Path(__file__).resolve().parents[1]
OVERVIEW_DIR = VIS_ROOT / "test" / "overview"
PROJECTED_PHISHING_PATH = VIS_ROOT / "test" / "projected_points.csv"
PROJECTED_MIXED_PATH = OVERVIEW_DIR / "projected_points_mixed_overview.csv"
FOCUS_METADATA_PATH = OVERVIEW_DIR / "overview_focus_metadata.json"
OUTPUT_ROOT = Path(__file__).resolve().parent / "overview"

PRINCIPLE_LABELS = [
    "Authority",
    "Liking",
    "Reciprocity",
    "Social Proof",
    "Scarcity",
    "Commitment",
]
PRINCIPLE_COLUMNS = [
    "principle_authority",
    "principle_liking",
    "principle_reciprocity",
    "principle_social_proof",
    "principle_scarcity",
    "principle_commitment",
]


@dataclass(frozen=True)
class ComparisonSpec:
    slug: str
    title: str
    group_a_name: str
    group_b_name: str
    group_a_frame: pd.DataFrame
    group_b_frame: pd.DataFrame

    @property
    def contrast_label(self) -> str:
        return f"{self.group_a_name} - {self.group_b_name}"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Run significance analysis for overview heatmap groups and write results "
            "under Visualization/statistic/overview."
        )
    )
    parser.add_argument("--permutations", type=int, default=300)
    parser.add_argument("--bootstraps", type=int, default=300)
    parser.add_argument("--seed", type=int, default=123)
    return parser.parse_args()


def region_mask(frame: pd.DataFrame, bounds: dict[str, float]) -> pd.Series:
    return (
        frame["proj_x"].between(bounds["x0"], bounds["x1"], inclusive="both")
        & frame["proj_y"].between(bounds["y0"], bounds["y1"], inclusive="both")
    )


def load_frames() -> tuple[pd.DataFrame, pd.DataFrame, dict[str, dict[str, float]]]:
    phishing_frame = pd.read_csv(PROJECTED_PHISHING_PATH)
    mixed_frame = pd.read_csv(PROJECTED_MIXED_PATH)
    focus_boxes = json.loads(FOCUS_METADATA_PATH.read_text(encoding="utf-8"))["focus_boxes"]

    phishing_frame = phishing_frame.copy()
    mixed_frame = mixed_frame.copy()
    for frame in [phishing_frame, mixed_frame]:
        for column in ["is_fn", "is_tp", "is_fn_phishing", "is_tp_phishing", "is_tn_benign"]:
            if column in frame.columns:
                frame[column] = frame[column].astype(bool)
    return phishing_frame, mixed_frame, focus_boxes


def extract_feature_matrix(frame: pd.DataFrame) -> np.ndarray:
    values = frame[PRINCIPLE_COLUMNS].dropna().to_numpy(dtype=float)
    if values.size == 0:
        return np.empty((0, 21), dtype=float)
    feature_columns: list[np.ndarray] = []
    for row_index, _ in enumerate(PRINCIPLE_COLUMNS):
        for col_index in range(row_index, len(PRINCIPLE_COLUMNS)):
            if row_index == col_index:
                feature_columns.append(values[:, row_index])
            else:
                feature_columns.append(values[:, row_index] * values[:, col_index])
    return np.column_stack(feature_columns)


def feature_index_table() -> list[tuple[int, int, str, str]]:
    feature_rows: list[tuple[int, int, str, str]] = []
    for row_index, row_name in enumerate(PRINCIPLE_LABELS):
        for col_index in range(row_index, len(PRINCIPLE_LABELS)):
            feature_rows.append((row_index, col_index, row_name, PRINCIPLE_LABELS[col_index]))
    return feature_rows


def vector_to_matrix(vector: np.ndarray) -> np.ndarray:
    matrix = np.full((len(PRINCIPLE_LABELS), len(PRINCIPLE_LABELS)), np.nan, dtype=float)
    for value, (row_index, col_index, _, _) in zip(vector, feature_index_table(), strict=True):
        matrix[row_index, col_index] = value
        matrix[col_index, row_index] = value
    return matrix


def compute_hedges_g(group_a: np.ndarray, group_b: np.ndarray) -> np.ndarray:
    n_a = group_a.shape[0]
    n_b = group_b.shape[0]
    if n_a < 2 or n_b < 2:
        return np.full(group_a.shape[1], np.nan, dtype=float)
    mean_diff = group_a.mean(axis=0) - group_b.mean(axis=0)
    var_a = group_a.var(axis=0, ddof=1)
    var_b = group_b.var(axis=0, ddof=1)
    pooled = np.sqrt(((n_a - 1) * var_a + (n_b - 1) * var_b) / max(1, n_a + n_b - 2))
    with np.errstate(divide="ignore", invalid="ignore"):
        d = mean_diff / pooled
    correction = 1.0 - (3.0 / max(1.0, 4.0 * (n_a + n_b) - 9.0))
    return d * correction


def bootstrap_confidence_intervals(
    group_a: np.ndarray,
    group_b: np.ndarray,
    *,
    bootstraps: int,
    rng: np.random.Generator,
) -> tuple[np.ndarray, np.ndarray]:
    boot_diffs = np.empty((bootstraps, group_a.shape[1]), dtype=float)
    for index in range(bootstraps):
        sample_a = group_a[rng.integers(0, group_a.shape[0], size=group_a.shape[0])]
        sample_b = group_b[rng.integers(0, group_b.shape[0], size=group_b.shape[0])]
        boot_diffs[index] = sample_a.mean(axis=0) - sample_b.mean(axis=0)
    return (
        np.percentile(boot_diffs, 2.5, axis=0),
        np.percentile(boot_diffs, 97.5, axis=0),
    )


def permutation_pvalues(
    group_a: np.ndarray,
    group_b: np.ndarray,
    *,
    permutations: int,
    rng: np.random.Generator,
) -> tuple[float, np.ndarray]:
    combined = np.vstack([group_a, group_b])
    n_a = group_a.shape[0]
    observed_diff = group_a.mean(axis=0) - group_b.mean(axis=0)
    observed_stat = float(np.linalg.norm(observed_diff))
    observed_abs = np.abs(observed_diff)

    overall_extreme = 0
    feature_extreme = np.zeros(group_a.shape[1], dtype=int)
    indices = np.arange(combined.shape[0])
    for _ in range(permutations):
        rng.shuffle(indices)
        perm_a = combined[indices[:n_a]]
        perm_b = combined[indices[n_a:]]
        perm_diff = perm_a.mean(axis=0) - perm_b.mean(axis=0)
        perm_stat = float(np.linalg.norm(perm_diff))
        overall_extreme += int(perm_stat >= observed_stat)
        feature_extreme += (np.abs(perm_diff) >= observed_abs).astype(int)

    overall_p = (overall_extreme + 1) / (permutations + 1)
    feature_p = (feature_extreme + 1) / (permutations + 1)
    return overall_p, feature_p


def draw_difference_heatmap(
    diff_matrix: np.ndarray,
    significance_mask: np.ndarray,
    output_path: Path,
    *,
    title: str,
    contrast_label: str,
) -> None:
    vmax = float(np.nanmax(np.abs(diff_matrix)))
    vmax = max(vmax, 1e-6)
    figure, axis = plt.subplots(figsize=(7.2, 6.2))
    image = axis.imshow(diff_matrix, cmap="RdBu_r", vmin=-vmax, vmax=vmax, aspect="equal")
    axis.set_xticks(range(len(PRINCIPLE_LABELS)))
    axis.set_xticklabels(PRINCIPLE_LABELS, rotation=45, ha="right", fontsize=9)
    axis.set_yticks(range(len(PRINCIPLE_LABELS)))
    axis.set_yticklabels(PRINCIPLE_LABELS, fontsize=9)

    for row_index in range(diff_matrix.shape[0]):
        for col_index in range(diff_matrix.shape[1]):
            if np.isnan(diff_matrix[row_index, col_index]):
                label = "NA"
            else:
                star = "*" if significance_mask[row_index, col_index] else ""
                label = f"{diff_matrix[row_index, col_index]:.2f}{star}"
            axis.text(col_index, row_index, label, ha="center", va="center", fontsize=7.8, color="black")

    colorbar = figure.colorbar(image, ax=axis, fraction=0.04, pad=0.03)
    colorbar.set_label(f"Mean difference [{contrast_label}]")
    figure.suptitle(f"{title}\nContrast: {contrast_label}", y=0.98, fontsize=14)
    figure.tight_layout(rect=(0, 0.03, 1, 0.94))
    figure.savefig(output_path, dpi=220)
    plt.close(figure)


def compare_groups(
    spec: ComparisonSpec,
    *,
    permutations: int,
    bootstraps: int,
    seed: int,
) -> dict[str, object]:
    output_dir = OUTPUT_ROOT / spec.slug
    output_dir.mkdir(parents=True, exist_ok=True)

    group_a = extract_feature_matrix(spec.group_a_frame)
    group_b = extract_feature_matrix(spec.group_b_frame)
    if len(group_a) == 0 or len(group_b) == 0:
        raise ValueError(f"{spec.slug} has an empty group and cannot be tested.")

    observed_a = group_a.mean(axis=0)
    observed_b = group_b.mean(axis=0)
    delta = observed_a - observed_b
    t_stat, welch_p = ttest_ind(group_a, group_b, axis=0, equal_var=False, nan_policy="omit")
    welch_p = np.nan_to_num(welch_p, nan=1.0)
    t_stat = np.nan_to_num(t_stat, nan=0.0)
    hedges_g = compute_hedges_g(group_a, group_b)

    rng = np.random.default_rng(seed)
    overall_p, permutation_p = permutation_pvalues(
        group_a,
        group_b,
        permutations=permutations,
        rng=rng,
    )
    ci_low, ci_high = bootstrap_confidence_intervals(
        group_a,
        group_b,
        bootstraps=bootstraps,
        rng=np.random.default_rng(seed + 10_000),
    )
    _, fdr_p, _, _ = multipletests(permutation_p, alpha=0.05, method="fdr_bh")

    feature_rows = []
    for index, (row_index, col_index, row_name, col_name) in enumerate(feature_index_table()):
        feature_rows.append(
            {
                "row_principle": row_name,
                "col_principle": col_name,
                "feature_type": "diagonal" if row_index == col_index else "cooccurrence",
                "mean_group_a": float(observed_a[index]),
                "mean_group_b": float(observed_b[index]),
                "delta_group_a_minus_group_b": float(delta[index]),
                "welch_t": float(t_stat[index]),
                "welch_p": float(welch_p[index]),
                "permutation_p": float(permutation_p[index]),
                "fdr_p": float(fdr_p[index]),
                "hedges_g": float(hedges_g[index]),
                "bootstrap_ci_low": float(ci_low[index]),
                "bootstrap_ci_high": float(ci_high[index]),
                "significant_fdr_0_05": bool(fdr_p[index] < 0.05),
            }
        )

    feature_table = pd.DataFrame(feature_rows)
    feature_table.to_csv(output_dir / "feature_stats.csv", index=False)

    diff_matrix = vector_to_matrix(delta)
    significance_matrix = vector_to_matrix((fdr_p < 0.05).astype(float)).astype(bool)
    draw_difference_heatmap(
        diff_matrix,
        significance_matrix,
        output_dir / "difference_heatmap.png",
        title=spec.title,
        contrast_label=spec.contrast_label,
    )

    summary = {
        "title": spec.title,
        "group_a_name": spec.group_a_name,
        "group_b_name": spec.group_b_name,
        "contrast_label": spec.contrast_label,
        "n_group_a": int(group_a.shape[0]),
        "n_group_b": int(group_b.shape[0]),
        "permutations": permutations,
        "bootstraps": bootstraps,
        "overall_difference_statistic_l2": float(np.linalg.norm(delta)),
        "overall_permutation_p": float(overall_p),
        "num_significant_features_fdr_0_05": int((fdr_p < 0.05).sum()),
        "num_total_features": int(len(feature_rows)),
        "files": {
            "feature_stats_csv": "feature_stats.csv",
            "difference_heatmap": "difference_heatmap.png",
        },
    }
    (output_dir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    return summary


def build_overview_comparisons() -> list[ComparisonSpec]:
    phishing_frame, mixed_frame, focus_boxes = load_frames()

    hw_focus_fn = phishing_frame[
        (phishing_frame["source"] == "HW")
        & phishing_frame["is_fn"]
        & region_mask(phishing_frame, focus_boxes["HW"])
    ].copy()
    llm_focus_fn = phishing_frame[
        (phishing_frame["source"] == "LLM")
        & phishing_frame["is_fn"]
        & region_mask(phishing_frame, focus_boxes["LLM"])
    ].copy()

    comparisons = [
        ComparisonSpec(
            slug="overview_focus_hw_p_fn_vs_llm_p_fn",
            title="Overview Focus FN Heatmap Difference: HW-P FN vs LLM-P FN",
            group_a_name="HW-P FN (focus)",
            group_b_name="LLM-P FN (focus)",
            group_a_frame=hw_focus_fn,
            group_b_frame=llm_focus_fn,
        ),
    ]

    global_groups: list[tuple[str, pd.DataFrame]] = [
        ("HW-P FN", mixed_frame[(mixed_frame["source"] == "HW") & mixed_frame["is_fn_phishing"]].copy()),
        ("HW-P TP", mixed_frame[(mixed_frame["source"] == "HW") & mixed_frame["is_tp_phishing"]].copy()),
        ("HW-B TN", mixed_frame[(mixed_frame["source"] == "HW") & mixed_frame["is_tn_benign"]].copy()),
        ("LLM-P FN", mixed_frame[(mixed_frame["source"] == "LLM") & mixed_frame["is_fn_phishing"]].copy()),
        ("LLM-P TP", mixed_frame[(mixed_frame["source"] == "LLM") & mixed_frame["is_tp_phishing"]].copy()),
        ("LLM-B TN", mixed_frame[(mixed_frame["source"] == "LLM") & mixed_frame["is_tn_benign"]].copy()),
    ]

    for (group_a_name, group_a_frame), (group_b_name, group_b_frame) in combinations(global_groups, 2):
        slug = (
            "overview_global_"
            + group_a_name.lower().replace("-", "_").replace(" ", "_")
            + "_vs_"
            + group_b_name.lower().replace("-", "_").replace(" ", "_")
        )
        comparisons.append(
            ComparisonSpec(
                slug=slug,
                title=f"Overview Global Heatmap Difference: {group_a_name} vs {group_b_name}",
                group_a_name=group_a_name,
                group_b_name=group_b_name,
                group_a_frame=group_a_frame,
                group_b_frame=group_b_frame,
            )
        )

    return comparisons


def write_summary_table(rows: list[dict[str, object]]) -> None:
    summary_frame = pd.DataFrame(rows)
    summary_frame.to_csv(OUTPUT_ROOT / "overview_summary.csv", index=False)

    markdown_lines = [
        "# Overview Heatmap Significance Summary",
        "",
        "| Comparison | Contrast (A-B) | n(A) | n(B) | Overall p | Significant cells (FDR<0.05) |",
        "|---|---|---:|---:|---:|---:|",
    ]
    for row in rows:
        markdown_lines.append(
            f"| {row['title']} | {row['contrast_label']} | {row['n_group_a']} | {row['n_group_b']} | "
            f"{row['overall_permutation_p']:.4f} | {row['num_significant_features_fdr_0_05']} |"
        )
    (OUTPUT_ROOT / "overview_summary.md").write_text("\n".join(markdown_lines) + "\n", encoding="utf-8")


def main() -> None:
    args = parse_args()
    OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)

    summaries: list[dict[str, object]] = []
    for index, spec in enumerate(build_overview_comparisons()):
        summaries.append(
            compare_groups(
                spec,
                permutations=args.permutations,
                bootstraps=args.bootstraps,
                seed=args.seed + index,
            )
        )
    write_summary_table(summaries)


if __name__ == "__main__":
    main()
