#!/usr/bin/env python3
from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.colors import LinearSegmentedColormap
from scipy.stats import kruskal, mannwhitneyu


SCRIPT_DIR = Path(__file__).resolve().parent
VIS_DIR = SCRIPT_DIR.parents[1]

EMAIL_INPUT = VIS_DIR / "test" / "overview" / "projected_points_mixed_overview.csv"
RQ2_SCORED = VIS_DIR / "RQ2" / "scored_inputs"
ROUND_INPUT = VIS_DIR / "test" / "fina_used" / "7" / "llm_multiturn_principle_turn_mean_trends_r6.csv"
ROUND_SCORED_INPUT = VIS_DIR / "test" / "fina_used" / "8" / "scored_inputs" / "llm_malicious_turns_scored.csv"

OUT_VALUES = SCRIPT_DIR / "rq2_llm_only_cross_setting_principle_means.csv"
OUT_KW = SCRIPT_DIR / "rq2_llm_only_cross_setting_kruskal_fdr.csv"
OUT_PAIRWISE = SCRIPT_DIR / "rq2_llm_only_cross_setting_pairwise_fdr.csv"
OUT_ROUND_KW = SCRIPT_DIR / "rq2_llm_multiturn_round_kruskal_fdr.csv"
OUT_FIG_PNG = SCRIPT_DIR / "rq2_llm_only_communication_persuasion_heatmap.png"
OUT_FIG_PDF = SCRIPT_DIR / "rq2_llm_only_communication_persuasion_heatmap.pdf"

PRINCIPLES = [
    ("Authority", "principle_authority"),
    ("Liking", "principle_liking"),
    ("Reciprocity", "principle_reciprocity"),
    ("Social Proof", "principle_social_proof"),
    ("Scarcity", "principle_scarcity"),
    ("Commitment", "principle_commitment"),
]
SETTINGS = ["Email", "Single-turn", "Multi-turn"]

CMAP = LinearSegmentedColormap.from_list(
    "blue_yellow_usenix",
    ["#fff7bc", "#d8eef0", "#9ecae1", "#2b8cbe", "#003d60"],
    N=256,
)


def bh_fdr(p_values: list[float]) -> list[float]:
    p = np.asarray(p_values, dtype=float)
    q = np.full_like(p, np.nan, dtype=float)
    valid = np.isfinite(p)
    if not valid.any():
        return q.tolist()
    valid_indices = np.where(valid)[0]
    order = valid_indices[np.argsort(p[valid])]
    ranked = p[order]
    adjusted = ranked * len(ranked) / np.arange(1, len(ranked) + 1)
    adjusted = np.minimum.accumulate(adjusted[::-1])[::-1]
    q[order] = np.clip(adjusted, 0.0, 1.0)
    return q.tolist()


def format_p(value: float) -> str:
    if not np.isfinite(value):
        return ""
    if value < 0.001:
        return "<0.001"
    return f"{value:.3f}"


def significance_stars(value: float) -> str:
    if not np.isfinite(value):
        return ""
    if value < 0.001:
        return "***"
    if value < 0.01:
        return "**"
    if value < 0.05:
        return "*"
    return ""


def load_llm_cross_setting_records() -> pd.DataFrame:
    email = pd.read_csv(EMAIL_INPUT, low_memory=False)
    email = email[
        (email["source"].astype(str) == "LLM")
        & (email["is_tp_phishing"].astype(bool) | email["is_fn_phishing"].astype(bool))
    ].copy()
    email["setting"] = "Email"

    single = pd.read_csv(RQ2_SCORED / "single_llm_vishing_persuasion.csv", low_memory=False)
    single["setting"] = "Single-turn"

    multi = pd.read_csv(RQ2_SCORED / "multi_llm_vishing_persuasion.csv", low_memory=False)
    multi["setting"] = "Multi-turn"

    columns = ["setting"] + [column for _, column in PRINCIPLES]
    return pd.concat([email[columns], single[columns], multi[columns]], ignore_index=True)


def build_value_table(records: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for principle, column in PRINCIPLES:
        setting_means = {}
        setting_ns = {}
        for setting in SETTINGS:
            values = pd.to_numeric(records.loc[records["setting"] == setting, column], errors="coerce").dropna()
            setting_means[setting] = float(values.mean())
            setting_ns[setting] = int(len(values))
        rows.append(
            {
                "Principle": principle,
                "Email": setting_means["Email"],
                "Single-turn": setting_means["Single-turn"],
                "Multi-turn": setting_means["Multi-turn"],
                "Max Difference": max(setting_means.values()) - min(setting_means.values()),
                "Email n": setting_ns["Email"],
                "Single-turn n": setting_ns["Single-turn"],
                "Multi-turn n": setting_ns["Multi-turn"],
            }
        )
    return pd.DataFrame(rows)


def build_stats(records: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    kw_rows: list[dict[str, object]] = []
    pair_rows: list[dict[str, object]] = []
    comparisons = [("Email", "Single-turn"), ("Email", "Multi-turn"), ("Single-turn", "Multi-turn")]
    for principle, column in PRINCIPLES:
        groups = [
            pd.to_numeric(records.loc[records["setting"] == setting, column], errors="coerce").dropna()
            for setting in SETTINGS
        ]
        h_stat, p_value = kruskal(*groups)
        kw_rows.append(
            {
                "Principle": principle,
                "H statistic": float(h_stat),
                "p": float(p_value),
                "p formatted": format_p(float(p_value)),
            }
        )
        for left, right in comparisons:
            left_values = pd.to_numeric(records.loc[records["setting"] == left, column], errors="coerce").dropna()
            right_values = pd.to_numeric(records.loc[records["setting"] == right, column], errors="coerce").dropna()
            result = mannwhitneyu(left_values, right_values, alternative="two-sided")
            u_stat = float(result.statistic)
            rank_biserial = 1.0 - 2.0 * (u_stat / float(len(left_values) * len(right_values)))
            pair_rows.append(
                {
                    "Principle": principle,
                    "Comparison": f"{left} vs {right}",
                    "Left mean": float(left_values.mean()),
                    "Right mean": float(right_values.mean()),
                    "Difference (right-left)": float(right_values.mean() - left_values.mean()),
                    "p": float(result.pvalue),
                    "p formatted": format_p(float(result.pvalue)),
                    "Effect Size (rank-biserial, right-left)": float(rank_biserial),
                    "n left": int(len(left_values)),
                    "n right": int(len(right_values)),
                }
            )

    kw = pd.DataFrame(kw_rows)
    kw["q (FDR)"] = bh_fdr(kw["p"].tolist())
    kw["q formatted"] = kw["q (FDR)"].apply(format_p)

    pairwise = pd.DataFrame(pair_rows)
    pairwise["q-value"] = bh_fdr(pairwise["p"].tolist())
    pairwise["q formatted"] = pairwise["q-value"].apply(format_p)
    return kw, pairwise


def round_matrix() -> pd.DataFrame:
    rounds = pd.read_csv(ROUND_INPUT)
    pivot = (
        rounds.pivot(index="principle", columns="round_number", values="mean_value")
        .reindex([name for name, _ in PRINCIPLES])
        .reindex(columns=range(1, 7))
    )
    pivot.columns = [f"R{column}" for column in pivot.columns]
    return pivot


def build_round_stats() -> pd.DataFrame:
    scored = pd.read_csv(ROUND_SCORED_INPUT, low_memory=False)
    scored["round_number"] = pd.to_numeric(scored["round_number"], errors="coerce")
    scored["total_malicious_turns"] = pd.to_numeric(scored["total_malicious_turns"], errors="coerce")
    scored = scored[scored["round_number"].notna()].copy()
    scored["round_number"] = scored["round_number"].astype(int)
    eligible = (
        scored[["dialogue_id", "total_malicious_turns"]]
        .drop_duplicates()
        .query("total_malicious_turns >= 6")["dialogue_id"]
    )
    scored = scored[scored["dialogue_id"].isin(eligible) & scored["round_number"].between(1, 6)].copy()

    rows: list[dict[str, object]] = []
    for principle, column in PRINCIPLES:
        groups = [
            pd.to_numeric(scored.loc[scored["round_number"] == round_number, column], errors="coerce").dropna()
            for round_number in range(1, 7)
        ]
        h_stat, p_value = kruskal(*groups)
        rows.append(
            {
                "Principle": principle,
                "H statistic": float(h_stat),
                "p": float(p_value),
                "p formatted": format_p(float(p_value)),
            }
        )
    stats = pd.DataFrame(rows)
    stats["q (FDR)"] = bh_fdr(stats["p"].tolist())
    stats["q formatted"] = stats["q (FDR)"].apply(format_p)
    return stats


def add_values(
    axis: plt.Axes,
    matrix: np.ndarray,
    stats: pd.DataFrame,
    fontsize: int = 18,
    star_fontsize: int = 18,
) -> None:
    stats_by_principle = stats.set_index("Principle")
    for row_index in range(matrix.shape[0]):
        for col_index in range(matrix.shape[1]):
            value = matrix[row_index, col_index]
            if np.isfinite(value):
                text_color = "white" if value >= 0.75 else "black"
                axis.text(
                    col_index,
                    row_index,
                    f"{value:.2f}",
                    ha="center",
                    va="center",
                    fontsize=fontsize,
                    color=text_color,
                )
                principle = PRINCIPLES[row_index][0]
                stars = significance_stars(float(stats_by_principle.loc[principle, "q (FDR)"]))
                if stars:
                    axis.text(
                        col_index + 0.31,
                        row_index - 0.30,
                        stars,
                        ha="center",
                        va="center",
                        fontsize=star_fontsize,
                        color=text_color,
                    )


def draw_heatmap(
    values: pd.DataFrame,
    rounds: pd.DataFrame,
    cross_stats: pd.DataFrame,
    round_stats: pd.DataFrame,
) -> None:
    plt.rcParams.update(
        {
            "font.family": "serif",
            "font.serif": ["Times New Roman", "Times", "Nimbus Roman", "DejaVu Serif"],
            "axes.linewidth": 1.1,
        }
    )

    cross = values.set_index("Principle")[SETTINGS].reindex([name for name, _ in PRINCIPLES])
    cross_matrix = cross.to_numpy(dtype=float)
    round_matrix_values = rounds.to_numpy(dtype=float)

    figure = plt.figure(figsize=(22.0, 7.8))
    grid = figure.add_gridspec(1, 3, width_ratios=[1.38, 1.42, 0.055], wspace=0.04)
    ax_cross = figure.add_subplot(grid[0, 0])
    ax_round = figure.add_subplot(grid[0, 1])
    cax = figure.add_subplot(grid[0, 2])

    image = ax_cross.imshow(cross_matrix, cmap=CMAP, vmin=0.0, vmax=1.0, aspect="auto")
    ax_round.imshow(round_matrix_values, cmap=CMAP, vmin=0.0, vmax=1.0, aspect="auto")

    principle_labels = [name for name, _ in PRINCIPLES]
    ax_cross.set_yticks(range(len(principle_labels)))
    ax_cross.set_yticklabels(principle_labels, fontsize=32)
    ax_round.set_yticks(range(len(principle_labels)))
    ax_round.set_yticklabels([])

    ax_cross.set_xticks(range(len(SETTINGS)))
    ax_cross.set_xticklabels(["Email", "Single-turn\nVishing", "Multi-turn\nVishing"], fontsize=32)
    ax_round.set_xticks(range(6))
    ax_round.set_xticklabels([f"R{index}" for index in range(1, 7)], fontsize=32)

    for axis in [ax_cross, ax_round]:
        axis.tick_params(axis="both", length=0)
        for spine in axis.spines.values():
            spine.set_visible(False)
        axis.set_xticks(np.arange(-0.5, axis.images[0].get_array().shape[1], 1), minor=True)
        axis.set_yticks(np.arange(-0.5, len(principle_labels), 1), minor=True)
        axis.grid(which="minor", color="white", linewidth=2.0)
        axis.tick_params(which="minor", bottom=False, left=False)

    ax_cross.set_xlabel("")
    ax_round.set_xlabel("Malicious turn round", fontsize=32, labelpad=10)

    add_values(ax_cross, cross_matrix, cross_stats, fontsize=28, star_fontsize=24)
    add_values(ax_round, round_matrix_values, round_stats, fontsize=28, star_fontsize=24)

    colorbar = figure.colorbar(image, cax=cax)
    colorbar.set_label("Mean persuasion score", fontsize=32, labelpad=12)
    colorbar.ax.tick_params(labelsize=28)

    figure.subplots_adjust(left=0.16, right=0.93, bottom=0.20, top=0.98)
    figure.savefig(OUT_FIG_PNG, dpi=300, bbox_inches="tight", pad_inches=0.04)
    figure.savefig(OUT_FIG_PDF, bbox_inches="tight", pad_inches=0.04)
    plt.close(figure)


def main() -> None:
    records = load_llm_cross_setting_records()
    values = build_value_table(records)
    kw, pairwise = build_stats(records)
    rounds = round_matrix()
    round_stats = build_round_stats()

    values.to_csv(OUT_VALUES, index=False)
    kw.to_csv(OUT_KW, index=False)
    pairwise.to_csv(OUT_PAIRWISE, index=False)
    round_stats.to_csv(OUT_ROUND_KW, index=False)
    draw_heatmap(values, rounds, kw, round_stats)


if __name__ == "__main__":
    main()
