#!/usr/bin/env python3
from __future__ import annotations

import json
from itertools import combinations_with_replacement
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import mannwhitneyu
from statsmodels.stats.multitest import multipletests


ROOT = Path(__file__).resolve().parent
PROJECT_ROOT = ROOT.parents[2]
STAGES_ROOT = PROJECT_ROOT / "Visualization" / "test" / "stages"
MCC_TABLE = PROJECT_ROOT / "Evaluation" / "stage-transfer-trend" / "mcc_stage_detector_overview_all.csv"

METHODS = ["S6-fuzzer", "S6-UTA", "S6-MPG"]
METHOD_LABELS = {
    "S6-fuzzer": "Fuzzer",
    "S6-UTA": "UTA",
    "S6-MPG": "MPG",
}
METHOD_COLORS = {
    "Fuzzer": "#003b4d",
    "UTA": "#f5c64f",
    "MPG": "#f28a17",
}

PRINCIPLES = [
    ("A", "Authority", "principle_authority"),
    ("L", "Liking", "principle_liking"),
    ("R", "Reciprocity", "principle_reciprocity"),
    ("SP", "Social Proof", "principle_social_proof"),
    ("S", "Scarcity", "principle_scarcity"),
    ("C", "Commitment", "principle_commitment"),
]


def pair_specs() -> list[dict[str, str]]:
    specs = []
    for left, right in combinations_with_replacement(PRINCIPLES, 2):
        left_short, left_name, left_col = left
        right_short, right_name, right_col = right
        specs.append(
            {
                "pair": f"{left_short}-{right_short}",
                "pair_full": f"{left_name} + {right_name}",
                "left_col": left_col,
                "right_col": right_col,
                "diagonal": left_col == right_col,
            }
        )
    return specs


def pair_values(frame: pd.DataFrame, spec: dict[str, str]) -> pd.Series:
    left = pd.to_numeric(frame[spec["left_col"]], errors="coerce").fillna(0.0)
    if spec["diagonal"]:
        return left
    right = pd.to_numeric(frame[spec["right_col"]], errors="coerce").fillna(0.0)
    return left * right


def bh_q_values(frame: pd.DataFrame, p_col: str, q_col: str) -> pd.DataFrame:
    out = frame.copy()
    out[q_col] = np.nan
    for method, idx in out.groupby("method").groups.items():
        p_values = out.loc[idx, p_col].fillna(1.0).to_numpy()
        out.loc[idx, q_col] = multipletests(p_values, method="fdr_bh")[1]
    return out


def rank_biserial_from_u(u_stat: float, n_a: int, n_b: int) -> float:
    if n_a == 0 or n_b == 0:
        return float("nan")
    return 2.0 * u_stat / (n_a * n_b) - 1.0


def load_method_frame(method: str) -> pd.DataFrame:
    path = STAGES_ROOT / method / "projected_points_mixed.csv"
    usecols = [
        "source",
        "true_label",
        "is_tp_phishing",
        "is_fn_phishing",
        *[column for _, _, column in PRINCIPLES],
    ]
    frame = pd.read_csv(path, usecols=usecols, low_memory=False)
    frame = frame[pd.to_numeric(frame["true_label"], errors="coerce").eq(1)].copy()
    frame["source"] = frame["source"].fillna("").astype(str)
    frame["is_tp_phishing"] = frame["is_tp_phishing"].astype(bool)
    frame["is_fn_phishing"] = frame["is_fn_phishing"].astype(bool)
    return frame


def summarize_detector_performance() -> pd.DataFrame:
    mcc = pd.read_csv(MCC_TABLE)
    cols = ["category", "detector", "llm_S6-fuzzer", "llm_S6-UTA", "llm_S6-MPG"]
    table = mcc[cols].rename(
        columns={
            "llm_S6-fuzzer": "Fuzzer_MCC",
            "llm_S6-UTA": "UTA_MCC",
            "llm_S6-MPG": "MPG_MCC",
        }
    )
    table["MCC_range_max_minus_min"] = table[["Fuzzer_MCC", "UTA_MCC", "MPG_MCC"]].max(axis=1) - table[
        ["Fuzzer_MCC", "UTA_MCC", "MPG_MCC"]
    ].min(axis=1)
    table = table.sort_values(["category", "detector"]).reset_index(drop=True)
    table.to_csv(ROOT / "s6_detector_performance_mcc.csv", index=False)
    return table


def compute_pair_tables() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    specs = pair_specs()
    mean_rows = []
    stat_rows = []

    for method in METHODS:
        method_label = METHOD_LABELS[method]
        frame = load_method_frame(method)
        groups = {
            "HW-FN": frame[frame["source"].eq("HW") & frame["is_fn_phishing"]],
            "LLM-FN": frame[frame["source"].eq("LLM") & frame["is_fn_phishing"]],
            "LLM-TP": frame[frame["source"].eq("LLM") & frame["is_tp_phishing"]],
        }
        for spec in specs:
            values = {name: pair_values(group, spec) for name, group in groups.items()}
            hw_fn_mean = float(values["HW-FN"].mean())
            llm_fn_mean = float(values["LLM-FN"].mean())
            llm_tp_mean = float(values["LLM-TP"].mean())
            delta_llm_fn_minus_hw_fn = llm_fn_mean - hw_fn_mean
            delta_llm_fn_minus_llm_tp = llm_fn_mean - llm_tp_mean

            mean_rows.append(
                {
                    "method": method_label,
                    "stage": method,
                    "pair": spec["pair"],
                    "pair_full": spec["pair_full"],
                    "HW-FN_mean": hw_fn_mean,
                    "LLM-FN_mean": llm_fn_mean,
                    "LLM-TP_mean": llm_tp_mean,
                    "delta_LLM_FN_minus_HW_FN": delta_llm_fn_minus_hw_fn,
                    "abs_delta_LLM_FN_minus_HW_FN": abs(delta_llm_fn_minus_hw_fn),
                    "delta_LLM_FN_minus_LLM_TP": delta_llm_fn_minus_llm_tp,
                    "abs_delta_LLM_FN_minus_LLM_TP": abs(delta_llm_fn_minus_llm_tp),
                    "n_HW_FN": int(len(values["HW-FN"])),
                    "n_LLM_FN": int(len(values["LLM-FN"])),
                    "n_LLM_TP": int(len(values["LLM-TP"])),
                }
            )

            comparisons = [
                ("LLM-FN_vs_HW-FN", values["LLM-FN"], values["HW-FN"]),
                ("LLM-FN_vs_LLM-TP", values["LLM-FN"], values["LLM-TP"]),
            ]
            for comparison, group_a, group_b in comparisons:
                if len(group_a) == 0 or len(group_b) == 0:
                    u_stat = float("nan")
                    p_value = 1.0
                    effect = float("nan")
                else:
                    result = mannwhitneyu(group_a, group_b, alternative="two-sided")
                    u_stat = float(result.statistic)
                    p_value = float(result.pvalue)
                    effect = rank_biserial_from_u(u_stat, len(group_a), len(group_b))
                stat_rows.append(
                    {
                        "method": method_label,
                        "stage": method,
                        "pair": spec["pair"],
                        "pair_full": spec["pair_full"],
                        "comparison": comparison,
                        "group_a": comparison.split("_vs_")[0],
                        "group_b": comparison.split("_vs_")[1],
                        "group_a_mean": float(group_a.mean()) if len(group_a) else float("nan"),
                        "group_a_median": float(group_a.median()) if len(group_a) else float("nan"),
                        "group_b_mean": float(group_b.mean()) if len(group_b) else float("nan"),
                        "group_b_median": float(group_b.median()) if len(group_b) else float("nan"),
                        "u_statistic": u_stat,
                        "p_value": p_value,
                        "rank_biserial_effect": effect,
                        "abs_rank_biserial_effect": abs(effect) if not np.isnan(effect) else float("nan"),
                        "n_group_a": int(len(group_a)),
                        "n_group_b": int(len(group_b)),
                    }
                )

    means = pd.DataFrame(mean_rows)
    stats = pd.DataFrame(stat_rows)
    stats = bh_q_values(stats, "p_value", "q_value")

    means.to_csv(ROOT / "s6_persuasion_pair_means.csv", index=False)
    stats.to_csv(ROOT / "s6_pair_mannwhitney_tests.csv", index=False)

    shift_summary = (
        means.groupby(["method", "stage"], as_index=False)
        .agg(
            delta_m=("abs_delta_LLM_FN_minus_HW_FN", "mean"),
            max_abs_pair_shift=("abs_delta_LLM_FN_minus_HW_FN", "max"),
            mean_abs_fn_tp_gap=("abs_delta_LLM_FN_minus_LLM_TP", "mean"),
            n_pairs=("pair", "count"),
        )
        .sort_values("delta_m", ascending=False)
    )
    shift_summary.to_csv(ROOT / "s6_method_shift_summary.csv", index=False)

    return means, stats, shift_summary


def build_top_tables(means: pd.DataFrame, stats: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    top_changed = (
        means.sort_values(["method", "abs_delta_LLM_FN_minus_HW_FN"], ascending=[True, False])
        .groupby("method", as_index=False)
        .head(8)
        .reset_index(drop=True)
    )
    top_changed.to_csv(ROOT / "s6_top_changed_pairs_by_method.csv", index=False)

    llm_hw = stats[stats["comparison"].eq("LLM-FN_vs_HW-FN")][
        ["method", "pair", "q_value", "p_value", "rank_biserial_effect"]
    ].rename(
        columns={
            "q_value": "q_LLM_FN_vs_HW_FN",
            "p_value": "p_LLM_FN_vs_HW_FN",
            "rank_biserial_effect": "effect_LLM_FN_vs_HW_FN",
        }
    )
    fn_tp = stats[stats["comparison"].eq("LLM-FN_vs_LLM-TP")][
        ["method", "pair", "q_value", "p_value", "rank_biserial_effect"]
    ].rename(
        columns={
            "q_value": "q_LLM_FN_vs_LLM_TP",
            "p_value": "p_LLM_FN_vs_LLM_TP",
            "rank_biserial_effect": "effect_LLM_FN_vs_LLM_TP",
        }
    )
    candidates = (
        means.merge(llm_hw, on=["method", "pair"], how="left")
        .merge(fn_tp, on=["method", "pair"], how="left")
        .assign(
            significant_LLM_vs_HW=lambda df: df["q_LLM_FN_vs_HW_FN"].lt(0.05),
            significant_FN_vs_TP=lambda df: df["q_LLM_FN_vs_LLM_TP"].lt(0.05),
        )
    )
    candidates = candidates[
        candidates["significant_LLM_vs_HW"] & candidates["significant_FN_vs_TP"]
    ].sort_values(["method", "abs_delta_LLM_FN_minus_HW_FN"], ascending=[True, False])
    candidates.to_csv(ROOT / "s6_candidate_pairs_for_final_figure.csv", index=False)
    return top_changed, candidates


def write_markdown(
    performance: pd.DataFrame,
    means: pd.DataFrame,
    stats: pd.DataFrame,
    shift_summary: pd.DataFrame,
    top_changed: pd.DataFrame,
    candidates: pd.DataFrame,
) -> None:
    securenet = performance[performance["detector"].eq("securenet_llama")].iloc[0]
    lines = [
        "# S6 Rewriting Mechanism Tables",
        "",
        "## Detector Performance",
        "",
        "| Detector | Fuzzer MCC | UTA MCC | MPG MCC | Range |",
        "|---|---:|---:|---:|---:|",
    ]
    for _, row in performance.iterrows():
        lines.append(
            f"| {row['category']} / {row['detector']} | {row['Fuzzer_MCC']:.4f} | "
            f"{row['UTA_MCC']:.4f} | {row['MPG_MCC']:.4f} | {row['MCC_range_max_minus_min']:.4f} |"
        )
    lines.extend(
        [
            "",
            f"Securenet: Fuzzer={securenet['Fuzzer_MCC']:.4f}, "
            f"UTA={securenet['UTA_MCC']:.4f}, MPG={securenet['MPG_MCC']:.4f}.",
            "",
            "## Method-Level Pair Shift",
            "",
            "| Method | Delta_m | Max pair shift | Mean FN-TP gap |",
            "|---|---:|---:|---:|",
        ]
    )
    for _, row in shift_summary.iterrows():
        lines.append(
            f"| {row['method']} | {row['delta_m']:.4f} | {row['max_abs_pair_shift']:.4f} | "
            f"{row['mean_abs_fn_tp_gap']:.4f} |"
        )
    lines.extend(["", "## Top Changed Pairs", "", "| Method | Pair | HW-FN | LLM-FN | LLM-TP | Delta LLM-FN - HW-FN |", "|---|---|---:|---:|---:|---:|"])
    for _, row in top_changed.groupby("method", as_index=False).head(5).iterrows():
        lines.append(
            f"| {row['method']} | {row['pair']} | {row['HW-FN_mean']:.4f} | "
            f"{row['LLM-FN_mean']:.4f} | {row['LLM-TP_mean']:.4f} | "
            f"{row['delta_LLM_FN_minus_HW_FN']:.4f} |"
        )
    lines.extend(
        [
            "",
            "## Candidate Pairs: LLM-vs-HW Changed and FN-vs-TP Significant",
            "",
            "| Method | Pair | HW-FN | LLM-FN | LLM-TP | q LLM-FN vs HW-FN | q LLM-FN vs LLM-TP | Effect FN vs TP |",
            "|---|---|---:|---:|---:|---:|---:|---:|",
        ]
    )
    for _, row in candidates.iterrows():
        lines.append(
            f"| {row['method']} | {row['pair']} | {row['HW-FN_mean']:.4f} | "
            f"{row['LLM-FN_mean']:.4f} | {row['LLM-TP_mean']:.4f} | "
            f"{row['q_LLM_FN_vs_HW_FN']:.3g} | {row['q_LLM_FN_vs_LLM_TP']:.3g} | "
            f"{row['effect_LLM_FN_vs_LLM_TP']:.4f} |"
        )
    (ROOT / "s6_rewriting_mechanism_summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def setup_plot_style() -> None:
    plt.rcParams.update(
        {
            "font.family": "serif",
            "font.serif": ["Times New Roman", "Times", "Nimbus Roman", "DejaVu Serif"],
            "axes.linewidth": 1.0,
            "xtick.major.width": 0.9,
            "ytick.major.width": 0.9,
        }
    )


def save_figure(fig: plt.Figure, stem: str) -> None:
    fig.savefig(ROOT / f"{stem}.png", dpi=300, bbox_inches="tight", pad_inches=0.06)
    fig.savefig(ROOT / f"{stem}.pdf", bbox_inches="tight", pad_inches=0.06)
    plt.close(fig)


def plot_detector_performance(performance: pd.DataFrame) -> None:
    setup_plot_style()
    selected_order = [
        "securenet_llama",
        "scamllm",
        "xgboost",
        "v3",
        "spamassassin",
        "rspamd",
    ]
    labels = {
        "securenet_llama": "Securenet",
        "scamllm": "ScamLLM",
        "xgboost": "XGBoost",
        "v3": "V3",
        "spamassassin": "SpamAssassin",
        "rspamd": "Rspamd",
    }
    data = performance[performance["detector"].isin(selected_order)].copy()
    data["detector"] = pd.Categorical(data["detector"], selected_order, ordered=True)
    data = data.sort_values("detector")

    fig, ax = plt.subplots(figsize=(10.8, 5.2))
    x = np.arange(len(data))
    width = 0.25
    bars = [
        ("Fuzzer", "Fuzzer_MCC", -width),
        ("UTA", "UTA_MCC", 0.0),
        ("MPG", "MPG_MCC", width),
    ]
    for label, col, offset in bars:
        ax.bar(x + offset, data[col], width=width, label=label, color=METHOD_COLORS[label])
    ax.axhline(0, color="#333333", linewidth=0.8)
    ax.set_ylabel("MCC", fontsize=15.5)
    ax.set_xticks(x)
    ax.set_xticklabels([labels[item] for item in data["detector"].astype(str)], rotation=20, ha="right", fontsize=14.5)
    ax.tick_params(axis="y", labelsize=14.5)
    ax.grid(axis="y", color="#d9d9d9", linewidth=0.7, alpha=0.8)
    ax.set_axisbelow(True)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.legend(frameon=False, fontsize=13.5, ncol=3, loc="upper left")
    fig.subplots_adjust(left=0.09, right=0.99, bottom=0.23, top=0.95)
    save_figure(fig, "fig_s6_detector_mcc")


def plot_method_shift_summary(shift_summary: pd.DataFrame) -> None:
    setup_plot_style()
    data = shift_summary.set_index("method").loc[["Fuzzer", "UTA", "MPG"]].reset_index()
    fig, ax = plt.subplots(figsize=(7.4, 4.8))
    bars = ax.bar(
        data["method"],
        data["delta_m"],
        color=[METHOD_COLORS[item] for item in data["method"]],
        width=0.58,
    )
    for bar, value in zip(bars, data["delta_m"], strict=True):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            value + 0.003,
            f"{value:.3f}",
            ha="center",
            va="bottom",
            fontsize=14.5,
        )
    ax.set_ylabel(r"Mean absolute pair shift $\Delta_m$", fontsize=15.5)
    ax.tick_params(axis="x", labelsize=15.0)
    ax.tick_params(axis="y", labelsize=14.5)
    ax.set_ylim(0, max(data["delta_m"]) * 1.25)
    ax.grid(axis="y", color="#d9d9d9", linewidth=0.7, alpha=0.8)
    ax.set_axisbelow(True)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    fig.subplots_adjust(left=0.17, right=0.98, bottom=0.16, top=0.94)
    save_figure(fig, "fig_s6_method_pair_shift")


def plot_top_changed_pairs(top_changed: pd.DataFrame) -> None:
    setup_plot_style()
    top_n = 6
    methods = ["Fuzzer", "UTA", "MPG"]
    fig, axes = plt.subplots(1, 3, figsize=(15.5, 5.7), sharex=False)
    for ax, method in zip(axes, methods, strict=True):
        data = (
            top_changed[top_changed["method"].eq(method)]
            .sort_values("abs_delta_LLM_FN_minus_HW_FN", ascending=False)
            .head(top_n)
            .sort_values("abs_delta_LLM_FN_minus_HW_FN", ascending=True)
        )
        colors = ["#003b4d" if value >= 0 else "#f5c64f" for value in data["delta_LLM_FN_minus_HW_FN"]]
        ax.barh(data["pair"], data["delta_LLM_FN_minus_HW_FN"], color=colors, height=0.62)
        ax.axvline(0, color="#333333", linewidth=0.8)
        ax.set_title(method, fontsize=16.5, pad=7)
        ax.set_xlabel(r"$\mu_{\mathrm{LLM-FN}} - \mu_{\mathrm{HW-FN}}$", fontsize=14.5)
        ax.tick_params(axis="x", labelsize=13.5)
        ax.tick_params(axis="y", labelsize=14.5)
        ax.grid(axis="x", color="#d9d9d9", linewidth=0.7, alpha=0.8)
        ax.set_axisbelow(True)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
    fig.subplots_adjust(left=0.07, right=0.995, bottom=0.18, top=0.90, wspace=0.35)
    save_figure(fig, "fig_s6_top_changed_pairs")


def plot_candidate_pairs(candidates: pd.DataFrame) -> None:
    setup_plot_style()
    methods = ["Fuzzer", "UTA", "MPG"]
    top_n = 5
    rows = []
    for method in methods:
        subset = candidates[candidates["method"].eq(method)].sort_values(
            "abs_delta_LLM_FN_minus_HW_FN", ascending=False
        )
        rows.append(subset.head(top_n))
    data = pd.concat(rows, ignore_index=True)

    fig, axes = plt.subplots(1, 3, figsize=(16.4, 6.0), sharex=False)
    for ax, method in zip(axes, methods, strict=True):
        subset = data[data["method"].eq(method)].copy()
        subset = subset.sort_values("abs_delta_LLM_FN_minus_HW_FN", ascending=True)
        y = np.arange(len(subset))
        height = 0.34
        ax.barh(y - height / 2, subset["LLM-FN_mean"], height=height, color="#f8d987", label="LLM-FN")
        ax.barh(y + height / 2, subset["LLM-TP_mean"], height=height, color="#003b4d", label="LLM-TP")
        ax.set_yticks(y)
        ax.set_yticklabels(subset["pair"], fontsize=14.5)
        ax.set_ylim(-0.6, top_n - 0.4)
        ax.set_title(method, fontsize=16.5, pad=7)
        ax.set_xlabel("Mean pair strength", fontsize=14.5)
        ax.tick_params(axis="x", labelsize=13.5)
        ax.grid(axis="x", color="#d9d9d9", linewidth=0.7, alpha=0.8)
        ax.set_axisbelow(True)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        xmax = max(float(subset["LLM-FN_mean"].max()), float(subset["LLM-TP_mean"].max()), 1e-6)
        ax.set_xlim(0, xmax * 1.22)
        for y_index, (_, row) in enumerate(subset.iterrows()):
            if row["q_LLM_FN_vs_LLM_TP"] < 0.001:
                label = "q<0.001"
            else:
                label = f"q={row['q_LLM_FN_vs_LLM_TP']:.3f}"
            ax.text(xmax * 1.19, y_index, label, ha="right", va="center", fontsize=12.5)
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, frameon=False, fontsize=13.5, loc="upper center", ncol=2, bbox_to_anchor=(0.52, 0.99))
    fig.subplots_adjust(left=0.07, right=0.995, bottom=0.18, top=0.84, wspace=0.35)
    save_figure(fig, "fig_s6_candidate_pairs_fn_vs_tp")


def make_visualizations(
    performance: pd.DataFrame,
    shift_summary: pd.DataFrame,
    top_changed: pd.DataFrame,
    candidates: pd.DataFrame,
) -> None:
    plot_detector_performance(performance)
    plot_method_shift_summary(shift_summary)
    plot_top_changed_pairs(top_changed)
    plot_candidate_pairs(candidates)


def main() -> None:
    ROOT.mkdir(parents=True, exist_ok=True)
    performance = summarize_detector_performance()
    means, stats, shift_summary = compute_pair_tables()
    top_changed, candidates = build_top_tables(means, stats)
    write_markdown(performance, means, stats, shift_summary, top_changed, candidates)
    make_visualizations(performance, shift_summary, top_changed, candidates)
    summary = {
        "outputs": {
            "detector_performance": "s6_detector_performance_mcc.csv",
            "pair_means": "s6_persuasion_pair_means.csv",
            "method_shift_summary": "s6_method_shift_summary.csv",
            "top_changed_pairs": "s6_top_changed_pairs_by_method.csv",
            "mannwhitney_tests": "s6_pair_mannwhitney_tests.csv",
            "candidate_pairs": "s6_candidate_pairs_for_final_figure.csv",
            "markdown_summary": "s6_rewriting_mechanism_summary.md",
            "detector_mcc_figure": "fig_s6_detector_mcc.png",
            "method_pair_shift_figure": "fig_s6_method_pair_shift.png",
            "top_changed_pairs_figure": "fig_s6_top_changed_pairs.png",
            "candidate_pairs_figure": "fig_s6_candidate_pairs_fn_vs_tp.png",
        }
    }
    (ROOT / "s6_rewriting_mechanism_outputs.json").write_text(
        json.dumps(summary, indent=2) + "\n",
        encoding="utf-8",
    )


if __name__ == "__main__":
    main()
