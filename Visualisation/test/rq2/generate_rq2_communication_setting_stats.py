#!/usr/bin/env python3
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import kruskal, mannwhitneyu


SCRIPT_DIR = Path(__file__).resolve().parent
VIS_DIR = SCRIPT_DIR.parents[1]
REPO_ROOT = VIS_DIR.parent

OVERVIEW_EMAIL = VIS_DIR / "test" / "overview" / "projected_points_mixed_overview.csv"
RQ2_SCORED = VIS_DIR / "RQ2" / "scored_inputs"
S7_DIR = REPO_ROOT / "Datasets" / "sublist" / "S7-Cross-channel Expansion"
FINAL_USED_7 = VIS_DIR / "test" / "fina_used" / "7"
FINAL_USED_8 = VIS_DIR / "test" / "fina_used" / "8"

OUTPUT_DIR = SCRIPT_DIR
OUT_SCALE = OUTPUT_DIR / "rq2_communication_setting_data_scale.csv"
OUT_MEANS = OUTPUT_DIR / "rq2_principle_mean_scores_by_setting.csv"
OUT_KW = OUTPUT_DIR / "rq2_principle_kruskal_fdr.csv"
OUT_PAIRWISE = OUTPUT_DIR / "rq2_principle_pairwise_mannwhitney_fdr.csv"
OUT_ROUNDS = OUTPUT_DIR / "rq2_multiturn_round_counts.csv"
OUT_SUMMARY = OUTPUT_DIR / "rq2_communication_setting_summary.md"
OUT_METADATA = OUTPUT_DIR / "rq2_communication_setting_metadata.json"

PRINCIPLES = [
    ("Authority", "principle_authority"),
    ("Liking", "principle_liking"),
    ("Reciprocity", "principle_reciprocity"),
    ("Social Proof", "principle_social_proof"),
    ("Scarcity", "principle_scarcity"),
    ("Commitment", "principle_commitment"),
]

SETTINGS = [
    "Email",
    "Single-turn Vishing",
    "Multi-turn Vishing",
]


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
    adjusted = np.clip(adjusted, 0.0, 1.0)
    q[order] = adjusted
    return q.tolist()


def format_p(value: float) -> str:
    if not np.isfinite(value):
        return ""
    if value < 0.001:
        return "<0.001"
    return f"{value:.3f}"


def load_email() -> pd.DataFrame:
    frame = pd.read_csv(OVERVIEW_EMAIL, low_memory=False)
    phishing_mask = frame["is_tp_phishing"].astype(bool) | frame["is_fn_phishing"].astype(bool)
    frame = frame[phishing_mask].copy()
    frame["setting"] = "Email"
    frame["dataset_source"] = frame.get("source_file", "").fillna("").astype(str)
    frame["generator_source"] = frame.get("source", "").fillna("").astype(str)
    return frame.reset_index(drop=True)


def load_vishing(turn_type: str) -> pd.DataFrame:
    specs = {
        "single": [
            ("HW", RQ2_SCORED / "single_hw_vishing_persuasion.csv", S7_DIR / "HW-Vishing-single.csv"),
            ("LLM", RQ2_SCORED / "single_llm_vishing_persuasion.csv", S7_DIR / "LLM-Vishing-Single.csv"),
        ],
        "multi": [
            ("HW", RQ2_SCORED / "multi_hw_vishing_persuasion.csv", S7_DIR / "HW-Vishing-Multi.csv"),
            ("LLM", RQ2_SCORED / "multi_llm_vishing_persuasion.csv", S7_DIR / "LLM-Vishing-Multi.csv"),
        ],
    }[turn_type]
    frames: list[pd.DataFrame] = []
    for source, scored_path, raw_path in specs:
        frame = pd.read_csv(scored_path, low_memory=False)
        frame["setting"] = "Single-turn Vishing" if turn_type == "single" else "Multi-turn Vishing"
        frame["generator_source"] = source
        frame["dataset_source"] = raw_path.name
        frames.append(frame)
    return pd.concat(frames, ignore_index=True)


def build_all_records() -> pd.DataFrame:
    frames = [load_email(), load_vishing("single"), load_vishing("multi")]
    common = ["setting", "generator_source", "dataset_source", "sentence_count_used", "token_count_used"]
    principle_columns = [column for _, column in PRINCIPLES]
    return pd.concat([frame[common + principle_columns].copy() for frame in frames], ignore_index=True)


def data_scale_table(records: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    labels = {
        "Email": "email samples",
        "Single-turn Vishing": "single-turn calls/messages",
        "Multi-turn Vishing": "full conversations scored for cross-setting comparison",
    }
    dataset_labels = {
        "Email": "projected_points_mixed_overview.csv phishing subset",
        "Single-turn Vishing": "HW-Vishing-single.csv; LLM-Vishing-Single.csv",
        "Multi-turn Vishing": "HW-Vishing-Multi.csv; LLM-Vishing-Multi.csv",
    }
    for setting in SETTINGS:
        subset = records[records["setting"] == setting].copy()
        generator_counts = subset["generator_source"].value_counts().sort_index()
        rows.append(
            {
                "Setting": setting,
                "# Samples / Conversations": int(len(subset)),
                "Unit label": labels[setting],
                "# Text Units Used": int(pd.to_numeric(subset["sentence_count_used"], errors="coerce").sum()),
                "# Tokens Used": int(pd.to_numeric(subset["token_count_used"], errors="coerce").sum()),
                "Generator(s)": "; ".join(f"{key}: {int(value)}" for key, value in generator_counts.items()),
                "Dataset": dataset_labels[setting],
            }
        )

    dialogue_summary_path = FINAL_USED_8 / "multiturn_dialogue_round_summary.csv"
    if dialogue_summary_path.exists():
        dialogue_summary = pd.read_csv(dialogue_summary_path)
        total_dialogues = int(dialogue_summary["dialogue_count"].sum())
        total_malicious_rounds = int(dialogue_summary["malicious_turn_rows"].sum())
        rows.append(
            {
                "Setting": "Multi-turn Vishing (round-level Fig. 2 source)",
                "# Samples / Conversations": total_dialogues,
                "Unit label": "dialogues used for round-level malicious-turn extraction",
                "# Text Units Used": total_malicious_rounds,
                "# Tokens Used": "",
                "Generator(s)": "; ".join(
                    f"{row['dataset']}: {int(row['dialogue_count'])}"
                    for _, row in dialogue_summary.iterrows()
                ),
                "Dataset": "HW-Vishing-Multi-ScamBaiter.csv; LLM-Vishing-Multi-BothBosu.csv",
            }
        )
    return pd.DataFrame(rows)


def principle_mean_table(records: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for principle, column in PRINCIPLES:
        values = {
            setting: pd.to_numeric(
                records.loc[records["setting"] == setting, column],
                errors="coerce",
            ).dropna()
            for setting in SETTINGS
        }
        means = {setting: float(series.mean()) for setting, series in values.items()}
        rows.append(
            {
                "Principle": principle,
                "Email": means["Email"],
                "Single-turn Vishing": means["Single-turn Vishing"],
                "Multi-turn Vishing": means["Multi-turn Vishing"],
                "Max Difference": max(means.values()) - min(means.values()),
            }
        )
    return pd.DataFrame(rows)


def statistical_tables(records: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    kw_rows: list[dict[str, object]] = []
    pairwise_rows: list[dict[str, object]] = []
    comparisons = [
        ("Email", "Single-turn Vishing"),
        ("Email", "Multi-turn Vishing"),
        ("Single-turn Vishing", "Multi-turn Vishing"),
    ]
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
            n_left = len(left_values)
            n_right = len(right_values)
            u_stat = float(result.statistic)
            auc_left_greater = u_stat / float(n_left * n_right)
            rank_biserial_right_minus_left = 1.0 - 2.0 * auc_left_greater
            pairwise_rows.append(
                {
                    "Principle": principle,
                    "Comparison": f"{left} vs {right}",
                    "Left group": left,
                    "Right group": right,
                    "Left mean": float(left_values.mean()),
                    "Right mean": float(right_values.mean()),
                    "Difference (right-left)": float(right_values.mean() - left_values.mean()),
                    "U statistic": u_stat,
                    "p": float(result.pvalue),
                    "p formatted": format_p(float(result.pvalue)),
                    "Effect Size (rank-biserial, right-left)": float(rank_biserial_right_minus_left),
                    "n left": int(n_left),
                    "n right": int(n_right),
                }
            )

    kw = pd.DataFrame(kw_rows)
    kw["q (FDR)"] = bh_fdr(kw["p"].tolist())
    kw["q formatted"] = kw["q (FDR)"].apply(format_p)

    pairwise = pd.DataFrame(pairwise_rows)
    pairwise["q-value"] = bh_fdr(pairwise["p"].tolist())
    pairwise["q formatted"] = pairwise["q-value"].apply(format_p)
    return kw, pairwise


def round_counts_table() -> pd.DataFrame:
    rows: list[pd.DataFrame] = []
    round_counts_path = FINAL_USED_8 / "multiturn_round_counts.csv"
    if round_counts_path.exists():
        round_counts = pd.read_csv(round_counts_path)
        rows.append(round_counts.assign(source_view="all extracted malicious turns"))

    llm_r6_path = FINAL_USED_7 / "llm_multiturn_principle_turn_mean_trends_r6.csv"
    if llm_r6_path.exists():
        llm_r6 = pd.read_csv(llm_r6_path)
        counts = (
            llm_r6[["round_number", "sample_count"]]
            .drop_duplicates()
            .rename(columns={"sample_count": "turn_count"})
        )
        counts["dataset"] = "LLM"
        counts["source_view"] = "Fig. 2 R1-R6 cohort (dialogues with at least 6 malicious turns)"
        rows.append(counts[["dataset", "round_number", "turn_count", "source_view"]])

    if not rows:
        return pd.DataFrame(columns=["dataset", "round_number", "turn_count", "source_view"])
    return pd.concat(rows, ignore_index=True).sort_values(["source_view", "dataset", "round_number"])


def write_markdown(
    scale: pd.DataFrame,
    means: pd.DataFrame,
    kw: pd.DataFrame,
    pairwise: pd.DataFrame,
    rounds: pd.DataFrame,
) -> None:
    md_lines = [
        "# RQ2 Communication-Setting Statistics",
        "",
        "Scope: Email vs. single-turn Vishing vs. multi-turn Vishing.",
        "",
        "Principle values are mean WVAE persuasion scores from the six `principle_*` columns.",
        "Pairwise differences are `right group mean - left group mean`; positive values mean the second group is larger.",
        "",
        "## A. Data Scale",
        "",
        scale.to_markdown(index=False),
        "",
        "## B. Mean Principle Scores",
        "",
        means.to_markdown(index=False, floatfmt=".4f"),
        "",
        "## C1. Kruskal-Wallis Tests",
        "",
        kw[["Principle", "H statistic", "p formatted", "q formatted"]].to_markdown(index=False, floatfmt=".4f"),
        "",
        "## C2. Pairwise Mann-Whitney Tests",
        "",
        pairwise[
            [
                "Principle",
                "Comparison",
                "Difference (right-left)",
                "q formatted",
                "Effect Size (rank-biserial, right-left)",
            ]
        ].to_markdown(index=False, floatfmt=".4f"),
        "",
        "## D. Multi-Turn Round Counts",
        "",
        rounds.to_markdown(index=False),
        "",
    ]
    OUT_SUMMARY.write_text("\n".join(md_lines), encoding="utf-8")


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    records = build_all_records()
    scale = data_scale_table(records)
    means = principle_mean_table(records)
    kw, pairwise = statistical_tables(records)
    rounds = round_counts_table()

    scale.to_csv(OUT_SCALE, index=False)
    means.to_csv(OUT_MEANS, index=False)
    kw.to_csv(OUT_KW, index=False)
    pairwise.to_csv(OUT_PAIRWISE, index=False)
    rounds.to_csv(OUT_ROUNDS, index=False)
    write_markdown(scale, means, kw, pairwise, rounds)

    metadata = {
        "email_input": str(OVERVIEW_EMAIL),
        "vishing_scored_dir": str(RQ2_SCORED),
        "principle_columns": {name: column for name, column in PRINCIPLES},
        "setting_rows": {setting: int((records["setting"] == setting).sum()) for setting in SETTINGS},
        "outputs": {
            "scale": OUT_SCALE.name,
            "means": OUT_MEANS.name,
            "kruskal": OUT_KW.name,
            "pairwise": OUT_PAIRWISE.name,
            "round_counts": OUT_ROUNDS.name,
            "summary": OUT_SUMMARY.name,
        },
    }
    OUT_METADATA.write_text(json.dumps(metadata, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
