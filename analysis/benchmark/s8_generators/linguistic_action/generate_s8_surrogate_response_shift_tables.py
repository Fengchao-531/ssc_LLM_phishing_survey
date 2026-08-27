#!/usr/bin/env python3
from __future__ import annotations

import json
from itertools import combinations
from pathlib import Path

import pandas as pd
from scipy.stats import wasserstein_distance


ROOT = Path(__file__).resolve().parent
STAGES_ROOT = ROOT.parent / "stages"

SCORE_SUMMARY_CSV = ROOT / "s8_surrogate_response_generator_summary.csv"
MEAN_SHIFT_CSV = ROOT / "s8_surrogate_response_mean_shift.csv"
WASSERSTEIN_CSV = ROOT / "s8_surrogate_response_pairwise_wasserstein.csv"
WASSERSTEIN_SUMMARY_CSV = ROOT / "s8_surrogate_response_wasserstein_summary.csv"
MARKDOWN_SUMMARY = ROOT / "s8_surrogate_response_shift_summary.md"
SUMMARY_JSON = ROOT / "s8_surrogate_response_shift_outputs.json"

GENERATORS = [
    "S8-claude",
    "S8-gpt",
    "S8-gemini",
    "S8-llama",
    "S8-ministral",
    "S8-deepseek",
]

GENERATOR_LABELS = {
    "S8-claude": "Claude",
    "S8-gpt": "GPT",
    "S8-gemini": "Gemini",
    "S8-llama": "Llama",
    "S8-ministral": "Ministral",
    "S8-deepseek": "DeepSeek",
}


def load_scores() -> pd.DataFrame:
    frames = []
    for generator in GENERATORS:
        path = STAGES_ROOT / generator / "projected_points.csv"
        frame = pd.read_csv(
            path,
            usecols=["source", "stage", "label", "surrogate_score"],
            low_memory=False,
        )
        frame["source"] = frame["source"].fillna("").astype(str)
        frame["stage"] = frame["stage"].fillna("").astype(str)
        frame["label"] = pd.to_numeric(frame["label"], errors="coerce").fillna(0.0)
        frame["surrogate_score"] = pd.to_numeric(frame["surrogate_score"], errors="coerce")
        selected = frame[
            frame["source"].eq("LLM")
            & frame["stage"].eq(generator)
            & frame["label"].eq(1.0)
            & frame["surrogate_score"].notna()
        ].copy()
        frames.append(selected)
    return pd.concat(frames, ignore_index=True)


def iqr(values: pd.Series) -> float:
    return float(values.quantile(0.75) - values.quantile(0.25))


def build_summary(frame: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for generator in GENERATORS:
        values = frame.loc[frame["stage"].eq(generator), "surrogate_score"].astype(float)
        rows.append(
            {
                "surrogate_space": "global_response_space",
                "generator": generator,
                "generator_label": GENERATOR_LABELS[generator],
                "n": int(len(values)),
                "mean_score": float(values.mean()),
                "median_score": float(values.median()),
                "iqr": iqr(values),
                "q1": float(values.quantile(0.25)),
                "q3": float(values.quantile(0.75)),
                "min_score": float(values.min()),
                "max_score": float(values.max()),
            }
        )
    return pd.DataFrame(rows)


def build_mean_shift(summary: pd.DataFrame) -> pd.DataFrame:
    min_row = summary.loc[summary["mean_score"].idxmin()]
    max_row = summary.loc[summary["mean_score"].idxmax()]
    return pd.DataFrame(
        [
            {
                "surrogate_space": "global_response_space",
                "min_generator": str(min_row["generator"]),
                "min_generator_label": str(min_row["generator_label"]),
                "min_mean_score": float(min_row["mean_score"]),
                "max_generator": str(max_row["generator"]),
                "max_generator_label": str(max_row["generator_label"]),
                "max_mean_score": float(max_row["mean_score"]),
                "R_max_minus_min_mean": float(max_row["mean_score"] - min_row["mean_score"]),
                "usable_generators": int(summary["generator"].nunique()),
            }
        ]
    )


def build_wasserstein(frame: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    values_by_generator = {
        generator: frame.loc[frame["stage"].eq(generator), "surrogate_score"].astype(float).to_numpy()
        for generator in GENERATORS
    }
    rows = []
    for left, right in combinations(GENERATORS, 2):
        distance = float(wasserstein_distance(values_by_generator[left], values_by_generator[right]))
        rows.append(
            {
                "surrogate_space": "global_response_space",
                "generator_a": left,
                "generator_a_label": GENERATOR_LABELS[left],
                "generator_b": right,
                "generator_b_label": GENERATOR_LABELS[right],
                "n_a": int(len(values_by_generator[left])),
                "n_b": int(len(values_by_generator[right])),
                "wasserstein_distance": distance,
            }
        )
    pairwise = pd.DataFrame(rows)
    max_row = pairwise.loc[pairwise["wasserstein_distance"].idxmax()]
    summary = pd.DataFrame(
        [
            {
                "surrogate_space": "global_response_space",
                "pair_count": int(len(pairwise)),
                "max_wasserstein": float(pairwise["wasserstein_distance"].max()),
                "max_pair": f"{max_row['generator_a_label']} vs {max_row['generator_b_label']}",
                "median_wasserstein": float(pairwise["wasserstein_distance"].median()),
                "mean_wasserstein": float(pairwise["wasserstein_distance"].mean()),
            }
        ]
    )
    return pairwise, summary


def write_markdown(summary: pd.DataFrame, mean_shift: pd.DataFrame, wasserstein_summary: pd.DataFrame) -> None:
    shift = mean_shift.iloc[0]
    w1 = wasserstein_summary.iloc[0]
    lines = [
        "# S8 Surrogate Response Shift Summary",
        "",
        "This table uses the continuous `surrogate_score` stored in each S8 stage's `projected_points.csv`.",
        "",
        "## Generator score summaries",
        "",
        "| LLM | n | Mean | Median | IQR |",
        "|---|---:|---:|---:|---:|",
    ]
    for row in summary.itertuples():
        lines.append(
            f"| {row.generator_label} | {row.n} | {row.mean_score:.4f} | {row.median_score:.4f} | {row.iqr:.4f} |"
        )
    lines.extend(
        [
            "",
            "## Cross-generator shift",
            "",
            f"- Mean-score range: `{shift.R_max_minus_min_mean:.4f}` ({shift.min_generator_label} `{shift.min_mean_score:.4f}` to {shift.max_generator_label} `{shift.max_mean_score:.4f}`).",
            f"- Median pairwise 1-Wasserstein distance: `{w1.median_wasserstein:.4f}`.",
            f"- Maximum pairwise 1-Wasserstein distance: `{w1.max_wasserstein:.4f}` ({w1.max_pair}).",
            "",
            "Suggested wording:",
            "",
            f"> Across LLM generators, the mean surrogate phishing score varies by {shift.R_max_minus_min_mean:.3f} in the global response space; the median pairwise Wasserstein distance is {w1.median_wasserstein:.3f}.",
            "",
        ]
    )
    MARKDOWN_SUMMARY.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    ROOT.mkdir(parents=True, exist_ok=True)
    frame = load_scores()
    summary = build_summary(frame)
    mean_shift = build_mean_shift(summary)
    pairwise_wasserstein, wasserstein_summary = build_wasserstein(frame)

    summary.to_csv(SCORE_SUMMARY_CSV, index=False)
    mean_shift.to_csv(MEAN_SHIFT_CSV, index=False)
    pairwise_wasserstein.to_csv(WASSERSTEIN_CSV, index=False)
    wasserstein_summary.to_csv(WASSERSTEIN_SUMMARY_CSV, index=False)
    write_markdown(summary, mean_shift, wasserstein_summary)
    SUMMARY_JSON.write_text(
        json.dumps(
            {
                "input_root": str(STAGES_ROOT),
                "filter": "source == LLM, stage == generator, label == phishing",
                "score_column": "surrogate_score",
                "outputs": {
                    "score_summary_csv": SCORE_SUMMARY_CSV.name,
                    "mean_shift_csv": MEAN_SHIFT_CSV.name,
                    "wasserstein_pairwise_csv": WASSERSTEIN_CSV.name,
                    "wasserstein_summary_csv": WASSERSTEIN_SUMMARY_CSV.name,
                    "markdown_summary": MARKDOWN_SUMMARY.name,
                },
            },
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )


if __name__ == "__main__":
    main()
