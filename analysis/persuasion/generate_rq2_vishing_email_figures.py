#!/usr/bin/env python3
"""Generate RQ2 persuasion figures for vishing vs. email phishing."""

from __future__ import annotations

import argparse
import json
import math
import re
import subprocess
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.lines import Line2D
from matplotlib.colors import LinearSegmentedColormap, TwoSlopeNorm
from scipy.stats import gaussian_kde
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import PolynomialFeatures, StandardScaler


SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parent
RQ2_OUTPUT_DIR = SCRIPT_DIR / "RQ2"
S7_DIR = REPO_ROOT / "Datasets" / "sublist" / "S7-Cross-channel Expansion"
WVAE_CODE_DIR = SCRIPT_DIR / "persuasion_strategy_wvae" / "code"
WVAE_MODEL_PATH = (
    SCRIPT_DIR / "persuasion_strategy_wvae" / "output" / "cialdini_wvae_full_v1" / "model.pkl"
)
FULL_RESULTS_DIR = SCRIPT_DIR / "persuasion_strategy_wvae" / "output" / "full_inference_results"
OVERVIEW_MIXED_INPUT = SCRIPT_DIR / "test" / "overview" / "projected_points_mixed_overview.csv"

PRINCIPLES = [
    ("Authority", "principle_authority"),
    ("Reciprocity", "principle_reciprocity"),
    ("Commitment", "principle_commitment"),
    ("Scarcity", "principle_scarcity"),
    ("Social Proof", "principle_social_proof"),
    ("Liking", "principle_liking"),
]
PRINCIPLE_LABELS = [label for label, _ in PRINCIPLES]
PRINCIPLE_COLUMNS = [column for _, column in PRINCIPLES]
SHARE_COLUMNS = [f"{column}_share" for column in PRINCIPLE_COLUMNS]
PRINCIPLE_COLUMN_BY_LABEL = {label: column for label, column in PRINCIPLES}

OVERVIEW_PRINCIPLE_LABELS = [
    "Authority",
    "Liking",
    "Reciprocity",
    "Social Proof",
    "Scarcity",
    "Commitment",
]
OVERVIEW_SHARE_COLUMNS = [
    f"{PRINCIPLE_COLUMN_BY_LABEL[label]}_share" for label in OVERVIEW_PRINCIPLE_LABELS
]
OVERVIEW_RAW_COLUMNS = [PRINCIPLE_COLUMN_BY_LABEL[label] for label in OVERVIEW_PRINCIPLE_LABELS]

VISHING_INPUTS = {
    "single": [
        ("HW", S7_DIR / "HW-Vishing-single.csv"),
        ("LLM", S7_DIR / "LLM-Vishing-Single.csv"),
    ],
    "multi": [
        ("HW", S7_DIR / "HW-Vishing-Multi.csv"),
        ("LLM", S7_DIR / "LLM-Vishing-Multi.csv"),
    ],
}

EMAIL_FILES = [
    "HW_S1_persuasion.csv",
    "HW_S2_persuasion.csv",
    "HW_S4_persuasion.csv",
    "HW_S5_persuasion.csv",
    "HW_S6_persuasion.csv",
    "HW_S8_persuasion.csv",
    "LLM_S1_persuasion.csv",
    "LLM_S2_persuasion.csv",
    "LLM_S4_persuasion.csv",
    "LLM_S5_persuasion.csv",
    "LLM_S6-MPG_persuasion.csv",
    "LLM_S6-UTA_persuasion.csv",
    "LLM_S6-fuzzer_persuasion.csv",
    "LLM_S8-claude_persuasion.csv",
    "LLM_S8-deepseek_persuasion.csv",
    "LLM_S8-gemini_persuasion.csv",
    "LLM_S8-gpt_persuasion.csv",
    "LLM_S8-llama_persuasion.csv",
    "LLM_S8-ministral_persuasion.csv",
]

GROUP_COLORS = {
    "ScamLLM Email": "#1f77b4",
    "Vishing Single-turn": "#d95f02",
    "Vishing Multi-turn": "#b22222",
}
GROUP_MARKERS = {
    "ScamLLM Email": "o",
    "Vishing Single-turn": "^",
    "Vishing Multi-turn": "s",
}
SOURCE_DISPLAY = {
    "HW": "HW phishing",
    "LLM": "LLM phishing",
}

TOKEN_RE = re.compile(r"[A-Za-z0-9_']+")
URGENCY_RE = re.compile(
    r"\b(urgent|urgently|immediately|asap|now|today|right away|act fast|hurry|"
    r"last chance|limited[- ]time|deadline|expires?|expiring|soon)\b",
    re.IGNORECASE,
)
THREAT_RE = re.compile(
    r"\b(blocked|freeze|frozen|suspend|suspended|warning|penalty|legal|risk|"
    r"consequences?|arrest|deactivated|terminate|termination|lose)\b",
    re.IGNORECASE,
)
ACTION_RE = re.compile(
    r"\b(click|call|reply|respond|visit|open|share|send|confirm|verify|transfer|"
    r"pay|download|login|log in|sign in|provide|press|submit)\b",
    re.IGNORECASE,
)
POLITENESS_RE = re.compile(
    r"\b(please|kindly|thank you|thanks|dear|sincerely|regards|hello|hi|namaste)\b",
    re.IGNORECASE,
)
ROLE_RE = re.compile(
    r"\b(bank|sbi|hdfc|manager|officer|agent|advisor|team|department|security|"
    r"support|government|police|customs|branch|service|representative|official)\b",
    re.IGNORECASE,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Generate RQ2 persuasion heatmaps and contour plots for "
            "vishing single-turn, vishing multi-turn, and ScamLLM email phishing."
        )
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=RQ2_OUTPUT_DIR,
        help="Directory for figures and summary tables.",
    )
    parser.add_argument(
        "--device",
        default="auto",
        help="Device passed to the WVAE scorer when S7 vishing scores are missing.",
    )
    parser.add_argument(
        "--density-sample-size",
        type=int,
        default=20000,
        help="Maximum rows per group used for KDE contour estimation.",
    )
    parser.add_argument(
        "--force-rescore",
        action="store_true",
        help="Re-run WVAE scoring for the S7 vishing CSVs even if outputs already exist.",
    )
    return parser.parse_args()


def ensure_scored_vishing(input_csv: Path, output_csv: Path, device: str, force_rescore: bool) -> None:
    metadata_json = output_csv.with_suffix(".metadata.json")
    if output_csv.exists() and metadata_json.exists() and not force_rescore:
        return
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    command = [
        sys.executable,
        str(WVAE_CODE_DIR / "score_email_csv.py"),
        "--input-csv",
        str(input_csv),
        "--output-csv",
        str(output_csv),
        "--model-path",
        str(WVAE_MODEL_PATH),
        "--body-column",
        "Body",
        "--device",
        device,
        "--metadata-json",
        str(metadata_json),
    ]
    subprocess.run(command, cwd=WVAE_CODE_DIR, check=True)


def load_vishing_turn(turn_type: str, output_dir: Path, device: str, force_rescore: bool) -> pd.DataFrame:
    frames: list[pd.DataFrame] = []
    scored_dir = output_dir / "scored_inputs"
    for source_name, input_csv in VISHING_INPUTS[turn_type]:
        score_name = f"{turn_type}_{source_name.lower()}_vishing_persuasion.csv"
        score_path = scored_dir / score_name
        ensure_scored_vishing(input_csv, score_path, device=device, force_rescore=force_rescore)
        frame = pd.read_csv(score_path)
        frame["group_name"] = f"Vishing {'Single-turn' if turn_type == 'single' else 'Multi-turn'}"
        frame["turn_type"] = turn_type
        frame["dataset_family"] = "vishing"
        frame["source_name"] = source_name
        frame["source_type"] = source_name
        frame["raw_text"] = frame.get("Body", "").fillna("")
        frame["sample_id"] = (
            frame["turn_type"]
            + "::"
            + frame["source_name"]
            + "::"
            + frame.index.astype(str)
        )
        frames.append(frame)
    return pd.concat(frames, ignore_index=True)


def load_email_group() -> pd.DataFrame:
    if not OVERVIEW_MIXED_INPUT.exists():
        raise FileNotFoundError(f"Overview mixed phishing input not found: {OVERVIEW_MIXED_INPUT}")
    frame = pd.read_csv(OVERVIEW_MIXED_INPUT)
    phishing_mask = frame["is_tp_phishing"].astype(bool) | frame["is_fn_phishing"].astype(bool)
    frame = frame[phishing_mask].copy()
    if frame.empty:
        raise FileNotFoundError("No phishing rows were loaded from projected_points_mixed_overview.csv")
    subject_series = frame.get("subject", pd.Series("", index=frame.index, dtype="object")).fillna("").astype(str)
    body_series = frame.get("body", pd.Series("", index=frame.index, dtype="object")).fillna("").astype(str)
    frame["group_name"] = "ScamLLM Email"
    frame["turn_type"] = "email"
    frame["dataset_family"] = "email"
    frame["source_name"] = frame.get("stage", "overview_email").fillna("overview_email").astype(str)
    frame["source_type"] = frame["source"].astype(str)
    frame["raw_text"] = (subject_series + "\n" + body_series).str.strip()
    frame["sample_id"] = (
        frame["source_type"].astype(str)
        + "::"
        + frame["source_name"].astype(str)
        + "::"
        + frame.index.astype(str)
    )
    return frame.reset_index(drop=True)


def normalize_principle_shares(frame: pd.DataFrame) -> pd.DataFrame:
    principle_values = (
        frame[PRINCIPLE_COLUMNS].apply(pd.to_numeric, errors="coerce").fillna(0.0).clip(lower=0.0)
    )
    totals = principle_values.sum(axis=1).replace(0.0, np.nan)
    shares = principle_values.div(totals, axis=0).fillna(0.0)
    shares.columns = SHARE_COLUMNS
    return pd.concat([frame.reset_index(drop=True), shares.reset_index(drop=True)], axis=1)


def token_count(text: str) -> int:
    return len(TOKEN_RE.findall(text.lower()))


def per_100_tokens(pattern: re.Pattern[str], text: str) -> float:
    tokens = max(token_count(text), 1)
    return 100.0 * len(pattern.findall(text)) / float(tokens)


def add_composite_features(frame: pd.DataFrame) -> pd.DataFrame:
    working = frame.copy()
    texts = working["raw_text"].fillna("").astype(str)
    working["urgency_rate"] = texts.apply(lambda value: per_100_tokens(URGENCY_RE, value))
    working["threat_rate"] = texts.apply(lambda value: per_100_tokens(THREAT_RE, value))
    working["action_request_rate"] = texts.apply(lambda value: per_100_tokens(ACTION_RE, value))
    working["politeness_rate"] = texts.apply(lambda value: per_100_tokens(POLITENESS_RE, value))
    working["role_credibility_rate"] = texts.apply(lambda value: per_100_tokens(ROLE_RE, value))

    rate_columns = [
        "urgency_rate",
        "threat_rate",
        "action_request_rate",
        "politeness_rate",
        "role_credibility_rate",
    ]
    rate_scales: dict[str, float] = {}
    for column in rate_columns:
        scale = float(working[column].quantile(0.95))
        if not math.isfinite(scale) or scale <= 0.0:
            scale = 1.0
        rate_scales[column] = scale
        working[f"{column}_scaled"] = (working[column] / scale).clip(lower=0.0, upper=1.0)

    working["action_oriented_persuasion"] = working[
        [
            "principle_reciprocity_share",
            "principle_scarcity_share",
            "principle_commitment_share",
            "urgency_rate_scaled",
            "threat_rate_scaled",
            "action_request_rate_scaled",
        ]
    ].mean(axis=1)
    working["trust_legitimacy_persuasion"] = working[
        [
            "principle_authority_share",
            "principle_liking_share",
            "principle_social_proof_share",
            "politeness_rate_scaled",
            "role_credibility_rate_scaled",
        ]
    ].mean(axis=1)
    working.attrs["rate_scales"] = rate_scales
    return working


def pair_strength_matrix(frame: pd.DataFrame) -> np.ndarray:
    values = frame[SHARE_COLUMNS].to_numpy(dtype=float)
    if values.size == 0:
        return np.full((len(SHARE_COLUMNS), len(SHARE_COLUMNS)), np.nan, dtype=float)
    return np.einsum("ni,nj->ij", values, values) / float(values.shape[0])


def pair_strength_matrix_for_columns(frame: pd.DataFrame, columns: list[str]) -> np.ndarray:
    values = frame[columns].to_numpy(dtype=float)
    if values.size == 0:
        return np.full((len(columns), len(columns)), np.nan, dtype=float)
    return np.einsum("ni,nj->ij", values, values) / float(values.shape[0])


def overview_style_matrix(frame: pd.DataFrame, columns: list[str]) -> np.ndarray:
    principle_frame = frame[columns].apply(pd.to_numeric, errors="coerce").dropna(how="all")
    if principle_frame.empty:
        return np.full((len(columns), len(columns)), np.nan, dtype=float)
    values = principle_frame.to_numpy(dtype=float)
    matrix = (values.T @ values) / max(1, values.shape[0])
    diagonal = np.nanmean(values, axis=0)
    np.fill_diagonal(matrix, diagonal)
    return matrix


def mean_share_vector(frame: pd.DataFrame) -> np.ndarray:
    return frame[SHARE_COLUMNS].mean(axis=0).to_numpy(dtype=float)


def flatten_pairwise_stats(
    comparison_name: str,
    left_name: str,
    right_name: str,
    left_matrix: np.ndarray,
    right_matrix: np.ndarray,
) -> list[dict[str, float | str]]:
    rows: list[dict[str, float | str]] = []
    delta = left_matrix - right_matrix
    for row_index, principle_i in enumerate(PRINCIPLE_LABELS):
        for col_index, principle_j in enumerate(PRINCIPLE_LABELS):
            rows.append(
                {
                    "comparison": comparison_name,
                    "left_group": left_name,
                    "right_group": right_name,
                    "principle_i": principle_i,
                    "principle_j": principle_j,
                    "left_pair_strength": float(left_matrix[row_index, col_index]),
                    "right_pair_strength": float(right_matrix[row_index, col_index]),
                    "delta_pair_strength": float(delta[row_index, col_index]),
                }
            )
    return rows


def summarize_top_differences(pairwise_rows: pd.DataFrame, principle_rows: pd.DataFrame) -> dict[str, object]:
    summary: dict[str, object] = {}
    for comparison_name, subset in pairwise_rows.groupby("comparison"):
        off_diagonal = subset[subset["principle_i"] != subset["principle_j"]]
        strongest_vishing = off_diagonal.nlargest(5, "delta_pair_strength")[
            ["principle_i", "principle_j", "delta_pair_strength"]
        ]
        strongest_email = off_diagonal.nsmallest(5, "delta_pair_strength")[
            ["principle_i", "principle_j", "delta_pair_strength"]
        ]
        principle_subset = principle_rows[principle_rows["comparison"] == comparison_name]
        summary[comparison_name] = {
            "top_vishing_pair_deltas": strongest_vishing.to_dict(orient="records"),
            "top_email_pair_deltas": strongest_email.to_dict(orient="records"),
            "top_vishing_principles": principle_subset.nlargest(3, "delta_mean_share")[
                ["principle", "delta_mean_share"]
            ].to_dict(orient="records"),
            "top_email_principles": principle_subset.nsmallest(3, "delta_mean_share")[
                ["principle", "delta_mean_share"]
            ].to_dict(orient="records"),
        }
    return summary


def style_heatmap_axis(ax: plt.Axes, matrix: np.ndarray, title: str, limit: float) -> None:
    image = ax.imshow(matrix, cmap="coolwarm", vmin=-limit, vmax=limit)
    ax.set_title(title, fontsize=11, pad=8)
    ax.set_xticks(range(len(PRINCIPLE_LABELS)))
    ax.set_yticks(range(len(PRINCIPLE_LABELS)))
    ax.set_xticklabels(PRINCIPLE_LABELS, rotation=45, ha="right")
    ax.set_yticklabels(PRINCIPLE_LABELS)
    for row_index in range(matrix.shape[0]):
        for col_index in range(matrix.shape[1]):
            value = matrix[row_index, col_index]
            if not math.isfinite(value):
                continue
            text_color = "white" if abs(value) > limit * 0.55 else "black"
            ax.text(
                col_index,
                row_index,
                f"{value:.3f}",
                ha="center",
                va="center",
                fontsize=7,
                color=text_color,
            )
    ax.set_xlim(-0.5, len(PRINCIPLE_LABELS) - 0.5)
    ax.set_ylim(len(PRINCIPLE_LABELS) - 0.5, -0.5)
    ax.grid(False)
    ax.set_aspect("equal")
    return image


def style_positive_heatmap_axis(ax: plt.Axes, matrix: np.ndarray, title: str, limit: float) -> None:
    image = ax.imshow(matrix, cmap="YlOrRd", vmin=0.0, vmax=limit)
    ax.set_title(title, fontsize=11, pad=8)
    ax.set_xticks(range(len(PRINCIPLE_LABELS)))
    ax.set_yticks(range(len(PRINCIPLE_LABELS)))
    ax.set_xticklabels(PRINCIPLE_LABELS, rotation=45, ha="right")
    ax.set_yticklabels(PRINCIPLE_LABELS)
    for row_index in range(matrix.shape[0]):
        for col_index in range(matrix.shape[1]):
            value = matrix[row_index, col_index]
            if not math.isfinite(value):
                continue
            text_color = "white" if value > limit * 0.55 else "black"
            ax.text(
                col_index,
                row_index,
                f"{value:.3f}",
                ha="center",
                va="center",
                fontsize=7,
                color=text_color,
            )
    ax.set_xlim(-0.5, len(PRINCIPLE_LABELS) - 0.5)
    ax.set_ylim(len(PRINCIPLE_LABELS) - 0.5, -0.5)
    ax.grid(False)
    ax.set_aspect("equal")
    return image


def overview_difference_cmap() -> LinearSegmentedColormap:
    # Keep the blue-green positive side visually aligned with test/overview's GnBu heatmaps.
    return LinearSegmentedColormap.from_list(
        "overview_difference",
        ["#c47a3b", "#f7f7f7", "#d8eef0", "#7bccc4", "#2b8cbe", "#084081"],
        N=256,
    )


def style_overview_difference_heatmap_axis(
    ax: plt.Axes,
    matrix: np.ndarray,
    title: str,
    limit: float,
) -> None:
    norm = TwoSlopeNorm(vmin=-limit, vcenter=0.0, vmax=limit)
    image = ax.imshow(matrix, cmap=overview_difference_cmap(), norm=norm, aspect="equal")
    ax.set_title(title, fontsize=11, pad=8)
    ax.set_xticks(range(len(OVERVIEW_PRINCIPLE_LABELS)))
    ax.set_xticklabels(OVERVIEW_PRINCIPLE_LABELS, rotation=45, ha="right", fontsize=9)
    ax.set_yticks(range(len(OVERVIEW_PRINCIPLE_LABELS)))
    ax.set_yticklabels(OVERVIEW_PRINCIPLE_LABELS, fontsize=9)
    for row_index in range(matrix.shape[0]):
        for col_index in range(matrix.shape[1]):
            value = matrix[row_index, col_index]
            if not math.isfinite(value):
                continue
            text_color = "white" if abs(value) > limit * 0.55 else "black"
            ax.text(
                col_index,
                row_index,
                f"{value:.3f}",
                ha="center",
                va="center",
                fontsize=7.2,
                color=text_color,
            )
    ax.set_xlim(-0.5, len(OVERVIEW_PRINCIPLE_LABELS) - 0.5)
    ax.set_ylim(len(OVERVIEW_PRINCIPLE_LABELS) - 0.5, -0.5)
    ax.grid(False)
    ax.set_aspect("equal")
    return image


def plot_source_split_difference_heatmaps(
    left_frame: pd.DataFrame,
    right_frame: pd.DataFrame,
    left_label: str,
    right_label: str,
    output_path: Path,
) -> dict[str, dict[str, float]]:
    difference_matrices: dict[str, np.ndarray] = {}
    summary: dict[str, dict[str, float]] = {}
    for source_type in ["HW", "LLM"]:
        left_subset = left_frame[left_frame["source_type"] == source_type].copy()
        right_subset = right_frame[right_frame["source_type"] == source_type].copy()
        left_matrix = overview_style_matrix(left_subset, OVERVIEW_RAW_COLUMNS)
        right_matrix = overview_style_matrix(right_subset, OVERVIEW_RAW_COLUMNS)
        difference_matrices[source_type] = left_matrix - right_matrix
        summary[source_type] = {
            "left_rows": int(len(left_subset)),
            "right_rows": int(len(right_subset)),
        }

    limit = max(float(np.nanmax(np.abs(matrix))) for matrix in difference_matrices.values())
    limit = max(limit, 1e-6)

    figure = plt.figure(figsize=(11.8, 5.8))
    grid = figure.add_gridspec(1, 3, width_ratios=[1.0, 1.0, 0.08], wspace=0.05)
    heatmap_axes = [figure.add_subplot(grid[0, 0]), figure.add_subplot(grid[0, 1])]
    colorbar_axis = figure.add_subplot(grid[0, 2])

    heatmap_images = []
    for axis, source_type in zip(heatmap_axes, ["HW", "LLM"], strict=True):
        image = style_overview_difference_heatmap_axis(
            axis,
            difference_matrices[source_type],
            (
                f"{SOURCE_DISPLAY[source_type]}\n"
                f"n={summary[source_type]['left_rows']} vs {summary[source_type]['right_rows']}"
            ),
            limit,
        )
        if source_type != "HW":
            axis.set_yticklabels([])
        heatmap_images.append(image)

    colorbar = figure.colorbar(heatmap_images[-1], cax=colorbar_axis)
    colorbar.set_label("Delta pair strength")
    figure.suptitle(
        f"RQ2: {left_label} - {right_label}",
        y=0.98,
        fontsize=15,
    )
    figure.subplots_adjust(left=0.06, right=0.95, bottom=0.10, top=0.88)
    figure.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close(figure)
    return summary


def plot_source_split_pair_strengths(
    frame: pd.DataFrame,
    dataset_label: str,
    output_path: Path,
) -> dict[str, dict[str, float]]:
    matrices: dict[str, np.ndarray] = {}
    summary: dict[str, dict[str, float]] = {}
    for source_type in ["HW", "LLM"]:
        subset = frame[frame["source_type"] == source_type].copy()
        matrices[source_type] = overview_style_matrix(subset, OVERVIEW_RAW_COLUMNS)
        summary[source_type] = {"rows": int(len(subset))}

    limit = max(float(np.nanmax(matrix)) for matrix in matrices.values())
    limit = max(limit, 1e-6)

    fig, axes = plt.subplots(1, 2, figsize=(12, 5.6))
    images = []
    for axis, source_type in zip(axes, ["HW", "LLM"]):
        images.append(
            style_positive_heatmap_axis(
                axis,
                matrices[source_type],
                f"{SOURCE_DISPLAY[source_type]}\nn={summary[source_type]['rows']}",
                limit,
            )
        )
    colorbar = fig.colorbar(images[-1], ax=axes, fraction=0.035, pad=0.03)
    colorbar.set_label("Mean pair strength E[p_i p_j]")
    fig.suptitle(
        f"RQ2: {dataset_label} Persuasion Pair-Strength Matrix",
        fontsize=14,
        y=1.02,
    )
    fig.tight_layout()
    fig.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    return summary


def build_kde_grid(frame: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
    x_values = frame["action_oriented_persuasion"].to_numpy(dtype=float)
    y_values = frame["trust_legitimacy_persuasion"].to_numpy(dtype=float)
    x_padding = max((x_values.max() - x_values.min()) * 0.15, 0.03)
    y_padding = max((y_values.max() - y_values.min()) * 0.15, 0.03)
    x_grid = np.linspace(x_values.min() - x_padding, x_values.max() + x_padding, 200)
    y_grid = np.linspace(y_values.min() - y_padding, y_values.max() + y_padding, 200)
    return np.meshgrid(x_grid, y_grid)


def density_surface(
    subset: pd.DataFrame,
    xx: np.ndarray,
    yy: np.ndarray,
    sample_size: int,
) -> np.ndarray | None:
    sample = subset
    if len(sample) > sample_size:
        sample = sample.sample(sample_size, random_state=42)
    x_values = sample["action_oriented_persuasion"].to_numpy(dtype=float)
    y_values = sample["trust_legitimacy_persuasion"].to_numpy(dtype=float)
    if len(sample) < 3:
        return None
    if np.isclose(x_values.std(), 0.0) or np.isclose(y_values.std(), 0.0):
        return None
    stacked = np.vstack([x_values, y_values])
    kde = gaussian_kde(stacked)
    positions = np.vstack([xx.ravel(), yy.ravel()])
    surface = kde(positions).reshape(xx.shape)
    return surface


def plot_density_contours(frame: pd.DataFrame, output_path: Path, sample_size: int) -> dict[str, dict[str, float]]:
    xx, yy = build_kde_grid(frame)
    fig, ax = plt.subplots(figsize=(10, 8))
    summary: dict[str, dict[str, float]] = {}

    for group_name in ["ScamLLM Email", "Vishing Single-turn", "Vishing Multi-turn"]:
        subset = frame[frame["group_name"] == group_name].copy()
        if subset.empty:
            continue
        surface = density_surface(subset, xx, yy, sample_size=sample_size)
        if surface is not None:
            levels = np.linspace(surface.max() * 0.25, surface.max() * 0.85, 4)
            ax.contourf(
                xx,
                yy,
                surface,
                levels=np.r_[levels, surface.max() * 1.01],
                colors=[GROUP_COLORS[group_name]],
                alpha=0.10,
            )
            ax.contour(
                xx,
                yy,
                surface,
                levels=levels,
                colors=[GROUP_COLORS[group_name]],
                linewidths=1.6,
                alpha=0.95,
            )

        centroid_x = float(subset["action_oriented_persuasion"].mean())
        centroid_y = float(subset["trust_legitimacy_persuasion"].mean())
        ax.scatter(
            [centroid_x],
            [centroid_y],
            color=GROUP_COLORS[group_name],
            s=65,
            edgecolor="white",
            linewidth=0.8,
            zorder=5,
        )
        summary[group_name] = {
            "rows": int(len(subset)),
            "action_mean": centroid_x,
            "trust_mean": centroid_y,
            "action_std": float(subset["action_oriented_persuasion"].std(ddof=0)),
            "trust_std": float(subset["trust_legitimacy_persuasion"].std(ddof=0)),
        }

    ax.set_title("RQ2: Vishing vs. ScamLLM Email in Persuasion Space", fontsize=13, pad=12)
    ax.set_xlabel("Action-oriented persuasion composite")
    ax.set_ylabel("Trust / legitimacy-oriented persuasion composite")
    ax.grid(alpha=0.2, linewidth=0.6)
    legend_items = [
        Line2D([0], [0], color=GROUP_COLORS[name], lw=2, marker="o", markersize=6, label=name)
        for name in ["ScamLLM Email", "Vishing Single-turn", "Vishing Multi-turn"]
        if name in summary
    ]
    ax.legend(handles=legend_items, loc="best", frameon=True)
    fig.tight_layout()
    fig.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    return summary


def build_pca_projection(frame: pd.DataFrame) -> tuple[pd.DataFrame, dict[str, object]]:
    feature_columns = [
        *SHARE_COLUMNS,
        "urgency_rate_scaled",
        "threat_rate_scaled",
        "action_request_rate_scaled",
        "politeness_rate_scaled",
        "role_credibility_rate_scaled",
    ]
    features = frame[feature_columns].to_numpy(dtype=float)
    feature_means = features.mean(axis=0)
    feature_stds = features.std(axis=0, ddof=0)
    feature_stds[feature_stds == 0.0] = 1.0
    standardized = (features - feature_means) / feature_stds

    # PCA via SVD keeps the script self-contained and avoids extra dependencies.
    _, singular_values, right_vectors = np.linalg.svd(standardized, full_matrices=False)
    components = right_vectors[:2]
    projected = standardized @ components.T

    explained_variance = (singular_values**2) / max(standardized.shape[0] - 1, 1)
    explained_variance_ratio = explained_variance / explained_variance.sum()

    projected_frame = frame.copy()
    projected_frame["pc1"] = projected[:, 0]
    projected_frame["pc2"] = projected[:, 1]

    metadata = {
        "feature_columns": feature_columns,
        "feature_means": {column: float(value) for column, value in zip(feature_columns, feature_means)},
        "feature_stds": {column: float(value) for column, value in zip(feature_columns, feature_stds)},
        "components": {
            "pc1": {column: float(value) for column, value in zip(feature_columns, components[0])},
            "pc2": {column: float(value) for column, value in zip(feature_columns, components[1])},
        },
        "explained_variance_ratio": {
            "pc1": float(explained_variance_ratio[0]),
            "pc2": float(explained_variance_ratio[1]),
        },
    }
    return projected_frame, metadata


def build_kde_grid_for_columns(frame: pd.DataFrame, x_column: str, y_column: str) -> tuple[np.ndarray, np.ndarray]:
    x_values = frame[x_column].to_numpy(dtype=float)
    y_values = frame[y_column].to_numpy(dtype=float)
    x_padding = max((x_values.max() - x_values.min()) * 0.15, 0.03)
    y_padding = max((y_values.max() - y_values.min()) * 0.15, 0.03)
    x_grid = np.linspace(x_values.min() - x_padding, x_values.max() + x_padding, 200)
    y_grid = np.linspace(y_values.min() - y_padding, y_values.max() + y_padding, 200)
    return np.meshgrid(x_grid, y_grid)


def density_surface_for_columns(
    subset: pd.DataFrame,
    xx: np.ndarray,
    yy: np.ndarray,
    x_column: str,
    y_column: str,
    sample_size: int,
) -> np.ndarray | None:
    sample = subset
    if len(sample) > sample_size:
        sample = sample.sample(sample_size, random_state=42)
    x_values = sample[x_column].to_numpy(dtype=float)
    y_values = sample[y_column].to_numpy(dtype=float)
    if len(sample) < 3:
        return None
    if np.isclose(x_values.std(), 0.0) or np.isclose(y_values.std(), 0.0):
        return None
    stacked = np.vstack([x_values, y_values])
    kde = gaussian_kde(stacked)
    positions = np.vstack([xx.ravel(), yy.ravel()])
    surface = kde(positions).reshape(xx.shape)
    return surface


def plot_pca_density_contours(frame: pd.DataFrame, output_path: Path, sample_size: int) -> dict[str, dict[str, float]]:
    xx, yy = build_kde_grid_for_columns(frame, "pc1", "pc2")
    fig, ax = plt.subplots(figsize=(10, 8))
    summary: dict[str, dict[str, float]] = {}
    scatter_sample_size = min(sample_size, 2500)

    for group_name in ["ScamLLM Email", "Vishing Single-turn", "Vishing Multi-turn"]:
        subset = frame[frame["group_name"] == group_name].copy()
        if subset.empty:
            continue

        scatter_sample = subset
        if len(scatter_sample) > scatter_sample_size:
            scatter_sample = scatter_sample.sample(scatter_sample_size, random_state=42)
        ax.scatter(
            scatter_sample["pc1"],
            scatter_sample["pc2"],
            s=8,
            color=GROUP_COLORS[group_name],
            alpha=0.14,
            linewidth=0,
            zorder=1,
        )

        surface = density_surface_for_columns(subset, xx, yy, "pc1", "pc2", sample_size=sample_size)
        if surface is not None:
            levels = np.linspace(surface.max() * 0.25, surface.max() * 0.85, 4)
            ax.contourf(
                xx,
                yy,
                surface,
                levels=np.r_[levels, surface.max() * 1.01],
                colors=[GROUP_COLORS[group_name]],
                alpha=0.10,
            )
            ax.contour(
                xx,
                yy,
                surface,
                levels=levels,
                colors=[GROUP_COLORS[group_name]],
                linewidths=1.6,
                alpha=0.95,
            )

        summary[group_name] = {
            "rows": int(len(subset)),
            "pc1_mean": float(subset["pc1"].mean()),
            "pc2_mean": float(subset["pc2"].mean()),
            "pc1_std": float(subset["pc1"].std(ddof=0)),
            "pc2_std": float(subset["pc2"].std(ddof=0)),
        }

    ax.set_title("RQ2: Phishing-only PCA Space", fontsize=13, pad=12)
    ax.set_xlabel("PC1")
    ax.set_ylabel("PC2")
    ax.set_xlim(-7.5, 5.0)
    ax.set_ylim(-5.0, 4.5)
    ax.set_xticks(np.arange(-7.5, 5.1, 2.5))
    ax.set_yticks(np.arange(-5.0, 4.6, 2.0))
    ax.grid(alpha=0.2, linewidth=0.6)
    legend_items = [
        Line2D(
            [0],
            [0],
            color=GROUP_COLORS[name],
            lw=2,
            marker="o",
            markersize=8,
            label=name,
        )
        for name in ["ScamLLM Email", "Vishing Single-turn", "Vishing Multi-turn"]
        if name in summary
    ]
    ax.legend(handles=legend_items, loc="best", frameon=True)
    fig.tight_layout()
    fig.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    return summary


def plot_source_split_pca_density_contours(
    frame: pd.DataFrame,
    source_type: str,
    output_path: Path,
    sample_size: int,
) -> dict[str, object]:
    subset_frame = frame[frame["source_type"] == source_type].copy()
    xx, yy = build_kde_grid_for_columns(subset_frame, "pc1", "pc2")
    fig, ax = plt.subplots(figsize=(10, 8))
    summary: dict[str, dict[str, float]] = {}
    scatter_sample_size = min(sample_size, 2500)

    for group_name in ["ScamLLM Email", "Vishing Single-turn", "Vishing Multi-turn"]:
        subset = subset_frame[subset_frame["group_name"] == group_name].copy()
        if subset.empty:
            continue

        scatter_sample = subset
        if len(scatter_sample) > scatter_sample_size:
            scatter_sample = scatter_sample.sample(scatter_sample_size, random_state=42)
        ax.scatter(
            scatter_sample["pc1"],
            scatter_sample["pc2"],
            s=10,
            color=GROUP_COLORS[group_name],
            alpha=0.16,
            linewidth=0,
            zorder=1,
        )

        surface = density_surface_for_columns(subset, xx, yy, "pc1", "pc2", sample_size=sample_size)
        if surface is not None:
            levels = np.linspace(surface.max() * 0.25, surface.max() * 0.85, 4)
            ax.contourf(
                xx,
                yy,
                surface,
                levels=np.r_[levels, surface.max() * 1.01],
                colors=[GROUP_COLORS[group_name]],
                alpha=0.10,
            )
            ax.contour(
                xx,
                yy,
                surface,
                levels=levels,
                colors=[GROUP_COLORS[group_name]],
                linewidths=1.6,
                alpha=0.95,
            )

        summary[group_name] = {
            "rows": int(len(subset)),
            "pc1_mean": float(subset["pc1"].mean()),
            "pc2_mean": float(subset["pc2"].mean()),
            "pc1_std": float(subset["pc1"].std(ddof=0)),
            "pc2_std": float(subset["pc2"].std(ddof=0)),
        }

    ax.set_title(f"RQ2: {SOURCE_DISPLAY[source_type]} PCA Space", fontsize=13, pad=12)
    ax.set_xlabel("PC1")
    ax.set_ylabel("PC2")
    ax.set_xlim(-7.5, 5.0)
    ax.set_ylim(-5.0, 4.5)
    ax.set_xticks(np.arange(-7.5, 5.1, 2.5))
    ax.set_yticks(np.arange(-5.0, 4.6, 2.0))
    ax.grid(alpha=0.2, linewidth=0.6)
    legend_items = [
        Line2D(
            [0],
            [0],
            color=GROUP_COLORS[name],
            lw=2,
            marker="o",
            markersize=8,
            label=name,
        )
        for name in ["ScamLLM Email", "Vishing Single-turn", "Vishing Multi-turn"]
        if name in summary
    ]
    ax.legend(handles=legend_items, loc="best", frameon=True)
    fig.tight_layout()
    fig.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    return {
        "source_type": source_type,
        "groups": summary,
    }


def fit_pca_surrogate_surface(frame: pd.DataFrame) -> Pipeline:
    working = frame.copy()
    working["is_vishing"] = working["group_name"].isin(["Vishing Single-turn", "Vishing Multi-turn"]).astype(int)
    features = working[["pc1", "pc2"]].to_numpy(dtype=float)
    labels = working["is_vishing"].to_numpy(dtype=int)
    model = Pipeline(
        [
            ("poly", PolynomialFeatures(degree=2, include_bias=False)),
            ("scale", StandardScaler()),
            (
                "logreg",
                LogisticRegression(
                    max_iter=2000,
                    class_weight="balanced",
                    random_state=42,
                ),
            ),
        ]
    )
    model.fit(features, labels)
    return model


def build_fixed_pca_grid() -> tuple[np.ndarray, np.ndarray]:
    grid_x, grid_y = np.meshgrid(
        np.linspace(-7.5, 5.0, 220),
        np.linspace(-5.0, 4.5, 220),
    )
    return grid_x, grid_y


def plot_reference_style_pca_contours(frame: pd.DataFrame, output_path: Path, sample_size: int) -> dict[str, object]:
    model = fit_pca_surrogate_surface(frame)
    grid_x, grid_y = build_fixed_pca_grid()
    mesh = np.column_stack([grid_x.ravel(), grid_y.ravel()])
    score_grid = model.predict_proba(mesh)[:, 1].reshape(grid_x.shape)

    fig, ax = plt.subplots(figsize=(12.6, 8.8))
    filled = ax.contourf(
        grid_x,
        grid_y,
        score_grid,
        levels=np.linspace(0.0, 1.0, 13),
        cmap="YlOrRd",
        alpha=0.72,
    )
    ax.contour(
        grid_x,
        grid_y,
        score_grid,
        levels=[0.2, 0.4, 0.6, 0.8],
        colors="#8a4f15",
        linewidths=0.65,
        alpha=0.55,
    )
    ax.contour(
        grid_x,
        grid_y,
        score_grid,
        levels=[0.5],
        colors="black",
        linewidths=1.8,
    )

    scatter_sample_size = min(sample_size, 3500)
    for group_name in ["ScamLLM Email", "Vishing Single-turn", "Vishing Multi-turn"]:
        subset = frame[frame["group_name"] == group_name].copy()
        if subset.empty:
            continue
        scatter_sample = subset
        if len(scatter_sample) > scatter_sample_size:
            scatter_sample = scatter_sample.sample(scatter_sample_size, random_state=42)
        ax.scatter(
            scatter_sample["pc1"],
            scatter_sample["pc2"],
            s=18,
            color=GROUP_COLORS[group_name],
            alpha=0.22,
            marker=GROUP_MARKERS[group_name],
            linewidth=0.3,
            edgecolors="white",
            zorder=2,
        )

    ax.set_title(
        "RQ2: Vishing vs ScamLLM Email PCA Contours\n"
        f"Email={int((frame['group_name'] == 'ScamLLM Email').sum())} | "
        f"Single-turn={int((frame['group_name'] == 'Vishing Single-turn').sum())} | "
        f"Multi-turn={int((frame['group_name'] == 'Vishing Multi-turn').sum())}",
        fontsize=14,
        pad=10,
    )
    ax.set_xlabel("PCA1", fontsize=11)
    ax.set_ylabel("PCA2", fontsize=11)
    ax.set_xlim(-7.5, 5.0)
    ax.set_ylim(-5.0, 4.5)
    ax.set_xticks(np.arange(-7.5, 5.1, 2.5))
    ax.set_yticks(np.arange(-5.0, 4.6, 1.5))
    ax.grid(alpha=0.10, linewidth=0.5)

    legend_items = [
        Line2D(
            [0],
            [0],
            marker=GROUP_MARKERS["ScamLLM Email"],
            color="none",
            markerfacecolor=GROUP_COLORS["ScamLLM Email"],
            markeredgecolor="white",
            markersize=8,
            linewidth=0,
            label="ScamLLM Email",
        ),
        Line2D(
            [0],
            [0],
            marker=GROUP_MARKERS["Vishing Single-turn"],
            color="none",
            markerfacecolor=GROUP_COLORS["Vishing Single-turn"],
            markeredgecolor="white",
            markersize=8,
            linewidth=0,
            label="Vishing Single-turn",
        ),
        Line2D(
            [0],
            [0],
            marker=GROUP_MARKERS["Vishing Multi-turn"],
            color="none",
            markerfacecolor=GROUP_COLORS["Vishing Multi-turn"],
            markeredgecolor="white",
            markersize=8,
            linewidth=0,
            label="Vishing Multi-turn",
        ),
        Line2D([0], [0], color="black", linewidth=1.8, label="threshold contour: s(x)=0.50"),
    ]
    ax.legend(handles=legend_items, loc="upper left", frameon=True)
    colorbar = fig.colorbar(filled, ax=ax, fraction=0.04, pad=0.02)
    colorbar.set_label("Vishing surrogate score")
    fig.tight_layout()
    fig.savefig(output_path, dpi=240, bbox_inches="tight")
    plt.close(fig)
    return {
        "email_rows": int((frame["group_name"] == "ScamLLM Email").sum()),
        "single_turn_rows": int((frame["group_name"] == "Vishing Single-turn").sum()),
        "multi_turn_rows": int((frame["group_name"] == "Vishing Multi-turn").sum()),
    }


def main() -> int:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    email_frame = load_email_group()
    single_frame = load_vishing_turn(
        "single",
        output_dir=args.output_dir,
        device=args.device,
        force_rescore=args.force_rescore,
    )
    multi_frame = load_vishing_turn(
        "multi",
        output_dir=args.output_dir,
        device=args.device,
        force_rescore=args.force_rescore,
    )
    combined_vishing = pd.concat([single_frame, multi_frame], ignore_index=True)
    combined_vishing["group_name"] = "Vishing Overall"
    combined_vishing["turn_type"] = "overall"

    all_records = pd.concat([email_frame, single_frame, multi_frame], ignore_index=True)
    all_records = normalize_principle_shares(all_records)
    all_records = add_composite_features(all_records)

    email_normalized = all_records[all_records["group_name"] == "ScamLLM Email"].copy()
    single_normalized = all_records[all_records["group_name"] == "Vishing Single-turn"].copy()
    multi_normalized = all_records[all_records["group_name"] == "Vishing Multi-turn"].copy()
    overall_normalized = pd.concat([single_normalized, multi_normalized], ignore_index=True)

    comparisons = {
        "single_vs_email": ("Vishing Single-turn", single_normalized, email_normalized),
        "multi_vs_email": ("Vishing Multi-turn", multi_normalized, email_normalized),
        "overall_vs_email": ("Vishing Overall", overall_normalized, email_normalized),
    }

    pairwise_rows: list[dict[str, float | str]] = []
    principle_rows: list[dict[str, float | str]] = []
    heatmap_matrices: list[np.ndarray] = []
    comparison_metadata: dict[str, object] = {}

    for comparison_name, (left_name, left_frame, right_frame) in comparisons.items():
        left_matrix = pair_strength_matrix(left_frame)
        right_matrix = pair_strength_matrix(right_frame)
        delta_matrix = left_matrix - right_matrix
        heatmap_matrices.append(delta_matrix)
        pairwise_rows.extend(
            flatten_pairwise_stats(
                comparison_name=comparison_name,
                left_name=left_name,
                right_name="ScamLLM Email",
                left_matrix=left_matrix,
                right_matrix=right_matrix,
            )
        )

        left_means = mean_share_vector(left_frame)
        right_means = mean_share_vector(right_frame)
        for principle_label, left_value, right_value in zip(PRINCIPLE_LABELS, left_means, right_means):
            principle_rows.append(
                {
                    "comparison": comparison_name,
                    "left_group": left_name,
                    "right_group": "ScamLLM Email",
                    "principle": principle_label,
                    "left_mean_share": float(left_value),
                    "right_mean_share": float(right_value),
                    "delta_mean_share": float(left_value - right_value),
                }
            )

        comparison_metadata[comparison_name] = {
            "left_group": left_name,
            "right_group": "ScamLLM Email",
            "left_rows": int(len(left_frame)),
            "right_rows": int(len(right_frame)),
        }

    pairwise_df = pd.DataFrame(pairwise_rows)
    principle_df = pd.DataFrame(principle_rows)
    pairwise_df.to_csv(args.output_dir / "rq2_pairwise_difference_stats.csv", index=False)
    principle_df.to_csv(args.output_dir / "rq2_principle_share_differences.csv", index=False)

    source_split_heatmap_summary = {
        "email": plot_source_split_pair_strengths(
            frame=email_normalized,
            dataset_label="ScamLLM Email",
            output_path=args.output_dir / "rq2_email_pair_strength_heatmaps.png",
        ),
        "single": plot_source_split_pair_strengths(
            frame=single_normalized,
            dataset_label="Vishing Single-turn",
            output_path=args.output_dir / "rq2_vishing_single_pair_strength_heatmaps.png",
        ),
        "multi": plot_source_split_pair_strengths(
            frame=multi_normalized,
            dataset_label="Vishing Multi-turn",
            output_path=args.output_dir / "rq2_vishing_multi_pair_strength_heatmaps.png",
        ),
    }
    source_split_difference_summary = {
        "single": plot_source_split_difference_heatmaps(
            left_frame=email_normalized,
            right_frame=single_normalized,
            left_label="ScamLLM Email",
            right_label="Vishing Single-turn",
            output_path=args.output_dir / "rq2_vishing_single_minus_email_difference_heatmaps.png",
        ),
        "multi": plot_source_split_difference_heatmaps(
            left_frame=email_normalized,
            right_frame=multi_normalized,
            left_label="ScamLLM Email",
            right_label="Vishing Multi-turn",
            output_path=args.output_dir / "rq2_vishing_multi_minus_email_difference_heatmaps.png",
        ),
    }

    density_summary = plot_density_contours(
        frame=all_records,
        output_path=args.output_dir / "rq2_vishing_email_density_contours.png",
        sample_size=args.density_sample_size,
    )

    pca_records, pca_metadata = build_pca_projection(all_records)
    pca_records[
        [
            "sample_id",
            "group_name",
            "turn_type",
            "dataset_family",
            "source_name",
            "pc1",
            "pc2",
            *SHARE_COLUMNS,
            "urgency_rate_scaled",
            "threat_rate_scaled",
            "action_request_rate_scaled",
            "politeness_rate_scaled",
            "role_credibility_rate_scaled",
        ]
    ].to_csv(args.output_dir / "rq2_vishing_email_pca_points.csv", index=False)
    pca_density_summary = plot_pca_density_contours(
        frame=pca_records,
        output_path=args.output_dir / "rq2_vishing_email_pca_contours.png",
        sample_size=args.density_sample_size,
    )
    pca_source_split_summary = {
        "HW": plot_source_split_pca_density_contours(
            frame=pca_records,
            source_type="HW",
            output_path=args.output_dir / "rq2_vishing_email_pca_contours_hw.png",
            sample_size=args.density_sample_size,
        ),
        "LLM": plot_source_split_pca_density_contours(
            frame=pca_records,
            source_type="LLM",
            output_path=args.output_dir / "rq2_vishing_email_pca_contours_llm.png",
            sample_size=args.density_sample_size,
        ),
    }
    pca_reference_style_summary = plot_reference_style_pca_contours(
        frame=pca_records,
        output_path=args.output_dir / "rq2_vishing_email_pca_contours_reference_style.png",
        sample_size=args.density_sample_size,
    )

    top_difference_summary = summarize_top_differences(pairwise_df, principle_df)
    metadata = {
        "normalization": {
            "principle_scores": "within-sample share normalization across the six persuasion principles",
            "pair_strength": "mean outer product of normalized principle shares",
            "density_axes": {
                "x": (
                    "mean of reciprocity/share, scarcity/share, commitment/share, "
                    "urgency cue rate, threat cue rate, and action-request cue rate"
                ),
                "y": (
                    "mean of authority/share, liking/share, social-proof/share, "
                    "politeness cue rate, and role-credibility cue rate"
                ),
            },
            "rate_scaling": all_records.attrs.get("rate_scales", {}),
        },
        "groups": {
            "email_rows": int(len(email_normalized)),
            "single_turn_vishing_rows": int(len(single_normalized)),
            "multi_turn_vishing_rows": int(len(multi_normalized)),
            "overall_vishing_rows": int(len(overall_normalized)),
        },
        "comparisons": comparison_metadata,
        "source_split_pair_strength": source_split_heatmap_summary,
        "source_split_difference": source_split_difference_summary,
        "density_summary": density_summary,
        "pca_summary": {
            "projection": pca_metadata,
            "group_density": pca_density_summary,
            "source_split_density": pca_source_split_summary,
            "reference_style": pca_reference_style_summary,
        },
        "top_differences": top_difference_summary,
        "outputs": {
            "email_pair_strength_heatmap": str(args.output_dir / "rq2_email_pair_strength_heatmaps.png"),
            "single_pair_strength_heatmap": str(args.output_dir / "rq2_vishing_single_pair_strength_heatmaps.png"),
            "multi_pair_strength_heatmap": str(args.output_dir / "rq2_vishing_multi_pair_strength_heatmaps.png"),
            "single_difference_heatmap": str(args.output_dir / "rq2_vishing_single_minus_email_difference_heatmaps.png"),
            "multi_difference_heatmap": str(args.output_dir / "rq2_vishing_multi_minus_email_difference_heatmaps.png"),
            "density_figure": str(args.output_dir / "rq2_vishing_email_density_contours.png"),
            "pca_density_figure": str(args.output_dir / "rq2_vishing_email_pca_contours.png"),
            "pca_density_hw_figure": str(args.output_dir / "rq2_vishing_email_pca_contours_hw.png"),
            "pca_density_llm_figure": str(args.output_dir / "rq2_vishing_email_pca_contours_llm.png"),
            "pca_reference_style_figure": str(args.output_dir / "rq2_vishing_email_pca_contours_reference_style.png"),
            "pca_points_csv": str(args.output_dir / "rq2_vishing_email_pca_points.csv"),
            "pairwise_csv": str(args.output_dir / "rq2_pairwise_difference_stats.csv"),
            "principle_csv": str(args.output_dir / "rq2_principle_share_differences.csv"),
        },
    }
    (args.output_dir / "rq2_summary.json").write_text(
        json.dumps(metadata, indent=2, ensure_ascii=True),
        encoding="utf-8",
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
