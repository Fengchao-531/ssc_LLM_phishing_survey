#!/usr/bin/env python3
"""Build stage-wise phishing pattern visualizations for the survey workspace.

This script creates a lightweight indicator space from email subject/body text,
projects the sampled records into 2D, and renders four figure families:

1. Pattern map: sampled points, density backdrop, and source confidence ellipses
2. Detector coverage: a surrogate ScamLLM response field over the 2D projection
3. Indicator composition: mean standardized indicator values for HW vs LLM
4. Motif delta: co-occurrence-rate differences between LLM and HW indicators

The current default is a preview mode that samples 100 rows per stage per source
from the processed academic evaluation datasets.
"""

from __future__ import annotations

import argparse
import json
import math
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.colors import Normalize
from matplotlib.patches import Ellipse
from sklearn.decomposition import PCA
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler


SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parent
DATA_ROOT = REPO_ROOT / "Visualization" / "persuasion_strategy_wvae" / "output" / "full_inference_results" / "phishing_only"
DEFAULT_OUTPUT_DIR = SCRIPT_DIR / "output"

STAGE_ORDER = [
    "S1",
    "S2",
    "S4",
    "S5",
    "S6-MPG",
    "S6-UTA",
    "S6-fuzzer",
    "S8-deepseek",
    "S8-llama",
    "S8-ministral",
]
SOURCE_PATHS = {
    "HW": DATA_ROOT,
    "LLM": DATA_ROOT,
}
SOURCE_COLORS = {"HW": "#1f77b4", "LLM": "#d95f02"}
SOURCE_MARKERS = {"HW": "o", "LLM": "^"}

INDICATOR_DISPLAY_NAMES = {
    "principle_authority": "Authority (WVAE)",
    "principle_reciprocity": "Reciprocity (WVAE)",
    "principle_commitment": "Commitment (WVAE)",
    "principle_scarcity": "Scarcity (WVAE)",
    "principle_social_proof": "Social Proof (WVAE)",
    "principle_liking": "Liking (WVAE)",
    "principle_Authority": "Authority",
    "principle_Reciprocity": "Reciprocity",
    "principle_Commitment": "Commitment",
    "principle_Scarcity": "Scarcity",
    "principle_Social Proof": "Social Proof",
    "principle_Liking": "Liking",
    "authority_score": "Authority",
    "urgency_score": "Urgency",
    "credential_score": "Credential",
    "payment_score": "Payment",
    "secrecy_score": "Secrecy",
    "link_score": "Link",
    "attachment_score": "Attach",
    "threat_score": "Threat",
    "contact_score": "Contact",
    "placeholder_score": "Placeholder",
    "llm_style_score": "LLM-style",
    "html_score": "HTML",
    "token_count_log": "Length",
    "uppercase_ratio": "Caps",
    "digit_ratio": "Digits",
    "punct_ratio": "Punct",
    "line_break_ratio": "Breaks",
}
COMPOSITION_COLUMNS = [
    "authority_score",
    "urgency_score",
    "credential_score",
    "payment_score",
    "secrecy_score",
    "link_score",
    "attachment_score",
    "threat_score",
    "placeholder_score",
    "llm_style_score",
]
MOTIF_COLUMNS = [
    "authority_flag",
    "urgency_flag",
    "credential_flag",
    "payment_flag",
    "secrecy_flag",
    "link_flag",
    "attachment_flag",
    "threat_flag",
]
MOTIF_DISPLAY_NAMES = {
    "authority_flag": "Auth",
    "urgency_flag": "Urg",
    "credential_flag": "Cred",
    "payment_flag": "Pay",
    "secrecy_flag": "Sec",
    "link_flag": "Link",
    "attachment_flag": "Att",
    "threat_flag": "Threat",
}

TOKEN_RE = re.compile(r"[A-Za-z0-9_]+")
HTML_TAG_RE = re.compile(r"<[^>]+>")
URL_RE = re.compile(r"(?:https?://|www\.)\S+", re.IGNORECASE)
BRACKET_PLACEHOLDER_RE = re.compile(
    r"(\[[^\]]{1,40}\]|<[^>]{1,40}>|X{3,}|x{3,}|_{3,})"
)

PATTERN_SPECS = {
    "authority": re.compile(
        r"\b(ceo|chief executive|manager|director|finance team|accounts payable|accounting "
        r"department|bank|it support|security team|administrator|admin|microsoft|paypal|hr)\b",
        re.IGNORECASE,
    ),
    "urgency": re.compile(
        r"\b(urgent|immediately|asap|right away|within \d+ (?:hours|days)|deadline|today|now|"
        r"time[- ]sensitive|promptly|immediate attention)\b",
        re.IGNORECASE,
    ),
    "credential": re.compile(
        r"\b(password|passcode|otp|mfa|2fa|verify your account|login|log in|sign in|username|"
        r"credential|confirm your identity|security code)\b",
        re.IGNORECASE,
    ),
    "payment": re.compile(
        r"\b(invoice|payment|wire transfer|bank account|routing number|swift code|remit|"
        r"transaction|settle the attached invoice|outstanding invoice|gift card|funds)\b",
        re.IGNORECASE,
    ),
    "secrecy": re.compile(
        r"\b(confidential|strictly confidential|secret|discreet|do not discuss|keep this "
        r"between us|quietly|privately|delete all correspondence)\b",
        re.IGNORECASE,
    ),
    "link": re.compile(
        r"\b(click|copy and paste|portal|website|web page|download here|review the link|visit)\b",
        re.IGNORECASE,
    ),
    "attachment": re.compile(
        r"\b(attached|attachment|pdf|docx|xlsx|zip file|invoice attached|open the file|"
        r"see attached)\b",
        re.IGNORECASE,
    ),
    "threat": re.compile(
        r"\b(suspended|termination|penalty|consequences|legal action|locked|final warning|"
        r"failure to comply|deactivated|disable)\b",
        re.IGNORECASE,
    ),
    "contact": re.compile(
        r"\b(reply|call me|phone|text me|whatsapp|reach out|contact me|send me the confirmation|"
        r"respond directly)\b",
        re.IGNORECASE,
    ),
    "llm_style": re.compile(
        r"(i hope this email finds you well|do not hesitate to reach out|kindly|warm regards|"
        r"thank you for your immediate attention|valued customer|dear \[[^\]]+\])",
        re.IGNORECASE,
    ),
}


@dataclass(frozen=True)
class ProjectionArtifacts:
    frame: pd.DataFrame
    indicator_columns: list[str]
    composition_columns: list[str]
    motif_columns: list[str]
    method: str


def build_run_suffix(sample_size: int, phishing_only: bool) -> str:
    base = "full" if sample_size == 0 else f"preview_n{sample_size}"
    if phishing_only:
        return f"{base}_phishing_only"
    return base


def source_display_name(source_name: str, phishing_only: bool) -> str:
    if phishing_only:
        return f"{source_name}-phishing"
    return source_name


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Generate stage-wise pattern, coverage, composition, and motif figures "
            "for ScamLLM using sampled HW/GD and LLM academic datasets."
        )
    )
    parser.add_argument(
        "--detector-column",
        default="scamllm",
        help="Prediction column to treat as the detector output (default: scamllm).",
    )
    parser.add_argument(
        "--sample-size",
        type=int,
        default=100,
        help="Rows sampled per stage per source. Use 0 for all rows.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=7,
        help="Random seed used for deterministic stage/source sampling.",
    )
    parser.add_argument(
        "--projection",
        choices=["pca", "umap"],
        default="pca",
        help="2D projection method. UMAP is optional and falls back to PCA if unavailable.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
        help=f"Output directory for figures and sampled data (default: {DEFAULT_OUTPUT_DIR}).",
    )
    parser.add_argument(
        "--phishing-only",
        action="store_true",
        help="Keep only phishing rows (label == 1) so the classes become HW-phishing vs LLM-phishing.",
    )
    return parser.parse_args()


def stage_seed(base_seed: int, stage_name: str, source_name: str) -> int:
    return base_seed + sum(ord(char) for char in f"{stage_name}:{source_name}")


def normalize_text(value: object) -> str:
    return str(value or "").replace("\r\n", "\n").replace("\r", "\n").strip()


def find_count(pattern: re.Pattern[str], text: str) -> int:
    return len(pattern.findall(text))


def safe_ratio(numerator: float, denominator: float) -> float:
    if denominator <= 0:
        return 0.0
    return float(numerator) / float(denominator)


def tokenize(text: str) -> list[str]:
    return TOKEN_RE.findall(text.lower())


def extract_indicator_row(subject: str, body: str) -> dict[str, float]:
    subject_text = normalize_text(subject)
    body_text = normalize_text(body)
    merged_text = "\n\n".join(part for part in [subject_text, body_text] if part)
    tokens = tokenize(merged_text)
    token_count = len(tokens)
    char_count = len(merged_text)
    line_count = max(1, merged_text.count("\n") + 1)

    uppercase_chars = sum(1 for char in merged_text if char.isupper())
    digit_chars = sum(1 for char in merged_text if char.isdigit())
    punct_chars = sum(1 for char in merged_text if not char.isalnum() and not char.isspace())
    avg_token_length = float(np.mean([len(token) for token in tokens])) if tokens else 0.0

    authority_count = find_count(PATTERN_SPECS["authority"], merged_text)
    urgency_count = find_count(PATTERN_SPECS["urgency"], merged_text)
    credential_count = find_count(PATTERN_SPECS["credential"], merged_text)
    payment_count = find_count(PATTERN_SPECS["payment"], merged_text)
    secrecy_count = find_count(PATTERN_SPECS["secrecy"], merged_text)
    link_count = find_count(PATTERN_SPECS["link"], merged_text) + find_count(URL_RE, merged_text)
    attachment_count = find_count(PATTERN_SPECS["attachment"], merged_text)
    threat_count = find_count(PATTERN_SPECS["threat"], merged_text)
    contact_count = find_count(PATTERN_SPECS["contact"], merged_text)
    llm_style_count = find_count(PATTERN_SPECS["llm_style"], merged_text)
    html_count = find_count(HTML_TAG_RE, merged_text)
    placeholder_count = find_count(BRACKET_PLACEHOLDER_RE, merged_text)

    token_norm = max(1.0, token_count / 50.0)

    features = {
        "authority_score": authority_count / token_norm,
        "urgency_score": urgency_count / token_norm,
        "credential_score": credential_count / token_norm,
        "payment_score": payment_count / token_norm,
        "secrecy_score": secrecy_count / token_norm,
        "link_score": link_count / token_norm,
        "attachment_score": attachment_count / token_norm,
        "threat_score": threat_count / token_norm,
        "contact_score": contact_count / token_norm,
        "placeholder_score": placeholder_count / token_norm,
        "llm_style_score": llm_style_count / token_norm,
        "html_score": html_count / token_norm,
        "token_count_log": math.log1p(token_count),
        "avg_token_length": avg_token_length,
        "uppercase_ratio": safe_ratio(uppercase_chars, char_count),
        "digit_ratio": safe_ratio(digit_chars, char_count),
        "punct_ratio": safe_ratio(punct_chars, char_count),
        "line_break_ratio": safe_ratio(line_count - 1, token_count + 1),
    }
    features.update(
        {
            "authority_flag": float(authority_count > 0),
            "urgency_flag": float(urgency_count > 0),
            "credential_flag": float(credential_count > 0),
            "payment_flag": float(payment_count > 0),
            "secrecy_flag": float(secrecy_count > 0),
            "link_flag": float(link_count > 0),
            "attachment_flag": float(attachment_count > 0),
            "threat_flag": float(threat_count > 0),
        }
    )
    return features


def load_stage_source_frame(
    *,
    stage_name: str,
    source_name: str,
    detector_column: str,
    sample_size: int,
    seed: int,
    phishing_only: bool,
) -> pd.DataFrame:
    csv_path = SOURCE_PATHS[source_name] / f"{source_name}_{stage_name}_persuasion.csv"
    if not csv_path.exists():
        if source_name == 'LLM' and stage_name in [f'S{i}' for i in range(1, 8)]:
            print(f"Skipping non-existent LLM stage: {stage_name}")
            return pd.DataFrame()
        print(f"Warning: {csv_path} not found, skipping."); return pd.DataFrame()

    frame = pd.read_csv(csv_path)
    required_columns = {"subject", "body", "label", detector_column}
    missing = sorted(required_columns.difference(frame.columns))
    if missing:
        raise ValueError(f"{csv_path} is missing required columns: {', '.join(missing)}")

    frame = frame.copy()
    frame["label"] = pd.to_numeric(frame["label"], errors="coerce")
    if phishing_only:
        frame = frame[frame["label"] == 1].copy()
        if frame.empty:
            raise ValueError(f"{csv_path} has no phishing rows after filtering label == 1")

    if sample_size > 0 and len(frame) > sample_size:
        frame = frame.sample(
            n=sample_size,
            random_state=stage_seed(seed, stage_name, source_name),
            replace=False,
        )

    frame["stage"] = stage_name
    frame["source"] = source_name
    frame["detector_prediction"] = pd.to_numeric(frame[detector_column], errors="coerce").fillna(0.0)
    frame["raw_label"] = frame["label"]
    frame["text"] = (
        frame["subject"].fillna("").astype(str).str.strip()
        + "\n\n"
        + frame["body"].fillna("").astype(str).str.strip()
    ).str.strip()
    return frame.reset_index(drop=True)


def build_sampled_frame(args: argparse.Namespace) -> pd.DataFrame:
    stage_frames: list[pd.DataFrame] = []
    for stage_name in STAGE_ORDER:
        for source_name in SOURCE_PATHS:
            stage_frames.append(
                load_stage_source_frame(
                    stage_name=stage_name,
                    source_name=source_name,
                    detector_column=args.detector_column,
                    sample_size=args.sample_size,
                    seed=args.seed,
                    phishing_only=args.phishing_only,
                )
            )
    return pd.concat(stage_frames, ignore_index=True)


def project_indicator_space(
    frame: pd.DataFrame,
    *,
    projection: str,
    seed: int,
) -> ProjectionArtifacts:
    feature_rows = [
        extract_indicator_row(subject, body)
        for subject, body in zip(frame["subject"], frame["body"], strict=False)
    ]
    indicator_frame = pd.DataFrame(feature_rows)
    indicator_columns = indicator_frame.columns.tolist()

    scaler = StandardScaler()
    scaled_matrix = scaler.fit_transform(indicator_frame[indicator_columns])

    method = projection
    if projection == "umap":
        try:
            import umap.umap_ as umap  # type: ignore
        except ImportError:
            method = "pca"
        else:
            reducer = umap.UMAP(
                n_components=2,
                n_neighbors=min(20, max(5, len(indicator_frame) - 1)),
                min_dist=0.25,
                random_state=seed,
            )
            coords = reducer.fit_transform(scaled_matrix)
    if method == "pca":
        reducer = PCA(n_components=2, random_state=seed)
        coords = reducer.fit_transform(scaled_matrix)

    projected = frame.reset_index(drop=True).copy()
    for column in indicator_columns:
        projected[column] = indicator_frame[column].values
    standardized_indicator_frame = pd.DataFrame(
        scaled_matrix,
        columns=[f"{column}__z" for column in indicator_columns],
    )
    for column in standardized_indicator_frame.columns:
        projected[column] = standardized_indicator_frame[column].values
    projected["proj_x"] = coords[:, 0]
    projected["proj_y"] = coords[:, 1]

    return ProjectionArtifacts(
        frame=projected,
        indicator_columns=indicator_columns,
        composition_columns=COMPOSITION_COLUMNS,
        motif_columns=MOTIF_COLUMNS,
        method=method,
    )


def confidence_ellipse(
    ax: plt.Axes,
    x_values: np.ndarray,
    y_values: np.ndarray,
    *,
    color: str,
    n_std: float = 1.6,
) -> None:
    if len(x_values) < 3 or len(y_values) < 3:
        return

    covariance = np.cov(x_values, y_values)
    if covariance.shape != (2, 2) or not np.all(np.isfinite(covariance)):
        return

    eigenvalues, eigenvectors = np.linalg.eigh(covariance)
    if np.any(eigenvalues <= 0):
        return

    order = eigenvalues.argsort()[::-1]
    eigenvalues = eigenvalues[order]
    eigenvectors = eigenvectors[:, order]
    angle = np.degrees(np.arctan2(*eigenvectors[:, 0][::-1]))
    width, height = 2 * n_std * np.sqrt(eigenvalues)

    ellipse = Ellipse(
        xy=(float(np.mean(x_values)), float(np.mean(y_values))),
        width=float(width),
        height=float(height),
        angle=float(angle),
        facecolor="none",
        edgecolor=color,
        linestyle="--",
        linewidth=1.5,
        alpha=0.95,
    )
    ax.add_patch(ellipse)


def compute_axis_limits(frame: pd.DataFrame) -> tuple[tuple[float, float], tuple[float, float]]:
    x_min, x_max = frame["proj_x"].min(), frame["proj_x"].max()
    y_min, y_max = frame["proj_y"].min(), frame["proj_y"].max()
    x_pad = (x_max - x_min) * 0.08 or 1.0
    y_pad = (y_max - y_min) * 0.08 or 1.0
    return (x_min - x_pad, x_max + x_pad), (y_min - y_pad, y_max + y_pad)


def format_stage_title(stage_frame: pd.DataFrame, stage_name: str) -> str:
    hw_count = int((stage_frame["source"] == "HW").sum())
    llm_count = int((stage_frame["source"] == "LLM").sum())
    return f"{stage_name}\nHW={hw_count} | LLM={llm_count}"


def draw_pattern_map(
    frame: pd.DataFrame,
    *,
    output_path: Path,
    detector_column: str,
    projection_method: str,
    phishing_only: bool,
) -> None:
    figure, axes = plt.subplots(1, len(STAGE_ORDER), figsize=(38, 4.8), sharex=True, sharey=True)
    x_limits, y_limits = compute_axis_limits(frame)

    for index, stage_name in enumerate(STAGE_ORDER):
        axis = axes[index]
        stage_frame = frame[frame["stage"] == stage_name]

        axis.hexbin(
            stage_frame["proj_x"],
            stage_frame["proj_y"],
            gridsize=18,
            cmap="Greys",
            mincnt=1,
            linewidths=0,
            alpha=0.22,
        )

        for source_name in SOURCE_PATHS:
            source_frame = stage_frame[stage_frame["source"] == source_name]
            axis.scatter(
                source_frame["proj_x"],
                source_frame["proj_y"],
                s=22,
                alpha=0.75,
                color=SOURCE_COLORS[source_name],
                marker=SOURCE_MARKERS[source_name],
                linewidths=0.3,
                edgecolors="white",
                label=source_display_name(source_name, phishing_only) if index == 0 else None,
            )
            confidence_ellipse(
                axis,
                source_frame["proj_x"].to_numpy(),
                source_frame["proj_y"].to_numpy(),
                color=SOURCE_COLORS[source_name],
            )

        axis.set_title(format_stage_title(stage_frame, stage_name), fontsize=10)
        axis.set_xlim(*x_limits)
        axis.set_ylim(*y_limits)
        axis.grid(alpha=0.14, linewidth=0.5)
        axis.tick_params(axis="both", labelsize=8)
        if index == 0:
            axis.set_ylabel(
                f"Indicator Projection Axis 2\n({projection_method.upper()}-2)",
                fontsize=11,
            )
        axis.set_xlabel(f"{projection_method.upper()}-1", fontsize=9)

    handles, labels = axes[0].get_legend_handles_labels()
    figure.legend(
        handles,
        labels,
        loc="lower center",
        ncol=2,
        frameon=False,
        bbox_to_anchor=(0.5, 0.03),
    )
    figure.suptitle(
        f"Figure 1. Pattern Map in Indicator Space ({detector_column}, {projection_method.upper()})",
        y=0.96,
        fontsize=14,
    )
    figure.subplots_adjust(left=0.06, right=0.995, bottom=0.24, top=0.82, wspace=0.06)
    figure.savefig(output_path, dpi=220, bbox_inches="tight")
    plt.close(figure)


def fit_detector_surface(stage_frame: pd.DataFrame, seed: int) -> Pipeline | None:
    y_values = stage_frame["detector_prediction"].astype(int).to_numpy()
    if len(np.unique(y_values)) < 2:
        return None

    model = Pipeline(
        [
            ("poly", PolynomialFeatures(degree=2, include_bias=False)),
            ("scale", StandardScaler()),
            (
                "logreg",
                LogisticRegression(
                    max_iter=1500,
                    class_weight="balanced",
                    random_state=seed,
                ),
            ),
        ]
    )
    x_values = stage_frame[["proj_x", "proj_y"]].to_numpy()
    model.fit(x_values, y_values)
    return model


def draw_detector_coverage(
    frame: pd.DataFrame,
    *,
    output_path: Path,
    detector_column: str,
    projection_method: str,
    seed: int,
    phishing_only: bool,
) -> None:
    figure, axes = plt.subplots(1, len(STAGE_ORDER), figsize=(38, 4.8), sharex=True, sharey=True)
    x_limits, y_limits = compute_axis_limits(frame)
    x_grid = np.linspace(*x_limits, 160)
    y_grid = np.linspace(*y_limits, 160)
    grid_x, grid_y = np.meshgrid(x_grid, y_grid)
    mesh = np.column_stack([grid_x.ravel(), grid_y.ravel()])

    legend_handles = []
    legend_labels = []
    contour_fill = None

    for index, stage_name in enumerate(STAGE_ORDER):
        axis = axes[index]
        stage_frame = frame[frame["stage"] == stage_name]
        surface_model = fit_detector_surface(stage_frame, seed + index)

        if surface_model is not None:
            probabilities = surface_model.predict_proba(mesh)[:, 1].reshape(grid_x.shape)
            contour_fill = axis.contourf(
                grid_x,
                grid_y,
                probabilities,
                levels=np.linspace(0.0, 1.0, 11),
                cmap="YlOrRd",
                alpha=0.68,
            )
            axis.contour(
                grid_x,
                grid_y,
                probabilities,
                levels=[0.5],
                colors=["#4d0000"],
                linewidths=[1.4],
            )
        else:
            contour_fill = None
            axis.text(
                0.5,
                0.5,
                "single-class\nresponse",
                transform=axis.transAxes,
                ha="center",
                va="center",
                fontsize=10,
                color="#7f2704",
            )

        for source_name in SOURCE_PATHS:
            source_frame = stage_frame[stage_frame["source"] == source_name]
            negative_frame = source_frame[source_frame["detector_prediction"] < 0.5]
            positive_frame = source_frame[source_frame["detector_prediction"] >= 0.5]

            scatter_negative = axis.scatter(
                negative_frame["proj_x"],
                negative_frame["proj_y"],
                s=22,
                color=SOURCE_COLORS[source_name],
                alpha=0.35,
                marker=SOURCE_MARKERS[source_name],
                linewidths=0.3,
                edgecolors="white",
                label=source_display_name(source_name, phishing_only) if index == 0 else None,
            )
            scatter_positive = axis.scatter(
                positive_frame["proj_x"],
                positive_frame["proj_y"],
                s=28,
                color=SOURCE_COLORS[source_name],
                alpha=0.88,
                marker=SOURCE_MARKERS[source_name],
                linewidths=0.9,
                edgecolors="black",
                label=(
                    f"{source_display_name(source_name, phishing_only)} + {detector_column}=1"
                    if index == 0
                    else None
                ),
            )
            if index == 0:
                display_name = source_display_name(source_name, phishing_only)
                if display_name not in legend_labels:
                    legend_handles.append(scatter_negative)
                    legend_labels.append(display_name)
                legend_handles.append(scatter_positive)
                legend_labels.append(f"{display_name} + {detector_column}=1")

        hw_flag_rate = stage_frame.loc[stage_frame["source"] == "HW", "detector_prediction"].mean()
        llm_flag_rate = stage_frame.loc[stage_frame["source"] == "LLM", "detector_prediction"].mean()
        axis.set_title(
            f"{stage_name}\nHW-flag={hw_flag_rate:.2f} | LLM-flag={llm_flag_rate:.2f}",
            fontsize=10,
        )
        axis.set_xlim(*x_limits)
        axis.set_ylim(*y_limits)
        axis.grid(alpha=0.12, linewidth=0.5)
        axis.tick_params(axis="both", labelsize=8)
        if index == 0:
            axis.set_ylabel(
                f"Indicator Projection Axis 2\n({projection_method.upper()}-2)",
                fontsize=11,
            )
        axis.set_xlabel(f"{projection_method.upper()}-1", fontsize=9)

    if contour_fill is not None:
        colorbar = figure.colorbar(contour_fill, ax=axes, fraction=0.02, pad=0.01)
        colorbar.set_label(f"Surrogate P({detector_column}=1)")
    figure.legend(
        legend_handles,
        legend_labels,
        loc="lower center",
        ncol=4,
        frameon=False,
        bbox_to_anchor=(0.5, 0.03),
    )
    figure.suptitle(
        f"Figure 2. Surrogate Detector Field ({detector_column}, {projection_method.upper()})",
        y=0.96,
        fontsize=14,
    )
    figure.subplots_adjust(left=0.06, right=0.985, bottom=0.24, top=0.80, wspace=0.06)
    figure.savefig(output_path, dpi=220, bbox_inches="tight")
    plt.close(figure)


def draw_indicator_composition(
    frame: pd.DataFrame,
    *,
    output_path: Path,
    phishing_only: bool,
) -> None:
    figure, axes = plt.subplots(1, len(STAGE_ORDER), figsize=(38, 5.2), sharey=True)
    stage_matrices: dict[str, np.ndarray] = {}
    matrix_values: list[float] = []

    for stage_name in STAGE_ORDER:
        stage_frame = frame[frame["stage"] == stage_name]
        matrix_rows = []
        for source_name in SOURCE_PATHS:
            source_frame = stage_frame[stage_frame["source"] == source_name]
            row_values = [
                float(source_frame[f"{column}"].mean()) if not source_frame.empty else 0.0
                for column in COMPOSITION_COLUMNS
            ]
            matrix_rows.append(row_values)
        matrix = np.array(matrix_rows, dtype=float)
        stage_matrices[stage_name] = matrix
        matrix_values.extend(matrix.ravel().tolist())

    mean_abs_values = np.abs(np.array(matrix_values, dtype=float))
    z_limit = max(0.45, float(np.quantile(mean_abs_values, 0.95)))

    image_artist = None
    for index, stage_name in enumerate(STAGE_ORDER):
        axis = axes[index]
        matrix = stage_matrices[stage_name]
        image_artist = axis.imshow(
            matrix,
            aspect="auto",
            cmap="coolwarm",
            vmin=-z_limit,
            vmax=z_limit,
        )
        axis.set_title(stage_name, fontsize=10)
        axis.set_xticks(range(len(COMPOSITION_COLUMNS)))
        axis.set_xticklabels(
            [INDICATOR_DISPLAY_NAMES[column] for column in COMPOSITION_COLUMNS],
            rotation=90,
            fontsize=8,
        )
        axis.set_yticks(range(len(SOURCE_PATHS)))
        axis.set_yticklabels(
            [source_display_name(name, phishing_only) for name in SOURCE_PATHS] if index == 0 else [],
            fontsize=9,
        )

    if image_artist is not None:
        colorbar = figure.colorbar(image_artist, ax=axes, fraction=0.02, pad=0.01)
        colorbar.set_label("Mean standardized indicator value")
    figure.suptitle(
        "Figure 3. Indicator Composition by Stage (HW-phishing vs LLM-phishing)"
        if phishing_only
        else "Figure 3. Indicator Composition by Stage (HW vs LLM)",
        y=1.02,
        fontsize=14,
    )
    figure.subplots_adjust(left=0.04, right=0.985, bottom=0.47, top=0.84, wspace=0.06)
    figure.savefig(output_path, dpi=220, bbox_inches="tight")
    plt.close(figure)


def compute_cooccurrence_matrix(frame: pd.DataFrame, columns: Iterable[str]) -> np.ndarray:
    binary_matrix = frame[list(columns)].to_numpy(dtype=float)
    if binary_matrix.size == 0:
        return np.zeros((len(list(columns)), len(list(columns))))
    row_count = max(1, binary_matrix.shape[0])
    return (binary_matrix.T @ binary_matrix) / row_count


def draw_motif_delta(
    frame: pd.DataFrame,
    *,
    output_path: Path,
) -> None:
    figure, axes = plt.subplots(1, len(STAGE_ORDER), figsize=(38, 5.2), sharex=True, sharey=True)
    delta_matrices: dict[str, np.ndarray] = {}

    for stage_name in STAGE_ORDER:
        stage_frame = frame[frame["stage"] == stage_name]
        hw_matrix = compute_cooccurrence_matrix(stage_frame[stage_frame["source"] == "HW"], MOTIF_COLUMNS)
        llm_matrix = compute_cooccurrence_matrix(stage_frame[stage_frame["source"] == "LLM"], MOTIF_COLUMNS)
        delta_matrices[stage_name] = llm_matrix - hw_matrix

    max_abs_delta = max(float(np.max(np.abs(matrix))) for matrix in delta_matrices.values()) or 1.0
    max_abs_delta = max(max_abs_delta, 0.15)
    image_artist = None

    for index, stage_name in enumerate(STAGE_ORDER):
        axis = axes[index]
        matrix = delta_matrices[stage_name]
        image_artist = axis.imshow(
            matrix,
            aspect="auto",
            cmap="RdBu_r",
            vmin=-max_abs_delta,
            vmax=max_abs_delta,
        )
        axis.set_title(stage_name, fontsize=10)
        axis.set_xticks(range(len(MOTIF_COLUMNS)))
        axis.set_xticklabels(
            [MOTIF_DISPLAY_NAMES[column] for column in MOTIF_COLUMNS],
            rotation=90,
            fontsize=8,
        )
        axis.set_yticks(range(len(MOTIF_COLUMNS)))
        axis.set_yticklabels(
            [MOTIF_DISPLAY_NAMES[column] for column in MOTIF_COLUMNS] if index == 0 else [],
            fontsize=9,
        )

    if image_artist is not None:
        colorbar = figure.colorbar(image_artist, ax=axes, fraction=0.02, pad=0.01)
        colorbar.set_label("LLM - HW co-occurrence rate")
    figure.suptitle("Figure 4. Motif Co-occurrence Delta (LLM minus HW)", y=1.02, fontsize=14)
    figure.subplots_adjust(left=0.04, right=0.985, bottom=0.42, top=0.84, wspace=0.06)
    figure.savefig(output_path, dpi=220, bbox_inches="tight")
    plt.close(figure)


def write_metadata(
    frame: pd.DataFrame,
    args: argparse.Namespace,
    output_dir: Path,
    method: str,
    run_suffix: str,
) -> None:
    stage_counts = (
        frame.groupby(["stage", "source"])
        .size()
        .rename("sample_count")
        .reset_index()
        .sort_values(["stage", "source"])
    )
    stage_counts.to_csv(output_dir / f"sample_manifest_{run_suffix}.csv", index=False)

    metadata = {
        "detector_column": args.detector_column,
        "sample_size_per_stage_per_source": args.sample_size,
        "phishing_only": args.phishing_only,
        "seed": args.seed,
        "projection_requested": args.projection,
        "projection_used": method,
        "stage_order": STAGE_ORDER,
        "sources": list(SOURCE_PATHS.keys()),
        "row_count": int(len(frame)),
        "figure_files": {
            "pattern_map": f"01_pattern_map_scamllm_{run_suffix}.png",
            "detector_coverage": f"02_detector_coverage_scamllm_{run_suffix}.png",
            "indicator_composition": f"03_indicator_composition_scamllm_{run_suffix}.png",
            "motif_delta": f"04_motif_delta_scamllm_{run_suffix}.png",
        },
    }
    (output_dir / f"run_metadata_{run_suffix}.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")


def main() -> int:
    args = parse_args()
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    run_suffix = build_run_suffix(args.sample_size, args.phishing_only)

    sampled_frame = build_sampled_frame(args)
    artifacts = project_indicator_space(sampled_frame, projection=args.projection, seed=args.seed)
    frame = artifacts.frame

    sampled_csv_path = output_dir / f"sampled_scamllm_{run_suffix}.csv"
    frame.to_csv(sampled_csv_path, index=False)

    draw_pattern_map(
        frame,
        output_path=output_dir / f"01_pattern_map_scamllm_{run_suffix}.png",
        detector_column=args.detector_column,
        projection_method=artifacts.method,
        phishing_only=args.phishing_only,
    )
    draw_detector_coverage(
        frame,
        output_path=output_dir / f"02_detector_coverage_scamllm_{run_suffix}.png",
        detector_column=args.detector_column,
        projection_method=artifacts.method,
        seed=args.seed,
        phishing_only=args.phishing_only,
    )
    draw_indicator_composition(
        frame,
        output_path=output_dir / f"03_indicator_composition_scamllm_{run_suffix}.png",
        phishing_only=args.phishing_only,
    )
    draw_motif_delta(
        frame,
        output_path=output_dir / f"04_motif_delta_scamllm_{run_suffix}.png",
    )
    write_metadata(frame, args, output_dir, artifacts.method, run_suffix)

    print(f"Saved sampled CSV: {sampled_csv_path}")
    print(f"Saved figures to: {output_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
