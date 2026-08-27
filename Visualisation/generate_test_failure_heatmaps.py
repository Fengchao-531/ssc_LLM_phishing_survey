from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

PRINCIPLE_NAMES = [
    "Authority",
    "Liking",
    "Reciprocity",
    "Social Proof",
    "Scarcity",
    "Commitment",
]
WVAE_COLUMN_BY_PRINCIPLE = {
    "Authority": "principle_authority",
    "Reciprocity": "principle_reciprocity",
    "Commitment": "principle_commitment",
    "Scarcity": "principle_scarcity",
    "Social Proof": "principle_social_proof",
    "Liking": "principle_liking",
}

SCRIPT_DIR = Path(__file__).resolve().parent
WVAE_OUTPUT_ROOT = SCRIPT_DIR / "persuasion_strategy_wvae" / "output" / "full_inference_results"

EXACT_STAGE_FILES = [
    ("HW", "S1", "HW_S1_persuasion.csv"),
    ("HW", "S2", "HW_S2_persuasion.csv"),
    ("HW", "S4", "HW_S4_persuasion.csv"),
    ("HW", "S5", "HW_S5_persuasion.csv"),
    ("LLM", "S1", "LLM_S1_persuasion.csv"),
    ("LLM", "S2", "LLM_S2_persuasion.csv"),
    ("LLM", "S4", "LLM_S4_persuasion.csv"),
    ("LLM", "S5", "LLM_S5_persuasion.csv"),
    ("LLM", "S6-MPG", "LLM_S6-MPG_persuasion.csv"),
    ("LLM", "S6-UTA", "LLM_S6-UTA_persuasion.csv"),
    ("LLM", "S6-fuzzer", "LLM_S6-fuzzer_persuasion.csv"),
    ("LLM", "S8-claude", "LLM_S8-claude_persuasion.csv"),
    ("LLM", "S8-deepseek", "LLM_S8-deepseek_persuasion.csv"),
    ("LLM", "S8-gemini", "LLM_S8-gemini_persuasion.csv"),
    ("LLM", "S8-gpt", "LLM_S8-gpt_persuasion.csv"),
    ("LLM", "S8-llama", "LLM_S8-llama_persuasion.csv"),
    ("LLM", "S8-ministral", "LLM_S8-ministral_persuasion.csv"),
]
SPLIT_STAGE_FILES = [
    ("HW", "HW_S6_persuasion.csv"),
    ("HW", "HW_S8_persuasion.csv"),
    ("LLM", "LLM_S6_persuasion.csv"),
    ("LLM", "LLM_S8_persuasion.csv"),
]


def _normalize_text(series: pd.Series) -> pd.Series:
    return series.fillna("").astype(str).str.strip()


def _build_key(frame: pd.DataFrame) -> pd.Series:
    return _normalize_text(frame["subject"]) + "\n||\n" + _normalize_text(frame["body"])


def _required_columns() -> set[str]:
    return {"subject", "body", *WVAE_COLUMN_BY_PRINCIPLE.values()}


def _load_exact_stage_rows(csv_path: Path, source_name: str, stage_name: str) -> pd.DataFrame:
    frame = pd.read_csv(csv_path)
    missing = sorted(_required_columns().difference(frame.columns))
    if missing:
        raise ValueError(f"{csv_path} is missing required columns: {', '.join(missing)}")
    frame = frame.copy()
    frame["source"] = source_name
    frame["stage"] = stage_name
    frame["merge_key"] = _build_key(frame)
    return frame[["merge_key", "subject", "body", "source", "stage", *WVAE_COLUMN_BY_PRINCIPLE.values()]]


def _load_split_stage_rows(csv_path: Path, source_name: str) -> pd.DataFrame:
    frame = pd.read_csv(csv_path)
    missing = sorted(_required_columns().union({"source_file"}).difference(frame.columns))
    if missing:
        raise ValueError(f"{csv_path} is missing required columns: {', '.join(missing)}")
    frame = frame.copy()
    frame["source"] = source_name
    frame["stage"] = frame["source_file"].astype(str).str.replace(".csv", "", regex=False)
    frame["merge_key"] = _build_key(frame)
    return frame[["merge_key", "subject", "body", "source", "stage", *WVAE_COLUMN_BY_PRINCIPLE.values()]]


def build_wvae_stage_frame() -> pd.DataFrame:
    frames = []
    exact_stage_names = set()
    for source_name, stage_name, filename in EXACT_STAGE_FILES:
        csv_path = WVAE_OUTPUT_ROOT / filename
        if not csv_path.exists():
            continue
        frames.append(_load_exact_stage_rows(csv_path, source_name, stage_name))
        exact_stage_names.add((source_name, stage_name))

    for source_name, filename in SPLIT_STAGE_FILES:
        csv_path = WVAE_OUTPUT_ROOT / filename
        if not csv_path.exists():
            continue
        split_frame = _load_split_stage_rows(csv_path, source_name)
        split_frame = split_frame[
            ~split_frame[["source", "stage"]].apply(tuple, axis=1).isin(exact_stage_names)
        ]
        if not split_frame.empty:
            frames.append(split_frame)

    if not frames:
        raise FileNotFoundError(f"No WVAE persuasion CSVs found under {WVAE_OUTPUT_ROOT}")

    merged = pd.concat(frames, ignore_index=True)
    merged = merged.drop_duplicates(subset=["merge_key", "source", "stage"], keep="first")
    return merged.reset_index(drop=True)


def merge_projected_with_wvae(projected_frame: pd.DataFrame) -> pd.DataFrame:
    projected = projected_frame.copy()
    required = {"subject", "body", "source", "stage"}
    missing = sorted(required.difference(projected.columns))
    if missing:
        raise ValueError(f"Projected frame is missing required columns: {', '.join(missing)}")
    projected["merge_key"] = _build_key(projected)
    wvae = build_wvae_stage_frame()
    merged = projected.merge(
        wvae[["merge_key", "source", "stage", *WVAE_COLUMN_BY_PRINCIPLE.values()]],
        on=["merge_key", "source", "stage"],
        how="left",
    )
    return merged


def compute_principle_matrix(
    merged_frame: pd.DataFrame,
    row_order: list[str],
    row_column: str = "group",
) -> tuple[np.ndarray, list[int]]:
    matrix_rows = []
    counts = []
    for row_name in row_order:
        row_frame = merged_frame[merged_frame[row_column] == row_name]
        principle_frame = row_frame[list(WVAE_COLUMN_BY_PRINCIPLE.values())].dropna(how="all")
        counts.append(int(len(principle_frame)))
        if principle_frame.empty:
            matrix_rows.append([np.nan] * len(PRINCIPLE_NAMES))
            continue
        matrix_rows.append(
            [float(principle_frame[WVAE_COLUMN_BY_PRINCIPLE[principle_name]].mean()) for principle_name in PRINCIPLE_NAMES]
        )
    return np.array(matrix_rows, dtype=float), counts


def draw_principle_heatmap(
    matrix: np.ndarray,
    row_labels: list[str],
    output_path: Path,
    title: str,
    *,
    count_labels: list[int] | None = None,
    no_data_note: str | None = None,
    colorbar_label: str = "Mean principle probability",
):
    safe_matrix = np.nan_to_num(matrix, nan=0.0, posinf=0.0, neginf=0.0)
    figure_height = max(3.2, 1.15 + 0.95 * len(row_labels))
    figure, axis = plt.subplots(figsize=(10.8, figure_height))
    image = axis.imshow(safe_matrix, aspect="auto", cmap="YlOrRd", vmin=0.0, vmax=1.0)

    if count_labels is not None:
        y_labels = [f"{label}  (n={count})" for label, count in zip(row_labels, count_labels, strict=True)]
    else:
        y_labels = row_labels
    axis.set_yticks(range(len(y_labels)))
    axis.set_yticklabels(y_labels, fontsize=10)
    axis.set_xticks(range(len(PRINCIPLE_NAMES)))
    axis.set_xticklabels(PRINCIPLE_NAMES, rotation=35, ha="right", fontsize=10)

    for row_index in range(safe_matrix.shape[0]):
        for col_index in range(safe_matrix.shape[1]):
            label = "NA" if np.isnan(matrix[row_index, col_index]) else f"{matrix[row_index, col_index]:.2f}"
            axis.text(
                col_index,
                row_index,
                label,
                ha="center",
                va="center",
                fontsize=8,
                color="black",
            )

    if no_data_note:
        axis.text(
            0.5,
            -0.26,
            no_data_note,
            transform=axis.transAxes,
            ha="center",
            va="top",
            fontsize=9,
            color="#6a6a6a",
        )

    colorbar = figure.colorbar(image, ax=axis, fraction=0.03, pad=0.02)
    colorbar.set_label(colorbar_label)
    figure.suptitle(title, y=0.98, fontsize=15)
    figure.tight_layout(rect=(0, 0.03, 1, 0.94))
    figure.savefig(output_path, dpi=220, bbox_inches="tight")
    plt.close(figure)


def draw_fig5_indicator_heatmap(frame: pd.DataFrame, output_path: Path, group_order: list[str]):
    merged = merge_projected_with_wvae(frame)
    matrix, counts = compute_principle_matrix(merged, group_order, row_column="group")
    no_data_note = None
    if any(count == 0 for count in counts):
        no_data_note = "Some stage-matched WVAE outputs are not available yet, so rows with no matched persuasion scores show as NA."
    draw_principle_heatmap(
        matrix,
        group_order,
        output_path,
        "Figure 5. Cialdini Persuasion Principle Comparison",
        count_labels=counts,
        no_data_note=no_data_note,
    )
