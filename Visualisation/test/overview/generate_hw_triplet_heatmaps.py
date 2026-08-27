#!/usr/bin/env python3
from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.colors import Normalize

SCRIPT_DIR = Path(__file__).resolve().parent
MIXED_INPUT = SCRIPT_DIR / "projected_points_mixed_overview.csv"
OUTPUT_PATH = SCRIPT_DIR / "fig_hw_phishing_fn_tp_benign_tn_heatmaps_pca.png"
METADATA_PATH = SCRIPT_DIR / "fig_hw_phishing_fn_tp_benign_tn_heatmaps_metadata.json"
PRINCIPLE_COLUMNS = [
    "principle_authority",
    "principle_liking",
    "principle_reciprocity",
    "principle_social_proof",
    "principle_scarcity",
    "principle_commitment",
]


def compute_principle_cooccurrence_matrix(frame: pd.DataFrame) -> tuple[np.ndarray, int]:
    principle_frame = frame[PRINCIPLE_COLUMNS].dropna(how="all")
    if principle_frame.empty:
        return np.full((len(PRINCIPLE_COLUMNS), len(PRINCIPLE_COLUMNS)), np.nan, dtype=float), 0
    values = principle_frame.to_numpy(dtype=float)
    matrix = (values.T @ values) / max(1, values.shape[0])
    diagonal = np.nanmean(values, axis=0)
    np.fill_diagonal(matrix, diagonal)
    return matrix, int(values.shape[0])


def main() -> None:
    frame = pd.read_csv(MIXED_INPUT)
    groups = [
        ("HW-P FN", frame[(frame["source"] == "HW") & frame["is_fn_phishing"].astype(bool)].copy()),
        ("HW-P TP", frame[(frame["source"] == "HW") & frame["is_tp_phishing"].astype(bool)].copy()),
        ("HW-B TN", frame[(frame["source"] == "HW") & frame["is_tn_benign"].astype(bool)].copy()),
    ]
    labels = [name.replace("principle_", "").replace("_", " ").title() for name in PRINCIPLE_COLUMNS]

    figure = plt.figure(figsize=(16.6, 5.8))
    grid = figure.add_gridspec(1, 4, width_ratios=[1.0, 1.0, 1.0, 0.08], wspace=0.05)
    axes = [figure.add_subplot(grid[0, i]) for i in range(3)]
    colorbar_axis = figure.add_subplot(grid[0, 3])

    counts: dict[str, int] = {}
    images = []
    for axis, (title, subset) in zip(axes, groups, strict=True):
        matrix, count = compute_principle_cooccurrence_matrix(subset)
        counts[title] = count
        safe_matrix = np.nan_to_num(matrix, nan=0.0, posinf=0.0, neginf=0.0)
        image = axis.imshow(safe_matrix, aspect="equal", cmap="GnBu", norm=Normalize(vmin=0.0, vmax=1.0))
        images.append(image)
        axis.set_xticks(range(len(PRINCIPLE_COLUMNS)))
        axis.set_xticklabels(labels, rotation=45, ha="right", fontsize=8.2)
        axis.set_yticks(range(len(PRINCIPLE_COLUMNS)))
        axis.set_yticklabels(labels if title == "HW-P FN" else [], fontsize=8.2)
        axis.set_title(title, fontsize=12)
        for row_index in range(safe_matrix.shape[0]):
            for col_index in range(safe_matrix.shape[1]):
                label = "NA" if np.isnan(matrix[row_index, col_index]) else f"{matrix[row_index, col_index]:.2f}"
                axis.text(col_index, row_index, label, ha="center", va="center", fontsize=7.2, color="black")

    figure.colorbar(images[-1], cax=colorbar_axis).set_label("Mean co-occurrence strength")
    figure.suptitle("Overview HW Heatmaps: HW-P FN, HW-P TP, and HW-B TN", y=0.98, fontsize=15)
    figure.subplots_adjust(left=0.05, right=0.95, bottom=0.10, top=0.88)
    figure.savefig(OUTPUT_PATH, dpi=220)
    plt.close(figure)

    metadata = {
        "mixed_input": str(MIXED_INPUT),
        "output_file": OUTPUT_PATH.name,
        "counts": {
            "hw_p_fn": int(counts["HW-P FN"]),
            "hw_p_tp": int(counts["HW-P TP"]),
            "hw_b_tn": int(counts["HW-B TN"]),
        },
    }
    METADATA_PATH.write_text(json.dumps(metadata, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
