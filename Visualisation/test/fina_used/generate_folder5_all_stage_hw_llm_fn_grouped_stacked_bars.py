#!/usr/bin/env python3
from __future__ import annotations

import json
import importlib.util
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.patches import Patch, Rectangle


SCRIPT_DIR = Path(__file__).resolve().parent
STAGES_DIR = SCRIPT_DIR.parent / "stages"
VIS_ROOT = SCRIPT_DIR.parents[1]
STAGE_FIG2_SCRIPT = VIS_ROOT / "generate_stage_fig2_fn_figures.py"
if str(VIS_ROOT) not in sys.path:
    sys.path.insert(0, str(VIS_ROOT))


def load_stage_fig2_module():
    spec = importlib.util.spec_from_file_location("stage_fig2_module", STAGE_FIG2_SCRIPT)
    if spec is None or spec.loader is None:
        raise ImportError(f"Unable to load stage fig2 script from {STAGE_FIG2_SCRIPT}")
    module = importlib.util.module_from_spec(spec)
    sys.modules.setdefault("stage_fig2_module", module)
    spec.loader.exec_module(module)
    return module


STAGE_FIG2_MODULE = load_stage_fig2_module()
compute_principle_cooccurrence_matrix = STAGE_FIG2_MODULE.compute_principle_cooccurrence_matrix

OUTPUT_DIR = SCRIPT_DIR / "5"
OUTPUT_PNG = OUTPUT_DIR / "all_stage_hw_llm_fn_grouped_stacked_bars.png"
OUTPUT_CSV = OUTPUT_DIR / "all_stage_hw_llm_fn_grouped_stacked_bars.csv"
OUTPUT_METADATA = OUTPUT_DIR / "all_stage_hw_llm_fn_grouped_stacked_bars_metadata.json"
X_AXIS_LABEL_FONT_SIZE = 33
TICK_LABEL_FONT_SIZE = 29
X_TICK_LABEL_FONT_SIZE = 34
PRINCIPLES: list[tuple[str, str, str]] = [
    ("Authority", "A", "principle_authority"),
    ("Liking", "L", "principle_liking"),
    ("Reciprocity", "R", "principle_reciprocity"),
    ("Social Proof", "SP", "principle_social_proof"),
    ("Scarcity", "SC", "principle_scarcity"),
    ("Commitment", "C", "principle_commitment"),
]
SOURCE_ORDER = ["HW", "LLM"]
STAGE_ORDER = [
    "S1",
    "S2",
    "S4",
    "S5",
    "S6-MPG",
    "S6-UTA",
    "S6-fuzzer",
    "S8-claude",
    "S8-deepseek",
    "S8-gemini",
    "S8-gpt",
    "S8-llama",
    "S8-ministral",
]
PAIR_COLORS = [
    "#143d73",
    "#2e5e95",
    "#4b7fb2",
    "#6a9cc3",
    "#88b6cf",
    "#a9cbd9",
    "#2f7f7f",
    "#5d9d8c",
    "#83b57b",
    "#a8c66b",
    "#c9cf77",
    "#d8bd74",
    "#e2aa76",
    "#e79a74",
    "#d97d62",
    "#be6851",
    "#b85a69",
    "#9b6f8f",
    "#7f73a8",
    "#676da8",
    "#4d78b7",
]


def build_pair_specs() -> list[tuple[str, str, str]]:
    pair_specs: list[tuple[str, str, str]] = []
    color_index = 0
    for index, (left_label, left_short, _) in enumerate(PRINCIPLES):
        for right_label, right_short, _ in PRINCIPLES[index:]:
            pair_specs.append(
                (
                    f"{left_label} | {right_label}",
                    f"{left_short}-{right_short}",
                    PAIR_COLORS[color_index],
                )
            )
            color_index += 1
    return pair_specs
def build_summary_table() -> pd.DataFrame:
    pair_specs = build_pair_specs()
    rows: list[dict[str, float | int | str]] = []
    for stage_name in STAGE_ORDER:
        stage_csv = STAGES_DIR / stage_name / "projected_points.csv"
        if not stage_csv.exists():
            continue
        frame = pd.read_csv(stage_csv, low_memory=False)
        for source_name in SOURCE_ORDER:
            subset = frame[(frame["source"] == source_name) & frame["is_fn"].astype(bool)].copy()
            matrix, sample_count = compute_principle_cooccurrence_matrix(subset)
            row: dict[str, float | int | str] = {
                "stage": stage_name,
                "source": source_name,
                "fn_sample_count": sample_count,
                "stack_total": 0.0,
                "source_heatmap": f"test/stages/{stage_name}/fig2_stage_fn_heatmaps_pca.png",
            }
            pair_index = 0
            for left_index, _ in enumerate(PRINCIPLES):
                for right_index in range(left_index, len(PRINCIPLES)):
                    value = float(matrix[left_index, right_index]) if sample_count else 0.0
                    value = value if np.isfinite(value) else 0.0
                    row[pair_specs[pair_index][0]] = value
                    row["stack_total"] += value
                    pair_index += 1
            rows.append(row)
    return pd.DataFrame(rows)


def draw_grouped_stacked_bars(summary: pd.DataFrame) -> None:
    pair_specs = build_pair_specs()
    pair_centers = np.arange(len(STAGE_ORDER), dtype=float) * 2.0 + 0.5
    bar_positions = {
        "HW": np.arange(0, len(STAGE_ORDER) * 2, 2, dtype=float),
        "LLM": np.arange(1, len(STAGE_ORDER) * 2, 2, dtype=float),
    }
    bar_width = 0.82

    plt.rcParams["font.family"] = "DejaVu Sans"
    figure, axis = plt.subplots(figsize=(28, 16))
    figure.patch.set_facecolor("#ffffff")
    axis.set_facecolor("#ffffff")

    max_total = float(summary["stack_total"].max()) if not summary.empty else 0.0
    y_limit = max(5.0, max_total + 0.45)

    hw_band_x = bar_positions["HW"]
    band_width = bar_width + 0.12
    for xpos in hw_band_x:
        axis.add_patch(
            Rectangle(
                (xpos - band_width / 2, 0.0),
                band_width,
                y_limit,
                facecolor="#cfcfcf",
                edgecolor="none",
                alpha=0.95,
                zorder=1.2,
            )
        )

    for source_name in SOURCE_ORDER:
        source_frame = (
            summary[summary["source"] == source_name]
            .set_index("stage")
            .reindex(STAGE_ORDER)
            .reset_index()
        )
        x_positions = bar_positions[source_name]
        bottoms = np.zeros(len(STAGE_ORDER), dtype=float)

        for label, short_label, color in pair_specs:
            heights = source_frame[label].to_numpy(dtype=float)
            axis.bar(
                x_positions,
                heights,
                width=bar_width,
                bottom=bottoms,
                color=color,
                edgecolor="#b4b4b4" if source_name == "HW" else "#fffdf8",
                linewidth=0.25 if source_name == "HW" else 0.35,
                linestyle="solid",
                label=short_label if source_name == "HW" else None,
                zorder=3,
            )
            bottoms += heights

        counts = source_frame["fn_sample_count"].fillna(0).to_numpy(dtype=int)
        for xpos, total, count in zip(x_positions, bottoms, counts, strict=True):
            continue

    for divider_index in range(len(STAGE_ORDER) - 1):
        midpoint = (pair_centers[divider_index] + pair_centers[divider_index + 1]) / 2
        axis.axvline(midpoint, color="#ddd5c9", linewidth=0.8, zorder=1)

    axis.grid(axis="y", color="#dddddd", linewidth=0.8, alpha=0.9, zorder=0)
    axis.spines["top"].set_visible(False)
    axis.spines["right"].set_visible(False)
    axis.spines["left"].set_color("#888888")
    axis.spines["bottom"].set_color("#888888")
    axis.tick_params(axis="x", length=0, pad=12, labelsize=X_TICK_LABEL_FONT_SIZE, colors="#444444")
    axis.tick_params(axis="y", colors="#444444", labelsize=TICK_LABEL_FONT_SIZE)

    axis.set_xticks(pair_centers)
    axis.set_xticklabels(STAGE_ORDER, fontsize=X_TICK_LABEL_FONT_SIZE)
    for tick_label in axis.get_xticklabels():
        tick_label.set_rotation(15)
        tick_label.set_ha("right")
        tick_label.set_rotation_mode("anchor")
    axis.set_xlim(-0.55, float(bar_positions["LLM"][-1]) + 0.55)
    axis.set_ylim(0.0, y_limit)
    axis.set_ylabel("Mean co-occurrence strength within FN emails", fontsize=36, color="#333333")
    axis.set_xlabel("")

    color_legend = axis.legend(
        ncol=3,
        loc="upper left",
        bbox_to_anchor=(0.012, 0.995),
        frameon=True,
        fontsize=30,
        handlelength=2.7,
        columnspacing=1.5,
        borderaxespad=0.35,
        facecolor="#ffffff",
        edgecolor="#cfcfcf",
    )
    axis.add_artist(color_legend)

    style_handles = [
        Patch(facecolor="#cfcfcf", edgecolor="#9a9a9a", label="HW-P FN - HW-P TP"),
        Patch(facecolor="#ffffff", edgecolor="#7f7f7f", linewidth=1.6, label="LLM-P FN - LLM-P TP"),
    ]
    style_legend = axis.legend(
        handles=style_handles,
        ncol=1,
        loc="upper left",
        bbox_to_anchor=(0.4, 0.955),
        frameon=True,
        fontsize=31,
        borderaxespad=0.35,
        facecolor="#ffffff",
        edgecolor="#cfcfcf",
        handlelength=3.0,
    )
    axis.add_artist(style_legend)

    figure.subplots_adjust(left=0.10, right=0.98, bottom=0.18, top=0.98)
    figure.savefig(OUTPUT_PNG, dpi=260, bbox_inches="tight", facecolor=figure.get_facecolor())
    plt.close(figure)


def write_outputs(summary: pd.DataFrame) -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    summary.to_csv(OUTPUT_CSV, index=False)
    metadata = {
        "source_note": "Stack heights come from the exact same upper-triangle values used by test/stages/<stage>/fig2_stage_fn_heatmaps_pca.",
        "wvae_note": "The principle columns in projected_points.csv are identical to the stage-matched WVAE merge values.",
        "stage_order": STAGE_ORDER,
        "source_order": SOURCE_ORDER,
        "principles": [label for label, _, _ in PRINCIPLES],
        "principle_groups": [full_label for full_label, _, _ in build_pair_specs()],
        "output_png": OUTPUT_PNG.name,
        "output_csv": OUTPUT_CSV.name,
        "rows": int(len(summary)),
    }
    OUTPUT_METADATA.write_text(json.dumps(metadata, indent=2), encoding="utf-8")


def main() -> None:
    summary = build_summary_table()
    if summary.empty:
        raise RuntimeError("No stage summary rows were generated for folder 5.")
    write_outputs(summary)
    draw_grouped_stacked_bars(summary)


if __name__ == "__main__":
    main()
