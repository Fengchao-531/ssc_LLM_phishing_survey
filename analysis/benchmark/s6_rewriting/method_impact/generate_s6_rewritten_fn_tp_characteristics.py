#!/usr/bin/env python3
import csv
import math
import sys
from collections import defaultdict
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import LinearSegmentedColormap, TwoSlopeNorm


csv.field_size_limit(sys.maxsize)

ROOT = Path(__file__).resolve().parent
INPUT = ROOT.parent / "A-I Differences" / "selected_llm_tp_detector_merged.csv"

OUTPUT_CSV = ROOT / "s6_rewritten_fn_tp_pair_characteristics.csv"
TOP_CSV = ROOT / "s6_rewritten_fn_tp_pair_characteristics_top.csv"
CONSISTENCY_CSV = ROOT / "s6_rewritten_fn_tp_detector_consistency.csv"
SUMMARY_MD = ROOT / "s6_rewritten_fn_tp_characteristics_summary.md"
OUTPUT_PNG = ROOT / "fig_s6_rewritten_fn_tp_pair_characteristics.png"
OUTPUT_PDF = ROOT / "fig_s6_rewritten_fn_tp_pair_characteristics.pdf"

METHODS = ["S6-fuzzer", "S6-UTA", "S6-MPG"]
METHOD_LABELS = {
    "S6-fuzzer": "Fuzzer",
    "S6-UTA": "UTA",
    "S6-MPG": "MPG",
}
DETECTORS = [
    ("securenet_llama", "SecureNet"),
    ("email_phishing_detection_v3_prediction", "V3"),
]
PRINCIPLES = [
    ("A", "Authority", "principle_authority"),
    ("L", "Liking", "principle_liking"),
    ("R", "Reciprocity", "principle_reciprocity"),
    ("SP", "Social Proof", "principle_social_proof"),
    ("S", "Scarcity", "principle_scarcity"),
    ("C", "Commitment", "principle_commitment"),
]
PAIR_ORDER = [
    "A-A",
    "A-L",
    "A-R",
    "A-SP",
    "A-S",
    "A-C",
    "L-L",
    "L-R",
    "L-SP",
    "L-S",
    "L-C",
    "R-R",
    "R-SP",
    "R-S",
    "R-C",
    "SP-SP",
    "SP-S",
    "SP-C",
    "S-S",
    "S-C",
    "C-C",
]
PAIR_LABELS = ["(" + pair.split("-", 1)[0] + ", " + pair.split("-", 1)[1] + ")" for pair in PAIR_ORDER]


def pair_specs():
    specs = []
    for left_index, left in enumerate(PRINCIPLES):
        for right in PRINCIPLES[left_index:]:
            left_short, left_name, left_col = left
            right_short, right_name, right_col = right
            specs.append(
                {
                    "pair": left_short + "-" + right_short,
                    "pair_full": left_name + " + " + right_name,
                    "left_col": left_col,
                    "right_col": right_col,
                    "diagonal": left_col == right_col,
                }
            )
    return specs


PAIR_SPECS = pair_specs()
PAIR_FULL = {spec["pair"]: spec["pair_full"] for spec in PAIR_SPECS}


def to_float(value):
    try:
        if value == "":
            return None
        return float(value)
    except Exception:
        return None


def mean(values):
    return sum(values) / len(values) if values else float("nan")


def median(values):
    if not values:
        return float("nan")
    ordered = sorted(values)
    middle = len(ordered) // 2
    if len(ordered) % 2:
        return ordered[middle]
    return (ordered[middle - 1] + ordered[middle]) / 2.0


def average_ranks(values):
    indexed = sorted(enumerate(values), key=lambda item: item[1])
    ranks = [0.0] * len(values)
    tie_counts = []
    index = 0
    while index < len(indexed):
        end = index + 1
        while end < len(indexed) and indexed[end][1] == indexed[index][1]:
            end += 1
        average_rank = (index + 1 + end) / 2.0
        for tied_index in range(index, end):
            ranks[indexed[tied_index][0]] = average_rank
        tie_counts.append(end - index)
        index = end
    return ranks, tie_counts


def mann_whitney(fn_values, tp_values):
    n_fn = len(fn_values)
    n_tp = len(tp_values)
    if n_fn == 0 or n_tp == 0:
        return float("nan"), 1.0, float("nan")

    combined = fn_values + tp_values
    ranks, tie_counts = average_ranks(combined)
    rank_sum_fn = sum(ranks[:n_fn])
    u_fn = rank_sum_fn - n_fn * (n_fn + 1) / 2.0
    effect = 2.0 * u_fn / (n_fn * n_tp) - 1.0

    total = n_fn + n_tp
    tie_term = sum(count ** 3 - count for count in tie_counts)
    if total > 1:
        variance = n_fn * n_tp / 12.0 * ((total + 1) - tie_term / (total * (total - 1)))
    else:
        variance = 0.0
    if variance <= 0:
        p_value = 1.0
    else:
        mean_u = n_fn * n_tp / 2.0
        continuity = 0.5 if u_fn > mean_u else -0.5
        z = (u_fn - mean_u - continuity) / math.sqrt(variance)
        p_value = math.erfc(abs(z) / math.sqrt(2.0))
    return u_fn, p_value, effect


def bh_adjust(rows):
    indexed = sorted(
        [(index, row["p_value"]) for index, row in enumerate(rows)],
        key=lambda item: item[1],
    )
    adjusted = [1.0] * len(rows)
    running = 1.0
    total = len(rows)
    for rank_from_end, (index, p_value) in enumerate(reversed(indexed), start=1):
        rank = total - rank_from_end + 1
        candidate = min(1.0, p_value * total / rank)
        running = min(running, candidate)
        adjusted[index] = running
    for index, row in enumerate(rows):
        row["q_value"] = adjusted[index]
        row["significant"] = adjusted[index] < 0.05


def pair_value(row, spec):
    left = to_float(row.get(spec["left_col"], ""))
    if left is None:
        left = 0.0
    if spec["diagonal"]:
        return left
    right = to_float(row.get(spec["right_col"], ""))
    if right is None:
        right = 0.0
    return left * right


def load_rows():
    groups = defaultdict(lambda: defaultdict(lambda: {"FN": [], "TP": []}))
    counts = defaultdict(lambda: {"FN": 0, "TP": 0})
    with INPUT.open(newline="", encoding="utf-8") as input_file:
        reader = csv.DictReader(input_file)
        for row in reader:
            if row.get("stage") not in METHODS:
                continue
            if row.get("source") != "LLM":
                continue
            if to_float(row.get("label_x", "")) != 1.0:
                continue
            for detector_col, detector_label in DETECTORS:
                prediction = to_float(row.get(detector_col, ""))
                if prediction is None:
                    continue
                outcome = "FN" if prediction == 0.0 else "TP"
                counts[(METHOD_LABELS[row["stage"]], detector_label)][outcome] += 1
                for spec in PAIR_SPECS:
                    groups[(METHOD_LABELS[row["stage"]], detector_label)][spec["pair"]][outcome].append(
                        pair_value(row, spec)
                    )
    return groups, counts


def build_tables():
    groups, counts = load_rows()
    rows = []
    for method in [METHOD_LABELS[stage] for stage in METHODS]:
        for _, detector_label in DETECTORS:
            family = []
            for pair in PAIR_ORDER:
                values = groups[(method, detector_label)][pair]
                fn_values = values["FN"]
                tp_values = values["TP"]
                u_stat, p_value, effect = mann_whitney(fn_values, tp_values)
                family.append(
                    {
                        "method": method,
                        "detector_label": detector_label,
                        "pair": pair,
                        "pair_full": PAIR_FULL[pair],
                        "FN_mean_score": mean(fn_values),
                        "FN_median_score": median(fn_values),
                        "TP_mean_score": mean(tp_values),
                        "TP_median_score": median(tp_values),
                        "FN_minus_TP_mean_score": mean(fn_values) - mean(tp_values),
                        "FN_minus_TP_median_score": median(fn_values) - median(tp_values),
                        "u_statistic": u_stat,
                        "p_value": p_value,
                        "rank_biserial_effect": effect,
                        "abs_rank_biserial_effect": abs(effect) if not math.isnan(effect) else float("nan"),
                        "n_FN": counts[(method, detector_label)]["FN"],
                        "n_TP": counts[(method, detector_label)]["TP"],
                    }
                )
            bh_adjust(family)
            rows.extend(family)
    rows.sort(key=lambda row: (row["method"], row["detector_label"], row["pair"]))

    fieldnames = [
        "method",
        "detector_label",
        "pair",
        "pair_full",
        "FN_mean_score",
        "FN_median_score",
        "TP_mean_score",
        "TP_median_score",
        "FN_minus_TP_mean_score",
        "FN_minus_TP_median_score",
        "u_statistic",
        "p_value",
        "q_value",
        "rank_biserial_effect",
        "abs_rank_biserial_effect",
        "n_FN",
        "n_TP",
        "significant",
    ]
    with OUTPUT_CSV.open("w", newline="", encoding="utf-8") as output_file:
        writer = csv.DictWriter(output_file, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    top_rows = [
        row
        for row in sorted(rows, key=lambda row: (row["method"], row["detector_label"], -row["abs_rank_biserial_effect"]))
        if row["significant"]
    ]
    limited = []
    seen = defaultdict(int)
    for row in top_rows:
        key = (row["method"], row["detector_label"])
        if seen[key] >= 5:
            continue
        limited.append(row)
        seen[key] += 1
    with TOP_CSV.open("w", newline="", encoding="utf-8") as output_file:
        writer = csv.DictWriter(output_file, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(limited)

    build_consistency(rows)
    write_summary(rows, limited)
    draw_figure(rows)


def build_consistency(rows):
    by_method_detector = defaultdict(list)
    for row in rows:
        if row["significant"]:
            by_method_detector[(row["method"], row["detector_label"])].append(row)

    out_rows = []
    for method in [METHOD_LABELS[stage] for stage in METHODS]:
        sn_pairs = set(row["pair"] for row in by_method_detector[(method, "SecureNet")])
        v3_pairs = set(row["pair"] for row in by_method_detector[(method, "V3")])
        overlap = sorted(sn_pairs & v3_pairs, key=lambda pair: PAIR_ORDER.index(pair))
        union = sorted(sn_pairs | v3_pairs, key=lambda pair: PAIR_ORDER.index(pair))
        out_rows.append(
            {
                "method": method,
                "SecureNet_significant_pairs": ";".join(sorted(sn_pairs, key=lambda pair: PAIR_ORDER.index(pair))),
                "V3_significant_pairs": ";".join(sorted(v3_pairs, key=lambda pair: PAIR_ORDER.index(pair))),
                "overlap_pairs": ";".join(overlap),
                "n_SecureNet": len(sn_pairs),
                "n_V3": len(v3_pairs),
                "n_overlap": len(overlap),
                "jaccard_overlap": len(overlap) / float(len(union)) if union else 0.0,
            }
        )
    with CONSISTENCY_CSV.open("w", newline="", encoding="utf-8") as output_file:
        fieldnames = [
            "method",
            "SecureNet_significant_pairs",
            "V3_significant_pairs",
            "overlap_pairs",
            "n_SecureNet",
            "n_V3",
            "n_overlap",
            "jaccard_overlap",
        ]
        writer = csv.DictWriter(output_file, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(out_rows)


def fmt(value, digits=4, signed=False):
    if value is None or (isinstance(value, float) and math.isnan(value)):
        return ""
    template = "{:+" if signed else "{:"
    return (template + "." + str(digits) + "f}").format(value)


def write_summary(rows, top_rows):
    lines = [
        "# S6 Rewritten FN-vs-TP Pair Characteristics",
        "",
        "This is the final pairing-independent S6 failure-characteristic analysis. It compares rewritten phishing rows that become false negatives with rewritten phishing rows that remain true positives, using rewritten persuasion-pair scores directly rather than original-to-rewrite deltas.",
        "",
        "Statistics: two-sided Mann-Whitney U per persuasion pair, with Benjamini-Hochberg FDR correction within each method-detector family. Positive rank-biserial effects mean the pair score is higher in FN than TP rewritten phishing.",
        "",
        "## Top Significant Pairs",
        "",
        "| Method | Detector | Pair | FN median | TP median | FN-TP median | q | Effect | n_FN | n_TP |",
        "|---|---|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in top_rows:
        lines.append(
            "| {method} | {detector_label} | {pair} | {fn} | {tp} | {diff} | {q:.3g} | {effect} | {n_FN} | {n_TP} |".format(
                method=row["method"],
                detector_label=row["detector_label"],
                pair=row["pair"],
                fn=fmt(row["FN_median_score"]),
                tp=fmt(row["TP_median_score"]),
                diff=fmt(row["FN_minus_TP_median_score"], signed=True),
                q=row["q_value"],
                effect=fmt(row["rank_biserial_effect"], digits=3, signed=True),
                n_FN=row["n_FN"],
                n_TP=row["n_TP"],
            )
        )
    lines.extend(
        [
            "",
            "## Outputs",
            "",
            "- Full table: `s6_rewritten_fn_tp_pair_characteristics.csv`",
            "- Top table: `s6_rewritten_fn_tp_pair_characteristics_top.csv`",
            "- Detector consistency table: `s6_rewritten_fn_tp_detector_consistency.csv`",
            "- Heatmap: `fig_s6_rewritten_fn_tp_pair_characteristics.png` / `.pdf`",
        ]
    )
    SUMMARY_MD.write_text("\n".join(lines) + "\n", encoding="utf-8")


def draw_figure(rows):
    plt.rcParams.update(
        {
            "font.family": "serif",
            "font.serif": ["Times New Roman", "Times", "Nimbus Roman", "DejaVu Serif"],
            "axes.linewidth": 1.0,
        }
    )
    row_order = [(METHOD_LABELS[stage], detector_label) for stage in METHODS for _, detector_label in DETECTORS]
    row_labels = [method + ", " + ("SN" if detector == "SecureNet" else detector) for method, detector in row_order]

    matrix = np.full((len(row_order), len(PAIR_ORDER)), np.nan)
    lookup = {(row["method"], row["detector_label"], row["pair"]): row for row in rows}
    for row_index, (method, detector) in enumerate(row_order):
        for col_index, pair in enumerate(PAIR_ORDER):
            row = lookup[(method, detector, pair)]
            if row["significant"]:
                matrix[row_index, col_index] = row["rank_biserial_effect"]

    masked = np.ma.masked_invalid(matrix)
    cmap = LinearSegmentedColormap.from_list(
        "s6_rewritten_fn_tp_effect",
        [
            (0.0, "#f6d681"),
            (0.47, "#fff7df"),
            (0.50, "#f7f7f7"),
            (0.53, "#eaf2f1"),
            (1.0, "#003b4d"),
        ],
    )
    cmap.set_bad("#eeeeee")
    norm = TwoSlopeNorm(vmin=-0.45, vcenter=0.0, vmax=0.45)

    figure, axis = plt.subplots(figsize=(18.8, 5.9))
    image = axis.imshow(masked, cmap=cmap, norm=norm, aspect="auto")
    axis.set_xticks(np.arange(len(PAIR_ORDER)))
    axis.set_xticklabels(PAIR_LABELS, fontsize=14.5)
    axis.set_yticks(np.arange(len(row_order)))
    axis.set_yticklabels(row_labels, fontsize=20)
    axis.tick_params(axis="x", pad=6)
    axis.tick_params(axis="y", pad=7)

    axis.set_xticks(np.arange(-0.5, len(PAIR_ORDER), 1), minor=True)
    axis.set_yticks(np.arange(-0.5, len(row_order), 1), minor=True)
    axis.grid(which="minor", color="white", linewidth=1.5)
    axis.tick_params(which="minor", bottom=False, left=False)
    for row_index in [1.5, 3.5]:
        axis.axhline(row_index, color="black", linewidth=1.0)

    for row_index in range(len(row_order)):
        for col_index in range(len(PAIR_ORDER)):
            value = matrix[row_index, col_index]
            if np.isnan(value):
                axis.text(col_index, row_index, "·", ha="center", va="center", fontsize=17, color="#666666")
            else:
                axis.text(col_index, row_index, "{:+.2f}".format(value), ha="center", va="center", fontsize=14)

    axis.set_xlabel("Persuasion pair", fontsize=21, labelpad=12)
    axis.set_ylabel("Method, detector", fontsize=21, labelpad=12)
    axis.set_title("Rewritten phishing FN-vs-TP persuasion-pair characteristics", fontsize=25, pad=18)
    colorbar = figure.colorbar(image, ax=axis, fraction=0.022, pad=0.015)
    colorbar.set_label("Rank-biserial effect size\n(gray/dot = q >= 0.05)", fontsize=19, labelpad=10)
    colorbar.ax.tick_params(labelsize=17)
    figure.subplots_adjust(left=0.115, right=0.925, bottom=0.19, top=0.82)
    figure.savefig(OUTPUT_PNG, dpi=300, bbox_inches="tight", pad_inches=0.06)
    figure.savefig(OUTPUT_PDF, bbox_inches="tight", pad_inches=0.06)
    plt.close(figure)


def main():
    build_tables()


if __name__ == "__main__":
    main()
