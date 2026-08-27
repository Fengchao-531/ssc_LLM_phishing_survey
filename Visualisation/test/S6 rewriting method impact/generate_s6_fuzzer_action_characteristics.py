#!/usr/bin/env python3
import csv
import json
import math
import re
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import LinearSegmentedColormap


ROOT = Path(__file__).resolve().parent
PROJECT = ROOT.parents[2]
MERGED = ROOT.parent / "A-I Differences" / "selected_llm_tp_detector_merged.csv"
S6_DATA = PROJECT / "Datasets" / "sublist" / "S6-Stealthy Rewriting"
FUZZER_LLM_P = S6_DATA / "fuzzer-LLM-P.csv"
EMAILS_NORMALIZED = S6_DATA / "emails_normalized.json"
EVAL_LLM_FUZZER = PROJECT / "Evaluation" / "processed-evaluation-datasets" / "llm" / "academic" / "S6-fuzzer.csv"
EVAL_GD_FUZZER = PROJECT / "Evaluation" / "processed-evaluation-datasets" / "gd" / "academic" / "S6-fuzzer.csv"
LLM_PERSUASION = PROJECT / "Visualization" / "persuasion_strategy_wvae" / "output" / "full_inference_results" / "LLM_S6-fuzzer_persuasion.csv"
HW_PERSUASION = PROJECT / "Visualization" / "persuasion_strategy_wvae" / "output" / "full_inference_results" / "HW_S6_persuasion.csv"

INPUT_ROWS_CSV = ROOT / "s6_fuzzer_action_characteristics_input_rows.csv"
FN_TP_CSV = ROOT / "s6_fuzzer_fn_tp_action_characteristics.csv"
AUDIT_CSV = ROOT / "s6_fuzzer_original_rewrite_mapping_audit.csv"
AUDIT_MD = ROOT / "s6_fuzzer_original_rewrite_mapping_audit.md"
SUMMARY_MD = ROOT / "s6_fuzzer_action_characteristics_summary.md"
FIGURE_PNG = ROOT / "fig_s6_fuzzer_fn_tp_action_characteristics.png"
FIGURE_PDF = ROOT / "fig_s6_fuzzer_fn_tp_action_characteristics.pdf"

DETECTORS = [
    ("SecureNet", "securenet_llama"),
    ("V3", "email_phishing_detection_v3_prediction"),
]

FEATURES = [
    (
        "Urgency",
        "Urgency",
        r"\b(urgent|immediately|as soon as possible|asap|deadline|expires?|expiring|suspended?|locked|limited time|within \d+|final notice|act now)\b",
    ),
    (
        "Login/account",
        "Login/\naccount",
        r"\b(log ?in|sign ?in|account|password|credential|username|authentication|verify your account|account verification|account update)\b",
    ),
    (
        "Information submission",
        "Info\nsubmission",
        r"\b(submit|provide|send|enter|input|fill|complete|confirm|verify|update).{0,45}\b(info|information|details|credential|password|account|address|payment|card|code|otp)\b",
    ),
    (
        "Click/open",
        "Click/\nopen",
        r"\b(click|tap|open|visit|follow (the )?link|use (the )?link|press (the )?button|button below|link below)\b",
    ),
    (
        "Explicit action",
        "Explicit\naction",
        r"\b(click|tap|open|visit|go to|follow|log ?in|sign ?in|submit|provide|enter|verify|confirm|update|download|review|complete)\b",
    ),
    (
        "Direct URL/page instruction",
        "URL/page\ninstruction",
        r"\b(https?://|www\.|url|link|webpage|website|portal|page|site|landing page|dashboard)\b",
    ),
    (
        "Conversational wording",
        "Conversational\nwording",
        r"\b(hi|hello|hey|dear|thanks|thank you|hope you|checking in|following up|best regards|regards|cheers)\b",
    ),
    (
        "Softened request",
        "Softened\nrequest",
        r"\b(please|kindly|could you|would you|when you have a chance|at your convenience|we would appreciate|if possible|just wanted|quick note)\b",
    ),
]

PAIR_COLS = {
    "A-S": ("principle_authority", "principle_scarcity"),
    "S-S": ("principle_scarcity", "principle_scarcity"),
    "SP-S": ("principle_social_proof", "principle_scarcity"),
}


def read_csv(path):
    with path.open(newline="", encoding="utf-8", errors="replace") as handle:
        return list(csv.DictReader(handle))


def normalize_text(value):
    return re.sub(r"\s+", " ", (value or "")).strip().lower()


def text_key(row, subject_col, body_col):
    subject = row.get(subject_col, "") or ""
    body = row.get(body_col, "") or ""
    return normalize_text(str(subject) + " " + str(body))


def to_float(value):
    try:
        if value is None or str(value).strip() == "":
            return None
        return float(value)
    except ValueError:
        return None


def to_prediction(value):
    parsed = to_float(value)
    if parsed is None:
        return None
    return 1 if parsed >= 0.5 else 0


def normal_two_sided_pvalue(z_score):
    return math.erfc(abs(z_score) / math.sqrt(2.0))


def two_proportion_pvalue(success_a, n_a, success_b, n_b):
    if n_a == 0 or n_b == 0:
        return 1.0
    pooled = float(success_a + success_b) / float(n_a + n_b)
    standard_error = math.sqrt(pooled * (1.0 - pooled) * (1.0 / n_a + 1.0 / n_b))
    if standard_error == 0.0:
        return 1.0
    rate_a = float(success_a) / float(n_a)
    rate_b = float(success_b) / float(n_b)
    return normal_two_sided_pvalue((rate_a - rate_b) / standard_error)


def bh_fdr(p_values):
    indexed = sorted(enumerate(p_values), key=lambda item: item[1])
    adjusted = [1.0] * len(p_values)
    running_min = 1.0
    total = len(p_values)
    for rank, (original_index, p_value) in reversed(list(enumerate(indexed, start=1))):
        running_min = min(running_min, p_value * total / rank)
        adjusted[original_index] = min(running_min, 1.0)
    return adjusted


def stars(q_value):
    if q_value < 0.001:
        return "***"
    if q_value < 0.01:
        return "**"
    if q_value < 0.05:
        return "*"
    return ""


def feature_values(subject, body):
    text = subject + "\n" + body
    values = {}
    for name, _, pattern in FEATURES:
        values[name] = 1 if re.search(pattern, text, flags=re.IGNORECASE) else 0
    return values


def load_fuzzer_rows():
    rows = []
    for row in read_csv(MERGED):
        if row.get("stage") != "S6-fuzzer":
            continue
        if row.get("source") != "LLM":
            continue
        if to_prediction(row.get("label_x")) != 1:
            continue
        rows.append(row)
    return rows


def write_input_rows(rows):
    fieldnames = [
        "sample_id",
        "subject",
        "body",
        "text",
        "principle_authority",
        "principle_reciprocity",
        "principle_commitment",
        "principle_scarcity",
        "principle_social_proof",
        "principle_liking",
        "A-S_score",
        "S-S_score",
        "SP-S_score",
        "securenet_llama_prediction",
        "email_phishing_detection_v3_prediction",
    ] + [name for name, _, _ in FEATURES]
    with INPUT_ROWS_CSV.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for idx, row in enumerate(rows):
            subject = row.get("subject", "")
            body = row.get("body", "")
            out = {
                "sample_id": "S6-fuzzer-LLM-{:04d}".format(idx),
                "subject": subject,
                "body": body,
                "text": (subject + "\n" + body).strip(),
                "principle_authority": row.get("principle_authority", ""),
                "principle_reciprocity": row.get("principle_reciprocity", ""),
                "principle_commitment": row.get("principle_commitment", ""),
                "principle_scarcity": row.get("principle_scarcity", ""),
                "principle_social_proof": row.get("principle_social_proof", ""),
                "principle_liking": row.get("principle_liking", ""),
                "securenet_llama_prediction": row.get("securenet_llama", ""),
                "email_phishing_detection_v3_prediction": row.get("email_phishing_detection_v3_prediction", ""),
            }
            for pair, (left, right) in PAIR_COLS.items():
                left_value = to_float(row.get(left))
                right_value = to_float(row.get(right))
                out[pair + "_score"] = "" if left_value is None or right_value is None else left_value * right_value
            out.update(feature_values(subject, body))
            writer.writerow(out)


def compute_fn_tp(rows):
    result_rows = []
    p_values = []
    for detector_name, detector_col in DETECTORS:
        usable = []
        for row in rows:
            pred = to_prediction(row.get(detector_col))
            if pred in (0, 1):
                usable.append((row, pred))
        for feature, _, _ in FEATURES:
            fn_values = []
            tp_values = []
            for row, pred in usable:
                value = feature_values(row.get("subject", ""), row.get("body", ""))[feature]
                if pred == 0:
                    fn_values.append(value)
                else:
                    tp_values.append(value)
            n_fn = len(fn_values)
            n_tp = len(tp_values)
            fn_count = sum(fn_values)
            tp_count = sum(tp_values)
            fn_prev = float(fn_count) / n_fn if n_fn else 0.0
            tp_prev = float(tp_count) / n_tp if n_tp else 0.0
            p_value = two_proportion_pvalue(fn_count, n_fn, tp_count, n_tp)
            p_values.append(p_value)
            result_rows.append(
                {
                    "detector": detector_name,
                    "characteristic": feature,
                    "FN_prevalence": fn_prev,
                    "TP_prevalence": tp_prev,
                    "FN_minus_TP_pp": (fn_prev - tp_prev) * 100.0,
                    "p_value": p_value,
                    "q_value": 1.0,
                    "n_FN": n_fn,
                    "n_TP": n_tp,
                }
            )
    q_values = bh_fdr(p_values)
    for row, q_value in zip(result_rows, q_values):
        row["q_value"] = q_value
        row["stars"] = stars(q_value)
    with FN_TP_CSV.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=[
                "detector",
                "characteristic",
                "FN_prevalence",
                "TP_prevalence",
                "FN_minus_TP_pp",
                "p_value",
                "q_value",
                "stars",
                "n_FN",
                "n_TP",
            ],
        )
        writer.writeheader()
        writer.writerows(result_rows)
    return result_rows


def draw_fn_tp_heatmap(result_rows):
    detectors = [name for name, _ in DETECTORS]
    feature_names = [name for name, _, _ in FEATURES]
    feature_labels = [label for _, label, _ in FEATURES]
    values = np.zeros((len(detectors), len(feature_names)))
    star_values = [["" for _ in feature_names] for _ in detectors]
    lookup = {(row["detector"], row["characteristic"]): row for row in result_rows}
    for y, detector in enumerate(detectors):
        for x, feature in enumerate(feature_names):
            row = lookup[(detector, feature)]
            values[y, x] = row["FN_minus_TP_pp"]
            star_values[y][x] = row["stars"]

    plt.rcParams.update(
        {
            "font.family": "serif",
            "font.serif": ["Times New Roman", "Times", "Nimbus Roman", "DejaVu Serif"],
            "axes.linewidth": 1.05,
            "xtick.major.width": 0.9,
            "ytick.major.width": 0.9,
        }
    )
    cmap = LinearSegmentedColormap.from_list(
        "s6_s8_aligned_difference",
        ["#f3c957", "#fff1bf", "#ffffff", "#6f9cac", "#003b4d"],
    )

    limit = max(5.0, float(np.nanmax(np.abs(values))))
    fig, ax = plt.subplots(figsize=(16.2, 5.8))
    image = ax.imshow(values, cmap=cmap, vmin=-limit, vmax=limit, aspect="auto")
    ax.set_xticks(np.arange(len(feature_names)))
    ax.set_xticklabels(feature_labels, fontsize=24, rotation=0, ha="center")
    ax.set_yticks(np.arange(len(detectors)))
    ax.set_yticklabels(detectors, fontsize=28)
    ax.tick_params(axis="x", pad=11)
    ax.tick_params(axis="y", pad=10)
    ax.set_xlabel("Linguistic/action feature", fontsize=28, labelpad=18)
    ax.set_ylabel("Detector", fontsize=28, labelpad=18)
    ax.set_xticks(np.arange(-0.5, len(feature_names), 1), minor=True)
    ax.set_yticks(np.arange(-0.5, len(detectors), 1), minor=True)
    ax.grid(which="minor", color="white", linestyle="-", linewidth=2.0)
    ax.tick_params(which="minor", bottom=False, left=False)
    for y in range(values.shape[0]):
        for x in range(values.shape[1]):
            label = "{:+.1f}{}".format(values[y, x], star_values[y][x])
            ax.text(x, y, label, ha="center", va="center", fontsize=25, color="black")
    cbar = fig.colorbar(image, ax=ax, fraction=0.028, pad=0.018)
    cbar.set_label("FN - TP prevalence (pp)", fontsize=26, labelpad=15)
    cbar.ax.tick_params(labelsize=23)
    fig.subplots_adjust(left=0.11, right=0.925, bottom=0.32, top=0.94)
    fig.savefig(str(FIGURE_PNG), dpi=600, bbox_inches="tight", pad_inches=0.08)
    fig.savefig(str(FIGURE_PDF), bbox_inches="tight", pad_inches=0.06)
    plt.close(fig)


def count_json_list(path):
    with path.open(encoding="utf-8", errors="replace") as handle:
        data = json.load(handle)
    return len(data), set(data[0].keys()) if data else set()


def audit_mapping(rows):
    candidates = [
        ("fuzzer-LLM-P.csv", FUZZER_LLM_P, "Subject", "Body", "candidate rewritten Fuzzer phishing rows"),
        ("emails_normalized.json", EMAILS_NORMALIZED, "Subject", "Body", "normalized Fuzzer rows; JSON list"),
        ("Evaluation llm/academic/S6-fuzzer.csv", EVAL_LLM_FUZZER, "subject", "body", "processed LLM Fuzzer rows"),
        ("Evaluation gd/academic/S6-fuzzer.csv", EVAL_GD_FUZZER, "subject", "body", "processed GD/HW Fuzzer rows"),
        ("LLM_S6-fuzzer_persuasion.csv", LLM_PERSUASION, "subject", "body", "persuasion-scored LLM Fuzzer rows"),
        ("HW_S6_persuasion.csv", HW_PERSUASION, "subject", "body", "all persuasion-scored HW-side S6 rows"),
    ]
    rewrite_keys = set(text_key(row, "subject", "body") for row in rows)
    audit_rows = []
    for name, path, subject_col, body_col, description in candidates:
        if not path.exists():
            audit_rows.append(
                {
                    "candidate_file": name,
                    "exists": "no",
                    "row_count": 0,
                    "unique_text_keys": 0,
                    "overlap_with_merged_s6_fuzzer_llm": 0,
                    "has_original_subject_body": "no",
                    "has_parent_or_row_id": "no",
                    "mapping_status": "missing file",
                    "note": description,
                }
            )
            continue
        if path.suffix.lower() == ".json":
            with path.open(encoding="utf-8", errors="replace") as handle:
                data = json.load(handle)
            file_rows = data if isinstance(data, list) else []
            fieldnames = set(file_rows[0].keys()) if file_rows else set()
        else:
            file_rows = read_csv(path)
            fieldnames = set(file_rows[0].keys()) if file_rows else set()
        keys = set(text_key(row, subject_col, body_col) for row in file_rows)
        has_original = {"original_subject", "original_body"}.issubset(fieldnames)
        has_id = bool({"parent_id", "sample_id", "row_id", "original_id"} & fieldnames)
        status = "text-overlap only; no original-to-rewrite key"
        if has_original:
            status = "contains original_subject/original_body fields"
        elif has_id:
            status = "contains candidate id field but no original text fields"
        audit_rows.append(
            {
                "candidate_file": name,
                "exists": "yes",
                "row_count": len(file_rows),
                "unique_text_keys": len(keys),
                "overlap_with_merged_s6_fuzzer_llm": len(keys & rewrite_keys),
                "has_original_subject_body": "yes" if has_original else "no",
                "has_parent_or_row_id": "yes" if has_id else "no",
                "mapping_status": status,
                "note": description,
            }
        )
    with AUDIT_CSV.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=[
                "candidate_file",
                "exists",
                "row_count",
                "unique_text_keys",
                "overlap_with_merged_s6_fuzzer_llm",
                "has_original_subject_body",
                "has_parent_or_row_id",
                "mapping_status",
                "note",
            ],
        )
        writer.writeheader()
        writer.writerows(audit_rows)
    return audit_rows


def write_markdown(rows, result_rows, audit_rows):
    best_by_detector = {}
    for detector_name, _ in DETECTORS:
        detector_rows = [row for row in result_rows if row["detector"] == detector_name]
        best_by_detector[detector_name] = sorted(
            detector_rows, key=lambda row: abs(row["FN_minus_TP_pp"]), reverse=True
        )[:3]

    audit_lines = [
        "# S6 Fuzzer action/linguistic characteristics",
        "",
        "This analysis reuses the S8 regex-based action/linguistic feature framework, with the x-axis ordered as: Urgency, Login/account, Information submission, Click/open, Explicit action, Direct URL/page instruction, Conversational wording, Softened request.",
        "",
        "## Completed output",
        "",
        "- `s6_fuzzer_action_characteristics_input_rows.csv`: 2,200 S6-fuzzer LLM phishing rows with persuasion scores, detector predictions, and 8 binary characteristics.",
        "- `s6_fuzzer_fn_tp_action_characteristics.csv`: FN-vs-TP characteristic comparison for SecureNet and V3.",
        "- `fig_s6_fuzzer_fn_tp_action_characteristics.png` / `.pdf`: detector by characteristic heatmap, cell value = FN minus TP prevalence in percentage points, stars = BH-FDR q<0.05.",
        "",
        "## Mapping status",
        "",
        "The current local files do not expose a verified Fuzzer original-to-rewrite key. Therefore the requested paired original/rewrite prevalence table and scarcity-mechanism matrix are not computed here; doing so by row order would be an unsupported controlled-pair assumption.",
        "",
        "| candidate_file | rows | overlap_with_rewrite_rows | original_fields | id_field | status |",
        "|---|---:|---:|---|---|---|",
    ]
    for row in audit_rows:
        audit_lines.append(
            "| {candidate_file} | {row_count} | {overlap_with_merged_s6_fuzzer_llm} | {has_original_subject_body} | {has_parent_or_row_id} | {mapping_status} |".format(
                **row
            )
        )
    audit_lines.extend(
        [
            "",
            "## FN-vs-TP highlights",
            "",
        ]
    )
    for detector_name in [name for name, _ in DETECTORS]:
        audit_lines.append("### {}".format(detector_name))
        for row in best_by_detector[detector_name]:
            audit_lines.append(
                "- {characteristic}: FN {fn:.3f}, TP {tp:.3f}, gap {gap:+.1f} pp, q={q:.3g}".format(
                    characteristic=row["characteristic"],
                    fn=row["FN_prevalence"],
                    tp=row["TP_prevalence"],
                    gap=row["FN_minus_TP_pp"],
                    q=row["q_value"],
                )
            )
        audit_lines.append("")
    SUMMARY_MD.write_text("\n".join(audit_lines), encoding="utf-8")

    audit_md = [
        "# S6 Fuzzer original-rewrite mapping audit",
        "",
        "Goal: determine whether the current local package supports paired original -> Fuzzer rewrite analysis for S-S/A-S/SP-S scarcity mechanisms.",
        "",
        "Conclusion: no verified Fuzzer original-to-rewrite mapping is present in the inspected files. The LLM Fuzzer rows are scored and detector-labeled, but the available candidate files provide text overlap rather than a durable sample_id/parent_id/original_subject/original_body relation.",
        "",
        "Implication: CSV1 and CSV2 requested for paired original/rewrite deltas should wait for a true mapping file. The generated FN/TP analysis is valid because it uses only rewritten Fuzzer rows and detector outcomes.",
        "",
        "| candidate_file | rows | unique_text_keys | overlap_with_rewrite_rows | original_fields | id_field | status |",
        "|---|---:|---:|---:|---|---|---|",
    ]
    for row in audit_rows:
        audit_md.append(
            "| {candidate_file} | {row_count} | {unique_text_keys} | {overlap_with_merged_s6_fuzzer_llm} | {has_original_subject_body} | {has_parent_or_row_id} | {mapping_status} |".format(
                **row
            )
        )
    AUDIT_MD.write_text("\n".join(audit_md), encoding="utf-8")


def main():
    csv.field_size_limit(sys.maxsize)
    rows = load_fuzzer_rows()
    write_input_rows(rows)
    result_rows = compute_fn_tp(rows)
    audit_rows = audit_mapping(rows)
    draw_fn_tp_heatmap(result_rows)
    write_markdown(rows, result_rows, audit_rows)
    print("Wrote {}".format(INPUT_ROWS_CSV))
    print("Wrote {}".format(FN_TP_CSV))
    print("Wrote {}".format(AUDIT_CSV))
    print("Wrote {}".format(AUDIT_MD))
    print("Wrote {}".format(SUMMARY_MD))
    print("Wrote {}".format(FIGURE_PNG))
    print("Wrote {}".format(FIGURE_PDF))


if __name__ == "__main__":
    main()
