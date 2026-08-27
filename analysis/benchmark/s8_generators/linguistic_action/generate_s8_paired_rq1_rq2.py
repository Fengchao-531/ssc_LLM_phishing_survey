#!/usr/bin/env python3
import csv
import math
import random
import re
from collections import defaultdict
from pathlib import Path


ROOT = Path(__file__).resolve().parent
REPO_ROOT = ROOT.parents[2]
SEED = 20260825

GENERATOR_ORDER = ["Claude", "GPT", "Gemini", "Llama", "Ministral", "DeepSeek"]
GENERATOR_STAGE = {
    "Claude": "S8-claude",
    "GPT": "S8-gpt",
    "Gemini": "S8-gemini",
    "Llama": "S8-llama",
    "Ministral": "S8-ministral",
    "DeepSeek": "S8-deepseek",
}
RAW_PATHS = {
    "Claude": REPO_ROOT / "Datasets/sublist/S8-Model-driven Automation/Models-Output/fill-bracket-api-generators-selected/claude-sonnet-4-generated_output.fill_bracket.csv",
    "GPT": REPO_ROOT / "Datasets/sublist/S8-Model-driven Automation/Models-Output/fill-bracket-api-generators-selected/gpt-5.4-generated_output.fill_bracket.csv",
    "Gemini": REPO_ROOT / "Datasets/sublist/S8-Model-driven Automation/Models-Output/fill-bracket-api-generators-fast/gemini-2.5-pro-generated_output.fill_bracket.csv",
    "Llama": REPO_ROOT / "Datasets/sublist/S8-Model-driven Automation/Models-Output/fill-bracket-llama3_8b-full/llama-3.1-8b-generated_output.fill_bracket.csv",
    "Ministral": REPO_ROOT / "Datasets/sublist/S8-Model-driven Automation/Models-Output/fill-bracket-llama3_8b-full/ministral-8b-generated_output.fill_bracket.csv",
    "DeepSeek": REPO_ROOT / "Datasets/sublist/S8-Model-driven Automation/Models-Output/fill-bracket-llama3_8b-full/deepseek-r1-distill-qwen-7b-generated_output.fill_bracket.csv",
}

ACADEMIC_DETECTORS = ["scamllm", "pimref", "t5phishing", "xgboost", "securenet_llama"]
INDUSTRY_DETECTORS = ["email_phishing_detection_v3_prediction"]
DETECTOR_LABELS = {
    "scamllm": "ScamLLM",
    "pimref": "PiMRef",
    "t5phishing": "T5",
    "xgboost": "XGBoost",
    "securenet_llama": "SecureNet",
    "email_phishing_detection_v3_prediction": "PhishingV3",
}

FEATURES = [
    ("Urgency wording", r"\b(urgent|immediately|as soon as possible|asap|deadline|expires?|expiring|suspended?|locked|limited time|within \d+|final notice|act now)\b"),
    ("Login/account action", r"\b(log ?in|sign ?in|account|password|credential|username|authentication|verify your account|account verification|account update)\b"),
    ("Information submission", r"\b(submit|provide|send|enter|input|fill|complete|confirm|verify|update).{0,45}\b(info|information|details|credential|password|account|address|payment|card|code|otp)\b"),
    ("Click/open request", r"\b(click|tap|open|visit|follow (the )?link|use (the )?link|press (the )?button|button below|link below)\b"),
    ("Softened request", r"\b(please|kindly|could you|would you|when you have a chance|at your convenience|we would appreciate|if possible|just wanted|quick note)\b"),
    ("Explicit action request", r"\b(click|tap|open|visit|go to|follow|log ?in|sign ?in|submit|provide|enter|verify|confirm|update|download|review|complete)\b"),
    ("Direct URL/page instruction", r"\b(https?://|www\.|url|link|webpage|website|portal|page|site|landing page|dashboard)\b"),
    ("Conversational wording", r"\b(hi|hello|hey|dear|thanks|thank you|hope you|checking in|following up|best regards|regards|cheers)\b"),
]
COMPILED_FEATURES = [(name, re.compile(pattern, re.IGNORECASE | re.DOTALL)) for name, pattern in FEATURES]

PAIRED_ROWS = ROOT / "s8_paired_common_dataset_seed20260825.csv"
DATASET_AUDIT = ROOT / "s8_paired_dataset_audit.csv"
RQ1_TABLE = ROOT / "s8_rq1_paired_detector_cochran.csv"
RQ1_PAIRWISE = ROOT / "s8_rq1_selected_mcnemar_pairs.csv"
RQ1_ALL_PAIRWISE = ROOT / "S8_rq1_all_pairwise_mcnemar.csv"
RQ2_TABLE = ROOT / "s8_rq2_paired_feature_cochran.csv"
RQ2_PAIRWISE = ROOT / "s8_rq2_feature_pairwise_mcnemar.csv"
SUMMARY = ROOT / "s8_paired_rq1_rq2_summary.md"
FIG_S8_A = ROOT / "Fig_S8_A_detector_generator_detection_rate_heatmap.png"
FIG_S8_B = ROOT / "Fig_S8_B_generator_feature_prevalence_heatmap.png"


def parse_binary(value):
    try:
        numeric = float(str(value).strip())
    except ValueError:
        return None
    if math.isnan(numeric):
        return None
    return 1 if numeric >= 0.5 else 0


def chi_square_sf_df5(x_value):
    if x_value <= 0:
        return 1.0
    z_value = x_value / 2.0
    root_z = math.sqrt(z_value)
    return min(
        1.0,
        math.erfc(root_z)
        + math.exp(-z_value) * (root_z / math.gamma(1.5) + z_value ** 1.5 / math.gamma(2.5)),
    )


def bh_fdr(rows, p_key="p_value", q_key="q_value"):
    indexed = sorted(enumerate(rows), key=lambda item: float(item[1][p_key]))
    adjusted = [1.0] * len(rows)
    running_min = 1.0
    total = len(rows)
    for rank, (index, row) in reversed(list(enumerate(indexed, start=1))):
        running_min = min(running_min, float(row[p_key]) * total / rank)
        adjusted[index] = min(running_min, 1.0)
    for row, q_value in zip(rows, adjusted):
        row[q_key] = q_value


def significance_stars(value):
    if value in ("", None, "NA"):
        return ""
    value = float(value)
    if value < 0.001:
        return "***"
    if value < 0.01:
        return "**"
    if value < 0.05:
        return "*"
    return ""


def cochran_q(matrix):
    n_rows = len(matrix)
    if n_rows == 0:
        return 0.0, 1.0
    k_value = len(matrix[0])
    col_sums = [sum(row[j] for row in matrix) for j in range(k_value)]
    row_sums = [sum(row) for row in matrix]
    total = sum(col_sums)
    denominator = k_value * total - sum(value * value for value in row_sums)
    if denominator == 0:
        return 0.0, 1.0
    q_value = (k_value - 1) * (k_value * sum(value * value for value in col_sums) - total * total) / denominator
    return q_value, chi_square_sf_df5(q_value)


def log_comb(n_value, k_value):
    if k_value < 0 or k_value > n_value:
        return float("-inf")
    return math.lgamma(n_value + 1) - math.lgamma(k_value + 1) - math.lgamma(n_value - k_value + 1)


def binomial_probability(k_value, n_value):
    return math.exp(log_comb(n_value, k_value) - n_value * math.log(2.0))


def mcnemar_exact(b_value, c_value):
    n_value = b_value + c_value
    if n_value == 0:
        return 1.0
    observed = min(b_value, c_value)
    p_value = 2.0 * sum(binomial_probability(k, n_value) for k in range(observed + 1))
    return min(1.0, p_value)


def norm_text(subject, body):
    return (subject or "").strip(), (body or "").strip()


def norm_body(body):
    text = re.sub(r"\s+", " ", (body or "").strip())
    if len(text) >= 2 and text[0] == '"' and text[-1] == '"':
        text = text[1:-1].strip()
    return text


def read_csv_dicts(path):
    with path.open(newline="", encoding="utf-8", errors="replace") as handle:
        return list(csv.DictReader(handle))


def load_raw_outputs():
    raw = {}
    raw_order = {}
    audit = {}
    for generator, path in RAW_PATHS.items():
        rows = read_csv_dicts(path)
        by_prompt = defaultdict(list)
        ordered_records = []
        invalid = 0
        for row_index, row in enumerate(rows):
            if parse_binary(row.get("label")) != 1:
                continue
            subject = row.get("generated_subject", "")
            body = row.get("generated_body", "")
            if not (subject or body):
                invalid += 1
                continue
            record = {
                "raw_index": row_index,
                "prompt_id": row.get("join_key", ""),
                "subject": subject,
                "body": body,
                "generated_text": row.get("generated_text", ""),
                "text": f"{subject}\n{body}".lower(),
                "label": 1,
            }
            by_prompt[record["prompt_id"]].append(record)
            ordered_records.append(record)
        raw[generator] = by_prompt
        raw_order[generator] = ordered_records
        raw_phishing_outputs = sum(len(values) for values in by_prompt.values())
        audit[generator] = {
            "raw_outputs": len(rows),
            "raw_phishing_outputs": raw_phishing_outputs,
            "unique_prompts": len(by_prompt),
            "repeated_outputs_removed": raw_phishing_outputs - len(by_prompt),
            "missing_invalid": invalid,
        }
    return raw, raw_order, audit


def attach_predictions(generator, selected_records):
    stage = GENERATOR_STAGE[generator]
    paths = [
        REPO_ROOT / "Evaluation/processed-evaluation-datasets/llm/academic" / f"{stage}.csv",
        REPO_ROOT / "Evaluation/processed-evaluation-datasets/llm/industry" / f"{stage}.csv",
    ]
    columns_by_path = [ACADEMIC_DETECTORS, INDUSTRY_DETECTORS]
    for path, detector_columns in zip(paths, columns_by_path):
        raw_by_text = defaultdict(list)
        raw_by_body = defaultdict(list)
        for record in selected_records:
            raw_by_text[norm_text(record["subject"], record["body"])].append(record)
            raw_by_body[norm_body(record["body"])].append(record)
            raw_by_body[norm_body(record.get("generated_text", ""))].append(record)
        rows = [row for row in read_csv_dicts(path) if parse_binary(row.get("label")) == 1]
        if len(rows) != len(selected_records):
            raise RuntimeError(f"{path} phishing row count {len(rows)} does not match raw phishing rows {len(selected_records)}")
        available = {key: list(values) for key, values in raw_by_text.items()}
        for index, row in enumerate(rows):
            key = norm_text(row.get("subject"), row.get("body"))
            if available.get(key):
                record = available[key].pop(0)
            else:
                body_key = norm_body(row.get("body"))
                body_available = [record for record in raw_by_body.get(body_key, []) if record in selected_records]
                if not body_available:
                    raise RuntimeError(f"{path} row {index} does not match raw generation output text for {generator}: {key[0][:80]}")
                record = body_available.pop(0)
                raw_by_body[body_key].remove(record)
            for detector in detector_columns:
                record[detector] = parse_binary(row.get(detector))
            record["subject"] = row.get("subject", record["subject"])
            record["body"] = row.get("body", record["body"])
            record["text"] = f"{record['subject']}\n{record['body']}".lower()


def build_paired_dataset():
    rng = random.Random(SEED)
    raw, raw_order, audit = load_raw_outputs()
    for generator in GENERATOR_ORDER:
        attach_predictions(generator, raw_order[generator])

    selected = {}
    for generator in GENERATOR_ORDER:
        selected[generator] = {}
        for prompt_id, records in raw[generator].items():
            selected[generator][prompt_id] = rng.choice(records)

    common_prompts = sorted(set.intersection(*(set(selected[generator]) for generator in GENERATOR_ORDER)))
    for generator in GENERATOR_ORDER:
        audit[generator]["final_common_n"] = len(common_prompts)

    output_rows = []
    for prompt_id in common_prompts:
        for generator in GENERATOR_ORDER:
            record = selected[generator][prompt_id]
            row = {
                "prompt_id": prompt_id,
                "generator": generator,
                "stage": GENERATOR_STAGE[generator],
                "subject": record["subject"],
                "body": record["body"],
            }
            for detector in ACADEMIC_DETECTORS + INDUSTRY_DETECTORS:
                row[detector] = record.get(detector)
            for feature, pattern in COMPILED_FEATURES:
                row[feature] = 1 if pattern.search(record["text"]) else 0
            output_rows.append(row)
    return output_rows, audit, common_prompts


def write_csv(path, rows, fieldnames):
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            formatted = {}
            for field in fieldnames:
                value = row.get(field, "")
                if isinstance(value, float):
                    value = f"{value:.6g}"
                formatted[field] = value
            writer.writerow(formatted)


def pivot_by_prompt(rows, value_key):
    by_prompt = defaultdict(dict)
    for row in rows:
        value = row.get(value_key)
        if value in ("", None):
            continue
        by_prompt[row["prompt_id"]][row["generator"]] = int(value)
    matrix = []
    for prompt_id in sorted(by_prompt):
        prompt_values = by_prompt[prompt_id]
        if all(generator in prompt_values for generator in GENERATOR_ORDER):
            matrix.append([prompt_values[generator] for generator in GENERATOR_ORDER])
    return matrix


def run_rq1(rows):
    table = []
    for detector in ACADEMIC_DETECTORS + INDUSTRY_DETECTORS:
        matrix = pivot_by_prompt(rows, detector)
        q_stat, p_value = cochran_q(matrix)
        result = {
            "detector": DETECTOR_LABELS[detector],
            "detector_column": detector,
            "N_common": len(matrix),
            "Cochran_Q": q_stat,
            "p_value": p_value,
        }
        for index, generator in enumerate(GENERATOR_ORDER):
            rate = sum(row[index] for row in matrix) / len(matrix) if matrix else 0.0
            result[generator] = rate
        table.append(result)
    bh_fdr(table)
    return table


def run_rq1_pairwise(rows):
    selected_pairs = [
        ("SecureNet", "Llama", "Claude"),
        ("SecureNet", "Llama", "GPT"),
        ("SecureNet", "Llama", "Gemini"),
        ("SecureNet", "Llama", "DeepSeek"),
        ("SecureNet", "Ministral", "Claude"),
        ("SecureNet", "Ministral", "GPT"),
        ("SecureNet", "Ministral", "Gemini"),
        ("SecureNet", "Ministral", "DeepSeek"),
        ("PhishingV3", "DeepSeek", "Claude"),
        ("PhishingV3", "DeepSeek", "GPT"),
        ("PhishingV3", "DeepSeek", "Gemini"),
        ("PhishingV3", "DeepSeek", "Llama"),
        ("PhishingV3", "DeepSeek", "Ministral"),
    ]
    label_to_detector = {value: key for key, value in DETECTOR_LABELS.items()}
    rows_out = []
    for detector_label, gen_a, gen_b in selected_pairs:
        detector = label_to_detector[detector_label]
        matrix = pivot_by_prompt(rows, detector)
        idx_a = GENERATOR_ORDER.index(gen_a)
        idx_b = GENERATOR_ORDER.index(gen_b)
        b_value = sum(1 for row in matrix if row[idx_a] == 1 and row[idx_b] == 0)
        c_value = sum(1 for row in matrix if row[idx_a] == 0 and row[idx_b] == 1)
        rate_a = sum(row[idx_a] for row in matrix) / len(matrix)
        rate_b = sum(row[idx_b] for row in matrix) / len(matrix)
        rows_out.append({
            "detector": detector_label,
            "generator_a": gen_a,
            "generator_b": gen_b,
            "detection_rate_a": rate_a,
            "detection_rate_b": rate_b,
            "detection_rate_difference_a_minus_b": rate_a - rate_b,
            "discordant_a_detected_b_missed": b_value,
            "discordant_a_missed_b_detected": c_value,
            "p_value": mcnemar_exact(b_value, c_value),
        })
    bh_fdr(rows_out)
    return rows_out


def run_rq1_all_pairwise(rows):
    rows_out = []
    for detector in ACADEMIC_DETECTORS + INDUSTRY_DETECTORS:
        matrix = pivot_by_prompt(rows, detector)
        for left_index, gen_a in enumerate(GENERATOR_ORDER):
            for right_index in range(left_index + 1, len(GENERATOR_ORDER)):
                gen_b = GENERATOR_ORDER[right_index]
                b_value = sum(1 for row in matrix if row[left_index] == 1 and row[right_index] == 0)
                c_value = sum(1 for row in matrix if row[left_index] == 0 and row[right_index] == 1)
                rate_a = sum(row[left_index] for row in matrix) / len(matrix)
                rate_b = sum(row[right_index] for row in matrix) / len(matrix)
                rows_out.append({
                    "detector": DETECTOR_LABELS[detector],
                    "generator_A": gen_a,
                    "generator_B": gen_b,
                    "rate_A": rate_a,
                    "rate_B": rate_b,
                    "delta_pp": (rate_a - rate_b) * 100.0,
                    "A_TP_B_FN": b_value,
                    "A_FN_B_TP": c_value,
                    "p_value": mcnemar_exact(b_value, c_value),
                })
    bh_fdr(rows_out)
    return rows_out


def run_rq2(rows):
    table = []
    for feature, _ in COMPILED_FEATURES:
        matrix = pivot_by_prompt(rows, feature)
        q_stat, p_value = cochran_q(matrix)
        rates = [sum(row[index] for row in matrix) / len(matrix) for index in range(len(GENERATOR_ORDER))]
        result = {
            "feature": feature,
            "N_common": len(matrix),
            "Cochran_Q": q_stat,
            "p_value": p_value,
            "range_pp": (max(rates) - min(rates)) * 100.0,
        }
        for generator, rate in zip(GENERATOR_ORDER, rates):
            result[generator] = rate
        table.append(result)
    bh_fdr(table)
    return table


def run_rq2_pairwise(rows):
    rows_out = []
    for feature, _ in COMPILED_FEATURES:
        matrix = pivot_by_prompt(rows, feature)
        for left_index, gen_a in enumerate(GENERATOR_ORDER):
            for right_index in range(left_index + 1, len(GENERATOR_ORDER)):
                gen_b = GENERATOR_ORDER[right_index]
                b_value = sum(1 for row in matrix if row[left_index] == 1 and row[right_index] == 0)
                c_value = sum(1 for row in matrix if row[left_index] == 0 and row[right_index] == 1)
                rate_a = sum(row[left_index] for row in matrix) / len(matrix)
                rate_b = sum(row[right_index] for row in matrix) / len(matrix)
                rows_out.append({
                    "feature": feature,
                    "generator_a": gen_a,
                    "generator_b": gen_b,
                    "prevalence_a": rate_a,
                    "prevalence_b": rate_b,
                    "prevalence_difference_a_minus_b": rate_a - rate_b,
                    "discordant_a_yes_b_no": b_value,
                    "discordant_a_no_b_yes": c_value,
                    "p_value": mcnemar_exact(b_value, c_value),
                })
    bh_fdr(rows_out)
    return rows_out


def write_summary(audit_rows, rq1, pairwise, rq2):
    lines = [
        "# S8 paired design RQ1-RQ2",
        "",
        f"Fixed seed: `{SEED}`. Common prompt set is phishing-labelled prompts present for all six generators after one output per prompt is selected.",
        "",
        "## Step 0 dataset audit",
        "",
        "| Generator | Raw outputs | Unique prompts | Repeated outputs removed | Missing/invalid | Final common N |",
        "| --- | ---: | ---: | ---: | ---: | ---: |",
    ]
    for row in audit_rows:
        lines.append(f"| {row['generator']} | {row['raw_outputs']} | {row['unique_prompts']} | {row['repeated_outputs_removed']} | {row['missing_invalid']} | {row['final_common_n']} |")
    lines.extend(["", "## RQ1 detector outcome Cochran Q", "", "| Detector | Claude | GPT | Gemini | Llama | Ministral | DeepSeek | Q | q |", "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |"])
    for row in rq1:
        rates = [float(row[g]) * 100.0 for g in GENERATOR_ORDER]
        lines.append("| {detector} | {rates} | {qstat:.2f} | {qval:.3g} |".format(
            detector=row["detector"],
            rates=" | ".join(f"{rate:.1f}%" for rate in rates),
            qstat=float(row["Cochran_Q"]),
            qval=float(row["q_value"]),
        ))
    lines.extend(["", "## Selected RQ1 McNemar pairs", "", "| Detector | A | B | A-B | p | q |", "| --- | --- | --- | ---: | ---: | ---: |"])
    for row in pairwise:
        lines.append(f"| {row['detector']} | {row['generator_a']} | {row['generator_b']} | {float(row['detection_rate_difference_a_minus_b']) * 100.0:+.1f} pp | {float(row['p_value']):.3g} | {float(row['q_value']):.3g} |")
    lines.extend(["", "## RQ2 feature Cochran Q", "", "| Feature | Claude | GPT | Gemini | Llama | Ministral | DeepSeek | Q | q | Range |", "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |"])
    for row in sorted(rq2, key=lambda item: -float(item["range_pp"])):
        rates = [float(row[g]) * 100.0 for g in GENERATOR_ORDER]
        lines.append("| {feature} | {rates} | {qstat:.2f} | {qval:.3g} | {range_pp:.1f} pp |".format(
            feature=row["feature"],
            rates=" | ".join(f"{rate:.1f}%" for rate in rates),
            qstat=float(row["Cochran_Q"]),
            qval=float(row["q_value"]),
            range_pp=float(row["range_pp"]),
        ))
    SUMMARY.write_text("\n".join(lines) + "\n", encoding="utf-8")


def two_line_label(label):
    if label == "Direct URL/page instruction":
        return "Direct URL/page\ninstruction"
    return label.replace(" ", "\n", 1)


def draw_heatmap(path, matrix, row_labels, col_labels, title, colorbar_label, fmt="{:.1f}%"):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    base_font = 36
    tick_font = base_font + 2
    number_font = 30 if any("\n" in label for label in row_labels) else 32 + 1

    plt.rcParams.update({
        "font.size": base_font,
        "axes.labelsize": base_font,
        "xtick.labelsize": tick_font,
        "ytick.labelsize": tick_font,
    })
    has_multiline_rows = any("\n" in label for label in row_labels)
    fig_width = max(30.0 if has_multiline_rows else 24.0, 2.9 * len(col_labels) + 12.0)
    fig_height = max(13.0, 1.45 * len(row_labels) + 4.2)
    figure, axis = plt.subplots(figsize=(fig_width, fig_height))
    values = numpy_array(matrix)
    image = axis.imshow(values, cmap="YlGnBu", vmin=0, vmax=max(100, float(values.max())))
    axis.set_xticks(range(len(col_labels)))
    axis.set_xticklabels(col_labels, rotation=35, ha="right", fontsize=tick_font)
    axis.set_yticks(range(len(row_labels)))
    axis.set_yticklabels(row_labels, fontsize=tick_font)
    for row_index in range(len(row_labels)):
        for col_index in range(len(col_labels)):
            value = float(values[row_index][col_index])
            color = "white" if value > 62 else "black"
            axis.text(col_index, row_index, fmt.format(value), ha="center", va="center", color=color, fontsize=number_font, clip_on=False)
    axis.set_xlim(-0.5, len(col_labels) - 0.5)
    axis.set_xticks([x - 0.5 for x in range(1, len(col_labels))], minor=True)
    axis.set_yticks([y - 0.5 for y in range(1, len(row_labels))], minor=True)
    axis.grid(which="minor", color="white", linewidth=1.5)
    axis.tick_params(which="minor", bottom=False, left=False)
    colorbar = figure.colorbar(image, ax=axis, fraction=0.04, pad=0.03)
    colorbar.set_label(colorbar_label, fontsize=base_font)
    colorbar.ax.tick_params(labelsize=number_font)
    left_margin = 0.32 if has_multiline_rows else 0.24
    figure.subplots_adjust(left=left_margin, bottom=0.26, right=0.88, top=0.96)
    figure.savefig(path, dpi=300, bbox_inches="tight")
    plt.close(figure)


def numpy_array(matrix):
    import numpy as np
    return np.asarray(matrix, dtype=float)


def draw_rq_figures(rq1, rq2):
    detector_rows = []
    detector_labels = []
    for row in rq1:
        detector_labels.append(row["detector"])
        detector_rows.append([float(row[generator]) * 100.0 for generator in GENERATOR_ORDER])
    draw_heatmap(
        FIG_S8_A,
        detector_rows,
        detector_labels,
        GENERATOR_ORDER,
        "Fig. S8-A. Paired phishing detection rate",
        "Detection rate (%)",
    )

    feature_order = [name for name, _ in FEATURES]
    feature_rows = []
    for feature in feature_order:
        row = next(item for item in rq2 if item["feature"] == feature)
        feature_rows.append([float(row[generator]) * 100.0 for generator in GENERATOR_ORDER])
    draw_heatmap(
        FIG_S8_B,
        feature_rows,
        feature_order,
        GENERATOR_ORDER,
        "Fig. S8-B. Paired feature prevalence",
        "Prevalence (%)",
    )


def main():
    rows, audit, common_prompts = build_paired_dataset()
    paired_fields = ["prompt_id", "generator", "stage", "subject", "body"] + ACADEMIC_DETECTORS + INDUSTRY_DETECTORS + [name for name, _ in FEATURES]
    write_csv(PAIRED_ROWS, rows, paired_fields)

    audit_rows = []
    for generator in GENERATOR_ORDER:
        row = {"generator": generator}
        row.update(audit[generator])
        row["total_raw_rows"] = row["raw_outputs"]
        row["raw_outputs"] = row["raw_phishing_outputs"]
        audit_rows.append(row)
    write_csv(DATASET_AUDIT, audit_rows, ["generator", "raw_outputs", "unique_prompts", "repeated_outputs_removed", "missing_invalid", "final_common_n", "total_raw_rows"])

    rq1 = run_rq1(rows)
    write_csv(RQ1_TABLE, rq1, ["detector", "detector_column", "N_common"] + GENERATOR_ORDER + ["Cochran_Q", "p_value", "q_value"])
    rq1_pairwise = run_rq1_pairwise(rows)
    write_csv(RQ1_PAIRWISE, rq1_pairwise, ["detector", "generator_a", "generator_b", "detection_rate_a", "detection_rate_b", "detection_rate_difference_a_minus_b", "discordant_a_detected_b_missed", "discordant_a_missed_b_detected", "p_value", "q_value"])
    rq1_all_pairwise = run_rq1_all_pairwise(rows)
    write_csv(RQ1_ALL_PAIRWISE, rq1_all_pairwise, ["detector", "generator_A", "generator_B", "rate_A", "rate_B", "delta_pp", "A_TP_B_FN", "A_FN_B_TP", "p_value", "q_value"])

    rq2 = run_rq2(rows)
    write_csv(RQ2_TABLE, rq2, ["feature", "N_common"] + GENERATOR_ORDER + ["Cochran_Q", "p_value", "q_value", "range_pp"])
    rq2_pairwise = run_rq2_pairwise(rows)
    write_csv(RQ2_PAIRWISE, rq2_pairwise, ["feature", "generator_a", "generator_b", "prevalence_a", "prevalence_b", "prevalence_difference_a_minus_b", "discordant_a_yes_b_no", "discordant_a_no_b_yes", "p_value", "q_value"])

    write_summary(audit_rows, rq1, rq1_pairwise, rq2)
    draw_rq_figures(rq1, rq2)
    print(f"Common prompt N: {len(common_prompts)}")
    print(f"Wrote {PAIRED_ROWS}")
    print(f"Wrote {DATASET_AUDIT}")
    print(f"Wrote {RQ1_TABLE}")
    print(f"Wrote {RQ1_PAIRWISE}")
    print(f"Wrote {RQ1_ALL_PAIRWISE}")
    print(f"Wrote {RQ2_TABLE}")
    print(f"Wrote {RQ2_PAIRWISE}")
    print(f"Wrote {SUMMARY}")
    print(f"Wrote {FIG_S8_A}")
    print(f"Wrote {FIG_S8_B}")


if __name__ == "__main__":
    main()
