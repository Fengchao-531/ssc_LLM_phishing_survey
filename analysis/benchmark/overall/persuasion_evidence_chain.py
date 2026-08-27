#!/usr/bin/env python3
"""Assemble and analyze the Section 5.3.2 persuasion evidence chain.

Run from the repository root:

    python3 Visualization/5.3.2/run_5_3_2_evidence_chain.py

The script does not regenerate phishing samples and does not rerun detectors. It
uses existing persuasion scores and detector predictions, writes sample-level
CSV files, runs E1-E3 over all 21 persuasion pairs independently, and renders a
three-panel figure for the main paper.
"""

import csv
import json
import math
import sys
import zipfile
from collections import defaultdict
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D

csv.field_size_limit(sys.maxsize)

SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parents[1]
WVAE_DIR = REPO_ROOT / "Visualization" / "persuasion_strategy_wvae" / "output" / "full_inference_results"
INDUSTRY_DIR = REPO_ROOT / "Evaluation" / "processed-evaluation-datasets"

HW_FILES = [
    ("HW", "S1", "", "HW_S1_persuasion.csv"),
    ("HW", "S2", "", "HW_S2_persuasion.csv"),
    ("HW", "S4", "", "HW_S4_persuasion.csv"),
    ("HW", "S5", "", "HW_S5_persuasion.csv"),
    ("HW", "S6", "", "HW_S6_persuasion.csv"),
    ("HW", "S8", "", "HW_S8_persuasion.csv"),
]
LLM_FILES = [
    ("LLM", "S1", "", "LLM_S1_persuasion.csv"),
    ("LLM", "S2", "", "LLM_S2_persuasion.csv"),
    ("LLM", "S4", "", "LLM_S4_persuasion.csv"),
    ("LLM", "S5", "", "LLM_S5_persuasion.csv"),
    ("LLM", "S6-MPG", "MPG", "LLM_S6-MPG_persuasion.csv"),
    ("LLM", "S6-UTA", "UTA", "LLM_S6-UTA_persuasion.csv"),
    ("LLM", "S6-fuzzer", "fuzzer", "LLM_S6-fuzzer_persuasion.csv"),
    ("LLM", "S8-claude", "claude", "LLM_S8-claude_persuasion.csv"),
    ("LLM", "S8-deepseek", "deepseek", "LLM_S8-deepseek_persuasion.csv"),
    ("LLM", "S8-gemini", "gemini", "LLM_S8-gemini_persuasion.csv"),
    ("LLM", "S8-gpt", "gpt", "LLM_S8-gpt_persuasion.csv"),
    ("LLM", "S8-llama", "llama", "LLM_S8-llama_persuasion.csv"),
    ("LLM", "S8-ministral", "ministral", "LLM_S8-ministral_persuasion.csv"),
]
INPUT_FILES = HW_FILES + LLM_FILES

PRINCIPLES = [
    ("A", "Authority", "principle_authority"),
    ("R", "Reciprocity", "principle_reciprocity"),
    ("C", "Commitment", "principle_commitment"),
    ("S", "Scarcity", "principle_scarcity"),
    ("SP", "Social Proof", "principle_social_proof"),
    ("L", "Liking", "principle_liking"),
]
PAIR_SPECS = [
    (i, j, f"{PRINCIPLES[i][0]}-{PRINCIPLES[j][0]}")
    for i in range(len(PRINCIPLES))
    for j in range(i, len(PRINCIPLES))
]
PAIR_LABELS = [label for _, _, label in PAIR_SPECS]

ACADEMIC_DETECTORS = {
    "scamllm": "scamllm",
    "pimref": "pimref",
    "t5phishing": "t5phishing",
    "xgboost": "xgboost",
    "securenet_llama": "securenet_llama",
}
INDUSTRY_DETECTORS = {
    "email_phishing_detection_v3": "email_phishing_detection_v3_prediction",
    "phishing_email_agent": "phishing_email_agent_prediction",
    "rspamd": "rspamd_prediction",
    "spamscanner": "spamscanner_prediction",
    "spamassassin": "spamassassin_prediction",
}
DETECTOR_ORDER = list(ACADEMIC_DETECTORS) + list(INDUSTRY_DETECTORS)
DETECTOR_SHORT_LABELS = [f"D{index}" for index in range(1, len(DETECTOR_ORDER) + 1)]
DETECTOR_DISPLAY_LABELS = [
    "scamllm",
    "pimref",
    "t5phishing",
    "xgboost",
    "securenet_llama",
    "Phishingv3",
    "email agent",
    "rspamd",
    "spamscanner",
    "spamassassin",
]
KAPPA_PERMUTATIONS = 1000
KAPPA_BOOTSTRAPS = 500


def to_float(value,  default= 0.0) :
    try:
        if value is None or value == "":
            return default
        result = float(value)
    except (TypeError, ValueError):
        return default
    return result if math.isfinite(result) else default


def to_label(value) :
    return "P" if to_float(value) >= 0.5 else "B"


def to_prediction(value) :
    if isinstance(value, str):
        stripped = value.strip().upper()
        if stripped in {"P", "PHISHING", "PHISH", "1", "TRUE"}:
            return "P"
        if stripped in {"B", "BENIGN", "0", "FALSE"}:
            return "B"
    return "P" if to_float(value) >= 0.5 else "B"


def read_csv_rows(path) :
    if not path.exists():
        return []
    with path.open(newline="", encoding="utf-8", errors="replace") as handle:
        return list(csv.DictReader(handle))


def industry_rows(source,  stage) :
    family = "llm" if source == "LLM" else "gd"
    path = INDUSTRY_DIR / family / "industry" / f"{stage}.csv"
    return read_csv_rows(path)


def pair_scores(row):
    values = np.array([to_float(row.get(col)) for _, _, col in PRINCIPLES], dtype=float)
    values = np.clip(values, 0.0, None)
    scores = []
    for i, j, _ in PAIR_SPECS:
        scores.append(float(values[i] * values[j]))
    return scores


def rankdata(values) :
    order = np.argsort(values, kind="mergesort")
    ranks = np.empty(len(values), dtype=float)
    sorted_values = values[order]
    start = 0
    while start < len(values):
        end = start + 1
        while end < len(values) and sorted_values[end] == sorted_values[start]:
            end += 1
        avg_rank = (start + 1 + end) / 2.0
        ranks[order[start:end]] = avg_rank
        start = end
    return ranks


def mann_whitney(group_a,  group_b) :
    group_a = np.asarray(group_a, dtype=float)
    group_b = np.asarray(group_b, dtype=float)
    group_a = group_a[np.isfinite(group_a)]
    group_b = group_b[np.isfinite(group_b)]
    n_a = int(group_a.size)
    n_b = int(group_b.size)
    if n_a == 0 or n_b == 0:
        return {"n_a": n_a, "n_b": n_b, "u": math.nan, "p": math.nan, "rank_biserial": math.nan}
    combined = np.concatenate([group_a, group_b])
    ranks = rankdata(combined)
    rank_sum_a = float(ranks[:n_a].sum())
    u_a = rank_sum_a - n_a * (n_a + 1) / 2.0
    effect = (2.0 * u_a / (n_a * n_b)) - 1.0

    _, counts = np.unique(combined, return_counts=True)
    tie_sum = float(np.sum(counts**3 - counts))
    n_total = n_a + n_b
    mean_u = n_a * n_b / 2.0
    var_u = n_a * n_b / 12.0 * ((n_total + 1.0) - tie_sum / (n_total * (n_total - 1.0)))
    if var_u <= 0:
        p_value = 1.0 if abs(u_a - mean_u) < 1e-12 else 0.0
    else:
        correction = 0.5 if u_a > mean_u else -0.5 if u_a < mean_u else 0.0
        z = (u_a - mean_u - correction) / math.sqrt(var_u)
        p_value = math.erfc(abs(z) / math.sqrt(2.0))
    return {"n_a": n_a, "n_b": n_b, "u": u_a, "p": p_value, "rank_biserial": effect}


def bh_fdr(p_values) :
    indexed = [(idx, p) for idx, p in enumerate(p_values) if math.isfinite(p)]
    q_values = [math.nan] * len(p_values)
    if not indexed:
        return q_values
    indexed.sort(key=lambda item: item[1])
    m = len(indexed)
    prev = 1.0
    for rank in range(m, 0, -1):
        idx, p = indexed[rank - 1]
        prev = min(prev, p * m / rank)
        q_values[idx] = min(prev, 1.0)
    return q_values


def significance_stars(q_value):
    if not math.isfinite(q_value):
        return ""
    if q_value < 0.001:
        return "***"
    if q_value < 0.01:
        return "**"
    if q_value < 0.05:
        return "*"
    return ""


def quantiles(values) :
    if len(values) == 0:
        return math.nan, math.nan, math.nan
    q1, med, q3 = np.percentile(values, [25, 50, 75])
    return float(med), float(q1), float(q3)


def bootstrap_effect_ci(group_a,  group_b,  *,  seed,  n_boot= 300) :
    if len(group_a) == 0 or len(group_b) == 0:
        return math.nan, math.nan
    rng = np.random.default_rng(seed)
    effects = np.empty(n_boot, dtype=float)
    for index in range(n_boot):
        sample_a = group_a[rng.integers(0, len(group_a), len(group_a))]
        sample_b = group_b[rng.integers(0, len(group_b), len(group_b))]
        effects[index] = fast_rank_biserial(sample_a, sample_b)
    return float(np.percentile(effects, 2.5)), float(np.percentile(effects, 97.5))


def auc_ci(group_pos,  group_neg,  *,  seed,  n_boot= 500) :
    stat = mann_whitney(group_pos, group_neg)
    auc = (stat["rank_biserial"] + 1.0) / 2.0 if math.isfinite(stat["rank_biserial"]) else math.nan
    if len(group_pos) == 0 or len(group_neg) == 0:
        return auc, math.nan, math.nan
    rng = np.random.default_rng(seed)
    aucs = np.empty(n_boot, dtype=float)
    for index in range(n_boot):
        sample_pos = group_pos[rng.integers(0, len(group_pos), len(group_pos))]
        sample_neg = group_neg[rng.integers(0, len(group_neg), len(group_neg))]
        aucs[index] = fast_auc(sample_pos, sample_neg)
    return auc, float(np.percentile(aucs, 2.5)), float(np.percentile(aucs, 97.5))


def fast_auc(group_pos, group_neg):
    group_pos = np.asarray(group_pos, dtype=float)
    group_neg = np.asarray(group_neg, dtype=float)
    group_pos = group_pos[np.isfinite(group_pos)]
    group_neg = group_neg[np.isfinite(group_neg)]
    if len(group_pos) == 0 or len(group_neg) == 0:
        return math.nan
    sorted_neg = np.sort(group_neg)
    less = np.searchsorted(sorted_neg, group_pos, side="left")
    less_equal = np.searchsorted(sorted_neg, group_pos, side="right")
    wins = less + 0.5 * (less_equal - less)
    return float(np.sum(wins) / (len(group_pos) * len(group_neg)))


def fast_rank_biserial(group_a, group_b):
    auc = fast_auc(group_a, group_b)
    if not math.isfinite(auc):
        return math.nan
    return 2.0 * auc - 1.0


def write_dict_rows(path, fieldnames, rows):
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def assemble() :
    metadata_rows = []
    persuasion_rows = []
    prediction_rows = []
    score_groups = {pair: defaultdict(list) for pair in PAIR_LABELS}
    llm_p_scores_by_sample = {}
    llm_p_predictions_by_sample = defaultdict(dict)
    counts = defaultdict(int)
    missing_industry = defaultdict(int)

    for source, stage, generator, filename in INPUT_FILES:
        path = WVAE_DIR / filename
        rows = read_csv_rows(path)
        ind_rows = industry_rows(source, stage)
        for index, row in enumerate(rows):
            sample_id = f"{source}_{stage}_{index:06d}"
            label = to_label(row.get("label"))
            group_key = f"{source}-{label}"
            scores = pair_scores(row)
            counts[group_key] += 1

            metadata_rows.append(
                {
                    "sample_id": sample_id,
                    "dataset": "email",
                    "source": source,
                    "label": label,
                    "stage": stage,
                    "generator": generator,
                    "parent_id": "",
                }
            )
            for pair, score in zip(PAIR_LABELS, scores):
                persuasion_rows.append({"sample_id": sample_id, "pair": pair, "score": f"{score:.10g}"})
                score_groups[pair][group_key].append(score)
            if source == "LLM" and label == "P":
                llm_p_scores_by_sample[sample_id] = scores

            for detector, column in ACADEMIC_DETECTORS.items():
                pred = to_prediction(row.get(column))
                prediction_rows.append({"sample_id": sample_id, "detector": detector, "prediction": pred})
                if source == "LLM" and label == "P":
                    llm_p_predictions_by_sample[sample_id][detector] = pred

            if index < len(ind_rows):
                ind_row = ind_rows[index]
                for detector, column in INDUSTRY_DETECTORS.items():
                    if column not in ind_row or ind_row.get(column) == "":
                        continue
                    pred = to_prediction(ind_row.get(column))
                    prediction_rows.append({"sample_id": sample_id, "detector": detector, "prediction": pred})
                    if source == "LLM" and label == "P":
                        llm_p_predictions_by_sample[sample_id][detector] = pred
            else:
                missing_industry[f"{source}_{stage}"] += 1

    write_dict_rows(
        SCRIPT_DIR / "metadata.csv",
        ["sample_id", "dataset", "source", "label", "stage", "generator", "parent_id"],
        metadata_rows,
    )
    write_dict_rows(SCRIPT_DIR / "persuasion_scores.csv", ["sample_id", "pair", "score"], persuasion_rows)
    write_dict_rows(SCRIPT_DIR / "predictions.csv", ["sample_id", "detector", "prediction"], prediction_rows)
    count_rows = [
        {"source": key.split("-")[0], "label": key.split("-")[1], "n": value}
        for key, value in sorted(counts.items())
    ]
    write_dict_rows(SCRIPT_DIR / "sample_counts.csv", ["source", "label", "n"], count_rows)

    with zipfile.ZipFile(SCRIPT_DIR / "sample_level_5_3_2.zip", "w", compression=zipfile.ZIP_DEFLATED) as archive:
        for name in ["metadata.csv", "persuasion_scores.csv", "predictions.csv"]:
            archive.write(SCRIPT_DIR / name, arcname=name)

    score_arrays = {
        pair: {group: np.asarray(values, dtype=float) for group, values in group_map.items()}
        for pair, group_map in score_groups.items()
    }
    common_llm_p_sample_ids = sorted(
        sample_id
        for sample_id, predictions in llm_p_predictions_by_sample.items()
        if sample_id in llm_p_scores_by_sample and all(detector in predictions for detector in DETECTOR_ORDER)
    )
    detector_scores = {
        detector: {pair: defaultdict(list) for pair in PAIR_LABELS} for detector in DETECTOR_ORDER
    }
    for sample_id in common_llm_p_sample_ids:
        scores = llm_p_scores_by_sample[sample_id]
        predictions = llm_p_predictions_by_sample[sample_id]
        for detector in DETECTOR_ORDER:
            outcome = "TP" if predictions[detector] == "P" else "FN"
            for pair, score in zip(PAIR_LABELS, scores):
                detector_scores[detector][pair][outcome].append(score)
    detector_arrays = {
        detector: {
            pair: {outcome: np.asarray(values, dtype=float) for outcome, values in pair_map.items()}
            for pair, pair_map in detector_map.items()
        }
        for detector, detector_map in detector_scores.items()
    }
    diagnostics = {
        "n_samples": len(metadata_rows),
        "n_persuasion_score_rows": len(persuasion_rows),
        "n_prediction_rows": len(prediction_rows),
        "source_label_counts": dict(sorted(counts.items())),
        "detectors": DETECTOR_ORDER,
        "pair_order": PAIR_LABELS,
        "e3_common_llm_p_samples": len(common_llm_p_sample_ids),
        "missing_industry_prediction_rows_by_source_stage": dict(sorted(missing_industry.items())),
        "bootstrap_ci_note": "Full-sample MWU/effect/AUC are reported. Bootstrap CIs resample from the complete comparison groups with replacement; no fixed subsample cap is used.",
        "kappa_interaction_test": {
            "permutations": KAPPA_PERMUTATIONS,
            "bootstraps": KAPPA_BOOTSTRAPS,
            "null": "P/B labels fixed; HW/LLM source labels permuted independently within P and B groups.",
        },
    }
    return diagnostics, score_arrays, detector_arrays


def run_e1(score_arrays):
    rows = []
    for pair in PAIR_LABELS:
        groups = score_arrays[pair]
        p = np.concatenate([groups.get("HW-P", np.array([])), groups.get("LLM-P", np.array([]))])
        b = np.concatenate([groups.get("HW-B", np.array([])), groups.get("LLM-B", np.array([]))])
        stat = mann_whitney(p, b)
        p_med, p_q1, p_q3 = quantiles(p)
        b_med, b_q1, b_q3 = quantiles(b)
        ci_low, ci_high = bootstrap_effect_ci(p, b, seed=1000 + PAIR_LABELS.index(pair), n_boot=300)
        rows.append(
            {
                "pair": pair,
                "P_median": p_med,
                "P_iqr": p_q3 - p_q1,
                "P_q1": p_q1,
                "P_q3": p_q3,
                "B_median": b_med,
                "B_iqr": b_q3 - b_q1,
                "B_q1": b_q1,
                "B_q3": b_q3,
                "U": stat["u"],
                "p": stat["p"],
                "rank_biserial": stat["rank_biserial"],
                "rank_biserial_ci_low": ci_low,
                "rank_biserial_ci_high": ci_high,
                "n_P": len(p),
                "n_B": len(b),
            }
        )
    q_values = bh_fdr([float(row["p"]) for row in rows])
    for row, q in zip(rows, q_values):
        row["q"] = q
    return rows


def run_e1_stratified(score_arrays):
    rows = []
    for source in ["HW", "LLM"]:
        raw_rows = []
        for pair in PAIR_LABELS:
            groups = score_arrays[pair]
            p = groups.get(f"{source}-P", np.array([]))
            b = groups.get(f"{source}-B", np.array([]))
            stat = mann_whitney(p, b)
            p_med, p_q1, p_q3 = quantiles(p)
            b_med, b_q1, b_q3 = quantiles(b)
            raw_rows.append(
                {
                    "source": source,
                    "pair": pair,
                    "P_median": p_med,
                    "P_iqr": p_q3 - p_q1,
                    "B_median": b_med,
                    "B_iqr": b_q3 - b_q1,
                    "U": stat["u"],
                    "p": stat["p"],
                    "rank_biserial": stat["rank_biserial"],
                    "n_P": len(p),
                    "n_B": len(b),
                }
            )
        q_values = bh_fdr([float(row["p"]) for row in raw_rows])
        for row, q in zip(raw_rows, q_values):
            row["q"] = q
        rows.extend(raw_rows)
    return rows


def run_e2(score_arrays):
    rows = []
    for pair in PAIR_LABELS:
        groups = score_arrays[pair]
        llm = groups.get("LLM-P", np.array([]))
        hw = groups.get("HW-P", np.array([]))
        stat = mann_whitney(llm, hw)
        llm_med, llm_q1, llm_q3 = quantiles(llm)
        hw_med, hw_q1, hw_q3 = quantiles(hw)
        auc, auc_low, auc_high = auc_ci(llm, hw, seed=2000 + PAIR_LABELS.index(pair), n_boot=500)
        ci_low, ci_high = bootstrap_effect_ci(llm, hw, seed=3000 + PAIR_LABELS.index(pair), n_boot=300)
        rows.append(
            {
                "pair": pair,
                "LLM_P_median": llm_med,
                "LLM_P_iqr": llm_q3 - llm_q1,
                "HW_P_median": hw_med,
                "HW_P_iqr": hw_q3 - hw_q1,
                "delta_median_LLM_minus_HW": llm_med - hw_med,
                "U": stat["u"],
                "p": stat["p"],
                "rank_biserial": stat["rank_biserial"],
                "rank_biserial_ci_low": ci_low,
                "rank_biserial_ci_high": ci_high,
                "source_auc": auc,
                "source_auc_ci_low": auc_low,
                "source_auc_ci_high": auc_high,
                "n_LLM_P": len(llm),
                "n_HW_P": len(hw),
            }
        )
    q_values = bh_fdr([float(row["p"]) for row in rows])
    for row, q in zip(rows, q_values):
        row["q"] = q
    return rows


def median_shift(group_a, group_b):
    if len(group_a) == 0 or len(group_b) == 0:
        return math.nan
    return float(np.median(group_a) - np.median(group_b))


def bootstrap_kappa_ci(llm_p, hw_p, llm_b, hw_b, seed, n_boot=KAPPA_BOOTSTRAPS):
    if min(len(llm_p), len(hw_p), len(llm_b), len(hw_b)) == 0:
        return math.nan, math.nan
    rng = np.random.default_rng(seed)
    values = np.empty(n_boot, dtype=float)
    for index in range(n_boot):
        sample_llm_p = llm_p[rng.integers(0, len(llm_p), len(llm_p))]
        sample_hw_p = hw_p[rng.integers(0, len(hw_p), len(hw_p))]
        sample_llm_b = llm_b[rng.integers(0, len(llm_b), len(llm_b))]
        sample_hw_b = hw_b[rng.integers(0, len(hw_b), len(hw_b))]
        values[index] = median_shift(sample_llm_p, sample_hw_p) - median_shift(sample_llm_b, sample_hw_b)
    return float(np.percentile(values, 2.5)), float(np.percentile(values, 97.5))


def permutation_kappa_p(llm_p, hw_p, llm_b, hw_b, observed, seed, n_perm=KAPPA_PERMUTATIONS):
    if min(len(llm_p), len(hw_p), len(llm_b), len(hw_b)) == 0 or not math.isfinite(observed):
        return math.nan
    rng = np.random.default_rng(seed)
    p_values = np.concatenate([llm_p, hw_p])
    b_values = np.concatenate([llm_b, hw_b])
    n_llm_p = len(llm_p)
    n_llm_b = len(llm_b)
    extreme = 0
    for _ in range(n_perm):
        p_perm = rng.permutation(len(p_values))
        b_perm = rng.permutation(len(b_values))
        perm_delta_p = median_shift(p_values[p_perm[:n_llm_p]], p_values[p_perm[n_llm_p:]])
        perm_delta_b = median_shift(b_values[b_perm[:n_llm_b]], b_values[b_perm[n_llm_b:]])
        if abs(perm_delta_p - perm_delta_b) >= abs(observed):
            extreme += 1
    return float((extreme + 1) / (n_perm + 1))


def run_e2_interaction(score_arrays):
    rows = []
    for pair in PAIR_LABELS:
        groups = score_arrays[pair]
        llm_p = groups.get("LLM-P", np.array([]))
        hw_p = groups.get("HW-P", np.array([]))
        llm_b = groups.get("LLM-B", np.array([]))
        hw_b = groups.get("HW-B", np.array([]))
        delta_p = median_shift(llm_p, hw_p)
        delta_b = median_shift(llm_b, hw_b)
        kappa = delta_p - delta_b
        ci_low, ci_high = bootstrap_kappa_ci(llm_p, hw_p, llm_b, hw_b, seed=4000 + PAIR_LABELS.index(pair))
        permutation_p = permutation_kappa_p(llm_p, hw_p, llm_b, hw_b, kappa, seed=5000 + PAIR_LABELS.index(pair))
        stat_p = mann_whitney(llm_p, hw_p)
        stat_b = mann_whitney(llm_b, hw_b)
        rows.append(
            {
                "pair": pair,
                "delta_P": delta_p,
                "delta_B": delta_b,
                "kappa": kappa,
                "kappa_ci_low": ci_low,
                "kappa_ci_high": ci_high,
                "permutation_p": permutation_p,
                "rank_biserial_LLM_P_vs_HW_P": stat_p["rank_biserial"],
                "rank_biserial_LLM_B_vs_HW_B": stat_b["rank_biserial"],
                "n_LLM_P": len(llm_p),
                "n_HW_P": len(hw_p),
                "n_LLM_B": len(llm_b),
                "n_HW_B": len(hw_b),
                "permutations": KAPPA_PERMUTATIONS,
                "bootstraps": KAPPA_BOOTSTRAPS,
            }
        )
    q_values = bh_fdr([float(row["permutation_p"]) for row in rows])
    for row, q in zip(rows, q_values):
        row["FDR_q"] = q
    return rows


def run_e3(detector_arrays):
    all_rows = []
    for detector in DETECTOR_ORDER:
        detector_rows = []
        for pair in PAIR_LABELS:
            tp = detector_arrays[detector][pair].get("TP", np.array([]))
            fn = detector_arrays[detector][pair].get("FN", np.array([]))
            stat = mann_whitney(fn, tp)
            tp_med, tp_q1, tp_q3 = quantiles(tp)
            fn_med, fn_q1, fn_q3 = quantiles(fn)
            detector_rows.append(
                {
                    "detector": detector,
                    "pair": pair,
                    "TP_n": len(tp),
                    "FN_n": len(fn),
                    "TP_median": tp_med,
                    "TP_iqr": tp_q3 - tp_q1,
                    "FN_median": fn_med,
                    "FN_iqr": fn_q3 - fn_q1,
                    "gamma_FN_minus_TP": fn_med - tp_med,
                    "U": stat["u"],
                    "p": stat["p"],
                    "rank_biserial_FN_vs_TP": stat["rank_biserial"],
                }
            )
        q_values = bh_fdr([float(row["p"]) for row in detector_rows])
        for row, q in zip(detector_rows, q_values):
            row["q"] = q
        all_rows.extend(detector_rows)
    return all_rows


def draw_main_figure(e1_rows, e1_stratified_rows, e2_rows, e2_interaction_rows, e3_rows):
    e1 = {row["pair"]: row for row in e1_rows}
    e1s = {(row["source"], row["pair"]): row for row in e1_stratified_rows}
    e2 = {row["pair"]: row for row in e2_rows}
    e2i = {row["pair"]: row for row in e2_interaction_rows}
    significant_both = [pair for pair in PAIR_LABELS if e1[pair]["q"] < 0.05 and e2i[pair]["FDR_q"] < 0.05]
    remaining = [pair for pair in PAIR_LABELS if pair not in significant_both]
    significant_both.sort(key=lambda pair: abs(float(e2i[pair]["kappa"])), reverse=True)
    pair_order = significant_both + remaining
    x = np.arange(len(pair_order))

    heat = np.full((len(pair_order), len(DETECTOR_ORDER)), np.nan)
    sig = np.zeros_like(heat, dtype=bool)
    e3_lookup = {(row["pair"], row["detector"]): row for row in e3_rows}
    for row_index, pair in enumerate(pair_order):
        for col_index, detector in enumerate(DETECTOR_ORDER):
            row = e3_lookup.get((pair, detector))
            if row:
                heat[row_index, col_index] = float(row["rank_biserial_FN_vs_TP"])
                sig[row_index, col_index] = float(row["q"]) < 0.05

    hw_effect = np.array([float(e1s[("HW", pair)]["rank_biserial"]) for pair in pair_order])
    llm_effect = np.array([float(e1s[("LLM", pair)]["rank_biserial"]) for pair in pair_order])
    hw_sig = np.array([float(e1s[("HW", pair)]["q"]) < 0.05 for pair in pair_order])
    llm_sig = np.array([float(e1s[("LLM", pair)]["q"]) < 0.05 for pair in pair_order])
    hw_color = "#1f77b4"
    llm_color = "#f2c94c"
    e2_color = "#bd1f36"

    kappa = np.array([float(e2i[pair]["kappa"]) for pair in pair_order])
    e2_sig = np.array([float(e2i[pair]["FDR_q"]) < 0.05 for pair in pair_order])

    fig, axes = plt.subplots(
        2,
        1,
        figsize=(32, 18),
        sharex=True,
        gridspec_kw={"height_ratios": [1.0, 1.1], "hspace": 0.20},
    )
    ax1, ax2 = axes
    offset = 0.16
    ax1.scatter(x[~hw_sig] - offset, hw_effect[~hw_sig], marker="o", s=365, facecolors="white", edgecolors=hw_color, linewidths=4.5, zorder=2)
    ax1.scatter(x[hw_sig] - offset, hw_effect[hw_sig], marker="o", s=385, color=hw_color, edgecolors=hw_color, linewidths=2.2, zorder=3)
    ax1.scatter(x[~llm_sig] + offset, llm_effect[~llm_sig], marker="^", s=390, facecolors="white", edgecolors=llm_color, linewidths=4.5, zorder=2)
    ax1.scatter(x[llm_sig] + offset, llm_effect[llm_sig], marker="^", s=420, color=llm_color, edgecolors="#9a7a00", linewidths=2.2, zorder=3)
    ax1.axhline(0, color="#555555", lw=2.2)
    ymin_sep = min(float(np.min(hw_effect)), float(np.min(llm_effect)), -0.03)
    ymax_sep = max(float(np.max(hw_effect)), float(np.max(llm_effect)), 0.18)
    pad_sep = (ymax_sep - ymin_sep) * 0.12
    ax1.set_ylim(ymin_sep - pad_sep, ymax_sep + pad_sep)
    ax1.set_ylabel("Rank-biserial\nPhishing - Benign", fontsize=46, labelpad=18)
    ax1.grid(axis="y", color="#e2e2e2", lw=2.0)
    ax1.set_title("(a) Phishing-benign separation", loc="left", fontsize=46, weight="bold", pad=6)

    ax2.axhline(0, color="#555555", lw=2.2)
    ax2.vlines(x, 0, kappa, color=e2_color, lw=6.2, alpha=0.90, zorder=1)
    ax2.scatter(x[~e2_sig], kappa[~e2_sig], marker="o", s=365, facecolors="white", edgecolors=e2_color, linewidths=4.5, zorder=2)
    ax2.scatter(x[e2_sig], kappa[e2_sig], marker="o", s=385, color=e2_color, edgecolors=e2_color, linewidths=2.2, zorder=3)
    ymax = max(kappa.max() * 1.18, 0.08)
    ymin = min(kappa.min() * 1.18, -0.08)
    ax2.set_ylim(ymin, ymax)
    ax2.set_ylabel("Interaction shift\n(delta Phishing - delta Benign)", fontsize=46, labelpad=20)
    ax2.set_title("(b) Phishing-specific LLM shift", loc="left", fontsize=46, weight="bold", pad=6)
    ax2.grid(axis="y", color="#e2e2e2", lw=2.0)

    for xi, yi in zip(x[~e2_sig], kappa[~e2_sig]):
        ax2.text(xi, yi + 0.006, "n.s.", ha="center", va="bottom", fontsize=28, color=e2_color)

    ax2.set_xticks(x)
    ax2.set_xticklabels(pair_order, rotation=45, ha="right", fontsize=43)
    for axis in axes:
        axis.tick_params(axis="x", labelsize=43, width=3.4, length=10)
        axis.tick_params(axis="y", labelsize=40, width=3.4, length=10)
        axis.set_xlim(-0.6, len(pair_order) - 0.4)
        axis.spines["top"].set_visible(False)
        axis.spines["right"].set_visible(False)
        for spine in ["left", "bottom"]:
            axis.spines[spine].set_linewidth(3.0)

    source_handles = [
        Line2D([0], [0], marker="o", color=hw_color, markerfacecolor=hw_color, markersize=27, linewidth=0, label="HW Phishing - Benign"),
        Line2D([0], [0], marker="^", color=llm_color, markerfacecolor=llm_color, markeredgecolor="#9a7a00", markersize=27, linewidth=0, label="LLM Phishing - Benign"),
    ]
    status_handles = [
        Line2D([0], [0], marker="o", color="#333333", markerfacecolor="#333333", markersize=23, linewidth=0, label="Significant (q < 0.05)"),
        Line2D([0], [0], marker="o", color="#333333", markerfacecolor="white", markeredgewidth=3.2, markersize=23, linewidth=0, label="Open marker = n.s. (q >= 0.05)"),
    ]
    source_legend = ax1.legend(
        handles=source_handles,
        loc="upper left",
        fontsize=34,
        frameon=False,
        handlelength=2.2,
        handletextpad=0.6,
        markerscale=1.2,
    )
    ax1.add_artist(source_legend)
    ax1.legend(
        handles=status_handles,
        loc="upper center",
        bbox_to_anchor=(0.50, 1.02),
        fontsize=31,
        frameon=False,
        handlelength=1.6,
        handletextpad=0.5,
        markerscale=1.1,
    )
    shift_handle = Line2D([0], [0], marker="o", color=e2_color, markerfacecolor=e2_color, markersize=27, linewidth=7, label="Phishing-specific LLM shift")
    shift_status_handles = [
        Line2D([0], [0], marker="o", color=e2_color, markerfacecolor=e2_color, markersize=23, linewidth=0, label="Significant (q < 0.05)"),
        Line2D([0], [0], marker="o", color=e2_color, markerfacecolor="white", markeredgewidth=3.2, markersize=23, linewidth=0, label="Open marker = n.s. (q >= 0.05)"),
    ]
    ax2.legend(
        handles=[shift_handle] + shift_status_handles,
        loc="lower right",
        fontsize=34,
        frameon=False,
        handlelength=2.2,
        handletextpad=0.6,
        markerscale=1.2,
    )
    fig.tight_layout(pad=0.45)
    fig.savefig(SCRIPT_DIR / "fig_5_3_2_e1_e2_split_compact.png", dpi=300, bbox_inches="tight", pad_inches=0.04)
    fig.savefig(SCRIPT_DIR / "fig_5_3_2_e1_e2_split_compact.pdf", bbox_inches="tight", pad_inches=0.04)
    plt.close(fig)

    y = np.arange(len(pair_order))
    fig, axes = plt.subplots(
        1,
        2,
        figsize=(18.5, 9.2),
        sharey=True,
        gridspec_kw={"width_ratios": [1.0, 1.05], "wspace": 0.10},
    )
    ax_v1, ax_v2 = axes
    voffset = 0.15
    ax_v1.scatter(hw_effect[~hw_sig], y[~hw_sig] - voffset, marker="o", s=90, facecolors="white", edgecolors=hw_color, linewidths=2.0, zorder=2)
    ax_v1.scatter(hw_effect[hw_sig], y[hw_sig] - voffset, marker="o", s=96, color=hw_color, edgecolors=hw_color, linewidths=1.0, zorder=3)
    ax_v1.scatter(llm_effect[~llm_sig], y[~llm_sig] + voffset, marker="^", s=104, facecolors="white", edgecolors=llm_color, linewidths=2.0, zorder=2)
    ax_v1.scatter(llm_effect[llm_sig], y[llm_sig] + voffset, marker="^", s=112, color=llm_color, edgecolors="#9a7a00", linewidths=1.0, zorder=3)
    ax_v1.axvline(0, color="#555555", lw=1.4)
    ax_v1.set_title("(a) Phishing-benign separation", loc="left", fontsize=22, weight="bold", pad=4)
    ax_v1.set_xlabel("Rank-biserial effect, Phishing - Benign", fontsize=20, labelpad=8)
    ax_v1.set_yticks(y)
    ax_v1.set_yticklabels(pair_order, fontsize=17)
    ax_v1.invert_yaxis()
    ax_v1.grid(axis="x", color="#e2e2e2", lw=1.0)
    ax_v1.tick_params(axis="x", labelsize=17, width=1.8, length=6)
    ax_v1.tick_params(axis="y", labelsize=17, width=1.8, length=6, pad=3)
    ax_v1.legend(
        handles=[
            Line2D([0], [0], marker="o", color=hw_color, markerfacecolor=hw_color, markersize=11, linewidth=0, label="HW"),
            Line2D([0], [0], marker="^", color=llm_color, markerfacecolor=llm_color, markeredgecolor="#9a7a00", markersize=11, linewidth=0, label="LLM"),
        ],
        loc="lower right",
        fontsize=17,
        frameon=False,
        handletextpad=0.4,
    )

    ax_v2.axvline(0, color="#555555", lw=1.4)
    ax_v2.hlines(y, 0, kappa, color=e2_color, lw=2.4, alpha=0.9, zorder=1)
    ax_v2.scatter(kappa[~e2_sig], y[~e2_sig], marker="o", s=90, facecolors="white", edgecolors=e2_color, linewidths=2.0, zorder=2)
    ax_v2.scatter(kappa[e2_sig], y[e2_sig], marker="o", s=96, color=e2_color, edgecolors=e2_color, linewidths=1.0, zorder=3)
    ax_v2.set_title("(b) Phishing-specific LLM shift", loc="left", fontsize=22, weight="bold", pad=4)
    ax_v2.set_xlabel("Interaction shift (delta Phishing - delta Benign)", fontsize=20, labelpad=8)
    ax_v2.grid(axis="x", color="#e2e2e2", lw=1.0)
    ax_v2.tick_params(axis="x", labelsize=17, width=1.8, length=6)
    ax_v2.tick_params(axis="y", labelleft=False, width=1.8, length=6)
    ax_v2.legend(
        handles=[Line2D([0], [0], marker="o", color=e2_color, markerfacecolor=e2_color, markersize=11, linewidth=2.4, label="kappa")],
        loc="lower right",
        fontsize=17,
        frameon=False,
        handletextpad=0.4,
    )
    ax_v1.set_xlim(min(float(np.min(hw_effect)), float(np.min(llm_effect)), -0.06) - 0.02, max(float(np.max(hw_effect)), float(np.max(llm_effect)), 0.30) + 0.025)
    ax_v2.set_xlim(min(float(np.min(kappa)), -0.12) - 0.015, max(float(np.max(kappa)), 0.02) + 0.02)
    for axis in axes:
        axis.spines["top"].set_visible(False)
        axis.spines["right"].set_visible(False)
        for spine in ["left", "bottom"]:
            axis.spines[spine].set_linewidth(1.8)
    fig.tight_layout(pad=0.25)
    fig.savefig(SCRIPT_DIR / "fig_5_3_2_e1_e2_vertical_compact.png", dpi=300, bbox_inches="tight", pad_inches=0.035)
    fig.savefig(SCRIPT_DIR / "fig_5_3_2_e1_e2_vertical_compact.pdf", bbox_inches="tight", pad_inches=0.035)
    plt.close(fig)

    fig, ax3 = plt.subplots(figsize=(16.8, 10.2))
    vmax = max(0.05, float(np.nanmax(np.abs(heat))))
    im = ax3.imshow(heat, cmap="RdBu_r", vmin=-vmax, vmax=vmax, aspect="auto")
    ax3.set_xlabel("Detector", fontsize=36, labelpad=12)
    ax3.set_ylabel("Persuasion pair", fontsize=36, labelpad=12)
    ax3.set_xticks(np.arange(len(DETECTOR_ORDER)))
    ax3.set_xticklabels(DETECTOR_DISPLAY_LABELS, rotation=15, ha="right", fontsize=28)
    ax3.set_yticks(np.arange(len(pair_order)))
    ax3.set_yticklabels(pair_order, fontsize=30)
    ax3.tick_params(axis="both", width=3, length=8)
    for row_index in range(heat.shape[0]):
        for col_index in range(heat.shape[1]):
            row = e3_lookup.get((pair_order[row_index], DETECTOR_ORDER[col_index]))
            stars = significance_stars(float(row["q"])) if row else ""
            if stars:
                ax3.text(col_index, row_index, stars, ha="center", va="center", color="black", fontsize=20, weight="bold")
    cbar = fig.colorbar(im, ax=ax3, fraction=0.030, pad=0.012)
    cbar.set_label("Rank-biserial effect, FN vs TP", fontsize=32, labelpad=14)
    cbar.ax.tick_params(labelsize=28, width=2.4, length=7)
    ax3.set_xticks(np.arange(-0.5, len(DETECTOR_ORDER), 1), minor=True)
    ax3.set_yticks(np.arange(-0.5, len(pair_order), 1), minor=True)
    ax3.grid(which="minor", color="white", linewidth=1.1)
    ax3.tick_params(which="minor", bottom=False, left=False)
    for spine in ["left", "bottom", "top", "right"]:
        ax3.spines[spine].set_linewidth(2.4)
    fig.tight_layout(pad=0.18)
    fig.savefig(SCRIPT_DIR / "fig_5_3_2_detector_heatmap.png", dpi=300, bbox_inches="tight", pad_inches=0.025)
    fig.savefig(SCRIPT_DIR / "fig_5_3_2_detector_heatmap.pdf", bbox_inches="tight", pad_inches=0.025)
    plt.close(fig)

    y = np.arange(len(pair_order))
    fig = plt.figure(figsize=(17.4, 10.5))
    gs = fig.add_gridspec(1, 3, width_ratios=[1.05, 2.15, 0.08], wspace=0.14)
    ax_a = fig.add_subplot(gs[0, 0])
    ax_h = fig.add_subplot(gs[0, 1], sharey=ax_a)
    cax = fig.add_subplot(gs[0, 2])

    ax_a.scatter(hw_effect[~hw_sig], y[~hw_sig] - 0.16, s=86, facecolors="white", edgecolors=hw_color, linewidths=1.8, zorder=2)
    ax_a.scatter(hw_effect[hw_sig], y[hw_sig] - 0.16, s=92, color=hw_color, zorder=3)
    ax_a.scatter(llm_effect[~llm_sig], y[~llm_sig] + 0.16, s=86, facecolors="white", edgecolors=llm_color, linewidths=1.8, zorder=2)
    ax_a.scatter(llm_effect[llm_sig], y[llm_sig] + 0.16, s=92, color=llm_color, zorder=3)
    ax_a.axvline(0, color="#555555", lw=1.4)
    ax_a.set_title("(a) Phishing-benign separation", loc="left", fontsize=20, weight="bold", pad=4)
    ax_a.set_xlabel("Rank-biserial effect,\nP vs B", fontsize=17)
    ax_a.set_yticks(y)
    ax_a.set_yticklabels(pair_order, fontsize=15)
    ax_a.invert_yaxis()
    ax_a.grid(axis="x", color="#e2e2e2", lw=1.0)
    ax_a.tick_params(axis="x", labelsize=14)

    vmax = max(0.05, float(np.nanmax(np.abs(heat))))
    im = ax_h.imshow(heat, cmap="RdBu_r", vmin=-vmax, vmax=vmax, aspect="auto")
    ax_h.set_title("(c) Detector relevance: LLM-P FN vs TP", loc="left", fontsize=20, weight="bold", pad=4)
    ax_h.set_xlabel("Detector", fontsize=17)
    ax_h.set_xticks(np.arange(len(DETECTOR_ORDER)))
    ax_h.set_xticklabels(DETECTOR_DISPLAY_LABELS, rotation=15, ha="right", fontsize=12.5)
    ax_h.tick_params(axis="y", labelleft=False)
    for row_index in range(heat.shape[0]):
        for col_index in range(heat.shape[1]):
            row = e3_lookup.get((pair_order[row_index], DETECTOR_ORDER[col_index]))
            stars = significance_stars(float(row["q"])) if row else ""
            if stars:
                ax_h.text(col_index, row_index, stars, ha="center", va="center", color="black", fontsize=9.5, weight="bold")
    ax_h.set_xticks(np.arange(-0.5, len(DETECTOR_ORDER), 1), minor=True)
    ax_h.set_yticks(np.arange(-0.5, len(pair_order), 1), minor=True)
    ax_h.grid(which="minor", color="white", linewidth=0.7)
    ax_h.tick_params(which="minor", bottom=False, left=False)
    cbar = fig.colorbar(im, cax=cax)
    cbar.set_label("Rank-biserial effect,\nFN vs TP", fontsize=15)
    cbar.ax.tick_params(labelsize=13)

    for axis in [ax_a, ax_h]:
        axis.spines["top"].set_visible(False)
        axis.spines["right"].set_visible(False)
    fig.tight_layout(pad=0.35)
    fig.savefig(SCRIPT_DIR / "fig_5_3_2_a_plus_detector_heatmap.png", dpi=300, bbox_inches="tight", pad_inches=0.04)
    fig.savefig(SCRIPT_DIR / "fig_5_3_2_a_plus_detector_heatmap.pdf", bbox_inches="tight", pad_inches=0.04)
    plt.close(fig)



def write_outputs(diagnostics, e1, e1s, e2, e2i, e3):
    write_dict_rows(SCRIPT_DIR / "E1_phishing_relevance.csv", list(e1[0].keys()), e1)
    write_dict_rows(SCRIPT_DIR / "E1_source_stratified_phishing_relevance.csv", list(e1s[0].keys()), e1s)
    write_dict_rows(SCRIPT_DIR / "E2_llm_specific_shift.csv", list(e2[0].keys()), e2)
    write_dict_rows(SCRIPT_DIR / "E2_phishing_specific_llm_shift.csv", list(e2i[0].keys()), e2i)
    write_dict_rows(SCRIPT_DIR / "E3_detector_relevance.csv", list(e3[0].keys()), e3)

    appendix_1 = []
    e1_by_pair = {row["pair"]: row for row in e1}
    e2_by_pair = {row["pair"]: row for row in e2}
    e2i_by_pair = {row["pair"]: row for row in e2i}
    for pair in PAIR_LABELS:
        r1 = e1_by_pair[pair]
        r2 = e2_by_pair[pair]
        r2i = e2i_by_pair[pair]
        appendix_1.append(
            {
                "Pair": pair,
                "P Med.": r1["P_median"],
                "B Med.": r1["B_median"],
                "q_P/B": r1["q"],
                "r_rb_P/B": r1["rank_biserial"],
                "LLM-P Med.": r2["LLM_P_median"],
                "HW-P Med.": r2["HW_P_median"],
                "delta_p": r2["delta_median_LLM_minus_HW"],
                "delta_b": r2i["delta_B"],
                "kappa": r2i["kappa"],
                "q_kappa": r2i["FDR_q"],
                "q_L/H": r2["q"],
                "r_rb_L/H": r2["rank_biserial"],
                "r_rb_LLM_B_vs_HW_B": r2i["rank_biserial_LLM_B_vs_HW_B"],
                "Source AUC [95% CI]": f"{float(r2['source_auc']):.3f} [{float(r2['source_auc_ci_low']):.3f}, {float(r2['source_auc_ci_high']):.3f}]",
            }
        )
    write_dict_rows(SCRIPT_DIR / "appendix_E1_E2_complete_table.csv", list(appendix_1[0].keys()), appendix_1)
    write_dict_rows(SCRIPT_DIR / "appendix_E3_complete_table.csv", list(e3[0].keys()), e3)

    diagnostics["outputs"] = [
        "metadata.csv",
        "sample_counts.csv",
        "persuasion_scores.csv",
        "predictions.csv",
        "detector_caption_mapping.txt",
        "sample_level_5_3_2.zip",
        "E1_phishing_relevance.csv",
        "E1_source_stratified_phishing_relevance.csv",
        "E2_llm_specific_shift.csv",
        "E2_phishing_specific_llm_shift.csv",
        "E3_detector_relevance.csv",
        "appendix_E1_E2_complete_table.csv",
        "appendix_E3_complete_table.csv",
        "fig_5_3_2_e1_e2_split_compact.png",
        "fig_5_3_2_e1_e2_split_compact.pdf",
        "fig_5_3_2_e1_e2_vertical_compact.png",
        "fig_5_3_2_e1_e2_vertical_compact.pdf",
        "fig_5_3_2_detector_heatmap.png",
        "fig_5_3_2_detector_heatmap.pdf",
        "fig_5_3_2_a_plus_detector_heatmap.png",
        "fig_5_3_2_a_plus_detector_heatmap.pdf",
    ]
    with (SCRIPT_DIR / "run_summary.json").open("w", encoding="utf-8") as handle:
        json.dump(diagnostics, handle, indent=2)


def main() :
    diagnostics, score_arrays, detector_arrays = assemble()
    e1 = run_e1(score_arrays)
    e1s = run_e1_stratified(score_arrays)
    e2 = run_e2(score_arrays)
    e2i = run_e2_interaction(score_arrays)
    e3 = run_e3(detector_arrays)
    write_outputs(diagnostics, e1, e1s, e2, e2i, e3)
    draw_main_figure(e1, e1s, e2, e2i, e3)
    print(json.dumps(diagnostics, indent=2))


if __name__ == "__main__":
    main()
