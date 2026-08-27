#!/usr/bin/env python3
import csv
import importlib.util
import random
from collections import defaultdict
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parent
PAIRED_SCRIPT = ROOT / "generate_s8_paired_rq1_rq2.py"
ITERATIONS = 500
SEED = 20260825

REPEAT_GENERATORS = ["Llama", "Ministral", "DeepSeek"]
DETECTORS = [
    ("SecureNet", "securenet_llama"),
    ("PhishingV3", "email_phishing_detection_v3_prediction"),
    ("XGBoost", "xgboost"),
]
FEATURES = [
    "Direct URL/page instruction",
    "Click/open request",
    "Urgency wording",
]

ITERATION_CSV = ROOT / "S8_repeat_selection_robustness_iterations.csv"
SUMMARY_CSV = ROOT / "S8_repeat_selection_robustness_summary.csv"
FIGURE = ROOT / "Fig_S8_App2_repeat_selection_robustness.png"


def load_paired_module():
    spec = importlib.util.spec_from_file_location("s8_paired", str(PAIRED_SCRIPT))
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


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


def quantile(values, q_value):
    if not values:
        return float("nan")
    return float(np.quantile(np.asarray(values, dtype=float), q_value))


def prepare_raw(module):
    raw, raw_order, _audit = module.load_raw_outputs()
    for generator in module.GENERATOR_ORDER:
        module.attach_predictions(generator, raw_order[generator])
        for record in raw_order[generator]:
            record["_robustness_features"] = {
                feature: 1 if dict(module.COMPILED_FEATURES)[feature].search(record["text"]) else 0
                for feature in FEATURES
            }
    common_prompts = sorted(set.intersection(*(set(raw[generator]) for generator in module.GENERATOR_ORDER)))
    return module, raw, common_prompts


def select_iteration(module, raw, common_prompts, rng):
    selected = {}
    for generator in module.GENERATOR_ORDER:
        selected[generator] = {}
        for prompt_id in common_prompts:
            if generator in REPEAT_GENERATORS:
                selected[generator][prompt_id] = rng.choice(raw[generator][prompt_id])
            else:
                selected[generator][prompt_id] = raw[generator][prompt_id][0]
    return selected


def detection_rate(records, detector_col):
    values = [module_parse_int(record.get(detector_col)) for record in records]
    values = [value for value in values if value is not None]
    return sum(values) / len(values) if values else float("nan")


def module_parse_int(value):
    if value in ("", None):
        return None
    try:
        return int(float(value))
    except ValueError:
        return None


def feature_rate(records, feature_name, module):
    return sum(record["_robustness_features"][feature_name] for record in records) / len(records) if records else float("nan")


def run_iterations():
    module, raw, common_prompts = prepare_raw(load_paired_module())
    rng = random.Random(SEED + 1009)
    rows = []

    for iteration in range(ITERATIONS):
        selected = select_iteration(module, raw, common_prompts, rng)
        for generator in REPEAT_GENERATORS:
            records = [selected[generator][prompt_id] for prompt_id in common_prompts]
            for detector_label, detector_col in DETECTORS:
                rows.append({
                    "iteration": iteration,
                    "metric_type": "detection_rate",
                    "detector": detector_label,
                    "feature": "",
                    "generator": generator,
                    "value": detection_rate(records, detector_col),
                })
            for feature in FEATURES:
                rows.append({
                    "iteration": iteration,
                    "metric_type": "feature_prevalence",
                    "detector": "",
                    "feature": feature,
                    "generator": generator,
                    "value": feature_rate(records, feature, module),
                })

        for feature in FEATURES:
            rates = {}
            for generator in module.GENERATOR_ORDER:
                records = [selected[generator][prompt_id] for prompt_id in common_prompts]
                rates[generator] = feature_rate(records, feature, module)
            rows.append({
                "iteration": iteration,
                "metric_type": "feature_range",
                "detector": "",
                "feature": feature,
                "generator": "all six",
                "value": max(rates.values()) - min(rates.values()),
            })
    return rows


def summarize(rows):
    grouped = defaultdict(list)
    for row in rows:
        key = (row["metric_type"], row["detector"], row["feature"], row["generator"])
        grouped[key].append(float(row["value"]))
    summary = []
    for key, values in sorted(grouped.items()):
        metric_type, detector, feature, generator = key
        summary.append({
            "metric_type": metric_type,
            "detector": detector,
            "feature": feature,
            "generator": generator,
            "iterations": len(values),
            "mean": float(np.mean(values)),
            "sd": float(np.std(values, ddof=1)) if len(values) > 1 else 0.0,
            "min": min(values),
            "p2_5": quantile(values, 0.025),
            "median": quantile(values, 0.5),
            "p97_5": quantile(values, 0.975),
            "max": max(values),
        })
    return summary


def draw_figure(rows):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    figure, axes = plt.subplots(1, 2, figsize=(14.5, 5.2))

    det_data = []
    det_labels = []
    for detector, _ in DETECTORS:
        for generator in REPEAT_GENERATORS:
            values = [
                float(row["value"]) * 100.0
                for row in rows
                if row["metric_type"] == "detection_rate"
                and row["detector"] == detector
                and row["generator"] == generator
            ]
            det_data.append(values)
            det_labels.append(f"{detector}\n{generator}")
    axes[0].boxplot(det_data, labels=det_labels, showfliers=False)
    axes[0].set_ylabel("Detection rate (%)")
    axes[0].set_title("A. Detection rates under repeat selection")
    axes[0].tick_params(axis="x", rotation=45)
    axes[0].grid(axis="y", color="#dddddd")

    feat_data = []
    feat_labels = []
    for feature in FEATURES:
        values = [
            float(row["value"]) * 100.0
            for row in rows
            if row["metric_type"] == "feature_range" and row["feature"] == feature
        ]
        feat_data.append(values)
        feat_labels.append(feature)
    axes[1].boxplot(feat_data, labels=feat_labels, showfliers=False)
    axes[1].set_ylabel("Generator range (pp)")
    axes[1].set_title("B. Feature generator-range stability")
    axes[1].tick_params(axis="x", rotation=25)
    axes[1].grid(axis="y", color="#dddddd")

    figure.suptitle("Fig. S8-App2. Repeat-selection robustness")
    figure.tight_layout()
    figure.savefig(FIGURE, dpi=300, bbox_inches="tight")
    plt.close(figure)


def main():
    rows = run_iterations()
    write_csv(ITERATION_CSV, rows, ["iteration", "metric_type", "detector", "feature", "generator", "value"])
    summary = summarize(rows)
    write_csv(SUMMARY_CSV, summary, ["metric_type", "detector", "feature", "generator", "iterations", "mean", "sd", "min", "p2_5", "median", "p97_5", "max"])
    draw_figure(rows)
    print(f"Wrote {ITERATION_CSV}")
    print(f"Wrote {SUMMARY_CSV}")
    print(f"Wrote {FIGURE}")


if __name__ == "__main__":
    main()
