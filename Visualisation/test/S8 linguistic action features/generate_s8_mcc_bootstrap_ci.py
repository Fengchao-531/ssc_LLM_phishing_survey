#!/usr/bin/env python3
import csv
import importlib.util
import math
import random
import re
from collections import defaultdict
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parent
PAIRED_SCRIPT = ROOT / "generate_s8_paired_rq1_rq2.py"
SEED = 20260825
BOOTSTRAPS = 2000

OUTPUT_CSV = ROOT / "S8_detector_generator_MCC_bootstrap.csv"
README = ROOT / "S8_detector_generator_MCC_bootstrap_README.md"
PAIRED_AUDIT = ROOT / "S8_mcc_bootstrap_dataset_audit.csv"


def load_paired_module():
    spec = importlib.util.spec_from_file_location("s8_paired", str(PAIRED_SCRIPT))
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def parse_binary(value):
    try:
        numeric = float(str(value).strip())
    except (TypeError, ValueError):
        return None
    if math.isnan(numeric):
        return None
    return 1 if numeric >= 0.5 else 0


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


def load_raw_outputs_all(module):
    raw = {}
    raw_order = {}
    audit = {}
    for generator, path in module.RAW_PATHS.items():
        rows = read_csv_dicts(path)
        by_prompt = defaultdict(list)
        ordered = []
        invalid = 0
        for row_index, row in enumerate(rows):
            label = parse_binary(row.get("label"))
            subject = row.get("generated_subject", "")
            body = row.get("generated_body", "")
            generated_text = row.get("generated_text", "")
            if label is None or not (subject or body or generated_text):
                invalid += 1
                continue
            record = {
                "raw_index": row_index,
                "prompt_id": row.get("join_key", ""),
                "subject": subject,
                "body": body,
                "generated_text": generated_text,
                "label": label,
                "text": f"{subject}\n{body}".lower(),
            }
            by_prompt[record["prompt_id"]].append(record)
            ordered.append(record)
        raw[generator] = by_prompt
        raw_order[generator] = ordered
        audit[generator] = {
            "generator": generator,
            "raw_rows": len(rows),
            "valid_raw_rows": len(ordered),
            "unique_prompts": len(by_prompt),
            "repeated_outputs": len(ordered) - len(by_prompt),
            "invalid_rows": invalid,
        }
    return raw, raw_order, audit


def attach_predictions_all(module, generator, records):
    stage = module.GENERATOR_STAGE[generator]
    paths = [
        module.REPO_ROOT / "Evaluation/processed-evaluation-datasets/llm/academic" / f"{stage}.csv",
        module.REPO_ROOT / "Evaluation/processed-evaluation-datasets/llm/industry" / f"{stage}.csv",
    ]
    detector_sets = [module.ACADEMIC_DETECTORS, module.INDUSTRY_DETECTORS]
    for path, detectors in zip(paths, detector_sets):
        raw_by_text = defaultdict(list)
        raw_by_body = defaultdict(list)
        for record in records:
            raw_by_text[norm_text(record["subject"], record["body"])].append(record)
            raw_by_body[norm_body(record["body"])].append(record)
            raw_by_body[norm_body(record.get("generated_text", ""))].append(record)

        processed_rows = read_csv_dicts(path)
        if len(processed_rows) != len(records):
            raise RuntimeError(f"{path} has {len(processed_rows)} rows but raw has {len(records)} for {generator}")
        available = {key: list(value) for key, value in raw_by_text.items()}
        for index, row in enumerate(processed_rows):
            key = norm_text(row.get("subject"), row.get("body"))
            if available.get(key):
                record = available[key].pop(0)
            else:
                body_key = norm_body(row.get("body"))
                candidates = raw_by_body.get(body_key, [])
                if not candidates:
                    raise RuntimeError(f"{path} row {index} does not match raw text for {generator}: {key[0][:80]}")
                record = candidates.pop(0)
            label = parse_binary(row.get("label"))
            if label is not None:
                record["label"] = label
            for detector in detectors:
                record[detector] = parse_binary(row.get(detector))
            record["subject"] = row.get("subject", record["subject"])
            record["body"] = row.get("body", record["body"])
            record["text"] = f"{record['subject']}\n{record['body']}".lower()


def build_final_rows(module):
    rng = random.Random(SEED)
    raw, raw_order, audit = load_raw_outputs_all(module)
    for generator in module.GENERATOR_ORDER:
        attach_predictions_all(module, generator, raw_order[generator])
    common_prompts = sorted(set.intersection(*(set(raw[generator]) for generator in module.GENERATOR_ORDER)))
    rows = []
    for generator in module.GENERATOR_ORDER:
        for prompt_id in common_prompts:
            record = rng.choice(raw[generator][prompt_id])
            output = {
                "prompt_id": prompt_id,
                "generator": generator,
                "label": record["label"],
            }
            for detector in module.ACADEMIC_DETECTORS + module.INDUSTRY_DETECTORS:
                output[detector] = record.get(detector)
            rows.append(output)
        audit[generator]["final_common_n"] = len(common_prompts)
    return rows, audit, common_prompts


def mcc_from_counts(tp, tn, fp, fn):
    denom = (tp + fp) * (tp + fn) * (tn + fp) * (tn + fn)
    if denom <= 0:
        return 0.0
    return (tp * tn - fp * fn) / math.sqrt(denom)


def mcc(records, detector):
    tp = tn = fp = fn = 0
    for record in records:
        label = parse_binary(record.get("label"))
        pred = parse_binary(record.get(detector))
        if label is None or pred is None:
            continue
        if label == 1 and pred == 1:
            tp += 1
        elif label == 0 and pred == 0:
            tn += 1
        elif label == 0 and pred == 1:
            fp += 1
        elif label == 1 and pred == 0:
            fn += 1
    return mcc_from_counts(tp, tn, fp, fn), tp, tn, fp, fn


def cluster_counts(records, detector):
    counts = defaultdict(lambda: [0, 0, 0, 0])
    for record in records:
        label = parse_binary(record.get("label"))
        pred = parse_binary(record.get(detector))
        if label is None or pred is None:
            continue
        if label == 1 and pred == 1:
            counts[record["prompt_id"]][0] += 1
        elif label == 0 and pred == 0:
            counts[record["prompt_id"]][1] += 1
        elif label == 0 and pred == 1:
            counts[record["prompt_id"]][2] += 1
        elif label == 1 and pred == 0:
            counts[record["prompt_id"]][3] += 1
    return np.asarray(list(counts.values()), dtype=float)


def bootstrap_ci(records, detector, rng):
    counts = cluster_counts(records, detector)
    if counts.size == 0:
        return float("nan"), float("nan")
    n_clusters = counts.shape[0]
    values = []
    for _ in range(BOOTSTRAPS):
        indices = [rng.randrange(n_clusters) for _ in range(n_clusters)]
        tp, tn, fp, fn = counts[indices].sum(axis=0)
        values.append(mcc_from_counts(tp, tn, fp, fn))
    return float(np.quantile(values, 0.025)), float(np.quantile(values, 0.975))


def main():
    module = load_paired_module()
    rows, audit, common_prompts = build_final_rows(module)
    rng = random.Random(SEED + 31337)
    output = []
    for detector in module.ACADEMIC_DETECTORS + module.INDUSTRY_DETECTORS:
        detector_label = module.DETECTOR_LABELS[detector]
        for generator in module.GENERATOR_ORDER:
            subset = [row for row in rows if row["generator"] == generator and parse_binary(row.get(detector)) is not None]
            point, tp, tn, fp, fn = mcc(subset, detector)
            ci_low, ci_high = bootstrap_ci(subset, detector, rng)
            output.append({
                "Detector": detector_label,
                "Detector_column": detector,
                "Generator": generator,
                "N": len(subset),
                "N_prompts": len(set(row["prompt_id"] for row in subset)),
                "TP": tp,
                "TN": tn,
                "FP": fp,
                "FN": fn,
                "MCC": point,
                "CI_low": ci_low,
                "CI_high": ci_high,
                "bootstrap_unit": "prompt_id",
                "bootstrap_iterations": BOOTSTRAPS,
                "seed": SEED,
            })
    write_csv(
        OUTPUT_CSV,
        output,
        ["Detector", "Detector_column", "Generator", "N", "N_prompts", "TP", "TN", "FP", "FN", "MCC", "CI_low", "CI_high", "bootstrap_unit", "bootstrap_iterations", "seed"],
    )
    audit_rows = [audit[generator] for generator in module.GENERATOR_ORDER]
    write_csv(PAIRED_AUDIT, audit_rows, ["generator", "raw_rows", "valid_raw_rows", "unique_prompts", "repeated_outputs", "invalid_rows", "final_common_n"])
    README.write_text(
        "\n".join(
            [
                "# S8 MCC bootstrap confidence intervals",
                "",
                f"Generated by `{Path(__file__).name}`.",
                "",
                "## Design",
                "",
                f"- Fixed seed: `{SEED}`.",
                f"- Bootstrap iterations: `{BOOTSTRAPS}`.",
                "- Bootstrap unit: `prompt_id` cluster.",
                "- Dataset: S8 LLM outputs, benign + phishing labels.",
                "- For Llama, Ministral, and DeepSeek, one output per prompt is selected with the fixed seed before bootstrapping.",
                "- The common prompt set includes prompts present for all six generators.",
                "- Rows with missing detector prediction are omitted for that detector-generator cell.",
                "",
                "## Outputs",
                "",
                f"- Main table: `{OUTPUT_CSV.name}`",
                f"- Dataset audit: `{PAIRED_AUDIT.name}`",
                "",
                "## Columns",
                "",
                "- `MCC`: point estimate on the fixed selected common prompt set.",
                "- `CI_low`, `CI_high`: percentile 95% cluster-bootstrap confidence interval.",
                "- `TP`, `TN`, `FP`, `FN`: confusion-matrix counts for the point estimate.",
                "- `N_prompts`: number of prompt clusters contributing to the detector-generator cell.",
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    print(f"Common prompt N: {len(common_prompts)}")
    print(f"Wrote {OUTPUT_CSV}")
    print(f"Wrote {PAIRED_AUDIT}")
    print(f"Wrote {README}")


if __name__ == "__main__":
    main()
