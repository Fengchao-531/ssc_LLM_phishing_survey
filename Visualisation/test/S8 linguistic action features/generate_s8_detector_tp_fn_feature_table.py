#!/usr/bin/env python3
import csv
import math
import re
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Optional, Set, Tuple


ROOT = Path(__file__).resolve().parent
REPO_ROOT = ROOT.parents[2]
OUTPUT_CSV = ROOT / "s8_detector_tp_fn_feature_prevalence.csv"
TOP_CSV = ROOT / "s8_detector_tp_fn_feature_top_differences.csv"
SUMMARY_MD = ROOT / "s8_detector_tp_fn_feature_summary.md"

GENERATORS = [
    "S8-claude",
    "S8-gpt",
    "S8-gemini",
    "S8-llama",
    "S8-ministral",
    "S8-deepseek",
]

GENERATOR_LABELS = {
    "S8-claude": "Claude",
    "S8-gpt": "GPT",
    "S8-gemini": "Gemini",
    "S8-llama": "Llama",
    "S8-ministral": "Ministral",
    "S8-deepseek": "DeepSeek",
}

DETECTORS = [
    {
        "detector": "securenet_llama",
        "detector_label": "SecureNet",
        "input_dir": REPO_ROOT / "Evaluation" / "processed-evaluation-datasets" / "llm" / "academic",
        "prediction_column": "securenet_llama",
    },
    {
        "detector": "email_phishing_detection_v3",
        "detector_label": "PhishingV3",
        "input_dir": REPO_ROOT / "Evaluation" / "processed-evaluation-datasets" / "llm" / "industry",
        "prediction_column": "email_phishing_detection_v3_prediction",
    },
]

FEATURES = [
    (
        "Urgency wording",
        r"\b(urgent|immediately|as soon as possible|asap|deadline|expires?|expiring|suspended?|locked|limited time|within \d+|final notice|act now)\b",
    ),
    (
        "Login/account action",
        r"\b(log ?in|sign ?in|account|password|credential|username|authentication|verify your account|account verification|account update)\b",
    ),
    (
        "Information submission",
        r"\b(submit|provide|send|enter|input|fill|complete|confirm|verify|update).{0,45}\b(info|information|details|credential|password|account|address|payment|card|code|otp)\b",
    ),
    (
        "Click/open request",
        r"\b(click|tap|open|visit|follow (the )?link|use (the )?link|press (the )?button|button below|link below)\b",
    ),
    (
        "Softened request",
        r"\b(please|kindly|could you|would you|when you have a chance|at your convenience|we would appreciate|if possible|just wanted|quick note)\b",
    ),
    (
        "Explicit action request",
        r"\b(click|tap|open|visit|go to|follow|log ?in|sign ?in|submit|provide|enter|verify|confirm|update|download|review|complete)\b",
    ),
    (
        "Direct URL/page instruction",
        r"\b(https?://|www\.|url|link|webpage|website|portal|page|site|landing page|dashboard)\b",
    ),
    (
        "Conversational wording",
        r"\b(hi|hello|hey|dear|thanks|thank you|hope you|checking in|following up|best regards|regards|cheers)\b",
    ),
]


def parse_binary(value: str) -> Optional[int]:
    try:
        numeric = float(str(value).strip())
    except ValueError:
        return None
    if math.isnan(numeric):
        return None
    return 1 if numeric >= 0.5 else 0


def bh_fdr(p_values: List[float]) -> List[float]:
    indexed = sorted(enumerate(p_values), key=lambda item: item[1])
    adjusted = [1.0] * len(p_values)
    running_min = 1.0
    total = len(p_values)
    for rank, (original_index, p_value) in reversed(list(enumerate(indexed, start=1))):
        running_min = min(running_min, p_value * total / rank)
        adjusted[original_index] = min(running_min, 1.0)
    return adjusted


def log_comb(n_value: int, k_value: int) -> float:
    if k_value < 0 or k_value > n_value:
        return float("-inf")
    return (
        math.lgamma(n_value + 1)
        - math.lgamma(k_value + 1)
        - math.lgamma(n_value - k_value + 1)
    )


def hypergeom_probability(x_value: int, row_1: int, row_2: int, col_1: int) -> float:
    total = row_1 + row_2
    return math.exp(
        log_comb(row_1, x_value)
        + log_comb(row_2, col_1 - x_value)
        - log_comb(total, col_1)
    )


def fisher_exact_two_sided(a_value: int, b_value: int, c_value: int, d_value: int) -> float:
    row_1 = a_value + b_value
    row_2 = c_value + d_value
    col_1 = a_value + c_value
    if row_1 == 0 or row_2 == 0:
        return 1.0

    lower = max(0, col_1 - row_2)
    upper = min(row_1, col_1)
    observed = hypergeom_probability(a_value, row_1, row_2, col_1)
    p_value = 0.0
    epsilon = 1e-12
    for x_value in range(lower, upper + 1):
        probability = hypergeom_probability(x_value, row_1, row_2, col_1)
        if probability <= observed + epsilon:
            p_value += probability
    return min(p_value, 1.0)


def two_proportion_z_test(success_a: int, n_a: int, success_b: int, n_b: int) -> float:
    if n_a == 0 or n_b == 0:
        return 1.0
    rate_a = success_a / n_a
    rate_b = success_b / n_b
    pooled = (success_a + success_b) / (n_a + n_b)
    standard_error = math.sqrt(pooled * (1.0 - pooled) * (1.0 / n_a + 1.0 / n_b))
    if standard_error == 0:
        return 1.0
    z_score = (rate_a - rate_b) / standard_error
    return math.erfc(abs(z_score) / math.sqrt(2.0))


def feature_difference_p_value(tp_count: int, tp_n: int, fn_count: int, fn_n: int) -> float:
    if min(tp_n, fn_n) <= 20:
        return fisher_exact_two_sided(
            tp_count,
            tp_n - tp_count,
            fn_count,
            fn_n - fn_count,
        )
    return two_proportion_z_test(tp_count, tp_n, fn_count, fn_n)


def read_detector_rows(detector_info: Dict[str, object]) -> List[Dict[str, object]]:
    rows = []
    input_dir = detector_info["input_dir"]
    prediction_column = str(detector_info["prediction_column"])
    for generator in GENERATORS:
        path = input_dir / f"{generator}.csv"
        with path.open(newline="", encoding="utf-8", errors="replace") as handle:
            reader = csv.DictReader(handle)
            required = {"subject", "body", "label", prediction_column}
            missing = required.difference(reader.fieldnames or [])
            if missing:
                raise RuntimeError(f"{path} missing columns: {sorted(missing)}")

            for row in reader:
                label = parse_binary(row.get("label", ""))
                prediction = parse_binary(row.get(prediction_column, ""))
                if label != 1 or prediction is None:
                    continue
                rows.append(
                    {
                        "generator": generator,
                        "generator_label": GENERATOR_LABELS[generator],
                        "detector": detector_info["detector"],
                        "detector_label": detector_info["detector_label"],
                        "outcome": "TP" if prediction == 1 else "FN",
                        "text": f"{row.get('subject', '')}\n{row.get('body', '')}".lower(),
                    }
                )
    return rows


def format_float(value: float) -> str:
    return f"{value:.6g}"


def build_table() -> List[Dict[str, object]]:
    compiled_features = [
        (feature, re.compile(pattern, flags=re.IGNORECASE | re.DOTALL))
        for feature, pattern in FEATURES
    ]
    output_rows = []

    for detector_info in DETECTORS:
        rows = read_detector_rows(detector_info)
        by_generator = defaultdict(list)
        for row in rows:
            by_generator[str(row["generator"])].append(row)

        for generator in GENERATORS:
            generator_rows = by_generator[generator]
            tp_rows = [row for row in generator_rows if row["outcome"] == "TP"]
            fn_rows = [row for row in generator_rows if row["outcome"] == "FN"]
            tp_n = len(tp_rows)
            fn_n = len(fn_rows)
            total_n = len(generator_rows)

            for feature, pattern in compiled_features:
                tp_count = sum(1 for row in tp_rows if pattern.search(str(row["text"])))
                fn_count = sum(1 for row in fn_rows if pattern.search(str(row["text"])))
                tp_prevalence = tp_count / tp_n if tp_n else 0.0
                fn_prevalence = fn_count / fn_n if fn_n else 0.0
                p_value = feature_difference_p_value(tp_count, tp_n, fn_count, fn_n)
                output_rows.append(
                    {
                        "generator": generator,
                        "generator_label": GENERATOR_LABELS[generator],
                        "detector": detector_info["detector"],
                        "detector_label": detector_info["detector_label"],
                        "N": total_n,
                        "TP": tp_n,
                        "FN": fn_n,
                        "feature": feature,
                        "tp_count": tp_count,
                        "fn_count": fn_count,
                        "tp_prevalence": tp_prevalence,
                        "fn_prevalence": fn_prevalence,
                        "difference_tp_minus_fn": tp_prevalence - fn_prevalence,
                        "p_value": p_value,
                    }
                )

    q_values = bh_fdr([float(row["p_value"]) for row in output_rows])
    for row, q_value in zip(output_rows, q_values):
        row["q_value"] = q_value
    return output_rows


def write_csv(path: Path, rows: List[Dict[str, object]], fieldnames: List[str]) -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            formatted = dict(row)
            for key in ["tp_prevalence", "fn_prevalence", "difference_tp_minus_fn", "p_value", "q_value"]:
                if key in formatted:
                    formatted[key] = format_float(float(formatted[key]))
            writer.writerow(formatted)


def write_summary(rows: List[Dict[str, object]], top_rows: List[Dict[str, object]]) -> None:
    lines = [
        "# S8 detector TP/FN feature prevalence",
        "",
        "This table links generator, detector outcome, and action/linguistic characteristics for phishing-labelled S8 LLM outputs.",
        "",
        "## Detector-generator sample counts",
        "",
        "| Detector | Generator | N | TP | FN |",
        "| --- | --- | ---: | ---: | ---: |",
    ]
    seen = set()  # type: Set[Tuple[str, str]]
    for row in rows:
        key = (str(row["detector_label"]), str(row["generator_label"]))
        if key in seen:
            continue
        seen.add(key)
        lines.append(
            f"| {row['detector_label']} | {row['generator_label']} | {row['N']} | {row['TP']} | {row['FN']} |"
        )

    lines.extend(
        [
            "",
            "## Largest TP-FN feature differences",
            "",
            "| Detector | Generator | Feature | TP prev. | FN prev. | TP-FN | p | q |",
            "| --- | --- | --- | ---: | ---: | ---: | ---: | ---: |",
        ]
    )
    for row in top_rows:
        lines.append(
            "| {detector_label} | {generator_label} | {feature} | {tp:.1f}% | {fn:.1f}% | {diff:+.1f} pp | {p} | {q} |".format(
                detector_label=row["detector_label"],
                generator_label=row["generator_label"],
                feature=row["feature"],
                tp=float(row["tp_prevalence"]) * 100,
                fn=float(row["fn_prevalence"]) * 100,
                diff=float(row["difference_tp_minus_fn"]) * 100,
                p=format_float(float(row["p_value"])),
                q=format_float(float(row["q_value"])),
            )
        )
    SUMMARY_MD.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    rows = build_table()
    fieldnames = [
        "generator",
        "generator_label",
        "detector",
        "detector_label",
        "N",
        "TP",
        "FN",
        "feature",
        "tp_count",
        "fn_count",
        "tp_prevalence",
        "fn_prevalence",
        "difference_tp_minus_fn",
        "p_value",
        "q_value",
    ]
    write_csv(OUTPUT_CSV, rows, fieldnames)

    top_rows = sorted(
        rows,
        key=lambda row: (
            str(row["detector"]),
            str(row["generator"]),
            -abs(float(row["difference_tp_minus_fn"])),
        ),
    )
    selected_top = []
    counts = defaultdict(int)  # type: Dict[Tuple[str, str], int]
    for row in top_rows:
        key = (str(row["detector"]), str(row["generator"]))
        if counts[key] >= 3:
            continue
        selected_top.append(row)
        counts[key] += 1

    write_csv(TOP_CSV, selected_top, fieldnames)
    write_summary(rows, selected_top)

    print(f"Wrote {OUTPUT_CSV}")
    print(f"Wrote {TOP_CSV}")
    print(f"Wrote {SUMMARY_MD}")


if __name__ == "__main__":
    main()
