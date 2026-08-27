#!/usr/bin/env python3
import csv
import math
import re
import sys
from pathlib import Path


csv.field_size_limit(sys.maxsize)

ROOT = Path(__file__).resolve().parent
PROJECT_ROOT = ROOT.parents[2]
RAW_ROOT = PROJECT_ROOT / "Datasets" / "sublist" / "S6-Stealthy Rewriting"
MERGED = ROOT.parent / "A-I Differences" / "selected_llm_tp_detector_merged.csv"

RAW_ORIGINAL = RAW_ROOT / "HW-P.csv"
RAW_UTA = RAW_ROOT / "UTA-LLM-P.csv"
RAW_MPG = RAW_ROOT / "MPG-LLM-P.csv"

JOINED_ROWS_CSV = ROOT / "s6_controlled_uta_mpg_joined_rows.csv"
PERSUASION_COMPARISON_CSV = ROOT / "s6_controlled_uta_mpg_persuasion_comparison.csv"
DETECTOR_CELLS_CSV = ROOT / "s6_controlled_uta_mpg_detector_outcome_cells.csv"
ORIGINAL_TRANSITION_AUDIT_CSV = ROOT / "s6_controlled_original_transition_join_audit.csv"
SUMMARY_MD = ROOT / "s6_controlled_uta_mpg_join_summary.md"

METHODS = ["UTA", "MPG"]
STAGE_BY_METHOD = {"UTA": "S6-UTA", "MPG": "S6-MPG"}
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


def read_csv(path):
    with path.open(newline="", encoding="utf-8", errors="replace") as input_file:
        return list(csv.DictReader(input_file))


def text_key(subject, body):
    text = (subject or "") + " " + (body or "")
    text = text.replace("\ufeff", "")
    text = re.sub(r"\s+", " ", text).strip().lower()
    return text


def raw_key(row):
    return text_key(row.get("Subject", ""), row.get("Body", ""))


def merged_key(row):
    return text_key(row.get("subject", ""), row.get("body", ""))


def to_float(value):
    try:
        if value == "":
            return None
        return float(value)
    except Exception:
        return None


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


def mean(values):
    return sum(values) / len(values) if values else float("nan")


def median(values):
    if not values:
        return float("nan")
    ordered = sorted(values)
    mid = len(ordered) // 2
    if len(ordered) % 2:
        return ordered[mid]
    return (ordered[mid - 1] + ordered[mid]) / 2.0


def normal_two_sided_from_z(z_value):
    return math.erfc(abs(z_value) / math.sqrt(2.0))


def sign_test_approx(differences):
    nonzero = [value for value in differences if value != 0]
    n = len(nonzero)
    if n == 0:
        return 1.0, 0, 0
    positives = sum(1 for value in nonzero if value > 0)
    negatives = n - positives
    z = (abs(positives - n / 2.0) - 0.5) / math.sqrt(n / 4.0)
    return normal_two_sided_from_z(z), positives, negatives


def mcnemar_approx(b, c):
    if b + c == 0:
        return 1.0
    z = (abs(b - c) - 1.0) / math.sqrt(b + c)
    return normal_two_sided_from_z(z)


def bh_adjust(rows):
    indexed = sorted(
        [(index, row["p_value"]) for index, row in enumerate(rows)],
        key=lambda item: item[1],
    )
    total = len(rows)
    adjusted = [1.0] * total
    running = 1.0
    for rank_from_end, (index, p_value) in enumerate(reversed(indexed), start=1):
        rank = total - rank_from_end + 1
        candidate = min(1.0, p_value * total / rank)
        running = min(running, candidate)
        adjusted[index] = running
    for index, row in enumerate(rows):
        row["q_value"] = adjusted[index]
        row["significant"] = adjusted[index] < 0.05


def prediction_to_outcome(value):
    pred = to_float(value)
    if pred is None:
        return ""
    return "FN" if pred == 0.0 else "TP"


def build_merged_index():
    index = {"UTA": {}, "MPG": {}}
    for row in read_csv(MERGED):
        if row.get("source") != "LLM":
            continue
        if to_float(row.get("label_x", "")) != 1.0:
            continue
        for method, stage in STAGE_BY_METHOD.items():
            if row.get("stage") == stage:
                index[method][merged_key(row)] = row
    return index


def build_joined_rows():
    originals = read_csv(RAW_ORIGINAL)
    uta_rows = read_csv(RAW_UTA)
    mpg_rows = read_csv(RAW_MPG)
    merged_index = build_merged_index()
    n = min(len(originals), len(uta_rows), len(mpg_rows))
    joined = []
    for index in range(n):
        raw_original = originals[index]
        raw_uta = uta_rows[index]
        raw_mpg = mpg_rows[index]
        uta = merged_index["UTA"].get(raw_key(raw_uta))
        mpg = merged_index["MPG"].get(raw_key(raw_mpg))
        row = {
            "row_id": index + 1,
            "original_subject": raw_original.get("Subject", ""),
            "original_body": raw_original.get("Body", ""),
            "uta_subject": raw_uta.get("Subject", ""),
            "uta_body": raw_uta.get("Body", ""),
            "mpg_subject": raw_mpg.get("Subject", ""),
            "mpg_body": raw_mpg.get("Body", ""),
            "uta_joined": bool(uta),
            "mpg_joined": bool(mpg),
        }
        for detector_col, detector_label in DETECTORS:
            row["UTA_" + detector_label + "_prediction"] = uta.get(detector_col, "") if uta else ""
            row["MPG_" + detector_label + "_prediction"] = mpg.get(detector_col, "") if mpg else ""
            row["UTA_" + detector_label + "_outcome"] = prediction_to_outcome(row["UTA_" + detector_label + "_prediction"])
            row["MPG_" + detector_label + "_outcome"] = prediction_to_outcome(row["MPG_" + detector_label + "_prediction"])
        for spec in PAIR_SPECS:
            pair = spec["pair"]
            row["UTA_" + pair] = pair_value(uta, spec) if uta else ""
            row["MPG_" + pair] = pair_value(mpg, spec) if mpg else ""
            row["UTA_minus_MPG_" + pair] = (
                row["UTA_" + pair] - row["MPG_" + pair]
                if uta and mpg
                else ""
            )
        joined.append(row)
    return joined


def write_joined_rows(joined):
    fieldnames = [
        "row_id",
        "original_subject",
        "original_body",
        "uta_subject",
        "uta_body",
        "mpg_subject",
        "mpg_body",
        "uta_joined",
        "mpg_joined",
    ]
    for _, detector_label in DETECTORS:
        fieldnames.extend(
            [
                "UTA_" + detector_label + "_prediction",
                "MPG_" + detector_label + "_prediction",
                "UTA_" + detector_label + "_outcome",
                "MPG_" + detector_label + "_outcome",
            ]
        )
    for pair in PAIR_ORDER:
        fieldnames.extend(["UTA_" + pair, "MPG_" + pair, "UTA_minus_MPG_" + pair])
    with JOINED_ROWS_CSV.open("w", newline="", encoding="utf-8") as output_file:
        writer = csv.DictWriter(output_file, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(joined)


def write_persuasion_comparison(joined):
    rows = []
    complete = [row for row in joined if row["uta_joined"] and row["mpg_joined"]]
    for pair in PAIR_ORDER:
        uta = [row["UTA_" + pair] for row in complete]
        mpg = [row["MPG_" + pair] for row in complete]
        diffs = [row["UTA_minus_MPG_" + pair] for row in complete]
        p_value, n_uta_gt_mpg, n_mpg_gt_uta = sign_test_approx(diffs)
        rows.append(
            {
                "pair": pair,
                "pair_full": PAIR_FULL[pair],
                "n": len(complete),
                "UTA_mean_score": mean(uta),
                "MPG_mean_score": mean(mpg),
                "UTA_minus_MPG_mean_score": mean(diffs),
                "UTA_median_score": median(uta),
                "MPG_median_score": median(mpg),
                "UTA_minus_MPG_median_score": median(diffs),
                "n_UTA_gt_MPG": n_uta_gt_mpg,
                "n_MPG_gt_UTA": n_mpg_gt_uta,
                "p_value": p_value,
            }
        )
    bh_adjust(rows)
    with PERSUASION_COMPARISON_CSV.open("w", newline="", encoding="utf-8") as output_file:
        fieldnames = [
            "pair",
            "pair_full",
            "n",
            "UTA_mean_score",
            "MPG_mean_score",
            "UTA_minus_MPG_mean_score",
            "UTA_median_score",
            "MPG_median_score",
            "UTA_minus_MPG_median_score",
            "n_UTA_gt_MPG",
            "n_MPG_gt_UTA",
            "p_value",
            "q_value",
            "significant",
        ]
        writer = csv.DictWriter(output_file, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    return rows


def write_detector_cells(joined):
    rows = []
    complete = [row for row in joined if row["uta_joined"] and row["mpg_joined"]]
    for _, detector_label in DETECTORS:
        counts = {
            "UTA_FN__MPG_FN": 0,
            "UTA_FN__MPG_TP": 0,
            "UTA_TP__MPG_FN": 0,
            "UTA_TP__MPG_TP": 0,
        }
        usable = 0
        for row in complete:
            uta = row["UTA_" + detector_label + "_outcome"]
            mpg = row["MPG_" + detector_label + "_outcome"]
            if not uta or not mpg:
                continue
            usable += 1
            counts["UTA_" + uta + "__MPG_" + mpg] += 1
        uta_fn = counts["UTA_FN__MPG_FN"] + counts["UTA_FN__MPG_TP"]
        mpg_fn = counts["UTA_FN__MPG_FN"] + counts["UTA_TP__MPG_FN"]
        rows.append(
            {
                "detector_label": detector_label,
                "n": usable,
                "UTA_FN__MPG_FN": counts["UTA_FN__MPG_FN"],
                "UTA_FN__MPG_TP": counts["UTA_FN__MPG_TP"],
                "UTA_TP__MPG_FN": counts["UTA_TP__MPG_FN"],
                "UTA_TP__MPG_TP": counts["UTA_TP__MPG_TP"],
                "UTA_FN_rate": uta_fn / float(usable) if usable else float("nan"),
                "MPG_FN_rate": mpg_fn / float(usable) if usable else float("nan"),
                "UTA_minus_MPG_FN_rate": (uta_fn - mpg_fn) / float(usable) if usable else float("nan"),
                "mcnemar_p_value": mcnemar_approx(counts["UTA_FN__MPG_TP"], counts["UTA_TP__MPG_FN"]),
                "transition_definition": "paired UTA-vs-MPG rewritten outcomes for the same raw row_id; original TP->FN requires original predictions and is audited separately",
            }
        )
    with DETECTOR_CELLS_CSV.open("w", newline="", encoding="utf-8") as output_file:
        fieldnames = [
            "detector_label",
            "n",
            "UTA_FN__MPG_FN",
            "UTA_FN__MPG_TP",
            "UTA_TP__MPG_FN",
            "UTA_TP__MPG_TP",
            "UTA_FN_rate",
            "MPG_FN_rate",
            "UTA_minus_MPG_FN_rate",
            "mcnemar_p_value",
            "transition_definition",
        ]
        writer = csv.DictWriter(output_file, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    return rows


def write_original_transition_audit(joined):
    rows = [
        {
            "needed_input": "raw HW-P original persuasion scores",
            "available_joined_rows": 0,
            "required_rows": len(joined),
            "status": "missing",
            "consequence": "cannot compute UTA_delta = UTA - original or MPG_delta = MPG - original",
        },
        {
            "needed_input": "raw HW-P original SecureNet/V3 predictions",
            "available_joined_rows": 0,
            "required_rows": len(joined),
            "status": "missing",
            "consequence": "cannot compute original TP->FN / TP->TP transitions yet",
        },
    ]
    with ORIGINAL_TRANSITION_AUDIT_CSV.open("w", newline="", encoding="utf-8") as output_file:
        fieldnames = ["needed_input", "available_joined_rows", "required_rows", "status", "consequence"]
        writer = csv.DictWriter(output_file, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    return rows


def fmt(value, digits=4, signed=False):
    if value is None or (isinstance(value, float) and math.isnan(value)):
        return ""
    template = "{:+" if signed else "{:"
    return (template + "." + str(digits) + "f}").format(value)


def write_summary(joined, persuasion_rows, detector_rows):
    n_uta = sum(1 for row in joined if row["uta_joined"])
    n_mpg = sum(1 for row in joined if row["mpg_joined"])
    top = sorted(persuasion_rows, key=lambda row: abs(row["UTA_minus_MPG_median_score"]), reverse=True)[:8]
    lines = [
        "# S6 Controlled UTA-vs-MPG Join",
        "",
        "Raw `HW-P.csv`, `UTA-LLM-P.csv`, and `MPG-LLM-P.csv` are joined by row_id. UTA and MPG rewritten rows are then matched back to the current merged scored file by normalized subject+body text.",
        "",
        "- UTA rewritten rows joined: {}/{}".format(n_uta, len(joined)),
        "- MPG rewritten rows joined: {}/{}".format(n_mpg, len(joined)),
        "",
        "Important limitation: raw HW-P original persuasion scores and SecureNet/V3 predictions are not present in the current scored outputs, so original->rewrite TP->FN / TP->TP transitions are not computed here. See `s6_controlled_original_transition_join_audit.csv`.",
        "",
        "## UTA vs MPG Paired Persuasion Scores",
        "",
        "| Pair | UTA median | MPG median | UTA-MPG median | q | UTA>MPG | MPG>UTA |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for row in top:
        lines.append(
            "| {pair} | {uta} | {mpg} | {diff} | {q:.3g} | {ugt} | {mgt} |".format(
                pair=row["pair"],
                uta=fmt(row["UTA_median_score"]),
                mpg=fmt(row["MPG_median_score"]),
                diff=fmt(row["UTA_minus_MPG_median_score"], signed=True),
                q=row["q_value"],
                ugt=row["n_UTA_gt_MPG"],
                mgt=row["n_MPG_gt_UTA"],
            )
        )
    lines.extend(
        [
            "",
            "## Paired Rewritten Detector Outcomes",
            "",
            "| Detector | n | UTA FN / MPG FN | UTA FN / MPG TP | UTA TP / MPG FN | UTA TP / MPG TP | UTA FN rate | MPG FN rate | McNemar p |",
            "|---|---:|---:|---:|---:|---:|---:|---:|---:|",
        ]
    )
    for row in detector_rows:
        lines.append(
            "| {detector_label} | {n} | {ff} | {ft} | {tf} | {tt} | {ufn} | {mfn} | {p:.3g} |".format(
                detector_label=row["detector_label"],
                n=row["n"],
                ff=row["UTA_FN__MPG_FN"],
                ft=row["UTA_FN__MPG_TP"],
                tf=row["UTA_TP__MPG_FN"],
                tt=row["UTA_TP__MPG_TP"],
                ufn=fmt(row["UTA_FN_rate"]),
                mfn=fmt(row["MPG_FN_rate"]),
                p=row["mcnemar_p_value"],
            )
        )
    lines.extend(
        [
            "",
            "## Outputs",
            "",
            "- Joined row table: `s6_controlled_uta_mpg_joined_rows.csv`",
            "- Paired persuasion comparison: `s6_controlled_uta_mpg_persuasion_comparison.csv`",
            "- Paired rewritten detector outcome cells: `s6_controlled_uta_mpg_detector_outcome_cells.csv`",
            "- Missing original-baseline audit: `s6_controlled_original_transition_join_audit.csv`",
        ]
    )
    SUMMARY_MD.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main():
    joined = build_joined_rows()
    write_joined_rows(joined)
    persuasion_rows = write_persuasion_comparison(joined)
    detector_rows = write_detector_cells(joined)
    write_original_transition_audit(joined)
    write_summary(joined, persuasion_rows, detector_rows)


if __name__ == "__main__":
    main()
