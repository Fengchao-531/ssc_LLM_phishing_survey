#!/usr/bin/env python3
from __future__ import annotations

import json
from itertools import combinations_with_replacement
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import mannwhitneyu, spearmanr
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, brier_score_loss, roc_auc_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import PolynomialFeatures, StandardScaler
from statsmodels.stats.multitest import multipletests


ROOT = Path(__file__).resolve().parent
PROJECT_ROOT = ROOT.parents[2]
INPUT = PROJECT_ROOT / "Visualization" / "test" / "A-I Differences" / "selected_llm_tp_detector_merged.csv"
MCC_TABLE = PROJECT_ROOT / "Evaluation" / "stage-transfer-trend" / "mcc_stage_detector_overview_all.csv"

SAMPLE_LEVEL_CSV = ROOT / "s6_sample_level_paired_rewriting_scores.csv"
PERFORMANCE_CSV = ROOT / "s6_detector_performance_with_score_change.csv"
PERFORMANCE_COMPACT_CSV = ROOT / "s6_detector_performance_compact_table.csv"
PERFORMANCE_COMPACT_MD = ROOT / "s6_detector_performance_compact_table.md"
OVERALL_SHIFT_CSV = ROOT / "s6_method_overall_persuasion_shift.csv"
FEATURE_SHIFT_CSV = ROOT / "s6_method_pair_feature_shift.csv"
TOP_FEATURE_SHIFT_CSV = ROOT / "s6_method_top_changed_pairs.csv"
FAILURE_ASSOC_CSV = ROOT / "s6_failure_associated_pair_changes.csv"
TOP_FAILURE_ASSOC_CSV = ROOT / "s6_top_failure_associated_pair_changes.csv"
CORRELATION_CSV = ROOT / "s6_feature_change_detector_score_correlation.csv"
FIT_METRICS_CSV = ROOT / "s6_detector_surrogate_fit_metrics.csv"
SCARCITY_DIRECTION_CSV = ROOT / "s6_fuzzer_scarcity_direction_by_outcome.csv"
SCARCITY_EXAMPLES_CSV = ROOT / "s6_fuzzer_scarcity_top_fn_examples.csv"
INDEPENDENCE_CSV = ROOT / "s6_rewriting_independence_check.csv"
METHODOLOGY_MD = ROOT / "s6_rewriting_methodology_notes.md"
REWRITTEN_FN_TP_CSV = ROOT / "s6_rewritten_fn_tp_pair_characteristics.csv"
REWRITTEN_FN_TP_FIGURE = ROOT / "fig_s6_rewritten_fn_tp_pair_characteristics.png"
REWRITTEN_FN_TP_CONSISTENCY_CSV = ROOT / "s6_rewritten_fn_tp_detector_consistency.csv"
CONTROLLED_MAPPING_AUDIT_MD = ROOT / "s6_uta_mpg_controlled_mapping_audit.md"
CONTROLLED_JOIN_SUMMARY_MD = ROOT / "s6_controlled_uta_mpg_join_summary.md"
CONTROLLED_PERSUASION_COMPARISON_CSV = ROOT / "s6_controlled_uta_mpg_persuasion_comparison.csv"
CONTROLLED_DETECTOR_CELLS_CSV = ROOT / "s6_controlled_uta_mpg_detector_outcome_cells.csv"
CONTROLLED_ORIGINAL_TRANSITION_AUDIT_CSV = ROOT / "s6_controlled_original_transition_join_audit.csv"
SUMMARY_MD = ROOT / "s6_rewriting_method_impact_summary.md"
SUMMARY_JSON = ROOT / "s6_rewriting_method_impact_outputs.json"

METHODS = ["S6-fuzzer", "S6-UTA", "S6-MPG"]
METHOD_LABELS = {
    "S6-fuzzer": "Fuzzer",
    "S6-UTA": "UTA",
    "S6-MPG": "MPG",
}

DETECTORS = {
    "scamllm": "ScamLLM",
    "pimref": "PiMRef",
    "t5phishing": "T5Phishing",
    "xgboost": "XGBoost",
    "securenet_llama": "SecureNet",
    "email_phishing_detection_v3_prediction": "V3",
}

MCC_DETECTOR_NAMES = {
    "email_phishing_detection_v3_prediction": "v3",
}

PRINCIPLES = [
    ("A", "Authority", "principle_authority"),
    ("L", "Liking", "principle_liking"),
    ("R", "Reciprocity", "principle_reciprocity"),
    ("SP", "Social Proof", "principle_social_proof"),
    ("S", "Scarcity", "principle_scarcity"),
    ("C", "Commitment", "principle_commitment"),
]


def pair_specs() -> list[dict[str, object]]:
    specs = []
    for left, right in combinations_with_replacement(PRINCIPLES, 2):
        left_short, left_name, left_col = left
        right_short, right_name, right_col = right
        specs.append(
            {
                "pair": f"{left_short}-{right_short}",
                "pair_full": f"{left_name} + {right_name}",
                "left_col": left_col,
                "right_col": right_col,
                "diagonal": left_col == right_col,
            }
        )
    return specs


def normalize_numeric(frame: pd.DataFrame, columns: list[str]) -> None:
    for column in columns:
        frame[column] = pd.to_numeric(frame[column], errors="coerce")


def load_frame() -> pd.DataFrame:
    usecols = [
        "source",
        "stage",
        "subject",
        "body",
        "text",
        "source_file",
        "label_x",
        "proj_x",
        "proj_y",
        "industry_pred_available",
        *DETECTORS.keys(),
        *[column for _, _, column in PRINCIPLES],
    ]
    frame = pd.read_csv(INPUT, usecols=usecols, low_memory=False)
    frame["source"] = frame["source"].fillna("").astype(str)
    frame["stage"] = frame["stage"].fillna("").astype(str)
    normalize_numeric(
        frame,
        [
            "label_x",
            "proj_x",
            "proj_y",
            "industry_pred_available",
            *DETECTORS.keys(),
            *[column for _, _, column in PRINCIPLES],
        ],
    )
    return frame[
        frame["label_x"].eq(1.0)
        & frame["stage"].isin(METHODS)
        & frame["source"].isin(["HW", "LLM"])
        & frame["proj_x"].notna()
        & frame["proj_y"].notna()
    ].copy()


def fit_detector_surfaces(frame: pd.DataFrame, seed: int = 7) -> tuple[dict[str, Pipeline], pd.DataFrame]:
    models: dict[str, Pipeline] = {}
    fit_rows = []
    for detector, detector_label in DETECTORS.items():
        train = frame[frame[detector].notna()].copy()
        if detector == "email_phishing_detection_v3_prediction":
            train = train[train["industry_pred_available"].eq(1.0)].copy()
        y = train[detector].astype(int).to_numpy()
        if np.unique(y).size < 2:
            continue
        model = Pipeline(
            [
                ("poly", PolynomialFeatures(degree=2, include_bias=False)),
                ("scale", StandardScaler()),
                (
                    "logreg",
                    LogisticRegression(max_iter=1500, class_weight="balanced", random_state=seed),
                ),
            ]
        )
        x = train[["proj_x", "proj_y"]].to_numpy()
        model.fit(x, y)
        train_scores = model.predict_proba(x)[:, 1]
        models[detector] = model
        fit_rows.append(
            {
                "detector": detector,
                "detector_label": detector_label,
                "n": int(len(train)),
                "positive_rate": float(y.mean()),
                "roc_auc": float(roc_auc_score(y, train_scores)),
                "average_precision": float(average_precision_score(y, train_scores)),
                "brier": float(brier_score_loss(y, train_scores)),
                "score_definition": "Pr(detector predicts phishing | proj_x, proj_y), degree-2 balanced logistic surrogate",
            }
        )
    return models, pd.DataFrame(fit_rows)


def add_pair_columns(frame: pd.DataFrame) -> pd.DataFrame:
    out = frame.copy()
    for spec in pair_specs():
        left = out[str(spec["left_col"])].fillna(0.0).astype(float)
        if bool(spec["diagonal"]):
            out[f"pair_{spec['pair']}"] = left
        else:
            right = out[str(spec["right_col"])].fillna(0.0).astype(float)
            out[f"pair_{spec['pair']}"] = left * right
    return out


def build_paired_rows(frame: pd.DataFrame, models: dict[str, Pipeline]) -> pd.DataFrame:
    scored = add_pair_columns(frame)
    for detector, model in models.items():
        scored[f"{detector}__surrogate_score"] = model.predict_proba(
            scored[["proj_x", "proj_y"]].to_numpy()
        )[:, 1]

    rows = []
    pair_columns = [f"pair_{spec['pair']}" for spec in pair_specs()]
    for method in METHODS:
        hw = scored[scored["stage"].eq(method) & scored["source"].eq("HW")].reset_index(drop=True)
        llm = scored[scored["stage"].eq(method) & scored["source"].eq("LLM")].reset_index(drop=True)
        n = min(len(hw), len(llm))
        hw = hw.iloc[:n].copy()
        llm = llm.iloc[:n].copy()
        for detector, detector_label in DETECTORS.items():
            if f"{detector}__surrogate_score" not in scored:
                continue
            original_pred = hw[detector].astype(float).reset_index(drop=True)
            rewritten_pred = llm[detector].astype(float).reset_index(drop=True)
            original_score = hw[f"{detector}__surrogate_score"].astype(float).reset_index(drop=True)
            rewritten_score = llm[f"{detector}__surrogate_score"].astype(float).reset_index(drop=True)
            for index in range(n):
                original_identity = f"{method}:HW:{index + 1}"
                rewritten_identity = f"{method}:LLM:{index + 1}"
                row = {
                    "sample_id": index + 1,
                    "original_id": original_identity,
                    "rewrite_id": rewritten_identity,
                    "method": METHOD_LABELS[method],
                    "stage": method,
                    "detector": detector,
                    "detector_label": detector_label,
                    "original_subject": hw["subject"].fillna("").astype(str).iloc[index],
                    "original_body": hw["body"].fillna("").astype(str).iloc[index],
                    "original_text": hw["text"].fillna("").astype(str).iloc[index],
                    "original_source_file": hw["source_file"].fillna("").astype(str).iloc[index],
                    "rewritten_subject": llm["subject"].fillna("").astype(str).iloc[index],
                    "rewritten_body": llm["body"].fillna("").astype(str).iloc[index],
                    "rewritten_text": llm["text"].fillna("").astype(str).iloc[index],
                    "rewritten_source_file": llm["source_file"].fillna("").astype(str).iloc[index],
                    "original_prediction": original_pred.iloc[index],
                    "rewritten_prediction": rewritten_pred.iloc[index],
                    "original_score": original_score.iloc[index],
                    "rewritten_score": rewritten_score.iloc[index],
                    "delta_detector_score": rewritten_score.iloc[index] - original_score.iloc[index],
                    "rewritten_is_fn": bool(rewritten_pred.iloc[index] == 0.0),
                    "rewritten_is_tp": bool(rewritten_pred.iloc[index] == 1.0),
                }
                for spec in pair_specs():
                    pair = str(spec["pair"])
                    pair_column = f"pair_{pair}"
                    original_value = float(hw[pair_column].iloc[index])
                    rewritten_value = float(llm[pair_column].iloc[index])
                    row[f"original_{pair}"] = original_value
                    row[f"rewritten_{pair}"] = rewritten_value
                    row[f"delta_{pair}"] = rewritten_value - original_value
                rows.append(row)
    sample = pd.DataFrame(rows)
    sample.to_csv(SAMPLE_LEVEL_CSV, index=False)
    return sample


def bh_by_group(frame: pd.DataFrame, group_cols: list[str]) -> pd.DataFrame:
    out = frame.copy()
    out["q_value"] = np.nan
    for _, idx in out.groupby(group_cols).groups.items():
        out.loc[idx, "q_value"] = multipletests(out.loc[idx, "p_value"].fillna(1.0), method="fdr_bh")[1]
    return out


def rank_biserial_from_u(u_stat: float, n_a: int, n_b: int) -> float:
    if n_a == 0 or n_b == 0:
        return float("nan")
    return 2.0 * u_stat / (n_a * n_b) - 1.0


def build_performance(sample: pd.DataFrame) -> pd.DataFrame:
    mcc = pd.read_csv(MCC_TABLE)
    mcc_long = []
    for detector in DETECTORS:
        mcc_detector = MCC_DETECTOR_NAMES.get(detector, detector)
        row = mcc[mcc["detector"].eq(mcc_detector)]
        if row.empty:
            continue
        for stage in METHODS:
            mcc_long.append(
                {
                    "detector": detector,
                    "stage": stage,
                    "method": METHOD_LABELS[stage],
                    "MCC": float(row[f"llm_{stage}"].iloc[0]),
                }
            )
    mcc_long = pd.DataFrame(mcc_long)
    rows = []
    for (method, stage, detector, detector_label), group in sample.groupby(
        ["method", "stage", "detector", "detector_label"], sort=False
    ):
        pred = group["rewritten_prediction"].dropna().astype(float)
        rows.append(
            {
                "method": method,
                "stage": stage,
                "detector": detector,
                "detector_label": detector_label,
                "n": int(len(pred)),
                "FN_rate": float((pred == 0.0).mean()) if len(pred) else float("nan"),
                "TP_rate": float((pred == 1.0).mean()) if len(pred) else float("nan"),
                "original_mean_score": float(group["original_score"].mean()),
                "rewritten_mean_score": float(group["rewritten_score"].mean()),
                "mean_detector_score_change": float(group["delta_detector_score"].mean()),
                "median_detector_score_change": float(group["delta_detector_score"].median()),
            }
        )
    perf = pd.DataFrame(rows).merge(mcc_long, on=["method", "stage", "detector"], how="left")
    perf = perf[
        [
            "method",
            "detector_label",
            "MCC",
            "FN_rate",
            "TP_rate",
            "mean_detector_score_change",
            "median_detector_score_change",
            "original_mean_score",
            "rewritten_mean_score",
            "n",
            "stage",
            "detector",
        ]
    ].sort_values(["method", "detector_label"])
    perf.to_csv(PERFORMANCE_CSV, index=False)
    compact = perf[
        perf["detector"].isin(["securenet_llama", "email_phishing_detection_v3_prediction"])
    ][["method", "detector_label", "MCC", "FN_rate", "mean_detector_score_change"]].copy()
    compact = compact.rename(
        columns={
            "method": "Method",
            "detector_label": "Detector",
            "FN_rate": "FN Rate",
            "mean_detector_score_change": "Mean detector-score change",
        }
    )
    compact.to_csv(PERFORMANCE_COMPACT_CSV, index=False)
    lines = [
        "| Method | Detector | MCC | FN Rate | Mean detector-score change |",
        "|---|---|---:|---:|---:|",
    ]
    for _, row in compact.iterrows():
        lines.append(
            f"| {row['Method']} | {row['Detector']} | {row['MCC']:.4f} | {row['FN Rate']:.3f} | {row['Mean detector-score change']:+.4f} |"
        )
    PERFORMANCE_COMPACT_MD.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return perf


def build_feature_shift(sample: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    specs = pair_specs()
    base = sample.drop_duplicates(["method", "sample_id"]).copy()
    rows = []
    for method, method_frame in base.groupby("method", sort=False):
        for spec in specs:
            pair = str(spec["pair"])
            original = method_frame[f"original_{pair}"].astype(float)
            rewritten = method_frame[f"rewritten_{pair}"].astype(float)
            delta = method_frame[f"delta_{pair}"].astype(float)
            rows.append(
                {
                    "method": method,
                    "pair": pair,
                    "pair_full": spec["pair_full"],
                    "original_mean": float(original.mean()),
                    "rewritten_mean": float(rewritten.mean()),
                    "delta_rewritten_minus_original": float(delta.mean()),
                    "abs_delta_rewritten_minus_original": float(abs(delta.mean())),
                    "mean_abs_sample_delta": float(delta.abs().mean()),
                    "n": int(len(delta)),
                }
            )
    feature = pd.DataFrame(rows)
    overall = (
        feature.groupby("method", as_index=False)
        .agg(
            D_m_mean_abs_pair_shift=("abs_delta_rewritten_minus_original", "mean"),
            mean_abs_sample_pair_shift=("mean_abs_sample_delta", "mean"),
            max_abs_pair_shift=("abs_delta_rewritten_minus_original", "max"),
            n_pairs=("pair", "count"),
        )
        .sort_values("D_m_mean_abs_pair_shift", ascending=False)
    )
    top = (
        feature.sort_values(["method", "abs_delta_rewritten_minus_original"], ascending=[True, False])
        .groupby("method", as_index=False)
        .head(8)
        .reset_index(drop=True)
    )
    feature.to_csv(FEATURE_SHIFT_CSV, index=False)
    overall.to_csv(OVERALL_SHIFT_CSV, index=False)
    top.to_csv(TOP_FEATURE_SHIFT_CSV, index=False)
    return feature, overall, top


def build_failure_association(sample: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    rows = []
    for (method, detector, detector_label), group in sample.groupby(
        ["method", "detector", "detector_label"], sort=False
    ):
        for spec in pair_specs():
            pair = str(spec["pair"])
            fn = group.loc[group["rewritten_is_fn"], f"delta_{pair}"].astype(float)
            tp = group.loc[group["rewritten_is_tp"], f"delta_{pair}"].astype(float)
            if len(fn) == 0 or len(tp) == 0:
                u_stat = float("nan")
                p_value = 1.0
                effect = float("nan")
            else:
                result = mannwhitneyu(fn, tp, alternative="two-sided")
                u_stat = float(result.statistic)
                p_value = float(result.pvalue)
                effect = rank_biserial_from_u(u_stat, len(fn), len(tp))
            rows.append(
                {
                    "method": method,
                    "detector": detector,
                    "detector_label": detector_label,
                    "pair": pair,
                    "pair_full": spec["pair_full"],
                    "FN_mean_delta": float(fn.mean()) if len(fn) else float("nan"),
                    "FN_median_delta": float(fn.median()) if len(fn) else float("nan"),
                    "TP_mean_delta": float(tp.mean()) if len(tp) else float("nan"),
                    "TP_median_delta": float(tp.median()) if len(tp) else float("nan"),
                    "FN_minus_TP_mean_delta": float(fn.mean() - tp.mean())
                    if len(fn) and len(tp)
                    else float("nan"),
                    "u_statistic": u_stat,
                    "p_value": p_value,
                    "rank_biserial_effect": effect,
                    "abs_rank_biserial_effect": abs(effect) if not np.isnan(effect) else float("nan"),
                    "n_FN": int(len(fn)),
                    "n_TP": int(len(tp)),
                }
            )
    assoc = bh_by_group(pd.DataFrame(rows), ["method", "detector"])
    top = (
        assoc[assoc["q_value"].lt(0.05)]
        .sort_values(["method", "detector_label", "abs_rank_biserial_effect"], ascending=[True, True, False])
        .groupby(["method", "detector"], as_index=False)
        .head(5)
        .reset_index(drop=True)
    )
    assoc.to_csv(FAILURE_ASSOC_CSV, index=False)
    top.to_csv(TOP_FAILURE_ASSOC_CSV, index=False)
    return assoc, top


def build_score_correlations(sample: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for (method, detector, detector_label), group in sample.groupby(
        ["method", "detector", "detector_label"], sort=False
    ):
        y = group["delta_detector_score"].astype(float)
        for spec in pair_specs():
            pair = str(spec["pair"])
            x = group[f"delta_{pair}"].astype(float)
            if x.nunique(dropna=True) < 2 or y.nunique(dropna=True) < 2:
                rho = float("nan")
                p_value = 1.0
            else:
                result = spearmanr(x, y, nan_policy="omit")
                rho = float(result.statistic)
                p_value = float(result.pvalue)
            rows.append(
                {
                    "method": method,
                    "detector": detector,
                    "detector_label": detector_label,
                    "pair": pair,
                    "pair_full": spec["pair_full"],
                    "spearman_rho_delta_pair_vs_delta_detector_score": rho,
                    "p_value": p_value,
                    "n": int(len(group)),
                }
            )
    corr = bh_by_group(pd.DataFrame(rows), ["method", "detector"])
    corr.to_csv(CORRELATION_CSV, index=False)
    return corr


def build_scarcity_direction_table(assoc: pd.DataFrame) -> pd.DataFrame:
    selected = assoc[
        assoc["method"].eq("Fuzzer")
        & assoc["detector_label"].isin(["SecureNet", "V3"])
        & assoc["pair"].isin(["S-S", "A-S", "SP-S"])
    ].copy()
    selected["FN_minus_TP_median_delta"] = selected["FN_median_delta"] - selected["TP_median_delta"]
    selected["direction_summary"] = np.select(
        [
            selected["FN_median_delta"].gt(0) & selected["TP_median_delta"].gt(0),
            selected["FN_median_delta"].gt(0) & selected["TP_median_delta"].le(0),
            selected["FN_median_delta"].le(0) & selected["TP_median_delta"].gt(0),
            selected["FN_median_delta"].le(0) & selected["TP_median_delta"].le(0),
        ],
        [
            "FN and TP both increase; FN increases more if FN-TP is positive",
            "FN increases while TP is flat/decreases",
            "FN is flat/decreases while TP increases",
            "FN and TP both flat/decrease; FN decreases less if FN-TP is positive",
        ],
        default="mixed/undefined",
    )
    selected = selected[
        [
            "detector_label",
            "pair",
            "FN_median_delta",
            "TP_median_delta",
            "FN_minus_TP_median_delta",
            "FN_mean_delta",
            "TP_mean_delta",
            "FN_minus_TP_mean_delta",
            "q_value",
            "rank_biserial_effect",
            "n_FN",
            "n_TP",
            "direction_summary",
        ]
    ].sort_values(["detector_label", "pair"])
    selected.to_csv(SCARCITY_DIRECTION_CSV, index=False)
    return selected


def compact_text(value: object, limit: int = 900) -> str:
    text = " ".join(str(value).split())
    if len(text) <= limit:
        return text
    return text[: limit - 3].rstrip() + "..."


def build_scarcity_examples(sample: pd.DataFrame, top_n: int = 10) -> pd.DataFrame:
    rows = []
    base = sample[
        sample["method"].eq("Fuzzer")
        & sample["detector_label"].isin(["SecureNet", "V3"])
        & sample["rewritten_is_fn"].astype(bool)
    ].copy()
    for detector_label in ["SecureNet", "V3"]:
        detector_frame = base[base["detector_label"].eq(detector_label)].copy()
        for pair in ["S-S", "A-S", "SP-S"]:
            ranked = detector_frame.assign(abs_pair_delta=detector_frame[f"delta_{pair}"].abs()).sort_values(
                ["abs_pair_delta", f"delta_{pair}"], ascending=[False, False]
            )
            for rank, (_, row) in enumerate(ranked.head(top_n).iterrows(), start=1):
                rows.append(
                    {
                        "detector_label": detector_label,
                        "pair": pair,
                        "rank": rank,
                        "sample_id": row["sample_id"],
                        "original_id": row["original_id"],
                        "rewrite_id": row["rewrite_id"],
                        "original_pair_score": row[f"original_{pair}"],
                        "rewritten_pair_score": row[f"rewritten_{pair}"],
                        "delta_pair": row[f"delta_{pair}"],
                        "delta_detector_score": row["delta_detector_score"],
                        "original_subject": compact_text(row["original_subject"], 240),
                        "rewritten_subject": compact_text(row["rewritten_subject"], 240),
                        "original_text_excerpt": compact_text(row["original_text"] or row["original_body"]),
                        "rewritten_text_excerpt": compact_text(row["rewritten_text"] or row["rewritten_body"]),
                        "original_source_file": row["original_source_file"],
                        "rewritten_source_file": row["rewritten_source_file"],
                    }
                )
    examples = pd.DataFrame(rows)
    examples.to_csv(SCARCITY_EXAMPLES_CSV, index=False)
    return examples


def build_independence_check(sample: pd.DataFrame) -> pd.DataFrame:
    base = sample.drop_duplicates(["method", "sample_id"]).copy()
    base["original_text_key"] = base["original_text"].fillna("").astype(str).str.strip()
    rows = []
    method_count_by_original = base.groupby("original_text_key")["method"].nunique()
    for method, group in base.groupby("method", sort=False):
        original_counts = group.groupby("original_text_key").size()
        rows.append(
            {
                "method": method,
                "paired_rewrite_rows": int(len(group)),
                "unique_original_texts_within_method": int(original_counts.size),
                "max_rewrites_per_original_within_method": int(original_counts.max()) if len(original_counts) else 0,
                "original_texts_repeated_within_method": int((original_counts > 1).sum()),
                "original_texts_shared_with_other_methods": int(
                    group["original_text_key"].map(method_count_by_original).gt(1).sum()
                ),
                "independence_note": (
                    "Existing sample-level tests treat row-paired rewrites as observations. "
                    "If the same original text appears across methods, cross-method method comparisons should be phrased descriptively "
                    "or rerun with original-level aggregation/cluster bootstrap."
                ),
            }
        )
    check = pd.DataFrame(rows)
    check.to_csv(INDEPENDENCE_CSV, index=False)
    return check


def write_methodology_notes(direction: pd.DataFrame, independence: pd.DataFrame) -> None:
    lines = [
        "# S6 Rewriting Methodology Notes",
        "",
        "## D_m and Fig. 5 definition",
        "",
        "In `generate_s6_rewriting_method_impact.py`, each persuasion-pair score is computed from the six principle scores. Diagonal pairs use the principle score itself; off-diagonal pairs use the product of the two principle scores.",
        "",
        "`delta_pair = rewritten_pair_strength - original_pair_strength` at the row-paired sample level.",
        "",
        "`D_m` in `s6_method_overall_persuasion_shift.csv` is `mean_p | mean_i(delta_{i,p}) |`: the mean, over 21 persuasion pairs, of the absolute rewritten-minus-original mean shift for each pair. The separate `mean_abs_sample_pair_shift` column is `mean_p mean_i |delta_{i,p}|`.",
        "",
        "`fig_s6_sample_persuasion_shift_by_outcome` uses `per_sample_persuasion_shift = mean_p |delta_{i,p}|` for every method-detector sample. Its stars compare FN vs TP within each method and detector with two-sided Mann-Whitney U tests, followed by Benjamini-Hochberg FDR correction over the six method-detector comparisons in that figure.",
        "",
        "The FN/TP-associated pair tables compare `delta_pair` distributions between rewritten FN and rewritten TP samples within each method-detector pair. FDR correction is applied separately within each method-detector family over the 21 pair tests.",
        "",
        "## Scarcity-direction result",
        "",
        "| Detector | Pair | FN median delta | TP median delta | FN-TP median | q | Effect |",
        "|---|---|---:|---:|---:|---:|---:|",
    ]
    for row in direction.itertuples():
        lines.append(
            f"| {row.detector_label} | {row.pair} | {row.FN_median_delta:+.4f} | {row.TP_median_delta:+.4f} | {row.FN_minus_TP_median_delta:+.4f} | {row.q_value:.3g} | {row.rank_biserial_effect:+.3f} |"
        )
    lines.extend(
        [
            "",
            "## Independence check",
            "",
            "| Method | Paired rows | Unique originals | Max rewrites/original within method | Repeated originals within method | Originals shared with other methods |",
            "|---|---:|---:|---:|---:|---:|",
        ]
    )
    for row in independence.itertuples():
        lines.append(
            f"| {row.method} | {row.paired_rewrite_rows} | {row.unique_original_texts_within_method} | {row.max_rewrites_per_original_within_method} | {row.original_texts_repeated_within_method} | {row.original_texts_shared_with_other_methods} |"
        )
    lines.extend(
        [
            "",
            "Interpretation guardrail: these analyses support co-occurrence between rewriting-induced persuasion-pair changes and detector FN outcomes. They should not be worded as causal evidence unless a controlled same-original generation design or original-level robustness analysis is added.",
        ]
    )
    METHODOLOGY_MD.write_text("\n".join(lines) + "\n", encoding="utf-8")


def write_summary(
    performance: pd.DataFrame,
    overall: pd.DataFrame,
    top_feature: pd.DataFrame,
    top_failure: pd.DataFrame,
) -> None:
    lines = [
        "# S6 Rewriting Method Impact",
        "",
        "## Detector-performance difference",
        "",
        "| Method | Detector | MCC | FN rate | Mean detector-score change |",
        "|---|---|---:|---:|---:|",
    ]
    selected = performance[performance["detector"].isin(["securenet_llama", "email_phishing_detection_v3_prediction"])].copy()
    for row in selected.sort_values(["detector_label", "method"]).itertuples():
        lines.append(
            f"| {row.method} | {row.detector_label} | {row.MCC:.4f} | {row.FN_rate:.3f} | {row.mean_detector_score_change:+.4f} |"
        )
    lines.extend(["", "## Overall persuasion shift", "", "| Method | D_m | Mean abs sample-pair shift | Max pair shift |", "|---|---:|---:|---:|"])
    for row in overall.itertuples():
        lines.append(
            f"| {row.method} | {row.D_m_mean_abs_pair_shift:.4f} | {row.mean_abs_sample_pair_shift:.4f} | {row.max_abs_pair_shift:.4f} |"
        )
    lines.extend(["", "## Top changed persuasion pairs", "", "| Method | Pair | Original mean | Rewritten mean | Delta |", "|---|---|---:|---:|---:|"])
    for row in top_feature.groupby("method", sort=False).head(5).itertuples():
        lines.append(
            f"| {row.method} | {row.pair} | {row.original_mean:.4f} | {row.rewritten_mean:.4f} | {row.delta_rewritten_minus_original:+.4f} |"
        )
    lines.extend(["", "## Top failure-associated changes", "", "| Method | Detector | Pair | FN mean delta | TP mean delta | FN-TP | q | Effect |", "|---|---|---|---:|---:|---:|---:|---:|"])
    for row in top_failure.groupby(["method", "detector_label"], sort=False).head(3).itertuples():
        lines.append(
            f"| {row.method} | {row.detector_label} | {row.pair} | {row.FN_mean_delta:+.4f} | {row.TP_mean_delta:+.4f} | {row.FN_minus_TP_mean_delta:+.4f} | {row.q_value:.3g} | {row.rank_biserial_effect:+.3f} |"
        )
    lines.extend(
        [
            "",
            "## Follow-up outputs",
            "",
            f"- Final rewritten FN-vs-TP pair-characteristic table: `{REWRITTEN_FN_TP_CSV.name}`",
            f"- Final rewritten FN-vs-TP pair-characteristic heatmap: `{REWRITTEN_FN_TP_FIGURE.name}`",
            f"- Detector-consistency table for final FN-vs-TP analysis: `{REWRITTEN_FN_TP_CONSISTENCY_CSV.name}`",
            f"- UTA-MPG controlled mapping audit: `{CONTROLLED_MAPPING_AUDIT_MD.name}`",
            f"- Corrected UTA-MPG row-id join summary: `{CONTROLLED_JOIN_SUMMARY_MD.name}`",
            f"- Controlled UTA-MPG paired persuasion comparison: `{CONTROLLED_PERSUASION_COMPARISON_CSV.name}`",
            f"- Controlled UTA-MPG paired rewritten detector outcome cells: `{CONTROLLED_DETECTOR_CELLS_CSV.name}`",
            f"- Missing original-transition audit: `{CONTROLLED_ORIGINAL_TRANSITION_AUDIT_CSV.name}`",
            f"- Scarcity direction table: `{SCARCITY_DIRECTION_CSV.name}`",
            f"- Top Fuzzer FN original/rewrite examples: `{SCARCITY_EXAMPLES_CSV.name}`",
            f"- Repeated-original independence check: `{INDEPENDENCE_CSV.name}`",
            f"- D_m/test definitions and interpretation guardrails: `{METHODOLOGY_MD.name}`",
            "",
            "Note: the old row-order `delta_pair` failure-association outputs are exploratory/deprecated for final controlled claims. Use the rewritten FN-vs-TP outputs above for the pairing-independent S6 failure-characteristic analysis.",
            "",
            "Fuzzer values can remain as external-setting observations, not as causal ranking evidence against UTA/MPG.",
        ]
    )
    SUMMARY_MD.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    ROOT.mkdir(parents=True, exist_ok=True)
    frame = load_frame()
    models, fit_metrics = fit_detector_surfaces(frame)
    sample = build_paired_rows(frame, models)
    performance = build_performance(sample)
    _, overall, top_feature = build_feature_shift(sample)
    assoc, top_failure = build_failure_association(sample)
    build_score_correlations(sample)
    direction = build_scarcity_direction_table(assoc)
    build_scarcity_examples(sample)
    independence = build_independence_check(sample)
    write_methodology_notes(direction, independence)
    fit_metrics.to_csv(FIT_METRICS_CSV, index=False)
    write_summary(performance, overall, top_feature, top_failure)
    SUMMARY_JSON.write_text(
        json.dumps(
            {
                "input": str(INPUT),
                "pairing": "Rows are paired by source-specific row order within each S6 stage; S6-fuzzer has 2200 pairs, S6-UTA and S6-MPG have 1600 pairs.",
                "detector_score_definition": "Pr(detector predicts phishing | proj_x, proj_y), degree-2 balanced logistic surrogate fitted separately for each detector on S6 phishing rows.",
                "feature_delta_definition": "delta_pair = rewritten_pair_strength - original_pair_strength.",
                "outputs": {
                    "sample_level_csv": SAMPLE_LEVEL_CSV.name,
                    "performance_csv": PERFORMANCE_CSV.name,
                    "performance_compact_csv": PERFORMANCE_COMPACT_CSV.name,
                    "performance_compact_md": PERFORMANCE_COMPACT_MD.name,
                    "overall_shift_csv": OVERALL_SHIFT_CSV.name,
                    "feature_shift_csv": FEATURE_SHIFT_CSV.name,
                    "top_feature_shift_csv": TOP_FEATURE_SHIFT_CSV.name,
                    "failure_association_csv": FAILURE_ASSOC_CSV.name,
                    "top_failure_association_csv": TOP_FAILURE_ASSOC_CSV.name,
                    "correlation_csv": CORRELATION_CSV.name,
                    "fit_metrics_csv": FIT_METRICS_CSV.name,
                    "scarcity_direction_csv": SCARCITY_DIRECTION_CSV.name,
                    "scarcity_examples_csv": SCARCITY_EXAMPLES_CSV.name,
                    "independence_csv": INDEPENDENCE_CSV.name,
                    "methodology_md": METHODOLOGY_MD.name,
                    "rewritten_fn_tp_csv": REWRITTEN_FN_TP_CSV.name,
                    "rewritten_fn_tp_figure": REWRITTEN_FN_TP_FIGURE.name,
                    "rewritten_fn_tp_consistency_csv": REWRITTEN_FN_TP_CONSISTENCY_CSV.name,
                    "controlled_mapping_audit_md": CONTROLLED_MAPPING_AUDIT_MD.name,
                    "controlled_join_summary_md": CONTROLLED_JOIN_SUMMARY_MD.name,
                    "controlled_persuasion_comparison_csv": CONTROLLED_PERSUASION_COMPARISON_CSV.name,
                    "controlled_detector_cells_csv": CONTROLLED_DETECTOR_CELLS_CSV.name,
                    "controlled_original_transition_audit_csv": CONTROLLED_ORIGINAL_TRANSITION_AUDIT_CSV.name,
                    "summary_md": SUMMARY_MD.name,
                },
            },
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )


if __name__ == "__main__":
    main()
