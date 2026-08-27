#!/usr/bin/env python3
import csv
import copy
import math
import re
from collections import defaultdict
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parent
PAIRED = ROOT / "s8_paired_common_dataset_seed20260825.csv"

GENERATOR_ORDER = ["Claude", "GPT", "Gemini", "Llama", "Ministral", "DeepSeek"]
FEATURES = [
    "Urgency wording",
    "Login/account action",
    "Information submission",
    "Click/open request",
    "Softened request",
    "Explicit action request",
    "Direct URL/page instruction",
    "Conversational wording",
]
DETECTORS = [
    ("ScamLLM", "scamllm"),
    ("PiMRef", "pimref"),
    ("T5", "t5phishing"),
    ("XGBoost", "xgboost"),
    ("SecureNet", "securenet_llama"),
    ("PhishingV3", "email_phishing_detection_v3_prediction"),
]

TP_FN_CSV = ROOT / "S8_rq3_tp_fn_features_all_detectors.csv"
GENERATOR_EFFECTS_CSV = ROOT / "S8_rq3_gee_generator_effects.csv"
FEATURE_EFFECTS_CSV = ROOT / "S8_rq3_gee_feature_effects.csv"
ADJUSTMENT_SUMMARY_CSV = ROOT / "S8_rq3_generator_adjustment_summary.csv"
MARGINAL_CSV = ROOT / "S8_rq3_observed_adjusted_marginal_detection.csv"
SUMMARY_MD = ROOT / "S8_rq3_paired_analysis_summary.md"
FIG_C = ROOT / "Fig_S8_C_detector_specific_tp_fn_feature_differences.png"
FIG_C_APP = ROOT / "Fig_S8_C_appendix_all_detectors_tp_fn_feature_differences.png"
FIG_D = ROOT / "Fig_S8_D_observed_vs_feature_adjusted_detection.png"


def read_rows():
    with PAIRED.open(newline="", encoding="utf-8", errors="replace") as handle:
        return list(csv.DictReader(handle))


def parse_int(value):
    if value in ("", None):
        return None
    try:
        return int(float(value))
    except ValueError:
        return None


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


def bh_fdr(rows, p_key="p_value", q_key="q_value"):
    valid = [(index, row) for index, row in enumerate(rows) if row.get(p_key) not in ("", None, "NA")]
    indexed = sorted(valid, key=lambda item: float(item[1][p_key]))
    adjusted = {}
    running_min = 1.0
    total = len(indexed)
    for rank, (index, row) in reversed(list(enumerate(indexed, start=1))):
        running_min = min(running_min, float(row[p_key]) * total / rank)
        adjusted[index] = min(1.0, running_min)
    for index, row in enumerate(rows):
        row[q_key] = adjusted.get(index, "NA")


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


def add_stars_to_test_label(label):
    match = re.search(r"p=([0-9.eE+-]+)", label)
    if not match:
        return label
    return f"{label}{significance_stars(match.group(1))}"


def gammq(a_value, x_value):
    if x_value < 0 or a_value <= 0:
        return float("nan")
    if x_value == 0:
        return 1.0
    if x_value < a_value + 1.0:
        ap = a_value
        delta = 1.0 / a_value
        total = delta
        for _ in range(1000):
            ap += 1.0
            delta *= x_value / ap
            total += delta
            if abs(delta) < abs(total) * 3e-14:
                break
        return max(0.0, min(1.0, 1.0 - total * math.exp(-x_value + a_value * math.log(x_value) - math.lgamma(a_value))))
    b_value = x_value + 1.0 - a_value
    c_value = 1.0 / 1e-300
    d_value = 1.0 / b_value
    h_value = d_value
    for i_value in range(1, 1000):
        an_value = -i_value * (i_value - a_value)
        b_value += 2.0
        d_value = an_value * d_value + b_value
        if abs(d_value) < 1e-300:
            d_value = 1e-300
        c_value = b_value + an_value / c_value
        if abs(c_value) < 1e-300:
            c_value = 1e-300
        d_value = 1.0 / d_value
        delta = d_value * c_value
        h_value *= delta
        if abs(delta - 1.0) < 3e-14:
            break
    return max(0.0, min(1.0, math.exp(-x_value + a_value * math.log(x_value) - math.lgamma(a_value)) * h_value))


def chi_square_sf(x_value, df_value):
    return gammq(df_value / 2.0, x_value / 2.0)


def normal_pvalue(z_value):
    return math.erfc(abs(z_value) / math.sqrt(2.0))


def log_comb(n_value, k_value):
    if k_value < 0 or k_value > n_value:
        return float("-inf")
    return math.lgamma(n_value + 1) - math.lgamma(k_value + 1) - math.lgamma(n_value - k_value + 1)


def hypergeom_probability(x_value, row_1, row_2, col_1):
    return math.exp(log_comb(row_1, x_value) + log_comb(row_2, col_1 - x_value) - log_comb(row_1 + row_2, col_1))


def fisher_exact_two_sided(a_value, b_value, c_value, d_value):
    row_1 = a_value + b_value
    row_2 = c_value + d_value
    col_1 = a_value + c_value
    if row_1 == 0 or row_2 == 0:
        return 1.0
    lower = max(0, col_1 - row_2)
    upper = min(row_1, col_1)
    observed = hypergeom_probability(a_value, row_1, row_2, col_1)
    p_value = 0.0
    for x_value in range(lower, upper + 1):
        probability = hypergeom_probability(x_value, row_1, row_2, col_1)
        if probability <= observed + 1e-12:
            p_value += probability
    return min(1.0, p_value)


def fisher_or(a_value, b_value, c_value, d_value):
    if min(a_value, b_value, c_value, d_value) == 0:
        a_value += 0.5
        b_value += 0.5
        c_value += 0.5
        d_value += 0.5
    return (a_value * d_value) / (b_value * c_value)


def rq3_tp_fn_features(rows):
    output = []
    for detector_label, detector_col in DETECTORS:
        detector_rows = []
        for generator in GENERATOR_ORDER:
            subset = [row for row in rows if row["generator"] == generator and parse_int(row.get(detector_col)) is not None]
            tp_rows = [row for row in subset if parse_int(row[detector_col]) == 1]
            fn_rows = [row for row in subset if parse_int(row[detector_col]) == 0]
            sparse = detector_label == "PhishingV3" and generator == "DeepSeek" and len(tp_rows) < 10
            for feature in FEATURES:
                tp_count = sum(parse_int(row[feature]) == 1 for row in tp_rows)
                fn_count = sum(parse_int(row[feature]) == 1 for row in fn_rows)
                tp_n = len(tp_rows)
                fn_n = len(fn_rows)
                tp_rate = tp_count / tp_n if tp_n else 0.0
                fn_rate = fn_count / fn_n if fn_n else 0.0
                p_value = "NA" if sparse else fisher_exact_two_sided(tp_count, tp_n - tp_count, fn_count, fn_n - fn_count)
                detector_rows.append({
                    "Detector": detector_label,
                    "Generator": generator,
                    "Feature": feature,
                    "N_TP": tp_n,
                    "N_FN": fn_n,
                    "TP_%": tp_rate * 100.0,
                    "FN_%": fn_rate * 100.0,
                    "Delta_pp": (tp_rate - fn_rate) * 100.0,
                    "Fisher_OR": fisher_or(tp_count, tp_n - tp_count, fn_count, fn_n - fn_count),
                    "p_value": p_value,
                    "sparse_note": "sparse TP; descriptive only" if sparse else "",
                })
        bh_fdr(detector_rows)
        output.extend(detector_rows)
    return output


def design_matrix(data_rows, include_features, generators):
    columns = ["Intercept"] + [f"Generator: {generator} vs Claude" for generator in generators if generator != "Claude"]
    if include_features:
        columns += FEATURES
    x_rows = []
    y_values = []
    clusters = []
    row_generators = []
    for row in data_rows:
        generator = row["generator"]
        vector = [1.0]
        vector += [1.0 if generator == candidate else 0.0 for candidate in generators if candidate != "Claude"]
        if include_features:
            vector += [float(parse_int(row[feature]) or 0) for feature in FEATURES]
        x_rows.append(vector)
        y_values.append(float(row["_y"]))
        clusters.append(row["prompt_id"])
        row_generators.append(generator)
    return np.asarray(x_rows, dtype=float), np.asarray(y_values, dtype=float), clusters, row_generators, columns


def logistic_fit_cluster(x_matrix, y_vector, clusters, max_iter=200):
    beta = np.zeros(x_matrix.shape[1], dtype=float)
    for _ in range(max_iter):
        eta = np.clip(x_matrix.dot(beta), -30.0, 30.0)
        mu = 1.0 / (1.0 + np.exp(-eta))
        weight = np.maximum(mu * (1.0 - mu), 1e-8)
        hessian = x_matrix.T.dot(weight[:, None] * x_matrix)
        score = x_matrix.T.dot(y_vector - mu)
        try:
            step = np.linalg.solve(hessian + np.eye(hessian.shape[0]) * 1e-8, score)
        except np.linalg.LinAlgError:
            step = np.linalg.pinv(hessian).dot(score)
        step = np.clip(step, -5.0, 5.0)
        beta += step
        if float(np.max(np.abs(step))) < 1e-7:
            break
    eta = np.clip(x_matrix.dot(beta), -30.0, 30.0)
    mu = 1.0 / (1.0 + np.exp(-eta))
    weight = np.maximum(mu * (1.0 - mu), 1e-8)
    bread = np.linalg.pinv(x_matrix.T.dot(weight[:, None] * x_matrix))
    meat = np.zeros((x_matrix.shape[1], x_matrix.shape[1]))
    by_cluster = defaultdict(list)
    for index, cluster in enumerate(clusters):
        by_cluster[cluster].append(index)
    residual = y_vector - mu
    for indices in by_cluster.values():
        x_cluster = x_matrix[indices, :]
        score_cluster = x_cluster.T.dot(residual[indices])
        meat += np.outer(score_cluster, score_cluster)
    covariance = bread.dot(meat).dot(bread)
    se = np.sqrt(np.maximum(np.diag(covariance), 0.0))
    return beta, se, covariance


def wald_joint(beta, covariance, indices):
    sub_beta = beta[indices]
    sub_cov = covariance[np.ix_(indices, indices)]
    statistic = float(sub_beta.T.dot(np.linalg.pinv(sub_cov)).dot(sub_beta))
    p_value = chi_square_sf(statistic, len(indices))
    return statistic, p_value


def model_rows_for_detector(rows, detector_label, detector_col):
    output = []
    for row in rows:
        y_value = parse_int(row.get(detector_col))
        if y_value is None:
            continue
        if detector_label == "PhishingV3" and row["generator"] == "DeepSeek":
            continue
        item = dict(row)
        item["_y"] = y_value
        output.append(item)
    return output


def gee_outputs(rows):
    gen_coef_rows = []
    feature_rows = []
    adjustment_rows = []
    marginal_rows = []

    for detector_label, detector_col in DETECTORS:
        all_available = [dict(row, _y=parse_int(row[detector_col])) for row in rows if parse_int(row.get(detector_col)) is not None]
        model_data = model_rows_for_detector(rows, detector_label, detector_col)
        generators = [generator for generator in GENERATOR_ORDER if any(row["generator"] == generator for row in model_data)]

        x1, y1, clusters1, _, cols1 = design_matrix(model_data, False, generators)
        beta1, se1, cov1 = logistic_fit_cluster(x1, y1, clusters1)
        x2, y2, clusters2, _, cols2 = design_matrix(model_data, True, generators)
        beta2, se2, cov2 = logistic_fit_cluster(x2, y2, clusters2)

        gen_indices1 = list(range(1, len(generators)))
        gen_indices2 = list(range(1, len(generators)))
        joint1 = wald_joint(beta1, cov1, gen_indices1) if gen_indices1 else (0.0, 1.0)
        joint2 = wald_joint(beta2, cov2, gen_indices2) if gen_indices2 else (0.0, 1.0)

        for model_name, beta, se, columns in [
            ("Generator only", beta1, se1, cols1),
            ("+ 8 features", beta2, se2, cols2),
        ]:
            for index, column in enumerate(columns):
                if not column.startswith("Generator:"):
                    continue
                estimate = float(beta[index])
                std_err = float(se[index]) if se[index] > 0 else float("nan")
                p_value = normal_pvalue(estimate / std_err) if std_err == std_err and std_err > 0 else "NA"
                gen_coef_rows.append({
                    "Detector": detector_label,
                    "Generator": column.replace("Generator: ", ""),
                    "Model": model_name,
                    "OR": math.exp(estimate),
                    "CI_low": math.exp(estimate - 1.96 * std_err) if std_err == std_err else "NA",
                    "CI_high": math.exp(estimate + 1.96 * std_err) if std_err == std_err else "NA",
                    "p_value": p_value,
                })

        for index, feature in enumerate(FEATURES, start=len(cols2) - len(FEATURES)):
            estimate = float(beta2[index])
            std_err = float(se2[index]) if se2[index] > 0 else float("nan")
            p_value = normal_pvalue(estimate / std_err) if std_err == std_err and std_err > 0 else "NA"
            feature_rows.append({
                "Detector": detector_label,
                "Feature": feature,
                "OR": math.exp(estimate),
                "CI_low": math.exp(estimate - 1.96 * std_err) if std_err == std_err else "NA",
                "CI_high": math.exp(estimate + 1.96 * std_err) if std_err == std_err else "NA",
                "p_value": p_value,
            })

        observed_rates = {}
        for generator in GENERATOR_ORDER:
            gen_rows = [row for row in all_available if row["generator"] == generator]
            if gen_rows:
                observed_rates[generator] = sum(row["_y"] for row in gen_rows) / len(gen_rows)
                se_obs = math.sqrt(max(observed_rates[generator] * (1 - observed_rates[generator]) / len(gen_rows), 0.0))
                marginal_rows.append({
                    "Detector": detector_label,
                    "Generator": generator,
                    "Type": "Observed",
                    "Probability": observed_rates[generator],
                    "CI_low": max(0.0, observed_rates[generator] - 1.96 * se_obs),
                    "CI_high": min(1.0, observed_rates[generator] + 1.96 * se_obs),
                    "Note": "excluded from adjusted model" if detector_label == "PhishingV3" and generator == "DeepSeek" else "",
                })

        adjusted_rates = {}
        for generator in generators:
            x_counter = x2.copy()
            for gen_index, gen_candidate in enumerate(generators[1:], start=1):
                x_counter[:, gen_index] = 1.0 if generator == gen_candidate else 0.0
            eta = np.clip(x_counter.dot(beta2), -30, 30)
            p_hat = 1.0 / (1.0 + np.exp(-eta))
            probability = float(p_hat.mean())
            gradients = (p_hat * (1 - p_hat))[:, None] * x_counter
            gradient = gradients.mean(axis=0)
            se_prob = math.sqrt(max(float(gradient.T.dot(cov2).dot(gradient)), 0.0))
            adjusted_rates[generator] = probability
            marginal_rows.append({
                "Detector": detector_label,
                "Generator": generator,
                "Type": "Adjusted for 8 characteristics",
                "Probability": probability,
                "CI_low": max(0.0, probability - 1.96 * se_prob),
                "CI_high": min(1.0, probability + 1.96 * se_prob),
                "Note": "",
            })

        raw_range = (max(observed_rates[g] for g in generators if g in observed_rates) - min(observed_rates[g] for g in generators if g in observed_rates)) * 100.0
        adjusted_range = (max(adjusted_rates.values()) - min(adjusted_rates.values())) * 100.0
        if detector_label == "PhishingV3":
            interpretation = "among the five non-DeepSeek generators, feature adjustment changes generator variation only slightly"
        elif adjusted_range < raw_range * 0.5:
            interpretation = "features account for a substantial share of generator variation"
        elif adjusted_range < raw_range * 0.85:
            interpretation = "features account for part of generator variation"
        else:
            interpretation = "generator variation remains strong after feature adjustment"
        if detector_label == "PhishingV3":
            interpretation += "; DeepSeek retained in RQ1 but excluded from adjusted model due to sparse TP"
        adjustment_rows.append({
            "Detector": detector_label,
            "Raw generator range": raw_range,
            "Adjusted generator range": adjusted_range,
            "Generator joint test M1": f"Wald={joint1[0]:.3g}; p={joint1[1]:.3g}",
            "Generator joint test M2": f"Wald={joint2[0]:.3g}; p={joint2[1]:.3g}",
            "Interpretation": interpretation,
        })

    bh_fdr(gen_coef_rows)
    bh_fdr(feature_rows)
    return gen_coef_rows, feature_rows, adjustment_rows, marginal_rows


def draw_delta_heatmap(path, tpfn_rows, detector_labels, title):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    # Base font sizes (user prefers large fonts)
    BASE_FONT = 36
    TICK_LABEL_FONT = BASE_FONT + 2  # x/y axis labels +2
    INNER_NUMBER_FONT = 32 + 1  # numbers inside heatmap +1 (was 32)
    STAR_FONT = 30 + 1

    plt.rcParams.update({"font.size": BASE_FONT})
    figure, axes = plt.subplots(1, len(detector_labels), figsize=(13.0 * len(detector_labels), 15.0), squeeze=False)
    max_abs = 1.0
    matrices = {}
    label_matrices = {}
    star_matrices = {}
    sparse_matrices = {}
    for detector in detector_labels:
        matrix = np.zeros((len(FEATURES), len(GENERATOR_ORDER)))
        labels = [["" for _ in GENERATOR_ORDER] for _ in FEATURES]
        stars = [["" for _ in GENERATOR_ORDER] for _ in FEATURES]
        sparse = [["" for _ in GENERATOR_ORDER] for _ in FEATURES]
        for row in tpfn_rows:
            if row["Detector"] != detector:
                continue
            i = FEATURES.index(row["Feature"])
            j = GENERATOR_ORDER.index(row["Generator"])
            delta = float(row["Delta_pp"])
            if row.get("sparse_note"):
                matrix[i, j] = np.nan
                labels[i][j] = f"{delta:+.1f}"
            else:
                matrix[i, j] = delta
                labels[i][j] = f"{delta:+.1f}"
            stars[i][j] = significance_stars(row["q_value"])
            if row.get("sparse_note"):
                sparse[i][j] = "sparse"
        matrices[detector] = matrix
        label_matrices[detector] = labels
        star_matrices[detector] = stars
        sparse_matrices[detector] = sparse
        finite = matrix[np.isfinite(matrix)]
        if finite.size:
            max_abs = max(max_abs, float(np.max(np.abs(finite))))

    for panel_index, (axis, detector) in enumerate(zip(axes[0], detector_labels)):
        matrix = matrices[detector]
        cmap = copy.copy(plt.get_cmap("RdBu"))
        cmap.set_bad(color="#d9d9d9")
        image = axis.imshow(matrix, cmap=cmap, vmin=-max_abs, vmax=max_abs)
        axis.set_xticks(range(len(GENERATOR_ORDER)))
        axis.set_xticklabels(GENERATOR_ORDER, rotation=35, ha="right", fontsize=TICK_LABEL_FONT)
        axis.set_yticks(range(len(FEATURES)))
        # Split each feature label into two lines (first space) for readability
        multiline_features = [f.replace(" ", "\n", 1) for f in FEATURES]
        if panel_index == 0:
            axis.set_yticklabels(multiline_features, fontsize=TICK_LABEL_FONT)
        else:
            axis.set_yticklabels([])
            axis.tick_params(axis="y", length=0)
        for i in range(len(FEATURES)):
            for j in range(len(GENERATOR_ORDER)):
                if sparse_matrices[detector][i][j]:
                    axis.text(j, i, label_matrices[detector][i][j], ha="center", va="center", fontsize=INNER_NUMBER_FONT, color="black")
                else:
                    stars = star_matrices[detector][i][j]
                    text_color = "white" if abs(matrix[i, j]) >= 30 else "black"
                    if stars:
                        axis.text(j, i - 0.13, label_matrices[detector][i][j], ha="center", va="center", fontsize=INNER_NUMBER_FONT, color=text_color)
                        axis.text(j, i + 0.20, stars, ha="center", va="center", fontsize=STAR_FONT, color=text_color)
                    else:
                        axis.text(j, i, label_matrices[detector][i][j], ha="center", va="center", fontsize=INNER_NUMBER_FONT, color=text_color)
        axis.set_xticks([x - 0.5 for x in range(1, len(GENERATOR_ORDER))], minor=True)
        axis.set_yticks([y - 0.5 for y in range(1, len(FEATURES))], minor=True)
        axis.grid(which="minor", color="white", linewidth=1.2)
        axis.tick_params(which="minor", bottom=False, left=False)
    figure.subplots_adjust(wspace=0.035, bottom=0.19, top=0.98, right=0.88)
    colorbar = figure.colorbar(image, ax=axes.ravel().tolist(), fraction=0.025, pad=0.065)
    colorbar.set_label("TP-FN feature prevalence (pp)", fontsize=BASE_FONT)
    colorbar.ax.tick_params(labelsize=INNER_NUMBER_FONT)
    figure.savefig(path, dpi=300, bbox_inches="tight")
    plt.close(figure)


def draw_fig_d(path, marginal_rows, adjustment_rows):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    # Set large font sizes: general text 36, numeric tick labels 32
    plt.rcParams.update({
        "font.size": 36,
        "axes.titlesize": 36,
        "axes.labelsize": 36,
        "xtick.labelsize": 32,
        "ytick.labelsize": 32,
        "legend.fontsize": 32,
    })

    figure, axes = plt.subplots(1, 2, figsize=(28.0, 11.0), sharey=True)
    panel_specs = [
        ("SecureNet", GENERATOR_ORDER),
        ("PhishingV3", [generator for generator in GENERATOR_ORDER if generator != "DeepSeek"]),
    ]
    styles = [
        ("Observed", "#4c78a8", "o", -0.16, "Observed"),
        ("Adjusted for 8 characteristics", "#f58518", "s", 0.16, "Adjusted"),
    ]

    for panel_index, (axis, (detector, generators)) in enumerate(zip(axes, panel_specs)):
        x_values = np.arange(len(generators))
        for series_type, color, marker, offset, label in styles:
            values = []
            yerr_low = []
            yerr_high = []
            valid_x = []
            for x_index, generator in enumerate(generators):
                matches = [
                    item for item in marginal_rows
                    if item["Detector"] == detector
                    and item["Generator"] == generator
                    and item["Type"] == series_type
                ]
                if not matches:
                    continue
                row = matches[0]
                value = float(row["Probability"])
                valid_x.append(x_values[x_index] + offset)
                values.append(value * 100.0)
                yerr_low.append((value - float(row["CI_low"])) * 100.0)
                yerr_high.append((float(row["CI_high"]) - value) * 100.0)
            axis.errorbar(
                valid_x,
                values,
                yerr=[yerr_low, yerr_high],
                fmt=marker,
                linestyle="none",
                color=color,
                capsize=7,
                markersize=13,
                label=label,
            )
        axis.set_title(detector, fontsize=36, pad=16)
        axis.set_xticks(x_values)
        axis.set_xticklabels(generators, rotation=35, ha="right")
        axis.set_ylim(0, 100)
        axis.grid(axis="y", color="#dddddd", linewidth=0.9)
        if panel_index == 0:
            axis.set_ylabel("Detection probability (%)")
        else:
            axis.tick_params(axis="y", labelleft=False)

    deep_items = [
        item for item in marginal_rows
        if item["Detector"] == "PhishingV3"
        and item["Generator"] == "DeepSeek"
        and item["Type"] == "Observed"
    ]
    if deep_items:
        deep = deep_items[0]
        axes[1].text(
            0.98,
            0.08,
            f"DeepSeek observed={float(deep['Probability']) * 100:.1f}%\nexcluded from adjusted model",
            transform=axes[1].transAxes,
            ha="right",
            va="bottom",
            fontsize=30,
            bbox={"facecolor": "white", "edgecolor": "#cccccc", "alpha": 0.88, "pad": 5},
        )

    handles, labels = axes[0].get_legend_handles_labels()
    figure.legend(handles, labels, loc="upper center", ncol=2, fontsize=32, frameon=False)
    figure.subplots_adjust(left=0.10, bottom=0.24, right=0.98, top=0.82, wspace=0.10)
    figure.savefig(path, dpi=300, bbox_inches="tight")
    plt.close(figure)


def write_summary(tpfn_rows, adjustment_rows):
    lines = [
        "# S8 RQ3 paired analysis",
        "",
        "TP/FN feature differences use the 499-prompt paired dataset. Fisher tests are FDR-corrected within detector.",
        "",
        "## Sparse cell",
        "",
        "PhishingV3 x DeepSeek has sparse TP and is descriptive only in TP/FN feature analysis; DeepSeek is excluded from PhishingV3 adjusted GEE.",
        "",
        "## Adjustment summary",
        "",
        "| Detector | Raw range | Adjusted range | Interpretation |",
        "| --- | ---: | ---: | --- |",
    ]
    for row in adjustment_rows:
        lines.append(f"| {row['Detector']} | {float(row['Raw generator range']):.1f} pp | {float(row['Adjusted generator range']):.1f} pp | {row['Interpretation']} |")
    SUMMARY_MD.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main():
    rows = read_rows()
    tpfn_rows = rq3_tp_fn_features(rows)
    write_csv(TP_FN_CSV, tpfn_rows, ["Detector", "Generator", "Feature", "N_TP", "N_FN", "TP_%", "FN_%", "Delta_pp", "Fisher_OR", "p_value", "q_value", "sparse_note"])

    gen_rows, feature_rows, adjustment_rows, marginal_rows = gee_outputs(rows)
    write_csv(GENERATOR_EFFECTS_CSV, gen_rows, ["Detector", "Generator", "Model", "OR", "CI_low", "CI_high", "p_value", "q_value"])
    write_csv(FEATURE_EFFECTS_CSV, feature_rows, ["Detector", "Feature", "OR", "CI_low", "CI_high", "p_value", "q_value"])
    write_csv(ADJUSTMENT_SUMMARY_CSV, adjustment_rows, ["Detector", "Raw generator range", "Adjusted generator range", "Generator joint test M1", "Generator joint test M2", "Interpretation"])
    write_csv(MARGINAL_CSV, marginal_rows, ["Detector", "Generator", "Type", "Probability", "CI_low", "CI_high", "Note"])

    draw_delta_heatmap(FIG_C, tpfn_rows, ["SecureNet", "PhishingV3"], "Fig. S8-C. Detector-specific TP-FN characteristic differences")
    draw_delta_heatmap(FIG_C_APP, tpfn_rows, ["ScamLLM", "PiMRef", "T5", "XGBoost"], "Appendix. TP-FN characteristic differences")
    draw_fig_d(FIG_D, marginal_rows, adjustment_rows)
    write_summary(tpfn_rows, adjustment_rows)

    print(f"Wrote {TP_FN_CSV}")
    print(f"Wrote {GENERATOR_EFFECTS_CSV}")
    print(f"Wrote {FEATURE_EFFECTS_CSV}")
    print(f"Wrote {ADJUSTMENT_SUMMARY_CSV}")
    print(f"Wrote {MARGINAL_CSV}")
    print(f"Wrote {FIG_C}")
    print(f"Wrote {FIG_C_APP}")
    print(f"Wrote {FIG_D}")
    print(f"Wrote {SUMMARY_MD}")


if __name__ == "__main__":
    main()
