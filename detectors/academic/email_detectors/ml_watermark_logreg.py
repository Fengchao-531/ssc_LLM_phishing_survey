#!/usr/bin/env python3
"""Paper-style TF-IDF + Logistic Regression detector for phishing emails."""

from __future__ import annotations

import argparse
import csv
import json
import pickle
import random
import sys
import zipfile
from pathlib import Path
from typing import Any


SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_DIR = SCRIPT_DIR.parents[2]
DEFAULT_INPUT_CSV = (
    PROJECT_DIR / "Datasets" / "sublist" / "S5-Personalization for Credibility" / "LLM-P.csv"
)
DEFAULT_OUTPUT_DIR = SCRIPT_DIR / "results"
DEFAULT_MODEL_DIR = SCRIPT_DIR / "trained_models"
DEFAULT_ARCHIVE4_ZIP = Path("/scratch2/pk79/fche0036/kaggle/archive (4).zip")
DEFAULT_ARCHIVE4_DIR = Path("/scratch2/pk79/fche0036/kaggle/archive4")
DEFAULT_TRAIN_LEGITIMATE_CSVS = [
    PROJECT_DIR / "Datasets" / "sublist" / "S4-Scenarios-driven Adaptation" / "HW-B.csv",
]
DEFAULT_TRAIN_PHISHING_CSVS = [
    PROJECT_DIR / "Datasets" / "sublist" / "S5-Personalization for Credibility" / "LLM-P.csv",
]
DEFAULT_MAX_FEATURES = 5000
DEFAULT_SAMPLE_SIZE_PER_CLASS = 1000
DEFAULT_RANDOM_STATE = 42
DEFAULT_MODEL_PATH = DEFAULT_MODEL_DIR / "ml_watermark_logreg_archive4_llm_only.pkl"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Approximate the Logistic Regression setup from "
            "'Machine Learning and Watermarking for Accurate Detection of AI-Generated "
            "Phishing Emails' and run phishing detection on a CSV file."
        )
    )
    parser.add_argument("--input-csv", type=Path, default=DEFAULT_INPUT_CSV)
    parser.add_argument("--subject-column", default="Subject")
    parser.add_argument("--body-column", default="Body")
    parser.add_argument("--label-column", default="label")
    parser.add_argument("--sample-size", type=int, default=0, help="Use 0 for all rows.")
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
        help="Directory for the output CSV. Output filename is always <input_stem>_results.csv.",
    )
    parser.add_argument(
        "--train-source",
        choices=["archive4_llm_only", "archive4_human_only", "archive4_all", "csvs"],
        default="archive4_llm_only",
        help=(
            "Training source. 'archive4_llm_only' uses the uploaded archive (4) data and trains on "
            "llm-generated legit vs llm-generated phishing."
        ),
    )
    parser.add_argument(
        "--archive4-zip",
        type=Path,
        default=DEFAULT_ARCHIVE4_ZIP,
        help="Path to the uploaded archive (4).zip dataset.",
    )
    parser.add_argument(
        "--archive4-dir",
        type=Path,
        default=DEFAULT_ARCHIVE4_DIR,
        help="Directory containing the extracted archive (4) dataset.",
    )
    parser.add_argument(
        "--train-legitimate-csvs",
        nargs="+",
        type=Path,
        default=list(DEFAULT_TRAIN_LEGITIMATE_CSVS),
        help="One or more legitimate-email CSVs used to fit the Logistic Regression detector.",
    )
    parser.add_argument(
        "--train-phishing-csvs",
        nargs="+",
        type=Path,
        default=list(DEFAULT_TRAIN_PHISHING_CSVS),
        help="One or more phishing-email CSVs used to fit the Logistic Regression detector.",
    )
    parser.add_argument(
        "--train-sample-size-per-class",
        type=int,
        default=DEFAULT_SAMPLE_SIZE_PER_CLASS,
        help="Rows sampled per class for training. Use 0 for all rows.",
    )
    parser.add_argument("--max-features", type=int, default=DEFAULT_MAX_FEATURES)
    parser.add_argument("--random-state", type=int, default=DEFAULT_RANDOM_STATE)
    parser.add_argument(
        "--watermark-token",
        default="",
        help=(
            "Optional visible watermark token to append to phishing training emails, "
            "mirroring the paper's watermark experiment."
        ),
    )
    parser.add_argument(
        "--skip-cross-val",
        action="store_true",
        help="Skip the paper-style 5-fold stratified cross-validation report.",
    )
    parser.add_argument(
        "--model-path",
        type=Path,
        default=DEFAULT_MODEL_PATH,
        help="Where to save/load the trained Logistic Regression pipeline.",
    )
    parser.add_argument(
        "--force-retrain",
        action="store_true",
        help="Ignore any saved model and retrain from the training corpus.",
    )
    parser.add_argument(
        "--train-only",
        action="store_true",
        help="Train and save the model, then exit without running inference on an input CSV.",
    )
    return parser.parse_args()


def normalize_text(value: Any) -> str:
    return (value or "").replace("\r\n", "\n").replace("\r", "\n").strip()


def build_content(subject: str, body: str) -> str:
    subject = normalize_text(subject)
    body = normalize_text(body)
    if subject and body:
        return f"{subject}\n\n{body}"
    return subject or body


def load_csv_rows(
    path: Path,
    *,
    subject_column: str,
    body_column: str,
    label_column: str,
    sample_size: int = 0,
) -> list[dict[str, str]]:
    csv.field_size_limit(sys.maxsize)
    if not path.exists():
        raise SystemExit(f"CSV not found: {path}")

    rows: list[dict[str, str]] = []
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        if not reader.fieldnames:
            raise SystemExit(f"CSV has no header row: {path}")
        required = [subject_column, body_column]
        missing = [name for name in required if name not in reader.fieldnames]
        if missing:
            raise SystemExit(f"CSV missing required columns {', '.join(missing)} in {path}")

        for index, record in enumerate(reader, start=1):
            if sample_size > 0 and index > sample_size:
                break
            subject = normalize_text(record.get(subject_column, ""))
            body = normalize_text(record.get(body_column, ""))
            rows.append(
                {
                    "subject": subject,
                    "body": body,
                    "label": str(record.get(label_column, "")),
                    "_content": build_content(subject, body),
                }
            )
    return rows


def sample_rows(
    rows: list[dict[str, str]],
    *,
    sample_size: int,
    random_state: int,
) -> list[dict[str, str]]:
    if sample_size <= 0 or len(rows) <= sample_size:
        return list(rows)
    rng = random.Random(random_state)
    sampled = list(rows)
    rng.shuffle(sampled)
    return sampled[:sample_size]


def ensure_archive4_extracted(archive_zip: Path, archive_dir: Path) -> Path:
    required = [
        archive_dir / "human-generated" / "legit.csv",
        archive_dir / "human-generated" / "phishing.csv",
        archive_dir / "llm-generated" / "legit.csv",
        archive_dir / "llm-generated" / "phishing.csv",
    ]
    if all(path.exists() for path in required):
        return archive_dir
    if not archive_zip.exists():
        raise SystemExit(f"Archive dataset not found: {archive_zip}")

    archive_dir.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(archive_zip) as zf:
        zf.extractall(archive_dir)
    if not all(path.exists() for path in required):
        raise SystemExit(f"Archive dataset extracted but required files are still missing in {archive_dir}")
    return archive_dir


def load_archive4_llm_rows(path: Path, *, label_override: int | None = None) -> list[dict[str, str]]:
    rows: list[dict[str, str]] = []
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        header = handle.readline().strip().lower()
        if header != "text,label":
            raise SystemExit(f"Unexpected LLM CSV header in {path}: {header}")

        for raw_line in handle:
            line = raw_line.strip()
            if not line:
                continue
            if "," not in line:
                continue
            text_part, label_part = line.rsplit(",", 1)
            text = normalize_text(text_part.strip())
            if text.startswith('"') and text.endswith('"') and len(text) >= 2:
                text = text[1:-1].replace('""', '"')
            label_text = label_part.strip()
            if label_override is None:
                try:
                    numeric_label = int(label_text)
                except ValueError as exc:
                    raise SystemExit(f"Invalid label '{label_text}' in {path}") from exc
            else:
                numeric_label = int(label_override)
            rows.append(
                {
                    "subject": "",
                    "body": text,
                    "label": str(numeric_label),
                    "_content": text,
                }
            )
    return rows


def load_archive4_human_rows(path: Path) -> list[dict[str, str]]:
    return load_csv_rows(
        path,
        subject_column="subject",
        body_column="body",
        label_column="label",
    )


def load_archive4_training_rows(args: argparse.Namespace) -> tuple[list[dict[str, str]], list[dict[str, str]]]:
    archive_dir = ensure_archive4_extracted(args.archive4_zip.resolve(), args.archive4_dir.resolve())

    human_legit = load_archive4_human_rows(archive_dir / "human-generated" / "legit.csv")
    human_phishing = load_archive4_human_rows(archive_dir / "human-generated" / "phishing.csv")
    llm_legit = load_archive4_llm_rows(archive_dir / "llm-generated" / "legit.csv", label_override=0)
    llm_phishing = load_archive4_llm_rows(archive_dir / "llm-generated" / "phishing.csv", label_override=1)

    if args.train_source == "archive4_llm_only":
        return llm_legit, llm_phishing
    if args.train_source == "archive4_human_only":
        return human_legit, human_phishing
    if args.train_source == "archive4_all":
        return human_legit + llm_legit, human_phishing + llm_phishing
    raise SystemExit(f"Unsupported archive-based train source: {args.train_source}")


def build_training_corpus(args: argparse.Namespace) -> tuple[list[str], list[int]]:
    if args.train_source == "csvs":
        legitimate_rows: list[dict[str, str]] = []
        phishing_rows: list[dict[str, str]] = []
        for path in args.train_legitimate_csvs:
            legitimate_rows.extend(
                load_csv_rows(
                    path.resolve(),
                    subject_column=args.subject_column,
                    body_column=args.body_column,
                    label_column=args.label_column,
                )
            )
        for path in args.train_phishing_csvs:
            phishing_rows.extend(
                load_csv_rows(
                    path.resolve(),
                    subject_column=args.subject_column,
                    body_column=args.body_column,
                    label_column=args.label_column,
                )
            )
    else:
        legitimate_rows, phishing_rows = load_archive4_training_rows(args)

    legitimate_rows = sample_rows(
        legitimate_rows,
        sample_size=args.train_sample_size_per_class,
        random_state=args.random_state,
    )
    phishing_rows = sample_rows(
        phishing_rows,
        sample_size=args.train_sample_size_per_class,
        random_state=args.random_state + 1,
    )

    watermark_token = str(args.watermark_token or "").strip()
    legitimate_texts = [row["_content"] for row in legitimate_rows]
    phishing_texts = []
    for row in phishing_rows:
        text = row["_content"]
        if watermark_token:
            text = f"{text}\n\n{watermark_token}".strip()
        phishing_texts.append(text)

    texts = legitimate_texts + phishing_texts
    labels = ([0] * len(legitimate_texts)) + ([1] * len(phishing_texts))
    if not texts:
        raise SystemExit("Training corpus is empty.")
    return texts, labels


def build_estimator(args: argparse.Namespace):
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.linear_model import LogisticRegression
    from sklearn.pipeline import Pipeline

    return Pipeline(
        steps=[
            (
                "tfidf",
                TfidfVectorizer(
                    ngram_range=(1, 2),
                    max_features=args.max_features,
                    lowercase=True,
                    strip_accents="unicode",
                ),
            ),
            (
                "classifier",
                LogisticRegression(
                    max_iter=1000,
                    random_state=args.random_state,
                    solver="liblinear",
                ),
            ),
        ]
    )


def run_cross_validation(args: argparse.Namespace, texts: list[str], labels: list[int]) -> None:
    from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
    from sklearn.model_selection import StratifiedKFold, cross_val_predict

    estimator = build_estimator(args)
    splitter = StratifiedKFold(n_splits=5, shuffle=True, random_state=args.random_state)
    predictions = cross_val_predict(estimator, texts, labels, cv=splitter)

    accuracy = accuracy_score(labels, predictions)
    report = classification_report(
        labels,
        predictions,
        target_names=["legitimate", "phishing"],
        digits=4,
        zero_division=0,
    )
    matrix = confusion_matrix(labels, predictions).tolist()

    print(f"cross_val_accuracy={accuracy:.4f}", flush=True)
    print(report, flush=True)
    print(f"cross_val_confusion_matrix={json.dumps(matrix)}", flush=True)


def fit_estimator(args: argparse.Namespace, texts: list[str], labels: list[int]):
    estimator = build_estimator(args)
    estimator.fit(texts, labels)
    return estimator


def train_and_save_estimator(args: argparse.Namespace):
    train_texts, train_labels = build_training_corpus(args)
    print(
        f"training_corpus legitimate={train_labels.count(0)} phishing={train_labels.count(1)} "
        f"max_features={args.max_features}",
        flush=True,
    )
    if not args.skip_cross_val:
        run_cross_validation(args, train_texts, train_labels)

    estimator = fit_estimator(args, train_texts, train_labels)
    model_path = args.model_path.resolve()
    model_path.parent.mkdir(parents=True, exist_ok=True)
    with model_path.open("wb") as handle:
        pickle.dump(estimator, handle)
    print(f"saved_model: {model_path}", flush=True)
    return estimator


def load_saved_estimator(model_path: Path):
    with model_path.open("rb") as handle:
        return pickle.load(handle)


def main() -> int:
    args = parse_args()
    model_path = args.model_path.resolve()
    if model_path.exists() and not args.force_retrain:
        estimator = load_saved_estimator(model_path)
        print(f"loaded_model: {model_path}", flush=True)
    else:
        estimator = train_and_save_estimator(args)

    if args.train_only:
        return 0

    input_csv = args.input_csv.resolve()
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    output_csv = output_dir / f"{input_csv.stem}_results.csv"

    input_rows = load_csv_rows(
        input_csv,
        subject_column=args.subject_column,
        body_column=args.body_column,
        label_column=args.label_column,
        sample_size=args.sample_size,
    )
    texts = [row["_content"] for row in input_rows]
    predictions = estimator.predict(texts).tolist()

    output_rows: list[dict[str, Any]] = []
    total = len(input_rows)
    for index, (row, prediction) in enumerate(zip(input_rows, predictions), start=1):
        output_rows.append(
            {
                "subject": row["subject"],
                "body": row["body"],
                "label": row["label"],
                "model_prediction": int(prediction),
            }
        )
        if index == total or index % 500 == 0:
            print(f"processed {index}/{total}", flush=True)

    with output_csv.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=["subject", "body", "label", "model_prediction"],
        )
        writer.writeheader()
        writer.writerows(output_rows)

    print(f"saved: {output_csv}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
