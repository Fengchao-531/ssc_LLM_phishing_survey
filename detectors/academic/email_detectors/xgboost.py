#!/usr/bin/env python3
"""Run a paper-style stylometric XGBoost detector on CSV email data."""

from __future__ import annotations

import argparse
import csv
import importlib
import json
import math
import pickle
import re
import statistics
import sys
from pathlib import Path
from typing import Any

import numpy as np


SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_DIR = SCRIPT_DIR.parents[2]
DEFAULT_INPUT_CSV = (
    PROJECT_DIR / "Datasets" / "sublist" / "S5-Personalization for Credibility" / "LLM-P.csv"
)
DEFAULT_OUTPUT_DIR = SCRIPT_DIR / "results"
DEFAULT_MODEL_DIR = SCRIPT_DIR / "trained_models"
DEFAULT_KAGGLE_CSV = Path("/scratch2/pk79/fche0036/kaggle/AI_Generated_Phishing_Legitimate_Emails.csv")
DEFAULT_RANDOM_STATE = 42
DEFAULT_TEST_SIZE = 0.2
DEFAULT_MODEL_PATH = DEFAULT_MODEL_DIR / "xgboost_kaggle.pkl"

WORD_RE = re.compile(r"[A-Za-z]+(?:'[A-Za-z]+)?")
EMAIL_RE = re.compile(r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}\b")
URL_RE = re.compile(r"\b(?:https?://|www\.)\S+\b", re.IGNORECASE)
PHONE_RE = re.compile(r"(?:\+?\d[\d\s().-]{6,}\d)")
UPPERCASE_WORD_RE = re.compile(r"\b[A-Z]{2,}\b")

FIRST_PERSON_PRONOUNS = {"i", "we", "me", "us", "my", "our", "ours", "mine"}
SECOND_PERSON_PRONOUNS = {"you", "your", "yours", "yourself", "yourselves"}
ALL_PRONOUNS = FIRST_PERSON_PRONOUNS | SECOND_PERSON_PRONOUNS | {
    "he",
    "she",
    "it",
    "they",
    "him",
    "her",
    "them",
    "his",
    "hers",
    "its",
    "their",
    "theirs",
}
PREPOSITIONS = {"in", "on", "at", "by", "with", "for", "from", "to", "of", "into", "over", "under"}
FUNCTION_WORDS = {
    "the",
    "is",
    "at",
    "which",
    "on",
    "a",
    "an",
    "and",
    "or",
    "but",
    "if",
    "this",
    "that",
    "to",
    "of",
}
SUBORDINATE_CONJUNCTIONS = {"and", "but", "or", "because", "if", "unless", "while", "although"}
IMPERATIVE_VERBS = {"click", "verify", "submit", "download", "update"}
MODAL_VERBS = {"can", "could", "should"}
UNCERTAINTY_ADVERBS = {"maybe", "possibly", "perhaps"}
TECHNICAL_JARGON = {"security", "account", "update", "technical", "system", "verification", "password"}
PROMOTIONAL_WORDS = {"offer", "deal", "free", "exclusive", "bonus", "limited"}
POLITENESS_MARKERS = {"please", "thank", "thanks", "appreciate", "kindly"}
AGGRESSIVENESS_MARKERS = {"must", "now", "immediately", "required", "urgent"}
URGENCY_MARKERS = {"urgent", "asap", "immediately", "today", "now", "expire", "expired", "deadline"}
CONDITIONAL_PHRASES = {"if", "unless"}
PERSONALIZATION_MARKERS = {"you", "your", "dear", "hello", "hi"}
EMOTIVE_WORDS = {"exciting", "amazing", "wonderful", "pleased", "congratulations", "fantastic"}
APOLOGY_MARKERS = {"sorry", "apologize", "apologies"}
AUTHORITY_MARKERS = {"official", "security", "support", "team", "administrator", "manager", "bank"}
ATTACHMENT_MARKERS = {"attachment", "attached", "pdf", "document", "file"}
CURRENCY_SYMBOLS = {"$", "£", "€"}

FEATURE_NAMES = [
    "word_count",
    "character_count",
    "average_word_length",
    "sentence_count",
    "paragraph_count",
    "average_sentence_length_words",
    "lexical_diversity",
    "complex_word_count",
    "complex_word_ratio",
    "average_syllables_per_word",
    "digit_count",
    "digit_ratio",
    "uppercase_word_count",
    "uppercase_word_ratio",
    "title_case_word_count",
    "comma_count",
    "semicolon_count",
    "colon_count",
    "exclamation_count",
    "question_count",
    "quotation_count",
    "dash_count",
    "ellipsis_count",
    "punctuation_variety",
    "sentence_complexity_ratio",
    "clause_density",
    "pronoun_density",
    "preposition_density",
    "function_word_density",
    "subordinate_conjunction_count",
    "period_frequency",
    "comma_frequency",
    "semicolon_frequency",
    "colon_frequency",
    "exclamation_frequency",
    "question_frequency",
    "quotation_frequency",
    "dash_frequency",
    "parenthesis_count",
    "slash_count",
    "flesch_reading_ease",
    "smog_index",
    "dale_chall_readability_score",
    "coleman_liau_index",
    "gunning_fog",
    "automated_readability_index",
    "difficult_words_count",
    "polysyllable_count",
    "pronoun_count",
    "first_person_pronoun_count",
    "second_person_pronoun_count",
    "imperative_verbs_count",
    "modal_verbs_count",
    "uncertainty_adverbs_count",
    "technical_jargon_count",
    "promotional_words_count",
    "email_address_count",
    "attachment_mentions_count",
    "bigram_count",
    "trigram_count",
    "word_length_variation",
    "politeness_markers_count",
    "aggressiveness_markers_count",
    "urgency_markers_count",
    "conditional_phrases_count",
    "personalization_markers_count",
    "url_count",
    "currency_symbol_count",
    "phone_number_count",
    "emotive_words_count",
    "apology_markers_count",
    "authority_markers_count",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Read subject/body from a CSV, fit a stylometric XGBoost detector on the original "
            "Kaggle AI_Generated_Phishing_Legitimate_Emails dataset, and save "
            "subject/body/label/model_prediction to <input_stem>_results.csv."
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
    parser.add_argument("--kaggle-csv", type=Path, default=DEFAULT_KAGGLE_CSV)
    parser.add_argument("--random-state", type=int, default=DEFAULT_RANDOM_STATE)
    parser.add_argument("--test-size", type=float, default=DEFAULT_TEST_SIZE)
    parser.add_argument(
        "--skip-eval",
        action="store_true",
        help="Skip the paper-style train/test evaluation report before fitting on the full dataset.",
    )
    parser.add_argument(
        "--model-path",
        type=Path,
        default=DEFAULT_MODEL_PATH,
        help="Where to save/load the trained XGBoost detector.",
    )
    parser.add_argument(
        "--force-retrain",
        action="store_true",
        help="Ignore any saved model and retrain from the Kaggle dataset.",
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


def split_sentences(text: str) -> list[str]:
    pieces = [piece.strip() for piece in re.split(r"[.!?]+", text) if piece.strip()]
    return pieces or [text.strip()] if text.strip() else []


def split_paragraphs(text: str) -> list[str]:
    return [piece.strip() for piece in re.split(r"\n\s*\n", text) if piece.strip()]


def tokenize_words(text: str) -> list[str]:
    return WORD_RE.findall(text)


def safe_divide(numerator: float, denominator: float) -> float:
    if denominator == 0:
        return 0.0
    return float(numerator) / float(denominator)


def count_syllables_in_word(word: str) -> int:
    lowered = re.sub(r"[^a-z]", "", word.lower())
    if not lowered:
        return 0
    vowels = "aeiouy"
    count = 0
    prev_is_vowel = False
    for char in lowered:
        is_vowel = char in vowels
        if is_vowel and not prev_is_vowel:
            count += 1
        prev_is_vowel = is_vowel
    if lowered.endswith("e") and count > 1:
        count -= 1
    return max(1, count)


def textstat_metrics(text: str) -> dict[str, float]:
    try:
        import textstat
    except ImportError as exc:
        raise SystemExit("textstat is not installed in the current Python environment.") from exc

    if not text.strip():
        return {
            "flesch_reading_ease": 0.0,
            "smog_index": 0.0,
            "dale_chall_readability_score": 0.0,
            "coleman_liau_index": 0.0,
            "gunning_fog": 0.0,
            "automated_readability_index": 0.0,
            "difficult_words_count": 0.0,
        }

    def safe_metric(fn_name: str) -> float:
        fn = getattr(textstat, fn_name)
        try:
            value = fn(text)
            return float(value)
        except Exception:
            return 0.0

    return {
        "flesch_reading_ease": safe_metric("flesch_reading_ease"),
        "smog_index": safe_metric("smog_index"),
        "dale_chall_readability_score": safe_metric("dale_chall_readability_score"),
        "coleman_liau_index": safe_metric("coleman_liau_index"),
        "gunning_fog": safe_metric("gunning_fog"),
        "automated_readability_index": safe_metric("automated_readability_index"),
        "difficult_words_count": safe_metric("difficult_words"),
    }


def count_membership(tokens_lower: list[str], vocabulary: set[str]) -> int:
    return sum(1 for token in tokens_lower if token in vocabulary)


def count_attachment_mentions(tokens_lower: list[str]) -> int:
    return count_membership(tokens_lower, ATTACHMENT_MARKERS)


def extract_features(text: str) -> dict[str, float]:
    normalized_text = normalize_text(text)
    tokens = tokenize_words(normalized_text)
    tokens_lower = [token.lower() for token in tokens]
    sentences = split_sentences(normalized_text)
    paragraphs = split_paragraphs(normalized_text)
    word_count = len(tokens)
    char_count = len(normalized_text)
    unique_words = len(set(tokens_lower))
    sentence_count = len(sentences)
    paragraph_count = len(paragraphs)
    word_lengths = [len(token) for token in tokens]
    syllable_counts = [count_syllables_in_word(token) for token in tokens]
    complex_word_count = sum(1 for token in tokens if len(token) > 6)
    polysyllable_count = sum(1 for value in syllable_counts if value >= 3)
    digit_count = sum(ch.isdigit() for ch in normalized_text)
    uppercase_word_count = len(UPPERCASE_WORD_RE.findall(normalized_text))
    title_case_word_count = sum(1 for token in tokens if len(token) > 1 and token.istitle())

    comma_count = normalized_text.count(",")
    semicolon_count = normalized_text.count(";")
    colon_count = normalized_text.count(":")
    exclamation_count = normalized_text.count("!")
    question_count = normalized_text.count("?")
    quotation_count = normalized_text.count('"') + normalized_text.count("'")
    dash_count = normalized_text.count("-")
    ellipsis_count = normalized_text.count("...")
    punctuation_counts = {
        ".": normalized_text.count("."),
        ",": comma_count,
        ";": semicolon_count,
        ":": colon_count,
        "!": exclamation_count,
        "?": question_count,
        "-": dash_count,
        '"': normalized_text.count('"'),
    }
    punctuation_variety = sum(1 for value in punctuation_counts.values() if value > 0)

    conjunction_count = count_membership(tokens_lower, SUBORDINATE_CONJUNCTIONS)
    pronoun_count = count_membership(tokens_lower, ALL_PRONOUNS)
    preposition_count = count_membership(tokens_lower, PREPOSITIONS)
    function_word_count = count_membership(tokens_lower, FUNCTION_WORDS)
    first_person_pronoun_count = count_membership(tokens_lower, FIRST_PERSON_PRONOUNS)
    second_person_pronoun_count = count_membership(tokens_lower, SECOND_PERSON_PRONOUNS)
    imperative_verbs_count = count_membership(tokens_lower, IMPERATIVE_VERBS)
    modal_verbs_count = count_membership(tokens_lower, MODAL_VERBS)
    uncertainty_adverbs_count = count_membership(tokens_lower, UNCERTAINTY_ADVERBS)
    technical_jargon_count = count_membership(tokens_lower, TECHNICAL_JARGON)
    promotional_words_count = count_membership(tokens_lower, PROMOTIONAL_WORDS)
    politeness_markers_count = count_membership(tokens_lower, POLITENESS_MARKERS)
    aggressiveness_markers_count = count_membership(tokens_lower, AGGRESSIVENESS_MARKERS)
    urgency_markers_count = count_membership(tokens_lower, URGENCY_MARKERS)
    conditional_phrases_count = count_membership(tokens_lower, CONDITIONAL_PHRASES)
    personalization_markers_count = count_membership(tokens_lower, PERSONALIZATION_MARKERS)
    emotive_words_count = count_membership(tokens_lower, EMOTIVE_WORDS)
    apology_markers_count = count_membership(tokens_lower, APOLOGY_MARKERS)
    authority_markers_count = count_membership(tokens_lower, AUTHORITY_MARKERS)

    email_address_count = len(EMAIL_RE.findall(normalized_text))
    url_count = len(URL_RE.findall(normalized_text))
    phone_number_count = len(PHONE_RE.findall(normalized_text))
    currency_symbol_count = sum(normalized_text.count(symbol) for symbol in CURRENCY_SYMBOLS)
    attachment_mentions_count = count_attachment_mentions(tokens_lower)

    bigram_count = max(word_count - 1, 0)
    trigram_count = max(word_count - 2, 0)
    word_length_variation = float(statistics.pstdev(word_lengths)) if len(word_lengths) > 1 else 0.0

    readability = textstat_metrics(normalized_text)

    features = {
        "word_count": float(word_count),
        "character_count": float(char_count),
        "average_word_length": safe_divide(sum(word_lengths), word_count),
        "sentence_count": float(sentence_count),
        "paragraph_count": float(paragraph_count),
        "average_sentence_length_words": safe_divide(word_count, sentence_count),
        "lexical_diversity": safe_divide(unique_words, word_count),
        "complex_word_count": float(complex_word_count),
        "complex_word_ratio": safe_divide(complex_word_count, word_count),
        "average_syllables_per_word": safe_divide(sum(syllable_counts), word_count),
        "digit_count": float(digit_count),
        "digit_ratio": safe_divide(digit_count, char_count),
        "uppercase_word_count": float(uppercase_word_count),
        "uppercase_word_ratio": safe_divide(uppercase_word_count, word_count),
        "title_case_word_count": float(title_case_word_count),
        "comma_count": float(comma_count),
        "semicolon_count": float(semicolon_count),
        "colon_count": float(colon_count),
        "exclamation_count": float(exclamation_count),
        "question_count": float(question_count),
        "quotation_count": float(quotation_count),
        "dash_count": float(dash_count),
        "ellipsis_count": float(ellipsis_count),
        "punctuation_variety": float(punctuation_variety),
        "sentence_complexity_ratio": safe_divide(conjunction_count, sentence_count),
        "clause_density": safe_divide(safe_divide(conjunction_count, sentence_count), sentence_count),
        "pronoun_density": safe_divide(pronoun_count, word_count),
        "preposition_density": safe_divide(preposition_count, word_count),
        "function_word_density": safe_divide(function_word_count, word_count),
        "subordinate_conjunction_count": float(conjunction_count),
        "period_frequency": safe_divide(punctuation_counts["."], char_count),
        "comma_frequency": safe_divide(comma_count, char_count),
        "semicolon_frequency": safe_divide(semicolon_count, char_count),
        "colon_frequency": safe_divide(colon_count, char_count),
        "exclamation_frequency": safe_divide(exclamation_count, char_count),
        "question_frequency": safe_divide(question_count, char_count),
        "quotation_frequency": safe_divide(quotation_count, char_count),
        "dash_frequency": safe_divide(dash_count, char_count),
        "parenthesis_count": float(normalized_text.count("(") + normalized_text.count(")")),
        "slash_count": float(normalized_text.count("/")),
        "flesch_reading_ease": readability["flesch_reading_ease"],
        "smog_index": readability["smog_index"],
        "dale_chall_readability_score": readability["dale_chall_readability_score"],
        "coleman_liau_index": readability["coleman_liau_index"],
        "gunning_fog": readability["gunning_fog"],
        "automated_readability_index": readability["automated_readability_index"],
        "difficult_words_count": readability["difficult_words_count"],
        "polysyllable_count": float(polysyllable_count),
        "pronoun_count": float(pronoun_count),
        "first_person_pronoun_count": float(first_person_pronoun_count),
        "second_person_pronoun_count": float(second_person_pronoun_count),
        "imperative_verbs_count": float(imperative_verbs_count),
        "modal_verbs_count": float(modal_verbs_count),
        "uncertainty_adverbs_count": float(uncertainty_adverbs_count),
        "technical_jargon_count": float(technical_jargon_count),
        "promotional_words_count": float(promotional_words_count),
        "email_address_count": float(email_address_count),
        "attachment_mentions_count": float(attachment_mentions_count),
        "bigram_count": float(bigram_count),
        "trigram_count": float(trigram_count),
        "word_length_variation": float(word_length_variation),
        "politeness_markers_count": float(politeness_markers_count),
        "aggressiveness_markers_count": float(aggressiveness_markers_count),
        "urgency_markers_count": float(urgency_markers_count),
        "conditional_phrases_count": float(conditional_phrases_count),
        "personalization_markers_count": float(personalization_markers_count),
        "url_count": float(url_count),
        "currency_symbol_count": float(currency_symbol_count),
        "phone_number_count": float(phone_number_count),
        "emotive_words_count": float(emotive_words_count),
        "apology_markers_count": float(apology_markers_count),
        "authority_markers_count": float(authority_markers_count),
    }

    missing = [name for name in FEATURE_NAMES if name not in features]
    if missing:
        raise SystemExit(f"Feature extraction is missing expected features: {missing}")
    return features


def vectorize_texts(texts: list[str]) -> np.ndarray:
    matrix = []
    for text in texts:
        feature_map = extract_features(text)
        matrix.append([feature_map[name] for name in FEATURE_NAMES])
    return np.asarray(matrix, dtype=np.float32)


def load_kaggle_dataset(path: Path) -> tuple[list[str], list[int]]:
    csv.field_size_limit(sys.maxsize)
    if not path.exists():
        raise SystemExit(f"Kaggle dataset not found: {path}")

    texts: list[str] = []
    labels: list[int] = []
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        if not reader.fieldnames:
            raise SystemExit(f"Kaggle dataset has no header row: {path}")
        required = ["Subject", "Body", "Label"]
        missing = [name for name in required if name not in reader.fieldnames]
        if missing:
            raise SystemExit(f"Kaggle dataset missing required columns {', '.join(missing)} in {path}")
        for record in reader:
            subject = normalize_text(record.get("Subject", ""))
            body = normalize_text(record.get("Body", ""))
            label_text = normalize_text(record.get("Label", "")).lower()
            if label_text not in {"phishing", "legitimate"}:
                continue
            texts.append(build_content(subject, body))
            labels.append(1 if label_text == "phishing" else 0)

    if not texts:
        raise SystemExit(f"No usable rows loaded from Kaggle dataset: {path}")
    return texts, labels


def build_estimator(random_state: int):
    try:
        xgboost_module = import_xgboost_module()
        XGBClassifier = xgboost_module.XGBClassifier
    except ImportError as exc:
        raise SystemExit("xgboost is not installed in the current Python environment.") from exc

    return XGBClassifier(
        use_label_encoder=False,
        eval_metric="logloss",
        random_state=random_state,
    )


def report_eval(args: argparse.Namespace, features: np.ndarray, labels: np.ndarray) -> None:
    from sklearn.metrics import accuracy_score, precision_score, recall_score, roc_auc_score
    from sklearn.model_selection import train_test_split

    x_train, x_test, y_train, y_test = train_test_split(
        features,
        labels,
        test_size=args.test_size,
        random_state=args.random_state,
        stratify=labels,
    )
    model = build_estimator(args.random_state)
    model.fit(x_train, y_train)
    predictions = model.predict(x_test)
    probabilities = model.predict_proba(x_test)[:, 1]

    accuracy = accuracy_score(y_test, predictions)
    precision = precision_score(y_test, predictions, zero_division=0)
    recall = recall_score(y_test, predictions, zero_division=0)
    auc = roc_auc_score(y_test, probabilities)

    print(f"eval_accuracy={accuracy:.4f}", flush=True)
    print(f"eval_precision={precision:.4f}", flush=True)
    print(f"eval_recall={recall:.4f}", flush=True)
    print(f"eval_auc={auc:.4f}", flush=True)

    importances = model.feature_importances_
    ranked = sorted(
        zip(FEATURE_NAMES, importances),
        key=lambda item: float(item[1]),
        reverse=True,
    )[:10]
    print("top_features=" + json.dumps([[name, float(score)] for name, score in ranked]), flush=True)


def train_and_save_estimator(args: argparse.Namespace):
    kaggle_texts, kaggle_labels = load_kaggle_dataset(args.kaggle_csv.resolve())
    kaggle_features = vectorize_texts(kaggle_texts)
    kaggle_labels_array = np.asarray(kaggle_labels, dtype=np.int32)

    print(
        f"kaggle_rows={len(kaggle_texts)} phishing={int(kaggle_labels_array.sum())} "
        f"legitimate={int(len(kaggle_labels_array) - kaggle_labels_array.sum())} features={len(FEATURE_NAMES)}",
        flush=True,
    )
    if not args.skip_eval:
        report_eval(args, kaggle_features, kaggle_labels_array)

    model = build_estimator(args.random_state)
    model.fit(kaggle_features, kaggle_labels_array)
    model_path = args.model_path.resolve()
    model_path.parent.mkdir(parents=True, exist_ok=True)
    with model_path.open("wb") as handle:
        pickle.dump(model, handle)
    print(f"saved_model: {model_path}", flush=True)
    return model


def load_saved_estimator(model_path: Path):
    import_xgboost_module()
    with model_path.open("rb") as handle:
        return pickle.load(handle)


def import_xgboost_module():
    current_dir = SCRIPT_DIR.resolve()
    cwd = Path.cwd().resolve()
    original_sys_path = list(sys.path)
    sys.modules.pop("xgboost", None)
    try:
        sys.path = [
            entry
            for entry in sys.path
            if Path(entry or ".").resolve() not in {current_dir, cwd}
        ]
        return importlib.import_module("xgboost")
    finally:
        sys.path = original_sys_path


def load_input_rows(args: argparse.Namespace) -> list[dict[str, str]]:
    csv.field_size_limit(sys.maxsize)
    input_csv = args.input_csv.resolve()
    if not input_csv.exists():
        raise SystemExit(f"Input CSV not found: {input_csv}")

    rows: list[dict[str, str]] = []
    with input_csv.open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        if not reader.fieldnames:
            raise SystemExit(f"CSV has no header row: {input_csv}")

        required = [args.subject_column, args.body_column, args.label_column]
        missing = [name for name in required if name not in reader.fieldnames]
        if missing:
            raise SystemExit(f"CSV missing required columns {', '.join(missing)} in {input_csv}")

        for index, record in enumerate(reader, start=1):
            if args.sample_size > 0 and index > args.sample_size:
                break
            subject = normalize_text(record.get(args.subject_column, ""))
            body = normalize_text(record.get(args.body_column, ""))
            rows.append(
                {
                    "subject": subject,
                    "body": body,
                    "label": str(record.get(args.label_column, "")),
                    "_content": build_content(subject, body),
                }
            )
    if not rows:
        raise SystemExit(f"No rows loaded from {input_csv}")
    return rows


def main() -> int:
    args = parse_args()
    model_path = args.model_path.resolve()
    if model_path.exists() and not args.force_retrain:
        model = load_saved_estimator(model_path)
        print(f"loaded_model: {model_path}", flush=True)
    else:
        model = train_and_save_estimator(args)

    if args.train_only:
        return 0

    input_rows = load_input_rows(args)
    input_texts = [row["_content"] for row in input_rows]
    input_features = vectorize_texts(input_texts)
    predictions = model.predict(input_features).tolist()

    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    output_csv = output_dir / f"{args.input_csv.resolve().stem}_results.csv"

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
