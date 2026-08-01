#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import math
import re
from collections import Counter
from functools import lru_cache
from pathlib import Path
import sys
from typing import Callable

import numpy as np
import pandas as pd
import nltk
from nltk.corpus import words as nltk_words
from nltk.corpus import wordnet as wn
from nltk.tokenize import sent_tokenize

VIS_ROOT = Path(__file__).resolve().parents[1]
if str(VIS_ROOT) not in sys.path:
    sys.path.insert(0, str(VIS_ROOT))
MIXED_INPUT = VIS_ROOT / "test" / "overview" / "projected_points_mixed_overview.csv"
OUTPUT_ROOT = Path(__file__).resolve().parent / "overview"

from persuasion_strategy_model.src.email_preprocessing import (
    FUNCTION_WORDS,
    SPLIT_SENTENCE_RE,
    ensure_nltk_resources,
    normalize_raw_text,
    sentence_is_noise,
    tokenize_sentence,
)

PRINCIPLE_LABELS = [
    "Authority",
    "Liking",
    "Reciprocity",
    "Social Proof",
    "Scarcity",
    "Commitment",
]
PRINCIPLE_COLUMNS = [
    "principle_authority",
    "principle_liking",
    "principle_reciprocity",
    "principle_social_proof",
    "principle_scarcity",
    "principle_commitment",
]
GROUP_ORDER = [
    "HW-P-FN",
    "HW-P-TP",
    "HW-B-TN",
    "HW-B-FP",
    "LLM-P-FN",
    "LLM-P-TP",
    "LLM-B-TN",
    "LLM-B-FP",
]
PAIR_INDEX = [(i, j) for i in range(len(PRINCIPLE_LABELS)) for j in range(i, len(PRINCIPLE_LABELS))]

TERM_WHITELIST = {
    "account",
    "accounting",
    "access",
    "airtable",
    "analytics",
    "antivirus",
    "architect",
    "audit",
    "audits",
    "asynchronous",
    "bank",
    "banking",
    "botnet",
    "breach",
    "breaches",
    "buyer",
    "campaign",
    "cloud",
    "compliance",
    "compiler",
    "compatibility",
    "configuration",
    "configurations",
    "compromise",
    "crash",
    "convincing",
    "credential",
    "credentials",
    "data",
    "dashboard",
    "department",
    "dependencies",
    "details",
    "deployment",
    "digital",
    "discretion",
    "encryption",
    "endpoint",
    "endpoints",
    "entice",
    "error",
    "followup",
    "followups",
    "funds",
    "firebase",
    "gift",
    "hipaa",
    "hubspot",
    "incident",
    "incidents",
    "incentive",
    "integration",
    "integrations",
    "invoice",
    "jargon",
    "latency",
    "lead",
    "leads",
    "link",
    "login",
    "marketing",
    "medical",
    "metric",
    "metrics",
    "monitoring",
    "opportunities",
    "order",
    "patient",
    "payment",
    "payroll",
    "perl5",
    "phishing",
    "pipeline",
    "project",
    "procurement",
    "promotional",
    "purchase",
    "postfix",
    "request",
    "reassure",
    "repository",
    "response",
    "responses",
    "redis",
    "root",
    "routing",
    "sales",
    "saas",
    "security",
    "seller",
    "shipment",
    "social",
    "solution",
    "solutions",
    "specialist",
    "strong",
    "structure",
    "survey",
    "support",
    "technical",
    "threat",
    "threats",
    "transfer",
    "unauthorized",
    "unusual",
    "urgent",
    "verify",
    "vpn",
    "warranty",
    "website",
}

TOKEN_BLACKLIST = {
    "aetheros",
    "accountank",
    "accounto",
    "amize",
    "anchordesk",
    "anmost",
    "api",
    "apis",
    "apolog",
    "asly",
    "bec",
    "brooks",
    "banksecurityupdate",
    "canon",
    "capitalone",
    "cen",
    "chen",
    "cios",
    "cnet",
    "com",
    "contact",
    "customer",
    "dev",
    "email",
    "emily",
    "emojis",
    "enthusiastic",
    "evelyn",
    "extraterrestrial",
    "facil",
    "finds",
    "fucking",
    "free",
    "full",
    "fungal",
    "gent",
    "hope",
    "imac",
    "information",
    "instanding",
    "iso17799",
    "jackson",
    "jensen",
    "jordan",
    "keen",
    "knee",
    "lawn",
    "leopard",
    "lily",
    "lvm",
    "mash",
    "maya",
    "mercury",
    "mold",
    "motley",
    "name",
    "nbest",
    "news",
    "nobodydear",
    "nobodyject",
    "numbery",
    "number",
    "nthank",
    "opportunityment",
    "paypal",
    "palm",
    "planets",
    "position",
    "powershot",
    "professionalism",
    "pyknic",
    "qqqqqqqqqq-zdnet",
    "quired",
    "read",
    "regent",
    "report",
    "requiresands",
    "resume",
    "sa-learn",
    "samsung",
    "sarm",
    "secretaryo",
    "securityshield",
    "sharinging",
    "shippingcompany",
    "shit",
    "sleep",
    "sofia",
    "solaris",
    "story",
    "suse",
    "swing",
    "syncmaster",
    "team",
    "techcorp",
    "tiktok",
    "toshiba",
    "trec",
    "turner",
    "update",
    "usaa",
    "well",
    "wells",
    "write",
    "writing",
    "xfs",
    "youro's",
    "zdnet",
    "clickup",
    "companyemed",
    "commitation",
    "deck",
    "disation",
    "disiality",
    "disism",
    "dorain",
    "dispatch",
    "dreamsleep",
    "gravity",
    "housing",
    "hylafax",
    "initi",
    "ipod",
    "macworld",
    "notebooks",
    "overgrowth",
    "persists",
    "planet",
    "pleasely",
    "scsidisks",
    "societies",
    "spins",
    "sirnameary",
    "sirreary",
    "susp",
    "themost",
    "whatsoever",
    "vista",
    "mayajensen",
    "ipods",
    "builder",
    "masher",
    "promptwavering",
    "pleaseictly",
    "ogether",
    "rece",
    "transferate",
    "ment",
    "youro",
    "requestquired",
    "supp",
    "anderson",
    "informationification",
    "comunsubscribe",
    "farber",
    "quilt",
    "crochet",
    "ebusiness",
}

PHRASE_BLACKLIST_SUBSTRINGS = {
    "account number",
    "address subscription",
    "appreciate request",
    "banksecurityupdate",
    "capitalone",
    "canon powershot",
    "contact information",
    "copyright networks",
    "cloud solutions architect",
    "customer support",
    "email finds",
    "employees about",
    "fool stock advisor",
    "free spins",
    "full story",
    "greatly appreciate",
    "hesitate contact",
    "hope email",
    "looking forward response",
    "next body",
    "name company",
    "name hope",
    "name position",
    "networks rights",
    "news com",
    "protect their",
    "questions concerns",
    "read full",
    "sharinging matter",
    "should friendly",
    "subscription example",
    "support contacting",
    "techcorp",
    "transfer transfer",
    "unsubscribe manage",
    "would like know",
    "would request",
    "write email",
    "about importance",
    "accountank bank",
    "brown cloud",
    "ebusiness security",
    "forward prompt response",
    "funds complete",
    "funds over",
    "funds over account",
    "issue digital",
    "like security alert",
    "money gift",
    "phishing urging accounting",
    "phishing urging",
    "phishing convincing",
    "provide solution",
    "regarding urgent",
    "require request",
    "request requires immediate",
    "secretary phishing",
    "security networking applications",
    "solution earliest",
    "strong unique",
    "something like security",
    "subject security",
    "their security",
    "again payment",
    "transaction request",
    "transfer some",
    "antivirus subscription",
    "subject urgent confidential",
    "website senior",
}

GENERIC_TOKENS = {
    "account",
    "best",
    "company",
    "contact",
    "customer",
    "email",
    "full",
    "help",
    "hope",
    "immediate",
    "information",
    "message",
    "name",
    "number",
    "please",
    "regards",
    "report",
    "story",
    "support",
    "team",
    "thanks",
    "well",
    "write",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build overall group evidence tables from the mixed overview dataset."
    )
    parser.add_argument("--top-docs", type=int, default=400)
    parser.add_argument("--top-words", type=int, default=15)
    parser.add_argument("--top-phrases", type=int, default=15)
    parser.add_argument("--top-sentences", type=int, default=10)
    return parser.parse_args()


def ensure_lexicon_resources() -> None:
    for pkg in ("words", "wordnet", "omw-1.4"):
        try:
            nltk.data.find(f"corpora/{pkg}")
        except LookupError:
            nltk.download(pkg, quiet=True)


@lru_cache(maxsize=1)
def english_lexicon() -> set[str]:
    ensure_lexicon_resources()
    lexicon = {word.lower() for word in nltk_words.words()}
    lexicon.update(lemma.lower().replace("_", " ") for lemma in wn.all_lemma_names())
    return lexicon


@lru_cache(maxsize=20000)
def normalize_term(term: str) -> str:
    return re.sub(r"[^a-z]+", " ", str(term).lower()).strip()


@lru_cache(maxsize=20000)
def token_is_meaningful(token: str) -> bool:
    token = normalize_term(token)
    if not token or " " in token:
        return False
    if token in TERM_WHITELIST:
        return True
    if token in TOKEN_BLACKLIST or token in FUNCTION_WORDS:
        return False
    if len(token) < 4:
        return False
    if not token.isalpha():
        return False
    if re.search(r"(.)\1\1", token):
        return False
    if not any(ch in "aeiou" for ch in token):
        return False
    if token.endswith(("corp", "tech", "company")):
        return False
    lexicon = english_lexicon()
    if token in lexicon:
        return True
    if token.endswith("s") and token[:-1] in lexicon:
        return True
    if token.endswith("ing") and token[:-3] in lexicon:
        return True
    if token.endswith("ed") and token[:-2] in lexicon:
        return True
    if token.endswith("ly") and token[:-2] in lexicon:
        return True
    return False


@lru_cache(maxsize=50000)
def phrase_is_meaningful(phrase: str) -> bool:
    normalized = normalize_term(phrase)
    if not normalized:
        return False
    if any(bad in normalized for bad in PHRASE_BLACKLIST_SUBSTRINGS):
        return False
    tokens = normalized.split()
    if len(tokens) < 2 or len(tokens) > 3:
        return False
    if len(set(tokens)) < len(tokens):
        return False
    if any(not token_is_meaningful(token) for token in tokens):
        return False
    non_generic = [token for token in tokens if token not in GENERIC_TOKENS]
    if len(non_generic) < 2:
        return False
    if sum(token in TERM_WHITELIST for token in tokens) < 1:
        return False
    return True


def is_redundant_term(term: str, selected_terms: list[str]) -> bool:
    current = set(normalize_term(term).split())
    if not current:
        return True
    for existing in selected_terms:
        existing_tokens = set(normalize_term(existing).split())
        if not existing_tokens:
            continue
        if current == existing_tokens:
            return True
        overlap = len(current & existing_tokens) / max(1, min(len(current), len(existing_tokens)))
        if overlap >= 0.8:
            return True
        if normalize_term(term) in normalize_term(existing) or normalize_term(existing) in normalize_term(term):
            return True
    return False


def aggregate_doc_presence(docs: pd.DataFrame, field: str) -> Counter[str]:
    counter: Counter[str] = Counter()
    for record in docs[field]:
        for term in record.keys():
            counter[term] += 1
    return counter


def iter_clean_sentences(subject: object, body: object, *, max_total_words: int = 3000) -> list[tuple[str, list[str]]]:
    ensure_nltk_resources()
    total_words = 0
    results: list[tuple[str, list[str]]] = []
    pieces = []
    subject_text = normalize_raw_text(subject)
    body_text = normalize_raw_text(body)
    if subject_text:
        pieces.append(subject_text)
    if body_text:
        pieces.append(body_text)

    for piece in pieces:
        candidate_sentences: list[str] = []
        for block in SPLIT_SENTENCE_RE.split(piece):
            block = block.strip()
            if not block:
                continue
            try:
                candidate_sentences.extend(sent_tokenize(block))
            except LookupError:
                ensure_nltk_resources()
                candidate_sentences.extend(sent_tokenize(block))

        for sentence in candidate_sentences:
            sentence = sentence.strip()
            if not sentence:
                continue
            tokens = tokenize_sentence(sentence)
            if sentence_is_noise(sentence, tokens):
                continue
            remaining_budget = max_total_words - total_words
            if remaining_budget <= 0:
                return results
            if len(tokens) > remaining_budget:
                tokens = tokens[:remaining_budget]
            if not tokens:
                continue
            visible_sentence = " ".join(tokens)
            results.append((visible_sentence, tokens))
            total_words += len(tokens)
            if total_words >= max_total_words:
                return results
    return results


def build_group_label(frame: pd.DataFrame) -> pd.Series:
    phishing = frame["raw_label"].astype(int).eq(1)
    positive_pred = frame["pred_label"].astype(int).eq(1)
    label_part = np.where(phishing, "P", "B")
    outcome_part = np.where(
        phishing & ~positive_pred,
        "FN",
        np.where(
            phishing & positive_pred,
            "TP",
            np.where(~phishing & ~positive_pred, "TN", "FP"),
        ),
    )
    return frame["source"].astype(str) + "-" + label_part + "-" + outcome_part


def add_document_features(frame: pd.DataFrame) -> pd.DataFrame:
    records = []
    for row in frame.itertuples(index=False):
        sentences = iter_clean_sentences(getattr(row, "subject", ""), getattr(row, "body", ""))
        doc_tokens = [token for _, tokens in sentences for token in tokens]
        unigram_counts = Counter(token for token in doc_tokens if token_is_meaningful(token))
        bigram_counts: Counter[str] = Counter()
        trigram_counts: Counter[str] = Counter()
        for _, tokens in sentences:
            filtered = [token for token in tokens if token_is_meaningful(token)]
            for index in range(len(filtered) - 1):
                phrase = " ".join(filtered[index : index + 2])
                if phrase_is_meaningful(phrase):
                    bigram_counts[phrase] += 1
            for index in range(len(filtered) - 2):
                phrase = " ".join(filtered[index : index + 3])
                if phrase_is_meaningful(phrase):
                    trigram_counts[phrase] += 1
        records.append(
            {
                "sentences_clean": [sentence for sentence, _ in sentences],
                "sentence_tokens": [tokens for _, tokens in sentences],
                "unigram_counts": unigram_counts,
                "bigram_counts": bigram_counts,
                "trigram_counts": trigram_counts,
            }
        )

    enriched = frame.reset_index(drop=True).copy()
    extra = pd.DataFrame.from_records(records)
    for column in extra.columns:
        enriched[column] = extra[column]
    return enriched


def compute_cell_contribution(frame: pd.DataFrame, row_index: int, col_index: int) -> np.ndarray:
    first = frame[PRINCIPLE_COLUMNS[row_index]].to_numpy(dtype=float)
    second = frame[PRINCIPLE_COLUMNS[col_index]].to_numpy(dtype=float)
    if row_index == col_index:
        return first
    return first * second


def top_indices(values: np.ndarray, k: int) -> np.ndarray:
    if len(values) == 0:
        return np.array([], dtype=int)
    k = min(k, len(values))
    order = np.argsort(values)
    return order[-k:][::-1]


def weighted_log_ratio(
    group_counter: Counter[str],
    bg_counter: Counter[str],
    *,
    top_n: int,
    min_group_count: float = 2.0,
    min_group_docs: int = 2,
    min_score: float = 0.0,
    filter_fn: Callable[[str], bool] | None = None,
    dedupe: bool = True,
    group_doc_counter: Counter[str] | None = None,
    bg_doc_counter: Counter[str] | None = None,
) -> list[tuple[str, float, float, float, int, int]]:
    vocabulary = set(group_counter) | set(bg_counter)
    total_group = float(sum(group_counter.values()))
    total_bg = float(sum(bg_counter.values()))
    alpha = 0.01
    vocab_size = max(1, len(vocabulary))
    rows = []
    for term in vocabulary:
        g = float(group_counter.get(term, 0.0))
        b = float(bg_counter.get(term, 0.0))
        if g < min_group_count:
            continue
        if group_doc_counter is not None and int(group_doc_counter.get(term, 0)) < min_group_docs:
            continue
        if filter_fn is not None and not filter_fn(term):
            continue
        score = math.log((g + alpha) / (total_group + alpha * vocab_size)) - math.log(
            (b + alpha) / (total_bg + alpha * vocab_size)
        )
        if score < min_score:
            continue
        rows.append((term, score, g, b, int(group_doc_counter.get(term, 0)) if group_doc_counter else 0, int(bg_doc_counter.get(term, 0)) if bg_doc_counter else 0))
    rows.sort(key=lambda item: (item[1], item[2]), reverse=True)
    if not dedupe:
        return rows[:top_n]

    selected: list[tuple[str, float, float, float, int, int]] = []
    selected_terms: list[str] = []
    for row in rows:
        if is_redundant_term(row[0], selected_terms):
            continue
        selected.append(row)
        selected_terms.append(row[0])
        if len(selected) >= top_n:
            break
    return selected


def aggregate_weighted_terms(
    docs: pd.DataFrame,
    weights: np.ndarray,
    field: str,
) -> Counter[str]:
    counter: Counter[str] = Counter()
    for record, weight in zip(docs[field], weights, strict=True):
        if weight <= 0:
            continue
        for term, count in record.items():
            counter[term] += float(count) * float(weight)
    return counter


def build_sentence_scores(
    docs: pd.DataFrame,
    weights: np.ndarray,
    phrase_score_lookup: dict[str, float],
    *,
    top_n: int,
) -> list[tuple[str, float, int]]:
    candidates: list[tuple[str, float, int]] = []
    for doc_row, weight in zip(docs.itertuples(index=False), weights, strict=True):
        if weight <= 0:
            continue
        seen_sentences: set[str] = set()
        for sentence, tokens in zip(doc_row.sentences_clean, doc_row.sentence_tokens, strict=True):
            filtered = [token for token in tokens if token_is_meaningful(token)]
            if len(filtered) < 3:
                continue
            terms = set(filtered)
            for index in range(len(filtered) - 1):
                phrase = " ".join(filtered[index : index + 2])
                if phrase_is_meaningful(phrase):
                    terms.add(phrase)
            for index in range(len(filtered) - 2):
                phrase = " ".join(filtered[index : index + 3])
                if phrase_is_meaningful(phrase):
                    terms.add(phrase)
            lexical_score = sum(phrase_score_lookup.get(term, 0.0) for term in terms)
            if lexical_score <= 0:
                continue
            normalized_sentence = " ".join(filtered[:60])
            if normalized_sentence in seen_sentences:
                continue
            seen_sentences.add(normalized_sentence)
            candidates.append((normalized_sentence, float(weight) * lexical_score, len(filtered)))
    candidates.sort(key=lambda item: item[1], reverse=True)
    unique_rows: list[tuple[str, float, int]] = []
    seen = set()
    for sentence, score, token_len in candidates:
        if sentence in seen:
            continue
        seen.add(sentence)
        unique_rows.append((sentence, score, token_len))
        if len(unique_rows) >= top_n:
            break
    return unique_rows


def group_title(group_label: str) -> str:
    source, truth, outcome = group_label.split("-")
    truth_text = "Phishing" if truth == "P" else "Benign"
    return f"{source} {truth_text} {outcome}"


def main() -> None:
    args = parse_args()
    OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)

    frame = pd.read_csv(MIXED_INPUT)
    frame = frame.copy()
    frame["group_label"] = build_group_label(frame)
    frame = add_document_features(frame)

    registry_rows = []
    for group_label in GROUP_ORDER:
        group_frame = frame[frame["group_label"] == group_label].copy().reset_index(drop=True)
        if group_frame.empty:
            continue
        background_frame = frame[frame["group_label"] != group_label].copy().reset_index(drop=True)
        group_dir = OUTPUT_ROOT / group_label
        group_dir.mkdir(parents=True, exist_ok=True)

        cell_rows = []
        word_rows = []
        phrase_rows = []
        sentence_rows = []

        for row_index, col_index in PAIR_INDEX:
            feature_a = PRINCIPLE_LABELS[row_index]
            feature_b = PRINCIPLE_LABELS[col_index]
            cell_slug = f"{feature_a.lower().replace(' ', '_')}__{feature_b.lower().replace(' ', '_')}"

            group_contrib = compute_cell_contribution(group_frame, row_index, col_index)
            bg_contrib = compute_cell_contribution(background_frame, row_index, col_index)

            top_group_idx = top_indices(group_contrib, args.top_docs)
            top_bg_idx = top_indices(bg_contrib, args.top_docs)
            top_group_docs = group_frame.iloc[top_group_idx].reset_index(drop=True)
            top_bg_docs = background_frame.iloc[top_bg_idx].reset_index(drop=True)
            top_group_weights = group_contrib[top_group_idx] if len(top_group_idx) else np.array([], dtype=float)
            top_bg_weights = bg_contrib[top_bg_idx] if len(top_bg_idx) else np.array([], dtype=float)

            group_word_counter = aggregate_weighted_terms(top_group_docs, top_group_weights, "unigram_counts")
            bg_word_counter = aggregate_weighted_terms(top_bg_docs, top_bg_weights, "unigram_counts")
            group_bigram_counter = aggregate_weighted_terms(top_group_docs, top_group_weights, "bigram_counts")
            bg_bigram_counter = aggregate_weighted_terms(top_bg_docs, top_bg_weights, "bigram_counts")
            group_trigram_counter = aggregate_weighted_terms(top_group_docs, top_group_weights, "trigram_counts")
            bg_trigram_counter = aggregate_weighted_terms(top_bg_docs, top_bg_weights, "trigram_counts")
            group_word_docs = aggregate_doc_presence(top_group_docs, "unigram_counts")
            bg_word_docs = aggregate_doc_presence(top_bg_docs, "unigram_counts")
            group_phrase_docs = aggregate_doc_presence(top_group_docs, "bigram_counts") + aggregate_doc_presence(top_group_docs, "trigram_counts")
            bg_phrase_docs = aggregate_doc_presence(top_bg_docs, "bigram_counts") + aggregate_doc_presence(top_bg_docs, "trigram_counts")

            top_words = weighted_log_ratio(
                group_word_counter,
                bg_word_counter,
                top_n=args.top_words,
                min_group_count=2.5,
                min_group_docs=3,
                min_score=0.2,
                filter_fn=token_is_meaningful,
                group_doc_counter=group_word_docs,
                bg_doc_counter=bg_word_docs,
            )
            phrase_candidates = weighted_log_ratio(
                group_bigram_counter + group_trigram_counter,
                bg_bigram_counter + bg_trigram_counter,
                top_n=args.top_phrases,
                min_group_count=1.5,
                min_group_docs=2,
                min_score=0.2,
                filter_fn=phrase_is_meaningful,
                group_doc_counter=group_phrase_docs,
                bg_doc_counter=bg_phrase_docs,
            )

            phrase_score_lookup = {term: score for term, score, *_ in top_words + phrase_candidates}
            top_sentences = build_sentence_scores(
                top_group_docs,
                top_group_weights,
                phrase_score_lookup,
                top_n=args.top_sentences,
            )

            cell_rows.append(
                {
                    "feature_a": feature_a,
                    "feature_b": feature_b,
                    "cell_slug": cell_slug,
                    "contribution_type": "diagonal" if row_index == col_index else "cooccurrence",
                    "group_size": int(len(group_frame)),
                    "background_size": int(len(background_frame)),
                    "mean_group_contribution": float(np.mean(group_contrib)) if len(group_contrib) else 0.0,
                    "mean_background_contribution": float(np.mean(bg_contrib)) if len(bg_contrib) else 0.0,
                    "max_group_contribution": float(np.max(group_contrib)) if len(group_contrib) else 0.0,
                    "top_doc_count_used": int(len(top_group_docs)),
                }
            )

            for rank, (term, score, group_weight, bg_weight, group_docs, bg_docs) in enumerate(top_words, start=1):
                word_rows.append(
                    {
                        "group_label": group_label,
                        "feature_a": feature_a,
                        "feature_b": feature_b,
                        "cell_slug": cell_slug,
                        "rank": rank,
                        "word": term,
                        "log_ratio_score": score,
                        "weighted_group_count": group_weight,
                        "weighted_background_count": bg_weight,
                        "group_doc_count": group_docs,
                        "background_doc_count": bg_docs,
                    }
                )

            for rank, (term, score, group_weight, bg_weight, group_docs, bg_docs) in enumerate(phrase_candidates, start=1):
                phrase_rows.append(
                    {
                        "group_label": group_label,
                        "feature_a": feature_a,
                        "feature_b": feature_b,
                        "cell_slug": cell_slug,
                        "rank": rank,
                        "phrase": term,
                        "phrase_len": len(term.split()),
                        "log_ratio_score": score,
                        "weighted_group_count": group_weight,
                        "weighted_background_count": bg_weight,
                        "group_doc_count": group_docs,
                        "background_doc_count": bg_docs,
                    }
                )

            for rank, (sentence, score, token_len) in enumerate(top_sentences, start=1):
                sentence_rows.append(
                    {
                        "group_label": group_label,
                        "feature_a": feature_a,
                        "feature_b": feature_b,
                        "cell_slug": cell_slug,
                        "rank": rank,
                        "sentence": sentence,
                        "sentence_score": score,
                        "token_count": token_len,
                    }
                )

        pd.DataFrame(cell_rows).to_csv(group_dir / "cell_summary.csv", index=False)
        pd.DataFrame(word_rows).to_csv(group_dir / "top_words_by_cell.csv", index=False)
        pd.DataFrame(phrase_rows).to_csv(group_dir / "top_phrases_by_cell.csv", index=False)
        pd.DataFrame(sentence_rows).to_csv(group_dir / "top_sentences_by_cell.csv", index=False)

        group_summary = {
            "group_label": group_label,
            "group_title": group_title(group_label),
            "row_count": int(len(group_frame)),
            "files": {
                "cell_summary": "cell_summary.csv",
                "top_words_by_cell": "top_words_by_cell.csv",
                "top_phrases_by_cell": "top_phrases_by_cell.csv",
                "top_sentences_by_cell": "top_sentences_by_cell.csv",
            },
        }
        (group_dir / "group_summary.json").write_text(json.dumps(group_summary, indent=2), encoding="utf-8")
        registry_rows.append({"group_label": group_label, "group_title": group_title(group_label), "row_count": int(len(group_frame))})

    pd.DataFrame(registry_rows).to_csv(OUTPUT_ROOT / "group_registry.csv", index=False)


if __name__ == "__main__":
    main()
