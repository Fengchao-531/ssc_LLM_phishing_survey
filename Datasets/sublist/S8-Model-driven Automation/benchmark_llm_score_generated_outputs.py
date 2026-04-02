#!/usr/bin/env python3
"""Score previously generated S8 outputs from a generation CSV."""

import argparse
import csv
import math
import re
import sys
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, Iterator, List, Optional, Sequence, Tuple


csv.field_size_limit(10**9)

WORD_RE = re.compile(r"[A-Za-z0-9_]+")
DEFAULT_PERPLEXITY_MODEL = "gpt2"
DEFAULT_TOPIC_COUNT = 5
DEFAULT_TOPIC_TOP_WORDS = 10

_PERPLEXITY_SCORER: Dict[str, Any] = {}


@dataclass
class GenerationRow:
    join_key: str
    input_row_number: int
    reference_row_number: int
    model_alias: str
    company: str
    family: str
    access_type: str
    provider: str
    prompt_text: str
    generated_text: str
    reference_text: str
    created_at: str


@dataclass
class ScoredRow:
    generation: GenerationRow
    bleu: float
    rouge1_precision: float
    rouge1_recall: float
    rouge1_f1: float
    generated_perplexity: float
    reference_perplexity: float


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Score an existing generated_outputs.csv file."
    )
    parser.add_argument("--input", required=True, help="Path to generated_outputs.csv")
    parser.add_argument(
        "--output",
        default="",
        help="Optional output CSV path. Defaults to benchmark_results.csv next to the input file.",
    )
    parser.add_argument(
        "--save-every",
        type=int,
        default=1,
        help="Checkpoint scored rows every N completed rows.",
    )
    parser.add_argument(
        "--print-every",
        type=int,
        default=1,
        help="Progress logging interval in completed rows.",
    )
    parser.add_argument(
        "--disable-perplexity",
        action="store_true",
        help="Skip perplexity scoring.",
    )
    parser.add_argument(
        "--perplexity-model",
        default=DEFAULT_PERPLEXITY_MODEL,
        help="Local scorer model used for perplexity, e.g. gpt2.",
    )
    parser.add_argument(
        "--disable-topic-coherence",
        action="store_true",
        help="Skip topic coherence scoring.",
    )
    parser.add_argument(
        "--topic-count",
        type=int,
        default=DEFAULT_TOPIC_COUNT,
        help="Number of topics used for corpus-level topic coherence.",
    )
    parser.add_argument(
        "--topic-top-words",
        type=int,
        default=DEFAULT_TOPIC_TOP_WORDS,
        help="Top words per topic used for topic coherence.",
    )
    return parser.parse_args()


def validate_args(args: argparse.Namespace) -> None:
    input_path = Path(args.input).resolve()
    if not input_path.exists():
        raise FileNotFoundError(f"Generated CSV not found: {input_path}")
    if args.save_every <= 0:
        raise ValueError("--save-every must be greater than 0.")
    if args.print_every <= 0:
        raise ValueError("--print-every must be greater than 0.")
    if args.topic_count <= 0:
        raise ValueError("--topic-count must be greater than 0.")
    if args.topic_top_words <= 1:
        raise ValueError("--topic-top-words must be greater than 1.")


def normalize_text(value: str) -> str:
    text = (value or "").replace("\r\n", "\n").replace("\r", "\n")
    text = text.replace("\ufeff", "").replace("\u200b", "").replace("\xa0", " ")
    text = re.sub(r"[ \t]+\n", "\n", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def tokenize_words(text: str) -> List[str]:
    return [token.lower() for token in WORD_RE.findall(text or "")]


def iter_csv_rows(path: Path) -> Iterator[Tuple[int, Dict[str, str]]]:
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        if not reader.fieldnames:
            raise ValueError(f"CSV has no header row: {path}")
        for row_number, row in enumerate(reader, start=2):
            yield row_number, row


def safe_float(value: Optional[float]) -> str:
    if value is None:
        return ""
    if isinstance(value, float) and (math.isnan(value) or math.isinf(value)):
        return ""
    return f"{value:.6f}"


def load_generation_rows(path: Path) -> List[GenerationRow]:
    rows: List[GenerationRow] = []
    for _, row in iter_csv_rows(path):
        rows.append(
            GenerationRow(
                join_key=str(row.get("join_key", "")),
                input_row_number=int(row.get("input_row_number", "0") or 0),
                reference_row_number=int(row.get("reference_row_number", "0") or 0),
                model_alias=str(row.get("model_alias", "")),
                company=str(row.get("company", "")),
                family=str(row.get("family", "")),
                access_type=str(row.get("access_type", "")),
                provider=str(row.get("provider", "")),
                prompt_text=normalize_text(str(row.get("prompt_text", ""))),
                generated_text=normalize_text(str(row.get("generated_text", ""))),
                reference_text=normalize_text(str(row.get("reference_text", ""))),
                created_at=str(row.get("created_at", "")),
            )
        )
    if not rows:
        raise ValueError(f"No generation rows found in {path}")
    return rows


def ngrams(tokens: Sequence[str], n: int) -> Counter:
    if n <= 0 or len(tokens) < n:
        return Counter()
    return Counter(tuple(tokens[index : index + n]) for index in range(len(tokens) - n + 1))


def compute_bleu(reference_text: str, candidate_text: str, max_n: int = 4) -> float:
    reference_tokens = tokenize_words(reference_text)
    candidate_tokens = tokenize_words(candidate_text)
    if not reference_tokens or not candidate_tokens:
        return 0.0

    precisions: List[float] = []
    for n in range(1, max_n + 1):
        ref_ngrams = ngrams(reference_tokens, n)
        cand_ngrams = ngrams(candidate_tokens, n)
        if not cand_ngrams:
            precisions.append(0.0)
            continue
        overlap = sum(min(count, ref_ngrams[gram]) for gram, count in cand_ngrams.items())
        precision = (overlap + 1.0) / (sum(cand_ngrams.values()) + 1.0)
        precisions.append(precision)

    if min(precisions) <= 0.0:
        geo_mean = 0.0
    else:
        geo_mean = math.exp(sum(math.log(value) for value in precisions) / max_n)

    ref_len = len(reference_tokens)
    cand_len = len(candidate_tokens)
    if cand_len == 0:
        return 0.0
    if cand_len > ref_len:
        brevity_penalty = 1.0
    else:
        brevity_penalty = math.exp(1.0 - (ref_len / cand_len))
    return 100.0 * brevity_penalty * geo_mean


def compute_rouge1(reference_text: str, candidate_text: str) -> Tuple[float, float, float]:
    reference_tokens = tokenize_words(reference_text)
    candidate_tokens = tokenize_words(candidate_text)
    if not reference_tokens or not candidate_tokens:
        return 0.0, 0.0, 0.0

    reference_counts = Counter(reference_tokens)
    candidate_counts = Counter(candidate_tokens)
    overlap = sum(min(count, candidate_counts[token]) for token, count in reference_counts.items())
    precision = overlap / max(1, sum(candidate_counts.values()))
    recall = overlap / max(1, sum(reference_counts.values()))
    if precision + recall == 0:
        f1 = 0.0
    else:
        f1 = 2.0 * precision * recall / (precision + recall)
    return 100.0 * precision, 100.0 * recall, 100.0 * f1


def get_perplexity_scorer(model_name: str) -> Dict[str, Any]:
    if model_name in _PERPLEXITY_SCORER:
        return _PERPLEXITY_SCORER[model_name]

    try:
        import torch  # type: ignore
        from transformers import AutoModelForCausalLM, AutoTokenizer  # type: ignore
    except ImportError as exc:
        raise RuntimeError(
            "transformers and torch are required for perplexity scoring. "
            "Install them with: python3 -m pip install transformers torch"
        ) from exc

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(model_name)
    model.eval()
    scorer = {"tokenizer": tokenizer, "model": model, "torch": torch}
    _PERPLEXITY_SCORER[model_name] = scorer
    return scorer


def compute_perplexity(text: str, model_name: str) -> float:
    text = normalize_text(text)
    if not text:
        return float("nan")

    scorer = get_perplexity_scorer(model_name)
    tokenizer = scorer["tokenizer"]
    model = scorer["model"]
    torch = scorer["torch"]

    encoded = tokenizer(
        text,
        return_tensors="pt",
        truncation=True,
        max_length=getattr(model.config, "max_position_embeddings", 1024) or 1024,
    )
    with torch.no_grad():
        output = model(**encoded, labels=encoded["input_ids"])
    loss = float(output.loss)
    return float(math.exp(min(loss, 20.0)))


def compute_document_frequencies(tokenized_docs: Sequence[Sequence[str]]) -> Dict[str, int]:
    frequencies: Dict[str, int] = {}
    for doc in tokenized_docs:
        for token in set(doc):
            frequencies[token] = frequencies.get(token, 0) + 1
    return frequencies


def compute_topic_coherence_for_corpus(
    texts: Sequence[str], topic_count: int, top_words: int
) -> float:
    tokenized_docs = [tokenize_words(text) for text in texts if tokenize_words(text)]
    if len(tokenized_docs) < 2:
        return float("nan")

    try:
        from sklearn.decomposition import LatentDirichletAllocation  # type: ignore
        from sklearn.feature_extraction.text import CountVectorizer  # type: ignore
    except ImportError as exc:
        raise RuntimeError(
            "scikit-learn is required for topic coherence scoring. "
            "Install it with: python3 -m pip install scikit-learn"
        ) from exc

    raw_docs = [" ".join(tokens) for tokens in tokenized_docs if tokens]
    vectorizer = CountVectorizer(token_pattern=r"(?u)\b\w+\b", min_df=1)
    matrix = vectorizer.fit_transform(raw_docs)
    vocab = vectorizer.get_feature_names_out()
    if matrix.shape[0] < 2 or matrix.shape[1] < 2:
        return float("nan")

    actual_topics = max(1, min(topic_count, matrix.shape[0], matrix.shape[1]))
    lda = LatentDirichletAllocation(
        n_components=actual_topics,
        random_state=42,
        learning_method="batch",
    )
    lda.fit(matrix)

    doc_freq = compute_document_frequencies(tokenized_docs)
    token_sets = [set(tokens) for tokens in tokenized_docs]

    topic_scores: List[float] = []
    for topic in lda.components_:
        top_indices = topic.argsort()[-top_words:][::-1]
        topic_words = [str(vocab[index]) for index in top_indices]
        pair_scores: List[float] = []
        for right in range(1, len(topic_words)):
            for left in range(right):
                wi = topic_words[left]
                wj = topic_words[right]
                d_wi = doc_freq.get(wi, 0)
                d_wi_wj = sum(1 for token_set in token_sets if wi in token_set and wj in token_set)
                if d_wi <= 0:
                    continue
                pair_scores.append(math.log((d_wi_wj + 1.0) / d_wi))
        if pair_scores:
            topic_scores.append(sum(pair_scores) / len(pair_scores))

    if not topic_scores:
        return float("nan")
    return float(sum(topic_scores) / len(topic_scores))


def average(values: Iterable[float]) -> float:
    cleaned = [value for value in values if not math.isnan(value) and not math.isinf(value)]
    if not cleaned:
        return float("nan")
    return float(sum(cleaned) / len(cleaned))


def build_summary_rows(
    rows: Sequence[ScoredRow], args: argparse.Namespace
) -> List[Dict[str, Any]]:
    by_model: Dict[str, List[ScoredRow]] = {}
    for row in rows:
        by_model.setdefault(row.generation.model_alias, []).append(row)

    summary_rows: List[Dict[str, Any]] = []
    for model_alias, model_rows in by_model.items():
        model_info = model_rows[0].generation
        generated_texts = [row.generation.generated_text for row in model_rows]
        reference_texts = [row.generation.reference_text for row in model_rows]

        if args.disable_topic_coherence:
            generated_topic_coherence = float("nan")
            reference_topic_coherence = float("nan")
        else:
            generated_topic_coherence = compute_topic_coherence_for_corpus(
                generated_texts, args.topic_count, args.topic_top_words
            )
            reference_topic_coherence = compute_topic_coherence_for_corpus(
                reference_texts, args.topic_count, args.topic_top_words
            )

        summary_rows.append(
            {
                "model_alias": model_alias,
                "company": model_info.company,
                "family": model_info.family,
                "access_type": model_info.access_type,
                "provider": model_info.provider,
                "sample_count": len(model_rows),
                "mean_bleu": safe_float(average(row.bleu for row in model_rows)),
                "mean_rouge1_precision": safe_float(
                    average(row.rouge1_precision for row in model_rows)
                ),
                "mean_rouge1_recall": safe_float(
                    average(row.rouge1_recall for row in model_rows)
                ),
                "mean_rouge1_f1": safe_float(average(row.rouge1_f1 for row in model_rows)),
                "mean_generated_perplexity": safe_float(
                    average(row.generated_perplexity for row in model_rows)
                ),
                "mean_reference_perplexity": safe_float(
                    average(row.reference_perplexity for row in model_rows)
                ),
                "topic_coherence_generated": safe_float(generated_topic_coherence),
                "topic_coherence_reference": safe_float(reference_topic_coherence),
                "topic_coherence_gap": safe_float(
                    generated_topic_coherence - reference_topic_coherence
                    if not math.isnan(generated_topic_coherence)
                    and not math.isnan(reference_topic_coherence)
                    else float("nan")
                ),
            }
        )

    summary_rows.sort(key=lambda row: row["model_alias"])
    return summary_rows


def write_scored_csv(
    path: Path, rows: Sequence[ScoredRow], summary_rows: Sequence[Dict[str, Any]]
) -> None:
    summary_by_model = {str(row["model_alias"]): row for row in summary_rows}
    fieldnames = [
        "join_key",
        "input_row_number",
        "reference_row_number",
        "model_alias",
        "company",
        "family",
        "access_type",
        "provider",
        "prompt_text",
        "generated_text",
        "reference_text",
        "bleu",
        "rouge1_precision",
        "rouge1_recall",
        "rouge1_f1",
        "generated_perplexity",
        "reference_perplexity",
        "created_at",
        "sample_count",
        "mean_bleu",
        "mean_rouge1_precision",
        "mean_rouge1_recall",
        "mean_rouge1_f1",
        "mean_generated_perplexity",
        "mean_reference_perplexity",
        "topic_coherence_generated",
        "topic_coherence_reference",
        "topic_coherence_gap",
    ]
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            generation = row.generation
            summary = summary_by_model.get(generation.model_alias, {})
            writer.writerow(
                {
                    "join_key": generation.join_key,
                    "input_row_number": generation.input_row_number,
                    "reference_row_number": generation.reference_row_number,
                    "model_alias": generation.model_alias,
                    "company": generation.company,
                    "family": generation.family,
                    "access_type": generation.access_type,
                    "provider": generation.provider,
                    "prompt_text": generation.prompt_text,
                    "generated_text": generation.generated_text,
                    "reference_text": generation.reference_text,
                    "bleu": safe_float(row.bleu),
                    "rouge1_precision": safe_float(row.rouge1_precision),
                    "rouge1_recall": safe_float(row.rouge1_recall),
                    "rouge1_f1": safe_float(row.rouge1_f1),
                    "generated_perplexity": safe_float(row.generated_perplexity),
                    "reference_perplexity": safe_float(row.reference_perplexity),
                    "created_at": generation.created_at,
                    "sample_count": summary.get("sample_count", ""),
                    "mean_bleu": summary.get("mean_bleu", ""),
                    "mean_rouge1_precision": summary.get("mean_rouge1_precision", ""),
                    "mean_rouge1_recall": summary.get("mean_rouge1_recall", ""),
                    "mean_rouge1_f1": summary.get("mean_rouge1_f1", ""),
                    "mean_generated_perplexity": summary.get(
                        "mean_generated_perplexity", ""
                    ),
                    "mean_reference_perplexity": summary.get(
                        "mean_reference_perplexity", ""
                    ),
                    "topic_coherence_generated": summary.get(
                        "topic_coherence_generated", ""
                    ),
                    "topic_coherence_reference": summary.get(
                        "topic_coherence_reference", ""
                    ),
                    "topic_coherence_gap": summary.get("topic_coherence_gap", ""),
                }
            )


def main() -> int:
    args = parse_args()
    validate_args(args)

    input_path = Path(args.input).resolve()
    output_path = (
        Path(args.output).resolve()
        if args.output
        else input_path.parent / "benchmark_results.csv"
    )

    generations = load_generation_rows(input_path)
    scored_rows: List[ScoredRow] = []
    total = len(generations)

    for index, generation in enumerate(generations, start=1):
        bleu = compute_bleu(generation.reference_text, generation.generated_text)
        rouge_precision, rouge_recall, rouge_f1 = compute_rouge1(
            generation.reference_text, generation.generated_text
        )

        if args.disable_perplexity:
            generated_perplexity = float("nan")
            reference_perplexity = float("nan")
        else:
            generated_perplexity = compute_perplexity(
                generation.generated_text, args.perplexity_model
            )
            reference_perplexity = compute_perplexity(
                generation.reference_text, args.perplexity_model
            )

        scored_rows.append(
            ScoredRow(
                generation=generation,
                bleu=bleu,
                rouge1_precision=rouge_precision,
                rouge1_recall=rouge_recall,
                rouge1_f1=rouge_f1,
                generated_perplexity=generated_perplexity,
                reference_perplexity=reference_perplexity,
            )
        )

        if index % args.save_every == 0:
            summary_rows = build_summary_rows(scored_rows, args)
            write_scored_csv(output_path, scored_rows, summary_rows)

        if index % args.print_every == 0:
            print(
                f"[progress] scored {index}/{total} rows for {generation.model_alias} on sample {generation.join_key}",
                file=sys.stderr,
            )

    summary_rows = build_summary_rows(scored_rows, args)
    write_scored_csv(output_path, scored_rows, summary_rows)
    print(f"Wrote scored benchmark results to {output_path}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception as exc:
        print(f"[error] {exc}", file=sys.stderr)
        raise SystemExit(1)
