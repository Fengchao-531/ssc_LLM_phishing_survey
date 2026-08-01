#!/usr/bin/env python3
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from build_overview_group_evidence import (
    MIXED_INPUT,
    OUTPUT_ROOT,
    add_document_features,
    aggregate_doc_presence,
    aggregate_weighted_terms,
    build_group_label,
    compute_cell_contribution,
    group_title,
    phrase_is_meaningful,
    token_is_meaningful,
    top_indices,
    weighted_log_ratio,
)

EVIDENCE_ROOT = Path(__file__).resolve().parent
OVERVIEW_ROOT = EVIDENCE_ROOT / "overview"
OUTPUT_PATH = OVERVIEW_ROOT / "README.md"
OUTPUT_TABLE = OVERVIEW_ROOT / "curated_overview_evidence_table.csv"
OUTPUT_JSON = OVERVIEW_ROOT / "curated_overview_evidence_table.json"

GROUPS = ["HW-B-TN", "LLM-B-TN", "LLM-P-FN", "LLM-P-TP"]
CELL_SPECS = [
    ("(A, A)", "authority", "authority"),
    ("(A, L)", "authority", "liking"),
    ("(A, R)", "authority", "reciprocity"),
    ("(A, S)", "authority", "social_proof"),
    ("(R, R)", "reciprocity", "reciprocity"),
    ("(L, L)", "liking", "liking"),
    ("(S, S)", "social_proof", "social_proof"),
]
PRINCIPLE_SLUGS = [
    "authority",
    "liking",
    "reciprocity",
    "social_proof",
    "scarcity",
    "commitment",
]
PRINCIPLE_INDEX = {slug: idx for idx, slug in enumerate(PRINCIPLE_SLUGS)}

TOP_DOCS = 700
TOP_WORDS = 8
TOP_PHRASES = 8
TOP_ITEMS_FOR_TABLE = 5


def collect_selected_row_ids(frame: pd.DataFrame) -> dict[tuple[str, str], dict[str, list[int]]]:
    selections: dict[tuple[str, str], dict[str, list[int]]] = {}
    for _, row_slug, col_slug in CELL_SPECS:
        row_idx = PRINCIPLE_INDEX[row_slug]
        col_idx = PRINCIPLE_INDEX[col_slug]
        cell_slug = f"{row_slug}__{col_slug}"
        for group in GROUPS:
            group_frame = frame[frame["group_label"] == group]
            bg_frame = frame[frame["group_label"] != group]
            group_contrib = compute_cell_contribution(group_frame, row_idx, col_idx)
            bg_contrib = compute_cell_contribution(bg_frame, row_idx, col_idx)
            group_ids = group_frame.iloc[top_indices(group_contrib, TOP_DOCS)]["row_id"].astype(int).tolist()
            bg_ids = bg_frame.iloc[top_indices(bg_contrib, TOP_DOCS)]["row_id"].astype(int).tolist()
            selections[(group, cell_slug)] = {"group_ids": group_ids, "background_ids": bg_ids}
    return selections


def compute_curated_items(
    docs: pd.DataFrame,
    bg_docs: pd.DataFrame,
    doc_weights: np.ndarray,
    bg_weights: np.ndarray,
) -> tuple[list[str], list[str]]:
    group_word_counter = aggregate_weighted_terms(docs, doc_weights, "unigram_counts")
    bg_word_counter = aggregate_weighted_terms(bg_docs, bg_weights, "unigram_counts")
    group_bigram_counter = aggregate_weighted_terms(docs, doc_weights, "bigram_counts")
    bg_bigram_counter = aggregate_weighted_terms(bg_docs, bg_weights, "bigram_counts")
    group_trigram_counter = aggregate_weighted_terms(docs, doc_weights, "trigram_counts")
    bg_trigram_counter = aggregate_weighted_terms(bg_docs, bg_weights, "trigram_counts")

    group_word_docs = aggregate_doc_presence(docs, "unigram_counts")
    bg_word_docs = aggregate_doc_presence(bg_docs, "unigram_counts")
    group_phrase_docs = aggregate_doc_presence(docs, "bigram_counts") + aggregate_doc_presence(docs, "trigram_counts")
    bg_phrase_docs = aggregate_doc_presence(bg_docs, "bigram_counts") + aggregate_doc_presence(bg_docs, "trigram_counts")

    top_words = weighted_log_ratio(
        group_word_counter,
        bg_word_counter,
        top_n=TOP_WORDS,
        min_group_count=2.5,
        min_group_docs=3,
        min_score=0.18,
        filter_fn=token_is_meaningful,
        group_doc_counter=group_word_docs,
        bg_doc_counter=bg_word_docs,
    )
    top_phrases = weighted_log_ratio(
        group_bigram_counter + group_trigram_counter,
        bg_bigram_counter + bg_trigram_counter,
        top_n=TOP_PHRASES,
        min_group_count=1.5,
        min_group_docs=2,
        min_score=0.15,
        filter_fn=phrase_is_meaningful,
        group_doc_counter=group_phrase_docs,
        bg_doc_counter=bg_phrase_docs,
    )
    return [term for term, *_ in top_words], [term for term, *_ in top_phrases]


def format_items(items: list[str]) -> str:
    if not items:
        return "-"
    return "<br>".join(f"{idx}. {item}" for idx, item in enumerate(items[:TOP_ITEMS_FOR_TABLE], start=1))


def main() -> None:
    OVERVIEW_ROOT.mkdir(parents=True, exist_ok=True)
    frame = pd.read_csv(MIXED_INPUT, low_memory=False).copy()
    frame["row_id"] = np.arange(len(frame))
    frame["group_label"] = build_group_label(frame)

    selections = collect_selected_row_ids(frame)
    selected_ids: set[int] = set()
    for selection in selections.values():
        selected_ids.update(selection["group_ids"])
        selected_ids.update(selection["background_ids"])

    enriched = add_document_features(frame[frame["row_id"].isin(selected_ids)].copy()).reset_index(drop=True)

    results: dict[str, dict[str, dict[str, list[str]]]] = {cell_label: {} for cell_label, _, _ in CELL_SPECS}
    table_rows = []

    for cell_label, row_slug, col_slug in CELL_SPECS:
        cell_slug = f"{row_slug}__{col_slug}"
        row_idx = PRINCIPLE_INDEX[row_slug]
        col_idx = PRINCIPLE_INDEX[col_slug]

        row_record = {"cell": cell_label}
        for group in GROUPS:
            selection = selections[(group, cell_slug)]
            group_docs = enriched[enriched["row_id"].isin(selection["group_ids"])].copy().reset_index(drop=True)
            bg_docs = enriched[enriched["row_id"].isin(selection["background_ids"])].copy().reset_index(drop=True)
            if group_docs.empty or bg_docs.empty:
                words, phrases = [], []
            else:
                weights = compute_cell_contribution(group_docs, row_idx, col_idx)
                bg_weights = compute_cell_contribution(bg_docs, row_idx, col_idx)
                words, phrases = compute_curated_items(group_docs, bg_docs, weights, bg_weights)
            results[cell_label][group] = {"words": words[:TOP_ITEMS_FOR_TABLE], "phrases": phrases[:TOP_ITEMS_FOR_TABLE]}
            row_record[f"{group}_words"] = "; ".join(words[:TOP_ITEMS_FOR_TABLE])
            row_record[f"{group}_phrases"] = "; ".join(phrases[:TOP_ITEMS_FOR_TABLE])
        table_rows.append(row_record)

    pd.DataFrame(table_rows).to_csv(OUTPUT_TABLE, index=False)
    OUTPUT_JSON.write_text(json.dumps(results, indent=2), encoding="utf-8")

    lines = [
        "# Overview Evidence Table",
        "",
        "This table summarizes the **top 5 cleaned group-associated evidence items** for selected heatmap cells.",
        "",
        "Notes:",
        "",
        "- Columns are the four overall groups requested for reporting: `HW-B TN`, `LLM-B TN`, `LLM-P FN`, `LLM-P TP`.",
        "- Each group is split into two subcolumns: `Words` and `Phrases`.",
        "- Rows are selected persuasion-principle cell combinations.",
        "- `S` here means **Social Proof**.",
        "- I filtered boilerplate, placeholder entities, repeated variants, and low-meaning strings so the table is easier to read.",
        "- These are **group/cell-associated evidence items**, not exact causal token attributions from the persuasion model.",
        "",
        "| Cell | HW-B TN Words | HW-B TN Phrases | LLM-B TN Words | LLM-B TN Phrases | LLM-P FN Words | LLM-P FN Phrases | LLM-P TP Words | LLM-P TP Phrases |",
        "|---|---|---|---|---|---|---|---|---|",
    ]

    for cell_label, _, _ in CELL_SPECS:
        row = [cell_label]
        for group in GROUPS:
            row.append(format_items(results[cell_label][group]["words"]))
            row.append(format_items(results[cell_label][group]["phrases"]))
        lines.append("| " + " | ".join(row) + " |")

    lines.extend(
        [
            "",
            "Supporting files:",
            "",
            f"- [Curated CSV]({OUTPUT_TABLE})",
            f"- [Curated JSON]({OUTPUT_JSON})",
            "",
            "Group directories:",
            "",
        ]
    )
    for group in GROUPS:
        lines.append(f"- [{group_title(group)}]({OVERVIEW_ROOT / group})")

    OUTPUT_PATH.write_text("\n".join(lines) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
