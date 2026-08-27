import argparse
import csv
import pickle
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F

from AttnModel import Request
from Config import Config
from email_preprocessing import split_clean_sentences


PRINCIPLE_LABELS = {
    1: "Concreteness",
    2: "Commitment",
    3: "Emotional",
    4: "Identity",
    5: "Impact",
    6: "Scarcity",
}


def parse_args():
    parser = argparse.ArgumentParser(
        description="Score email CSV rows with the trained persuasion strategy model."
    )
    parser.add_argument("--input-csv", type=Path, required=True)
    parser.add_argument("--output-csv", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--vocab", type=Path, required=True)
    parser.add_argument("--subject-column", default="subject")
    parser.add_argument("--body-column", default="body")
    parser.add_argument("--max-total-words", type=int, default=3000)
    return parser.parse_args()


def resolve_device():
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def load_model(checkpoint_path: Path, vocab_size: int, device: torch.device):
    config = Config()
    checkpoint = torch.load(checkpoint_path, map_location=device)
    model = Request(config, vocab_size=vocab_size).to(device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()
    return model


def sentence_tokens_to_tensor(vocab, sentence_tokens):
    max_sentence_num = len(sentence_tokens)
    max_len = max(len(tokens) for tokens in sentence_tokens)
    matrix = np.zeros((1, max_sentence_num, max_len), dtype=np.int64)
    lengths = np.zeros((1, max_sentence_num), dtype=np.int64)

    for sentence_index, tokens in enumerate(sentence_tokens):
        lengths[0, sentence_index] = len(tokens)
        for token_index, token in enumerate(tokens):
            matrix[0, sentence_index, token_index] = vocab.word2id.get(token.lower(), 1)

    return (
        torch.from_numpy(matrix),
        torch.LongTensor([max_sentence_num]),
        torch.from_numpy(lengths),
    )


def score_row(model, vocab, subject, body, max_total_words, device):
    sentence_tokens = split_clean_sentences(
        subject=subject,
        body=body,
        max_total_words=max_total_words,
    )
    if not sentence_tokens:
        return {f"principle_{label.lower()}": 0.0 for label in PRINCIPLE_LABELS.values()}

    message_input, message_num, sentence_lengths = sentence_tokens_to_tensor(vocab, sentence_tokens)
    message_input = message_input.to(device)

    with torch.no_grad():
        sentence_out, _ = model(message_input, message_num, sentence_lengths)
        sentence_probs = F.softmax(sentence_out, dim=1).cpu().numpy()

    principle_scores = {}
    for index, label in PRINCIPLE_LABELS.items():
        principle_scores[f"principle_{label.lower()}"] = float(sentence_probs[:, index].mean())
    return principle_scores


def main():
    args = parse_args()
    device = resolve_device()

    with args.vocab.open("rb") as handle:
        vocab = pickle.load(handle)

    model = load_model(args.checkpoint, vocab.vocab_size, device)

    frame = pd.read_csv(args.input_csv)
    if args.subject_column not in frame.columns or args.body_column not in frame.columns:
        raise SystemExit(
            f"Input CSV missing required columns: {args.subject_column}, {args.body_column}"
        )

    output_rows = []
    for _, row in frame.iterrows():
        principle_scores = score_row(
            model,
            vocab,
            row.get(args.subject_column, ""),
            row.get(args.body_column, ""),
            args.max_total_words,
            device,
        )
        record = dict(row)
        record.update(principle_scores)
        output_rows.append(record)

    args.output_csv.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = list(output_rows[0].keys()) if output_rows else list(frame.columns)
    with args.output_csv.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(output_rows)

    print(f"saved scores to {args.output_csv}")


if __name__ == "__main__":
    main()
