import argparse
import csv
import json
import math
import os
import sys

import numpy as np
import torch
import torch.nn.functional as F
from transformers import BertConfig, BertTokenizer

# Handle potential Transformers version mismatch in pickled models
try:
    import transformers.models.bert.modeling_bert as modeling_bert
    if not hasattr(modeling_bert, "BertSdpaSelfAttention"):
        modeling_bert.BertSdpaSelfAttention = modeling_bert.BertSelfAttention
except (ImportError, AttributeError):
    pass

from cialdini_config import PRINCIPLE_ORDER, PRINCIPLE_STRATEGY_MAP, RAW_STRATEGY_NAMES
from email_preprocessing import build_fallback_sentence, split_clean_sentences


SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.dirname(SCRIPT_DIR)
DEFAULT_MODEL_PATH = os.path.join(REPO_ROOT, "output", "cialdini_wvae_run", "model.pkl")


def parse_args():
    parser = argparse.ArgumentParser(
        description="Score email CSV rows with the trained WVAE-BERT strategy model and aggregate them into six Cialdini-style principle scores."
    )
    parser.add_argument("--input-csv", required=True)
    parser.add_argument("--output-csv", required=True)
    parser.add_argument("--model-path", default=DEFAULT_MODEL_PATH)
    parser.add_argument("--subject-column", default="Subject")
    parser.add_argument("--body-column", default="Body")
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--max-seq-len", type=int, default=96)
    parser.add_argument("--max-total-words", type=int, default=3000)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--metadata-json", default="")
    return parser.parse_args()


def choose_device(requested):
    if requested != "auto":
        return torch.device(requested)
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def refresh_legacy_bert_config(model):
    encoder = getattr(model, "encoder", None)
    bert_model = getattr(encoder, "bert", None)
    legacy_config = getattr(bert_model, "config", None)
    if bert_model is None or legacy_config is None:
        return
    try:
        config_dict = legacy_config.to_dict()
    except AttributeError:
        return
    refreshed_config = BertConfig(**config_dict)
    if not hasattr(refreshed_config, "_attn_implementation"):
        refreshed_config._attn_implementation = "eager"
    bert_model.config = refreshed_config
    if not hasattr(bert_model, "attn_implementation"):
        bert_model.attn_implementation = "eager"


def load_rows(path):
    csv.field_size_limit(sys.maxsize)
    with open(path, "r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        return reader.fieldnames, list(reader)


def pack_sentences(tokenizer, sentence_tokens, max_seq_len):
    packed = []
    sent_lens = []
    for tokens in sentence_tokens:
        wordpieces = tokenizer.tokenize(" ".join(tokens))
        current = ["[CLS]"]
        current.extend(wordpieces[: max_seq_len - 2])
        current.append("[SEP]")
        sent_lens.append(len(current) - 1)
        packed.append(tokenizer.convert_tokens_to_ids(current))
    return packed, sent_lens


def batch_encode(model, sentence_batches, sent_len_batches, device):
    outputs = []
    with torch.no_grad():
        for batch_ids, batch_sent_lens in zip(sentence_batches, sent_len_batches):
            width = max(len(item) for item in batch_ids)
            encoded = torch.zeros((len(batch_ids), width), dtype=torch.long)
            for row_index, input_ids in enumerate(batch_ids):
                encoded[row_index, : len(input_ids)] = torch.tensor(input_ids, dtype=torch.long)
            encoded = encoded.to(device)
            sent_lens_tensor = torch.tensor(batch_sent_lens, dtype=torch.long, device=device)
            logits, _ = model.encode(encoded, sent_len=sent_lens_tensor)
            outputs.append(F.softmax(logits, dim=-1).cpu().numpy())
    return np.concatenate(outputs, axis=0) if outputs else np.zeros((0, len(RAW_STRATEGY_NAMES)))


def principle_scores_from_sentence_probs(sentence_probs):
    if sentence_probs.size == 0:
        return {name: 0.0 for name in PRINCIPLE_ORDER}
    scores = {}
    for principle_name in PRINCIPLE_ORDER:
        raw_indices = PRINCIPLE_STRATEGY_MAP[principle_name]
        principle_sentence_probs = sentence_probs[:, raw_indices].sum(axis=1)
        scores[principle_name] = float(1.0 - np.prod(1.0 - principle_sentence_probs))
    return scores


def other_probability_mean(sentence_probs):
    if sentence_probs.size == 0:
        return 0.0
    return float(sentence_probs[:, 0].mean())


def main():
    args = parse_args()
    fieldnames, rows = load_rows(args.input_csv)
    tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
    device = choose_device(args.device)
    model = torch.load(args.model_path, map_location=device, weights_only=False)
    refresh_legacy_bert_config(model)
    model = model.to(device)
    model.eval()

    output_rows = []
    total_sentences = 0
    total_tokens = 0
    empty_fallbacks = 0

    for row in rows:
        sentence_tokens = split_clean_sentences(
            row.get(args.subject_column, ""),
            row.get(args.body_column, ""),
            max_total_words=args.max_total_words,
        )
        if not sentence_tokens:
            sentence_tokens = build_fallback_sentence(
                row.get(args.subject_column, ""),
                row.get(args.body_column, ""),
                max_total_words=args.max_total_words,
            )
            empty_fallbacks += 1

        packed_sentences, sent_lens = pack_sentences(tokenizer, sentence_tokens, args.max_seq_len)
        total_sentences += len(packed_sentences)
        total_tokens += sum(len(tokens) for tokens in sentence_tokens)

        sentence_batches = []
        sent_len_batches = []
        for start in range(0, len(packed_sentences), args.batch_size):
            end = start + args.batch_size
            sentence_batches.append(packed_sentences[start:end])
            sent_len_batches.append(sent_lens[start:end])

        sentence_probs = batch_encode(model, sentence_batches, sent_len_batches, device)
        principle_scores = principle_scores_from_sentence_probs(sentence_probs)

        enriched = dict(row)
        for principle_name in PRINCIPLE_ORDER:
            enriched[f"principle_{principle_name}"] = f"{principle_scores[principle_name]:.6f}"
        enriched["other_probability_mean"] = f"{other_probability_mean(sentence_probs):.6f}"
        enriched["sentence_count_used"] = str(len(sentence_tokens))
        enriched["token_count_used"] = str(sum(len(tokens) for tokens in sentence_tokens))
        output_rows.append(enriched)

    extra_fields = [
        *(f"principle_{name}" for name in PRINCIPLE_ORDER),
        "other_probability_mean",
        "sentence_count_used",
        "token_count_used",
    ]
    output_fields = list(fieldnames)
    for extra_field in extra_fields:
        if extra_field not in output_fields:
            output_fields.append(extra_field)

    os.makedirs(os.path.dirname(args.output_csv), exist_ok=True)
    with open(args.output_csv, "w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=output_fields)
        writer.writeheader()
        writer.writerows(output_rows)

    metadata = {
        "input_csv": args.input_csv,
        "output_csv": args.output_csv,
        "model_path": args.model_path,
        "row_count": len(rows),
        "total_sentences_used": total_sentences,
        "total_tokens_used": total_tokens,
        "empty_fallback_rows": empty_fallbacks,
        "max_total_words": args.max_total_words,
        "max_seq_len": args.max_seq_len,
        "principle_order": PRINCIPLE_ORDER,
        "principle_strategy_map": PRINCIPLE_STRATEGY_MAP,
    }
    metadata_path = args.metadata_json or os.path.splitext(args.output_csv)[0] + ".metadata.json"
    with open(metadata_path, "w", encoding="utf-8") as handle:
        json.dump(metadata, handle, indent=2, sort_keys=True)
    print(json.dumps(metadata, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
