import argparse
import json
import os
import pickle

from cialdini_config import IDENTITY_LABEL_MAPPING, PRINCIPLE_NOTES, PRINCIPLE_STRATEGY_MAP, RAW_STRATEGY_NAMES


SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.dirname(SCRIPT_DIR)
DEFAULT_OUTPUT_DIR = os.path.join(REPO_ROOT, "data", "cialdini_combined_raw10")


def parse_args():
    parser = argparse.ArgumentParser(
        description="Build a combined Borrow+RAOP dataset for WVAE training with raw strategy labels preserved."
    )
    parser.add_argument(
        "--sources",
        nargs="+",
        default=["borrow", "raop"],
        help="Source datasets under the local data directory.",
    )
    parser.add_argument(
        "--output-dir",
        default=DEFAULT_OUTPUT_DIR,
        help="Directory where the combined pickles should be written.",
    )
    return parser.parse_args()


def load_pickle(path):
    with open(path, "rb") as handle:
        return pickle.load(handle)


def dump_pickle(path, value):
    with open(path, "wb") as handle:
        pickle.dump(value, handle)


def build_dataset(source_names, output_dir):
    labeled_data = {}
    unlabeled_data = {}
    mid2target = {}
    metadata = {"sources": {}, "label_mapping": IDENTITY_LABEL_MAPPING}

    for source_name in source_names:
        source_dir = os.path.join(REPO_ROOT, "data", source_name)
        source_labeled = load_pickle(os.path.join(source_dir, "labeled_data.pkl"))
        source_unlabeled = load_pickle(os.path.join(source_dir, "unlabeled_data.pkl"))
        source_target = load_pickle(os.path.join(source_dir, "mid2target.pkl"))

        metadata["sources"][source_name] = {
            "labeled_messages": len(source_labeled),
            "unlabeled_messages": len(source_unlabeled),
            "doc_targets": len(source_target),
        }

        for mid, value in source_labeled.items():
            labeled_data[f"{source_name}::{mid}"] = value
        for mid, value in source_unlabeled.items():
            unlabeled_data[f"{source_name}::{mid}"] = value
        for mid, value in source_target.items():
            mid2target[f"{source_name}::{mid}"] = value

    os.makedirs(output_dir, exist_ok=True)
    dump_pickle(os.path.join(output_dir, "labeled_data.pkl"), labeled_data)
    dump_pickle(os.path.join(output_dir, "unlabeled_data.pkl"), unlabeled_data)
    dump_pickle(os.path.join(output_dir, "mid2target.pkl"), mid2target)
    dump_pickle(os.path.join(output_dir, "label_mapping.pkl"), IDENTITY_LABEL_MAPPING)

    metadata["raw_strategy_names"] = RAW_STRATEGY_NAMES
    metadata["principle_strategy_map"] = PRINCIPLE_STRATEGY_MAP
    metadata["principle_notes"] = PRINCIPLE_NOTES
    with open(os.path.join(output_dir, "dataset_metadata.json"), "w", encoding="utf-8") as handle:
        json.dump(metadata, handle, indent=2, sort_keys=True)

    return metadata


def main():
    args = parse_args()
    metadata = build_dataset(args.sources, args.output_dir)
    print(json.dumps(metadata, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
