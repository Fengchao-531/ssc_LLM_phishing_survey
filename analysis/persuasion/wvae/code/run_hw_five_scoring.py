import argparse
import os
import subprocess
import sys


SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(SCRIPT_DIR)))
DEFAULT_DATASET_DIR = os.path.join(
    REPO_ROOT,
    "Detectors",
    "Industry",
    "email_detectors",
    "output",
    "HW-result",
    "datasets",
)
DEFAULT_OUTPUT_DIR = os.path.join(
    os.path.dirname(SCRIPT_DIR),
    "output",
    "hw_five_scores",
)
DEFAULT_MODEL_PATH = os.path.join(
    os.path.dirname(SCRIPT_DIR),
    "output",
    "cialdini_wvae_run",
    "model.pkl",
)

TARGET_DATASETS = [
    "S6-UTA-GD",
    "S6-fuzzer-GD",
    "S8-deepseek-GD",
    "S8-llama-GD",
    "S8-ministral-GD",
]


def parse_args():
    parser = argparse.ArgumentParser(
        description="Score the five HW datasets used for the current persuasion visualization pass."
    )
    parser.add_argument("--dataset-dir", default=DEFAULT_DATASET_DIR)
    parser.add_argument("--output-dir", default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--model-path", default=DEFAULT_MODEL_PATH)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--max-seq-len", type=int, default=96)
    parser.add_argument("--max-total-words", type=int, default=3000)
    return parser.parse_args()


def main():
    args = parse_args()
    os.makedirs(args.output_dir, exist_ok=True)

    for dataset_name in TARGET_DATASETS:
        input_csv = os.path.join(args.dataset_dir, dataset_name + ".csv")
        output_csv = os.path.join(args.output_dir, dataset_name + "__principles.csv")
        command = [
            sys.executable,
            os.path.join(SCRIPT_DIR, "score_email_csv.py"),
            "--input-csv",
            input_csv,
            "--output-csv",
            output_csv,
            "--model-path",
            args.model_path,
            "--device",
            args.device,
            "--batch-size",
            str(args.batch_size),
            "--max-seq-len",
            str(args.max_seq_len),
            "--max-total-words",
            str(args.max_total_words),
        ]
        print(" ".join(command))
        subprocess.run(command, check=True)


if __name__ == "__main__":
    main()
