import argparse
import os
import subprocess
import sys

from build_cialdini_wvae_dataset import DEFAULT_OUTPUT_DIR, build_dataset


SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.dirname(SCRIPT_DIR)
DEFAULT_RUN_DIR = os.path.join(REPO_ROOT, "output", "cialdini_wvae_run")


def parse_args():
    parser = argparse.ArgumentParser(
        description="Build the combined raw-strategy dataset and train the WVAE-BERT model for downstream Cialdini scoring."
    )
    parser.add_argument("--dataset-dir", default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--output-dir", default=DEFAULT_RUN_DIR)
    parser.add_argument("--epochs", type=int, default=6)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--batch-size-u", type=int, default=8)
    parser.add_argument("--val-iteration", type=int, default=120)
    parser.add_argument("--max-seq-num", type=int, default=24)
    parser.add_argument("--max-seq-len", type=int, default=96)
    parser.add_argument("--n-labeled-data", type=int, default=999999)
    parser.add_argument("--n-unlabeled-data", type=int, default=20000)
    parser.add_argument("--gpu", default="0")
    parser.add_argument("--word-dropout", type=float, default=0.6)
    parser.add_argument("--predict-weight", type=float, default=0.1)
    parser.add_argument("--class-weight", type=float, default=5.0)
    parser.add_argument("--kld-weight-y", type=float, default=1.0)
    parser.add_argument("--kld-weight-z", type=float, default=1.0)
    parser.add_argument("--kld-y-thres", type=float, default=1.2)
    parser.add_argument("--rec-coef", type=float, default=1.0)
    parser.add_argument("--lrmain", type=float, default=0.00001)
    parser.add_argument("--lrlast", type=float, default=0.001)
    return parser.parse_args()


def main():
    args = parse_args()
    build_dataset(["borrow", "raop"], args.dataset_dir)
    os.makedirs(args.output_dir, exist_ok=True)

    command = [
        sys.executable,
        os.path.join(SCRIPT_DIR, "train_bert_vae.py"),
        "--data-path",
        args.dataset_dir + "/",
        "--output-dir",
        args.output_dir,
        "--epochs",
        str(args.epochs),
        "--batch-size",
        str(args.batch_size),
        "--batch-size-u",
        str(args.batch_size_u),
        "--val-iteration",
        str(args.val_iteration),
        "--gpu",
        args.gpu,
        "--max-seq-num",
        str(args.max_seq_num),
        "--max-seq-len",
        str(args.max_seq_len),
        "--n-labeled-data",
        str(args.n_labeled_data),
        "--n-unlabeled-data",
        str(args.n_unlabeled_data),
        "--word-dropout",
        str(args.word_dropout),
        "--predict-weight",
        str(args.predict_weight),
        "--class-weight",
        str(args.class_weight),
        "--kld-weight-y",
        str(args.kld_weight_y),
        "--kld-weight-z",
        str(args.kld_weight_z),
        "--kld-y-thres",
        str(args.kld_y_thres),
        "--rec-coef",
        str(args.rec_coef),
        "--lrmain",
        str(args.lrmain),
        "--lrlast",
        str(args.lrlast),
        "--warm-up",
        "False",
        "--tsa-type",
        "no",
        "--hard",
        "True",
    ]
    print(" ".join(command))
    subprocess.run(command, check=True)


if __name__ == "__main__":
    main()
