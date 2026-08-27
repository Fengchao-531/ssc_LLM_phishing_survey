import argparse
import pickle
from pathlib import Path

import torch
import torch.utils.data as Data
from torch.optim.lr_scheduler import ExponentialLR

from AttnModel import Request
from Config import Config
from DataLoader import Vocab, dataLoaderANN, dataLoaderUnann
from MessageLoss import MessageLoss


def parse_args():
    parser = argparse.ArgumentParser(
        description="Train the Persuasion_Strategy model with configurable data paths."
    )
    parser.add_argument("--dataset-text", type=Path, required=True)
    parser.add_argument("--dataset-with-annotation", type=Path, required=True)
    parser.add_argument("--target-pickle", type=Path, default=None)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--ann-batch-size", type=int, default=64)
    parser.add_argument("--unann-batch-size", type=int, default=64)
    parser.add_argument("--dev-batch-size", type=int, default=128)
    parser.add_argument("--learning-rate", type=float, default=5e-5)
    parser.add_argument("--max-total-words", type=int, default=3000)
    parser.add_argument("--without-unann", action="store_true")
    return parser.parse_args()


def resolve_device() -> torch.device:
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def evaluation(model, loader_dev, message_loss, device):
    model.eval()
    correct_total = 0
    count_total = 0
    last_loss = None
    last_rmse = None
    last_sent_loss = None
    last_p = 0.0
    last_r = 0.0

    with torch.no_grad():
        for x, y, l, num, length in loader_dev:
            message_input = x.long().to(device)
            message_target = y.float().to(device)
            sentence_label = l.long().to(device)

            sentence_out, message_out = model(message_input, num, length)
            loss, rmse, sent_loss, _, _, _, correct, count, p, r = message_loss(
                labeled_doc=message_out,
                target1=message_target,
                labeled_sent=sentence_out,
                target2=sentence_label,
                mode="dev",
            )
            correct_total += correct
            count_total += count
            last_loss = float(loss.detach().cpu().item())
            last_rmse = float(rmse)
            last_sent_loss = float(sent_loss.detach().cpu().item()) if hasattr(sent_loss, "detach") else float(sent_loss)
            last_p = float(p)
            last_r = float(r)

    accuracy = correct_total / max(1, count_total)
    f1 = (2 * last_p * last_r / (last_p + last_r)) if (last_p + last_r) else 0.0
    metrics = {
        "accuracy": accuracy,
        "loss": last_loss if last_loss is not None else 0.0,
        "rmse": last_rmse if last_rmse is not None else 0.0,
        "sentence_loss": last_sent_loss if last_sent_loss is not None else 0.0,
        "precision_macro": last_p,
        "recall_macro": last_r,
        "f1_macro": f1,
    }
    return metrics


def main():
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    device = resolve_device()

    config = Config()

    vocab = Vocab(
        dataset_with_annotation=str(args.dataset_with_annotation),
        dataset_text=str(args.dataset_text),
        max_total_words=args.max_total_words,
    )

    train_ann = dataLoaderANN(
        vocab,
        str(args.dataset_with_annotation),
        mode="train",
        target_pickle_path=args.target_pickle,
    )
    dev_ann = dataLoaderANN(
        vocab,
        str(args.dataset_with_annotation),
        mode="dev",
        target_pickle_path=args.target_pickle,
    )
    train_unann = dataLoaderUnann(
        vocab,
        str(args.dataset_text),
        target_pickle_path=args.target_pickle,
        max_total_words=args.max_total_words,
    )

    loader_train_ann = Data.DataLoader(
        dataset=train_ann,
        batch_size=args.ann_batch_size,
        shuffle=True,
        num_workers=0,
    )
    loader_train_unann = Data.DataLoader(
        dataset=train_unann,
        batch_size=args.unann_batch_size,
        shuffle=True,
        num_workers=0,
    )
    loader_dev = Data.DataLoader(
        dataset=dev_ann,
        batch_size=args.dev_batch_size,
        shuffle=False,
        num_workers=0,
    )

    model = Request(config, vocab_size=vocab.vocab_size).to(device)
    message_loss = MessageLoss(w2=10).to(device)
    optimizer = torch.optim.Adam(params=model.parameters(), lr=args.learning_rate)
    scheduler = ExponentialLR(optimizer, gamma=0.9)

    best_accuracy = -1.0
    best_metrics = None
    best_checkpoint = args.output_dir / "model_best.pt"
    vocab_path = args.output_dir / "vocab.pkl"
    metadata_path = args.output_dir / "training_metadata.pkl"

    with vocab_path.open("wb") as handle:
        pickle.dump(vocab, handle)

    unann_batches = list(loader_train_unann) if not args.without_unann else []

    for epoch in range(1, args.epochs + 1):
        model.train()
        unann_index = 0
        for step, ann_batch in enumerate(loader_train_ann, start=1):
            x, y, l, num, length = ann_batch
            ann_message_input = x.long().to(device)
            ann_message_target = y.float().to(device)
            ann_sentence_label = l.long().to(device)
            ann_sentence_out, ann_message_out = model(ann_message_input, num, length)

            if args.without_unann or not unann_batches:
                loss, labeled_sent_loss, *_ = message_loss(
                    labeled_doc=ann_message_out,
                    target1=ann_message_target,
                    labeled_sent=ann_sentence_out,
                    target2=ann_sentence_label,
                    w1=1.0,
                    unlabeled_doc=None,
                    target3=None,
                    mode="train",
                )
            else:
                ux, uy, _, unum, ulength = unann_batches[unann_index % len(unann_batches)]
                unann_index += 1
                unann_message_input = ux.long().to(device)
                unann_message_target = uy.float().to(device)
                _, unann_message_out = model(unann_message_input, unum, ulength)
                w1 = ann_message_input.shape[0] / (
                    ann_message_input.shape[0] + unann_message_input.shape[0]
                )
                loss, labeled_sent_loss, *_ = message_loss(
                    labeled_doc=ann_message_out,
                    target1=ann_message_target,
                    labeled_sent=ann_sentence_out,
                    target2=ann_sentence_label,
                    w1=w1,
                    unlabeled_doc=unann_message_out,
                    target3=unann_message_target,
                    mode="train",
                )

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            if step % 20 == 0 or step == 1:
                sentence_loss_value = (
                    float(labeled_sent_loss.detach().cpu().item())
                    if hasattr(labeled_sent_loss, "detach")
                    else float(labeled_sent_loss)
                )
                print(
                    f"epoch={epoch} step={step} loss={float(loss.detach().cpu().item()):.4f} "
                    f"sentence_loss={sentence_loss_value:.4f}",
                    flush=True,
                )

        metrics = evaluation(model, loader_dev, message_loss, device)
        scheduler.step()
        print(
            f"[dev] epoch={epoch} accuracy={metrics['accuracy']:.4f} "
            f"loss={metrics['loss']:.4f} rmse={metrics['rmse']:.4f} "
            f"f1_macro={metrics['f1_macro']:.4f}",
            flush=True,
        )

        if metrics["accuracy"] >= best_accuracy:
            best_accuracy = metrics["accuracy"]
            best_metrics = metrics
            torch.save(
                {
                    "model_state_dict": model.state_dict(),
                    "config": vars(config),
                    "vocab_size": vocab.vocab_size,
                    "max_total_words": args.max_total_words,
                },
                best_checkpoint,
            )

    with metadata_path.open("wb") as handle:
        pickle.dump(
            {
                "dataset_text": str(args.dataset_text),
                "dataset_with_annotation": str(args.dataset_with_annotation),
                "target_pickle": str(args.target_pickle) if args.target_pickle else "",
                "best_metrics": best_metrics,
                "best_accuracy": best_accuracy,
                "max_total_words": args.max_total_words,
                "device": str(device),
            },
            handle,
        )
    print(f"saved model to {best_checkpoint}")
    print(f"saved vocab to {vocab_path}")


if __name__ == "__main__":
    main()
