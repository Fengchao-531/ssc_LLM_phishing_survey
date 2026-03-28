#!/usr/bin/env python3
"""Consolidated S7 quishing processor.

Replaces:
- process_quishing_s7_images.py
- process_malurlbench_s7_llm_quishing.py
"""

from __future__ import annotations

import argparse
import pickle
import re
from pathlib import Path

import numpy as np
import qrcode
from PIL import Image

SCRIPT_DIR = Path(__file__).resolve().parent
DATASETS_DIR = SCRIPT_DIR.parent


def run_hw_quishing() -> None:
    quishing_dir = DATASETS_DIR / "HW-Phishing" / "7-QuishingDataset"
    images_path = quishing_dir / "qr_codes_29.pickle"
    labels_path = quishing_dir / "qr_codes_29_labels.pickle"

    s7_dir = SCRIPT_DIR / "S7-Cross-channel Expansion"
    phish_dir = s7_dir / "HW-Quishing"
    benign_dir = s7_dir / "HW-QRCode"

    def load_pickle(path: Path):
        with path.open("rb") as handle:
            return pickle.load(handle)

    def clear_existing_pngs(folder: Path) -> None:
        folder.mkdir(parents=True, exist_ok=True)
        for png_file in folder.glob("*.png"):
            png_file.unlink()

    def to_uint8_image(sample: np.ndarray) -> Image.Image:
        array = np.asarray(sample)
        if array.ndim != 2:
            raise ValueError(f"Expected 2D QR sample, got shape {array.shape}")
        if array.dtype != np.uint8:
            max_value = int(array.max()) if array.size else 0
            if max_value <= 1:
                array = array.astype(np.uint8) * 255
            else:
                array = np.clip(array, 0, 255).astype(np.uint8)
        return Image.fromarray(array, mode="L")

    images = np.asarray(load_pickle(images_path))
    labels = np.asarray(load_pickle(labels_path))

    if len(images) != len(labels):
        raise ValueError(
            f"Sample/label count mismatch: {len(images)} images vs {len(labels)} labels"
        )

    clear_existing_pngs(phish_dir)
    clear_existing_pngs(benign_dir)

    phish_count = 0
    benign_count = 0
    total = len(images)

    for index, (sample, label) in enumerate(zip(images, labels)):
        label_value = int(label)
        if label_value == 1:
            output_dir = phish_dir
            prefix = "quishing"
            phish_count += 1
        elif label_value == 0:
            output_dir = benign_dir
            prefix = "qrcode"
            benign_count += 1
        else:
            raise ValueError(f"Unsupported label {label_value} at index {index}")

        image = to_uint8_image(sample)
        filename = f"{prefix}_{index:05d}.png"
        image.save(output_dir / filename, compress_level=0)

        if (index + 1) % 1000 == 0:
            print(f"Processed {index + 1}/{total} QR samples...")

    print(f"Exported {phish_count} phishing QR images to {phish_dir}")
    print(f"Exported {benign_count} benign QR images to {benign_dir}")


def run_llm_quishing() -> None:
    examples_dir = DATASETS_DIR / "LLM-Phishing" / "MalURLBench" / "Examples"
    output_dir = SCRIPT_DIR / "S7-Cross-channel Expansion" / "LLM-Quishing"
    target_total = 5300

    def sanitize_name(value: str) -> str:
        sanitized = re.sub(r"[^A-Za-z0-9._-]+", "_", value.strip())
        return sanitized.strip("._") or "item"

    def iter_urls(txt_path: Path):
        with txt_path.open("r", encoding="utf-8", errors="replace") as handle:
            for line_number, line in enumerate(handle, start=1):
                url = line.strip()
                if not url:
                    continue
                yield line_number, url

    def clear_existing_pngs(folder: Path) -> None:
        folder.mkdir(parents=True, exist_ok=True)
        for png_file in folder.glob("*.png"):
            png_file.unlink()

    def build_qr_image(url: str):
        qr = qrcode.QRCode(
            version=None,
            error_correction=qrcode.constants.ERROR_CORRECT_M,
            box_size=8,
            border=4,
        )
        qr.add_data(url)
        qr.make(fit=True)
        return qr.make_image(fill_color="black", back_color="white")

    def allocate_targets(model_names: list[str], total: int) -> dict[str, int]:
        model_count = len(model_names)
        if model_count == 0:
            return {}
        base = total // model_count
        remainder = total % model_count
        targets: dict[str, int] = {}
        for index, model_name in enumerate(model_names):
            targets[model_name] = base + (1 if index < remainder else 0)
        return targets

    model_dirs = sorted([path for path in examples_dir.iterdir() if path.is_dir()])
    if not model_dirs:
        raise FileNotFoundError(f"No txt files found under {examples_dir}")

    clear_existing_pngs(output_dir)

    model_names = [sanitize_name(path.name) for path in model_dirs]
    targets = allocate_targets(model_names, target_total)

    image_count = 0
    processed_models = 0

    for model_dir in model_dirs:
        model_name = sanitize_name(model_dir.name)
        model_target = targets[model_name]
        model_count = 0
        txt_files = sorted(model_dir.glob("*.txt"))

        for txt_path in txt_files:
            if model_count >= model_target:
                break
            template_name = sanitize_name(txt_path.stem)
            for url_index, (_, url) in enumerate(iter_urls(txt_path), start=1):
                if model_count >= model_target:
                    break
                filename = f"{model_name}__{template_name}__{url_index:04d}.png"
                output_path = output_dir / filename
                image = build_qr_image(url)
                image.save(output_path, compress_level=0)
                image_count += 1
                model_count += 1
                if image_count % 1000 == 0:
                    print(f"Created {image_count}/{target_total} QR images...")

        processed_models += 1
        print(f"{model_name}: {model_count}/{model_target} images ready")

    print(f"Processed {processed_models} model folders from {examples_dir}")
    print(f"Exported {image_count} QR images to {output_dir}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Consolidated S7 quishing processor.")
    parser.add_argument(
        "--mode",
        choices=("hw", "llm", "all"),
        default="all",
        help="Which S7 quishing pipeline to run.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.mode in {"hw", "all"}:
        run_hw_quishing()
    if args.mode in {"llm", "all"}:
        run_llm_quishing()


if __name__ == "__main__":
    main()
