#!/usr/bin/env python3
"""Export 7-QuishingDataset pickle samples into per-image folders for S7."""

from __future__ import annotations

import pickle
from pathlib import Path

import numpy as np
from PIL import Image

SCRIPT_DIR = Path(__file__).resolve().parent
DATASETS_DIR = SCRIPT_DIR.parent

QUISHING_DIR = DATASETS_DIR / "HW-Phishing" / "7-QuishingDataset"
IMAGES_PATH = QUISHING_DIR / "qr_codes_29.pickle"
LABELS_PATH = QUISHING_DIR / "qr_codes_29_labels.pickle"

S7_DIR = SCRIPT_DIR / "S7-Cross-channel Expansion"
PHISH_DIR = S7_DIR / "HW-Quishing"
BENIGN_DIR = S7_DIR / "HW-QRCode"


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


def main() -> None:
    images = np.asarray(load_pickle(IMAGES_PATH))
    labels = np.asarray(load_pickle(LABELS_PATH))

    if len(images) != len(labels):
        raise ValueError(
            f"Sample/label count mismatch: {len(images)} images vs {len(labels)} labels"
        )

    clear_existing_pngs(PHISH_DIR)
    clear_existing_pngs(BENIGN_DIR)

    phish_count = 0
    benign_count = 0

    total = len(images)

    for index, (sample, label) in enumerate(zip(images, labels)):
        label_value = int(label)
        if label_value == 1:
            output_dir = PHISH_DIR
            prefix = "quishing"
            phish_count += 1
        elif label_value == 0:
            output_dir = BENIGN_DIR
            prefix = "qrcode"
            benign_count += 1
        else:
            raise ValueError(f"Unsupported label {label_value} at index {index}")

        image = to_uint8_image(sample)
        filename = f"{prefix}_{index:05d}.png"
        image.save(output_dir / filename, compress_level=0)

        if (index + 1) % 1000 == 0:
            print(f"Processed {index + 1}/{total} QR samples...")

    print(f"Exported {phish_count} phishing QR images to {PHISH_DIR}")
    print(f"Exported {benign_count} benign QR images to {BENIGN_DIR}")


if __name__ == "__main__":
    main()
