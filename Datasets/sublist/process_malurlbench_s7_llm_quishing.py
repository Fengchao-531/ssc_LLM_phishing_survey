#!/usr/bin/env python3
"""Convert MalURLBench example URLs into per-image QR code files for S7."""

from __future__ import annotations

import re
from pathlib import Path

import qrcode

SCRIPT_DIR = Path(__file__).resolve().parent
DATASETS_DIR = SCRIPT_DIR.parent

EXAMPLES_DIR = DATASETS_DIR / "LLM-Phishing" / "MalURLBench" / "Examples"
OUTPUT_DIR = SCRIPT_DIR / "S7-Cross-channel Expansion" / "LLM-Quishing"
TARGET_TOTAL = 5300


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


def allocate_targets(model_names: list[str], target_total: int) -> dict[str, int]:
    model_count = len(model_names)
    if model_count == 0:
        return {}
    base = target_total // model_count
    remainder = target_total % model_count
    targets: dict[str, int] = {}
    for index, model_name in enumerate(model_names):
        targets[model_name] = base + (1 if index < remainder else 0)
    return targets


def main() -> None:
    model_dirs = sorted([path for path in EXAMPLES_DIR.iterdir() if path.is_dir()])
    if not model_dirs:
        raise FileNotFoundError(f"No txt files found under {EXAMPLES_DIR}")

    clear_existing_pngs(OUTPUT_DIR)

    model_names = [sanitize_name(path.name) for path in model_dirs]
    targets = allocate_targets(model_names, TARGET_TOTAL)

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
                output_path = OUTPUT_DIR / filename
                image = build_qr_image(url)
                image.save(output_path, compress_level=0)
                image_count += 1
                model_count += 1
                if image_count % 1000 == 0:
                    print(f"Created {image_count}/{TARGET_TOTAL} QR images...")

        processed_models += 1
        print(f"{model_name}: created {model_count} images")

    print(f"Processed {processed_models} model folders from {EXAMPLES_DIR}")
    print(f"Exported {image_count} QR images to {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
