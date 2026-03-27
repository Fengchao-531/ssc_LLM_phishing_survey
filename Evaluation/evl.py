#!/usr/bin/env python3
"""Dataset normalizer for the phishing survey project.

This script ingests a subset of the heterogeneous corpora found under the
``Datasets`` directory (currently S4*, 5Ham, and LegitPhish) and emits:

1. A consolidated CSV file whose header is the union of every column observed
   across the supported sources.
2. A normalized ``.eml`` message for every record, stored under a dedicated
   output directory so the raw email bodies can be inspected downstream.

The implementation favours the standard library so it can run in constrained
environments.  Each loader is intentionally modular; extending the
``REGISTERED_LOADERS`` list with new handlers is the quickest way to add more
datasets later.
"""

from __future__ import annotations

import argparse
import csv
import html
import io
import json
import logging
import os
import re
import shutil
from dataclasses import dataclass
from email import policy
from email.message import EmailMessage
from email.parser import BytesParser
from pathlib import Path
from typing import Dict, Iterable, Iterator, List, MutableMapping, Optional, Sequence, Tuple
import zipfile

LOGGER = logging.getLogger(__name__)

SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parent
DEFAULT_DATA_ROOT = REPO_ROOT / "Datasets"
DEFAULT_OUTPUT_ROOT = SCRIPT_DIR / "processed"
DEFAULT_EML_SUBDIR = "eml"
DEFAULT_CSV_NAME = "combined_records.csv"

DEFAULT_ENCODING = "utf-8"
SAFE_FILENAME_RE = re.compile(r"[^A-Za-z0-9_.-]+")
HTML_TAG_RE = re.compile(r"<[^>]+>")

# Fields to keep at the front of the CSV; the rest will follow alphabetically.
BASE_FIELD_ORDER: Sequence[str] = (
    "source_dataset",
    "source_detail",
    "record_id",
    "label",
    "eml_path",
    "template_key",
    "From",
    "Fullname",
    "To",
    "Cc",
    "Bcc",
    "Reply-To",
    "Subject",
    "Date",
    "Return_Path",
    "Delivered_to",
    "Message_Id",
    "IP",
    "Content_type",
    "Content_Length",
    "Body",
    "body_text",
    "body_html",
    "Raw_Headers",
    "Language",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Normalize S4, 5Ham, and LegitPhish datasets into CSV + EML outputs."
    )
    parser.add_argument(
        "--data-root",
        type=Path,
        default=DEFAULT_DATA_ROOT,
        help=f"Root folder that contains the raw datasets (default: {DEFAULT_DATA_ROOT})",
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=DEFAULT_OUTPUT_ROOT,
        help=f"Directory where the CSV and eml/ tree are written (default: {DEFAULT_OUTPUT_ROOT})",
    )
    parser.add_argument(
        "--eml-subdir",
        type=str,
        default=DEFAULT_EML_SUBDIR,
        help="Name of the folder (within output-root) that stores the generated .eml files.",
    )
    parser.add_argument(
        "--csv-name",
        type=str,
        default=DEFAULT_CSV_NAME,
        help="Filename (relative to output-root) for the consolidated CSV.",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Optional per-dataset record cap useful for quick smoke tests.",
    )
    parser.add_argument(
        "--clean-output",
        action="store_true",
        help="Delete the target output-root before running to avoid mixing stale files.",
    )
    parser.add_argument(
        "--log-level",
        default="INFO",
        choices=("DEBUG", "INFO", "WARNING", "ERROR"),
        help="Runtime logging verbosity.",
    )
    return parser.parse_args()


def configure_logging(level: str) -> None:
    logging.basicConfig(
        level=getattr(logging, level.upper(), logging.INFO),
        format="%(asctime)s | %(levelname)s | %(message)s",
    )


def safe_slug(value: str, fallback: str) -> str:
    """Return a filesystem-friendly slug."""
    cleaned = SAFE_FILENAME_RE.sub("_", value).strip("._")
    return cleaned or fallback


def looks_like_html(value: str) -> bool:
    return bool(value and HTML_TAG_RE.search(value))


def html_to_text(value: str) -> str:
    if not value:
        return ""
    sanitized = value.replace("<br>", "\n").replace("<br/>", "\n").replace("<br />", "\n")
    sanitized = re.sub(r"<(script|style)[^>]*>.*?</\1>", "", sanitized, flags=re.IGNORECASE | re.DOTALL)
    stripped = HTML_TAG_RE.sub("", sanitized)
    return html.unescape(stripped)


def coerce_bodies(raw: str) -> Tuple[str, str]:
    """Return (text_body, html_body) for the incoming payload."""
    if not raw:
        return "", ""
    if looks_like_html(raw):
        html_payload = raw.strip()
        text_payload = html_to_text(raw).strip()
        return text_payload, html_payload
    return raw.strip(), ""


def split_subject_and_body(raw: str) -> Tuple[str, str]:
    lines = [line.rstrip() for line in raw.splitlines()]
    while lines and not lines[0].strip():
        lines.pop(0)
    subject = ""
    if lines:
        first = lines.pop(0).strip()
        if first.lower().startswith("subject:"):
            subject = first.split(":", 1)[1].strip()
        else:
            subject = first
    body = "\n".join(lines).strip()
    return subject, body


def format_header_block(headers: Sequence[Tuple[str, str]]) -> str:
    return "\n".join(f"{key}: {value}" for key, value in headers)


def relative_path(path: Path, base: Path) -> str:
    try:
        return path.relative_to(base).as_posix()
    except ValueError:
        return path.as_posix()


class RecordCollector:
    """Accumulates flattened dictionaries and knows how to emit CSV + EML artifacts."""

    def __init__(self, output_root: Path, eml_subdir: str) -> None:
        self.output_root = output_root
        self.csv_path = (output_root / DEFAULT_CSV_NAME).resolve()
        self.eml_root = (output_root / eml_subdir).resolve()
        self.records: List[Dict[str, str]] = []
        self.all_fields: set[str] = set()

    def set_csv_path(self, csv_name: str) -> None:
        self.csv_path = (self.output_root / csv_name).resolve()

    def register(self, row: MutableMapping[str, object]) -> None:
        normalized: Dict[str, str] = {}
        for key, value in row.items():
            if value is None:
                normalized[key] = ""
            elif isinstance(value, Path):
                normalized[key] = relative_path(value.resolve(), self.output_root)
            else:
                normalized[key] = str(value)
        self.records.append(normalized)
        self.all_fields.update(normalized.keys())

    def write_csv(self) -> None:
        if not self.records:
            LOGGER.warning("No rows were collected; CSV will not be written.")
            return
        self.csv_path.parent.mkdir(parents=True, exist_ok=True)
        ordered_fields = build_field_order(self.all_fields)
        with self.csv_path.open("w", newline="", encoding=DEFAULT_ENCODING) as handle:
            writer = csv.DictWriter(handle, fieldnames=ordered_fields)
            writer.writeheader()
            for row in self.records:
                writer.writerow({field: row.get(field, "") for field in ordered_fields})
        LOGGER.info("CSV written to %s (%d rows)", self.csv_path, len(self.records))

    def write_eml_message(self, dataset_slug: str, record_id: str, message: EmailMessage) -> Path:
        dataset_dir = self.eml_root / dataset_slug
        dataset_dir.mkdir(parents=True, exist_ok=True)
        filename = safe_slug(f"{record_id}.eml", f"{record_id}.eml")
        path = dataset_dir / filename
        path.write_bytes(message.as_bytes(policy=policy.default))
        return path

    def write_eml_bytes(self, dataset_slug: str, record_id: str, payload: bytes) -> Path:
        dataset_dir = self.eml_root / dataset_slug
        dataset_dir.mkdir(parents=True, exist_ok=True)
        filename = safe_slug(f"{record_id}.eml", f"{record_id}.eml")
        path = dataset_dir / filename
        path.write_bytes(payload)
        return path


def build_field_order(all_fields: Iterable[str]) -> List[str]:
    result: List[str] = []
    seen = set()
    for field in BASE_FIELD_ORDER:
        if field in all_fields and field not in seen:
            result.append(field)
            seen.add(field)
    for field in sorted(all_fields):
        if field not in seen:
            result.append(field)
            seen.add(field)
    return result


def build_email_message(
    subject: str,
    from_addr: str,
    to_addr: Optional[str],
    cc_addr: Optional[str],
    bcc_addr: Optional[str],
    reply_to: Optional[str],
    text_body: str,
    html_body: str,
    extra_headers: Optional[Dict[str, str]] = None,
) -> EmailMessage:
    msg = EmailMessage()
    if subject:
        msg["Subject"] = subject
    if from_addr:
        msg["From"] = from_addr
    if to_addr:
        msg["To"] = to_addr
    if cc_addr:
        msg["Cc"] = cc_addr
    if bcc_addr:
        msg["Bcc"] = bcc_addr
    if reply_to:
        msg["Reply-To"] = reply_to
    if text_body and html_body:
        msg.set_content(text_body)
        msg.add_alternative(html_body, subtype="html")
    elif html_body:
        fallback = html_to_text(html_body)
        msg.set_content(fallback or "")
        msg.add_alternative(html_body, subtype="html")
    else:
        msg.set_content(text_body or "")
    for key, value in (extra_headers or {}).items():
        if value:
            msg[key] = value
    return msg


def find_first(root: Path, filename: str) -> Optional[Path]:
    matches = sorted(root.rglob(filename))
    return matches[0] if matches else None


def process_s4_csv_files(data_root: Path, collector: RecordCollector, limit: Optional[int]) -> None:
    candidates: Sequence[Tuple[str, str, str]] = (
        ("S4-phishing.csv", "S4_csv_phishing", "phishing"),
        ("S4-data_extracted_easy_ham.csv", "S4_csv_easy_ham", "ham"),
        ("S4-data_extracted_hard_ham.csv", "S4_csv_hard_ham", "ham"),
    )
    for filename, slug, label in candidates:
        path = find_first(data_root, filename)
        if not path:
            LOGGER.warning("Skipping %s (file not found).", filename)
            continue
        LOGGER.info("Processing %s", path)
        with path.open("r", encoding=DEFAULT_ENCODING, errors="replace", newline="") as handle:
            reader = csv.DictReader(handle)
            for idx, row in enumerate(reader, start=1):
                if limit and idx > limit:
                    LOGGER.info("Limit reached for %s; stopping at %d rows.", filename, limit)
                    break
                record_id = f"{slug}-{idx:06d}"
                body = row.get("Body", "") or ""
                text_body, html_body = coerce_bodies(body)
                subject = (row.get("Subject") or "").strip()
                sender = (row.get("Fullname") or row.get("From") or "").strip()
                message = build_email_message(
                    subject=subject,
                    from_addr=sender or "s4-dataset@example.com",
                    to_addr=(row.get("To") or "").strip() or None,
                    cc_addr=(row.get("Cc") or "").strip() or None,
                    bcc_addr=(row.get("Bcc") or "").strip() or None,
                    reply_to=(row.get("Reply-To") or "").strip() or None,
                    text_body=text_body,
                    html_body=html_body,
                    extra_headers={
                        "Date": (row.get("Date") or "").strip(),
                        "Return-Path": (row.get("Return_Path") or "").strip(),
                        "Delivered-To": (row.get("Delivered_to") or "").strip(),
                        "Message-ID": (row.get("Message_Id") or "").strip(),
                        "X-UID": (row.get("X_uid") or "").strip(),
                        "Content-Type": (row.get("Content_type") or "").strip(),
                        "X-Source-File": path.name,
                    },
                )
                eml_path = collector.write_eml_message("S4", record_id, message)
                flattened = {
                    "source_dataset": "S4",
                    "source_detail": path.name,
                    "record_id": record_id,
                    "label": label,
                    "eml_path": relative_path(eml_path, collector.output_root),
                    "Body": body.strip(),
                    "body_text": text_body,
                    "body_html": html_body,
                    "Fullname": sender,
                }
                flattened.update({key: value for key, value in row.items() if value})
                collector.register(flattened)


def parse_s4_template_file(path: Path) -> List[Tuple[str, str]]:
    text = path.read_text(encoding=DEFAULT_ENCODING, errors="replace")
    pattern = re.compile(r"(sample\d+)\s*:\s*\"(.*?)\",", re.DOTALL)
    return [(match.group(1), html.unescape(match.group(2))) for match in pattern.finditer(text)]


def process_s4_templates(path: Path, collector: RecordCollector, limit: Optional[int]) -> None:
    entries = parse_s4_template_file(path)
    LOGGER.info("Processing %s (%d templates)", path, len(entries))
    for idx, (key, payload) in enumerate(entries, start=1):
        if limit and idx > limit:
            LOGGER.info("Limit reached for %s at template %d", path.name, limit)
            break
        subject, body = split_subject_and_body(payload)
        text_body, html_body = coerce_bodies(body or payload)
        record_id = f"{path.stem}-{key}"
        message = build_email_message(
            subject=subject or f"{key} template",
            from_addr="template@s4.local",
            to_addr=None,
            cc_addr=None,
            bcc_addr=None,
            reply_to=None,
            text_body=text_body or payload.strip(),
            html_body=html_body,
            extra_headers={"X-Template-Key": key, "X-Source-File": path.name},
        )
        eml_path = collector.write_eml_message("S4Templates", record_id, message)
        collector.register(
            {
                "source_dataset": "S4",
                "source_detail": path.name,
                "record_id": record_id,
                "label": "phishing_template",
                "template_key": key,
                "Subject": subject,
                "Body": payload.strip(),
                "body_text": text_body or payload.strip(),
                "body_html": html_body,
                "eml_path": relative_path(eml_path, collector.output_root),
            }
        )


def parse_outlook_samples(path: Path) -> List[Tuple[str, str]]:
    text = path.read_text(encoding=DEFAULT_ENCODING, errors="replace")
    pattern = re.compile(r"header(\d*)\s*:\s*\"(.*?)\"\s*,\s*rawdata\1\s*:\s*\"(.*?)\"", re.DOTALL)
    results: List[Tuple[str, str]] = []
    for match in pattern.finditer(text):
        subject = html.unescape(match.group(2))
        body = html.unescape(match.group(3))
        suffix = match.group(1) or "1"
        key = f"outlook_{suffix}"
        results.append((key, subject.strip(), body.strip()))
    return [(key, subject, body) for key, subject, body in results]


def process_outlook_samples(path: Path, collector: RecordCollector, limit: Optional[int]) -> None:
    entries = parse_outlook_samples(path)
    LOGGER.info("Processing %s (%d outlook samples)", path, len(entries))
    for idx, (key, subject, body) in enumerate(entries, start=1):
        if limit and idx > limit:
            LOGGER.info("Limit reached for %s at entry %d", path.name, limit)
            break
        text_body, html_body = coerce_bodies(body)
        record_id = f"{path.stem}-{key}"
        message = build_email_message(
            subject=subject or f"Outlook sample {key}",
            from_addr="outlook-sample@s4.local",
            to_addr=None,
            cc_addr=None,
            bcc_addr=None,
            reply_to=None,
            text_body=text_body or body,
            html_body=html_body,
            extra_headers={"X-Template-Key": key, "X-Source-File": path.name},
        )
        eml_path = collector.write_eml_message("S4Outlook", record_id, message)
        collector.register(
            {
                "source_dataset": "S4",
                "source_detail": path.name,
                "record_id": record_id,
                "label": "phishing_template",
                "template_key": key,
                "Subject": subject,
                "Body": body,
                "body_text": text_body or body,
                "body_html": html_body,
                "eml_path": relative_path(eml_path, collector.output_root),
            }
        )


def process_s4_ephish(path: Path, collector: RecordCollector, limit: Optional[int]) -> None:
    try:
        data = json.loads(path.read_text(encoding=DEFAULT_ENCODING, errors="replace"))
    except json.JSONDecodeError as exc:
        LOGGER.error("Failed to parse %s: %s", path, exc)
        return
    LOGGER.info("Processing %s (%d records)", path, len(data))
    label_map = {1: "phishing", 0: "benign"}
    for idx, entry in enumerate(data, start=1):
        if limit and idx > limit:
            LOGGER.info("Limit reached for %s at row %d", path.name, limit)
            break
        subject = entry.get("Subject", "").strip()
        body = entry.get("Body", "").strip()
        text_body, html_body = coerce_bodies(body)
        record_id = f"{path.stem}-{idx:06d}"
        label = label_map.get(entry.get("type"), "unknown")
        language = entry.get("Language", "")
        message = build_email_message(
            subject=subject or f"S4 ephish sample {idx}",
            from_addr="s4-ephish@dataset.local",
            to_addr=None,
            cc_addr=None,
            bcc_addr=None,
            reply_to=None,
            text_body=text_body or body,
            html_body=html_body,
            extra_headers={
                "X-Source-File": path.name,
                "X-S4-Language": language,
            },
        )
        eml_path = collector.write_eml_message("S4ePhish", record_id, message)
        row = {
            "source_dataset": "S4",
            "source_detail": path.name,
            "record_id": record_id,
            "label": label,
            "Language": language,
            "Subject": subject,
            "Body": body,
            "body_text": text_body or body,
            "body_html": html_body,
            "type": entry.get("type", ""),
            "eml_path": relative_path(eml_path, collector.output_root),
        }
        collector.register(row)


def process_5ham_archive(data_root: Path, collector: RecordCollector, limit: Optional[int]) -> None:
    zip_path = find_first(data_root, "5Ham.zip")
    if not zip_path:
        LOGGER.warning("5Ham.zip was not found under %s; skipping 5Ham dataset.", data_root)
        return
    LOGGER.info("Processing %s", zip_path)
    parser = BytesParser(policy=policy.default)
    processed = 0
    with zipfile.ZipFile(zip_path) as archive:
        for info in archive.infolist():
            if info.is_dir() or not info.filename.lower().endswith(".eml"):
                continue
            processed += 1
            if limit and processed > limit:
                LOGGER.info("Limit reached for 5Ham at %d files.", limit)
                break
            raw_bytes = archive.read(info)
            record_id = safe_slug(Path(info.filename).stem, f"5Ham-{processed:06d}")
            eml_path = collector.write_eml_bytes("5Ham", record_id, raw_bytes)
            try:
                message = parser.parsebytes(raw_bytes)
            except Exception as exc:  # pragma: no cover - defensive
                LOGGER.warning("Failed to parse %s: %s", info.filename, exc)
                continue
            text_body, html_body = extract_message_bodies(message)
            record = {
                "source_dataset": "5Ham",
                "source_detail": zip_path.name,
                "record_id": record_id,
                "label": "ham",
                "From": message.get("From", ""),
                "To": message.get("To", ""),
                "Cc": message.get("Cc", ""),
                "Bcc": message.get("Bcc", ""),
                "Reply-To": message.get("Reply-To", ""),
                "Subject": message.get("Subject", ""),
                "Date": message.get("Date", ""),
                "Message_Id": message.get("Message-ID", ""),
                "Body": text_body or html_body,
                "body_text": text_body,
                "body_html": html_body,
                "Raw_Headers": format_header_block(message.items()),
                "eml_path": relative_path(eml_path, collector.output_root),
            }
            collector.register(record)
    LOGGER.info("Finished 5Ham ingestion (%d emails).", processed)


def extract_message_bodies(message: EmailMessage) -> Tuple[str, str]:
    text_parts: List[str] = []
    html_parts: List[str] = []
    if message.is_multipart():
        for part in message.walk():
            if part.get_content_maintype() == "multipart":
                continue
            if (part.get_content_disposition() or "").lower() == "attachment":
                continue
            content_type = part.get_content_type()
            charset = part.get_content_charset() or DEFAULT_ENCODING
            try:
                payload = part.get_content()
            except Exception:
                payload_bytes = part.get_payload(decode=True) or b""
                payload = payload_bytes.decode(charset, errors="replace")
            if content_type == "text/html":
                html_parts.append(payload.strip())
            else:
                text_parts.append(payload.strip())
    else:
        payload = message.get_content()
        if message.get_content_type() == "text/html":
            html_parts.append(payload.strip())
        else:
            text_parts.append(payload.strip())
    text_body = "\n\n".join(filter(None, text_parts))
    html_body = "\n\n".join(filter(None, html_parts))
    if not text_body and html_body:
        text_body = html_to_text(html_body)
    return text_body, html_body


def process_legitphish_archive(data_root: Path, collector: RecordCollector, limit: Optional[int]) -> None:
    zip_path = find_first(data_root, "LegitPhish Dataset.zip")
    if not zip_path:
        LOGGER.warning("LegitPhish Dataset.zip missing under %s; skipping dataset.", data_root)
        return
    LOGGER.info("Processing %s", zip_path)
    processed = 0
    with zipfile.ZipFile(zip_path) as archive:
        csv_names = [name for name in archive.namelist() if name.lower().endswith(".csv")]
        if not csv_names:
            LOGGER.warning("No CSV found inside %s", zip_path)
            return
        csv_name = csv_names[0]
        with archive.open(csv_name, "r") as raw_handle:
            text_handle = io.TextIOWrapper(raw_handle, encoding=DEFAULT_ENCODING, errors="replace", newline="")
            reader = csv.DictReader(text_handle)
            for idx, row in enumerate(reader, start=1):
                processed += 1
                if limit and processed > limit:
                    LOGGER.info("Limit reached for LegitPhish at %d rows.", limit)
                    break
                record_id = f"LegitPhish-{idx:06d}"
                body_lines = [f"{key}: {value}" for key, value in row.items() if value]
                body_text = "\n".join(body_lines)
                text_body, html_body = coerce_bodies(body_text)
                class_label = row.get("ClassLabel", "")
                try:
                    label = "phishing" if int(float(class_label)) == 1 else "legit"
                except (ValueError, TypeError):
                    label = "unknown"
                message = build_email_message(
                    subject=f"LegitPhish sample #{idx}",
                    from_addr="legitphish@dataset.local",
                    to_addr="recipient@example.com",
                    cc_addr=None,
                    bcc_addr=None,
                    reply_to=None,
                    text_body=text_body or body_text,
                    html_body=html_body,
                    extra_headers={"X-Source-File": csv_name},
                )
                eml_path = collector.write_eml_message("LegitPhish", record_id, message)
                row_copy = {key: value for key, value in row.items()}
                row_copy.update(
                    {
                        "source_dataset": "LegitPhish",
                        "source_detail": csv_name,
                        "record_id": record_id,
                        "label": label,
                        "Body": body_text,
                        "body_text": body_text,
                        "body_html": html_body,
                        "eml_path": relative_path(eml_path, collector.output_root),
                    }
                )
                collector.register(row_copy)
    LOGGER.info("Finished LegitPhish ingestion (%d rows).", processed)


def ensure_inputs_exist(data_root: Path) -> None:
    if not data_root.exists():
        raise FileNotFoundError(f"Data root {data_root} does not exist.")


def maybe_clean_output_dir(output_root: Path, clean: bool) -> None:
    if clean and output_root.exists():
        LOGGER.info("Removing existing output directory %s", output_root)
        shutil.rmtree(output_root)
    output_root.mkdir(parents=True, exist_ok=True)


def main() -> None:
    args = parse_args()
    configure_logging(args.log_level)
    ensure_inputs_exist(args.data_root)
    maybe_clean_output_dir(args.output_root, args.clean_output)
    collector = RecordCollector(args.output_root, args.eml_subdir)
    collector.set_csv_path(args.csv_name)
    collector.eml_root.mkdir(parents=True, exist_ok=True)

    process_s4_csv_files(args.data_root, collector, args.limit)

    s4_emails_path = find_first(args.data_root, "S4-emails.json")
    if s4_emails_path:
        process_s4_templates(s4_emails_path, collector, args.limit)
    else:
        LOGGER.warning("S4-emails.json not located under %s.", args.data_root)

    s4_outlook_path = find_first(args.data_root, "S4-outlook-samples.json")
    if s4_outlook_path:
        process_outlook_samples(s4_outlook_path, collector, args.limit)
    else:
        LOGGER.warning("S4-outlook-samples.json not located under %s.", args.data_root)

    s4_ephish_path = find_first(args.data_root, "S4-ephishLLM.json")
    if s4_ephish_path:
        process_s4_ephish(s4_ephish_path, collector, args.limit)
    else:
        LOGGER.warning("S4-ephishLLM.json not located under %s.", args.data_root)

    process_5ham_archive(args.data_root, collector, args.limit)
    process_legitphish_archive(args.data_root, collector, args.limit)

    collector.write_csv()
    LOGGER.info("Done.")


if __name__ == "__main__":
    main()
