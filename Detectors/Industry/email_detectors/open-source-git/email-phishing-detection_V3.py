#!/usr/bin/env python3
"""Batch runner for the official email-phishing-detection_V3 with AI verdict export.

This wrapper calls the upstream command exactly in the style:
  python main.py -f "<email>.eml" --ai -o "<result>.json"

It supports:
- looping over many `.eml` files
- converting CSV rows into synthetic `.eml` files first
- exporting only the short AI summary columns you asked for
"""

import argparse
import csv
import json
import os
import subprocess
import sys
import tempfile
from datetime import datetime
from email import policy
from email.message import EmailMessage
from email.parser import BytesParser
from pathlib import Path


SCRIPT_DIR = Path(__file__).resolve().parent
DEFAULT_REPO_DIR = SCRIPT_DIR / "email-phishing-detection_V3"
MODEL_NAME = "email-phishing-detection_V3"
DEFAULT_EML_DIR = (
    Path("/scratch3/che489/FC-W2-SoK/ssc_LLM_phishing_survey")
    / "Datasets/sublist/S5-Personalization for Credibility/5-HW-P_EML"
)
DEFAULT_SAMPLE_SIZE = 20
DEFAULT_FROM = "sender@example.com"
DEFAULT_TO = "recipient@example.com"


def detect_default_python():
    env_python = os.environ.get("PYTHON_BIN", "").strip()
    if env_python:
        return env_python

    preferred = Path("/scratch3/che489/Ha/.conda/envs/FC-W2-gpu-p39/bin/python")
    if preferred.exists():
        return str(preferred)
    return sys.executable


def parse_args():
    parser = argparse.ArgumentParser(
        description="Batch-run the official email-phishing-detection_V3 with --ai and export short CSV summaries."
    )
    parser.add_argument("--repo-dir", type=Path, default=DEFAULT_REPO_DIR)
    parser.add_argument("--python-bin", default=detect_default_python())
    parser.add_argument("--input-eml-dir", type=Path, default=DEFAULT_EML_DIR)
    parser.add_argument("--input-csv", type=Path, default=None)
    parser.add_argument("--subject-column", default="Subject")
    parser.add_argument("--body-column", default="Body")
    parser.add_argument("--sample-size", type=int, default=DEFAULT_SAMPLE_SIZE)
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--from-address", default=DEFAULT_FROM)
    parser.add_argument("--to-address", default=DEFAULT_TO)
    return parser.parse_args()


def now_stamp():
    return datetime.now().strftime("%Y%m%d_%H%M%S")


def sanitize_name(value):
    cleaned = "".join(ch if ch.isalnum() or ch in ("-", "_") else "_" for ch in str(value))
    while "__" in cleaned:
        cleaned = cleaned.replace("__", "_")
    return cleaned.strip("_") or "run"


def infer_dataset_name(args):
    if args.input_csv:
        return sanitize_name("{}__{}".format(args.input_csv.parent.name, args.input_csv.stem))
    return sanitize_name(args.input_eml_dir.name)


def build_output_dir(args):
    output_dir_arg = args.output_dir
    if output_dir_arg:
        return output_dir_arg.resolve()
    run_name = "{}__{}__{}".format(infer_dataset_name(args), MODEL_NAME, now_stamp())
    return (SCRIPT_DIR / "runs" / run_name).resolve()


def normalize_text(value):
    text = (value or "").replace("\r\n", "\n").replace("\r", "\n")
    return text.strip()


def read_eml_subject_body(eml_path):
    with eml_path.open("rb") as handle:
        msg = BytesParser(policy=policy.default).parse(handle)

    subject = str(msg.get("Subject", "") or "")
    text_parts = []
    html_parts = []

    if msg.is_multipart():
        for part in msg.walk():
            if part.get_content_maintype() == "multipart":
                continue
            content_type = part.get_content_type()
            try:
                payload = part.get_content()
            except Exception:
                payload = ""
            if content_type == "text/plain":
                text_parts.append(str(payload))
            elif content_type == "text/html":
                html_parts.append(str(payload))
    else:
        content_type = msg.get_content_type()
        try:
            payload = msg.get_content()
        except Exception:
            payload = ""
        if content_type == "text/html":
            html_parts.append(str(payload))
        else:
            text_parts.append(str(payload))

    return {
        "subject": normalize_text(subject),
        "body_text": normalize_text("\n\n".join(text_parts)),
        "body_html": normalize_text("\n\n".join(html_parts)),
    }


def create_synthetic_eml(out_path, subject, body, from_address, to_address):
    msg = EmailMessage()
    msg["Subject"] = subject or "(no subject)"
    msg["From"] = from_address
    msg["To"] = to_address
    msg.set_content(body or "")
    out_path.write_bytes(msg.as_bytes())


def build_sources_from_csv(args, generated_dir):
    rows = []
    with args.input_csv.open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        if not reader.fieldnames:
            raise SystemExit("CSV has no header row: {}".format(args.input_csv))
        missing = [name for name in [args.subject_column, args.body_column] if name not in reader.fieldnames]
        if missing:
            raise SystemExit(
                "CSV missing required columns {} in {}".format(", ".join(missing), args.input_csv)
            )

        for index, row in enumerate(reader, start=1):
            if index > args.sample_size:
                break
            subject = normalize_text(row.get(args.subject_column, ""))
            body = normalize_text(row.get(args.body_column, ""))
            eml_path = generated_dir / ("row_{:04d}.eml".format(index))
            create_synthetic_eml(eml_path, subject, body, args.from_address, args.to_address)
            rows.append({
                "row_number": index,
                "source_kind": "csv",
                "source_path": str(args.input_csv),
                "source_row_number": index,
                "eml_path": eml_path,
                "subject": subject,
                "body_text": body,
            })
    return rows


def build_sources_from_eml_dir(args):
    if not args.input_eml_dir.exists():
        raise SystemExit("EML directory not found: {}".format(args.input_eml_dir))

    eml_files = sorted(args.input_eml_dir.glob("*.eml"))[: args.sample_size]
    if not eml_files:
        raise SystemExit("No .eml files found in {}".format(args.input_eml_dir))

    rows = []
    for index, eml_path in enumerate(eml_files, start=1):
        parsed = read_eml_subject_body(eml_path)
        rows.append({
            "row_number": index,
            "source_kind": "eml",
            "source_path": str(eml_path),
            "source_row_number": "",
            "eml_path": eml_path,
            "subject": parsed["subject"],
            "body_text": parsed["body_text"],
        })
    return rows


def load_result_json(path):
    if not path.exists():
        return {}
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def parse_ai_summary(result_json):
    info = result_json.get("Information", {})
    analysis = result_json.get("Analysis", {})
    ai = analysis.get("AI_Analysis", {})

    brands = ai.get("identified_brands", []) if isinstance(ai, dict) else []
    suspicious_elements = ai.get("suspicious_elements", []) if isinstance(ai, dict) else []
    confidence = ai.get("confidence", "") if isinstance(ai, dict) else ""
    if confidence != "":
        try:
            confidence = round(float(confidence) * 100.0, 1)
        except Exception:
            pass

    return {
        "official_status": info.get("Status", ""),
        "official_error": result_json.get("Error", ""),
        "AI Verdict": ai.get("verdict", "") if isinstance(ai, dict) else "",
        "AI Confidence": confidence,
        "AI Phishing Score (0-10)": ai.get("phishing_score", "") if isinstance(ai, dict) else "",
        "AI Identified Brands": " | ".join(brands),
        "Key Suspicious Elements Cited by AI": " | ".join(suspicious_elements),
        "AI Explanation": ai.get("explanation", "") if isinstance(ai, dict) else "",
    }


def run_detector(repo_dir, python_bin, eml_path, result_json_path, result_log_path):
    cmd = [
        python_bin,
        "main.py",
        "-f",
        str(eml_path),
        "--ai",
        "-o",
        str(result_json_path),
    ]
    completed = subprocess.run(
        cmd,
        cwd=str(repo_dir),
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        universal_newlines=True,
    )
    with result_log_path.open("w", encoding="utf-8") as handle:
        handle.write(completed.stdout or "")
    return completed.returncode


def write_manifest(path, payload):
    with path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, ensure_ascii=False)


def main():
    args = parse_args()
    if not args.repo_dir.exists():
        raise SystemExit("Official repo directory not found: {}".format(args.repo_dir))

    output_dir = build_output_dir(args)
    summary_dir = output_dir / "summary"
    output_dir.mkdir(parents=True, exist_ok=True)
    summary_dir.mkdir(parents=True, exist_ok=True)
    output_csv = summary_dir / "email-phishing-detection_V3_ai_summary.csv"
    manifest_path = output_dir / "run_manifest.json"

    fieldnames = [
        "row_number",
        "source_kind",
        "source_path",
        "source_row_number",
        "subject",
        "AI Verdict",
        "AI Confidence",
        "AI Phishing Score (0-10)",
        "AI Identified Brands",
        "Key Suspicious Elements Cited by AI",
        "AI Explanation",
        "official_status",
        "official_error",
        "official_returncode",
    ]

    rows = []
    with tempfile.TemporaryDirectory(prefix="email_phishing_detection_v3_") as temp_dir:
        temp_dir = Path(temp_dir)
        generated_eml_dir = temp_dir / "generated_eml"
        result_json_dir = temp_dir / "result_json"
        result_json_dir.mkdir(parents=True, exist_ok=True)

        if args.input_csv:
            generated_eml_dir.mkdir(parents=True, exist_ok=True)
            sources = build_sources_from_csv(args, generated_eml_dir)
        else:
            sources = build_sources_from_eml_dir(args)

        for item in sources:
            filename = item["eml_path"].stem
            result_json_path = result_json_dir / (filename + ".json")
            result_log_path = temp_dir / (filename + ".log")
            returncode = run_detector(
                args.repo_dir,
                args.python_bin,
                item["eml_path"],
                result_json_path,
                result_log_path,
            )
            result_json = load_result_json(result_json_path)
            ai = parse_ai_summary(result_json)

            row = {
                "row_number": item["row_number"],
                "source_kind": item["source_kind"],
                "source_path": item["source_path"],
                "source_row_number": item["source_row_number"],
                "subject": item["subject"],
                "AI Verdict": ai["AI Verdict"],
                "AI Confidence": ai["AI Confidence"],
                "AI Phishing Score (0-10)": ai["AI Phishing Score (0-10)"],
                "AI Identified Brands": ai["AI Identified Brands"],
                "Key Suspicious Elements Cited by AI": ai["Key Suspicious Elements Cited by AI"],
                "AI Explanation": ai["AI Explanation"],
                "official_status": ai["official_status"],
                "official_error": ai["official_error"],
                "official_returncode": returncode,
            }
            rows.append(row)
            print("[{}/{}] {} -> {}".format(
                item["row_number"],
                len(sources),
                Path(item["eml_path"]).name,
                row["AI Verdict"] or "NO_AI_VERDICT",
            ))

    with output_csv.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    write_manifest(manifest_path, {
        "timestamp": datetime.utcnow().isoformat() + "Z",
        "repo_dir": str(args.repo_dir),
        "python_bin": args.python_bin,
        "input_mode": "csv" if args.input_csv else "eml",
        "input_eml_dir": str(args.input_eml_dir),
        "input_csv": str(args.input_csv) if args.input_csv else "",
        "sample_size": len(sources),
        "dataset_name": infer_dataset_name(args),
        "model_name": MODEL_NAME,
        "output_csv": str(output_csv),
        "output_dir": str(output_dir),
        "summary_dir": str(summary_dir),
        "saved_artifacts": [
            str(manifest_path),
            str(output_csv),
        ],
        "intermediate_artifacts_saved": False,
    })

    print(json.dumps({
        "output_csv": str(output_csv),
        "output_dir": str(output_dir),
        "manifest": str(manifest_path),
        "rows": len(rows),
    }, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
