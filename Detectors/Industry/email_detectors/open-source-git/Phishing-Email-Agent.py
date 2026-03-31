#!/usr/bin/env python3
"""Batch runner for the official Phishing-Email-Agent Flask backend."""

import argparse
import csv
import json
import subprocess
import sys
import time
import urllib.error
import urllib.request
from datetime import datetime
from pathlib import Path


SCRIPT_DIR = Path(__file__).resolve().parent
MODEL_NAME = "Phishing-Email-Agent"
DEFAULT_REPO_DIR = SCRIPT_DIR / "Phishing-Email-Agent"
DEFAULT_INPUT_CSV = (
    Path("/scratch3/che489/FC-W2-SoK/ssc_LLM_phishing_survey")
    / "Datasets/sublist/S5-Personalization for Credibility/LLM-P.csv"
)
DEFAULT_BACKEND_ROOT = "http://127.0.0.1:5000"
DEFAULT_SAMPLE_SIZE = 20
FEATURE_COLUMNS = [
    "digit_count",
    "domain_age",
    "domain_length",
    "email_length",
    "has_attachment",
    "html_tags",
    "hyphen_count",
    "link_density",
    "links_count",
    "sender_domain",
    "special_chars",
    "subdomain_count",
    "subject_length",
    "urgent_keywords",
]


def detect_default_python():
    preferred = Path("/scratch3/che489/Ha/.conda/envs/FC-W2-gpu-p39/bin/python")
    if preferred.exists():
        return str(preferred)
    return sys.executable


def sanitize_name(value):
    cleaned = "".join(ch if ch.isalnum() or ch in ("-", "_") else "_" for ch in str(value))
    while "__" in cleaned:
        cleaned = cleaned.replace("__", "_")
    return cleaned.strip("_") or "run"


def now_stamp():
    return datetime.now().strftime("%Y%m%d_%H%M%S")


def parse_args():
    parser = argparse.ArgumentParser(
        description="Batch-run the official Phishing-Email-Agent Flask backend and export a compact CSV summary."
    )
    parser.add_argument("--repo-dir", type=Path, default=DEFAULT_REPO_DIR)
    parser.add_argument("--python-bin", default=detect_default_python())
    parser.add_argument("--input-csv", type=Path, default=DEFAULT_INPUT_CSV)
    parser.add_argument("--subject-column", default="Subject")
    parser.add_argument("--body-column", default="Body")
    parser.add_argument("--sample-size", type=int, default=DEFAULT_SAMPLE_SIZE)
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--backend-root", default=DEFAULT_BACKEND_ROOT)
    parser.add_argument("--startup-timeout-seconds", type=int, default=45)
    parser.add_argument("--request-timeout-seconds", type=int, default=60)
    parser.add_argument("--auto-start-backend", action="store_true")
    return parser.parse_args()


def infer_dataset_name(input_csv):
    return sanitize_name("{}__{}".format(input_csv.parent.name, input_csv.stem))


def build_output_dir(args):
    if args.output_dir:
        return args.output_dir.resolve()
    run_name = "{}__{}__{}".format(infer_dataset_name(args.input_csv), MODEL_NAME, now_stamp())
    return (SCRIPT_DIR / "runs" / run_name).resolve()


def http_get_json(url, timeout_seconds):
    req = urllib.request.Request(url, method="GET")
    with urllib.request.urlopen(req, timeout=timeout_seconds) as response:
        body = response.read().decode("utf-8", errors="replace")
        return response.status, body


def backend_ready(backend_root, timeout_seconds):
    try:
        status, _ = http_get_json(backend_root.rstrip("/") + "/", timeout_seconds)
        return 200 <= status < 500
    except Exception:
        return False


def start_backend(args):
    process = subprocess.Popen(
        [args.python_bin, "app.py"],
        cwd=str(args.repo_dir),
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )
    deadline = time.time() + args.startup_timeout_seconds
    while time.time() < deadline:
        if backend_ready(args.backend_root, 3):
            return process
        if process.poll() is not None:
            raise SystemExit("Backend exited before becoming ready.")
        time.sleep(1)
    process.terminate()
    raise SystemExit("Timed out waiting for backend: {}".format(args.backend_root))


def post_predict(backend_root, payload, timeout_seconds):
    url = backend_root.rstrip("/") + "/predict"
    data = json.dumps(payload).encode("utf-8")
    request = urllib.request.Request(
        url,
        data=data,
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    try:
        with urllib.request.urlopen(request, timeout=timeout_seconds) as response:
            body = response.read().decode("utf-8", errors="replace")
            return response.status, json.loads(body)
    except urllib.error.HTTPError as exc:
        body = exc.read().decode("utf-8", errors="replace")
        try:
            payload = json.loads(body)
        except Exception:
            payload = {"error": body}
        return exc.code, payload


def write_manifest(path, payload):
    with path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, ensure_ascii=False)


def normalize_text(value):
    return (value or "").replace("\r\n", "\n").replace("\r", "\n").strip()


def main():
    args = parse_args()
    if not args.repo_dir.exists():
        raise SystemExit("Repo directory not found: {}".format(args.repo_dir))
    if not args.input_csv.exists():
        raise SystemExit("Input CSV not found: {}".format(args.input_csv))

    output_dir = build_output_dir(args)
    summary_dir = output_dir / "summary"
    output_dir.mkdir(parents=True, exist_ok=True)
    summary_dir.mkdir(parents=True, exist_ok=True)

    backend_process = None
    backend_started_by_script = False
    if not backend_ready(args.backend_root, 3):
        if not args.auto_start_backend:
            raise SystemExit("Backend is not running at {}.".format(args.backend_root))
        backend_process = start_backend(args)
        backend_started_by_script = True

    output_csv = summary_dir / "Phishing-Email-Agent_summary.csv"
    manifest_path = output_dir / "run_manifest.json"
    dataset_name = infer_dataset_name(args.input_csv)

    fieldnames = [
        "row_number",
        "source_path",
        "source_row_number",
        "dataset_name",
        "subject",
        "label",
        "data_source",
        "prediction",
        "probability",
        "confidence",
    ] + FEATURE_COLUMNS + [
        "http_status",
        "response_error",
    ]

    rows = []
    total_rows = 0
    try:
        with args.input_csv.open("r", encoding="utf-8-sig", newline="") as handle:
            reader = csv.DictReader(handle)
            if not reader.fieldnames:
                raise SystemExit("CSV has no header row: {}".format(args.input_csv))
            missing = [name for name in [args.subject_column, args.body_column] if name not in reader.fieldnames]
            if missing:
                raise SystemExit(
                    "CSV missing required columns {} in {}".format(", ".join(missing), args.input_csv)
                )

            for index, record in enumerate(reader, start=1):
                if args.sample_size > 0 and index > args.sample_size:
                    break
                total_rows += 1
                subject = normalize_text(record.get(args.subject_column, ""))
                body = normalize_text(record.get(args.body_column, ""))
                payload = {
                    "email_text": body,
                    "subject": subject,
                }
                status_code, response_json = post_predict(
                    args.backend_root,
                    payload,
                    args.request_timeout_seconds,
                )
                features = response_json.get("features_used", {}) if isinstance(response_json, dict) else {}
                row = {
                    "row_number": index,
                    "source_path": str(args.input_csv),
                    "source_row_number": index,
                    "dataset_name": dataset_name,
                    "subject": subject,
                    "label": record.get("label", ""),
                    "data_source": record.get("data_source", ""),
                    "prediction": response_json.get("prediction", "") if isinstance(response_json, dict) else "",
                    "probability": response_json.get("probability", "") if isinstance(response_json, dict) else "",
                    "confidence": response_json.get("confidence", "") if isinstance(response_json, dict) else "",
                    "http_status": status_code,
                    "response_error": response_json.get("error", "") if isinstance(response_json, dict) else "",
                }
                for feature_name in FEATURE_COLUMNS:
                    row[feature_name] = features.get(feature_name, "")
                rows.append(row)
                print("[{}/?] row {} -> {}".format(index, index, row["prediction"] or "NO_PREDICTION"))
    finally:
        if backend_process is not None:
            backend_process.terminate()
            try:
                backend_process.wait(timeout=10)
            except subprocess.TimeoutExpired:
                backend_process.kill()

    with output_csv.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    write_manifest(manifest_path, {
        "timestamp": datetime.utcnow().isoformat() + "Z",
        "repo_dir": str(args.repo_dir),
        "python_bin": args.python_bin,
        "backend_root": args.backend_root,
        "backend_started_by_script": backend_started_by_script,
        "input_mode": "csv",
        "input_csv": str(args.input_csv),
        "dataset_name": dataset_name,
        "model_name": MODEL_NAME,
        "sample_size": total_rows,
        "output_csv": str(output_csv),
        "output_dir": str(output_dir),
        "summary_dir": str(summary_dir),
        "saved_artifacts": [
            str(manifest_path),
            str(output_csv),
        ],
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
