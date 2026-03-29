#!/usr/bin/env python3
"""Phishing detection using Azure Content Safety blocklists with scan-like outputs."""

from __future__ import annotations

import argparse
import csv
import enum
import json
import os
import sys
import time
from pathlib import Path
from typing import Any, Union

import requests


PHISHING_BLOCKLIST_NAME = "phishing-detection"
PHISHING_BLOCKLIST_ITEMS = [
    "verify your account",
    "confirm your account",
    "confirm your password",
    "reset your password",
    "update your password",
    "change your password",
    "log in to your account",
    "login to your account",
    "sign in to your account",
    "validate your identity",
    "validate your mailbox",
    "reauthenticate your account",
    "verify your identity",
    "confirm your login",
    "check your account status",
    "unlock your account",
    "recover your account",
    "your account needs verification",
    "your mailbox needs verification",
    "account verification required",
    "office 365 team",
    "microsoft security",
    "microsoft support",
    "outlook support",
    "it helpdesk",
    "technical support",
    "system administrator",
    "security team",
    "mail administrator",
    "email administrator",
    "payroll department",
    "finance department",
    "human resources",
    "hr department",
    "accounts department",
    "customer support",
    "service desk",
    "ceo request",
    "executive office",
    "internal security team",
    "wire transfer",
    "bank transfer",
    "gift card",
    "payment details",
    "overdue invoice",
    "invoice attached",
    "payment failed",
    "refund pending",
    "tax refund",
    "outstanding payment",
    "billing update",
    "payment confirmation",
    "remittance advice",
    "salary adjustment",
    "payroll update",
    "crypto wallet",
    "wallet verification",
    "purchase order",
    "urgent payment",
    "vendor payment",
    "urgent",
    "immediately",
    "asap",
    "act now",
    "action required",
    "final warning",
    "within 24 hours",
    "today",
    "failure to respond",
    "your account will be closed",
    "your mailbox will be suspended",
    "avoid suspension",
    "limited time",
    "deadline today",
    "respond immediately",
    "security alert",
    "important notice",
    "unusual activity detected",
    "unauthorized login attempt",
    "verify now",
    "click the link below",
    "click here to verify",
    "open the secure message",
    "review the shared file",
    "download attachment",
    "download the document",
    "view document",
    "view invoice",
    "open attachment",
    "see attached",
    "access the file",
    "use the secure link",
    "confirm using the link below",
    "follow the link",
    "open the document",
    "review the payment link",
    "sign in using the link below",
    "visit the secure portal",
    "open the secure portal",
    "check the attached file",
]

DEFAULT_BLOCKLIST_NAMES = [PHISHING_BLOCKLIST_NAME]


class MediaType(enum.Enum):
    Text = 1
    Image = 2


class Category(enum.Enum):
    Hate = 1
    SelfHarm = 2
    Sexual = 3
    Violence = 4


class Action(enum.Enum):
    Accept = 1
    Reject = 2


class DetectionError(Exception):
    def __init__(self, code: str, message: str) -> None:
        self.code = code
        self.message = message

    def __repr__(self) -> str:
        return f"DetectionError(code={self.code}, message={self.message})"


class Decision:
    def __init__(self, suggested_action: Action, action_by_category: dict[Category, Action]) -> None:
        self.suggested_action = suggested_action
        self.action_by_category = action_by_category


class ContentSafety:
    def __init__(self, endpoint: str, subscription_key: str, api_version: str) -> None:
        self.endpoint = endpoint.rstrip("/")
        self.subscription_key = subscription_key
        self.api_version = api_version

    def build_url(self, media_type: MediaType) -> str:
        if media_type == MediaType.Text:
            return f"{self.endpoint}/contentsafety/text:analyze?api-version={self.api_version}"
        if media_type == MediaType.Image:
            return f"{self.endpoint}/contentsafety/image:analyze?api-version={self.api_version}"
        raise ValueError(f"Invalid Media Type {media_type}")

    def build_headers(self) -> dict[str, str]:
        return {
            "Ocp-Apim-Subscription-Key": self.subscription_key,
            "Content-Type": "application/json",
        }

    def build_request_body(self, media_type: MediaType, content: str, blocklists: list[str]) -> dict[str, Any]:
        if media_type == MediaType.Text:
            return {
                "text": content,
                "blocklistNames": blocklists,
            }
        if media_type == MediaType.Image:
            return {"image": {"content": content}}
        raise ValueError(f"Invalid Media Type {media_type}")

    def detect(self, media_type: MediaType, content: str, blocklists: list[str] | None = None) -> dict[str, Any]:
        url = self.build_url(media_type)
        headers = self.build_headers()
        request_body = self.build_request_body(media_type, content, blocklists or [])
        payload = json.dumps(request_body)

        response = requests.post(url, headers=headers, data=payload, timeout=60)
        res_content = response.json()

        if response.status_code != 200:
            raise DetectionError(
                res_content["error"]["code"],
                res_content["error"]["message"],
            )

        return res_content

    def get_detect_result_by_category(self, category: Category, detect_result: dict[str, Any]) -> Union[dict[str, Any], None]:
        category_res = detect_result.get("categoriesAnalysis", None)
        if not category_res:
            return None

        for res in category_res:
            if category.name == res.get("category", None):
                return res
        raise ValueError(f"Invalid Category {category}")

    def make_decision(self, detection_result: dict[str, Any], reject_thresholds: dict[Category, int]) -> Decision:
        action_result: dict[Category, Action] = {}
        final_action = Action.Accept

        for category, threshold in reject_thresholds.items():
            if threshold not in (-1, 0, 2, 4, 6):
                raise ValueError("RejectThreshold can only be in (-1, 0, 2, 4, 6)")

            cate_detect_res = self.get_detect_result_by_category(category, detection_result)
            if cate_detect_res is None or "severity" not in cate_detect_res:
                raise ValueError(f"Can not find detection result for {category}")

            severity = cate_detect_res["severity"]
            action = Action.Reject if threshold != -1 and severity >= threshold else Action.Accept
            action_result[category] = action
            if action.value > final_action.value:
                final_action = action

        if detection_result.get("blocklistsMatch"):
            final_action = Action.Reject

        return Decision(final_action, action_result)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Run phishing-oriented detection with Azure Content Safety blocklists. "
            "Output format and CSV logging are aligned with the original scan script."
        )
    )
    parser.add_argument("--text", help="Text to scan. If omitted, the script reads from stdin.")
    parser.add_argument("--input-csv", type=Path, help="Input CSV file for batch scanning.")
    parser.add_argument(
        "--output-csv",
        type=Path,
        help="Output CSV file path for batch scanning. Defaults to <input>.phishing_detection.csv.",
    )
    parser.add_argument(
        "--text-column",
        default="text",
        help="CSV column name containing the text to scan. Default: text",
    )
    parser.add_argument(
        "--endpoint",
        default=os.getenv("CONTENT_SAFETY_ENDPOINT") or os.getenv("AZURE_CONTENT_SAFETY_API_ENDPOINT"),
        help="Content Safety endpoint. Defaults to CONTENT_SAFETY_ENDPOINT or AZURE_CONTENT_SAFETY_API_ENDPOINT.",
    )
    parser.add_argument(
        "--api-key",
        default=os.getenv("CONTENT_SAFETY_KEY") or os.getenv("AZURE_CONTENT_SAFETY_API_KEY"),
        help="Content Safety API key. Defaults to CONTENT_SAFETY_KEY or AZURE_CONTENT_SAFETY_API_KEY.",
    )
    parser.add_argument(
        "--api-version",
        default="2024-09-01",
        help="Content Safety API version. Default: 2024-09-01",
    )
    parser.add_argument(
        "--blocklists",
        nargs="+",
        default=None,
        help=(
            "Blocklist names already created in Azure Content Safety. "
            "If omitted, the script uses the built-in phishing blocklist blueprint names."
        ),
    )
    return parser.parse_args()


def read_text(cli_text: str | None) -> str:
    if cli_text:
        return cli_text

    if not sys.stdin.isatty():
        piped_text = sys.stdin.read().strip()
        if piped_text:
            return piped_text

    raise SystemExit("No text provided. Use --text or pipe text into the script.")


def resolve_output_csv_path(input_csv: Path, output_csv: Path | None) -> Path:
    if output_csv:
        return output_csv
    return input_csv.with_name(f"{input_csv.stem}.phishing_detection.csv")


def build_content_safety(args: argparse.Namespace) -> ContentSafety:
    if not args.endpoint:
        raise SystemExit(
            "Missing Content Safety endpoint. Set CONTENT_SAFETY_ENDPOINT or pass --endpoint."
        )
    if not args.api_key:
        raise SystemExit(
            "Missing Content Safety API key. Set CONTENT_SAFETY_KEY or pass --api-key."
        )

    return ContentSafety(args.endpoint, args.api_key, args.api_version)


def resolve_blocklists(blocklists: list[str] | None) -> list[str]:
    if blocklists:
        return blocklists

    env_blocklists = os.getenv("CONTENT_SAFETY_BLOCKLISTS", "").strip()
    if env_blocklists:
        return [item.strip() for item in env_blocklists.split(",") if item.strip()]

    return DEFAULT_BLOCKLIST_NAMES.copy()


def serialize_decision(decision: Decision) -> dict[str, Any]:
    return {
        "suggested_action": decision.suggested_action.name,
        "action_by_category": {
            category.name: action.name for category, action in decision.action_by_category.items()
        },
    }


def build_detection_summary(
    *,
    content_safety: ContentSafety,
    text: str,
    blocklists: list[str],
) -> dict[str, Any]:
    detection_result = content_safety.detect(MediaType.Text, text, blocklists)
    reject_thresholds = {
        Category.Hate: 4,
        Category.SelfHarm: 4,
        Category.Sexual: 4,
        Category.Violence: 4,
    }
    decision_result = content_safety.make_decision(detection_result, reject_thresholds)

    return {
        "blocklists_used": blocklists,
        "phishing_blocklist_blueprint": {
            PHISHING_BLOCKLIST_NAME: PHISHING_BLOCKLIST_ITEMS,
        },
        "detection_result": detection_result,
        "decision": serialize_decision(decision_result),
    }


def detect_single_text(text: str, content_safety: ContentSafety, blocklists: list[str]) -> dict[str, Any]:
    started_at = time.perf_counter()
    results = build_detection_summary(content_safety=content_safety, text=text, blocklists=blocklists)
    elapsed_seconds = time.perf_counter() - started_at

    return {
        "input_text": text,
        "elapsed_seconds": round(elapsed_seconds, 6),
        "results": results,
    }


def scan_csv(
    *,
    input_csv: Path,
    output_csv: Path,
    text_column: str,
    content_safety: ContentSafety,
    blocklists: list[str],
) -> dict[str, Any]:
    if not input_csv.exists():
        raise SystemExit(f"Input CSV not found: {input_csv}")

    with input_csv.open("r", encoding="utf-8-sig", newline="") as infile:
        reader = csv.DictReader(infile)
        if not reader.fieldnames:
            raise SystemExit(f"CSV has no header row: {input_csv}")
        if text_column not in reader.fieldnames:
            available = ", ".join(reader.fieldnames)
            raise SystemExit(
                f"CSV column '{text_column}' not found in {input_csv}. Available columns: {available}"
            )

        results_column = "phishing_detection_results_json"
        elapsed_column = "phishing_detection_elapsed_seconds"
        error_column = "phishing_detection_error"
        fieldnames = list(reader.fieldnames)
        for extra_column in [results_column, elapsed_column, error_column]:
            if extra_column not in fieldnames:
                fieldnames.append(extra_column)

        row_count = 0
        success_count = 0

        with output_csv.open("w", encoding="utf-8", newline="") as outfile:
            writer = csv.DictWriter(outfile, fieldnames=fieldnames)
            writer.writeheader()

            for row in reader:
                row_count += 1
                text = (row.get(text_column) or "").strip()
                row[results_column] = ""
                row[elapsed_column] = ""
                row[error_column] = ""

                if not text:
                    row[error_column] = f"Empty text in column '{text_column}'"
                    writer.writerow(row)
                    continue

                started_at = time.perf_counter()
                try:
                    results = build_detection_summary(
                        content_safety=content_safety,
                        text=text,
                        blocklists=blocklists,
                    )
                    elapsed_seconds = time.perf_counter() - started_at
                    row[results_column] = json.dumps(results, ensure_ascii=False)
                    row[elapsed_column] = f"{elapsed_seconds:.6f}"
                    success_count += 1
                except Exception as exc:
                    elapsed_seconds = time.perf_counter() - started_at
                    row[elapsed_column] = f"{elapsed_seconds:.6f}"
                    row[error_column] = str(exc)

                writer.writerow(row)

    return {
        "input_csv": str(input_csv),
        "output_csv": str(output_csv),
        "rows_total": row_count,
        "rows_scanned": success_count,
        "rows_failed": row_count - success_count,
    }


def main() -> int:
    args = parse_args()
    content_safety = build_content_safety(args)
    blocklists = resolve_blocklists(args.blocklists)

    if args.input_csv:
        summary = scan_csv(
            input_csv=args.input_csv,
            output_csv=resolve_output_csv_path(args.input_csv, args.output_csv),
            text_column=args.text_column,
            content_safety=content_safety,
            blocklists=blocklists,
        )
        print(json.dumps(summary, ensure_ascii=False, indent=2))
        return 0

    text = read_text(args.text)
    result = detect_single_text(text, content_safety, blocklists)
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
