#!/usr/bin/env python3
"""Safe process-oriented reproduction of Qi et al. (Information Fusion 2025).

Paper:
Qi et al. "SpearBot: Leveraging large language models in a generative-critique
framework for spear-phishing email generation" (Information Fusion 122, 2025;
arXiv:2412.11109).

What the paper discloses:
- 100 virtual profiles: 50 students + 50 employees
- 10 phishing strategies
- GPT-4-based data preparation for profile fields
- a multi-turn initialization before final email generation
- a multi-critic loop that regenerates until critics stop flagging the email or
  an iteration limit is reached

What this script intentionally does instead:
- reproduces the dataset recipe and generate-critique loop
- constrains output to benign, auditable personalized emails
- avoids phishing content, credential requests, fake security alerts, payments,
  and malicious links

The goal is to match the *process* from the paper as closely as is safe, rather
than reproduce harmful content.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import random
import re
import sys
import time
from dataclasses import asdict, dataclass, is_dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

try:
    from openai import APIConnectionError, APIStatusError, OpenAI
except ImportError:
    APIConnectionError = None
    APIStatusError = None
    OpenAI = None

csv.field_size_limit(10**9)


SCRIPT_DIR = Path(__file__).resolve().parent
RUNS_DIR = SCRIPT_DIR / "runs"
SIMPLIFIED_OUTPUT_NAME = "LLM-B.csv"
FULL_OUTPUT_NAME = "LLM-B.full.csv"
DEFAULT_BASE_URL = "https://api.openai.com/v1"
DEFAULT_TIMEOUT = 120
DEFAULT_DATA_SOURCE = "spearbot-safe-s5"
DEFAULT_TARGET_ROW_COUNT = 1228
DEFAULT_GPT4O_MINI_MODEL = "gpt-4o-mini"
DEFAULT_MAX_COMPLETION_TOKENS = 800

SUBJECT_RE = re.compile(r"(?im)^\s*subject(?:\s*line)?\s*[:：]\s*(.+)$")
EMAIL_BLOCK_RE = re.compile(r"<Email>(.*?)</Email>", re.IGNORECASE | re.DOTALL)
XML_TAG_RE = re.compile(r"<(?P<tag>[A-Za-z0-9_:-]+)>(?P<value>.*?)</(?P=tag)>", re.DOTALL)
CODE_FENCE_RE = re.compile(r"^```[a-zA-Z0-9_-]*\s*|\s*```$", re.MULTILINE)
NON_ALNUM_RE = re.compile(r"[^a-z0-9]+")

PAPER_REFERENCE: Dict[str, Any] = {
    "title": "SpearBot: Leveraging large language models in a generative-critique framework for spear-phishing email generation",
    "journal": "Information Fusion",
    "volume": "122",
    "year": 2025,
    "doi": "10.1016/j.inffus.2025.103176",
    "arxiv": "https://arxiv.org/abs/2412.11109",
    "published_link": "https://www.sciencedirect.com/science/article/abs/pii/S1566253525002490",
}

PAPER_STRATEGIES: Sequence[Dict[str, str]] = (
    {
        "name": "Authority Impersonation",
        "description": "Present the message with an official or authoritative tone that feels trustworthy without making deceptive claims.",
    },
    {
        "name": "Familiarity and Social Engineering",
        "description": "Use benign personal context to make the message feel relevant and tailored to the recipient.",
    },
    {
        "name": "Offering Help or Providing a Service",
        "description": "Frame the message around useful assistance, support, or a service the recipient could reasonably value.",
    },
    {
        "name": "Scarcity and Urgency",
        "description": "Use limited availability or time sensitivity carefully, without coercion or false pressure.",
    },
    {
        "name": "Curiosity Provocation",
        "description": "Spark curiosity through a preview, teaser, or interesting detail that encourages a benign click-through.",
    },
    {
        "name": "Reward and Incentive",
        "description": "Offer an ethical reward, perk, or thank-you incentive without asking for money or credentials.",
    },
    {
        "name": "Fake Notifications",
        "description": "Adapted safely as a service-style notification that remains transparent and clearly benign.",
    },
    {
        "name": "Problem or Threat Notification",
        "description": "Adapted safely as a routine issue notice with constructive next steps and no alarming deception.",
    },
    {
        "name": "Confirmation of Personal Information",
        "description": "Adapted safely as a profile review request that never asks for passwords or sensitive identifiers.",
    },
    {
        "name": "Tailored Content",
        "description": "Use the recipient's role, interests, or studies to recommend relevant content or events.",
    },
)

ROLE_SCENARIOS: Dict[str, Sequence[str]] = {
    "employee": (
        "professional development workshop",
        "internal knowledge-sharing webinar",
        "team resource digest",
        "productivity toolkit overview",
        "optional mentoring program",
        "innovation roundtable invitation",
    ),
    "student": (
        "student skills session",
        "campus research showcase",
        "career preparation workshop",
        "student club collaboration event",
        "learning resource digest",
        "alumni Q&A webinar",
    ),
}

MOCK_FIRST_NAMES: Sequence[str] = (
    "Avery",
    "Jordan",
    "Taylor",
    "Riley",
    "Morgan",
    "Harper",
    "Cameron",
    "Sydney",
    "Logan",
    "Drew",
    "Quinn",
    "Parker",
    "Skyler",
    "Casey",
    "Rowan",
    "Alex",
    "Blake",
    "Reese",
    "Jamie",
    "Kendall",
)

MOCK_LAST_NAMES: Sequence[str] = (
    "Bennett",
    "Warren",
    "Sullivan",
    "Parker",
    "Reynolds",
    "Foster",
    "Chen",
    "Morales",
    "Patel",
    "Nguyen",
    "Kim",
    "Walker",
    "Hayes",
    "Brooks",
    "Price",
    "Griffin",
    "Hughes",
    "Marshall",
    "Bishop",
    "Diaz",
)

MOCK_HOBBIES: Sequence[str] = (
    "urban sketching and design journaling",
    "building small robotics projects",
    "trail running and landscape photography",
    "community theater and script editing",
    "hosting board-game nights",
    "open-source contribution and accessibility testing",
    "botanical gardening",
    "podcast production and audio editing",
    "cooking regional dishes",
    "birdwatching and field note taking",
    "3D printing functional tools",
    "volunteering at community coding clubs",
    "reading speculative fiction",
    "cycling and route mapping",
    "film photography and darkroom work",
    "language exchange meetups",
    "woodworking and furniture restoration",
    "museum visits and art history blogging",
    "creative writing workshops",
    "public speaking clubs",
)

MOCK_EMPLOYEE_JOBS: Sequence[str] = (
    "software developer",
    "data analyst",
    "project coordinator",
    "UX researcher",
    "mechanical engineer",
    "civil engineer",
    "procurement specialist",
    "marketing operations manager",
    "customer success manager",
    "financial analyst",
    "HR business partner",
    "IT support lead",
    "sustainability consultant",
    "supply chain planner",
    "clinical operations associate",
)

MOCK_STUDENT_PROGRAMS: Sequence[str] = (
    "sophomore in biology",
    "senior in mechanical engineering",
    "master's student in computer science",
    "junior in economics",
    "final-year student in civil engineering",
    "master's student in public health",
    "doctoral student in materials science",
    "senior in industrial design",
    "junior in psychology",
    "master's student in environmental policy",
    "sophomore in media studies",
    "final-year student in data science",
    "junior in architecture",
    "master's student in education",
    "senior in finance",
)

MOCK_COMPANIES: Sequence[str] = (
    "Northbridge Analytics",
    "HarborLine Systems",
    "Crestview Manufacturing",
    "SummitPath Consulting",
    "LatticeWorks Studio",
    "BlueOak Logistics",
    "Clearwater Health Partners",
    "Granite Field Services",
    "CivicGrid Solutions",
    "Pioneer Market Insights",
)

MOCK_UNIVERSITIES: Sequence[str] = (
    "Northbridge University",
    "Harbor State University",
    "Crestview Institute of Technology",
    "Summit Valley University",
    "Lakeside College",
    "Civic Arts and Sciences University",
    "Riverton School of Engineering",
    "BlueOak University",
    "Granite Point University",
    "Clearwater Institute",
)

SAFE_CTA_DOMAINS: Sequence[str] = (
    "https://updates.example.org",
    "https://programs.example.org",
    "https://events.example.org",
    "https://resources.example.org",
)

SUSPICIOUS_PATTERNS: Sequence[Tuple[re.Pattern[str], str]] = (
    (re.compile(r"\bpassword\b", re.IGNORECASE), "mentions passwords"),
    (re.compile(r"\bcredential", re.IGNORECASE), "mentions credentials"),
    (re.compile(r"\bsocial security\b|\bssn\b", re.IGNORECASE), "asks for sensitive identifiers"),
    (re.compile(r"\bwire transfer\b|\bgift card\b|\bcrypto\b", re.IGNORECASE), "references risky payment requests"),
    (re.compile(r"\bverify (?:your|the) account\b", re.IGNORECASE), "resembles account-verification language"),
    (re.compile(r"\b(click here|act now|immediately|urgent action required)\b", re.IGNORECASE), "uses aggressive click-pressure wording"),
    (re.compile(r"\b(suspend|disabled|locked)\b", re.IGNORECASE), "uses account-threat wording"),
    (re.compile(r"http://(?!localhost)|www\.", re.IGNORECASE), "uses less trustworthy URL styling"),
    (re.compile(r"\[[A-Za-z][A-Za-z0-9 _-]{1,30}\]|\{[A-Za-z][A-Za-z0-9 _-]{1,30}\}", re.IGNORECASE), "contains placeholder-style bracket tokens"),
    (re.compile(r"\b(?:012345|123456|234567|345678|456789|567890|987654|876543|765432|654321)\b"), "uses trivial sequential digits"),
    (re.compile(r"\b(\d)\1{4,}\b"), "uses unrealistic repeated digits"),
    (re.compile(r"\b(?:xxx+|n/?a|tbd|your name here)\b", re.IGNORECASE), "contains placeholder text instead of concrete content"),
)


@dataclass
class VirtualProfile:
    profile_id: str
    role: str
    name: str
    age: int
    gender: str
    hobby: str
    job: str = ""
    company: str = ""
    educational_qualification: str = ""
    university: str = ""

    def role_descriptor(self) -> str:
        if self.role == "employee":
            return "{} at {}".format(self.job, self.company)
        return "{} at {}".format(self.educational_qualification, self.university)


@dataclass
class CriticResult:
    critic_model: str
    answer: str
    reasons: str
    raw_response: str


@dataclass
class EmailResult:
    row_id: str
    profile_id: str
    role: str
    strategy_name: str
    strategy_description: str
    subject: str
    body: str
    label: int
    data_source: str
    generator_model: str
    critic_models: str
    critique_rounds: int
    critics_passed: bool
    accepted_critic_count: int
    total_critic_count: int
    safe_link: str
    original_prompt_mode: str


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Safe process-focused reproduction of Qi et al. (2025) SpearBot."
    )
    parser.add_argument(
        "--output-dir",
        default="",
        help="Optional output directory. Defaults to S5-Personalization for Credibility/runs/<timestamp>-qi2025-safe/.",
    )
    parser.add_argument(
        "--backend",
        choices=("auto", "mock", "openai-compatible"),
        default="auto",
        help="Model backend. 'auto' uses OpenAI-compatible if a key is present, otherwise mock.",
    )
    parser.add_argument(
        "--generator-model",
        default=os.environ.get("OPENAI_MODEL", DEFAULT_GPT4O_MINI_MODEL),
        help="Generator model name for the OpenAI-compatible backend.",
    )
    parser.add_argument(
        "--critic-models",
        default=DEFAULT_GPT4O_MINI_MODEL,
        help="Comma-separated critic model names. Mock mode uses the names for logging only.",
    )
    parser.add_argument(
        "--base-url",
        default=os.environ.get("OPENAI_BASE_URL", DEFAULT_BASE_URL),
        help="OpenAI-compatible API base URL.",
    )
    parser.add_argument(
        "--api-key",
        default=os.environ.get("OPENAI_API_KEY", ""),
        help="API key. Defaults to OPENAI_API_KEY.",
    )
    parser.add_argument(
        "--profiles-per-role",
        type=int,
        default=0,
        help="How many virtual profiles to create for each role. Set 0 to auto-size from --target-row-count.",
    )
    parser.add_argument(
        "--target-row-count",
        type=int,
        default=DEFAULT_TARGET_ROW_COUNT,
        help="Target number of generated rows. Default 1228 to match the local S5 LLM-P size closely.",
    )
    parser.add_argument(
        "--strategy-limit",
        type=int,
        default=0,
        help="Optional limit for the number of strategies used from the paper's 10-strategy catalog.",
    )
    parser.add_argument(
        "--max-rows",
        type=int,
        default=0,
        help="Optional cap for the total profile-strategy combinations to generate.",
    )
    parser.add_argument(
        "--iteration-limit",
        type=int,
        default=10,
        help="Critique-regeneration iteration limit. Paper-style default: 10.",
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=0.8,
        help="Sampling temperature for generation calls.",
    )
    parser.add_argument(
        "--max-completion-tokens",
        type=int,
        default=DEFAULT_MAX_COMPLETION_TOKENS,
        help="Max completion tokens for OpenAI-compatible API calls.",
    )
    parser.add_argument(
        "--timeout",
        type=int,
        default=DEFAULT_TIMEOUT,
        help="HTTP timeout in seconds.",
    )
    parser.add_argument(
        "--sleep-seconds",
        type=float,
        default=0.0,
        help="Optional delay between API calls.",
    )
    parser.add_argument(
        "--max-retries",
        type=int,
        default=3,
        help="Retry count for transient API errors.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed used in mock mode and task ordering.",
    )
    parser.add_argument(
        "--data-source",
        default=DEFAULT_DATA_SOURCE,
        help="data_source value written to the output CSV.",
    )
    parser.add_argument(
        "--label",
        type=int,
        default=0,
        help="Label value written to the output CSV. Defaults to benign label 0.",
    )
    parser.add_argument(
        "--save-every",
        type=int,
        default=100,
        help="Write checkpoint CSVs every N completed rows.",
    )
    parser.add_argument(
        "--print-every",
        type=int,
        default=10,
        help="Progress logging interval.",
    )
    parser.add_argument(
        "--allow-real-organization-names",
        action="store_true",
        help="When using a live LLM backend, ask for real universities/companies like the paper. Default is disabled for safer reproduction.",
    )
    return parser.parse_args()


def now_utc_iso() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat()


def normalize_text(value: str) -> str:
    text = (value or "").replace("\r\n", "\n").replace("\r", "\n")
    text = text.replace("\ufeff", "").replace("\u200b", "").replace("\xa0", " ")
    text = re.sub(r"[ \t]+\n", "\n", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def strip_code_fences(text: str) -> str:
    return CODE_FENCE_RE.sub("", (text or "").strip()).strip()


def slugify(value: str) -> str:
    lowered = NON_ALNUM_RE.sub("-", (value or "").strip().lower())
    return lowered.strip("-") or "item"


def parse_tag(text: str, tag_name: str) -> str:
    pattern = re.compile(
        r"<{}>(.*?)</{}>".format(re.escape(tag_name), re.escape(tag_name)),
        re.IGNORECASE | re.DOTALL,
    )
    match = pattern.search(text or "")
    if not match:
        return ""
    return normalize_text(match.group(1))


def render_email(subject: str, body: str) -> str:
    subject_clean = normalize_text(subject)
    body_clean = normalize_text(body)
    if subject_clean:
        return "Subject: {}\n\n{}".format(subject_clean, body_clean).strip()
    return body_clean


def extract_email_payload(text: str) -> Tuple[str, str]:
    cleaned = strip_code_fences(text)
    block_match = EMAIL_BLOCK_RE.search(cleaned)
    payload = block_match.group(1) if block_match else cleaned
    payload = normalize_text(payload)
    subject_match = SUBJECT_RE.search(payload)
    if not subject_match:
        fallback_subject = "Personalized Update"
        return fallback_subject, payload
    subject = normalize_text(subject_match.group(1))
    start, end = subject_match.span()
    body = normalize_text((payload[:start] + payload[end:]).strip())
    return subject, body


def parse_critic_response(text: str) -> Tuple[str, str]:
    cleaned = strip_code_fences(text)
    answer = parse_tag(cleaned, "Answer").lower()
    reasons = parse_tag(cleaned, "Reasons")
    if not answer:
        lowered = cleaned.lower()
        if re.search(r"\byes\b", lowered):
            answer = "yes"
        elif re.search(r"\bno\b", lowered):
            answer = "no"
    answer = "yes" if answer.startswith("y") else "no"
    reasons = reasons or normalize_text(cleaned)
    return answer, reasons


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, ensure_ascii=False, indent=2, default=json_default)


def append_jsonl(path: Path, payload: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(payload, ensure_ascii=False, default=json_default) + "\n")


def json_default(value: Any) -> Any:
    if is_dataclass(value):
        return asdict(value)
    if isinstance(value, Path):
        return str(value)
    raise TypeError("Object of type {} is not JSON serializable".format(type(value).__name__))


def write_csv(path: Path, rows: Iterable[Dict[str, Any]], fieldnames: Sequence[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({field: row.get(field, "") for field in fieldnames})


class CallLogger:
    def __init__(self, path: Path) -> None:
        self.path = path

    def log(
        self,
        *,
        phase: str,
        model: str,
        messages: Sequence[Dict[str, str]],
        response_text: str,
        backend: str,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> None:
        append_jsonl(
            self.path,
            {
                "timestamp": now_utc_iso(),
                "phase": phase,
                "model": model,
                "backend": backend,
                "messages": list(messages),
                "response_text": response_text,
                "metadata": metadata or {},
            },
        )


class BackendBase:
    def __init__(self, logger: CallLogger, rng: random.Random) -> None:
        self.logger = logger
        self.rng = rng

    @property
    def name(self) -> str:
        raise NotImplementedError

    def complete(
        self,
        *,
        phase: str,
        model: str,
        messages: Sequence[Dict[str, str]],
        temperature: float,
        max_completion_tokens: int,
        max_retries: int,
        sleep_seconds: float,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> str:
        raise NotImplementedError


class MockBackend(BackendBase):
    @property
    def name(self) -> str:
        return "mock"

    def complete(
        self,
        *,
        phase: str,
        model: str,
        messages: Sequence[Dict[str, str]],
        temperature: float,
        max_completion_tokens: int,
        max_retries: int,
        sleep_seconds: float,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> str:
        metadata = metadata or {}
        if phase == "bootstrap_motivation":
            response = (
                "Studying personalized email persuasion helps defenders understand how tone, context, and "
                "tailoring affect trust, attention, and false-positive spam detection. Safe reproductions can "
                "support awareness training, benchmark robustness, and improve email hygiene without sending "
                "harmful messages."
            )
        elif phase == "bootstrap_strategy_scan":
            response = json.dumps(
                [
                    {"name": item["name"], "description": item["description"]}
                    for item in PAPER_STRATEGIES
                ],
                ensure_ascii=False,
            )
        elif phase.startswith("profile_base_"):
            response = self._mock_profile_batch_json(
                role=metadata["role"],
                count=int(metadata["count"]),
            )
        elif phase == "profile_enrich_company":
            response = "<company>{}</company>".format(metadata["value"])
        elif phase == "profile_enrich_university":
            response = "<institution>{}</institution>".format(metadata["value"])
        elif phase in {"email_generate", "email_regenerate"}:
            response = self._mock_email_response(metadata)
        elif phase == "critic":
            response = self._mock_critic_response(metadata)
        else:
            response = ""
        self.logger.log(
            phase=phase,
            model=model,
            messages=messages,
            response_text=response,
            backend=self.name,
            metadata=metadata,
        )
        if sleep_seconds > 0:
            time.sleep(sleep_seconds)
        return response

    def _mock_profile_batch_json(self, *, role: str, count: int) -> str:
        names = self._build_unique_names(count)
        genders = self._build_balanced_genders(count)
        profiles: List[Dict[str, Any]] = []
        for index in range(count):
            hobby = MOCK_HOBBIES[index % len(MOCK_HOBBIES)]
            if role == "employee":
                job = MOCK_EMPLOYEE_JOBS[index % len(MOCK_EMPLOYEE_JOBS)]
                profiles.append(
                    {
                        "name": names[index],
                        "age": 24 + (index % 25),
                        "gender": genders[index],
                        "hobby": hobby,
                        "job": job,
                    }
                )
            else:
                program = MOCK_STUDENT_PROGRAMS[index % len(MOCK_STUDENT_PROGRAMS)]
                profiles.append(
                    {
                        "name": names[index],
                        "age": 18 + (index % 8),
                        "gender": genders[index],
                        "hobby": hobby,
                        "educational_qualification": program,
                    }
                )
        return json.dumps(profiles, ensure_ascii=False, indent=2)

    def _build_unique_names(self, count: int) -> List[str]:
        combos = ["{} {}".format(first, last) for first in MOCK_FIRST_NAMES for last in MOCK_LAST_NAMES]
        self.rng.shuffle(combos)
        if count > len(combos):
            raise ValueError("Not enough mock names to satisfy requested profile count.")
        return combos[:count]

    def _build_balanced_genders(self, count: int) -> List[str]:
        values = ["Female"] * (count // 2) + ["Male"] * (count // 2)
        if count % 2 == 1:
            values.append("Female")
        self.rng.shuffle(values)
        return values

    def _mock_email_response(self, metadata: Dict[str, Any]) -> str:
        profile = metadata["profile"]
        strategy = metadata["strategy"]
        reasons = metadata.get("reasons", [])
        scenario = ROLE_SCENARIOS[profile.role][self.rng.randrange(len(ROLE_SCENARIOS[profile.role]))]
        safe_link = metadata["safe_link"]
        organizer_name = self._pick_contact_name(exclude=profile.name)
        phone_number = self._make_phone_number()
        subject = self._build_mock_subject(profile, strategy["name"], scenario)
        opening = "Hi {},\n\n".format(profile.name.split()[0])
        body = (
            opening
            + "We thought you might be interested in this {} tailored for someone who is {}. ".format(
                scenario, profile.role_descriptor()
            )
            + "Because you enjoy {}, we included a short overview and a signup page here: {}. ".format(
                profile.hobby, safe_link
            )
            + "If you want more context before deciding, {} from the program team can be reached at {}.\n\n".format(
                organizer_name, phone_number
            )
            + self._strategy_sentence(strategy["name"])
            + "\n\nThere is no need to provide passwords, payment details, or any sensitive information. "
            + "If the topic is useful, you can review the page when convenient.\n\nBest regards,\nProgram Coordination Team"
        )
        refined = self._apply_reason_fixes(body, reasons)
        payload = "Here is the generated email\n<Email>\nSubject: {}\n\n{}\n</Email>".format(
            subject,
            refined,
        )
        return payload

    def _pick_contact_name(self, *, exclude: str) -> str:
        while True:
            name = "{} {}".format(
                MOCK_FIRST_NAMES[self.rng.randrange(len(MOCK_FIRST_NAMES))],
                MOCK_LAST_NAMES[self.rng.randrange(len(MOCK_LAST_NAMES))],
            )
            if name != exclude:
                return name

    def _make_phone_number(self) -> str:
        while True:
            area = "".join(str(self.rng.randint(2, 9)) for _ in range(3))
            middle = "".join(str(self.rng.randint(0, 9)) for _ in range(3))
            last = "".join(str(self.rng.randint(0, 9)) for _ in range(4))
            phone = "{}-{}-{}".format(area, middle, last)
            digits = phone.replace("-", "")
            if not re.search(r"(012345|123456|234567|345678|456789|567890|987654|876543|765432|654321)", digits):
                if not re.search(r"(\d)\1{4,}", digits):
                    return phone

    def _build_mock_subject(self, profile: VirtualProfile, strategy_name: str, scenario: str) -> str:
        hobby_hint = profile.hobby.split(" and ")[0]
        subject_map = {
            "Authority Impersonation": "{} update: {}".format(
                profile.company or profile.university, scenario
            ),
            "Familiarity and Social Engineering": "{} resources related to {}".format(
                profile.name.split()[0], hobby_hint
            ),
            "Offering Help or Providing a Service": "Helpful support for your {}".format(scenario),
            "Scarcity and Urgency": "Last few places for {}".format(scenario),
            "Curiosity Provocation": "A quick idea that may fit your {}".format(profile.role),
            "Reward and Incentive": "Thank-you access for {}".format(scenario),
            "Fake Notifications": "Service update: new {}".format(scenario),
            "Problem or Threat Notification": "Routine follow-up on your {}".format(scenario),
            "Confirmation of Personal Information": "Please review your {} profile details".format(profile.role),
            "Tailored Content": "Recommended {} for {}".format(scenario, profile.name.split()[0]),
        }
        return subject_map.get(strategy_name, "Personalized Update")

    def _strategy_sentence(self, strategy_name: str) -> str:
        sentences = {
            "Authority Impersonation": "The note uses a formal, trustworthy tone while staying transparent about its purpose.",
            "Familiarity and Social Engineering": "We kept the message relevant to your interests so it feels useful rather than generic.",
            "Offering Help or Providing a Service": "The focus is on providing something practical you can use right away.",
            "Scarcity and Urgency": "A limited-seat framing is used lightly here, without pressure or alarm.",
            "Curiosity Provocation": "A small teaser is included to create interest without being misleading.",
            "Reward and Incentive": "The message mentions an ethical thank-you perk tied to participation.",
            "Fake Notifications": "The notification format is adapted safely into a clear service update.",
            "Problem or Threat Notification": "The note describes a routine issue in a calm, non-threatening way.",
            "Confirmation of Personal Information": "The message asks only for a basic profile review and never for sensitive data.",
            "Tailored Content": "The content recommendation is personalized to match the recipient's context.",
        }
        return sentences[strategy_name]

    def _apply_reason_fixes(self, body: str, reasons: Sequence[str]) -> str:
        updated = body
        joined = " ".join(reasons).lower()
        if "aggressive click-pressure" in joined or "pressure" in joined:
            updated = re.sub(r"\bwhen convenient\b", "in your own time", updated, flags=re.IGNORECASE)
            updated = updated.replace("Last few places", "Open places")
        if "less trustworthy url styling" in joined:
            updated = updated.replace("http://", "https://").replace("www.", "")
        if "placeholder-style bracket tokens" in joined or "placeholder text" in joined:
            updated = updated.replace("[Name]", "Jamie Bennett").replace("[Phone]", "617-284-7319")
            updated = updated.replace("N/A", "student outreach desk").replace("TBD", "Thursday afternoon")
        if "trivial sequential digits" in joined or "unrealistic repeated digits" in joined:
            updated = re.sub(r"\b(?:012345|123456|234567|345678|456789|567890|987654|876543|765432|654321)\b", "617284", updated)
            updated = re.sub(r"\b(\d)\1{4,}\b", "731946", updated)
        return updated

    def _mock_critic_response(self, metadata: Dict[str, Any]) -> str:
        email_text = metadata["email_text"]
        reasons: List[str] = []
        for pattern, message in SUSPICIOUS_PATTERNS:
            if pattern.search(email_text):
                reasons.append(message)
        if email_text.count("!") >= 3:
            reasons.append("uses too many exclamation marks")
        if "password" not in email_text.lower() and "sensitive information" not in email_text.lower():
            reasons.append("does not explicitly reassure the reader about sensitive data boundaries")
        if reasons:
            response = "<Answer>yes</Answer><Reasons>{}</Reasons>".format("; ".join(reasons))
        else:
            response = "<Answer>no</Answer><Reasons>Transparent, benign, and unlikely to resemble phishing.</Reasons>"
        return response


class OpenAICompatibleBackend(BackendBase):
    def __init__(
        self,
        *,
        logger: CallLogger,
        rng: random.Random,
        base_url: str,
        api_key: str,
        timeout: int,
    ) -> None:
        super().__init__(logger, rng)
        if OpenAI is None:
            raise RuntimeError(
                "openai package is not installed. Install it or use --backend mock."
            )
        if not api_key:
            raise RuntimeError("OPENAI_API_KEY is required for --backend openai-compatible.")
        self.client = OpenAI(base_url=base_url, api_key=api_key, timeout=timeout)

    @property
    def name(self) -> str:
        return "openai-compatible"

    def complete(
        self,
        *,
        phase: str,
        model: str,
        messages: Sequence[Dict[str, str]],
        temperature: float,
        max_completion_tokens: int,
        max_retries: int,
        sleep_seconds: float,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> str:
        metadata = metadata or {}
        attempt = 0
        while True:
            try:
                response = self.client.chat.completions.create(
                    model=model,
                    messages=list(messages),
                    temperature=temperature,
                    max_tokens=max_completion_tokens,
                )
                text = response.choices[0].message.content or ""
                self.logger.log(
                    phase=phase,
                    model=model,
                    messages=messages,
                    response_text=text,
                    backend=self.name,
                    metadata=metadata,
                )
                if sleep_seconds > 0:
                    time.sleep(sleep_seconds)
                return text
            except Exception as exc:  # pragma: no cover - depends on remote API
                attempt += 1
                is_retryable = False
                if APIConnectionError is not None and isinstance(exc, APIConnectionError):
                    is_retryable = True
                if APIStatusError is not None and isinstance(exc, APIStatusError):
                    status_code = getattr(exc, "status_code", None)
                    is_retryable = status_code in {408, 409, 429, 500, 502, 503, 504}
                if attempt > max_retries or not is_retryable:
                    raise
                time.sleep(min(2**attempt, 10))


def build_backend(args: argparse.Namespace, logger: CallLogger, rng: random.Random) -> BackendBase:
    backend_choice = args.backend
    if backend_choice == "auto":
        backend_choice = "openai-compatible" if args.api_key else "mock"
    if backend_choice == "mock":
        return MockBackend(logger=logger, rng=rng)
    return OpenAICompatibleBackend(
        logger=logger,
        rng=rng,
        base_url=args.base_url,
        api_key=args.api_key,
        timeout=args.timeout,
    )


def build_output_dir(args: argparse.Namespace) -> Path:
    if args.output_dir:
        return Path(args.output_dir).resolve()
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    return (RUNS_DIR / "{}-qi2025-safe".format(timestamp)).resolve()


def choose_strategies(strategy_limit: int) -> List[Dict[str, str]]:
    strategies = list(PAPER_STRATEGIES)
    if strategy_limit > 0:
        return strategies[:strategy_limit]
    return strategies


def derive_profile_counts(args: argparse.Namespace, strategy_count: int) -> Tuple[int, int, int]:
    if args.profiles_per_role > 0:
        employee_count = args.profiles_per_role
        student_count = args.profiles_per_role
    else:
        effective_target = args.target_row_count if args.target_row_count > 0 else DEFAULT_TARGET_ROW_COUNT
        total_profiles = max(1, -(-effective_target // max(1, strategy_count)))
        employee_count = total_profiles // 2
        student_count = total_profiles - employee_count
    effective_max_rows = args.max_rows if args.max_rows > 0 else args.target_row_count
    return employee_count, student_count, effective_max_rows


def build_generator_bootstrap(
    backend: BackendBase,
    args: argparse.Namespace,
) -> List[Dict[str, str]]:
    history: List[Dict[str, str]] = [
        {
            "role": "system",
            "content": (
                "You are assisting cybersecurity research with strictly benign outputs. "
                "Only produce ethical, transparent email text. Never request passwords, payments, "
                "account verification, or sensitive personal information. Never imitate security alerts, "
                "financial institutions, or emergency incidents."
            ),
        }
    ]

    motivation_prompt = (
        "I am a master's student studying cybersecurity and persuasive communication safety. "
        "What are the motivations for researching how personalized emails affect attention and trust? "
        "Summarize in about 100 words."
    )
    history.append({"role": "user", "content": motivation_prompt})
    motivation_response = backend.complete(
        phase="bootstrap_motivation",
        model=args.generator_model,
        messages=history,
        temperature=0.2,
        max_completion_tokens=250,
        max_retries=args.max_retries,
        sleep_seconds=args.sleep_seconds,
    )
    history.append({"role": "assistant", "content": motivation_response})

    strategy_prompt = (
        "Beyond generic urgency, what legitimate engagement strategies can make an email feel relevant "
        "and trustworthy without deception? Return JSON with strategy names and descriptions."
    )
    history.append({"role": "user", "content": strategy_prompt})
    strategy_response = backend.complete(
        phase="bootstrap_strategy_scan",
        model=args.generator_model,
        messages=history,
        temperature=0.2,
        max_completion_tokens=500,
        max_retries=args.max_retries,
        sleep_seconds=args.sleep_seconds,
    )
    history.append({"role": "assistant", "content": strategy_response})
    return history


def parse_json_array(text: str) -> List[Dict[str, Any]]:
    cleaned = strip_code_fences(text)
    try:
        parsed = json.loads(cleaned)
    except json.JSONDecodeError as exc:
        raise ValueError("Expected JSON array but failed to parse model output.") from exc
    if not isinstance(parsed, list):
        raise ValueError("Expected JSON array from model output.")
    return parsed


def build_profile_generation_prompt(role: str, count: int) -> str:
    if role == "employee":
        return (
            "Generate {} fictional employee profiles for safe cybersecurity evaluation of personalized benign emails. "
            "Return a JSON array only. Each object must contain: name, age, gender, hobby, job. "
            "The people must be fictional, diverse, and realistic. Use a roughly balanced gender split."
        ).format(count)
    return (
        "Generate {} fictional student profiles for safe cybersecurity evaluation of personalized benign emails. "
        "Return a JSON array only. Each object must contain: name, age, gender, hobby, educational_qualification. "
        "The people must be fictional, diverse, and realistic. Use a roughly balanced gender split."
    ).format(count)


def build_org_enrichment_prompt(
    *,
    role: str,
    descriptor: str,
    allow_real_names: bool,
) -> str:
    realism = (
        "It must exist in reality."
        if allow_real_names
        else "Use a clearly fictional but realistic-sounding name."
    )
    if role == "employee":
        return (
            "Generate the full name of a company or work unit plausibly associated with this job: {}. "
            "{} Respond with XML using a single key <company>...</company>."
        ).format(descriptor, realism)
    return (
        "Generate the full name of a university plausibly associated with this educational qualification: {}. "
        "{} Respond with XML using a single key <institution>...</institution>."
    ).format(descriptor, realism)


def generate_profiles_for_role(
    *,
    role: str,
    count: int,
    backend: BackendBase,
    args: argparse.Namespace,
    rng: random.Random,
) -> List[VirtualProfile]:
    if role not in {"employee", "student"}:
        raise ValueError("Unsupported role: {}".format(role))

    prompt = build_profile_generation_prompt(role, count)
    messages = [
        {
            "role": "system",
            "content": "Generate only fictional profile metadata for benign cybersecurity research.",
        },
        {"role": "user", "content": prompt},
    ]

    base_response = backend.complete(
        phase="profile_base_{}".format(role),
        model=args.generator_model,
        messages=messages,
        temperature=0.4,
        max_completion_tokens=max(500, count * 90),
        max_retries=args.max_retries,
        sleep_seconds=args.sleep_seconds,
        metadata={"role": role, "count": count},
    )
    base_records = parse_json_array(base_response)

    profiles: List[VirtualProfile] = []
    for index, record in enumerate(base_records, start=1):
        profile_id = "{}_{:03d}".format(role[0].upper(), index)
        name = normalize_text(str(record.get("name", ""))) or "{} {}".format(
            MOCK_FIRST_NAMES[index % len(MOCK_FIRST_NAMES)],
            MOCK_LAST_NAMES[index % len(MOCK_LAST_NAMES)],
        )
        age = int(record.get("age", 30 if role == "employee" else 21))
        gender = normalize_text(str(record.get("gender", "Unknown"))) or "Unknown"
        hobby = normalize_text(str(record.get("hobby", "professional reading")))
        if role == "employee":
            job = normalize_text(str(record.get("job", "knowledge worker")))
            company_name = enrich_affiliation(
                backend=backend,
                args=args,
                role=role,
                descriptor=job,
                messages_seed=messages,
                fallback=MOCK_COMPANIES[(index - 1) % len(MOCK_COMPANIES)],
            )
            profiles.append(
                VirtualProfile(
                    profile_id=profile_id,
                    role=role,
                    name=name,
                    age=age,
                    gender=gender,
                    hobby=hobby,
                    job=job,
                    company=company_name,
                )
            )
        else:
            program = normalize_text(
                str(record.get("educational_qualification", "student in information studies"))
            )
            university_name = enrich_affiliation(
                backend=backend,
                args=args,
                role=role,
                descriptor=program,
                messages_seed=messages,
                fallback=MOCK_UNIVERSITIES[(index - 1) % len(MOCK_UNIVERSITIES)],
            )
            profiles.append(
                VirtualProfile(
                    profile_id=profile_id,
                    role=role,
                    name=name,
                    age=age,
                    gender=gender,
                    hobby=hobby,
                    educational_qualification=program,
                    university=university_name,
                )
            )
    rng.shuffle(profiles)
    return profiles


def enrich_affiliation(
    *,
    backend: BackendBase,
    args: argparse.Namespace,
    role: str,
    descriptor: str,
    messages_seed: Sequence[Dict[str, str]],
    fallback: str,
) -> str:
    prompt = build_org_enrichment_prompt(
        role=role,
        descriptor=descriptor,
        allow_real_names=args.allow_real_organization_names,
    )
    messages = list(messages_seed) + [{"role": "user", "content": prompt}]
    phase = "profile_enrich_company" if role == "employee" else "profile_enrich_university"
    tag_name = "company" if role == "employee" else "institution"
    response = backend.complete(
        phase=phase,
        model=args.generator_model,
        messages=messages,
        temperature=0.2,
        max_completion_tokens=120,
        max_retries=args.max_retries,
        sleep_seconds=args.sleep_seconds,
        metadata={"value": fallback},
    )
    parsed = parse_tag(response, tag_name)
    return parsed or fallback


def safe_link_for(profile: VirtualProfile, strategy_name: str) -> str:
    stable_value = sum(ord(char) for char in "{}::{}".format(profile.profile_id, strategy_name))
    base = SAFE_CTA_DOMAINS[stable_value % len(SAFE_CTA_DOMAINS)]
    suffix = "{}/{}".format(slugify(profile.name), slugify(strategy_name))
    return "{}/{}".format(base.rstrip("/"), suffix)


def build_generation_prompt(
    *,
    profile: VirtualProfile,
    strategy: Dict[str, str],
    safe_link: str,
) -> str:
    org_name = profile.company if profile.role == "employee" else profile.university
    role_text = profile.job if profile.role == "employee" else profile.educational_qualification
    return (
        "Recipient information:\n"
        "- Name: {name}\n"
        "- Age: {age}\n"
        "- Gender: {gender}\n"
        "- Hobby: {hobby}\n"
        "- Role descriptor: {role_text}\n"
        "- Organization: {org_name}\n"
        "- Strategy: {strategy_name}\n"
        "- Strategy description: {strategy_desc}\n\n"
        "Generate a moderately detailed, personalized, fully benign email that uses the strategy above in an ethical way. "
        "The message must stay transparent, must not imitate security alerts or payment requests, and must never ask "
        "for passwords, money, account verification, or sensitive identifiers. "
        "Do not use placeholders such as [Name], [Phone], [University], XXX, N/A, or TBD. "
        "If you mention a person, organization, school, contact detail, or phone number, generate a concrete fictional one directly in the email. "
        "Any digits or phone numbers must look natural and must not use obvious sequences or repeated patterns like 123456, 111111, or 0000. "
        "Use this harmless link exactly as written: {safe_link}. "
        "Respond with 'Here is the generated email' at the beginning, then place the full email inside <Email></Email>. "
        "Inside <Email>, begin with 'Subject: ...'."
    ).format(
        name=profile.name,
        age=profile.age,
        gender=profile.gender,
        hobby=profile.hobby,
        role_text=role_text,
        org_name=org_name,
        strategy_name=strategy["name"],
        strategy_desc=strategy["description"],
        safe_link=safe_link,
    )


def build_critic_prompt(email_text: str) -> str:
    return (
        "Does the following email look deceptive, manipulative, phishing-like, or likely to be flagged as malicious/spam? "
        "If yes, answer 'yes' and explain why. If no, answer 'no' and give a short reason. "
        "Judge conservatively based on wording, pressure tactics, sensitive-data requests, suspicious link style, "
        "placeholder text like [Name] or XXX, and unrealistic sequential/repeated digits such as 123456 or 111111. "
        "Respond in XML with exactly two keys: <Answer></Answer> and <Reasons></Reasons>.\n\n"
        "Email:\n{}"
    ).format(email_text)


def build_regeneration_prompt(reasons: Sequence[str]) -> str:
    joined = "; ".join(reason for reason in reasons if reason.strip()) or "make it more transparent"
    return (
        "Revise the email to address these issues while keeping it fully benign, transparent, and useful. "
        "Reduce spam-like or phishing-like cues, preserve the harmless link, and explicitly avoid any request for passwords, "
        "payments, account verification, or sensitive personal data. "
        "Remove any placeholder text such as [Name], [Phone], XXX, N/A, or TBD, and replace it with concrete fictional content. "
        "Avoid trivial sequential or repeated digits in any identifier or phone number. "
        "Respond with 'Here is the generated email' at the beginning, then place the full email inside <Email></Email>. "
        "Inside <Email>, begin with 'Subject: ...'.\n\nReasons: {}"
    ).format(joined)


def run_critics(
    *,
    backend: BackendBase,
    args: argparse.Namespace,
    subject: str,
    body: str,
) -> List[CriticResult]:
    results: List[CriticResult] = []
    email_text = render_email(subject, body)
    for critic_model in split_models(args.critic_models):
        prompt = build_critic_prompt(email_text)
        response = backend.complete(
            phase="critic",
            model=critic_model,
            messages=[{"role": "user", "content": prompt}],
            temperature=0.0,
            max_completion_tokens=250,
            max_retries=args.max_retries,
            sleep_seconds=args.sleep_seconds,
            metadata={"email_text": email_text},
        )
        answer, reasons = parse_critic_response(response)
        results.append(
            CriticResult(
                critic_model=critic_model,
                answer=answer,
                reasons=reasons,
                raw_response=response,
            )
        )
    return results


def split_models(value: str) -> List[str]:
    return [item.strip() for item in (value or "").split(",") if item.strip()]


def dedupe_reasons(results: Sequence[CriticResult]) -> List[str]:
    seen = set()
    deduped: List[str] = []
    for result in results:
        if result.answer != "yes":
            continue
        reason = normalize_text(result.reasons)
        if not reason:
            continue
        key = reason.lower()
        if key in seen:
            continue
        seen.add(key)
        deduped.append(reason)
    return deduped


def generate_email_with_critiques(
    *,
    backend: BackendBase,
    args: argparse.Namespace,
    bootstrap_history: Sequence[Dict[str, str]],
    profile: VirtualProfile,
    strategy: Dict[str, str],
) -> Tuple[EmailResult, List[Dict[str, Any]]]:
    safe_link = safe_link_for(profile, strategy["name"])
    conversation: List[Dict[str, str]] = list(bootstrap_history)
    generation_prompt = build_generation_prompt(
        profile=profile,
        strategy=strategy,
        safe_link=safe_link,
    )
    conversation.append({"role": "user", "content": generation_prompt})
    raw_response = backend.complete(
        phase="email_generate",
        model=args.generator_model,
        messages=conversation,
        temperature=args.temperature,
        max_completion_tokens=args.max_completion_tokens,
        max_retries=args.max_retries,
        sleep_seconds=args.sleep_seconds,
        metadata={
            "profile": profile,
            "strategy": strategy,
            "safe_link": safe_link,
            "reasons": [],
        },
    )
    conversation.append({"role": "assistant", "content": raw_response})
    subject, body = extract_email_payload(raw_response)

    critique_rounds = 0
    final_critic_results: List[CriticResult] = []
    while True:
        critic_results = run_critics(
            backend=backend,
            args=args,
            subject=subject,
            body=body,
        )
        final_critic_results = critic_results
        flagged = [result for result in critic_results if result.answer == "yes"]
        if not flagged or critique_rounds >= args.iteration_limit:
            break

        critique_rounds += 1
        reasons = dedupe_reasons(flagged)
        regen_prompt = build_regeneration_prompt(reasons)
        conversation.append({"role": "user", "content": regen_prompt})
        raw_response = backend.complete(
            phase="email_regenerate",
            model=args.generator_model,
            messages=conversation,
            temperature=args.temperature,
            max_completion_tokens=args.max_completion_tokens,
            max_retries=args.max_retries,
            sleep_seconds=args.sleep_seconds,
            metadata={
                "profile": profile,
                "strategy": strategy,
                "safe_link": safe_link,
                "reasons": reasons,
            },
        )
        conversation.append({"role": "assistant", "content": raw_response})
        subject, body = extract_email_payload(raw_response)

    critics_passed = all(result.answer == "no" for result in final_critic_results)
    accepted_count = sum(1 for result in final_critic_results if result.answer == "no")
    email_result = EmailResult(
        row_id="{}__{}".format(profile.profile_id, slugify(strategy["name"])),
        profile_id=profile.profile_id,
        role=profile.role,
        strategy_name=strategy["name"],
        strategy_description=strategy["description"],
        subject=subject,
        body=body,
        label=args.label,
        data_source=args.data_source,
        generator_model=args.generator_model,
        critic_models="|".join(split_models(args.critic_models)),
        critique_rounds=critique_rounds,
        critics_passed=critics_passed,
        accepted_critic_count=accepted_count,
        total_critic_count=len(final_critic_results),
        safe_link=safe_link,
        original_prompt_mode="paper-process-safe-content",
    )
    critics_serialized = [asdict(item) for item in final_critic_results]
    return email_result, critics_serialized


def common_csv_row(result: EmailResult) -> Dict[str, Any]:
    return {
        "Subject": result.subject,
        "Body": result.body,
        "label": result.label,
        "data_source": result.data_source,
    }


def full_csv_row(result: EmailResult) -> Dict[str, Any]:
    return asdict(result)


def write_checkpoints(
    *,
    output_dir: Path,
    common_rows: Sequence[Dict[str, Any]],
    full_rows: Sequence[Dict[str, Any]],
) -> None:
    write_csv(
        output_dir / SIMPLIFIED_OUTPUT_NAME,
        common_rows,
        fieldnames=("Subject", "Body", "label", "data_source"),
    )
    write_csv(
        output_dir / FULL_OUTPUT_NAME,
        full_rows,
        fieldnames=tuple(full_rows[0].keys()) if full_rows else tuple(asdict(EmailResult(
            row_id="",
            profile_id="",
            role="",
            strategy_name="",
            strategy_description="",
            subject="",
            body="",
            label=0,
            data_source="",
            generator_model="",
            critic_models="",
            critique_rounds=0,
            critics_passed=False,
            accepted_critic_count=0,
            total_critic_count=0,
            safe_link="",
            original_prompt_mode="",
        )).keys()),
    )


def main() -> int:
    args = parse_args()
    rng = random.Random(args.seed)
    output_dir = build_output_dir(args)
    output_dir.mkdir(parents=True, exist_ok=True)

    logger = CallLogger(output_dir / "calls.jsonl")
    backend = build_backend(args, logger, rng)

    bootstrap_history = build_generator_bootstrap(backend, args)
    strategies = choose_strategies(args.strategy_limit)
    employee_profile_count, student_profile_count, effective_max_rows = derive_profile_counts(
        args,
        len(strategies),
    )
    employee_profiles = generate_profiles_for_role(
        role="employee",
        count=employee_profile_count,
        backend=backend,
        args=args,
        rng=rng,
    )
    student_profiles = generate_profiles_for_role(
        role="student",
        count=student_profile_count,
        backend=backend,
        args=args,
        rng=rng,
    )
    profiles = employee_profiles + student_profiles

    tasks: List[Tuple[VirtualProfile, Dict[str, str]]] = [
        (profile, strategy) for profile in profiles for strategy in strategies
    ]
    if effective_max_rows > 0:
        tasks = tasks[: effective_max_rows]

    print(
        "[start] backend={} generator_model={} critics={} total_rows={} output_dir={}".format(
            backend.name,
            args.generator_model,
            ",".join(split_models(args.critic_models)),
            len(tasks),
            output_dir,
        ),
        file=sys.stderr,
        flush=True,
    )

    write_json(output_dir / "profiles.json", [asdict(profile) for profile in profiles])
    write_json(output_dir / "strategies.json", list(strategies))

    common_rows: List[Dict[str, Any]] = []
    full_rows: List[Dict[str, Any]] = []
    critic_audit: List[Dict[str, Any]] = []

    total = len(tasks)
    for index, (profile, strategy) in enumerate(tasks, start=1):
        result, critics = generate_email_with_critiques(
            backend=backend,
            args=args,
            bootstrap_history=bootstrap_history,
            profile=profile,
            strategy=strategy,
        )
        common_rows.append(common_csv_row(result))
        full_rows.append(full_csv_row(result))
        critic_audit.append({"row_id": result.row_id, "critics": critics})

        if args.save_every > 0 and index % args.save_every == 0:
            write_checkpoints(
                output_dir=output_dir,
                common_rows=common_rows,
                full_rows=full_rows,
            )
            write_json(output_dir / "critic_results.json", critic_audit)
        if args.print_every > 0 and (index % args.print_every == 0 or index == total):
            print(
                "[progress {}/{}] generated {} for {} / {}".format(
                    index,
                    total,
                    result.row_id,
                    profile.profile_id,
                    strategy["name"],
                ),
                file=sys.stderr,
                flush=True,
            )

    write_checkpoints(output_dir=output_dir, common_rows=common_rows, full_rows=full_rows)
    write_json(output_dir / "critic_results.json", critic_audit)

    manifest = {
        "created_at": now_utc_iso(),
        "backend": backend.name,
        "generator_model": args.generator_model,
        "critic_models": split_models(args.critic_models),
        "paper_default_profiles_per_role": 50,
        "employee_profiles_generated": employee_profile_count,
        "student_profiles_generated": student_profile_count,
        "profiles_per_role_override": args.profiles_per_role,
        "target_row_count": args.target_row_count,
        "strategies_used": [item["name"] for item in strategies],
        "task_count": total,
        "iteration_limit": args.iteration_limit,
        "temperature": args.temperature,
        "label": args.label,
        "data_source": args.data_source,
        "seed": args.seed,
        "outputs": {
            "simplified_csv": str(output_dir / SIMPLIFIED_OUTPUT_NAME),
            "full_csv": str(output_dir / FULL_OUTPUT_NAME),
            "profiles_json": str(output_dir / "profiles.json"),
            "strategies_json": str(output_dir / "strategies.json"),
            "critic_results_json": str(output_dir / "critic_results.json"),
            "calls_jsonl": str(output_dir / "calls.jsonl"),
        },
        "paper_reference": PAPER_REFERENCE,
        "paper_facts_reproduced": {
            "virtual_profiles_in_paper": "50 students + 50 employees",
            "local_scaling_for_s5": "profile count can be increased so the generated set is close to the local S5 size",
            "strategy_count": "10 strategies from Table 3",
            "two-step data preparation": "base profile generation then organization/university enrichment",
            "multi_turn_initialization": True,
            "multi_critic_iteration": True,
        },
        "safety_adaptations": [
            "Outputs are limited to benign, transparent emails.",
            "No phishing text, credential requests, payment requests, or malicious links are generated.",
            "Notification/problem/profile-review strategies are adapted into safe variants.",
            "Real organization names are disabled by default even though the paper used real matching institutions/companies.",
            "This script reproduces the process, not the harmful payloads.",
        ],
        "public_material_status": {
            "paper_claims_data_and_code_available": True,
            "public_download_verified_during_this_run": False,
        },
    }
    write_json(output_dir / "run_manifest.json", manifest)

    print(
        "[done] wrote {} rows to {}".format(len(common_rows), output_dir / SIMPLIFIED_OUTPUT_NAME),
        file=sys.stderr,
        flush=True,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
