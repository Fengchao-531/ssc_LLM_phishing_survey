import html
import re
from typing import Iterable

from utils import sentence_tokenize


WORD_PATTERN = r"""(?x)
    (?:[A-Z]\.)+
    |\$?\d+(?:\.\d+)?%?
    |\w+(?:[-']\w+)*
    |\.\.\.
    |(?:[.,;"'?():_`])
"""

URL_RE = re.compile(r"https?://\S+|www\.\S+", re.IGNORECASE)
STYLE_BLOCK_RE = re.compile(r"<style\b.*?</style>", re.IGNORECASE | re.DOTALL)
SCRIPT_BLOCK_RE = re.compile(r"<script\b.*?</script>", re.IGNORECASE | re.DOTALL)
TAG_RE = re.compile(r"<[^>]+>")
MULTISPACE_RE = re.compile(r"\s+")
TOKEN_WITH_ALPHA_RE = re.compile(r"[A-Za-z]")
TOKEN_WITH_DIGIT_RE = re.compile(r"\d")
TOKEN_WITH_UNDERSCORE_RE = re.compile(r"_")
SPLIT_SENTENCE_RE = re.compile(r"[\n\r]+")

NOISE_KEYWORDS = {
    "style",
    "font",
    "padding",
    "margin",
    "background",
    "width",
    "height",
    "img",
    "table",
    "tbody",
    "tr",
    "td",
    "href",
    "src",
    "doctype",
    "html",
    "meta",
    "viewport",
    "charset",
}
FUNCTION_WORDS = {
    "the",
    "a",
    "an",
    "to",
    "and",
    "or",
    "for",
    "of",
    "in",
    "on",
    "is",
    "are",
    "you",
    "your",
    "we",
    "our",
    "please",
    "dear",
    "hello",
    "thank",
    "thanks",
    "regards",
    "with",
    "that",
    "this",
    "from",
    "have",
    "will",
    "can",
    "not",
    "be",
    "as",
    "if",
    "by",
    "at",
}
def transform(text):
    text = re.sub(r"Let\'s", " Let us ", text)
    text = re.sub(r"let\'s", " let us ", text)
    text = re.sub(r"\'m", " am ", text)
    text = re.sub(r"\'ve", " have ", text)
    text = re.sub(r"can\'t", " can not ", text)
    text = re.sub(r"n\'t", " not ", text)
    text = re.sub(r"\'re", " are ", text)
    text = re.sub(r"\'d", " would ", text)
    text = re.sub(r"\'ll", " will ", text)
    text = re.sub(r"y\'all", " you all ", text)
    return text


def check_ascii_word(word):
    for char in word:
        if ord(char) >= 128:
            return False
    return True


def normalize_raw_text(text):
    normalized = "" if text is None else str(text)
    normalized = html.unescape(normalized)
    normalized = normalized.replace("\r\n", "\n").replace("\r", "\n")
    normalized = STYLE_BLOCK_RE.sub(" ", normalized)
    normalized = SCRIPT_BLOCK_RE.sub(" ", normalized)
    normalized = URL_RE.sub(" website ", normalized)
    normalized = TAG_RE.sub(" ", normalized)
    normalized = MULTISPACE_RE.sub(" ", normalized)
    return normalized.strip()


def tokenize_sentence(sentence):
    tokens = re.findall(WORD_PATTERN, transform(sentence))
    cleaned = []
    for token in tokens:
        if not check_ascii_word(token):
            continue
        if token.isdigit():
            token = "number"
        elif token.startswith("$"):
            token = "money"
        elif token.endswith("%"):
            token = "percentage"
        cleaned.append(token.lower())
    return cleaned


def sentence_is_noise(raw_sentence, tokens: Iterable[str]):
    tokens = list(tokens)
    if not tokens:
        return True

    alpha_tokens = [token for token in tokens if TOKEN_WITH_ALPHA_RE.search(token)]
    natural_alpha_tokens = [
        token
        for token in alpha_tokens
        if token.isalpha() and len(token) >= 3 and token not in NOISE_KEYWORDS
    ]
    function_word_count = sum(token in FUNCTION_WORDS for token in alpha_tokens)
    digit_or_symbol_tokens = sum(
        TOKEN_WITH_DIGIT_RE.search(token) is not None
        or TOKEN_WITH_UNDERSCORE_RE.search(token) is not None
        for token in tokens
    )
    comma_count = raw_sentence.count(",")
    semicolon_count = raw_sentence.count(";")
    punctuation_density = (comma_count + semicolon_count) / max(1, len(tokens))
    weird_ratio = digit_or_symbol_tokens / max(1, len(tokens))
    avg_alpha_token_len = (
        sum(len(token) for token in natural_alpha_tokens) / len(natural_alpha_tokens)
        if natural_alpha_tokens
        else 0.0
    )
    lowercase_sentence = raw_sentence.lower()

    if any(marker in lowercase_sentence for marker in ("</style", "{", "}", "/*", "*/")):
        return True
    if len(natural_alpha_tokens) < 2 and weird_ratio >= 0.2:
        return True
    if function_word_count == 0 and weird_ratio >= 0.25:
        return True
    if function_word_count == 0 and punctuation_density >= 0.25 and avg_alpha_token_len <= 9.0:
        return True
    if len(alpha_tokens) >= 6 and function_word_count == 0 and avg_alpha_token_len <= 10.0:
        return True
    return False


def split_clean_sentences(subject, body, max_total_words=3000):
    sentence_tokens = []
    total_words = 0

    pieces = []
    subject_text = normalize_raw_text(subject)
    body_text = normalize_raw_text(body)
    if subject_text:
        pieces.append(subject_text)
    if body_text:
        pieces.append(body_text)

    for piece in pieces:
        candidate_sentences = []
        for block in SPLIT_SENTENCE_RE.split(piece):
            block = block.strip()
            if not block:
                continue
            candidate_sentences.extend(sentence_tokenize(block))

        for sentence in candidate_sentences:
            sentence = sentence.strip()
            if not sentence:
                continue
            tokens = tokenize_sentence(sentence)
            if sentence_is_noise(sentence, tokens):
                continue
            remaining_budget = max_total_words - total_words
            if remaining_budget <= 0:
                return sentence_tokens
            if len(tokens) > remaining_budget:
                tokens = tokens[:remaining_budget]
            if not tokens:
                continue
            sentence_tokens.append(tokens)
            total_words += len(tokens)
            if total_words >= max_total_words:
                return sentence_tokens

    return sentence_tokens


def build_fallback_sentence(subject, body, max_total_words=3000):
    merged = " ".join(
        value for value in (normalize_raw_text(subject), normalize_raw_text(body)) if value
    ).strip()
    if not merged:
        return [["empty"]]
    tokens = tokenize_sentence(merged)
    if not tokens:
        return [["empty"]]
    return [tokens[:max_total_words]]
