"""Tokenization, page splitting, chunking and deterministic chunk IDs."""
from __future__ import annotations

import re
import unicodedata
import uuid

CHUNK_NAMESPACE = uuid.UUID("3b8f6f2a-5c1d-4f7e-9a51-0d2e6c9b7a11")

STOPWORDS = frozenset(
    """a about all an and any are as at be been but by can could do does for from had has have
    how i if in into is it its me off many may much my of on or our please should so than that the
    their them then there these they this to us was we were what when where which who whom why
    will with would you your""".split()
)

TOKEN_RE = re.compile(r"[a-z0-9]+(?:[-_][a-z0-9]+)*")
IDENTIFIER_RE = re.compile(r"[a-z]+[-_]?\d+[a-z0-9_-]*")
PAGE_MARKER_RE = re.compile(r"<!--\s*page\s+(\d+)\s*-->", re.I)

# Tiny hand-written concept map. It lets the *simulated* dense embedder and reranker
# generalize across a few paraphrases (vacation ~ pto), which real neural models learn.
# The sparse/BM25 branch deliberately does NOT use it: BM25 is lexical.
CONCEPTS = {
    "vacation": "pto", "holidays": "pto", "carry": "carryover", "rollover": "carryover",
    "roll": "carryover", "approve": "approval", "approved": "approval", "approver": "approval",
    "signoff": "approval", "sign": "approval", "deploy": "deployment", "release": "deployment", "ship": "deployment",
    "breach": "incident", "hacked": "incident", "phishing": "incident", "2fa": "mfa",
    "supplier": "vendor", "contractor": "vendor", "food": "meal", "lunch": "meal",
    "dinner": "meal", "reimbursed": "reimburse", "reimbursement": "reimburse",
    "reimbursable": "reimburse", "christmas": "dec", "december": "dec", "earn": "accrue",
    "rotated": "rotate", "rotation": "rotate", "advance": "ahead",
}


def normalize(text: str) -> str:
    return unicodedata.normalize("NFKC", text).lower()


def is_identifier(token: str) -> bool:
    """Codes such as deploy-4021: letters followed by digits."""
    return bool(IDENTIFIER_RE.fullmatch(token)) and any(c.isdigit() for c in token)


def stem(token: str) -> str:
    """Minimal plural stripper; enough for a demo, not Porter."""
    if is_identifier(token) or token.isdigit() or len(token) <= 3:
        return token
    if token.endswith("ies") and len(token) > 4:
        return token[:-3] + "y"
    if token.endswith("s") and not token.endswith("ss"):
        return token[:-1]
    return token


def tokenize(text: str) -> list[str]:
    """Lowercased, stopword-free, lightly stemmed tokens. Compounds keep the whole form
    (so `deploy-4021` is one exact term) and also emit their parts."""
    out: list[str] = []
    for match in TOKEN_RE.finditer(normalize(text)):
        tok = match.group(0)
        if tok in STOPWORDS:
            continue
        out.append(stem(tok))
        if "-" in tok or "_" in tok:
            out.extend(stem(p) for p in re.split(r"[-_]", tok) if p and p not in STOPWORDS)
    return out


def concept_terms(tokens: list[str]) -> list[str]:
    return [CONCEPTS.get(t, t) for t in tokens]


def split_pages(text: str) -> list[tuple[int, str]]:
    """Split on `<!-- page N -->` markers or form feeds. Drops a leading `# Title` line."""
    text = re.sub(r"^\s*#\s+[^\n]*\n", "", text, count=1)
    if PAGE_MARKER_RE.search(text):
        parts = PAGE_MARKER_RE.split(text)  # [pre, n1, body1, n2, body2, ...]
        pages = [(1, parts[0])] if parts[0].strip() else []
        pages += [(int(parts[i]), parts[i + 1]) for i in range(1, len(parts), 2)]
    else:
        pages = list(enumerate(text.split("\f"), start=1))
    return [(n, body.strip()) for n, body in pages if body.strip()]


def extract_title(text: str, fallback: str) -> str:
    match = re.match(r"^\s*#\s+([^\n]+)", text)
    return match.group(1).strip() if match else fallback


def chunk_text(text: str, size: int = 900, overlap: int = 140) -> list[str]:
    """Character windows of about `size` with `overlap` characters shared between
    neighbours. Window ends snap back to a sentence end (if one is in the second half
    of the window) or a space; window starts snap forward to a word boundary."""
    if size <= 0 or overlap < 0 or overlap >= size:
        raise ValueError(f"invalid chunking: size={size} overlap={overlap}")
    text = re.sub(r"\s+", " ", text).strip()
    chunks: list[str] = []
    start, n = 0, len(text)
    while start < n:
        end = min(start + size, n)
        if end < n:
            window = text[start:end]
            cut = max(window.rfind(". "), window.rfind("? "), window.rfind("! "))
            if cut >= size // 2:
                end = start + cut + 1
            elif (space := window.rfind(" ")) > overlap:
                end = start + space
        piece = text[start:end].strip()
        if piece:
            chunks.append(piece)
        if end >= n:
            break
        nxt = end - overlap
        space = text.find(" ", nxt, end)
        start = max(space + 1 if space != -1 else nxt, start + 1)
    return chunks


def chunk_id(tenant_id: str, doc_id: str, page: int, index: int) -> str:
    """Stable across retries and processes, so upserts are idempotent."""
    return str(uuid.uuid5(CHUNK_NAMESPACE, f"{tenant_id}/{doc_id}/{page}/{index}"))


def manifest_id(tenant_id: str, doc_id: str) -> str:
    return str(uuid.uuid5(CHUNK_NAMESPACE, f"manifest/{tenant_id}/{doc_id}"))
