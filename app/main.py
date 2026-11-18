"""FastAPI app: auth, request validation, HTTP mapping. Logic lives in pipeline.py."""
from __future__ import annotations

import io
import logging
import re
from contextlib import asynccontextmanager

from fastapi import Depends, FastAPI, HTTPException, Request
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer
from pydantic import BaseModel, Field, ValidationError
from pypdf import PdfReader
from pypdf.errors import PdfReadError

from app.config import TOKENS, Principal, Settings
from app.llm import LLMError
from app.pipeline import RagPipeline

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s %(message)s")
log = logging.getLogger("rag")
logging.getLogger("httpx").setLevel(logging.WARNING)
DOC_ID_RE = re.compile(r"^[a-z0-9][a-z0-9-]{0,63}$")


@asynccontextmanager
async def lifespan(app: FastAPI):
    settings = Settings()
    pipeline = RagPipeline(settings)
    pipeline.store.wait_ready()
    pipeline.store.ensure_collections()
    for res in pipeline.ingest_corpus():
        log.info("seed ingest %s", res)
    app.state.pipeline = pipeline
    yield


app = FastAPI(title="P1 Hybrid RAG", version="1.0.0", lifespan=lifespan)
bearer = HTTPBearer(auto_error=False)


def principal(creds: HTTPAuthorizationCredentials | None = Depends(bearer)) -> Principal:
    """The tenant comes from the verified token only, never from the request body."""
    p = TOKENS.get(creds.credentials) if creds else None
    if p is None:
        raise HTTPException(401, "missing or invalid bearer token", headers={"WWW-Authenticate": "Bearer"})
    return p


class AskRequest(BaseModel):
    question: str = Field(min_length=1, max_length=2000)
    top_k: int = Field(default=5, ge=1, le=10)


class IngestText(BaseModel):
    doc_id: str
    text: str = Field(min_length=1, max_length=500_000)
    title: str | None = Field(default=None, max_length=200)


@app.get("/healthz")
def healthz() -> dict:
    return {"status": "ok"}


@app.post("/v1/rag/ask")
def ask(body: AskRequest, request: Request, who: Principal = Depends(principal)) -> dict:
    try:
        return request.app.state.pipeline.ask(who, body.question, body.top_k)
    except LLMError as exc:  # gateway mode: surface the failure, never fall back to mock
        raise HTTPException(502, {"error": "llm_gateway_error", "reason": str(exc)}) from exc


def _validate_doc_id(doc_id: str | None) -> str:
    if not doc_id or not DOC_ID_RE.match(doc_id):
        raise HTTPException(422, {"error": "invalid_doc_id", "reason": "use lowercase letters, digits and '-' (max 64)"})
    return doc_id


def _pdf_pages(data: bytes) -> list[tuple[int, str]]:
    if not data.startswith(b"%PDF"):
        raise HTTPException(415, {"error": "not_a_pdf"})
    try:
        reader = PdfReader(io.BytesIO(data))
        if reader.is_encrypted:
            raise HTTPException(422, {"error": "encrypted_pdf"})
        pages = [(i, (page.extract_text() or "").strip()) for i, page in enumerate(reader.pages, start=1)]
    except PdfReadError as exc:
        raise HTTPException(422, {"error": "corrupt_pdf", "reason": str(exc)[:200]}) from exc
    pages = [(n, t) for n, t in pages if t]
    if not pages:
        raise HTTPException(422, {"error": "no_text", "reason": "no extractable text; scanned PDFs need OCR"})
    return pages


@app.post("/v1/rag/ingest", openapi_extra={"requestBody": {"content": {
    "application/json": {"schema": IngestText.model_json_schema()},
    "multipart/form-data": {"schema": {"type": "object", "required": ["file", "doc_id"], "properties": {
        "file": {"type": "string", "format": "binary"}, "doc_id": {"type": "string"},
        "title": {"type": "string"}}}}}}})
async def ingest(request: Request, who: Principal = Depends(principal)) -> dict:
    """JSON `{doc_id, text, title?}` or multipart PDF upload (`file`, `doc_id`, `title?`).
    Documents land in the caller's tenant. Re-ingesting a doc_id replaces it (stale chunks removed)."""
    pipeline: RagPipeline = request.app.state.pipeline
    limit = pipeline.s.max_upload_bytes
    if int(request.headers.get("content-length") or 0) > limit:
        raise HTTPException(413, {"error": "too_large", "limit_bytes": limit})
    if request.headers.get("content-type", "").startswith("multipart/form-data"):
        form = await request.form()
        upload = form.get("file")
        if upload is None or isinstance(upload, str):
            raise HTTPException(422, {"error": "missing_file"})
        doc_id = _validate_doc_id(form.get("doc_id"))
        data = await upload.read(limit + 1)
        if len(data) > limit:
            raise HTTPException(413, {"error": "too_large", "limit_bytes": limit})
        pages = _pdf_pages(data)
        title = str(form.get("title") or upload.filename or doc_id)[:200]
        result = pipeline.ingest_pages(who.tenant_id, doc_id, title, upload.filename or f"{doc_id}.pdf", pages)
    else:
        try:
            body = IngestText.model_validate(await request.json())
        except (ValidationError, ValueError) as exc:
            raise HTTPException(422, {"error": "invalid_body", "reason": str(exc)[:300]}) from exc
        _validate_doc_id(body.doc_id)
        result = pipeline.ingest_text(who.tenant_id, body.doc_id, body.text, body.title)
    return {"tenant_id": who.tenant_id, **result}


@app.get("/v1/rag/documents")
def documents(request: Request, who: Principal = Depends(principal)) -> dict:
    manifests = request.app.state.pipeline.store.list_manifests(who.tenant_id)
    return {"tenant_id": who.tenant_id, "documents": [
        {k: m[k] for k in ("doc_id", "title", "source", "checksum", "ingested_at")} | {"chunks": len(m["chunk_ids"])}
        for m in manifests]}
