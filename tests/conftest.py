import os

import httpx
import pytest

# Integration tests use their own collections so they never disturb the running API's data.
os.environ.setdefault("RAG_COLLECTION", "test_rag_chunks")
os.environ.setdefault("RAG_MANIFEST_COLLECTION", "test_rag_manifests")
os.environ.setdefault("RAG_CORPUS_DIR", os.path.join(os.path.dirname(__file__), "..", "data", "corpus"))
os.environ["RAG_LLM_MODE"] = "mock"

QDRANT_URL = os.environ.setdefault("QDRANT_URL", "http://rag-qdrant:6333")


def _qdrant_up() -> bool:
    try:
        return httpx.get(f"{QDRANT_URL}/readyz", timeout=2).status_code == 200
    except httpx.HTTPError:
        return False


@pytest.fixture(scope="session")
def live_client():
    if not _qdrant_up():
        pytest.skip(f"Qdrant not reachable at {QDRANT_URL}")
    from fastapi.testclient import TestClient

    from app.main import app
    from app.pipeline import RagPipeline
    from app.config import Settings

    RagPipeline(Settings()).store.drop_collections()  # start from an empty index
    with TestClient(app) as client:  # lifespan creates collections and ingests the corpus
        yield client
    client.app.state.pipeline.store.drop_collections()


ACME = {"Authorization": "Bearer token-acme-alice"}
GLOBEX = {"Authorization": "Bearer token-globex-bob"}
