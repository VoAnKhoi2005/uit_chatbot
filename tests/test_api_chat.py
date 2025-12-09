import pytest
from fastapi.testclient import TestClient

from backend.api import main


class DummyPipeline:
    async def answer_question(self, question: str):
        return {
            "answer": "ok",
            "question_type": "EXACT_RULE",
            "sources": [{"article_id": "A1", "clause_id": None, "text": "sample"}],
            "ontology_facts": [],
        }


@pytest.fixture(autouse=True)
def patch_pipeline(monkeypatch):
    monkeypatch.setattr(main, "pipeline", DummyPipeline())
    yield


def test_health():
    client = TestClient(main.app)
    resp = client.get("/health")
    assert resp.status_code == 200
    assert resp.json().get("status") == "ok"


def test_chat_endpoint():
    client = TestClient(main.app)
    resp = client.post("/chat", json={"question": "test"})
    assert resp.status_code == 200
    data = resp.json()
    assert data["answer"] == "ok"
    assert data["question_type"] == "EXACT_RULE"
    assert data["sources"]

