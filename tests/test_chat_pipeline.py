import pytest
from rdflib import Graph

from backend.llm.orchestrator import ChatPipeline
from backend.llm.question_types import QuestionType


class DummyLLM:
    async def generate(self, system_prompt: str, user_prompt: str, context: str = ""):
        return "dummy answer"

    async def generate_json(self, system_prompt: str, user_prompt: str):
        return {"label": QuestionType.EXACT_RULE.value, "reason": "test"}


class DummyVectorStore:
    def search(self, query, embedder=None, top_k=5, alpha=0.5, candidate_k=None):
        return [
            {
                "article_id": "A-1",
                "clause_id": "C-1",
                "text": "Khoản 1 nội dung.",
                "score": 0.9,
            }
        ]

    def get_chunks_by_so_hieu(self, so_hieu):
        return []


class DummyEmbedder:
    pass


@pytest.mark.asyncio
async def test_chat_pipeline_returns_structure():
    pipeline = ChatPipeline(
        llm_client=DummyLLM(),
        vector_store=DummyVectorStore(),
        embedder=DummyEmbedder(),
        ontology_graph=Graph(),
    )
    result = await pipeline.answer_question("Câu hỏi thử")
    assert "answer" in result
    assert "question_type" in result
    assert isinstance(result.get("sources"), list)

