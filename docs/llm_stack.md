# LLM stack

- All chatbot LLM calls go through one generic OpenAI-protocol client:
  `backend/llm/client.py::LLMClient`, exposing `generate` / `generate_json`.
- It's not tied to any one provider - it points the `openai` SDK's client at
  whatever `LLM_BASE_URL` is configured, so it works against OpenAI, Groq,
  OpenRouter, a local vLLM/Ollama server, or anything else that speaks the
  OpenAI chat-completions API.
- `ChatPipeline` (retrieval routing, the in/out-of-scope gate in
  `llm/scope_gate.py`), FastAPI's `/chat` (`backend/main.py`), and offline
  scripts (`create_batch_queries.py`, `eval/graphrag_pipeline`'s `collect`
  node) all share this one client.

## Environment variables
- `LLM_BASE_URL` (required) - the OpenAI-protocol endpoint to call, e.g.
  `https://api.openai.com/v1`, `https://api.groq.com/openai/v1`, or a local
  server's `/v1` URL.
- `LLM_API_KEY` (required; `OPENAI_API_KEY` also accepted as a fallback so an
  existing `.env` doesn't need every key renamed at once)
- `LLM_MODEL` (required; `OPENAI_MODEL` also accepted as a fallback)

## Embeddings

Same idea, separate client: `backend/retrieval/text_rag/embeddings.py::TextEmbedder`
calls an OpenAI-protocol embeddings endpoint (`client.embeddings.create(...)`)
server-side, instead of loading a model locally - no torch/sentence-transformers
needed for retrieval. Used by `ChunkVectorStore`'s dense signal (Path 1) and
`GraphRetriever`'s node/triple embeddings (Path 2); `eval/graphrag_pipeline`'s
RAGAS judge (`answer_correctness`) uses the same config for its embedding-dependent
scoring, via `ragas.embeddings.OpenAIEmbeddings`.

- `EMBEDDING_BASE_URL` / `EMBEDDING_API_KEY` (optional) - fall back to
  `LLM_BASE_URL`/`LLM_API_KEY`, since a provider like OpenRouter serves both
  chat and embeddings from the same account/endpoint.
- `EMBEDDING_MODEL` (default: `qwen/qwen3-embedding-8b`)

Switching the embedding model changes the vector space, so `vector_store.db`
(text chunks) and the graph embedding cache
(`backend/retrieval/src/db/faiss_indexes/`) need rebuilding after a change -
see the main README's "Data rebuild" section.

## Legacy note
- `graph/src/triplet_extraction/llm/client.py` uses its own OpenAI client for
  offline triplet extraction workflows (not the chat path, not `LLMClient`).
- `retrieval.src.db.vector_db.ConceptRelationDB` still loads a local
  `sentence-transformers` model directly - superseded by `GraphRetriever`
  (see `retrieval.src.retrieval.graph_retriever`) and unused by the live
  pipeline; `sentence-transformers` is not in `backend/requirements.txt`
  because of this, install it separately if you need that legacy path.
