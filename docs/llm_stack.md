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

## Legacy note
- `graph/src/triplet_extraction/llm/client.py` uses its own OpenAI client for
  offline triplet extraction workflows (not the chat path, not `LLMClient`).
