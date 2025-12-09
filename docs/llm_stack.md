# LLM stack (Groq)

- All chatbot LLM calls go through Groq using the shared helper `groq_client.call_groq_llm`.
- `backend/llm/LLMClient` wraps Groq and exposes `generate` / `generate_json`.
- Question classifier, ChatPipeline, and FastAPI `/chat` all rely on this Groq-backed client.

## Environment variables
- `GROQ_API_KEY` (required; `GROK_API_KEY` also accepted for backward compatibility)
- `GROQ_MODEL` (default: `llama3-8b-8192`)

## Legacy note
- `graph/src/triplet_extraction/llm/client.py` still uses OpenAI but is not used by the chatbot path; kept for triplet extraction workflows.

