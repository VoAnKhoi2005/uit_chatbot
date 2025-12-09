# uit_chatbot

## Quick start (API)
1) Generate ontology (if not already):
```
python -m ontology.from_jsonl
```
2) Build RAG index:
```
python -m retrieval.text_rag.build_index
```
3) Start FastAPI server:
```
uvicorn backend.api.main:app --host 0.0.0.0 --port 8000
```

Environment variables:
- `OPENAI_API_KEY` (required for LLM)
- `OPENAI_MODEL` (default: gpt-4o-mini)
- `UIT_TTL_PATH` (default: ontology/uit_regulations.ttl)
- `UIT_VECTOR_DB` (default: retrieval/text_rag/vector_store.db)

LLM backend (chatbot) uses Groq:
- `GROQ_API_KEY` (or `GROK_API_KEY`) required
- `GROQ_MODEL` (default: llama3-8b-8192)
