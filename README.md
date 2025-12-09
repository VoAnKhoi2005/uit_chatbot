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

## Docker

### Backend

Before building the Docker image, generate the required data files:

```bash
# Generate ontology TTL from triplets JSON
python -m ontology.from_jsonl

# Build RAG vector index from items JSON
python -m retrieval.text_rag.build_index
```

Build and run the backend container:

```bash
# Build the image
docker build -t uit-chatbot-backend .

# Run the container (use --env-file to load .env)
docker run -p 8000:8000 --env-file .env uit-chatbot-backend
```

The backend will be available at `http://localhost:8000`.

### Frontend (Optional)

Build and run the frontend container:

```bash
# Build the frontend image
docker build -t uit-chatbot-frontend ./web

# Run the frontend container
# Set VITE_API_BASE_URL to point to your backend (default: http://localhost:8000)
docker run -p 4173:4173 -e VITE_API_BASE_URL="http://localhost:8000" uit-chatbot-frontend
```

The frontend will be available at `http://localhost:4173`.

**Note:** If the backend is running in a different container or host, adjust `VITE_API_BASE_URL` accordingly (e.g., `http://backend:8000` for Docker Compose, or your actual backend URL).
