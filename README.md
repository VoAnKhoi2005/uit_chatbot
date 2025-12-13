# uit_chatbot

## 🚀 GPT Version Available!
Want to use OpenAI GPT instead of Groq? See **[README_GPT.md](README_GPT.md)** for setup instructions with Docker support.

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
uvicorn backend.api.main:app --host 0.0.0.0 --port 10000
```

Environment variables:
- `OPENAI_API_KEY` (required for LLM)
- `OPENAI_MODEL` (default: gpt-4o-mini)
- `UIT_TTL_PATH` (default: ontology/uit_regulations.ttl)
- `UIT_VECTOR_DB` (default: retrieval/text_rag/vector_store.db)

LLM backend (chatbot) uses Groq:
- `GROQ_API_KEY` (or `GROK_API_KEY`) required
- `GROQ_MODEL` (default: llama3-8b-8192)

RAG retrieval:
- `UIT_DISABLE_LOCAL_EMBEDDER` (default: `false`) - Set to `true` to disable heavy embedding model for lightweight deployment (e.g., Render free tier). Uses text-only search instead of vector similarity.

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
docker run -p 10000:10000 --env-file .env uit-chatbot-backend
```

The backend will be available at `http://localhost:10000`.

### Frontend (Optional)

Build and run the frontend container:

```bash
# Build the frontend image
docker build -t uit-chatbot-frontend ./web

# Run the frontend container
# Set VITE_API_BASE_URL to point to your backend (default: http://localhost:10000)
docker run -p 4173:4173 -e VITE_API_BASE_URL="http://localhost:10000" uit-chatbot-frontend
```

The frontend will be available at `http://localhost:4173`.

**Note:** If the backend is running in a different container or host, adjust `VITE_API_BASE_URL` accordingly (e.g., `http://backend:10000` for Docker Compose, or your actual backend URL).

### Docker Compose (Recommended)

Run both backend and frontend together with a single command:

```bash
# Generate data files first (if not already done)
python -m ontology.from_jsonl
python -m retrieval.text_rag.build_index

# Build and start both services
docker compose build
docker compose up

# Or run in detached mode
docker compose up -d

# Stop services
docker compose down
```

This will start:
- **Backend API** at `http://localhost:10000`
- **Frontend UI** at `http://localhost:4173`

The frontend is automatically configured to connect to the backend service within the Docker network.
