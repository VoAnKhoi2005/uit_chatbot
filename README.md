![UIT](https://img.shields.io/badge/from-UIT%20VNUHCM-blue?style=for-the-badge&link=https%3A%2F%2Fwww.uit.edu.vn%2F)
# UIT Chatbot

UIT Chatbot is a regulation assistant for the University of Information Technology (UIT).  
It uses a hybrid Retrieval-Augmented Generation (RAG) pipeline that combines:

- ontology graph data
- text chunk retrieval from regulation documents
- LLM-based answer generation

## Product overview

### Web interface preview
The main chat page where users ask questions and receive regulation-grounded answers.

![UIT Chatbot main page](./Picture1%20crop.png)

### End-to-end processing pipeline
High-level flow from user question through retrieval, grounding, and final response generation.

![UIT Chatbot pipeline](./Pipeline%20UIT%20Chatbot.png)

## Repository main layout:
- `backend/`: FastAPI app and chatbot pipeline
- `frontend/`: Vite + React web client
- `backend/ontology/`: ontology loader and generated `uit_regulations.ttl`
- `backend/retrieval/`: text and graph retrieval logic
- `docker-compose.gpt.yml`: GPT backend + frontend

## Prerequisites

- Python 3.11+
- Node.js 20+
- npm
- Docker (optional, for Compose-based run)

## Quick start (recommended: Docker Compose)

1. Create `.env` in the repository root:
   - `OPENAI_API_KEY=...`
2. Start services:
   - `docker compose -f docker-compose.gpt.yml up --build`
3. Open:
   - Frontend: `http://localhost:4173`
   - Backend health: `http://localhost:10000/health`

Stop with `docker compose -f docker-compose.gpt.yml down`.

## Local run (without Docker)

### 1) Backend

Install dependencies:

```powershell
python -m pip install -r backend\requirements.txt
```

For GPT variant:

```powershell
python -m pip install -r backend\requirements.gpt.txt
```

Run API from repo root:

```powershell
uvicorn main:app --app-dir backend --host 0.0.0.0 --port 10000
```

Run GPT API variant:

```powershell
uvicorn main_gpt:app --app-dir backend --host 0.0.0.0 --port 10000
```

### 2) Frontend

```powershell
Set-Location frontend
npm install
npm run dev
```

Frontend default URL: `http://localhost:5173`  
Set API endpoint with `VITE_API_BASE_URL` if needed.

## Data rebuild (only when source data changes)

If you update exported UIT source files and need to regenerate backend artifacts:

```powershell
$env:UIT_TRIPLETS_PATH="backend/graph/mongo_export_uit/KB_UIT.triplets.json"
$env:UIT_ITEMS_PATH="backend/graph/mongo_export_uit/KB_UIT.items.json"
$env:UIT_TTL_PATH="backend/ontology/uit_regulations.ttl"
python -m ontology.from_jsonl

$env:UIT_CONTENT_JSON="backend/graph/mongo_export_uit/KB_UIT.items.json"
$env:UIT_VECTOR_DB="backend/retrieval/text_rag/vector_store.db"
python -m retrieval.text_rag.build_index
```

## Environment variables

Common runtime variables:

- `OPENAI_API_KEY` (GPT backend)
- `OPENAI_MODEL` (default: `gpt-4o-mini`)
- `UIT_TTL_PATH` (default expects `backend/ontology/uit_regulations.ttl`)
- `UIT_VECTOR_DB` (default expects backend vector DB path)
- `UIT_DISABLE_LOCAL_EMBEDDER` (`true`/`false`, lightweight retrieval mode)
- `UIT_DISABLE_TRIPLET_RAG` (`true`/`false`, disables graph triplet retrieval)

## API endpoints

- `GET /health`
- `POST /chat`
- `GET /document/{doc_id}`

## Contributors

- Vo Hong Luong - 23520905 - https://github.com/luong-vh
- Vo An Khoi - 23520790 - https://github.com/VoAnKhoi2005
- Pham Thi Kieu Diem - 23520286 - https://github.com/korobe0906

Supervisor: Dr. Do Trong Hop

## Disclaimer

This project is for educational purposes.  
Always verify important academic decisions with official UIT sources.
