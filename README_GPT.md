# UIT Chatbot - GPT Version Setup Guide

This guide explains how to run the UIT Chatbot with GPT (OpenAI) instead of Groq.

## Quick Start with Docker (Recommended)

### Prerequisites

1. **Docker and Docker Compose** installed on your system
2. **OpenAI API Key** - Get it from [OpenAI Platform](https://platform.openai.com/api-keys)
3. **Generated data files** (run these commands before building Docker images):

```bash
# Generate ontology TTL from triplets JSON
python -m ontology.from_jsonl

# Build RAG vector index from items JSON
python -m retrieval.text_rag.build_index
```

### Setup Steps

1. **Create environment file**:
```bash
# Copy the example file
cp .env.gpt.example .env

# Edit .env and add your OpenAI API key
# OPENAI_API_KEY=your_actual_api_key_here
```

2. **Build and run with Docker Compose**:
```bash
# Build both backend and frontend images
docker compose -f docker-compose.gpt.yml build

# Start both services
docker compose -f docker-compose.gpt.yml up

# Or run in detached mode
docker compose -f docker-compose.gpt.yml up -d
```

3. **Access the application**:
   - **Frontend UI**: http://localhost:4173
   - **Backend API**: http://localhost:10000
   - **API Docs**: http://localhost:10000/docs

4. **Stop services**:
```bash
docker compose -f docker-compose.gpt.yml down
```

## Running Locally (Without Docker)

### Prerequisites

1. Python 3.11+
2. Node.js 18+ (for frontend)
3. OpenAI API Key

### Backend Setup

1. **Install Python dependencies**:
```bash
pip install -r requirements.txt
```

2. **Create `.env` file**:
```bash
cp .env.gpt.example .env
# Edit .env and add your OPENAI_API_KEY
```

3. **Generate required data files**:
```bash
python -m ontology.from_jsonl
python -m retrieval.text_rag.build_index
```

4. **Start the backend server**:
```bash
uvicorn backend.api.main_gpt:app --host 0.0.0.0 --port 10000
```

Backend will run at `http://localhost:10000`

### Frontend Setup

1. **Navigate to web directory**:
```bash
cd frontend
```

2. **Install dependencies**:
```bash
npm install
```

3. **Start development server**:
```bash
npm run dev
```

Frontend will run at `http://localhost:5173` (Vite dev server)

4. **Or build for production**:
```bash
npm run build
npm run preview
```

Production build will run at `http://localhost:4173`

## Environment Variables

Key environment variables for GPT version:

| Variable | Default | Description |
|----------|---------|-------------|
| `OPENAI_API_KEY` | (required) | Your OpenAI API key |
| `OPENAI_MODEL` | `gpt-4o-mini` | OpenAI model to use |
| `UIT_TTL_PATH` | `ontology/uit_regulations.ttl` | Path to ontology file |
| `UIT_VECTOR_DB` | `retrieval/text_rag/vector_store.db` | Path to vector database |
| `UIT_DISABLE_LOCAL_EMBEDDER` | `false` | Set to `true` for lightweight deployment |

## Features

### Markdown Formatting
The frontend now supports rich markdown formatting in bot responses:
- **Bold** and *italic* text
- `Code blocks` and syntax highlighting
- Lists (ordered and unordered)
- Tables
- Blockquotes
- Headings
- Links

### API Endpoints

- `GET /health` - Health check
- `POST /chat` - Send a question and get an answer
  ```json
  {
    "question": "Sinh viên đăng ký tối đa bao nhiêu tín chỉ?",
    "conversation_history": []
  }
  ```

## Differences from Groq Version

1. **Backend API**:
   - Uses `backend/api/main_gpt.py` instead of `backend/api/main.py`
   - Uses `backend/llm/gpt_client.py` (OpenAI API) instead of `groq_client.py`

2. **Docker Files**:
   - `Dockerfile.gpt` - Backend Dockerfile for GPT version
   - `docker-compose.gpt.yml` - Docker Compose for GPT version

3. **Environment Variables**:
   - Uses `OPENAI_API_KEY` instead of `GROQ_API_KEY`
   - Uses `OPENAI_MODEL` instead of `GROQ_MODEL`

## Troubleshooting

### API Key Issues
- Make sure your `.env` file contains a valid `OPENAI_API_KEY`
- Verify the key works: https://platform.openai.com/api-keys

### Docker Issues
- Ensure data files are generated before building: Run `python -m ontology.from_jsonl` and `python -m retrieval.text_rag.build_index`
- Check logs: `docker compose -f docker-compose.gpt.yml logs -f`

### Port Conflicts
- Backend port 10000 or frontend port 4173 already in use?
- Change ports in `docker-compose.gpt.yml` or stop conflicting services

## Support

For issues or questions, please check the main README.md or create an issue in the repository.
