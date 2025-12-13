# UIT Chatbot - Regulations Assistant

An AI-powered chatbot for UIT (University of Information Technology) regulations and academic policies.

## 📋 Project Overview

This chatbot helps students and staff quickly find information about UIT's training regulations, academic policies, and administrative procedures. It uses RAG (Retrieval-Augmented Generation) with ontology-based knowledge graphs and vector search to provide accurate, cited answers.

## 👥 Project Team

| Name | Class | Student ID |
|------|-------|------------|
| Võ An Khôi | KTPM2023.2 | 23520790 |
| Võ Hồng Lương | KTPM2023.2 | 23520905 |
| Phạm Thị Kiều Diễm | KTPM2023.1 | 23520286 |

## ⚖️ Disclaimer

This project is developed for **educational purposes only** as part of an academic assignment at the University of Information Technology (UIT).

**Important Notes:**
- The information provided by this chatbot is based on available UIT regulation documents and may not always be up-to-date or complete
- Users should always verify critical information with official UIT sources and academic advisors
- This is a prototype system and should not be used as the sole source for making important academic decisions
- The developers are not responsible for any decisions made based on the chatbot's responses
- All regulation documents and data belong to their respective copyright holders

For official and authoritative information, please refer to:
- [UIT Official Website](https://www.uit.edu.vn/)
- UIT Student Affairs Office
- Your academic advisor

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

## 🎓 Academic Context

This project was developed as part of the coursework at the University of Information Technology, VNU-HCM. It demonstrates the application of:
- Natural Language Processing (NLP)
- Retrieval-Augmented Generation (RAG)
- Knowledge Graphs and Ontologies
- Modern web development practices
- Docker containerization

## 📝 License

This project is for educational purposes. All UIT regulation documents and related content are property of the University of Information Technology.

---

**Developed by KTPM2023 students at University of Information Technology (UIT), VNU-HCM**
