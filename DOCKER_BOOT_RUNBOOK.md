# UIT_CHATBOT Docker Boot Doctor Runbook

## Files Changed
- Dockerfile.gpt: Cleaned requirements install, set Docker-safe envs, removed pre-downloads.
- docker-compose.gpt.yml: Fixed backend service envs, volumes, fallback always-boot flags.
- requirements.gpt.txt / requirements.txt: Minimal, CPU-safe, Docker-bootable deps only.
- retrieval/src/retrieval/triplet_retriever.py: Typing modernized, Docker importable, no Mongo.
- retrieval/src/db/vector_db.py: Typing modernized, Docker importable, no Mongo.
- backend/llm/orchestrator.py: ChatPipeline constructor flexible, always boots.
- backend/tools/smoke_imports.py: Added boot smoke test for critical imports.

## How to Build & Boot

1. **Build backend image:**
   ```sh
   docker compose -f docker-compose.gpt.yml build backend
   ```
2. **Start backend service:**
   ```sh
   docker compose -f docker-compose.gpt.yml up -d backend
   ```
3. **Tail backend logs:**
   ```sh
   docker compose -f docker-compose.gpt.yml logs -f backend --tail 200
   ```
4. **Check API docs:**
   - Open: http://localhost:10000/docs
   - Or: `curl http://localhost:10000/docs`

## Fallback Always-Boot Mode
- To force backend to boot even if retrieval/embedding fails:
  - Set in `.env` or compose env:
    - `UIT_DISABLE_TRIPLET_RAG=1`
    - `UIT_DISABLE_LOCAL_EMBEDDER=1`

## Smoke Test (optional, local)
   ```sh
   python backend/tools/smoke_imports.py
   ```

---

**If backend fails to boot, check logs for import errors or missing envs.**

**All MongoDB, CUDA, and Windows path dependencies are removed.**

**You are ready to run UIT_CHATBOT in Docker!**
