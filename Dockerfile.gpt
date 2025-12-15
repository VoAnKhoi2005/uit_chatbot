# Dockerfile for UIT Chatbot Backend with GPT

FROM python:3.11-slim

WORKDIR /app

# (Optional) ổn định pip khi mạng yếu
ENV PIP_DEFAULT_TIMEOUT=300 \
    PYTHONUNBUFFERED=1 \
    HF_HOME=/app/.cache/huggingface \
    TRANSFORMERS_CACHE=/app/.cache/huggingface \
    SENTENCE_TRANSFORMERS_HOME=/app/.cache/sentence-transformers


# Copy dependency file
COPY requirements.gpt.txt ./

# Install Python dependencies (CPU only, no CUDA)
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.gpt.txt


# NOTE: Do not pre-download models in build to avoid network/torch failures.
# If you want to cache models, run download after container is up.

# Copy backend code
COPY backend/ ./backend/
COPY ontology/ ./ontology/
COPY retrieval/ ./retrieval/
COPY groq_client.py .
COPY normailizer/ ./normailizer/


# Expose port 10000
EXPOSE 10000


# Set Docker-safe envs
ENV PYTHONPATH=/app \
    HF_HOME=/app/.cache/huggingface

# Run the FastAPI app with GPT client
CMD ["uvicorn", "backend.api.main_gpt:app", "--host", "0.0.0.0", "--port", "10000"]
