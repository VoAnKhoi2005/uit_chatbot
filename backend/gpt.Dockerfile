# Dockerfile for UIT Chatbot Backend with GPT
# All backend dependencies are now contained within the backend/ folder

FROM python:3.11-slim

WORKDIR /app

# Install system dependencies including Java for VnCoreNLP
RUN apt-get update && apt-get install -y --no-install-recommends \
    openjdk-21-jdk \
    && rm -rf /var/lib/apt/lists/*

# Copy ONLY dependency file first (for better caching)
COPY requirements.gpt.txt ./requirements.txt

# Install Python dependencies - CPU-only (cached unless requirements changes)
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

# Pre-download the embedding model during build (cached in Docker layer)
# This speeds up container startup significantly
RUN python -c "from sentence_transformers import SentenceTransformer; SentenceTransformer('keepitreal/vietnamese-sbert')"

# Copy entire backend folder (contains all necessary code and data)
COPY . ./backend/

# Expose port 10000
EXPOSE 10000

# Set Docker-safe envs
ENV PYTHONUNBUFFERED=1 \
    PYTHONPATH=/app \
    HF_HOME=/app/.cache/huggingface

# Run the FastAPI app with GPT client
CMD ["uvicorn", "backend.api.main_gpt:app", "--host", "0.0.0.0", "--port", "10000"]
