# Dockerfile for UIT Chatbot Backend
#
# All backend dependencies are now contained within the backend/ folder

FROM python:3.11-slim

WORKDIR /app

# ============================================
# LAYER 1: Install dependencies (CACHED)
# This layer only rebuilds when requirements.txt changes
# ============================================

# Copy ONLY requirements file first
COPY backend/requirements.txt .

# Install Python dependencies with no-cache to reduce image size
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

# ============================================
# LAYER 2: Copy application code (CHANGES OFTEN)
# Code changes won't invalidate pip install layer above
# ============================================

# Copy entire backend folder (contains all necessary code and data)
COPY backend/ ./backend/

# ============================================
# LAYER 3: Configuration
# ============================================

# Expose port 10000
EXPOSE 10000

# Use .env file passed at runtime (via --env-file)
# Do not bake secrets into the image
ENV PYTHONUNBUFFERED=1 \
    PYTHONPATH=/app

# Run the FastAPI app
CMD ["uvicorn", "backend.api.main:app", "--host", "0.0.0.0", "--port", "10000"]

