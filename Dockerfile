# ==============================================================================
# Project VaniShield AI — Production Cloud Streaming Server Dockerfile
# Target: Voroa / Render / Railway / AWS / GCP Cloud Run / RunPod / Kubernetes
# ==============================================================================

# --- Stage 1: Build Frontend React Dashboard ---
FROM node:20-alpine AS frontend-builder
WORKDIR /build/frontend
COPY frontend/package*.json ./
RUN npm install
COPY frontend/ ./
RUN npm run build

# --- Stage 2: Production Python Streaming Server ---
FROM python:3.11-slim

# Set environment flags
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    DEBIAN_FRONTEND=noninteractive \
    PORT=8000

# Install essential system audio & networking dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    ffmpeg \
    libsndfile1 \
    curl \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Install Python dependencies (using PyTorch CPU wheels for fast, lightweight cloud build)
COPY requirements.txt .
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt --extra-index-url https://download.pytorch.org/whl/cpu

# Copy project source code
COPY . /app

# Copy built frontend from Stage 1 into /app/frontend/dist
COPY --from=frontend-builder /build/frontend/dist /app/frontend/dist

# Expose backend port
EXPOSE 8000

# Health check with dynamic port fallback
HEALTHCHECK --interval=30s --timeout=5s --start-period=30s --retries=3 \
    CMD sh -c "curl -f http://localhost:${PORT:-8000}/health || exit 1"

# Launch FastAPI ASGI streaming server with dynamic PORT evaluation
CMD ["sh", "-c", "uvicorn backend.main:app --host 0.0.0.0 --port ${PORT:-8000}"]
