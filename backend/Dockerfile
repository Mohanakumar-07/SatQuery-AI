# SatQuery AI — API image (web layer only, no model weights, no CUDA).
# The GPU worker image is the same code with a different command; see
# docker-compose.yml and requirements-ml.txt for the model-bearing variant.
FROM python:3.11-slim

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1

WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends \
    curl \
    libgomp1 \
    && rm -rf /var/lib/apt/lists/*

COPY backend/requirements.txt ./requirements.txt
RUN pip install --no-cache-dir -r requirements.txt

# App package + config + tests. Repo root files (.env) are mounted, never baked.
COPY backend/ ./

RUN mkdir -p /app/artifacts/uploads /app/artifacts/masks /app/artifacts/geojson \
             /app/artifacts/reports /app/artifacts/checkpoints /app/var

EXPOSE 8000

HEALTHCHECK --interval=30s --timeout=5s --start-period=10s --retries=3 \
    CMD curl -fsS http://localhost:8000/api/v1/health || exit 1

CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
