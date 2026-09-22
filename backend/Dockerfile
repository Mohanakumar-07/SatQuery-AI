# SatQuery AI — API image (web layer, async engine, and remote VLM gateway)
FROM python:3.11-slim

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1 \
    PYTHONPATH=/app:/app/backend:/app/src

WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends \
    curl \
    libgomp1 \
    && rm -rf /var/lib/apt/lists/*

COPY backend/requirements.txt ./requirements.txt
COPY backend/requirements-geospatial.txt ./requirements-geospatial.txt
RUN pip install --no-cache-dir -r requirements.txt && \
    pip install --no-cache-dir rasterio shapely pyproj httpx

COPY src/ ./src/
COPY backend/ ./backend/

WORKDIR /app/backend

RUN mkdir -p /app/artifacts/uploads /app/artifacts/masks /app/artifacts/geojson \
             /app/artifacts/reports /app/artifacts/checkpoints /app/var

EXPOSE 8000

CMD ["sh", "-c", "uvicorn app.main:app --host 0.0.0.0 --port ${PORT:-8000}"]
