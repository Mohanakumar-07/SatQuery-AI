# SatQuery AI — Production API Container for Render / Cloud Deployment
FROM python:3.11-slim

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1 \
    PYTHONPATH=/app:/app/backend:/app/src

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    curl \
    libgomp1 \
    && rm -rf /var/lib/apt/lists/*

# Install python dependencies
COPY backend/requirements.txt ./requirements.txt
COPY backend/requirements-geospatial.txt ./requirements-geospatial.txt
RUN pip install --no-cache-dir -r requirements.txt && \
    pip install --no-cache-dir rasterio shapely pyproj scipy httpx

# libexpat1 is required at runtime by rasterio's manylinux wheel (GDAL XML parsing).
# Separate RUN step so this layer is never served stale from Docker cache.
RUN apt-get update && apt-get install -y --no-install-recommends libexpat1 && rm -rf /var/lib/apt/lists/*

# Copy source tree and backend application
COPY src/ ./src/
COPY backend/ ./backend/

WORKDIR /app/backend

# Create runtime directories for evidence and artifacts
RUN mkdir -p /app/artifacts/uploads /app/artifacts/masks /app/artifacts/geojson \
             /app/artifacts/reports /app/artifacts/checkpoints /app/var

EXPOSE 8000

# Respect Render's dynamic $PORT environment variable
CMD ["sh", "-c", "uvicorn app.main:app --host 0.0.0.0 --port ${PORT:-8000}"]
