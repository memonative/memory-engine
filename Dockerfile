# syntax=docker/dockerfile:1

# ─── Build stage ──────────────────────────────────────────────────────
FROM python:3.11-slim AS build

WORKDIR /app

COPY requirements.txt .
RUN pip install --upgrade pip && \
    pip install --no-cache-dir --prefix=/install -r requirements.txt

# ─── Runtime stage ────────────────────────────────────────────────────
FROM python:3.11-slim AS runtime

RUN groupadd -r memonative && useradd -r -g memonative -s /sbin/nologin memonative

WORKDIR /app

COPY --from=build /install /usr/local
COPY src/ ./src/
COPY alembic/ ./alembic/
COPY alembic.ini .

ENV PYTHONPATH=/app/src \
    PYTHONUNBUFFERED=1

USER memonative

# ─── Targets (select via --target or docker-compose command override) ─

# API: gunicorn with uvicorn workers for production concurrency
FROM runtime AS api
EXPOSE 8000
CMD ["gunicorn", "memonative.main:app", \
     "-k", "uvicorn.workers.UvicornWorker", \
     "--bind", "0.0.0.0:8000", \
     "--workers", "4", \
     "--graceful-timeout", "30", \
     "--timeout", "120", \
     "--access-logfile", "-"]

# Worker: Celery task consumer
FROM runtime AS worker
CMD ["celery", "-A", "memonative.worker", "worker", \
     "--loglevel=info", \
     "--concurrency=4", \
     "--without-heartbeat", \
     "--without-mingle"]

# Beat: Celery scheduler (single instance only)
FROM runtime AS beat
CMD ["celery", "-A", "memonative.worker", "beat", \
     "--loglevel=info"]

# Migrate: run alembic upgrade head then exit
FROM runtime AS migrate
USER root
RUN apt-get update && apt-get install -y --no-install-recommends postgresql-client && rm -rf /var/lib/apt/lists/*
USER memonative
COPY scripts/migrate.sh ./scripts/migrate.sh
CMD ["bash", "scripts/migrate.sh"]
