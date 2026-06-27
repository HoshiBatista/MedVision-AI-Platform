# ── Build stage: deps into an isolated venv ──────────────────────────────────
FROM python:3.11-slim@sha256:cdbd05fb6f457ca275ff51ce00d93d865ca0b6a25f5ffb08262d94f6835771e5 AS builder
ENV PIP_NO_CACHE_DIR=1
RUN python -m venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"
WORKDIR /app
COPY requirements.txt .
RUN pip install -r requirements.txt

# ── Runtime stage ────────────────────────────────────────────────────────────
# No torch/transformers here — generation is delegated to the Ollama service.
FROM python:3.11-slim@sha256:cdbd05fb6f457ca275ff51ce00d93d865ca0b6a25f5ffb08262d94f6835771e5 AS runtime
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PATH="/opt/venv/bin:$PATH"
RUN groupadd --gid 10001 appuser \
 && useradd --uid 10001 --gid 10001 --create-home appuser
COPY --from=builder /opt/venv /opt/venv
WORKDIR /app
COPY app/ ./app/
COPY templates/ ./templates/
COPY alembic.ini ./
COPY migrations/ ./migrations/
USER appuser
EXPOSE 8005
HEALTHCHECK --interval=20s --timeout=5s --start-period=20s --retries=6 \
  CMD ["python", "-c", "import urllib.request; urllib.request.urlopen('http://localhost:8005/health', timeout=3)"]
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8005"]
