# ── Build stage: install deps into an isolated venv ──────────────────────────
FROM python:3.13-slim@sha256:2b7445fb71ca9cb15e9aab053fe8cb3162796f8e1d92ada12a49c766a811bc1e AS builder
ENV PIP_NO_CACHE_DIR=1
RUN python -m venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"
WORKDIR /app
COPY requirements.txt .
RUN pip install -r requirements.txt

# ── Runtime stage: copy the venv, run as a non-root user ─────────────────────
FROM python:3.13-slim@sha256:2b7445fb71ca9cb15e9aab053fe8cb3162796f8e1d92ada12a49c766a811bc1e AS runtime
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PATH="/opt/venv/bin:$PATH" \
    STORAGE_ROOT=/data/studies
RUN groupadd --gid 10001 appuser \
 && useradd --uid 10001 --gid 10001 --create-home appuser \
 && mkdir -p /data/studies \
 && chown -R appuser:appuser /data
COPY --from=builder /opt/venv /opt/venv
WORKDIR /app
COPY app/ ./app/
COPY alembic.ini ./
COPY migrations/ ./migrations/
USER appuser
EXPOSE 8002
HEALTHCHECK --interval=15s --timeout=5s --start-period=25s --retries=5 \
  CMD ["python", "-c", "import urllib.request; urllib.request.urlopen('http://localhost:8002/health', timeout=3)"]
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8002"]
