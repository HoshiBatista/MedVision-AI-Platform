FROM python:3.12-slim

WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends \
    libpq-dev gcc \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY app/ ./app/
COPY alembic.ini ./
COPY migrations/ ./migrations/

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1

# Default: run the FastAPI service.
# To run the Celery worker instead, override CMD:
#   celery -A app.workers.celery_app.celery worker -Q analysis -c 2 -l info
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8002"]
