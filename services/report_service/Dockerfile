FROM python:3.11-slim

WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Install CPU-only torch first to keep image lean (~800MB vs 3GB for CUDA)
RUN pip install --no-cache-dir \
    torch==2.5.1 --index-url https://download.pytorch.org/whl/cpu

COPY requirements.txt .
# Skip torch re-install (already installed above)
RUN grep -v "^torch" requirements.txt | pip install --no-cache-dir -r /dev/stdin

COPY app/ ./app/
COPY templates/ ./templates/

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1

EXPOSE 8003

CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8003"]
