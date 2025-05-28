# syntax=docker/dockerfile:1

FROM python:3.10-slim AS base

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

WORKDIR /app

# Install system deps (for torch / transformers CPU fallback)
RUN apt-get update && \
    apt-get install -y --no-install-recommends git gcc && \
    rm -rf /var/lib/apt/lists/*

COPY requirements.txt ./
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

COPY . .

# Expose vLLM port
EXPOSE 8000

CMD ["python", "main.py", "serve", "--config", "config/base.yaml"] 