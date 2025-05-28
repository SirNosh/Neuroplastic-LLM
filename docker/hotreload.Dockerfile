# syntax=docker/dockerfile:1

FROM python:3.10-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

WORKDIR /app

COPY requirements.txt ./
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

COPY hot_reload ./hot_reload
COPY config ./config
# COPY serving ./serving # Not needed if controller uses API

CMD ["python", "hot_reload/controller.py", "--config", "config/base.yaml"] 