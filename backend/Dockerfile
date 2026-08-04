# Build context is the repo root (see docker-compose.yml) -- this image needs
# both backend/app and the ml/ml package (the backend imports ml directly,
# it is not a separate service).
FROM python:3.11-slim

WORKDIR /app

COPY backend/requirements.txt /app/requirements.txt
RUN pip install --no-cache-dir --extra-index-url https://download.pytorch.org/whl/cpu \
    -r /app/requirements.txt

COPY ml/ml /app/ml/ml
COPY backend/app /app/backend/app

ENV PYTHONPATH=/app/ml:/app/backend
WORKDIR /app/backend

EXPOSE 8000
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
