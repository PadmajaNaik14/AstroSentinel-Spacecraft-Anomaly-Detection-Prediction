# One-shot trainer image. Build context is the repo root (see docker-compose.yml).
# Not part of the default `docker compose up` service set -- run explicitly via
# `docker compose --profile train run --rm trainer` (or `make train`).
FROM python:3.11-slim

WORKDIR /app

COPY ml/requirements.txt /app/requirements.txt
RUN pip install --no-cache-dir --extra-index-url https://download.pytorch.org/whl/cpu \
    -r /app/requirements.txt

COPY ml/ml /app/ml/ml

ENV PYTHONPATH=/app/ml

ENTRYPOINT ["python", "-m", "ml.training.train_all"]
