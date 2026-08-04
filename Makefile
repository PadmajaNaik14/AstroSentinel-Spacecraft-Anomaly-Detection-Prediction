.PHONY: data train up down logs test

data:
	python scripts/download_dataset.py
	python scripts/stage_data.py

train:
	docker compose --profile train run --rm trainer

up:
	docker compose up --build

down:
	docker compose down

logs:
	docker compose logs -f

test:
	cd ml && PYTHONPATH=. python -m pytest tests/ -q
