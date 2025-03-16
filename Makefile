.PHONY: setup run test lint clean docker-build docker-run docker-stop

setup:
	@echo "Creating virtual environment..."
	python3 -m venv venv
	. venv/bin/activate && \
	pip install --upgrade pip && \
	pip install -r requirements.txt || (echo "Failed to install dependencies" && exit 1)

run:
	python launch.py

test:
	pytest tests/

lint:
	black .
	mypy .
	flake8 .

clean:
	rm -rf venv
	rm -rf __pycache__
	rm -rf .pytest_cache
	find . -type d -name "__pycache__" -exec rm -r {} +
	find . -type f -name "*.pyc" -delete
	rm -rf kaleidoscope_data/*

docker-build:
	docker-compose build

docker-run:
	docker-compose up

docker-stop:
	docker-compose down
