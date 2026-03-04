export PYTHONPATH := $(shell pwd)
.PHONY: help install run test docker-build docker-up docker-down clean lint format

help:
	@echo "LLM Gateway Makefile"
	@echo ""
	@echo "Available commands:"
	@echo "  make install      Install dependencies"
	@echo "  make run          Run the gateway server"
	@echo "  make test         Run tests"
	@echo "  make docker-build Build Docker image"
	@echo "  make docker-up    Start with Docker Compose"
	@echo "  make docker-down  Stop Docker Compose"
	@echo "  make clean        Clean build artifacts"
	@echo "  make lint         Run linters"
	@echo "  make format       Format code"

install:
	pip install -r requirements.txt

run:
	python -m gateway.server --host 0.0.0.0 --port 8000

dev:
	python -m gateway.server --host 0.0.0.0 --port 8000 --log-level DEBUG

test:
	pytest tests/ -v

docker-build:
	docker-compose -f docker/docker-compose.yml build

docker-up:
	docker-compose -f docker/docker-compose.yml up -d

docker-down:
	docker-compose -f docker/docker-compose.yml down

docker-logs:
	docker-compose -f docker/docker-compose.yml logs -f

clean:
	find . -type f -name "*.pyc" -delete
	find . -type d -name "__pycache__" -delete
	find . -type d -name "*.egg-info" -exec rm -rf {} +
	rm -rf build/ dist/ .pytest_cache/

lint:
	flake8 gateway/ --max-line-length=120
	mypy gateway/

format:
	black gateway/ --line-length=120
	isort gateway/ --profile black

health:
	curl http://localhost:8000/health | jq .

models:
	curl http://localhost:8000/v1/models | jq .

example-chat:
	curl http://localhost:8000/v1/chat/completions \
		-H "Content-Type: application/json" \
		-d '{"model": "smart", "messages": [{"role": "user", "content": "Hello!"}], "stream": false}' | jq .

example-stream:
	curl http://localhost:8000/v1/chat/completions \
		-H "Content-Type: application/json" \
		-d '{"model": "smart", "messages": [{"role": "user", "content": "Hello!"}], "stream": true}'
