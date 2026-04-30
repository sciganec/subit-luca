# Makefile for subit-luca
# Targets: install, test, lint, format, clean, experiments, help

.PHONY: help install install-dev install-all test lint format clean \
        run-luca-projection run-subit-umap run-cross-species run-phylogeny \
        run-all-experiments

# Detect Python interpreter
PYTHON ?= python3

# Default target
help:
	@echo "Available targets:"
	@echo "  install               Install core dependencies"
	@echo "  install-dev           Install core + dev dependencies (pytest, black, ruff, mypy)"
	@echo "  install-all           Install all dependencies (dev + viz + notebooks)"
	@echo "  test                  Run pytest with coverage"
	@echo "  lint                  Run ruff linter"
	@echo "  format                Format code with black"
	@echo "  clean                 Remove cache, build artifacts, and .pyc files"
	@echo "  run-luca-projection   Run experiment 01_luca_projection.py"
	@echo "  run-subit-umap        Run experiment 02_subit_umap.py"
	@echo "  run-cross-species     Run experiment 03_cross_species.py"
	@echo "  run-phylogeny         Run experiment 04_phylogeny.py"
	@echo "  run-all-experiments   Run all experiments (sequential)"
	@echo "  help                  Show this help"

install:
	$(PYTHON) -m pip install -e .

install-dev:
	$(PYTHON) -m pip install -e .[dev]

install-all:
	$(PYTHON) -m pip install -e .[all]

test:
	$(PYTHON) -m pytest tests/ --cov=subit --cov-report=term-missing

lint:
	$(PYTHON) -m ruff check .

format:
	$(PYTHON) -m black .

clean:
	find . -type d -name "__pycache__" -exec rm -rf {} + 2>/dev/null || true
	find . -type f -name "*.pyc" -delete
	find . -type f -name "*.pyo" -delete
	find . -type f -name ".coverage" -delete
	find . -type d -name ".pytest_cache" -exec rm -rf {} + 2>/dev/null || true
	find . -type d -name ".ruff_cache" -exec rm -rf {} + 2>/dev/null || true
	find . -type d -name ".mypy_cache" -exec rm -rf {} + 2>/dev/null || true
	find . -type d -name "htmlcov" -exec rm -rf {} + 2>/dev/null || true
	rm -rf build/ dist/ *.egg-info/

run-luca-projection:
	$(PYTHON) experiments/01_luca_projection.py

run-subit-umap:
	$(PYTHON) experiments/02_subit_umap.py

run-cross-species:
	$(PYTHON) experiments/03_cross_species.py

run-phylogeny:
	$(PYTHON) experiments/04_phylogeny.py

run-all-experiments: run-luca-projection run-subit-umap run-cross-species run-phylogeny