# List all available commands
default:
    @just --list

# Lock dependencies with 7-day quarantine to mitigate supply-chain attacks
lock-safe:
    uv lock --exclude-newer '1 week'

# Run standard tests
test:
    uv run pytest

# Run tests with coverage reporting
test-cov:
    uv run pytest --cov=spade_llm --cov-report=term-missing

# Lint the codebase
lint:
    uv run ruff check spade_llm

# Format and automatically fix linting errors
fix:
    uv run ruff format .
    uv run ruff check --fix .

# Build and upload to PyPI
ship:
    uv build
    uv run twine check dist/*
    uv run twine upload dist/*

docs:
    cd docs && uv run mkdocs build --strict