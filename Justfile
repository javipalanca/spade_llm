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
    uvx ruff check spade_llm

# Format and automatically fix linting errors
fix:
    uvx ruff format .
    uvx ruff check --fix .

# Build and upload to PyPI
ship:
    uv build
    uvx twine check dist/*
    uvx twine upload dist/*

# Build documentation
build-docs:
    uv run mkdocs build --strict

# Serve documentation locally for development
see-docs:
    uv run mkdocs serve --strict