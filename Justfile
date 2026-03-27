# --- Variables ---
package := "spade_llm"
pytest  := "uv run pytest"
ruff    := "uvx ruff"
mkdocs  := "uv run mkdocs"
twine   := "uvx twine"

# --- Recipes ---

# List all available commands
default:
    @just --list

# Lock dependencies with 7-day quarantine to mitigate supply-chain attacks
lock-safe:
    uv lock --exclude-newer '1 week'

# Sync dev dependencies
sync-dev:
    uv sync --extra dev

# Run standard tests
test: sync-dev
    {{pytest}}

# Run tests with coverage reporting
test-cov: sync-dev
    {{pytest}} --cov={{package}} --cov-report=term-missing

# Lint the codebase
lint:
    {{ruff}} check {{package}}

# Format and automatically fix linting errors
fix:
    {{ruff}} format .
    {{ruff}} check --fix .

# Build and upload to PyPI
ship:
    uv build
    {{twine}} check dist/*
    {{twine}} upload dist/*

# Build documentation
build-docs: sync-dev
    {{mkdocs}} build --strict

# Serve documentation locally for development
see-docs: sync-dev
    {{mkdocs}} serve --strict