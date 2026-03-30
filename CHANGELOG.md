# Changelog

## [0.3.0] - 2026-03-31

### Added

- **LiteLLM Provider Migration**: Unified `LLMProvider` powered by LiteLLM, supporting all LiteLLM-compatible models (OpenAI, Anthropic, Hugging Face, LM Studio, vLLM, etc.) through a single interface with `model="provider/model"` format.
- **RAG System**:
  - Document model with serialization support.
  - Document loaders: `TextLoader`, `DirectoryLoader`.
  - Text splitters: `CharacterTextSplitter`, `RecursiveCharacterTextSplitter`.
  - Vector stores: abstract `VectorStore` with optional `Chroma` (ChromaDB) backend.
  - Retrievers: `VectorStoreRetriever` for semantic similarity search.
  - `RetrievalAgent` and `RetrievalBehaviour` for SPADE-integrated document retrieval.
  - `RetrievalTool` enabling LLM agents to query retrieval agents via XMPP.
- **Structured Outputs**: Structured output generation using Pydantic models.
  - `ReadyForStructuredOutputTool` signal tool injection when using both tools and structured output.
  - `output_schema` parameter in `LLMProvider` and `LLMAgent`.
- **Parallel Subagents**: Coordinator can now manage multiple subagents in parallel, configurable per-subagent response timeouts.
- **OpenTelemetry Support**: Conversation metadata handling with session and conversation IDs.
- **New examples**: `rag_system_ollama_chroma.py`, `rag_vs_no_rag.py`, `coordinator_params_predict/`.

### Changed

- **Breaking**: Replaced individual provider classes with a single `LLMProvider` backed by LiteLLM. Old provider-specific classes are no longer available.
- **Breaking**: `BaseLLMProvider.get_llm_response()` now returns a dict with `text`, `tool_calls`, and `structured` keys instead of the previous format.
- Migrated build system from `setup.py` to `pyproject.toml` with `hatchling`.
- Replaced `Makefile` with `Justfile` (using `just` command runner).
- Replaced `flake8` + `black` + `isort` with `ruff` for linting and formatting.
- Updated all CI workflows to use `uv` package manager.
- Restructured docs directory from `docs/docs/` to `docs/`.
- Restricted Python version support to `>=3.11, <3.13`.
- Updated `pyjabber>=0.4.1` (0.4.0 had Python 3.12+ f-string syntax issues).
- MCP adapter class names simplified (removed transport prefix for clarity).
- Updated example scripts: removed `_example` suffix, added automated CLI listing, standardized configuration.

### Removed

- `setup.py`, `requirements.txt`, `requirements_dev.txt`, `MANIFEST.in`, `override.txt`, `tox.ini`.
- Individual LLM provider implementations (replaced by LiteLLM-based `LLMProvider`).

## [0.2.0] - 2025-10-02

- Added new examples, CoordinatorAgent, and adapted tests and documentation.

## [0.1.1] - 2025-08-10

- Modified documentation and examples.

## [0.1.0] - 2025-07-30

- Initial release.
