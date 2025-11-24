# Installation

## Requirements

- Python 3.10+
- 4GB+ RAM (8GB+ for local models)

## Install

```bash
pip install spade_llm
```

## Verify Installation

```python
import spade_llm
from spade_llm import LLMAgent, LLMProvider

print(f"SPADE_LLM version: {spade_llm.__version__}")
```

## XMPP Server Setup

### Built-in SPADE Server (Recommended)

SPADE 4.0+ includes a built-in XMPP server - **no external server setup required!**

```bash
# Start SPADE's built-in server (simplest setup)
spade run
```
The built-in server provides everything you need to run SPADE-LLM agents locally. Simply start the server in one terminal and run your agents in another.

### Advanced XMPP Server Configuration

For custom setups, you can specify different ports and hosts:

```bash
# Custom host and ports
spade run --host localhost --client_port 6222 --server_port 6269

# Use IP address instead of localhost
spade run --host 127.0.0.1 --client_port 6222 --server_port 6269

# Custom ports if defaults are in use
spade run --client_port 6223 --server_port 6270
```

### Alternative: External XMPP Servers

For production or if you prefer external servers:

- **Public servers**: `jabber.at`, `jabber.org` (require manual account creation)
- **Local Prosody**: Install [Prosody](https://prosody.im/) for local hosting
- **Other servers**: Any XMPP-compliant server

## LLM Provider Setup

Choose one provider:

### OpenAI
```bash
export OPENAI_API_KEY="your-api-key"
```

### Ollama (Local)
```bash
# Install Ollama
curl -fsSL https://ollama.ai/install.sh | sh

# Download a model
ollama pull llama3.1:8b
ollama serve
```

### LM Studio (Local)
1. Download [LM Studio](https://lmstudio.ai/)
2. Download a model through the GUI
3. Start the local server

## RAG Dependencies (Optional)

For RAG (Retrieval-Augmented Generation) support, install with ChromaDB:

```bash
# Install with ChromaDB support
pip install spade_llm[chroma]
```

### Prerequisites for RAG

**Embedding Model** - Choose one:

**Ollama** (recommended for local/private use):
```bash
# Pull embedding model
ollama pull nomic-embed-text

# Verify
curl http://localhost:11434/api/tags
```

**OpenAI**:
```bash
# Use with your OpenAI API key
export OPENAI_API_KEY="your-api-key"

# Models: text-embedding-3-small, text-embedding-3-large
```

## Development Install

```bash
git clone https://github.com/sosanzma/spade_llm.git
cd spade_llm
pip install -e ".[dev]"
```

## Troubleshooting

**SPADE server not starting**: 
```bash
# Check if default ports are already in use
netstat -an | grep 5222

# Try different ports if needed
spade run --client_port 6222 --server_port 6269
```

**Agent connection issues**: Ensure SPADE server is running first
```bash
# Terminal 1: Start server
spade run

# Terminal 2: Run your agent
python your_agent.py
```

**Import errors**: Ensure you're in the correct Python environment
```bash
python -m pip install spade_llm
```

**SSL errors**: For development with built-in server, disable SSL verification:
```python
agent = LLMAgent(..., verify_security=False)
```

**Ollama connection**: Check if Ollama is running:
```bash
curl http://localhost:11434/v1/models
```
