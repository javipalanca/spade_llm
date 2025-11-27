# Development Guide

Detailed guide for SPADE_LLM development and testing.

## Development Environment

### Setup

```bash
# Clone repository
git clone https://github.com/javipalanca/spade_llm.git
cd spade_llm

# Create virtual environment
python -m venv .venv
source .venv/bin/activate

# Install development dependencies
pip install -e ".[dev,docs]"

# Setup pre-commit hooks
pre-commit install
```

### Project Structure

```
spade_llm/
├── spade_llm/          # Main package
│   ├── agent/          # Agent implementations
│   ├── behaviour/      # Behaviour classes
│   ├── context/        # Context management
│   ├── providers/      # LLM providers
│   ├── tools/          # Tool system
│   ├── routing/        # Message routing
│   ├── mcp/           # MCP integration
│   └── utils/         # Utilities
├── tests/             # Test suite
├── examples/          # Usage examples
├── docs/             # Documentation
└── requirements*.txt  # Dependencies
```

## Testing

### Test Organization

```
tests/
├── test_agent/        # Agent tests
├── test_behaviour/    # Behaviour tests
├── test_context/      # Context tests
├── test_providers/    # Provider tests
├── test_tools/        # Tool tests
├── test_routing/      # Routing tests
├── conftest.py        # Test configuration
└── factories.py       # Test factories
```

### Running Tests

```bash
# Run all tests
pytest

# Run with coverage
pytest --cov=spade_llm --cov-report=html

# Run specific test module
pytest tests/test_agent/

# Run specific test
pytest tests/test_agent/test_llm_agent.py::test_agent_creation

# Run with verbose output
pytest -v -s
```


### Documentation Standards

#### Docstring Format

```python
def example_function(param1: str, param2: int = 0) -> str:
    """Brief description of the function.
    
    Longer description if needed. Explain the purpose,
    behavior, and any important details.
    
    Args:
        param1: Description of first parameter
        param2: Description of second parameter with default value
        
    Returns:
        Description of return value
        
    Raises:
        ValueError: When invalid input is provided
        ConnectionError: When service is unavailable
        
    Example:
        ```python
        result = example_function("hello", 42)
        print(result)  # Output: processed result
        ```
    """
    # Implementation here
    pass
```

#### Class Documentation

```python
class ExampleClass:
    """Brief description of the class.
    
    Longer description explaining the class purpose,
    usage patterns, and important behavior.
    
    Attributes:
        attribute1: Description of attribute
        attribute2: Description of another attribute
        
    Example:
        ```python
        instance = ExampleClass(param="value")
        result = instance.method()
        ```
    """
    
    def __init__(self, param: str):
        """Initialize the class.
        
        Args:
            param: Configuration parameter
        """
        self.attribute1 = param
```


This development guide should help you contribute effectively to SPADE_LLM. For specific questions, check the existing issues or create a new one.
