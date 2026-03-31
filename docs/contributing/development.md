# Development Guide

Detailed guide for SPADE_LLM development and testing.

## Development Environment

### Prerequisites
| Tool | Requirement | Purpose |
| :--- | :--- | :--- |
| [**git**](https://git-scm.com/install/) | **Required** | Version control |
| [**uv**](https://docs.astral.sh/uv/getting-started/installation/) | **Required** | Python management |
| [**just**](https://www.google.com/search?q=https://github.com/casey/just%23installation) | *Optional* | Task runner |


### Setup

```bash
# Clone repository
git clone https://github.com/javipalanca/spade_llm.git
# If you forked, clone your fork instead
cd spade_llm

# Install development dependencies
uv sync --extra dev
```


## Testing


### Running Tests

```bash
# Run all tests
uv run pytest

# Run with coverage
uv run pytest --cov=spade_llm

# Run just to see available tasks (tests, linting, etc.)
just
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


This development guide should help you contribute to SPADE_LLM. For specific questions, check the existing issues or create a new one.
