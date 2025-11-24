"""SPADE_LLM setup script."""

from setuptools import setup, find_packages
import pathlib

# The directory containing this file
HERE = pathlib.Path(__file__).parent

# Get the long description from the README file
README = (HERE / "README.md").read_text(encoding="utf-8")

# Import version
about = {}
with open(HERE / "spade_llm" / "version.py", encoding="utf-8") as f:
    exec(f.read(), about)

setup(
    name="spade_llm",
    version=about["__version__"],
    description="Extension for SPADE to integrate Large Language Models in agents",
    long_description=README,
    long_description_content_type="text/markdown",
    author="Manel Soler Sanz",
    author_email="masosan9@upvnet.upv.es",
    url="https://github.com/sosanzma/spade_llm",
    packages=find_packages(exclude=["tests*", "examples*"]),
    include_package_data=True,
    install_requires=[
        "spade>=4.0.0",
        "openai>=1.0.0",
        "pydantic>=2.0.0",
        "aiohttp>=3.8.0",
        "python-dateutil>=2.8.2",
        "mcp>=1.8.0",
        "aiosqlite>=0.17.0",
        "aiofiles>=23.0.0",
    ],
    extras_require={
        "langchain": [
            "langchain_community>=0.3.2",
        ],
        "chroma": [
            "chromadb>=0.4.0",
        ],
        "dev": [
            "pytest>=7.0.0",
            "pytest-asyncio>=0.21.0",
            "pytest-cov>=4.0.0",
            "pytest-mock>=3.10.0",
            "flake8>=6.0.0",
            "black>=23.0.0",
            "isort>=5.12.0",
            "coverage>=7.0.0",
            "tox>=4.0.0",
            "pre-commit>=3.0.0",
            "chromadb>=0.4.0",
        ],
        "docs": [
            "sphinx>=6.0.0",
            "sphinx-rtd-theme>=1.2.0",
        ],
        "all": [
            "langchain_community>=0.3.2",
            "google-generativeai>=0.3.0",
            "anthropic>=0.5.0",
            "chromadb>=0.4.0",
        ]
    },
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Developers",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Programming Language :: Python :: 3.12",
        "Topic :: Software Development :: Libraries :: Python Modules",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "Topic :: Internet :: XMPP",
        "Operating System :: OS Independent",
    ],
    python_requires=">=3.10",
    keywords="spade llm ai agents multi-agent-systems openai ollama",
    project_urls={
        "Bug Reports": "https://github.com/sosanzma/spade_llm/issues",
        "Source": "https://github.com/sosanzma/spade_llm",
        "Documentation": "https://sosanzma.github.io/spade_llm",
    },
    entry_points={
        "console_scripts": [
            "spade-llm=spade_llm.__main__:main",
        ],
    },
)
