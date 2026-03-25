"""SPADE_LLM command line interface."""

import argparse
import pathlib
from .version import __version__

def has_examples(path):
    if path.is_file():
        return path.suffix == ".py" and path.name != "__init__.py"
    return any(has_examples(child) for child in path.iterdir())

def print_tree(directory, prefix=""):
    # Filter: must be .py OR a dir with .py files
    items = sorted([
        f for f in directory.iterdir()
        if (f.is_file() and f.suffix == ".py" and f.name != "__init__.py")
        or (f.is_dir() and not f.name.startswith("__") and has_examples(f))
    ])

    for i, path in enumerate(items):
        is_last = (i == len(items) - 1)
        connector = "└── " if is_last else "├── "
        
        print(f"{prefix}{connector}{path.name}")
        
        if path.is_dir():
            extension = "    " if is_last else "│   "
            print_tree(path, prefix + extension)

def main():
    parser = argparse.ArgumentParser(
        description="SPADE_LLM - Extension for SPADE to integrate Large Language Models",
        prog="spade-llm",
    )

    parser.add_argument(
        "--version", action="version", version=f"SPADE_LLM {__version__}"
    )

    subparsers = parser.add_subparsers(dest="command", help="Available commands")

    # Info command
    subparsers.add_parser("info", help="Show SPADE_LLM information")

    # Example command
    subparsers.add_parser("examples", help="List available examples")

    args = parser.parse_args()

    if args.command == "info":
        print(f"SPADE_LLM version {__version__}")
        print("Extension for SPADE to integrate Large Language Models in agents")
        print("Visit https://github.com/javipalanca/spade_llm for more information")
    elif args.command == "examples":
        examples_dir = pathlib.Path(__file__).parent.parent.resolve() / "examples"
        print(f"Available examples:\n{examples_dir.name}/")
        print_tree(examples_dir)
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
