"""SPADE_LLM command line interface."""

import argparse

from .version import __version__


def main():
    """Main entry point for SPADE_LLM command line interface."""
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
        print("Available examples:")
        print("- spanish_to_english_translator.py")
        print("- ollama_with_tools_example.py")
        print("- multi_provider_chat_example.py")
        print("- valencia_smartCity_mcp_example.py")
        print("\nSee examples/ directory for complete code")
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
