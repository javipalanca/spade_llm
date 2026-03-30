"""
Valencia Smart City Assistant

An LLM agent with access to real-time Valencia city data via MCP:
- Weather forecasts
- Traffic conditions
- Valenbisi bike availability
- Air quality

Setup:
  1. Install uv: https://docs.astral.sh/uv/getting-started/installation/
  2. cp .env.example .env  (fill in LLM_MODEL and VALENCIA_MCP_PATH)
  3. spade run             (in a separate terminal)
  4. python examples/valencia_smartCity_mcp.py

The Valencia Smart City MCP server runs automatically from GitHub (requires uv).
Set VALENCIA_MCP_PATH in .env to use a local clone instead.
"""

import os
import spade

from spade_llm.agent import LLMAgent, ChatAgent
from spade_llm.providers import LLMProvider
from spade_llm.mcp import StdioServerConfig
from spade_llm.utils import load_env_vars

from rich.console import Console
from rich.panel import Panel
from rich.markdown import Markdown
from rich import box

load_env_vars()

LLM_MODEL = os.environ.get("LLM_MODEL")
if not LLM_MODEL:
    raise SystemExit("LLM_MODEL is not set — copy .env.example to .env and configure it.")
LOCAL_MCP_PATH = os.environ.get("VALENCIA_MCP_PATH")

console = Console()

async def main():
    console.print("\n")
    console.print(Panel(
        "[bold cyan]Valencia Smart City Assistant[/bold cyan]\n"
        "[bold]Powered by MCP & SPADE[/bold]",
        box=box.DOUBLE,
        border_style="cyan"
    ))

    xmpp_server = os.environ.get("XMPP_SERVER", "localhost")

    llm_jid = f"llm_agent@{xmpp_server}"
    llm_password = "llm_pass"

    # Valencia Smart City MCP server configuration
    console.print("\n[cyan]MCP Server Configuration[/cyan]")

    if not LOCAL_MCP_PATH:
        # Option 1: Run directly from GitHub (recommended)
        valencia_mcp = StdioServerConfig(
            name="ValenciaSmart",
            command="uv",
            args=[
                "run",
                "https://raw.githubusercontent.com/olafmeneses/SmartCityMCP/refs/heads/master/valencia_smart_city_mcp.py"
            ],
            cache_tools=True
        )
        console.print("   [green]✓[/green] Valencia Smart City MCP configured (running from GitHub)")
        console.print("   [dim]Note: First run downloads dependencies automatically[/dim]")
    else:
        # Option 2: Run from local clone
        # For users who prefer to clone the repository manually:
        # git clone https://github.com/olafmeneses/SmartCityMCP.git
        valencia_mcp = StdioServerConfig(
            name="ValenciaSmart",
            command="uv",
            args=["run", LOCAL_MCP_PATH],
            cache_tools=True
        )
        console.print(f"   [green]✓[/green] Valencia Smart City MCP configured (running from {LOCAL_MCP_PATH})")

    # Create provider
    console.print("\n[cyan]Initializing LLM Provider...[/cyan]")
    provider = LLMProvider(model=LLM_MODEL)
    console.print(f"   [green]✓[/green] Provider ready: {LLM_MODEL}")

    # Create LLM agent with MCP
    console.print("\n[cyan]Starting agents...[/cyan]")
    llm_agent = LLMAgent(
        jid=llm_jid,
        password=llm_password,
        provider=provider,
        system_prompt="You are a helpful assistant with access to Valencia city data tools. Provide weather, traffic, bike availability, air quality, and city info.",
        mcp_servers=[valencia_mcp]
    )

    await llm_agent.start()
    console.print(f"   [green]✓[/green] LLM agent started: {llm_jid}")

    # Human agent setup
    human_jid = f"human@{xmpp_server}"
    human_password = "human_pass"

    # Rich console display callback with markdown support
    def display_response(message: str, sender: str):
        console.print("\n")
        console.print(Panel(
            Markdown(message),
            title="[bold green]Valencia Smart Assistant[/bold green]",
            border_style="green",
            box=box.ROUNDED,
        ))

    chat = ChatAgent(
        jid=human_jid,
        password=human_password,
        target_agent_jid=llm_jid,
        display_callback=display_response
    )

    await chat.start()
    console.print(f"   [green]✓[/green] Chat agent started: {human_jid}")

    console.print("\n")
    console.print(Panel(
        "[bold green]System Ready[/bold green]\n\n"
        "[yellow]Ask about:[/yellow]\n"
        "  • Valencia weather conditions\n"
        "  • Traffic information\n"
        "  • Bike station availability\n"
        "  • Air quality monitoring\n"
        "  • City data and services\n\n"
        "[dim]Type 'exit' to quit[/dim]",
        border_style="green",
        box=box.ROUNDED,
    ))

    # Run interactive chat with custom prompt
    try:
        await chat.run_interactive(
            input_prompt="> ",
            exit_command="exit",
            response_timeout=60.0,
        )
    except KeyboardInterrupt:
        console.print("\n[yellow]Interrupted by user[/yellow]")
    finally:
        # Cleanup
        console.print("\n[cyan]Cleaning up...[/cyan]")
        await chat.stop()
        console.print("   [green]✓[/green] Chat agent stopped")
        await llm_agent.stop()
        console.print("   [green]✓[/green] LLM agent stopped")
        console.print("\n[bold green]Demo completed[/bold green]\n")



if __name__ == "__main__":
    spade.run(main())