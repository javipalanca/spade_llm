"""
Valencia Smart City MCP Example

Demonstrates using Valencia Smart City MCP server with SPADE agents.
The MCP server provides real-time data about Valencia, Spain including:
- Traffic conditions and congestion
- Valenbisi bike station availability
- Air quality monitoring
- Weather forecasts

PREREQUISITES:
1. Install dependencies:
   pip install spade_llm

2. Install uv (Python package installer) if you don't have it:
   https://docs.astral.sh/uv/getting-started/installation/

3. Start SPADE built-in server in another terminal:
   spade run

4. (Optional) Configure your LLM:
   - Default: Ollama with gpt-oss:20b model
   - Modify OLLAMA_BASE_URL and LLM_MODEL variables if needed

QUICK START:
- The MCP server runs automatically from GitHub - no manual setup required!
- You can also clone the SmartCityMCP repo and run locally by changing LOCAL_MCP_PATH.

This example uses SPADE's default built-in server (localhost:5222) - no account registration needed!
"""

import spade

from spade_llm.agent import LLMAgent, ChatAgent
from spade_llm.providers import LLMProvider
from spade_llm.mcp import StdioServerConfig

from rich.console import Console
from rich.panel import Panel
from rich.markdown import Markdown
from rich import box

OLLAMA_BASE_URL = "http://localhost:11434/v1"
LLM_MODEL = "gpt-oss:20b"

# Change this path if you've cloned the repo from https://github.com/olafmeneses/SmartCityMCP)
LOCAL_MCP_PATH = None  # example -> "../SmartCityMCP/valencia_smart_city_mcp.py"

console = Console()

async def main():
    console.print("\n")
    console.print(Panel(
        "[bold cyan]🌆 Valencia Smart City Assistant[/bold cyan]\n"
        "[bold]Powered by MCP & SPADE[/bold]",
        box=box.DOUBLE,
        border_style="cyan"
    ))
    
    # XMPP server configuration - using default SPADE settings
    xmpp_server = "localhost"
    console.print("\n[cyan]Server Configuration[/cyan]")
    console.print("   [green]✓[/green] Using SPADE built-in server (localhost:5222)")
    console.print("   [green]✓[/green] No account registration needed!")
    
    # Agent credentials
    llm_jid = f"llm_agent@{xmpp_server}"
    llm_password = "llm_pass"  # Simple password (auto-registration with SPADE server)

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
    provider = LLMProvider.create_ollama(
        model=LLM_MODEL,
        base_url=OLLAMA_BASE_URL,
    )
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
    human_password = "human_pass"  # Simple password (auto-registration with SPADE server)

    # Rich console display callback with markdown support
    def display_response(message: str, sender: str):
        console.print("\n")
        console.print(Panel(
            Markdown(message),
            title="[bold green]🌆 Valencia Smart Assistant[/bold green]",
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