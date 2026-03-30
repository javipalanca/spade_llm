"""
GitHub Issues/PRs Monitor with Email Notifications

A 4-agent workflow that:
1. Analyzes GitHub issues and pull requests via GitHub MCP
2. Stores summaries in Notion via Notion MCP
3. Asks a human expert for email confirmation (human-in-the-loop)
4. Sends the summary via Gmail MCP

Setup:
  1. cp .env.example .env  (fill in LLM_MODEL, GITHUB_MCP_URL, NOTION_MCP_URL, GMAIL_MCP_URL)
  2. Start the human expert interface: python -m spade_llm.human_interface.web_server
  3. spade run             (in a separate terminal)
  4. python examples/github_issues_monitor_complex.py

Obtain MCP URLs from composio.dev after connecting your GitHub, Notion, and Gmail accounts.
"""

import asyncio
import os
import spade
import logging
from typing import Dict, Any

from spade_llm.agent import LLMAgent, ChatAgent
from spade_llm.providers import LLMProvider
from spade_llm.mcp import StreamableHttpServerConfig
from spade_llm.guardrails.base import Guardrail, GuardrailResult, GuardrailAction
from spade_llm.tools import HumanInTheLoopTool
from spade_llm.utils import load_env_vars

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# 1. AGENT PROMPTS
CHAT_PROMPT = """
You are a GitHub monitoring interface agent. Forward all user messages to the GitHub analyzer
and relay responses back to the user in a clear and helpful manner.

Focus on GitHub-related requests: issues, pull requests, repository monitoring, and analysis.
"""

GITHUB_ANALYZER_PROMPT = """You are a GitHub analysis specialist. You receive GitHub monitoring requests and provide comprehensive repository analysis.

Your workflow:
1. Identify the repository to analyze (ask user if not specified)
2. Use GitHub MCP tools to gather data:
   - List recent issues (open and closed, last 30 days)
   - List recent pull requests (all states, last 30 days)
   - Get repository details if available
3. Analyze the collected data for patterns, priorities, and insights
4. Generate a structured summary with actionable information

IMPORTANT: 
- Always specify which repository you're analyzing
- Include actual numbers and real data from the GitHub API
- Focus on actionable insights and trends
- Identify urgent items that need attention

Response format:
=== GITHUB REPOSITORY ANALYSIS ===

🏪 Repository: [owner/repo-name]
📅 Analysis Date: [current date and time]
🔍 Analysis Period: Last 30 days

📊 SUMMARY METRICS
- Issues: [X] total ([X] open, [X] closed)
- Pull Requests: [X] total ([X] open, [X] merged, [X] draft, [X] closed)
- Recent Activity Level: [High/Medium/Low]
- Last Updated: [when]

🚨 URGENT ITEMS (High Priority)
[List critical issues/PRs that need immediate attention, include #numbers and titles]

📈 RECENT TRENDS (Last 30 days)
- New Issues Created: [X]
- Issues Closed: [X] 
- PRs Merged: [X]
- Most Active Contributors: [list top 3]
- Common Labels/Categories: [list most frequent]

🔍 KEY INSIGHTS
[Notable patterns, recurring issues, areas needing attention]

💡 RECOMMENDATIONS
[Actionable suggestions based on the analysis]

<GITHUB_SUMMARY>
{
  "repository": "[owner/repo]",
  "analysis_date": "[ISO date]",
  "period_days": 30,
  "summary": {
    "total_issues": X,
    "open_issues": X,
    "closed_issues": X,
    "total_prs": X,
    "open_prs": X,
    "merged_prs": X,
    "draft_prs": X
  },
  "urgent_items": [
    {"type": "issue/pr", "number": X, "title": "...", "priority": "high", "url": "..."}
  ],
  "trends": {
    "new_issues": X,
    "closed_issues": X,
    "merged_prs": X,
    "top_contributors": ["...", "..."],
    "common_labels": ["...", "..."]
  },
  "insights": ["...", "..."],
  "recommendations": ["...", "..."]
}
</GITHUB_SUMMARY>

This analysis will now be stored in Notion and potentially sent via email."""

NOTION_MANAGER_PROMPT = """You are a Notion storage specialist. You receive GitHub analysis summaries and store them systematically.

Your workflow:
1. Receive complete GitHub analysis from GitHubAnalyzer agent
2. Use Notion MCP tools to:
   - Search for "Spade monitoring" page
   - Add a new entry with all analysis data
   - Structure the data for easy reading
3. After successful storage, prepare the summary for email notification

IMPORTANT:
- Create consistent Notion entries for easy tracking over time
- Include the full analysis text for complete context
- Verify the data was stored before forwarding

Response format:
=== NOTION STORAGE COMPLETED ===

📚 **Notion Database Updated**
✅ Entry created: "[Repository] Analysis - [Date]"
🗂️  Database: GitHub Repository Monitoring
📊 Data stored:
   - Repository: [owner/repo]
   - Analysis Date: [date]
   - Issues: [X] total ([X] open)
   - PRs: [X] total ([X] open)

🔗 Notion URL: [if available]

📧 **Forwarding to Email Manager**
The complete analysis is now ready for potential email notification.

[Include the FULL original analysis text here for the Email Manager]"""

EMAIL_MANAGER_PROMPT = """You are an email notification specialist with human-in-the-loop confirmation. You receive GitHub analysis summaries and handle email notifications with human oversight.

Your workflow:
1. Receive complete GitHub analysis from Notion Manager
2. Extract key information and prepare a concise summary for human review
3. Use ask_human_expert tool to get human confirmation and email details
4. If approved, format and send professional email via Gmail MCP
5. ALWAYS end with termination marker after completing the process

HUMAN INTERACTION PROCESS:
1. Present a concise executive summary to the human
2. Ask: "Would you like to send this GitHub analysis via email?"
3. If YES: Ask "Please provide the recipient's email address(es)"
4. If NO: Acknowledge and end with termination marker
5. If email provided: Send formatted email, confirm delivery, and end with termination marker

EMAIL FORMAT (when sending):
Subject: "GitHub Repository Analysis - [Repository Name] - [Date]"

Email Content:
---
# GitHub Repository Analysis Report

**Repository:** [owner/repo-name]  
**Analysis Date:** [date]  
**Period Analyzed:** Last 30 days

## Executive Summary
- **Issues:** [X] total ([X] open, [X] closed)
- **Pull Requests:** [X] total ([X] open, [X] merged)
- **Activity Level:** [High/Medium/Low]
- **Urgent Items:** [X] items need attention

## Key Insights
[3-4 most important insights from analysis]

## Urgent Items Requiring Attention
[List critical issues/PRs with numbers and titles]

## Recommendations
[Top 3 actionable recommendations]

## Full Analysis
[Include complete detailed analysis from GitHubAnalyzer]

---
*This report was generated automatically and stored in Notion for tracking.*

INTERACTION EXAMPLE:
When you receive analysis, use ask_human_expert with message like:
"GitHub analysis ready for [repo-name]. Key findings: [X] open issues, [Y] urgent items. 
Activity level: [level]. Would you like me to email this summary to someone?"

TERMINATION:
- After sending email successfully: "Email sent successfully to [recipient]. <EMAIL_PROCESS_COMPLETE>"
- After human declines email: "GitHub analysis completed and stored in Notion. No email sent. <EMAIL_PROCESS_COMPLETE>"

IMPORTANT:
- Always summarize key points for human decision-making
- Wait for explicit human approval before sending emails
- Include repository name and key metrics in human interaction
- Use professional email formatting
- Confirm successful email delivery
- ALWAYS end with <EMAIL_PROCESS_COMPLETE> termination marker"""


class GitHubOnlyGuardrail(Guardrail):
    """Custom guardrail that only allows GitHub-related requests."""
    
    def __init__(self, name: str = "github_only_filter", enabled: bool = True):
        super().__init__(name, enabled, "I only help with GitHub-related requests. Please ask about issues, pull requests, or repository monitoring.")
        self.github_keywords = [
            "github", "issue", "issues", "pull request", "pr", "prs", 
            "repository", "repo", "commit", "branch", "merge", "review",
            "bug", "feature", "enhancement", "milestone", "project",
            "analyze", "monitor", "check", "status", "activity"
        ]
    
    async def check(self, content: str, context: Dict[str, Any]) -> GuardrailResult:
        """Check if content is GitHub-related."""
        content_lower = content.lower()
        
        # Check if any GitHub keyword is present
        if any(keyword in content_lower for keyword in self.github_keywords):
            return GuardrailResult(
                action=GuardrailAction.PASS,
                content=content,
                reason="GitHub-related request detected"
            )
        else:
            return GuardrailResult(
                action=GuardrailAction.BLOCK,
                content=self.blocked_message,
                reason="Non-GitHub request blocked"
            )


async def main():
    """Main function of the GitHub Issues Monitor example."""
    
    load_env_vars()

    print("=== GitHub Issues/PRs Monitor ===")

    xmpp_server = os.environ.get("XMPP_SERVER", "localhost")

    model = os.environ.get("LLM_MODEL")
    if not model:
        raise SystemExit("LLM_MODEL is not set — copy .env.example to .env and configure it.")

    # MCP server URLs - configure in .env (see .env.example)
    github_mcp_url = os.environ.get("GITHUB_MCP_URL")
    notion_mcp_url = os.environ.get("NOTION_MCP_URL")
    gmail_mcp_url = os.environ.get("GMAIL_MCP_URL")
    if not all([github_mcp_url, notion_mcp_url, gmail_mcp_url]):
        raise ValueError(
            "GITHUB_MCP_URL, NOTION_MCP_URL, and GMAIL_MCP_URL must be set. "
            "See .env.example for details."
        )
    
    # 3. DECLARE THE PROVIDER
    provider = LLMProvider(
        model=model,
        temperature=0.7,
    )

    # MCP Server configurations
    github_mcp = StreamableHttpServerConfig(
        name="GitHubMCP",
        url=github_mcp_url,
        headers={
            "Content-Type": "application/json",
            "User-Agent": "SPADE_LLM/1.0"
        },
        timeout=30.0,
        sse_read_timeout=300.0,
        terminate_on_close=True,
        cache_tools=True
    )
    
    notion_mcp = StreamableHttpServerConfig(
        name="NotionMCP",
        url=notion_mcp_url,
        headers={
            "Content-Type": "application/json",
            "User-Agent": "SPADE_LLM/1.0"
        },
        timeout=30.0,
        sse_read_timeout=300.0,
        terminate_on_close=True,
        cache_tools=True
    )
    
    gmail_mcp = StreamableHttpServerConfig(
        name="GmailMCP",
        url=gmail_mcp_url,
        headers={
            "Content-Type": "application/json",
            "User-Agent": "SPADE_LLM/1.0"
        },
        timeout=30.0,
        sse_read_timeout=300.0,
        terminate_on_close=True,
        cache_tools=True
    )
    
    # 4. AGENT CONFIGURATION (no hardcoded JIDs)
    chat_jid = f"github_chat@{xmpp_server}"
    analyzer_jid = f"github_analyzer@{xmpp_server}"
    notion_jid = f"notion_manager@{xmpp_server}"
    email_jid = f"email_manager@{xmpp_server}"
    human_jid = f"human_expert@{xmpp_server}"
    
    # Simple passwords (auto-registration with SPADE server)
    chat_password = "chat_pass"
    analyzer_password = "analyzer_pass"
    notion_password = "notion_pass"
    email_password = "email_pass"
    print("✓ Using auto-registration with built-in server")
    
    # Create guardrails and tools
    input_guardrails = [GitHubOnlyGuardrail()]
    
    human_tool = HumanInTheLoopTool(
        human_expert_jid=human_jid,
        timeout=300.0,  # 5 minutes
        name="ask_human_expert",
        description="Ask human expert for email sending confirmation and recipient details"
    )
    
    def display_response(message: str, sender: str):
        print(f"\nGitHub Monitor: {message}")
        print("-" * 50)
    
    # 5. INITIALIZE AGENTS WITH LLMAgent()
    # WORKFLOW: User → Chat → Analyzer → Notion → Email → Human Expert
    
    # Chat Agent with Guardrails (Entry Point)
    chat_agent = ChatAgent(
        jid=chat_jid,
        password=chat_password,
        target_agent_jid=analyzer_jid,
        display_callback=display_response,
        verify_security=False
    )
    
    # GitHub Analyzer Agent (Data Collection & Analysis)
    analyzer_agent = LLMAgent(
        jid=analyzer_jid,
        password=analyzer_password,
        provider=provider,
        system_prompt=GITHUB_ANALYZER_PROMPT,
        input_guardrails=input_guardrails,
        mcp_servers=[github_mcp],
        reply_to=notion_jid,
        verify_security=False
    )
    
    # Notion Manager Agent (Storage & Forwarding)
    notion_agent = LLMAgent(
        jid=notion_jid,
        password=notion_password,
        provider=provider,
        system_prompt=NOTION_MANAGER_PROMPT,
        mcp_servers=[notion_mcp],
        reply_to=email_jid,
        verify_security=False
    )
    
    # Email Manager Agent (HITL & Email Sending)
    email_agent = LLMAgent(
        jid=email_jid,
        password=email_password,
        provider=provider,
        system_prompt=EMAIL_MANAGER_PROMPT,
        tools=[human_tool],
        mcp_servers=[gmail_mcp],
        termination_markers=["<EMAIL_PROCESS_COMPLETE>"],
        verify_security=False
    )
    
    try:
        print("Starting agents...")
        agents = {
            "chat": chat_agent,
            "analyzer": analyzer_agent,
            "notion": notion_agent,
            "email": email_agent,
        }

        for name, agent in agents.items():
            await agent.start()

        await asyncio.sleep(3)

        print("All agents ready.")
        print("\n" + "=" * 70)
        print("GITHUB ISSUES/PRS MONITOR")
        print("=" * 70)
        print("\nGuardrails: Only GitHub-related requests accepted")
        print("\nExample requests:")
        print("  - 'Show me recent issues in the repository'")
        print("  - 'Analyze pull requests from this week'")
        print("  - 'Review GitHub activity and send summary'")
        print("\nNote: All MCP services use HTTP streaming.")
        print("Ensure the human expert web interface is running.")
        print(f"  Human expert: http://localhost:8080  (connect as: {human_jid})")
        print("\nType 'exit' to quit.")
        print("=" * 70 + "\n")

        await chat_agent.run_interactive(
            input_prompt="GitHub> ",
            exit_command="exit",
            response_timeout=120.0,
        )
        
    except KeyboardInterrupt:
        print("\nShutting down...")

    except Exception as e:
        print(f"Error: {e}")

    finally:
        print("\nStopping agents...")
        for name, agent in agents.items():
            await agent.stop()
        print("Done.")


if __name__ == "__main__":
    spade.run(main())