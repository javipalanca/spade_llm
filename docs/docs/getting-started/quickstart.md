# Quick Start Tutorials

Learn SPADE-LLM step by step with our comprehensive tutorial series. Each tutorial builds on the previous one, taking you from basic concepts to advanced multi-agent systems.

## 📚 Tutorial Series

!!! tip "Learning Path"
    Each tutorial builds on the previous one. Start with Tutorial 1 if you're new to SPADE-LLM.

### 🚀 Tutorial 1: Your First Agent

!!! abstract "Start here if you're new to SPADE-LLM"
    Learn the basics of creating and running your first LLM-powered agent.

**What you'll learn:**

- Basic agent setup and configuration
- LLM provider integration (OpenAI, Ollama)
- Interactive chat interfaces
- Error handling and best practices

**Duration:** 15-20 minutes

[**Start Tutorial →**](first-agent.md){ .md-button .md-button--primary }

---

### 🛡️ Tutorial 2: Guardrails and Safety

!!! abstract "Add safety and content filtering to your agents"
    Implement comprehensive protection systems for your AI agents.

**What you'll learn:**

- Input and output content filtering
- Custom guardrail creation
- LLM-based safety validation
- Monitoring and logging guardrail actions

**Duration:** 20-25 minutes

[**Start Tutorial →**](guardrails-tutorial.md){ .md-button .md-button--primary }

---

### 🔧 Tutorial 3: Custom Tools

!!! abstract "Extend your agents with function calling"
    Give your agents the ability to perform actions beyond text generation.

**What you'll learn:**

- Creating custom tools and functions
- Tool parameter schemas
- Async tool execution
- External API integration
- Tool composition and chaining

**Duration:** 25-30 minutes

[**Start Tutorial →**](tools-tutorial.md){ .md-button .md-button--primary }

---

### 🏗️ Tutorial 4: Advanced Multi-Agent Systems

!!! abstract "Build production-ready multi-agent workflows"
    Create sophisticated systems with multiple specialized agents working together.

**What you'll learn:**

- Agent-to-agent communication
- MCP server integration
- Human-in-the-loop workflows
- Custom guardrails and routing
- Production deployment patterns

**Duration:** 45-60 minutes

[**Start Tutorial →**](advanced-agent.md){ .md-button .md-button--primary }

## 🎯 Choose Your Path

### 👶 **Complete Beginner**
Start with **Tutorial 1** and work through all tutorials in order.

### 🔧 **Have Basic Knowledge**
Already familiar with SPADE-LLM basics? Jump to **Tutorial 2** or **Tutorial 3**.

### 🏆 **Advanced User**
Building production systems? Go directly to **Tutorial 4** for advanced patterns.

## 🛠️ Quick Setup

Before starting any tutorial, make sure you have:

### 1. Prerequisites
- Python 3.10+ installed
- SPADE-LLM installed: `pip install spade_llm`
- **SPADE built-in server running** (recommended for beginners)

### 2. Start SPADE Server
**New in SPADE 4.0 - No external server needed!**

```bash
# Terminal 1: Start the built-in SPADE server
spade run
```

This provides everything you need - no complex XMPP setup required!

### 3. LLM Provider Access
Choose one:

**OpenAI** (easiest for beginners):
```bash
export OPENAI_API_KEY="your-api-key"
```

**Ollama** (free, local option):
```bash
ollama pull llama3.1:8b
ollama serve
```

### 4. Ready to Go!
With your SPADE server running and LLM provider configured, pick a tutorial above and start building! 🚀

## 💡 Tips for Success

- **Follow the order**: Each tutorial builds on previous concepts
- **Try the examples**: Run all code examples as you go
- **Experiment**: Modify examples to understand how they work
- **Ask questions**: Use the examples as a foundation for your own projects

## 🔗 Additional Resources

- **[Installation Guide](installation.md)** - Detailed setup instructions
- **[API Reference](../reference/index.md)** - Complete documentation
- **[Examples](../reference/examples.md)** - Working code examples
- **[Guides](../guides/index.md)** - In-depth feature explanations

---

*Ready to build intelligent multi-agent systems? Start with Tutorial 1 and begin your journey!*
