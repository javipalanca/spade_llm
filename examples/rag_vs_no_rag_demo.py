"""
RAG Comparison Demo

Demonstrates RAG by comparing LLM responses with and without retrieval.
The knowledge base includes classic literature and the announcement of Claude Haiku 4.5.

PREREQUISITES:
1. Install dependencies:
   pip install spade_llm[chroma]

2. Ollama setup:
   ollama serve
   ollama pull gpt-oss:20b
   ollama pull nomic-embed-text

3. SPADE server:
   spade run

USAGE:
   python examples/rag_retrieval_vs_no_retrieval_demo.py
"""

import asyncio
import time
import urllib.request
from pathlib import Path
from typing import Tuple, List, Optional

from spade.message import Message
from spade_llm import LLMAgent, RetrievalAgent
from spade_llm.providers import LLMProvider
from spade_llm.rag import (
    Document,
    Chroma,
    RecursiveCharacterTextSplitter,
    VectorStoreRetriever,
)
from spade_llm.tools import RetrievalTool

from rich.console import Console
from rich.panel import Panel
from rich.table import Table
from rich.progress import Progress, SpinnerColumn, TextColumn, BarColumn
from rich.markdown import Markdown
from rich.prompt import Confirm
from rich import box


# Configuration
XMPP_SERVER = "localhost"
RETRIEVAL_AGENT_JID = f"retrieval_demo@{XMPP_SERVER}"
RETRIEVAL_AGENT_PASSWORD = "retrieval_pass"
LLM_WITH_RAG_JID = f"llm_with_rag@{XMPP_SERVER}"
LLM_WITH_RAG_PASSWORD = "rag_pass"
LLM_NO_RAG_JID = f"llm_no_rag@{XMPP_SERVER}"
LLM_NO_RAG_PASSWORD = "no_rag_pass"

OLLAMA_BASE_URL = "http://localhost:11434/v1"
LLM_MODEL = "gpt-oss:20b"
EMBEDDING_MODEL = "nomic-embed-text"

KNOWLEDGE_BASE_DIR = Path(__file__).parent / ".data/gutenberg_books"
VECTOR_DB_DIR = Path(__file__).parent / ".vector_db/retrieval_vs_no_retrieval_demo"

console = Console()


def download_gutenberg_book(book_id: int, title: str, author: str) -> Optional[Document]:
    url = f"https://www.gutenberg.org/files/{book_id}/{book_id}-0.txt"
    
    try:
        console.print(f"   [blue]Downloading:[/blue] {title} by {author}...")
        with urllib.request.urlopen(url, timeout=30) as response:
            content = response.read().decode('utf-8')
        
        # Strip Project Gutenberg header and footer
        start_markers = [
            "*** START OF THE PROJECT GUTENBERG EBOOK",
            "*** START OF THIS PROJECT GUTENBERG EBOOK",
            "***START OF THE PROJECT GUTENBERG EBOOK"
        ]
        end_markers = [
            "*** END OF THE PROJECT GUTENBERG EBOOK",
            "*** END OF THIS PROJECT GUTENBERG EBOOK",
            "***END OF THE PROJECT GUTENBERG EBOOK"
        ]
        
        for marker in start_markers:
            if marker in content:
                content = content.split(marker, 1)[1]
                if '\n' in content:
                    content = content.split('\n', 1)[1]
                break
        
        for marker in end_markers:
            if marker in content:
                content = content.split(marker, 1)[0]
                break
        
        content = content.strip()
        
        console.print(f"      [green]Success[/green] - Downloaded {len(content)} characters")
        
        return Document(
            content=content,
            metadata={
                "source": "project_gutenberg",
                "title": title,
                "author": author,
                "gutenberg_id": book_id,
                "type": "book"
            }
        )
        
    except Exception as e:
        console.print(f"      [red]Failed[/red] - {title}: {e}")
        return None


def get_gutenberg_books() -> List[Document]:
    console.print("\n[blue]Downloading books from Project Gutenberg...[/blue]")
    
    books_info = [
        {"id": 84, "title": "Frankenstein", "author": "Mary Shelley"},
        {"id": 1342, "title": "Pride and Prejudice", "author": "Jane Austen"},
        {"id": 11, "title": "Alice's Adventures in Wonderland", "author": "Lewis Carroll"},
        {"id": 1661, "title": "The Adventures of Sherlock Holmes", "author": "Arthur Conan Doyle"},
        {"id": 1400, "title": "Great Expectations", "author": "Charles Dickens"},
    ]
    
    documents = []
    for book_info in books_info:
        doc = download_gutenberg_book(book_info["id"], book_info["title"], book_info["author"])
        if doc:
            documents.append(doc)
        time.sleep(0.5)
    
    console.print(f"   [green]Complete[/green] - Downloaded {len(documents)}/5 books")
    return documents


def get_anthropic_news() -> Document:
    console.print("[blue]Loading Anthropic news...[/blue]")
    
    content = """# Introducing Claude Haiku 4.5

Oct 15, 2025 ● 2 min read

Claude Haiku 4.5, our latest small model, is available today to all users.

What was recently at the frontier is now cheaper and faster. Five months ago, Claude Sonnet 4 was a state-of-the-art model. Today, Claude Haiku 4.5 gives you similar levels of coding performance but at one-third the cost and more than twice the speed.

*Chart comparing frontier models on SWE-bench Verified which measures performance on real-world coding tasks*

Claude Haiku 4.5 even surpasses Claude Sonnet 4 at certain tasks, like using computers. These advances make applications like Claude for Chrome faster and more useful than ever before.

Users who rely on AI for real-time, low-latency tasks like chat assistants, customer service agents, or pair programming will appreciate Haiku 4.5's combination of high intelligence and remarkable speed. And users of Claude Code will find that Haiku 4.5 makes the coding experience—from multiple-agent projects to rapid prototyping—markedly more responsive.

**Claude Sonnet 4.5**, released two weeks ago, remains our frontier model and the best coding model in the world. Claude Haiku 4.5 gives users a new option for when they want near-frontier performance with much greater cost-efficiency. It also opens up new ways of using our models together. For example, Sonnet 4.5 can break down a complex problem into multi-step plans, then orchestrate a team of multiple Haiku 4.5s to complete subtasks in parallel.

Claude Haiku 4.5 is available everywhere today. If you're a developer, simply use `claude-haiku-4-5` via the Claude API. Pricing is now **$1/$5 per million input and output tokens**."""
    
    doc = Document(
        content=content,
        metadata={
            "source": "anthropic_news",
            "url": "https://www.anthropic.com/news/claude-haiku-4-5",
            "title": "Introducing Claude Haiku 4.5",
            "date": "Oct 15, 2025",
            "type": "news"
        }
    )
    
    console.print("   [green]Complete[/green] - Loaded Claude Haiku 4.5 announcement")
    return doc


async def setup_knowledge_base(embedding_provider: LLMProvider) -> Tuple[VectorStoreRetriever, int, int]:
    console.print("\n[cyan]Building Knowledge Base...[/cyan]")
    
    vector_store = Chroma(
        collection_name="rag_demo",
        embedding_fn=embedding_provider.get_embeddings,
        persist_directory=str(VECTOR_DB_DIR),
    )
    
    await vector_store.initialize()
    
    doc_count = await vector_store.get_document_count()
    if doc_count > 0:
        console.print(f"   [green]Found existing database[/green] - {doc_count} chunks indexed")
        reindex = Confirm.ask("   Do you want to re-download and re-index?", default=False)
        
        if not reindex:
            retriever = VectorStoreRetriever(vector_store=vector_store)
            return retriever, 6, doc_count
        else:
            await vector_store.delete_collection()
            await vector_store.cleanup()
            await vector_store.initialize()
    
    books = get_gutenberg_books()
    news = get_anthropic_news()
    all_docs = books + [news]
    
    table = Table(title="[bold blue]Knowledge Base Contents[/bold blue]", box=box.ROUNDED)
    table.add_column("Type", style="yellow")
    table.add_column("Title", style="cyan")
    table.add_column("Source", style="green")
    
    for doc in all_docs:
        doc_type = doc.metadata.get("type", "unknown")
        title = doc.metadata.get("title", "Unknown")
        source = doc.metadata.get("source", "unknown")
        table.add_row(doc_type.capitalize(), title[:50], source)
    
    console.print("\n")
    console.print(table)
    
    console.print("\n[blue]Processing documents...[/blue]")
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200,
    )
    
    with console.status("[blue]Splitting documents into chunks...[/blue]", spinner="dots"):
        chunks = text_splitter.split_documents(all_docs)
    console.print(f"   [green]Complete[/green] - Created {len(chunks)} chunks")
    
    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        BarColumn(),
        TextColumn("({task.completed}/{task.total})"),
        console=console,
    ) as progress:
        task = progress.add_task("[cyan]Indexing chunks...", total=len(chunks))
        
        batch_size = 100
        for i in range(0, len(chunks), batch_size):
            batch = chunks[i:i + batch_size]
            await vector_store.add_documents(batch)
            progress.update(task, advance=len(batch))
    
    console.print("   [green]Complete[/green] - All chunks indexed")
    
    retriever = VectorStoreRetriever(vector_store=vector_store)
    return retriever, len(all_docs), len(chunks)


async def setup_agents(retriever: VectorStoreRetriever, llm_provider: LLMProvider) -> Tuple[RetrievalAgent, LLMAgent, LLMAgent]:
    console.print("\n[cyan]Setting up agents...[/cyan]")
    
    retrieval_agent = RetrievalAgent(
        jid=RETRIEVAL_AGENT_JID,
        password=RETRIEVAL_AGENT_PASSWORD,
        retriever=retriever,
        default_k=5,
        verify_security=False
    )
    
    await retrieval_agent.start()
    await asyncio.sleep(1)
    console.print(f"   [green]Started[/green] - Retrieval Agent: {RETRIEVAL_AGENT_JID}")
    
    retrieval_tool = RetrievalTool(
        retrieval_agent_jid=RETRIEVAL_AGENT_JID,
        default_k=5,
        timeout=30,
    )
    
    llm_with_rag = LLMAgent(
        jid=LLM_WITH_RAG_JID,
        password=LLM_WITH_RAG_PASSWORD,
        provider=llm_provider,
        tools=[retrieval_tool],
        system_prompt="""You are a helpful AI assistant with access to a knowledge base.
Always use the retrieve_documents tool to search for relevant information before answering any question.
Provide accurate, detailed responses based on the retrieved documents.""",
        verify_security=False
    )
    
    await llm_with_rag.start()
    await asyncio.sleep(1)
    console.print(f"   [green]Started[/green] - LLM with RAG: {LLM_WITH_RAG_JID}")
    
    llm_no_rag = LLMAgent(
        jid=LLM_NO_RAG_JID,
        password=LLM_NO_RAG_PASSWORD,
        provider=llm_provider,
        tools=[],
        system_prompt="""You are a helpful AI assistant. Answer questions based on your training data.
Be honest about what you know and don't know.""",
        verify_security=False
    )
    
    await llm_no_rag.start()
    await asyncio.sleep(1)
    console.print(f"   [green]Started[/green] - LLM without RAG: {LLM_NO_RAG_JID}")
    
    return retrieval_agent, llm_with_rag, llm_no_rag


async def get_response(agent: LLMAgent, question: str, timeout: int = 60) -> Tuple[str, float]:
    behaviours = list(agent.behaviours)
    if not behaviours:
        return "Error: No behaviours available", 0.0
    
    behaviour = behaviours[0]
    
    msg = Message(to=agent.jid)
    msg.body = question
    msg.set_metadata("message_type", "llm")
    
    start_time = time.time()
    await behaviour.send(msg)
    
    response = await behaviour.receive(timeout=timeout)
    elapsed = time.time() - start_time
    
    if response:
        response_text = response.body
        
        # Strip thinking tags if present
        if "<think>" in response_text:
            think_start = response_text.find("<think>")
            think_end = response_text.find("</think>") + len("</think>")
            response_text = response_text[:think_start] + response_text[think_end:]
            response_text = response_text.strip()
        
        return response_text, elapsed
    else:
        return "No response (timeout)", elapsed


async def run_comparison_demo(llm_with_rag: LLMAgent, llm_no_rag: LLMAgent, retriever: VectorStoreRetriever):
    console.print("\n")
    console.print(Panel(
        "[bold cyan]Comparison Demo: With vs Without RAG[/bold cyan]\n\n"
        "[dim]Testing the same question with both agents to compare responses[/dim]",
        box=box.DOUBLE,
        border_style="cyan",
    ))
    
    question = "What is the latest LLM released by Anthropic?"
    
    console.print("\n")
    console.print(Panel(
        f"[bold yellow]Question:[/bold yellow]\n{question}",
        box=box.ROUNDED,
        border_style="yellow"
    ))
    
    # Without RAG
    console.print("\n[bold red]Without RAG (Base Model Only)[/bold red]")
    with console.status("[blue]Generating response...[/blue]", spinner="dots"):
        response_no_rag, time_no_rag = await get_response(llm_no_rag, question)
    
    console.print(Panel(
        Markdown(response_no_rag),
        title=f"[bold red]Response (No RAG)[/bold red] [dim]({time_no_rag:.1f}s)[/dim]",
        border_style="red",
        box=box.ROUNDED,
    ))
    
    console.print("\n[bold green]With RAG (Retrieval-Augmented)[/bold green]")
    
    # Document retrieval
    console.print("\n[bold cyan]Document Retrieval[/bold cyan]")
    with console.status("[blue]Searching knowledge base...[/blue]", spinner="dots"):
        retrieved_docs = await retriever.retrieve(question, search_type="similarity")
    
    if retrieved_docs:
        console.print(Panel(
            f"[bold green]Found {len(retrieved_docs)} relevant documents[/bold green]",
            border_style="green",
            box=box.ROUNDED,
        ))
        
        docs_table = Table(box=box.ROUNDED, show_header=True, border_style="cyan")
        docs_table.add_column("Rank", justify="center", style="cyan", width=6)
        docs_table.add_column("Source", style="yellow", width=25)
        docs_table.add_column("Content preview", style="white", width=60)
        
        for idx, doc in enumerate(retrieved_docs, 1):
            source = doc.metadata.get('source', 'Unknown')
            if source.startswith('book_'):
                content_preview = doc.content[:150]
                if '\n\n' in content_preview:
                    book_title = content_preview.split('\n\n')[0]
                else:
                    book_title = source.replace('_', ' ').title()
                source_display = f"{book_title}"
            elif 'anthropic' in source.lower():
                source_display = "Anthropic"
            else:
                source_display = source.replace('_', ' ').title()
            
            preview = doc.content.replace('\n', ' ')[:150]
            if len(doc.content) > 150:
                preview += "..."
            
            docs_table.add_row(
                f"#{idx}",
                source_display,
                preview
            )
        
        console.print(docs_table)
        console.print()
    else:
        console.print("[yellow]No documents retrieved[/yellow]")

    # Generate response with RAG
    with console.status("[blue]Generating response from retrieved context...[/blue]", spinner="dots"):
        response_with_rag, time_with_rag = await get_response(llm_with_rag, question)
    
    console.print(Panel(
        Markdown(response_with_rag),
        title=f"[bold green]Response (With RAG)[/bold green] [dim]({time_with_rag:.1f}s)[/dim]",
        border_style="green",
        box=box.ROUNDED,
    ))


async def main():
    console.print("\n")
    console.print(Panel(
        "[bold cyan]RAG Comparison Demo[/bold cyan]\n"
        "[bold]With vs Without Retrieval[/bold]",
        box=box.DOUBLE,
        border_style="cyan"
    ))
    
    console.print("[cyan]Initializing LLM providers...[/cyan]")
    
    embedding_provider = LLMProvider.create_ollama(
        model=EMBEDDING_MODEL,
        base_url=OLLAMA_BASE_URL,
    )
    console.print(f"   [green]Ready[/green] - Embedding: {EMBEDDING_MODEL}")
    
    llm_provider = LLMProvider.create_ollama(
        model=LLM_MODEL,
        base_url=OLLAMA_BASE_URL,
        temperature=0.7,
    )
    console.print(f"   [green]Ready[/green] - Chat: {LLM_MODEL}")
    
    retrieval_agent = None
    llm_with_rag = None
    llm_no_rag = None
    
    try:
        retriever, doc_count, chunk_count = await setup_knowledge_base(embedding_provider)
        
        retrieval_agent, llm_with_rag, llm_no_rag = await setup_agents(retriever, llm_provider)
        await asyncio.sleep(1)

        console.print("\n")
        console.print(Panel(
            f"[bold green]System Ready[/bold green]\n\n"
            f"Documents: [yellow]{doc_count}[/yellow] (5 books + 1 news article)\n"
            f"Chunks: [yellow]{chunk_count}[/yellow]\n"
            f"Agents: [yellow]3[/yellow] (1 retrieval + 2 chat)",
            border_style="green",
            box=box.ROUNDED,
        ))
        
        await run_comparison_demo(llm_with_rag, llm_no_rag, retriever)
        
    except KeyboardInterrupt:
        console.print("\n[yellow]Interrupted by user[/yellow]")
    except Exception as e:
        console.print(f"\n[red]Error: {e}[/red]")
        import traceback
        traceback.print_exc()
    finally:
        console.print("\n[cyan]Cleaning up...[/cyan]")
        
        if llm_with_rag:
            await llm_with_rag.stop()
            console.print("   [green]Stopped[/green] - LLM with RAG")
        if llm_no_rag:
            await llm_no_rag.stop()
            console.print("   [green]Stopped[/green] - LLM without RAG")
        if retrieval_agent:
            await retrieval_agent.stop()
            console.print("   [green]Stopped[/green] - Retrieval agent")
        
        console.print("\n[bold green]Demo completed[/bold green]\n")


if __name__ == "__main__":
    asyncio.run(main())
