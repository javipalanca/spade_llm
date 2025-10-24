"""
RAG System Example with Ollama and ChromaDB.

Demonstrates the core RAG (Retrieval-Augmented Generation) components of SPADE-LLM:
- Document creation and text splitting
- Vector embeddings using Ollama
- ChromaDB vector store operations (add, search, delete)
- Semantic retrieval with filtering and scoring

PREREQUISITES:
1. Install dependencies with Chroma support:
   pip install spade_llm[chroma]

2. Ollama setup:
   ollama serve
   ollama pull nomic-embed-text

3. Configure Ollama base URL:
   - For local: http://localhost:11434
   - Update OLLAMA_BASE_URL variable below if needed.

NOTE: This example runs standalone without SPADE server - it only demonstrates RAG components.
"""

import asyncio
from spade_llm.rag import (
    Document,
    Chroma,
    RecursiveCharacterTextSplitter,
    VectorStoreRetriever,
)
from spade_llm.providers import LLMProvider
from rich.console import Console
from rich.panel import Panel
from rich.rule import Rule
from rich.table import Table
from rich.text import Text


OLLAMA_BASE_URL = "http://localhost:11434/v1"
EMBEDDING_MODEL = "nomic-embed-text"

console = Console()


def create_sample_documents() -> list[Document]:
    """Create sample documents about Ollama, embeddings, and SPADE."""
    return [
        Document(
            content="""Ollama is an open-source platform for running large language models locally.
            It enables easy download, installation, and interaction with LLMs without cloud services.
            Supports models like Llama, Mistral, CodeLlama, and embedding models like nomic-embed-text.
            
            Benefits: complete privacy, offline usage, cost-effective, easy model switching, 
            REST API integration, and custom model support.""",
            metadata={"document_id": "ollama_info", "source": "tech_docs", "category": "tools"}
        ),
        Document(
            content="""Text embeddings are numerical representations of text that capture semantic meaning
            in high-dimensional vector space. Similar texts have similar embeddings, enabling semantic
            search, clustering, and recommendations.
            
            Popular models: OpenAI embeddings, Sentence-BERT, Nomic Embed, All-MiniLM, and E5 models.
            Essential for RAG systems to find relevant context for language model queries.""",
            metadata={"document_id": "embeddings_info", "source": "tech_docs", "category": "ai"}
        ),
        Document(
            content="""SPADE (Smart Python Agent Development Environment) is a multi-agent system platform
            based on XMPP technology. It enables intelligent agents to communicate in distributed 
            environments using behaviors, templates, and various communication patterns.
            
            Features: XMPP communication, behavior-driven design, message filtering, web monitoring,
            external service integration, and LLM support via SPADE-LLM.""",
            metadata={"document_id": "spade_info", "source": "tech_docs", "category": "frameworks"}
        )
    ]


async def demonstrate_basic_rag(retriever: VectorStoreRetriever):
    """Demonstrate basic RAG retrieval operations."""
    console.print(Rule("[cyan]Basic Retrieval Examples[/cyan]", style="cyan"))
    
    queries = [
        "How does Ollama enable local LLM usage?",
        "Explain text embeddings and their applications",
        "What is SPADE and how does it work?",
    ]
    
    for query in queries:
        results = await retriever.retrieve(query, k=2, search_type="similarity")
        doc_id = results[0].metadata.get('document_id', 'unknown') if results else 'none'
        console.print(f"   [yellow]Query:[/yellow] {query}")
        console.print(f"   → [green]Best match:[/green] [blue]{doc_id}[/blue]\n")


async def demonstrate_filtered_search(retriever: VectorStoreRetriever):
    """Demonstrate metadata filtering in retrieval."""
    console.print(Rule("[cyan]Filtered Search Example[/cyan]", style="cyan"))
    
    results = await retriever.retrieve(
        "programming and development tools",
        k=3,
        search_type="similarity",
        filters={"category": "tools"}
    )
    
    console.print("   [yellow]Query:[/yellow] 'programming and development tools'")
    console.print("   [yellow]Filter:[/yellow] category='tools'")
    console.print(f"   → [green]Found {len(results)} results[/green]")
    for i, doc in enumerate(results, 1):
        doc_id = doc.metadata.get('document_id', 'unknown')
        console.print(f"      {i}. [blue]{doc_id}[/blue]")


async def demonstrate_scored_retrieval(retriever: VectorStoreRetriever):
    """Demonstrate retrieval with similarity scores."""
    console.print(Rule("[cyan]Semantic Similarity Scoring[/cyan]", style="cyan"))
    
    test_queries = [
        ("running models locally", "ollama_info"),
        ("vector representations", "embeddings_info"),
        ("agent communication", "spade_info"),
    ]
    
    for query, expected_doc in test_queries:
        results = await retriever.retrieve(query, k=1, search_type="similarity_score")
        if results:
            doc, score = results[0]
            doc_id = doc.metadata.get('document_id', 'unknown')
            match = "MATCH" if expected_doc in doc_id else "NO MATCH"
            match_style = "green" if match == "MATCH" else "red"
            console.print(
                f"   [{match_style}]{match}[/{match_style}] '[yellow]{query}[/yellow]' → "
                f"[blue]{doc_id}[/blue] (score: {score:.3f})"
            )


async def demonstrate_document_management(vector_store: Chroma):
    """Demonstrate document deletion and statistics."""
    console.print(Rule("[cyan]Document Management[/cyan]", style="cyan"))
    
    count_before = await vector_store.get_document_count()
    console.print(f"   [blue]Documents before deletion:[/blue] {count_before}")
    
    # Find documents to delete by metadata
    result = await vector_store.get(where={"document_id": "ollama_info"})
    ids_to_delete = result.get("ids", [])
    
    if ids_to_delete:
        deleted = await vector_store.delete(ids_to_delete)
        status = "DELETED" if deleted else "FAILED"
        status_style = "green" if deleted else "red"
        console.print(
            f"   [{status_style}]{status}[/{status_style}]: {len(ids_to_delete)} chunk(s) from 'ollama_info'"
        )
    else:
        console.print(f"   [red]Not found: 'ollama_info'[/red]")
    
    count_after = await vector_store.get_document_count()
    console.print(f"   [blue]Documents after deletion:[/blue] {count_after}")


async def main():
    """Run the RAG system demonstration."""
    console.print(Panel(
        "[bold cyan]SPADE-LLM RAG System Example[/bold cyan]",
        expand=False,
        border_style="cyan"
    ))
    
    llm_provider = LLMProvider.create_ollama(
        model=EMBEDDING_MODEL,
        base_url=OLLAMA_BASE_URL,
        timeout=60
    )
    
    vector_store = Chroma(
        collection_name="rag_example",
        embedding_fn=llm_provider.get_embeddings
    )
    
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=500,
        chunk_overlap=100
    )
    
    retriever = VectorStoreRetriever(vector_store=vector_store)
    
    try:
        console.print("\n[blue]Initializing RAG components...[/blue]")
        await vector_store.initialize()
        
        console.print("[blue]Creating and processing documents...[/blue]")
        documents = create_sample_documents()
        chunks = text_splitter.split_documents(documents)
        
        console.print(f"[blue]Storing {len(chunks)} chunks in vector store...[/blue]")
        await vector_store.add_documents(chunks)
        
        console.print(Panel(
            f"[green]Indexed {len(documents)} documents ({len(chunks)} chunks)[/green]\n"
            f"   [bold]Model:[/] {EMBEDDING_MODEL}\n"
            f"   [bold]Provider:[/] Ollama",
            title="[bold green]Setup Complete[/bold green]",
            expand=False,
            border_style="green"
        ))

        # Create a table for document previews
        table = Table(title="[bold blue]Document Previews[/bold blue]", title_justify="left")
        table.add_column("Document ID", style="yellow", no_wrap=True)
        table.add_column("Content Preview", style="default")
        
        for doc in documents:
            doc_id = doc.metadata.get('document_id', 'unknown')
            preview = (doc.content[:100] + "...") if len(doc.content) > 100 else doc.content
            table.add_row(doc_id, preview.replace('\n', ' '))
        console.print(table)
        
        await demonstrate_basic_rag(retriever)
        await demonstrate_filtered_search(retriever)
        await demonstrate_scored_retrieval(retriever)
        await demonstrate_document_management(vector_store)
        
        console.print("\n[bold green]Example completed successfully![/bold green]")
        
    except Exception as e:
        console.print(f"\n[bold red]Error: {e}[/bold red]")
        
        if "Cannot connect" in str(e) or "Ollama" in str(e):
            troubleshooting_text = Text()
            troubleshooting_text.append("1. Install Ollama: https://ollama.com\n", style="default")
            troubleshooting_text.append(f"2. Pull model: ollama pull {EMBEDDING_MODEL}\n", style="default")
            troubleshooting_text.append(f"3. Verify server: curl {OLLAMA_BASE_URL}/api/tags", style="default")
            
            console.print(Panel(
                troubleshooting_text,
                title="[yellow]Troubleshooting[/yellow]",
                border_style="yellow",
                padding=(1, 2)
            ))
    
    finally:
        console.print("\n[blue]Cleaning up...[/blue]")
        await vector_store.cleanup()


if __name__ == "__main__":
    asyncio.run(main())