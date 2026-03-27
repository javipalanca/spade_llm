# Memory Architecture

This guide provides detailed architectural diagrams and explanations of SPADE_LLM's memory system components and their interactions.

## System Overview

```mermaid
graph TB
    subgraph "SPADE_LLM Agent"
        A[LLMAgent] --> B[LLMBehaviour]
        B --> C[Memory Manager]
    end
    
    subgraph "Dual Memory System"
        C --> D[Interaction Memory]
        C --> E[Agent Base Memory]
    end
    
    subgraph "Storage Backends"
        D --> F[JSON Files]
        E --> G[SQLite Database]
        G --> G1[Persistent Mode]
        G --> G2[In-Memory Mode]
    end
    
    subgraph "Memory Tools"
        H[remember_interaction_info] --> D
        I[store_memory] --> E
        J[search_memories] --> E
        K[list_memories] --> E
    end
    
    subgraph "Integration Points"
        L[Context Manager] --> D
        M[LLM Provider] --> H
        M --> I
        M --> J
        M --> K
    end
    
    style A fill:#e1f5fe
    style D fill:#f3e5f5
    style E fill:#e8f5e8
    style F fill:#fff3e0
    style G fill:#e8f5e8
```

## Memory Types Architecture

### Interaction Memory Flow

```mermaid
sequenceDiagram
    participant U as User
    participant A as LLMAgent
    participant IM as Interaction Memory
    participant CM as Context Manager
    participant LLM as LLM Provider
    
    U->>A: Send message
    A->>IM: Check existing memory
    IM->>CM: Inject memory context
    CM->>LLM: Send enriched context
    LLM->>A: Response + tool calls
    A->>IM: Store new information
    IM->>U: Enhanced response
    
    Note over IM: JSON-based storage
    Note over CM: Auto-injection
```

### Agent Base Memory Flow

#### Persistent Mode
```mermaid
sequenceDiagram
    participant LLM as LLM Provider
    participant ABM as Agent Base Memory
    participant SB as SQLite Backend
    participant FS as File System
    
    LLM->>ABM: store_memory(content, category)
    ABM->>SB: Execute INSERT query
    SB->>FS: Write to database
    FS->>SB: Confirm write
    SB->>ABM: Return memory ID
    ABM->>LLM: Confirmation message
    
    LLM->>ABM: search_memories(query)
    ABM->>SB: Execute SELECT query
    SB->>FS: Read from database
    FS->>SB: Return results
    SB->>ABM: Memory entries
    ABM->>LLM: Formatted results
```

#### In-Memory Mode
```mermaid
sequenceDiagram
    participant LLM as LLM Provider
    participant ABM as Agent Base Memory
    participant SB as SQLite Backend
    participant RAM as Memory (RAM)
    
    LLM->>ABM: store_memory(content, category)
    ABM->>SB: Execute INSERT query
    SB->>RAM: Write to in-memory database
    RAM->>SB: Confirm write
    SB->>ABM: Return memory ID
    ABM->>LLM: Confirmation message
    
    LLM->>ABM: search_memories(query)
    ABM->>SB: Execute SELECT query
    SB->>RAM: Read from memory
    RAM->>SB: Return results
    SB->>ABM: Memory entries
    ABM->>LLM: Formatted results
    
    Note over RAM: Automatically deleted on agent stop
```

## Data Storage Architecture

### File System Organization

#### Persistent Storage
```mermaid
graph LR
    subgraph "Memory Root Path"
        A[/memory/path/]
    end
    
    subgraph "Agent 1 Files"
        A --> B[agent1_example_com_interactions.json]
        A --> C[agent1_example_com_base_memory.db]
    end
    
    subgraph "Agent 2 Files"
        A --> D[agent2_example_com_interactions.json]
        A --> E[agent2_example_com_base_memory.db]
    end
    
    subgraph "Shared Resources"
        A --> F[.memory_metadata.json]
        A --> G[backup/]
    end
    
    style A fill:#fff3e0
    style B fill:#f3e5f5
    style C fill:#e8f5e8
    style D fill:#f3e5f5
    style E fill:#e8f5e8
```

#### In-Memory Storage
```mermaid
graph LR
    subgraph "RAM Memory Space"
        A[In-Memory Databases]
    end
    
    subgraph "Agent 1 Memory"
        A --> B[:memory: database]
        B --> C[agent_memories table]
        C --> D[Temporary data]
    end
    
    subgraph "Agent 2 Memory"
        A --> E[:memory: database]
        E --> F[agent_memories table]
        F --> G[Temporary data]
    end
    
    subgraph "Lifecycle"
        H[Agent Start] --> I[Create :memory: DB]
        I --> J[Store/Retrieve Data]
        J --> K[Agent Stop]
        K --> L[Automatic Cleanup]
    end
    
    style A fill:#e3f2fd
    style B fill:#e8f5e8
    style E fill:#e8f5e8
    style L fill:#ffebee
```

### Database Schema Architecture

```mermaid
erDiagram
    AGENT_MEMORIES {
        string id PK
        string agent_id FK
        string category
        string content
        string context
        real confidence
        timestamp created_at
        timestamp last_accessed
        integer access_count
    }
    
    MEMORY_CATEGORIES {
        string category PK
        string description
        string usage_examples
    }
    
    MEMORY_STATS {
        string agent_id PK
        integer total_memories
        real avg_confidence
        timestamp last_update
        json category_distribution
    }
    
    AGENT_MEMORIES ||--o{ MEMORY_CATEGORIES : belongs_to
    AGENT_MEMORIES ||--o{ MEMORY_STATS : contributes_to
```

## Memory Tool Integration

### Tool Registration Flow

```mermaid
graph TB
    subgraph "Agent Initialization"
        A[LLMAgent.__init__] --> B{Memory Config?}
        B -->|interaction_memory=True| C[Register interaction tools]
        B -->|agent_base_memory=True| D[Register base memory tools]
        B -->|Both enabled| E[Register all tools]
    end
    
    subgraph "Tool Registration"
        C --> F[remember_interaction_info]
        D --> G[store_memory]
        D --> H[search_memories]
        D --> I[list_memories]
        E --> F
        E --> G
        E --> H
        E --> I
    end
    
    subgraph "Tool Execution"
        F --> J[Interaction Memory API]
        G --> K[Base Memory API]
        H --> K
        I --> K
    end
    
    style A fill:#e1f5fe
    style J fill:#f3e5f5
    style K fill:#e8f5e8
```

### Tool Call Processing

```mermaid
sequenceDiagram
    participant LLM as LLM Provider
    participant TB as Tool Bridge
    participant MT as Memory Tool
    participant MS as Memory System
    
    LLM->>TB: Tool call request
    TB->>MT: Parse and validate
    MT->>MS: Execute memory operation
    MS->>MT: Return result
    MT->>TB: Formatted response
    TB->>LLM: Tool execution result
    
    Note over MT: Handles validation
    Note over MS: Performs storage/retrieval
```

## Memory Categories System

### Category Organization

```mermaid
graph TB
    subgraph "Memory Categories"
        A[Agent Base Memory] --> B[Facts]
        A --> C[Patterns]
        A --> D[Preferences]
        A --> E[Capabilities]
    end
    
    subgraph "Fact Examples"
        B --> F["API endpoints"]
        B --> G["Configuration values"]
        B --> H["Database schemas"]
    end
    
    subgraph "Pattern Examples"
        C --> I["User behavior trends"]
        C --> J["Error patterns"]
        C --> K["Usage patterns"]
    end
    
    subgraph "Preference Examples"
        D --> L["Response formats"]
        D --> M["Communication styles"]
        D --> N["Tool preferences"]
    end
    
    subgraph "Capability Examples"
        E --> O["Skills and abilities"]
        E --> P["Limitations"]
        E --> Q["Specializations"]
    end
    
    style A fill:#e8f5e8
    style B fill:#e3f2fd
    style C fill:#f3e5f5
    style D fill:#fff3e0
    style E fill:#e8f5e8
```



## Context Integration

### Agent interaction Memory Context Injection

```mermaid
graph TB
    subgraph "Conversation Processing"
        A[New Message] --> B[Context Manager]
        B --> C{Memory Available?}
    end
    
    subgraph "Memory Injection"
        C -->|Yes| D[Retrieve Memories]
        D --> E[Format Context]
        E --> F[Inject into Prompt]
        C -->|No| G[Standard Context]
    end
    
    subgraph "Context Types"
        F --> H[System Message]
        F --> I[User Context]
        F --> J[Tool Context]
        G --> K[Basic Context]
    end
    
    subgraph "LLM Processing"
        H --> L[LLM Provider]
        I --> L
        J --> L
        K --> L
    end
    
    style A fill:#e1f5fe
    style D fill:#e8f5e8
    style L fill:#fff3e0
```

## Multi-Agent Memory Architecture

### Agent Memory Isolation

```mermaid
graph TB
    subgraph "Agent 1"
        A1[LLMAgent 1] --> B1[Memory Manager 1]
        B1 --> C1[Interaction Memory 1]
        B1 --> D1[Base Memory 1]
    end
    
    subgraph "Agent 2"
        A2[LLMAgent 2] --> B2[Memory Manager 2]
        B2 --> C2[Interaction Memory 2]
        B2 --> D2[Base Memory 2]
    end
    
    subgraph "Shared Infrastructure"
        E[Memory Backend Pool] --> F[SQLite Connections]
        E --> G[File System Access]
        E --> H[Configuration Manager]
    end
    
    subgraph "Isolation Boundaries"
        C1 -.-> I[Conversation Isolation]
        C2 -.-> I
        D1 -.-> J[Agent Data Isolation]
        D2 -.-> J
    end
    
    style I fill:#ffebee
    style J fill:#ffebee
    style E fill:#e8f5e8
```

## Next Steps

- **[Memory System Guide](memory.md)** - Complete memory system documentation
- **[API Reference](../reference/api/memory.md)** - Detailed API documentation