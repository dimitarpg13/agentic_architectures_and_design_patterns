# Multi-Agent Agentic Workflow with MCP Protocol — Architecture & Design Documentation

This document provides a detailed architectural analysis of the
`agentic_workflow_tool_use.ipynb` notebook, which implements a **multi-agent
system** communicating through the **Model Context Protocol (MCP)**. It covers
the four core MCP capabilities — **tool calling**, **dynamic tool discovery**,
**contextual data sharing with persistence**, and **agent registration** — with
UML class diagrams, sequence diagrams, and workflow flowcharts rendered in
Mermaid.

---

## Table of Contents

1. [System Overview](#1-system-overview)
2. [Static Architecture — UML Class Diagrams](#2-static-architecture--uml-class-diagrams)
   - 2.1 [MCP Protocol Layer](#21-mcp-protocol-layer)
   - 2.2 [Agent Hierarchy](#22-agent-hierarchy)
   - 2.3 [Full System Class Diagram](#23-full-system-class-diagram)
3. [Dynamic Behaviour — UML Sequence Diagrams](#3-dynamic-behaviour--uml-sequence-diagrams)
   - 3.1 [Agent Registration Sequence](#31-agent-registration-sequence)
   - 3.2 [Dynamic Tool Discovery Sequence](#32-dynamic-tool-discovery-sequence)
   - 3.3 [Tool Calling Sequence](#33-tool-calling-sequence)
   - 3.4 [Contextual Data Sharing Sequence](#34-contextual-data-sharing-sequence)
   - 3.5 [End-to-End Workflow Sequence](#35-end-to-end-workflow-sequence)
4. [Workflow Flowcharts](#4-workflow-flowcharts)
   - 4.1 [LangGraph Workflow DAG](#41-langgraph-workflow-dag)
   - 4.2 [MCP Server Request Routing](#42-mcp-server-request-routing)
   - 4.3 [Orchestrator Planning Flowchart](#43-orchestrator-planning-flowchart)
   - 4.4 [Research Agent Flowchart](#44-research-agent-flowchart)
   - 4.5 [Analysis Agent Flowchart](#45-analysis-agent-flowchart)
   - 4.6 [Writer Agent Flowchart](#46-writer-agent-flowchart)
   - 4.7 [Dynamic Runtime Tool Registration Flowchart](#47-dynamic-runtime-tool-registration-flowchart)
   - 4.8 [Context Persistence Lifecycle](#48-context-persistence-lifecycle)
5. [Component Descriptions](#5-component-descriptions)
6. [Design Patterns & Key Decisions](#6-design-patterns--key-decisions)
7. [MCP Capability Mapping](#7-mcp-capability-mapping)

---

## 1. System Overview

The system consists of four **LLM-powered agents** that collaborate to answer
complex user queries. All inter-agent data exchange, tool invocation, and
agent discovery is routed through a lightweight, in-process **MCP server**.

| Layer | Components |
|---|---|
| **LLM** | OpenAI (`gpt-4o-mini`) or Anthropic (`claude-sonnet-4-20250514`), configurable at startup |
| **Agents** | `OrchestratorAgent`, `ResearchAgent`, `AnalysisAgent`, `WriterAgent` |
| **MCP Server** | `MCPToolRegistry`, `MCPAgentRegistry`, `MCPContextStore`, request router |
| **Tools** | `web_search`, `calculator`, `text_summariser`, `format_report`, (+ runtime-registered `sentiment_analyser`) |
| **Orchestration** | LangGraph `StateGraph` wiring agents into a linear DAG |

```mermaid
graph TD
    subgraph Orchestration
        LG[LangGraph StateGraph]
    end

    subgraph Agents
        O[Orchestrator]
        R[Researcher]
        A[Analyst]
        W[Writer]
    end

    subgraph MCP Server
        TR[Tool Registry]
        AR[Agent Registry]
        CS[Context Store]
        RR[Request Router]
    end

    subgraph External
        LLM[LLM Provider]
        DISK[JSON Persistence File]
    end

    LG --> O
    LG --> R
    LG --> A
    LG --> W

    O --> RR
    R --> RR
    A --> RR
    W --> RR

    RR --> TR
    RR --> AR
    RR --> CS

    CS --> DISK

    O -.-> LLM
    R -.-> LLM
    A -.-> LLM
    W -.-> LLM
```

---

## 2. Static Architecture — UML Class Diagrams

### 2.1 MCP Protocol Layer

The protocol layer uses a **JSON-RPC 2.0 inspired** message envelope
(`MCPMessage` / `MCPResponse`) and three registries managed by a central
`MCPServer` facade.

```mermaid
classDiagram
    class MCPMessage {
        +str method
        +Dict~str,Any~ params
        +str id
        +str source
        +str timestamp
    }

    class MCPResponse {
        +str id
        +Any result
        +Optional~str~ error
        +str timestamp
        +ok() bool
    }

    class ToolDefinition {
        +str name
        +str description
        +Dict~str,Any~ parameters
        +Callable handler
        +List~str~ tags
        +schema_dict() Dict
    }

    class MCPToolRegistry {
        -Dict~str,ToolDefinition~ _tools
        +register(ToolDefinition) None
        +unregister(str) bool
        +list_tools(tag_filter) List~Dict~
        +call(name, arguments) Any
    }

    class AgentDescriptor {
        +str agent_id
        +str name
        +str description
        +List~str~ capabilities
        +str registered_at
        +str status
    }

    class MCPAgentRegistry {
        -Dict~str,AgentDescriptor~ _agents
        +register(AgentDescriptor) None
        +deregister(str) bool
        +list_agents() List~Dict~
        +get(agent_id) AgentDescriptor
        +find_by_capability(str) List~AgentDescriptor~
    }

    class MCPContextStore {
        -Dict~str,Any~ _store
        -List~Dict~ _history
        -Optional~str~ _persist_path
        +set(key, value, source) None
        +get(key, default) Any
        +keys() List~str~
        +get_history(key) List~Dict~
        +snapshot() Dict
        -_flush() None
    }

    class MCPServer {
        +MCPToolRegistry tool_registry
        +MCPAgentRegistry agent_registry
        +MCPContextStore context_store
        -List~Dict~ _request_log
        +handle(MCPMessage) MCPResponse
        -_dispatch(MCPMessage) Any
        +get_request_log() List~Dict~
    }

    MCPServer *-- MCPToolRegistry : owns
    MCPServer *-- MCPAgentRegistry : owns
    MCPServer *-- MCPContextStore : owns
    MCPToolRegistry o-- ToolDefinition : stores
    MCPAgentRegistry o-- AgentDescriptor : stores
    MCPServer ..> MCPMessage : receives
    MCPServer ..> MCPResponse : returns
```

### 2.2 Agent Hierarchy

All agents extend `MCPAgent`, which provides the LLM integration and MCP
helper methods. Each specialised agent adds a domain-specific `run()` or
`plan()` method.

```mermaid
classDiagram
    class MCPAgent {
        +str agent_id
        +str name
        +MCPServer server
        +str system_prompt
        +LLM llm
        -_send(method, params) MCPResponse
        -_register(description, capabilities) None
        +discover_tools(tag) List~Dict~
        +call_tool(name, arguments) Any
        +set_context(key, value) None
        +get_context(key, default) Any
        +list_context_keys() List~str~
        +list_agents() List~Dict~
        +find_agents(capability) List~Dict~
        +invoke_llm(prompt) str
    }

    class OrchestratorAgent {
        +plan(query) Dict~str,str~
    }

    class ResearchAgent {
        +run(task) str
    }

    class AnalysisAgent {
        +run(task) str
    }

    class WriterAgent {
        +run(task) str
    }

    OrchestratorAgent --|> MCPAgent
    ResearchAgent --|> MCPAgent
    AnalysisAgent --|> MCPAgent
    WriterAgent --|> MCPAgent

    MCPAgent --> MCPServer : communicates via
    MCPAgent --> LLM : invokes
```

### 2.3 Full System Class Diagram

Combines all classes and relationships into a single view.

```mermaid
classDiagram
    class MCPMessage {
        +str method
        +Dict params
        +str id
        +str source
        +str timestamp
    }

    class MCPResponse {
        +str id
        +Any result
        +str error
        +ok() bool
    }

    class ToolDefinition {
        +str name
        +str description
        +Dict parameters
        +Callable handler
        +List~str~ tags
        +schema_dict() Dict
    }

    class MCPToolRegistry {
        +register(ToolDefinition)
        +unregister(name) bool
        +list_tools(tag) List
        +call(name, args) Any
    }

    class AgentDescriptor {
        +str agent_id
        +str name
        +str description
        +List~str~ capabilities
        +str status
    }

    class MCPAgentRegistry {
        +register(AgentDescriptor)
        +deregister(agent_id) bool
        +list_agents() List
        +get(agent_id) AgentDescriptor
        +find_by_capability(str) List
    }

    class MCPContextStore {
        +set(key, value, source)
        +get(key, default) Any
        +keys() List~str~
        +get_history(key) List
        +snapshot() Dict
    }

    class MCPServer {
        +MCPToolRegistry tool_registry
        +MCPAgentRegistry agent_registry
        +MCPContextStore context_store
        +handle(MCPMessage) MCPResponse
        +get_request_log() List
    }

    class MCPAgent {
        +str agent_id
        +str name
        +LLM llm
        +discover_tools(tag) List
        +call_tool(name, args) Any
        +set_context(key, value)
        +get_context(key) Any
        +list_context_keys() List
        +list_agents() List
        +find_agents(capability) List
        +invoke_llm(prompt) str
    }

    class OrchestratorAgent {
        +plan(query) Dict
    }

    class ResearchAgent {
        +run(task) str
    }

    class AnalysisAgent {
        +run(task) str
    }

    class WriterAgent {
        +run(task) str
    }

    class WorkflowState {
        +str query
        +Dict plan
        +str research
        +str analysis
        +str report
        +List~str~ messages
    }

    MCPServer *-- MCPToolRegistry
    MCPServer *-- MCPAgentRegistry
    MCPServer *-- MCPContextStore
    MCPToolRegistry o-- ToolDefinition
    MCPAgentRegistry o-- AgentDescriptor

    MCPAgent --> MCPServer : sends MCPMessage
    OrchestratorAgent --|> MCPAgent
    ResearchAgent --|> MCPAgent
    AnalysisAgent --|> MCPAgent
    WriterAgent --|> MCPAgent

    MCPServer ..> MCPMessage : receives
    MCPServer ..> MCPResponse : returns
```

---

## 3. Dynamic Behaviour — UML Sequence Diagrams

### 3.1 Agent Registration Sequence

When `build_workflow()` is called, each agent constructor triggers a
`agents/register` MCP message. This happens before any user query is processed.

```mermaid
sequenceDiagram
    participant BW as build_workflow()
    participant O as OrchestratorAgent.__init__
    participant R as ResearchAgent.__init__
    participant A as AnalysisAgent.__init__
    participant W as WriterAgent.__init__
    participant MCP as MCPServer
    participant AR as MCPAgentRegistry

    BW->>O: create
    O->>MCP: MCPMessage(agents/register, orchestrator, planning+coordination)
    MCP->>AR: register(AgentDescriptor)
    AR-->>MCP: ok
    MCP-->>O: MCPResponse(registered: orchestrator)

    BW->>R: create
    R->>MCP: MCPMessage(agents/register, researcher, research+search)
    MCP->>AR: register(AgentDescriptor)
    AR-->>MCP: ok
    MCP-->>R: MCPResponse(registered: researcher)

    BW->>A: create
    A->>MCP: MCPMessage(agents/register, analyst, analysis+math+summarisation)
    MCP->>AR: register(AgentDescriptor)
    AR-->>MCP: ok
    MCP-->>A: MCPResponse(registered: analyst)

    BW->>W: create
    W->>MCP: MCPMessage(agents/register, writer, writing+formatting)
    MCP->>AR: register(AgentDescriptor)
    AR-->>MCP: ok
    MCP-->>W: MCPResponse(registered: writer)
```

### 3.2 Dynamic Tool Discovery Sequence

Before performing work, each agent issues a `tools/list` request with an
optional tag filter to discover relevant tools at runtime.

```mermaid
sequenceDiagram
    participant Agent as Any MCPAgent
    participant MCP as MCPServer
    participant TR as MCPToolRegistry

    Agent->>MCP: MCPMessage(tools/list, tag=search)
    MCP->>TR: list_tools(tag_filter=search)
    TR-->>MCP: filtered tool schemas
    MCP-->>Agent: MCPResponse(result=[web_search schema])

    Note over Agent: Agent now knows which tools match the tag

    Agent->>MCP: MCPMessage(tools/list)
    MCP->>TR: list_tools(tag_filter=None)
    TR-->>MCP: all tool schemas
    MCP-->>Agent: MCPResponse(result=[web_search, calculator, text_summariser, format_report])
```

### 3.3 Tool Calling Sequence

Agents call tools by sending a `tools/call` MCP message with the tool name
and arguments. The MCP server delegates to the registered handler.

```mermaid
sequenceDiagram
    participant R as ResearchAgent
    participant MCP as MCPServer
    participant TR as MCPToolRegistry
    participant WS as web_search()

    R->>MCP: MCPMessage(tools/call, name=web_search, arguments={query, max_results})
    MCP->>TR: call(web_search, {query, max_results})
    TR->>WS: web_search(query=..., max_results=4)
    WS-->>TR: JSON search results
    TR-->>MCP: results
    MCP-->>R: MCPResponse(result=JSON search results)

    Note over R: Agent feeds results into LLM for synthesis
```

### 3.4 Contextual Data Sharing Sequence

Agents write data to and read data from the shared `MCPContextStore` via
`context/set` and `context/get` messages. The store flushes to disk on
every write.

```mermaid
sequenceDiagram
    participant R as ResearchAgent
    participant A as AnalysisAgent
    participant MCP as MCPServer
    participant CS as MCPContextStore
    participant DISK as JSON File

    Note over R,DISK: ResearchAgent writes results

    R->>MCP: MCPMessage(context/set, key=research_results, value=...)
    MCP->>CS: set(research_results, value, source=researcher)
    CS->>DISK: flush to _mcp_context_store.json
    CS-->>MCP: ok
    MCP-->>R: MCPResponse(stored: research_results)

    Note over A,DISK: AnalysisAgent reads them

    A->>MCP: MCPMessage(context/get, key=research_results)
    MCP->>CS: get(research_results)
    CS-->>MCP: stored value
    MCP-->>A: MCPResponse(result=research data)

    Note over A: Agent uses data for analysis
```

### 3.5 End-to-End Workflow Sequence

The complete workflow from user query through all four agents to final report.

```mermaid
sequenceDiagram
    participant U as User
    participant LG as LangGraph
    participant O as Orchestrator
    participant MCP as MCP Server
    participant R as Researcher
    participant A as Analyst
    participant W as Writer
    participant LLM as LLM Provider

    U->>LG: invoke(query)

    rect rgb(240, 248, 255)
        Note over LG,O: Orchestrate Node
        LG->>O: plan(query)
        O->>MCP: tools/list()
        MCP-->>O: all tool schemas
        O->>MCP: agents/list()
        MCP-->>O: registered agents
        O->>LLM: decompose query into sub-tasks
        LLM-->>O: JSON plan
        O->>MCP: context/set(plan)
        O-->>LG: plan dict
    end

    rect rgb(240, 255, 240)
        Note over LG,R: Research Node
        LG->>R: run(research_task)
        R->>MCP: tools/list(tag=search)
        MCP-->>R: search tools
        R->>MCP: tools/call(web_search, query)
        MCP-->>R: search results
        R->>LLM: synthesise findings
        LLM-->>R: synthesis text
        R->>MCP: context/set(research_results)
        R-->>LG: research string
    end

    rect rgb(255, 248, 240)
        Note over LG,A: Analyse Node
        LG->>A: run(analysis_task)
        A->>MCP: tools/list(tag=analysis)
        MCP-->>A: analysis tools
        A->>MCP: context/get(research_results)
        MCP-->>A: research data
        A->>MCP: tools/call(calculator, expression)
        MCP-->>A: calc result
        A->>MCP: tools/call(text_summariser, text)
        MCP-->>A: summary
        A->>LLM: structured analysis
        LLM-->>A: analysis text
        A->>MCP: context/set(analysis_results)
        A-->>LG: analysis string
    end

    rect rgb(255, 240, 255)
        Note over LG,W: Write Node
        LG->>W: run(writing_task)
        W->>MCP: tools/list(tag=writing)
        MCP-->>W: writing tools
        W->>MCP: context/keys()
        MCP-->>W: all context keys
        W->>MCP: context/get(research_results)
        W->>MCP: context/get(analysis_results)
        W->>MCP: context/get(plan)
        W->>LLM: draft report
        LLM-->>W: report text
        W->>MCP: tools/call(format_report, title, sections)
        MCP-->>W: formatted markdown
        W->>MCP: context/set(final_report)
        W-->>LG: report string
    end

    LG-->>U: final result with report
```

---

## 4. Workflow Flowcharts

### 4.1 LangGraph Workflow DAG

The four agents are wired into a strictly sequential DAG via LangGraph's
`StateGraph`.

```mermaid
graph LR
    S((START)) --> O[orchestrate]
    O --> R[research]
    R --> A[analyse]
    A --> W[write]
    W --> E((END))

    style S fill:#4CAF50,color:#fff,stroke:none
    style E fill:#f44336,color:#fff,stroke:none
    style O fill:#2196F3,color:#fff
    style R fill:#FF9800,color:#fff
    style A fill:#9C27B0,color:#fff
    style W fill:#00BCD4,color:#fff
```

Each node receives the shared `WorkflowState` and returns a partial update:

| Node | Reads from state | Writes to state |
|---|---|---|
| `orchestrate` | `query` | `plan`, `messages` |
| `research` | `plan.research_task` | `research`, `messages` |
| `analyse` | `plan.analysis_task` | `analysis`, `messages` |
| `write` | `plan.writing_task` | `report`, `messages` |

### 4.2 MCP Server Request Routing

The `MCPServer._dispatch()` method routes incoming messages to the correct
subsystem based on the `method` prefix.

```mermaid
flowchart TD
    IN[Incoming MCPMessage] --> LOG[Append to request log]
    LOG --> PARSE{Parse method prefix}

    PARSE -->|tools/register| TR_REG[ToolRegistry.register]
    PARSE -->|tools/list| TR_LIST[ToolRegistry.list_tools]
    PARSE -->|tools/call| TR_CALL[ToolRegistry.call]

    PARSE -->|agents/register| AR_REG[AgentRegistry.register]
    PARSE -->|agents/list| AR_LIST[AgentRegistry.list_agents]
    PARSE -->|agents/find| AR_FIND[AgentRegistry.find_by_capability]

    PARSE -->|context/set| CS_SET[ContextStore.set]
    PARSE -->|context/get| CS_GET[ContextStore.get]
    PARSE -->|context/keys| CS_KEYS[ContextStore.keys]
    PARSE -->|context/history| CS_HIST[ContextStore.get_history]

    PARSE -->|unknown| ERR[Raise ValueError]

    TR_REG --> OK[MCPResponse result]
    TR_LIST --> OK
    TR_CALL --> OK
    AR_REG --> OK
    AR_LIST --> OK
    AR_FIND --> OK
    CS_SET --> OK
    CS_GET --> OK
    CS_KEYS --> OK
    CS_HIST --> OK
    ERR --> FAIL[MCPResponse error]
```

### 4.3 Orchestrator Planning Flowchart

```mermaid
flowchart TD
    START([plan called with query]) --> DISC_TOOLS[Discover all tools via tools/list]
    DISC_TOOLS --> DISC_AGENTS[List registered agents via agents/list]
    DISC_AGENTS --> BUILD[Build LLM prompt with query + tools + agents]
    BUILD --> LLM[Invoke LLM for plan decomposition]
    LLM --> EXTRACT[Extract JSON plan from response]
    EXTRACT --> FENCE{Starts with code fence?}
    FENCE -->|Yes| STRIP[Strip markdown fences]
    FENCE -->|No| PARSE[Parse JSON directly]
    STRIP --> PARSE
    PARSE --> STORE[Store plan in context via context/set]
    STORE --> RETURN([Return plan dict])
```

### 4.4 Research Agent Flowchart

```mermaid
flowchart TD
    START([run called with task]) --> DISC[Discover search tools via tools/list tag=search]
    DISC --> SEARCH[Call web_search tool via tools/call]
    SEARCH --> RESULTS[Receive search results JSON]
    RESULTS --> PROMPT[Build LLM prompt with task + results]
    PROMPT --> LLM[Invoke LLM to synthesise findings]
    LLM --> STORE[Store synthesis in context as research_results]
    STORE --> RETURN([Return synthesis text])
```

### 4.5 Analysis Agent Flowchart

```mermaid
flowchart TD
    START([run called with task]) --> DISC[Discover analysis tools via tools/list tag=analysis]
    DISC --> CTX[Retrieve research_results from context/get]
    CTX --> CALC[Call calculator tool via tools/call]
    CALC --> SUMM[Call text_summariser tool via tools/call]
    SUMM --> PROMPT[Build LLM prompt with task + research + tool outputs]
    PROMPT --> LLM[Invoke LLM for structured analysis]
    LLM --> STORE[Store analysis in context as analysis_results]
    STORE --> RETURN([Return analysis text])
```

### 4.6 Writer Agent Flowchart

```mermaid
flowchart TD
    START([run called with task]) --> DISC[Discover writing tools via tools/list tag=writing]
    DISC --> KEYS[List all context keys via context/keys]
    KEYS --> GET_R[Retrieve research_results from context]
    GET_R --> GET_A[Retrieve analysis_results from context]
    GET_A --> GET_P[Retrieve plan from context]
    GET_P --> PROMPT[Build LLM prompt with all context data]
    PROMPT --> LLM[Invoke LLM to draft report]
    LLM --> FORMAT[Call format_report tool via tools/call]
    FORMAT --> STORE[Store formatted report in context as final_report]
    STORE --> RETURN([Return formatted report])
```

### 4.7 Dynamic Runtime Tool Registration Flowchart

After the main workflow finishes, Section 12 of the notebook demonstrates
adding a tool at runtime and using it immediately.

```mermaid
flowchart TD
    START([Runtime admin defines sentiment_analyser function]) --> REG[Send tools/register MCP message]
    REG --> ADDED[Tool added to MCPToolRegistry]
    ADDED --> LIST[Any agent calls tools/list]
    LIST --> SEE[sentiment_analyser appears in results]
    SEE --> CALL[Send tools/call with final report text]
    CALL --> HANDLER[sentiment_analyser handler executes]
    HANDLER --> RESULT([Sentiment classification returned])
```

### 4.8 Context Persistence Lifecycle

```mermaid
flowchart TD
    subgraph Session 1
        A1[Agent writes via context/set] --> MEM1[In-memory store updated]
        MEM1 --> FLUSH[_flush writes to JSON file]
    end

    FLUSH --> DISK[(JSON persistence file on disk)]

    subgraph Session 2
        DISK --> INIT[New MCPServer reads file on init]
        INIT --> MEM2[In-memory store restored]
        MEM2 --> A2[Agent reads via context/get]
    end
```

---

## 5. Component Descriptions

### MCPMessage / MCPResponse

The communication primitive. Modelled after **JSON-RPC 2.0**: every request
carries a `method` string (e.g. `tools/call`), a `params` dict, a unique
`id`, and the `source` agent that sent it. Responses echo the `id` and carry
either a `result` or an `error`.

### MCPToolRegistry

A **dynamic service catalogue** for tools. Tools are stored as
`ToolDefinition` objects that bundle a name, description, JSON-Schema style
parameter spec, a callable handler, and a set of tags for filtering. Agents
can:
- **Register** new tools at any time (`tools/register`).
- **Discover** tools, optionally filtered by tag (`tools/list`).
- **Call** tools by name with arguments (`tools/call`).

### MCPAgentRegistry

Maintains an inventory of all **live agents** and their self-declared
capabilities. Supports:
- **Registration** with full metadata (`agents/register`).
- **Listing** all agents (`agents/list`).
- **Capability search** to find agents with a specific skill (`agents/find`).

### MCPContextStore

A **key-value store** shared across agents with built-in audit history. Each
`set()` appends a timestamped history entry. The store optionally persists to
a JSON file on every write, enabling **cross-session state recovery**.

### MCPServer

The **routing facade**. It logs every request, dispatches to the correct
subsystem based on the method prefix (`tools/`, `agents/`, `context/`), and
wraps results in `MCPResponse` envelopes. All error handling is centralised
here.

### MCPAgent (Base)

Wraps an LLM instance and provides convenience methods for every MCP
operation: tool discovery, tool calling, context read/write, agent listing.
The constructor **automatically registers** the agent with the MCP server.

### OrchestratorAgent

Uses the LLM to decompose a user query into three sub-tasks. Before planning,
it discovers available tools and registered agents to inform the LLM. The plan
is stored in the shared context for downstream agents to reference.

### ResearchAgent

Discovers search-tagged tools, calls `web_search` through MCP, and feeds the
results to the LLM for synthesis. Stores findings in context under
`research_results`.

### AnalysisAgent

Reads the researcher's findings from context, calls `calculator` and
`text_summariser` for quantitative and textual analysis, and produces a
structured analysis stored as `analysis_results`.

### WriterAgent

Reads all accumulated context (plan, research, analysis), uses the LLM to
draft a report, formats it via the `format_report` tool, and stores the final
output as `final_report`.

### WorkflowState

A `TypedDict` flowing through LangGraph nodes. Fields are: `query`, `plan`,
`research`, `analysis`, `report`, and an append-only `messages` list for
tracing.

---

## 6. Design Patterns & Key Decisions

| Pattern | Application |
|---|---|
| **Facade** | `MCPServer` provides a single entry point for three subsystems |
| **Service Locator** | `MCPToolRegistry` and `MCPAgentRegistry` let agents discover services at runtime |
| **Template Method** | `MCPAgent` defines the registration and communication skeleton; subclasses override `run()` / `plan()` |
| **Observer (audit)** | `MCPContextStore._history` and `MCPServer._request_log` record all mutations |
| **Strategy** | `_build_llm()` factory selects OpenAI or Anthropic based on configuration |
| **Pipeline** | LangGraph DAG chains agents in a linear pipeline with shared state |
| **Message Passing** | All cross-component calls use `MCPMessage` / `MCPResponse` rather than direct method calls |

### Why an in-process MCP server?

For educational clarity and zero external dependencies. The protocol boundary
(message envelopes, routing, registries) mirrors the real MCP specification
closely enough that upgrading to an out-of-process server (HTTP/SSE or
stdio transport) would only require replacing the `_send()` method on
`MCPAgent` and the transport layer on `MCPServer`, without changing any agent
logic.

### Why LangGraph for orchestration?

LangGraph provides a declarative way to define the agent pipeline as a
`StateGraph` with typed state. This makes the execution order explicit,
enables future branching or parallelism by changing edges, and provides
built-in support for checkpointing.

---

## 7. MCP Capability Mapping

| MCP Capability | Protocol Methods | Notebook Sections | Components Involved |
|---|---|---|---|
| **Tool Calling** | `tools/call` | 8 (agent `run` methods), 10 (workflow execution), 12 (runtime tool call) | `MCPAgent.call_tool()` -> `MCPServer` -> `MCPToolRegistry.call()` -> handler |
| **Dynamic Tool Discovery** | `tools/list`, `tools/register` | 8 (every agent's `run`/`plan`), 12 (runtime registration) | `MCPAgent.discover_tools()` -> `MCPServer` -> `MCPToolRegistry.list_tools()` |
| **Contextual Data Sharing & Persistence** | `context/set`, `context/get`, `context/keys`, `context/history` | 8 (agents store/retrieve results), 11 (inspect store), 14 (cross-session restore) | `MCPAgent.set_context()` / `get_context()` -> `MCPServer` -> `MCPContextStore` -> JSON file |
| **Agent Registration** | `agents/register`, `agents/list`, `agents/find` | 7-8 (auto-registration in constructor), 10 (orchestrator queries agents), 13 (capability search) | `MCPAgent._register()` -> `MCPServer` -> `MCPAgentRegistry` |
