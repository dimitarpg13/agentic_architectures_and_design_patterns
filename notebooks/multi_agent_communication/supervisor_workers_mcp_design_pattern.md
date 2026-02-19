# Multi-Agent Supervisor-Worker Design Pattern with MCP-style Messaging - Architecture Documentation

This document provides comprehensive UML diagrams and flowcharts describing the architecture and workflow of the Multi-Agent Supervisor-Worker design pattern and mock workflow using Model Context Protocol (MCP) style messaging patterns.

---

## Table of Contents

1. [Overview](#overview)
2. [Class Diagram](#class-diagram)
3. [Component Diagram](#component-diagram)
4. [Message Format Diagram](#message-format-diagram)
5. [Sequence Diagram](#sequence-diagram)
6. [MCPBus Routing Diagram](#mcpbus-routing-diagram)
7. [Supervisor Planning Flowchart](#supervisor-planning-flowchart)
8. [Worker Tool Execution Flowchart](#worker-tool-execution-flowchart)
9. [End-to-End Workflow Diagram](#end-to-end-workflow-diagram)
10. [Real MCP Architecture](#real-mcp-architecture)

---

## Overview

This system demonstrates a **Supervisor Agent** orchestrating multiple **Worker Agents** using a lightweight simulation of the **Model Context Protocol (MCP)** message pattern.

### Key Components

| Component | Role | Description |
|-----------|------|-------------|
| **MCPBus** | Message Router | In-memory transport mimicking MCP protocol |
| **Supervisor** | Orchestrator | Plans tasks, dispatches to workers, fuses results |
| **MathWorker** | Specialized Worker | Provides `add` and `avg` numeric tools |
| **TextWorker** | Specialized Worker | Provides `upper` and `keywords` text tools |

### Architecture Overview

```
+-------------------+          JSON-RPC-ish messages          +------------------+
|   Supervisor      |  ------------------------------------>  |  Worker A (Math) |
|   Agent           |  <------------------------------------  |  Tools: add, avg |
+---------+---------+                                          +------------------+
          |                                                                
          |                                  +------------------+
          |--------------------------------->|  Worker B (Text) |
          |<---------------------------------|  Tools: upper, kw|
                                             +------------------+
```

---

## Class Diagram

This diagram shows all classes, their attributes, methods, and inheritance relationships.

```mermaid
classDiagram
    class MCPRequest {
        <<dataclass>>
        +str jsonrpc = "2.0"
        +str method = "tools/call"
        +str id
        +Dict params
    }

    class MCPResponse {
        <<dataclass>>
        +str jsonrpc = "2.0"
        +Optional~str~ id
        +Any result
        +Optional~Dict~ error
    }

    class MCPBus {
        -Dict _handlers
        +register(name, handler)
        +send(req) MCPResponse
    }

    class BaseAgent {
        <<abstract>>
        +str name
        +MCPBus bus
        +handle(req)* MCPResponse
    }

    class MathWorker {
        +str name = "math"
        +handle(req) MCPResponse
        -_add(numbers) float
        -_avg(numbers) float
    }

    class TextWorker {
        +str name = "text"
        +handle(req) MCPResponse
        -_upper(text) str
        -_keywords(text, k) List~str~
    }

    class Supervisor {
        +str name = "supervisor"
        +List~str~ log
        +plan(user_goal) List~Dict~
        +call_tool(target, tool, args) Dict
        +handle_user_goal(goal) Dict
    }

    BaseAgent <|-- MathWorker : extends
    BaseAgent <|-- TextWorker : extends
    BaseAgent <|-- Supervisor : extends

    MCPBus --> MCPRequest : routes
    MCPBus --> MCPResponse : returns

    BaseAgent --> MCPBus : uses
    MathWorker ..> MCPRequest : processes
    MathWorker ..> MCPResponse : produces
    TextWorker ..> MCPRequest : processes
    TextWorker ..> MCPResponse : produces
    Supervisor ..> MCPRequest : creates
    Supervisor ..> MCPResponse : receives
```

### Description

The class diagram illustrates the inheritance hierarchy and relationships:

- **MCPRequest/MCPResponse**: Dataclasses following JSON-RPC 2.0 message format
- **MCPBus**: Central message router with handler registration
- **BaseAgent**: Abstract base class defining the agent interface
- **MathWorker**: Concrete worker with numeric computation tools
- **TextWorker**: Concrete worker with text processing tools
- **Supervisor**: Orchestrator that plans, dispatches, and aggregates results

---

## Component Diagram

This diagram shows the high-level system architecture and component interactions.

```mermaid
flowchart TB
    subgraph UserInterface["👤 User Interface"]
        UG["User Goal<br/>(Natural Language)"]
    end

    subgraph SupervisorComponent["🎯 Supervisor Agent"]
        P["Planner"]
        D["Dispatcher"]
        F["Fuser"]
        L["Logger"]
    end

    subgraph MCPTransport["📡 MCP Transport Layer"]
        BUS["MCPBus<br/>─────────────<br/>In-memory Router<br/>Handler Registry"]
    end

    subgraph Workers["👷 Worker Agents"]
        subgraph MW["🔢 MathWorker"]
            MT1["add(numbers)"]
            MT2["avg(numbers)"]
        end
        subgraph TW["📝 TextWorker"]
            TT1["upper(text)"]
            TT2["keywords(text, k)"]
        end
    end

    subgraph Output["📤 Output"]
        R["Fused Results"]
    end

    UG --> P
    P --> D
    D --> BUS
    BUS --> MW
    BUS --> TW
    MW --> BUS
    TW --> BUS
    BUS --> F
    F --> R
    D -.-> L
    F -.-> L

    style SupervisorComponent fill:#e3f2fd
    style MCPTransport fill:#fff3e0
    style Workers fill:#e8f5e9
```

### Description

The component diagram shows:

- **User Interface**: Entry point for natural language goals
- **Supervisor Agent**: Contains planning, dispatching, fusing, and logging capabilities
- **MCP Transport Layer**: The MCPBus routes messages between agents
- **Worker Agents**: Specialized tools registered with the bus
- **Output**: Aggregated results from all tool executions

---

## Message Format Diagram

This diagram details the MCP-style JSON-RPC message structure.

```mermaid
flowchart LR
    subgraph Request["📤 MCPRequest"]
        R1["jsonrpc: '2.0'"]
        R2["method: 'tools/call'"]
        R3["id: 'uuid-string'"]
        R4["params: {<br/>  target: 'math',<br/>  tool: 'add',<br/>  args: {numbers: [1,2,3]}<br/>}"]
    end

    subgraph SuccessResponse["✅ MCPResponse (Success)"]
        S1["jsonrpc: '2.0'"]
        S2["id: 'uuid-string'"]
        S3["result: {<br/>  ok: true,<br/>  tool: 'add',<br/>  value: 6<br/>}"]
        S4["error: null"]
    end

    subgraph ErrorResponse["❌ MCPResponse (Error)"]
        E1["jsonrpc: '2.0'"]
        E2["id: 'uuid-string'"]
        E3["result: null"]
        E4["error: {<br/>  code: -32601,<br/>  message: 'Unknown tool'<br/>}"]
    end

    Request -->|"Success"| SuccessResponse
    Request -->|"Failure"| ErrorResponse

    style Request fill:#e3f2fd
    style SuccessResponse fill:#e8f5e9
    style ErrorResponse fill:#ffebee
```

### Description

The message format follows JSON-RPC 2.0 conventions:

**MCPRequest Fields:**
- `jsonrpc`: Protocol version ("2.0")
- `method`: Operation type ("tools/call")
- `id`: Unique request identifier (UUID)
- `params`: Contains `target` (worker), `tool` (function), and `args` (parameters)

**MCPResponse Fields:**
- `jsonrpc`: Protocol version
- `id`: Matching request ID
- `result`: Tool execution result (on success)
- `error`: Error details with code and message (on failure)

---

## Sequence Diagram

This diagram shows the complete message flow from user goal to final result.

```mermaid
sequenceDiagram
    autonumber
    participant User
    participant Supervisor
    participant Planner as Planner (internal)
    participant Bus as MCPBus
    participant Math as MathWorker
    participant Text as TextWorker

    User->>Supervisor: handle_user_goal(goal)
    
    Note over Supervisor: Log: "User goal: ..."
    
    rect rgb(227, 242, 253)
        Note over Planner: Planning Phase
        Supervisor->>Planner: plan(goal)
        Planner->>Planner: Extract numbers
        Planner->>Planner: Extract quoted text
        Planner->>Planner: Detect intent (avg/sum/upper/keywords)
        Planner-->>Supervisor: List of tasks
        Note over Supervisor: Log: "Plan: [...]"
    end
    
    rect rgb(255, 243, 224)
        Note over Bus: Execution Phase - Math
        Supervisor->>Supervisor: call_tool("math", "avg", {numbers})
        Note over Supervisor: Log: "→ call math.avg(...)"
        Supervisor->>Bus: send(MCPRequest)
        Bus->>Bus: Lookup handler for "math"
        Bus->>Math: handler(MCPRequest)
        Math->>Math: Calculate average
        Math-->>Bus: MCPResponse(result)
        Bus-->>Supervisor: MCPResponse
        Note over Supervisor: Log: "← result math.avg: ..."
    end
    
    rect rgb(232, 245, 233)
        Note over Bus: Execution Phase - Text (upper)
        Supervisor->>Supervisor: call_tool("text", "upper", {text})
        Note over Supervisor: Log: "→ call text.upper(...)"
        Supervisor->>Bus: send(MCPRequest)
        Bus->>Bus: Lookup handler for "text"
        Bus->>Text: handler(MCPRequest)
        Text->>Text: Convert to uppercase
        Text-->>Bus: MCPResponse(result)
        Bus-->>Supervisor: MCPResponse
        Note over Supervisor: Log: "← result text.upper: ..."
    end
    
    rect rgb(243, 229, 245)
        Note over Bus: Execution Phase - Text (keywords)
        Supervisor->>Supervisor: call_tool("text", "keywords", {text, k})
        Supervisor->>Bus: send(MCPRequest)
        Bus->>Text: handler(MCPRequest)
        Text->>Text: Extract keywords
        Text-->>Bus: MCPResponse(result)
        Bus-->>Supervisor: MCPResponse
    end
    
    rect rgb(255, 248, 225)
        Note over Supervisor: Fusion Phase
        Supervisor->>Supervisor: Aggregate results by tool name
        Supervisor->>Supervisor: Build fused output
    end
    
    Supervisor-->>User: {plan, results, fused, log}
```

### Description

The sequence diagram shows the complete flow:

1. **User Input**: Natural language goal is submitted
2. **Planning Phase** (blue): Supervisor analyzes goal and creates task list
3. **Execution Phase - Math** (orange): Numeric computation dispatched to MathWorker
4. **Execution Phase - Text** (green): Text processing dispatched to TextWorker
5. **Fusion Phase** (yellow): Results aggregated into final output

Each tool call is logged with request and response details.

---

## MCPBus Routing Diagram

This diagram details how the MCPBus routes messages to registered handlers.

```mermaid
flowchart TD
    subgraph Input["📥 Incoming Request"]
        REQ["MCPRequest<br/>───────────<br/>params.target: 'math'<br/>params.tool: 'add'<br/>params.args: {...}"]
    end

    subgraph MCPBus["📡 MCPBus"]
        RECV["Receive Request"]
        EXTRACT["Extract target<br/>from params"]
        LOOKUP{"target in<br/>_handlers?"}
        DISPATCH["Dispatch to handler"]
        ERROR["Create Error Response<br/>code: -32601<br/>message: 'Unknown target'"]
        CATCH["Catch Exception"]
        ERRMSG["Create Error Response<br/>code: -32000<br/>message: str(exception)"]
    end

    subgraph Registry["🗂️ Handler Registry"]
        H1["'math' → MathWorker.handle"]
        H2["'text' → TextWorker.handle"]
    end

    subgraph Workers["👷 Workers"]
        MW["MathWorker.handle()"]
        TW["TextWorker.handle()"]
    end

    subgraph Output["📤 Output"]
        RESP["MCPResponse"]
    end

    REQ --> RECV
    RECV --> EXTRACT
    EXTRACT --> LOOKUP
    LOOKUP -->|"No"| ERROR
    LOOKUP -->|"Yes"| DISPATCH
    DISPATCH --> Registry
    Registry --> MW
    Registry --> TW
    MW --> RESP
    TW --> RESP
    DISPATCH -->|"Exception"| CATCH
    CATCH --> ERRMSG
    ERROR --> RESP
    ERRMSG --> RESP

    style MCPBus fill:#fff3e0
    style Registry fill:#e3f2fd
    style Workers fill:#e8f5e9
```

### Description

The MCPBus routing logic:

1. **Receive**: Accept incoming MCPRequest
2. **Extract**: Get `target` from `params`
3. **Lookup**: Check if target exists in handler registry
4. **Dispatch**: Route to registered handler function
5. **Error Handling**: 
   - Unknown target → Error code -32601
   - Handler exception → Error code -32000
6. **Return**: MCPResponse with result or error

---

## Supervisor Planning Flowchart

This diagram details the heuristic planning logic in the Supervisor.

```mermaid
flowchart TD
    START((Start<br/>Planning)) --> A["Receive user_goal"]
    
    A --> B["Extract numbers using regex<br/>-?\d+(?:\.\d+)?"]
    
    B --> C["Detect numeric intent"]
    
    C --> D{"'average' or 'avg'<br/>or 'mean' in goal?"}
    D -->|Yes| E["wants_avg = True"]
    D -->|No| F{"'sum' or 'total'<br/>in goal?"}
    F -->|Yes| G["wants_sum = True"]
    F -->|No| H["No numeric task"]
    
    E --> I["Check text intent"]
    G --> I
    H --> I
    
    I --> J["Extract quoted text<br/>using double or single quotes"]
    
    J --> K{"'uppercase' or 'upper'<br/>in goal?"}
    K -->|Yes| L["wants_upper = True"]
    K -->|No| M{"'keywords'<br/>in goal?"}
    
    L --> M
    M -->|Yes| N["wants_keywords = True"]
    M -->|No| O["Build task list"]
    N --> O
    
    O --> P{"numbers AND<br/>wants_avg?"}
    P -->|Yes| Q["Add task:<br/>math.avg(numbers)"]
    P -->|No| R{"numbers AND<br/>wants_sum?"}
    R -->|Yes| S["Add task:<br/>math.add(numbers)"]
    R -->|No| T["No math task"]
    
    Q --> U{"text_segments<br/>exist?"}
    S --> U
    T --> U
    
    U -->|Yes| V{"wants_upper?"}
    U -->|No| Z["Return tasks"]
    
    V -->|Yes| W["Add task:<br/>text.upper(text)"]
    V -->|No| X{"wants_keywords?"}
    
    W --> X
    X -->|Yes| Y["Add task:<br/>text.keywords(text, k=5)"]
    X -->|No| Z
    Y --> Z
    
    Z --> END((Return<br/>task list))

    style D fill:#fff3e0
    style K fill:#e3f2fd
    style P fill:#e8f5e9
    style V fill:#f3e5f5
```

### Description

The Supervisor's planning algorithm:

1. **Number Extraction**: Use regex to find all numeric values
2. **Numeric Intent Detection**: Check for keywords like "average", "sum", "total"
3. **Text Extraction**: Find text within quotes
4. **Text Intent Detection**: Check for "uppercase" or "keywords"
5. **Task Generation**: Build list of tool calls based on detected intents

This is a simple heuristic planner; a production system would use an LLM for more sophisticated planning.

---

## Worker Tool Execution Flowchart

This diagram shows how workers process tool requests.

```mermaid
flowchart TD
    subgraph MathWorkerFlow["🔢 MathWorker.handle()"]
        MA["Receive MCPRequest"]
        MB["Extract tool & args"]
        MC{"tool == 'add'?"}
        MD["sum(numbers)"]
        ME{"tool == 'avg'?"}
        MF["sum/len(numbers)"]
        MG["Return error:<br/>Unknown tool"]
        MH["Return MCPResponse<br/>{ok: true, tool, value}"]
        
        MA --> MB --> MC
        MC -->|Yes| MD --> MH
        MC -->|No| ME
        ME -->|Yes| MF --> MH
        ME -->|No| MG
    end
    
    subgraph TextWorkerFlow["📝 TextWorker.handle()"]
        TA["Receive MCPRequest"]
        TB["Extract tool & args"]
        TC{"tool == 'upper'?"}
        TD["text.upper()"]
        TE{"tool == 'keywords'?"}
        TF["Extract & count words<br/>Sort by frequency<br/>Return top k"]
        TG["Return error:<br/>Unknown tool"]
        TH["Return MCPResponse<br/>{ok: true, tool, value}"]
        
        TA --> TB --> TC
        TC -->|Yes| TD --> TH
        TC -->|No| TE
        TE -->|Yes| TF --> TH
        TE -->|No| TG
    end

    style MathWorkerFlow fill:#e3f2fd
    style TextWorkerFlow fill:#e8f5e9
```

### Description

**MathWorker Tools:**
| Tool | Input | Output | Description |
|------|-------|--------|-------------|
| `add` | `numbers: List[float]` | `float` | Sum of all numbers |
| `avg` | `numbers: List[float]` | `float` | Arithmetic mean |

**TextWorker Tools:**
| Tool | Input | Output | Description |
|------|-------|--------|-------------|
| `upper` | `text: str` | `str` | Uppercase conversion |
| `keywords` | `text: str, k: int` | `List[str]` | Top k keywords by frequency |

---

## End-to-End Workflow Diagram

This diagram shows a complete example execution.

```mermaid
flowchart TD
    subgraph UserInput["👤 User Input"]
        GOAL["Compute the average of 10, 20, 30<br/>and extract keywords and uppercase<br/>the phrase: Agents coordinate via MCP messages"]
    end

    subgraph Planning["📋 Planning"]
        PARSE["Parse goal:<br/>───────────────<br/>numbers: [10, 20, 30]<br/>text: 'Agents coordinate via MCP messages'<br/>wants_avg: true<br/>wants_keywords: true<br/>wants_upper: true"]
        
        TASKS["Generated Tasks:<br/>───────────────<br/>1. math.avg([10,20,30])<br/>2. text.upper(...)<br/>3. text.keywords(..., k=5)"]
    end

    subgraph Execution["⚡ Execution"]
        E1["→ math.avg([10,20,30])<br/>← result: 20.0"]
        E2["→ text.upper('Agents...')<br/>← result: 'AGENTS COORDINATE VIA MCP MESSAGES'"]
        E3["→ text.keywords('Agents...', k=5)<br/>← result: ['agents', 'coordinate', 'mcp', 'messages', 'via']"]
    end

    subgraph Fusion["🔗 Fusion"]
        FUSED["Fused Results:<br/>───────────────<br/>{<br/>  'avg': 20.0,<br/>  'upper': 'AGENTS COORDINATE...',<br/>  'keywords': ['agents', ...]<br/>}"]
    end

    subgraph Output["📤 Final Output"]
        RESULT["{<br/>  plan: [...],<br/>  results: [...],<br/>  fused: {...},<br/>  log: [...]<br/>}"]
    end

    GOAL --> PARSE
    PARSE --> TASKS
    TASKS --> E1
    E1 --> E2
    E2 --> E3
    E3 --> FUSED
    FUSED --> RESULT

    style Planning fill:#e3f2fd
    style Execution fill:#fff3e0
    style Fusion fill:#e8f5e9
```

### Description

This example shows the complete flow for a multi-intent goal:

1. **Input**: User provides a complex goal with numeric and text operations
2. **Planning**: Supervisor extracts data and detects multiple intents
3. **Execution**: Three sequential tool calls via MCPBus
4. **Fusion**: Results aggregated by tool name
5. **Output**: Complete response with plan, results, fused data, and logs

---

## Real MCP Architecture

This diagram shows how to transition from the simulated MCPBus to real MCP servers.

```mermaid
flowchart TB
    subgraph SimulatedMCP["🔬 Simulated (In-Notebook)"]
        SB["MCPBus<br/>(In-memory)"]
        SM["MathWorker<br/>(Python class)"]
        ST["TextWorker<br/>(Python class)"]
        SS["Supervisor<br/>(Python class)"]
        
        SS <-->|"MCPRequest/Response"| SB
        SB <--> SM
        SB <--> ST
    end

    subgraph RealMCP["🚀 Real MCP (Production)"]
        subgraph Client["MCP Client (Supervisor)"]
            RC["Session Manager"]
            RP["Task Planner"]
        end
        
        subgraph Transport["Transport Layer"]
            STDIO["stdio transport"]
            SSE["SSE transport"]
            HTTP["HTTP transport"]
        end
        
        subgraph Servers["MCP Servers (Workers)"]
            RM["math_server.py<br/>───────────<br/>@server.tool()<br/>async def add(...)<br/>async def avg(...)"]
            RT["text_server.py<br/>───────────<br/>@server.tool()<br/>async def upper(...)<br/>async def keywords(...)"]
        end
        
        RC --> STDIO
        RC --> SSE
        RC --> HTTP
        STDIO --> RM
        STDIO --> RT
        SSE --> RM
        SSE --> RT
        HTTP --> RM
        HTTP --> RT
    end

    SimulatedMCP -.->|"Swap transport"| RealMCP

    style SimulatedMCP fill:#fff3e0
    style RealMCP fill:#e8f5e9
```

### Description

**Simulated MCP (Current Implementation):**
- In-memory MCPBus for message routing
- Python class handlers for tool execution
- No network, runs entirely in notebook

**Real MCP (Production):**
- MCP SDK with async client/server
- Multiple transport options (stdio, SSE, HTTP)
- Separate server processes for each worker
- JSON-RPC over network/IPC

**Migration Path:**
1. Install MCP SDK: `pip install mcp anyio`
2. Convert workers to MCP servers with `@server.tool()` decorators
3. Use `stdio_client()` to spawn server processes
4. Replace `bus.send()` with `session.call_tool()`

---

## State Flow Diagram

This diagram shows how state changes through the supervisor workflow.

```mermaid
stateDiagram-v2
    [*] --> Idle
    
    Idle --> Planning: handle_user_goal(goal)
    
    Planning --> Executing: plan generated
    
    state Executing {
        [*] --> NextTask
        NextTask --> CallTool: task exists
        CallTool --> AwaitResponse: send to MCPBus
        AwaitResponse --> LogResult: response received
        LogResult --> NextTask: more tasks
        LogResult --> [*]: no more tasks
    }
    
    Executing --> Fusing: all tasks complete
    
    Fusing --> Complete: results merged
    
    Complete --> [*]
    
    note right of Planning
        Extract numbers
        Extract text
        Detect intents
        Build task list
    end note
    
    note right of Executing
        Sequential tool calls
        Each logged with ID
    end note
    
    note right of Fusing
        Aggregate by tool name
        Build final output dict
    end note
```

### Description

The Supervisor's state transitions:

1. **Idle**: Waiting for user goal
2. **Planning**: Analyzing goal and generating tasks
3. **Executing**: Processing each task sequentially
   - Call tool via MCPBus
   - Await response
   - Log result
   - Repeat for remaining tasks
4. **Fusing**: Combining all results
5. **Complete**: Returning final output

---

## Summary

This documentation covers the complete architecture of the Multi-Agent MCP-style Supervisor system:

| Diagram Type | Purpose |
|-------------|---------|
| Class Diagram | Data structures and inheritance hierarchy |
| Component Diagram | High-level system organization |
| Message Format | JSON-RPC request/response structure |
| Sequence Diagram | Complete message flow |
| MCPBus Routing | Handler dispatch logic |
| Planning Flowchart | Heuristic task decomposition |
| Worker Execution | Tool processing logic |
| End-to-End | Complete example walkthrough |
| Real MCP Architecture | Production migration path |
| State Flow | Supervisor state transitions |

### Key Architecture Decisions

1. **JSON-RPC 2.0 Style**: Messages follow standard protocol for interoperability
2. **Handler Registry**: Dynamic registration enables extensible worker pool
3. **Simulated Transport**: In-memory bus for easy development and testing
4. **Heuristic Planning**: Simple regex-based planning (replaceable with LLM)
5. **Sequential Execution**: Tasks processed one at a time (parallelizable in production)
6. **Result Fusion**: Simple aggregation by tool name

### MCP Protocol Key Concepts

| Concept | Description |
|---------|-------------|
| **JSON-RPC** | Standard protocol for request/response messaging |
| **Method** | Operation type (e.g., "tools/call") |
| **Params** | Request parameters including target and args |
| **Transport** | Communication channel (stdio, SSE, HTTP) |
| **Server** | Process exposing tools via MCP |
| **Client** | Process consuming tools via MCP |

### Extension Points

- Replace heuristic planner with LLM-based decomposition
- Add parallel task execution
- Implement real MCP transport (stdio/SSE/HTTP)
- Add more specialized workers (database, API, file system)
- Implement tool discovery via `tools/list` method
- Add authentication and access control
- Implement streaming responses for long-running tools
