# Multi-Agent System with Supervisor - Architecture Documentation

This document provides comprehensive UML diagrams and flowcharts describing the architecture and workflow of the Multi-Agent System with a Supervisor Agent that coordinates specialized worker agents using LangGraph.

---

## Table of Contents

1. [Overview](#overview)
2. [Class Diagram](#class-diagram)
3. [Component Diagram](#component-diagram)
4. [State Graph Architecture](#state-graph-architecture)
5. [Sequence Diagram - Simple Task](#sequence-diagram---simple-task)
6. [Sequence Diagram - Complex Task](#sequence-diagram---complex-task)
7. [Supervisor Decision Flowchart](#supervisor-decision-flowchart)
8. [Agent Node Execution Flowchart](#agent-node-execution-flowchart)
9. [Routing Logic Flowchart](#routing-logic-flowchart)
10. [State Transition Diagram](#state-transition-diagram)
11. [Message Flow Diagram](#message-flow-diagram)

---

## Overview

This system implements a **Supervisor Pattern** for multi-agent orchestration where a central Supervisor Agent coordinates multiple specialized worker agents.

### System Components

| Component | Role | Description |
|-----------|------|-------------|
| **Supervisor Agent** | Orchestrator | Analyzes tasks and routes to appropriate workers |
| **Research Agent** | Worker | Gathers information, facts, and research |
| **Code Agent** | Worker | Writes code and technical implementations |
| **Writer Agent** | Worker | Creates articles, documentation, content |

### Architecture Pattern

```
┌─────────────────────────────────────────────────────────────┐
│                    SUPERVISOR PATTERN                        │
├─────────────────────────────────────────────────────────────┤
│                                                             │
│                    ┌──────────────┐                         │
│         ┌─────────►│  Supervisor  │◄─────────┐              │
│         │          │    Agent     │          │              │
│         │          └──────┬───────┘          │              │
│         │                 │                  │              │
│    ┌────┴────┐      ┌────┴────┐       ┌────┴────┐         │
│    │Research │      │  Code   │       │ Writer  │          │
│    │ Agent   │      │  Agent  │       │  Agent  │          │
│    └─────────┘      └─────────┘       └─────────┘          │
│                                                             │
└─────────────────────────────────────────────────────────────┘
```

---

## Class Diagram

This diagram shows the data structures, agent functions, and their relationships.

```mermaid
classDiagram
    class AgentState {
        <<TypedDict>>
        +list messages
        +str next_agent
        +str task
    }

    class StateGraph {
        +AgentState state_schema
        +add_node(name, func)
        +add_edge(source, target)
        +add_conditional_edges(source, router, mapping)
        +compile() CompiledGraph
    }

    class CompiledGraph {
        +invoke(state) AgentState
        +stream(state) Iterator
        +get_graph() Graph
    }

    class ChatAnthropic {
        +str model
        +float temperature
        +invoke(messages) AIMessage
    }

    class SupervisorNode {
        <<function>>
        +__call__(state) Dict
        -analyze_task()
        -determine_next_agent()
    }

    class ResearchAgentNode {
        <<function>>
        +system_message: str
        +__call__(state) Dict
    }

    class CodeAgentNode {
        <<function>>
        +system_message: str
        +__call__(state) Dict
    }

    class WriterAgentNode {
        <<function>>
        +system_message: str
        +__call__(state) Dict
    }

    class RouteToAgent {
        <<function>>
        +__call__(state) Literal
    }

    StateGraph --> AgentState : uses
    StateGraph --> CompiledGraph : compiles to
    
    SupervisorNode --> ChatAnthropic : uses
    ResearchAgentNode --> ChatAnthropic : uses
    CodeAgentNode --> ChatAnthropic : uses
    WriterAgentNode --> ChatAnthropic : uses

    StateGraph --> SupervisorNode : contains
    StateGraph --> ResearchAgentNode : contains
    StateGraph --> CodeAgentNode : contains
    StateGraph --> WriterAgentNode : contains
    StateGraph --> RouteToAgent : uses for routing
```

### Description

The class diagram shows:

- **AgentState**: TypedDict holding messages (accumulating), next_agent routing decision, and the original task
- **StateGraph**: LangGraph workflow builder for defining nodes and edges
- **CompiledGraph**: Executable workflow that can be invoked
- **ChatAnthropic**: Claude LLM for all agent reasoning
- **Agent Nodes**: Functions that process state and return updates
- **RouteToAgent**: Routing function for conditional edges

---

## Component Diagram

This diagram shows the high-level system organization.

```mermaid
flowchart TB
    subgraph UserInterface["👤 User Interface"]
        TASK["Task Input"]
        RESULT["Results Output"]
    end

    subgraph LangGraphWorkflow["🔄 LangGraph Workflow"]
        subgraph SupervisorLayer["🎯 Supervisor Layer"]
            SUP["Supervisor Node<br/>─────────────<br/>• Analyze task<br/>• Review history<br/>• Route to agent"]
        end

        subgraph WorkerLayer["👷 Worker Layer"]
            RES["Research Agent<br/>─────────────<br/>• Gather facts<br/>• Cite sources"]
            CODE["Code Agent<br/>─────────────<br/>• Write code<br/>• Best practices"]
            WRITE["Writer Agent<br/>─────────────<br/>• Create content<br/>• Format text"]
        end

        subgraph RoutingLayer["⚡ Routing Layer"]
            ROUTE["route_to_agent()<br/>─────────────<br/>• Parse decision<br/>• Map to node<br/>• Handle FINISH"]
        end
    end

    subgraph StateLayer["📦 State Layer"]
        STATE["AgentState<br/>─────────────<br/>• messages: List<br/>• next_agent: str<br/>• task: str"]
    end

    subgraph LLMLayer["🤖 LLM Layer"]
        LLM["ChatAnthropic<br/>Claude 3.5 Sonnet"]
    end

    TASK --> SUP
    SUP --> ROUTE
    ROUTE --> RES & CODE & WRITE
    RES & CODE & WRITE --> SUP
    ROUTE -->|"FINISH"| RESULT

    SUP & RES & CODE & WRITE <--> STATE
    SUP & RES & CODE & WRITE --> LLM

    style SupervisorLayer fill:#e3f2fd
    style WorkerLayer fill:#e8f5e9
    style RoutingLayer fill:#fff3e0
```

### Description

The system is organized into layers:

- **User Interface**: Task input and results output
- **Supervisor Layer**: Central routing and orchestration
- **Worker Layer**: Three specialized agents
- **Routing Layer**: Conditional edge logic
- **State Layer**: Shared state across all nodes
- **LLM Layer**: Claude model for all reasoning

---

## State Graph Architecture

This diagram shows the LangGraph workflow structure.

```mermaid
flowchart TD
    subgraph LangGraphWorkflow["LangGraph StateGraph"]
        START((START))
        
        SUP["supervisor<br/>─────────────<br/>supervisor_node()"]
        
        RES["research<br/>─────────────<br/>research_agent()"]
        
        CODE["code<br/>─────────────<br/>code_agent()"]
        
        WRITE["writer<br/>─────────────<br/>writer_agent()"]
        
        ENDN((END))
        
        START --> SUP
        
        SUP -->|"next_agent == 'research'"| RES
        SUP -->|"next_agent == 'code'"| CODE
        SUP -->|"next_agent == 'writer'"| WRITE
        SUP -->|"next_agent == 'FINISH'"| ENDN
        
        RES --> SUP
        CODE --> SUP
        WRITE --> SUP
    end

    style SUP fill:#e3f2fd
    style RES fill:#fff3e0
    style CODE fill:#e8f5e9
    style WRITE fill:#f3e5f5
```

### Description

The graph structure:

- **START → Supervisor**: All workflows begin at the supervisor
- **Conditional Edges**: Supervisor routes to appropriate worker based on `next_agent`
- **Worker → Supervisor**: All workers return to supervisor after completing
- **Supervisor → END**: Workflow terminates when supervisor decides "FINISH"

This creates a hub-and-spoke pattern with the supervisor at the center.

---

## Sequence Diagram - Simple Task

This diagram shows a simple task that requires only one agent.

```mermaid
sequenceDiagram
    autonumber
    participant User
    participant App as CompiledGraph
    participant Sup as Supervisor
    participant LLM as ChatAnthropic
    participant Code as Code Agent

    User->>App: invoke({task: "Write fibonacci function"})
    
    Note over App: Initialize state:<br/>messages: []<br/>task: "Write fibonacci..."<br/>next_agent: ""

    rect rgb(227, 242, 253)
        Note over Sup: First Supervisor Call
        App->>Sup: supervisor_node(state)
        Sup->>LLM: Analyze task + history
        LLM-->>Sup: "Code Agent"
        Sup->>Sup: Parse response
        Note over Sup: 🎯 Decision: Code Agent
        Sup-->>App: {next_agent: "code"}
    end

    rect rgb(232, 245, 233)
        Note over Code: Code Agent Execution
        App->>Code: code_agent(state)
        Code->>LLM: Generate fibonacci code
        LLM-->>Code: Python function + explanation
        Code-->>App: {messages: ["[Code Agent]: ..."]}
    end

    rect rgb(227, 242, 253)
        Note over Sup: Second Supervisor Call
        App->>Sup: supervisor_node(state)
        Sup->>LLM: Analyze task + history
        LLM-->>Sup: "FINISH"
        Note over Sup: 🎯 Decision: FINISH
        Sup-->>App: {next_agent: "FINISH"}
    end

    App->>App: route_to_agent() returns "__end__"
    App-->>User: Final state with results
```

### Description

Simple task flow:

1. **User Input**: Task submitted to compiled graph
2. **First Supervisor**: Analyzes task, routes to Code Agent
3. **Code Agent**: Generates fibonacci function
4. **Second Supervisor**: Reviews completed work, decides FINISH
5. **Termination**: Workflow ends, results returned

---

## Sequence Diagram - Complex Task

This diagram shows a complex task requiring multiple agents.

```mermaid
sequenceDiagram
    autonumber
    participant User
    participant App as CompiledGraph
    participant Sup as Supervisor
    participant LLM as ChatAnthropic
    participant Res as Research Agent
    participant Code as Code Agent
    participant Write as Writer Agent

    User->>App: invoke({task: "Research BST, implement it, explain"})
    
    Note over App: Complex multi-agent task

    rect rgb(255, 243, 224)
        Note over Sup: Supervisor → Research
        App->>Sup: supervisor_node(state)
        Sup->>LLM: Analyze task
        LLM-->>Sup: "Research Agent"
        Sup-->>App: {next_agent: "research"}
    end

    rect rgb(255, 248, 225)
        Note over Res: Research Execution
        App->>Res: research_agent(state)
        Res->>LLM: Research binary search trees
        LLM-->>Res: BST concepts, properties, operations
        Res-->>App: {messages: ["[Research Agent]: BST info..."]}
    end

    rect rgb(232, 245, 233)
        Note over Sup: Supervisor → Code
        App->>Sup: supervisor_node(state)
        Sup->>LLM: Analyze task + research results
        LLM-->>Sup: "Code Agent"
        Sup-->>App: {next_agent: "code"}
    end

    rect rgb(200, 230, 201)
        Note over Code: Code Execution
        App->>Code: code_agent(state)
        Code->>LLM: Implement BST in Python
        LLM-->>Code: BST class with insert, search, traverse
        Code-->>App: {messages: ["[Code Agent]: class BST..."]}
    end

    rect rgb(243, 229, 245)
        Note over Sup: Supervisor → Writer
        App->>Sup: supervisor_node(state)
        Sup->>LLM: Analyze task + code results
        LLM-->>Sup: "Writer Agent"
        Sup-->>App: {next_agent: "writer"}
    end

    rect rgb(225, 190, 231)
        Note over Write: Writer Execution
        App->>Write: writer_agent(state)
        Write->>LLM: Write explanation of BST
        LLM-->>Write: Clear documentation + examples
        Write-->>App: {messages: ["[Writer Agent]: BST explanation..."]}
    end

    rect rgb(227, 242, 253)
        Note over Sup: Supervisor → FINISH
        App->>Sup: supervisor_node(state)
        Sup->>LLM: Analyze completed work
        LLM-->>Sup: "FINISH"
        Sup-->>App: {next_agent: "FINISH"}
    end

    App-->>User: Complete results from all agents
```

### Description

Complex task flow with multiple agents:

1. **Research Phase**: Gather information about BST concepts
2. **Code Phase**: Implement BST based on research
3. **Writing Phase**: Create explanation based on research and code
4. **Completion**: All sub-tasks done, supervisor finishes

Messages accumulate, giving each subsequent agent full context.

---

## Supervisor Decision Flowchart

This diagram details the supervisor's decision-making logic.

```mermaid
flowchart TD
    START((Start)) --> A["Receive state:<br/>task, messages, next_agent"]
    
    A --> B["Build supervisor prompt:<br/>─────────────────<br/>• List available agents<br/>• Include current task<br/>• Include conversation history"]
    
    B --> C["Invoke LLM with prompt"]
    
    C --> D["Parse LLM response"]
    
    D --> E{"Response contains<br/>'Research Agent'?"}
    E -->|Yes| F["next_node = 'research'"]
    
    E -->|No| G{"Response contains<br/>'Code Agent'?"}
    G -->|Yes| H["next_node = 'code'"]
    
    G -->|No| I{"Response contains<br/>'Writer Agent'?"}
    I -->|Yes| J["next_node = 'writer'"]
    
    I -->|No| K{"Response contains<br/>'FINISH'?"}
    K -->|Yes| L["next_node = 'FINISH'"]
    
    K -->|No| L
    
    F & H & J & L --> M["Print decision:<br/>🎯 Supervisor Decision"]
    
    M --> N["Return {next_agent: next_node}"]
    
    N --> END((End))

    style E fill:#fff3e0
    style G fill:#e8f5e9
    style I fill:#f3e5f5
    style K fill:#ffebee
```

### Description

Supervisor decision process:

1. **Build Prompt**: Include agent descriptions, task, and full history
2. **LLM Reasoning**: Claude analyzes what's needed next
3. **Parse Response**: Match response to known agent names
4. **Default Handling**: Unknown responses default to FINISH
5. **Return Decision**: Update `next_agent` in state

The supervisor uses case-insensitive matching for flexibility.

---

## Agent Node Execution Flowchart

This diagram shows how worker agents process their tasks.

```mermaid
flowchart TD
    subgraph CreateAgentNode["🔧 create_agent_node(name, system_message)"]
        FACTORY["Factory function<br/>Returns agent_node()"]
    end

    subgraph AgentNodeExecution["⚡ agent_node(state) Execution"]
        A["Receive state"]
        
        B["Build message list:<br/>───────────────<br/>1. SystemMessage(system_message)<br/>2. HumanMessage(task)<br/>3. + existing messages"]
        
        C["Invoke LLM"]
        
        D["Format response:<br/>[Agent Name]: content"]
        
        E["Return {messages: [response]}"]
    end

    subgraph SystemMessages["📝 System Messages"]
        SM1["Research Agent:<br/>'You are a research agent...<br/>Focus on accuracy and cite sources.'"]
        
        SM2["Code Agent:<br/>'You are a coding expert...<br/>Write clean, efficient code.'"]
        
        SM3["Writer Agent:<br/>'You are a professional writer...<br/>Focus on clarity and tone.'"]
    end

    FACTORY --> AgentNodeExecution
    SM1 & SM2 & SM3 --> B
    A --> B --> C --> D --> E

    style CreateAgentNode fill:#e3f2fd
    style AgentNodeExecution fill:#e8f5e9
```

### Description

Agent node creation and execution:

**Factory Pattern:**
- `create_agent_node()` returns a configured agent function
- Each agent has a unique system message defining its role

**Execution Flow:**
1. Receive current state
2. Build messages: system prompt + task + history
3. Invoke LLM for response
4. Format with agent name prefix
5. Return message to accumulate in state

---

## Routing Logic Flowchart

This diagram shows the conditional routing implementation.

```mermaid
flowchart TD
    subgraph ConditionalEdges["📡 add_conditional_edges()"]
        CE["workflow.add_conditional_edges(<br/>  'supervisor',<br/>  route_to_agent,<br/>  {...mapping...}<br/>)"]
    end

    subgraph RouteFunction["⚡ route_to_agent(state)"]
        A["Get state['next_agent']"]
        
        B{"next_agent<br/>== 'FINISH'?"}
        
        B -->|Yes| C["Return '__end__'"]
        B -->|No| D["Return next_agent"]
    end

    subgraph EdgeMapping["🗺️ Edge Mapping"]
        MAP["Routing Map:<br/>───────────────<br/>'research' → research node<br/>'code' → code node<br/>'writer' → writer node<br/>'__end__' → END"]
    end

    subgraph Destinations["🎯 Destinations"]
        RES["research"]
        CODE["code"]
        WRITE["writer"]
        ENDN["END"]
    end

    CE --> RouteFunction
    RouteFunction --> EdgeMapping
    MAP --> RES & CODE & WRITE & ENDN

    style ConditionalEdges fill:#e3f2fd
    style RouteFunction fill:#fff3e0
    style EdgeMapping fill:#e8f5e9
```

### Description

Routing implementation:

**Conditional Edges:**
- Defined on the supervisor node
- Uses `route_to_agent()` function
- Maps return values to destination nodes

**route_to_agent():**
- Checks if `next_agent` is "FINISH"
- Returns `"__end__"` for termination
- Otherwise returns the agent node name

**Edge Mapping:**
- Dictionary mapping route values to actual nodes
- `"__end__"` maps to LangGraph's END constant

---

## State Transition Diagram

This diagram shows all possible state transitions.

```mermaid
stateDiagram-v2
    [*] --> Supervisor: START
    
    state Supervisor {
        [*] --> Analyzing
        Analyzing --> Deciding: LLM response
        Deciding --> [*]: Return next_agent
    }
    
    Supervisor --> Research: next_agent == "research"
    Supervisor --> Code: next_agent == "code"
    Supervisor --> Writer: next_agent == "writer"
    Supervisor --> [*]: next_agent == "FINISH"
    
    state Research {
        [*] --> Researching
        Researching --> Formatting: LLM response
        Formatting --> [*]: Return messages
    }
    
    state Code {
        [*] --> Coding
        Coding --> Formatting2: LLM response
        Formatting2 --> [*]: Return messages
    }
    
    state Writer {
        [*] --> Writing
        Writing --> Formatting3: LLM response
        Formatting3 --> [*]: Return messages
    }
    
    Research --> Supervisor: Complete
    Code --> Supervisor: Complete
    Writer --> Supervisor: Complete
    
    note right of Supervisor
        Central routing hub
        Analyzes task + history
        Decides next step
    end note
```

### Description

State transitions:

- **Supervisor**: Always the entry point and routing hub
- **Worker Agents**: Execute tasks and return to supervisor
- **Termination**: Only supervisor can end the workflow
- **Cycling**: Workers always return to supervisor for next decision

---

## Message Flow Diagram

This diagram shows how messages accumulate through the workflow.

```mermaid
flowchart TD
    subgraph Initial["📥 Initial State"]
        I1["messages: []<br/>task: 'Research BST...'<br/>next_agent: ''"]
    end

    subgraph AfterSup1["After Supervisor #1"]
        S1["messages: []<br/>task: 'Research BST...'<br/>next_agent: 'research'"]
    end

    subgraph AfterResearch["After Research Agent"]
        R1["messages: [<br/>  '[Research]: BST info...'<br/>]<br/>task: 'Research BST...'<br/>next_agent: 'research'"]
    end

    subgraph AfterSup2["After Supervisor #2"]
        S2["messages: [<br/>  '[Research]: BST info...'<br/>]<br/>task: 'Research BST...'<br/>next_agent: 'code'"]
    end

    subgraph AfterCode["After Code Agent"]
        C1["messages: [<br/>  '[Research]: BST info...',<br/>  '[Code]: class BST...'<br/>]<br/>task: 'Research BST...'<br/>next_agent: 'code'"]
    end

    subgraph AfterSup3["After Supervisor #3"]
        S3["messages: [<br/>  '[Research]: BST info...',<br/>  '[Code]: class BST...'<br/>]<br/>task: 'Research BST...'<br/>next_agent: 'writer'"]
    end

    subgraph AfterWriter["After Writer Agent"]
        W1["messages: [<br/>  '[Research]: BST info...',<br/>  '[Code]: class BST...',<br/>  '[Writer]: BST explanation...'<br/>]<br/>task: 'Research BST...'<br/>next_agent: 'writer'"]
    end

    subgraph Final["📤 Final State"]
        F1["messages: [<br/>  '[Research]: BST info...',<br/>  '[Code]: class BST...',<br/>  '[Writer]: BST explanation...'<br/>]<br/>task: 'Research BST...'<br/>next_agent: 'FINISH'"]
    end

    Initial --> AfterSup1
    AfterSup1 --> AfterResearch
    AfterResearch --> AfterSup2
    AfterSup2 --> AfterCode
    AfterCode --> AfterSup3
    AfterSup3 --> AfterWriter
    AfterWriter --> Final

    style Initial fill:#e3f2fd
    style Final fill:#e8f5e9
```

### Description

Message accumulation pattern:

- **Initial**: Empty messages list
- **After Each Agent**: New message appended via `operator.add`
- **Context Building**: Each agent sees all previous messages
- **Final State**: Complete conversation history preserved

The `Annotated[list, operator.add]` enables automatic message accumulation.

---

## Complete Workflow Example

This diagram shows a full end-to-end example.

```mermaid
flowchart TD
    subgraph Input["📥 User Input"]
        TASK["Task: 'Write a Python function<br/>that calculates fibonacci'"]
    end

    subgraph Execution["⚡ Workflow Execution"]
        subgraph Cycle1["Cycle 1"]
            SUP1["Supervisor<br/>─────────────<br/>Analyzes: coding task<br/>Decision: Code Agent"]
            CODE1["Code Agent<br/>─────────────<br/>Generates fibonacci<br/>function with docs"]
        end

        subgraph Cycle2["Cycle 2"]
            SUP2["Supervisor<br/>─────────────<br/>Reviews: code complete<br/>Decision: FINISH"]
        end
    end

    subgraph Output["📤 Final Output"]
        RESULT["Results:<br/>─────────────<br/>[Code Agent]:<br/>def fibonacci(n):<br/>    '''Calculate fibonacci...'''<br/>    if n <= 1:<br/>        return n<br/>    return fibonacci(n-1) + fibonacci(n-2)"]
    end

    TASK --> SUP1
    SUP1 -->|"route: code"| CODE1
    CODE1 -->|"return to supervisor"| SUP2
    SUP2 -->|"route: __end__"| RESULT

    style Cycle1 fill:#e8f5e9
    style Cycle2 fill:#fff3e0
```

### Description

Complete workflow for a coding task:

1. **Input**: User submits fibonacci function request
2. **Cycle 1**: Supervisor → Code Agent → generates function
3. **Cycle 2**: Supervisor reviews and decides FINISH
4. **Output**: Complete code with documentation

---

## Summary

This documentation covers the complete architecture of the Multi-Agent Supervisor System:

| Diagram Type | Purpose |
|-------------|---------|
| Class Diagram | Data structures and relationships |
| Component Diagram | High-level system layers |
| State Graph | LangGraph workflow structure |
| Simple Sequence | Single-agent task flow |
| Complex Sequence | Multi-agent task flow |
| Supervisor Decision | Routing decision logic |
| Agent Execution | Worker node implementation |
| Routing Logic | Conditional edge implementation |
| State Transitions | All possible state changes |
| Message Flow | State evolution through workflow |
| Complete Example | End-to-end execution |

### Key Architecture Patterns

1. **Supervisor Pattern**: Central coordinator routes to specialized workers
2. **Hub-and-Spoke**: All paths go through the supervisor
3. **Message Accumulation**: Context builds via `operator.add`
4. **Factory Pattern**: `create_agent_node()` generates configured agents
5. **Conditional Routing**: Dynamic edge selection based on state

### Supervisor Pattern Benefits

| Benefit | Description |
|---------|-------------|
| **Centralized Control** | Single point for routing decisions |
| **Context Awareness** | Supervisor sees full conversation history |
| **Dynamic Routing** | Any agent can be called at any time |
| **Graceful Termination** | Supervisor decides when task is complete |
| **Scalability** | Easy to add new specialized agents |

### Extension Points

- **Add More Agents**: Data analyst, translator, validator, etc.
- **Agent Tools**: Give agents access to web search, calculators, databases
- **Memory Systems**: Long-term context across sessions
- **Human-in-the-Loop**: Approval steps before certain actions
- **Hierarchical Supervisors**: Sub-supervisors for complex workflows
- **Parallel Execution**: Multiple agents working simultaneously
- **Error Recovery**: Retry logic and fallback agents
