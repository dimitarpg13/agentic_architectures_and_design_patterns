# 🔬 Researcher Agent with LangGraph - Architecture Documentation

This document provides comprehensive UML diagrams and flowcharts describing the architecture and workflow of the Researcher Agent implementation using LangGraph.

---

## Table of Contents

1. [Overview](#overview)
2. [Class Diagram](#class-diagram)
3. [Component Diagram](#component-diagram)
4. [Sequence Diagram](#sequence-diagram)
5. [State Machine Diagram](#state-machine-diagram)
6. [Workflow Flowchart](#workflow-flowchart)
7. [Data Flow Diagram](#data-flow-diagram)
8. [Tool Interaction Diagram](#tool-interaction-diagram)

---

## Overview

The Researcher Agent is an intelligent system built using LangGraph and LangChain that:
- Accepts natural language research queries
- Creates structured research plans
- Searches the web for relevant information
- Synthesizes findings into comprehensive research reports

The agent uses a graph-based workflow with multiple specialized nodes that collaborate to produce high-quality research outputs.

---

## Class Diagram

This diagram shows the main classes, their attributes, methods, and relationships within the Researcher Agent system.

```mermaid
classDiagram
    class ResearcherState {
        <<TypedDict>>
        +list messages
        +str research_query
        +str research_plan
        +List~str~ search_results
        +str final_report
        +int iteration_count
    }

    class StateGraph {
        +ResearcherState state_schema
        +add_node(name, func)
        +add_edge(source, target)
        +add_conditional_edges(source, condition, mapping)
        +compile() CompiledGraph
    }

    class CompiledGraph {
        +invoke(state) ResearcherState
        +stream(state) Iterator
        +get_graph() Graph
    }

    class ToolNode {
        +list tools
        +invoke(state) dict
    }

    class ChatOpenAI {
        +str model
        +float temperature
        +bind_tools(tools) LLMWithTools
        +invoke(messages) AIMessage
    }

    class ChatAnthropic {
        +str model
        +float temperature
        +bind_tools(tools) LLMWithTools
        +invoke(messages) AIMessage
    }

    class ChatGoogleGenerativeAI {
        +str model
        +float temperature
        +bind_tools(tools) LLMWithTools
        +invoke(messages) AIMessage
    }

    class TavilyClient {
        +str api_key
        +search(query, search_depth, max_results) dict
    }

    class WebSearchTool {
        <<tool>>
        +str name
        +str description
        +__call__(query) str
    }

    class GetCurrentDateTool {
        <<tool>>
        +str name
        +str description
        +__call__() str
    }

    class ChatPromptTemplate {
        +from_template(template) ChatPromptTemplate
        +invoke(variables) PromptValue
    }

    class StrOutputParser {
        +invoke(message) str
    }

    StateGraph --> ResearcherState : uses
    StateGraph --> CompiledGraph : compiles to
    CompiledGraph --> ToolNode : contains
    ToolNode --> WebSearchTool : executes
    ToolNode --> GetCurrentDateTool : executes
    WebSearchTool --> TavilyClient : uses
    ChatOpenAI --|> LLM : implements
    ChatAnthropic --|> LLM : implements
    ChatGoogleGenerativeAI --|> LLM : implements

    class LLM {
        <<interface>>
        +bind_tools(tools)
        +invoke(messages)
    }
```

### Description

The class diagram illustrates the core components of the Researcher Agent:

- **ResearcherState**: A TypedDict that maintains the agent's state throughout execution, including messages, query, plan, results, and iteration tracking.
- **StateGraph**: The LangGraph workflow builder that defines nodes and edges for the agent graph.
- **CompiledGraph**: The executable version of the state graph that can be invoked or streamed.
- **LLM Providers**: Three interchangeable LLM implementations (OpenAI, Anthropic, Google) that share a common interface.
- **Tools**: Specialized functions (`web_search`, `get_current_date`) that the agent can call during research.
- **TavilyClient**: The web search client that provides real-time internet search capabilities.

---

## Component Diagram

This diagram shows the high-level components and their dependencies.

```mermaid
classDiagram
    direction TB
    
    class UserInterface {
        <<component>>
        run_research()
        run_research_streaming()
    }

    class ResearcherAgent {
        <<component>>
        StateGraph workflow
        invoke()
        stream()
    }

    class PlannerNode {
        <<component>>
        PLANNER_PROMPT
        plan_research()
    }

    class ResearcherNode {
        <<component>>
        RESEARCHER_PROMPT
        conduct_research()
    }

    class SynthesizerNode {
        <<component>>
        SYNTHESIZER_PROMPT
        synthesize_report()
    }

    class ToolsNode {
        <<component>>
        web_search
        get_current_date
    }

    class LLMProvider {
        <<component>>
        get_llm()
        OpenAI | Anthropic | Google
    }

    class WebSearchService {
        <<external>>
        Tavily API
    }

    UserInterface --> ResearcherAgent : queries
    ResearcherAgent --> PlannerNode : contains
    ResearcherAgent --> ResearcherNode : contains
    ResearcherAgent --> SynthesizerNode : contains
    ResearcherAgent --> ToolsNode : contains
    PlannerNode --> LLMProvider : uses
    ResearcherNode --> LLMProvider : uses
    SynthesizerNode --> LLMProvider : uses
    ToolsNode --> WebSearchService : calls
```

### Description

The component diagram shows how the Researcher Agent is organized into modular components:

- **UserInterface**: Entry point functions for executing research queries
- **ResearcherAgent**: The main orchestrator containing all workflow nodes
- **PlannerNode**: Responsible for analyzing queries and creating research plans
- **ResearcherNode**: Executes the research plan using available tools
- **SynthesizerNode**: Compiles findings into coherent reports
- **ToolsNode**: Container for executable tools (web search, date retrieval)
- **LLMProvider**: Abstraction layer for LLM selection and initialization
- **WebSearchService**: External Tavily API integration

---

## Sequence Diagram

This diagram shows the temporal flow of a research query through the system.

```mermaid
sequenceDiagram
    autonumber
    participant User
    participant Agent as Researcher Agent
    participant Planner as Planner Node
    participant Researcher as Researcher Node
    participant Tools as Tools Node
    participant Tavily as Tavily API
    participant Synthesizer as Synthesizer Node
    participant LLM as LLM Provider

    User->>Agent: run_research(query)
    
    Note over Agent: Initialize State
    Agent->>Agent: Create initial ResearcherState
    
    rect rgb(240, 248, 255)
        Note over Planner: Planning Phase
        Agent->>Planner: plan_research(state)
        Planner->>LLM: Generate research plan
        LLM-->>Planner: Research plan text
        Planner-->>Agent: Updated state with plan
    end
    
    rect rgb(255, 248, 240)
        Note over Researcher: Research Loop
        loop Until RESEARCH COMPLETE or max iterations
            Agent->>Researcher: conduct_research(state)
            Researcher->>LLM: Invoke with tools bound
            LLM-->>Researcher: Response (may include tool calls)
            
            alt Tool calls present
                Researcher-->>Agent: State with tool calls
                Agent->>Tools: Execute tool calls
                Tools->>Tavily: web_search(query)
                Tavily-->>Tools: Search results
                Tools-->>Agent: State with tool results
                Agent->>Researcher: Continue research
            else RESEARCH COMPLETE
                Researcher-->>Agent: Signal completion
            end
        end
    end
    
    rect rgb(240, 255, 240)
        Note over Synthesizer: Synthesis Phase
        Agent->>Synthesizer: synthesize_report(state)
        Synthesizer->>Synthesizer: Extract findings from messages
        Synthesizer->>LLM: Generate final report
        LLM-->>Synthesizer: Synthesized report
        Synthesizer-->>Agent: State with final_report
    end
    
    Agent-->>User: Return final_report
```

### Description

The sequence diagram illustrates the complete lifecycle of a research query:

1. **Initialization**: User submits a query, agent creates initial state
2. **Planning Phase**: The planner node analyzes the query and generates a structured research plan with specific search queries
3. **Research Loop**: The researcher node iteratively:
   - Decides whether to search or synthesize
   - Executes web searches via the Tools node
   - Accumulates findings in the message history
   - Continues until "RESEARCH COMPLETE" or max iterations reached
4. **Synthesis Phase**: The synthesizer compiles all findings into a professional report
5. **Return**: Final report is returned to the user

---

## State Machine Diagram

This diagram shows the states and transitions of the Researcher Agent workflow.

```mermaid
stateDiagram-v2
    [*] --> START
    START --> Planner: Begin workflow
    
    Planner --> Researcher: Plan created
    
    state ResearchCycle {
        Researcher --> CheckCondition: Response generated
        
        CheckCondition --> Tools: tool_calls present
        CheckCondition --> Synthesizer: RESEARCH COMPLETE
        CheckCondition --> Synthesizer: max iterations (5)
        
        Tools --> Researcher: Tool results returned
    }
    
    Synthesizer --> END: Report generated
    END --> [*]
    
    note right of Planner
        Creates research plan
        with specific queries
    end note
    
    note right of Researcher
        Executes searches
        based on plan
    end note
    
    note right of Synthesizer
        Compiles findings
        into final report
    end note
```

### Description

The state machine diagram shows the possible states and transitions:

- **START**: Initial entry point
- **Planner**: Creates the research strategy
- **Researcher**: Core research execution state
- **CheckCondition**: Decision point for routing
- **Tools**: Tool execution state
- **Synthesizer**: Report generation state
- **END**: Terminal state

Transitions are governed by:
- Presence of tool calls in the LLM response
- Detection of "RESEARCH COMPLETE" signal
- Maximum iteration count (5) reached

---

## Workflow Flowchart

This diagram provides a detailed view of the decision logic and data flow.

```mermaid
flowchart TD
    subgraph Input
        A[/"📌 User Research Query"/]
    end
    
    subgraph Initialization
        B["🔧 Initialize ResearcherState<br/>• messages: []<br/>• research_query: query<br/>• iteration_count: 0"]
    end
    
    subgraph Planning["📋 Planning Phase"]
        C["Planner Node"]
        C1["Generate research plan<br/>using PLANNER_PROMPT"]
        C2["Extract key aspects<br/>Define 2-4 search queries"]
        C --> C1 --> C2
    end
    
    subgraph Research["🔍 Research Phase"]
        D["Researcher Node"]
        D1{"Check Routing<br/>Condition"}
        D2["Execute Tool"]
        D3["Process Results"]
        
        D --> D1
        D1 -->|"tool_calls present"| D2
        D2 --> D3
        D3 --> D
        
        D1 -->|"RESEARCH COMPLETE<br/>OR iteration >= 5"| E
    end
    
    subgraph Tools["🔧 Tools Node"]
        T1["web_search()"]
        T2["get_current_date()"]
        D2 -.-> T1
        D2 -.-> T2
    end
    
    subgraph Synthesis["📝 Synthesis Phase"]
        E["Synthesizer Node"]
        E1["Extract findings<br/>from messages"]
        E2["Generate report<br/>using SYNTHESIZER_PROMPT"]
        E3["Format with sections:<br/>• Executive Summary<br/>• Key Findings<br/>• Limitations<br/>• Sources"]
        E --> E1 --> E2 --> E3
    end
    
    subgraph Output
        F[/"📄 Final Research Report"/]
    end
    
    A --> B
    B --> C
    C2 --> D
    E3 --> F
    
    style A fill:#e1f5fe
    style F fill:#e8f5e9
    style C fill:#fff3e0
    style D fill:#fce4ec
    style E fill:#f3e5f5
```

### Description

The workflow flowchart provides a comprehensive view of the agent's execution:

1. **Input**: User provides a natural language research query
2. **Initialization**: State object is created with default values
3. **Planning Phase**: 
   - Analyzes the query
   - Identifies key aspects to investigate
   - Generates 2-4 specific search queries
4. **Research Phase**:
   - Iteratively executes searches
   - Routes based on tool calls or completion signals
   - Maximum 5 iterations as safety limit
5. **Tools Node**: Executes web searches via Tavily
6. **Synthesis Phase**:
   - Extracts findings from message history
   - Generates structured report with sections
7. **Output**: Returns the final research report

---

## Data Flow Diagram

This diagram shows how data transforms as it flows through the system.

```mermaid
flowchart LR
    subgraph Inputs
        Q["Research Query<br/>(string)"]
    end
    
    subgraph StateTransformations["State Transformations"]
        S1["Initial State<br/>───────────<br/>messages: []<br/>query: input<br/>plan: ''<br/>results: []<br/>report: ''<br/>iterations: 0"]
        
        S2["After Planning<br/>───────────<br/>messages: [System, Human]<br/>query: input<br/>plan: 'structured plan'<br/>results: []<br/>report: ''<br/>iterations: 0"]
        
        S3["During Research<br/>───────────<br/>messages: [+AI, +Tool]<br/>query: input<br/>plan: 'structured plan'<br/>results: ['search1']<br/>report: ''<br/>iterations: 1-5"]
        
        S4["After Synthesis<br/>───────────<br/>messages: [+Final AI]<br/>query: input<br/>plan: 'structured plan'<br/>results: ['all searches']<br/>report: 'final report'<br/>iterations: n"]
    end
    
    subgraph Outputs
        R["Research Report<br/>(markdown string)"]
    end
    
    Q --> S1
    S1 -->|"plan_research()"| S2
    S2 -->|"conduct_research()<br/>+ tools"| S3
    S3 -->|"loop"| S3
    S3 -->|"synthesize_report()"| S4
    S4 --> R
```

### Description

The data flow diagram traces how the `ResearcherState` evolves:

1. **Initial State**: Empty state with only the research query populated
2. **After Planning**: State gains a research plan and initial messages (system prompt + human query)
3. **During Research**: Messages accumulate AI responses and tool results; iteration counter increases
4. **After Synthesis**: Final report is generated and stored in state

Key data transformations:
- Query → Research Plan (via LLM)
- Plan → Search Results (via Tavily)
- Search Results → Synthesized Report (via LLM)

---

## Tool Interaction Diagram

This diagram details the tool execution flow and Tavily API interaction.

```mermaid
flowchart TD
    subgraph Agent["Researcher Agent"]
        R["Researcher Node"]
        LLM["LLM with Tools Bound"]
        R --> LLM
    end
    
    subgraph ToolExecution["Tool Execution Layer"]
        TN["ToolNode"]
        
        subgraph AvailableTools["Available Tools"]
            WS["web_search<br/>────────<br/>@tool decorator<br/>query: str → str"]
            GD["get_current_date<br/>────────<br/>@tool decorator<br/>() → str"]
        end
        
        TN --> WS
        TN --> GD
    end
    
    subgraph ExternalServices["External Services"]
        TC["TavilyClient"]
        TA["Tavily API<br/>────────<br/>search_depth: 'advanced'<br/>max_results: 5<br/>include_answer: True"]
        
        TC --> TA
    end
    
    LLM -->|"tool_calls: [web_search]"| TN
    WS -->|"tavily_client.search()"| TC
    
    TA -->|"JSON Response"| TC
    TC -->|"Formatted Results"| WS
    WS -->|"Tool Result"| TN
    TN -->|"Add to messages"| R
    
    subgraph ResponseFormat["Search Response Format"]
        RF["**Quick Answer:** AI summary<br/>────────<br/>**Search Results:**<br/>1. Title, URL, Content<br/>2. Title, URL, Content<br/>..."]
    end
    
    WS -.-> RF
    
    style TA fill:#e3f2fd
    style RF fill:#f5f5f5
```

### Description

The tool interaction diagram shows how tools are invoked and executed:

1. **LLM Decision**: The LLM decides to call a tool and includes tool_calls in its response
2. **ToolNode Processing**: LangGraph's ToolNode receives the tool call request
3. **Tool Execution**: The appropriate tool function is invoked:
   - `web_search`: Calls Tavily API with advanced search depth
   - `get_current_date`: Returns current timestamp
4. **Response Formatting**: Search results are formatted into a readable string
5. **State Update**: Results are added to the message history

### Tavily API Parameters:
- `search_depth`: "advanced" for comprehensive results
- `max_results`: 5 results per query
- `include_answer`: True for AI-generated quick answers
- `include_raw_content`: False to reduce response size

---

## Routing Decision Flowchart

This diagram details the conditional routing logic in the research phase.

```mermaid
flowchart TD
    START((Start)) --> A["Get last message<br/>from state"]
    
    A --> B{"iteration_count<br/>>= 5?"}
    
    B -->|Yes| C["Return 'synthesize'<br/>⏹️ Max iterations reached"]
    
    B -->|No| D{"Has tool_calls<br/>attribute?"}
    
    D -->|No| G{"Contains<br/>'RESEARCH COMPLETE'?"}
    
    D -->|Yes| E{"tool_calls<br/>not empty?"}
    
    E -->|Yes| F["Return 'tools'<br/>🔧 Execute tool call"]
    
    E -->|No| G
    
    G -->|Yes| H["Return 'synthesize'<br/>✅ Research complete"]
    
    G -->|No| I["Return 'synthesize'<br/>Default: no tools needed"]
    
    C --> END((End))
    F --> END
    H --> END
    I --> END
    
    style F fill:#e3f2fd
    style C fill:#ffebee
    style H fill:#e8f5e9
    style I fill:#fff8e1
```

### Description

The routing decision flowchart shows the `should_continue_research()` logic:

1. **Check Iteration Limit**: If 5 or more iterations, force synthesis
2. **Check Tool Calls**: If the LLM's response contains tool calls, route to tools
3. **Check Completion Signal**: If "RESEARCH COMPLETE" is in the response, proceed to synthesis
4. **Default**: If none of the above, default to synthesis

This ensures:
- Research doesn't run indefinitely (max 5 iterations)
- Tool calls are properly executed
- Agent can signal when it has enough information
- Graceful fallback to synthesis when uncertain

---

## Summary

This documentation covers the complete architecture of the Researcher Agent:

| Diagram Type | Purpose |
|-------------|---------|
| Class Diagram | Shows data structures and their relationships |
| Component Diagram | High-level system organization |
| Sequence Diagram | Temporal flow of operations |
| State Machine | Workflow states and transitions |
| Workflow Flowchart | Detailed execution logic |
| Data Flow | State transformations |
| Tool Interaction | External API integration |
| Routing Decision | Conditional logic details |

### Key Architecture Decisions

1. **Graph-based Workflow**: LangGraph enables clear separation of concerns and conditional routing
2. **Multi-provider LLM**: Abstraction allows switching between OpenAI, Anthropic, and Google
3. **Tool-augmented Research**: Web search provides real-time information retrieval
4. **Iterative Research Loop**: Agent can conduct multiple searches before synthesizing
5. **Structured State**: TypedDict provides type safety and clear data contracts
6. **Streaming Support**: Real-time output for better user experience

### Extension Points

- Add more tools (academic search, document analysis)
- Implement memory for multi-session research
- Add citation verification and fact-checking
- Create web interface with Gradio/Streamlit
- Integrate additional LLM providers
