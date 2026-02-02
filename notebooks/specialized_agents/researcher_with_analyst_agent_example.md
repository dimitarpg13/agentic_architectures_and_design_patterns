# 🔬📊 Multi-Agent Research & Analysis System - Architecture Documentation

This document provides comprehensive UML diagrams and flowcharts describing the architecture and workflow of the Multi-Agent Research & Analysis System using LangGraph, featuring collaborative Researcher and Analyst agents.

---

## Table of Contents

1. [Overview](#overview)
2. [Class Diagram](#class-diagram)
3. [Component Diagram](#component-diagram)
4. [Agent Interaction Diagram](#agent-interaction-diagram)
5. [Sequence Diagram](#sequence-diagram)
6. [State Machine Diagram](#state-machine-diagram)
7. [Workflow Flowchart](#workflow-flowchart)
8. [Data Flow Diagram](#data-flow-diagram)
9. [Tool Architecture Diagram](#tool-architecture-diagram)
10. [Iterative Loop Flowchart](#iterative-loop-flowchart)

---

## Overview

The Multi-Agent Research & Analysis System is a collaborative AI system featuring two specialized agents:

| Agent | Role | Capabilities |
|-------|------|--------------|
| 🔬 **Researcher Agent** | Information Gathering | Search docs, papers, benchmarks, examples, costs |
| 📊 **Analyst Agent** | Critical Analysis | Compare trade-offs, assess costs, identify challenges, recommend |

The system uses an iterative workflow where the Analyst can request additional research until sufficient information is gathered to make a well-supported recommendation.

---

## Class Diagram

This diagram shows all classes, their attributes, methods, and relationships within the multi-agent system.

```mermaid
classDiagram
    class MultiAgentState {
        <<TypedDict>>
        +list messages
        +str user_query
        +List~str~ research_findings
        +List~str~ analysis_notes
        +List~str~ research_requests
        +int iteration_count
        +str current_agent
        +str final_recommendation
    }

    class AgentAction {
        <<Enum>>
        +NEED_MORE_RESEARCH
        +READY_TO_CONCLUDE
    }

    class ResearchRequest {
        <<BaseModel>>
        +str request_type
        +str specific_query
        +str reason
    }

    class StateGraph {
        +MultiAgentState state_schema
        +add_node(name, func)
        +add_edge(source, target)
        +add_conditional_edges(source, condition, mapping)
        +compile() CompiledGraph
    }

    class CompiledGraph {
        +invoke(state) MultiAgentState
        +stream(state) Iterator
        +get_graph() Graph
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

    class SearchTechnicalDocs {
        <<tool>>
        +name: "search_technical_docs"
        +__call__(query) str
    }

    class SearchAcademicPapers {
        <<tool>>
        +name: "search_academic_papers"
        +__call__(query) str
    }

    class SearchBenchmarks {
        <<tool>>
        +name: "search_benchmarks"
        +__call__(query) str
    }

    class SearchImplementationExamples {
        <<tool>>
        +name: "search_implementation_examples"
        +__call__(query) str
    }

    class SearchCostAnalysis {
        <<tool>>
        +name: "search_cost_analysis"
        +__call__(query) str
    }

    class ToolNode {
        +list tools
        +invoke(state) dict
    }

    StateGraph --> MultiAgentState : uses
    StateGraph --> CompiledGraph : compiles to
    MultiAgentState --> AgentAction : tracks
    MultiAgentState --> ResearchRequest : contains

    ChatOpenAI --|> LLM : implements
    ChatAnthropic --|> LLM : implements
    ChatGoogleGenerativeAI --|> LLM : implements

    class LLM {
        <<interface>>
        +bind_tools(tools)
        +invoke(messages)
    }

    SearchTechnicalDocs --> TavilyClient : uses
    SearchAcademicPapers --> TavilyClient : uses
    SearchBenchmarks --> TavilyClient : uses
    SearchImplementationExamples --> TavilyClient : uses
    SearchCostAnalysis --> TavilyClient : uses

    ToolNode --> SearchTechnicalDocs : executes
    ToolNode --> SearchAcademicPapers : executes
    ToolNode --> SearchBenchmarks : executes
    ToolNode --> SearchImplementationExamples : executes
    ToolNode --> SearchCostAnalysis : executes
```

### Description

The class diagram illustrates the core components:

- **MultiAgentState**: Extended state tracking conversation, findings, analysis notes, and current agent
- **AgentAction**: Enum defining possible analyst decisions (need more research vs ready to conclude)
- **ResearchRequest**: Pydantic model for structured research requests from analyst
- **Five Specialized Tools**: Each tool targets a specific type of information search
- **LLM Providers**: Interchangeable implementations (OpenAI, Anthropic, Google)
- **TavilyClient**: Shared web search client used by all research tools

---

## Component Diagram

This diagram shows the high-level system architecture and component relationships.

```mermaid
classDiagram
    direction TB

    class UserInterface {
        <<component>>
        run_analysis()
    }

    class MultiAgentSystem {
        <<component>>
        StateGraph workflow
        invoke()
        stream()
    }

    class ResearcherAgent {
        <<component>>
        RESEARCHER_SYSTEM_PROMPT
        initial_research()
        additional_research()
    }

    class AnalystAgent {
        <<component>>
        ANALYST_SYSTEM_PROMPT
        analyze_research()
    }

    class ReportGenerator {
        <<component>>
        generate_final_report()
    }

    class ResearchToolkit {
        <<component>>
        search_technical_docs
        search_academic_papers
        search_benchmarks
        search_implementation_examples
        search_cost_analysis
    }

    class LLMProvider {
        <<component>>
        get_llm()
        llm (temp=0.7)
        llm_structured (temp=0.3)
    }

    class TavilyService {
        <<external>>
        Tavily API
    }

    UserInterface --> MultiAgentSystem : queries
    MultiAgentSystem --> ResearcherAgent : contains
    MultiAgentSystem --> AnalystAgent : contains
    MultiAgentSystem --> ReportGenerator : contains
    ResearcherAgent --> ResearchToolkit : uses
    ResearcherAgent --> LLMProvider : uses
    AnalystAgent --> LLMProvider : uses
    ReportGenerator --> LLMProvider : uses
    ResearchToolkit --> TavilyService : calls
```

### Description

The component diagram shows the modular organization:

- **UserInterface**: Entry point for running analysis queries
- **MultiAgentSystem**: Main orchestrator managing the workflow graph
- **ResearcherAgent**: Handles initial and additional research phases
- **AnalystAgent**: Reviews research, identifies gaps, makes decisions
- **ReportGenerator**: Creates the final comprehensive recommendation
- **ResearchToolkit**: Collection of 5 specialized search tools
- **LLMProvider**: Dual LLM configuration (standard + structured output)
- **TavilyService**: External API for web searches

---

## Agent Interaction Diagram

This diagram illustrates how the two agents collaborate and communicate.

```mermaid
flowchart TD
    subgraph User["👤 User"]
        Q["Technical Decision Query"]
    end

    subgraph ResearcherAgent["🔬 Researcher Agent"]
        R1["Initial Research"]
        R2["Additional Research"]
        RT["Research Toolkit<br/>─────────────<br/>📚 Technical Docs<br/>📄 Academic Papers<br/>📊 Benchmarks<br/>💻 Examples<br/>💰 Cost Analysis"]
    end

    subgraph AnalystAgent["📊 Analyst Agent"]
        A1["Analyze Findings"]
        A2{"Sufficient<br/>Information?"}
        A3["Request More<br/>Research"]
        A4["Ready to<br/>Conclude"]
    end

    subgraph Output["📄 Output"]
        FR["Final Recommendation<br/>Report"]
    end

    Q --> R1
    R1 --> RT
    RT --> A1
    A1 --> A2
    A2 -->|"No"| A3
    A3 -->|"Specific Request"| R2
    R2 --> RT
    RT -->|"Additional Findings"| A1
    A2 -->|"Yes"| A4
    A4 --> FR

    style ResearcherAgent fill:#e3f2fd
    style AnalystAgent fill:#fff3e0
    style Output fill:#e8f5e9
```

### Description

The agent interaction diagram shows the collaborative workflow:

1. **User Query** flows to the Researcher Agent
2. **Researcher** gathers initial information using the toolkit
3. **Analyst** reviews findings and determines if sufficient
4. **Iterative Loop**: Analyst requests specific additional research if needed
5. **Final Output**: Comprehensive recommendation when Analyst is satisfied

---

## Sequence Diagram

This diagram shows the complete temporal flow including iterative cycles.

```mermaid
sequenceDiagram
    autonumber
    participant User
    participant System as Multi-Agent System
    participant Researcher as 🔬 Researcher Agent
    participant Tools as Research Toolkit
    participant Tavily as Tavily API
    participant Analyst as 📊 Analyst Agent
    participant Report as Report Generator

    User->>System: run_analysis(query)
    
    Note over System: Initialize MultiAgentState
    
    rect rgb(227, 242, 253)
        Note over Researcher: Initial Research Phase
        System->>Researcher: initial_research(state)
        
        loop Up to 5 tool calls
            Researcher->>Tools: Select appropriate tool
            Tools->>Tavily: search(query + modifiers)
            Tavily-->>Tools: Search results
            Tools-->>Researcher: Formatted findings
        end
        
        Researcher->>Researcher: Summarize findings
        Researcher-->>System: State with research_findings
    end
    
    rect rgb(255, 243, 224)
        Note over Analyst: Analysis Phase
        System->>Analyst: analyze_research(state)
        Analyst->>Analyst: Review all findings
        Analyst->>Analyst: Identify gaps
        
        alt Needs more research
            Analyst-->>System: NEED_MORE_RESEARCH
            Note over System: current_agent = "researcher"
        else Ready to conclude
            Analyst-->>System: READY_TO_CONCLUDE
            Note over System: current_agent = "final"
        end
    end
    
    opt Additional Research Loop (max 3 iterations)
        rect rgb(232, 245, 233)
            Note over Researcher: Additional Research Phase
            System->>Researcher: additional_research(state)
            
            loop Up to 3 tool calls
                Researcher->>Tools: Targeted search
                Tools->>Tavily: search(specific_query)
                Tavily-->>Tools: Results
                Tools-->>Researcher: Findings
            end
            
            Researcher-->>System: Updated research_findings
        end
        
        System->>Analyst: analyze_research(state)
        Note over Analyst: Re-evaluate with new data
    end
    
    rect rgb(243, 229, 245)
        Note over Report: Final Report Phase
        System->>Report: generate_final_report(state)
        Report->>Report: Compile all research
        Report->>Report: Synthesize analysis
        Report->>Report: Generate structured report
        Report-->>System: final_recommendation
    end
    
    System-->>User: Return final report
```

### Description

The sequence diagram shows the complete lifecycle:

1. **Initialization**: System creates MultiAgentState with user query
2. **Initial Research** (blue): Researcher uses up to 5 tool calls to gather comprehensive information
3. **Analysis** (orange): Analyst reviews findings and decides next action
4. **Additional Research** (green, optional): If gaps identified, Researcher conducts targeted searches
5. **Iterative Loop**: Analysis and additional research can repeat up to 3 times
6. **Final Report** (purple): All findings compiled into comprehensive recommendation

---

## State Machine Diagram

This diagram shows all possible states and transitions in the workflow.

```mermaid
stateDiagram-v2
    [*] --> START
    
    START --> InitialResearch: Begin workflow
    
    InitialResearch --> Analyze: Findings gathered
    
    state AnalysisLoop {
        Analyze --> CheckDecision: Analysis complete
        
        CheckDecision --> AdditionalResearch: NEED_MORE_RESEARCH
        CheckDecision --> FinalReport: READY_TO_CONCLUDE
        CheckDecision --> FinalReport: iteration >= 3
        
        AdditionalResearch --> Analyze: New findings added
    }
    
    FinalReport --> END: Report generated
    
    END --> [*]
    
    note right of InitialResearch
        Uses all 5 research tools
        Up to 5 tool calls
    end note
    
    note right of Analyze
        Reviews findings
        Identifies gaps
        Decides next step
    end note
    
    note right of AdditionalResearch
        Targeted searches
        Up to 3 tool calls
    end note
    
    note right of FinalReport
        Compiles all research
        Generates recommendation
    end note
```

### Description

The state machine shows:

- **START**: Entry point
- **InitialResearch**: Comprehensive information gathering
- **Analyze**: Analyst evaluation state
- **CheckDecision**: Decision point based on analyst's assessment
- **AdditionalResearch**: Targeted follow-up research
- **FinalReport**: Report generation state
- **END**: Terminal state

Key transitions:
- `NEED_MORE_RESEARCH` triggers additional research loop
- `READY_TO_CONCLUDE` proceeds to final report
- Maximum 3 iterations enforced as safety limit

---

## Workflow Flowchart

This diagram provides a detailed view of the complete workflow logic.

```mermaid
flowchart TD
    subgraph Input["📥 Input"]
        A[/"👤 User Query<br/>(Technical Decision Question)"/]
    end
    
    subgraph Init["🔧 Initialization"]
        B["Create MultiAgentState<br/>───────────────<br/>• messages: []<br/>• user_query: query<br/>• research_findings: []<br/>• analysis_notes: []<br/>• iteration_count: 0<br/>• current_agent: 'researcher'"]
    end
    
    subgraph InitialResearch["🔬 Initial Research Phase"]
        C["Researcher Agent"]
        C1["Bind LLM with tools"]
        C2["Execute research prompt"]
        C3{"Tool calls<br/>in response?"}
        C4["Execute tool<br/>(max 5 calls)"]
        C5["Accumulate findings"]
        C6["Generate summary"]
        
        C --> C1 --> C2 --> C3
        C3 -->|Yes| C4
        C4 --> C5 --> C3
        C3 -->|No/Max reached| C6
    end
    
    subgraph Analysis["📊 Analysis Phase"]
        D["Analyst Agent"]
        D1["Compile all findings"]
        D2["Analyze trade-offs"]
        D3["Identify gaps"]
        D4{"Decision?"}
        D5["Mark: NEED_MORE_RESEARCH"]
        D6["Mark: READY_TO_CONCLUDE"]
        
        D --> D1 --> D2 --> D3 --> D4
        D4 -->|"Gaps found"| D5
        D4 -->|"Sufficient data"| D6
    end
    
    subgraph AdditionalResearch["🔄 Additional Research"]
        E["Extract analyst request"]
        E1["Targeted tool calls<br/>(max 3)"]
        E2["Summarize new findings"]
        E3["Increment iteration"]
    end
    
    subgraph Routing["⚡ Routing Logic"]
        R{"Check<br/>condition"}
        R1["iteration >= 3?"]
        R2["current_agent?"]
    end
    
    subgraph FinalReport["📄 Final Report"]
        F["Compile all research"]
        F1["Compile all analysis"]
        F2["Generate structured report:<br/>───────────────<br/>1. Executive Summary<br/>2. Technology Comparison<br/>3. Performance Analysis<br/>4. Cost Analysis<br/>5. Integration Considerations<br/>6. Risk Assessment<br/>7. Final Recommendation<br/>8. Next Steps"]
    end
    
    subgraph Output["📤 Output"]
        G[/"📊 Final Recommendation<br/>Report"/]
    end
    
    A --> B
    B --> C
    C6 --> D
    D5 --> R
    D6 --> R
    R --> R1
    R1 -->|Yes| F
    R1 -->|No| R2
    R2 -->|"researcher"| E
    R2 -->|"final"| F
    E --> E1 --> E2 --> E3
    E3 --> D
    F --> F1 --> F2
    F2 --> G
    
    style InitialResearch fill:#e3f2fd
    style Analysis fill:#fff3e0
    style AdditionalResearch fill:#e8f5e9
    style FinalReport fill:#f3e5f5
```

### Description

The workflow flowchart shows:

1. **Input**: User provides a technical decision query
2. **Initialization**: MultiAgentState created with all tracking fields
3. **Initial Research**: Comprehensive gathering using all 5 tools
4. **Analysis**: Analyst evaluates and decides on next action
5. **Routing**: Conditional logic for iteration limits and agent decisions
6. **Additional Research**: Targeted searches to fill identified gaps
7. **Final Report**: Structured recommendation with 8 sections

---

## Data Flow Diagram

This diagram shows how data transforms through the system.

```mermaid
flowchart LR
    subgraph Input
        Q["User Query<br/>(string)"]
    end
    
    subgraph S1["State After Init"]
        SI["messages: []<br/>query: input<br/>findings: []<br/>notes: []<br/>iteration: 0<br/>agent: 'researcher'"]
    end
    
    subgraph S2["State After Initial Research"]
        SR["messages: [System, Human, AI]<br/>query: input<br/>findings: ['tech_docs', 'papers',<br/>'benchmarks', 'examples', 'costs']<br/>notes: []<br/>iteration: 1<br/>agent: 'analyst'"]
    end
    
    subgraph S3["State After Analysis"]
        SA["messages: [+Analysis]<br/>query: input<br/>findings: [...]<br/>notes: ['analysis_1']<br/>iteration: 1<br/>agent: 'researcher' | 'final'"]
    end
    
    subgraph S4["State After Additional Research"]
        SAR["messages: [+New findings]<br/>query: input<br/>findings: [..., 'additional']<br/>notes: ['analysis_1']<br/>iteration: 2<br/>agent: 'analyst'"]
    end
    
    subgraph S5["State After Final Report"]
        SF["messages: [...]<br/>query: input<br/>findings: [all]<br/>notes: [all]<br/>iteration: n<br/>agent: 'complete'<br/>final_recommendation: 'report'"]
    end
    
    subgraph Output
        R["Final Report<br/>(markdown string)"]
    end
    
    Q --> SI
    SI -->|"initial_research()"| SR
    SR -->|"analyze_research()"| SA
    SA -->|"additional_research()<br/>(if needed)"| SAR
    SAR -->|"analyze_research()"| SA
    SA -->|"generate_final_report()"| SF
    SF --> R
```

### Description

The data flow diagram traces how `MultiAgentState` evolves:

1. **After Init**: Empty state with only query populated
2. **After Initial Research**: Findings populated from 5 tool types, agent switches to analyst
3. **After Analysis**: Analysis notes added, agent may switch back to researcher or to final
4. **After Additional Research**: New findings appended, iteration incremented
5. **After Final Report**: Complete state with final_recommendation

---

## Tool Architecture Diagram

This diagram details the research toolkit and how each tool specializes.

```mermaid
flowchart TD
    subgraph ResearcherAgent["🔬 Researcher Agent"]
        LLM["LLM with Tools Bound"]
    end
    
    subgraph ResearchToolkit["📦 Research Toolkit"]
        T1["📚 search_technical_docs<br/>──────────────────<br/>Query modifier:<br/>'technical documentation<br/>guide tutorial'"]
        
        T2["📄 search_academic_papers<br/>──────────────────<br/>Query modifier:<br/>'research paper arxiv<br/>academic study'"]
        
        T3["📊 search_benchmarks<br/>──────────────────<br/>Query modifier:<br/>'benchmark performance<br/>comparison metrics'"]
        
        T4["💻 search_implementation_examples<br/>──────────────────<br/>Query modifier:<br/>'implementation example<br/>code github tutorial'"]
        
        T5["💰 search_cost_analysis<br/>──────────────────<br/>Query modifier:<br/>'cost pricing analysis<br/>resources requirements'"]
    end
    
    subgraph TavilyConfig["⚙️ Tavily Configuration"]
        TC["search_depth: 'advanced'<br/>max_results: 5<br/>include_answer: True"]
    end
    
    subgraph TavilyAPI["🌐 Tavily API"]
        TA["Web Search Service"]
    end
    
    subgraph ResponseFormat["📋 Response Format"]
        RF1["📚 Summary: AI answer<br/>───────────────<br/>Technical Documentation Found:<br/>1. Title, URL, Content<br/>2. ..."]
        
        RF2["📄 Research Summary: AI answer<br/>───────────────<br/>Academic Papers & Research:<br/>1. Title, URL, Content<br/>2. ..."]
        
        RF3["📊 Benchmark Summary: AI answer<br/>───────────────<br/>Benchmarks & Performance Data:<br/>1. Title, URL, Content<br/>2. ..."]
        
        RF4["💻 Implementation Summary: AI answer<br/>───────────────<br/>Implementation Examples:<br/>1. Title, URL, Content<br/>2. ..."]
        
        RF5["💰 Cost Summary: AI answer<br/>───────────────<br/>Cost & Resource Analysis:<br/>1. Title, URL, Content<br/>2. ..."]
    end
    
    LLM --> T1 & T2 & T3 & T4 & T5
    T1 & T2 & T3 & T4 & T5 --> TC
    TC --> TA
    
    T1 -.-> RF1
    T2 -.-> RF2
    T3 -.-> RF3
    T4 -.-> RF4
    T5 -.-> RF5
    
    style T1 fill:#e3f2fd
    style T2 fill:#fff8e1
    style T3 fill:#e8f5e9
    style T4 fill:#fce4ec
    style T5 fill:#f3e5f5
```

### Description

The tool architecture shows the 5 specialized research tools:

| Tool | Purpose | Query Modifiers |
|------|---------|-----------------|
| `search_technical_docs` | Official docs, guides | "technical documentation guide tutorial" |
| `search_academic_papers` | Research, studies | "research paper arxiv academic study" |
| `search_benchmarks` | Performance metrics | "benchmark performance comparison metrics" |
| `search_implementation_examples` | Code samples | "implementation example code github" |
| `search_cost_analysis` | Pricing, resources | "cost pricing analysis resources" |

All tools share:
- Tavily advanced search depth
- 5 results maximum
- AI-generated answer included

---

## Iterative Loop Flowchart

This diagram details the analyst decision logic and iteration control.

```mermaid
flowchart TD
    START((Analyze<br/>Start)) --> A["Get all research<br/>findings"]
    
    A --> B["Compile findings<br/>into analysis prompt"]
    
    B --> C["Analyst LLM<br/>processes findings"]
    
    C --> D["Generate analysis<br/>with decision"]
    
    D --> E{"Contains<br/>'NEED_MORE_RESEARCH'?"}
    
    E -->|Yes| F["Set current_agent<br/>= 'researcher'"]
    E -->|No| G["Set current_agent<br/>= 'final'"]
    
    F --> H["Add analysis<br/>to notes"]
    G --> H
    
    H --> I{"Route<br/>Decision"}
    
    I --> J{"iteration_count<br/>>= 3?"}
    
    J -->|Yes| K["Force: 'final_report'<br/>⏹️ Max iterations"]
    
    J -->|No| L{"current_agent<br/>== 'researcher'?"}
    
    L -->|Yes| M["Route: 'additional_research'<br/>🔄 More research needed"]
    
    L -->|No| N["Route: 'final_report'<br/>✅ Ready to conclude"]
    
    K --> END1((To Final<br/>Report))
    M --> END2((To Additional<br/>Research))
    N --> END1
    
    style E fill:#fff3e0
    style J fill:#ffebee
    style L fill:#e8f5e9
    style M fill:#e3f2fd
    style N fill:#f3e5f5
    style K fill:#ffcdd2
```

### Description

The iterative loop flowchart shows the analyst's decision process:

1. **Compile Findings**: Gather all research data
2. **LLM Analysis**: Analyst evaluates trade-offs and gaps
3. **Decision Detection**: Check for "NEED_MORE_RESEARCH" in response
4. **State Update**: Set `current_agent` based on decision
5. **Iteration Check**: Enforce maximum 3 iterations
6. **Routing**: Direct to additional research or final report

Safety mechanisms:
- Maximum 3 research-analysis cycles
- Clear decision keywords for routing
- Graceful fallback to final report

---

## Agent Communication Protocol

This diagram shows how agents communicate through state.

```mermaid
sequenceDiagram
    participant State as MultiAgentState
    participant R as 🔬 Researcher
    participant A as 📊 Analyst
    
    Note over State: Initial State
    
    R->>State: Update research_findings[]
    R->>State: Set current_agent = "analyst"
    
    State->>A: Pass accumulated findings
    
    A->>A: Analyze findings
    
    alt Gaps identified
        A->>State: Add to analysis_notes[]
        A->>State: Set current_agent = "researcher"
        A->>State: (Implicit) research_requests in notes
        
        State->>R: Pass analysis with request
        
        R->>R: Parse what's needed
        R->>State: Append to research_findings[]
        R->>State: Set current_agent = "analyst"
        
        State->>A: Pass updated findings
    else Sufficient data
        A->>State: Add final analysis to notes
        A->>State: Set current_agent = "final"
    end
    
    Note over State: Final State Ready
```

### Description

Agents communicate indirectly through state:

- **Researcher → Analyst**: Via `research_findings` list
- **Analyst → Researcher**: Via `analysis_notes` (containing research requests)
- **Coordination**: Via `current_agent` field

This decoupled design enables:
- Clear separation of concerns
- Traceable communication history
- Easy extension with additional agents

---

## Summary

This documentation covers the complete architecture of the Multi-Agent Research & Analysis System:

| Diagram Type | Purpose |
|-------------|---------|
| Class Diagram | Data structures and type relationships |
| Component Diagram | High-level system organization |
| Agent Interaction | Collaboration between agents |
| Sequence Diagram | Complete temporal flow with iterations |
| State Machine | All states and valid transitions |
| Workflow Flowchart | Detailed execution logic |
| Data Flow | State transformations through phases |
| Tool Architecture | Specialized research tool details |
| Iterative Loop | Analyst decision and routing logic |
| Communication Protocol | Inter-agent state-based messaging |

### Key Architecture Decisions

1. **Dual-Agent Design**: Separation of research (information gathering) and analysis (critical evaluation)
2. **Specialized Tools**: 5 tools targeting different information types for comprehensive coverage
3. **Iterative Refinement**: Analyst can request additional research to fill gaps
4. **State-Based Communication**: Agents communicate through shared state rather than direct messaging
5. **Bounded Iteration**: Maximum 3 cycles prevents infinite loops
6. **Dual LLM Configuration**: Standard temperature for creativity, lower temperature for structured outputs

### Use Cases

- Technology adoption decisions (e.g., GraphRAG vs Traditional RAG)
- Framework comparison (e.g., LangChain vs LlamaIndex)
- Infrastructure selection (e.g., Vector database comparison)
- Any technical decision requiring research and analysis

### Extension Points

- Add more specialized search tools (e.g., GitHub code search, patent search)
- Implement memory for multi-session research projects
- Add fact-checking/verification agent
- Create domain-specific analyst variants
- Integrate with internal knowledge bases
- Add confidence scoring to recommendations
