# Design Patterns for an Agent Coordinator Communicating With Specialized Worker Agents via MCP - Architecture Documentation

This document provides comprehensive UML diagrams and flowcharts describing the architecture and workflow of the Multi-Agent System of a Coordinator and Specialized Workers using Model Context Protocol (MCP) for inter-agent communication, resource sharing, and workflow orchestration.

---

## Table of Contents

1. [Overview](#overview)
2. [Class Diagram](#class-diagram)
3. [Component Diagram](#component-diagram)
4. [MessageBus Architecture](#messagebus-architecture)
5. [Agent Hierarchy Diagram](#agent-hierarchy-diagram)
6. [Sequence Diagram - Sequential Workflow](#sequence-diagram---sequential-workflow)
7. [Sequence Diagram - Parallel Execution](#sequence-diagram---parallel-execution)
8. [Resource Sharing Flowchart](#resource-sharing-flowchart)
9. [Coordinator Orchestration Flowchart](#coordinator-orchestration-flowchart)
10. [Agent Discovery Diagram](#agent-discovery-diagram)
11. [Tool Registry Pattern](#tool-registry-pattern)
12. [End-to-End Data Pipeline](#end-to-end-data-pipeline)

---

## Overview

This system demonstrates a comprehensive multi-agent architecture using the **Model Context Protocol (MCP)** for standardized communication between AI agents.

### MCP Protocol Benefits

| Benefit | Description |
|---------|-------------|
| **Standardized Communication** | All agents use the same protocol |
| **Resource Sharing** | Agents can share data, tools, and context |
| **Loose Coupling** | Agents don't need to know internals of others |
| **Discoverability** | Agents can discover capabilities of other agents |
| **Scalability** | Easy to add new agents without changes |
| **Interoperability** | Different AI systems can work together |

### System Components

| Component | Role | Description |
|-----------|------|-------------|
| **MessageBus** | Communication Hub | Pub/sub messaging and resource storage |
| **MCPAgent** | Base Agent | Client/server hybrid with MCP capabilities |
| **DataCollectorAgent** | Worker | Web scraping, API calls, data extraction |
| **AnalysisAgent** | Worker | Statistical analysis, pattern recognition |
| **ReportGeneratorAgent** | Worker | Report writing, visualization |
| **ValidationAgent** | Worker | Quality checks, verification |
| **MCPCoordinator** | Orchestrator | Workflow management and execution |
| **MCPToolRegistry** | Service | Tool registration and discovery |

---

## Class Diagram

This diagram shows all classes, their attributes, methods, and relationships.

```mermaid
classDiagram
    class MessageBus {
        +List~Dict~ messages
        +Dict~str,Any~ resources
        +publish(agent_id, type, content) str
        +subscribe(agent_id, type?) List~Dict~
        +store_resource(resource_id, data)
        +get_resource(resource_id) Any?
    }

    class MCPAgent {
        <<abstract>>
        +str agent_id
        +str role
        +List~str~ capabilities
        +MessageBus message_bus
        +expose_tool(name, func) Tool
        +publish_result(type, data) str
        +get_messages(type?) List~Dict~
        +share_resource(name, data) str
        +access_resource(resource_id) Any?
        +process_task(task)* Dict
    }

    class DataCollectorAgent {
        +agent_id = "data_collector"
        +role = "Data Collection"
        +capabilities = [web_scraping, api_calls, data_extraction]
        +process_task(task) Dict
    }

    class AnalysisAgent {
        +agent_id = "analyzer"
        +role = "Data Analysis"
        +capabilities = [statistical_analysis, pattern_recognition, insights]
        +process_task(task) Dict
    }

    class ReportGeneratorAgent {
        +agent_id = "report_generator"
        +role = "Report Generation"
        +capabilities = [report_writing, visualization, documentation]
        +process_task(task) Dict
    }

    class ValidationAgent {
        +agent_id = "validator"
        +role = "Validation"
        +capabilities = [quality_check, verification, compliance]
        +process_task(task) Dict
    }

    class MCPCoordinator {
        +Dict~str,MCPAgent~ agents
        +MessageBus message_bus
        +register_agent(agent)
        +execute_workflow(task, sequence) Dict
        +get_message_history() List~Dict~
        +get_shared_resources() Dict
        +display_communication_log()
    }

    class MCPToolRegistry {
        +Dict~str,Dict~ tools
        +register_tool(agent_id, name, func, desc)
        +discover_tools(capability?) List~Dict~
        +call_tool(tool_id, args) Any
    }

    class Tool {
        <<MCP Type>>
        +str name
        +str description
        +Dict inputSchema
    }

    class Resource {
        <<MCP Type>>
        +str uri
        +str name
        +str mimeType
    }

    MCPAgent <|-- DataCollectorAgent
    MCPAgent <|-- AnalysisAgent
    MCPAgent <|-- ReportGeneratorAgent
    MCPAgent <|-- ValidationAgent

    MCPAgent --> MessageBus : uses
    MCPCoordinator --> MCPAgent : manages
    MCPCoordinator --> MessageBus : monitors
    MCPAgent ..> Tool : exposes
    MCPAgent ..> Resource : shares
    MCPToolRegistry --> Tool : registers
```

### Description

The class diagram shows:

- **MessageBus**: Central pub/sub system with resource storage
- **MCPAgent**: Abstract base class defining the MCP agent interface
- **Specialized Agents**: Four concrete implementations with specific capabilities
- **MCPCoordinator**: Orchestrator managing agent registration and workflow execution
- **MCPToolRegistry**: Service for tool registration and discovery
- **MCP Types**: Tool and Resource from the MCP protocol specification

---

## Component Diagram

This diagram shows the high-level system architecture.

```mermaid
flowchart TB
    subgraph UserLayer["👤 User Layer"]
        USER["User Request"]
    end

    subgraph CoordinatorLayer["🎯 Coordinator Layer"]
        COORD["MCPCoordinator<br/>─────────────<br/>• Agent Registry<br/>• Workflow Execution<br/>• Communication Log"]
    end

    subgraph CommunicationLayer["📡 MCP Communication Layer"]
        BUS["MessageBus<br/>─────────────<br/>• Pub/Sub Messaging<br/>• Resource Storage<br/>• Message History"]
        
        REGISTRY["MCPToolRegistry<br/>─────────────<br/>• Tool Registration<br/>• Tool Discovery<br/>• Tool Invocation"]
    end

    subgraph AgentLayer["🤖 Agent Layer"]
        DC["DataCollectorAgent<br/>─────────────<br/>📊 Data Collection"]
        AN["AnalysisAgent<br/>─────────────<br/>📈 Data Analysis"]
        RG["ReportGeneratorAgent<br/>─────────────<br/>📝 Report Generation"]
        VA["ValidationAgent<br/>─────────────<br/>✅ Validation"]
    end

    subgraph OutputLayer["📤 Output Layer"]
        RES["Shared Resources"]
        MSG["Message History"]
        REP["Final Reports"]
    end

    USER --> COORD
    COORD --> DC & AN & RG & VA
    DC & AN & RG & VA <--> BUS
    DC & AN & RG & VA <--> REGISTRY
    BUS --> RES & MSG
    VA --> REP

    style CoordinatorLayer fill:#e3f2fd
    style CommunicationLayer fill:#fff3e0
    style AgentLayer fill:#e8f5e9
```

### Description

The system is organized into layers:

- **User Layer**: Entry point for workflow requests
- **Coordinator Layer**: Manages agent registration and workflow execution
- **MCP Communication Layer**: MessageBus for messaging, ToolRegistry for tool discovery
- **Agent Layer**: Four specialized agents with specific capabilities
- **Output Layer**: Shared resources, message logs, and final reports

---

## MessageBus Architecture

This diagram details the MessageBus publish/subscribe and resource sharing mechanisms.

```mermaid
flowchart TD
    subgraph Publishers["📤 Publishers"]
        P1["Agent A<br/>publish()"]
        P2["Agent B<br/>publish()"]
        P3["Agent C<br/>publish()"]
    end

    subgraph MessageBus["📡 MessageBus"]
        subgraph MessageStore["Message Store"]
            MS["messages: List[Dict]<br/>─────────────────<br/>{id, agent_id, type,<br/>content, timestamp}"]
        end

        subgraph ResourceStore["Resource Store"]
            RS["resources: Dict[str, Any]<br/>─────────────────<br/>{resource_id: data}"]
        end

        PUB["publish()<br/>─────────────────<br/>• Create message<br/>• Assign UUID<br/>• Add timestamp<br/>• Append to list"]

        SUB["subscribe()<br/>─────────────────<br/>• Filter by type<br/>• Exclude self<br/>• Return matches"]

        STORE["store_resource()<br/>─────────────────<br/>• Store by ID<br/>• Print confirmation"]

        GET["get_resource()<br/>─────────────────<br/>• Lookup by ID<br/>• Return or None"]
    end

    subgraph Subscribers["📥 Subscribers"]
        S1["Agent A<br/>subscribe()"]
        S2["Agent B<br/>subscribe()"]
        S3["Agent C<br/>subscribe()"]
    end

    P1 & P2 & P3 --> PUB
    PUB --> MS
    MS --> SUB
    SUB --> S1 & S2 & S3

    P1 & P2 & P3 --> STORE
    STORE --> RS
    RS --> GET
    GET --> S1 & S2 & S3

    style MessageStore fill:#e3f2fd
    style ResourceStore fill:#e8f5e9
```

### Description

The MessageBus provides two key services:

**Pub/Sub Messaging:**
- `publish()`: Create message with UUID, timestamp, and content
- `subscribe()`: Filter messages by type, exclude sender's own messages

**Resource Sharing:**
- `store_resource()`: Store data with unique ID
- `get_resource()`: Retrieve data by ID

---

## Agent Hierarchy Diagram

This diagram shows the inheritance hierarchy and capabilities of each agent.

```mermaid
flowchart TD
    subgraph BaseClass["🔷 Base Class"]
        MCP["MCPAgent<br/>─────────────<br/>• agent_id<br/>• role<br/>• capabilities<br/>• message_bus<br/>─────────────<br/>• expose_tool()<br/>• publish_result()<br/>• get_messages()<br/>• share_resource()<br/>• access_resource()<br/>• process_task()*"]
    end

    subgraph Implementations["🔶 Specialized Implementations"]
        DC["DataCollectorAgent<br/>─────────────<br/>🔍 Role: Data Collection<br/>─────────────<br/>Capabilities:<br/>• web_scraping<br/>• api_calls<br/>• data_extraction<br/>─────────────<br/>Outputs:<br/>• collected_data resource<br/>• data_collected message"]

        AN["AnalysisAgent<br/>─────────────<br/>📊 Role: Data Analysis<br/>─────────────<br/>Capabilities:<br/>• statistical_analysis<br/>• pattern_recognition<br/>• insights<br/>─────────────<br/>Outputs:<br/>• analysis_results resource<br/>• analysis_complete message"]

        RG["ReportGeneratorAgent<br/>─────────────<br/>📝 Role: Report Generation<br/>─────────────<br/>Capabilities:<br/>• report_writing<br/>• visualization<br/>• documentation<br/>─────────────<br/>Outputs:<br/>• final_report resource<br/>• report_generated message"]

        VA["ValidationAgent<br/>─────────────<br/>✅ Role: Validation<br/>─────────────<br/>Capabilities:<br/>• quality_check<br/>• verification<br/>• compliance<br/>─────────────<br/>Outputs:<br/>• validation_complete message"]
    end

    MCP --> DC
    MCP --> AN
    MCP --> RG
    MCP --> VA

    style BaseClass fill:#e3f2fd
    style DC fill:#fff3e0
    style AN fill:#e8f5e9
    style RG fill:#f3e5f5
    style VA fill:#ffebee
```

### Description

Each agent extends MCPAgent and specializes in a specific domain:

| Agent | Input | Output |
|-------|-------|--------|
| **DataCollector** | Task description | Raw data resource |
| **Analysis** | Data resource | Statistics + insights |
| **ReportGenerator** | Analysis resource | Formatted report |
| **Validation** | All messages | Validation status |

---

## Sequence Diagram - Sequential Workflow

This diagram shows the complete sequential workflow execution.

```mermaid
sequenceDiagram
    autonumber
    participant User
    participant Coord as MCPCoordinator
    participant Bus as MessageBus
    participant DC as DataCollectorAgent
    participant AN as AnalysisAgent
    participant RG as ReportGeneratorAgent
    participant VA as ValidationAgent

    User->>Coord: execute_workflow(task, sequence)
    
    Note over Coord: Start workflow logging
    
    rect rgb(255, 243, 224)
        Note over DC: Step 1: Data Collection
        Coord->>DC: process_task(task)
        DC->>DC: Simulate data collection
        DC->>Bus: store_resource("collected_data", data)
        Bus-->>DC: resource_id
        DC->>Bus: publish("data_collected", {resource_id})
        DC-->>Coord: Return data
    end

    rect rgb(232, 245, 233)
        Note over AN: Step 2: Analysis
        Coord->>AN: process_task(task)
        AN->>Bus: subscribe("data_collected")
        Bus-->>AN: messages with resource_id
        AN->>Bus: get_resource(resource_id)
        Bus-->>AN: collected data
        AN->>AN: Calculate statistics
        AN->>Bus: store_resource("analysis_results", analysis)
        AN->>Bus: publish("analysis_complete", {resource_id})
        AN-->>Coord: Return analysis
    end

    rect rgb(243, 229, 245)
        Note over RG: Step 3: Report Generation
        Coord->>RG: process_task(task)
        RG->>Bus: subscribe("analysis_complete")
        Bus-->>RG: messages with resource_id
        RG->>Bus: get_resource(resource_id)
        Bus-->>RG: analysis results
        RG->>RG: Generate report
        RG->>Bus: store_resource("final_report", report)
        RG->>Bus: publish("report_generated", {resource_id})
        RG-->>Coord: Return report
    end

    rect rgb(255, 235, 238)
        Note over VA: Step 4: Validation
        Coord->>VA: process_task(task)
        VA->>Bus: subscribe() - all messages
        Bus-->>VA: all workflow messages
        VA->>VA: Validate each step
        VA->>Bus: publish("validation_complete", results)
        VA-->>Coord: Return validation
    end

    Coord-->>User: {all_results}
```

### Description

The sequential workflow executes agents in order:

1. **Data Collection**: Collects data, shares via resource, publishes completion
2. **Analysis**: Subscribes to data_collected, retrieves resource, analyzes, shares results
3. **Report Generation**: Subscribes to analysis_complete, generates formatted report
4. **Validation**: Reviews all messages to validate workflow completeness

---

## Sequence Diagram - Parallel Execution

This diagram shows multiple agents executing in parallel.

```mermaid
sequenceDiagram
    autonumber
    participant Main as Main Process
    participant Bus as MessageBus
    participant DC1 as DataCollector_1
    participant DC2 as DataCollector_2

    Main->>Main: Create parallel tasks

    par Parallel Execution
        Main->>DC1: process_task("Dataset A")
        DC1->>DC1: Collect data A
        DC1->>Bus: store_resource("dc1_collected_data")
        DC1->>Bus: publish("data_collected", {dataset: "A"})
    and
        Main->>DC2: process_task("Dataset B")
        DC2->>DC2: Collect data B
        DC2->>Bus: store_resource("dc2_collected_data")
        DC2->>Bus: publish("data_collected", {dataset: "B"})
    end

    Note over Main: asyncio.gather() completes
    
    Main->>Bus: Get all resources
    Bus-->>Main: {dc1_collected_data, dc2_collected_data}
    
    Main->>Main: Combine results
```

### Description

MCP enables parallel agent execution:

- Multiple agents run concurrently via `asyncio.gather()`
- Each agent publishes to the shared MessageBus
- Results are combined after all agents complete
- Enables horizontal scaling of workloads

---

## Resource Sharing Flowchart

This diagram details how agents share resources via MCP.

```mermaid
flowchart TD
    subgraph ProducerAgent["🔷 Producer Agent"]
        P1["Process data"]
        P2["Create resource<br/>resource_id = agent_id + name"]
        P3["store_resource(id, data)"]
        P4["publish('resource_shared',<br/>{resource_id})"]
    end

    subgraph MessageBus["📡 MessageBus"]
        RS["resources: Dict"]
        MS["messages: List"]
    end

    subgraph ConsumerAgent["🔶 Consumer Agent"]
        C1["subscribe('resource_shared')"]
        C2["Extract resource_id<br/>from message"]
        C3["access_resource(resource_id)"]
        C4["Use data"]
    end

    P1 --> P2
    P2 --> P3
    P3 --> RS
    P3 --> P4
    P4 --> MS

    MS --> C1
    C1 --> C2
    C2 --> C3
    RS --> C3
    C3 --> C4

    style ProducerAgent fill:#e3f2fd
    style ConsumerAgent fill:#e8f5e9
    style MessageBus fill:#fff3e0
```

### Description

Resource sharing follows this pattern:

**Producer Side:**
1. Process data to be shared
2. Generate unique resource_id (agent_id + resource_name)
3. Store in MessageBus resources dict
4. Publish notification with resource_id

**Consumer Side:**
1. Subscribe to resource_shared messages
2. Extract resource_id from message content
3. Access resource from MessageBus
4. Use the retrieved data

---

## Coordinator Orchestration Flowchart

This diagram shows how the MCPCoordinator manages workflows.

```mermaid
flowchart TD
    START((Start)) --> A["Receive task and<br/>agent_sequence"]
    
    A --> B["Log workflow start"]
    
    B --> C["Initialize results = {}"]
    
    C --> D{"More agents<br/>in sequence?"}
    
    D -->|No| J["Log workflow complete"]
    
    D -->|Yes| E["Get next agent_id"]
    
    E --> F{"agent_id in<br/>registered agents?"}
    
    F -->|No| G["Log: Agent not found"]
    G --> D
    
    F -->|Yes| H["Get agent instance"]
    
    H --> I["await agent.process_task(task)"]
    
    I --> K["Store result in results[agent_id]"]
    
    K --> L["await asyncio.sleep(0.3)"]
    
    L --> D
    
    J --> M["Return results dict"]
    
    M --> END((End))

    style D fill:#fff3e0
    style F fill:#e3f2fd
    style I fill:#e8f5e9
```

### Description

The MCPCoordinator orchestration logic:

1. **Receive Request**: Task description and agent sequence
2. **Initialize**: Empty results dictionary
3. **Iterate Agents**: Process each agent in sequence
4. **Validate**: Check if agent is registered
5. **Execute**: Call agent's `process_task()` asynchronously
6. **Store**: Save result keyed by agent_id
7. **Delay**: Brief pause between agents
8. **Return**: Complete results dictionary

---

## Agent Discovery Diagram

This diagram shows how agents discover each other via MCP.

```mermaid
flowchart TD
    subgraph Registration["📝 Agent Registration"]
        R1["Agent created with<br/>agent_id, role, capabilities"]
        R2["__post_init__() called"]
        R3["publish('agent_registered',<br/>{role, capabilities})"]
    end

    subgraph MessageBus["📡 MessageBus"]
        MS["messages: [<br/>  {type: 'agent_registered',<br/>   agent_id: 'data_collector',<br/>   content: {role, capabilities}},<br/>  {type: 'agent_registered',<br/>   agent_id: 'analyzer',<br/>   content: {...}},<br/>  ...<br/>]"]
    end

    subgraph Discovery["🔍 Agent Discovery"]
        D1["Filter messages where<br/>type == 'agent_registered'"]
        D2["Extract agent info:<br/>• agent_id<br/>• role<br/>• capabilities"]
        D3["Display discovered agents"]
    end

    subgraph DiscoveredAgents["🤖 Discovered Agents"]
        DA1["data_collector<br/>Role: Data Collection<br/>web_scraping, api_calls, data_extraction"]
        DA2["analyzer<br/>Role: Data Analysis<br/>statistical_analysis, pattern_recognition"]
        DA3["report_generator<br/>Role: Report Generation<br/>report_writing, visualization"]
        DA4["validator<br/>Role: Validation<br/>quality_check, verification"]
    end

    R1 --> R2 --> R3
    R3 --> MS
    MS --> D1
    D1 --> D2
    D2 --> D3
    D3 --> DA1 & DA2 & DA3 & DA4

    style Registration fill:#e3f2fd
    style Discovery fill:#e8f5e9
    style DiscoveredAgents fill:#fff3e0
```

### Description

Agent discovery workflow:

**Registration (Automatic):**
- Each agent publishes `agent_registered` message on creation
- Message contains role and capabilities list

**Discovery:**
- Query MessageBus for `agent_registered` messages
- Extract agent details from each message
- Build registry of available agents and their capabilities

This enables dynamic agent composition without hardcoding dependencies.

---

## Tool Registry Pattern

This diagram shows the MCPToolRegistry for tool management.

```mermaid
flowchart TD
    subgraph ToolRegistration["🔧 Tool Registration"]
        TR1["Agent defines function"]
        TR2["register_tool(<br/>  agent_id,<br/>  tool_name,<br/>  tool_func,<br/>  description<br/>)"]
        TR3["Store in tools dict<br/>key = 'agent_id.tool_name'"]
    end

    subgraph MCPToolRegistry["📦 MCPToolRegistry"]
        TOOLS["tools: Dict[str, Dict]<br/>─────────────────────<br/>{<br/>  'analyzer.calculate_mean': {<br/>    agent_id, name,<br/>    function, description<br/>  },<br/>  'report_gen.format_report': {<br/>    ...<br/>  }<br/>}"]
    end

    subgraph ToolDiscovery["🔍 Tool Discovery"]
        TD1["discover_tools(capability?)"]
        TD2{"capability<br/>provided?"}
        TD3["Filter by description"]
        TD4["Return all tools"]
        TD5["Return List[Dict]"]
    end

    subgraph ToolInvocation["⚡ Tool Invocation"]
        TI1["call_tool(tool_id, *args, **kwargs)"]
        TI2{"tool_id in<br/>tools?"}
        TI3["Execute function"]
        TI4["Raise ValueError"]
        TI5["Return result"]
    end

    TR1 --> TR2 --> TR3
    TR3 --> TOOLS

    TOOLS --> TD1
    TD1 --> TD2
    TD2 -->|Yes| TD3 --> TD5
    TD2 -->|No| TD4 --> TD5

    TOOLS --> TI1
    TI1 --> TI2
    TI2 -->|Yes| TI3 --> TI5
    TI2 -->|No| TI4

    style ToolRegistration fill:#e3f2fd
    style ToolDiscovery fill:#e8f5e9
    style ToolInvocation fill:#fff3e0
```

### Description

The MCPToolRegistry provides:

**Registration:**
- Tools registered with unique ID: `{agent_id}.{tool_name}`
- Stores function reference and description

**Discovery:**
- `discover_tools()`: List all available tools
- `discover_tools(capability)`: Filter by description keyword

**Invocation:**
- `call_tool(tool_id, args)`: Execute registered function
- Raises error if tool not found

---

## End-to-End Data Pipeline

This diagram shows a complete workflow example with all components.

```mermaid
flowchart TD
    subgraph Input["📥 Input"]
        USER["User Request:<br/>'Quarterly Sales Analysis'"]
    end

    subgraph Coordinator["🎯 Coordinator"]
        INIT["Initialize workflow<br/>agent_sequence = [<br/>  data_collector,<br/>  analyzer,<br/>  report_generator,<br/>  validator<br/>]"]
    end

    subgraph Step1["Step 1: Data Collection"]
        DC["DataCollectorAgent"]
        DC_OUT["Output:<br/>─────────────<br/>Resource: data_collector_collected_data<br/>Data: {data_points: [10,25,30,45,50]}<br/>Message: data_collected"]
    end

    subgraph Step2["Step 2: Analysis"]
        AN["AnalysisAgent"]
        AN_IN["Input:<br/>─────────────<br/>Subscribe: data_collected<br/>Resource: collected_data"]
        AN_OUT["Output:<br/>─────────────<br/>Resource: analyzer_analysis_results<br/>Data: {mean: 32, max: 50, min: 10}<br/>Message: analysis_complete"]
    end

    subgraph Step3["Step 3: Report Generation"]
        RG["ReportGeneratorAgent"]
        RG_IN["Input:<br/>─────────────<br/>Subscribe: analysis_complete<br/>Resource: analysis_results"]
        RG_OUT["Output:<br/>─────────────<br/>Resource: report_generator_final_report<br/>Data: {title, summary, details}<br/>Message: report_generated"]
    end

    subgraph Step4["Step 4: Validation"]
        VA["ValidationAgent"]
        VA_IN["Input:<br/>─────────────<br/>Subscribe: all messages"]
        VA_OUT["Output:<br/>─────────────<br/>Validated steps:<br/>✓ data_collected<br/>✓ analysis_complete<br/>✓ report_generated<br/>workflow_complete: true"]
    end

    subgraph Output["📤 Output"]
        RESULT["Final Results:<br/>─────────────<br/>{<br/>  data_collector: {...},<br/>  analyzer: {...},<br/>  report_generator: {...},<br/>  validator: {complete: true}<br/>}"]
    end

    USER --> INIT
    INIT --> DC
    DC --> DC_OUT
    DC_OUT --> AN_IN
    AN_IN --> AN
    AN --> AN_OUT
    AN_OUT --> RG_IN
    RG_IN --> RG
    RG --> RG_OUT
    RG_OUT --> VA_IN
    VA_IN --> VA
    VA --> VA_OUT
    VA_OUT --> RESULT

    style Step1 fill:#fff3e0
    style Step2 fill:#e8f5e9
    style Step3 fill:#f3e5f5
    style Step4 fill:#ffebee
```

### Description

Complete pipeline execution:

1. **Data Collection**: Simulates API call, stores 5 data points
2. **Analysis**: Calculates mean (32), max (50), min (10), identifies trends
3. **Report Generation**: Creates formatted report with title, summary, details
4. **Validation**: Verifies all 3 required steps completed successfully

Each step communicates via MessageBus publish/subscribe and shares data via resources.

---

## Real-Time Communication Flowchart

This diagram shows live agent-to-agent messaging.

```mermaid
sequenceDiagram
    autonumber
    participant A as AgentA (Requester)
    participant Bus as MessageBus
    participant B as AgentB (Processor)
    participant C as AgentC (Validator)

    Note over A,C: Real-time Communication Demo

    A->>Bus: publish("request", {task: "Process customer data"})
    Note over A: 📤 Requesting data processing
    
    Bus-->>B: subscribe("request")
    B->>B: Process request
    B->>Bus: publish("processing", {status: "in_progress"})
    Note over B: 📥 Received request, processing...
    
    B->>Bus: store_resource("processed_data", {result})
    Note over B: 📤 Sharing processed results
    
    Bus-->>C: subscribe("resource_shared")
    C->>Bus: get_resource("processed_data")
    Bus-->>C: {result: "Customer insights"}
    C->>C: Validate results
    C->>Bus: publish("validated", {status: "approved"})
    Note over C: 📥 Validating results
    
    Note over A,C: ✅ Communication sequence complete
```

### Description

Real-time inter-agent communication:

1. **AgentA** publishes a request for data processing
2. **AgentB** subscribes, processes, and shares results as a resource
3. **AgentC** subscribes to resource notifications, retrieves data, validates, and publishes approval

This demonstrates the pub/sub pattern enabling loose coupling between agents.

---

## Summary

This documentation covers the complete architecture of the MCP-based Multi-Agent System:

| Diagram Type | Purpose |
|-------------|---------|
| Class Diagram | Data structures and relationships |
| Component Diagram | High-level system layers |
| MessageBus Architecture | Pub/sub and resource sharing |
| Agent Hierarchy | Inheritance and capabilities |
| Sequential Workflow | Step-by-step execution |
| Parallel Execution | Concurrent agent processing |
| Resource Sharing | Data exchange pattern |
| Coordinator Orchestration | Workflow management logic |
| Agent Discovery | Dynamic agent registration |
| Tool Registry | Tool management pattern |
| End-to-End Pipeline | Complete example walkthrough |
| Real-Time Communication | Live messaging demo |

### Key Architecture Patterns

1. **Pub/Sub Messaging**: Loose coupling via MessageBus
2. **Resource Sharing**: Data exchange through shared storage
3. **Agent Registration**: Self-announcing agents on creation
4. **Tool Registry**: Discoverable tool capabilities
5. **Sequential Orchestration**: Coordinator-managed workflows
6. **Parallel Execution**: Async concurrent processing

### MCP Protocol Implementation

| MCP Concept | Implementation |
|-------------|----------------|
| **Tools** | Agent methods exposed via `expose_tool()` |
| **Resources** | Data stored via `share_resource()` |
| **Messages** | Pub/sub via `publish()`/`subscribe()` |
| **Discovery** | Agent registration messages |
| **Interoperability** | Standardized message format |

### Real-World Applications

- **Data Processing Pipelines**: ETL with specialized agents
- **Autonomous Systems**: Robotics and IoT coordination
- **AI Assistants**: Multiple specialized AI agents collaborating
- **Microservices**: Agent-based service architectures
- **Distributed Computing**: Cross-system agent communication

### Extension Points

- Integrate with real MCP SDK servers and clients
- Add authentication and authorization
- Implement persistent message storage (Redis, Kafka)
- Add monitoring and observability (OpenTelemetry)
- Scale to distributed systems with network transport
- Add error recovery and retry logic
- Implement agent health checks and heartbeats
