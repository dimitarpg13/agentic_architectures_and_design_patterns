# Task-Driven Agentic Workflow Visualization

This document provides comprehensive UML diagrams and flowcharts to visualize the task-driven agentic work distribution in the reinforcement learning workflow defined in `simulated_agentic_rl_workflow.ipynb`.

## Table of Contents
1. [System Overview](#1-system-overview)
2. [Class Diagram](#2-class-diagram)
3. [Agent Roles and Hierarchy](#3-agent-roles-and-hierarchy)
4. [Task Lifecycle Flowchart](#4-task-lifecycle-flowchart)
5. [Sequence Diagram: Task Processing](#5-sequence-diagram-task-processing)
6. [Sequence Diagram: Training Loop](#6-sequence-diagram-training-loop)
7. [State Diagram: Task States](#7-state-diagram-task-states)
8. [Activity Diagram: Agent Task Execution](#8-activity-diagram-agent-task-execution)
9. [Component Interaction Flowchart](#9-component-interaction-flowchart)
10. [Tool Selection and Q-Learning Flow](#10-tool-selection-and-q-learning-flow)
11. [Production System Architecture](#11-production-system-architecture)
12. [MLflow Experiment Tracking Flow](#12-mlflow-experiment-tracking-flow)

---

## 1. System Overview

High-level view of the agentic RL system showing the main components and their interactions.

```mermaid
flowchart TB
    subgraph UserLayer["User Layer"]
        User((User Request))
    end
    
    subgraph OrchestratorLayer["Orchestration Layer"]
        ARS[AgenticRLSystem]
        PAS[ProductionAgenticSystem]
    end
    
    subgraph AgentLayer["Multi-Agent Layer"]
        AG1[Researcher Agent]
        AG2[Analyzer Agent]
        AG3[Executor Agent]
        AG4[Validator Agent]
        AG5[Coordinator Agent]
    end
    
    subgraph EnvironmentLayer["Environment Layer"]
        ENV[MultiAgentTaskEnvironment]
        TQ[(Task Queue)]
        AT[(Active Tasks)]
        CT[(Completed Tasks)]
    end
    
    subgraph LearningLayer["Learning Layer"]
        PPO[PPO Model]
        QL[Q-Learning<br/>Tool Selection]
    end
    
    subgraph ToolLayer["Tool Layer"]
        TR[ToolRegistry]
        WS[Web Search]
        DA[Data Analysis]
        CG[Code Generation]
    end
    
    subgraph MonitoringLayer["Monitoring Layer"]
        PM[PerformanceMonitor]
        MLF[MLflow Tracker]
    end
    
    User --> PAS
    PAS --> ARS
    ARS --> ENV
    ARS --> PPO
    
    ENV --> TQ
    ENV --> AT
    ENV --> CT
    
    AG1 & AG2 & AG3 & AG4 & AG5 --> ENV
    AG1 & AG2 & AG3 & AG4 & AG5 --> QL
    QL --> TR
    
    TR --> WS & DA & CG
    
    ENV --> PM
    PPO --> MLF
```

---

## 2. Class Diagram

Complete class diagram showing all classes, their attributes, methods, and relationships.

```mermaid
classDiagram
    class AgentRole {
        <<enumeration>>
        RESEARCHER
        ANALYZER
        EXECUTOR
        VALIDATOR
        COORDINATOR
    }
    
    class Task {
        +str id
        +str type
        +float complexity
        +List~str~ requirements
        +float deadline
        +float priority
        +str status
        +Optional~str~ assigned_agent
        +Optional~float~ completion_time
        +Optional~float~ quality_score
    }
    
    class AgentState {
        +str id
        +AgentRole role
        +float capacity
        +Dict~str,float~ expertise
        +float current_load
        +int completed_tasks
        +float success_rate
        +float collaboration_score
    }
    
    class MultiAgentTaskEnvironment {
        +int n_agents
        +int max_tasks
        +int current_step
        +int max_steps
        +List~AgentState~ agents
        +deque task_queue
        +Dict active_tasks
        +List completed_tasks
        +spaces observation_space
        +spaces action_space
        +ndarray collaboration_matrix
        +_initialize_agents() List~AgentState~
        +_generate_task() Task
        +_get_observation() ndarray
        +step(action) Tuple
        +_assign_task(agent, task)
        +_complete_task(task, agent)
        +reset() Tuple
        +render()
    }
    
    class AgenticRLSystem {
        +env_class
        +int n_envs
        +DummyVecEnv env
        +DummyVecEnv eval_env
        +PPO model
        +Dict training_history
        +List callbacks
        +_create_model() PPO
        +_setup_callbacks() List
        +train(total_timesteps) PPO
        +evaluate(n_episodes) Dict
        +save_model(path)
        +load_model(path)
    }
    
    class ToolRegistry {
        +Dict tools
        +_register_default_tools()
        +register_tool(name, func, description, cost)
        +use_tool(tool_name, kwargs) Dict
        +_mock_web_search(query) str
        +_mock_data_analysis(data) Dict
        +_mock_code_generation(prompt) str
    }
    
    class SmartAgent {
        +str id
        +AgentRole role
        +ToolRegistry tool_registry
        +Dict q_table
        +float epsilon
        +float learning_rate
        +float discount_factor
        +List task_history
        +Dict tool_effectiveness
        +process_task(task) Dict
        +_select_tool(task) str
        +_calculate_reward(tool_result, time, task) float
        +_update_q_table(state, action, reward)
    }
    
    class PerformanceMonitor {
        +Dict metrics_history
        +record_metrics(timestamp, env_info, agent_metrics)
        +plot_performance()
        +generate_report() DataFrame
    }
    
    class ProductionAgenticSystem {
        +Dict config
        +str model_path
        +model
        +ToolRegistry tool_registry
        +PerformanceMonitor monitor
        +int error_count
        +int max_errors
        +bool circuit_breaker_open
        +deque request_times
        +int max_requests_per_minute
        +process_request(request) Dict
        +_check_rate_limit() bool
        +_validate_request(request) Dict
        +_process_with_agents(request) Dict
        +_handle_error(error_message)
        +_reset_circuit_breaker()
        +_record_metrics(result)
        +get_health_status() Dict
    }
    
    class MLflowExperimentTracker {
        +str experiment_name
        +MlflowClient client
        +start_run(run_name, tags)
        +log_hyperparameters(params)
        +log_metrics(metrics, step)
        +log_model(model, model_name)
        +log_artifacts(artifact_path)
        +end_run()
        +compare_runs(metric_name) DataFrame
    }
    
    AgentState --> AgentRole : has
    MultiAgentTaskEnvironment --> AgentState : manages
    MultiAgentTaskEnvironment --> Task : processes
    AgenticRLSystem --> MultiAgentTaskEnvironment : uses
    SmartAgent --> AgentRole : has
    SmartAgent --> ToolRegistry : uses
    SmartAgent --> Task : processes
    ProductionAgenticSystem --> ToolRegistry : uses
    ProductionAgenticSystem --> PerformanceMonitor : uses
    AgenticRLSystem --> MLflowExperimentTracker : tracks with
```

---

## 3. Agent Roles and Hierarchy

Diagram showing the different agent roles and their specializations.

```mermaid
flowchart TB
    subgraph AgentSystem["Multi-Agent System"]
        direction TB
        
        subgraph Roles["Agent Roles"]
            R[🔬 RESEARCHER<br/>Expertise: Research +30%]
            A[📊 ANALYZER<br/>Expertise: Analysis +30%]
            E[⚡ EXECUTOR<br/>Expertise: Execution]
            V[✅ VALIDATOR<br/>Expertise: Validation]
            C[🎯 COORDINATOR<br/>Expertise: Coordination]
        end
        
        subgraph Capabilities["Shared Capabilities"]
            CAP1[Research: 0.5-1.0]
            CAP2[Analysis: 0.5-1.0]
            CAP3[Execution: 0.5-1.0]
            CAP4[Validation: 0.5-1.0]
        end
        
        subgraph Properties["Agent Properties"]
            P1[Capacity: 0.8-1.0]
            P2[Current Load: 0.0+]
            P3[Success Rate: 0.0-1.0]
            P4[Collaboration Score: 0.0-1.0]
        end
    end
    
    R --> CAP1
    A --> CAP2
    E --> CAP3
    V --> CAP4
    
    Capabilities --> Properties
    
    style R fill:#e1f5fe
    style A fill:#f3e5f5
    style E fill:#fff3e0
    style V fill:#e8f5e9
    style C fill:#fce4ec
```

---

## 4. Task Lifecycle Flowchart

Complete flowchart showing a task's journey from generation to completion.

```mermaid
flowchart TD
    Start([Task Generated]) --> TQ[Add to Task Queue]
    TQ --> Check{Task Queue<br/>Not Empty?}
    
    Check -->|No| Wait[Wait for<br/>New Tasks]
    Wait --> TQ
    
    Check -->|Yes| SelectAgent{Select Best<br/>Agent via PPO}
    
    SelectAgent --> Assign[Assign Task<br/>to Agent]
    Assign --> UpdateLoad[Update Agent<br/>Current Load]
    UpdateLoad --> Active[Task Status:<br/>ACTIVE]
    
    Active --> Process[Agent Processes<br/>Task]
    Process --> ToolSelect{Select Tool<br/>ε-greedy}
    
    ToolSelect -->|Explore| RandomTool[Random Tool<br/>Selection]
    ToolSelect -->|Exploit| BestTool[Best Tool from<br/>Q-Table]
    
    RandomTool --> UseTool
    BestTool --> UseTool
    
    UseTool[Execute Tool] --> Progress{Check<br/>Progress}
    
    Progress -->|Random < expertise| Complete[Task Completed]
    Progress -->|Otherwise| Process
    
    Complete --> Quality[Calculate<br/>Quality Score]
    Quality --> UpdateQ[Update Q-Table]
    UpdateQ --> UpdateAgent[Update Agent<br/>Metrics]
    UpdateAgent --> Reward[Calculate<br/>Reward]
    
    Reward --> TimeBonus{Within<br/>Deadline?}
    TimeBonus -->|Yes| BonusReward[Add Time Bonus<br/>to Reward]
    TimeBonus -->|No| PenaltyReward[Apply Deadline<br/>Penalty]
    
    BonusReward --> Final
    PenaltyReward --> Final
    
    Final[Add to<br/>Completed Tasks] --> End([Task Lifecycle<br/>Complete])
    
    style Start fill:#c8e6c9
    style End fill:#c8e6c9
    style Active fill:#fff3e0
    style Complete fill:#e1f5fe
    style Reward fill:#f3e5f5
```

---

## 5. Sequence Diagram: Task Processing

Sequence diagram showing the interaction between components during task processing.

```mermaid
sequenceDiagram
    autonumber
    participant TQ as Task Queue
    participant ENV as Environment
    participant PPO as PPO Policy
    participant AG as Agent
    participant TR as Tool Registry
    participant Tool as Tool (Web/Data/Code)
    participant PM as Performance Monitor
    
    rect rgb(240, 248, 255)
        Note over TQ,PM: Task Assignment Phase
        TQ->>ENV: Get next task from queue
        ENV->>PPO: Request action for current state
        PPO-->>ENV: Return action vector
        ENV->>AG: Assign task based on action
        AG->>AG: Update current_load += complexity
    end
    
    rect rgb(255, 248, 240)
        Note over TQ,PM: Task Execution Phase
        AG->>AG: _select_tool() using ε-greedy
        
        alt Exploration (random < ε)
            AG->>AG: Select random tool
        else Exploitation
            AG->>AG: Select best tool from Q-table
        end
        
        AG->>TR: use_tool(selected_tool, params)
        TR->>Tool: Execute tool function
        Tool-->>TR: Return result
        TR-->>AG: Return tool result + cost
    end
    
    rect rgb(240, 255, 240)
        Note over TQ,PM: Completion Phase
        AG->>AG: Calculate reward
        AG->>AG: Update Q-table
        AG->>ENV: Report task completion
        ENV->>ENV: Update collaboration_matrix
        ENV->>PM: Record metrics
    end
    
    rect rgb(255, 240, 255)
        Note over TQ,PM: Learning Update
        ENV->>PPO: Return observation, reward, done
        PPO->>PPO: Update policy parameters
    end
```

---

## 6. Sequence Diagram: Training Loop

Detailed sequence diagram of the PPO training loop with the multi-agent environment.

```mermaid
sequenceDiagram
    autonumber
    participant User
    participant ARS as AgenticRLSystem
    participant VecEnv as VectorizedEnv
    participant ENV as MultiAgentTaskEnv
    participant PPO as PPO Model
    participant CB as Callbacks
    participant MLF as MLflow
    
    User->>ARS: train(total_timesteps)
    ARS->>MLF: start_run()
    ARS->>MLF: log_hyperparameters()
    
    loop For each training step
        ARS->>PPO: learn(n_steps)
        
        loop Collect n_steps experiences
            PPO->>VecEnv: step(action)
            VecEnv->>ENV: step(action)
            
            Note over ENV: Process task assignments
            ENV->>ENV: _assign_task()
            
            Note over ENV: Process active tasks
            ENV->>ENV: _complete_task()
            
            Note over ENV: Generate new tasks
            ENV->>ENV: _generate_task()
            
            ENV-->>VecEnv: obs, reward, done, info
            VecEnv-->>PPO: batched results
            
            PPO->>PPO: Store in rollout buffer
        end
        
        PPO->>PPO: Compute advantages
        
        loop For n_epochs
            PPO->>PPO: Sample minibatch
            PPO->>PPO: Compute policy loss
            PPO->>PPO: Compute value loss
            PPO->>PPO: Update network
        end
        
        ARS->>CB: on_step()
        
        alt Eval frequency reached
            CB->>ARS: evaluate()
            ARS->>MLF: log_metrics()
        end
        
        alt Checkpoint frequency reached
            CB->>ARS: save_checkpoint()
        end
    end
    
    ARS->>MLF: log_model()
    ARS->>MLF: end_run()
    ARS-->>User: Return trained model
```

---

## 7. State Diagram: Task States

State diagram showing all possible states of a task and transitions.

```mermaid
stateDiagram-v2
    [*] --> Generated: Task Created
    
    Generated --> Pending: Add to Queue
    
    Pending --> Active: Agent Assignment<br/>(PPO Action > 0.5)
    
    Pending --> Pending: No Suitable<br/>Agent Available
    
    Active --> Processing: Agent Starts<br/>Processing
    
    Processing --> Processing: Using Tools<br/>(Q-Learning)
    
    Processing --> Completed: Progress Check<br/>Succeeds
    
    Processing --> Active: Tool Execution<br/>Complete
    
    Active --> Overdue: Deadline<br/>Exceeded
    
    Overdue --> Completed: Eventually<br/>Completed
    
    Completed --> [*]: Task Archived
    
    state Active {
        [*] --> Assigned
        Assigned --> InProgress: Start Processing
        InProgress --> ToolSelection: Need Tool
        ToolSelection --> ToolExecution: Tool Selected
        ToolExecution --> InProgress: Result Received
    }
    
    note right of Processing
        Tools: web_search,
        data_analysis,
        code_generation
    end note
    
    note right of Completed
        Quality Score =
        expertise × success_rate
    end note
```

---

## 8. Activity Diagram: Agent Task Execution

Activity diagram showing the detailed flow of an agent executing a task.

```mermaid
flowchart TD
    subgraph AgentExecution["Agent Task Execution"]
        Start([Receive Task<br/>Assignment]) --> CheckLoad{Current Load<br/>Below Capacity?}
        
        CheckLoad -->|No| QueueTask[Queue Task<br/>Locally]
        QueueTask --> Wait[Wait for<br/>Capacity]
        Wait --> CheckLoad
        
        CheckLoad -->|Yes| AcceptTask[Accept Task]
        AcceptTask --> AnalyzeTask[Analyze Task<br/>Requirements]
        
        AnalyzeTask --> EpsilonCheck{Random less than<br/>Epsilon?}
        
        EpsilonCheck -->|Yes| Explore[Explore:<br/>Random Tool]
        EpsilonCheck -->|No| Exploit[Exploit:<br/>Q-Table Lookup]
        
        Explore --> GetQValues[Get Q-Values<br/>for Task Type]
        Exploit --> GetQValues
        
        GetQValues --> SelectTool[Select Tool]
        SelectTool --> PrepareTool[Prepare Tool<br/>Parameters]
        
        PrepareTool --> CallTool[Call Tool<br/>via Registry]
        
        CallTool --> ToolResult{Tool<br/>Success?}
        
        ToolResult -->|Yes| ProcessResult[Process Tool<br/>Result]
        ToolResult -->|No| HandleError[Handle Tool<br/>Error]
        
        HandleError --> CalculateReward
        
        ProcessResult --> CalculateReward[Calculate<br/>Reward]
        
        CalculateReward --> BaseReward[Base: +1.0 success<br/>-0.5 failure]
        BaseReward --> TimePenalty[Time Penalty:<br/>-time × 0.1]
        TimePenalty --> CostPenalty[Cost Penalty:<br/>-cost × 0.5]
        CostPenalty --> PriorityBonus[Priority Bonus:<br/>+priority × 0.5]
        
        PriorityBonus --> UpdateQ[Update Q-Table<br/>using Learning Rate]
        
        UpdateQ --> UpdateEffectiveness[Update Tool<br/>Effectiveness]
        UpdateEffectiveness --> RecordHistory[Record in<br/>Task History]
        
        RecordHistory --> UpdateMetrics[Update Agent<br/>Metrics]
        UpdateMetrics --> End([Task Complete])
    end
    
    style Start fill:#c8e6c9
    style End fill:#c8e6c9
    style SelectTool fill:#e1f5fe
    style UpdateQ fill:#f3e5f5
```

---

## 9. Component Interaction Flowchart

Flowchart showing how all system components interact during runtime.

```mermaid
flowchart TB
    subgraph Input["Input Layer"]
        REQ[User Request]
        CONFIG[Configuration]
    end
    
    subgraph Production["Production Layer"]
        RATE[Rate Limiter]
        CB[Circuit Breaker]
        VAL[Request Validator]
    end
    
    subgraph Core["Core Processing"]
        direction TB
        
        subgraph Environment["RL Environment"]
            ENV_INIT[Initialize Env]
            OBS[Generate Observation]
            STEP[Environment Step]
            REWARD[Calculate Reward]
        end
        
        subgraph Agents["Multi-Agent System"]
            A1[Agent 1: Researcher]
            A2[Agent 2: Analyzer]
            A3[Agent 3: Executor]
            A4[Agent 4: Validator]
        end
        
        subgraph Learning["Learning Components"]
            PPO_POLICY[PPO Policy Network]
            Q_LEARN[Q-Learning Tables]
            COLLAB[Collaboration Matrix]
        end
    end
    
    subgraph Tools["Tool Layer"]
        WEB[🌐 Web Search<br/>Cost: 0.1]
        DATA[📊 Data Analysis<br/>Cost: 0.2]
        CODE[💻 Code Generation<br/>Cost: 0.3]
    end
    
    subgraph Monitor["Monitoring"]
        PERF[Performance Monitor]
        MLFLOW[MLflow Tracker]
        LOGS[Logging System]
    end
    
    REQ --> RATE
    CONFIG --> CB
    RATE --> CB
    CB --> VAL
    VAL --> ENV_INIT
    
    ENV_INIT --> OBS
    OBS --> PPO_POLICY
    PPO_POLICY --> STEP
    
    STEP --> A1 & A2 & A3 & A4
    
    A1 & A2 & A3 & A4 --> Q_LEARN
    Q_LEARN --> WEB & DATA & CODE
    
    WEB & DATA & CODE --> REWARD
    
    A1 & A2 & A3 & A4 --> COLLAB
    COLLAB --> REWARD
    
    REWARD --> OBS
    
    STEP --> PERF
    PPO_POLICY --> MLFLOW
    VAL --> LOGS
    
    style REQ fill:#e1f5fe
    style PPO_POLICY fill:#f3e5f5
    style REWARD fill:#e8f5e9
```

---

## 10. Tool Selection and Q-Learning Flow

Detailed flowchart of the Q-learning based tool selection mechanism.

```mermaid
flowchart TD
    subgraph QLearning["Q-Learning Tool Selection"]
        Start([Task Received]) --> GetState[Extract State:<br/>task.type]
        
        GetState --> CheckQTable{State in<br/>Q-Table?}
        
        CheckQTable -->|No| InitQ[Initialize Q-Values<br/>to 0.0 for all tools]
        CheckQTable -->|Yes| Proceed
        InitQ --> Proceed
        
        Proceed[Get Q-Values] --> Random{random less than<br/>epsilon?}
        
        Random -->|Yes| RandomSelect[Random Tool<br/>Selection]
        Random -->|No| GreedySelect[Select Best Tool<br/>from Q-Table]
        
        RandomSelect --> ExecuteTool
        GreedySelect --> ExecuteTool
        
        ExecuteTool[Execute Tool] --> GetResult{Success?}
        
        GetResult -->|Yes| BasePos[Base Reward: +1.0]
        GetResult -->|No| BaseNeg[Base Reward: -0.5]
        
        BasePos --> CalcTime[Time Penalty:<br/>-0.1 × time]
        BaseNeg --> CalcTime
        
        CalcTime --> CalcCost[Cost Penalty:<br/>-0.5 × tool_cost]
        CalcCost --> CalcPriority[Priority Bonus:<br/>+0.5 × priority]
        
        CalcPriority --> TotalReward[Total Reward: R]
        
        TotalReward --> UpdateRule[Q-Value Update<br/>Q = Q + lr * R - Q]
        
        UpdateRule --> UpdateEffect[Update Effectiveness<br/>0.95 * old + 0.05 * success]
        
        UpdateEffect --> End([Return Result])
    end
    
    subgraph Parameters["Learning Parameters"]
        P1[epsilon = 0.1<br/>Exploration Rate]
        P2[alpha = 0.1<br/>Learning Rate]
        P3[gamma = 0.95<br/>Discount Factor]
    end
    
    Parameters -.-> QLearning
    
    style Start fill:#c8e6c9
    style End fill:#c8e6c9
    style UpdateRule fill:#f3e5f5
    style TotalReward fill:#e1f5fe
```

---

## 11. Production System Architecture

Architecture diagram for the production-ready system with all safeguards.

```mermaid
flowchart TB
    subgraph External["External Interface"]
        CLIENT[Client Request]
    end
    
    subgraph SafetyLayer["Safety & Control Layer"]
        direction LR
        RL[Rate Limiter<br/>60 req/min]
        CB[Circuit Breaker<br/>max 10 errors]
        VALID[Request<br/>Validator]
    end
    
    subgraph Processing["Processing Pipeline"]
        direction TB
        
        TIMEOUT[Timeout Handler<br/>30 seconds]
        
        subgraph AgentPipeline["Agent Processing"]
            DISPATCH[Task Dispatcher]
            AGENT1[Agent Pool]
            TOOLS[Tool Execution]
        end
        
        RESULT[Result Aggregator]
    end
    
    subgraph ErrorHandling["Error Handling"]
        ERR_LOG[Error Logger]
        ERR_COUNT[Error Counter]
        CB_OPEN[Open Circuit]
        CB_RESET[Reset Timer<br/>60 sec cooldown]
    end
    
    subgraph Health["Health Monitoring"]
        HEALTH[Health Check<br/>Endpoint]
        STATUS[System Status]
        METRICS[Metrics Export]
    end
    
    CLIENT --> RL
    RL -->|pass| CB
    RL -->|fail| RATE_ERR[Rate Limit Error]
    
    CB -->|closed| VALID
    CB -->|open| CB_ERR[Circuit Open Error]
    
    VALID -->|valid| TIMEOUT
    VALID -->|invalid| VAL_ERR[Validation Error]
    
    TIMEOUT --> DISPATCH
    DISPATCH --> AGENT1
    AGENT1 --> TOOLS
    TOOLS --> RESULT
    
    RESULT -->|success| RESPONSE[Response]
    RESULT -->|error| ERR_LOG
    TIMEOUT -->|timeout| ERR_LOG
    
    ERR_LOG --> ERR_COUNT
    ERR_COUNT -->|count >= max| CB_OPEN
    CB_OPEN --> CB
    CB_OPEN --> CB_RESET
    CB_RESET -->|after cooldown| CB
    
    HEALTH --> STATUS
    STATUS --> METRICS
    
    RESPONSE --> CLIENT
    
    style CLIENT fill:#e1f5fe
    style RESPONSE fill:#e8f5e9
    style CB_OPEN fill:#ffcdd2
    style ERR_LOG fill:#fff3e0
```

---

## 12. MLflow Experiment Tracking Flow

Sequence showing how experiments are tracked with MLflow.

```mermaid
sequenceDiagram
    autonumber
    participant User
    participant Tracker as MLflowTracker
    participant MLflow as MLflow Server
    participant Model as PPO Model
    participant Artifacts as Artifact Store
    
    rect rgb(240, 248, 255)
        Note over User,Artifacts: Experiment Setup
        User->>Tracker: Create tracker(experiment_name)
        Tracker->>MLflow: set_experiment()
        MLflow-->>Tracker: Experiment ID
    end
    
    rect rgb(255, 248, 240)
        Note over User,Artifacts: Run Initialization
        User->>Tracker: start_run(run_name, tags)
        Tracker->>MLflow: start_run()
        MLflow-->>Tracker: Run ID
        
        User->>Tracker: log_hyperparameters(params)
        Tracker->>MLflow: log_param(key, value)
    end
    
    rect rgb(240, 255, 240)
        Note over User,Artifacts: Training Loop
        loop For each training step
            Model->>Model: Train step
            Model-->>Tracker: Metrics
            Tracker->>MLflow: log_metric(metric, value, step)
        end
    end
    
    rect rgb(255, 240, 255)
        Note over User,Artifacts: Model Saving
        User->>Tracker: log_model(model, name)
        Tracker->>MLflow: pytorch.log_model()
        MLflow->>Artifacts: Store model files
    end
    
    rect rgb(248, 248, 248)
        Note over User,Artifacts: Experiment Completion
        User->>Tracker: end_run()
        Tracker->>MLflow: end_run()
        
        User->>Tracker: compare_runs(metric)
        Tracker->>MLflow: search_runs()
        MLflow-->>Tracker: Run comparison data
        Tracker-->>User: DataFrame with results
    end
```

---

## Summary: Task-Driven Work Distribution

```mermaid
flowchart LR
    subgraph TaskFlow["Task-Driven Distribution"]
        direction TB
        
        T1[📋 Task Generated] --> T2[📥 Queue Assignment]
        T2 --> T3[🤖 PPO Policy<br/>Selects Agent]
        T3 --> T4[👤 Agent Assigned]
        T4 --> T5[🔧 Q-Learning<br/>Selects Tool]
        T5 --> T6[⚙️ Tool Executed]
        T6 --> T7[📊 Reward Calculated]
        T7 --> T8[🧠 Models Updated]
        T8 --> T9[✅ Task Complete]
    end
    
    subgraph Agents["Agent Specialization"]
        direction TB
        AG1[🔬 Researcher<br/>Research Tasks]
        AG2[📊 Analyzer<br/>Analysis Tasks]
        AG3[⚡ Executor<br/>Execution Tasks]
        AG4[✅ Validator<br/>Validation Tasks]
    end
    
    subgraph Learning["Learning Mechanisms"]
        direction TB
        L1[PPO: Task Assignment]
        L2[Q-Learning: Tool Selection]
        L3[Collaboration Matrix]
    end
    
    T3 -.-> AG1 & AG2 & AG3 & AG4
    T5 -.-> L2
    T3 -.-> L1
    T7 -.-> L3
    
    style T1 fill:#c8e6c9
    style T9 fill:#c8e6c9
    style L1 fill:#e1f5fe
    style L2 fill:#f3e5f5
    style L3 fill:#fff3e0
```

---

## Key Insights

### Task Distribution Strategy
1. **PPO Policy** decides which agent receives each task based on:
   - Agent capacity and current load
   - Task complexity and requirements
   - Historical success rates
   - Collaboration opportunities

2. **Q-Learning** enables each agent to learn optimal tool selection:
   - ε-greedy exploration balances learning vs exploitation
   - Rewards shape tool preferences per task type
   - Tool effectiveness tracking improves over time

3. **Collaboration Matrix** captures inter-agent synergies:
   - Agents requesting collaboration simultaneously get bonuses
   - Matrix values increase with successful collaborations
   - Enables emergent team behaviors

### Reward Structure
| Component | Value | Description |
|-----------|-------|-------------|
| Task Match | +0.0 to +1.0 | Agent expertise × task priority |
| Quality | +0.0 to +10.0 | (expertise × success_rate × priority) × time_bonus |
| Collaboration | +0.5 | Mutual collaboration bonus |
| Queue Penalty | -0.1/task | Discourages queue buildup |
| Overdue Penalty | -0.5/task | Penalizes missed deadlines |
