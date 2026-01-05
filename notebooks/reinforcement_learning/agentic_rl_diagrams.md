# Agentic RL Workflow - Architecture Diagrams

## Table of Contents
1. [System Overview Flowchart](#system-overview-flowchart)
2. [Training Pipeline Flowchart](#training-pipeline-flowchart)
3. [Agent Decision Process Flowchart](#agent-decision-process-flowchart)
4. [Class UML Diagrams](#class-uml-diagrams)
5. [Sequence Diagrams](#sequence-diagrams)

---

## System Overview Flowchart

```mermaid
flowchart TB
    Start([Start]) --> Init[Initialize System]
    
    Init --> CreateEnv[Create Multi-Agent Environment]
    Init --> CreateAgents[Initialize Agents with Roles]
    Init --> InitRL[Initialize PPO Model]
    
    CreateEnv --> EnvReady{Environment Ready?}
    CreateAgents --> AgentsReady{Agents Ready?}
    InitRL --> ModelReady{Model Ready?}
    
    EnvReady -->|Yes| SystemReady
    AgentsReady -->|Yes| SystemReady
    ModelReady -->|Yes| SystemReady
    
    SystemReady[System Ready] --> MainLoop{Training or Production?}
    
    MainLoop -->|Training| TrainingLoop
    MainLoop -->|Production| ProductionLoop
    
    TrainingLoop[Training Loop] --> CollectExp[Collect Experience]
    CollectExp --> UpdatePolicy[Update PPO Policy]
    UpdatePolicy --> EvalPerf[Evaluate Performance]
    EvalPerf --> CheckConv{Converged?}
    
    CheckConv -->|No| CollectExp
    CheckConv -->|Yes| SaveModel[Save Model]
    
    ProductionLoop[Production Loop] --> ReceiveReq[Receive Request]
    ReceiveReq --> ValidateReq[Validate Request]
    ValidateReq --> RateLimit{Rate Limit OK?}
    
    RateLimit -->|No| RejectReq[Reject Request]
    RateLimit -->|Yes| ProcessReq[Process with Agents]
    
    ProcessReq --> ReturnResp[Return Response]
    ReturnResp --> LogMetrics[Log Metrics]
    LogMetrics --> ReceiveReq
    
    SaveModel --> End([End])
    RejectReq --> ReceiveReq
```

---

## Training Pipeline Flowchart

```mermaid
flowchart TD
    Start([Start Training]) --> InitSystem[Initialize AgenticRLSystem]
    
    InitSystem --> CreateVecEnv[Create Vectorized Environments]
    CreateVecEnv --> InitPPO[Initialize PPO Model]
    
    InitPPO --> SetCallbacks[Setup Callbacks]
    SetCallbacks --> StartTrain[Start Training Loop]
    
    StartTrain --> Episode[New Episode]
    Episode --> Reset[Reset Environment]
    Reset --> GetObs[Get Initial Observation]
    
    GetObs --> ActionLoop{Max Steps Reached?}
    
    ActionLoop -->|No| PredictAction[Model Predicts Action]
    PredictAction --> ParseAction[Parse Action Vector]
    
    ParseAction --> TaskAssign[Task Assignment]
    ParseAction --> ResourceAlloc[Resource Allocation]
    ParseAction --> CollabReq[Collaboration Request]
    
    TaskAssign --> EnvStep[Environment Step]
    ResourceAlloc --> EnvStep
    CollabReq --> EnvStep
    
    EnvStep --> CalcReward[Calculate Reward]
    CalcReward --> UpdateState[Update Agent States]
    UpdateState --> ProcessTasks[Process Active Tasks]
    
    ProcessTasks --> TaskComplete{Task Completed?}
    TaskComplete -->|Yes| QualityScore[Calculate Quality Score]
    TaskComplete -->|No| Continue
    
    QualityScore --> RewardBonus[Add Completion Bonus]
    RewardBonus --> Continue
    
    Continue[Continue] --> GenerateTasks{Generate New Tasks?}
    GenerateTasks -->|Yes| AddQueue[Add to Task Queue]
    GenerateTasks -->|No| NextStep
    
    AddQueue --> NextStep
    NextStep[Next Step] --> GetObs
    
    ActionLoop -->|Yes| EpisodeEnd[Episode End]
    EpisodeEnd --> StoreBuffer[Store in Replay Buffer]
    
    StoreBuffer --> BufferFull{Buffer Full?}
    BufferFull -->|No| Episode
    BufferFull -->|Yes| UpdatePPO[Update PPO Policy]
    
    UpdatePPO --> CalcAdvantage[Calculate Advantages]
    CalcAdvantage --> OptimizePolicy[Optimize Policy Network]
    OptimizePolicy --> OptimizeValue[Optimize Value Network]
    
    OptimizeValue --> EvalCallback{Evaluation Time?}
    EvalCallback -->|Yes| Evaluate[Evaluate Performance]
    EvalCallback -->|No| CheckDone
    
    Evaluate --> SaveBest{Best Model?}
    SaveBest -->|Yes| SaveModel[Save Best Model]
    SaveBest -->|No| CheckDone
    
    SaveModel --> CheckDone
    CheckDone{Training Complete?}
    CheckDone -->|No| Episode
    CheckDone -->|Yes| End([End Training])
```

---

## Agent Decision Process Flowchart

```mermaid
flowchart LR
    Start([Agent Receives Task]) --> CheckCapacity{Check Capacity}
    
    CheckCapacity -->|Full| Decline[Decline Task]
    CheckCapacity -->|Available| EvalTask[Evaluate Task]
    
    EvalTask --> MatchExpertise[Match Task to Expertise]
    MatchExpertise --> CheckComplexity[Assess Complexity]
    
    CheckComplexity --> NeedCollab{Need Collaboration?}
    
    NeedCollab -->|Yes| RequestCollab[Request Collaboration]
    NeedCollab -->|No| SelectTool[Select Tool]
    
    RequestCollab --> FindPartner[Find Available Partner]
    FindPartner --> PartnerFound{Partner Found?}
    
    PartnerFound -->|Yes| CollabProcess[Collaborative Processing]
    PartnerFound -->|No| SoloProcess[Solo Processing]
    
    SelectTool --> CheckQTable{Q-Table Entry Exists?}
    CheckQTable -->|Yes| UseEpsilon{Explore or Exploit?}
    CheckQTable -->|No| InitQValues[Initialize Q-Values]
    
    InitQValues --> RandomTool[Select Random Tool]
    
    UseEpsilon -->|Explore| RandomTool
    UseEpsilon -->|Exploit| BestTool[Select Best Tool]
    
    RandomTool --> UseTool[Use Selected Tool]
    BestTool --> UseTool
    CollabProcess --> UseTool
    SoloProcess --> UseTool
    
    UseTool --> ProcessResult[Process Tool Result]
    ProcessResult --> CalcReward[Calculate Reward]
    
    CalcReward --> UpdateQTable[Update Q-Table]
    UpdateQTable --> UpdateMetrics[Update Agent Metrics]
    
    UpdateMetrics --> TaskDone{Task Complete?}
    TaskDone -->|No| SelectTool
    TaskDone -->|Yes| RecordCompletion[Record Completion]
    
    RecordCompletion --> UpdateSuccess[Update Success Rate]
    UpdateSuccess --> End([Task Completed])
    
    Decline --> End
```

---

## Class UML Diagrams

### Core System Classes

```mermaid
classDiagram
    class AgenticRLSystem {
        -env_class: type
        -n_envs: int
        -env: DummyVecEnv
        -eval_env: DummyVecEnv
        -model: PPO
        -training_history: Dict
        -callbacks: List
        +__init__(env_class, n_envs)
        +train(total_timesteps): PPO
        +evaluate(n_episodes): Dict
        +save_model(path): void
        +load_model(path): void
        -_create_model(): PPO
        -_setup_callbacks(): List
    }
    
    class MultiAgentTaskEnvironment {
        -n_agents: int
        -max_tasks: int
        -current_step: int
        -max_steps: int
        -agents: List~AgentState~
        -task_queue: deque
        -active_tasks: Dict
        -completed_tasks: List
        -collaboration_matrix: ndarray
        -observation_space: Box
        -action_space: Box
        +__init__(n_agents, max_tasks)
        +step(action): Tuple
        +reset(seed, options): Tuple
        +render(): void
        -_initialize_agents(): List
        -_generate_task(): Task
        -_get_observation(): ndarray
        -_assign_task(agent, task): void
        -_complete_task(task, agent): void
    }
    
    class Task {
        +id: str
        +type: str
        +complexity: float
        +requirements: List~str~
        +deadline: float
        +priority: float
        +status: str
        +assigned_agent: Optional~str~
        +completion_time: Optional~float~
        +quality_score: Optional~float~
    }
    
    class AgentState {
        +id: str
        +role: AgentRole
        +capacity: float
        +expertise: Dict~str, float~
        +current_load: float
        +completed_tasks: int
        +success_rate: float
        +collaboration_score: float
    }
    
    class AgentRole {
        <<enumeration>>
        RESEARCHER
        ANALYZER
        EXECUTOR
        VALIDATOR
        COORDINATOR
    }
    
    AgenticRLSystem --> MultiAgentTaskEnvironment : uses
    MultiAgentTaskEnvironment --> AgentState : manages
    MultiAgentTaskEnvironment --> Task : processes
    AgentState --> AgentRole : has
    MultiAgentTaskEnvironment --> Task : creates
```

### Smart Agent and Tool System Classes

```mermaid
classDiagram
    class SmartAgent {
        -id: str
        -role: AgentRole
        -tool_registry: ToolRegistry
        -q_table: Dict
        -epsilon: float
        -learning_rate: float
        -discount_factor: float
        -task_history: List
        -tool_effectiveness: Dict
        +__init__(agent_id, role, tool_registry)
        +process_task(task): Dict
        -_select_tool(task): str
        -_calculate_reward(tool_result, time, task): float
        -_update_q_table(state, action, reward): void
    }
    
    class ToolRegistry {
        -tools: Dict
        +__init__()
        +register_tool(name, func, description, cost): void
        +use_tool(tool_name, kwargs): Dict
        -_register_default_tools(): void
        -_mock_web_search(query): str
        -_mock_data_analysis(data): Dict
        -_mock_code_generation(prompt): str
    }
    
    class PerformanceMonitor {
        -metrics_history: Dict
        +__init__()
        +record_metrics(timestamp, env_info, agent_metrics): void
        +plot_performance(): void
        +generate_report(): DataFrame
    }
    
    class ProductionAgenticSystem {
        -config: Dict
        -model_path: str
        -model: Optional~PPO~
        -tool_registry: ToolRegistry
        -monitor: PerformanceMonitor
        -error_count: int
        -max_errors: int
        -circuit_breaker_open: bool
        -request_times: deque
        -max_requests_per_minute: int
        +__init__(model_path, config)
        +process_request(request): Dict
        +get_health_status(): Dict
        -_check_rate_limit(): bool
        -_validate_request(request): Dict
        -_process_with_agents(request): Dict
        -_handle_error(error_message): void
        -_reset_circuit_breaker(): void
        -_record_metrics(result): void
    }
    
    class MLflowExperimentTracker {
        -experiment_name: str
        -client: MlflowClient
        +__init__(experiment_name)
        +start_run(run_name, tags): void
        +log_hyperparameters(params): void
        +log_metrics(metrics, step): void
        +log_model(model, model_name): void
        +log_artifacts(artifact_path): void
        +end_run(): void
        +compare_runs(metric_name): DataFrame
    }
    
    SmartAgent --> ToolRegistry : uses
    SmartAgent --> AgentRole : has
    ProductionAgenticSystem --> ToolRegistry : contains
    ProductionAgenticSystem --> PerformanceMonitor : contains
    ProductionAgenticSystem --> SmartAgent : manages
    AgenticRLSystem --> MLflowExperimentTracker : uses
```

---

## Sequence Diagrams

### Training Sequence

```mermaid
sequenceDiagram
    participant User
    participant System as AgenticRLSystem
    participant Env as Environment
    participant Model as PPO Model
    participant Agent as SmartAgent
    participant MLflow
    
    User->>System: train(timesteps)
    System->>MLflow: start_run()
    System->>MLflow: log_hyperparameters()
    
    loop Training Episodes
        System->>Env: reset()
        Env-->>System: initial_observation
        
        loop Episode Steps
            System->>Model: predict(observation)
            Model-->>System: action
            
            System->>Env: step(action)
            
            Env->>Agent: assign_task()
            Agent->>Agent: select_tool()
            Agent->>Agent: process_task()
            Agent-->>Env: task_result
            
            Env->>Env: calculate_reward()
            Env->>Env: update_states()
            Env-->>System: obs, reward, done, info
            
            System->>System: store_experience()
        end
        
        System->>Model: update_policy()
        Model->>Model: optimize_networks()
        
        System->>MLflow: log_metrics()
    end
    
    System->>System: evaluate()
    System->>Model: save()
    System->>MLflow: log_model()
    System->>MLflow: end_run()
    System-->>User: trained_model
```

### Production Request Processing

```mermaid
sequenceDiagram
    participant Client
    participant ProdSystem as ProductionSystem
    participant RateLimit as Rate Limiter
    participant CircuitBreaker
    participant Agent
    participant Tool as ToolRegistry
    participant Monitor
    
    Client->>ProdSystem: process_request(request)
    
    ProdSystem->>CircuitBreaker: check_status()
    alt Circuit Open
        CircuitBreaker-->>ProdSystem: breaker_open
        ProdSystem-->>Client: error: unavailable
    else Circuit Closed
        CircuitBreaker-->>ProdSystem: breaker_closed
        
        ProdSystem->>RateLimit: check_rate_limit()
        alt Rate Exceeded
            RateLimit-->>ProdSystem: limit_exceeded
            ProdSystem-->>Client: error: rate_limit
        else Within Limit
            RateLimit-->>ProdSystem: ok
            
            ProdSystem->>ProdSystem: validate_request()
            
            alt Invalid Request
                ProdSystem-->>Client: error: validation_failed
            else Valid Request
                ProdSystem->>Agent: process_task()
                
                Agent->>Tool: use_tool()
                Tool->>Tool: execute_function()
                Tool-->>Agent: tool_result
                
                Agent->>Agent: update_q_table()
                Agent-->>ProdSystem: task_result
                
                ProdSystem->>Monitor: record_metrics()
                Monitor->>Monitor: store_metrics()
                
                ProdSystem-->>Client: success_response
            end
        end
    end
    
    opt On Error
        ProdSystem->>CircuitBreaker: increment_error()
        CircuitBreaker->>CircuitBreaker: check_threshold()
        alt Threshold Exceeded
            CircuitBreaker->>CircuitBreaker: open_breaker()
            CircuitBreaker->>CircuitBreaker: schedule_reset()
        end
    end
```

### Multi-Agent Collaboration Sequence

```mermaid
sequenceDiagram
    participant Coordinator
    participant Agent1 as Agent 1 (Researcher)
    participant Agent2 as Agent 2 (Analyzer)
    participant Agent3 as Agent 3 (Executor)
    participant TaskQueue
    participant CollabMatrix as Collaboration Matrix
    
    Coordinator->>TaskQueue: get_next_task()
    TaskQueue-->>Coordinator: task
    
    Coordinator->>Agent1: evaluate_task(task)
    Agent1->>Agent1: check_expertise()
    Agent1-->>Coordinator: capability_score
    
    Coordinator->>Agent2: evaluate_task(task)
    Agent2->>Agent2: check_expertise()
    Agent2-->>Coordinator: capability_score
    
    Coordinator->>Agent3: evaluate_task(task)
    Agent3->>Agent3: check_expertise()
    Agent3-->>Coordinator: capability_score
    
    Coordinator->>Coordinator: select_best_agent()
    
    alt Single Agent Capable
        Coordinator->>Agent2: assign_task(task)
        Agent2->>Agent2: process_independently()
        Agent2-->>Coordinator: result
    else Collaboration Required
        Coordinator->>Agent1: request_collaboration(task)
        Coordinator->>Agent2: request_collaboration(task)
        
        Agent1->>Agent2: share_context()
        Agent2->>Agent1: share_analysis()
        
        par Parallel Processing
            Agent1->>Agent1: research_phase()
        and
            Agent2->>Agent2: analysis_phase()
        end
        
        Agent1-->>Agent2: research_results
        Agent2->>Agent2: integrate_results()
        Agent2-->>Coordinator: combined_result
        
        Coordinator->>CollabMatrix: update_scores(agent1, agent2)
        CollabMatrix->>CollabMatrix: increase_collaboration_weight()
    end
    
    Coordinator->>TaskQueue: mark_complete(task)
    Coordinator->>Coordinator: calculate_rewards()
```

### Tool Selection Learning Sequence

```mermaid
sequenceDiagram
    participant Agent
    participant QTable as Q-Table
    participant ToolReg as ToolRegistry
    participant Task
    
    Task->>Agent: new_task(type, complexity)
    
    Agent->>Agent: get_state(task.type)
    Agent->>QTable: lookup(state)
    
    alt State Not Found
        Agent->>QTable: initialize_values(state)
        QTable-->>Agent: initial_q_values
    else State Exists
        QTable-->>Agent: q_values
    end
    
    Agent->>Agent: epsilon_greedy_selection()
    
    alt Explore (random < epsilon)
        Agent->>Agent: random_tool_selection()
    else Exploit
        Agent->>Agent: argmax(q_values)
    end
    
    Agent->>ToolReg: use_tool(selected_tool, params)
    ToolReg->>ToolReg: execute_tool()
    ToolReg-->>Agent: tool_result
    
    Agent->>Agent: calculate_reward(result, time, cost)
    
    Agent->>QTable: get_current_value(state, action)
    QTable-->>Agent: old_q_value
    
    Agent->>Agent: compute_new_q_value(old, reward, learning_rate)
    
    Agent->>QTable: update(state, action, new_q_value)
    QTable->>QTable: store_updated_value()
    
    Agent->>Agent: update_tool_effectiveness(tool, reward)
    Agent->>Agent: record_history(task, tool, reward)
    
    Agent-->>Task: completion_status
```

---

## Component Interaction Overview

```mermaid
graph TB
    subgraph Training Phase
        DataGen[Data Generation] --> Env[Environment]
        Env --> PPO[PPO Algorithm]
        PPO --> Policy[Policy Network]
        Policy --> Actions[Action Selection]
        Actions --> Env
    end
    
    subgraph Agent System
        Actions --> MA[Multi-Agent Coordinator]
        MA --> A1[Agent 1]
        MA --> A2[Agent 2]
        MA --> A3[Agent 3]
        MA --> A4[Agent 4]
        
        A1 --> Tools[Tool Registry]
        A2 --> Tools
        A3 --> Tools
        A4 --> Tools
    end
    
    subgraph Monitoring
        Env --> Metrics[Metrics Collector]
        MA --> Metrics
        Metrics --> MLflow[MLflow Tracking]
        Metrics --> Visualizer[Dashboard]
    end
    
    subgraph Production
        Model[Trained Model] --> ProdSys[Production System]
        ProdSys --> RateLimit[Rate Limiter]
        ProdSys --> Circuit[Circuit Breaker]
        ProdSys --> MA
        ProdSys --> Monitor[Health Monitor]
    end
    
    PPO --> Model
    Model --> Checkpoint[Model Checkpoint]
    Checkpoint --> ProdSys
```

---

## State Machine Diagram

```mermaid
stateDiagram-v2
    [*] --> Initialization
    
    Initialization --> Ready: System Initialized
    
    Ready --> Training: Start Training
    Ready --> Production: Deploy to Production
    
    state Training {
        [*] --> CollectingData
        CollectingData --> UpdatingPolicy: Buffer Full
        UpdatingPolicy --> Evaluating: Policy Updated
        Evaluating --> CollectingData: Continue Training
        Evaluating --> ModelSaved: Training Complete
        ModelSaved --> [*]
    }
    
    state Production {
        [*] --> Healthy
        
        Healthy --> Processing: Request Received
        Processing --> Healthy: Success
        Processing --> ErrorState: Failure
        
        ErrorState --> Healthy: Error Recovered
        ErrorState --> CircuitOpen: Max Errors
        
        CircuitOpen --> Cooldown: Breaker Opened
        Cooldown --> Healthy: Reset After Timeout
        
        state Processing {
            [*] --> Validating
            Validating --> RateLimiting: Valid
            RateLimiting --> TaskAssignment: Within Limit
            TaskAssignment --> ToolExecution
            ToolExecution --> ResultGeneration
            ResultGeneration --> [*]
        }
    }
    
    Training --> Ready: Training Aborted
    Production --> Ready: Shutdown
    Ready --> [*]: System Exit
```

---

## Data Flow Diagram

```mermaid
flowchart LR
    subgraph Input
        TR[Training Data]
        PR[Production Requests]
        CF[Configuration]
    end
    
    subgraph Processing
        ENV[Environment]
        PPO[PPO Model]
        AG[Agents]
        TL[Tools]
        
        TR --> ENV
        ENV <--> PPO
        PPO --> AG
        AG <--> TL
        PR --> AG
        CF --> ENV
        CF --> PPO
    end
    
    subgraph Storage
        QB[Q-Tables]
        MD[Model Weights]
        MT[Metrics Store]
        
        AG --> QB
        PPO --> MD
        AG --> MT
        ENV --> MT
    end
    
    subgraph Output
        RS[Response]
        VZ[Visualizations]
        RP[Reports]
        ML[MLflow Logs]
        
        AG --> RS
        MT --> VZ
        MT --> RP
        MT --> ML
    end
    
    PR --> RS
```

---

## Error Handling Flow

```mermaid
flowchart TD
    Start([Request Received]) --> Process[Process Request]
    
    Process --> Error{Error Occurred?}
    
    Error -->|No| Success[Return Success Response]
    Error -->|Yes| ErrorType{Error Type?}
    
    ErrorType -->|Timeout| TimeoutHandler[Handle Timeout]
    ErrorType -->|Validation| ValidationHandler[Handle Validation Error]
    ErrorType -->|RateLimit| RateLimitHandler[Handle Rate Limit]
    ErrorType -->|System| SystemHandler[Handle System Error]
    
    TimeoutHandler --> LogError[Log Error]
    ValidationHandler --> LogError
    RateLimitHandler --> LogError
    SystemHandler --> LogError
    
    LogError --> IncrementCount[Increment Error Count]
    
    IncrementCount --> CheckThreshold{Above Threshold?}
    
    CheckThreshold -->|No| ReturnError[Return Error Response]
    CheckThreshold -->|Yes| OpenBreaker[Open Circuit Breaker]
    
    OpenBreaker --> NotifyOps[Notify Operations]
    NotifyOps --> ScheduleReset[Schedule Reset]
    
    ScheduleReset --> ReturnError
    ReturnError --> End([End])
    Success --> End
    
    ScheduleReset --> WaitCooldown[Wait Cooldown Period]
    WaitCooldown --> ResetBreaker[Reset Circuit Breaker]
    ResetBreaker --> ResetCount[Reset Error Count]
    ResetCount --> BreakerClosed[Breaker Closed]
```

---

## Deployment Architecture

```mermaid
graph TB
    subgraph Client Layer
        WebApp[Web Application]
        API[REST API]
        SDK[Python SDK]
    end
    
    subgraph Load Balancer
        LB[Load Balancer]
    end
    
    subgraph Application Servers
        AS1[App Server 1]
        AS2[App Server 2]
        AS3[App Server 3]
        
        subgraph AS1_Components
            PD1[Production System]
            AG1[Agent Pool]
            CB1[Circuit Breaker]
        end
        
        AS1 --> AS1_Components
    end
    
    subgraph ML Infrastructure
        ModelReg[Model Registry]
        MLflow[MLflow Server]
        TensorBoard[TensorBoard]
    end
    
    subgraph Storage Layer
        Redis[Redis Cache]
        PostgreSQL[PostgreSQL DB]
        S3[S3 Model Storage]
    end
    
    subgraph Monitoring
        Prometheus[Prometheus]
        Grafana[Grafana]
        AlertManager[Alert Manager]
    end
    
    WebApp --> LB
    API --> LB
    SDK --> LB
    
    LB --> AS1
    LB --> AS2
    LB --> AS3
    
    AS1 --> ModelReg
    AS2 --> ModelReg
    AS3 --> ModelReg
    
    ModelReg --> S3
    
    AS1 --> Redis
    AS2 --> Redis
    AS3 --> Redis
    
    AS1 --> PostgreSQL
    AS2 --> PostgreSQL
    AS3 --> PostgreSQL
    
    AS1 --> Prometheus
    AS2 --> Prometheus
    AS3 --> Prometheus
    
    Prometheus --> Grafana
    Prometheus --> AlertManager
    
    MLflow --> S3
    MLflow --> PostgreSQL
```

---

## Summary

These diagrams provide a comprehensive view of the agentic RL workflow system architecture:

1. **Flowcharts** show the logical flow of training, agent decisions, and system operations
2. **Class UML Diagrams** detail the object-oriented structure and relationships
3. **Sequence Diagrams** illustrate the temporal interactions between components
4. **State Machine Diagram** captures the system's behavioral states
5. **Data Flow Diagram** shows how information moves through the system
6. **Deployment Architecture** presents the production infrastructure

The diagrams cover all major aspects:
- Multi-agent coordination
- Reinforcement learning pipeline
- Tool selection and Q-learning
- Production patterns (circuit breakers, rate limiting)
- Monitoring and observability
- MLflow integration for experiment tracking

These visualizations can be rendered in any Markdown viewer that supports Mermaid diagrams (GitHub, GitLab, many documentation tools).
