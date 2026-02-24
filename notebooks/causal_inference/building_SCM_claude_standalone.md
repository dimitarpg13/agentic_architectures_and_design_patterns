# Building Structural Causal Models (SCMs) from Text Corpora Using Agentic Workflows

## Overview

This document extends the architecture presented in [building_causal_DAG.md](building_causal_DAG.md), which covered the construction of Causal DAGs — the qualitative skeleton of cause-and-effect relationships. Here we address the next, harder problem: constructing full **Structural Causal Models (SCMs)** — the quantitative machinery needed to answer *Why?*-type questions from Rung 3 of Pearl's causal ladder.

A Causal DAG tells us *what causes what*. An SCM tells us *how*, *how much*, and *what would have happened if things had been different*.

---

## Why SCMs Are Necessary Beyond Causal DAGs

An SCM $\mathcal{M} = \langle \mathbf{U}, \mathbf{V}, \mathbf{F}, P(\mathbf{U}) \rangle$ consists of four components:

| Symbol | Name | Meaning |
|--------|------|---------|
| **U** | Exogenous variables | Unobserved noise / background conditions |
| **V** | Endogenous variables | Observed variables (the nodes of the DAG) |
| **F** | Structural equations | $V_i = f_i(\text{Pa}(V_i), U_i)$ — one equation per node |
| **P(U)** | Exogenous distribution | Joint probability over background noise |

The Causal DAG — as constructed by the pipeline in `building_causal_DAG.md` — gives us the graph structure: which variables appear as parents in each equation. But the DAG alone is a *qualitative* object. To simulate interventions (`do`-operator) and reason about counterfactuals ("What would Y have been if X had been different?"), we need the **structural equations F** and the **exogenous distributions P(U)**.

```mermaid
flowchart LR
    subgraph DAG_Only["Causal DAG Only<br/>(from building_causal_DAG.md)"]
        direction TB
        D1["Nodes: {X, Y, Z}"]
        D2["Edges: X → Y, Z → Y"]
        D3["Confidence scores"]
        D4["Evidence map"]
        D1 ~~~ D2 ~~~ D3 ~~~ D4
    end

    subgraph Full_SCM["Full SCM<br/>(this document)"]
        direction TB
        S1["Nodes: {X, Y, Z}"]
        S2["Edges: X → Y, Z → Y"]
        S3["Equations:<br/>Y = 0.7X + 0.3Z + ε"]
        S4["Noise: ε ~ N(0, σ²)"]
        S5["Abduction mechanism<br/>for counterfactuals"]
        S1 ~~~ S2 ~~~ S3 ~~~ S4 ~~~ S5
    end

    DAG_Only -->|"SCM extends DAG<br/>with quantitative layer"| Full_SCM

    style DAG_Only fill:#e3f2fd,stroke:#0984e3,color:#2d3436
    style Full_SCM fill:#fce4ec,stroke:#e94560,color:#2d3436
```

### What Each Rung Requires

| Pearl's Rung | Question Type | What It Requires | DAG Sufficient? |
|-------------|--------------|-----------------|----------------|
| **Rung 1** — Association | "What is?" P(Y\|X) | Joint distribution | ✅ With data |
| **Rung 2** — Intervention | "What if I do X?" P(Y\|do(X)) | DAG + adjustment formulae | ⚠️ Partially (needs identifiability) |
| **Rung 3** — Counterfactual | "What if X had been different?" | Full SCM (F + P(U) + abduction) | ❌ Insufficient |

Rung 3 — the *Why?* rung — is where the DAG-only pipeline from `building_causal_DAG.md` reaches its limit. We need the full SCM.

---

## Research Landscape

### Deep Structural Causal Models (DSCMs)

**Pawlowski, Castro, and Glocker (NeurIPS 2020)** introduced DSCMs — the first framework enabling tractable counterfactual inference across all three rungs of Pearl's hierarchy. They use normalizing flows and variational inference to model causal mechanisms as deep generative models. The critical innovation is enabling **abduction** — inferring the exogenous noise posterior $P(\mathbf{U} \mid \mathbf{X} = \mathbf{x})$ — which is the step that makes counterfactual reasoning mechanically possible. Demonstrated on Morpho-MNIST and brain MRI.

A comprehensive IJCAI 2024 survey classifies DSCMs by their deep generative component (VAEs, normalizing flows, GANs, diffusion models) and their identifiability guarantees for L3 (counterfactual) queries.

### Neural Causal Models (NCMs)

**Xia, Lee, Bengio, and Bareinboim (NeurIPS 2021)** defined Neural Causal Models — SCMs where causal mechanisms are parameterized by feedforward neural networks. They proved NCMs are as expressive as arbitrary SCMs (L3-consistent). In a follow-up at **ICLR 2023**, Xia, Pan, and Bareinboim extended NCMs to counterfactual identification and estimation using GAN-based implementations, providing the first general counterfactual estimation technique that handles arbitrary combinations of observational and experimental data with unobserved confounders.

### Causal Modelling Agent (CMA)

**Brown et al. (ICLR 2024)** introduced the Causal Modelling Agent — the closest existing system to a full agentic SCM construction pipeline. The CMA combines LLM metadata-based reasoning with data-driven DSCMs. An LLM proposes an initial causal graph, a DSCM is fitted to data based on that graph, and the LLM acts as a critic to propose amendments in global and local phases. The framework maintains a **memory of changes and their impact on model fit** — a pattern closely related to the validation log used in the DAG pipeline from `building_causal_DAG.md`, but extended to track quantitative model fit rather than purely structural properties.

### Agentic Stream of Thought (ASoT)

**ASoT (ScienceDirect, 2025)** orchestrates multiple smaller open-source LLMs for causal discovery through hierarchical query decomposition, dual-stream processing of competing causal hypotheses (affirmative vs. negative), and two-tiered consensus mechanisms (Delphi protocol and Ensemble Synthesis). While focused on DAG discovery, the dual-stream pattern is directly transferable to SCM equation specification — where competing functional forms can be debated by parallel agents.

### Causal-Copilot

**Wang et al. (arXiv, April 2025)** built the most complete autonomous causal analysis agent to date. It automates the full pipeline — data preprocessing, algorithm selection, causal discovery, causal inference, counterfactual estimation — orchestrated by an LLM. It integrates 20+ causal analysis methods and generates PDF reports. Causal-Copilot demonstrates that LLM-orchestrated causal workflows are viable at production scale, though it treats SCM as a downstream tool rather than a construction target.

### Causal-Aware LLM Agents

**Kirubanandan et al. (PHM Conference, 2025)** propose a dual-agent architecture for predictive health management where one agent constructs localized causal structures from case-based retrieval and a second agent simulates interventions and counterfactual effects. This demonstrates the **localized SCM** pattern — building small, context-specific SCMs rather than attempting a global model — which is a practical strategy for production systems.

### CausalRAG and HugRAG

**CausalRAG (Wang et al., ACL Findings 2025)** — also referenced in `building_causal_DAG.md` under Paradigm 3 — integrates causal graphs into the retrieval process. While CausalRAG focuses on using causal structure for better retrieval, it demonstrates a principle critical for SCM construction: causal knowledge can be extracted, stored, and traversed in graph form, with retrieval precision improving when it follows causal pathways rather than pure semantic similarity.

**HugRAG (2025)** pushes this further with hierarchical causal knowledge graphs that explicitly organize retrieved knowledge along causal chains — a natural foundation for retrieving functional form information during equation specification.

### Language Agents Meet Causality

**Gou et al. (2024)** propose a framework bridging LLMs and Causal World Models (CWMs). The CWM acts as a simulator that the LLM queries — enabling causal inference and planning via Monte Carlo Tree Search over the causal model. This is the clearest demonstration that LLM + SCM integration enables capabilities neither component achieves alone.

---

## How SCM Construction Differs from DAG Construction

The DAG pipeline from `building_causal_DAG.md` outputs a validated directed acyclic graph with confidence scores and an evidence map. The SCM pipeline starts where that pipeline ends — but introduces fundamentally different challenges.

### Side-by-Side Comparison

| Dimension | DAG Pipeline (`building_causal_DAG.md`) | SCM Pipeline (this document) |
|-----------|----------------------------------------|------------------------------|
| **Core output** | Directed graph (nodes + edges + confidence) | Graph + structural equations + noise distributions |
| **Knowledge type** | Qualitative: "X causes Y" (confidence: 0.85) | Quantitative: "Y = 0.7X + 0.3Z + ε, ε ~ N(0, 0.4)" |
| **Data requirements** | Can work from text alone (Paradigm 1 in DAG doc) | Requires observational data to fit equations; text alone is insufficient |
| **Agent count** | 6 agents (Ingester → Variable Extractor → Causal Extractor → DAG Assembler → Validator → Reporter) | 6 DAG agents + 4–5 new SCM-specific agents |
| **Validation target** | Structural (acyclicity, connectivity, degree bounds) + Semantic (transitivity, domain coherence, evidence coverage) | All DAG checks + model fit (AIC/BIC), intervention prediction, counterfactual plausibility |
| **Iteration loop** | Refine edges and orientations | Refine edges + functional forms + noise distributions |
| **Memory schema** | `CausalDAGState` (edges, confidence, evidence map) | `CausalDAGState` + equations, parameters, exogenous distributions, abduction models, fit history |
| **Primary LLM role** | Edge extraction, orientation, validation | Additionally: functional form proposal, parameter initialization guidance, counterfactual interpretation |
| **RAG usage** | Retrieve evidence for edge existence | Additionally: retrieve domain-specific functional forms, known equations, mechanism types |
| **Identifiability concern** | Markov equivalence class (multiple DAGs fit same data) | L3-identifiability (can counterfactuals be uniquely determined?) |

### The Three Additional Challenges for SCM

The DAG pipeline in `building_causal_DAG.md` solves the *graph structure* problem. SCM construction adds three challenges that the DAG pipeline does not face:

```mermaid
flowchart TD
    subgraph DAG_Challenges["DAG Challenges<br/>(solved in building_causal_DAG.md)"]
        DC1["Variable identification"]
        DC2["Edge direction<br/>(orientation)"]
        DC3["Hidden confounders"]
        DC4["Cycle prevention"]
        DC5["Markov equivalence<br/>resolution"]
    end

    subgraph SCM_New["Additional SCM Challenges<br/>(this document)"]
        SC1["<b>Challenge 1</b><br/>Structural equation<br/>specification<br/><i>What is f_i?</i>"]
        SC2["<b>Challenge 2</b><br/>Exogenous distribution<br/>modeling<br/><i>What is P(U)?</i>"]
        SC3["<b>Challenge 3</b><br/>Counterfactual<br/>validation<br/><i>How to test Rung 3?</i>"]
    end

    DAG_Challenges -->|"DAG is prerequisite<br/>input to SCM"| SCM_New

    style DAG_Challenges fill:#e3f2fd,stroke:#0984e3,color:#2d3436
    style SCM_New fill:#fce4ec,stroke:#e94560,color:#2d3436
    style SC1 fill:#fab1a0,stroke:#e17055,color:#2d3436
    style SC2 fill:#fab1a0,stroke:#e17055,color:#2d3436
    style SC3 fill:#fab1a0,stroke:#e17055,color:#2d3436
```

**Challenge 1 — Structural Equation Specification**: For every edge $X_i \to X_j$ in the DAG, the SCM requires a concrete function $X_j = f_j(\text{Pa}(X_j), U_j)$. This could be linear, polynomial, sigmoidal, threshold, or a neural network. The DAG pipeline only needed to determine *whether* an edge exists; the SCM pipeline must determine *how that edge works quantitatively*.

**Challenge 2 — Exogenous Distribution Modeling**: Each structural equation has a noise term $U_i$ with a specific distribution. For counterfactual reasoning, we need to perform *abduction* — inferring the individual-specific noise $P(U_i \mid \text{observed data})$. This requires either invertible mechanisms (normalizing flows) for exact abduction or amortized variational inference for approximation.

**Challenge 3 — Counterfactual Validation**: The DAG pipeline (per `building_causal_DAG.md`) validates against structural checks (acyclicity), semantic checks (domain coherence), and optionally statistical checks (conditional independence). But counterfactual predictions concern *unobservable* alternate worlds — we can never directly observe what *would have* happened. This makes validation fundamentally harder than anything in the DAG pipeline.

### How the DAG State Schema Extends to SCM

The `building_causal_DAG.md` document defines a `CausalDAGState` for LangGraph:

```python
# From building_causal_DAG.md — DAG state
class CausalDAGState(TypedDict):
    corpus_chunks: list[str]
    embeddings_indexed: bool
    candidate_nodes: list[str]
    extracted_edges: list[dict]       # {source, target, confidence, evidence}
    dag: nx.DiGraph
    iteration: int
    max_iterations: int
    validation_log: list[dict]
    confidence_scores: dict[str, float]
    evidence_map: dict[str, list[str]]
```

The SCM state *wraps and extends* this:

```python
# SCM state — extends CausalDAGState
class SCMState(TypedDict):
    # ── Inherited from CausalDAGState ──────────────────────
    corpus_chunks: list[str]
    embeddings_indexed: bool
    candidate_nodes: list[str]
    extracted_edges: list[dict]
    dag: nx.DiGraph
    dag_iteration: int
    dag_max_iterations: int
    dag_validation_log: list[dict]
    confidence_scores: dict[str, float]
    evidence_map: dict[str, list[str]]

    # ── New: Structural Equations ──────────────────────────
    structural_equations: dict[str, dict]  # node → {form, params, fitted}
    # e.g. {"Y": {"form": "linear", "params": {"β_X": 0.7, "β_Z": 0.3},
    #              "fitted": True, "r_squared": 0.82}}
    candidate_forms: dict[str, list[str]]  # node → candidate functional forms
    equation_fit_scores: dict[str, float]  # node → AIC/BIC score

    # ── New: Exogenous Variables ───────────────────────────
    exogenous_distributions: dict[str, dict]  # node → {dist_type, params}
    # e.g. {"Y": {"dist_type": "normal", "params": {"mu": 0, "sigma": 0.4}}}
    abduction_models: dict[str, Any]  # node → trained encoder / inverse

    # ── New: SCM-Specific Validation ───────────────────────
    scm_iteration: int
    scm_max_iterations: int
    model_fit_history: list[dict]     # fit metric trajectory
    intervention_tests: list[dict]    # predicted vs actual intervention results
    counterfactual_log: list[dict]    # counterfactual queries + results
    observational_data: Any           # DataFrame or path to data
```

---

## Proposed Architecture: Agentic SCM Builder

### Design Principle: DAG-First, Then Quantify

The architecture uses the DAG pipeline from `building_causal_DAG.md` as **Stage 1** — producing a validated Causal DAG — then hands that DAG to the SCM-specific agents in **Stage 2**. A third **Stage 3** handles validation, intervention simulation, and counterfactual reasoning.

```mermaid
flowchart TB
    USER(("User Query")) --> SUPERVISOR

    subgraph ORCHESTRATION["Orchestration Layer<br/>(LangGraph StateGraph)"]
        SUPERVISOR["Supervisor Agent"]
        PLANNER["Planning Agent"]
        MEMORY[("Persistent State<br/>(SCMState)")]
    end

    subgraph STAGE1["Stage 1 · DAG Construction<br/>(from building_causal_DAG.md)"]
        direction TB
        A1["🔍 Ingester"]
        A2["📋 Variable Extractor"]
        A3["⚡ Causal Extractor"]
        A4["🏗️ DAG Assembler"]
        A5["✅ DAG Validator"]
        A1 --> A2 --> A3 --> A4 --> A5
        A5 -.->|"refine loop"| A3
    end

    subgraph STAGE2["Stage 2 · SCM Construction<br/>(new in this document)"]
        direction TB
        B1["📐 Equation Specification<br/>Agent"]
        B2["📊 Parameter Estimation<br/>Agent"]
        B3["🎲 Exogenous Distribution<br/>Agent"]
        B4["🔄 Abduction Mechanism<br/>Agent"]
        B1 --> B2 --> B3 --> B4
        B4 -.->|"refine loop"| B1
    end

    subgraph STAGE3["Stage 3 · Validation & Inference"]
        direction TB
        C1["🧪 Model Fit Validator"]
        C2["💉 Intervention Simulator"]
        C3["🔮 Counterfactual Reasoner"]
        C4["📝 Report Generator"]
        C1 --> C2 --> C3 --> C4
    end

    subgraph KNOWLEDGE["Knowledge Retrieval Layer"]
        direction TB
        VEC[("Vector Store<br/>(ChromaDB /<br/>Pinecone)")]
        GRAPH[("Causal KG<br/>(Neo4j /<br/>FalkorDB)")]
        DATA[("Observational<br/>Data Store")]
    end

    SUPERVISOR --> PLANNER
    PLANNER --> STAGE1
    PLANNER --> STAGE2
    PLANNER --> STAGE3

    KNOWLEDGE <--> STAGE1
    KNOWLEDGE <--> STAGE2
    KNOWLEDGE <--> STAGE3

    MEMORY <--> STAGE1
    MEMORY <--> STAGE2
    MEMORY <--> STAGE3

    STAGE1 -->|"Validated DAG"| STAGE2
    STAGE2 -->|"Candidate SCM"| STAGE3
    STAGE3 -->|"Refinement signals"| STAGE2

    STAGE3 --> USER

    style ORCHESTRATION fill:#dfe6e9,stroke:#636e72,color:#2d3436
    style STAGE1 fill:#e3f2fd,stroke:#0984e3,color:#2d3436
    style STAGE2 fill:#fce4ec,stroke:#e94560,color:#2d3436
    style STAGE3 fill:#ede7f6,stroke:#6c5ce7,color:#2d3436
    style KNOWLEDGE fill:#ffeaa7,stroke:#fdcb6e,color:#2d3436
```

### LangGraph Control Flow

Extending the control flow from `building_causal_DAG.md` — which routes from `validate_dag` back to `extract_causal_relations` when issues are found — the SCM pipeline adds a second refinement loop over the quantitative layer:

```mermaid
graph TD
    START(("__start__")) --> ingest["ingest_corpus"]

    subgraph DAG_Phase["Stage 1: DAG (from building_causal_DAG.md)"]
        ingest --> extract_vars["extract_variables"]
        extract_vars --> extract_causal["extract_causal_relations"]
        extract_causal --> assemble["assemble_dag"]
        assemble --> validate_dag["validate_dag"]
        validate_dag --> dag_route{"dag_route"}
        dag_route -- "issues & iter < max" --> extract_causal
        dag_route -- "pass" --> dag_done["dag_finalized"]
    end

    subgraph SCM_Phase["Stage 2: SCM (new)"]
        dag_done --> specify_eqs["specify_equations"]
        specify_eqs --> estimate_params["estimate_parameters"]
        estimate_params --> model_exogenous["model_exogenous_dist"]
        model_exogenous --> build_abduction["build_abduction_mechanism"]
        build_abduction --> validate_scm["validate_scm"]
        validate_scm --> scm_route{"scm_route"}
        scm_route -- "poor fit & iter < max" --> specify_eqs
        scm_route -- "pass" --> scm_done["scm_finalized"]
    end

    subgraph Inference_Phase["Stage 3: Inference"]
        scm_done --> intervention["simulate_interventions"]
        intervention --> counterfactual["run_counterfactuals"]
        counterfactual --> report["generate_report"]
    end

    report --> END(("__end__"))

    style START fill:#6c5ce7,color:#fff
    style END fill:#6c5ce7,color:#fff
    style dag_route fill:#fdcb6e,stroke:#f39c12,color:#2d3436
    style scm_route fill:#fdcb6e,stroke:#f39c12,color:#2d3436
    style DAG_Phase fill:#e3f2fd,stroke:#0984e3,color:#2d3436
    style SCM_Phase fill:#fce4ec,stroke:#e94560,color:#2d3436
    style Inference_Phase fill:#ede7f6,stroke:#6c5ce7,color:#2d3436
```

---

### Stage 2 Agent Specifications (New)

#### Equation Specification Agent

This is the most novel and challenging agent — nothing in the DAG pipeline has an equivalent. The DAG pipeline's **Causal Extractor** (agent ③ in `building_causal_DAG.md`) asks "Does X cause Y?" and outputs a confidence score. The Equation Specification Agent asks the harder question: "How does X affect Y, quantitatively?"

**Strategy: Retrieve → Propose → Compete → Select**

```mermaid
flowchart TD
    EDGE["Edge from DAG:<br/>X → Y with confidence 0.85<br/>and evidence passages"] --> RETRIEVE

    subgraph RETRIEVE["Retrieve Domain Knowledge"]
        R1["GraphRAG: Traverse causal KG<br/>for known mechanisms<br/>involving X and Y"]
        R2["RAG: Retrieve papers /<br/>domain texts on X–Y<br/>functional relationship"]
        R1 ~~~ R2
    end

    RETRIEVE --> PROPOSE

    subgraph PROPOSE["Propose Candidate Forms"]
        P1["Linear: Y = βX + ε"]
        P2["Polynomial: Y = β₁X + β₂X² + ε"]
        P3["Monotonic nonlinear:<br/>Y = α · sigmoid(βX) + ε"]
        P4["Neural (DSCM):<br/>Y = NN(X, U)"]
    end

    PROPOSE --> COMPETE

    subgraph COMPETE["Compete on Data"]
        F1["Fit each candidate<br/>to observational data"]
        F2["Score: AIC / BIC /<br/>cross-validated RMSE"]
        F3["Domain plausibility<br/>(LLM judge)"]
        F1 --> F2 --> F3
    end

    COMPETE --> SELECT["Select winning<br/>functional form"]
    SELECT --> STORE["Store in<br/>structural_equations<br/>state field"]

    style EDGE fill:#e3f2fd,stroke:#0984e3,color:#2d3436
    style RETRIEVE fill:#ffeaa7,stroke:#fdcb6e,color:#2d3436
    style PROPOSE fill:#dfe6e9,stroke:#636e72,color:#2d3436
    style COMPETE fill:#fce4ec,stroke:#e94560,color:#2d3436
    style SELECT fill:#00b894,stroke:#dfe6e9,color:#fff
```

Note the critical RAG asymmetry: the same vector store used by the DAG pipeline's Causal Extractor is now queried for a different kind of information — not "does X cause Y?" but "what is the functional form of X's effect on Y?"

#### Parameter Estimation Agent

Fits parameters for the chosen functional forms using computational tools:

- For parametric models: `scipy.optimize`, `statsmodels`
- For semi-parametric models: `PyGAM` (generalized additive models)
- For neural mechanisms: PyTorch / Pyro training loops
- Computes confidence intervals via bootstrap or Bayesian posterior

#### Exogenous Distribution Agent

Models the noise distribution $P(U_i)$ for each structural equation:

1. Compute residuals from fitted equations
2. Test distributional assumptions (normality, heavy tails, multimodality)
3. For DSCM approaches: learn exogenous distributions implicitly through normalizing flows
4. Check independence between exogenous variables (Markovian assumption)

#### Abduction Mechanism Agent

Builds the machinery needed for Pearl's three-step counterfactual procedure:

```mermaid
sequenceDiagram
    participant OBS as Observed Data<br/>(V = v)
    participant ABD as Abduction<br/>(Step 1)
    participant ACT as Action<br/>(Step 2)
    participant PRD as Prediction<br/>(Step 3)

    rect rgba(108, 117, 125, 0.15)
    Note over OBS,ABD: Step 1 · Abduction
    OBS->>ABD: Given observed V = v
    ABD->>ABD: Infer P(U | V = v)<br/>using inverse mechanisms<br/>or variational encoder
    ABD-->>ACT: Inferred û
    end

    rect rgba(9, 132, 227, 0.15)
    Note over ACT,ACT: Step 2 · Action
    ACT->>ACT: Modify SCM:<br/>replace f_X with do(X = x')
    end

    rect rgba(162, 155, 254, 0.15)
    Note over ACT,PRD: Step 3 · Prediction
    ACT->>PRD: Forward-propagate<br/>through modified SCM<br/>with inferred û
    PRD-->>PRD: Compute Y_{X=x'}(û)<br/>= counterfactual outcome
    end
```

The agent selects the abduction approach per variable:

| Mechanism Type | Abduction Method | Exactness | When to Use |
|---------------|-----------------|-----------|-------------|
| Invertible (normalizing flow) | Direct inverse | Exact | When invertibility is achievable |
| Non-invertible parametric | Algebraic solve for U given V and parents | Exact (if solvable) | Simple functional forms |
| Non-invertible neural | Amortized variational encoder | Approximate | Complex, high-dimensional mechanisms |
| GAN-based (NCM-style) | Adversarial posterior estimation | Approximate | When unobserved confounders exist |

---

### Role of RAG and GraphRAG: DAG vs. SCM

The DAG pipeline in `building_causal_DAG.md` uses RAG in one primary mode: **edge evidence retrieval** — the Causal Extractor agent retrieves relevant chunks for each variable pair and asks the LLM to assess the causal relationship. The SCM pipeline adds three new retrieval modes:

```mermaid
flowchart TD
    subgraph DAG_RAG["RAG in DAG Pipeline<br/>(building_causal_DAG.md)"]
        DR1["<b>Mode 1: Edge Evidence</b><br/>Retrieve passages supporting<br/>'Does X cause Y?'"]
    end

    subgraph SCM_RAG["RAG in SCM Pipeline<br/>(this document)"]
        SR1["<b>Mode 1: Edge Evidence</b><br/><i>(inherited from DAG)</i>"]
        SR2["<b>Mode 2: Mechanism Retrieval</b><br/>Retrieve known functional forms<br/>'What is the dose-response<br/>curve for X → Y?'"]
        SR3["<b>Mode 3: Parameter Priors</b><br/>Retrieve published parameter<br/>estimates as initialization<br/>'Typical β for X → Y is 0.3–0.7'"]
        SR4["<b>Mode 4: Counterfactual Grounding</b><br/>Retrieve domain knowledge<br/>to validate counterfactual<br/>plausibility"]
    end

    DAG_RAG --> SCM_RAG

    style DAG_RAG fill:#e3f2fd,stroke:#0984e3,color:#2d3436
    style SCM_RAG fill:#fce4ec,stroke:#e94560,color:#2d3436
```

GraphRAG is particularly valuable for **Mode 2 (Mechanism Retrieval)** because functional forms often require multi-hop reasoning over causal pathways. For example, understanding that "insulin → β-cell function → glucose regulation" follows a Hill equation requires traversing the causal knowledge graph to find the intermediate mechanism, then retrieving pharmacokinetic literature on that specific pathway. This goes well beyond the single-hop retrieval sufficient for the DAG pipeline's edge-existence queries.

---

### Validation: Extending the DAG Validator

The `building_causal_DAG.md` document specifies three tiers of validation checks — Structural, Semantic, and Statistical (optional). The SCM validator inherits all three tiers, expands the Statistical tier substantially, and adds a fourth tier:

```mermaid
flowchart TD
    subgraph Structural["Structural Checks<br/>(inherited from DAG pipeline)"]
        S1["Acyclicity ✓"]
        S2["Connectivity ✓"]
        S3["Degree bounds ✓"]
    end

    subgraph Semantic["Semantic Checks<br/>(inherited + extended)"]
        SE1["Transitivity consistency ✓"]
        SE2["Domain coherence ✓"]
        SE3["Evidence coverage ✓"]
        SE4["<b>NEW:</b> Equation plausibility<br/><i>Do functional forms<br/>match domain expectations?</i>"]
    end

    subgraph Statistical["Statistical Checks<br/>(greatly expanded)"]
        ST1["Conditional independence ✓"]
        ST2["<b>NEW:</b> Observational fit<br/><i>R², AIC/BIC for<br/>each equation</i>"]
        ST3["<b>NEW:</b> Residual diagnostics<br/><i>Independence, normality,<br/>homoscedasticity</i>"]
        ST4["<b>NEW:</b> Intervention prediction<br/><i>If experimental data exists,<br/>compare do(X) predictions</i>"]
    end

    subgraph Counterfactual["Counterfactual Checks<br/>(entirely new tier)"]
        CF1["Abduction reconstruction<br/><i>Can we recover observed<br/>data from inferred U?</i>"]
        CF2["Sensitivity analysis<br/><i>How stable are counterfactuals<br/>to SCM specification?</i>"]
        CF3["Expert plausibility review<br/><i>Do counterfactual results<br/>pass domain sanity checks?</i>"]
    end

    Structural --> PASS{"All tiers<br/>acceptable?"}
    Semantic --> PASS
    Statistical --> PASS
    Counterfactual --> PASS
    PASS -- Yes --> DONE["✅ SCM Accepted"]
    PASS -- No --> LOOP["🔄 Refine equations,<br/>distributions, or DAG"]

    style Counterfactual fill:#ede7f6,stroke:#6c5ce7,color:#2d3436
    style SE4 fill:#fab1a0,stroke:#e17055,color:#2d3436
    style ST2 fill:#fab1a0,stroke:#e17055,color:#2d3436
    style ST3 fill:#fab1a0,stroke:#e17055,color:#2d3436
    style ST4 fill:#fab1a0,stroke:#e17055,color:#2d3436
```

---

## The Hybrid Approach: Parametric + Neural Mechanisms

A practical SCM builder should not commit to a single mechanism type. The Equation Specification Agent should deploy a spectrum, guided by RAG-retrieved domain knowledge:

```mermaid
flowchart LR
    subgraph Spectrum["Mechanism Spectrum"]
        direction TB
        PARAM["<b>Parametric</b><br/>Y = βX + ε<br/><i>Strong domain knowledge</i><br/><i>Interpretable, auditable</i>"]
        SEMI["<b>Semi-parametric</b><br/>Y = s(X) + ε<br/><i>Known shape class,<br/>flexible fit (GAMs, splines)</i>"]
        NEURAL["<b>Fully Neural (DSCM)</b><br/>Y = NN(X, U)<br/><i>Weak domain knowledge</i><br/><i>Maximum flexibility</i>"]
    end

    PARAM ---|"Increasing flexibility →<br/>← Increasing interpretability"| SEMI
    SEMI ---|"Increasing flexibility →<br/>← Increasing interpretability"| NEURAL

    RAG_Signal["RAG retrieval signals<br/>which level is appropriate<br/>per edge"] --> Spectrum

    style PARAM fill:#00b894,stroke:#dfe6e9,color:#fff
    style SEMI fill:#fdcb6e,stroke:#f39c12,color:#2d3436
    style NEURAL fill:#e17055,stroke:#dfe6e9,color:#fff
    style RAG_Signal fill:#ffeaa7,stroke:#fdcb6e,color:#2d3436
```

The decision is data- and knowledge-driven per edge: if RAG retrieves well-established functional forms from domain literature, use parametric; if the relationship is known to be nonlinear but the exact form is uncertain, use semi-parametric; if the mechanism is poorly understood, fall back to neural.

---

## Connecting to the Iterative RL Loop

The `building_causal_DAG.md` document includes an advanced section on an **Iterative Causal-Aware RL Loop** where the SCM is used to generate interventional experiments that refine the DAG. The SCM pipeline proposed here provides the quantitative substrate that makes this loop operational:

```mermaid
sequenceDiagram
    participant DAG as DAG Pipeline<br/>(building_causal_DAG.md)
    participant SCM as SCM Pipeline<br/>(this document)
    participant ENV as Environment / Data

    rect rgba(108, 117, 125, 0.15)
    Note over DAG,SCM: Phase 1 · Qualitative Structure
    DAG->>DAG: Build causal DAG<br/>from text corpus
    DAG-->>SCM: Validated DAG + evidence map
    end

    rect rgba(233, 69, 96, 0.15)
    Note over SCM,SCM: Phase 2 · Quantitative Equations
    SCM->>SCM: Specify equations,<br/>fit parameters,<br/>model exogenous dist
    SCM->>ENV: do-operator experiments<br/>(intervention simulation)
    ENV-->>SCM: Validate / refute<br/>predicted outcomes
    end

    rect rgba(162, 155, 254, 0.15)
    Note over SCM,DAG: Phase 3 · Feedback
    SCM-->>DAG: Intervention results<br/>may refute DAG edges
    DAG->>DAG: Revise structure
    DAG-->>SCM: Updated DAG
    SCM->>SCM: Re-fit equations
    end

    Note over DAG,ENV: Loop until convergence
```

The DAG pipeline from `building_causal_DAG.md` described this loop conceptually using an LLM Agent ↔ Environment ↔ SCM sequence diagram. The SCM pipeline makes the **Adapting Phase** concrete — the SCM now has actual structural equations that generate testable interventional predictions, which either confirm or refute the DAG structure.

---

## Tools & Stack Recommendations

Extending the stack table from `building_causal_DAG.md`:

| Component | DAG Pipeline (from DAG doc) | Additional for SCM |
|-----------|---------------------------|-------------------|
| **Orchestration** | LangGraph + LangSmith | Same (extended state schema) |
| **Graph Backend** | NetworkX → Neo4j | Same + equation metadata per edge/node |
| **Vector Store** | ChromaDB → Pinecone | Same (additional indexing for mechanism literature) |
| **LLM** | Claude Sonnet (extraction) + Opus (validation) | + Opus for equation specification & counterfactual reasoning |
| **Statistical CD** | `causal-learn` (PC, GES) | `DoWhy`, `EconML`, `CausalPy` |
| **Deep SCM training** | — | PyTorch, Pyro, `nflows`, `zuko` |
| **Parameter estimation** | — | `scipy.optimize`, `statsmodels`, `PyGAM` |
| **Tracking** | MLflow + LangSmith | Same (+ model fit metrics tracking) |
| **Visualization** | Graphviz / pyvis | + equation rendering, counterfactual plots |

---

## Open Challenges and Research Gaps

### 1. No End-to-End Agentic SCM Builder Exists

As of early 2026, **no published system provides a fully agentic end-to-end SCM construction pipeline from text**. The closest works are:
- CMA (Brown et al., ICLR 2024) — iterates between LLM and DSCM but requires pre-specified variable sets and structured data
- Causal-Copilot (Wang et al., 2025) — automates analysis but treats SCM as downstream, not as a construction target
- The DAG pipeline pattern from `building_causal_DAG.md` — solves the qualitative half of the problem

The gap is in connecting **LLM-driven knowledge extraction** (the strength of the DAG pipeline) with **automated structural equation specification and neural mechanism training** in a single orchestrated workflow.

### 2. Automated Functional Form Selection Is Underexplored

Current agentic causal discovery focuses almost exclusively on graph structure. The problem of automatically choosing between linear, polynomial, neural, etc. functional forms for each structural equation — guided by domain knowledge retrieval — has no established solution in the multi-agent literature.

### 3. Counterfactual Validation Remains Fundamentally Hard

Counterfactual outcomes are never directly observed. The IJCAI 2024 DSCM survey underscores that the field lacks standardized benchmarks for counterfactual accuracy beyond synthetic datasets with known ground truth. Production systems will need to rely heavily on sensitivity analysis and domain expert review.

### 4. Text-to-Equation Is a Frontier Problem

The DAG pipeline from `building_causal_DAG.md` works with qualitative causal language ("X causes Y", "increasing X leads to Y"). Extracting *quantitative functional forms* from text ("X has a logarithmic effect on Y with saturation around X=100") requires a fundamentally different extraction capability that LLMs may possess but has not been systematically evaluated.

### 5. Scalability

Current DSCM and NCM implementations work on relatively small graphs (3–10 variables). The DAG pipeline in `building_causal_DAG.md` already notes fragility beyond ~15 variables for direct extraction. SCM construction compounds this — each additional variable adds an equation, a noise model, and an abduction mechanism.

---

## Summary: From DAG to SCM Pipeline

```mermaid
flowchart LR
    subgraph DAG_Doc["building_causal_DAG.md"]
        D_IN["Text Corpus"] --> D_PIPE["6-Agent DAG Pipeline<br/>(Ingest → Extract → Assemble<br/>→ Validate → Report)"]
        D_PIPE --> D_OUT["Validated Causal DAG<br/>+ confidence scores<br/>+ evidence map"]
    end

    subgraph SCM_Doc["building_SCM.md<br/>(this document)"]
        S_IN["Validated DAG<br/>+ Observational Data"] --> S_PIPE["4-Agent SCM Pipeline<br/>(Equations → Parameters<br/>→ Exogenous → Abduction)"]
        S_PIPE --> S_VAL["3-Agent Validation<br/>(Fit → Intervention<br/>→ Counterfactual)"]
        S_VAL --> S_OUT["Full SCM<br/>capable of answering<br/>'Why?' questions<br/>(Pearl's Rung 3)"]
    end

    D_OUT -->|"DAG is prerequisite<br/>input to SCM"| S_IN

    style DAG_Doc fill:#e3f2fd,stroke:#0984e3,color:#2d3436
    style SCM_Doc fill:#fce4ec,stroke:#e94560,color:#2d3436
    style D_OUT fill:#74b9ff,stroke:#0984e3,color:#2d3436
    style S_OUT fill:#ff7675,stroke:#e94560,color:#fff
```

The two documents together describe a **complete path from raw text to counterfactual reasoning**: `building_causal_DAG.md` handles the qualitative structure (Rungs 1–2), and this document handles the quantitative machinery (Rung 3). The architecture is modular — the DAG pipeline can run independently, and the SCM pipeline consumes its output — preserving the separation of concerns that makes agentic systems debuggable and iterable.

---

## References

### Foundational SCM / DSCM / NCM

- Pawlowski, N., Castro, D. C., & Glocker, B. (2020). Deep Structural Causal Models for Tractable Counterfactual Inference. *NeurIPS 2020*. [arXiv:2006.06485](https://arxiv.org/abs/2006.06485)
- Xia, K., Lee, K.-Z., Bengio, Y., & Bareinboim, E. (2021). The Causal-Neural Connection: Expressiveness, Learnability, and Inference. *NeurIPS 2021*.
- Xia, K., Pan, Y., & Bareinboim, E. (2023). Neural Causal Models for Counterfactual Identification and Estimation. *ICLR 2023*. [arXiv:2210.00035](https://arxiv.org/abs/2210.00035)
- DSCM Survey — Learning Structural Causal Models through Deep Generative Models: Methods, Guarantees, and Challenges. *IJCAI 2024*. [Proceedings](https://www.ijcai.org/proceedings/2024/0907.pdf)
- Pearl, J. (2009). *Causality: Models, Reasoning and Inference* (2nd ed.). Cambridge University Press.
- Bareinboim, E. et al. (2022). On Pearl's Hierarchy and the Foundations of Causal Inference. *ACM*.

### Agentic Causal Systems

- Brown, D. C. et al. (2024). Causal Modelling Agents: Causal Graph Discovery through Synergising Metadata- and Data-Driven Reasoning. *ICLR 2024*. [OpenReview](https://openreview.net/pdf?id=pAoqRlTBtY)
- Wang, X. et al. (2025). Causal-Copilot: An Autonomous Causal Analysis Agent. *arXiv:2504.13263*. [GitHub](https://github.com/Lancelot39/Causal-Copilot)
- Han, K. et al. (2024). Causal Agent based on Large Language Model. *arXiv:2408.06849*. [arXiv](https://arxiv.org/abs/2408.06849v1)
- ASoT — Structured Knowledge-Based Causal Discovery: Agentic Streams of Thought. *ScienceDirect*, May 2025.
- Causal MAS Survey — A Survey of Large Language Model Multi-Agent Systems for Causal Inference. *arXiv:2509.00987* (2025).
- Kirubanandan, R. et al. (2025). Causal-Aware LLM Agents for PHM Co-Pilots. *PHM Conference 2025*.

### LLM + Causality Integration

- Gou, J. et al. (2024). Language Agents Meet Causality — Bridging LLMs and Causal World Models. [Project](https://j0hngou.github.io/LLMCWM/)
- Du, H. et al. (2025). Causal Discovery through Synergizing LLM and Data-Driven Reasoning (LLM-CD). *KDD 2025*.
- IJCAI 2025 Survey — Large Language Models for Causal Discovery: Current Landscape and Future Directions. [arXiv:2402.11068](https://arxiv.org/abs/2402.11068)

### RAG / GraphRAG for Causal Knowledge

- Wang, N. et al. (2025). CausalRAG: Integrating Causal Graphs into Retrieval-Augmented Generation. *ACL Findings 2025*. [ACL Anthology](https://aclanthology.org/2025.findings-acl.1165/)
- HugRAG — Hierarchical Causal Knowledge Graph Design for RAG. *arXiv*, 2025.

### Referenced Companion Document

- [building_causal_DAG.md](building_causal_DAG.md) — Building Causal DAGs from Text Corpora Using Agentic Workflows.
