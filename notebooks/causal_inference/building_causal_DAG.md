# Building Causal DAGs from Text Corpora Using Agentic Workflows

## Overview

Constructing causal Directed Acyclic Graphs (DAGs) from unstructured text is an emerging capability at the intersection of causal discovery, NLP, and agentic AI orchestration. This document synthesizes the current landscape — from practical architecture patterns to research-frontier methods — drawing on key recent work including DEMOCRITUS (Mahadevan, Dec 2025), ARCADIA, CausalRAG (ACL 2025), The Agentic Leash (Panda et al., Jan 2026), and the IJCAI 2025 survey on LLMs for causal discovery.

---

## Three Paradigms for Causal DAG Construction

The IJCAI 2025 survey identifies three distinct roles LLMs play in causal discovery:

1. **Direct causal inference** without observational data
2. **Post-hoc refinement** of statistically derived causal structures
3. **Integration of prior knowledge** into traditional statistical methods

For text corpora specifically, paradigm (1) is the most direct entry point, but the strongest results emerge from hybrid approaches that combine all three.

```mermaid
graph LR
    subgraph Paradigm1["1 · Direct Extraction"]
        direction TB
        T1[Text Corpus] --> LLM1["LLM Causal<br/>Extraction"]
        LLM1 --> G1["Candidate<br/>DAG"]
    end

    subgraph Paradigm2["2 · Post-hoc Refinement"]
        direction TB
        D2[Observational<br/>Data] --> STAT["Statistical CD<br/>(PC / GES / NOTEARS)"]
        STAT --> DRAFT["Draft<br/>CPDAG"]
        DRAFT --> LLM2["LLM Refines<br/>Orientations"]
        LLM2 --> G2["Refined<br/>DAG"]
    end

    subgraph Paradigm3["3 · Prior Knowledge Integration"]
        direction TB
        T3[Text + Data] --> LLM3["LLM Encodes<br/>Domain Priors"]
        LLM3 --> PRIOR["Soft / Hard<br/>Constraints"]
        PRIOR --> STAT3["Constrained<br/>Statistical CD"]
        STAT3 --> G3["Knowledge-Informed<br/>DAG"]
    end

    style Paradigm1 fill:#1a1a2e,stroke:#e94560,color:#eee
    style Paradigm2 fill:#1a1a2e,stroke:#0f3460,color:#eee
    style Paradigm3 fill:#1a1a2e,stroke:#16213e,color:#eee
```

---

## Paradigm 1 — Direct LLM-Based Causal Extraction

The baseline approach uses LLMs to directly extract `(cause, effect)` pairs from text. Variables along with their descriptive texts are provided as input to an LLM, which performs causal discovery by identifying direct relationships after comprehending the semantic meaning of each variable (Ban et al., 2023).

### Pipeline

```mermaid
flowchart TD
    A["Raw Text Corpus"] --> B["Chunking &<br/>Preprocessing"]
    B --> C["Structured Prompt:<br/>'Extract causal statements<br/>of the form X → Y'"]
    C --> D["LLM Extraction<br/>(per chunk)"]
    D --> E["Aggregate Edges<br/>Across Chunks"]
    E --> F{"Cycles<br/>Detected?"}
    F -- Yes --> G["Cycle Removal<br/>(topological sort /<br/>edge-weight threshold)"]
    G --> H["Second LLM Pass:<br/>Validate & Orient"]
    F -- No --> H
    H --> I["Candidate<br/>Causal DAG"]

    style A fill:#2d3436,stroke:#dfe6e9,color:#dfe6e9
    style I fill:#00b894,stroke:#dfe6e9,color:#fff
```

### Limitations

- LLMs hallucinate edges and miss implicit causation
- Transitivity consistency is not guaranteed
- No grounding in observational evidence
- Fragile on its own for graphs with more than ~15 variables

---

## Paradigm 2 — Agentic Multi-Step Pipeline

This is where multi-agent orchestration (e.g., LangGraph) delivers the most value. The strongest recent systems all use iterative refinement loops with specialized agents.

### High-Level Agentic Architecture

```mermaid
flowchart TD
    START(("Start")) --> INGEST

    subgraph INGEST["① Corpus Ingester"]
        direction LR
        I1["Raw Docs"] --> I2["Chunk +<br/>Sentence Split"]
        I2 --> I3["Embed &<br/>Index in<br/>Vector Store"]
    end

    INGEST --> VAREX

    subgraph VAREX["② Variable Extractor Agent"]
        direction LR
        V1["NER + Noun Phrase<br/>Extraction"] --> V2["Deduplication &<br/>Canonicalization"]
        V2 --> V3["Candidate<br/>Node Set V"]
    end

    VAREX --> CAUSAL

    subgraph CAUSAL["③ Causal Relation Extractor Agent"]
        direction LR
        C1["For each pair<br/>(Xᵢ, Xⱼ) ∈ V×V"] --> C2["Retrieve evidence<br/>via RAG"]
        C2 --> C3["Prompt: 'Given this<br/>evidence, does Xᵢ<br/>cause Xⱼ?'"]
        C3 --> C4["Scored edge list<br/>with confidence"]
    end

    CAUSAL --> ASSEMBLY

    subgraph ASSEMBLY["④ DAG Assembler"]
        direction LR
        A1["Build weighted<br/>DiGraph"] --> A2["Enforce acyclicity<br/>(greedy cycle removal<br/>on lowest-weight edges)"]
        A2 --> A3["Threshold<br/>low-confidence<br/>edges"]
    end

    ASSEMBLY --> VALIDATE

    subgraph VALIDATE["⑤ Validator Agent"]
        direction TB
        VA1["Check acyclicity ✓"]
        VA2["Check transitivity<br/>consistency"]
        VA3["Check domain<br/>coherence"]
        VA4["Identify missing<br/>edges via retrieval"]
        VA1 --> VA2 --> VA3 --> VA4
    end

    VALIDATE --> DECISION{"Passes all<br/>criteria?"}
    DECISION -- "No (iteration < max)" --> CAUSAL
    DECISION -- "Yes" --> OUTPUT

    subgraph OUTPUT["⑥ Output & Reporting"]
        direction LR
        O1["Final DAG<br/>(NetworkX / Neo4j)"] --> O2["Confidence<br/>Scores per Edge"]
        O2 --> O3["Evidence Map<br/>(edge → source passages)"]
        O3 --> O4["Visualization<br/>& Export"]
    end

    OUTPUT --> END(("End"))

    style START fill:#6c5ce7,stroke:#a29bfe,color:#fff
    style END fill:#6c5ce7,stroke:#a29bfe,color:#fff
    style DECISION fill:#fdcb6e,stroke:#f39c12,color:#2d3436
```

---

### Reference Systems

#### DEMOCRITUS (Mahadevan, Dec 2025)

The most ambitious text-to-causal-model pipeline. A strong LLM acts as a discovery engine for domain topics, causal questions, and statements, followed by a Geometric Transformer layer that produces manifold embeddings over the resulting relational graph.

```mermaid
flowchart LR
    subgraph Module1["Module 1<br/>Topic Graph"]
        T1["Root<br/>Topic"] --> T2["LLM-guided<br/>BFS Expansion"]
        T2 --> T3["Topic<br/>Graph"]
    end

    subgraph Module2["Module 2<br/>Causal Questions"]
        Q1["Per-topic<br/>prompts"] --> Q2["Causal query<br/>candidates"]
    end

    subgraph Module3["Module 3<br/>Causal Statements"]
        S1["Structured<br/>prompts"] --> S2["'X causes Y'<br/>assertions"]
    end

    subgraph Module4["Module 4<br/>Triple Extraction"]
        E1["OpenIE-style<br/>parsing"] --> E2["(subj, rel, obj)<br/>triples"]
    end

    subgraph Module5["Module 5<br/>Graph Assembly"]
        G1["Directed multi-<br/>relational causal<br/>graph G"]
    end

    subgraph Module6["Module 6<br/>Geometric Refinement"]
        R1["Geometric<br/>Transformer"] --> R2["Manifold<br/>embeddings"]
        R2 --> R3["Topos-organized<br/>slices"]
    end

    Module1 --> Module2 --> Module3 --> Module4 --> Module5 --> Module6

    style Module1 fill:#2d3436,stroke:#74b9ff,color:#dfe6e9
    style Module2 fill:#2d3436,stroke:#74b9ff,color:#dfe6e9
    style Module3 fill:#2d3436,stroke:#74b9ff,color:#dfe6e9
    style Module4 fill:#2d3436,stroke:#74b9ff,color:#dfe6e9
    style Module5 fill:#2d3436,stroke:#74b9ff,color:#dfe6e9
    style Module6 fill:#2d3436,stroke:#00cec9,color:#dfe6e9
```

#### ARCADIA (Scalable Causal Discovery)

Uses an agentic LLM as an orchestration layer that constrains statistical discovery. The agent documents its theory at each step, and the workflow loops until the DAG satisfies six evaluation criteria.

```mermaid
stateDiagram-v2
    [*] --> Propose: Agent proposes initial DAG
    Propose --> Evaluate: Statistical diagnostics
    Evaluate --> Check: Six criteria met?

    Check --> Refine: No — issues found
    Refine --> Propose: Agent revises theory

    Check --> Finish: Yes — all pass
    Finish --> [*]: Emit DAG + JSON transcript + report

    note right of Evaluate
        - Back-door sets
        - ΔBIC orientation
        - Temporal constraints
        - Binary vs continuous fit
        - Node-level regression R²
        - Sample balance check
    end note
```

#### The Agentic Leash (Panda et al., Jan 2026)

Builds Fuzzy Cognitive Maps (FCMs) where the dynamical system's equilibria drive the LLM agent to fetch and process additional causal text — a bidirectional loop between extraction and the evolving causal structure.

```mermaid
flowchart TD
    A["Raw Text"] --> B["Step 1: Extract<br/>Nouns & Noun Phrases"]
    B --> C["Step 2: Identify<br/>Causal Relations +<br/>Edge Weights"]
    C --> D["Step 3: Assemble<br/>FCM Dynamical System"]
    D --> E{"Equilibrium<br/>Analysis"}
    E -- "Limit cycles /<br/>fixed-point attractors" --> F["FCM drives agent<br/>to fetch more text"]
    F --> B
    E -- "Converged" --> G["Final FCM"]

    style G fill:#00b894,stroke:#dfe6e9,color:#fff
```

---

## Paradigm 3 — RAG-Enhanced Causal Discovery

CausalRAG (ACL 2025 Findings) combines causal graph structure with retrieval in a bidirectional manner: RAG grounds causal extraction in evidence, and the causal graph improves retrieval quality.

```mermaid
flowchart TD
    subgraph Ingestion["Corpus Ingestion"]
        D1["Documents"] --> D2["Chunk &<br/>Embed"]
        D2 --> D3["Vector<br/>Store"]
    end

    subgraph CausalDiscovery["Causal Discovery Layer"]
        CD1["Retrieve relevant<br/>chunks for variable<br/>pair (X, Y)"]
        CD2["LLM: Assess causal<br/>relation + strength"]
        CD3["Build / update<br/>causal graph G"]
        CD1 --> CD2 --> CD3
    end

    subgraph CausalRAG["Causal-Aware Retrieval"]
        CR1["User Query Q"]
        CR2["Traverse causal<br/>graph to identify<br/>relevant causal chains"]
        CR3["Retrieve evidence<br/>along causal paths"]
        CR4["Generate causal<br/>analysis report"]
        CR1 --> CR2 --> CR3 --> CR4
    end

    D3 --> CD1
    CD3 --> CR2
    D3 --> CR3

    style CausalDiscovery fill:#2d3436,stroke:#e17055,color:#dfe6e9
    style CausalRAG fill:#2d3436,stroke:#0984e3,color:#dfe6e9
```

---

## Practical LangGraph Implementation Blueprint

### State Schema

```python
from typing import TypedDict, Annotated
from langgraph.graph import StateGraph
import networkx as nx

class CausalDAGState(TypedDict):
    corpus_chunks: list[str]              # Processed text chunks
    embeddings_indexed: bool              # Whether vector store is ready
    candidate_nodes: list[str]            # Extracted variable names
    extracted_edges: list[dict]           # {source, target, confidence, evidence}
    dag: nx.DiGraph                       # Current DAG state
    iteration: int                        # Refinement loop counter
    max_iterations: int                   # Loop bound
    validation_log: list[dict]            # Issues found per iteration
    confidence_scores: dict[str, float]   # Edge → confidence mapping
    evidence_map: dict[str, list[str]]    # Edge → supporting passages
```

### Agent Definitions

```mermaid
flowchart LR
    subgraph Agents["Agent Roster"]
        direction TB
        A1["🔍 Ingester<br/><i>Chunks, embeds,<br/>indexes corpus</i>"]
        A2["📋 Variable Extractor<br/><i>NER + dedup +<br/>canonicalize</i>"]
        A3["⚡ Causal Extractor<br/><i>Pairwise or batch<br/>edge discovery</i>"]
        A4["🏗️ DAG Assembler<br/><i>Graph construction +<br/>cycle removal</i>"]
        A5["✅ Validator<br/><i>Structural + semantic<br/>consistency checks</i>"]
        A6["📊 Reporter<br/><i>Export DAG +<br/>evidence + viz</i>"]
    end

    subgraph Tools["Tool Access"]
        direction TB
        T1["Vector Store<br/>(ChromaDB / Pinecone)"]
        T2["LLM API<br/>(Claude / GPT)"]
        T3["NetworkX<br/>Graph Ops"]
        T4["causal-learn<br/>Statistical CD"]
        T5["MLflow<br/>Experiment Tracking"]
    end

    A1 --> T1
    A2 --> T2
    A3 --> T1 & T2
    A4 --> T3
    A5 --> T2 & T4
    A6 --> T3 & T5
```

### Graph Topology (LangGraph Control Flow)

```mermaid
graph TD
    START(("__start__")) --> ingest["ingest_corpus"]
    ingest --> extract_vars["extract_variables"]
    extract_vars --> extract_causal["extract_causal_relations"]
    extract_causal --> assemble["assemble_dag"]
    assemble --> validate["validate_dag"]
    validate --> route{"route_decision"}

    route -- "issues found &<br/>iteration < max" --> extract_causal
    route -- "all checks pass OR<br/>max iterations reached" --> report["generate_report"]
    report --> END(("__end__"))

    style START fill:#6c5ce7,color:#fff
    style END fill:#6c5ce7,color:#fff
    style route fill:#fdcb6e,stroke:#f39c12,color:#2d3436
```

---

## Key Design Decisions

### With RAG vs. Without RAG

```mermaid
flowchart LR
    subgraph WithRAG["With RAG (Recommended)"]
        direction TB
        WR1["Large corpus<br/>(100s–1000s of docs)"]
        WR2["Vector store indexes<br/>all chunks"]
        WR3["Causal Extractor retrieves<br/>evidence for each (X,Y) pair"]
        WR4["Grounds claims in<br/>actual corpus content"]
        WR1 --> WR2 --> WR3 --> WR4
    end

    subgraph WithoutRAG["Without RAG"]
        direction TB
        NR1["Small corpus<br/>(fits in context)"]
        NR2["Pass full chunks<br/>to extraction agents"]
        NR3["Exhaustive chunk-by-chunk<br/>processing"]
        NR4["Relies more on LLM<br/>parametric knowledge"]
        NR1 --> NR2 --> NR3 --> NR4
    end

    style WithRAG fill:#00b894,stroke:#dfe6e9,color:#fff
    style WithoutRAG fill:#636e72,stroke:#dfe6e9,color:#fff
```

### Pairwise vs. Batch Extraction

| Strategy | Complexity | Accuracy | Best For |
|----------|-----------|----------|----------|
| **Pairwise** ("Does X cause Y?") | O(n²) in variables | Higher — focused reasoning | Graphs with < 30 nodes |
| **Batch** ("Output full adjacency matrix") | O(1) LLM calls | Lower — degrades > 15 nodes | Quick prototyping, small graphs |
| **Hybrid** (batch first, pairwise refinement) | O(n) + selective O(k²) | Best of both worlds | Production systems |

### Validation Criteria

The validator agent should enforce both structural and semantic checks:

```mermaid
flowchart TD
    subgraph Structural["Structural Checks"]
        S1["Acyclicity<br/><i>(hard constraint)</i>"]
        S2["Connectivity<br/><i>(no isolated nodes)</i>"]
        S3["Degree bounds<br/><i>(no unrealistic hubs)</i>"]
    end

    subgraph Semantic["Semantic Checks"]
        SE1["Transitivity consistency<br/><i>If A→B and B→C,<br/>is A→C present<br/>or reasonably absent?</i>"]
        SE2["Domain coherence<br/><i>Do edges align with<br/>known domain knowledge?</i>"]
        SE3["Evidence coverage<br/><i>Every edge backed by<br/>≥1 source passage</i>"]
    end

    subgraph Statistical["Statistical Checks (Optional)"]
        ST1["Conditional independence<br/><i>If data available,<br/>test implied d-separations</i>"]
        ST2["BIC/AIC scoring<br/><i>Model fit of implied<br/>factorization</i>"]
    end

    Structural --> PASS{"All pass?"}
    Semantic --> PASS
    Statistical --> PASS
    PASS -- Yes --> DONE["✅ DAG Accepted"]
    PASS -- No --> LOOP["🔄 Refine"]
```

---

## Tools & Stack Recommendations

| Component | Prototyping | Production |
|-----------|------------|------------|
| **Orchestration** | LangGraph | LangGraph + LangSmith |
| **Graph Backend** | NetworkX | Neo4j / FalkorDB |
| **Vector Store** | ChromaDB | Pinecone / Weaviate |
| **LLM** | Claude Sonnet 4.5 | Claude Sonnet (extraction) + Opus (validation) |
| **Statistical CD** | `causal-learn` (PC, GES) | `causal-learn` + custom NOTEARS |
| **Tracking** | MLflow | MLflow + LangSmith traces |
| **Visualization** | Graphviz / pyvis | D3.js / Cytoscape.js |

---

## Iterative Causal-Aware RL Loop (Advanced)

For scenarios where observational data complements the text corpus, the iterative RL pattern offers an additional refinement mechanism:

```mermaid
sequenceDiagram
    participant LLM as LLM Agent
    participant ENV as Environment / Data
    participant SCM as Causal DAG (SCM)

    rect rgb(45, 52, 54)
    Note over LLM,SCM: Learning Phase
    LLM->>ENV: Extract candidate causal<br/>variables from text observations
    ENV-->>LLM: Candidate variables +<br/>textual evidence
    LLM->>SCM: Propose initial edges
    end

    rect rgb(39, 60, 117)
    Note over LLM,SCM: Adapting Phase
    SCM->>ENV: Interventional do-operator<br/>experiments
    ENV-->>SCM: Validate / refute<br/>proposed edges
    SCM->>SCM: Refine DAG structure
    end

    rect rgb(106, 27, 154)
    Note over LLM,SCM: Acting Phase
    SCM->>LLM: Refined causal structure
    LLM->>LLM: Generate adaptive policies<br/>with causal-aware reward shaping
    LLM-->>SCM: Updated edge proposals
    end

    Note over LLM,SCM: Loop until convergence
```

---

## References

- **DEMOCRITUS**: Mahadevan (Dec 2025) — "Large Causal Models from Large Language Models" — [arXiv:2512.07796](https://arxiv.org/abs/2512.07796)
- **ARCADIA**: "Scalable Causal Discovery for Corporate" — [arXiv:2512.00839](https://arxiv.org/abs/2512.00839)
- **The Agentic Leash**: Panda et al. (Jan 2026) — "Extracting Causal Feedback Fuzzy Cognitive Maps with LLMs" — [arXiv:2601.00097](https://arxiv.org/abs/2601.00097)
- **CausalRAG**: ACL 2025 Findings — "Integrating Causal Graphs into Retrieval-Augmented Generation" — [ACL Anthology](https://aclanthology.org/2025.findings-acl.1165.pdf)
- **Causal-LLM**: EMNLP 2025 Findings — "A Unified One-Shot Framework for Prompt" — [ACL Anthology](https://aclanthology.org/2025.findings-emnlp.439.pdf)
- **IJCAI 2025 Survey**: "Large Language Models for Causal Discovery: Current Landscape and Future Directions" — [arXiv:2402.11068](https://arxiv.org/abs/2402.11068)
- **LCMs Overview**: EmergentMind — [Large Causal Models topic page](https://www.emergentmind.com/topics/large-causal-models-lcms)
