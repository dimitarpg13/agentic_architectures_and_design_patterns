# Guardrail Design and Causal Analysis

> **Series**: Actor-Critic Agent Design Pattern
> **Document**: 05 — Guardrail Design & Causal Analysis
> **Scope**: Guardrail interactions, the confounder problem, and Structural Causal Models (SCMs) for diagnosing and resolving multi-agent deadlocks

---

## 1. The General Confounder Problem

### What Is Confounding?

In causal inference, **confounding** occurs when a variable $U$ causally influences both the treatment $X$ and the outcome $Y$, creating a spurious statistical association between $X$ and $Y$ that does not reflect a direct causal effect.

```mermaid
graph LR
    U["Confounder (U)"]
    X["Treatment (X)"]
    Y["Outcome (Y)"]

    U -->|causal| X
    U -->|causal| Y
    X -->|causal| Y
    X -.-|"spurious association<br/>via U"| Y

    style U fill:#ffcccc,stroke:#cc0000
    style X fill:#cce5ff,stroke:#0066cc
    style Y fill:#d4edda,stroke:#28a745
```

An observer who measures only $X$ and $Y$ cannot distinguish the direct causal effect $X \to Y$ from the backdoor path $X \leftarrow U \to Y$. The standard remedy is Pearl's **backdoor adjustment**, which conditions on the confounder to block the spurious path:

$$P(Y \mid do(X)) = \sum_{U} P(Y \mid X, U) \cdot P(U)$$

The $do(\cdot)$ operator denotes an *intervention* — physically setting $X$ to a value rather than passively observing it. Without adjustment, observational data conflates causal and confounded effects.

### Confounding in Adversarial Multi-Agent Systems

In an Actor-Critic architecture, the Critic observes the Actor's output and issues a pass/fail signal. This signal drives the Actor's next attempt. But when a **hidden variable** influences both the Actor's output quality and the Critic's ability to evaluate it, the feedback loop becomes confounded:

```mermaid
graph TB
    U["Confounder (U)<br/><i>unobserved by Critic</i>"]
    Gen["Actor<br/>(generates output)"]
    Val["Critic<br/>(validates output)"]
    FB["Feedback Signal"]
    Adj["Actor Adjusts Strategy"]

    U -.->|degrades output| Gen
    U -.->|distorts evaluation| Val
    Gen -->|output| Val
    Val -->|pass/fail + rationale| FB
    FB -->|"correction prompt"| Adj
    Adj -->|"next attempt"| Gen

    style U fill:#ffcccc,stroke:#cc0000,stroke-dasharray: 5 5
    style FB fill:#fff3cd,stroke:#856404
```

The Critic cannot distinguish whether a bad outcome was caused by the **Actor's strategy** (fixable via correction) or by an **environmental variable** (not fixable by the Actor). When the Critic attributes an environmentally caused failure to the Actor's strategy, the correction prompt sends the Actor searching for a solution in a space that contains none — the beginning of oscillation.

---

## 2. Flavors of Confounding in Actor-Critic Systems

### Flavor 1: Task-Difficulty Confounding

**Confounder:** intrinsic task complexity.

```mermaid
graph LR
    TC["Task Complexity<br/>(confounder)"]
    AS["Actor Output Quality"]
    CR["Critic Rejection Rate"]

    TC -->|"complex task →<br/>lower quality output"| AS
    TC -->|"complex task →<br/>stricter evaluation"| CR
    AS -->|"quality drives<br/>pass/fail"| CR

    style TC fill:#ffcccc,stroke:#cc0000
```

A complex task simultaneously causes lower-quality Actor output (the task is genuinely harder to solve) AND a higher Critic rejection rate (the Critic's rubric penalizes incompleteness or subtlety). The Critic cannot distinguish "the Actor chose a bad strategy" from "the task is inherently hard and no strategy would score well on the first pass."

**Example — Code Generation:** The Actor observes over many interactions that verbose code with extensive inline comments correlates with Critic approval. It shifts its strategy toward generating verbose, heavily commented code. The real explanation: simple tasks — where verbose commenting is easy and natural — are also the tasks the Critic passes on the first attempt. For complex tasks, the verbosity is a cargo-cult adaptation that doesn't address the actual difficulty.

### Flavor 2: Physical-Constraint Confounding

**Confounder:** system guardrails (output size limits, execution timeouts, memory limits).

```mermaid
graph LR
    PC["Physical Constraint<br/>(output size limit,<br/>timeout, memory)"]
    AO["Actor Output<br/>(truncated / incomplete)"]
    CE["Critic Evaluation<br/>(flags incompleteness)"]

    PC -->|"prevents complete<br/>output"| AO
    PC -->|"invisible to Critic"| CE
    AO -->|"incomplete →<br/>rejection"| CE

    style PC fill:#ffcccc,stroke:#cc0000
```

A physical constraint prevents the Actor from producing complete output AND causes the Critic to flag the output as incomplete. The Critic cannot distinguish a **lazy Actor** (chose not to be thorough) from a **physically constrained Actor** (tried to be thorough but was blocked).

**Example — Code Generation:** A user asks the Actor to generate a comprehensive integration test suite covering all 200 API endpoints in a microservices platform. The full suite would require ~50,000 tokens. The output guardrail caps responses at 5,000 tokens. The Actor generates the most critical 20 tests and offers to continue in batches. The Critic rejects: "Test suite is incomplete — covers only 10% of endpoints." The Actor tries pagination. The Critic rejects each page: "Partial output — does not answer the user's request for a complete suite." Context fills with failed attempts. Deadlock.

### Flavor 3: Temporal-State Confounding

**Confounder:** accumulated context window state.

```mermaid
graph TB
    CW["Context Window State<br/>(accumulated history)"]
    GQ["Actor Generation<br/>Quality"]
    EQ["Critic Evaluation<br/>Quality"]
    CA["Correction Attempts"]

    CW -->|"noise from prior<br/>failed attempts"| GQ
    CW -->|"conflicting signals<br/>degrade rubric application"| EQ
    GQ -->|"lower quality →<br/>rejection"| CA
    EQ -->|"noisier evaluation →<br/>less useful feedback"| CA
    CA -->|"more messages<br/>appended"| CW

    style CW fill:#ffcccc,stroke:#cc0000
```

As the correction loop iterates, the context window fills with prior attempts, error messages, correction prompts, and Critic feedback. This accumulated state degrades **both** the Actor's generation quality (the model attends to contradictory prior instructions) and the Critic's evaluation quality (the Critic's context is polluted with failed attempts that bias its rubric application). Each correction attempt makes the next one worse — a positive feedback loop toward context exhaustion.

---

## 3. Guardrail Taxonomy

### The Guardrail Interaction Graph

Every Actor-Critic system operates under multiple simultaneous guardrails. These guardrails are not independent — they interact, and some interactions create infeasible constraint regions.

```mermaid
graph TB
    subgraph Output["Output Guardrails"]
        OG1["Max output size<br/>(token limit)"]
        OG2["Context window limit"]
        OG3["Tool output cap"]
    end

    subgraph Security["Security Guardrails"]
        SG1["Code validation<br/>(no eval, exec, open)"]
        SG2["Import restrictions<br/>(whitelist only)"]
        SG3["Dangerous operation<br/>blacklist"]
    end

    subgraph Critic["Critic Guardrails"]
        CG1["Completeness check"]
        CG2["Correctness check"]
        CG3["Hallucination check"]
    end

    subgraph Behavioral["Behavioral Guardrails"]
        BG1["Instruction compliance"]
        BG2["Correction loop limit<br/>(max N attempts)"]
        BG3["Style / format<br/>compliance"]
    end

    OG1 <-->|"INCOMPATIBLE<br/>for large outputs"| CG1
    OG2 <-->|"INCOMPATIBLE<br/>for deep correction"| BG2
    SG1 ---|"compatible"| CG2
    BG1 ---|"compatible"| BG3

    style OG1 fill:#fff3cd,stroke:#856404
    style OG2 fill:#fff3cd,stroke:#856404
    style CG1 fill:#f8d7da,stroke:#721c24
    style BG2 fill:#f8d7da,stroke:#721c24
```

The **incompatibility edges** are the dangerous ones:
- **Output size ↔ Completeness check**: When the complete answer exceeds the output limit, no Actor strategy can simultaneously satisfy both.
- **Context window ↔ Correction loop**: Each correction attempt consumes context. A long correction loop can exhaust the window before convergence.

### Taxonomy of Guardrail Interactions

| Type | Definition | Example | Risk |
|---|---|---|---|
| **Compatible** | Both constraints satisfiable simultaneously for all task classes | Security validation + style check — secure code and well-formatted code are independent properties | None |
| **Hierarchical** | One guardrail takes precedence; the subordinate yields | Context window crash overrides the correction loop — the system halts rather than crashing | Low — predictable degradation |
| **Incompatible** | No feasible Actor strategy satisfies both for some task class | Output token limit + completeness check when the complete answer exceeds the limit | **HIGH** — deadlock |
| **Confounding** | One guardrail's effect is unobserved by the evaluating agent | Output size limit invisible to the Critic's completeness check — the Critic sees incompleteness but not its cause | **HIGH** — causal misattribution |

Incompatible and Confounding guardrails often co-occur: the constraint that creates the infeasibility is typically the same one hidden from the Critic.

---

## 4. Case Study: Guardrail Deadlock

### The Scenario

A user asks the Actor to generate integration tests for all 200 API endpoints in a platform's REST API. The full test suite — with setup, teardown, assertions, and edge cases for each endpoint — requires approximately 50,000 tokens. The system's output guardrail limits any single response to 5,000 tokens.

| Parameter | Value |
|---|---|
| User request | "Generate integration tests for all 200 API endpoints" |
| Required output size | ~50,000 tokens |
| Output guardrail | 5,000 tokens max |
| Critic rubric | "Response must answer the user's question completely" |
| Correction loop limit | 3 attempts |

### The Deadlock Sequence

```mermaid
flowchart TD
    U["User: Generate tests for all 200 endpoints"]

    subgraph Attempt1["Attempt 1 - Direct Generation"]
        A1["Actor generates tests"]
        G1["Output Guardrail truncates at 5,000 tokens<br/>Only 20 of 200 endpoints covered"]
        C1["Actor: Here are tests for the 20<br/>highest-priority endpoints..."]
        F1["Critic: FAIL - Only 10% coverage. Incomplete."]
        A1 --> G1 --> C1 --> F1
    end

    subgraph Attempt2["Attempt 2 - Pagination"]
        A2["Actor tries continuing from endpoint 21"]
        G2["Output Guardrail truncates again"]
        C2["Actor: Endpoints 21-40.<br/>See prior message for 1-20."]
        F2["Critic: FAIL - Partial output.<br/>Does not constitute a complete test suite."]
        A2 --> G2 --> C2 --> F2
    end

    subgraph Attempt3["Attempt 3 - Summarized Approach"]
        A3["Actor generates template + endpoint list"]
        G3["Fits within guardrail limit"]
        C3["Actor: Parameterized test template<br/>with an endpoint table."]
        F3["Critic: FAIL - Template is not executable tests.<br/>User asked for actual tests."]
        A3 --> G3 --> C3 --> F3
    end

    CRASH["Context exhausted.<br/>System returns best-effort with warning."]

    U --> Attempt1
    F1 --> Attempt2
    F2 --> Attempt3
    F3 --> CRASH

    style F1 fill:#f8d7da,stroke:#721c24
    style F2 fill:#f8d7da,stroke:#721c24
    style F3 fill:#f8d7da,stroke:#721c24
    style CRASH fill:#f5c6cb,stroke:#721c24,stroke-width:2px
```

### The Causal Misattribution

| Observation | Critic's Interpretation | True Cause |
|---|---|---|
| Test suite covers only 20 of 200 endpoints | Actor strategy failure — Actor should have generated all tests (fixable) | Output guardrail prevents returning more than ~20 endpoints worth of tests (unfixable by Actor) |
| Paginated output across multiple messages | Actor formatting failure — should consolidate into single response | Single-response output limit makes consolidation physically impossible |
| Template-based alternative offered | Actor deviated from user's request | Actor correctly adapted to physical constraint; Critic's rubric doesn't account for this |

### Why the System Cannot Converge

Both agents reason correctly **within their respective observation spaces**. The Actor correctly determines that the full output is physically impossible and offers reasonable alternatives. The Critic correctly determines that none of the alternatives fully answer the user's question. The intersection of their individually correct judgments is **empty** — no strategy exists that the Actor can execute and the Critic will accept.

```mermaid
graph TB
    subgraph ActorSpace["Actor's Feasible Strategies"]
        S1["Full output<br/>(blocked by guardrail)"]
        S2["Paginated output"]
        S3["Summarized template"]
        S4["Top-N prioritized"]
    end

    subgraph CriticSpace["Critic's Acceptance Criterion"]
        C1["Complete test suite<br/>covering all 200 endpoints"]
    end

    subgraph GuardrailSpace["Guardrail Constraint"]
        G1["≤ 5,000 tokens<br/>per response"]
    end

    S1 -.->|"blocked"| G1
    S2 -.->|"rejected"| C1
    S3 -.->|"rejected"| C1
    S4 -.->|"rejected"| C1

    style S1 fill:#f8d7da,stroke:#721c24
    style S2 fill:#fff3cd,stroke:#856404
    style S3 fill:#fff3cd,stroke:#856404
    style S4 fill:#fff3cd,stroke:#856404
    style C1 fill:#cce5ff,stroke:#0066cc
    style G1 fill:#ffcccc,stroke:#cc0000
```

**Formal convergence requires three conditions.** The deadlock violates all of them:

| Condition | Requirement | Status |
|---|---|---|
| **Feasibility** | There exists at least one Actor strategy that the Critic will accept | **Violated** — No strategy fits within 5,000 tokens AND constitutes a complete 200-endpoint test suite |
| **Observability** | The Critic observes all variables that affect the Actor's output | **Violated** — The Critic does not observe the output guardrail's activation |
| **Monotonicity** | Each correction attempt moves the Actor closer to acceptance | **Violated** — Attempts cycle between truncation, pagination, and summarization without approaching the acceptance boundary |

---

## 5. Deconfounding: Making Hidden Constraints Observable

### The Fix as Informal Backdoor Adjustment

The Critic's evaluation without deconfounding:

$$P(\text{FAIL} \mid \text{incomplete output})$$

This conflates two very different situations — the Actor **chose** to produce incomplete output (strategy failure) and the Actor **was prevented** from producing complete output (physical constraint). The backdoor adjustment conditions on the constraint:

$$P(\text{FAIL} \mid \text{incomplete output},\ \text{physical constraint})$$

When we decompose by constraint status, the Critic's judgment changes:

| Physical Constraint | Output Completeness | Correct Critic Judgment |
|---|---|---|
| **Inactive** | Complete | PASS |
| **Inactive** | Incomplete | FAIL — Actor strategy failure |
| **Active** | Complete (within constraint) | PASS |
| **Active** | Incomplete (due to constraint) | **PASS with caveat** — Actor adapted correctly to physical limitation |

Without conditioning on the constraint, the Critic collapses rows 2 and 4 into the same judgment: FAIL. With the constraint observable, the Critic can distinguish row 4 (correct Actor behavior under constraint) from row 2 (genuine Actor failure).

### Formal Deconfounding via SCM

The deconfounded system makes the physical constraint **observable** to the Critic, blocking the backdoor path:

```mermaid
graph TB
    PC["Physical Constraint<br/>(NOW OBSERVED)"]
    AO["Actor Output"]
    CE["Critic Evaluation"]
    AS["Actor Strategy"]

    AS -->|"determines approach"| AO
    PC -->|"limits output"| AO
    PC ==>|"constraint metadata<br/>passed to Critic"| CE
    AO -->|"content evaluated"| CE

    style PC fill:#d4edda,stroke:#28a745,stroke-width:3px
    style CE fill:#cce5ff,stroke:#0066cc
```

In the original (confounded) system, the path $\text{Physical Constraint} \to \text{Actor Output} \leftarrow \text{Actor Strategy} \to \text{Critic Evaluation}$ was confounded because the Critic could not condition on the Physical Constraint. In the deconfounded system, the Critic receives the constraint signal directly (the bold edge), blocking the backdoor and enabling correct causal attribution.

---

## 6. Building an SCM for Guardrail Interactions

### Step 1: Enumerate All Guardrails

Before constructing the causal model, catalog every constraint that could influence Actor output or Critic evaluation:

```mermaid
graph TB
    subgraph Physical["Physical Constraints"]
        P1["Output token limit"]
        P2["Context window capacity"]
        P3["Execution timeout"]
        P4["Memory limit"]
        P5["Tool output cap"]
    end

    subgraph BehavioralC["Behavioral Constraints"]
        B1["Max correction attempts"]
        B2["Instruction compliance rules"]
        B3["Format / style requirements"]
    end

    subgraph SecurityC["Security Constraints"]
        S1["Allowed import whitelist"]
        S2["Blocked operation blacklist"]
        S3["Sandbox restrictions"]
    end

    subgraph EvalC["Evaluation Constraints"]
        E1["Completeness rubric"]
        E2["Correctness rubric"]
        E3["Hallucination detection"]
    end

    style Physical fill:#fff3cd,stroke:#856404
    style BehavioralC fill:#d1ecf1,stroke:#0c5460
    style SecurityC fill:#f8d7da,stroke:#721c24
    style EvalC fill:#cce5ff,stroke:#0066cc
```

### Step 2: Construct the Causal DAG

Map the causal relationships between user request, guardrails, Actor strategy, and Critic evaluation. **Dashed edges** indicate relationships that are real but unobserved by the Critic:

```mermaid
graph TB
    UR["User Request<br/>(complexity, scope)"]
    AS["Actor Strategy"]
    AO["Actor Output"]
    CE["Critic Evaluation"]
    CA["Correction Attempt"]

    OTL["Output Token Limit"]
    CWC["Context Window Capacity"]
    ET["Execution Timeout"]
    TOC["Tool Output Cap"]
    MCA["Max Correction Attempts"]

    UR -->|"determines scope"| AS
    AS -->|"generates"| AO
    AO -->|"evaluated by"| CE
    CE -->|"feedback"| CA
    CA -->|"next attempt"| AS

    OTL -.->|"truncates<br/>(UNOBSERVED)"| AO
    CWC -.->|"degrades quality<br/>(UNOBSERVED)"| AS
    CWC -.->|"degrades evaluation<br/>(UNOBSERVED)"| CE
    ET -.->|"kills execution<br/>(UNOBSERVED)"| AO
    TOC -.->|"truncates tool result<br/>(UNOBSERVED)"| AO
    MCA -->|"halts loop"| CA
    UR -->|"determines difficulty"| CE

    CA -->|"fills context"| CWC

    style OTL fill:#ffcccc,stroke:#cc0000,stroke-dasharray: 5 5
    style CWC fill:#ffcccc,stroke:#cc0000,stroke-dasharray: 5 5
    style ET fill:#ffcccc,stroke:#cc0000,stroke-dasharray: 5 5
    style TOC fill:#ffcccc,stroke:#cc0000,stroke-dasharray: 5 5
```

### Step 3: Identify Backdoor Paths

Three backdoor paths create confounded feedback signals:

**Path 1 — Output Truncation Confounding:**

$$\text{Actor Strategy} \leftarrow \text{User Request} \to \text{Output Token Limit activation} \to \text{Actor Output} \to \text{Critic Evaluation}$$

The Critic sees incomplete output but does not observe whether the Output Token Limit was the cause. Attribution: Actor strategy failure. True cause: physical truncation.

**Path 2 — Context Degradation Confounding:**

$$\text{Actor Strategy} \leftarrow \text{Context Window Capacity} \to \text{Critic Evaluation}$$

As context fills (from correction attempts), both the Actor's generation quality and the Critic's evaluation quality degrade. The Critic sees declining Actor output quality but does not observe that its own evaluation is simultaneously degrading.

**Path 3 — Tool Execution Confounding:**

$$\text{Actor Strategy} \leftarrow \text{User Request} \to \text{Execution Timeout / Tool Output Cap} \to \text{Actor Output} \to \text{Critic Evaluation}$$

A tool execution times out or its output is capped. The Actor receives an error and must adapt. The Critic sees the adapted (incomplete) output but does not observe the tool-level constraint that forced the adaptation.

### Step 4: Apply Backdoor Adjustment

For each backdoor path, identify the confounder to observe and the mechanism for making it observable:

| Backdoor Path | Confounder | Adjustment Mechanism |
|---|---|---|
| Output Truncation | Output Token Limit activation | Attach `constraint_type: SIZE_CONSTRAINED` and `max_tokens` / `actual_tokens` metadata to Critic input |
| Context Degradation | Context Window Capacity utilization | Attach `context_utilization: 0.87` and `correction_attempt: 2 of 3` metadata to Critic input |
| Tool Execution Constraint | Execution Timeout / Tool Output Cap | Attach `tool_constraint: TIMEOUT` or `tool_constraint: OUTPUT_TRUNCATED` with details to Critic input |

### The Problem: Constraint Signals Exist but Don't Reach the Critic

In many implementations, the execution layer already detects and signals these constraints — but the signals are returned as **plain text error messages to the Actor**, not as **structured metadata to the Critic**:

```python
def execute_code(code: str, timeout: int = 30) -> str:
    result = sandbox.run(code, timeout=timeout)

    if result.truncated:
        return "Result too large. Please reduce output size or summarize."

    if result.token_count > MAX_TOOL_OUTPUT_TOKENS:
        return f"Output exceeds {MAX_TOOL_OUTPUT_TOKENS} token limit. Modify your code."

    if result.timed_out:
        return f"Execution timed out after {timeout}s. Optimize or reduce scope."

    return result.output
```

These messages give the Actor enough information to adapt its strategy, but the Critic never sees them. The Critic receives the Actor's **adapted output** — which looks like incomplete work — without knowing that the adaptation was forced by a physical constraint. The causal signal is present in the system but routed to the wrong consumer.

---

## 7. Guardrail-Aware Validation Architecture

### The Deconfounding Layer

Insert a **constraint collection layer** between the Actor's output and the Critic's evaluation. This layer gathers active guardrail signals and attaches them as structured metadata.

```mermaid
flowchart TB
    AO["Actor Output"]
    CL["Constraint Collector"]
    MD["Constraint Metadata"]
    CI["Critic Input<br/>(output + metadata)"]
    CE["Critic Evaluation<br/>(constraint-aware)"]

    OG["Output Guardrails"]
    TG["Tool Guardrails"]
    CWG["Context Window Monitor"]
    LG["Loop Counter"]

    AO --> CL
    OG -->|"size_constrained?"| CL
    TG -->|"timeout / truncation?"| CL
    CWG -->|"utilization %"| CL
    LG -->|"attempt N of M"| CL

    CL --> MD

    subgraph CriticPackage["Critic Receives"]
        MD --> CI
        AO --> CI
    end

    CI --> CE

    style CL fill:#d4edda,stroke:#28a745,stroke-width:2px
    style MD fill:#d4edda,stroke:#28a745
    style CE fill:#cce5ff,stroke:#0066cc
```

The constraint collector classifies the Actor's operating condition:

| Constraint Class | Trigger | Critic Behavior |
|---|---|---|
| `UNCONSTRAINED` | No guardrails active | Normal rubric — hold Actor to full standard |
| `SIZE_CONSTRAINED` | Output exceeds token limit | Accept partial output if Actor's prioritization and coverage strategy are sound |
| `TIMEOUT_CONSTRAINED` | Tool execution timed out | Evaluate Actor's fallback strategy rather than output completeness |
| `CONTEXT_EXHAUSTED` | Context utilization > 90% or on final correction attempt | Accept best-effort output; flag for user that full answer requires a fresh session |

### Constraint-Aware Validation Formula

The Critic's evaluation function becomes conditional on the constraint class:

$$
P(\text{FAIL} \mid \text{output}, \text{constraints}) = \begin{cases}
P(\text{FAIL} \mid \text{output}, \text{UNCONSTRAINED}) & \text{full rubric} \\\\
P(\text{FAIL} \mid \text{output}, \texttt{SIZE CONSTRAINED}) & \text{relaxed completeness, strict prioritization} \\\\
P(\text{FAIL} \mid \text{output}, \texttt{TIMEOUT CONSTRAINED}) & \text{evaluate fallback quality} \\\\
P(\text{FAIL} \mid \text{output}, \texttt{CONTEXT EXHAUSTED}) & \text{accept best-effort}
\end{cases}
$$

This is the backdoor adjustment in practice: the Critic conditions on the constraint variable, separating Actor strategy quality from environmental limitation.

---

## 8. From Oscillation to Convergence

### The Oscillation Mechanism

Without deconfounding, the correction loop follows a predictable path to context exhaustion:

```mermaid
stateDiagram-v2
    [*] --> Generate
    Generate --> GuardrailBlock: output exceeds limit
    GuardrailBlock --> OfferAlternative: Actor adapts
    OfferAlternative --> CriticReject: Critic sees incompleteness
    CriticReject --> CorrectionPrompt: "make it complete"
    CorrectionPrompt --> Generate: Actor tries again

    Generate --> GuardrailBlock: same constraint hit

    CorrectionPrompt --> ContextExhaustion: after N iterations
    ContextExhaustion --> Crash: return best-effort with warning

    note right of CriticReject
        Critic does not know
        why output is incomplete
    end note

    note right of ContextExhaustion
        Context filled with
        failed attempts
    end note
```

Each iteration adds ~1,000–3,000 tokens of correction prompt and failed output to the context window, accelerating the approach to exhaustion without moving closer to acceptance.

### Causal Conditions for Convergence

A correction loop converges if and only if all three conditions hold:

**Condition 1 — Feasibility:**

$$\exists\ s \in \mathcal{S}_{\text{Actor}} \text{ such that } \text{Critic}(s) = \text{PASS}$$

There must exist at least one strategy in the Actor's feasible set that the Critic will accept. When a guardrail makes the Critic's acceptance criterion physically unreachable, no amount of iteration helps.

*Deconfounding restores feasibility* by allowing the Critic to relax its acceptance criterion when it observes an active constraint — expanding the acceptance region to include constrained-but-well-adapted strategies.

**Condition 2 — Observability:**

$$\forall\ v \in \text{Vars}(\text{Actor Output}),\ v \in \text{Obs}(\text{Critic})$$

Every variable that causally affects the Actor's output must be observable by the Critic. Hidden guardrails violate this condition.

*Deconfounding restores observability* by routing constraint signals to the Critic as structured metadata.

**Condition 3 — Monotonicity:**

$$d(\text{Critic}(s_{t+1}),\ \text{PASS}) < d(\text{Critic}(s_t),\ \text{PASS})$$

Each correction attempt must move the Actor's output closer to the acceptance boundary. When the Actor oscillates between incompatible strategies (full output → truncated, summarized → rejected, full output → truncated again), the distance to acceptance does not decrease.

*Deconfounding restores monotonicity* because the Critic's feedback becomes actionable — it targets the Actor's actual strategy choices rather than the symptoms of hidden constraints, so each correction addresses a real deficiency.

---

## 9. Generalizing the Methodology

The guardrail analysis methodology applies to any Actor-Critic system, regardless of domain. The procedure is:

```mermaid
flowchart TB
    E["1. Enumerate all guardrails<br/>(physical, behavioral,<br/>security, evaluation)"]
    D["2. Construct causal DAG<br/>(map all causal relationships,<br/>mark unobserved edges)"]
    B["3. Identify backdoor paths<br/>(confounders hidden<br/>from Critic)"]
    A["4. Apply backdoor adjustment<br/>(route constraint signals<br/>to Critic as metadata)"]
    V["5. Verify convergence<br/>(check feasibility,<br/>observability, monotonicity)"]
    M["6. Monitor for new<br/>incompatibilities<br/>(new guardrails, new<br/>task classes)"]

    E --> D --> B --> A --> V --> M
    M -->|"new guardrail<br/>added"| E

    style E fill:#cce5ff,stroke:#0066cc
    style D fill:#cce5ff,stroke:#0066cc
    style B fill:#f8d7da,stroke:#721c24
    style A fill:#d4edda,stroke:#28a745
    style V fill:#d4edda,stroke:#28a745
    style M fill:#fff3cd,stroke:#856404
```

| Step | Action | Output |
|---|---|---|
| **Enumerate** | Catalog every constraint the Actor operates under | Complete guardrail inventory |
| **Construct DAG** | Map causal relationships; mark edges invisible to the Critic | Causal graph with observed/unobserved edge annotations |
| **Identify backdoor paths** | Trace every path from Actor Strategy to Critic Evaluation that passes through an unobserved variable | List of confounded feedback channels |
| **Apply adjustment** | For each backdoor path, route the confounder's signal to the Critic as structured metadata | Deconfounded Critic input schema |
| **Verify convergence** | Check that feasibility, observability, and monotonicity hold for all supported task classes | Convergence proof or identification of remaining infeasible regions |
| **Monitor** | When new guardrails are added or task classes change, re-enter the loop | Ongoing system health |

**Key insight:** Guardrails are not independent safety checks — they form a **causal system** whose interactions determine whether the Actor-Critic loop converges. Treating each guardrail in isolation misses the emergent incompatibilities that arise from their joint effect on the feedback loop. The SCM formalism provides a principled method for discovering, diagnosing, and resolving these interactions before they manifest as production deadlocks.

> **Real-world note:** A production platform implementing this Actor-Critic pattern for natural language analytics encountered exactly the deadlock described in Section 4 when users requested exhaustive item listings that exceeded output guardrails. Applying the deconfounding methodology (constraint metadata passed to the Critic) resolved the oscillation and enabled the system to gracefully degrade under physical constraints instead of entering correction loops that inevitably crashed.

---

## Summary

This document established a causal framework for understanding and resolving guardrail interactions in Actor-Critic systems:

- **Confounding** occurs when hidden variables (physical constraints, task complexity, context degradation) influence both Actor output and Critic evaluation, creating misattributed feedback signals.
- **Three flavors** — task-difficulty, physical-constraint, and temporal-state confounding — cover the major failure modes.
- **Guardrail interactions** follow a taxonomy: compatible, hierarchical, incompatible, and confounding — with incompatible and confounding interactions creating deadlock risk.
- **Deadlocks** arise when the intersection of the Actor's feasible strategies and the Critic's acceptance criteria is empty due to hidden constraints.
- **Deconfounding** via backdoor adjustment — making constraint signals observable to the Critic — restores feasibility, observability, and monotonicity.
- **The SCM methodology** (enumerate → DAG → backdoor paths → adjust → verify → monitor) provides a repeatable procedure for any Actor-Critic deployment.

---

<div align="center">

**Actor-Critic Agent Design Pattern — Document Series**

| Document | Title |
|----------|-------|
| [01](./01_actor_critic_architecture.md) | Actor-Critic Architecture |
| [02](./02_actor_critic_workflow.md) | Agentic Workflow Deep-Dive |
| [03](./03_critic_validation_system.md) | Critic Validation System |
| [04](./04_tool_calling_system.md) | Tool Calling System |
| **05** | **Guardrail Design & Causal Analysis** *(this document)* |
| [06](./06_adversarial_dynamics_and_convergence.md) | Adversarial Dynamics & Convergence |
| [07](./07_limitations_and_enhancements.md) | Limitations & Enhancements |
| [08](./08_causal_nash_equilibrium_convergence.md) | Causal Nash Equilibrium Convergence |

</div>
