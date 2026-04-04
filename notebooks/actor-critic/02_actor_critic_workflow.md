# Actor-Critic Agentic Workflow

> **Series**: Actor-Critic Agent Design Pattern
> **Document**: 02 — End-to-End Workflow

This document traces the complete lifecycle of a request through a dual-agent system that follows the **Actor-Critic reinforcement-learning paradigm**. The Actor (Generator) produces outputs using tools; the Critic (Validator) evaluates those outputs and triggers corrections when necessary.

Throughout, we use a **code-generation assistant** as the running example: the Actor writes code, calls linters and test-runners, and the Critic validates correctness, security, and grounding.

---

## 1. The Dual-Agent Paradigm

The architecture separates *generation* from *validation* into two independently-reasoned agents, each backed by a distinct LLM. This mirrors the Actor-Critic split in reinforcement learning, where the Actor selects actions and the Critic estimates value.

```mermaid
flowchart TB
    subgraph Actor["Actor (Generator)"]
        A1[Reason about request]
        A2[Select & call tools]
        A3[Synthesize response]
        A1 --> A2 --> A3
    end

    subgraph Critic["Critic (Validator)"]
        C1[Evaluate correctness]
        C2[Check grounding & hallucination]
        C3[Assess security & compliance]
        C1 --> C2 --> C3
    end

    A3 -->|response + tool artifacts| C1

    C3 -->|PASS| DONE["✓ Return to user"]
    C3 -->|SALVAGEABLE| FIX["Critic rewrites text<br/>(no new tool calls)"]
    C3 -->|NON-SALVAGEABLE| FB["Feedback → Actor<br/>(re-runs tools)"]

    FIX --> DONE
    FB --> A1

    style Actor fill:#e8f4f8,stroke:#2980b9,stroke-width:2px
    style Critic fill:#fdf2e9,stroke:#e67e22,stroke-width:2px
```

### Why Cross-Model Diversity Matters

| Concern | Mitigation |
|---|---|
| Shared blind spots | Use different model families (e.g., GPT-series Actor, Claude-series Critic) so systematic biases do not overlap |
| Sycophantic agreement | A Critic from a different provider is less likely to rubber-stamp the Actor's reasoning |
| Cost efficiency | The Critic can run on a smaller, cheaper model since validation is narrower than generation |
| Latency control | Critic evaluation runs only once per attempt and can use a faster model tier |

> **Example from practice**: A production analytics platform may pair a high-capability Actor model with a cost-efficient Critic from a different family, achieving both quality and budget goals.

---

## 2. End-to-End Sequence Diagram

The following diagram traces a single user request from submission through Actor generation, tool execution, Critic validation (with correction branches), and final rendering.

```mermaid
sequenceDiagram
    participant User
    participant Interface
    participant Orchestrator
    participant PromptBuilder
    participant ActorLLM
    participant ToolHandlers
    participant CriticValidator
    participant CriticLLM
    participant Persistence
    participant FileStorage

    User->>Interface: Submit request
    Interface->>Orchestrator: dispatch(request, config)

    %% --- System prompt assembly ---
    Orchestrator->>PromptBuilder: build_system_message(use_case, context)
    PromptBuilder-->>Orchestrator: system_message

    %% --- Actor first call ---
    Orchestrator->>ActorLLM: chat.completions.create(messages, tools)
    ActorLLM-->>Orchestrator: response (may contain tool_calls)

    Orchestrator->>Persistence: log_usage(model, tokens, cost)

    %% --- Tool call loop ---
    loop While response contains tool_calls
        Orchestrator->>ToolHandlers: execute(tool_name, args)
        ToolHandlers-->>Orchestrator: tool_result

        Orchestrator->>Orchestrator: append tool_result to messages
        Orchestrator->>FileStorage: checkpoint(messages)
        Orchestrator->>Persistence: checkpoint(session)

        Orchestrator->>ActorLLM: chat.completions.create(messages, tools)
        ActorLLM-->>Orchestrator: response (may contain more tool_calls)

        Orchestrator->>Persistence: log_usage(model, tokens, cost)
    end

    %% --- Critic validation ---
    alt Critic enabled
        loop attempt = 1 to max_attempts
            Orchestrator->>CriticValidator: validate(question, response, tool_results)
            CriticValidator->>CriticLLM: evaluate(rubric, evidence)
            CriticLLM-->>CriticValidator: verdict {pass | salvageable | non_salvageable}
            CriticValidator-->>Orchestrator: validation_result

            Orchestrator->>Persistence: log_usage(critic_model, tokens, cost)

            alt PASS
                Orchestrator->>Orchestrator: accept response
            else SALVAGEABLE
                CriticValidator-->>Orchestrator: corrected_text
                Orchestrator->>Orchestrator: replace response text
            else NON-SALVAGEABLE
                Orchestrator->>Orchestrator: format correction prompt
                Orchestrator->>Orchestrator: shallow copy messages + _internal tag

                rect rgb(255, 245, 238)
                    Note over Orchestrator,ActorLLM: Recursive self-correction call
                    Orchestrator->>ActorLLM: run_completion(type=critic_feedback)
                    ActorLLM-->>Orchestrator: corrected response (own tool loop + critic)
                end

                Orchestrator->>Orchestrator: strip _internal messages
            end

            Note over Orchestrator: Exit loop when validation<br/>passed or max attempts reached
        end
    end

    %% --- Final delivery ---
    Orchestrator->>FileStorage: save_final(messages, artifacts)
    Orchestrator->>Persistence: save_session(messages, metadata)
    Orchestrator-->>Interface: final_response
    Interface-->>User: render response
```

### Key Observations

- **Tool calls are iterative**: the Actor may chain multiple rounds of tool calls before producing a final text response.
- **Checkpointing after every tool round** enables crash recovery without re-executing expensive tool calls.
- **Non-salvageable correction is recursive**: the inner `run_completion` call runs its own independent tool loop and Critic validation, converging on a corrected response.
- **Internal messages never leak**: `_internal`-tagged messages are stripped before the response reaches the user.

---

## 3. System Prompt Construction

The system message is the single most important lever for controlling agent behavior. It is assembled dynamically at request time from composable fragments.

```mermaid
flowchart LR
    BP["Base Behavioral Prompt<br/><i>Core rules & persona</i>"]
    DI["Domain Instructions<br/><i>Task-specific behavior</i>"]
    DK["Domain Knowledge<br/><i>Schemas, APIs, types</i>"]
    MD["Metadata<br/><i>Data sources, tool descriptions</i>"]
    UC["User Context<br/><i>Preferences, history</i>"]
    TS["Timestamp<br/><i>Current date/time</i>"]

    BP --> ASM["Prompt Builder<br/><b>assemble()</b>"]
    DI --> ASM
    DK --> ASM
    MD --> ASM
    UC --> ASM
    TS --> ASM

    ASM --> SM["system_message"]

    style ASM fill:#d5f5e3,stroke:#27ae60,stroke-width:2px
```

### Assembly Pattern

```python
def build_system_message(
    base_prompt: str,
    domain_instructions: str,
    domain_knowledge: str,
    sources: list[dict],
    user_context: dict | None = None,
    current_timestamp: str | None = None,
) -> str:
    current_timestamp = current_timestamp or datetime.now().isoformat()

    system_message = f"""# Instructions
{base_prompt}

## Domain Context
{domain_instructions}

## Domain Knowledge
{domain_knowledge}

## Available Data Sources
{format_data_sources(sources)}

## Current Timestamp
{current_timestamp}
"""

    if user_context:
        system_message += f"\n## User Context\n{format_user_context(user_context)}\n"

    return system_message
```

### Key Design Decisions Embedded in the System Prompt

| Decision | Rationale |
|---|---|
| **Data-driven only — no recommendations** | The agent presents facts and computed results; it does not speculate beyond what tools return. Prevents hallucinated advice. |
| **Clarify before acting** | When the request is ambiguous, the agent asks a clarifying question rather than guessing intent. Reduces wasted tool calls. |
| **Self-validation checklist** | The prompt includes a pre-flight checklist the Actor must mentally run before responding (e.g., "Did I answer the exact question asked?"). Acts as a lightweight inner critic. |
| **Tool-first computation** | Arithmetic, aggregations, and lookups must go through tools — the LLM must not perform calculations in-context. Eliminates a class of numerical hallucinations. |
| **Sequential tool calls** | Tools are called one at a time (no parallel dispatch) so each call can use results from the previous one. Simplifies debugging and cost attribution. |

---

## 4. The LLM Interaction Cycle

### API Argument Preparation

Different model families require different API arguments. The orchestrator normalizes this behind a model-family dispatcher.

```mermaid
flowchart TD
    REQ["Prepare API call"]
    REQ --> DETECT{"Detect model family"}

    DETECT -->|OpenAI-compatible| OA["Set temperature, top_p<br/>Set tool_choice if forced<br/>Set max_tokens"]
    DETECT -->|Anthropic-compatible| AN["Set temperature<br/>Set max_tokens<br/>Configure extended thinking<br/>(budget_tokens)"]
    DETECT -->|Custom endpoint| CU["Apply endpoint-specific<br/>parameter mapping"]

    OA --> CALL["chat.completions.create()"]
    AN --> CALL
    CU --> CALL

    style DETECT fill:#fdebd0,stroke:#f39c12,stroke-width:2px
```

### Response Processing Pipeline

Every LLM response — whether from the Actor or the Critic — passes through the same 6-step pipeline.

```mermaid
flowchart TD
    RESP["Raw LLM Response"] --> S1

    S1["1 · Extract usage & cost<br/><i>tokens_in, tokens_out, cache</i>"]
    S1 --> S2["2 · Log usage to persistence<br/><i>model, tokens, cost, latency</i>"]
    S2 --> S3{"3 · Critic validation?"}

    S3 -->|Yes, this is an Actor response<br/>and Critic is enabled| CV["Run CriticValidator<br/>(see §6 for correction flow)"]
    S3 -->|No| S4

    CV --> S4["4 · Append output message<br/><i>assistant role, content, reasoning</i>"]
    S4 --> S5["5 · Append tool calls<br/><i>if response contains tool_calls</i>"]
    S5 --> S6["6 · Fire context update callback<br/><i>UI re-render, progress indicator</i>"]

    S6 --> RET["Return to orchestrator loop"]

    style S3 fill:#fdebd0,stroke:#e67e22,stroke-width:2px
```

**Step 1 — Cost extraction** deserves special attention: the raw `usage` object varies by provider, so a normalizer maps provider-specific fields (e.g., `cache_creation_input_tokens`, `prompt_tokens_details.cached_tokens`) to a canonical schema before cost calculation.

---

## 5. Tool Call Loop Mechanics

### State Machine

The tool-call loop is the inner engine of the Actor agent. It runs until the LLM produces a response with no further tool calls, then hands off to the Critic.

```mermaid
stateDiagram-v2
    [*] --> InitialCall: submit messages + tools

    InitialCall --> ProcessResponse: LLM returns response

    ProcessResponse --> CheckTools: extract tool_calls

    CheckTools --> ExecuteTools: tool_calls present
    CheckTools --> CriticValidation: no tool_calls

    ExecuteTools --> Checkpoint: all tools executed
    Checkpoint --> ReSubmit: append results to messages
    ReSubmit --> ProcessResponse: LLM returns response

    CriticValidation --> [*]: return final response

    note right of ExecuteTools
        Each tool runs sequentially.
        Results appended as role=tool messages.
    end note

    note right of Checkpoint
        Messages saved to file storage
        and relational DB after every round.
    end note
```

### Tool Execution Pattern

```python
for tool_call in response.tool_calls:
    tool_name = tool_call.function.name
    args = json.loads(tool_call.function.arguments)

    if tool_name in tool_handlers:
        result = tool_handlers[tool_name](**args)
    else:
        result = f"Unknown tool: {tool_name}"

    messages.append({
        "role": "tool",
        "tool_call_id": tool_call.id,
        "content": str(result),
    })
```

### Tool Registration Pattern

Tools are defined as paired *specification* + *handler* entries. A factory function splits them into the two structures the orchestrator needs.

```mermaid
flowchart LR
    TC["tool_config: list of dicts<br/><code>[{spec, handler}, ...]</code>"]
    TC --> EXT["extract_tools_and_handlers()"]

    EXT --> SPECS["tools: list[dict]<br/><i>OpenAI function specs</i><br/>passed to API call"]
    EXT --> HANDLERS["handlers: dict[str, callable]<br/><i>name → function mapping</i><br/>used at execution time"]

    style EXT fill:#d5f5e3,stroke:#27ae60,stroke-width:2px
```

```python
def extract_tools_and_handlers(
    tool_config: list[dict],
) -> tuple[list[dict], dict[str, callable]]:
    tools = [entry["spec"] for entry in tool_config]
    handlers = {
        entry["spec"]["function"]["name"]: entry["handler"]
        for entry in tool_config
    }
    return tools, handlers
```

This separation keeps tool definitions declarative and co-located (easy to audit) while giving the orchestrator efficient O(1) handler lookup at runtime.

---

## 6. Self-Correction Flow

Self-correction is the mechanism by which non-salvageable Critic feedback triggers the Actor to re-generate its response from scratch, potentially re-running tools. The design uses **recursive orchestrator invocation** to reuse all existing logic (tool loop, checkpointing, Critic validation) without duplication.

### Detailed Sequence

```mermaid
sequenceDiagram
    participant Orchestrator as Orchestrator (outer)
    participant ActorLLM
    participant CriticValidator
    participant CriticLLM
    participant InnerOrch as Orchestrator (inner / recursive)

    Note over Orchestrator: Actor has produced a response

    Orchestrator->>CriticValidator: validate(response, tool_results)
    CriticValidator->>CriticLLM: evaluate(rubric, evidence)
    CriticLLM-->>CriticValidator: verdict = NON_SALVAGEABLE
    CriticValidator-->>Orchestrator: {needs_correction: true, issues, severity}

    Note over Orchestrator: Format correction prompt

    Orchestrator->>Orchestrator: shallow_copy(messages)
    Orchestrator->>Orchestrator: append correction_message(_internal=True)

    rect rgb(240, 248, 255)
        Note over InnerOrch,ActorLLM: Recursive run_completion(type=critic_feedback)
        Orchestrator->>InnerOrch: run_completion(messages_copy, type=critic_feedback)

        InnerOrch->>ActorLLM: chat.completions.create(messages, tools)
        ActorLLM-->>InnerOrch: response (may include tool_calls)

        loop Tool call loop (inner)
            InnerOrch->>InnerOrch: execute tools, checkpoint, re-submit
        end

        InnerOrch->>CriticValidator: validate(inner_response)
        CriticValidator->>CriticLLM: evaluate(rubric, evidence)
        CriticLLM-->>CriticValidator: verdict (PASS or SALVAGEABLE)
        CriticValidator-->>InnerOrch: validation_result

        InnerOrch-->>Orchestrator: corrected_response
    end

    Orchestrator->>Orchestrator: strip _internal messages
    Orchestrator->>Orchestrator: set assistant message content = None (tool calls only)
    Orchestrator->>Orchestrator: adopt corrected_response as final
```

### Two-Layer Prompt Guardrailing

The user must never see evidence of corrections. Two independent guardrails enforce this.

```mermaid
flowchart TD
    subgraph Layer1["Layer 1 — System Prompt Rule"]
        SP["System prompt contains:<br/><i>'Never reveal that your response<br/>was corrected or validated.<br/>Do not reference internal processes.'</i>"]
    end

    subgraph Layer2["Layer 2 — Correction Prompt Directive"]
        CP["Correction message contains:<br/><i>'Generate a response as if answering<br/>for the first time. Do not mention<br/>any prior attempt or correction.'</i>"]
    end

    SP --> COMBINED["Combined effect"]
    CP --> COMBINED

    COMBINED --> RESULT["User receives a clean, natural response<br/>with no trace of internal correction"]

    style Layer1 fill:#ebdef0,stroke:#8e44ad,stroke-width:2px
    style Layer2 fill:#d4efdf,stroke:#27ae60,stroke-width:2px
```

Having two independent layers means the guardrail holds even if one layer is partially ignored by the LLM. This is defense-in-depth applied to prompt engineering.

### Internal Message Lifecycle

Internal messages exist only for the duration of a correction cycle and are never persisted or shown to the user.

```mermaid
flowchart LR
    CREATE["1 · Creation<br/><i>Tagged _internal: True</i><br/><i>role: user (correction prompt)</i>"]
    CREATE --> API["2 · API Call<br/><i>Stripped from messages<br/>before any external send</i>"]
    API --> CLEANUP["3 · Cleanup<br/><i>Filtered from conversation<br/>history after correction</i>"]
    CLEANUP --> NEVER["4 · Never Persisted<br/><i>Not saved to DB<br/>Not saved to file storage<br/>Not visible in session replay</i>"]

    style CREATE fill:#d6eaf8,stroke:#2980b9
    style NEVER fill:#fadbd8,stroke:#e74c3c
```

### Correction Prompt Template

```text
VALIDATION FAILED - CORRECTION REQUIRED

Issues detected: {detected_issues}
Root Cause: {root_cause}
Severity: {severity}
Recommendation: {recommendation}

Generate a corrected response addressing all identified issues.
Respond as if answering the user's question for the first time.
Do not reference any prior attempt, validation, or correction process.
```

The template is deliberately terse and structured. Verbose correction prompts risk the LLM echoing the correction language back to the user. Bullet-point issues with a clear directive minimize that risk.

---

## 7. Session Checkpointing

Checkpointing preserves conversation state after every tool-call round, enabling crash recovery without re-executing expensive or non-idempotent tool calls.

```mermaid
flowchart TD
    TRIGGER["Checkpoint trigger:<br/>after each tool-call round"]
    TRIGGER --> MODE{"Execution mode?"}

    MODE -->|Interactive| INT["Save via UI callback<br/><i>Session state updated in-place</i>"]
    MODE -->|Headless| HEAD["Save via direct call<br/><i>No UI dependency</i>"]

    INT --> FS["File Storage<br/><i>Full message history as JSON<br/>Artifact files (code, CSVs, etc.)</i>"]
    INT --> DB["Relational DB<br/><i>Session metadata<br/>Tool call log<br/>Cost accumulator</i>"]

    HEAD --> FS
    HEAD --> DB

    FS --> RECOVERY["Recovery benefits"]
    DB --> RECOVERY

    RECOVERY --> R1["Resume from last checkpoint<br/>on crash or timeout"]
    RECOVERY --> R2["Replay tool results<br/>without re-execution"]
    RECOVERY --> R3["Audit trail for<br/>every intermediate state"]

    style TRIGGER fill:#d5f5e3,stroke:#27ae60,stroke-width:2px
    style RECOVERY fill:#fdebd0,stroke:#f39c12,stroke-width:2px
```

---

## 8. Cost Tracking Pipeline

Every LLM call — Actor or Critic, outer or recursive — feeds into a unified cost-tracking pipeline.

```mermaid
flowchart LR
    subgraph Config["Pricing Configuration"]
        PC["pricing_config: dict<br/><i>model → rate per token type</i><br/><i>(input, output, cache_read, cache_write)</i>"]
    end

    subgraph Calc["Cost Calculation"]
        USAGE["Raw usage from API<br/><i>tokens_in, tokens_out,<br/>cache_read, cache_write</i>"]
        USAGE --> NORM["Normalize to<br/>canonical schema"]
        NORM --> MULT["Multiply each token type<br/>by its rate from config"]
        MULT --> SUM["Sum = call cost"]
    end

    subgraph Accum["Accumulation"]
        SUM --> SESS["Session accumulator<br/><i>running total across all calls</i>"]
        SESS --> DISPLAY["Display to user<br/><i>(interactive mode)</i>"]
    end

    subgraph Log["Logging"]
        SUM --> PERSIST["Persist to DB<br/><i>model, tokens, cost,<br/>latency, call_type</i>"]
    end

    PC --> MULT

    style Config fill:#ebdef0,stroke:#8e44ad
    style Calc fill:#d6eaf8,stroke:#2980b9
    style Accum fill:#d5f5e3,stroke:#27ae60
    style Log fill:#fdebd0,stroke:#e67e22
```

### Rate Normalization

Providers express rates differently (per-million tokens, per-thousand tokens, in dollars, in platform-specific units). The cost module normalizes everything to a single canonical unit (e.g., USD) using a conversion factor from the pricing config.

```python
def calculate_cost(usage: dict, model: str, pricing: dict) -> float:
    rates = pricing[model]
    cost = (
        usage["input_tokens"] * rates["input"]
        + usage["output_tokens"] * rates["output"]
        + usage.get("cache_read_tokens", 0) * rates.get("cache_read", 0)
        + usage.get("cache_write_tokens", 0) * rates.get("cache_write", 0)
    )
    return cost * rates.get("unit_conversion", 1.0)
```

---

## 9. Headless vs Interactive Execution

The same orchestrator backend drives both interactive (UI-driven) and headless (API/batch) execution modes. The difference is entirely in how callbacks and configuration are wired.

```mermaid
flowchart TD
    subgraph Shared["Shared Backend"]
        ORCH["Orchestrator<br/><i>run_completion()</i>"]
        TOOLS["Tool Loop"]
        CRITIC["Critic Validation"]
        COST["Cost Tracking"]
        CHECK["Checkpointing"]

        ORCH --> TOOLS --> CRITIC --> COST --> CHECK
    end

    subgraph Interactive["Interactive Mode"]
        ICFG["from_interactive(config)<br/><i>UI session state as context</i>"]
        ICB["Callbacks wired:<br/>on_progress → update spinner<br/>on_message → render in chat<br/>on_checkpoint → save session state<br/>on_cost → update cost display"]
    end

    subgraph Headless["Headless Mode"]
        HCFG["from_headless(config)<br/><i>Standalone config dict</i>"]
        HCB["All callbacks = None<br/><i>Results returned as<br/>structured response object</i>"]
    end

    ICFG --> ORCH
    ICB --> ORCH
    HCFG --> ORCH
    HCB --> ORCH

    ORCH -->|Interactive| IOUT["UI renders incrementally<br/><i>Streaming messages,<br/>progress indicators</i>"]
    ORCH -->|Headless| HOUT["Structured return value<br/><i>JSON with response,<br/>tool_results, cost, metadata</i>"]

    style Shared fill:#eaf2f8,stroke:#2c3e50,stroke-width:2px
    style Interactive fill:#d5f5e3,stroke:#27ae60,stroke-width:2px
    style Headless fill:#fdebd0,stroke:#e67e22,stroke-width:2px
```

### Configuration Factory Pattern

```python
@classmethod
def from_interactive(cls, session_state: dict) -> "Orchestrator":
    return cls(
        messages=session_state["messages"],
        tools=session_state["tools"],
        on_progress=lambda msg: update_spinner(msg),
        on_message=lambda msg: render_chat_message(msg),
        on_checkpoint=lambda: save_session_state(),
        on_cost=lambda c: update_cost_display(c),
    )

@classmethod
def from_headless(cls, config: dict) -> "Orchestrator":
    return cls(
        messages=config["messages"],
        tools=config["tools"],
        on_progress=None,
        on_message=None,
        on_checkpoint=None,
        on_cost=None,
    )
```

The callback-or-None pattern means the orchestrator's core logic contains no UI imports and no conditional branches for "am I headless?". Each callback site is simply:

```python
if self.on_progress:
    self.on_progress("Executing tool: " + tool_name)
```

This makes the orchestrator independently testable and deployable in serverless / batch contexts without any UI framework dependency.

---

## Summary

| Phase | Actor | Critic | Key Artifact |
|---|---|---|---|
| Prompt construction | Receives assembled system message | — | `system_message` |
| Generation | Calls LLM, executes tools iteratively | — | `messages[]`, tool results |
| Validation | — | Evaluates response against rubric | `validation_result` |
| Salvageable fix | — | Rewrites text in-place | Corrected `content` |
| Non-salvageable fix | Re-runs via recursive call with feedback | Re-validates inner response | Corrected response (internal messages stripped) |
| Checkpointing | After every tool round | — | File storage + DB snapshot |
| Cost tracking | Every LLM call logged | Every LLM call logged | `cost_accumulator` |

The workflow's power comes from composability: the same `run_completion` function handles first attempts, recursive corrections, and headless batch runs. The Critic is an opt-in layer that bolts onto the existing tool loop without modifying it. And checkpointing ensures that even long, multi-tool conversations can recover gracefully from failures.
