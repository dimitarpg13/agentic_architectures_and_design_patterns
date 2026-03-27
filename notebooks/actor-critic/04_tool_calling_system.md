# 04 — Tool Calling System

> **Series**: Actor-Critic Agent Design Pattern
> **Running Example**: A *code generation* agent where the Actor produces source code and can execute it, run tests, perform static analysis, and lint — while the Critic validates correctness, style, and safety.

---

## 1. Tool Architecture Overview

The tool calling subsystem bridges two execution environments — **remote** (cloud-sandboxed, higher latency, stronger isolation) and **local** (in-process, low latency, weaker isolation) — unified by shared infrastructure for authentication, security, and token accounting.

```mermaid
graph TB
    subgraph Actor["Actor Agent"]
        LLM["LLM with Tool Calling"]
    end

    subgraph SharedInfra["Shared Infrastructure"]
        Auth["Auth Manager"]
        Security["Security Validator"]
        TokenCounter["Token Counter"]
    end

    subgraph Remote["Remote Execution Environment"]
        CodeExec["Code Executor<br/>(Cloud Sandbox)"]
        DBQuery["Database Query Engine"]
    end

    subgraph Local["Local Execution Environment"]
        StaticAnalyzer["Static Analyzer"]
        TestRunner["Unit Test Runner"]
        Linter["Linter"]
    end

    LLM -->|tool_call| Auth
    Auth --> Security
    Security -->|remote| CodeExec
    Security -->|remote| DBQuery
    Security -->|local| StaticAnalyzer
    Security -->|local| TestRunner
    Security -->|local| Linter
    CodeExec --> TokenCounter
    DBQuery --> TokenCounter
    StaticAnalyzer --> TokenCounter
    TestRunner --> TokenCounter
    Linter --> TokenCounter
    TokenCounter -->|tool result| LLM
```

### Tool Comparison Matrix

| Tool | Environment | Typical Latency | Isolation Level | Output Size Risk | Auth Required |
|------|-------------|----------------|-----------------|------------------|---------------|
| **Code Executor** | Remote | 2–30 s | Full sandbox | High (stdout/files) | Yes |
| **Static Analyzer** | Local | 0.1–1 s | Process-level | Low (findings list) | No |
| **Unit Test Runner** | Local | 1–10 s | Process-level | Medium (test report) | No |
| **Linter** | Local | 0.1–0.5 s | Process-level | Low (diagnostics) | No |
| **Custom domain tools** | Varies | Varies | Varies | Varies | Varies |

Remote tools carry higher latency but provide strong isolation guarantees — critical when executing LLM-generated code. Local tools trade isolation for speed and are appropriate for read-only analysis that cannot modify system state.

---

## 2. Tool Registration and Dispatch

### Spec + Handler Pattern

Every tool is a **spec + handler** pair. The spec is an OpenAI-compatible function definition sent to the LLM; the handler is the callable that runs when the LLM invokes that tool.

```mermaid
classDiagram
    class ToolConfig {
        +ToolSpec spec
        +Handler handler
    }

    class ToolSpec {
        +String type = "function"
        +FunctionDef function
    }

    class FunctionDef {
        +String name
        +String description
        +JSONSchema parameters
    }

    class Handler {
        +__call__(args: dict) str
    }

    ToolConfig --> ToolSpec
    ToolConfig --> Handler
    ToolSpec --> FunctionDef
```

This separation keeps the LLM-facing contract (spec) decoupled from execution logic (handler), enabling independent evolution of either side.

### Registration in the Actor Agent

Tools are assembled into a list of `ToolConfig` dicts during agent setup. Domain-specific tools — loaded from a preprocessor — are appended after the core set.

```python
tool_config = [
    {
        'spec': code_executor_spec,
        'handler': lambda args: execute_code(
            code=args.get('code'),
            language=args.get('language'),
            reason=args.get('reason'),
        )
    },
    {
        'spec': static_analyzer_spec,
        'handler': lambda args: run_static_analysis(
            code=args.get('code'),
            language=args.get('language'),
            rules=args.get('rules', 'default'),
        )
    },
    {
        'spec': test_runner_spec,
        'handler': lambda args: run_tests(
            code=args.get('code'),
            test_code=args.get('test_code'),
            framework=args.get('framework', 'pytest'),
        )
    },
]

# Domain-specific tools appended
for tool in config.additional_tool_specs:
    tool_config.append(tool)
```

### Extraction and Dispatch

Before the first LLM call, the unified `tool_config` list is split into two structures:

```mermaid
flowchart LR
    TC["tool_config<br/>(list of spec+handler dicts)"]
    EX["extract_tools_and_handlers()"]
    SPECS["tools<br/>(list of specs)<br/>→ sent to LLM"]
    HANDLERS["handlers<br/>(dict: name → callable)<br/>→ used at dispatch time"]

    TC --> EX
    EX --> SPECS
    EX --> HANDLERS
```

At dispatch time, the orchestrator reads the `function.name` from the LLM's `tool_call`, looks it up in the `handlers` dict, deserializes the JSON arguments, and invokes the callable. The string result is wrapped in a `tool` role message and appended to the conversation.

---

## 3. Code Execution: Remote Sandbox

Remote code execution is the highest-risk tool. The sequence below shows every validation and isolation gate between the Actor's tool call and the sandbox.

```mermaid
flowchart TD
    A1["1. Actor sends code to Execution Wrapper"]
    A2["2. Wrapper forwards to Security Validator"]
    A3["3. Validator performs checks:<br/>Parse AST<br/>Check imports against allowlist<br/>Check for dangerous calls<br/>Verify structural validity"]
    A3 -->|Fails| A4["SecurityError returned to Actor"]
    A3 -->|Passes| A5["4. Wrapper sends code to Cloud Sandbox"]
    A5 --> A6["Isolated environment:<br/>no network, no filesystem,<br/>restricted stdlib"]
    A6 --> A7["5. Sandbox returns execution result"]
    A7 --> A8{"Output within size limit?"}
    A8 -->|Yes| A9["6. Formatted result returned to Actor"]
    A8 -->|No| A10["Truncate + warn"] --> A9

    A1 --> A2 --> A3

    style A4 fill:#f8d7da,stroke:#721c24
    style A9 fill:#d4edda,stroke:#28a745
```

### Code Execution Guardrails

Every piece of LLM-generated code passes through a multi-stage validation pipeline before it reaches the sandbox.

```mermaid
flowchart TD
    A["Receive code from Actor"] --> B["Strip string literals<br/>(prevent false positives<br/>in static checks)"]
    B --> C{"Forbidden operations?<br/>file I/O · network · system calls"}
    C -->|Yes| REJECT["Reject with<br/>SecurityError"]
    C -->|No| D{"Imports on allowlist?"}
    D -->|No| REJECT
    D -->|Yes| E{"Valid program?<br/>(parseable AST)"}
    E -->|No| REJECT
    E -->|Yes| F["Execute in sandboxed<br/>environment with timeout"]
    F --> G{"Output within<br/>size limit?"}
    G -->|No| H["Truncate + warn"]
    G -->|Yes| I["Return result<br/>to Actor"]
    H --> I
```

String literal stripping (step 1) prevents false positives — the code `print("do not call os.system()")` should not be flagged. The actual AST walk operates on the structural representation, not raw text.

---

## 4. Static Analysis: Local Execution

Static analysis runs in-process with no sandbox overhead, making it suitable for rapid feedback loops during iterative code generation.

```mermaid
flowchart TD
    B1["1. Actor calls static analysis tool<br/>with code, language, rules"]
    B2["2. run_static_analysis forwards<br/>code to validate_code_security"]
    B3["3. Security validator parses AST"]
    B4["4. Walk AST nodes, apply rules"]
    B5["5. Return findings list"]
    B6["6. Formatted findings report<br/>returned to Actor"]

    B1 --> B2 --> B3 --> B4 --> B5 --> B6

    style B1 fill:#e8f4f8,stroke:#2980b9
    style B6 fill:#d4edda,stroke:#28a745
```

### Three Execution Variants

The code analysis subsystem supports three modes, selected based on the nature of the task the Actor is performing.

```mermaid
flowchart TD
    INPUT["Code from Actor"] --> MODE{"Execution mode?"}

    MODE -->|expression| EXPR["Expression Mode"]
    EXPR --> EXPR_DESC["Single-line evaluation<br/>(eval-style)<br/>Returns computed value"]

    MODE -->|function| FUNC["Function Mode"]
    FUNC --> FUNC_DESC["Multi-line function definition<br/>(exec to define, then call)<br/>Returns function output"]

    MODE -->|analysis| ANAL["Analysis Mode"]
    ANAL --> ANAL_DESC["Parse and analyze only<br/>(AST walk, no execution)<br/>Returns structural findings"]

    EXPR_DESC --> RESULT["Result → Actor"]
    FUNC_DESC --> RESULT
    ANAL_DESC --> RESULT
```

| Mode | Use Case | Executes Code? | Typical Latency |
|------|----------|---------------|-----------------|
| **Expression** | Quick computations, metric formulas | Yes (eval) | < 100 ms |
| **Function** | Complex transformations, multi-step logic | Yes (exec + call) | 100 ms – 1 s |
| **Analysis** | Style checks, complexity metrics, import audits | No (AST only) | 50–200 ms |

Analysis mode is the only variant that can safely run on untrusted code without a sandbox, since it never executes — it only parses and inspects the tree.

---

## 5. Security Model

### Defense Layers

Security is not a single gate but a series of independent layers, each catching a different class of threat. A failure at any layer halts execution.

```mermaid
flowchart TD
    CODE["LLM-generated code"] --> L1

    subgraph L1["Layer 1: AST-Based Static Analysis"]
        PARSE["Parse source → AST"]
        WALK["Walk all nodes"]
        IMPORTS["Check imports against allowlist"]
        CALLS["Check function calls against blocklist"]
        PARSE --> WALK --> IMPORTS --> CALLS
    end

    L1 -->|pass| L2

    subgraph L2["Layer 2: Restricted Execution Environment"]
        GLOBALS["Sandboxed globals<br/>(curated __builtins__)"]
        NO_IO["No file I/O primitives"]
        NO_NET["No network access"]
        GLOBALS --- NO_IO --- NO_NET
    end

    L2 -->|pass| L3

    subgraph L3["Layer 3: Execution Constraints"]
        TIMEOUT["Timeout enforcement<br/>(default 10s)"]
        RESOURCES["Resource limits<br/>(memory, CPU)"]
        OUTPUT["Output size cap<br/>(bytes + lines)"]
        TIMEOUT --- RESOURCES --- OUTPUT
    end

    L3 -->|pass| L4

    subgraph L4["Layer 4: Code Validation Guardrails"]
        BLACKLIST["Dangerous operation blacklist<br/>(eval, exec, compile, __import__)"]
        STRUCTURAL["Structural checks<br/>(no infinite loops heuristic,<br/>no recursive depth bombs)"]
        BLACKLIST --- STRUCTURAL
    end

    L4 -->|pass| SAFE["Safe to return result"]
    L1 -->|fail| BLOCK["Block execution +<br/>return SecurityError"]
    L2 -->|fail| BLOCK
    L3 -->|fail| BLOCK
    L4 -->|fail| BLOCK
```

### Allowed vs. Forbidden Operations

```mermaid
graph LR
    subgraph Allowed["✅ Allowed"]
        MATH["math · statistics<br/>standard arithmetic"]
        STRING["String operations<br/>formatting · regex"]
        DATA["Data structures<br/>list · dict · set · tuple"]
        DT["datetime · collections<br/>itertools · functools"]
    end

    subgraph Forbidden["❌ Forbidden"]
        EVAL["eval() · exec() · compile()"]
        OPEN["open() · pathlib.Path.read"]
        IMPORT["__import__() · importlib"]
        PROC["subprocess · os.system<br/>os.popen · shutil"]
        NET["socket · urllib · requests<br/>http.client"]
    end

    subgraph Controlled["⚠️ Controlled Access"]
        VIZ["Visualization libraries<br/>(only via dedicated tool)"]
        HEAVY["Heavy computation<br/>(only in remote sandbox)"]
        EXTERN["External API calls<br/>(only via approved tools)"]
    end
```

The **Controlled** category covers operations that are legitimate but must go through a specific tool rather than raw code execution. For example, generating a chart is allowed — but only through a dedicated visualization tool that handles rendering in a controlled context, not via arbitrary `matplotlib` calls in user code.

---

## 6. Custom Domain Tools

The tool system is extensible. Each domain-specific preprocessor (one per use case) can register additional tools that the Actor discovers at setup time.

```mermaid
flowchart TD
    UC["Use Case Configuration"] --> PP["Preprocessor"]
    PP --> CHECK{"Has additional<br/>tool specs?"}
    CHECK -->|No| CORE["Use core tools only"]
    CHECK -->|Yes| LOAD["Load additional_tool_specs"]
    LOAD --> MERGE["Append to tool_config"]
    MERGE --> EXTRACT["extract_tools_and_handlers()"]
    EXTRACT --> SPECS["Updated specs → LLM"]
    EXTRACT --> HANDLERS["Updated handlers → dispatch"]
    CORE --> EXTRACT
```

Below is an example custom tool that generates a type-safe API client from an OpenAPI specification — a domain-specific capability that makes sense for a code generation use case.

```python
custom_tool_spec = {
    'type': 'function',
    'function': {
        'name': 'generate_api_client',
        'description': 'Generate a type-safe API client from an OpenAPI specification',
        'parameters': {
            'type': 'object',
            'properties': {
                'spec_url': {'type': 'string', 'description': 'URL to OpenAPI spec'},
                'language': {'type': 'string', 'enum': ['python', 'typescript', 'go']},
                'auth_method': {'type': 'string', 'enum': ['bearer', 'api_key', 'oauth2']},
            },
            'required': ['spec_url', 'language'],
        },
    },
}


def handle_generate_api_client(args: dict) -> str:
    spec_url = args['spec_url']
    language = args['language']
    auth_method = args.get('auth_method', 'bearer')

    spec = fetch_and_validate_openapi_spec(spec_url)
    client_code = render_client_template(spec, language, auth_method)
    return client_code


custom_tool_config = {
    'spec': custom_tool_spec,
    'handler': handle_generate_api_client,
}
```

Custom tools follow the same spec + handler contract as core tools. The only additional requirement is that they must be **self-contained** — they cannot depend on internal agent state or assume a particular conversation history.

---

## 7. Result Size Management

LLM context windows are finite and expensive. Tool results that exceed token limits degrade reasoning quality or cause hard failures. The system enforces limits at multiple levels.

```mermaid
flowchart TD
    RESULT["Tool execution result"] --> EL{"Execution level:<br/>max_output_lines?<br/>max_output_bytes?"}
    EL -->|Exceeds| TRUNC_EXEC["Truncate at execution level<br/>+ append warning"]
    EL -->|Within limits| TL{"Token level:<br/>len(result) ><br/>MAX_TOOL_OUTPUT_TOKENS?"}
    TRUNC_EXEC --> TL
    TL -->|Exceeds| TRUNC_TOKEN["Truncate to token limit<br/>+ append warning"]
    TL -->|Within limits| OK["Append full result<br/>to conversation"]
    TRUNC_TOKEN --> WARN["Agent sees truncation warning<br/>and can adjust strategy"]
    OK --> CONTINUE["Continue generation"]
    WARN --> CONTINUE
```

### The Pagination Protocol

When a tool result exceeds limits, the Actor is expected to adapt its strategy rather than accept a truncated result.

```mermaid
flowchart TD
    LARGE["Result too large"] --> S1["Strategy 1:<br/>Optimize Output"]
    S1 --> S1_DESC["Rewrite code to produce<br/>only needed columns/rows<br/>(e.g., add filters, select fewer fields)"]
    S1_DESC --> S1_CHECK{"Fits now?"}
    S1_CHECK -->|Yes| DONE["Use result"]
    S1_CHECK -->|No| S2["Strategy 2:<br/>Smart Pagination"]
    S2 --> S2_DESC["Paginate along a meaningful<br/>dimension (e.g., by category,<br/>by time window, by module)"]
    S2_DESC --> S2_CHECK{"Fits now?"}
    S2_CHECK -->|Yes| DONE
    S2_CHECK -->|No| S3["Strategy 3:<br/>Chunk-Based Retrieval"]
    S3 --> S3_DESC["Retrieve fixed-size chunks<br/>with OFFSET/LIMIT or<br/>cursor-based iteration"]
    S3_DESC --> DONE
```

The protocol is encoded in the Actor's system prompt as a behavioral guideline. The Critic can detect violations — for example, when the Actor accepts a truncated result without attempting optimization — and flag them during validation.

---

## 8. Error Handling Patterns

### Tool Execution Error Flow

Every tool call passes through a unified dispatch function that handles deserialization failures, missing handlers, and execution errors uniformly.

```mermaid
flowchart TD
    CALL["_execute_tool_call(tool_call)"] --> PARSE{"Parse JSON<br/>arguments?"}
    PARSE -->|Fail| PARSE_ERR["Return error message:<br/>'Invalid JSON arguments'"]
    PARSE -->|OK| HANDLER{"Handler exists<br/>for tool name?"}
    HANDLER -->|No| HANDLER_ERR["Return error message:<br/>'Unknown tool: {name}'"]
    HANDLER -->|Yes| EXEC["Execute handler(args)"]
    EXEC --> TYPE{"Result type?"}
    TYPE -->|str| MSG_STR["Append tool message<br/>with string content"]
    TYPE -->|dict| MSG_DICT["Serialize to JSON<br/>+ append tool message"]
    TYPE -->|Exception| MSG_EXC["Format error<br/>+ append tool message"]
    MSG_STR --> APPEND["Append to<br/>conversation messages"]
    MSG_DICT --> APPEND
    MSG_EXC --> APPEND
    PARSE_ERR --> APPEND
    HANDLER_ERR --> APPEND
```

All paths converge on appending a `tool` role message. The Actor always sees a result — even if that result is an error description. This prevents the conversation from entering an inconsistent state where a `tool_call` has no corresponding `tool` response.

### Authentication Error Recovery

Two strategies address authentication failures, chosen based on the cost profile of the operation.

```mermaid
flowchart LR
    subgraph Cheap["Cheap Operations<br/>(LLM calls, metadata queries)"]
        C1["Retry with<br/>exponential backoff"]
        C2["Token refresh<br/>on each attempt"]
        C3["Up to N retries"]
        C1 --> C2 --> C3
    end

    subgraph Expensive["Expensive Operations<br/>(large code executions, batch jobs)"]
        E1["Single attempt"]
        E2["Pre-flight token<br/>freshness check"]
        E3["On failure: refresh<br/>token + retry once"]
        E1 --> E2 --> E3
    end
```

**Retry with backoff** — used for LLM completions and lightweight API calls where individual attempts are cheap but transient auth failures are common (e.g., token expiry during a long conversation):

```python
@retry(wait=wait_random_exponential(min=5, max=10), stop=stop_after_attempt(5))
def completion_with_retry(self, **kwargs):
    try:
        return self.client.chat.completions.create(**kwargs)
    except AuthenticationError:
        self.client.api_key = self.auth_manager.get_token(force_refresh=True)
        raise  # Trigger retry
```

**Single-attempt with pre-flight check** — used for expensive operations where blind retries waste significant resources:

```python
def execute_expensive_operation(self, payload):
    if self.auth_manager.is_token_expiring_soon(threshold_seconds=300):
        self.auth_manager.get_token(force_refresh=True)

    try:
        return self.executor.run(payload)
    except AuthenticationError:
        self.auth_manager.get_token(force_refresh=True)
        return self.executor.run(payload)  # Single retry
```

The choice between strategies is made at registration time — each handler knows its own cost profile and wraps itself accordingly. The dispatch layer does not impose a universal retry policy.

---

## Summary

| Concept | Key Mechanism |
|---------|---------------|
| **Tool Architecture** | Remote (sandboxed) vs. local (in-process) with shared auth/security/token infrastructure |
| **Registration** | Spec + handler pairs, extracted into parallel structures for LLM and dispatch |
| **Code Execution** | Multi-stage validation pipeline → isolated sandbox with timeout |
| **Static Analysis** | Three modes (expression, function, analysis) with AST-based safety |
| **Security** | Four independent defense layers; allowlist + blocklist + sandbox + constraints |
| **Custom Tools** | Domain preprocessors register additional spec + handler pairs at setup |
| **Result Size** | Execution-level and token-level caps; three-strategy pagination protocol |
| **Error Handling** | Unified dispatch with guaranteed tool-role response; auth strategy per cost profile |

> **Next**: [05 — Observability and Cost Tracking](./05_observability_and_cost_tracking.md) covers token accounting, logging, and production monitoring.
