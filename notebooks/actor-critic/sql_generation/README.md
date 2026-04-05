# Actor-Critic SQL Generation Workflow

A dual-agent system for generating validated SQL queries from natural language, deployed on **Google Cloud Vertex AI**.

| Component | Technology |
|-----------|-----------|
| **Actor** (Generator) | Anthropic Claude via Vertex AI Model Garden |
| **Critic** (Validator) | Google Gemini via Vertex AI |
| **Orchestration** | LangGraph |
| **Tracing & Observability** | LangSmith |
| **Runtime** | Vertex AI Workbench (JupyterLab) |
| **Dataset** | TPC-H benchmark |

---

## Architecture

```
User Question
      │
      ▼
┌─────────────────┐
│ assemble_context │  Reads DOMAIN_RULES.md, DATA_DICTIONARY.md,
│                  │  actor/critic prompts → builds system messages
└────────┬────────┘
         ▼
┌─────────────────┐
│  generate_sql   │  Actor (Claude) generates SQL + explanation
│                 │  from user question + assembled context
└────────┬────────┘
         ▼
┌─────────────────┐
│  validate_sql   │  Critic (Gemini) evaluates SQL against
│                 │  schema, logic, domain rules, best practices
└────────┬────────┘
         │
    ┌────┴────────────────────┐
    │    route_verdict         │
    ├──────────┬──────────────┤
    │          │              │
  PASS    SALVAGEABLE   NON_SALVAGEABLE
    │          │              │
    ▼          ▼              │
 finalize  apply_correction   │
               │              │
               ▼              │
          validate_sql ◄──────┘
          (re-validate)   (Actor re-generates
                           with feedback)
```

### Verdict Categories

| Verdict | Action | Example |
|---------|--------|---------|
| **pass** | Deliver SQL as-is | All checks satisfied |
| **salvageable** | Critic fixes the SQL, then re-validates its own correction | Wrong column alias, missing GROUP BY column |
| **non_salvageable** | Feedback sent to Actor for full re-generation | Wrong tables, hallucinated columns, fundamentally wrong approach |
| **max_attempts** | Deliver best available SQL with a warning | Exhausted retry budget |

---

## Directory Structure

```
sql_generation/
├── README.md                          ← You are here
├── requirements.txt                   ← Python dependencies
├── .env.example                       ← Template environment variables
├── demo_sql_generation.ipynb          ← Runnable demo notebook
│
├── prompts/
│   ├── actor_system_prompt.md         ← Actor behavioral guidelines
│   └── critic_system_prompt.md        ← Critic validation rubric
│
├── usecases/
│   └── tpch/
│       ├── DOMAIN_RULES.md            ← TPC-H business rules
│       └── DATA_DICTIONARY.md         ← TPC-H table schemas
│
├── utils/
│   ├── __init__.py
│   └── prompt_builder.py             ← System prompt assembly (PromptBuilder)
│
└── workflow/
    ├── __init__.py
    ├── config.py                      ← Configuration & secret management
    ├── state.py                       ← LangGraph state schema
    ├── graph.py                       ← LangGraph graph definition
    └── nodes/
        ├── __init__.py
        ├── actor.py                   ← SQL generation (Claude)
        ├── critic.py                  ← SQL validation (Gemini)
        └── router.py                  ← Routing, correction, finalization

tests/
├── __init__.py
├── conftest.py                        ← Shared fixtures and mock LLMs
├── test_prompt_builder.py             ← PromptBuilder unit tests
├── test_actor.py                      ← ActorNode unit tests
├── test_critic.py                     ← CriticNode unit tests
├── test_router.py                     ← Routing / correction / finalization tests
├── test_config.py                     ← WorkflowConfig factory tests
└── test_graph_integration.py          ← End-to-end graph tests (mocked LLMs)
```

---

## Running Tests

All tests use **mocked LLMs** — no GCP credentials, API keys, or network access required. They run identically on a local machine and on Vertex AI Workbench.

### Quick run

```bash
cd actor-critic/sql_generation
pip install -r requirements.txt
pytest -v
```

### Run a specific test module

```bash
pytest tests/test_router.py -v
pytest tests/test_graph_integration.py -v
```

### What the tests cover

| Module | Tests | What is verified |
|--------|-------|-----------------|
| `test_prompt_builder` | 10 | File reading, section assembly, metadata injection, missing-file fallback |
| `test_actor` | 13 | SQL extraction (fenced / unfenced), explanation parsing, prompt construction with/without feedback, state updates |
| `test_critic` | 11 | JSON parsing (clean / fenced / embedded / broken), verdict extraction, prompt structure |
| `test_router` | 12 | All routing branches (pass, salvageable, non_salvageable, max_attempts), correction state update, finalize status |
| `test_config` | 6 | Factory methods (`from_values`, `from_env`), LangSmith env propagation, base_dir handling |
| `test_graph_integration` | 7 | Full pipeline: pass-on-first-try, salvageable→pass, non_salvageable→regenerate→pass, max-attempts exhaustion, salvageable loop exhaustion |

---

## Setup on GCP

### 1. Create a Vertex AI Workbench Instance

```bash
gcloud workbench instances create sql-gen-notebook \
    --location=us-central1-a \
    --machine-type=e2-standard-4
```

Open the JupyterLab URL from the Cloud Console.

### 2. Clone the Repository

In a JupyterLab terminal:

```bash
git clone <your-repo-url>
cd agentic_architectures_and_design_patterns/notebooks/actor-critic/sql_generation
```

### 3. Enable Required APIs

```bash
gcloud services enable aiplatform.googleapis.com
gcloud services enable secretmanager.googleapis.com
```

### 4. Grant Model Access

Claude is available through **Vertex AI Model Garden**. Enable access in the Cloud Console:

1. Navigate to **Vertex AI → Model Garden**
2. Search for "Claude" and click **Enable**
3. Note the supported regions (e.g., `us-east5`, `europe-west1`)

Gemini is available by default in all Vertex AI regions.

### 5. Configure Secrets

Choose one of three methods:

#### Method A — `.env` file

```bash
cp .env.example .env
# Edit .env with your values
```

#### Method B — Hardcoded in notebook

Edit the `WorkflowConfig.from_values()` cell directly in the notebook.

#### Method C — Google Cloud Secret Manager

Create secrets with the `sql-gen-` prefix:

```bash
PROJECT_ID="your-project-id"

echo -n "us-central1"              | gcloud secrets create sql-gen-gcp-location-gemini --data-file=- --project=$PROJECT_ID
echo -n "us-east5"                 | gcloud secrets create sql-gen-gcp-location-claude --data-file=- --project=$PROJECT_ID
echo -n "claude-sonnet-4-20250514" | gcloud secrets create sql-gen-actor-model --data-file=- --project=$PROJECT_ID
echo -n "gemini-2.5-flash"         | gcloud secrets create sql-gen-critic-model --data-file=- --project=$PROJECT_ID
echo -n "3"                        | gcloud secrets create sql-gen-max-attempts --data-file=- --project=$PROJECT_ID
echo -n "tpch"                     | gcloud secrets create sql-gen-use-case --data-file=- --project=$PROJECT_ID
echo -n "lsv2_your_key_here"       | gcloud secrets create sql-gen-langsmith-api-key --data-file=- --project=$PROJECT_ID
echo -n "sql-generation-actor-critic" | gcloud secrets create sql-gen-langsmith-project --data-file=- --project=$PROJECT_ID
```

Grant the Workbench service account access:

```bash
SA="<workbench-service-account>@<project>.iam.gserviceaccount.com"

gcloud secrets add-iam-policy-binding sql-gen-langsmith-api-key \
    --member="serviceAccount:$SA" \
    --role="roles/secretmanager.secretAccessor" \
    --project=$PROJECT_ID

# Repeat for each secret, or use a wildcard IAM binding
```

### 6. Run the Demo Notebook

Open `demo_sql_generation.ipynb` in Workbench and execute the cells sequentially.

---

## Tracing with LangSmith

LangGraph has **built-in LangSmith integration**. When the environment variables are set, every graph invocation is traced automatically — no code changes required.

### What Gets Traced

Each `graph.invoke()` call creates a hierarchical trace:

```
sql-generation-actor-critic (project)
└── RunnableSequence (graph invocation)
    ├── assemble_context          ← prompt assembly, no LLM call
    ├── generate_sql              ← Claude invocation
    │   └── ChatAnthropicVertex   ← raw LLM request/response
    ├── validate_sql              ← Gemini invocation
    │   └── ChatVertexAI          ← raw LLM request/response
    ├── apply_correction          ← (only if salvageable)
    ├── validate_sql              ← (re-validation after correction)
    │   └── ChatVertexAI
    └── finalize                  ← terminal state
```

### Viewing Traces

1. Open [smith.langchain.com](https://smith.langchain.com)
2. Select the project (default: `sql-generation-actor-critic`)
3. Each row is one `graph.invoke()` call — click to expand
4. Inside each trace you can see:
   - **Input/output** for every node
   - **Full LLM prompts and responses** (system message, user message, model output)
   - **Latency** per node and per LLM call
   - **Token usage** for cost tracking
   - **Error details** if a node fails

### Identifying the Sequential Steps

The LangSmith trace view shows nodes in execution order. To identify the correction loop:

| Trace Pattern | Meaning |
|---------------|---------|
| `assemble_context → generate_sql → validate_sql → finalize` | First-pass acceptance (PASS) |
| `... → validate_sql → apply_correction → validate_sql → finalize` | Critic-corrected (SALVAGEABLE) |
| `... → validate_sql → generate_sql → validate_sql → finalize` | Actor re-generated (NON_SALVAGEABLE) |

### Console Logging

The workflow also emits structured Python log messages at each stage. In the notebook, these appear in the cell output. Example:

```
14:23:01 │ workflow.graph                      │ INFO     │ Assembling context for use-case 'tpch'
14:23:01 │ workflow.nodes.actor                │ INFO     │ Actor generating SQL — attempt 1
14:23:04 │ workflow.nodes.actor                │ INFO     │ Actor produced 342-char SQL
14:23:04 │ workflow.nodes.critic               │ INFO     │ Critic validating SQL — attempt 1
14:23:07 │ workflow.nodes.critic               │ INFO     │ Critic verdict: salvageable (1 issues)
14:23:07 │ workflow.nodes.router               │ INFO     │ Routing → apply_correction (SALVAGEABLE, attempt 1/3)
14:23:07 │ workflow.nodes.router               │ INFO     │ Applying Critic correction (358 chars)
14:23:07 │ workflow.nodes.critic               │ INFO     │ Critic validating SQL — attempt 1
14:23:09 │ workflow.nodes.critic               │ INFO     │ Critic verdict: pass (0 issues)
14:23:09 │ workflow.nodes.router               │ INFO     │ Routing → finalize (PASS)
14:23:09 │ workflow.nodes.router               │ INFO     │ Finalized: ACCEPTED after 1 attempt(s)
```

---

## Extending the Workflow

### Adding a New Use Case

1. Create a new directory under `usecases/`:
   ```
   usecases/my_dataset/
   ├── DOMAIN_RULES.md
   └── DATA_DICTIONARY.md
   ```
2. Set `use_case="my_dataset"` in the config
3. The workflow will automatically read the new grounding documents

### Adjusting Models

Update the model names in your config:

```python
config = WorkflowConfig.from_values(
    gcp_project_id="...",
    actor_model="claude-sonnet-4-20250514",  # or any Claude model on Model Garden
    critic_model="gemini-2.5-pro",           # or gemini-2.5-flash for lower cost
)
```

### Modifying the Validation Rubric

Edit `prompts/critic_system_prompt.md` to adjust what the Critic checks for. The structured JSON output format ensures the routing logic continues to work regardless of rubric changes.

### Adding SQL Execution

The current workflow generates and validates SQL but does not execute it against a database. To add execution:

1. Create a new node (e.g., `execute_sql`) in `workflow/nodes/`
2. Add it to the graph after `finalize`
3. Connect to your database (e.g., BigQuery, Cloud SQL, AlloyDB) using the appropriate Python client

---

## Troubleshooting

| Symptom | Likely Cause | Fix |
|---------|-------------|-----|
| `PermissionDenied` on Claude call | Model Garden access not enabled | Enable Claude in Vertex AI Model Garden console |
| `PermissionDenied` on Gemini call | Vertex AI API not enabled | Run `gcloud services enable aiplatform.googleapis.com` |
| `InvalidArgument: location` | Wrong region for the model | Claude: use `us-east5` or `europe-west1`; Gemini: use `us-central1` |
| Empty `correction_history` | First-pass acceptance | The Actor's SQL passed validation on the first try |
| `status: best_effort` | Exhausted `max_attempts` | Increase `max_attempts` or refine prompts |
| LangSmith traces not appearing | API key not set or invalid | Verify `LANGSMITH_API_KEY` is correct and `LANGSMITH_TRACING=true` |
