# Actor-Critic Agent Design Pattern

> A comprehensive reference for building adversarial dual-agent systems following the Actor-Critic Reinforcement Learning paradigm, where an Actor (Generator) produces outputs and a Critic (Validator) evaluates them — creating productive tension that yields higher-quality results than either agent alone.

---

## Overview

This document set describes a **generic Actor-Critic adversarial architecture** for LLM-based agentic systems. The Actor generates tool-augmented outputs (code, analysis, text) from natural language requests, while the Critic independently validates those outputs for correctness, completeness, security, and compliance. The architecture draws on concepts from reinforcement learning, game theory, and causal inference.

A **code generation** use case serves as the running example throughout: an Actor that produces code from natural language specifications, and a Critic that validates correctness, security, and style. The pattern generalizes to any domain where generation and validation benefit from adversarial separation.

---

## Documents

| # | Document | Description |
|---|---|---|
| 1 | [Architecture Overview](01_actor_critic_architecture.md) | High-level system design, static class diagrams (Core Agent, Critic Validation, Tool Execution, Infrastructure), PipelineConfig bridge pattern, configuration registry, storage architecture, environment-aware design |
| 2 | [Agentic Workflow Deep-Dive](02_actor_critic_workflow.md) | End-to-end sequence diagrams, system prompt construction, LLM interaction cycle, tool call loop mechanics, recursive self-correction flow, two-layer prompt guardrailing, session checkpointing, cost tracking, headless vs interactive execution |
| 3 | [Critic Validation System](03_critic_validation_system.md) | Validation architecture, structured output schema (code quality, security, correctness, compliance checks), salvageable vs non-salvageable errors, retry decision logic, failure pattern taxonomy, logging/observability, Critic prompt engineering |
| 4 | [Tool Calling System](04_tool_calling_system.md) | Spec + handler registration pattern, remote code execution (cloud sandbox), local static analysis, security model (4-layer defense), custom domain tools, result size management, pagination protocol, error handling patterns |
| 5 | [Guardrail Design & Causal Analysis](05_guardrail_design_and_causal_analysis.md) | Confounder problem in causal inference, three confounding flavors (task-difficulty, physical-constraint, temporal-state), guardrail taxonomy and interaction graph, SCM construction for guardrail interactions, backdoor path identification, deconfounding strategies, guardrail-aware validation architecture |
| 6 | [Adversarial Dynamics & Convergence](06_adversarial_dynamics_and_convergence.md) | Why the setup is adversarial (opposing objectives, cross-model diversity, veto power), intra-episode refinement vs inter-episode co-evolution, adversarial spectrum classification, co-evolution mechanisms (prompt augmentation, adversarial memory, parameterized state), causal inference integration (interventional best responses, counterfactual credit assignment, Causal SHAP for equilibrium verification) |
| 7 | [Limitations & Enhancements](07_limitations_and_enhancements.md) | 10 identified limitations (pass rate, instruction-following, latency, sandbox isolation, provider coupling), 10 enhancement proposals with Mermaid diagrams (graduated validation, instruction distillation, async streaming, multi-agent routing, process sandbox, cross-session memory, model-agnostic backend, observability dashboard), prioritized Gantt roadmap |
| 8 | [Causal Nash Equilibrium Convergence](08_causal_nash_equilibrium_convergence.md) | Formal game-theoretic foundations: Nash Equilibrium in MARL (Nash-Q, satisficing paths, ATMGs), confounding problem with structural causal model, four causal mechanisms for convergence (interventional best responses, game decomposition via d-separation, counterfactual credit assignment, Causal SHAP verification), complete walkthrough comparing standard MARL oscillation vs causal convergence, implementation architecture with pseudocode |

---

## Design Principles

The Actor-Critic pattern is built on five core principles:

1. **Adversarial Separation** — The Actor and Critic use different model families with different reasoning patterns, training data, and failure modes. This ensemble diversity catches errors that self-review would miss.

2. **Structured Evaluation** — The Critic returns machine-parseable JSON, not free text. This enables programmatic decision-making (pass/salvage/reject) rather than subjective judgment.

3. **Graduated Correction** — Three response tiers (pass, salvageable fix, non-salvageable feedback) route corrections to the most efficient path. Text-level fixes stay with the Critic; fundamental errors go back to the Actor for new tool calls.

4. **Causal Awareness** — Guardrails form a causal system whose interactions determine whether the adversarial loop converges. Building SCMs of guardrail interactions prevents deadlocks caused by unobserved confounders.

5. **Fail-Safe Defaults** — After exhausting correction attempts, the system presents the best available response with a warning rather than failing silently. The user always gets an answer.

---

## Reading Order

For a complete understanding, read the documents in order (1 → 8). For specific topics:

- **"How does the Actor-Critic loop work?"** — Start with [02](02_actor_critic_workflow.md), then [03](03_critic_validation_system.md)
- **"How are tools designed?"** — [04](04_tool_calling_system.md)
- **"Why do correction loops sometimes fail?"** — [05](05_guardrail_design_and_causal_analysis.md)
- **"Is this truly adversarial?"** — [06](06_adversarial_dynamics_and_convergence.md)
- **"What should I improve first?"** — [07](07_limitations_and_enhancements.md)
- **"Why does causal inference enable Nash Equilibrium convergence?"** — [08](08_causal_nash_equilibrium_convergence.md)

---

## Diagram Reference

All diagrams use [Mermaid](https://mermaid.js.org/) syntax and render natively in GitHub, GitLab, and most Markdown editors. Diagram types used:

- **Class diagrams** — Static structure of agents, validators, tools, and infrastructure
- **Sequence diagrams** — Dynamic interaction flows (end-to-end workflow, self-correction, validation)
- **Flowcharts** — Decision logic, security model, prompt construction, guardrail analysis
- **State diagrams** — Tool call loop, oscillation mechanisms
- **Causal DAGs** — Confounding analysis, backdoor paths, deconfounding
- **Gantt charts** — Enhancement roadmap

---

## Prompt to Generate Specific Workflow

**SQL Generation**

In folder `actor-critic/sql_generation/` could you create me an example dual agent workflow for sql query generation utilizing complex analytical expressions based on user supplied information in a query? The critic should be doing error correction of the generated analytical expressions following the principles and workflows discussed in the markdowns in folder `actor-critic`. The whole workflow should be deployable on Databricks in Nike Sole account under my account. The two agents should use different LLM models - say the generator should use Antrhopic's Claude while the critic should use Google Gemini inside Vertex AI ecosystem. The workflow can contain multiple modules spread into subfolders of `actor-critic/sql_generation` and should have at least one demo notebook in the folder `actor-critic/sql_generation` which I should be able to run from within Vertex AI (i guess using Vertex AI Workbench). I should be able supply keys for the LLMs used in the agents through `.env` file, hardcoded in a demo notebook showing how the workflow works, or using Google CLoud Secret Manager. In terms of agentic framework to implement this I am flexible - you could use LangGraph as I am familiar with it. The user input to the demo notebook should be user request for creating a SQL query which meets certain specified objectives and format specifications in a set of predefined markdown documents which will be available to the workflow at the time of the user query. For details of the user input please read the markdown in `actor-critic/user_input.md` which explains what the user input should be used in general. Try to reuse the subfolder structure explained in the section _"Step 2: System Prompt Construction - Grounding the LLM"_. Assume that such files are created in the specified subfolders of actor-critic/sql_generation. `DOMAIN_RULES.md` and `DATA_DICTIONARY.md` - we will create those later based on the familiar RDBMS dataset **TPC-H**. So  the user will ask questions about how to obtain particular information from the TPC-H tables and the workflow should generate SQL query delivering the data. If we are using LangGraph as an orchestration framework for this workflow then the tracing and observability can be provided by LangSmith as it is already integrated with Vertex AI ecosystem. Log messages indicating the execution of different stages of the workflow by using LangSmith tracing and osbervability functionality. Create a `README.md` inside `actor-critic/sql_generation` which details how to run this workflow inside GCP and trace the logged messages thereby identifying the sequential steps of its execution. 

*Generated: April 2026*
