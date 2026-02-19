# Multi-Agent Collaboration and Load Balancing via Reinforcement Learning

## Overview

This folder contains a series of notebooks and supporting documents that explore how **Reinforcement Learning (RL)** can coordinate a pool of specialized LLM-backed agents — routing tasks to the best agent, balancing workload across agents, and (in future work) enabling inter-agent collaboration.

The experiments progressed through two major phases:

1. **Simulated workflow with PPO** — a fully simulated multi-agent Gymnasium environment using Proximal Policy Optimization and a collaboration matrix.
2. **Real LLM calls with Q-Learning** — a practical redesign that replaces the simulation with actual LLM API calls (OpenAI / Anthropic) orchestrated by LangGraph, using tabular Q-Learning for task routing and load balancing.

---

## Phase 1: Simulated Workflow with PPO

### Notebooks

| Notebook | Description |
|----------|-------------|
| [`simulated_agentic_rl_workflow.ipynb`](simulated_agentic_rl_workflow.ipynb) | Core PPO-based multi-agent RL system with a custom Gymnasium environment, 4 agents (researcher, analyst, coder, validator), and fully simulated task processing. |
| [`simulated_agentic_rl_workflow_with_google_colab.ipynb`](simulated_agentic_rl_workflow_with_google_colab.ipynb) | Google Colab-ready variant with MLflow experiment tracking. |
| [`simulated_agentic_rl_workflow_worker_pool_semaphore.ipynb`](simulated_agentic_rl_workflow_worker_pool_semaphore.ipynb) | Extended variant with worker-pool concurrency control via semaphores. |

### Design

The simulated workflow used **PPO** (from `stable-baselines3`) with:

- A **67-dimensional continuous observation** vector that included task features, agent availability, performance history, and a flattened **4×4 collaboration matrix**.
- A **12-dimensional continuous action space** representing task assignment probabilities, resource allocation, and collaboration request signals across 4 agents.
- A **collaboration matrix** (`np.ones((4, 4))`) that was updated during training when pairs of agents signalled willingness to collaborate (action dimension > 0.7). Successful collaboration earned a bonus reward, encouraging the policy to learn emergent multi-agent cooperation patterns.

The goal was to simultaneously optimise for:
- **Task routing quality** — assigning each task to the most suitable agent.
- **Load balancing** — distributing work evenly across agents.
- **Agent collaboration** — learning which agent pairs benefit from joint work.

### Supporting Documents

| Document | Description |
|----------|-------------|
| [`simulated_agentic_rl_workflow.md`](simulated_agentic_rl_workflow.md) | UML class diagrams, sequence diagrams, and workflow flowcharts for the simulated notebook. |
| [`simulated_agentic_rl_workflow_training_loop.md`](simulated_agentic_rl_workflow_training_loop.md) | Detailed analysis of the PPO training loop with Mermaid diagrams. |
| [`pytorch_policy_net_diagrams.md`](pytorch_policy_net_diagrams.md) | Architecture diagrams for the PyTorch policy network used by PPO. |
| [`task_driven_multi-agent_colab_workflow.md`](task_driven_multi-agent_colab_workflow.md) | Task-driven workflow analysis with UML and flowcharts. |

---

## Phase 2: Real LLM Calls with Q-Learning

### Why PPO Was Replaced

When the workflow was rewritten to use **real LLM API calls** instead of simulation, PPO became fundamentally unsuitable:

| Issue | PPO (Simulated) | Q-Learning (Real LLM Calls) |
|-------|-----------------|------------------------------|
| **Sample efficiency** | Requires 100k–1M+ environment steps | Learns useful policies in 30–50 iterations |
| **Cost per step** | Free (simulated) | ~$0.003 per step (2 LLM calls) |
| **Action space** | 12-dim continuous (neural network) | 8 discrete agent-provider pairs (lookup table) |
| **State space** | 67-dim continuous observation | Discrete task type (4 values) or task type × load state (20 values) |
| **Dependencies** | PyTorch, stable-baselines3, Gymnasium | None beyond NumPy |
| **Training time** | Minutes (simulated) but would be days/weeks with real API calls | Minutes with real API calls |
| **Interpretability** | Opaque neural network | Transparent Q-table, directly inspectable |

> **For a comprehensive analysis** of why PPO was replaced, including code excerpts, architectural comparisons, Mermaid diagrams, and the discussion of online RL interaction costs, see [`problems_with_ppo_algorithm.md`](problems_with_ppo_algorithm.md).

### What Happened to the Collaboration Matrix

The collaboration matrix — central to the PPO simulation — was **dropped** in the Q-Learning redesign because:

- The real LLM workflow makes **independent single-agent calls** per task; there is no shared multi-agent state within a single step.
- The Q-Learning state is intentionally minimal (`task_type` or `task_type × load_state`), and a 4×4 matrix cannot be directly represented in a tabular Q-table without exponential state explosion.
- The collaboration signal was an emergent property of the simulated environment's continuous action space, which has no analogue in the discrete action design.

See **Section 16** of [`problems_with_ppo_algorithm.md`](problems_with_ppo_algorithm.md) for a detailed discussion and three proposed approaches for reintroducing collaboration in future designs.

### Notebooks

| Notebook | Description |
|----------|-------------|
| [`agentic_rl_workflow_LLM_calls.ipynb`](agentic_rl_workflow_LLM_calls.ipynb) | Base Q-Learning coordinator with real LLM calls (OpenAI + Anthropic) via LangChain/LangGraph. State = task type (4 states), 8 agent-provider actions, 32 Q-values. LLM-as-Judge reward model. |
| [`agentic_rl_workflow_LLM_calls_lb.ipynb`](agentic_rl_workflow_LLM_calls_lb.ipynb) | **Load-balanced** variant. State = task type × load state (20 states), 160 Q-values. Adds a `LoadTracker` with sliding-window utilization, hot-agent detection, and a load penalty subtracted from the quality reward. The coordinator learns conditional policies: e.g., "when the coder is overloaded, route coding tasks to the analyst instead." |

### How Load Balancing Works

The load-balanced notebook (`agentic_rl_workflow_LLM_calls_lb.ipynb`) augments the base Q-Learning design with:

1. **LoadTracker** — maintains a sliding window (size 8) of recent agent assignments and computes per-agent utilization.
2. **Load state** — a discrete label added to the Q-table key:
   - `"balanced"` — all agents within 1.5× fair share.
   - `"<agent>_hot"` — that agent exceeds the threshold.
3. **Load penalty** — subtracted from the quality reward when assigning to an over-utilized agent. Grows linearly with excess utilization, capped at 0.20.
4. **Conditional policies** — the coordinator learns different routing strategies depending on which (if any) agent is overloaded.

### Supporting Documents

| Document | Description |
|----------|-------------|
| [`agentic_rl_workflow_LLM_calls.md`](agentic_rl_workflow_LLM_calls.md) | UML class diagrams, sequence diagrams, and workflow flowcharts for the real LLM calls notebook. |
| [`problems_with_ppo_algorithm.md`](problems_with_ppo_algorithm.md) | Comprehensive analysis of why PPO was replaced with Q-Learning, including sample efficiency, action/state space design, dependency bloat, online RL interaction costs, and the fate of the collaboration matrix. |

---

## Future Directions

Several extensions are under consideration for future exploration:

### 1. Agent Chains in RL Scenarios

Current designs route each task to a **single agent**. A natural extension is to let the RL coordinator learn **multi-step agent chains** — e.g., route a complex task through `researcher → analyst → validator`, where each agent refines the previous output. This would require:
- Extending the action space to include chain definitions (ordered sequences of agents).
- A reward signal that evaluates the final chain output rather than individual agent outputs.
- Possible integration with LangGraph's branching and conditional edges to orchestrate multi-hop workflows.

### 2. Reintroducing the Collaboration Matrix

The collaboration matrix from the PPO simulation captured valuable inter-agent synergy information. Three approaches to reintroduce it in a Q-Learning or more advanced RL framework are discussed in detail in [`problems_with_ppo_algorithm.md` § Section 16](problems_with_ppo_algorithm.md):

- **State augmentation** — encode a discretised collaboration signal (e.g., "which agent pair last collaborated?") as part of the Q-table state.
- **Pairwise collaboration Q-tables** — maintain separate Q-tables for agent pairs, learning which pairs produce better joint outcomes.
- **LLM-native shared context** — instead of a numeric matrix, use prompt chaining to pass one agent's output as context to another, letting the LLMs themselves "collaborate" through shared information.

### 3. Deeper Load Balancing

- Incorporate **provider-level load** (e.g., OpenAI vs Anthropic rate limits) into the load state.
- Add **latency-aware routing** — prefer agents/providers with lower current latency.
- Explore **multi-objective RL** to explicitly balance quality, cost, latency, and load fairness as separate reward dimensions.

---

## File Index

| File | Type | Description |
|------|------|-------------|
| [`simulated_agentic_rl_workflow.ipynb`](simulated_agentic_rl_workflow.ipynb) | Notebook | PPO-based simulated multi-agent RL system |
| [`simulated_agentic_rl_workflow_with_google_colab.ipynb`](simulated_agentic_rl_workflow_with_google_colab.ipynb) | Notebook | Google Colab variant with MLflow tracking |
| [`simulated_agentic_rl_workflow_worker_pool_semaphore.ipynb`](simulated_agentic_rl_workflow_worker_pool_semaphore.ipynb) | Notebook | Worker-pool variant with semaphore concurrency |
| [`agentic_rl_workflow_LLM_calls.ipynb`](agentic_rl_workflow_LLM_calls.ipynb) | Notebook | Q-Learning with real LLM calls (base) |
| [`agentic_rl_workflow_LLM_calls_lb.ipynb`](agentic_rl_workflow_LLM_calls_lb.ipynb) | Notebook | Q-Learning with real LLM calls + load balancing |
| [`simulated_agentic_rl_workflow.md`](simulated_agentic_rl_workflow.md) | Docs | UML diagrams for simulated workflow |
| [`simulated_agentic_rl_workflow_training_loop.md`](simulated_agentic_rl_workflow_training_loop.md) | Docs | PPO training loop analysis |
| [`pytorch_policy_net_diagrams.md`](pytorch_policy_net_diagrams.md) | Docs | Policy network architecture diagrams |
| [`task_driven_multi-agent_colab_workflow.md`](task_driven_multi-agent_colab_workflow.md) | Docs | Task-driven workflow analysis |
| [`agentic_rl_workflow_LLM_calls.md`](agentic_rl_workflow_LLM_calls.md) | Docs | UML diagrams for real LLM calls workflow |
| [`problems_with_ppo_algorithm.md`](problems_with_ppo_algorithm.md) | Docs | PPO to Q-Learning migration analysis |
