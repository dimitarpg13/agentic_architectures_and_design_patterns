# MCP Multi-Agent Workflow Example

This example demonstrates:
- An orchestrator coordinating specialized agents
- Agent registration with MCP-based capability discovery (`list_tools`)
- Tool invocation via MCP (`call_tool`)
- LLM provider switch between OpenAI ChatGPT and Anthropic Claude via `.env`

## Files

- `main.py` - entrypoint
- `orchestrator.py` - orchestrator and agent registry
- `agents.py` - specialized agent logic (tool selection + tool use)
- `mcp_client.py` - MCP stdio client wrapper
- `research_tools_server.py` - MCP server with research tools
- `analytics_tools_server.py` - MCP server with analytics tools
- `config.py` / `llm.py` - env config and LLM provider abstraction

## Setup with uv

```bash
cd /<your_path_to_folder>/agentic_architectures_and_design_patterns/src/mcp/multi_agent_workflow
uv sync
```

This creates a local virtual environment (`.venv`) based on `pyproject.toml`.

## Environment

Create `.env` at the repository root:

`/<your_path_to_folder>/agentic_architectures_and_design_patterns/.env`

```env
LLM_PROVIDER=openai
OPENAI_API_KEY=your-openai-key
ANTHROPIC_API_KEY=your-anthropic-key
OPENAI_MODEL=gpt-4o-mini
ANTHROPIC_MODEL=claude-3-5-sonnet-latest
```

Set `LLM_PROVIDER=anthropic` to use Claude.

## Run with uv

From `src/mcp/multi_agent_workflow`:

```bash
uv run mcp-multi-agent
```

Or with a custom objective:

```bash
uv run mcp-multi-agent --objective "Design a short plan for launching an AI support assistant and include one KPI calculation."
```


