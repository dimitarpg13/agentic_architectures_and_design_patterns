# ADK + A2A + MCP Orchestrator Workflow

This folder contains a notebook-first multi-agent example:
- **Orchestrator agent** coordinates specialized agents
- **Specialized agents** use Google ADK agent objects
- **A2A communication** is modeled with an A2A channel between agents
- **Tools are discovered and called over MCP** (`list_tools`, `call_tool`)
- **LLM provider** can be OpenAI ChatGPT or Anthropic Claude

## Files

- `a2a_adk_mcp_orchestrator_workflow.ipynb` (main entrypoint)
- `workflow.py` (runner)
- `orchestrator.py` (orchestrator + registry)
- `specialized_agents.py` (ADK-specialized agents)
- `a2a_channel.py` (A2A message exchange abstraction)
- `mcp_client.py` (MCP protocol client)
- `research_tools_server.py` / `analytics_tools_server.py` (MCP tool servers)
- `settings.py` / `llm_clients.py` (config and LLM provider adapters)

## Install

```bash
pip install mcp python-dotenv openai anthropic google-adk a2a-sdk jupyter
```

## API keys and models

Option 1: put keys in repo-root `.env`:

```env
LLM_PROVIDER=openai
OPENAI_API_KEY=your-openai-key
ANTHROPIC_API_KEY=your-anthropic-key
OPENAI_MODEL=gpt-4o-mini
ANTHROPIC_MODEL=claude-3-5-sonnet-latest
```

Option 2: set keys directly in notebook config cell.

## Run

Open:
`notebooks/multi_agent_communication/a2a/orchestrator_and_specialized_agents/a2a_adk_mcp_orchestrator_workflow.ipynb`


