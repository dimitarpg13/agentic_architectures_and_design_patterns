# LangGraph + MCP Multi-Agent Workflow (Notebook)

This example includes:
- LangGraph orchestrator (`orchestrator_langgraph.py`)
- Specialized agents (`agents.py`)
- MCP tool servers (`research_tools_server.py`, `analytics_tools_server.py`)
- Main notebook entrypoint (`langgraph_mcp_multi_agent_workflow.ipynb`)

## Install dependencies

```bash
pip install langgraph mcp openai anthropic python-dotenv jupyter
```

## API keys

You can provide keys either way:

1) In repo-root `.env`:

```env
LLM_PROVIDER=openai
OPENAI_API_KEY=your-openai-key
ANTHROPIC_API_KEY=your-anthropic-key
OPENAI_MODEL=gpt-4o-mini
ANTHROPIC_MODEL=claude-3-5-sonnet-latest
```

2) Directly in the notebook configuration cell.

## Run

Open and run:

`notebooks/multi_agent_workflow/mcp/langgraph_mcp_multi_agent_workflow.ipynb`


