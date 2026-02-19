from __future__ import annotations

from mcp.server.fastmcp import FastMCP

mcp = FastMCP("research-tools")

_KB = {
    "langgraph": (
        "LangGraph models agentic workflows as stateful graphs where nodes represent "
        "reasoning/actions and edges drive control flow."
    ),
    "mcp": (
        "MCP allows standardized tool discovery and invocation. Agents call list_tools "
        "for capability discovery and call_tool for execution."
    ),
    "orchestrator": (
        "An orchestrator decomposes objectives, delegates to specialists, and synthesizes "
        "final output from intermediate results."
    ),
}


@mcp.tool()
def search_knowledge_base(query: str) -> str:
    q = query.lower()
    hits = [text for keyword, text in _KB.items() if keyword in q]
    if hits:
        return " ".join(hits)
    return " ".join(_KB.values())


@mcp.tool()
def create_research_outline(topic: str) -> str:
    return (
        f"Research outline for '{topic}':\n"
        "1) Objective and constraints\n"
        "2) Agent roles\n"
        "3) MCP registration + capability discovery\n"
        "4) Tool calling pattern\n"
        "5) Risks and mitigations"
    )


if __name__ == "__main__":
    mcp.run(transport="stdio")


