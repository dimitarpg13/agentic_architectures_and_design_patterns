from __future__ import annotations

from mcp.server.fastmcp import FastMCP

mcp = FastMCP("research-tools")

_KNOWLEDGE = {
    "a2a": (
        "A2A provides a protocol for structured communication between agents. "
        "It helps orchestrators and specialists exchange messages with clear boundaries."
    ),
    "adk": (
        "Google ADK provides agent primitives and orchestration-friendly abstractions "
        "for building multi-agent systems."
    ),
    "mcp": (
        "MCP standardizes tool discovery and invocation. Agents inspect available tools "
        "and invoke them over a consistent protocol."
    ),
}


@mcp.tool()
def search_research_notes(query: str) -> str:
    q = query.lower()
    matches = [text for key, text in _KNOWLEDGE.items() if key in q]
    if matches:
        return " ".join(matches)
    return " ".join(_KNOWLEDGE.values())


@mcp.tool()
def create_brief_outline(topic: str) -> str:
    return (
        f"Brief outline for '{topic}':\n"
        "1) Objective\n"
        "2) Orchestrator and specialized roles\n"
        "3) ADK + A2A communication design\n"
        "4) MCP tool lifecycle\n"
        "5) Metrics and next actions"
    )


if __name__ == "__main__":
    mcp.run(transport="stdio")


