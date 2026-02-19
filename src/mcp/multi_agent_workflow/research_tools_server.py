from __future__ import annotations

from mcp.server.fastmcp import FastMCP

mcp = FastMCP("research-tools")

_CORPUS = {
    "orchestrator": (
        "An orchestrator coordinates specialists, decomposes objectives into tasks, "
        "tracks intermediate state, and synthesizes final output."
    ),
    "mcp": (
        "Model Context Protocol standardizes how clients discover and invoke tools "
        "across servers. Agents discover capabilities with list_tools and execute with call_tool."
    ),
    "multi-agent": (
        "Specialized agents reduce context overload and improve reliability by focusing "
        "on narrower tools and responsibilities."
    ),
}


@mcp.tool()
def search_knowledge_base(query: str) -> str:
    q = query.lower()
    matches = [value for key, value in _CORPUS.items() if key in q]
    if not matches:
        return (
            "No exact keyword match found. Relevant defaults: "
            + " ".join(_CORPUS.values())
        )
    return " ".join(matches)


@mcp.tool()
def outline_report(topic: str) -> str:
    return (
        f"Report outline for '{topic}':\n"
        "1) Problem framing\n"
        "2) Agent roles and orchestration strategy\n"
        "3) MCP registration and capability discovery flow\n"
        "4) Tool invocation lifecycle\n"
        "5) Risks and mitigations"
    )


if __name__ == "__main__":
    mcp.run(transport="stdio")


