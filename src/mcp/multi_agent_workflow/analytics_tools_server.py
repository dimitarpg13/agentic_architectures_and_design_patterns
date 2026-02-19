from __future__ import annotations

from mcp.server.fastmcp import FastMCP

mcp = FastMCP("analytics-tools")


@mcp.tool()
def calculate_growth_rate(start_value: float, end_value: float) -> str:
    if start_value == 0:
        return "Growth rate undefined because start_value is 0."
    growth = ((end_value - start_value) / start_value) * 100
    return f"Growth rate: {growth:.2f}%"


@mcp.tool()
def format_kpi_summary(metric_name: str, current_value: float, target_value: float) -> str:
    gap = target_value - current_value
    status = "on-track" if gap <= 0 else "behind-target"
    return (
        f"KPI: {metric_name}\n"
        f"Current: {current_value:.2f}\n"
        f"Target: {target_value:.2f}\n"
        f"Gap: {gap:.2f}\n"
        f"Status: {status}"
    )


if __name__ == "__main__":
    mcp.run(transport="stdio")


