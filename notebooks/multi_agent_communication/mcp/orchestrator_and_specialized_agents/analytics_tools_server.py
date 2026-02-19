from __future__ import annotations

from mcp.server.fastmcp import FastMCP

mcp = FastMCP("analytics-tools")


@mcp.tool()
def calculate_growth_rate(start_value: float, end_value: float) -> str:
    if start_value == 0:
        return "Cannot calculate growth rate because start_value is 0."
    growth = ((end_value - start_value) / start_value) * 100
    return f"Growth rate: {growth:.2f}%"


@mcp.tool()
def summarize_kpi(metric_name: str, actual: float, target: float) -> str:
    delta = actual - target
    status = "on-track" if actual >= target else "below-target"
    return (
        f"KPI={metric_name}; actual={actual:.2f}; target={target:.2f}; "
        f"delta={delta:.2f}; status={status}"
    )


if __name__ == "__main__":
    mcp.run(transport="stdio")


