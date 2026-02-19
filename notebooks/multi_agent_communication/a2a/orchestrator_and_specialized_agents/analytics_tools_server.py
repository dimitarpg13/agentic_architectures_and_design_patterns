from __future__ import annotations

from mcp.server.fastmcp import FastMCP

mcp = FastMCP("analytics-tools")


@mcp.tool()
def compute_growth(start_value: float, end_value: float) -> str:
    if start_value == 0:
        return "Cannot compute growth because start_value is 0."
    growth = ((end_value - start_value) / start_value) * 100
    return f"Growth={growth:.2f}%"


@mcp.tool()
def kpi_status(metric: str, actual: float, target: float) -> str:
    delta = actual - target
    status = "on-track" if actual >= target else "off-track"
    return (
        f"KPI {metric}: actual={actual:.2f}, target={target:.2f}, "
        f"delta={delta:.2f}, status={status}"
    )


if __name__ == "__main__":
    mcp.run(transport="stdio")


