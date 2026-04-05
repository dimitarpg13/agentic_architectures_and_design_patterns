# TPC-H Domain Rules

> **Status**: Placeholder — populate with TPC-H-specific business rules before running the workflow.

## Date Handling

- "Last quarter" means the most recent completed calendar quarter relative to the current date.
- "Year-to-date" means from January 1st of the current year through the current date.
- Date columns use the format `YYYY-MM-DD`.

## Default Behaviors

- When the user does not specify a sort order, default to descending by the primary metric.
- When the user asks for "top N" without specifying N, default to 10.
- Currency amounts are stored in the database without currency symbols; do not apply formatting in SQL.

## Business Definitions

- **Revenue** = `l_extendedprice * (1 - l_discount)`
- **Discounted revenue** = `l_extendedprice * l_discount`
- **Total cost** = `ps_supplycost * ps_availqty`
- **Order priority**: urgency is ranked as 1-URGENT > 2-HIGH > 3-MEDIUM > 4-NOT SPECIFIED > 5-LOW.

## Query Constraints

- Always use `SELECT` or `WITH ... SELECT` — no mutations.
- Results should be limited to a reasonable number of rows (≤ 1000) unless the user explicitly requests all rows.
- Use the table aliases defined in the Data Dictionary for consistency.

## TPC-H Specific Notes

- The `LINEITEM` table is the fact table; most revenue and quantity metrics derive from it.
- `ORDERS` and `LINEITEM` are linked by `o_orderkey = l_orderkey`.
- `CUSTOMER` and `ORDERS` are linked by `c_custkey = o_custkey`.
- `NATION` and `REGION` provide geographic hierarchy: customer → nation → region.
- `PARTSUPP` links `PART` and `SUPPLIER` as a many-to-many relationship.
