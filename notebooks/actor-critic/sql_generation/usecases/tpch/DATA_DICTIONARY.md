# TPC-H Data Dictionary

> **Status**: Placeholder — expand with full column-level descriptions, data types, and example values before running the workflow.

## Tables Overview

| Table | Alias | Description | Approximate Row Count (SF=1) |
|-------|-------|-------------|------------------------------|
| `region` | `r` | Geographic regions (5 rows) | 5 |
| `nation` | `n` | Nations within regions | 25 |
| `supplier` | `s` | Part suppliers | 10,000 |
| `customer` | `c` | Customers who place orders | 150,000 |
| `part` | `p` | Parts catalog | 200,000 |
| `partsupp` | `ps` | Part-supplier relationships (costs, availability) | 800,000 |
| `orders` | `o` | Customer orders | 1,500,000 |
| `lineitem` | `l` | Order line items (fact table) | 6,000,000 |

## Table Schemas

### region
| Column | Type | Description |
|--------|------|-------------|
| `r_regionkey` | INTEGER | Primary key |
| `r_name` | VARCHAR(25) | Region name (AFRICA, AMERICA, ASIA, EUROPE, MIDDLE EAST) |
| `r_comment` | VARCHAR(152) | Free-text comment |

### nation
| Column | Type | Description |
|--------|------|-------------|
| `n_nationkey` | INTEGER | Primary key |
| `n_name` | VARCHAR(25) | Nation name |
| `n_regionkey` | INTEGER | FK → region.r_regionkey |
| `n_comment` | VARCHAR(152) | Free-text comment |

### supplier
| Column | Type | Description |
|--------|------|-------------|
| `s_suppkey` | INTEGER | Primary key |
| `s_name` | VARCHAR(25) | Supplier name |
| `s_address` | VARCHAR(40) | Street address |
| `s_nationkey` | INTEGER | FK → nation.n_nationkey |
| `s_phone` | VARCHAR(15) | Phone number |
| `s_acctbal` | DECIMAL(15,2) | Account balance |
| `s_comment` | VARCHAR(101) | Free-text comment |

### customer
| Column | Type | Description |
|--------|------|-------------|
| `c_custkey` | INTEGER | Primary key |
| `c_name` | VARCHAR(25) | Customer name |
| `c_address` | VARCHAR(40) | Street address |
| `c_nationkey` | INTEGER | FK → nation.n_nationkey |
| `c_phone` | VARCHAR(15) | Phone number |
| `c_acctbal` | DECIMAL(15,2) | Account balance |
| `c_mktsegment` | VARCHAR(10) | Market segment (AUTOMOBILE, BUILDING, FURNITURE, HOUSEHOLD, MACHINERY) |
| `c_comment` | VARCHAR(117) | Free-text comment |

### orders
| Column | Type | Description |
|--------|------|-------------|
| `o_orderkey` | INTEGER | Primary key |
| `o_custkey` | INTEGER | FK → customer.c_custkey |
| `o_orderstatus` | CHAR(1) | F = fulfilled, O = open, P = partial |
| `o_totalprice` | DECIMAL(15,2) | Total order price |
| `o_orderdate` | DATE | Date the order was placed |
| `o_orderpriority` | VARCHAR(15) | 1-URGENT, 2-HIGH, 3-MEDIUM, 4-NOT SPECIFIED, 5-LOW |
| `o_clerk` | VARCHAR(15) | Clerk identifier |
| `o_shippriority` | INTEGER | Shipping priority |
| `o_comment` | VARCHAR(79) | Free-text comment |

### lineitem
| Column | Type | Description |
|--------|------|-------------|
| `l_orderkey` | INTEGER | FK → orders.o_orderkey (composite PK part 1) |
| `l_partkey` | INTEGER | FK → part.p_partkey |
| `l_suppkey` | INTEGER | FK → supplier.s_suppkey |
| `l_linenumber` | INTEGER | Line number within order (composite PK part 2) |
| `l_quantity` | DECIMAL(15,2) | Quantity ordered |
| `l_extendedprice` | DECIMAL(15,2) | Extended price (quantity × list price) |
| `l_discount` | DECIMAL(15,2) | Discount percentage (0.00–0.10) |
| `l_tax` | DECIMAL(15,2) | Tax rate |
| `l_returnflag` | CHAR(1) | R = returned, A = accepted, N = none |
| `l_linestatus` | CHAR(1) | O = open, F = fulfilled |
| `l_shipdate` | DATE | Ship date |
| `l_commitdate` | DATE | Commit date |
| `l_receiptdate` | DATE | Receipt date |
| `l_shipinstruct` | VARCHAR(25) | Shipping instructions |
| `l_shipmode` | VARCHAR(10) | SHIP, TRUCK, AIR, MAIL, RAIL, REG AIR, FOB |
| `l_comment` | VARCHAR(44) | Free-text comment |

### part
| Column | Type | Description |
|--------|------|-------------|
| `p_partkey` | INTEGER | Primary key |
| `p_name` | VARCHAR(55) | Part name |
| `p_mfgr` | VARCHAR(25) | Manufacturer |
| `p_brand` | VARCHAR(10) | Brand (e.g., Brand#13) |
| `p_type` | VARCHAR(25) | Part type |
| `p_size` | INTEGER | Part size |
| `p_container` | VARCHAR(10) | Container type |
| `p_retailprice` | DECIMAL(15,2) | Retail price |
| `p_comment` | VARCHAR(23) | Free-text comment |

### partsupp
| Column | Type | Description |
|--------|------|-------------|
| `ps_partkey` | INTEGER | FK → part.p_partkey (composite PK part 1) |
| `ps_suppkey` | INTEGER | FK → supplier.s_suppkey (composite PK part 2) |
| `ps_availqty` | INTEGER | Available quantity from this supplier |
| `ps_supplycost` | DECIMAL(15,2) | Supply cost |
| `ps_comment` | VARCHAR(199) | Free-text comment |

## Key Relationships

```
region (1) ──── (N) nation (1) ──── (N) customer (1) ──── (N) orders (1) ──── (N) lineitem
                         │                                                          │
                         └──── (N) supplier (1) ──── (N) partsupp (N) ──── (1) part ┘
                                                          │
                                                     lineitem references
                                                     both l_partkey and l_suppkey
```
