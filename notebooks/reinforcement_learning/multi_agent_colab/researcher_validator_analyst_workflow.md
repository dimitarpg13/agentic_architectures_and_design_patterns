## Researcher agent, Validator Agent and Analyst agent workflow

```
User Query: "Should we adopt GraphRAG for our search system?"

┌─────────────────────────────────────────────────────────┐
│ 1. RESEARCH AGENT                                       │
│ - Web search for GraphRAG papers, benchmarks           │
│ - Document retrieval from technical blogs              │
│ - API calls to gather implementation examples          │
│ Output: Raw findings with sources                      │
└─────────────────────────────────────────────────────────┘
                         ↓
┌─────────────────────────────────────────────────────────┐
│ 2. VALIDATOR AGENT (First Pass)                        │
│ - Source credibility check                              │
│ - Detect contradictory claims in research               │
│ - Flag unsupported assertions                           │
│ - Identify information gaps                             │
│ Output: Validated findings + refinement requests       │
└─────────────────────────────────────────────────────────┘
                         ↓
            ┌────────────┴────────────┐
            │ Research refinement     │
            │ if gaps found           │
            └────────────┬────────────┘
                         ↓
┌─────────────────────────────────────────────────────────┐
│ 3. ANALYST AGENT                                        │
│ - Compare GraphRAG vs traditional RAG trade-offs        │
│ - Cost/performance analysis                             │
│ - Integration complexity assessment                     │
│ - Generate recommendation with reasoning                │
│ Output: Structured analysis + recommendation            │
└─────────────────────────────────────────────────────────┘
                         ↓
┌─────────────────────────────────────────────────────────┐
│ 4. VALIDATOR AGENT (Second Pass)                       │
│ - Verify all claims trace back to research evidence    │
│ - Check logical consistency of conclusions             │
│ - Ensure recommendation aligns with stated constraints │
│ - Validate numerical accuracy (costs, metrics)         │
│ - Detect reasoning fallacies or unsupported leaps      │
│ Output: Approved analysis OR feedback for revision     │
└─────────────────────────────────────────────────────────┘
                         ↓
            ┌────────────┴────────────┐
            │ Analyst revision        │
            │ if validation fails     │
            └────────────┬────────────┘
                         ↓
                  Final Output
```

### Validator Agent's Unique Value

**Evidence Traceability**
The validator ensures every claim in the analyst's output can be traced to specific research findings:

```python
# Pseudocode for validation check
for claim in analyst_output.claims:
    if not validator.find_supporting_evidence(claim, research_findings):
        validator.flag_issue(claim, "Unsupported assertion")
```

**Logical Consistency Checks**

* **Internal contradictions**: "The analysis says GraphRAG is cost-effective but also recommends against it due to high costs"

* **Reasoning gaps**: "Conclusion X doesn't follow from premises A and B"

* **Scope violations**: Analyst making claims beyond what research covered

**Quantitative Validation**

* Cross-check numerical claims (e.g., "30% performance improvement") against source data

* Verify calculations in cost estimates

* Ensure benchmark comparisons are apples-to-apples

**Hallucination Detection**
Critical for catching when the analyst:

* Invents technical details not in research

* Misattributes features to wrong tools

* Creates synthetic "facts" from pattern matching


### LangGraph Implementation Pattern


```python
from langgraph.graph import StateGraph, END
from typing import TypedDict, List

class WorkflowState(TypedDict):
    query: str
    research_findings: List[dict]
    validation_issues: List[str]
    analysis: dict
    final_output: dict
    iteration_count: int

def should_refine_research(state):
    """Route based on first validation pass"""
    return "refine_research" if state["validation_issues"] else "analyze"

def should_revise_analysis(state):
    """Route based on second validation pass"""
    if state["validation_issues"]:
        if state["iteration_count"] < 2:  # Max 2 revision loops
            return "revise_analysis"
        else:
            return "human_review"  # Escalate if validation keeps failing
    return END

workflow = StateGraph(WorkflowState)

workflow.add_node("research", research_agent)
workflow.add_node("validate_research", validator_agent_research_mode)
workflow.add_node("analyze", analyst_agent)
workflow.add_node("validate_analysis", validator_agent_analysis_mode)
workflow.add_node("refine_research", research_refinement)
workflow.add_node("revise_analysis", analyst_revision)

workflow.set_entry_point("research")
workflow.add_edge("research", "validate_research")
workflow.add_conditional_edges("validate_research", should_refine_research)
workflow.add_edge("refine_research", "validate_research")
workflow.add_edge("analyze", "validate_analysis")
workflow.add_conditional_edges("validate_analysis", should_revise_analysis)
workflow.add_edge("revise_analysis", "validate_analysis")
```
