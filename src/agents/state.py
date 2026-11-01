"""Graph state (TypedDict) and the Pydantic contracts exchanged between agents.

Two graphs share these definitions:

* ``AgentState``  - the copilot run (intake -> triage/fan-out -> briefing).
* ``CaseState``   - one investigation (Data Analyst -> ML Diagnostics ->
  Supply Chain Strategist -> grounding check). Morning triage runs several
  cases in parallel with ``Send``; a single question runs exactly one.

The two only share reducer channels (``cases``, ``trace``) so parallel
investigations merge cleanly into the parent run.
"""
from __future__ import annotations

import operator
from typing import Annotated, Any, Literal

from pydantic import BaseModel, Field
from typing_extensions import TypedDict

Intent = Literal["sku_investigation", "analytics_question", "triage", "clarify"]


class TraceStep(TypedDict, total=False):
    case_id: str
    agent: str          # Intake | Data Analyst | ML Diagnostics | Supply Chain Strategist | Grounding Guard | Briefing
    action: str         # short title shown in the UI
    detail: str         # one-line explanation
    tool: str           # tool / query name, if any
    sql: str            # SQL text, if any
    output: Any         # small JSON-able payload for the UI expander
    duration_ms: int
    status: Literal["ok", "warning", "error"]


class CaseResult(TypedDict, total=False):
    case_id: str
    intent: Intent
    question: str
    sku_id: str | None
    dc_id: str | None
    sku_name: str | None
    risk_class: str | None
    brief: dict                    # ExecutiveBrief.model_dump() or AnalyticsAnswer.model_dump()
    brief_markdown: str
    evidence: dict
    grounding: dict
    author: str                    # "llm:<provider>" or "deterministic"


# --------------------------------------------------------------------------- #
# LangGraph states
# --------------------------------------------------------------------------- #
class CaseState(TypedDict, total=False):
    case_id: str
    case_question: str
    intent: Intent
    sku_id: str | None
    dc_id: str | None
    risk_context: dict | None      # triage row that triggered the case, if any
    sql_evidence: list[dict]       # [{name, sql, rows, row_count}]
    sql_attempts: int
    sql_error: str | None
    ml_evidence: dict
    brief: dict | None
    author: str
    strategist_attempts: int
    grounding: dict | None
    grounding_feedback: str | None
    trace: Annotated[list[TraceStep], operator.add]
    cases: Annotated[list[CaseResult], operator.add]


class CaseOutput(TypedDict, total=False):
    """What an investigation hands back to the parent graph."""
    trace: Annotated[list[TraceStep], operator.add]
    cases: Annotated[list[CaseResult], operator.add]


class AgentState(TypedDict, total=False):
    question: str
    mode: Literal["question", "triage"]
    intent: Intent
    entities: dict
    triage: dict | None
    clarification: str | None
    max_cases: int
    trace: Annotated[list[TraceStep], operator.add]
    cases: Annotated[list[CaseResult], operator.add]
    final_report: str


# --------------------------------------------------------------------------- #
# Structured LLM outputs
# --------------------------------------------------------------------------- #
class SQLDraft(BaseModel):
    """A single read-only DuckDB query answering the user's question."""
    sql: str = Field(description="One DuckDB SELECT statement (CTEs allowed). No semicolons, no DDL/DML.")
    rationale: str = Field(description="One sentence on how the query answers the question.")


class RootCause(BaseModel):
    factor: str = Field(description="Short name of the driver, e.g. 'Demand acceleration'.")
    evidence: str = Field(description="The specific facts (numbers copied verbatim from the evidence) proving it.")
    impact: Literal["HIGH", "MEDIUM", "LOW"]


class Action(BaseModel):
    action_type: Literal["PURCHASE_ORDER", "STOCK_TRANSFER", "EXPEDITE", "PROMO_ADJUSTMENT",
                         "POLICY_UPDATE", "PAUSE_REPLENISHMENT", "INVESTIGATE", "NO_ACTION"]
    description: str
    quantity_units: int | None = Field(default=None, description="Units, copied from the replenishment tool output.")
    from_dc_id: str | None = None
    to_dc_id: str | None = None
    due_date: str | None = Field(default=None, description="ISO date YYYY-MM-DD")
    owner: str = Field(default="Supply Planner", description="Role accountable for the action.")


class ExecutiveBrief(BaseModel):
    """Root-cause analysis and decision for one SKU at one DC."""
    headline: str = Field(description="One sentence a VP can read in five seconds.")
    situation: str = Field(description="2-3 sentences: current position and what happens if nothing changes.")
    root_causes: list[RootCause] = Field(min_length=1, max_length=5)
    recommended_actions: list[Action] = Field(min_length=1, max_length=5)
    risk_if_no_action: str
    confidence: Literal["HIGH", "MEDIUM", "LOW"]
    confidence_rationale: str


class AnalyticsAnswer(BaseModel):
    """Answer to an ad-hoc analytics question, based only on the SQL result."""
    headline: str
    answer: str = Field(description="3-5 sentences citing numbers verbatim from the query result.")
    key_figures: list[str] = Field(default_factory=list, max_length=5)
    caveats: str | None = None
