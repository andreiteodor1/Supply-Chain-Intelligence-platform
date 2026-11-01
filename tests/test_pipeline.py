"""End-to-end checks: SQL guard, inventory maths, and the agent graph in both modes.

The LLM path is exercised with a scripted stub model, so these tests prove the
Grounding Guard rejects an invented number and the analyst self-corrects a
failing query, without needing an API key.
"""
from __future__ import annotations

import pytest

from src import ml_pipeline as ml
from src import tools as T
from src.agents.graph import build_graph, check_grounding, run_copilot
from src.agents.state import AnalyticsAnswer, ExecutiveBrief, SQLDraft
from src.db import UnsafeQueryError, run_query, validate_sql


# --------------------------------------------------------------------------- #
# SQL guard
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("sql", [
    "DROP TABLE dim_sku",
    "DELETE FROM fact_daily_sales",
    "SELECT 1; SELECT 2",
    "SELECT * FROM read_csv('C:/secrets.csv')",
    "ATTACH 'other.duckdb'",
    "COPY dim_sku TO 'out.csv'",
    "PRAGMA database_list",
    "INSTALL httpfs",
    "WITH x AS (SELECT 1) INSERT INTO dim_sku SELECT * FROM x",
])
def test_guard_rejects(sql):
    with pytest.raises(UnsafeQueryError):
        validate_sql(sql)


def test_guard_allows_reads_and_keywords_inside_strings():
    assert run_query("SELECT sku_name FROM dim_sku WHERE sku_name LIKE '%delete%'").row_count == 0
    res = run_query("WITH t AS (SELECT * FROM dim_sku) SELECT COUNT(*) AS n FROM t")
    assert int(res.df.n.iloc[0]) == 14


def test_row_cap():
    res = run_query("SELECT * FROM fact_daily_sales", max_rows=10)
    assert res.truncated and len(res.df) == 10


# --------------------------------------------------------------------------- #
# Entity resolution & deterministic maths
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("text,sku,dc", [
    ("Why is Ariel Pods 40ct at risk of stocking out in Bucharest?", "FC-ARL-PODS40", "DC-BUC"),
    ("pampers active baby in cluj", "BC-PMP-AB4-76", "DC-CLJ"),
    ("Fusion5 blades Iasi", "GR-GIL-F5-4", "DC-IAS"),
    ("head & shoulders bucuresti", "HC-HS-CLS400", "DC-BUC"),
])
def test_entity_resolution(text, sku, dc):
    ent = T.resolve_entities(text)
    assert (ent["sku_id"], ent["dc_id"]) == (sku, dc)


def test_replenishment_is_case_rounded_and_consistent():
    rep = ml.recommend_replenishment("FC-ARL-PODS40", "DC-BUC")
    assert rep["recommended_order_units"] % rep["case_pack"] == 0
    assert rep["effective_lead_time_days"] == rep["nominal_lead_time_days"] + rep["disruption_added_days"]
    assert rep["shortfall_before_po_arrival_units"] == rep["stockout_projection"]["lost_units_before_new_po"]
    assert rep["inventory_position_units"] == rep["on_hand_units"] + rep["on_order_units"]


def test_engineered_scenarios_are_detected():
    tri = ml.triage_scan().set_index(["sku_id", "dc_id"])
    assert tri.loc[("FC-ARL-PODS40", "DC-BUC"), "risk_class"] == "CRITICAL_STOCKOUT"
    assert tri.loc[("BC-PMP-AB4-76", "DC-CLJ"), "risk_class"] == "OVERSTOCK"
    recent = ml.score_anomalies(lookback_days=7)
    assert {("HC-HS-CLS400", "DC-BUC"), ("GR-GIL-F5-4", "DC-TIM")} <= set(zip(recent.sku_id, recent.dc_id))


def test_grounding_check():
    ev = {"a": 2214, "b": 0.523, "d": "2026-11-02", "c": 36979.59}
    assert check_grounding(["Order 2,214 units by 2026-11-02 (+52.3%), 36,980 RON at risk"], ev)["passed"]
    bad = check_grounding(["Order 3,000 units by 2026-11-09"], ev)
    assert not bad["passed"] and bad["unsupported_numbers"] == ["3,000"] and bad["unsupported_dates"] == ["2026-11-09"]


# --------------------------------------------------------------------------- #
# Agent graph
# --------------------------------------------------------------------------- #
def test_deterministic_question():
    out = run_copilot("Why is Ariel Pods 40ct at risk in Bucharest?", graph=build_graph(llm=None))
    case = out["cases"][0]
    assert case["grounding"]["passed"]
    assert {s["agent"] for s in out["trace"]} >= {"Intake", "Data Analyst", "ML Diagnostics",
                                                 "Supply Chain Strategist", "Grounding Guard"}
    qty = ml.recommend_replenishment("FC-ARL-PODS40", "DC-BUC")["recommended_order_units"]
    assert f"{qty:,}" in case["brief_markdown"]


def test_deterministic_triage_fans_out():
    out = run_copilot(triage=True, max_cases=3, graph=build_graph(llm=None))
    assert len(out["cases"]) == 3
    assert any(c["risk_class"] == "OVERSTOCK" for c in out["cases"])
    assert all(c["grounding"]["passed"] for c in out["cases"])


def test_template_analytics_and_clarify():
    g = build_graph(llm=None)
    assert "Baby Care" in run_copilot("lost sales by category", graph=g)["final_report"]
    assert run_copilot("what is the meaning of life", graph=g)["intent"] == "clarify"


class ScriptedLLM:
    """Minimal stand-in for a LangChain chat model: returns scripted structured outputs in order."""

    def __init__(self, script):
        self.script = list(script)
        self.calls = []
        self.model_name = "scripted-stub"

    def with_structured_output(self, schema):
        llm = self

        class Runnable:
            def invoke(self, messages):
                llm.calls.append((schema.__name__, messages))
                item = llm.script.pop(0)
                return item(messages) if callable(item) else item
        return Runnable()


def _brief(headline: str, qty: int) -> ExecutiveBrief:
    return ExecutiveBrief(
        headline=headline, situation="Stock runs out before the next PO can land.",
        root_causes=[{"factor": "Supply disruption", "evidence": "Lead time extended by the active disruption.",
                      "impact": "HIGH"}],
        recommended_actions=[{"action_type": "PURCHASE_ORDER", "description": "Raise a PO.", "quantity_units": qty}],
        risk_if_no_action="Lost sales.", confidence="HIGH", confidence_rationale="Model and stock data agree.")


def test_llm_brief_with_invented_number_is_rejected_then_fixed():
    qty = ml.recommend_replenishment("FC-ARL-PODS40", "DC-BUC")["recommended_order_units"]
    llm = ScriptedLLM([_brief("Order 9,876 units now.", 9876), _brief(f"Order {qty:,} units now.", qty)])
    out = run_copilot("Why is Ariel Pods 40ct at risk in Bucharest?", graph=build_graph(llm=llm))
    case = out["cases"][0]
    assert len(llm.calls) == 2
    assert "9,876" in llm.calls[1][1][-1].content          # guard feedback names the invented value
    assert case["author"].startswith("llm") and case["grounding"]["passed"]
    assert any(s["action"] == "Rejected draft" for s in out["trace"])


def test_llm_brief_falls_back_after_second_failure():
    llm = ScriptedLLM([_brief("Order 9,876 units.", 9876), _brief("Order 8,765 units.", 8765)])
    case = run_copilot("Ariel Pods 40ct Bucharest", graph=build_graph(llm=llm))["cases"][0]
    assert case["author"] == "deterministic" and case["grounding"]["passed"]


def test_text_to_sql_self_corrects():
    llm = ScriptedLLM([
        SQLDraft(sql="SELECT category, SUM(lost_units) FROM fact_daily_sales GROUP BY 1", rationale="first try"),
        SQLDraft(sql="SELECT k.category, SUM(s.lost_sales_units) AS lost_units FROM fact_daily_sales s "
                     "JOIN dim_sku k USING (sku_id) GROUP BY 1 ORDER BY 2 DESC", rationale="fixed join"),
        lambda msgs: AnalyticsAnswer(headline="Baby Care leads lost sales",
                                     answer="Baby Care has the most lost units."),
    ])
    out = run_copilot("Which category has the most lost units over two years?", graph=build_graph(llm=llm))
    sql_steps = [s for s in out["trace"] if s["action"].startswith("Text-to-SQL")]
    assert [s["status"] for s in sql_steps] == ["error", "ok"]
    assert "ERROR" in llm.calls[1][1][-1].content
    assert out["cases"][0]["grounding"]["passed"]
