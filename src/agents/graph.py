"""LangGraph multi-agent copilot.

Parent graph (``AgentState``)::

    START -> intake --(question)--> Send(investigate) ------------------> briefing -> END
                    \\-(triage)----> triage_scan -> Send(investigate) x N -/
                    \\-(clarify)-------------------------------------------/

Investigation subgraph (``CaseState``), one per SKU x DC or analytics question::

    START -> data_analyst --(sql error, retry <= 3)--> data_analyst
                          \\-(sku case)--> ml_diagnostics -> strategist
                          \\-(analytics)-------------------> strategist
    strategist -> grounding_guard --(unsupported numbers, retry once)--> strategist
                                  \\-> END

Design rules
------------
* Routing, entity resolution and every calculation are deterministic.
  The LLM is used where language is needed: writing ad-hoc SQL and writing
  the executive narrative.
* The Grounding Guard extracts every number and date in the narrative and
  rejects the draft if one is not present in the tool evidence, so the LLM
  cannot invent a figure. After one failed retry the deterministic brief is
  used instead.
* With no API key the graph runs end to end in deterministic mode, so the
  demo always works.
"""
from __future__ import annotations

import json
import math
import os
import re
import time
import uuid
from contextlib import contextmanager
from functools import lru_cache
from typing import Any, Iterator

from dotenv import load_dotenv
from langchain_core.messages import HumanMessage, SystemMessage
from langgraph.graph import END, START, StateGraph
from langgraph.types import Send

from src import PROJECT_ROOT
from src import tools as T
from src.agents.state import (AgentState, AnalyticsAnswer, CaseOutput, CaseResult, CaseState, ExecutiveBrief,
                              SQLDraft, TraceStep)
from src.db import as_of_date, schema_description

load_dotenv(PROJECT_ROOT / ".env")

MAX_SQL_ATTEMPTS = 3
MAX_STRATEGIST_ATTEMPTS = 2
DEFAULT_TRIAGE_CASES = 3

AGENT_ANALYST = "Data Analyst"
AGENT_ML = "ML Diagnostics"
AGENT_STRATEGIST = "Supply Chain Strategist"
AGENT_GUARD = "Grounding Guard"


# --------------------------------------------------------------------------- #
# LLM provider (configurable via .env; None => deterministic mode)
# --------------------------------------------------------------------------- #
def _provider() -> str:
    p = os.getenv("LLM_PROVIDER", "auto").strip().lower()
    if p != "auto":
        return p
    if os.getenv("OPENAI_API_KEY"):
        return "openai"
    if os.getenv("GROQ_API_KEY"):
        return "groq"
    return "mock"


@lru_cache(maxsize=1)
def get_llm():
    """Return a LangChain chat model, or None for deterministic (mock) mode."""
    provider = _provider()
    temperature = float(os.getenv("LLM_TEMPERATURE", "0"))
    try:
        if provider == "openai" and os.getenv("OPENAI_API_KEY"):
            from langchain_openai import ChatOpenAI
            return ChatOpenAI(model=os.getenv("OPENAI_MODEL", "gpt-4o-mini"), temperature=temperature, timeout=60)
        if provider == "groq" and os.getenv("GROQ_API_KEY"):
            from langchain_groq import ChatGroq
            return ChatGroq(model=os.getenv("GROQ_MODEL", "llama-3.3-70b-versatile"), temperature=temperature)
        if provider == "ollama":
            from langchain_ollama import ChatOllama
            return ChatOllama(model=os.getenv("OLLAMA_MODEL", "llama3.1"),
                              base_url=os.getenv("OLLAMA_BASE_URL", "http://localhost:11434"),
                              temperature=temperature)
    except Exception as exc:  # missing package or bad config -> keep the demo alive
        print(f"[cpg-pulse] LLM provider '{provider}' unavailable ({exc}); using deterministic mode.")
    return None


def describe_llm(llm=None) -> str:
    llm = get_llm() if llm is None else llm
    if llm is None or llm is False:
        return "deterministic (no LLM configured)"
    name = getattr(llm, "model_name", None) or getattr(llm, "model", None) or type(llm).__name__
    return f"{_provider()}:{name}"


# --------------------------------------------------------------------------- #
# Helpers
# --------------------------------------------------------------------------- #
def _step(case_id: str, agent: str, action: str, detail: str = "", **kw) -> TraceStep:
    return TraceStep(case_id=case_id, agent=agent, action=action, detail=detail, status=kw.pop("status", "ok"), **kw)


@contextmanager
def _timer() -> Iterator[dict]:
    box = {"ms": 0}
    t0 = time.perf_counter()
    try:
        yield box
    finally:
        box["ms"] = int((time.perf_counter() - t0) * 1000)


def _preview(obj: Any, limit: int = 4000) -> Any:
    """Keep trace payloads small enough for the UI."""
    text = json.dumps(obj, default=str)
    if len(text) <= limit:
        return obj
    if isinstance(obj, dict):
        return {k: (v if len(json.dumps(v, default=str)) < 600 else f"<{type(v).__name__}, truncated>")
                for k, v in obj.items()}
    return text[:limit] + " ..."


def _first(rows: list[dict] | None) -> dict:
    return rows[0] if rows else {}


def _pct(a: float | None, b: float | None) -> float | None:
    if a is None or b in (None, 0):
        return None
    return round((a / b - 1) * 100, 1)


def _fmt(n: float | int | None, dec: int = 0) -> str:
    if n is None:
        return "n/a"
    return f"{n:,.{dec}f}"


# --------------------------------------------------------------------------- #
# Deterministic analytics shared by the template brief and the LLM prompt
# --------------------------------------------------------------------------- #
def derive_metrics(sql: dict[str, list[dict]], ml: dict) -> dict:
    """Pre-compute the comparisons a narrative would want, so the LLM can quote rather than calculate."""
    trend, inv = _first(sql.get("demand_trend")), _first(sql.get("inventory_position"))
    comp = _first(sql.get("competitor_pricing"))
    proj, rep, fc = ml.get("projection", {}), ml.get("replenishment", {}), ml.get("forecast", {})
    promos = [p for p in sql.get("promo_calendar", []) if p.get("status") in ("PLANNED", "ACTIVE")]
    cancelled = [p for p in sql.get("promo_calendar", []) if p.get("status") == "CANCELLED"]
    d = {
        "demand_change_last_7d_vs_prior_28d_pct": _pct(trend.get("avg_daily_demand_last_7d"),
                                                       trend.get("avg_daily_demand_prior_28d")),
        "demand_change_vs_same_period_last_year_pct": _pct(trend.get("avg_daily_demand_prior_28d"),
                                                           trend.get("avg_daily_demand_same_period_ly")),
        "forecast_daily_vs_policy_daily_pct": _pct(fc.get("avg_daily_forecast"), inv.get("planned_daily_demand")),
        "competitor_index_change_pct": _pct(comp.get("competitor_index_last_7d"),
                                            comp.get("competitor_index_4_8_weeks_ago")),
        "next_promo_start_date": promos[0]["start_date"] if promos else None,
        "next_promo_depth_pct": round(promos[0]["promo_depth_pct"] * 100) if promos else None,
        "cancelled_promo": cancelled[0] if cancelled else None,
    }
    if rep:
        eff_lt = rep.get("effective_lead_time_days", 0)
        d["suggested_reorder_point_units"] = int(round(fc.get("avg_daily_forecast", 0) * eff_lt
                                                       + rep.get("safety_stock_units", 0)))
        d["current_reorder_point_units"] = rep.get("current_policy", {}).get("reorder_point_units")
    if proj:
        d["days_until_stockout"] = proj.get("days_until_stockout")
        d["days_of_cover_forecast"] = proj.get("days_of_cover")
    return {k: v for k, v in d.items() if v is not None}


def template_brief(case: CaseState) -> ExecutiveBrief:
    """Rule-based RCA used in deterministic mode and as the guard's fallback. Every number comes from evidence."""
    sql = {q["name"]: q["rows"] for q in case.get("sql_evidence", [])}
    ml = case.get("ml_evidence", {})
    dm = ml.get("derived", {})
    proj, rep, an = ml.get("projection", {}), ml.get("replenishment", {}), ml.get("anomalies", {})
    trend, inv = _first(sql.get("demand_trend")), _first(sql.get("inventory_position"))
    name, dc = rep.get("sku_name", case.get("sku_id")), case.get("dc_id")

    causes: list[dict] = []
    chg = dm.get("demand_change_last_7d_vs_prior_28d_pct")
    if chg is not None and chg >= 15:
        causes.append(dict(factor="Demand acceleration", impact="HIGH" if chg >= 30 else "MEDIUM",
                           evidence=f"Last 7 days averaged {_fmt(trend['avg_daily_demand_last_7d'], 1)} units/day vs "
                                    f"{_fmt(trend['avg_daily_demand_prior_28d'], 1)} in the prior 28 days ({chg:+.1f}%)."))
    elif chg is not None and chg <= -15:
        causes.append(dict(factor="Demand slowdown", impact="MEDIUM",
                           evidence=f"Last 7 days averaged {_fmt(trend['avg_daily_demand_last_7d'], 1)} units/day vs "
                                    f"{_fmt(trend['avg_daily_demand_prior_28d'], 1)} in the prior 28 days ({chg:+.1f}%)."))
    if dm.get("next_promo_start_date"):
        causes.append(dict(factor="Planned promotion", impact="HIGH",
                           evidence=f"Promo at {dm['next_promo_depth_pct']}% off starts {dm['next_promo_start_date']}; "
                                    f"the forecast includes the lift: {_fmt(proj.get('forecast_14d_units'))} units over "
                                    f"14 days (P90 {_fmt(proj.get('forecast_14d_p90_units'))})."))
    if rep.get("disruption_added_days"):
        ev = rep["active_disruptions"][0]
        causes.append(dict(factor="Supply disruption", impact="HIGH",
                           evidence=f"{ev['event_id']} ({ev['disruption_type']}, {ev['severity']}) adds "
                                    f"{rep['disruption_added_days']} days: effective lead time is "
                                    f"{rep['effective_lead_time_days']} days vs {rep['nominal_lead_time_days']} nominal."))
    late = [p for p in sql.get("open_purchase_orders", []) if p.get("status") == "DELAYED"]
    if late:
        p = late[0]
        causes.append(dict(factor="Late inbound PO", impact="MEDIUM",
                           evidence=f"{p['po_id']} ({_fmt(p['quantity_units'])} units) slipped {p['delay_days']} day{'s' if p['delay_days'] != 1 else ''} "
                                    f"to {p['revised_eta_date']}."))
    fvp = dm.get("forecast_daily_vs_policy_daily_pct")
    if fvp is not None and fvp >= 20:
        causes.append(dict(factor="Stale inventory policy", impact="MEDIUM",
                           evidence=f"Reorder point of {_fmt(dm.get('current_reorder_point_units'))} units is sized on "
                                    f"{_fmt(inv.get('planned_daily_demand'), 1)} units/day; the model forecasts "
                                    f"{_fmt(proj.get('avg_daily_forecast'), 1)} units/day ({fvp:+.1f}%)."))
    cc = dm.get("competitor_index_change_pct")
    if cc is not None and cc <= -5:
        causes.append(dict(factor="Competitor price cut", impact="MEDIUM",
                           evidence=f"Competitor price index moved {cc:+.1f}% vs 4-8 weeks ago."))
    if dm.get("cancelled_promo"):
        cp = dm["cancelled_promo"]
        causes.append(dict(factor="Cancelled promotion", impact="HIGH",
                           evidence=f"{cp['promo_id']} ({cp['mechanic']}, {cp['start_date']} to {cp['end_date']}) was "
                                    f"cancelled after pre-build stock had landed."))
    if an.get("count"):
        a = an["anomalies"][0]
        causes.append(dict(factor="Anomaly detected", impact="MEDIUM",
                           evidence=f"IsolationForest flagged a {a['anomaly_type']} on {a['date']}: "
                                    f"{_fmt(a['units_demanded'])} units vs a 28-day baseline of "
                                    f"{_fmt(a['baseline_28d'], 1)} (score {a['anomaly_score']})."))
    if not causes:
        causes.append(dict(factor="No material driver", impact="LOW",
                           evidence=f"Demand and supply are in line with plan; {_fmt(proj.get('days_of_cover'), 1)} "
                                    f"days of forecast cover."))

    actions: list[dict] = []
    stockout = proj.get("stockout_date_expected")
    overstock = (proj.get("days_of_cover") or 0) > 30
    if rep.get("recommended_order_units", 0) > 0:
        actions.append(dict(action_type="PURCHASE_ORDER", owner="Supply Planner", due_date=proj.get("as_of_date"),
                            quantity_units=rep["recommended_order_units"], to_dc_id=dc,
                            description=f"Raise a PO to {rep['supplier_id']} for {_fmt(rep['recommended_order_units'])} "
                                        f"units ({rep['recommended_order_cases']} cases, "
                                        f"{_fmt(rep['order_value_ron'])} RON); expected arrival "
                                        f"{rep['expected_po_arrival_date']}."))
    for t in rep.get("transfer_options", [])[:2]:
        actions.append(dict(action_type="STOCK_TRANSFER", owner="Logistics Planner", quantity_units=t["transfer_units"],
                            from_dc_id=t["from_dc_id"], to_dc_id=dc, due_date=proj.get("as_of_date"),
                            description=f"Transfer {_fmt(t['transfer_units'])} units from {t['from_dc_id']} "
                                        f"({t['transit_days']}-day transit, {_fmt(t['cost_ron'], 2)} RON); donor keeps "
                                        f"{_fmt(t['donor_days_of_cover_after'], 1)} days of cover."))
    gap = rep.get("shortfall_before_po_arrival_units", 0)
    if gap > 0 and not rep.get("transfer_options"):
        actions.append(dict(action_type="EXPEDITE", owner="Supply Planner", quantity_units=gap,
                            description=f"No DC can spare stock: expedite the open inbound and agree customer "
                                        f"allocation for the {_fmt(gap)}-unit gap before the new PO lands."))
    if dm.get("next_promo_start_date") and proj.get("stockout_before_new_po_can_land"):
        actions.append(dict(action_type="PROMO_ADJUSTMENT", owner="Customer / Revenue Manager",
                            due_date=dm["next_promo_start_date"],
                            description=f"Align with the customer team on the promo starting "
                                        f"{dm['next_promo_start_date']}: cap leaflet volumes or phase the start "
                                        f"until supply recovers."))
    if (fvp or 0) >= 20 and dm.get("suggested_reorder_point_units"):
        actions.append(dict(action_type="POLICY_UPDATE", owner="Demand Planner",
                            description=f"Raise the reorder point from {_fmt(dm['current_reorder_point_units'])} to "
                                        f"{_fmt(dm['suggested_reorder_point_units'])} units to reflect current demand."))
    if overstock:
        actions.append(dict(action_type="PAUSE_REPLENISHMENT", owner="Supply Planner",
                            description=f"Hold replenishment: {_fmt(proj.get('days_of_cover'), 1)} days of cover. "
                                        f"Offer the excess to DCs at risk via inter-DC transfer."))
    if not actions:
        actions.append(dict(action_type="NO_ACTION", owner="Supply Planner",
                            description="No action required; keep on standard review cycle."))

    if stockout:
        headline = (f"{name} at {dc} runs out on {stockout} ({proj['days_until_stockout']} days) and a PO raised "
                    f"today cannot land before {proj.get('earliest_new_po_arrival_date')}."
                    if proj.get("stockout_before_new_po_can_land") else
                    f"{name} at {dc} projected to run out on {stockout}; a PO raised today lands in time.")
    elif overstock:
        headline = f"{name} at {dc} is overstocked with {_fmt(proj.get('days_of_cover'), 1)} days of cover."
    else:
        headline = f"{name} at {dc} is covered for the next 14 days."
    situation = (f"On hand {_fmt(proj.get('on_hand_units'))} units with {_fmt(proj.get('on_order_units'))} on order, "
                 f"against forecast demand of {_fmt(proj.get('forecast_14d_units'))} units over 14 days "
                 f"({_fmt(proj.get('avg_daily_forecast'), 1)}/day). ")
    if proj.get("lost_units_before_new_po"):
        situation += (f"Without intervention about {_fmt(proj['lost_units_before_new_po'])} units "
                      f"({_fmt(proj['revenue_at_risk_ron'])} RON) are lost before replenishment can arrive.")
    risk = (f"{_fmt(proj.get('lost_units_before_new_po'))} units of lost sales ({_fmt(proj.get('revenue_at_risk_ron'))} "
            f"RON) and empty shelves during the promo window." if proj.get("lost_units_before_new_po")
            else "Working capital tied up and ageing stock." if overstock else "Low.")
    conf = "HIGH" if stockout and len(causes) >= 2 else "MEDIUM"
    return ExecutiveBrief(
        headline=headline, situation=situation.strip(), root_causes=causes[:5], recommended_actions=actions[:5],
        risk_if_no_action=risk, confidence=conf,
        confidence_rationale="Based on the HistGradientBoosting forecast (backtest WAPE in the model card), current "
                             "stock and open POs; the P90 scenario is shown alongside the expected case.")


def template_analytics_answer(title: str, rows: list[dict]) -> AnalyticsAnswer:
    if not rows:
        return AnalyticsAnswer(headline=title, answer="The query returned no rows.")
    cols = list(rows[0].keys())
    label, metric = cols[0], next((c for c in cols[1:] if isinstance(rows[0][c], (int, float))), None)
    num = lambda v: f"{v:,.0f}" if float(v).is_integer() else f"{v:,}"  # noqa: E731
    figs = []
    for r in rows[:3]:
        figs.append(", ".join(f"{k} {num(v)}" if isinstance(v, (int, float)) else f"{v}" for k, v in r.items()))
    lead = f"Top {label}: {rows[0][label]}" + (f" with {metric} {num(rows[0][metric])}." if metric else ".")
    return AnalyticsAnswer(headline=title, answer=f"{lead} {len(rows)} rows returned; full table below.",
                           key_figures=figs)


# --------------------------------------------------------------------------- #
# Grounding guard
# --------------------------------------------------------------------------- #
_DATE = re.compile(r"\b\d{4}-\d{2}-\d{2}\b")
_NUM = re.compile(r"(?<![\w.])[-+]?\d[\d,]*(?:\.\d+)?(?:\s?[kKmM]\b)?(?![\w-])")
# Small integers (days, counts of causes, "14-day") and common constants are not treated as claims.
_FREE_NUMBERS = {90.0, 95.0, 100.0}
_FREE_INT_MAX = 31


def _collect_evidence(obj: Any, nums: set[float], dates: set[str]) -> None:
    if isinstance(obj, bool) or obj is None:
        return
    if isinstance(obj, (int, float)):
        if not (isinstance(obj, float) and math.isnan(obj)):
            nums.add(float(obj))
    elif isinstance(obj, str):
        dates.update(_DATE.findall(obj))
        for m in _NUM.findall(_DATE.sub(" ", obj)):
            v = _parse_num(m)
            if v is not None:
                nums.add(v)
    elif isinstance(obj, dict):
        for v in obj.values():
            _collect_evidence(v, nums, dates)
    elif isinstance(obj, (list, tuple)):
        for v in obj:
            _collect_evidence(v, nums, dates)


def _parse_num(tok: str) -> float | None:
    t = tok.replace(",", "").replace(" ", "").lstrip("+")
    mult = 1.0
    if t[-1:] in "kK":
        mult, t = 1e3, t[:-1]
    elif t[-1:] in "mM":
        mult, t = 1e6, t[:-1]
    try:
        return float(t) * mult
    except ValueError:
        return None


def _supported(v: float, evidence: list[float], scaled: bool) -> bool:
    if abs(v) <= _FREE_INT_MAX and float(v).is_integer() or abs(v) in _FREE_NUMBERS:
        return True
    tol_rel = 0.02 if scaled else 0.006
    for e in evidence:
        for cand in (e, e * 100):  # fractions quoted as percentages
            if abs(abs(v) - abs(cand)) <= max(tol_rel * abs(cand), 0.051):
                return True
    return False


def check_grounding(texts: list[str], evidence: dict) -> dict:
    nums: set[float] = set()
    dates: set[str] = set()
    _collect_evidence(evidence, nums, dates)
    ev = sorted(nums)
    checked, bad, bad_dates = 0, [], []
    for text in texts:
        for d in _DATE.findall(text):
            if d not in dates:
                bad_dates.append(d)
        for tok in _NUM.findall(_DATE.sub(" ", text)):
            v = _parse_num(tok)
            if v is None:
                continue
            checked += 1
            if not _supported(v, ev, scaled=tok.strip()[-1:] in "kKmM"):
                bad.append(tok.strip())
    return {"passed": not bad and not bad_dates, "numbers_checked": checked,
            "unsupported_numbers": sorted(set(bad)), "unsupported_dates": sorted(set(bad_dates)),
            "evidence_values": len(ev)}


def _brief_texts(brief: dict) -> list[str]:
    out = []

    def walk(o):
        if isinstance(o, str):
            out.append(o)
        elif isinstance(o, dict):
            for k, v in o.items():
                if k == "quantity_units" and v is not None:
                    out.append(str(v))
                else:
                    walk(v)
        elif isinstance(o, list):
            for v in o:
                walk(v)
    walk(brief)
    return out


# --------------------------------------------------------------------------- #
# Rendering
# --------------------------------------------------------------------------- #
ACTION_ICONS = {"PURCHASE_ORDER": "PO", "STOCK_TRANSFER": "Transfer", "EXPEDITE": "Expedite",
                "PROMO_ADJUSTMENT": "Promo", "POLICY_UPDATE": "Policy", "PAUSE_REPLENISHMENT": "Hold",
                "INVESTIGATE": "Investigate", "NO_ACTION": "No action"}


def render_brief(brief: dict) -> str:
    if "answer" in brief:  # AnalyticsAnswer
        md = f"### {brief['headline']}\n\n{brief['answer']}\n"
        if brief.get("key_figures"):
            md += "\n" + "\n".join(f"- {f}" for f in brief["key_figures"]) + "\n"
        if brief.get("caveats"):
            md += f"\n_Caveat: {brief['caveats']}_\n"
        return md
    md = [f"### {brief['headline']}", "", brief["situation"], "", "**Root causes**", ""]
    for c in brief["root_causes"]:
        md.append(f"- **{c['factor']}** ({c['impact']}): {c['evidence']}")
    md += ["", "**Recommended actions**", "", "| # | Type | Action | Qty | Owner | Due |", "|---|---|---|---|---|---|"]
    for i, a in enumerate(brief["recommended_actions"], 1):
        qty = f"{a['quantity_units']:,}" if a.get("quantity_units") is not None else ""
        md.append(f"| {i} | {ACTION_ICONS.get(a['action_type'], a['action_type'])} | {a['description']} | {qty} | "
                  f"{a.get('owner', '')} | {a.get('due_date') or ''} |")
    md += ["", f"**Risk if no action:** {brief['risk_if_no_action']}", "",
           f"**Confidence:** {brief['confidence']}. {brief['confidence_rationale']}"]
    return "\n".join(md)


# --------------------------------------------------------------------------- #
# Prompts
# --------------------------------------------------------------------------- #
SQL_SYSTEM = """You are the Data Analyst agent for a P&G-style FMCG supply chain in Romania.
Write ONE DuckDB SELECT query that answers the question using only the schema below.
Rules:
- Read-only: a single SELECT (CTEs allowed). No DDL/DML, PRAGMA, SET, file functions or semicolons.
- "Today" is the as-of date {asof}; express relative periods against it, e.g. date > DATE '{asof}' - 28.
- Prefer aggregates over raw rows; ROUND money to 0 decimals and ratios to 3; add ORDER BY and LIMIT 50.
- Join dim_sku / dim_dc for readable names. Unconstrained demand = units_demanded; shipped = units_sold.

SCHEMA
{schema}"""

STRATEGIST_SYSTEM = """You are the Supply Chain Strategist agent at a P&G distribution hub in Romania.
Write an executive root-cause analysis and replenishment decision for ONE SKU at ONE distribution centre.

HARD RULES (a Grounding Guard rejects your answer if broken):
1. Every number and date you write must appear verbatim in the EVIDENCE JSON (rounding to fewer decimals is ok).
   Do not add, subtract, multiply or compute percentages yourself; use `derived` for comparisons.
2. Quantities in recommended_actions come from `replenishment` (recommended_order_units, transfer_options,
   shortfall_before_po_arrival_units). Never invent a quantity.
3. Write dates as YYYY-MM-DD. Money is in RON.
4. Root causes must cite the evidence that proves them. Prefer 2-4 causes, ranked by impact.
5. Be concise and decisive: a VP reads this in 30 seconds."""

ANALYTICS_SYSTEM = """You are the Supply Chain Strategist agent. Answer the business question using ONLY the
SQL result provided. Every number you write must appear in the result (rounding allowed); do not compute
new figures. If the result cannot fully answer the question, say so in `caveats`."""


def _compact_evidence(case: CaseState) -> dict:
    ml = dict(case.get("ml_evidence", {}))
    if "projection" in ml:
        proj = dict(ml["projection"])
        proj["daily_projection"] = [
            {k: r[k] for k in ("date", "forecast_units", "forecast_p90_units", "receipts_units", "projected_on_hand")}
            for r in proj.get("daily_projection", [])]
        ml["projection"] = proj
    if "forecast" in ml:
        ml["forecast"] = {k: v for k, v in ml["forecast"].items() if k != "daily"}
    return {"sql_facts": {q["name"]: q["rows"] for q in case.get("sql_evidence", [])}, **ml,
            "risk_context": case.get("risk_context")}


# --------------------------------------------------------------------------- #
# Investigation subgraph
# --------------------------------------------------------------------------- #
def build_case_graph(llm):
    def data_analyst(state: CaseState) -> dict:
        cid, intent = state["case_id"], state["intent"]
        if intent == "sku_investigation":
            steps, evidence = [], []
            for name, sql in T.sql_playbook(state["sku_id"], state["dc_id"]):
                sql = re.sub(r"\n\s*\n", "\n", sql).strip()
                with _timer() as t:
                    out = T.run_sql_query.invoke({"sql": sql})
                rows = out.get("rows", []) if out.get("ok") else []
                evidence.append({"name": name, "sql": sql, "rows": rows, "row_count": out.get("row_count", 0)})
                steps.append(_step(cid, AGENT_ANALYST, f"SQL: {name.replace('_', ' ')}",
                                   f"{out.get('row_count', 0)} row(s)" if out.get("ok") else out.get("error"),
                                   tool="run_sql_query", sql=sql, output=_preview(rows),
                                   duration_ms=t["ms"], status="ok" if out.get("ok") else "error"))
            return {"sql_evidence": evidence, "trace": steps}

        # Ad-hoc analytics question: LLM text-to-SQL with self-correction, or template match.
        attempts = state.get("sql_attempts", 0) + 1
        question = state["case_question"]
        if llm is None:
            match = T.match_analytics_template(question)
            if match is None:
                return {"sql_attempts": attempts, "sql_error": "no_template",
                        "trace": [_step(cid, AGENT_ANALYST, "No matching SQL template",
                                        "Free-form questions need an LLM; set LLM_PROVIDER in .env.",
                                        status="warning")]}
            title, sql = match
            rationale = f"Template: {title}"
        else:
            msgs = [SystemMessage(SQL_SYSTEM.format(asof=as_of_date().date(), schema=schema_description())),
                    HumanMessage(question)]
            if state.get("sql_error"):
                prev = state["sql_evidence"][-1]["sql"] if state.get("sql_evidence") else ""
                msgs.append(HumanMessage(f"Your previous query failed.\nQUERY:\n{prev}\nERROR:\n{state['sql_error']}\n"
                                         "Fix it and return a corrected query."))
            try:
                with _timer() as t:
                    draft: SQLDraft = llm.with_structured_output(SQLDraft).invoke(msgs)
                sql, rationale, title = draft.sql, draft.rationale, question
            except Exception as exc:
                return {"sql_attempts": MAX_SQL_ATTEMPTS, "sql_error": f"LLM error: {exc}",
                        "trace": [_step(cid, AGENT_ANALYST, "LLM call failed", str(exc)[:300], status="error")]}
        with _timer() as t:
            out = T.run_sql_query.invoke({"sql": sql})
        step = _step(cid, AGENT_ANALYST, f"Text-to-SQL (attempt {attempts})", rationale, tool="run_sql_query",
                     sql=sql, output=_preview(out.get("rows") if out.get("ok") else out.get("error")),
                     duration_ms=t["ms"], status="ok" if out.get("ok") else "error")
        ev = list(state.get("sql_evidence", [])) + [{"name": title, "sql": sql, "rows": out.get("rows", []),
                                                     "row_count": out.get("row_count", 0)}]
        return {"sql_attempts": attempts, "sql_error": None if out.get("ok") else out.get("error"),
                "sql_evidence": ev, "trace": [step]}

    def route_after_analyst(state: CaseState) -> str:
        if state["intent"] == "sku_investigation":
            return "ml_diagnostics"
        if state.get("sql_error") and state["sql_error"] != "no_template" and llm is not None \
                and state.get("sql_attempts", 0) < MAX_SQL_ATTEMPTS:
            return "data_analyst"
        return "strategist"

    def ml_diagnostics(state: CaseState) -> dict:
        cid, args = state["case_id"], {"sku_id": state["sku_id"], "dc_id": state["dc_id"]}
        calls = [
            ("forecast", T.forecast_demand, args,
             lambda o: f"{o['forecast_14d_units']:,} units over 14 days (P90 {o['forecast_14d_p90_units']:,}), "
                       f"{o['promo_days_in_horizon']} promo day(s) in horizon"),
            ("projection", T.project_stockout, args,
             lambda o: (f"Stockout {o['stockout_date_expected']} (P90 {o['stockout_date_p90']}); "
                        f"new PO lands {o['earliest_new_po_arrival_date']}"
                        if o["stockout_date_expected"] else
                        f"No stockout in 14 days; {o['days_of_cover']} days of cover")),
            ("anomalies", T.detect_anomalies, {**args, "lookback_days": 14},
             lambda o: f"{o['count']} anomaly flag(s) in the last 14 days"),
            ("replenishment", T.recommend_replenishment, args,
             lambda o: f"Order {o['recommended_order_units']:,} units ({o['recommended_order_cases']} cases); "
                       f"{len(o['transfer_options'])} transfer option(s)"),
        ]
        evidence, steps = {}, []
        for key, tool_fn, tool_args, summarise in calls:
            with _timer() as t:
                out = tool_fn.invoke(tool_args)
            evidence[key] = out
            steps.append(_step(cid, AGENT_ML, tool_fn.name.replace("_", " ").capitalize(), summarise(out),
                               tool=tool_fn.name, output=_preview(out), duration_ms=t["ms"]))
        sql = {q["name"]: q["rows"] for q in state.get("sql_evidence", [])}
        evidence["derived"] = derive_metrics(sql, evidence)
        steps.append(_step(cid, AGENT_ML, "Derived comparisons", "Deterministic deltas for the strategist to quote",
                           output=evidence["derived"]))
        return {"ml_evidence": evidence, "trace": steps}

    def strategist(state: CaseState) -> dict:
        cid, attempts = state["case_id"], state.get("strategist_attempts", 0) + 1
        analytics = state["intent"] != "sku_investigation"
        if analytics:
            ev = state.get("sql_evidence", [])
            last = ev[-1] if ev else None
            if last is None or state.get("sql_error"):
                msg = ("I could not map this question to a product/DC or a SQL template. Try naming a SKU and a DC "
                       "(e.g. 'Ariel Pods 40ct in Bucharest'), run the morning triage, or configure an LLM for "
                       "free-form analytics." if state.get("sql_error") == "no_template" or last is None
                       else f"The query failed after {state.get('sql_attempts')} attempt(s): {state.get('sql_error')}")
                brief = AnalyticsAnswer(headline="Could not answer", answer=msg).model_dump()
                return {"brief": brief, "author": "deterministic", "strategist_attempts": attempts,
                        "trace": [_step(cid, AGENT_STRATEGIST, "No answer", msg, status="warning")]}
            if llm is None:
                brief = template_analytics_answer(last["name"], last["rows"]).model_dump()
                return {"brief": brief, "author": "deterministic", "strategist_attempts": attempts,
                        "trace": [_step(cid, AGENT_STRATEGIST, "Summarised query result", "Deterministic summary")]}
            msgs = [SystemMessage(ANALYTICS_SYSTEM),
                    HumanMessage(f"QUESTION: {state['case_question']}\nSQL:\n{last['sql']}\n"
                                 f"RESULT (JSON):\n{json.dumps(last['rows'][:50], default=str)}")]
            schema = AnalyticsAnswer
        else:
            if llm is None:
                brief = template_brief(state).model_dump()
                return {"brief": brief, "author": "deterministic", "strategist_attempts": attempts,
                        "trace": [_step(cid, AGENT_STRATEGIST, "Drafted RCA and decision",
                                        f"Rule-based brief: {len(brief['root_causes'])} cause(s), "
                                        f"{len(brief['recommended_actions'])} action(s)", output=brief)]}
            msgs = [SystemMessage(STRATEGIST_SYSTEM),
                    HumanMessage(f"QUESTION / TRIGGER: {state['case_question']}\n\nEVIDENCE JSON:\n"
                                 f"{json.dumps(_compact_evidence(state), default=str)}")]
            schema = ExecutiveBrief
        if state.get("grounding_feedback"):
            msgs.append(HumanMessage(state["grounding_feedback"]))
        try:
            with _timer() as t:
                out = llm.with_structured_output(schema).invoke(msgs)
            brief = out.model_dump()
        except Exception as exc:
            fallback = (template_analytics_answer(state["sql_evidence"][-1]["name"], state["sql_evidence"][-1]["rows"])
                        if analytics else template_brief(state)).model_dump()
            return {"brief": fallback, "author": "deterministic", "strategist_attempts": MAX_STRATEGIST_ATTEMPTS,
                    "trace": [_step(cid, AGENT_STRATEGIST, "LLM call failed, used rule-based brief",
                                    str(exc)[:300], status="warning")]}
        return {"brief": brief, "author": f"llm:{describe_llm(llm)}", "strategist_attempts": attempts,
                "trace": [_step(cid, AGENT_STRATEGIST, f"Drafted brief with LLM (attempt {attempts})",
                                brief.get("headline", ""), output=brief, duration_ms=t["ms"])]}

    def grounding_guard(state: CaseState) -> dict:
        cid, brief = state["case_id"], state["brief"]
        evidence = _compact_evidence(state)
        evidence["sql_full"] = state.get("sql_evidence", [])
        result = check_grounding(_brief_texts(brief), evidence)
        author = state.get("author", "deterministic")
        if not result["passed"] and author.startswith("llm") and state.get("strategist_attempts", 0) < MAX_STRATEGIST_ATTEMPTS:
            fb = ("GROUNDING GUARD REJECTED YOUR DRAFT. These values are not in the evidence: "
                  f"{result['unsupported_numbers'] + result['unsupported_dates']}. Rewrite using only numbers and "
                  "dates that appear verbatim in the evidence.")
            return {"grounding": result, "grounding_feedback": fb,
                    "trace": [_step(cid, AGENT_GUARD, "Rejected draft", fb, output=result, status="warning")]}
        steps = []
        if not result["passed"] and author.startswith("llm"):
            analytics = state["intent"] != "sku_investigation"
            ev = state.get("sql_evidence", [])
            brief = (template_analytics_answer(ev[-1]["name"], ev[-1]["rows"]) if analytics and ev
                     else template_brief(state)).model_dump()
            result = {**check_grounding(_brief_texts(brief), evidence), "llm_draft_replaced": True}
            author = "deterministic"
            steps.append(_step(cid, AGENT_GUARD, "LLM draft replaced",
                               "Second draft still contained unsupported numbers; using the rule-based brief.",
                               status="warning"))
        steps.append(_step(cid, AGENT_GUARD, "Verified numbers against evidence" if result["passed"] else
                           "Unverified values remain",
                           f"{result['numbers_checked']} number(s) checked; unsupported: "
                           f"{result['unsupported_numbers'] + result['unsupported_dates'] or 'none'}",
                           output=result, status="ok" if result["passed"] else "warning"))
        rep = state.get("ml_evidence", {}).get("replenishment", {})
        case = CaseResult(case_id=cid, intent=state["intent"], question=state["case_question"],
                          sku_id=state.get("sku_id"), dc_id=state.get("dc_id"), sku_name=rep.get("sku_name"),
                          risk_class=(state.get("risk_context") or {}).get("risk_class"),
                          brief=brief, brief_markdown=render_brief(brief), evidence=_preview(evidence, 60000),
                          grounding=result, author=author)
        return {"grounding": result, "brief": brief, "author": author, "cases": [case], "trace": steps}

    def route_after_guard(state: CaseState) -> str:
        return "strategist" if state.get("grounding_feedback") and not state.get("cases") else END

    g = StateGraph(CaseState, output_schema=CaseOutput)
    g.add_node("data_analyst", data_analyst)
    g.add_node("ml_diagnostics", ml_diagnostics)
    g.add_node("strategist", strategist)
    g.add_node("grounding_guard", grounding_guard)
    g.add_edge(START, "data_analyst")
    g.add_conditional_edges("data_analyst", route_after_analyst, ["data_analyst", "ml_diagnostics", "strategist"])
    g.add_edge("ml_diagnostics", "strategist")
    g.add_edge("strategist", "grounding_guard")
    g.add_conditional_edges("grounding_guard", route_after_guard, ["strategist", END])
    return g.compile()


# --------------------------------------------------------------------------- #
# Parent graph
# --------------------------------------------------------------------------- #
def _case_payload(question: str, intent: str, sku_id=None, dc_id=None, risk_context=None) -> dict:
    return {"case_id": uuid.uuid4().hex[:6], "case_question": question, "intent": intent, "sku_id": sku_id,
            "dc_id": dc_id, "risk_context": risk_context, "sql_attempts": 0, "strategist_attempts": 0}


def build_graph(llm: Any = "default"):
    """Compile the copilot graph. ``llm=None`` forces deterministic mode; default reads .env."""
    llm = get_llm() if llm == "default" else llm
    case_graph = build_case_graph(llm)

    def intake(state: AgentState) -> dict:
        if state.get("mode") == "triage":
            return {"intent": "triage", "trace": [_step("main", "Intake", "Morning triage requested",
                                                        "Scan all SKU x DC series and investigate the top risks")]}
        q = state.get("question", "").strip()
        with _timer() as t:
            ent = T.lookup_sku_and_dc.invoke({"text": q})
        sku, dc, cands = ent["sku_id"], ent["dc_id"], ent["sku_candidates"]
        note = ""
        if (sku or cands) and not (sku and dc):
            # Fill the missing dimension with the riskiest match from the triage ranking.
            ranking = T.ml.triage_scan()
            pool = ranking[ranking.sku_id.isin([sku] if sku else cands)]
            if dc:
                pool = pool[pool.dc_id == dc]
            if not pool.empty and (sku or len(set(pool.sku_id)) <= 3):
                top = pool.iloc[0]
                note = (f"Resolved to the highest-risk match: {top.sku_id} @ {top.dc_id}"
                        f" (candidates: {', '.join(cands) if not sku else top.sku_id})")
                sku, dc = top.sku_id, top.dc_id
        if sku and dc:
            intent, detail = "sku_investigation", note or f"Resolved {sku} @ {dc}"
        elif llm is not None or T.match_analytics_template(q):
            intent, detail = "analytics_question", "No single SKU/DC: routing to text-to-SQL"
        else:
            intent, detail = "clarify", "Could not resolve a product or DC, and no LLM is configured"
        ent.update(sku_id=sku, dc_id=dc)
        return {"intent": intent, "entities": ent,
                "trace": [_step("main", "Intake", f"Intent: {intent.replace('_', ' ')}", detail,
                                tool="lookup_sku_and_dc", output=ent, duration_ms=t["ms"])]}

    def route_intake(state: AgentState):
        intent = state["intent"]
        if intent == "triage":
            return "triage_scan"
        if intent == "clarify":
            return "briefing"
        ent = state["entities"]
        return [Send("investigate", _case_payload(state["question"], intent, ent.get("sku_id"), ent.get("dc_id")))]

    def triage_scan(state: AgentState) -> dict:
        with _timer() as t:
            scan = T.run_triage_scan.invoke({"top_n": 100})
        n = state.get("max_cases") or DEFAULT_TRIAGE_CASES
        risky = [r for r in scan["top"] if r["risk_class"] != "HEALTHY"]
        picked = [r for r in risky if r["risk_class"] != "OVERSTOCK"][: max(n - 1, 1)]
        over = [r for r in risky if r["risk_class"] == "OVERSTOCK"][:1]
        picked = (picked + over) if n > 1 else picked
        if len(picked) < n:
            picked += [r for r in risky if r not in picked][: n - len(picked)]
        scan["selected"] = picked[:n]
        detail = (f"{scan['series_scanned']} series scanned: "
                  + ", ".join(f"{v} {k.lower().replace('_', ' ')}" for k, v in scan["risk_class_counts"].items())
                  + f". Investigating {len(scan['selected'])}.")
        return {"triage": scan, "trace": [_step("main", AGENT_ML, "Triage scan (forecast + anomaly + stock maths)",
                                                detail, tool="run_triage_scan", output=_preview(scan),
                                                duration_ms=t["ms"])]}

    def fan_out(state: AgentState):
        sends = []
        for r in state["triage"]["selected"]:
            q = (f"Morning triage alert: {r['risk_class']} for {r['sku_name']} ({r['sku_id']}) at {r['dc_id']}. "
                 f"Diagnose the root cause and recommend replenishment actions.")
            sends.append(Send("investigate", _case_payload(q, "sku_investigation", r["sku_id"], r["dc_id"], r)))
        return sends or "briefing"

    def briefing(state: AgentState) -> dict:
        cases = state.get("cases", [])
        if state.get("intent") == "clarify":
            report = ("I couldn't identify a product and distribution centre in that question. Try e.g. "
                      "*\"Why is Ariel Pods 40ct at risk in Bucharest?\"*, ask about lost sales by category, fill "
                      "rate, promo uplift, supplier delays or channel mix, or run the morning triage. Configure an "
                      "LLM in `.env` for free-form analytics.")
        elif state.get("intent") == "triage":
            tri = state["triage"]
            order = {r["sku_id"] + r["dc_id"]: i for i, r in enumerate(tri["selected"])}
            cases = sorted(cases, key=lambda c: order.get((c.get("sku_id") or "") + (c.get("dc_id") or ""), 99))
            counts = tri["risk_class_counts"]
            lines = [f"## Morning triage, as of {tri['as_of_date']}", "",
                     f"**{counts.get('CRITICAL_STOCKOUT', 0)} critical stockouts**, "
                     f"{counts.get('STOCKOUT_RISK', 0)} stockout risks, {counts.get('OVERSTOCK', 0)} overstock and "
                     f"{counts.get('WATCH_P90', 0)} P90 watch items across {tri['series_scanned']} SKU x DC series. "
                     f"Revenue at risk before replenishment can land: **{tri['total_revenue_at_risk_ron']:,.0f} RON**; "
                     f"excess stock value: **{tri['total_excess_value_ron']:,.0f} RON**.", "",
                     "| SKU | DC | Risk | Days to stockout | Revenue at risk (RON) | Excess (RON) |",
                     "|---|---|---|---|---|---|"]
            for r in tri["top"][:10]:
                dts = "" if r["days_until_stockout"] is None else f"{r['days_until_stockout']:.0f}"
                lines.append(f"| {r['sku_name']} | {r['dc_id']} | {r['risk_class']} | {dts} | "
                             f"{r['revenue_at_risk_ron']:,.0f} | {r['excess_value_ron']:,.0f} |")
            for c in cases:
                lines += ["", "---", "", f"#### {c.get('sku_name') or c.get('sku_id')} @ {c.get('dc_id')}"
                          f" ({c.get('risk_class')})", "", c["brief_markdown"]]
            report = "\n".join(lines)
        else:
            report = cases[0]["brief_markdown"] if cases else "No result."
        return {"final_report": report, "trace": [_step("main", "Briefing", "Compiled report",
                                                        f"{len(cases)} investigation(s)")]}

    g = StateGraph(AgentState)
    g.add_node("intake", intake)
    g.add_node("triage_scan", triage_scan)
    g.add_node("investigate", case_graph)
    g.add_node("briefing", briefing)
    g.add_edge(START, "intake")
    g.add_conditional_edges("intake", route_intake, ["triage_scan", "investigate", "briefing"])
    g.add_conditional_edges("triage_scan", fan_out, ["investigate", "briefing"])
    g.add_edge("investigate", "briefing")
    g.add_edge("briefing", END)
    return g.compile()


# --------------------------------------------------------------------------- #
# Entry points
# --------------------------------------------------------------------------- #
def _inputs(question: str | None, triage: bool, max_cases: int) -> dict:
    return {"question": question or "", "mode": "triage" if triage else "question", "max_cases": max_cases,
            "trace": [], "cases": []}


def stream_copilot(question: str | None = None, triage: bool = False, max_cases: int = DEFAULT_TRIAGE_CASES,
                   graph=None) -> Iterator[tuple[str, Any]]:
    """Yield ("step", TraceStep) as agents work, then ("final", AgentState)."""
    graph = graph or build_graph()
    final = None
    for ns, mode, chunk in graph.stream(_inputs(question, triage, max_cases),
                                        stream_mode=["updates", "values"], subgraphs=True):
        if mode == "values" and not ns:
            final = chunk
        elif mode == "updates":
            for node, update in chunk.items():
                if not ns and node == "investigate":
                    continue  # already streamed from inside the subgraph
                for step in (update or {}).get("trace", []):
                    yield "step", step
    yield "final", final


def run_copilot(question: str | None = None, triage: bool = False, max_cases: int = DEFAULT_TRIAGE_CASES,
                graph=None) -> dict:
    graph = graph or build_graph()
    return graph.invoke(_inputs(question, triage, max_cases))


if __name__ == "__main__":
    import sys
    q = " ".join(sys.argv[1:]) or None
    for kind, payload in stream_copilot(q, triage=q is None):
        if kind == "step":
            print(f"[{payload['agent']}] {payload['action']}: {payload.get('detail', '')}")
        else:
            print("\n" + payload["final_report"])
