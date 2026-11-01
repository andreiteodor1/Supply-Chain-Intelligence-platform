"""LangChain tools: the only way agents touch numbers.

Every figure an agent quotes is produced here, either by a read-only SQL query
against DuckDB or by the scikit-learn models and the deterministic inventory
maths in ``ml_pipeline``. The LLM narrates; it never calculates.

Tools return plain dicts so they serialise cleanly into the agent trace and
into the evidence set used by the grounding check.
"""
from __future__ import annotations

import re
import unicodedata
from functools import lru_cache

import pandas as pd
from langchain_core.tools import tool

from src import ml_pipeline as ml
from src.db import UnsafeQueryError, as_of_date, query_df, run_query, schema_description

# --------------------------------------------------------------------------- #
# Entity resolution (deterministic: no LLM needed to find "Ariel Pods in Bucharest")
# --------------------------------------------------------------------------- #
DC_ALIASES = {
    "DC-BUC": ["bucharest", "bucuresti", "buc", "muntenia"],
    "DC-CLJ": ["cluj", "cluj-napoca", "cluj napoca", "clj", "transilvania", "transylvania"],
    "DC-TIM": ["timisoara", "tim", "banat"],
    "DC-IAS": ["iasi", "ias", "moldova"],
}
_STOP = {"the", "and", "in", "of", "for", "all", "1", "with", "pro", "v", "ml", "l", "kg", "ct", "s"}


def _norm(text: str) -> str:
    text = unicodedata.normalize("NFKD", text).encode("ascii", "ignore").decode().lower()
    return re.sub(r"[^a-z0-9&.\- ]+", " ", text)


def _tokens(text: str) -> set[str]:
    return {t for t in re.split(r"[\s\-]+", _norm(text)) if t and t not in _STOP}


@lru_cache(maxsize=1)
def _sku_master() -> pd.DataFrame:
    return query_df("SELECT sku_id, sku_name, brand, category, pack_size FROM dim_sku ORDER BY sku_id")


def resolve_entities(text: str) -> dict:
    """Find SKU and DC mentions in free text. Returns best matches plus ambiguity info."""
    norm = _norm(text)
    toks = _tokens(text)
    dc_id = None
    for dc, aliases in DC_ALIASES.items():
        if dc.lower() in norm or any(re.search(rf"\b{re.escape(a)}\b", norm) for a in aliases):
            dc_id = dc
            break

    skus = _sku_master()
    scored = []
    for r in skus.itertuples():
        if r.sku_id.lower() in norm:
            scored.append((100, r.sku_id))
            continue
        name_toks = _tokens(r.sku_name)
        brand_toks = _tokens(r.brand)
        brand_hit = bool(brand_toks) and brand_toks <= toks
        overlap = len((name_toks - brand_toks) & toks)
        # Pack-size style tokens like "40ct", "76ct", "4kg" are highly distinctive
        size_hit = any(t in toks for t in name_toks if re.match(r"^\d+[a-z]*$", t))
        score = 3 * brand_hit + 2 * overlap + 2 * size_hit
        if score >= 2 and (brand_hit or overlap >= 2):
            scored.append((score, r.sku_id))
    scored.sort(reverse=True)
    sku_id = scored[0][1] if scored and (len(scored) == 1 or scored[0][0] > scored[1][0]) else None
    candidates = [s for _, s in scored[:5]]
    return {"sku_id": sku_id, "dc_id": dc_id, "sku_candidates": candidates}


def _check_ids(sku_id: str, dc_id: str | None = None) -> None:
    skus = set(_sku_master().sku_id)
    if sku_id not in skus:
        raise ValueError(f"Unknown sku_id '{sku_id}'. Valid: {sorted(skus)}")
    if dc_id is not None and dc_id not in DC_ALIASES:
        raise ValueError(f"Unknown dc_id '{dc_id}'. Valid: {sorted(DC_ALIASES)}")


def _records(df: pd.DataFrame, limit: int = 60) -> list[dict]:
    df = df.head(limit).copy()
    for c in df.columns:
        if pd.api.types.is_datetime64_any_dtype(df[c]):
            df[c] = df[c].dt.strftime("%Y-%m-%d")
    df = df.astype(object).where(df.notna(), None)
    out = []
    for row in df.to_dict("records"):
        out.append({k: (round(v, 4) if isinstance(v, float) else (v.isoformat() if hasattr(v, "isoformat") else v))
                    for k, v in row.items()})
    return out


# --------------------------------------------------------------------------- #
# SQL playbook used by the Data Analyst agent (ids are whitelisted, then the
# query still goes through the read-only validator like any LLM query would)
# --------------------------------------------------------------------------- #
def sql_playbook(sku_id: str, dc_id: str) -> list[tuple[str, str]]:
    _check_ids(sku_id, dc_id)
    asof = as_of_date().date().isoformat()
    p = f"sku_id = '{sku_id}' AND dc_id = '{dc_id}'"
    return [
        ("demand_trend", f"""
SELECT
  ROUND(AVG(units_demanded) FILTER (WHERE date > DATE '{asof}' - 7), 1)                         AS avg_daily_demand_last_7d,
  ROUND(AVG(units_demanded) FILTER (WHERE date <= DATE '{asof}' - 7 AND date > DATE '{asof}' - 35), 1) AS avg_daily_demand_prior_28d,
  ROUND(AVG(units_demanded) FILTER (WHERE date > DATE '{asof}' - 365 - 28 AND date <= DATE '{asof}' - 365), 1) AS avg_daily_demand_same_period_ly,
  SUM(lost_sales_units) FILTER (WHERE date > DATE '{asof}' - 28)                                AS lost_sales_units_28d,
  ROUND(SUM(lost_sales_units * net_price_ron) FILTER (WHERE date > DATE '{asof}' - 28), 0)      AS lost_revenue_ron_28d,
  ROUND(SUM(units_sold) FILTER (WHERE date > DATE '{asof}' - 28) * 1.0
        / NULLIF(SUM(units_demanded) FILTER (WHERE date > DATE '{asof}' - 28), 0), 3)           AS fill_rate_28d,
  SUM(CASE WHEN lost_sales_units > 0 AND date > DATE '{asof}' - 28 THEN 1 ELSE 0 END)           AS stockout_days_28d,
  ROUND(SUM(net_revenue_ron) FILTER (WHERE date > DATE '{asof}' - 28), 0)                      AS net_revenue_ron_28d
FROM fact_daily_sales WHERE {p}"""),
        ("inventory_position", f"""
SELECT i.on_hand_close AS on_hand_units, i.on_order_units, i.days_of_cover AS days_of_cover_trailing,
       p.reorder_point_units, p.safety_stock_units, p.order_qty_units, p.planned_daily_demand,
       s.supplier_id, s.supplier_name, s.lead_time_mean_days, s.on_time_reliability
FROM fact_inventory_daily i
JOIN dim_inventory_policy p USING (sku_id, dc_id)
JOIN dim_sku k USING (sku_id)
JOIN dim_supplier s ON s.supplier_id = k.supplier_id
WHERE i.sku_id = '{sku_id}' AND i.dc_id = '{dc_id}' AND i.date = DATE '{asof}'"""),
        ("open_purchase_orders", f"""
SELECT po_id, po_type, order_date, expected_arrival_date, revised_eta_date, quantity_units, status, delay_days
FROM fact_purchase_orders WHERE {p} AND status <> 'RECEIVED' ORDER BY revised_eta_date"""),
        ("promo_calendar", f"""
SELECT promo_id, start_date, end_date, promo_depth_pct, mechanic, status
FROM fact_promo_calendar
WHERE {p} AND end_date >= DATE '{asof}' - 14 ORDER BY start_date"""),
        ("active_disruptions", f"""
SELECT d.event_id, d.disruption_type, d.severity, d.added_lead_time_days, d.start_date, d.end_date, d.description
FROM fact_logistics_disruptions d, dim_sku k
WHERE k.sku_id = '{sku_id}' AND d.status = 'ACTIVE'
  AND (d.supplier_id IS NULL OR d.supplier_id = k.supplier_id)
  AND (d.dc_id IS NULL OR d.dc_id = '{dc_id}')
  AND (d.sku_id IS NULL OR d.sku_id = '{sku_id}')"""),
        ("network_stock", f"""
SELECT i.dc_id, i.on_hand_close AS on_hand_units, i.on_order_units, i.days_of_cover AS days_of_cover_trailing
FROM fact_inventory_daily i
WHERE i.sku_id = '{sku_id}' AND i.date = DATE '{asof}' ORDER BY i.dc_id"""),
        ("competitor_pricing", f"""
SELECT ROUND(AVG(competitor_price_index) FILTER (WHERE date > DATE '{asof}' - 7), 3)  AS competitor_index_last_7d,
       ROUND(AVG(competitor_price_index) FILTER (WHERE date <= DATE '{asof}' - 28
                                                  AND date > DATE '{asof}' - 56), 3) AS competitor_index_4_8_weeks_ago
FROM fact_daily_sales WHERE {p}"""),
    ]


def run_playbook(sku_id: str, dc_id: str) -> list[dict]:
    out = []
    for name, sql in sql_playbook(sku_id, dc_id):
        sql = re.sub(r"\n\s*\n", "\n", sql).strip()
        res = run_query(sql)
        out.append({"name": name, "sql": sql, "rows": _records(res.df), "row_count": res.row_count})
    return out


# Template answers for common analytics questions when no LLM is configured.
ANALYTICS_TEMPLATES = [
    (("lost", "category"), "Lost sales by category, last 28 days", """
SELECT k.category, SUM(s.lost_sales_units) AS lost_units, ROUND(SUM(s.lost_sales_units * s.net_price_ron), 0) AS lost_revenue_ron,
       ROUND(SUM(s.units_sold) * 1.0 / SUM(s.units_demanded), 3) AS fill_rate
FROM fact_daily_sales s JOIN dim_sku k USING (sku_id)
WHERE s.date > (SELECT MAX(date) FROM fact_daily_sales) - 28
GROUP BY 1 ORDER BY lost_revenue_ron DESC"""),
    (("fill", "rate"), "Fill rate by distribution centre, last 28 days", """
SELECT d.city, ROUND(SUM(s.units_sold) * 1.0 / SUM(s.units_demanded), 3) AS fill_rate,
       SUM(s.lost_sales_units) AS lost_units
FROM fact_daily_sales s JOIN dim_dc d USING (dc_id)
WHERE s.date > (SELECT MAX(date) FROM fact_daily_sales) - 28
GROUP BY 1 ORDER BY fill_rate"""),
    (("revenue", "brand"), "Net revenue by brand, last 28 days vs prior 28 days", """
SELECT k.brand,
       ROUND(SUM(s.net_revenue_ron) FILTER (WHERE s.date > m.d - 28), 0) AS revenue_last_28d,
       ROUND(SUM(s.net_revenue_ron) FILTER (WHERE s.date <= m.d - 28 AND s.date > m.d - 56), 0) AS revenue_prior_28d
FROM fact_daily_sales s JOIN dim_sku k USING (sku_id), (SELECT MAX(date) AS d FROM fact_daily_sales) m
GROUP BY 1 ORDER BY revenue_last_28d DESC"""),
    (("promo",), "Average promo uplift by category (promo vs non-promo days, last 365 days)", """
SELECT k.category,
       ROUND(AVG(s.units_demanded) FILTER (WHERE s.promo_flag = 1) / AVG(s.units_demanded) FILTER (WHERE s.promo_flag = 0), 2) AS promo_uplift_x,
       ROUND(AVG(s.promo_depth_pct) FILTER (WHERE s.promo_flag = 1), 2) AS avg_depth
FROM fact_daily_sales s JOIN dim_sku k USING (sku_id)
WHERE s.date > (SELECT MAX(date) FROM fact_daily_sales) - 365
GROUP BY 1 ORDER BY promo_uplift_x DESC"""),
    (("delay",), "Purchase-order delays by supplier (last 180 days)", """
SELECT p.supplier_id, s.supplier_name, COUNT(*) AS pos, ROUND(AVG(p.delay_days), 2) AS avg_delay_days,
       ROUND(AVG(CASE WHEN p.delay_days > 0 THEN 1 ELSE 0 END), 3) AS late_share
FROM fact_purchase_orders p JOIN dim_supplier s USING (supplier_id)
WHERE p.order_date > (SELECT MAX(date) FROM fact_daily_sales) - 180
GROUP BY 1, 2 ORDER BY avg_delay_days DESC"""),
    (("channel",), "Channel mix, last 8 weeks", """
SELECT channel, SUM(units_sold) AS units, ROUND(SUM(net_revenue_ron), 0) AS revenue_ron,
       ROUND(SUM(net_revenue_ron) * 1.0 / SUM(SUM(net_revenue_ron)) OVER (), 3) AS revenue_share
FROM fact_channel_sales_weekly
WHERE week_start > (SELECT MAX(week_start) FROM fact_channel_sales_weekly) - 56
GROUP BY 1 ORDER BY revenue_ron DESC"""),
    (("stockout",), "SKU x DC with the most stockout days, last 28 days", """
SELECT s.sku_id, k.sku_name, s.dc_id, SUM(CASE WHEN s.lost_sales_units > 0 THEN 1 ELSE 0 END) AS stockout_days,
       SUM(s.lost_sales_units) AS lost_units
FROM fact_daily_sales s JOIN dim_sku k USING (sku_id)
WHERE s.date > (SELECT MAX(date) FROM fact_daily_sales) - 28
GROUP BY 1, 2, 3 HAVING stockout_days > 0 ORDER BY stockout_days DESC, lost_units DESC LIMIT 15"""),
]


def match_analytics_template(question: str) -> tuple[str, str] | None:
    q = _norm(question)
    best, best_hits = None, 0
    for keys, title, sql in ANALYTICS_TEMPLATES:
        hits = sum(k in q for k in keys)
        if hits == len(keys) and hits > best_hits:
            best, best_hits = (title, sql.strip()), hits
    return best


# --------------------------------------------------------------------------- #
# @tool definitions
# --------------------------------------------------------------------------- #
@tool
def lookup_sku_and_dc(text: str) -> dict:
    """Resolve product and distribution-centre mentions in free text to sku_id / dc_id codes."""
    return resolve_entities(text)


@tool
def get_database_schema() -> str:
    """Return the warehouse schema (tables, columns, types and business descriptions)."""
    return schema_description()


@tool
def run_sql_query(sql: str) -> dict:
    """Run ONE read-only DuckDB SELECT query against the CPG warehouse and return up to 500 rows.
    Writes, DDL, PRAGMA/SET, file readers and multiple statements are rejected."""
    try:
        res = run_query(sql)
    except UnsafeQueryError as exc:
        return {"ok": False, "error": f"Rejected by read-only guard: {exc}"}
    except Exception as exc:  # surfaced to the LLM for self-correction
        return {"ok": False, "error": f"{type(exc).__name__}: {exc}"}
    return {"ok": True, "sql": res.sql, "row_count": res.row_count, "truncated": res.truncated,
            "rows": _records(res.df, limit=100)}


@tool
def get_sku_dc_facts(sku_id: str, dc_id: str) -> dict:
    """Run the standard diagnostic SQL playbook (demand trend, inventory, open POs, promos,
    disruptions, network stock, competitor pricing) for one SKU at one DC."""
    return {q["name"]: q["rows"] for q in run_playbook(sku_id, dc_id)}


@tool
def forecast_demand(sku_id: str, dc_id: str) -> dict:
    """14-day daily demand forecast (mean and P90) from the HistGradientBoosting model."""
    _check_ids(sku_id, dc_id)
    fc = ml.forecast_series(sku_id, dc_id)
    return {
        "sku_id": sku_id, "dc_id": dc_id, "model": "HistGradientBoostingRegressor (direct multi-horizon)",
        "forecast_14d_units": round(float(fc.forecast_units.sum())),
        "forecast_14d_p90_units": round(float(fc.forecast_p90_units.sum())),
        "avg_daily_forecast": round(float(fc.forecast_units.mean()), 1),
        "promo_days_in_horizon": int((fc.promo_depth_pct > 0).sum()),
        "daily": _records(fc[["date", "forecast_units", "forecast_p90_units", "promo_depth_pct"]]),
    }


@tool
def project_stockout(sku_id: str, dc_id: str) -> dict:
    """Project on-hand inventory day by day using the ML forecast and open-PO ETAs; returns the
    expected and P90 stockout dates, whether a PO placed today can land in time, and revenue at risk."""
    _check_ids(sku_id, dc_id)
    return ml.project_stockout(sku_id, dc_id)


@tool
def detect_anomalies(sku_id: str | None = None, dc_id: str | None = None, lookback_days: int = 14) -> dict:
    """IsolationForest anomaly flags (velocity spikes/drops, unexplained inventory drops) in the lookback window."""
    if sku_id:
        _check_ids(sku_id, dc_id)
    df = ml.score_anomalies(lookback_days=lookback_days, sku_id=sku_id, dc_id=dc_id)
    cols = ["date", "sku_id", "dc_id", "anomaly_type", "anomaly_score", "units_demanded", "baseline_28d",
            "vel_ratio28", "inv_residual_units", "promo_depth"]
    return {"lookback_days": lookback_days, "count": int(len(df)), "anomalies": _records(df[cols].round({c: 3 for c in cols if c != "date"}))}


@tool
def recommend_replenishment(sku_id: str, dc_id: str) -> dict:
    """Order-up-to purchase recommendation sized on the ML forecast (lead time incl. active disruptions,
    review period, P90-based safety stock, case-pack rounding) plus inter-DC transfer options."""
    _check_ids(sku_id, dc_id)
    return ml.recommend_replenishment(sku_id, dc_id)


@tool
def run_triage_scan(top_n: int = 10) -> dict:
    """Rank every SKU x DC by business risk (stockout before replenishment, overstock, anomalies)."""
    df = ml.triage_scan()
    cols = ["sku_id", "sku_name", "dc_id", "risk_class", "risk_score", "on_hand_close", "forecast_14d",
            "days_of_cover", "days_until_stockout", "effective_lead_time_days", "lost_units_before_new_po",
            "revenue_at_risk_ron", "excess_units", "excess_value_ron", "anomalies_7d"]
    counts = df.risk_class.value_counts().to_dict()
    return {"as_of_date": as_of_date().date().isoformat(), "series_scanned": int(len(df)),
            "risk_class_counts": {k: int(v) for k, v in counts.items()},
            "total_revenue_at_risk_ron": round(float(df.revenue_at_risk_ron.sum()), 2),
            "total_excess_value_ron": round(float(df.excess_value_ron.sum()), 2),
            "top": _records(df[cols].round(1), limit=top_n)}


ALL_TOOLS = [lookup_sku_and_dc, get_database_schema, run_sql_query, get_sku_dc_facts, forecast_demand,
             project_stockout, detect_anomalies, recommend_replenishment, run_triage_scan]
