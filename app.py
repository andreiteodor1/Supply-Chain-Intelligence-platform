"""CPG Pulse: Streamlit front end.

Tabs
  1. Executive KPI & ML Dashboard - KPIs, stockout-risk grid, forecast drill-down, anomalies
  2. Agentic Copilot - autonomous morning triage and natural-language questions with a live agent trace
  3. Model & Data Lab - model card, guarded SQL explorer, schema and graph topology

Run:  streamlit run app.py
"""
from __future__ import annotations

import json
import time

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

st.set_page_config(page_title="CPG Pulse", page_icon=":material/monitoring:", layout="wide")

from src.db import UnsafeQueryError, db_path, query_df, reset_connection, run_query, schema_description  # noqa: E402

# --------------------------------------------------------------------------- #
# Palette (validated reference palette; light/dark steps chosen per mode)
# --------------------------------------------------------------------------- #
LIGHT = dict(surface="#fcfcfb", ink="#0b0b0b", ink2="#52514e", muted="#898781", grid="#e1e0d9", axis="#c3c2b7",
             series=["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#6250d6", "#e34948"],
             band="rgba(235,104,52,0.16)", promo="rgba(137,135,129,0.12)")
DARK = dict(surface="#1a1a19", ink="#f0efec", ink2="#c3c2b7", muted="#898781", grid="#2c2c2a", axis="#383835",
            series=["#3987e5", "#d95926", "#199e70", "#c98500", "#d55181", "#008300", "#9085e9", "#e66767"],
            band="rgba(217,89,38,0.22)", promo="rgba(137,135,129,0.18)")
STATUS = {"CRITICAL_STOCKOUT": "#d03b3b", "STOCKOUT_RISK": "#ec835a", "WATCH_P90": "#fab219",
          "HEALTHY": "#0ca30c", "OVERSTOCK": "#2a78d6", "ANOMALY": "#898781"}
STATUS_LABEL = {"CRITICAL_STOCKOUT": "Critical", "STOCKOUT_RISK": "Stockout risk", "WATCH_P90": "Watch (P90)",
                "HEALTHY": "Healthy", "OVERSTOCK": "Overstock", "ANOMALY": "Anomaly"}
STATUS_ICON = {"CRITICAL_STOCKOUT": "⛔", "STOCKOUT_RISK": "▲", "WATCH_P90": "◆", "HEALTHY": "✓",
               "OVERSTOCK": "■", "ANOMALY": "●"}


def pal() -> dict:
    try:
        return DARK if st.context.theme.type == "dark" else LIGHT
    except Exception:
        return LIGHT


def style(fig: go.Figure, height: int = 360, **kw) -> go.Figure:
    p = pal()
    kw.setdefault("margin", dict(l=8, r=8, t=36, b=8))
    fig.update_layout(
        height=height, paper_bgcolor=p["surface"], plot_bgcolor=p["surface"],
        font=dict(family='system-ui, -apple-system, "Segoe UI", sans-serif', color=p["ink2"], size=12),
        title_font=dict(color=p["ink"], size=14), hoverlabel=dict(font_size=12),
        legend=dict(orientation="h", yanchor="bottom", y=1.0, xanchor="right", x=1, title=None,
                    font=dict(color=p["ink2"])), **kw)
    fig.update_xaxes(gridcolor=p["grid"], linecolor=p["axis"], tickfont=dict(color=p["muted"]), zeroline=False)
    fig.update_yaxes(gridcolor=p["grid"], linecolor=p["axis"], tickfont=dict(color=p["muted"]), zeroline=False)
    return fig


# --------------------------------------------------------------------------- #
# Bootstrap: build the warehouse and models on first run
# --------------------------------------------------------------------------- #
@st.cache_resource(show_spinner=False)
def bootstrap() -> dict:
    from src import ml_pipeline as ml
    if not db_path().exists():
        from data.generate_cpg_data import main as generate
        with st.spinner("First run: generating the synthetic FMCG warehouse (about 20 s)..."):
            generate([])
        reset_connection()
    if not ml.models_ready():
        with st.spinner("First run: training the demand forecaster and anomaly detector (about 2 min)..."):
            ml.train_all(verbose=False)
    return ml.model_card()


@st.cache_resource(show_spinner=False)
def copilot_graph():
    from src.agents.graph import build_graph
    return build_graph()


@st.cache_data(show_spinner=False)
def load_triage() -> pd.DataFrame:
    from src import ml_pipeline as ml
    return ml.triage_scan()


@st.cache_data(show_spinner=False)
def load_master() -> tuple[pd.DataFrame, pd.DataFrame, str]:
    skus = query_df("SELECT sku_id, sku_name, brand, category FROM dim_sku ORDER BY category, sku_name")
    dcs = query_df("SELECT dc_id, city FROM dim_dc ORDER BY dc_id")
    asof = query_df("SELECT value FROM meta_dataset_info WHERE key = 'as_of_date'").iloc[0, 0]
    return skus, dcs, asof


@st.cache_data(show_spinner=False)
def load_kpis(categories: tuple[str, ...], dcs: tuple[str, ...]) -> pd.DataFrame:
    return query_df(f"""
        WITH m AS (SELECT MAX(date) AS d FROM fact_daily_sales),
        f AS (SELECT s.*, CASE WHEN s.date > m.d - 28 THEN 'cur' WHEN s.date > m.d - 56 THEN 'prev' END AS period
              FROM fact_daily_sales s JOIN dim_sku k USING (sku_id), m
              WHERE k.category IN ({",".join("?" * len(categories))}) AND s.dc_id IN ({",".join("?" * len(dcs))}))
        SELECT period, SUM(net_revenue_ron) AS revenue, SUM(gross_margin_ron) AS margin,
               SUM(units_sold) AS units, SUM(units_sold) * 1.0 / NULLIF(SUM(units_demanded), 0) AS fill_rate,
               SUM(lost_sales_units * net_price_ron) AS lost_revenue
        FROM f WHERE period IS NOT NULL GROUP BY period""", [*categories, *dcs]).set_index("period")


@st.cache_data(show_spinner=False)
def load_category_weekly(dcs: tuple[str, ...]) -> pd.DataFrame:
    return query_df(f"""
        SELECT date_trunc('week', s.date) AS week, k.category, SUM(s.net_revenue_ron) AS revenue
        FROM fact_daily_sales s JOIN dim_sku k USING (sku_id)
        WHERE s.dc_id IN ({",".join("?" * len(dcs))})
          AND date_trunc('week', s.date) > date_trunc('week', (SELECT MAX(date) FROM fact_daily_sales) - 364)
          AND date_trunc('week', s.date) < date_trunc('week', (SELECT MAX(date) FROM fact_daily_sales))
        GROUP BY 1, 2 ORDER BY 1""", list(dcs))


@st.cache_data(show_spinner=False)
def load_history(sku_id: str, dc_id: str, days: int = 120) -> pd.DataFrame:
    return query_df("""
        SELECT s.date, s.units_demanded, s.units_sold, s.promo_depth_pct, i.on_hand_close
        FROM fact_daily_sales s JOIN fact_inventory_daily i USING (date, sku_id, dc_id)
        WHERE s.sku_id = ? AND s.dc_id = ? AND s.date > (SELECT MAX(date) FROM fact_daily_sales) - ?
        ORDER BY s.date""", [sku_id, dc_id, days])


@st.cache_data(show_spinner=False)
def load_projection(sku_id: str, dc_id: str) -> dict:
    from src import ml_pipeline as ml
    return ml.recommend_replenishment(sku_id, dc_id) | {"projection": ml.project_stockout(sku_id, dc_id)}


@st.cache_data(show_spinner=False)
def load_anomalies(days: int) -> pd.DataFrame:
    from src import ml_pipeline as ml
    return ml.score_anomalies(lookback_days=days)


@st.cache_data(show_spinner=False)
def load_policy(sku_id: str, dc_id: str) -> dict:
    return query_df("SELECT * FROM dim_inventory_policy WHERE sku_id = ? AND dc_id = ?", [sku_id, dc_id]).iloc[0].to_dict()


card = bootstrap()
skus, dcs_df, as_of = load_master()
sku_name = dict(zip(skus.sku_id, skus.sku_name))
dc_city = dict(zip(dcs_df.dc_id, dcs_df.city))

# --------------------------------------------------------------------------- #
# Sidebar
# --------------------------------------------------------------------------- #
with st.sidebar:
    st.markdown("### CPG Pulse")
    st.caption("Supply chain & retail demand intelligence")
    st.markdown(f"**Data as of:** {as_of}")
    from src.agents.graph import describe_llm
    llm_desc = describe_llm()
    st.markdown(f"**Agent LLM:** `{llm_desc}`")
    if llm_desc.startswith("deterministic"):
        st.caption("No API key found: agents run in deterministic mode (rule-based narrative, same tools and "
                   "numbers). Set `LLM_PROVIDER` and a key in `.env` to enable LLM reasoning.")
    st.divider()
    sel_cats = st.multiselect("Category", sorted(skus.category.unique()), default=sorted(skus.category.unique()))
    sel_dcs = st.multiselect("Distribution centre", list(dc_city), default=list(dc_city),
                             format_func=lambda d: f"{dc_city[d]} ({d})")
    if not sel_cats or not sel_dcs:
        st.warning("Select at least one category and one DC.")
        st.stop()
    st.divider()
    st.caption("All data is synthetic, generated for this portfolio project. Brand names illustrate the "
               "category structure only; nothing here is real P&G data.")

tab_dash, tab_agent, tab_lab = st.tabs(["Executive KPI & ML Dashboard", "Agentic Copilot & Autonomous Triage",
                                        "Model & Data Lab"])

# =========================================================================== #
# Tab 1 - Dashboard
# =========================================================================== #
with tab_dash:
    triage = load_triage()
    tri = triage[triage.category.isin(sel_cats) & triage.dc_id.isin(sel_dcs)]
    k = load_kpis(tuple(sel_cats), tuple(sel_dcs))
    cur, prev = k.loc["cur"], k.loc["prev"]

    c = st.columns(6)
    c[0].metric("Net revenue, 28d (RON)", f"{cur.revenue / 1e6:,.2f}M", f"{cur.revenue / prev.revenue - 1:+.1%} vs prior 28d")
    c[1].metric("Gross margin, 28d", f"{cur.margin / cur.revenue:.1%}",
                f"{(cur.margin / cur.revenue - prev.margin / prev.revenue) * 100:+.1f} pts")
    c[2].metric("Fill rate, 28d", f"{cur.fill_rate:.1%}", f"{(cur.fill_rate - prev.fill_rate) * 100:+.1f} pts")
    c[3].metric("Lost sales, 28d (RON)", f"{cur.lost_revenue / 1e3:,.0f}k",
                f"{cur.lost_revenue / max(prev.lost_revenue, 1) - 1:+.0%}", delta_color="inverse")
    n_crit = int((tri.risk_class == "CRITICAL_STOCKOUT").sum())
    c[4].metric("Critical stockouts", f"{n_crit}", f"{tri.revenue_at_risk_ron.sum() / 1e3:,.0f}k RON at risk",
                delta_color="off")
    c[5].metric("Excess stock (RON)", f"{tri.excess_value_ron.sum() / 1e3:,.1f}k",
                f"{int((tri.risk_class == 'OVERSTOCK').sum())} SKU x DC overstocked", delta_color="off")

    # ---- Stockout risk grid + priority list -----------------------------
    left, right = st.columns([1.15, 1])
    with left:
        grid = tri.assign(sku_label=tri.sku_id.map(sku_name))
        order_sku = [s for s in skus.sku_id if s in set(grid.sku_id)]
        dcs_sorted = [d for d in dc_city if d in sel_dcs]
        z, text, hover = [], [], []
        codes = list(STATUS)
        for s in order_sku:
            zr, tr, hr = [], [], []
            for d in dcs_sorted:
                r = grid[(grid.sku_id == s) & (grid.dc_id == d)]
                if r.empty:
                    zr.append(None)
                    tr.append("")
                    hr.append("")
                    continue
                r = r.iloc[0]
                zr.append(codes.index(r.risk_class))
                dts = "" if pd.isna(r.days_until_stockout) else f"{r.days_until_stockout:.0f}d"
                tr.append(f"{STATUS_ICON[r.risk_class]} {dts or f'{r.days_of_cover:.0f}c'}")
                hr.append(f"<b>{sku_name[s]}</b> @ {dc_city[d]}<br>{STATUS_LABEL[r.risk_class]}<br>"
                          f"On hand {r.on_hand_close:,.0f} | 14d forecast {r.forecast_14d:,.0f}<br>"
                          f"Days of cover {r.days_of_cover:.1f}<br>"
                          + (f"Stockout in {r.days_until_stockout:.0f} days<br>" if dts else "")
                          + f"Revenue at risk {r.revenue_at_risk_ron:,.0f} RON")
            z.append(zr)
            text.append(tr)
            hover.append(hr)
        n = len(codes)
        p = pal()
        # Healthy cells recede to a neutral so the eye lands on the exceptions.
        cell_fill = {**STATUS, "HEALTHY": p["grid"]}
        scale = []
        for i, code in enumerate(codes):
            scale += [[i / n, cell_fill[code]], [(i + 1) / n, cell_fill[code]]]
        xs, ys = [dc_city[d] for d in dcs_sorted], [sku_name[s] for s in order_sku]
        fig = go.Figure(go.Heatmap(
            z=z, x=xs, y=ys, customdata=hover, hovertemplate="%{customdata}<extra></extra>",
            colorscale=scale, zmin=-0.5, zmax=n - 0.5, showscale=False, xgap=2, ygap=2))
        ink_on = {"HEALTHY": p["ink2"], "WATCH_P90": "#0b0b0b"}
        for yi, row in enumerate(z):
            for xi, code_idx in enumerate(row):
                if code_idx is None:
                    continue
                fig.add_annotation(x=xs[xi], y=ys[yi], text=text[yi][xi], showarrow=False,
                                   font=dict(size=11, color=ink_on.get(codes[code_idx], "#ffffff")))
        fig.update_yaxes(autorange="reversed", showgrid=False)
        fig.update_xaxes(side="top", showgrid=False)
        style(fig, height=110 + 30 * len(order_sku), title=None, margin=dict(l=8, r=8, t=30, b=8))
        st.markdown("##### Stockout risk grid")
        st.caption("Cell = days until projected stockout (d) or days of forecast cover (c). "
                   + "  ".join(f"{STATUS_ICON[c]} {STATUS_LABEL[c]}" for c in codes if c != "ANOMALY"))
        st.plotly_chart(fig, use_container_width=True, config={"displayModeBar": False})
    with right:
        st.markdown("##### Priority list")
        pri = tri[tri.risk_class != "HEALTHY"].head(14)
        st.dataframe(
            pd.DataFrame({
                "Status": pri.risk_class.map(lambda c: f"{STATUS_ICON[c]} {STATUS_LABEL[c]}"),
                "SKU": pri.sku_name, "DC": pri.dc_id.map(dc_city),
                "Days to stockout": pri.days_until_stockout,
                "Lost units before new PO": pri.lost_units_before_new_po,
                "Revenue at risk (RON)": pri.revenue_at_risk_ron,
                "Excess (RON)": pri.excess_value_ron}),
            hide_index=True, use_container_width=True, height=110 + 30 * min(len(order_sku), 14),
            column_config={
                "Days to stockout": st.column_config.NumberColumn(format="%.0f"),
                "Lost units before new PO": st.column_config.NumberColumn(format="%d"),
                "Revenue at risk (RON)": st.column_config.ProgressColumn(
                    format="%.0f", min_value=0, max_value=float(max(pri.revenue_at_risk_ron.max(), 1))),
                "Excess (RON)": st.column_config.NumberColumn(format="%.0f")})

    # ---- Forecast drill-down -------------------------------------------
    st.markdown("##### Demand forecast & inventory projection")
    options = [(r.sku_id, r.dc_id) for r in tri.itertuples()]
    default = options.index(("FC-ARL-PODS40", "DC-BUC")) if ("FC-ARL-PODS40", "DC-BUC") in options else 0
    sku_id, dc_id = st.selectbox("SKU @ DC", options, index=default,
                                 format_func=lambda o: f"{sku_name[o[0]]} @ {dc_city[o[1]]}")
    hist = load_history(sku_id, dc_id)
    rep = load_projection(sku_id, dc_id)
    proj = pd.DataFrame(rep["projection"]["daily_projection"])
    proj["date"] = pd.to_datetime(proj.date)
    policy = load_policy(sku_id, dc_id)
    p = pal()

    def promo_spans(dates: pd.Series, depth: pd.Series) -> list[tuple]:
        """Contiguous promo date ranges, for shading."""
        spans, start, last = [], None, None
        for d, v in zip(dates, depth):
            if v > 0 and start is None:
                start = d
            if v == 0 and start is not None:
                spans.append((start, last))
                start = None
            last = d
        if start is not None:
            spans.append((start, last))
        return spans

    all_dates = pd.concat([hist.date, proj.date])
    all_depth = pd.concat([hist.promo_depth_pct, proj.promo_depth_pct])
    spans = promo_spans(all_dates.reset_index(drop=True), all_depth.reset_index(drop=True))

    f1 = go.Figure()
    for a, b in spans:
        f1.add_vrect(x0=a - pd.Timedelta(hours=12), x1=b + pd.Timedelta(hours=12), fillcolor=p["promo"],
                     line_width=0, layer="below")
    f1.add_trace(go.Scatter(x=proj.date, y=proj.forecast_p90_units, mode="lines", line=dict(width=0), showlegend=False,
                            hoverinfo="skip"))
    f1.add_trace(go.Scatter(x=proj.date, y=proj.forecast_units, mode="lines", fill="tonexty", fillcolor=p["band"],
                            line=dict(width=0), showlegend=False, hoverinfo="skip"))
    f1.add_trace(go.Scatter(x=hist.date, y=hist.units_demanded, name="Actual demand", mode="lines",
                            line=dict(color=p["series"][0], width=2),
                            hovertemplate="%{y:,.0f} units<extra>Actual demand</extra>"))
    f1.add_trace(go.Scatter(x=proj.date, y=proj.forecast_units, name="Forecast (P90 band)", mode="lines",
                            line=dict(color=p["series"][1], width=2),
                            customdata=proj.forecast_p90_units,
                            hovertemplate="%{y:,.0f} units (P90 %{customdata:,.0f})<extra>Forecast</extra>"))
    f1.add_vline(x=pd.Timestamp(as_of), line=dict(color=p["axis"], width=1, dash="dot"))
    style(f1, height=320, hovermode="x unified", title=None, margin=dict(l=8, r=8, t=28, b=8))
    f1.update_yaxes(title=None, rangemode="tozero")

    hist_tail = hist.tail(28)
    f2 = go.Figure()
    f2.add_trace(go.Scatter(x=hist_tail.date, y=hist_tail.on_hand_close, name="On hand (actual)", mode="lines",
                            line=dict(color=p["series"][0], width=2),
                            hovertemplate="%{y:,.0f} units<extra>On hand</extra>"))
    f2.add_trace(go.Scatter(x=proj.date, y=proj.projected_on_hand, name="Projected (expected)", mode="lines",
                            line=dict(color=p["series"][1], width=2),
                            hovertemplate="%{y:,.0f} units<extra>Projected</extra>"))
    f2.add_trace(go.Scatter(x=proj.date, y=proj.projected_on_hand_p90, name="Projected (P90 demand)", mode="lines",
                            line=dict(color=p["series"][1], width=2, dash="dot"),
                            hovertemplate="%{y:,.0f} units<extra>Projected at P90</extra>"))
    f2.add_hline(y=policy["reorder_point_units"], line=dict(color=p["muted"], width=1, dash="dash"),
                 annotation_text=f"Reorder point {policy['reorder_point_units']:,}", annotation_position="top left",
                 annotation_font=dict(color=p["ink2"], size=11))
    f2.add_vline(x=pd.Timestamp(as_of), line=dict(color=p["axis"], width=1, dash="dot"))
    style(f2, height=320, hovermode="x unified", title=None, margin=dict(l=8, r=8, t=28, b=8))
    f2.update_yaxes(rangemode="tozero")

    g1, g2 = st.columns([1.4, 1])
    g1.markdown("**Daily demand**: last 120 days and 14-day forecast with P90 band; grey = promo")
    g1.plotly_chart(f1, use_container_width=True)
    g2.markdown("**Inventory**: actual and projected with no new orders")
    g2.plotly_chart(f2, use_container_width=True)

    pr = rep["projection"]
    m = st.columns(5)
    m[0].metric("14-day forecast", f"{pr['forecast_14d_units']:,} units", f"P90 {pr['forecast_14d_p90_units']:,}",
                delta_color="off")
    m[1].metric("Projected stockout", pr["stockout_date_expected"] or "none in 14d",
                f"P90: {pr['stockout_date_p90'] or 'none'}", delta_color="off")
    m[2].metric("Effective lead time", f"{rep['effective_lead_time_days']} days",
                f"+{rep['disruption_added_days']} disruption" if rep["disruption_added_days"] else "nominal",
                delta_color="inverse" if rep["disruption_added_days"] else "off")
    m[3].metric("Recommended PO", f"{rep['recommended_order_units']:,} units",
                f"{rep['recommended_order_cases']} cases · {rep['order_value_ron']:,.0f} RON", delta_color="off")
    m[4].metric("Revenue at risk", f"{pr['revenue_at_risk_ron']:,.0f} RON",
                f"{pr['lost_units_before_new_po']:,} units before new PO lands", delta_color="off")
    if rep["transfer_options"]:
        st.caption("Transfer options: " + "; ".join(
            f"{t['transfer_units']:,} units from {dc_city[t['from_dc_id']]} ({t['transit_days']}d, "
            f"{t['cost_ron']:,.0f} RON)" for t in rep["transfer_options"]))

    # ---- Anomalies & category trend --------------------------------------
    a_col, t_col = st.columns([1, 1])
    with a_col:
        st.markdown("##### Anomaly alerts (IsolationForest)")
        days = st.segmented_control("Window", [7, 30, 90], default=30, format_func=lambda d: f"Last {d} days",
                                    key="an_window") or 30
        an = load_anomalies(days)
        an = an[an.sku_id.isin(skus[skus.category.isin(sel_cats)].sku_id) & an.dc_id.isin(sel_dcs)]
        if an.empty:
            st.info("No anomalies flagged in this window.")
        else:
            st.dataframe(pd.DataFrame({
                "Date": pd.to_datetime(an.date).dt.date, "SKU": an.sku_id.map(sku_name), "DC": an.dc_id.map(dc_city),
                "Type": an.anomaly_type.str.replace("_", " ").str.title(),
                "Demand": an.units_demanded, "28d baseline": an.baseline_28d.round(1),
                "Inventory residual": an.inv_residual_units, "Score": an.anomaly_score.round(3)}),
                hide_index=True, use_container_width=True, height=300,
                column_config={"Score": st.column_config.ProgressColumn(format="%.3f", min_value=0.5, max_value=0.9)})
    with t_col:
        st.markdown("##### Weekly net revenue by category")
        cw = load_category_weekly(tuple(sel_dcs))
        cw = cw[cw.category.isin(sel_cats)]
        f3 = go.Figure()
        all_cats = sorted(skus.category.unique())  # fixed slot per category, never re-coloured by filters
        for cat in all_cats:
            d = cw[cw.category == cat]
            if d.empty:
                continue
            f3.add_trace(go.Scatter(x=d.week, y=d.revenue / 1e3, name=cat, mode="lines",
                                    line=dict(color=p["series"][all_cats.index(cat)], width=2),
                                    hovertemplate="%{y:,.0f}k RON<extra>" + cat + "</extra>"))
        style(f3, height=350, hovermode="x unified", title=None, margin=dict(l=8, r=8, t=40, b=8))
        f3.update_yaxes(ticksuffix="k", rangemode="tozero")
        st.plotly_chart(f3, use_container_width=True)

# =========================================================================== #
# Tab 2 - Agentic copilot
# =========================================================================== #
AGENT_ICON = {"Intake": ":material/input:", "Data Analyst": ":material/database:",
              "ML Diagnostics": ":material/model_training:", "Supply Chain Strategist": ":material/strategy:",
              "Grounding Guard": ":material/verified_user:", "Briefing": ":material/summarize:"}
EXAMPLES = [
    "Why is Ariel Pods 40ct at risk of stocking out in Bucharest next week, and how much should we reorder?",
    "Is Pampers Active Baby overstocked in Cluj?",
    "What's going on with Gillette Fusion5 blades in Timisoara?",
    "Show lost sales by category",
    "Which suppliers have the most delays?",
]


def render_step(step: dict) -> None:
    icon = AGENT_ICON.get(step.get("agent"), ":material/bolt:")
    status = {"ok": "", "warning": " :orange[warning]", "error": " :red[error]"}[step.get("status", "ok")]
    ms = f" · {step['duration_ms']} ms" if step.get("duration_ms") else ""
    with st.expander(f"{icon} **{step['agent']}**: {step['action']}{status}{ms}", expanded=False):
        if step.get("detail"):
            st.write(step["detail"])
        if step.get("sql"):
            st.code(step["sql"], language="sql")
        if step.get("output") not in (None, "", [], {}):
            out = step["output"]
            if isinstance(out, list) and out and isinstance(out[0], dict):
                st.dataframe(pd.DataFrame(out), hide_index=True, use_container_width=True)
            else:
                st.json(out, expanded=False)


def render_result(result: dict) -> None:
    cases = result.get("cases", [])
    if not cases:
        st.markdown(result["final_report"])
        return
    if len(cases) > 1:  # triage: network summary first, then each case with its own verification badge
        st.markdown(result["final_report"].split("\n---\n", 1)[0])
        rank = {(r["sku_id"], r["dc_id"]): i for i, r in enumerate((result.get("triage") or {}).get("selected", []))}
        cases = sorted(cases, key=lambda c: rank.get((c.get("sku_id"), c.get("dc_id")), 99))
    for c in cases:
        if len(result.get("cases", [])) > 1:
            st.divider()
            st.markdown(f"#### {c.get('sku_name') or c.get('sku_id')} @ {dc_city.get(c.get('dc_id'), c.get('dc_id'))}"
                        f" · {STATUS_ICON.get(c.get('risk_class'), '')} {STATUS_LABEL.get(c.get('risk_class'), '')}")
        st.markdown(c["brief_markdown"])
        g = c.get("grounding") or {}
        badge = (f":green-badge[:material/verified: {g.get('numbers_checked', 0)} numbers verified against evidence]"
                 if g.get("passed") else
                 f":orange-badge[:material/warning: unverified: {', '.join(g.get('unsupported_numbers', []) + g.get('unsupported_dates', []))}]")
        author = c.get("author", "")
        who = f":blue-badge[{'LLM: ' + author.split(':', 1)[1] if author.startswith('llm') else 'Rule-based narrative'}]"
        label = f"{sku_name.get(c.get('sku_id'), '')} @ {c.get('dc_id')}" if c.get("sku_id") else "Analytics"
        cols = st.columns([4, 1])
        cols[0].markdown(f"{badge} {who} &nbsp; _{label}_")
        cols[1].download_button("Evidence JSON", json.dumps(c.get("evidence", {}), default=str, indent=1),
                                file_name=f"evidence_{c.get('case_id')}.json", mime="application/json",
                                key=f"dl_{c.get('case_id')}", use_container_width=True)
        if c.get("intent") == "analytics_question":
            sql_full = c.get("evidence", {}).get("sql_full")
            rows = sql_full[-1].get("rows") if isinstance(sql_full, list) and sql_full else None
            if rows:
                st.dataframe(pd.DataFrame(rows), hide_index=True, use_container_width=True)


def run_agents(question: str | None, triage: bool, max_cases: int) -> None:
    from src.agents.graph import stream_copilot
    st.session_state.history.append({"role": "user", "content": question or f"Run autonomous morning triage "
                                                                            f"(top {max_cases} risks)"})
    with st.chat_message("user"):
        st.markdown(st.session_state.history[-1]["content"])
    with st.chat_message("assistant"):
        steps, final = [], None
        t0 = time.time()
        with st.status("Agents working...", expanded=True) as status:
            for kind, payload in stream_copilot(question, triage=triage, max_cases=max_cases, graph=copilot_graph()):
                if kind == "step":
                    steps.append(payload)
                    status.update(label=f"{payload['agent']}: {payload['action']}")
                    render_step(payload)
                else:
                    final = payload
            status.update(label=f"Done in {time.time() - t0:.1f}s · {len(steps)} agent steps",
                          state="complete", expanded=False)
        render_result(final)
    st.session_state.history.append({"role": "assistant", "steps": steps, "result": final,
                                     "elapsed": time.time() - t0})


with tab_agent:
    st.session_state.setdefault("history", [])
    head = st.columns([2, 1, 1])
    head[0].markdown("Ask about a product, a DC or the network. Each answer runs **Data Analyst -> ML Diagnostics "
                     "-> Supply Chain Strategist -> Grounding Guard**; every number is produced by a SQL query or a "
                     "model tool and verified before you see it.")
    n_cases = head[1].number_input("Triage investigations", 1, 6, 3)
    triage_clicked = head[2].button("Run Autonomous Morning Triage", type="primary", use_container_width=True,
                                    icon=":material/play_arrow:")
    ex_cols = st.columns(len(EXAMPLES))
    example = None
    for col, ex in zip(ex_cols, EXAMPLES):
        if col.button(ex, use_container_width=True, key=f"ex_{hash(ex)}"):
            example = ex

    for msg in st.session_state.history:
        with st.chat_message(msg["role"]):
            if msg["role"] == "user":
                st.markdown(msg["content"])
            else:
                with st.expander(f"Agent trace · {len(msg['steps'])} steps · {msg['elapsed']:.1f}s"):
                    for s in msg["steps"]:
                        render_step(s)
                render_result(msg["result"])

    prompt = st.chat_input("e.g. Why is Head & Shoulders at risk in Bucharest?")
    if triage_clicked:
        run_agents(None, True, int(n_cases))
    elif example or prompt:
        run_agents(example or prompt, False, int(n_cases))

# =========================================================================== #
# Tab 3 - Model & data lab
# =========================================================================== #
with tab_lab:
    fc, ad = card["forecaster"], card["anomaly_detector"]
    st.markdown("##### Demand forecaster: backtest on the last 56 forecast origins")
    c = st.columns(4)
    c[0].metric("WAPE", f"{fc['wape']:.1%}", f"{-fc['improvement_vs_seasonal_naive_pct']:.0%} vs seasonal naive",
                delta_color="inverse")
    c[1].metric("Bias", f"{fc['bias_pct']:+.1%}")
    c[2].metric("WAPE on promo days", f"{fc['wape_promo_days']:.1%}")
    c[3].metric("P90 interval coverage", f"{fc['p90_coverage']:.1%}", "target 90%", delta_color="off")
    p = pal()
    l, r = st.columns(2)
    with l:
        names = ["Seasonal naive (same weekday)", "28-day moving average", "HistGradientBoosting (ours)"]
        vals = [fc["wape_baseline_seasonal_naive"], fc["wape_baseline_ma28"], fc["wape"]]
        fb = go.Figure(go.Bar(x=vals, y=names, orientation="h", marker=dict(color=[p["muted"], p["muted"],
                                                                                    p["series"][0]],
                                                                             cornerradius=4),
                              text=[f"{v:.1%}" for v in vals], textposition="outside",
                              hovertemplate="%{y}: WAPE %{x:.1%}<extra></extra>"))
        style(fb, height=240, title="WAPE vs baselines (lower is better)", showlegend=False)
        fb.update_xaxes(tickformat=".0%", range=[0, max(vals) * 1.25])
        st.plotly_chart(fb, use_container_width=True)
    with r:
        h = fc["wape_by_horizon"]
        fh = go.Figure(go.Scatter(x=[int(k) for k in h], y=list(h.values()), mode="lines+markers",
                                  line=dict(color=p["series"][0], width=2), marker=dict(size=8),
                                  hovertemplate="Day +%{x}: WAPE %{y:.1%}<extra></extra>"))
        style(fh, height=240, title="WAPE by forecast horizon (days ahead)", showlegend=False)
        fh.update_yaxes(tickformat=".0%", rangemode="tozero")
        st.plotly_chart(fh, use_container_width=True)
    st.caption(f"{fc['model']}. Trained on {fc['train_rows']:,} (origin, horizon) rows; target is unconstrained "
               "demand so stockout days don't teach the model that demand was low.")

    st.markdown("##### Anomaly detector: scored against the generator's injected ground truth")
    c = st.columns(4)
    c[0].metric("Flags (2 years, 56 series)", f"{ad['flagged']}")
    c[1].metric("Recall vs injected", f"{ad['recall_vs_injected']:.0%}")
    c[2].metric("Precision vs injected", f"{ad['precision_vs_injected']:.0%}")
    c[3].metric("Recall: velocity spikes / inventory drops",
                f"{ad['recall_by_type'].get('VELOCITY_SPIKE', 0):.0%} / {ad['recall_by_type'].get('INVENTORY_DROP', 0):.0%}")
    st.caption(f"{ad['model']}. {ad['note']}")

    st.divider()
    st.markdown("##### SQL explorer (same read-only guard the Data Analyst agent uses)")
    default_sql = ("SELECT k.brand, d.city, SUM(s.lost_sales_units) AS lost_units,\n"
                   "       ROUND(SUM(s.lost_sales_units * s.net_price_ron), 0) AS lost_revenue_ron\n"
                   "FROM fact_daily_sales s JOIN dim_sku k USING (sku_id) JOIN dim_dc d USING (dc_id)\n"
                   "WHERE s.date > DATE '" + as_of + "' - 28\nGROUP BY 1, 2 ORDER BY lost_revenue_ron DESC")
    sql = st.text_area("Query", default_sql, height=140, label_visibility="collapsed")
    q1, q2 = st.columns([1, 5])
    if q1.button("Run query", icon=":material/play_arrow:"):
        try:
            res = run_query(sql)
            st.dataframe(res.df, hide_index=True, use_container_width=True)
            st.caption(f"{res.row_count} rows{' (truncated)' if res.truncated else ''} · tables: {', '.join(res.tables)}")
        except UnsafeQueryError as exc:
            st.error(f"Rejected by read-only guard: {exc}")
        except Exception as exc:
            st.error(f"{type(exc).__name__}: {exc}")
    q2.caption("Try `DROP TABLE dim_sku`, `SELECT 1; SELECT 2` or `SELECT * FROM read_csv('secrets.csv')` "
               "to see the guard reject it.")

    with st.expander("Warehouse schema (generated from DuckDB COMMENTs, also fed to the text-to-SQL agent)"):
        st.code(schema_description(), language="sql")
    with st.expander("LangGraph topology"):
        g = copilot_graph()
        st.code(g.get_graph(xray=True).draw_mermaid(), language="text")
