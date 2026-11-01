"""Synthetic FMCG warehouse generator for CPG Pulse.

Simulates two years of daily sell-out, inventory and replenishment for a
P&G-style portfolio across four Romanian distribution centres, then writes
everything to a DuckDB file (``data/cpg_warehouse.duckdb``).

The simulation is mechanistic rather than random noise:
  demand  = base x trend x annual season x weekday x promo lift x post-promo dip
            x competitor-price effect x gamma-Poisson noise
  stock   = daily (s, Q) reorder policy with promo pre-builds, stochastic
            supplier lead times and logistics disruptions that delay POs
  sales   = min(demand, available stock); the remainder is lost sales

A handful of scenarios are engineered at the end of the history so the
agents have something real to investigate (Ariel Pods in Bucharest, Pampers
in Iasi/Cluj, Gillette in Timisoara, Head & Shoulders in Bucharest). Every
injected anomaly is logged in ``meta_injected_anomalies`` so the unsupervised
detector can be scored against ground truth.

All names, plants, quantities and prices are synthetic.

Usage:
    python data/generate_cpg_data.py [--db PATH] [--end-date YYYY-MM-DD] [--seed N]
"""
from __future__ import annotations

import argparse
import math
from datetime import datetime, timezone
from pathlib import Path

import duckdb
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_DB = ROOT / "data" / "cpg_warehouse.duckdb"
DEFAULT_END = "2026-10-31"
HISTORY_DAYS = 730
PLAN_HORIZON_DAYS = 42  # promo calendar extends this far past the as-of date

# --------------------------------------------------------------------------- #
# Master data
# --------------------------------------------------------------------------- #
# sku_id, name, brand, category, pack_size, list_price_ron, unit_cost_ron,
# case_pack, base_units_per_day (Bucharest), supplier_id, promo_elasticity,
# competitor_elasticity, annual_growth
SKUS = [
    ("FC-ARL-PODS40", "Ariel All-in-1 Pods 40ct", "Ariel", "Fabric Care", "40 pods", 89.99, 47.50, 6, 70, "SUP-URL", 2.6, 1.6, 0.06),
    ("FC-ARL-LIQ22", "Ariel Liquid Color 2.2L", "Ariel", "Fabric Care", "2.2 L", 64.99, 33.00, 4, 85, "SUP-URL", 2.2, 1.4, 0.02),
    ("FC-TIDE-PWD4", "Tide Powder Original 4kg", "Tide", "Fabric Care", "4 kg", 54.99, 27.50, 4, 60, "SUP-URL", 2.0, 1.8, -0.03),
    ("FC-LEN-SOFT18", "Lenor Fabric Softener Spring 1.8L", "Lenor", "Fabric Care", "1.8 L", 29.99, 14.00, 8, 95, "SUP-URL", 2.4, 1.3, 0.03),
    ("BC-PMP-PC3-60", "Pampers Premium Care S3 60ct", "Pampers", "Baby Care", "60 diapers", 79.99, 41.00, 4, 55, "SUP-BBC", 1.6, 1.2, 0.04),
    ("BC-PMP-AB4-76", "Pampers Active Baby S4 76ct", "Pampers", "Baby Care", "76 diapers", 84.99, 43.00, 4, 65, "SUP-BBC", 1.7, 1.5, 0.01),
    ("BC-PMP-WIPES", "Pampers Sensitive Wipes 4x52", "Pampers", "Baby Care", "4 x 52 wipes", 24.99, 11.50, 12, 110, "SUP-BBC", 2.0, 1.1, 0.05),
    ("GR-GIL-F5-4", "Gillette Fusion5 Blades 4ct", "Gillette", "Grooming", "4 cartridges", 74.99, 30.00, 10, 40, "SUP-GRM", 1.9, 1.0, 0.02),
    ("GR-GIL-M3", "Gillette Mach3 Razor", "Gillette", "Grooming", "1 razor + 2 blades", 39.99, 15.00, 12, 35, "SUP-GRM", 1.8, 1.0, -0.02),
    ("GR-GIL-GEL200", "Gillette Series Sensitive Gel 200ml", "Gillette", "Grooming", "200 ml", 19.99, 8.00, 12, 50, "SUP-GRM", 2.1, 1.2, 0.01),
    ("HC-HS-CLS400", "Head & Shoulders Classic Clean 400ml", "Head & Shoulders", "Hair Care", "400 ml", 32.99, 14.00, 12, 75, "SUP-HHC", 2.0, 1.3, 0.03),
    ("HC-PNT-REP400", "Pantene Pro-V Repair & Protect 400ml", "Pantene", "Hair Care", "400 ml", 29.99, 12.50, 12, 60, "SUP-HHC", 2.2, 1.4, 0.00),
    ("OC-ORB-PE75", "Oral-B Pro-Expert Toothpaste 75ml", "Oral-B", "Oral Care", "75 ml", 17.99, 6.50, 24, 90, "SUP-HHC", 1.8, 1.2, 0.04),
    ("HM-FRY-ORG900", "Fairy Original Dishwashing Liquid 900ml", "Fairy", "Home Care", "900 ml", 16.99, 7.00, 12, 130, "SUP-URL", 1.9, 1.1, 0.02),
]
SKU_COLS = [
    "sku_id", "sku_name", "brand", "category", "pack_size", "list_price_ron", "unit_cost_ron",
    "case_pack", "base_units_per_day", "supplier_id", "promo_elasticity", "competitor_elasticity",
    "annual_growth",
]

# dc_id, dc_name, city, region, size_factor, latitude, longitude, capacity_units
DCS = [
    ("DC-BUC", "Bucharest DC", "Bucharest", "Muntenia", 1.00, 44.4268, 26.1025, 250_000),
    ("DC-CLJ", "Cluj-Napoca DC", "Cluj-Napoca", "Transilvania", 0.60, 46.7712, 23.6236, 150_000),
    ("DC-TIM", "Timisoara DC", "Timisoara", "Banat", 0.50, 45.7489, 21.2087, 120_000),
    ("DC-IAS", "Iasi DC", "Iasi", "Moldova", 0.45, 47.1585, 27.6014, 110_000),
]
DC_COLS = ["dc_id", "dc_name", "city", "region", "size_factor", "latitude", "longitude", "capacity_units"]

# supplier_id, name, country, lead_time_mean_days, lead_time_std_days, on_time_reliability
SUPPLIERS = [
    ("SUP-URL", "Fabric & Home Care Plant (RO)", "Romania", 4, 1.0, 0.96),
    ("SUP-BBC", "Baby Care Plant (PL)", "Poland", 7, 1.5, 0.93),
    ("SUP-GRM", "Grooming Plant (DE)", "Germany", 10, 2.0, 0.95),
    ("SUP-HHC", "Hair & Oral Care Plant (CZ)", "Czechia", 8, 1.5, 0.94),
]
SUPPLIER_COLS = ["supplier_id", "supplier_name", "country", "lead_time_mean_days", "lead_time_std_days", "on_time_reliability"]

# Inter-DC transfer lanes (symmetric): transit days, cost per unit in RON
LANES = {
    ("DC-BUC", "DC-CLJ"): (2, 1.10), ("DC-BUC", "DC-TIM"): (2, 1.20), ("DC-BUC", "DC-IAS"): (1, 0.90),
    ("DC-CLJ", "DC-TIM"): (1, 0.80), ("DC-CLJ", "DC-IAS"): (2, 1.00), ("DC-TIM", "DC-IAS"): (3, 1.40),
}

# Category seasonality: (amplitude, peak day-of-year)
SEASONALITY = {
    "Fabric Care": (0.08, 100), "Baby Care": (0.03, 15), "Grooming": (0.18, 350),
    "Hair Care": (0.10, 200), "Oral Care": (0.04, 30), "Home Care": (0.07, 355),
}
WEEKDAY = np.array([0.90, 0.92, 0.95, 1.00, 1.12, 1.20, 0.91])  # Mon..Sun
CHANNELS = {"Hypermarket": 0.32, "Supermarket": 0.24, "Discounter": 0.22, "Proximity": 0.14, "E-commerce": 0.08}
PROMO_MECHANICS = ["Temporary Price Reduction", "Leaflet + Gondola End", "Multibuy 2+1", "Loyalty Card Discount"]
SERVICE_Z = 1.65  # ~95% cycle service level used by the (synthetic) planners


# --------------------------------------------------------------------------- #
# Table & column documentation (written as DuckDB COMMENTs -> self-describing
# schema that the SQL agent reads at runtime)
# --------------------------------------------------------------------------- #
TABLE_DOCS = {
    "dim_sku": ("Product master: one row per SKU.", {
        "sku_id": "SKU code, e.g. FC-ARL-PODS40",
        "sku_name": "Full product name",
        "brand": "Brand (Ariel, Tide, Lenor, Pampers, Gillette, Head & Shoulders, Pantene, Oral-B, Fairy)",
        "category": "Category (Fabric Care, Baby Care, Grooming, Hair Care, Oral Care, Home Care)",
        "list_price_ron": "Shelf list price per unit in RON",
        "unit_cost_ron": "Landed cost per unit in RON",
        "case_pack": "Units per case; purchase orders are rounded up to whole cases",
        "supplier_id": "Sourcing plant (FK dim_supplier)",
    }),
    "dim_dc": ("Distribution centres in Romania.", {
        "dc_id": "DC code: DC-BUC, DC-CLJ, DC-TIM, DC-IAS",
        "city": "City (Bucharest, Cluj-Napoca, Timisoara, Iasi)",
        "capacity_units": "Storage capacity in units",
    }),
    "dim_supplier": ("Sourcing plants and their nominal lead-time profile.", {
        "lead_time_mean_days": "Nominal plant-to-DC lead time in days",
        "lead_time_std_days": "Std. deviation of lead time in days",
        "on_time_reliability": "Share of POs delivered on time (0-1)",
    }),
    "dim_inventory_policy": ("Current replenishment parameters per SKU x DC as configured by planners.", {
        "reorder_point_units": "Reorder point: a PO is raised when inventory position falls to this level",
        "safety_stock_units": "Safety stock component of the reorder point",
        "order_qty_units": "Standard replenishment order quantity",
        "planned_daily_demand": "Daily demand the policy was sized on (can go stale)",
    }),
    "dim_transfer_lanes": ("Inter-DC stock transfer lanes.", {
        "transit_days": "Truck transit time in days",
        "cost_per_unit_ron": "Transfer cost per unit in RON",
    }),
    "fact_daily_sales": ("Daily sell-out per SKU x DC (grain: date, sku_id, dc_id).", {
        "date": "Business date",
        "units_sold": "Units shipped to stores (constrained by stock)",
        "units_demanded": "Unconstrained demand = units_sold + lost_sales_units",
        "lost_sales_units": "Estimated demand lost because the DC was out of stock",
        "promo_flag": "1 if a promotion was live that day",
        "promo_depth_pct": "Promotional discount depth (0-1)",
        "net_price_ron": "Realised price per unit after promo discount",
        "net_revenue_ron": "units_sold x net_price_ron",
        "gross_margin_ron": "net_revenue_ron - units_sold x unit_cost_ron",
        "competitor_price_index": "Competitor shelf price / our list price (below 1 = competitor cheaper)",
    }),
    "fact_inventory_daily": ("End-of-day inventory per SKU x DC (grain: date, sku_id, dc_id).", {
        "on_hand_open": "Units on hand at start of day",
        "receipts_units": "Units received from purchase orders or transfers",
        "units_shipped": "Units shipped (= units_sold)",
        "adjustment_units": "Inventory adjustments (negative = damage, shrink, write-off)",
        "on_hand_close": "Units on hand at end of day",
        "on_order_units": "Units on open purchase orders at end of day",
        "stockout_flag": "1 if demand was not fully served that day",
        "days_of_cover": "on_hand_close / trailing 28-day average units_sold",
    }),
    "fact_purchase_orders": ("Purchase orders from plants to DCs.", {
        "po_type": "REPLENISHMENT, PROMO_PREBUILD or MANUAL",
        "expected_arrival_date": "Arrival date promised at order time",
        "revised_eta_date": "Latest ETA (differs from expected when delayed)",
        "actual_arrival_date": "Date received (NULL if still open)",
        "status": "RECEIVED, IN_TRANSIT or DELAYED",
        "delay_days": "revised/actual arrival minus expected arrival, in days",
    }),
    "fact_promo_calendar": ("Trade promotion calendar per SKU x DC, including planned future events.", {
        "promo_depth_pct": "Discount depth (0-1)",
        "status": "COMPLETED, ACTIVE, PLANNED or CANCELLED",
    }),
    "fact_logistics_disruptions": ("Supply-chain disruption log. Scope columns (supplier_id, dc_id, sku_id) are NULL when not restricted.", {
        "added_lead_time_days": "Extra lead time imposed on POs placed or in transit during the event",
        "severity": "LOW, MEDIUM or HIGH",
        "status": "ACTIVE or RESOLVED as of the as-of date",
    }),
    "fact_channel_sales_weekly": ("Weekly sell-out split by retail channel (grain: week_start, sku_id, dc_id, channel).", {
        "channel": "Hypermarket, Supermarket, Discounter, Proximity, E-commerce",
    }),
    "meta_dataset_info": ("Dataset metadata key/value pairs (as_of_date = last closed business day).", {}),
    "meta_injected_anomalies": ("Ground-truth log of anomalies injected by the generator (used only to evaluate the anomaly detector).", {}),
}


# --------------------------------------------------------------------------- #
# Helpers
# --------------------------------------------------------------------------- #
def _round_up(x: float, multiple: int) -> int:
    return int(math.ceil(max(x, 0) / multiple) * multiple)


def _make_promos(rng: np.random.Generator, end: pd.Timestamp, start: pd.Timestamp) -> pd.DataFrame:
    rows = []
    horizon_end = end + pd.Timedelta(days=PLAN_HORIZON_DAYS)
    dc_ids = [d[0] for d in DCS]
    for sku in SKUS:
        sku_id = sku[0]
        t = start + pd.Timedelta(days=int(rng.integers(5, 40)))
        while t < horizon_end:
            duration = int(rng.choice([7, 10, 14]))
            depth = float(rng.choice([0.15, 0.20, 0.25, 0.30, 0.35]))
            mechanic = str(rng.choice(PROMO_MECHANICS))
            scope = dc_ids if rng.random() < 0.7 else [str(rng.choice(dc_ids))]
            for dc_id in scope:
                rows.append((sku_id, dc_id, t, t + pd.Timedelta(days=duration - 1), depth, mechanic, None))
            t = t + pd.Timedelta(days=duration + int(rng.integers(35, 75)))
    promos = pd.DataFrame(rows, columns=["sku_id", "dc_id", "start_date", "end_date", "promo_depth_pct", "mechanic", "status"])

    # Clear the window around the engineered scenarios so the story is unambiguous.
    def clear(sku_id, lo, hi, dcs=None):
        mask = (promos.sku_id == sku_id) & (promos.end_date >= lo) & (promos.start_date <= hi)
        if dcs is not None:
            mask &= promos.dc_id.isin(dcs)
        return promos[~mask]

    d = lambda n: end + pd.Timedelta(days=n)  # noqa: E731
    promos = clear("FC-ARL-PODS40", d(-35), d(PLAN_HORIZON_DAYS))
    promos = clear("BC-PMP-AB4-76", d(-35), d(PLAN_HORIZON_DAYS))
    promos = clear("GR-GIL-F5-4", d(-30), d(20))
    promos = clear("HC-HS-CLS400", d(-20), d(5))
    engineered = [
        # National Ariel Pods leaflet promo starting in 4 days, 30% off for 14 days
        *[("FC-ARL-PODS40", dc, d(4), d(17), 0.30, "Leaflet + Gondola End", None) for dc in dc_ids],
        # Regional Pampers promo in Iasi starting in 3 days
        ("BC-PMP-AB4-76", "DC-IAS", d(3), d(13), 0.25, "Loyalty Card Discount", None),
        # Cancelled Pampers promo in Cluj (pre-build stock already landed -> overstock)
        ("BC-PMP-AB4-76", "DC-CLJ", d(-4), d(6), 0.25, "Multibuy 2+1", "CANCELLED"),
    ]
    promos = pd.concat([promos, pd.DataFrame(engineered, columns=promos.columns)], ignore_index=True)
    promos = promos.sort_values(["sku_id", "dc_id", "start_date"]).reset_index(drop=True)

    def status(r):
        if r.status == "CANCELLED":
            return "CANCELLED"
        if r.end_date < end:
            return "COMPLETED"
        if r.start_date <= end <= r.end_date:
            return "ACTIVE"
        return "PLANNED"

    promos["status"] = promos.apply(status, axis=1)
    promos.insert(0, "promo_id", [f"PRM-{i:05d}" for i in range(1, len(promos) + 1)])
    return promos


def _make_disruptions(rng: np.random.Generator, end: pd.Timestamp, start: pd.Timestamp) -> pd.DataFrame:
    rows = []
    types = [
        ("Port congestion", "Inbound containers held at Constanta port"),
        ("Production line maintenance", "Unplanned line stoppage at plant"),
        ("Raw material shortage", "Packaging material allocation from vendor"),
        ("Carrier capacity shortage", "Contracted haulier short of trucks"),
        ("Severe weather", "Road closures on main inbound route"),
    ]
    sup_ids = [s[0] for s in SUPPLIERS]
    dc_ids = [d[0] for d in DCS]
    for _ in range(9):
        s = start + pd.Timedelta(days=int(rng.integers(30, HISTORY_DAYS - 60)))
        dur = int(rng.integers(5, 16))
        dtype, desc = types[int(rng.integers(0, len(types)))]
        if dtype in ("Carrier capacity shortage", "Severe weather"):
            sup, dc = None, str(rng.choice(dc_ids))
            added = int(rng.integers(1, 4))
        else:
            sup, dc = str(rng.choice(sup_ids)), None
            added = int(rng.integers(3, 8))
        sev = "HIGH" if added >= 6 else "MEDIUM" if added >= 3 else "LOW"
        rows.append((sup, dc, None, dtype, s, s + pd.Timedelta(days=dur - 1), sev, added, desc))

    d = lambda n: end + pd.Timedelta(days=n)  # noqa: E731
    rows += [
        ("SUP-URL", None, "FC-ARL-PODS40", "Raw material shortage", d(-9), d(12), "HIGH", 7,
         "PVA film supplier allocation cut pods line output; Ariel Pods POs re-scheduled +7 days"),
        (None, "DC-IAS", None, "Carrier capacity shortage", d(-3), d(5), "MEDIUM", 2,
         "Contracted haulier for Iasi inbound short of trucks"),
    ]
    df = pd.DataFrame(rows, columns=["supplier_id", "dc_id", "sku_id", "disruption_type", "start_date", "end_date",
                                     "severity", "added_lead_time_days", "description"])
    df = df.sort_values("start_date").reset_index(drop=True)
    df["status"] = np.where(df.end_date >= end, "ACTIVE", "RESOLVED")
    df.insert(0, "event_id", [f"DSR-{i:04d}" for i in range(1, len(df) + 1)])
    return df


# --------------------------------------------------------------------------- #
# Simulation
# --------------------------------------------------------------------------- #
def simulate(end: pd.Timestamp, seed: int) -> dict[str, pd.DataFrame]:
    rng = np.random.default_rng(seed)
    start = end - pd.Timedelta(days=HISTORY_DAYS - 1)
    dates = pd.date_range(start, end, freq="D")
    T = len(dates)

    dim_sku = pd.DataFrame(SKUS, columns=SKU_COLS)
    dim_dc = pd.DataFrame(DCS, columns=DC_COLS)
    dim_supplier = pd.DataFrame(SUPPLIERS, columns=SUPPLIER_COLS)
    sup_by_id = dim_supplier.set_index("supplier_id")

    promos = _make_promos(rng, end, start)
    disruptions = _make_disruptions(rng, end, start)

    series = [(sku, dc) for sku in dim_sku.itertuples(index=False) for dc in dim_dc.itertuples(index=False)]
    S = len(series)
    sidx = {(sku.sku_id, dc.dc_id): i for i, (sku, dc) in enumerate(series)}
    t_of = lambda ts: (ts - start).days  # noqa: E731  (may fall outside [0, T))

    # ---- Promo arrays (history only drives simulation) -------------------
    promo_depth = np.zeros((S, T))
    post_promo = np.ones((S, T))
    for r in promos[promos.status != "CANCELLED"].itertuples():
        i = sidx[(r.sku_id, r.dc_id)]
        a, b = max(t_of(r.start_date), 0), min(t_of(r.end_date), T - 1)
        if a <= b:
            promo_depth[i, a:b + 1] = r.promo_depth_pct
        pa, pb = t_of(r.end_date) + 1, min(t_of(r.end_date) + 7, T - 1)
        if 0 <= pa <= pb:
            post_promo[i, pa:pb + 1] = 0.88

    # ---- Competitor price index ------------------------------------------
    comp = np.ones((S, T))
    for k, sku in enumerate(dim_sku.itertuples(index=False)):
        x = np.empty(T)
        x[0] = 1.0
        shocks = rng.normal(0, 0.006, T)
        for t in range(1, T):
            x[t] = 1 + 0.97 * (x[t - 1] - 1) + shocks[t]
        for _ in range(3):  # competitor price-war events
            a = int(rng.integers(30, T - 60))
            x[a:a + int(rng.integers(14, 29))] *= 0.86
        for j in range(len(DCS)):
            comp[k * len(DCS) + j] = x + rng.normal(0, 0.004, T)
    # Engineered: aggressive competitor price cut on diapers in Cluj
    comp[sidx[("BC-PMP-AB4-76", "DC-CLJ")], T - 30:] *= 0.82
    comp = np.clip(comp, 0.6, 1.3)

    # ---- Demand ----------------------------------------------------------
    doy = dates.day_of_year.to_numpy()
    wd = WEEKDAY[dates.day_of_week.to_numpy()]
    mean = np.zeros((S, T))
    base = np.zeros(S)
    for i, (sku, dc) in enumerate(series):
        amp, peak = SEASONALITY[sku.category]
        season = 1 + amp * np.cos(2 * np.pi * (doy - peak) / 365.25)
        trend = (1 + sku.annual_growth) ** (np.arange(T) / 365.25)
        lift = np.exp(sku.promo_elasticity * promo_depth[i])
        base[i] = sku.base_units_per_day * dc.size_factor
        mean[i] = base[i] * trend * season * wd * lift * post_promo[i] * comp[i] ** sku.competitor_elasticity

    # Engineered: social-media driven acceleration of Ariel Pods in Bucharest
    i = sidx[("FC-ARL-PODS40", "DC-BUC")]
    mean[i, T - 21:] *= np.linspace(1.05, 1.40, 21)

    noise = rng.gamma(shape=12, scale=1 / 12, size=(S, T))
    demand = rng.poisson(mean * noise).astype(float)

    # ---- Injected anomalies ----------------------------------------------
    anomalies = []  # (t, series, type, magnitude, note)
    no_promo = promo_depth == 0
    for _ in range(26):
        i, t = int(rng.integers(0, S)), int(rng.integers(35, T - 30))
        if not no_promo[i, t]:
            continue
        f = float(rng.uniform(3.0, 5.0))
        demand[i, t] = np.round(demand[i, t] * f)
        anomalies.append((t, i, "VELOCITY_SPIKE", round(f, 2), "Unexplained bulk order / listing error"))
    shrink = {}
    for _ in range(14):
        i, t = int(rng.integers(0, S)), int(rng.integers(35, T - 30))
        f = float(rng.uniform(0.25, 0.50))
        shrink[(i, t)] = f
        anomalies.append((t, i, "INVENTORY_DROP", round(f, 2), "Damage / shrink write-off"))
    # Engineered recent anomalies
    for key, t, f, kind, note in [
        (("FC-ARL-PODS40", "DC-BUC"), T - 2, 3.2, "VELOCITY_SPIKE", "Viral cleaning-hack video drove a sell-out spike"),
        (("HC-HS-CLS400", "DC-BUC"), T - 1, 4.0, "VELOCITY_SPIKE", "One-off B2B bulk order from HoReCa distributor"),
    ]:
        i = sidx[key]
        demand[i, t] = np.round(demand[i, t] * f)
        anomalies.append((t, i, kind, f, note))
    i, t = sidx[("GR-GIL-F5-4", "DC-TIM")], T - 3
    shrink[(i, t)] = 0.45
    anomalies.append((t, i, "INVENTORY_DROP", 0.45, "Pallet damage during forklift handling"))

    # ---- Inventory policy (sized on base demand: goes stale by design) ---
    policy_rows = []
    rop = np.zeros(S)
    oq = np.zeros(S)
    for i, (sku, dc) in enumerate(series):
        sup = sup_by_id.loc[sku.supplier_id]
        lt = sup.lead_time_mean_days
        ss = SERVICE_Z * (base[i] * 0.35) * math.sqrt(lt)
        rop[i] = round(base[i] * lt + ss)
        oq[i] = _round_up(base[i] * 14, sku.case_pack)
        policy_rows.append((sku.sku_id, dc.dc_id, int(rop[i]), int(round(ss)), int(oq[i]), round(base[i], 1), 7, 0.95))
    dim_policy = pd.DataFrame(policy_rows, columns=[
        "sku_id", "dc_id", "reorder_point_units", "safety_stock_units", "order_qty_units",
        "planned_daily_demand", "review_period_days", "target_service_level"])

    def added_delay(sku, dc_id, t_order):
        ts = start + pd.Timedelta(days=t_order)
        m = disruptions[(disruptions.start_date <= ts) & (disruptions.end_date >= ts)]
        tot = 0
        for r in m.itertuples():
            if r.supplier_id is not None and r.supplier_id != sku.supplier_id:
                continue
            if r.dc_id is not None and r.dc_id != dc_id:
                continue
            if r.sku_id is not None and r.sku_id != sku.sku_id:
                continue
            tot += r.added_lead_time_days
        return tot

    # Pre-build triggers: (series, t_order) -> promo uplift qty
    prebuild = {}
    for r in promos[promos.status != "CANCELLED"].itertuples():
        i = sidx[(r.sku_id, r.dc_id)]
        sku = series[i][0]
        lt = sup_by_id.loc[sku.supplier_id].lead_time_mean_days
        t_order = t_of(r.start_date) - lt - 2
        if 0 <= t_order < T:
            dur = (r.end_date - r.start_date).days + 1
            uplift = base[i] * (math.exp(sku.promo_elasticity * r.promo_depth_pct) - 1) * dur * 0.8
            prebuild[(i, t_order)] = _round_up(uplift, sku.case_pack)

    # ---- Day-by-day stock simulation -------------------------------------
    pos = []  # dicts
    arrivals: dict[tuple[int, int], list[int]] = {}  # (series, t) -> PO indices
    on_order = np.zeros(S)
    on_hand = rop + oq / 2
    rec = {k: np.zeros((S, T)) for k in ("open", "receipts", "sold", "adj", "close", "on_order", "lost")}

    def place(i, t, qty, po_type):
        sku, dc = series[i]
        sup = sup_by_id.loc[sku.supplier_id]
        nominal = int(sup.lead_time_mean_days)
        lt = max(1, int(round(rng.normal(sup.lead_time_mean_days, sup.lead_time_std_days))))
        if rng.random() > sup.on_time_reliability:
            lt += int(rng.integers(1, 4))
        lt += added_delay(sku, dc.dc_id, t)
        po = dict(sku_id=sku.sku_id, dc_id=dc.dc_id, supplier_id=sku.supplier_id, po_type=po_type,
                  order_t=t, expected_t=t + nominal, arrival_t=t + lt, quantity_units=int(qty))
        pos.append(po)
        arrivals.setdefault((i, t + lt), []).append(len(pos) - 1)
        on_order[i] += qty

    # Engineered: Cluj pre-build for the (later cancelled) Pampers promo, landed 6 days ago
    i_clj = sidx[("BC-PMP-AB4-76", "DC-CLJ")]
    pos.append(dict(sku_id="BC-PMP-AB4-76", dc_id="DC-CLJ", supplier_id="SUP-BBC", po_type="PROMO_PREBUILD",
                    order_t=T - 13, expected_t=T - 6, arrival_t=T - 6,
                    quantity_units=_round_up(base[i_clj] * 32, 4)))
    arrivals.setdefault((i_clj, T - 6), []).append(len(pos) - 1)
    on_order[i_clj] += pos[-1]["quantity_units"]

    for t in range(T):
        for i in range(S):
            rec["open"][i, t] = on_hand[i]
            rcv = 0
            for p in arrivals.get((i, t), []):
                rcv += pos[p]["quantity_units"]
            on_order[i] -= rcv
            avail = on_hand[i] + rcv
            adj = 0.0
            if (i, t) in shrink:
                adj = -round(avail * shrink[(i, t)])
                avail += adj
            sold = min(demand[i, t], avail)
            on_hand[i] = avail - sold
            rec["receipts"][i, t] = rcv
            rec["adj"][i, t] = adj
            rec["sold"][i, t] = sold
            rec["lost"][i, t] = demand[i, t] - sold
            rec["close"][i, t] = on_hand[i]

            if (i, t) in prebuild:
                place(i, t, prebuild[(i, t)], "PROMO_PREBUILD")
            if on_hand[i] + on_order[i] <= rop[i]:
                place(i, t, oq[i], "REPLENISHMENT")
            rec["on_order"][i, t] = on_order[i]

    # ---- Assemble fact tables -------------------------------------------
    sku_df = dim_sku.set_index("sku_id")
    keys = pd.DataFrame([(s.sku_id, d.dc_id) for s, d in series], columns=["sku_id", "dc_id"])
    long = lambda arr: arr.reshape(-1)  # noqa: E731  (series-major, then date)
    frame = pd.DataFrame({
        "date": np.tile(dates.values, S),
        "sku_id": np.repeat(keys.sku_id.values, T),
        "dc_id": np.repeat(keys.dc_id.values, T),
    })
    list_price = frame.sku_id.map(sku_df.list_price_ron).values
    unit_cost = frame.sku_id.map(sku_df.unit_cost_ron).values
    depth = long(promo_depth)
    sold = long(rec["sold"]).astype(int)
    lost = long(rec["lost"]).astype(int)
    net_price = np.round(list_price * (1 - depth), 2)

    sales = frame.copy()
    sales["units_sold"] = sold
    sales["units_demanded"] = sold + lost
    sales["lost_sales_units"] = lost
    sales["promo_flag"] = (depth > 0).astype(int)
    sales["promo_depth_pct"] = depth
    sales["list_price_ron"] = list_price
    sales["net_price_ron"] = net_price
    sales["net_revenue_ron"] = np.round(sold * net_price, 2)
    sales["gross_margin_ron"] = np.round(sold * (net_price - unit_cost), 2)
    sales["competitor_price_index"] = np.round(long(comp), 4)

    inv = frame.copy()
    inv["on_hand_open"] = long(rec["open"]).astype(int)
    inv["receipts_units"] = long(rec["receipts"]).astype(int)
    inv["units_shipped"] = sold
    inv["adjustment_units"] = long(rec["adj"]).astype(int)
    inv["on_hand_close"] = long(rec["close"]).astype(int)
    inv["on_order_units"] = long(rec["on_order"]).astype(int)
    inv["stockout_flag"] = (lost > 0).astype(int)
    avg28 = (sales.groupby(["sku_id", "dc_id"]).units_sold
             .transform(lambda s: s.rolling(28, min_periods=7).mean()).fillna(sales.units_sold))
    inv["days_of_cover"] = np.round(inv.on_hand_close / np.maximum(avg28, 1.0), 1)

    to_date = lambda t: (start + pd.Timedelta(days=int(t))).normalize()  # noqa: E731
    po_rows = []
    for k, p in enumerate(pos, start=1):
        arrived = p["arrival_t"] <= T - 1
        delay = max(0, p["arrival_t"] - p["expected_t"])
        if arrived:
            status = "RECEIVED"
        else:
            status = "DELAYED" if delay > 0 else "IN_TRANSIT"
        po_rows.append(dict(
            po_id=f"PO-{k:06d}", sku_id=p["sku_id"], dc_id=p["dc_id"], supplier_id=p["supplier_id"],
            po_type=p["po_type"], order_date=to_date(p["order_t"]),
            expected_arrival_date=to_date(p["expected_t"]), revised_eta_date=to_date(p["arrival_t"]),
            actual_arrival_date=to_date(p["arrival_t"]) if arrived else pd.NaT,
            quantity_units=p["quantity_units"], status=status, delay_days=int(delay)))
    po_df = pd.DataFrame(po_rows)

    anomalies_df = pd.DataFrame([
        dict(date=dates[t], sku_id=series[i][0].sku_id, dc_id=series[i][1].dc_id,
             anomaly_type=kind, magnitude=mag, note=note)
        for t, i, kind, mag, note in anomalies]).sort_values("date").reset_index(drop=True)

    # ---- Weekly channel split -------------------------------------------
    wk = sales.assign(week_start=sales.date - pd.to_timedelta(sales.date.dt.day_of_week, unit="D"))
    wk = wk.groupby(["week_start", "sku_id", "dc_id"], as_index=False).agg(units=("units_sold", "sum"),
                                                                          rev=("net_revenue_ron", "sum"))
    shares = np.array(list(CHANNELS.values()))
    ch_rows = []
    weeks_idx = (wk.week_start - wk.week_start.min()).dt.days.values / 7
    ecom_growth = 1 + 0.004 * weeks_idx  # e-commerce gains share over time
    for r, g in zip(wk.itertuples(index=False), ecom_growth):
        sh = shares.copy()
        sh[-1] *= g
        sh = rng.dirichlet(sh / sh.sum() * 300)
        units = np.floor(sh * r.units).astype(int)
        units[0] += r.units - units.sum()
        for ch, u, s in zip(CHANNELS, units, sh):
            ch_rows.append((r.week_start, r.sku_id, r.dc_id, ch, int(u), round(r.rev * s, 2)))
    channel_df = pd.DataFrame(ch_rows, columns=["week_start", "sku_id", "dc_id", "channel", "units_sold", "net_revenue_ron"])

    lanes = []
    for (a, b), (days, cost) in LANES.items():
        lanes += [(a, b, days, cost), (b, a, days, cost)]
    lanes_df = pd.DataFrame(lanes, columns=["from_dc_id", "to_dc_id", "transit_days", "cost_per_unit_ron"])

    meta = pd.DataFrame([
        ("as_of_date", end.strftime("%Y-%m-%d")),
        ("history_start_date", start.strftime("%Y-%m-%d")),
        ("currency", "RON"),
        ("seed", str(seed)),
        ("generated_at_utc", datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")),
        ("disclaimer", "Fully synthetic data generated for a portfolio project; not real P&G data."),
    ], columns=["key", "value"])

    dim_sku_out = dim_sku.drop(columns=["base_units_per_day", "promo_elasticity", "competitor_elasticity", "annual_growth"])
    return {
        "dim_sku": dim_sku_out,
        "dim_dc": dim_dc.drop(columns=["size_factor"]),
        "dim_supplier": dim_supplier,
        "dim_inventory_policy": dim_policy,
        "dim_transfer_lanes": lanes_df,
        "fact_daily_sales": sales,
        "fact_inventory_daily": inv,
        "fact_purchase_orders": po_df,
        "fact_promo_calendar": promos,
        "fact_logistics_disruptions": disruptions,
        "fact_channel_sales_weekly": channel_df,
        "meta_dataset_info": meta,
        "meta_injected_anomalies": anomalies_df,
    }


# --------------------------------------------------------------------------- #
# Persistence
# --------------------------------------------------------------------------- #
DATE_COLS = {"date", "start_date", "end_date", "order_date", "expected_arrival_date", "revised_eta_date",
             "actual_arrival_date", "week_start"}


def write_duckdb(tables: dict[str, pd.DataFrame], db_path: Path) -> None:
    db_path.parent.mkdir(parents=True, exist_ok=True)
    tmp = db_path.with_suffix(".tmp.duckdb")
    tmp.unlink(missing_ok=True)
    con = duckdb.connect(str(tmp))
    try:
        for name, df in tables.items():
            con.register("_df", df)
            cols = ", ".join(f'CAST("{c}" AS DATE) AS "{c}"' if c in DATE_COLS else f'"{c}"' for c in df.columns)
            con.execute(f"CREATE TABLE {name} AS SELECT {cols} FROM _df")
            con.unregister("_df")
            doc, col_docs = TABLE_DOCS.get(name, ("", {}))
            if doc:
                con.execute(f"COMMENT ON TABLE {name} IS '{doc.replace(chr(39), chr(39) * 2)}'")
            for col, cdoc in col_docs.items():
                if col in df.columns:
                    con.execute(f"COMMENT ON COLUMN {name}.{col} IS '{cdoc.replace(chr(39), chr(39) * 2)}'")
        con.execute("CREATE INDEX idx_sales ON fact_daily_sales(sku_id, dc_id, date)")
        con.execute("CREATE INDEX idx_inv ON fact_inventory_daily(sku_id, dc_id, date)")
        con.execute("CHECKPOINT")
    finally:
        con.close()
    tmp.replace(db_path)
    tmp.with_suffix(".duckdb.wal").unlink(missing_ok=True)


def main(argv: list[str] | None = None) -> Path:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--db", type=Path, default=DEFAULT_DB)
    ap.add_argument("--end-date", default=DEFAULT_END, help="As-of date (last closed business day)")
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args(argv)

    end = pd.Timestamp(args.end_date).normalize()
    tables = simulate(end, args.seed)
    write_duckdb(tables, args.db)
    print(f"Wrote {args.db}")
    for name, df in tables.items():
        print(f"  {name:<28} {len(df):>8,} rows")
    return args.db


if __name__ == "__main__":
    main()
