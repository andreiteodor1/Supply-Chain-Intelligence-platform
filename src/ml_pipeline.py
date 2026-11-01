"""Machine-learning layer: 14-day demand forecaster, anomaly detector and the
deterministic inventory maths built on top of them.

Demand forecaster
-----------------
One *global* ``HistGradientBoostingRegressor`` across all SKU x DC series,
trained as a **direct multi-horizon** model: each training row is
(forecast origin t0, horizon h in 1..14) -> demand on day t0 + h.

* Target is unconstrained demand (``units_demanded``), not ``units_sold``:
  training on censored sales would teach the model that stockouts are low
  demand and bake them into the next forecast.
* Every demand feature and the target are divided by the series' trailing
  28-day mean at the origin, so one model serves 20-unit and 130-unit SKUs.
* Known-in-advance drivers (promo calendar, weekday, seasonality) are taken
  at the *target* date, which is why the model can anticipate a planned promo.
* A second model with quantile loss gives the P90 used for safety stock and
  the pessimistic stockout scenario.

Anomaly detector
----------------
``IsolationForest`` on scale-free daily signals: demand velocity vs the
trailing 28-day and same-weekday baselines, the unexplained inventory
movement (close - (open + receipts - shipped)), promo depth and competitor
price moves. Scored against the generator's ground-truth anomaly log.

Run ``python -m src.ml_pipeline`` to train, evaluate and save the models.
"""
from __future__ import annotations

import json
import math
import time
from dataclasses import dataclass
from functools import lru_cache

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor, IsolationForest

from src import MODELS_DIR
from src.db import as_of_date, db_path, query_df

HORIZON = 14
MIN_HISTORY = 56
TEST_DAYS = 56
Z_SERVICE = 1.65  # ~95% cycle service level
Z_P90 = 1.2816
OVERSTOCK_COVER_DAYS = 30   # DC stock beyond this many days of forecast demand is tied-up working capital
TARGET_COVER_DAYS = 21

FORECAST_FEATURES = [
    "horizon", "r_lag0", "r_lag1", "r_roll7", "r_roll14", "r_std28", "r_same_dow", "r_same_dow_avg4",
    "promo_share28", "depth_origin", "depth_target", "promo_day_idx", "post_promo_target",
    "dow_target", "doy_sin", "doy_cos", "comp_origin", "comp_change28", "log_scale",
    "sku_code", "dc_code", "cat_code",
]
CATEGORICAL = ["dow_target", "sku_code", "dc_code", "cat_code"]
# Deliberately small: correlated velocity features would out-vote the inventory
# signal, and promo/competitor effects are explained before scoring.
ANOMALY_FEATURES = ["log_vel_promo_adj", "inv_residual_ratio"]
ANOMALY_CONTAMINATION = 0.002

FORECAST_PATH = MODELS_DIR / "demand_forecaster.joblib"
ANOMALY_PATH = MODELS_DIR / "anomaly_detector.joblib"
METRICS_PATH = MODELS_DIR / "model_card.json"


# --------------------------------------------------------------------------- #
# Data access
# --------------------------------------------------------------------------- #
@dataclass
class Panel:
    """Dense (series x day) arrays for fast feature construction."""
    keys: pd.DataFrame          # sku_id, dc_id, category, sku_code, dc_code, cat_code
    dates: pd.DatetimeIndex     # history dates
    y: np.ndarray               # units_demanded [S, T]
    sold: np.ndarray            # units_sold [S, T]
    depth: np.ndarray           # promo depth [S, T + HORIZON + 28] incl. planned promos
    comp: np.ndarray            # competitor price index [S, T]

    @property
    def T(self) -> int:
        return len(self.dates)

    def index_of(self, sku_id: str, dc_id: str) -> int:
        m = (self.keys.sku_id == sku_id) & (self.keys.dc_id == dc_id)
        if not m.any():
            raise KeyError(f"Unknown series {sku_id} @ {dc_id}")
        return int(np.flatnonzero(m.values)[0])


def _cache_key() -> tuple[str, float]:
    p = db_path()
    return str(p), p.stat().st_mtime if p.exists() else 0.0


@lru_cache(maxsize=2)
def _load_panel(_key: tuple[str, float]) -> Panel:
    sales = query_df("""
        SELECT s.date, s.sku_id, s.dc_id, k.category, s.units_demanded, s.units_sold,
               s.promo_depth_pct, s.competitor_price_index
        FROM fact_daily_sales s JOIN dim_sku k USING (sku_id)
        ORDER BY s.sku_id, s.dc_id, s.date""")
    keys = sales[["sku_id", "dc_id", "category"]].drop_duplicates().reset_index(drop=True)
    for col, src in [("sku_code", "sku_id"), ("dc_code", "dc_id"), ("cat_code", "category")]:
        keys[col] = keys[src].astype("category").cat.codes.astype(int)
    dates = pd.DatetimeIndex(sorted(sales.date.unique()))
    S, T = len(keys), len(dates)
    pivot = lambda c: sales[c].to_numpy(dtype=float).reshape(S, T)  # noqa: E731
    y, sold, comp = pivot("units_demanded"), pivot("units_sold"), pivot("competitor_price_index")

    ext = T + HORIZON + 28
    depth = np.zeros((S, ext))
    depth[:, :T] = pivot("promo_depth_pct")
    promos = query_df("""SELECT sku_id, dc_id, start_date, end_date, promo_depth_pct
                         FROM fact_promo_calendar WHERE status IN ('ACTIVE', 'PLANNED')""")
    idx = {(r.sku_id, r.dc_id): i for i, r in keys.iterrows()}
    t0 = dates[0]
    for r in promos.itertuples():
        i = idx[(r.sku_id, r.dc_id)]
        a = max((pd.Timestamp(r.start_date) - t0).days, T)
        b = min((pd.Timestamp(r.end_date) - t0).days, ext - 1)
        if a <= b:
            depth[i, a:b + 1] = r.promo_depth_pct
    return Panel(keys, dates, y, sold, depth, comp)


def load_panel() -> Panel:
    return _load_panel(_cache_key())


# --------------------------------------------------------------------------- #
# Feature engineering
# --------------------------------------------------------------------------- #
def _rolling(a: np.ndarray, w: int) -> tuple[np.ndarray, np.ndarray]:
    """Trailing mean / std over window w ending at t (inclusive), NaN before."""
    S, T = a.shape
    c = np.cumsum(np.pad(a, ((0, 0), (1, 0))), axis=1)
    c2 = np.cumsum(np.pad(a ** 2, ((0, 0), (1, 0))), axis=1)
    mean = np.full((S, T), np.nan)
    std = np.full((S, T), np.nan)
    s1 = c[:, w:] - c[:, :-w]
    s2 = c2[:, w:] - c2[:, :-w]
    mean[:, w - 1:] = s1 / w
    std[:, w - 1:] = np.sqrt(np.maximum(s2 / w - (s1 / w) ** 2, 0))
    return mean, std


def _promo_position(depth: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Day index within the current promo (1-based, 0 = none) and post-promo flag (7 days)."""
    S, L = depth.shape
    day_idx = np.zeros((S, L))
    post = np.zeros((S, L))
    on = depth > 0
    for t in range(L):
        if t > 0:
            day_idx[:, t] = np.where(on[:, t], day_idx[:, t - 1] + 1, 0)
            ended = on[:, t - 1] & ~on[:, t]
            post[:, t] = np.where(on[:, t], 0, np.where(ended, 1, np.where(post[:, t - 1] > 0, post[:, t - 1] + 1, 0)))
        else:
            day_idx[:, t] = on[:, t].astype(float)
    post = ((post > 0) & (post <= 7)).astype(float)
    return day_idx, post


def build_forecast_features(panel: Panel, origins: np.ndarray, with_target: bool = True) -> pd.DataFrame:
    """Rows for every series x origin x horizon (1..HORIZON)."""
    y, S, T = panel.y, len(panel.keys), panel.T
    m7, _ = _rolling(y, 7)
    m14, _ = _rolling(y, 14)
    m28, s28 = _rolling(y, 28)
    promo28, _ = _rolling((panel.depth[:, :T] > 0).astype(float), 28)
    comp28, _ = _rolling(panel.comp, 28)
    day_idx, post = _promo_position(panel.depth)
    base_date = panel.dates[0]

    frames = []
    for h in range(1, HORIZON + 1):
        o = origins[origins + h < (T if with_target else T + HORIZON)]
        if len(o) == 0:
            continue
        td = o + h
        k = math.ceil(h / 7) * 7
        scale = np.maximum(m28[:, o], 1.0)
        same = [y[:, td - k - 7 * j] for j in range(4)]
        tdates = base_date + pd.to_timedelta(td, unit="D")
        doy = tdates.day_of_year.to_numpy()
        f = {
            "series": np.repeat(np.arange(S), len(o)),
            "origin": np.tile(o, S),
            "horizon": np.full(S * len(o), h),
            "scale": scale.ravel(),
            "r_lag0": (y[:, o] / scale).ravel(),
            "r_lag1": (y[:, o - 1] / scale).ravel(),
            "r_roll7": (m7[:, o] / scale).ravel(),
            "r_roll14": (m14[:, o] / scale).ravel(),
            "r_std28": (s28[:, o] / scale).ravel(),
            "r_same_dow": (same[0] / scale).ravel(),
            "r_same_dow_avg4": (np.mean(same, axis=0) / scale).ravel(),
            "promo_share28": promo28[:, o].ravel(),
            "depth_origin": panel.depth[:, o].ravel(),
            "depth_target": panel.depth[:, td].ravel(),
            "promo_day_idx": day_idx[:, td].ravel(),
            "post_promo_target": post[:, td].ravel(),
            "dow_target": np.tile(tdates.day_of_week.to_numpy(), S),
            "doy_sin": np.tile(np.sin(2 * np.pi * doy / 365.25), S),
            "doy_cos": np.tile(np.cos(2 * np.pi * doy / 365.25), S),
            "comp_origin": panel.comp[:, o].ravel(),
            "comp_change28": (panel.comp[:, o] / comp28[:, o]).ravel(),
            "log_scale": np.log(scale).ravel(),
            "sku_code": np.repeat(panel.keys.sku_code.to_numpy(), len(o)),
            "dc_code": np.repeat(panel.keys.dc_code.to_numpy(), len(o)),
            "cat_code": np.repeat(panel.keys.cat_code.to_numpy(), len(o)),
        }
        if with_target:
            f["y"] = y[:, td].ravel()
            f["seasonal_naive"] = same[0].ravel()
            f["ma28"] = m28[:, o].ravel()
        frames.append(pd.DataFrame(f))
    return pd.concat(frames, ignore_index=True)


def build_anomaly_features(panel: Panel, inv: pd.DataFrame) -> pd.DataFrame:
    """One row per series x day (after a 35-day warm-up)."""
    y, S, T = panel.y, len(panel.keys), panel.T
    m28, s28 = _rolling(y, 28)
    prev_m28 = np.pad(m28, ((0, 0), (1, 0)), constant_values=np.nan)[:, :T]
    prev_s28 = np.pad(s28, ((0, 0), (1, 0)), constant_values=np.nan)[:, :T]
    dow_base = np.full((S, T), np.nan)
    dow_base[:, 28:] = np.mean([y[:, 28 - 7 * j:T - 7 * j] for j in range(1, 5)], axis=0)
    comp_prev7 = np.pad(panel.comp, ((0, 0), (7, 0)), mode="edge")[:, :T]

    inv_arr = lambda c: inv[c].to_numpy(dtype=float).reshape(S, T)  # noqa: E731
    open_, rec, shipped, close = (inv_arr(c) for c in ("on_hand_open", "receipts_units", "units_shipped", "on_hand_close"))
    residual = close - (open_ + rec - shipped)

    df = pd.DataFrame({
        "series": np.repeat(np.arange(S), T),
        "t": np.tile(np.arange(T), S),
        "date": np.tile(panel.dates.values, S),
        "sku_id": np.repeat(panel.keys.sku_id.values, T),
        "dc_id": np.repeat(panel.keys.dc_id.values, T),
        "units_demanded": y.ravel(),
        "baseline_28d": prev_m28.ravel(),
        "vel_ratio28": (y / np.maximum(prev_m28, 1)).ravel(),
        "vel_z28": ((y - prev_m28) / np.maximum(prev_s28, 1)).ravel(),
        "vel_ratio_dow": (y / np.maximum(dow_base, 1)).ravel(),
        "inv_residual_units": residual.ravel(),
        "inv_residual_ratio": (residual / np.maximum(open_ + rec, 1)).ravel(),
        "promo_depth": panel.depth[:, :T].ravel(),
        "comp_change7": (panel.comp / comp_prev7).ravel(),
    })
    return df[df.t >= 35].reset_index(drop=True)


def fit_promo_elasticity(feats: pd.DataFrame) -> dict[str, float]:
    """Per-SKU promo elasticity b in velocity ~ exp(b * depth), least squares through the origin."""
    lv = np.log(np.clip(feats.vel_ratio28, 0.05, None))
    out = {}
    for sku, g in feats.assign(lv=lv).groupby("sku_id"):
        g = g[g.promo_depth > 0]
        out[sku] = float((g.lv * g.promo_depth).sum() / (g.promo_depth ** 2).sum()) if len(g) else 0.0
    return out


def anomaly_matrix(feats: pd.DataFrame, elasticity: dict[str, float]) -> pd.DataFrame:
    """Adds the promo-explained velocity column and returns the model input matrix."""
    lift = np.exp(feats.sku_id.map(elasticity).fillna(0) * feats.promo_depth)
    feats["expected_promo_lift"] = lift
    feats["log_vel_promo_adj"] = np.log(np.clip(feats.vel_ratio28 / lift, 0.05, None))
    return feats[ANOMALY_FEATURES].fillna(0.0)


# --------------------------------------------------------------------------- #
# Training
# --------------------------------------------------------------------------- #
def _hgb(loss: str = "squared_error", **kw) -> HistGradientBoostingRegressor:
    return HistGradientBoostingRegressor(
        loss=loss, learning_rate=0.06, max_iter=400, max_leaf_nodes=48, min_samples_leaf=60,
        l2_regularization=1.0, categorical_features=[FORECAST_FEATURES.index(c) for c in CATEGORICAL],
        random_state=7, **kw)


def _wape(y: np.ndarray, yhat: np.ndarray) -> float:
    return float(np.abs(y - yhat).sum() / max(y.sum(), 1e-9))


def _fit_forecasters(df: pd.DataFrame) -> tuple[HistGradientBoostingRegressor, HistGradientBoostingRegressor]:
    X, r = df[FORECAST_FEATURES], df.y / df.scale
    mean_model = _hgb().fit(X, r)
    p90_model = _hgb("quantile", quantile=0.9).fit(X, r)
    return mean_model, p90_model


def train_forecaster(verbose: bool = True) -> dict:
    panel = load_panel()
    T = panel.T
    origins = np.arange(MIN_HISTORY, T - 1)
    data = build_forecast_features(panel, origins)
    data = data.dropna(subset=FORECAST_FEATURES)
    target_t = data.origin + data.horizon
    cutoff = T - TEST_DAYS

    # Time-based backtest: train only on rows whose target is before the cutoff
    # and whose origin is before it, evaluate on origins inside the final window.
    train = data[target_t < cutoff]
    test = data[data.origin >= cutoff]
    t0 = time.time()
    m_mean, m_p90 = _fit_forecasters(train)
    pred = m_mean.predict(test[FORECAST_FEATURES]) * test.scale
    p90 = np.maximum(m_p90.predict(test[FORECAST_FEATURES]) * test.scale, pred)
    y = test.y.to_numpy()

    by_h = (test.assign(err=np.abs(y - pred)).groupby("horizon")
            .apply(lambda g: g.err.sum() / g.y.sum(), include_groups=False))
    promo_mask = test.depth_target > 0
    metrics = {
        "model": "HistGradientBoostingRegressor (global, direct multi-horizon, scale-normalised target)",
        "train_rows": int(len(train)), "test_rows": int(len(test)),
        "test_window": f"last {TEST_DAYS} forecast origins",
        "wape": _wape(y, pred),
        "bias_pct": float((pred.sum() - y.sum()) / y.sum()),
        "wape_promo_days": _wape(y[promo_mask], pred[promo_mask]),
        "wape_baseline_seasonal_naive": _wape(y, test.seasonal_naive.to_numpy()),
        "wape_baseline_ma28": _wape(y, test.ma28.to_numpy()),
        "p90_coverage": float((y <= p90).mean()),
        "wape_by_horizon": {int(h): float(v) for h, v in by_h.items()},
        "fit_seconds": round(time.time() - t0, 1),
    }
    metrics["improvement_vs_seasonal_naive_pct"] = float(
        1 - metrics["wape"] / metrics["wape_baseline_seasonal_naive"])

    # Production models are refit on all available history.
    final_mean, final_p90 = _fit_forecasters(data)
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    joblib.dump({"mean": final_mean, "p90": final_p90, "features": FORECAST_FEATURES,
                 "as_of": str(panel.dates[-1].date())}, FORECAST_PATH)
    if verbose:
        print(f"Forecaster WAPE {metrics['wape']:.1%} (seasonal naive {metrics['wape_baseline_seasonal_naive']:.1%}, "
              f"MA28 {metrics['wape_baseline_ma28']:.1%}), bias {metrics['bias_pct']:+.1%}, "
              f"P90 coverage {metrics['p90_coverage']:.1%}")
    return metrics


def _inventory_frame() -> pd.DataFrame:
    return query_df("""SELECT sku_id, dc_id, date, on_hand_open, receipts_units, units_shipped, on_hand_close
                       FROM fact_inventory_daily ORDER BY sku_id, dc_id, date""")


def train_anomaly_detector(verbose: bool = True) -> dict:
    panel = load_panel()
    feats = build_anomaly_features(panel, _inventory_frame())
    elasticity = fit_promo_elasticity(feats)
    X = anomaly_matrix(feats, elasticity)
    model = IsolationForest(n_estimators=400, contamination=ANOMALY_CONTAMINATION, max_samples=4096,
                            random_state=7).fit(X)
    feats["anomaly_score"] = -model.score_samples(X)
    feats["is_anomaly"] = model.predict(X) == -1

    truth = query_df("SELECT date, sku_id, dc_id, anomaly_type FROM meta_injected_anomalies")
    truth["date"] = pd.to_datetime(truth.date)
    flagged = feats[feats.is_anomaly][["date", "sku_id", "dc_id"]]
    hits = flagged.merge(truth, on=["date", "sku_id", "dc_id"], how="inner")
    in_scope = truth[truth.date >= panel.dates[35]]
    metrics = {
        "model": f"IsolationForest (400 trees, contamination {ANOMALY_CONTAMINATION:.1%}) on "
                 "promo-adjusted velocity + unexplained inventory movement",
        "promo_elasticity_by_sku": {k: round(v, 3) for k, v in elasticity.items()},
        "rows_scored": int(len(feats)), "flagged": int(len(flagged)),
        "injected_anomalies": int(len(in_scope)),
        "precision_vs_injected": float(len(hits) / max(len(flagged), 1)),
        "recall_vs_injected": float(len(hits) / max(len(in_scope), 1)),
        "recall_by_type": {k: float(len(hits[hits.anomaly_type == k]) / max((in_scope.anomaly_type == k).sum(), 1))
                           for k in sorted(in_scope.anomaly_type.unique())},
        "note": "Flags that are not injected anomalies are mostly genuine extremes (promo launches, "
                "competitor price wars, post-stockout rebounds), so precision is a lower bound.",
    }
    joblib.dump({"model": model, "features": ANOMALY_FEATURES, "threshold": float(-model.offset_),
                 "elasticity": elasticity}, ANOMALY_PATH)
    if verbose:
        print(f"Anomaly detector: {metrics['flagged']} flags, precision {metrics['precision_vs_injected']:.0%}, "
              f"recall {metrics['recall_vs_injected']:.0%} vs injected ground truth")
    return metrics


def train_all(verbose: bool = True) -> dict:
    card = {"forecaster": train_forecaster(verbose), "anomaly_detector": train_anomaly_detector(verbose),
            "data_as_of": str(as_of_date().date())}
    METRICS_PATH.write_text(json.dumps(card, indent=2))
    clear_caches()
    return card


def models_ready() -> bool:
    return FORECAST_PATH.exists() and ANOMALY_PATH.exists() and METRICS_PATH.exists()


def ensure_models() -> None:
    if not models_ready():
        train_all(verbose=False)


def model_card() -> dict:
    ensure_models()
    return json.loads(METRICS_PATH.read_text())


def clear_caches() -> None:
    _load_panel.cache_clear()
    _forecast_all.cache_clear()
    _anomaly_scores.cache_clear()
    _load_models.cache_clear()


# --------------------------------------------------------------------------- #
# Inference
# --------------------------------------------------------------------------- #
@lru_cache(maxsize=1)
def _load_models(_mtime: float) -> tuple[dict, dict]:
    return joblib.load(FORECAST_PATH), joblib.load(ANOMALY_PATH)


def _models() -> tuple[dict, dict]:
    ensure_models()
    return _load_models(FORECAST_PATH.stat().st_mtime + ANOMALY_PATH.stat().st_mtime)


@lru_cache(maxsize=2)
def _forecast_all(_key: tuple[str, float], _mtime: float) -> pd.DataFrame:
    panel = load_panel()
    fc, _ = _models()
    X = build_forecast_features(panel, np.array([panel.T - 1]), with_target=False)
    mean = np.maximum(fc["mean"].predict(X[FORECAST_FEATURES]) * X.scale, 0)
    p90 = np.maximum(fc["p90"].predict(X[FORECAST_FEATURES]) * X.scale, mean)
    out = pd.DataFrame({
        "sku_id": panel.keys.sku_id.to_numpy()[X.series],
        "dc_id": panel.keys.dc_id.to_numpy()[X.series],
        "date": panel.dates[-1] + pd.to_timedelta(X.horizon, unit="D"),
        "horizon": X.horizon,
        "forecast_units": np.round(mean, 1),
        "forecast_p90_units": np.round(p90, 1),
        "promo_depth_pct": X.depth_target,
    })
    return out.sort_values(["sku_id", "dc_id", "date"]).reset_index(drop=True)


def forecast_all() -> pd.DataFrame:
    """14-day daily forecast (mean and P90) for every SKU x DC from the as-of date."""
    _models()
    return _forecast_all(_cache_key(), FORECAST_PATH.stat().st_mtime)


def forecast_series(sku_id: str, dc_id: str) -> pd.DataFrame:
    f = forecast_all()
    out = f[(f.sku_id == sku_id) & (f.dc_id == dc_id)].reset_index(drop=True)
    if out.empty:
        raise KeyError(f"No forecast for {sku_id} @ {dc_id}")
    return out


@lru_cache(maxsize=2)
def _anomaly_scores(_key: tuple[str, float], _mtime: float) -> pd.DataFrame:
    panel = load_panel()
    _, an = _models()
    feats = build_anomaly_features(panel, _inventory_frame())
    X = anomaly_matrix(feats, an["elasticity"])
    feats["anomaly_score"] = -an["model"].score_samples(X)
    feats["is_anomaly"] = feats.anomaly_score >= an["threshold"]
    feats["anomaly_type"] = np.select(
        [feats.inv_residual_ratio < -0.1, feats.log_vel_promo_adj >= np.log(1.5),
         feats.log_vel_promo_adj <= np.log(0.5)],
        ["INVENTORY_DROP", "VELOCITY_SPIKE", "VELOCITY_DROP"], "MULTIVARIATE_OUTLIER")
    return feats


def score_anomalies(lookback_days: int | None = None, only_flagged: bool = True,
                    sku_id: str | None = None, dc_id: str | None = None) -> pd.DataFrame:
    _models()
    df = _anomaly_scores(_cache_key(), ANOMALY_PATH.stat().st_mtime)
    if lookback_days:
        df = df[df.date > df.date.max() - pd.Timedelta(days=lookback_days)]
    if sku_id:
        df = df[df.sku_id == sku_id]
    if dc_id:
        df = df[df.dc_id == dc_id]
    if only_flagged:
        df = df[df.is_anomaly]
    cols = ["date", "sku_id", "dc_id", "anomaly_type", "anomaly_score", "units_demanded", "baseline_28d",
            "vel_ratio28", "expected_promo_lift", "inv_residual_units", "promo_depth", "comp_change7", "is_anomaly"]
    return df[cols].sort_values("anomaly_score", ascending=False).reset_index(drop=True)


# --------------------------------------------------------------------------- #
# Deterministic inventory maths (all numbers the agents quote come from here)
# --------------------------------------------------------------------------- #
def _snapshot(sku_id: str, dc_id: str) -> dict:
    row = query_df("""
        SELECT i.on_hand_close, i.on_order_units, k.sku_name, k.case_pack, k.unit_cost_ron, k.list_price_ron,
               k.supplier_id, sp.lead_time_mean_days, sp.lead_time_std_days,
               p.reorder_point_units, p.safety_stock_units, p.order_qty_units, p.review_period_days
        FROM fact_inventory_daily i
        JOIN dim_sku k USING (sku_id)
        JOIN dim_supplier sp USING (supplier_id)
        JOIN dim_inventory_policy p USING (sku_id, dc_id)
        WHERE i.sku_id = ? AND i.dc_id = ? AND i.date = (SELECT max(date) FROM fact_inventory_daily)""",
                    [sku_id, dc_id])
    if row.empty:
        raise KeyError(f"Unknown series {sku_id} @ {dc_id}")
    return row.iloc[0].to_dict()


def _open_pos(sku_id: str | None = None, dc_id: str | None = None) -> pd.DataFrame:
    sql = """SELECT po_id, sku_id, dc_id, po_type, order_date, expected_arrival_date, revised_eta_date,
                    quantity_units, status, delay_days
             FROM fact_purchase_orders WHERE status <> 'RECEIVED'"""
    params = []
    if sku_id:
        sql += " AND sku_id = ?"
        params.append(sku_id)
    if dc_id:
        sql += " AND dc_id = ?"
        params.append(dc_id)
    return query_df(sql + " ORDER BY revised_eta_date", params)


def active_disruption_days(sku_id: str, dc_id: str, supplier_id: str) -> tuple[int, pd.DataFrame]:
    d = query_df("""SELECT event_id, disruption_type, severity, added_lead_time_days, start_date, end_date,
                           description, supplier_id, dc_id, sku_id
                    FROM fact_logistics_disruptions
                    WHERE status = 'ACTIVE'
                      AND (supplier_id IS NULL OR supplier_id = ?)
                      AND (dc_id IS NULL OR dc_id = ?)
                      AND (sku_id IS NULL OR sku_id = ?)""", [supplier_id, dc_id, sku_id])
    return int(d.added_lead_time_days.sum()) if not d.empty else 0, d


def _json_rows(df: pd.DataFrame) -> list[dict]:
    return [{k: (v.date().isoformat() if isinstance(v, pd.Timestamp) else v) for k, v in r.items()
             if v is not None and not (isinstance(v, float) and math.isnan(v))} for r in df.to_dict("records")]


def project_stockout(sku_id: str, dc_id: str) -> dict:
    """Roll on-hand stock forward with the ML forecast and open-PO ETAs.

    The projection assumes no *new* orders. What matters operationally is
    whether stock runs out before a PO raised today could land (the effective
    lead time, including active disruptions): only that gap is unavoidable
    through normal replenishment, so lost sales and revenue at risk are
    counted inside that window.
    """
    snap = _snapshot(sku_id, dc_id)
    fc = forecast_series(sku_id, dc_id)
    pos = _open_pos(sku_id, dc_id)
    asof = as_of_date()
    extra_lt, _ = active_disruption_days(sku_id, dc_id, snap["supplier_id"])
    lead_time = int(snap["lead_time_mean_days"]) + extra_lt
    replenish_date = asof + pd.Timedelta(days=lead_time)

    receipts: dict[pd.Timestamp, int] = {}
    for r in pos.itertuples():
        eta = max(pd.Timestamp(r.revised_eta_date), asof + pd.Timedelta(days=1))  # overdue -> tomorrow
        receipts[eta] = receipts.get(eta, 0) + int(r.quantity_units)

    rows, inv_mean, inv_p90 = [], float(snap["on_hand_close"]), float(snap["on_hand_close"])
    so_mean = so_p90 = None
    lost_lt = lost_lt_p90 = rev_lt = 0.0
    for r in fc.itertuples():
        rec = receipts.get(r.date, 0)
        inv_mean += rec
        inv_p90 += rec
        short, short_p90 = max(r.forecast_units - inv_mean, 0), max(r.forecast_p90_units - inv_p90, 0)
        if r.date <= replenish_date:
            lost_lt += short
            lost_lt_p90 += short_p90
            rev_lt += short * snap["list_price_ron"] * (1 - r.promo_depth_pct)
        if so_mean is None and short > 0:
            so_mean = r.date
        if so_p90 is None and short_p90 > 0:
            so_p90 = r.date
        inv_mean = max(inv_mean - r.forecast_units, 0)
        inv_p90 = max(inv_p90 - r.forecast_p90_units, 0)
        rows.append(dict(date=r.date.date().isoformat(), forecast_units=r.forecast_units,
                         forecast_p90_units=r.forecast_p90_units, promo_depth_pct=r.promo_depth_pct,
                         receipts_units=rec, projected_on_hand=round(inv_mean), projected_on_hand_p90=round(inv_p90)))
    demand14 = float(fc.forecast_units.sum())
    return {
        "sku_id": sku_id, "dc_id": dc_id, "sku_name": snap["sku_name"],
        "as_of_date": asof.date().isoformat(),
        "on_hand_units": int(snap["on_hand_close"]),
        "on_order_units": int(snap["on_order_units"]),
        "forecast_14d_units": round(demand14),
        "forecast_14d_p90_units": round(float(fc.forecast_p90_units.sum())),
        "avg_daily_forecast": round(demand14 / HORIZON, 1),
        "days_of_cover": round(snap["on_hand_close"] / max(demand14 / HORIZON, 0.1), 1),
        "effective_lead_time_days": lead_time,
        "earliest_new_po_arrival_date": replenish_date.date().isoformat(),
        "stockout_date_expected": so_mean.date().isoformat() if so_mean is not None else None,
        "stockout_date_p90": so_p90.date().isoformat() if so_p90 is not None else None,
        "days_until_stockout": (so_mean - asof).days if so_mean is not None else None,
        "stockout_before_new_po_can_land": bool(so_mean is not None and so_mean <= replenish_date),
        "stockout_before_new_po_can_land_p90": bool(so_p90 is not None and so_p90 <= replenish_date),
        "lost_units_before_new_po": round(lost_lt),
        "lost_units_before_new_po_p90": round(lost_lt_p90),
        "revenue_at_risk_ron": round(rev_lt, 2),
        "open_purchase_orders": _json_rows(pos),
        "daily_projection": rows,
    }


def _transfer_options(sku_id: str, to_dc: str, needed: int) -> list[dict]:
    lanes = query_df("SELECT from_dc_id, transit_days, cost_per_unit_ron FROM dim_transfer_lanes WHERE to_dc_id = ?",
                     [to_dc])
    opts = []
    for lane in lanes.itertuples():
        snap = _snapshot(sku_id, lane.from_dc_id)
        fc = forecast_series(sku_id, lane.from_dc_id)
        demand = float(fc.forecast_p90_units.sum())
        incoming = int(_open_pos(sku_id, lane.from_dc_id).quantity_units.sum())
        # Donor keeps its own P90 14-day demand plus safety stock before releasing anything.
        keep = demand + snap["safety_stock_units"]
        excess = int(snap["on_hand_close"] + incoming - keep)
        if excess <= 0:
            continue
        qty = int(min(excess, max(needed, 0)) // snap["case_pack"] * snap["case_pack"])
        if qty <= 0:
            continue
        opts.append(dict(from_dc_id=lane.from_dc_id, available_excess_units=excess, transfer_units=qty,
                         transit_days=int(lane.transit_days), cost_ron=round(qty * lane.cost_per_unit_ron, 2),
                         donor_days_of_cover_after=round((snap["on_hand_close"] - qty) / max(demand / HORIZON, 0.1), 1)))
    return sorted(opts, key=lambda o: (o["transit_days"], o["cost_ron"]))


def recommend_replenishment(sku_id: str, dc_id: str) -> dict:
    """Order-up-to recommendation sized on the ML forecast, plus inter-DC transfer options.

    order_qty = demand over (lead time + review period) + safety stock - inventory position,
    rounded up to whole cases. Safety stock uses the forecast's P90 spread as sigma.
    """
    snap = _snapshot(sku_id, dc_id)
    proj = project_stockout(sku_id, dc_id)
    fc = forecast_series(sku_id, dc_id)
    extra_lt, disruptions = active_disruption_days(sku_id, dc_id, snap["supplier_id"])
    lead_time = int(snap["lead_time_mean_days"]) + extra_lt
    review = int(snap["review_period_days"])
    window = lead_time + review

    daily = fc.forecast_units.to_numpy()
    tail = daily[-7:].mean()
    demand_window = float(daily[:window].sum() + max(window - HORIZON, 0) * tail)
    sigma_daily = float(((fc.forecast_p90_units - fc.forecast_units) / Z_P90).mean())
    sigma_lt = math.sqrt(window * sigma_daily ** 2 + (daily.mean() * snap["lead_time_std_days"]) ** 2)
    safety = Z_SERVICE * sigma_lt
    inventory_position = snap["on_hand_close"] + snap["on_order_units"]
    raw = demand_window + safety - inventory_position
    case = int(snap["case_pack"])
    order_qty = int(math.ceil(max(raw, 0) / case) * case)

    # Units a new PO cannot cover in time (from the day-by-day projection) -> candidates for a transfer.
    arrival = as_of_date() + pd.Timedelta(days=lead_time)
    gap = int(proj["lost_units_before_new_po"])
    transfers = _transfer_options(sku_id, dc_id, gap) if gap > 0 else []

    return {
        "sku_id": sku_id, "dc_id": dc_id, "sku_name": snap["sku_name"],
        "supplier_id": snap["supplier_id"],
        "nominal_lead_time_days": int(snap["lead_time_mean_days"]),
        "disruption_added_days": extra_lt,
        "effective_lead_time_days": lead_time,
        "review_period_days": review,
        "coverage_window_days": window,
        "forecast_demand_in_window_units": round(demand_window),
        "safety_stock_units": round(safety),
        "on_hand_units": int(snap["on_hand_close"]),
        "on_order_units": int(snap["on_order_units"]),
        "inventory_position_units": int(inventory_position),
        "recommended_order_units": order_qty,
        "recommended_order_cases": order_qty // case,
        "case_pack": case,
        "order_value_ron": round(order_qty * snap["unit_cost_ron"], 2),
        "expected_po_arrival_date": arrival.date().isoformat(),
        "shortfall_before_po_arrival_units": gap,
        "transfer_options": transfers,
        "current_policy": {"reorder_point_units": int(snap["reorder_point_units"]),
                           "safety_stock_units": int(snap["safety_stock_units"]),
                           "order_qty_units": int(snap["order_qty_units"])},
        "active_disruptions": _json_rows(disruptions),
        "stockout_projection": {k: proj[k] for k in (
            "stockout_date_expected", "stockout_date_p90", "days_until_stockout", "days_of_cover",
            "stockout_before_new_po_can_land", "lost_units_before_new_po", "revenue_at_risk_ron",
            "forecast_14d_units")},
    }


def triage_scan(top_n: int | None = None) -> pd.DataFrame:
    """Rank every SKU x DC by business risk for the morning triage."""
    fc = forecast_all()
    snap = query_df("""
        SELECT i.sku_id, i.dc_id, k.sku_name, k.category, k.list_price_ron, k.unit_cost_ron,
               i.on_hand_close, i.on_order_units
        FROM fact_inventory_daily i JOIN dim_sku k USING (sku_id)
        WHERE i.date = (SELECT max(date) FROM fact_inventory_daily)""")
    agg = fc.groupby(["sku_id", "dc_id"], as_index=False).agg(
        forecast_14d=("forecast_units", "sum"), forecast_14d_p90=("forecast_p90_units", "sum"))
    df = snap.merge(agg, on=["sku_id", "dc_id"])
    an = score_anomalies(lookback_days=7)
    an_count = an.groupby(["sku_id", "dc_id"]).size().rename("anomalies_7d")
    df = df.merge(an_count, on=["sku_id", "dc_id"], how="left").fillna({"anomalies_7d": 0})

    projections = [project_stockout(r.sku_id, r.dc_id) for r in df.itertuples()]
    for key in ("days_until_stockout", "stockout_date_expected", "effective_lead_time_days",
                "stockout_before_new_po_can_land", "stockout_before_new_po_can_land_p90",
                "lost_units_before_new_po", "revenue_at_risk_ron", "days_of_cover"):
        df[key] = [p[key] for p in projections]
    df["excess_units"] = np.where(df.days_of_cover > OVERSTOCK_COVER_DAYS,
                                  df.on_hand_close - df.forecast_14d / HORIZON * TARGET_COVER_DAYS, 0).round()
    df["excess_value_ron"] = (df.excess_units * df.unit_cost_ron).round(2)

    def classify(r) -> str:
        if r.stockout_before_new_po_can_land and r.days_until_stockout <= 7:
            return "CRITICAL_STOCKOUT"      # runs dry within a week and normal ordering cannot save it
        if r.stockout_before_new_po_can_land:
            return "STOCKOUT_RISK"
        if r.days_of_cover > OVERSTOCK_COVER_DAYS:
            return "OVERSTOCK"
        if r.stockout_before_new_po_can_land_p90:
            return "WATCH_P90"              # only fails if demand lands at the P90
        if r.anomalies_7d > 0:
            return "ANOMALY"
        return "HEALTHY"

    df["risk_class"] = df.apply(classify, axis=1)
    weight = {"CRITICAL_STOCKOUT": 4, "STOCKOUT_RISK": 3, "OVERSTOCK": 2, "WATCH_P90": 1, "ANOMALY": 1, "HEALTHY": 0}
    df["risk_score"] = (df.risk_class.map(weight) * 1000 + df.revenue_at_risk_ron / 100 + df.excess_value_ron / 200
                        + df.anomalies_7d * 50).round(1)
    df = df.sort_values("risk_score", ascending=False).reset_index(drop=True)
    return df.head(top_n) if top_n else df


if __name__ == "__main__":
    card = train_all()
    print(json.dumps(card, indent=2))
