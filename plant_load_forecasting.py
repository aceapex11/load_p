"""Plant load forecasting: notebook-matched model comparison and 7-day forecasts.

The six forecasters and saved hyperparameters mirror energy_forecasting_final.ipynb.
The app uses the saved tuned parameters (the notebook's default TUNE=0 workflow).
Expected CSV columns: Timestamps, kw. Hourly observations are required.
"""
from __future__ import annotations

import hashlib
import math
import time
import traceback
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import plotly.graph_objects as go

HORIZON_HOURS = 168
MIN_HISTORY = 168
BEST_MODEL_NAME = "sktime_XGBoost"
MODEL_ORDER = [
    "sklearn_XGBoost_manual",
    "sktime_XGBoost",
    "darts_XGBoost",
    "mlforecast_XGBoost",
    "sktime_HoltWinters",
    "TBATS",
]

# Saved best parameters from the uploaded notebook (TUNE=0).
BEST_PARAMS = {
    "sklearn_XGBoost_manual": {
        "n_estimators": 250, "max_depth": 5, "learning_rate": 0.05,
        "subsample": 0.7, "colsample_bytree": 0.9, "min_child_weight": 2,
        "reg_lambda": 8.9, "reg_alpha": 1.2, "gamma": 0.99,
    },
    "sktime_XGBoost": {
        "n_estimators": 100, "max_depth": 7, "learning_rate": 0.1,
        "subsample": 0.9, "colsample_bytree": 0.9, "min_child_weight": 2,
        "reg_lambda": 0.03, "reg_alpha": 0.003, "gamma": 4.3,
        "lags": "w24",
    },
    "darts_XGBoost": {
        "n_estimators": 200, "max_depth": 7, "learning_rate": 0.06,
        "subsample": 0.8, "colsample_bytree": 0.56, "min_child_weight": 14,
        "reg_lambda": 0.02, "reg_alpha": 3.4, "gamma": 1.6,
        "lags": "w24",
    },
    "mlforecast_XGBoost": {
        "n_estimators": 400, "max_depth": 3, "learning_rate": 0.01,
        "subsample": 0.7, "colsample_bytree": 0.6, "min_child_weight": 18,
        "reg_lambda": 0.3, "reg_alpha": 0.02, "gamma": 0.3,
        "lags": "sel2", "roll": "basic",
    },
    "sktime_HoltWinters": {
        "trend": "add", "damped": False, "seasonal": "mul", "sp": 7,
    },
    "TBATS": {"periods": "daily", "box_cox": True, "trend": False},
}

TEXT = "#0F172A"
MUTED = "#64748B"
GRID = "#EDF0F6"
ACCENT = "#6366F1"
GREEN = "#10B981"
BLUE = "#3B82F6"
AMBER = "#F59E0B"


def read_hourly_series(file_path: Path | str) -> pd.Series:
    """Read the repository's hourly CSV; accept common column-name aliases."""
    file_path = Path(file_path)
    if not file_path.exists():
        raise FileNotFoundError(f"Required hourly data file was not found: {file_path}")
    raw = pd.read_csv(file_path)
    lookup = {str(c).strip().lower(): c for c in raw.columns}
    time_col = next((lookup[k] for k in ("timestamps", "timestamp", "datetime", "ds") if k in lookup), None)
    load_col = next((lookup[k] for k in ("kw", "load_kw", "load", "total_kw") if k in lookup), None)
    if time_col is None or load_col is None:
        raise ValueError(
            "CSV must contain a timestamp column (Timestamps) and a load column (kw). "
            f"Columns found: {list(raw.columns)}"
        )
    raw[time_col] = pd.to_datetime(raw[time_col], errors="coerce")
    raw[load_col] = pd.to_numeric(raw[load_col], errors="coerce")
    raw = raw.dropna(subset=[time_col, load_col]).sort_values(time_col)
    if raw.empty:
        raise ValueError("No valid timestamp/load rows were found in the CSV.")
    if raw[time_col].duplicated().any():
        raise ValueError("Duplicate timestamps found. Please provide one load value per hour.")
    y = raw.set_index(time_col)[load_col].astype(float).sort_index().asfreq("h")
    if y.isna().any():
        missing = int(y.isna().sum())
        raise ValueError(f"Hourly data has {missing} missing hour(s). Fill or remove gaps before forecasting.")
    if (y < 0).any():
        raise ValueError("Negative load values were found. Please check the source data.")
    if len(y) < 2 * HORIZON_HOURS + 1:
        raise ValueError(
            f"At least {2 * HORIZON_HOURS + 1} continuous hourly observations are required "
            "for the 7-day holdout test and model training."
        )
    y.name = "kw"
    return y


def _xgb_kwargs(params: dict[str, Any]) -> dict[str, Any]:
    from xgboost import XGBRegressor
    keys = ["n_estimators", "max_depth", "learning_rate", "subsample", "colsample_bytree",
            "min_child_weight", "reg_lambda", "reg_alpha", "gamma"]
    kwargs = {k: params[k] for k in keys}
    kwargs.update(random_state=42, n_jobs=1, tree_method="hist", verbosity=0)
    return kwargs


def _row_features(hist: np.ndarray, ts: pd.Timestamp) -> list[float]:
    lags = [1, 2, 3, 6, 12, 24, 48, 72, 168]
    f = [hist[-lag] for lag in lags]
    w24 = hist[-24:]
    f += [hist[-6:].mean(), w24.mean(), w24.std(), hist[-168:].mean(), w24.min(), w24.max()]
    hr, dw = ts.hour, ts.dayofweek
    f += [hr, dw, int(dw >= 5), math.sin(2 * math.pi * hr / 24), math.cos(2 * math.pi * hr / 24),
          math.sin(2 * math.pi * dw / 7), math.cos(2 * math.pi * dw / 7)]
    return f


def _run_model(name: str, y_hist: pd.Series, horizon: int = HORIZON_HOURS):
    """Return forecast array and fit/predict/feature-build timings for one model."""
    p = BEST_PARAMS[name]
    future_idx = pd.date_range(y_hist.index[-1] + pd.Timedelta(hours=1), periods=horizon, freq="h")
    feature_build_s = 0.0

    if name == "sklearn_XGBoost_manual":
        from xgboost import XGBRegressor
        vals = y_hist.to_numpy(dtype=float)
        t_feat = time.perf_counter()
        X_train = np.asarray([_row_features(vals[:i], y_hist.index[i]) for i in range(MIN_HISTORY, len(vals))])
        y_train = vals[MIN_HISTORY:]
        feature_build_s = time.perf_counter() - t_feat
        model = XGBRegressor(**_xgb_kwargs(p))
        t = time.perf_counter(); model.fit(X_train, y_train); fit_s = time.perf_counter() - t
        hist = list(vals)
        out = []
        t = time.perf_counter()
        for ts in future_idx:
            pred = float(model.predict(np.asarray(_row_features(np.asarray(hist), ts)).reshape(1, -1))[0])
            out.append(pred)
            hist.append(pred)
        predict_s = time.perf_counter() - t
        return np.asarray(out), fit_s, predict_s, feature_build_s

    if name == "sktime_XGBoost":
        from xgboost import XGBRegressor
        from sktime.forecasting.compose import make_reduction, ForecastingPipeline
        from sktime.transformations.series.date import DateTimeFeatures
        lag_cfg = {"w24": 24, "w48": 48, "w168": 168}[p["lags"]]
        reg = make_reduction(XGBRegressor(**_xgb_kwargs(p)), window_length=lag_cfg, strategy="recursive")
        fc = ForecastingPipeline(steps=[
            ("dt", DateTimeFeatures(ts_freq="H", manual_selection=["hour_of_day", "day_of_week"])),
            ("reg", reg),
        ])
        X_train = pd.DataFrame({"dummy": 0.0}, index=y_hist.index)
        X_future = pd.DataFrame({"dummy": 0.0}, index=future_idx)
        t = time.perf_counter(); fc.fit(y_hist, X=X_train); fit_s = time.perf_counter() - t
        t = time.perf_counter(); out = fc.predict(fh=np.arange(1, horizon + 1), X=X_future).values.ravel(); predict_s = time.perf_counter() - t
        return np.asarray(out), fit_s, predict_s, feature_build_s

    if name == "darts_XGBoost":
        from darts import TimeSeries
        from darts.models import XGBModel
        lag_cfg = {"w24": 24, "w48": 48, "sel": [-1, -2, -3, -6, -12, -24, -48, -168]}[p["lags"]]
        ts = TimeSeries.from_series(y_hist, freq="h")
        model = XGBModel(
            lags=lag_cfg, lags_future_covariates=[0], output_chunk_length=1,
            add_encoders={"cyclic": {"future": ["hour", "dayofweek"]}}, **_xgb_kwargs(p)
        )
        t = time.perf_counter(); model.fit(ts); fit_s = time.perf_counter() - t
        t = time.perf_counter(); out = model.predict(horizon).values().ravel(); predict_s = time.perf_counter() - t
        return np.asarray(out), fit_s, predict_s, feature_build_s

    if name == "mlforecast_XGBoost":
        from xgboost import XGBRegressor
        from mlforecast import MLForecast
        from mlforecast.lag_transforms import RollingMean, RollingStd
        lags = {
            "w24": list(range(1, 25)),
            "sel": [1, 2, 3, 6, 12, 24, 48, 168],
            "sel2": [1, 2, 3, 4, 5, 6, 12, 18, 24, 48, 72, 168],
        }[p["lags"]]
        lag_transforms = {
            "none": {},
            "basic": {1: [RollingMean(window_size=24)], 24: [RollingMean(window_size=24)]},
            "rich": {1: [RollingMean(window_size=6), RollingMean(window_size=24), RollingStd(window_size=24)],
                     24: [RollingMean(window_size=24), RollingMean(window_size=168)]},
        }[p["roll"]]
        fcst = MLForecast(
            models={"xgb": XGBRegressor(**_xgb_kwargs(p))}, freq="h", lags=lags,
            lag_transforms=lag_transforms, date_features=["hour", "dayofweek"]
        )
        train_df = pd.DataFrame({"unique_id": "kw", "ds": y_hist.index, "y": y_hist.values})
        t = time.perf_counter(); fcst.fit(train_df, static_features=[]); fit_s = time.perf_counter() - t
        t = time.perf_counter(); out = fcst.predict(horizon)["xgb"].values; predict_s = time.perf_counter() - t
        return np.asarray(out), fit_s, predict_s, feature_build_s

    if name == "sktime_HoltWinters":
        from sktime.forecasting.exp_smoothing import ExponentialSmoothing
        trend = None if p["trend"] == "none" else p["trend"]
        damped = bool(p["damped"]) if trend else False
        # The notebook implementation uses daily seasonality (sp=24).
        model = ExponentialSmoothing(
            trend=trend, damped_trend=damped, seasonal=p["seasonal"], sp=24,
            initialization_method="estimated"
        )
        t = time.perf_counter(); model.fit(y_hist); fit_s = time.perf_counter() - t
        t = time.perf_counter(); out = model.predict(fh=np.arange(1, horizon + 1)).values.ravel(); predict_s = time.perf_counter() - t
        return np.asarray(out), fit_s, predict_s, feature_build_s

    if name == "TBATS":
        from tbats import TBATS
        periods = [24] if p["periods"] == "daily" else [24, 168]
        estimator = TBATS(
            seasonal_periods=periods, use_box_cox=bool(p["box_cox"]),
            use_trend=bool(p["trend"]), use_damped_trend=bool(p["trend"]),
            use_arma_errors=True, n_jobs=1
        )
        t = time.perf_counter(); fitted = estimator.fit(y_hist.values); fit_s = time.perf_counter() - t
        t = time.perf_counter(); out = np.asarray(fitted.forecast(steps=horizon)); predict_s = time.perf_counter() - t
        return out, fit_s, predict_s, feature_build_s

    raise KeyError(f"Unknown forecasting model: {name}")


def _scores(actual, predicted) -> dict[str, float]:
    a = np.asarray(actual, dtype=float)
    p = np.asarray(predicted, dtype=float)
    ok = np.isfinite(a) & np.isfinite(p)
    a, p = a[ok], p[ok]
    if not len(a):
        return {"MAE": np.nan, "RMSE": np.nan, "MAPE_pct": np.nan, "WAPE_pct": np.nan}
    err = np.abs(a - p)
    nonzero = np.abs(a) > 1e-9
    mape = float(np.mean(err[nonzero] / np.abs(a[nonzero])) * 100) if nonzero.any() else np.nan
    wape = float(err.sum() / np.abs(a).sum() * 100) if np.abs(a).sum() > 1e-9 else np.nan
    return {
        "MAE": float(err.mean()),
        "RMSE": float(np.sqrt(np.mean((a - p) ** 2))),
        "MAPE_pct": mape,
        "WAPE_pct": wape,
    }


def run_model_comparison(y: pd.Series) -> dict[str, Any]:
    """Hold out last 168h, score models, then refit each on full data for next 168h."""
    if len(y) < 2 * HORIZON_HOURS + 1:
        raise ValueError("Not enough history for the 7-day test and training window.")
    split = len(y) - HORIZON_HOURS
    train_y = y.iloc[:split]
    actual = y.iloc[split:]
    test_index = actual.index
    future_index = pd.date_range(y.index[-1] + pd.Timedelta(hours=1), periods=HORIZON_HOURS, freq="h")

    predictions: dict[str, np.ndarray] = {}
    future_predictions: dict[str, np.ndarray] = {}
    timing_rows = []
    status_rows = []

    for name in MODEL_ORDER:
        try:
            test_pred, fit_test, predict_test, feature_test = _run_model(name, train_y, HORIZON_HOURS)
            future_pred, fit_full, predict_full, feature_full = _run_model(name, y, HORIZON_HOURS)
            if len(test_pred) != HORIZON_HOURS or len(future_pred) != HORIZON_HOURS:
                raise ValueError("Model returned a forecast with an unexpected length.")
            predictions[name] = np.asarray(test_pred, dtype=float)
            future_predictions[name] = np.asarray(future_pred, dtype=float)
            timing_rows.append({
                "model": name,
                "fit_time_s": float(fit_full),
                "manual_feature_build_s": float(feature_full),
                "train_total_s": float(fit_full + feature_full),
                "fit_time_train_only_s": float(fit_test),
                "predict_7d_s": float(predict_full),
                "test_predict_7d_s": float(predict_test),
            })
            status_rows.append({"model": name, "status": "Completed", "details": ""})
        except Exception as exc:
            status_rows.append({"model": name, "status": "Failed", "details": f"{type(exc).__name__}: {exc}"})

    if BEST_MODEL_NAME not in future_predictions:
        errors = [r["details"] for r in status_rows if r["model"] == BEST_MODEL_NAME]
        raise RuntimeError(
            "The required cost-optimization model sktime_XGBoost could not run. "
            + (errors[0] if errors else "Install the notebook dependencies and retry.")
        )

    # Notebook baselines (not trainable models, so training time is not applicable).
    baselines = {
        "Baseline_naive_24h": np.tile(y.iloc[split - 24:split].values, HORIZON_HOURS // 24),
        "Baseline_naive_168h": y.iloc[split - HORIZON_HOURS:split].values,
    }
    actual_vs_pred = pd.DataFrame({"Actual": actual.to_numpy()}, index=test_index)
    for name, pred in predictions.items():
        actual_vs_pred[name] = pred
    for name, pred in baselines.items():
        actual_vs_pred[name] = pred

    metric_rows = []
    for name in actual_vs_pred.columns[1:]:
        metric_rows.append({"model": name, **_scores(actual_vs_pred["Actual"], actual_vs_pred[name])})
    metrics = pd.DataFrame(metric_rows).sort_values("MAE", na_position="last").reset_index(drop=True)

    future_df = pd.DataFrame(future_predictions, index=future_index)
    future_df.index.name = "timestamp"
    actual_vs_pred.index.name = "timestamp"
    timing = pd.DataFrame(timing_rows)
    if not timing.empty:
        timing = timing.sort_values("train_total_s").reset_index(drop=True)
        timing.insert(1, "speed_rank", timing["train_total_s"].rank(method="min").astype(int))
        timing["x_slower_than_fastest"] = timing["train_total_s"] / max(timing["train_total_s"].min(), 1e-9)

    return {
        "series": y,
        "actual_vs_pred": actual_vs_pred,
        "future_forecasts": future_df,
        "metrics": metrics,
        "timing": timing,
        "status": pd.DataFrame(status_rows),
        "best_model": BEST_MODEL_NAME,
    }


def _style_plot(fig: go.Figure, height: int = 390) -> go.Figure:
    fig.update_layout(
        template="plotly_white", height=height, paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)", font=dict(family="Inter, sans-serif", color=TEXT, size=12),
        margin=dict(l=12, r=15, t=72, b=20), hovermode="x unified",
        title=dict(x=0, xanchor="left", font=dict(size=16, color=TEXT)),
        legend=dict(orientation="h", y=1.12, x=0, yanchor="bottom"),
        hoverlabel=dict(bgcolor="white", bordercolor=GRID, font=dict(color=TEXT)),
    )
    fig.update_xaxes(gridcolor=GRID, linecolor="#E2E8F0", tickfont=dict(color=MUTED))
    fig.update_yaxes(gridcolor=GRID, zeroline=False, tickfont=dict(color=MUTED))
    return fig


def _kpi(st_module, col, label, value, sub, icon):
    with col:
        st_module.markdown(
            f'<div class="kpi"><div class="top"><div class="ic">{icon}</div><div class="l">{label}</div></div>'
            f'<div class="v">{value}</div><div class="s">{sub}</div></div>',
            unsafe_allow_html=True,
        )


def _render_forecast_kpis(st_module, result: dict[str, Any]) -> None:
    y = result["series"]
    fc = result["future_forecasts"][BEST_MODEL_NAME]
    forecast_energy = float(fc.sum())  # hourly kW values summed over 1-hour intervals -> kWh
    previous_energy = float(y.iloc[-HORIZON_HOURS:].sum())
    peak = float(fc.max())
    low_count = max(1, int(np.ceil(len(fc) * 0.25)))
    low_average = float(np.sort(fc.to_numpy())[:low_count].mean())
    st_module.markdown('<div class="ph"><b>Plant load forecast · key indicators</b><span>Cost optimization uses sktime_XGBoost</span></div>', unsafe_allow_html=True)
    cols = st_module.columns(4)
    _kpi(st_module, cols[0], "Forecast load · next 7 days", f"{forecast_energy / 1000:,.1f} MWh", f"{forecast_energy:,.0f} kWh · sktime_XGBoost", "🔮")
    _kpi(st_module, cols[1], "Actual load · last 7 days", f"{previous_energy / 1000:,.1f} MWh", f"{previous_energy:,.0f} kWh · historical meter data", "📊")
    _kpi(st_module, cols[2], "Forecast peak demand", f"{peak:,.1f} kW", f"Highest hourly average over next 7 days", "📈")
    _kpi(st_module, cols[3], "Low-demand average", f"{low_average:,.1f} kW", f"Average of the lowest 25% forecast hours ({low_count} hours)", "🌙")


def _render_model_comparison(st_module, result: dict[str, Any]) -> None:
    st_module.markdown('<div class="ph"><b>Model comparison · last 7-day holdout</b><span>Lower error and shorter training time are better</span></div>', unsafe_allow_html=True)
    metrics = result["metrics"].copy()
    timing = result["timing"].copy()
    if not timing.empty:
        metrics = metrics.merge(
            timing[["model", "train_total_s", "predict_7d_s", "speed_rank"]],
            on="model", how="left"
        )
    else:
        metrics["train_total_s"] = np.nan
        metrics["predict_7d_s"] = np.nan
        metrics["speed_rank"] = np.nan
    metrics["Optimization model"] = np.where(metrics["model"] == BEST_MODEL_NAME, "Selected for cost optimization", "Comparison only")
    metrics = metrics.rename(columns={
        "model": "Model", "MAE": "MAE (kW)", "RMSE": "RMSE (kW)",
        "Optimization model": "Dashboard role",
        "MAPE_pct": "MAPE (%)", "WAPE_pct": "WAPE (%)",
        "train_total_s": "Training time (s)", "predict_7d_s": "7-day forecast time (s)",
        "speed_rank": "Speed rank",
    })
    st_module.dataframe(metrics.round(3), use_container_width=True, hide_index=True)
    st_module.caption("Accuracy is measured on the last 168 actual hours. Each model is fitted on earlier hours for testing, then refitted on all available data for the next 168-hour forecast. Baseline rows have no model-training time.")

    status = result["status"]
    failed = status[status["status"] == "Failed"]
    if not failed.empty:
        with st_module.expander(f"Models that could not run ({len(failed)})"):
            st_module.dataframe(failed, use_container_width=True, hide_index=True)

    if not timing.empty:
        fig = go.Figure()
        t = timing.sort_values("train_total_s", ascending=True)
        fig.add_trace(go.Bar(
            x=t["model"], y=t["train_total_s"], name="Training time",
            marker_color=ACCENT, text=t["train_total_s"].map(lambda v: f"{v:.2f}s"), textposition="outside"
        ))
        fig.update_layout(title="Training-time comparison · full historical dataset", xaxis_title="Model", yaxis_title="Seconds")
        st_module.plotly_chart(_style_plot(fig, 330), use_container_width=True, config={"displayModeBar": False})

    # Actual vs predicted on holdout window for every model and baseline.
    test = result["actual_vs_pred"]
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=test.index, y=test["Actual"], name="Actual", line=dict(color=TEXT, width=2.8)))
    palette = [ACCENT, GREEN, BLUE, AMBER, "#8B5CF6", "#EC4899", "#64748B", "#0EA5E9"]
    for i, col in enumerate(test.columns[1:]):
        fig.add_trace(go.Scatter(x=test.index, y=test[col], name=col, line=dict(color=palette[i % len(palette)], width=1.5)))
    fig.update_layout(title="Actual vs predicted · last 7 days", xaxis_title="Timestamp", yaxis_title="Load (kW)")
    st_module.plotly_chart(_style_plot(fig, 430), use_container_width=True, config={"displayModeBar": False})

    # All successful model forecasts for the next 7 days.
    future = result["future_forecasts"]
    fig = go.Figure()
    for i, col in enumerate(future.columns):
        fig.add_trace(go.Scatter(
            x=future.index, y=future[col], name=col,
            line=dict(color=palette[i % len(palette)], width=2.0 if col == BEST_MODEL_NAME else 1.5)
        ))
    fig.update_layout(title="Next 7 days forecast · all successful models", xaxis_title="Timestamp", yaxis_title="Forecast load (kW)")
    st_module.plotly_chart(_style_plot(fig, 430), use_container_width=True, config={"displayModeBar": False})


def render_plant_load_tab(
    st_module, app_dir: Path, cost_module, data_file_name: str = "hourly_energy_consumption.csv"
) -> None:
    """Render the new Plant Load Forecasting & Optimization tab."""
    st_module.markdown(
        '<div class="ph"><b>Plant load forecasting & optimization</b>'
        '<span>7-day demand outlook · cost-saving opportunities · model performance</span></div>',
        unsafe_allow_html=True,
    )
    data_path = Path(app_dir) / data_file_name
    st_module.caption(
        f"Data source: {data_file_name} · stored beside app.py in the GitHub repository. "
        "Required fields: Timestamps and kw."
    )

    if not data_path.exists():
        st_module.error(
            f"Required file not found: {data_path}. Add {data_file_name} "
            "to the same GitHub folder as app.py."
        )
        return

    file_hash = hashlib.sha256(data_path.read_bytes()).hexdigest()
    try:
        y = read_hourly_series(data_path)
    except Exception as exc:
        st_module.error(str(exc))
        return

    top_cols = st_module.columns([1, 1, 2])
    with top_cols[0]:
        st_module.metric("Hourly observations", f"{len(y):,}")
    with top_cols[1]:
        st_module.metric("Data coverage", f"{(y.index[-1] - y.index[0]).days + 1:,} days")
    with top_cols[2]:
        st_module.caption(f"{y.index.min():%d %b %Y %H:%M} → {y.index.max():%d %b %Y %H:%M}")

    session_key = "plant_load_model_result"
    hash_key = "plant_load_model_hash"
    existing = st_module.session_state.get(session_key)
    has_current_result = existing is not None and st_module.session_state.get(hash_key) == file_hash

    if st_module.button("Run model comparison & 7-day optimization", type="primary", key="plant_run_models"):
        try:
            with st_module.spinner("Fitting the forecasting models, evaluating the last 7 days and forecasting the next 7 days…"):
                result = run_model_comparison(y)
            st_module.session_state[session_key] = result
            st_module.session_state[hash_key] = file_hash
            has_current_result = True
        except Exception as exc:
            st_module.error(f"Could not complete the model run: {type(exc).__name__}: {exc}")
            return

    if not has_current_result:
        st_module.info("Click **Run model comparison & 7-day optimization** to train/evaluate the models and build the dashboard.")
        return

    result = st_module.session_state[session_key]
    if BEST_MODEL_NAME not in result["future_forecasts"].columns:
        st_module.error("sktime_XGBoost forecast is unavailable; cost optimization cannot run.")
        return

    with st_module.spinner("Calculating load-shifting opportunities using the sktime_XGBoost forecast…"):
        opt_result = cost_module.optimize_load_shifting(result["future_forecasts"][BEST_MODEL_NAME])

    # Recommendation first, as requested.
    cost_module.render_recommendation(st_module, opt_result)
    st_module.write("")

    # Forecast KPIs.
    _render_forecast_kpis(st_module, result)
    st_module.write("")

    # Forecast path selected for the cost optimization model.
    with st_module.container(border=True):
        st_module.markdown('<div class="ph"><b>Selected model forecast · sktime_XGBoost</b><span>Next 168 hourly values</span></div>', unsafe_allow_html=True)
        hist = result["series"].iloc[-14 * 24:]
        future = result["future_forecasts"][BEST_MODEL_NAME]
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=hist.index, y=hist.values, name="Historical load", line=dict(color=TEXT, width=2)))
        fig.add_trace(go.Scatter(x=future.index, y=future.values, name="7-day forecast", line=dict(color=ACCENT, width=2.7)))
        fig.add_vline(x=result["series"].index[-1], line_dash="dot", line_color=MUTED)
        fig.update_layout(title="Historical load and next 7-day forecast", xaxis_title="Timestamp", yaxis_title="Load (kW)")
        st_module.plotly_chart(_style_plot(fig, 390), use_container_width=True, config={"displayModeBar": False})

    # Optimization KPIs and Plotly graphs.
    cost_module.render_optimization_results(st_module, opt_result)

    # Model evaluation and all-model plots at the end of the page.
    with st_module.container(border=True):
        _render_model_comparison(st_module, result)
