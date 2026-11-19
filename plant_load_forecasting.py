"""Plant hourly-load forecasting: data loading, ML models, DL models, evaluation, P90 uncertainty.

This module has NO Streamlit code, NO charts and NO cost optimisation.
The dashboard layer (plant_load_visualization.py) imports the public functions below.

Input CSV columns: Timestamps and kw (hourly observations).

File layout
-----------
1. SETTINGS            constants, model lists, saved best parameters
2. DATA LOADING        read_hourly_series
3. PART A - ML MODELS  XGBoost (sklearn / sktime / darts / mlforecast), Holt-Winters, TBATS
4. PART B - DL MODELS  Darts TiDE / TSMixer / TFT
5. METRICS + UNCERTAINTY  error scores and the rolling P90 error band
6. ORCHESTRATION       run_model_comparison (holdout test + refit + next 7 days)
"""
from __future__ import annotations

import math
import time
from contextlib import contextmanager
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

# =====================================================================================
# 1. SETTINGS
# =====================================================================================
HORIZON_HOURS = 168                 # forecast length and holdout length (7 days)
MIN_HISTORY = 168                   # hours of history the manual sklearn features need
BEST_MODEL_NAME = "sktime_XGBoost"  # model used by the cost optimisation and the P90 band

# Rolling uncertainty (same method as the notebook): P90 of the absolute error over a rolling window.
UNCERTAINTY_WINDOW_HOURS = 60 * 24  # 60 days of hourly errors (uses fewer when less history exists)
UNCERTAINTY_QUANTILE = 0.90         # P90

# Model lists are kept apart so ML and DL are two independent parts of the code.
ML_MODEL_ORDER = [
    "sklearn_XGBoost_manual",
    "sktime_XGBoost",
    "darts_XGBoost",
    "mlforecast_XGBoost",
    "sktime_HoltWinters",
    "TBATS",
]
DL_MODEL_ORDER = ["darts_TiDE", "darts_TSMixer", "darts_TFT"]
MODEL_ORDER = ML_MODEL_ORDER + DL_MODEL_ORDER
DEEP_MODELS = DL_MODEL_ORDER        # kept under the old name for the dashboard layer

# Saved best parameters from the notebook (TUNE=0) - ML models.
ML_PARAMS = {
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
    # The notebook implementation uses a DAILY season (24 hours) for Holt-Winters.
    "sktime_HoltWinters": {"trend": "add", "damped": False, "seasonal": "mul", "sp": 24},
    "TBATS": {"periods": "daily", "box_cox": True, "trend": False},
}

# Deep-learning models: hyperparameters copied from the framework-comparison notebook.
DL_PARAMS = {
    "darts_TiDE": {"hidden_size": 256, "num_encoder_layers": 2, "num_decoder_layers": 2, "dropout": 0.1, "lr": 1e-3},
    "darts_TSMixer": {"hidden_size": 64, "ff_size": 64, "num_blocks": 2, "dropout": 0.1, "lr": 1e-3},
    "darts_TFT": {"hidden_size": 32, "lstm_layers": 1, "num_attention_heads": 4, "hidden_continuous_size": 16,
                  "dropout": 0.1, "lr": 1e-3},
}
BEST_PARAMS = {**ML_PARAMS, **DL_PARAMS}   # one merged lookup for code that does not care about the family

# Shared training settings for the deep models (same values as the notebook configuration cell).
DEEP_INPUT_HOURS = 48        # history window seen by the models (2 days)
DEEP_EPOCHS = 15             # notebook used 60 on a GPU; lowered so CPU hosts finish in reasonable time
DEEP_BATCH_SIZE = 64
DEEP_LR = 1e-3
DEEP_PATIENCE = 3
DEEP_VAL_HOURS = 240         # last 10 days of the training data, used for early stopping only
DEEP_SEED = 42
CAL_COLS = ["tod_sin", "tod_cos", "dow_sin", "dow_cos", "weekend"]


# =====================================================================================
# 2. DATA LOADING
# =====================================================================================
def read_hourly_series(file_path: Path | str) -> pd.Series:
    """Read the hourly CSV and return a clean, gap-free hourly kW series.

    Accepts common column-name aliases (Timestamps/Timestamp/datetime/ds and kw/load_kw/load/total_kw).
    Raises ValueError for duplicates, gaps, negative values or too little history.
    """
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


# =====================================================================================
# 3. PART A - MACHINE-LEARNING (ML) MODELS
#    Every runner returns (forecast_array, fit_seconds, predict_seconds, feature_build_seconds)
# =====================================================================================
def _xgb_kwargs(params: dict[str, Any]) -> dict[str, Any]:
    """Pick the XGBoost hyperparameters out of a parameter dict and add the fixed runtime options."""
    keys = ["n_estimators", "max_depth", "learning_rate", "subsample", "colsample_bytree",
            "min_child_weight", "reg_lambda", "reg_alpha", "gamma"]
    kwargs = {k: params[k] for k in keys}
    kwargs.update(random_state=42, n_jobs=1, tree_method="hist", verbosity=0)
    return kwargs


def _row_features(hist: np.ndarray, ts: pd.Timestamp) -> list[float]:
    """Hand-made feature row for the sklearn model: lags, rolling stats and calendar features for time `ts`."""
    lags = [1, 2, 3, 6, 12, 24, 48, 72, 168]
    f = [hist[-lag] for lag in lags]
    w24 = hist[-24:]
    f += [hist[-6:].mean(), w24.mean(), w24.std(), hist[-168:].mean(), w24.min(), w24.max()]
    hr, dw = ts.hour, ts.dayofweek
    f += [hr, dw, int(dw >= 5), math.sin(2 * math.pi * hr / 24), math.cos(2 * math.pi * hr / 24),
          math.sin(2 * math.pi * dw / 7), math.cos(2 * math.pi * dw / 7)]
    return f


def _run_sklearn_xgb_manual(y_hist: pd.Series, horizon: int):
    """sklearn XGBoost with manually built features; forecasts recursively (each prediction feeds the next row)."""
    from xgboost import XGBRegressor
    p = ML_PARAMS["sklearn_XGBoost_manual"]
    future_idx = pd.date_range(y_hist.index[-1] + pd.Timedelta(hours=1), periods=horizon, freq="h")
    vals = y_hist.to_numpy(dtype=float)
    # build the training matrix by hand
    t_feat = time.perf_counter()
    X_train = np.asarray([_row_features(vals[:i], y_hist.index[i]) for i in range(MIN_HISTORY, len(vals))])
    y_train = vals[MIN_HISTORY:]
    feature_build_s = time.perf_counter() - t_feat
    model = XGBRegressor(**_xgb_kwargs(p))
    t = time.perf_counter(); model.fit(X_train, y_train); fit_s = time.perf_counter() - t
    # recursive forecast
    hist, out = list(vals), []
    t = time.perf_counter()
    for ts in future_idx:
        pred = float(model.predict(np.asarray(_row_features(np.asarray(hist), ts)).reshape(1, -1))[0])
        out.append(pred)
        hist.append(pred)
    predict_s = time.perf_counter() - t
    return np.asarray(out), fit_s, predict_s, feature_build_s


def _run_sktime_xgb(y_hist: pd.Series, horizon: int):
    """sktime XGBoost: make_reduction on a 24-hour window plus hour-of-day / day-of-week features (recursive)."""
    from xgboost import XGBRegressor
    from sktime.forecasting.compose import make_reduction, ForecastingPipeline
    from sktime.transformations.series.date import DateTimeFeatures
    p = ML_PARAMS["sktime_XGBoost"]
    future_idx = pd.date_range(y_hist.index[-1] + pd.Timedelta(hours=1), periods=horizon, freq="h")
    window = {"w24": 24, "w48": 48, "w168": 168}[p["lags"]]
    reg = make_reduction(XGBRegressor(**_xgb_kwargs(p)), window_length=window, strategy="recursive")
    fc = ForecastingPipeline(steps=[
        ("dt", DateTimeFeatures(ts_freq="H", manual_selection=["hour_of_day", "day_of_week"])),
        ("reg", reg),
    ])
    X_train = pd.DataFrame({"dummy": 0.0}, index=y_hist.index)    # dummy exogenous frame so the calendar step has an index
    X_future = pd.DataFrame({"dummy": 0.0}, index=future_idx)
    t = time.perf_counter(); fc.fit(y_hist, X=X_train); fit_s = time.perf_counter() - t
    t = time.perf_counter(); out = fc.predict(fh=np.arange(1, horizon + 1), X=X_future).values.ravel(); predict_s = time.perf_counter() - t
    return np.asarray(out), fit_s, predict_s, 0.0


def _run_darts_xgb(y_hist: pd.Series, horizon: int):
    """Darts XGBModel with automatic lags and cyclic hour / day-of-week encoders."""
    from darts import TimeSeries
    from darts.models import XGBModel
    p = ML_PARAMS["darts_XGBoost"]
    lags = {"w24": 24, "w48": 48, "sel": [-1, -2, -3, -6, -12, -24, -48, -168]}[p["lags"]]
    ts = TimeSeries.from_series(y_hist, freq="h")
    model = XGBModel(
        lags=lags, lags_future_covariates=[0], output_chunk_length=1,
        add_encoders={"cyclic": {"future": ["hour", "dayofweek"]}}, **_xgb_kwargs(p)
    )
    t = time.perf_counter(); model.fit(ts); fit_s = time.perf_counter() - t
    t = time.perf_counter(); out = model.predict(horizon).values().ravel(); predict_s = time.perf_counter() - t
    return np.asarray(out), fit_s, predict_s, 0.0


def _run_mlforecast_xgb(y_hist: pd.Series, horizon: int):
    """mlforecast XGBoost with selected lags, rolling-mean lag transforms and hour / day-of-week date features."""
    from xgboost import XGBRegressor
    from mlforecast import MLForecast
    from mlforecast.lag_transforms import RollingMean, RollingStd
    p = ML_PARAMS["mlforecast_XGBoost"]
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
    fcst = MLForecast(models={"xgb": XGBRegressor(**_xgb_kwargs(p))}, freq="h", lags=lags,
                      lag_transforms=lag_transforms, date_features=["hour", "dayofweek"])
    train_df = pd.DataFrame({"unique_id": "kw", "ds": y_hist.index, "y": y_hist.values})
    t = time.perf_counter(); fcst.fit(train_df, static_features=[]); fit_s = time.perf_counter() - t
    t = time.perf_counter(); out = fcst.predict(horizon)["xgb"].values; predict_s = time.perf_counter() - t
    return np.asarray(out), fit_s, predict_s, 0.0


def _run_sktime_holtwinters(y_hist: pd.Series, horizon: int):
    """sktime Holt-Winters (Exponential Smoothing) with a daily season."""
    from sktime.forecasting.exp_smoothing import ExponentialSmoothing
    p = ML_PARAMS["sktime_HoltWinters"]
    trend = None if p["trend"] == "none" else p["trend"]
    damped = bool(p["damped"]) if trend else False
    model = ExponentialSmoothing(trend=trend, damped_trend=damped, seasonal=p["seasonal"], sp=p["sp"],
                                 initialization_method="estimated")
    t = time.perf_counter(); model.fit(y_hist); fit_s = time.perf_counter() - t
    t = time.perf_counter(); out = model.predict(fh=np.arange(1, horizon + 1)).values.ravel(); predict_s = time.perf_counter() - t
    return np.asarray(out), fit_s, predict_s, 0.0


def _run_tbats(y_hist: pd.Series, horizon: int):
    """TBATS statistical model (Box-Cox, ARMA errors, daily season)."""
    from tbats import TBATS
    p = ML_PARAMS["TBATS"]
    periods = [24] if p["periods"] == "daily" else [24, 168]
    estimator = TBATS(seasonal_periods=periods, use_box_cox=bool(p["box_cox"]),
                      use_trend=bool(p["trend"]), use_damped_trend=bool(p["trend"]),
                      use_arma_errors=True, n_jobs=1)
    t = time.perf_counter(); fitted = estimator.fit(y_hist.values); fit_s = time.perf_counter() - t
    t = time.perf_counter(); out = np.asarray(fitted.forecast(steps=horizon)); predict_s = time.perf_counter() - t
    return out, fit_s, predict_s, 0.0


# name -> runner, so adding an ML model means writing one function and one line here
ML_RUNNERS = {
    "sklearn_XGBoost_manual": _run_sklearn_xgb_manual,
    "sktime_XGBoost": _run_sktime_xgb,
    "darts_XGBoost": _run_darts_xgb,
    "mlforecast_XGBoost": _run_mlforecast_xgb,
    "sktime_HoltWinters": _run_sktime_holtwinters,
    "TBATS": _run_tbats,
}


def run_ml_model(name: str, y_hist: pd.Series, horizon: int = HORIZON_HOURS):
    """Fit one ML model on `y_hist` and forecast `horizon` hours. Returns (forecast, fit_s, predict_s, feature_s)."""
    if name not in ML_RUNNERS:
        raise KeyError(f"Unknown ML model: {name}")
    return ML_RUNNERS[name](y_hist, horizon)


# =====================================================================================
# 4. PART B - DEEP-LEARNING (DL) MODELS  (Darts TiDE / TSMixer / TFT)
# =====================================================================================
def _calendar(index: pd.DatetimeIndex) -> pd.DataFrame:
    """Calendar covariates (time-of-day and day-of-week sin/cos, weekend flag) used by the deep models."""
    tod = (index.hour * 60 + index.minute) / 1440.0
    dow = index.dayofweek
    return pd.DataFrame({
        "tod_sin": np.sin(2 * np.pi * tod), "tod_cos": np.cos(2 * np.pi * tod),
        "dow_sin": np.sin(2 * np.pi * dow / 7), "dow_cos": np.cos(2 * np.pi * dow / 7),
        "weekend": (dow >= 5).astype(float),
    }, index=index)


def _deep_val_len(n: int, horizon: int = HORIZON_HOURS) -> int:
    """Hours held out for early stopping; 0 (no validation) when the history is too short."""
    if n - DEEP_VAL_HOURS >= 2 * (DEEP_INPUT_HOURS + horizon):
        return DEEP_VAL_HOURS
    return 0


@contextmanager
def _trusted_torch_load():
    """Temporarily allow torch.load(weights_only=False).

    PyTorch >= 2.6 refuses to reload Lightning checkpoints. The checkpoints reloaded here were written by this
    process a moment earlier, so it is safe - and only inside this block.
    """
    import torch
    original = torch.load

    def patched(*args, **kwargs):
        kwargs["weights_only"] = False
        return original(*args, **kwargs)

    torch.load = patched
    try:
        yield
    finally:
        torch.load = original


def _run_darts_deep(name: str, y_hist: pd.Series, horizon: int):
    """Train one Darts deep model (TiDE / TSMixer / TFT) and make a direct `horizon`-step forecast.

    Uses calendar future covariates, scaling, early stopping on the last 10 days of the training data and
    restores the best-validation weights when a validation set exists.
    """
    import shutil
    import tempfile
    import torch
    import pytorch_lightning as pl
    from pytorch_lightning.callbacks import EarlyStopping
    from darts import TimeSeries
    from darts.dataprocessing.transformers import Scaler
    from darts.models import TFTModel, TiDEModel, TSMixerModel

    kind = name.split("_", 1)[1]
    cls = {"TiDE": TiDEModel, "TSMixer": TSMixerModel, "TFT": TFTModel}[kind]
    ps = dict(DL_PARAMS[name])
    lr = ps.pop("lr")
    pl.seed_everything(DEEP_SEED, workers=True)

    n = len(y_hist)
    val_len = _deep_val_len(n, horizon)
    i_val = n - val_len                       # train on [:i_val], early-stop on [i_val:]
    full_idx = pd.date_range(y_hist.index[0], periods=n + horizon, freq="h")
    y_ts = TimeSeries.from_series(y_hist, freq="h")
    fut_ts = TimeSeries.from_dataframe(_calendar(full_idx), freq="h")
    sc_y = Scaler().fit(y_ts[:i_val])         # scalers are fitted on training hours only
    y_s = sc_y.transform(y_ts)
    fut_s = Scaler().fit(fut_ts[:i_val]).transform(fut_ts)

    callbacks = [EarlyStopping(monitor="val_loss", patience=DEEP_PATIENCE, min_delta=1e-4, mode="min")] if val_len else []
    work_dir = tempfile.mkdtemp(prefix="plant_darts_")
    model_name = f"{kind.lower()}_app"
    try:
        model = cls(
            input_chunk_length=DEEP_INPUT_HOURS, output_chunk_length=horizon, n_epochs=DEEP_EPOCHS,
            batch_size=DEEP_BATCH_SIZE, optimizer_kwargs={"lr": lr}, use_reversible_instance_norm=True,
            random_state=DEEP_SEED, save_checkpoints=True, force_reset=True, model_name=model_name, work_dir=work_dir,
            pl_trainer_kwargs={"accelerator": "auto", "enable_progress_bar": False,
                               "enable_model_summary": False, "callbacks": callbacks},
            **({"loss_fn": torch.nn.MSELoss()} if kind == "TFT" else {}), **ps,
        )
        fit_kwargs = dict(series=y_s[:i_val], future_covariates=fut_s)
        if val_len:
            fit_kwargs.update(val_series=y_s[i_val:], val_future_covariates=fut_s)
        t = time.perf_counter()
        model.fit(**fit_kwargs)
        if val_len:                            # restore the best-validation weights
            try:
                with _trusted_torch_load():
                    model = cls.load_from_checkpoint(model_name, work_dir=work_dir, best=True)
            except Exception:
                pass                           # keep the last-epoch weights
        fit_s = time.perf_counter() - t
        t = time.perf_counter()
        pred = model.predict(n=horizon, series=y_s, future_covariates=fut_s)
        out = sc_y.inverse_transform(pred).values().ravel()
        predict_s = time.perf_counter() - t
    finally:
        shutil.rmtree(work_dir, ignore_errors=True)   # always remove the temporary checkpoints
    return np.asarray(out, dtype=float), fit_s, predict_s, 0.0


def run_dl_model(name: str, y_hist: pd.Series, horizon: int = HORIZON_HOURS):
    """Fit one DL model on `y_hist` and forecast `horizon` hours. Returns (forecast, fit_s, predict_s, feature_s)."""
    if name not in DL_PARAMS:
        raise KeyError(f"Unknown DL model: {name}")
    return _run_darts_deep(name, y_hist, horizon)


# =====================================================================================
# 5. METRICS + ROLLING P90 UNCERTAINTY
# =====================================================================================
def _run_model(name: str, y_hist: pd.Series, horizon: int = HORIZON_HOURS):
    """Send a model name to the ML part or the DL part. Returns (forecast, fit_s, predict_s, feature_s)."""
    if name in ML_RUNNERS:
        return run_ml_model(name, y_hist, horizon)
    if name in DL_PARAMS:
        return run_dl_model(name, y_hist, horizon)
    raise KeyError(f"Unknown forecasting model: {name}")


def _scores(actual, predicted) -> dict[str, float]:
    """MAE, RMSE, MAPE (%) and WAPE (%) on the hours where both actual and predicted values exist."""
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


def rolling_error_uncertainty(actual: pd.Series, predicted: np.ndarray,
                              window_hours: int = UNCERTAINTY_WINDOW_HOURS,
                              quantile: float = UNCERTAINTY_QUANTILE) -> pd.DataFrame:
    """Rolling MAE and rolling P90 of the absolute forecast error (notebook method).

    Returns a table indexed like `actual` with columns abs_error, rolling_mae, rolling_p90.
    The window is `window_hours` (60 days) with min_periods=1, so it simply uses all available
    errors while less history than the window exists.
    """
    abs_error = (actual - pd.Series(np.asarray(predicted, dtype=float), index=actual.index)).abs()
    rolling = abs_error.rolling(window=window_hours, min_periods=1)
    return pd.DataFrame({
        "abs_error": abs_error,
        "rolling_mae": rolling.mean(),
        "rolling_p90": rolling.quantile(quantile),
    })


# =====================================================================================
# 6. ORCHESTRATION
# =====================================================================================
def _evaluate_model_group(names: list[str], family: str, y: pd.Series, split: int,
                          predictions: dict, future_predictions: dict,
                          timing_rows: list, status_rows: list, families: dict) -> None:
    """Run every model in `names`: score it on the held-out last 168 h, refit on all data, forecast the next 168 h.

    Results are appended in place to the shared dicts / lists. A model that raises is recorded as Failed
    and does not stop the others.
    """
    train_y = y.iloc[:split]
    for name in names:
        try:
            test_pred, fit_test, predict_test, _ = _run_model(name, train_y, HORIZON_HOURS)
            future_pred, fit_full, predict_full, feature_full = _run_model(name, y, HORIZON_HOURS)
            if len(test_pred) != HORIZON_HOURS or len(future_pred) != HORIZON_HOURS:
                raise ValueError("Model returned a forecast with an unexpected length.")
            predictions[name] = np.asarray(test_pred, dtype=float)
            future_predictions[name] = np.asarray(future_pred, dtype=float)
            families[name] = family
            timing_rows.append({
                "model": name,
                "fit_time_s": float(fit_full),
                "manual_feature_build_s": float(feature_full),
                "train_total_s": float(fit_full + feature_full),
                "fit_time_train_only_s": float(fit_test),
                "predict_7d_s": float(predict_full),
                "test_predict_7d_s": float(predict_test),
            })
            status_rows.append({"model": name, "family": family, "status": "Completed", "details": ""})
        except Exception as exc:
            status_rows.append({"model": name, "family": family, "status": "Failed",
                                "details": f"{type(exc).__name__}: {exc}"})


def run_model_comparison(y: pd.Series, include_deep: bool = False) -> dict[str, Any]:
    """Hold out the last 168 h, score all models, refit each on full data and forecast the next 168 h.

    Part A (ML models) always runs; Part B (DL models) runs only when include_deep=True.
    Also builds the rolling P90 uncertainty of the selected model (BEST_MODEL_NAME):
      - "uncertainty": hourly rolling MAE / P90 on the 7-day test window
      - "future_band": next-168-hour forecast +/- the latest rolling P90 (lower edge clipped at 0)
    """
    if len(y) < 2 * HORIZON_HOURS + 1:
        raise ValueError("Not enough history for the 7-day test and training window.")
    split = len(y) - HORIZON_HOURS
    actual = y.iloc[split:]
    test_index = actual.index
    future_index = pd.date_range(y.index[-1] + pd.Timedelta(hours=1), periods=HORIZON_HOURS, freq="h")

    predictions: dict[str, np.ndarray] = {}
    future_predictions: dict[str, np.ndarray] = {}
    timing_rows: list = []
    status_rows: list = []
    families: dict[str, str] = {}

    # ---- Part A: ML models
    _evaluate_model_group(ML_MODEL_ORDER, "ML", y, split, predictions, future_predictions, timing_rows, status_rows, families)
    # ---- Part B: DL models (optional, slow)
    if include_deep:
        _evaluate_model_group(DL_MODEL_ORDER, "DL", y, split, predictions, future_predictions, timing_rows, status_rows, families)

    if BEST_MODEL_NAME not in future_predictions:
        errors = [r["details"] for r in status_rows if r["model"] == BEST_MODEL_NAME]
        raise RuntimeError(
            f"The required cost-optimization model {BEST_MODEL_NAME} could not run. "
            + (errors[0] if errors else "Install the notebook dependencies and retry.")
        )

    # Notebook baselines (not trainable, so no training time).
    baselines = {
        "Baseline_naive_24h": np.tile(y.iloc[split - 24:split].values, HORIZON_HOURS // 24),
        "Baseline_naive_168h": y.iloc[split - HORIZON_HOURS:split].values,
    }
    actual_vs_pred = pd.DataFrame({"Actual": actual.to_numpy()}, index=test_index)
    for name, pred in predictions.items():
        actual_vs_pred[name] = pred
    for name, pred in baselines.items():
        actual_vs_pred[name] = pred
        families[name] = "Baseline"

    metric_rows = [{"model": name, "family": families.get(name, ""), **_scores(actual_vs_pred["Actual"], actual_vs_pred[name])}
                   for name in actual_vs_pred.columns[1:]]
    metrics = pd.DataFrame(metric_rows).sort_values("MAE", na_position="last").reset_index(drop=True)

    future_df = pd.DataFrame(future_predictions, index=future_index)
    future_df.index.name = "timestamp"
    actual_vs_pred.index.name = "timestamp"
    timing = pd.DataFrame(timing_rows)
    if not timing.empty:
        timing = timing.sort_values("train_total_s").reset_index(drop=True)
        timing.insert(1, "speed_rank", timing["train_total_s"].rank(method="min").astype(int))
        timing["x_slower_than_fastest"] = timing["train_total_s"] / max(timing["train_total_s"].min(), 1e-9)

    # ---- rolling P90 uncertainty of the selected model
    uncertainty = rolling_error_uncertainty(actual, predictions[BEST_MODEL_NAME])
    latest_p90 = float(uncertainty["rolling_p90"].iloc[-1])
    latest_mae = float(uncertainty["rolling_mae"].iloc[-1])
    best_future = future_df[BEST_MODEL_NAME]
    future_band = pd.DataFrame({"lower": (best_future - latest_p90).clip(lower=0.0),
                                "upper": best_future + latest_p90}, index=future_index)

    return {
        "series": y,
        "actual_vs_pred": actual_vs_pred,
        "future_forecasts": future_df,
        "metrics": metrics,
        "timing": timing,
        "status": pd.DataFrame(status_rows),
        "best_model": BEST_MODEL_NAME,
        "uncertainty": uncertainty,
        "future_band": future_band,
        "latest_p90": latest_p90,
        "latest_mae": latest_mae,
        "uncertainty_hours_used": int(min(UNCERTAINTY_WINDOW_HOURS, len(uncertainty))),
    }
