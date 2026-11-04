"""GridSense forecasting engine - the two notebooks, as functions (no Streamlit import).

exponential_smoothing_load_forecasting.ipynb       -> total kW: hourly mean, damped-trend additive Holt-Winters (fixed smoothing values), last 168 h held out for testing
three_phase_imbalance_detection_forecasting.ipynb  -> Red / Yellow / Blue kW: hourly mean, Exponential Smoothing (damped trend, additive season),
                                                      learns from the last 28 days, tested on past weeks, imbalance is then calculated from the forecast phases

Forecasting uses sktime's ExponentialSmoothing (sktime wraps statsmodels' Holt-Winters, so statsmodels must still be installed as its backend).
"""
import numpy as np
import pandas as pd
from sktime.forecasting.base import ForecastingHorizon
from sktime.forecasting.exp_smoothing import ExponentialSmoothing

# ------------------------------------------------------------------ constants
HOURS_PER_DAY = 24
HOURS_PER_WEEK = 168
PHASE_CODES = ["R", "Y", "B"]                                            # column names of the hourly phase tables
NOTEBOOK_SMOOTHING_PARAMS = dict(smoothing_level=0.7, smoothing_trend=0.05, smoothing_seasonal=0.1, damping_trend=0.95)   # alpha / beta / gamma / phi of the load notebook


# ================================================================== 1. HOURLY DATA (same idea as both notebooks)
def build_hourly_frame(minute_rows, value_columns, min_readings_per_hour=45, max_fill_hours=6):
    """minute_rows = cleaned one-minute rows (column Timestamps + value_columns).  Hourly mean; an hour with fewer than `min_readings_per_hour` readings counts as missing.
    Short gaps (<= max_fill_hours hours) are filled by a straight line, longer gaps repeat the nearest hour - for TRAINING ONLY; accuracy is scored on the measured hours.
    Returns dict(measured, filled, n_gap_hours, n_unfilled_hours) or None when there is no usable hour."""
    hourly_groups = minute_rows.set_index("Timestamps")[list(value_columns)].resample("1h")
    hourly_mean, readings_per_hour = hourly_groups.mean(), hourly_groups[value_columns[0]].count()
    enough_readings = np.repeat((readings_per_hour.to_numpy() >= min_readings_per_hour)[:, None], len(value_columns), axis=1)
    measured = hourly_mean.where(enough_readings)                        # hours with too few readings become NaN
    hour_is_valid = measured.notna().all(axis=1)
    if not hour_is_valid.any(): return None
    measured = measured.loc[hour_is_valid[hour_is_valid].index[0]:hour_is_valid[hour_is_valid].index[-1]]      # trim leading / trailing missing hours
    straight_line_filled = measured.interpolate(method="time", limit=max_fill_hours, limit_area="inside")
    return dict(measured=measured, filled=straight_line_filled.ffill().bfill(),
                n_gap_hours=int(measured.isna().any(axis=1).sum()), n_unfilled_hours=int(straight_line_filled.isna().any(axis=1).sum()))


# ================================================================== 2. THE FORECASTING MODEL (sktime Holt-Winters)
def _fit_holt_winters_and_forecast(values, steps_ahead, season_length, fixed_smoothing):
    """Damped-trend additive Holt-Winters through sktime.  A seasonal part is used only when the history holds at least two full seasons.
    fixed_smoothing = dict of smoothing values to keep as they are (no optimisation), or None to let the model optimise them."""
    use_season = len(values) >= 2 * season_length
    smoothing = {}
    if fixed_smoothing: smoothing = {name: value for name, value in fixed_smoothing.items() if use_season or name != "smoothing_seasonal"}   # no gamma without a season
    forecaster = ExponentialSmoothing(trend="add", damped_trend=True, seasonal="add" if use_season else None, sp=season_length if use_season else 1,
                                      initialization_method="estimated", optimized=not fixed_smoothing, **smoothing)
    forecaster.fit(pd.Series(np.asarray(values, float)))
    horizon = ForecastingHorizon(np.arange(1, steps_ahead + 1), is_relative=True)      # 1, 2, ... steps after the last training hour
    return np.asarray(forecaster.predict(horizon), float)


def _model_label(history_length, season_length):
    """Short text that tells which seasonal setting was really used."""
    if history_length < 2 * season_length: return "Holt-Winters (no season: too little data)"
    if season_length == HOURS_PER_WEEK: return "Holt-Winters (weekly)"
    if season_length == HOURS_PER_DAY: return "Holt-Winters (daily)"
    return f"Holt-Winters (season {season_length} h)"


def forecast_one_series(values, steps_ahead, season_length=HOURS_PER_WEEK, fixed_smoothing=None):
    """One series -> (forecast >= 0, model label, fit_ok).  A failed fit repeats the last value (and says so)."""
    values = np.asarray(values, float)
    try:
        forecast = _fit_holt_winters_and_forecast(values, steps_ahead, season_length, fixed_smoothing)
        if not np.all(np.isfinite(forecast)): raise ValueError("non-finite forecast")
        return np.clip(forecast, 0, None), _model_label(len(values), season_length), True      # kW can never be negative
    except Exception as error:
        return np.repeat(max(float(values[-1]), 0.0), steps_ahead), f"failed ({type(error).__name__}): last value used", False


def forecast_three_phases(history, steps_ahead, season_length=HOURS_PER_WEEK, fixed_smoothing=None):
    """history = DataFrame with columns R, Y, B (gap-free hours).  Returns (steps_ahead x 3 array, {phase: label}, all_ok).
    Each phase is forecast separately (the notebook method)."""
    forecast, labels, all_ok = np.zeros((steps_ahead, 3)), {}, True
    for i, phase in enumerate(PHASE_CODES):
        forecast[:, i], labels[phase], fit_ok = forecast_one_series(history[phase].to_numpy(float), steps_ahead, season_length, fixed_smoothing); all_ok &= fit_ok
    return forecast, labels, all_ok


# ================================================================== 4. SCORING
def score_forecast(actual, predicted):
    """MAE, RMSE, WAPE %, SMAPE % (and R2) on the hours where both values exist."""
    actual, predicted = np.asarray(actual, float), np.asarray(predicted, float)
    both_exist = np.isfinite(actual) & np.isfinite(predicted); actual, predicted = actual[both_exist], predicted[both_exist]
    if len(actual) == 0: return dict(MAE=np.nan, RMSE=np.nan, WAPE=np.nan, SMAPE=np.nan, R2=np.nan, n=0)
    error = predicted - actual; abs_sum = np.abs(actual) + np.abs(predicted); total_variance = np.sum((actual - actual.mean()) ** 2)
    wape = float(np.abs(error).sum() / np.abs(actual).sum() * 100) if np.abs(actual).sum() > 0 else np.nan
    smape = float(np.mean(np.where(abs_sum > 1e-9, 2 * np.abs(error) / np.where(abs_sum > 1e-9, abs_sum, 1), 0)) * 100)
    return dict(MAE=float(np.mean(np.abs(error))), RMSE=float(np.sqrt(np.mean(error ** 2))), WAPE=wape, SMAPE=smape,
                R2=float(1 - np.sum(error ** 2) / total_variance) if total_variance > 0 else np.nan, n=int(len(actual)))


def holdout_test_total_load(measured, filled, season_length, fixed_smoothing, max_training_hours, test_hours=HOURS_PER_WEEK):
    """The load notebook's test: hide the last `test_hours`, learn from the hours before (at most max_training_hours), forecast them, compare with what really happened.
    Returns dict(actual, pred, label, ok, scores) or None when the history is too short."""
    n_hours = len(filled)
    if n_hours < test_hours + 48: return None
    training = filled.iloc[max(0, n_hours - test_hours - max_training_hours): n_hours - test_hours]
    forecast, label, fit_ok = forecast_one_series(training.to_numpy(), test_hours, season_length, fixed_smoothing)
    test_index = filled.index[n_hours - test_hours:]
    predicted = pd.Series(forecast, index=test_index); actual = measured.reindex(test_index)
    return dict(actual=actual, pred=predicted, label=label, ok=fit_ok, scores=score_forecast(actual.to_numpy(), predicted.to_numpy()))
