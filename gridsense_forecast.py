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
PHASE_FULL_NAME = {"R": "Red", "Y": "Yellow", "B": "Blue"}
NOTEBOOK_SMOOTHING_PARAMS = dict(smoothing_level=0.7, smoothing_trend=0.05, smoothing_seasonal=0.1, damping_trend=0.95)   # alpha / beta / gamma / phi of the load notebook
AVAILABLE_MODELS = ["Exponential Smoothing"]
PHASE_FORECAST_METHODS = ["Each phase separately"]


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


def forecast_one_series(values, steps_ahead, model_name="Exponential Smoothing", season_length=HOURS_PER_WEEK, fixed_smoothing=None):
    """One series -> (forecast >= 0, model label, fit_ok).  A failed fit repeats the last value (and says so)."""
    values = np.asarray(values, float)
    try:
        forecast = _fit_holt_winters_and_forecast(values, steps_ahead, season_length, fixed_smoothing)
        if not np.all(np.isfinite(forecast)): raise ValueError("non-finite forecast")
        return np.clip(forecast, 0, None), _model_label(len(values), season_length), True      # kW can never be negative
    except Exception as error:
        return np.repeat(max(float(values[-1]), 0.0), steps_ahead), f"failed ({type(error).__name__}): last value used", False


def forecast_three_phases(history, steps_ahead, model_name="Exponential Smoothing", season_length=HOURS_PER_WEEK, fixed_smoothing=None, method=PHASE_FORECAST_METHODS[0]):
    """history = DataFrame with columns R, Y, B (gap-free hours).  Returns (steps_ahead x 3 array, {phase: label}, all_ok).
    Each phase is forecast separately (the notebook method)."""
    forecast, labels, all_ok = np.zeros((steps_ahead, 3)), {}, True
    for i, phase in enumerate(PHASE_CODES):
        forecast[:, i], labels[phase], fit_ok = forecast_one_series(history[phase].to_numpy(float), steps_ahead, model_name, season_length, fixed_smoothing); all_ok &= fit_ok
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


def holdout_test_total_load(measured, filled, model_name, season_length, fixed_smoothing, max_training_hours, test_hours=HOURS_PER_WEEK):
    """The load notebook's test: hide the last `test_hours`, learn from the hours before (at most max_training_hours), forecast them, compare with what really happened.
    Returns dict(actual, pred, label, ok, scores) or None when the history is too short."""
    n_hours = len(filled)
    if n_hours < test_hours + 48: return None
    training = filled.iloc[max(0, n_hours - test_hours - max_training_hours): n_hours - test_hours]
    forecast, label, fit_ok = forecast_one_series(training.to_numpy(), test_hours, model_name, season_length, fixed_smoothing)
    test_index = filled.index[n_hours - test_hours:]
    predicted = pd.Series(forecast, index=test_index); actual = measured.reindex(test_index)
    return dict(actual=actual, pred=predicted, label=label, ok=fit_ok, scores=score_forecast(actual.to_numpy(), predicted.to_numpy()))


# ================================================================== 5. BACKTEST OF THE PHASE FORECAST
def _compare_imbalance_tables(actual_table, forecast_table):
    """Score a forecast imbalance table against the actual one; both are gridsense_rules.analyse_phase_balance tables on the same hours."""
    actual_loaded, forecast_loaded = actual_table.Loaded.to_numpy(bool), forecast_table.Loaded.to_numpy(bool); both_loaded = actual_loaded & forecast_loaded
    if not both_loaded.any(): return dict(hours=0, imb_mae=np.nan, alarm_ok=np.nan, culprit_ok=np.nan, culprit_dir_ok=np.nan)
    actual_alarm, forecast_alarm = actual_table.Alarm.to_numpy(bool), forecast_table.Alarm.to_numpy(bool)
    was_in_alarm = both_loaded & actual_alarm
    actual_worst, forecast_worst = actual_table.Worst_phase.to_numpy(), forecast_table.Worst_phase.to_numpy()
    actual_direction, forecast_direction = actual_table.Direction.to_numpy(), forecast_table.Direction.to_numpy()
    right_phase = actual_worst[was_in_alarm] == forecast_worst[was_in_alarm]
    right_phase_and_direction = right_phase & (actual_direction[was_in_alarm] == forecast_direction[was_in_alarm])
    return dict(hours=int(both_loaded.sum()),
                imb_mae=float(np.nanmean(np.abs(actual_table.Imbalance_pct.to_numpy()[both_loaded] - forecast_table.Imbalance_pct.to_numpy()[both_loaded]))),
                alarm_ok=float((actual_alarm[both_loaded] == forecast_alarm[both_loaded]).mean() * 100),      # alarm / no alarm called right
                culprit_ok=float(right_phase.mean() * 100) if was_in_alarm.any() else np.nan,                  # right phase, on the hours that really were in alarm
                culprit_dir_ok=float(right_phase_and_direction.mean() * 100) if was_in_alarm.any() else np.nan)


def backtest_phase_forecast(measured_hours, filled_hours, model_name, season_length, max_training_hours, horizon_hours, n_weeks, analyse_function, progress_callback=None, method=PHASE_FORECAST_METHODS[0]):
    """The imbalance notebook's test: forecast the last n_weeks weeks (one week at a time, using only the hours before each week) and score
    the phase kW, the imbalance % and the 'which phase is the culprit' call - AND score two plain baselines so the forecast has something to beat:
      same hour last week  = the actual phase kW of the week before, repeated
      usual culprit        = always name the phase that was most often in alarm in the training window
    analyse_function(phase_kw_array) -> DataFrame like gridsense_rules.analyse_phase_balance.
    Returns dict(actual, pred, per_phase, imbalance, baseline) or None."""
    n_hours = len(filled_hours); n_weeks = int(min(n_weeks, max(0, (n_hours - max_training_hours) // horizon_hours)))
    if n_weeks < 1: return None
    actual_parts, forecast_parts, same_hour_last_week_parts, usual_culprit_parts = [], [], [], []
    for week_number, weeks_from_end in enumerate(range(n_weeks, 0, -1), 1):
        week_start = n_hours - weeks_from_end * horizon_hours
        history = filled_hours.iloc[max(0, week_start - max_training_hours):week_start]          # only hours BEFORE the tested week
        forecast, _, _ = forecast_three_phases(history, horizon_hours, model_name, season_length, None, method)
        week_index = filled_hours.index[week_start:week_start + horizon_hours]
        actual_parts.append(measured_hours.loc[week_index, PHASE_CODES]); forecast_parts.append(pd.DataFrame(forecast[:len(week_index)], index=week_index, columns=PHASE_CODES))
        last_week = history.iloc[-HOURS_PER_WEEK:][PHASE_CODES].to_numpy(float)                  # same hour last week (hourly data -> 168 h)
        same_hour_last_week_parts.append(pd.DataFrame(np.resize(last_week, (len(week_index), 3)), index=week_index, columns=PHASE_CODES))
        history_table = analyse_function(history[PHASE_CODES].to_numpy(float)); history_alarm = history_table.Alarm.to_numpy(bool)
        culprit_counts = history_table.Worst_phase[history_alarm].value_counts() if history_alarm.any() else history_table.Worst_phase[history_table.Loaded.to_numpy(bool)].value_counts()
        usual_culprit_parts.append(pd.Series(culprit_counts.idxmax() if len(culprit_counts) else None, index=week_index, dtype=object))
        if progress_callback: progress_callback(week_number / n_weeks)
    actual_kw, forecast_kw = pd.concat(actual_parts), pd.concat(forecast_parts)
    same_hour_last_week_kw, usual_culprit = pd.concat(same_hour_last_week_parts), pd.concat(usual_culprit_parts)
    per_phase_scores = pd.DataFrame([{"Phase": PHASE_FULL_NAME[p], **score_forecast(actual_kw[p].to_numpy(), forecast_kw[p].to_numpy())} for p in PHASE_CODES])
    actual_table, forecast_table, naive_table = analyse_function(actual_kw.to_numpy()), analyse_function(forecast_kw.to_numpy()), analyse_function(same_hour_last_week_kw.to_numpy())
    forecast_result = _compare_imbalance_tables(actual_table, forecast_table); baseline_result = _compare_imbalance_tables(actual_table, naive_table)
    was_in_alarm = actual_table.Loaded.to_numpy(bool) & actual_table.Alarm.to_numpy(bool)
    baseline_result["usual_culprit_ok"] = float((actual_table.Worst_phase.to_numpy()[was_in_alarm] == usual_culprit.to_numpy()[was_in_alarm]).mean() * 100) if was_in_alarm.any() else np.nan
    baseline_result["usual_culprit"] = usual_culprit.dropna().mode().iloc[0] if usual_culprit.notna().any() else None
    forecast_result.update(status_ok=forecast_result["alarm_ok"], alarm_hours_actual=int(actual_table.Alarm.sum()), alarm_hours_forecast=int(forecast_table.Alarm.sum()), weeks=n_weeks)
    return dict(actual=actual_kw, pred=forecast_kw, per_phase=per_phase_scores, imbalance=forecast_result, baseline=baseline_result)


# ================================================================== 6. HOW FAR CAN THE FORECAST BE TRUSTED
def classify_forecast_reliability(backtest, alarm_limit_pct):
    """RELIABLE / DEGRADED / UNRELIABLE / INSUFFICIENT DATA from a backtest result - project rules, not universal ones.
    alarm_limit_pct = the alarm limit in %.  A forecast that beats none of the plain baselines is moved one level down."""
    if backtest is None: return None
    forecast_scores, baseline_scores = backtest["imbalance"], backtest["baseline"]
    if forecast_scores["hours"] < 48 or not np.isfinite(forecast_scores["imb_mae"]): return ("INSUFFICIENT DATA", "too few loaded test hours to judge the forecast")
    phase_wape = float(backtest["per_phase"]["WAPE"].mean()); imbalance_mae = forecast_scores["imb_mae"]; alarm_right_pct = forecast_scores["alarm_ok"]; culprit_right_pct = forecast_scores["culprit_ok"]
    baseline_alarm_right_pct = baseline_scores["alarm_ok"]; baseline_imbalance_mae = baseline_scores["imb_mae"]; baseline_culprit_pct = baseline_scores.get("usual_culprit_ok", np.nan)
    baselines_beaten = (int(imbalance_mae < baseline_imbalance_mae) + int(alarm_right_pct > baseline_alarm_right_pct + 2)
                        + int(np.isfinite(culprit_right_pct) and np.isfinite(baseline_culprit_pct) and culprit_right_pct > baseline_culprit_pct + 2))
    culprit_text = (f", right phase named in {culprit_right_pct:.0f} % of alarm hours (always saying {baseline_scores.get('usual_culprit')}: {baseline_culprit_pct:.0f} %)"
                    if np.isfinite(culprit_right_pct) and np.isfinite(baseline_culprit_pct) else "")
    summary = (f"phase kW error {phase_wape:.0f} % (WAPE), imbalance error {imbalance_mae:.1f} points (same hour last week: {baseline_imbalance_mae:.1f}), alarm call right in {alarm_right_pct:.0f} % "
               f"of {forecast_scores['hours']} test hours (same hour last week: {baseline_alarm_right_pct:.0f} %){culprit_text}")
    levels = ["UNRELIABLE", "DEGRADED", "RELIABLE"]
    tolerance = max(2.0, alarm_limit_pct)
    if alarm_right_pct >= 85 and imbalance_mae <= tolerance: level_index = 2
    elif alarm_right_pct < 65 or imbalance_mae > 2 * tolerance: level_index = 0
    else: level_index = 1
    if baselines_beaten == 0 and level_index > 0: level_index -= 1; summary += " - beats none of the simple baselines"      # no better than a naive guess
    return (levels[level_index], summary)
