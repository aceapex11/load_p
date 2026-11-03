"""GridSense forecasting engine (no Streamlit import).

Hourly means of the total kW and of the Red / Yellow / Blue kW are forecast with damped-trend additive Holt-Winters (Exponential Smoothing),
tested on hours the model has not seen, and the phase imbalance is calculated from the forecast phases.
"""
import numpy as np
import pandas as pd

PH = ["R", "Y", "B"]


# ------------------------------------------------------------------ hourly data
def hourly_frame(tl, cols, min_readings=45, max_fill=6):
    """tl = cleaned one-minute rows (column Timestamps + cols).  Hourly mean; an hour with fewer than `min_readings` readings counts as missing.
    Short gaps (<= max_fill hours) are filled by a straight line, the rest repeats the nearest hour - for TRAINING ONLY; accuracy is scored on the real hours.
    Returns dict(real, model, n_gap, n_left) or None when there is no usable hour."""
    g = tl.set_index("Timestamps")[list(cols)].resample("1h")
    mean, cnt = g.mean(), g[cols[0]].count()
    real = mean.where(np.repeat((cnt.to_numpy() >= min_readings)[:, None], len(cols), axis=1))
    ok = real.notna().all(axis=1)
    if not ok.any(): return None
    real = real.loc[ok[ok].index[0]:ok[ok].index[-1]]
    filled = real.interpolate(method="time", limit=max_fill, limit_area="inside")
    return dict(real=real, model=filled.ffill().bfill(), n_gap=int(real.isna().any(axis=1).sum()), n_left=int(filled.isna().any(axis=1).sum()))


# ------------------------------------------------------------------ the model
def _holt_winters(y, steps, season, fixed):
    from statsmodels.tsa.holtwinters import ExponentialSmoothing
    seasonal = len(y) >= 2 * season
    model = ExponentialSmoothing(y, trend="add", damped_trend=True, seasonal="add" if seasonal else None, seasonal_periods=season if seasonal else None, initialization_method="estimated")
    if fixed:
        kw = {k: v for k, v in fixed.items() if seasonal or k != "smoothing_seasonal"}
        res = model.fit(optimized=False, **kw)
    else: res = model.fit()
    return np.asarray(res.forecast(steps), float)


def forecast_series(y, steps, season=168, fixed=None):
    """One series -> (forecast >= 0, model label, ok).  A failed fit repeats the last value (and says so)."""
    y = np.asarray(y, float)
    try:
        fc = _holt_winters(y, steps, season, fixed)
        label = ("Holt-Winters" + (" (weekly)" if season == 168 else " (daily)" if season == 24 else f" (season {season} h)")) if len(y) >= 2 * season else "Holt-Winters (no season: too little data)"
        if not np.all(np.isfinite(fc)): raise ValueError("non-finite forecast")
        return np.clip(fc, 0, None), label, True
    except Exception as ex:
        return np.repeat(max(float(y[-1]), 0.0), steps), f"failed ({type(ex).__name__}): last value used", False


def forecast_phases(window, steps, season=168, fixed=None):
    """window = DataFrame with columns R, Y, B (gap-free hours).  Returns (steps x 3 array, {phase: label}, all_ok)."""
    out, labels, ok = np.zeros((steps, 3)), {}, True
    for j, p in enumerate(PH):
        out[:, j], labels[p], good = forecast_series(window[p].to_numpy(float), steps, season, fixed); ok &= good
    return out, labels, ok


# ------------------------------------------------------------------ scoring
def score(actual, pred):
    """MAE, RMSE, WAPE %, SMAPE % (and R2) on the hours where both exist."""
    a, f = np.asarray(actual, float), np.asarray(pred, float); ok = np.isfinite(a) & np.isfinite(f); a, f = a[ok], f[ok]
    if len(a) == 0: return dict(MAE=np.nan, RMSE=np.nan, WAPE=np.nan, SMAPE=np.nan, R2=np.nan, n=0)
    err = f - a; den = np.abs(a) + np.abs(f); ss = np.sum((a - a.mean()) ** 2)
    return dict(MAE=float(np.mean(np.abs(err))), RMSE=float(np.sqrt(np.mean(err ** 2))), WAPE=float(np.abs(err).sum() / np.abs(a).sum() * 100) if np.abs(a).sum() > 0 else np.nan,
                SMAPE=float(np.mean(np.where(den > 1e-9, 2 * np.abs(err) / np.where(den > 1e-9, den, 1), 0)) * 100), R2=float(1 - np.sum(err ** 2) / ss) if ss > 0 else np.nan, n=int(len(a)))


def holdout_total(y_real, y_model, season, fixed, fit_steps, test_hours=168):
    """Hide the last `test_hours`, learn from the hours before (at most fit_steps), forecast them, compare with what really happened.
    Returns dict(actual, pred, label, scores) or None when the history is too short."""
    n = len(y_model)
    if n < test_hours + 48: return None
    tr = y_model.iloc[max(0, n - test_hours - fit_steps): n - test_hours]
    fc, label, ok = forecast_series(tr.to_numpy(), test_hours, season, fixed)
    idx = y_model.index[n - test_hours:]; pred = pd.Series(fc, index=idx); act = y_real.reindex(idx)
    return dict(actual=act, pred=pred, label=label, ok=ok, scores=score(act.to_numpy(), pred.to_numpy()))
