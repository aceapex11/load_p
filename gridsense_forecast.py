"""GridSense forecasting engine - the two notebooks, as functions (no Streamlit import).

exponential_smoothing_load_forecasting.ipynb          -> total kW: hourly mean, damped-trend additive Holt-Winters with fixed alpha / beta / gamma / phi, last 168 h held out for testing
three_phase_imbalance_detection_forecasting.ipynb     -> Red / Yellow / Blue kW: hourly mean, Exponential Smoothing (damped trend, additive season) and Auto ETS (lowest AICc),
                                                         learns from the last 28 days, tested on past weeks, then the imbalance is calculated from the forecast phases
"""
import numpy as np
import pandas as pd

PH = ["R", "Y", "B"]
NOTEBOOK_FIXED = dict(smoothing_level=0.7, smoothing_trend=0.05, smoothing_seasonal=0.1, damping_trend=0.95)   # ES_ALPHA / BETA / GAMMA / PHI of the load notebook
MODELS = ["Exponential Smoothing", "Auto ETS"]
ETS_SEASON = 24                                                                                                  # Auto ETS: daily pattern (as in the notebook)
ETS_GRID = [(e, t, d, s) for e in ("add", "mul") for (t, d) in ((None, False), ("add", False), ("add", True)) for s in (None, "add", "mul")]


# ------------------------------------------------------------------ hourly data (same idea as both notebooks)
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


# ------------------------------------------------------------------ the two models
def _holt_winters(y, steps, season, fixed):
    from statsmodels.tsa.holtwinters import ExponentialSmoothing
    seasonal = len(y) >= 2 * season
    model = ExponentialSmoothing(y, trend="add", damped_trend=True, seasonal="add" if seasonal else None, seasonal_periods=season if seasonal else None, initialization_method="estimated")
    if fixed:
        kw = {k: v for k, v in fixed.items() if seasonal or k != "smoothing_seasonal"}
        res = model.fit(optimized=False, **kw)
    else: res = model.fit()
    return np.asarray(res.forecast(steps), float)


def _ets_name(e, t, d, s):
    return "ETS({},{},{})".format("A" if e == "add" else "M", "N" if t is None else ("Ad" if d else "A"), "N" if s is None else ("A" if s == "add" else "M"))


def _auto_ets(y, steps, season=ETS_SEASON):
    """Tries every ETS model and keeps the one with the lowest AICc (best fit without over-fitting)."""
    from statsmodels.tsa.exponential_smoothing.ets import ETSModel
    positive = bool(np.all(y > 0)); best = None
    for e, t, d, s in ETS_GRID:
        if not positive and (e == "mul" or s == "mul"): continue
        if s and len(y) < 2 * season: continue
        try:
            res = ETSModel(y, error=e, trend=t, damped_trend=d, seasonal=s, seasonal_periods=season if s else None).fit(disp=False, maxiter=100)
            fc = np.asarray(res.forecast(steps), float)
            if not (np.isfinite(res.aicc) and np.all(np.isfinite(fc))): continue
            if best is None or res.aicc < best[0]: best = (res.aicc, fc, _ets_name(e, t, d, s))
        except Exception: continue
    if best is None: raise RuntimeError("no ETS model could be fitted")
    return best[1], best[2]


def forecast_series(y, steps, model="Exponential Smoothing", season=168, fixed=None):
    """One series -> (forecast >= 0, model label, ok).  A failed fit repeats the last value (and says so)."""
    y = np.asarray(y, float)
    try:
        if model == "Auto ETS": fc, label = _auto_ets(y, steps)
        else: fc, label = _holt_winters(y, steps, season, fixed), ("Holt-Winters" + (" (weekly)" if season == 168 else " (daily)" if season == 24 else f" (season {season} h)") if len(y) >= 2 * season else "Holt-Winters (no season: too little data)")
        if not np.all(np.isfinite(fc)): raise ValueError("non-finite forecast")
        return np.clip(fc, 0, None), label, True
    except Exception as ex:
        return np.repeat(max(float(y[-1]), 0.0), steps), f"failed ({type(ex).__name__}): last value used", False


def forecast_phases(window, steps, model="Exponential Smoothing", season=168, fixed=None):
    """window = DataFrame with columns R, Y, B (gap-free hours).  Returns (steps x 3 array, {phase: label}, all_ok)."""
    out, labels, ok = np.zeros((steps, 3)), {}, True
    for j, p in enumerate(PH):
        out[:, j], labels[p], good = forecast_series(window[p].to_numpy(float), steps, model, season, fixed); ok &= good
    return out, labels, ok


# ------------------------------------------------------------------ scoring
def score(actual, pred):
    """MAE, RMSE, WAPE %, SMAPE % (and R2) on the hours where both exist."""
    a, f = np.asarray(actual, float), np.asarray(pred, float); ok = np.isfinite(a) & np.isfinite(f); a, f = a[ok], f[ok]
    if len(a) == 0: return dict(MAE=np.nan, RMSE=np.nan, WAPE=np.nan, SMAPE=np.nan, R2=np.nan, n=0)
    err = f - a; den = np.abs(a) + np.abs(f); ss = np.sum((a - a.mean()) ** 2)
    return dict(MAE=float(np.mean(np.abs(err))), RMSE=float(np.sqrt(np.mean(err ** 2))), WAPE=float(np.abs(err).sum() / np.abs(a).sum() * 100) if np.abs(a).sum() > 0 else np.nan,
                SMAPE=float(np.mean(np.where(den > 1e-9, 2 * np.abs(err) / np.where(den > 1e-9, den, 1), 0)) * 100), R2=float(1 - np.sum(err ** 2) / ss) if ss > 0 else np.nan, n=int(len(a)))


def holdout_total(y_real, y_model, model, season, fixed, fit_steps, test_hours=168):
    """The load notebook's test: hide the last `test_hours`, learn from the hours before (at most fit_steps), forecast them, compare with what really happened.
    Returns dict(actual, pred, label, scores) or None when the history is too short."""
    n = len(y_model)
    if n < test_hours + 48: return None
    tr = y_model.iloc[max(0, n - test_hours - fit_steps): n - test_hours]
    fc, label, ok = forecast_series(tr.to_numpy(), test_hours, model, season, fixed)
    idx = y_model.index[n - test_hours:]; pred = pd.Series(fc, index=idx); act = y_real.reindex(idx)
    return dict(actual=act, pred=pred, label=label, ok=ok, scores=score(act.to_numpy(), pred.to_numpy()))


def backtest_phases(load_real, load_model, model, season, fit_steps, horizon, n_weeks, analyse_fn, progress=None):
    """The imbalance notebook's test: forecast the last n_weeks weeks (one week at a time, using only the hours before each week) and score
    the phase kW, the imbalance % and the 'which phase is the culprit' call.  analyse_fn(P) -> DataFrame like gridsense_rules.analyse.
    Returns dict(actual, pred, per_phase, imbalance) or None."""
    n = len(load_model); n_weeks = int(min(n_weeks, max(0, (n - fit_steps) // horizon)))
    if n_weeks < 1: return None
    acts, preds = [], []
    for i, k in enumerate(range(n_weeks, 0, -1), 1):
        p_end = n - k * horizon; hist = load_model.iloc[max(0, p_end - fit_steps):p_end]
        fc, _, _ = forecast_phases(hist, horizon, model, season)
        idx = load_model.index[p_end:p_end + horizon]
        acts.append(load_real.loc[idx, PH]); preds.append(pd.DataFrame(fc[:len(idx)], index=idx, columns=PH))
        if progress: progress(i / n_weeks)
    A, F = pd.concat(acts), pd.concat(preds)
    per = pd.DataFrame([{"Phase": {"R": "Red", "Y": "Yellow", "B": "Blue"}[p], **score(A[p].to_numpy(), F[p].to_numpy())} for p in PH])
    ra, rf = analyse_fn(A.to_numpy()), analyse_fn(F.to_numpy()); both = ra.Loaded.to_numpy() & rf.Loaded.to_numpy()
    was = both & ra.Alarm.to_numpy()
    imb = dict(imb_mae=float(np.nanmean(np.abs(ra.Imbalance_pct.to_numpy()[both] - rf.Imbalance_pct.to_numpy()[both]))) if both.any() else np.nan,
               status_ok=float((ra.Alarm.to_numpy()[both] == rf.Alarm.to_numpy()[both]).mean() * 100) if both.any() else np.nan,
               culprit_ok=float((ra.Worst_phase.to_numpy()[was] == rf.Worst_phase.to_numpy()[was]).mean() * 100) if was.any() else np.nan,
               alarm_hours_actual=int(ra.Alarm.sum()), alarm_hours_forecast=int(rf.Alarm.sum()), hours=int(both.sum()), weeks=n_weeks)
    return dict(actual=A, pred=F, per_phase=per, imbalance=imb)


def reliability(bt, thr):
    """RELIABLE / DEGRADED / UNRELIABLE / INSUFFICIENT DATA from a backtest result - project rules, not universal ones."""
    if bt is None: return None
    im = bt["imbalance"]
    if im["hours"] < 48 or not np.isfinite(im["imb_mae"]): return ("INSUFFICIENT DATA", "too few loaded test hours to judge the forecast")
    wape = float(bt["per_phase"]["WAPE"].mean()); ok, mae = im["status_ok"], im["imb_mae"]
    txt = f"phase kW error {wape:.0f} % (WAPE), imbalance error {mae:.1f} points, alarm / no-alarm call right in {ok:.0f} % of {im['hours']} test hours"
    if ok >= 85 and mae <= max(2.0, thr): return ("RELIABLE", txt)
    if ok < 65 or mae > 2 * max(2.0, thr): return ("UNRELIABLE", txt)
    return ("DEGRADED", txt)
