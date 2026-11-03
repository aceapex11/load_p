"""GridSense forecasting engine - the two notebooks, as functions (no Streamlit import).

exponential_smoothing_load_forecasting.ipynb          -> total kW: hourly mean, damped-trend additive Holt-Winters with fixed alpha / beta / gamma / phi, last 168 h held out for testing
three_phase_imbalance_detection_forecasting.ipynb     -> Red / Yellow / Blue kW: hourly mean, Exponential Smoothing (damped trend, additive season)
                                                         learns from the last 28 days, tested on past weeks, then the imbalance is calculated from the forecast phases
"""
import numpy as np
import pandas as pd

PH = ["R", "Y", "B"]
NOTEBOOK_FIXED = dict(smoothing_level=0.7, smoothing_trend=0.05, smoothing_seasonal=0.1, damping_trend=0.95)   # ES_ALPHA / BETA / GAMMA / PHI of the load notebook
MODELS = ["Exponential Smoothing"]

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


def forecast_series(y, steps, model="Exponential Smoothing", season=168, fixed=None):
    """One series -> (forecast >= 0, model label, ok).  A failed fit repeats the last value (and says so)."""
    y = np.asarray(y, float)
    try:
        fc, label = _holt_winters(y, steps, season, fixed), ("Holt-Winters" + (" (weekly)" if season == 168 else " (daily)" if season == 24 else f" (season {season} h)") if len(y) >= 2 * season else "Holt-Winters (no season: too little data)")        if not np.all(np.isfinite(fc)): raise ValueError("non-finite forecast")
        return np.clip(fc, 0, None), label, True
    except Exception as ex:
        return np.repeat(max(float(y[-1]), 0.0), steps), f"failed ({type(ex).__name__}): last value used", False


PHASE_METHODS = ["Each phase separately", "Total × phase shares"]


def _forecast_shares(window, steps, model, season, fixed):
    """The imbalance % only depends on each phase's SHARE of the total, not on the total itself.  So forecast the total kW and the three shares
    separately, then rebuild R / Y / B = total x share.  Errors in the total no longer leak into the imbalance."""
    W = window[PH].to_numpy(float); tot = W.sum(axis=1); pos = tot > 0
    sh = np.where(pos[:, None], W / np.where(pos, tot, 1.0)[:, None], 1 / 3)             # a zero-load hour counts as perfectly balanced
    ftot, ltot, ok = forecast_series(tot, steps, model, season, fixed)
    fs, labels = np.zeros((steps, 3)), {}
    for j, p in enumerate(PH):
        fs[:, j], lab, good = forecast_series(sh[:, j], steps, model, season, fixed); ok &= good; labels[p] = "share: " + lab
    s = fs.sum(axis=1, keepdims=True); fs = np.where(s > 1e-9, fs / np.where(s > 1e-9, s, 1.0), 1 / 3)      # shares must add up to 1
    labels["total"] = ltot
    return ftot[:, None] * fs, labels, ok


def forecast_phases(window, steps, model="Exponential Smoothing", season=168, fixed=None, method=PHASE_METHODS[0]):
    """window = DataFrame with columns R, Y, B (gap-free hours).  Returns (steps x 3 array, {phase: label}, all_ok).
    method "Each phase separately" = the notebook; "Total × phase shares" = forecast the total and the shares, then rebuild the phases."""
    if method == PHASE_METHODS[1]: return _forecast_shares(window, steps, model, season, fixed)
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


def _compare(ra, rf):
    """Score a forecast imbalance table (rf) against the actual one (ra); both are gridsense_rules.analyse tables on the same hours."""
    A_ld, F_ld = ra.Loaded.to_numpy(bool), rf.Loaded.to_numpy(bool); both = A_ld & F_ld
    nan = dict(hours=0, imb_mae=np.nan, alarm_ok=np.nan, sev_ok=np.nan, culprit_ok=np.nan, culprit_dir_ok=np.nan)
    if not both.any(): return nan
    a_al, f_al = ra.Alarm.to_numpy(bool), rf.Alarm.to_numpy(bool)
    a_sv, f_sv = (ra.Status.to_numpy() == "SEVERE"), (rf.Status.to_numpy() == "SEVERE")
    was = both & a_al
    aw, fw = ra.Worst_phase.to_numpy(), rf.Worst_phase.to_numpy(); ad, fd = ra.Direction.to_numpy(), rf.Direction.to_numpy()
    return dict(hours=int(both.sum()),
                imb_mae=float(np.nanmean(np.abs(ra.Imbalance_pct.to_numpy()[both] - rf.Imbalance_pct.to_numpy()[both]))),
                alarm_ok=float((a_al[both] == f_al[both]).mean() * 100),                       # alarm / no alarm called right
                sev_ok=float((a_sv[both] == f_sv[both]).mean() * 100),                         # severe / not severe called right
                culprit_ok=float((aw[was] == fw[was]).mean() * 100) if was.any() else np.nan,   # right phase, on the hours that really were in alarm
                culprit_dir_ok=float(((aw[was] == fw[was]) & (ad[was] == fd[was])).mean() * 100) if was.any() else np.nan)


def backtest_phases(load_real, load_model, model, season, fit_steps, horizon, n_weeks, analyse_fn, progress=None, method=PHASE_METHODS[0]):
    """The imbalance notebook's test: forecast the last n_weeks weeks (one week at a time, using only the hours before each week) and score
    the phase kW, the imbalance % and the 'which phase is the culprit' call - AND score two plain baselines so the forecast has something to beat:
      same hour last week  = the actual phase kW of the week before, repeated
      usual culprit        = always name the phase that was most often in alarm in the training window
    analyse_fn(P) -> DataFrame like gridsense_rules.analyse.
    Returns dict(actual, pred, per_phase, imbalance, baseline) or None."""
    n = len(load_model); n_weeks = int(min(n_weeks, max(0, (n - fit_steps) // horizon)))
    if n_weeks < 1: return None
    acts, preds, naive, usual = [], [], [], []
    for i, k in enumerate(range(n_weeks, 0, -1), 1):
        p_end = n - k * horizon; hist = load_model.iloc[max(0, p_end - fit_steps):p_end]
        fc, _, _ = forecast_phases(hist, horizon, model, season, None, method)
        idx = load_model.index[p_end:p_end + horizon]
        acts.append(load_real.loc[idx, PH]); preds.append(pd.DataFrame(fc[:len(idx)], index=idx, columns=PH))
        last_week = hist.iloc[-168:][PH].to_numpy(float)                                   # same hour last week (hourly data -> 168 h)
        naive.append(pd.DataFrame(np.resize(last_week, (len(idx), 3)), index=idx, columns=PH))
        rh = analyse_fn(hist[PH].to_numpy(float)); al = rh.Alarm.to_numpy(bool)
        top = rh.Worst_phase[al].value_counts() if al.any() else rh.Worst_phase[rh.Loaded.to_numpy(bool)].value_counts()
        usual.append(pd.Series(top.idxmax() if len(top) else None, index=idx, dtype=object))
        if progress: progress(i / n_weeks)
    A, F, N, U = pd.concat(acts), pd.concat(preds), pd.concat(naive), pd.concat(usual)
    per = pd.DataFrame([{"Phase": {"R": "Red", "Y": "Yellow", "B": "Blue"}[p], **score(A[p].to_numpy(), F[p].to_numpy())} for p in PH])
    ra, rf, rn = analyse_fn(A.to_numpy()), analyse_fn(F.to_numpy()), analyse_fn(N.to_numpy())
    imb = _compare(ra, rf); base = _compare(ra, rn)
    was = ra.Loaded.to_numpy(bool) & ra.Alarm.to_numpy(bool)
    base["usual_culprit_ok"] = float((ra.Worst_phase.to_numpy()[was] == U.to_numpy()[was]).mean() * 100) if was.any() else np.nan
    base["usual_culprit"] = U.dropna().mode().iloc[0] if U.notna().any() else None
    imb.update(status_ok=imb["alarm_ok"], alarm_hours_actual=int(ra.Alarm.sum()), alarm_hours_forecast=int(rf.Alarm.sum()),
               severe_hours_actual=int((ra.Status == "SEVERE").sum()), severe_hours_forecast=int((rf.Status == "SEVERE").sum()), weeks=n_weeks)
    return dict(actual=A, pred=F, per_phase=per, imbalance=imb, baseline=base)


def reliability(bt, thr):
    """RELIABLE / DEGRADED / UNRELIABLE / INSUFFICIENT DATA from a backtest result - project rules, not universal ones.
    thr = the alarm limit in %.  The alarm / severe call is judged by the weaker of the two (a chronic 8 % imbalance would make the alarm call
    trivially right).  A forecast that beats none of the plain baselines is moved one level down."""
    if bt is None: return None
    im, bs = bt["imbalance"], bt["baseline"]
    if im["hours"] < 48 or not np.isfinite(im["imb_mae"]): return ("INSUFFICIENT DATA", "too few loaded test hours to judge the forecast")
    wape = float(bt["per_phase"]["WAPE"].mean()); mae = im["imb_mae"]; ok = min(im["alarm_ok"], im["sev_ok"]); cul = im["culprit_ok"]
    b_ok = min(bs["alarm_ok"], bs["sev_ok"]); b_mae = bs["imb_mae"]; b_cul = bs.get("usual_culprit_ok", np.nan)
    wins = int(mae < b_mae) + int(ok > b_ok + 2) + int(np.isfinite(cul) and np.isfinite(b_cul) and cul > b_cul + 2)
    cul_txt = f", right phase named in {cul:.0f} % of alarm hours (always saying {bs.get('usual_culprit')}: {b_cul:.0f} %)" if np.isfinite(cul) and np.isfinite(b_cul) else ""
    txt = (f"phase kW error {wape:.0f} % (WAPE), imbalance error {mae:.1f} points (same hour last week: {b_mae:.1f}), alarm / severe call right in {ok:.0f} % "
           f"of {im['hours']} test hours (same hour last week: {b_ok:.0f} %){cul_txt}")
    lv = ["UNRELIABLE", "DEGRADED", "RELIABLE"]
    if ok >= 85 and mae <= max(2.0, thr): i = 2
    elif ok < 65 or mae > 2 * max(2.0, thr): i = 0
    else: i = 1
    if wins == 0 and i > 0: i -= 1; txt += " - beats none of the simple baselines"
    return (lv[i], txt)
