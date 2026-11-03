"""GridSense rule module - the phase-balance rules used by app.py (no Streamlit, no AI service).

Rebuilt from how app.py and gridsense_forecast.py call it, because the original file was missing.  Functions:
  analyse(P, thr, min_kw)       -> one row per hour/minute: Loaded, Imbalance_pct, Worst_phase, Worst_dev, Direction, Status, Alarm
  transfers(vals, names)        -> [(from, to, amount)] that levels the three phases
  playbook(...)                 -> {"head": text, "level": "good" | "amber" | "bad"}
  best_windows(total, index)    -> quietest and busiest 3-hour window of a forecast
"""
import numpy as np
import pandas as pd

PHASE_NAMES = ["Red", "Yellow", "Blue"]          # P columns are always in the order R, Y, B


def analyse(P, thr=5.0, min_kw=5.0):
    """imbalance % = max(|R-avg|, |Y-avg|, |B-avg|) / avg x 100.   170 / 169 / 160 kW -> 3.81 % (Blue, LOW).
    Loaded  : average kW per phase is at least `min_kw` (below that the machine is off and a % is meaningless).
    Status  : IDLE (not loaded) / OK / ALARM (> thr).   Alarm = ALARM."""
    a = np.atleast_2d(np.asarray(P, float))
    avg = a.mean(axis=1)
    loaded = np.isfinite(avg) & (avg > 0) & (avg >= min_kw)
    dev = a - avg[:, None]
    idx = np.where(loaded[:, None], np.abs(dev), -1.0).argmax(axis=1)
    rows = np.arange(len(a))
    worst_dev = np.where(loaded, dev[rows, idx] / np.where(loaded, avg, 1.0) * 100, np.nan)
    imb = np.abs(worst_dev)
    status = np.where(~loaded, "IDLE", np.where(imb > thr, "ALARM", "OK"))
    return pd.DataFrame({
        "Loaded": loaded,
        "Imbalance_pct": imb,
        "Worst_phase": np.where(loaded, np.array(PHASE_NAMES, dtype=object)[idx], None),
        "Worst_dev": worst_dev,
        "Direction": np.where(loaded, np.where(worst_dev > 0, "HIGH", "LOW"), None),
        "Status": status,
        "Alarm": status == "ALARM",
    })


def transfers(vals, names):
    """Amounts to move from phases above the average to phases below it so that all three end level.
    Returns [(from_name, to_name, amount)], the biggest move first."""
    v = np.asarray(vals, float); mean = v.mean()
    give = sorted([(v[i] - mean, i) for i in range(len(v)) if v[i] > mean], reverse=True)
    take = sorted([[mean - v[i], i] for i in range(len(v)) if v[i] < mean], reverse=True)
    out = []
    for g, i in give:
        for t in take:
            if g <= 1e-9 or t[0] <= 1e-9: continue
            m = min(g, t[0]); out.append((names[i], names[t[1]], float(m))); g -= m; t[0] -= m
    return sorted(out, key=lambda x: -x[2])


def best_windows(total, index, hours=3):
    """Quietest and busiest `hours`-hour window of a forecast (start, end).  None when the forecast is too short."""
    x = np.asarray(total, float)
    if len(x) < hours or not np.isfinite(x).all(): return None
    s = np.convolve(x, np.ones(hours), "valid")
    lo, hi = int(s.argmin()), int(s.argmax()); step = pd.Timedelta(hours=1)
    return dict(quiet=(index[lo], index[lo + hours - 1] + step), busy=(index[hi], index[hi + hours - 1] + step))


def playbook(dg, now, fc, alarm_pct, min_kw, rating=0.0, demand=0.0, extra=None, real_clock=True):
    """One headline for the page: how bad is the balance now, on average, and in the forecast.  Rule-based."""
    if dg is None:
        return dict(level="amber", head="Not enough loaded readings to judge the phase balance yet.")
    rep = fc["rep"]; n = len(rep); n_al = int(rep.Alarm.sum())
    mean, main = dg["mean_imb"], dg["main_name"]
    if mean > alarm_pct: level, head = "amber", f"{main} is out of balance: {mean:.1f} % on average (limit {alarm_pct:g} %)."
    else: level, head = "good", f"The three phases are in balance: {mean:.1f} % on average (limit {alarm_pct:g} %)."
    if now.get("loaded") and now.get("status") == "ALARM":
        level = "amber"; head += f" Right now it is in alarm ({now['imb']:.1f} %)."
    if n_al: head += f" The forecast has {n_al} alarm hours of {n}."
    return dict(level=level, head=head)
