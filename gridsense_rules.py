"""GridSense rule module - the phase-balance rules used by app.py (no Streamlit, no AI service).

Rebuilt from how app.py and gridsense_forecast.py call it, because the original file was missing.  Functions:
  analyse_phase_balance(phase_kw, alarm_limit_pct, min_phase_kw) -> one row per hour/minute: Loaded, Imbalance_pct, Worst_phase, Worst_dev, Direction, Status, Alarm
  plan_phase_transfers(values, phase_names)                     -> [(from, to, amount)] that levels the three phases
  build_balance_headline(...)                                   -> {"head": text, "level": "good" | "amber" | "bad"}
  find_quietest_and_busiest_windows(total_kw, time_index)       -> quietest and busiest 3-hour window of a forecast
"""
import numpy as np
import pandas as pd

PHASE_NAMES = ["Red", "Yellow", "Blue"]          # phase columns are always in the order R, Y, B


def analyse_phase_balance(phase_kw, alarm_limit_pct=5.0, min_phase_kw=5.0):
    """imbalance % = max(|R-avg|, |Y-avg|, |B-avg|) / avg x 100.   170 / 169 / 160 kW -> 3.81 % (Blue, LOW).
    Loaded  : average kW per phase is at least `min_phase_kw` (below that the machine is off and a % is meaningless).
    Status  : IDLE (not loaded) / OK / ALARM (> alarm_limit_pct).   Alarm = ALARM."""
    kw = np.atleast_2d(np.asarray(phase_kw, float))
    average_kw = kw.mean(axis=1)
    is_loaded = np.isfinite(average_kw) & (average_kw > 0) & (average_kw >= min_phase_kw)
    deviation_kw = kw - average_kw[:, None]                                                    # each phase minus the three-phase average
    worst_phase_index = np.where(is_loaded[:, None], np.abs(deviation_kw), -1.0).argmax(axis=1)      # idle rows are ignored
    row_numbers = np.arange(len(kw))
    worst_deviation_pct = np.where(is_loaded, deviation_kw[row_numbers, worst_phase_index] / np.where(is_loaded, average_kw, 1.0) * 100, np.nan)
    imbalance_pct = np.abs(worst_deviation_pct)
    status = np.where(~is_loaded, "IDLE", np.where(imbalance_pct > alarm_limit_pct, "ALARM", "OK"))
    return pd.DataFrame({
        "Loaded": is_loaded,
        "Imbalance_pct": imbalance_pct,
        "Worst_phase": np.where(is_loaded, np.array(PHASE_NAMES, dtype=object)[worst_phase_index], None),
        "Worst_dev": worst_deviation_pct,
        "Direction": np.where(is_loaded, np.where(worst_deviation_pct > 0, "HIGH", "LOW"), None),
        "Status": status,
        "Alarm": status == "ALARM",
    })


def plan_phase_transfers(values, phase_names):
    """Amounts to move from phases above the average to phases below it so that all three end level.
    Returns [(from_name, to_name, amount)], the biggest move first."""
    values = np.asarray(values, float); average = values.mean()
    givers = sorted([(values[i] - average, i) for i in range(len(values)) if values[i] > average], reverse=True)       # phases above average: (surplus, index)
    takers = sorted([[average - values[i], i] for i in range(len(values)) if values[i] < average], reverse=True)       # phases below average: [room, index]
    moves = []
    for surplus, giver_index in givers:
        for taker in takers:
            if surplus <= 1e-9 or taker[0] <= 1e-9: continue
            amount = min(surplus, taker[0]); moves.append((phase_names[giver_index], phase_names[taker[1]], float(amount))); surplus -= amount; taker[0] -= amount
    return sorted(moves, key=lambda move: -move[2])


def find_quietest_and_busiest_windows(total_kw, time_index, window_hours=3):
    """Quietest and busiest `window_hours`-hour window of a forecast (start, end).  None when the forecast is too short."""
    total_kw = np.asarray(total_kw, float)
    if len(total_kw) < window_hours or not np.isfinite(total_kw).all(): return None
    window_sums = np.convolve(total_kw, np.ones(window_hours), "valid")                        # kW summed over every possible window
    quiet_start, busy_start = int(window_sums.argmin()), int(window_sums.argmax()); one_hour = pd.Timedelta(hours=1)
    return dict(quiet=(time_index[quiet_start], time_index[quiet_start + window_hours - 1] + one_hour),
                busy=(time_index[busy_start], time_index[busy_start + window_hours - 1] + one_hour))


def build_balance_headline(diagnosis, now, forecast, alarm_limit_pct, min_phase_kw, rating=0.0, demand=0.0, extra=None, real_clock=True):
    """One headline for the page: how bad is the balance now, on average, and in the forecast.  Rule-based."""
    if diagnosis is None:
        return dict(level="amber", head="Not enough loaded readings to judge the phase balance yet.")
    forecast_table = forecast["rep"]; n_hours = len(forecast_table); n_alarm_hours = int(forecast_table.Alarm.sum())
    mean_imbalance, main_phase = diagnosis["mean_imb"], diagnosis["main_name"]
    if mean_imbalance > alarm_limit_pct: level, head = "amber", f"{main_phase} is out of balance: {mean_imbalance:.1f} % on average (limit {alarm_limit_pct:g} %)."
    else: level, head = "good", f"The three phases are in balance: {mean_imbalance:.1f} % on average (limit {alarm_limit_pct:g} %)."
    if now.get("loaded") and now.get("status") == "ALARM":
        level = "amber"; head += f" Right now it is in alarm ({now['imb']:.1f} %)."
    if n_alarm_hours: head += f" The forecast has {n_alarm_hours} alarm hours of {n_hours}."
    return dict(level=level, head=head)
