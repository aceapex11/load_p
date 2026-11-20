"""TOU cost optimization for the 168-hour sktime_XGBoost forecast.

Operating rules built into the logic
------------------------------------
1. OFF-PEAK IS UNTOUCHED (22:00-06:00): the plant already runs at high capacity there under contract, so no load
   is ever shifted INTO or OUT OF Off-peak. Its maximum hourly forecast load is only the CEILING (benchmark)
   that no receiving hour may exceed.
2. SOURCES, highest tariff first:
     Evening peak (18-22, Rs 10.00) -> Morning / Day (Rs 8.00) first, then Morning peak (Rs 9.50) if capacity remains.
     Morning peak (09-12, Rs 9.50)  -> Morning / Day (Rs 8.00) only, and only when the transfer is big enough
                                       to be operationally worthwhile (its own, higher kW minimum).
3. MEANINGFUL SHIFTS ONLY: a source hour is reduced by 15-20 % of its own load, and by at least a minimum kW
   (default 100 kW Evening, 150 kW Morning peak). The other 80 %+ (low-power machines) keeps running, so the
   plant never stops. If the receiving hours cannot absorb at least the minimum, the hour is left unchanged -
   no 5-20 kW micro-shifts. Every single transfer between two hours is also at least MIN_TRANSFER_KW.
4. Same-day shifting only. An hour never both gives and receives load.

This module has NO Streamlit code and NO charts. All plant-tab frontend (banner, KPI cards,
charts, tables) lives in app.py, which calls optimize_load_shifting() below.

File layout
-----------
1. SETTINGS + TARIFF       constants, tariff lookup
2. OPTIMISATION            optimize_load_shifting (pure pandas)
"""
from __future__ import annotations

from typing import Any
import numpy as np
import pandas as pd

# =====================================================================================
# 1. SETTINGS + TARIFF
# =====================================================================================
# Single source of truth for the tariff: (band, start hour, end hour, rate in INR per kWh). Off-peak wraps past midnight.
TARIFF_SCHEDULE = [
    ("Off-peak",     22, 6,  6.50),
    ("Morning",       6, 9,  8.00),
    ("Morning peak",  9, 12, 9.50),
    ("Day",          12, 18, 8.00),
    ("Evening peak", 18, 22, 10.00),
]
TARIFF_BANDS = [b[0] for b in TARIFF_SCHEDULE]
TARIFF_ORDER = {name: i for i, name in enumerate(TARIFF_BANDS)}
TARIFF_RATES = {b: rate for b, _, _, rate in TARIFF_SCHEDULE}
TARIFF_TIME_WINDOWS = {b: f"{start:02d}:00–{end:02d}:00" for b, start, end, _ in TARIFF_SCHEDULE}   # e.g. "22:00–06:00"
INTERVAL_HOURS = 1.0

# ---- shifting rules - EDIT HERE
OFF_PEAK_BAND = "Off-peak"                       # never a source, never a destination; only the ceiling benchmark
SOURCE_BANDS = ["Evening peak", "Morning peak"]  # processed in this order (highest tariff first)
DESTINATION_BANDS = {                            # allowed receiving bands, in preference order (cheapest first)
    "Evening peak": ["Morning", "Day", "Morning peak"],   # Morning peak only if capacity remains after Morning / Day
    "Morning peak": ["Morning", "Day"],
}
MIN_TARIFF_DIFF = 0.5            # Rs/kWh: a destination must be at least this much cheaper (Evening -> Morning peak is Rs 0.50)
MAX_SHIFT_PCT = 20.0             # a source hour is reduced by at most 20 % of its own load (the rest keeps running)
MIN_SHIFT_PCT = 15.0             # ...and by at least 15 %, otherwise the hour is left unchanged
MIN_SHIFT_KW = {"Evening peak": 100.0, "Morning peak": 150.0}   # absolute minimum per source hour (Morning peak needs a bigger, worthwhile move)
MIN_TRANSFER_KW = 50.0           # smallest single transfer between two hours (no micro-shifts)


def get_tariff_rate(dt: pd.Timestamp) -> tuple[str, float]:
    """Return (tariff band name, rate in INR per kWh) for the hour of `dt`."""
    hr = pd.Timestamp(dt).hour
    for band, start, end, rate in TARIFF_SCHEDULE:
        in_band = (start <= hr < end) if start < end else (hr >= start or hr < end)   # wrap-around band (22:00-06:00)
        if in_band:
            return band, rate
    raise ValueError(f"No tariff band covers hour {hr}.")


# =====================================================================================
# 2. OPTIMISATION
# =====================================================================================
def optimize_load_shifting(forecast: pd.Series) -> dict[str, Any]:
    """Shift heavy load out of the peak bands into cheaper bands, day by day, under the operating rules above.

    For every source hour (Evening peak first, then Morning peak):
      1. its maximum shift is MAX_SHIFT_PCT of its own load, its minimum is the larger of MIN_SHIFT_PCT and the kW minimum;
      2. cheaper same-day hours are tried cheapest first (then lowest load first), each limited by the Off-peak ceiling;
      3. the plan is applied only if the total reaches the minimum - otherwise the hour is untouched.
    Returns a dict with the hourly table, shift log, summaries, costs and peak figures.
    """
    if not isinstance(forecast, pd.Series):
        forecast = pd.Series(forecast)
    forecast = forecast.astype(float).sort_index()
    if forecast.empty or not isinstance(forecast.index, pd.DatetimeIndex):
        raise ValueError("The selected model forecast must have a non-empty DatetimeIndex.")
    if forecast.isna().any() or (forecast < 0).any():
        raise ValueError("Forecast values must be complete and non-negative.")
    if forecast.index.has_duplicates:
        raise ValueError("Duplicate forecast timestamps found.")

    df = pd.DataFrame(index=forecast.index)
    df["original_load_kw"] = forecast
    tariff_info = [get_tariff_rate(ts) for ts in df.index]
    df["tariff_band"] = [x[0] for x in tariff_info]
    df["rate_inr"] = [x[1] for x in tariff_info]

    # Off-peak is only the ceiling: the highest hourly Off-peak forecast load.
    offpeak = df[df["tariff_band"] == OFF_PEAK_BAND]
    if offpeak.empty:
        raise ValueError("No Off-Peak forecast observations are available to calculate the benchmark.")
    benchmark = float(offpeak["original_load_kw"].max())

    df["source_candidate"] = df["tariff_band"].isin(SOURCE_BANDS)
    df["dest_candidate"] = df["tariff_band"].isin({b for bands in DESTINATION_BANDS.values() for b in bands})
    df["optimised_load_kw"] = df["original_load_kw"].copy()

    # Per source hour: largest allowed shift (20 %), smallest worthwhile shift (15 % and the kW minimum), load that keeps running.
    min_kw = df["tariff_band"].map(MIN_SHIFT_KW).fillna(0.0)
    df["max_shift_kw"] = np.where(df["source_candidate"], df["original_load_kw"] * MAX_SHIFT_PCT / 100.0, 0.0)
    df["min_shift_kw"] = np.where(df["source_candidate"], np.maximum(df["original_load_kw"] * MIN_SHIFT_PCT / 100.0, min_kw), 0.0)
    df["protected_load_kw"] = np.where(df["source_candidate"], df["original_load_kw"] - df["max_shift_kw"], 0.0)   # low-power load that never moves
    df["shiftable_kw"] = df["max_shift_kw"]                                                                        # heavy load allowed to move
    df["shifted_out_kw"] = 0.0
    df["shifted_in_kw"] = 0.0

    shift_log: list[dict[str, Any]] = []
    total_shifted_kwh = 0.0
    skipped_hours = 0                                                           # source hours left unchanged (move too small / no capacity)

    for day in df.index.normalize().unique():
        day_hours = list(df.index[df.index.normalize() == day])
        gave, received = set(), set()                                           # an hour never both gives and receives

        # highest tariff first, then the heaviest hour first
        sources = [ts for ts in day_hours if df.at[ts, "source_candidate"]]
        sources.sort(key=lambda ts: (-float(df.at[ts, "rate_inr"]), -float(df.at[ts, "original_load_kw"])))

        for src in sources:
            if src in received:                                                 # it already took load; do not also send load away
                continue
            band, src_rate = df.at[src, "tariff_band"], float(df.at[src, "rate_inr"])
            max_shift, min_shift = float(df.at[src, "max_shift_kw"]), float(df.at[src, "min_shift_kw"])
            if max_shift + 1e-9 < min_shift:                                    # hour too small for a meaningful shift
                skipped_hours += 1
                continue

            # candidate receiving hours: allowed band, cheaper by the minimum spread, not Off-peak, not a giver
            preference = {b: i for i, b in enumerate(DESTINATION_BANDS[band])}
            candidates = [ts for ts in day_hours
                          if ts != src and ts not in gave and df.at[ts, "tariff_band"] in preference
                          and src_rate - float(df.at[ts, "rate_inr"]) >= MIN_TARIFF_DIFF - 1e-9]
            # cheapest tariff first (Morning / Day before Morning peak), then the lowest-loaded hour first
            candidates.sort(key=lambda ts: (float(df.at[ts, "rate_inr"]), preference[df.at[ts, "tariff_band"]], float(df.at[ts, "optimised_load_kw"])))

            # build the plan WITHOUT changing anything yet
            plan, remaining = [], max_shift
            for dst in candidates:
                if remaining <= 1e-9:
                    break
                capacity = max(0.0, benchmark - float(df.at[dst, "optimised_load_kw"]))   # Off-peak ceiling
                amount = min(remaining, capacity)
                if amount < MIN_TRANSFER_KW:                                    # no micro-transfers
                    continue
                plan.append((dst, amount)); remaining -= amount
            moved = sum(a for _, a in plan)
            if not plan or moved + 1e-9 < min_shift:                            # cannot reach a meaningful shift -> leave the hour alone
                skipped_hours += 1
                continue

            # apply the plan
            for dst, amount in plan:
                df.at[src, "optimised_load_kw"] -= amount
                df.at[dst, "optimised_load_kw"] += amount
                df.at[dst, "shifted_in_kw"] += amount
                tariff_difference = src_rate - float(df.at[dst, "rate_inr"])
                shifted_kwh = amount * INTERVAL_HOURS
                total_shifted_kwh += shifted_kwh
                received.add(dst)
                shift_log.append({
                    "source_time": src,
                    "source_band": band,
                    "destination_time": dst,
                    "destination_band": df.at[dst, "tariff_band"],
                    "shifted_kw": amount,
                    "shifted_kwh": shifted_kwh,
                    "source_shift_pct": moved / float(df.at[src, "original_load_kw"]) * 100.0,
                    "tariff_difference": tariff_difference,
                    "estimated_savings_inr": shifted_kwh * tariff_difference,
                })
            df.at[src, "shifted_out_kw"] = moved
            gave.add(src)

    df["original_energy_kwh"] = df["original_load_kw"] * INTERVAL_HOURS
    df["optimised_energy_kwh"] = df["optimised_load_kw"] * INTERVAL_HOURS
    df["original_cost_inr"] = df["original_energy_kwh"] * df["rate_inr"]
    df["optimised_cost_inr"] = df["optimised_energy_kwh"] * df["rate_inr"]

    # ---- safety checks: every operating rule is verified on the final table
    original_energy = float(df["original_energy_kwh"].sum())
    optimised_energy = float(df["optimised_energy_kwh"].sum())
    if not np.isclose(original_energy, optimised_energy, rtol=1e-9, atol=1e-6):
        raise AssertionError("Energy balance failed after load shifting.")
    if (df["optimised_load_kw"] < -1e-8).any():
        raise AssertionError("Source load became negative.")
    is_off = df["tariff_band"] == OFF_PEAK_BAND
    if not np.allclose(df.loc[is_off, "optimised_load_kw"], df.loc[is_off, "original_load_kw"], atol=1e-7):
        raise AssertionError("Off-peak load was changed; it must stay untouched.")
    got = df["shifted_in_kw"] > 0
    if (df.loc[got, "optimised_load_kw"] > benchmark + 1e-7).any():
        raise AssertionError("A receiving hour exceeded the Off-Peak benchmark.")
    gave_mask = df["shifted_out_kw"] > 0
    if (gave_mask & got).any():
        raise AssertionError("An hour both gave and received load.")
    if (df.loc[gave_mask, "optimised_load_kw"] < df.loc[gave_mask, "protected_load_kw"] - 1e-7).any():
        raise AssertionError("A peak hour fell below its protected running load.")
    if (df.loc[gave_mask, "shifted_out_kw"] < df.loc[gave_mask, "min_shift_kw"] - 1e-7).any() or \
       (df.loc[gave_mask, "shifted_out_kw"] > df.loc[gave_mask, "max_shift_kw"] + 1e-7).any():
        raise AssertionError("A shift fell outside the allowed 15-20 % / minimum-kW range.")

    original_cost = float(df["original_cost_inr"].sum())
    optimised_cost = float(df["optimised_cost_inr"].sum())
    savings = original_cost - optimised_cost
    original_peak = float(df["original_load_kw"].max())
    optimised_peak = float(df["optimised_load_kw"].max())
    df["shift_pct_of_hour"] = np.where(df["original_load_kw"] > 0, df["shifted_out_kw"] / df["original_load_kw"].where(df["original_load_kw"] > 0, 1.0) * 100.0, 0.0)

    band_summary = df.groupby("tariff_band", observed=False).agg(
        original_energy_kwh=("original_energy_kwh", "sum"),
        optimised_energy_kwh=("optimised_energy_kwh", "sum"),
        original_cost_inr=("original_cost_inr", "sum"),
        optimised_cost_inr=("optimised_cost_inr", "sum"),
    ).reindex(TARIFF_BANDS).fillna(0).reset_index()
    band_summary.insert(1, "time_window", band_summary["tariff_band"].map(TARIFF_TIME_WINDOWS))   # shown in the tariff-band table
    band_summary.insert(2, "rate_inr", band_summary["tariff_band"].map(TARIFF_RATES))
    band_summary["energy_change_kwh"] = band_summary["optimised_energy_kwh"] - band_summary["original_energy_kwh"]
    band_summary["cost_change_inr"] = band_summary["optimised_cost_inr"] - band_summary["original_cost_inr"]

    shifts = pd.DataFrame(shift_log)
    if shifts.empty:
        route_summary = pd.DataFrame(columns=["route", "shifted_kwh"])
        source_summary = pd.DataFrame(columns=["source_band", "shifted_kwh"])
    else:
        shifts["route"] = shifts["source_band"] + " → " + shifts["destination_band"]
        route_summary = shifts.groupby("route", as_index=False)["shifted_kwh"].sum()
        source_summary = shifts.groupby("source_band", as_index=False)["shifted_kwh"].sum()

    src_mask = df["source_candidate"]
    return {
        "df": df,
        "benchmark_kw": benchmark,                                                  # Off-peak ceiling (Off-peak itself is untouched)
        "off_peak_untouched": True,
        "base_load_kw": float(df["original_load_kw"].quantile(0.10)),               # information only (10th percentile of the week)
        "min_peak_running_pct": 100.0 - MAX_SHIFT_PCT,                              # % of each shifted hour that keeps running
        "lowest_peak_hour_kw": float(df.loc[src_mask, "optimised_load_kw"].min()) if src_mask.any() else float("nan"),   # proof the plant never hits 0
        "shiftable_energy_kwh": float(df["shiftable_kw"].sum() * INTERVAL_HOURS),   # heavy load that was allowed to move (<= 20 % of each peak hour)
        "shifted_hours": int((df["shifted_out_kw"] > 0).sum()),                     # source hours that really shifted
        "skipped_hours": int(skipped_hours),                                        # source hours left unchanged (too small / no capacity)
        "shifts": shifts,
        "band_summary": band_summary,
        "route_summary": route_summary,
        "source_summary": source_summary,
        "total_shifted_kwh": float(total_shifted_kwh),
        "original_energy_kwh": original_energy,
        "optimised_energy_kwh": optimised_energy,
        "original_cost_inr": original_cost,
        "optimised_cost_inr": optimised_cost,
        "savings_inr": float(savings),
        "savings_pct": float(savings / original_cost * 100) if original_cost else 0.0,
        "original_peak_kw": original_peak,
        "optimised_peak_kw": optimised_peak,
        "peak_reduction_kw": original_peak - optimised_peak,
    }
