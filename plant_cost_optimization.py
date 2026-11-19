"""TOU cost optimization for the 168-hour sktime_XGBoost forecast.

Uses the tariff windows and load-shifting logic from the user's notebook/current
agreed logic: no 15% cap and no fixed kW cap; same-day shifting only; destinations
cannot exceed the maximum hourly Off-Peak forecast load.

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
MIN_TARIFF_DIFF = 1.0
INTERVAL_HOURS = 1.0


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
    """Apply same-day max-possible greedy shifting to an hourly forecast series.

    Moves load from Morning peak / Evening peak hours into cheaper Morning / Day hours of the SAME day,
    never lifting a receiving hour above the maximum Off-Peak forecast load (the benchmark).
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

    offpeak = df[df["tariff_band"] == "Off-peak"]
    if offpeak.empty:
        raise ValueError("No Off-Peak forecast observations are available to calculate the benchmark.")
    benchmark = float(offpeak["original_load_kw"].max())

    source_bands = ["Morning peak", "Evening peak"]
    destination_bands = ["Morning", "Day"]
    df["source_candidate"] = df["tariff_band"].isin(source_bands)
    df["dest_candidate"] = df["tariff_band"].isin(destination_bands)
    df["optimised_load_kw"] = df["original_load_kw"].copy()

    shift_log: list[dict[str, Any]] = []
    total_shifted_kwh = 0.0

    for day in df.index.normalize().unique():
        day_data = df[df.index.normalize() == day]
        source_indices = list(day_data[day_data["source_candidate"]].index)
        destination_indices = list(day_data[day_data["dest_candidate"]].index)

        # Keep the notebook's priority: highest original source load first.
        source_indices.sort(
            key=lambda ts: float(df.loc[ts, "original_load_kw"]), reverse=True
        )
        # Fill the lowest-original-load receiving hour first.
        destination_indices.sort(
            key=lambda ts: float(df.loc[ts, "original_load_kw"])
        )

        for src in source_indices:
            remaining = float(df.loc[src, "optimised_load_kw"])
            src_rate = float(df.loc[src, "rate_inr"])

            for dst in destination_indices:
                if remaining <= 1e-9:
                    break

                tariff_difference = src_rate - float(df.loc[dst, "rate_inr"])
                if tariff_difference < MIN_TARIFF_DIFF:
                    continue

                dst_current = float(df.loc[dst, "optimised_load_kw"])
                available_capacity = max(0.0, benchmark - dst_current)
                if available_capacity <= 1e-9:
                    continue

                amount_to_shift = min(remaining, available_capacity)
                if amount_to_shift <= 1e-9:
                    continue

                df.loc[src, "optimised_load_kw"] -= amount_to_shift
                df.loc[dst, "optimised_load_kw"] += amount_to_shift

                shifted_kwh = amount_to_shift * INTERVAL_HOURS
                total_shifted_kwh += shifted_kwh
                remaining -= amount_to_shift

                shift_log.append({
                    "source_time": src,
                    "source_band": df.loc[src, "tariff_band"],
                    "destination_time": dst,
                    "destination_band": df.loc[dst, "tariff_band"],
                    "shifted_kw": amount_to_shift,
                    "shifted_kwh": shifted_kwh,
                    "tariff_difference": tariff_difference,
                    "estimated_savings_inr": shifted_kwh * tariff_difference,
                })

    df["original_energy_kwh"] = df["original_load_kw"] * INTERVAL_HOURS
    df["optimised_energy_kwh"] = df["optimised_load_kw"] * INTERVAL_HOURS
    df["original_cost_inr"] = df["original_energy_kwh"] * df["rate_inr"]
    df["optimised_cost_inr"] = df["optimised_energy_kwh"] * df["rate_inr"]

    original_energy = float(df["original_energy_kwh"].sum())
    optimised_energy = float(df["optimised_energy_kwh"].sum())
    if not np.isclose(original_energy, optimised_energy, rtol=1e-9, atol=1e-6):
        raise AssertionError("Energy balance failed after load shifting.")
    if (df["optimised_load_kw"] < -1e-8).any():
        raise AssertionError("Source load became negative.")
    dest_loads = df.loc[df["dest_candidate"], "optimised_load_kw"]
    if (dest_loads > benchmark + 1e-7).any():
        raise AssertionError("A receiving hour exceeded the Off-Peak benchmark.")

    original_cost = float(df["original_cost_inr"].sum())
    optimised_cost = float(df["optimised_cost_inr"].sum())
    savings = original_cost - optimised_cost
    original_peak = float(df["original_load_kw"].max())
    optimised_peak = float(df["optimised_load_kw"].max())

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

    return {
        "df": df,
        "benchmark_kw": benchmark,
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
