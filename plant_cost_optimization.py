"""TOU cost optimization for the 168-hour sktime_XGBoost forecast.

Uses the tariff windows and load-shifting logic from the user's notebook/current
agreed logic: no 15% cap and no fixed kW cap; same-day shifting only; destinations
cannot exceed the maximum hourly Off-Peak forecast load.
"""
from __future__ import annotations

from typing import Any
import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots

TARIFF_BANDS = ["Off-peak", "Morning", "Morning peak", "Day", "Evening peak"]
TARIFF_ORDER = {name: i for i, name in enumerate(TARIFF_BANDS)}
MIN_TARIFF_DIFF = 1.0
INTERVAL_HOURS = 1.0
TEXT = "#0F172A"
MUTED = "#64748B"
GRID = "#EDF0F6"
ACCENT = "#6366F1"
GREEN = "#10B981"
BLUE = "#3B82F6"
AMBER = "#F59E0B"


def get_tariff_rate(dt: pd.Timestamp) -> tuple[str, float]:
    hr = pd.Timestamp(dt).hour
    if 6 <= hr < 9:
        return "Morning", 8.00
    if 9 <= hr < 12:
        return "Morning peak", 9.50
    if 12 <= hr < 18:
        return "Day", 8.00
    if 18 <= hr < 22:
        return "Evening peak", 10.00
    return "Off-peak", 6.50


def optimize_load_shifting(forecast: pd.Series) -> dict[str, Any]:
    """Apply same-day max-possible greedy shifting to an hourly forecast series."""
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


def _style(fig: go.Figure, height: int = 390) -> go.Figure:
    fig.update_layout(
        template="plotly_white", height=height, paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)", font=dict(family="Inter, sans-serif", size=12, color=TEXT),
        margin=dict(l=14, r=14, t=78, b=18), hovermode="x unified",
        title=dict(x=0, xanchor="left", font=dict(size=16, color=TEXT)),
        legend=dict(orientation="h", x=0, y=1.13, yanchor="bottom"),
        hoverlabel=dict(bgcolor="white", bordercolor=GRID, font=dict(color=TEXT)),
    )
    fig.update_xaxes(gridcolor=GRID, linecolor="#E2E8F0", tickfont=dict(color=MUTED))
    fig.update_yaxes(gridcolor=GRID, zeroline=False, tickfont=dict(color=MUTED))
    return fig


def render_recommendation(st_module, result: dict[str, Any]) -> None:
    shifted = result["total_shifted_kwh"]
    saving = result["savings_inr"]
    if shifted > 1e-6 and saving > 0:
        title = "Recommended: shift eligible peak-period load"
        text = (
            f"The sktime_XGBoost 7-day forecast indicates that approximately {shifted:,.0f} kWh can be shifted "
            f"from Morning Peak / Evening Peak into eligible Morning / Day hours. Estimated energy-charge saving: "
            f"₹{saving:,.0f} ({result['savings_pct']:.2f}%)."
        )
        status = "Recommended opportunity"
        bg, border, fg = "#E8FAF3", "#A7F3D0", "#047857"
    else:
        title = "No cost-saving shift identified"
        text = "Under the configured time-of-use tariffs and Off-Peak receiving-load benchmark, no eligible load transfer produced a positive saving."
        status = "No shift opportunity"
        bg, border, fg = "#FFF4DD", "#FDE68A", "#B45309"

    st_module.markdown(
        f'<div style="background:{bg};border:1px solid {border};border-radius:16px;padding:16px 18px;margin:4px 0 14px;">'
        f'<div style="display:flex;justify-content:space-between;gap:12px;align-items:center;flex-wrap:wrap;">'
        f'<div style="font-size:1rem;font-weight:750;color:{TEXT};">{title}</div>'
        f'<span style="font-size:.75rem;font-weight:700;color:{fg};">{status}</span></div>'
        f'<div style="font-size:.88rem;color:#475569;margin-top:7px;line-height:1.5;">{text}</div>'
        f'<div style="font-size:.75rem;color:#64748B;margin-top:6px;">Indicative forecast-based estimate using the configured tariffs. Confirm equipment/process flexibility before applying any schedule change.</div>'
        f'</div>', unsafe_allow_html=True
    )


def render_optimization_results(st_module, result: dict[str, Any]) -> None:
    st_module.markdown('<div class="ph"><b>Cost optimization · 7-day forecast</b><span>TOU tariff shifting · sktime_XGBoost</span></div>', unsafe_allow_html=True)
    cols = st_module.columns(4)
    metrics = [
        ("Energy shifted", f"{result['total_shifted_kwh']:,.0f} kWh", "Across eligible peak hours", "🔁"),
        ("Estimated savings", f"₹{result['savings_inr']:,.0f}", f"{result['savings_pct']:.2f}% of forecast energy cost", "💰"),
        ("Original → optimized cost", f"₹{result['original_cost_inr']:,.0f} → ₹{result['optimised_cost_inr']:,.0f}", "7-day energy charge estimate", "🧾"),
        ("Peak hourly demand", f"{result['original_peak_kw']:,.0f} → {result['optimised_peak_kw']:,.0f} kW", f"Change: {result['peak_reduction_kw']:,.1f} kW", "📉"),
    ]
    for col, (label, value, sub, icon) in zip(cols, metrics):
        with col:
            st_module.markdown(
                f'<div class="kpi"><div class="top"><div class="ic">{icon}</div><div class="l">{label}</div></div>'
                f'<div class="v" style="font-size:1.22rem;letter-spacing:-.4px;">{value}</div>'
                f'<div class="s">{sub}</div></div>', unsafe_allow_html=True
            )

    df = result["df"]
    summary = result["band_summary"]
    st_module.write("")

    # Original vs optimized energy and cost, grouped by tariff band.
    fig = make_subplots(
        rows=1, cols=2,
        subplot_titles=("Energy consumption breakdown", "Electricity cost breakdown"),
        horizontal_spacing=0.15,
    )
    fig.add_trace(go.Bar(
        name="Original Energy", x=summary["tariff_band"], y=summary["original_energy_kwh"],
        marker_color="#94A3B8", hovertemplate="%{x}<br>Original: %{y:,.1f} kWh<extra></extra>"
    ), row=1, col=1)
    fig.add_trace(go.Bar(
        name="Optimised Energy", x=summary["tariff_band"], y=summary["optimised_energy_kwh"],
        marker_color=GREEN, hovertemplate="%{x}<br>Optimised: %{y:,.1f} kWh<extra></extra>"
    ), row=1, col=1)
    fig.add_trace(go.Bar(
        name="Original Cost", x=summary["tariff_band"], y=summary["original_cost_inr"],
        marker_color="#94A3B8", showlegend=False,
        hovertemplate="%{x}<br>Original: ₹%{y:,.0f}<extra></extra>"
    ), row=1, col=2)
    fig.add_trace(go.Bar(
        name="Optimised Cost", x=summary["tariff_band"], y=summary["optimised_cost_inr"],
        marker_color=BLUE, showlegend=False,
        hovertemplate="%{x}<br>Optimised: ₹%{y:,.0f}<extra></extra>"
    ), row=1, col=2)
    fig.update_layout(title="Original vs optimized energy and cost", barmode="group")
    fig.update_yaxes(title_text="Energy (kWh)", row=1, col=1)
    fig.update_yaxes(title_text="Cost (₹)", row=1, col=2)
    st_module.plotly_chart(_style(fig, 430), use_container_width=True, config={"displayModeBar": False})

    # Combined original/optimized demand with tariff on the secondary axis.
    fig = make_subplots(specs=[[{"secondary_y": True}]])
    fig.add_trace(go.Scatter(
        x=df.index, y=df["original_load_kw"], name="Original Demand",
        mode="lines", line=dict(color="#94A3B8", width=1.7, dash="dash"),
    ), secondary_y=False)
    fig.add_trace(go.Scatter(
        x=df.index, y=df["optimised_load_kw"], name="Optimised Demand",
        mode="lines", line=dict(color=GREEN, width=2.3),
    ), secondary_y=False)
    fig.add_trace(go.Scatter(
        x=df.index, y=df["rate_inr"], name="Tariff (₹/kWh)",
        mode="lines", line=dict(color=AMBER, width=2, shape="hv"),
    ), secondary_y=True)
    fig.add_hline(y=result["benchmark_kw"], line_dash="dot", line_color=BLUE,
                  annotation_text=f"Off-Peak max benchmark {result['benchmark_kw']:,.0f} kW",
                  annotation_position="top left", secondary_y=False)
    fig.update_layout(title="Original vs optimized demand profile with tariff overlay", xaxis_title="Timestamp")
    fig.update_yaxes(title_text="Demand (kW)", secondary_y=False)
    fig.update_yaxes(title_text="Tariff (₹/kWh)", secondary_y=True)
    st_module.plotly_chart(_style(fig, 430), use_container_width=True, config={"displayModeBar": False})

    # Route-level bar chart makes source-to-destination shifts explicit; no Sankey chart.
    st_module.markdown('<div class="ph"><b>Where the load was shifted</b><span>kWh moved from each source period into each receiving period</span></div>', unsafe_allow_html=True)
    route = result["route_summary"]
    if route.empty:
        st_module.info("No eligible load shift was made under the current tariff and receiving-capacity rules.")
    else:
        route = route.sort_values("shifted_kwh", ascending=False)
        fig = go.Figure(go.Bar(
            x=route["route"], y=route["shifted_kwh"],
            marker_color=ACCENT,
            text=route["shifted_kwh"].map(lambda v: f"{v:,.0f} kWh"),
            textposition="outside",
            hovertemplate="%{x}<br>Shifted: %{y:,.2f} kWh<extra></extra>",
        ))
        fig.update_layout(title="Load shifted by source and destination", xaxis_title="Shift route", yaxis_title="Shifted energy (kWh)")
        st_module.plotly_chart(_style(fig, 330), use_container_width=True, config={"displayModeBar": False})
        st_module.dataframe(route.rename(columns={"route": "Shift route", "shifted_kwh": "Shifted energy (kWh)"}).round(2), use_container_width=True, hide_index=True)

    with st_module.expander("Tariff-band detail"):
        show = summary.rename(columns={
            "tariff_band": "Tariff band",
            "original_energy_kwh": "Original energy (kWh)",
            "optimised_energy_kwh": "Optimized energy (kWh)",
            "original_cost_inr": "Original cost (₹)",
            "optimised_cost_inr": "Optimized cost (₹)",
            "energy_change_kwh": "Energy change (kWh)",
            "cost_change_inr": "Cost change (₹)",
        })
        st_module.dataframe(show.round(2), use_container_width=True, hide_index=True)

    st_module.caption(
        f"Off-Peak benchmark: {result['benchmark_kw']:,.2f} kW (maximum individual hourly Off-Peak forecast). "
        "Each hourly kW value is treated as kWh for a one-hour interval. Demand-charge components are not included."
    )
