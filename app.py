"""GridSense - machine load intelligence (light mode).
Data            : device_16_prediction_dataset.csv - one row per minute with V1-V3, I1-I3, PF, RKW, YKW, BKW
                  (an optional Timestamps / Timestamp / Timestamp_Index column is used when present; you can also upload the file in the app)
Overview        : load timeline (kW, phases, current, voltage, imbalance), activity calendar, uptime
Load prediction : ARIMA(2,0,2) on the 30-min rolling total kW, future forecast + advice (power, kVA, power factor, sanctioned demand)
Phase imbalance : measured imbalance % (max deviation from the 3-phase average), WHY it happens, WHAT to do (which phase to move how many amps)
Phase forecast  : SARIMAX per phase on the last 24 hours (5-min steps), forecast imbalance + alarms, backtest against "same as now"
"""
import datetime as dt
import io
import warnings
from contextlib import contextmanager
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from plotly.subplots import make_subplots
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.statespace.sarimax import SARIMAX

warnings.filterwarnings("ignore")
st.set_page_config(page_title="GridSense · Load intelligence", page_icon="⚡", layout="wide", initial_sidebar_state="collapsed")

HERE = Path(__file__).resolve().parent
DATA_NAME = "device_16_prediction_dataset.csv"       # put this file next to app.py (or upload it in "Data & limits")
PHASES = ["RKW", "YKW", "BKW"]
PNAME = {"RKW": "Red", "YKW": "Yellow", "BKW": "Blue"}
NAMES = [PNAME[p] for p in PHASES]

# limits - the real values come from the "Data & limits" panel further down; these are only the defaults
ALARM_PCT, SEVERE_PCT, MIN_KW, SMOOTH, DEMAND, RATING = 3.0, 10.0, 5.0, 30, 0.0, 0.0

# COLOUR THEME  (one palette everywhere)
#   neutrals : slate  (INK text / actual values, MUTED secondary, GRID lines)
#   accent   : indigo (ACCENT = measured load + forecasts + selected buttons; ACCENT_LIGHT / ACCENT_DARK = its tints)
#   status   : green = good, amber = warning, rose = alarm / bad   (status only)
#   phases   : R rose, Y amber, B blue                              (phase charts only)
INK, MUTED, GRID = "#0F172A", "#64748B", "#EDF0F6"
ACCENT, ACCENT_LIGHT, ACCENT_DARK = "#6366F1", "#818CF8", "#4F46E5"
LOAD = FORECAST = VIOLET = ACCENT
CYAN = ACCENT_LIGHT
GOOD, WARN, BAD = "#10B981", "#F59E0B", "#F43F5E"
AMBER, RED, BLUE = WARN, BAD, "#3B82F6"
IDLE, NODATA, AXIS = "#CBD5E1", BAD, "#E2E8F0"            # idle share (slate), NO DATA (red), axis lines / empty bars


def tint(hex_color, alpha):
    """'#6366F1', .16  ->  'rgba(99,102,241,.16)'  (transparent fills that follow the theme colours)."""
    h = hex_color.lstrip("#"); return f"rgba({int(h[0:2], 16)},{int(h[2:4], 16)},{int(h[4:6], 16)},{alpha})"


GREY = MUTED
PHASE_COL = {"RKW": "#F43F5E", "YKW": "#F59E0B", "BKW": "#3B82F6"}

# ------------------------------------------------------------------ design system (light + indigo accent, forced in CSS - no config file needed)
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&family=JetBrains+Mono:wght@500;600&display=swap');
:root { color-scheme: light; }
html, body, .stApp, [data-testid="stAppViewContainer"], [data-testid="stMain"] { background:#F5F7FB !important; color:#0F172A !important; font-family:'Inter',sans-serif; }
[data-testid="stHeader"] { background:transparent !important; } #MainMenu, footer { visibility:hidden; }
.block-container, [data-testid="stMainBlockContainer"] { max-width:100% !important; padding:3.4rem 2.4rem 3rem 2.4rem !important; }
.stApp p, .stApp label, .stApp li, [data-testid="stWidgetLabel"] p { color:#0F172A !important; font-family:'Inter',sans-serif; }
[data-testid="stCaptionContainer"] p { color:#64748B !important; }

/* app bar */
.appbar { display:flex; align-items:center; gap:.9rem; background:#FFFFFF; border:1px solid #E6EAF2; border-radius:18px; padding:.85rem 1.2rem; margin-bottom:1rem; box-shadow:0 1px 2px rgba(15,23,42,.04), 0 8px 24px -12px rgba(15,23,42,.10); }
.appbar .logo { width:42px; height:42px; border-radius:12px; display:grid; place-items:center; font-size:1.3rem; background:linear-gradient(135deg,#818CF8,#6366F1 55%,#4F46E5); box-shadow:0 6px 16px -6px rgba(99,102,241,.7); }
.appbar .t { font-weight:800; font-size:1.2rem; letter-spacing:-.4px; color:#0F172A !important; } .appbar .s { font-size:.78rem; color:#64748B !important; margin-top:-1px; }
.appbar .sp { flex:1; }
.chip { display:inline-flex; align-items:center; gap:.35rem; padding:.28rem .7rem; border-radius:999px; font-size:.76rem; font-weight:500; border:1px solid #E6EAF2; background:#F8FAFD; color:#475569 !important; }
.pill { display:inline-flex; align-items:center; gap:.45rem; padding:.3rem .8rem; border-radius:999px; font-size:.78rem; font-weight:600; }
.pill.run { background:#E8FAF3; color:#047857 !important; } .pill.idle { background:#F1F4F9; color:#475569 !important; } .pill.warn { background:#FFF4DD; color:#B45309 !important; } .pill.bad { background:#FFE9ED; color:#BE123C !important; }
.dot { width:8px; height:8px; border-radius:50%; background:currentColor; } .run .dot { animation:pulse 1.6s infinite; }
@keyframes pulse { 0%{box-shadow:0 0 0 0 rgba(16,185,129,.55);} 70%{box-shadow:0 0 0 8px rgba(16,185,129,0);} 100%{box-shadow:0 0 0 0 rgba(16,185,129,0);} }

/* tabs as a segmented pill nav */
.stTabs [data-baseweb="tab-list"] { gap:.25rem; background:#E9EDF5; padding:.3rem; border-radius:14px; width:fit-content; border:0; }
.stTabs [data-baseweb="tab"] { height:2.4rem; padding:0 1.15rem; border-radius:10px; background:transparent !important; color:#64748B !important; font-weight:600; }
.stTabs [aria-selected="true"] { background:#FFFFFF !important; color:#0F172A !important; box-shadow:0 1px 3px rgba(15,23,42,.12); }
.stTabs [data-baseweb="tab-highlight"], .stTabs [data-baseweb="tab-border"] { display:none !important; }
.stTabs [data-baseweb="tab-panel"] { padding-top:1.1rem; }

/* cards */
[data-testid="stVerticalBlockBorderWrapper"], [class*="st-key-panel"] { background:#FFFFFF !important; border:1px solid #E6EAF2 !important; border-radius:18px !important; box-shadow:0 1px 2px rgba(15,23,42,.04), 0 10px 28px -16px rgba(15,23,42,.12); }
.ph { display:flex; align-items:baseline; gap:.6rem; margin:.1rem 0 .2rem; } .ph b { font-size:1rem; font-weight:700; color:#0F172A !important; letter-spacing:-.2px; } .ph span { font-size:.8rem; color:#64748B !important; }
.kpi { position:relative; overflow:hidden; background:#FFFFFF; border:1px solid #E6EAF2; border-radius:18px; padding:1rem 1.1rem .9rem; height:100%; box-shadow:0 1px 2px rgba(15,23,42,.04), 0 10px 28px -16px rgba(15,23,42,.12); }
.kpi .top { display:flex; align-items:center; gap:.55rem; } .kpi .ic { width:30px; height:30px; border-radius:9px; display:grid; place-items:center; font-size:.95rem; background:#F3F5FB; }
.kpi .l { color:#64748B !important; font-size:.8rem; font-weight:500; }
.kpi .v { font-family:'JetBrains Mono',monospace; font-size:1.75rem; font-weight:600; letter-spacing:-1px; margin-top:.55rem; color:#0F172A !important; }
.kpi .s { font-size:.78rem; margin-top:.1rem; color:#64748B !important; }
.kpi.good .v { color:#059669 !important; } .kpi.warn .v { color:#D97706 !important; } .kpi.bad .v { color:#E11D48 !important; } .kpi.info .v { color:#4F46E5 !important; }
.kpi .spark { position:absolute; right:0; bottom:0; width:46%; height:38px; opacity:.9; }
.hero-f { border-radius:18px; padding:1.3rem 1.4rem; height:100%; background:radial-gradient(420px 200px at 100% 0%, rgba(99,102,241,.16), transparent 70%), linear-gradient(180deg,#FFFFFF,#F7F8FF); border:1px solid #E3E6FA; }
.hero-f .l { font-size:.8rem; font-weight:600; color:#4F46E5 !important; text-transform:uppercase; letter-spacing:.06em; }
.hero-f .big { font-family:'JetBrains Mono',monospace; font-size:2.9rem; font-weight:600; letter-spacing:-2px; color:#0F172A !important; line-height:1.1; margin-top:.35rem; } .hero-f .big small { font-size:1.1rem; color:#64748B !important; letter-spacing:0; }
.hero-f .r { color:#475569 !important; font-size:.86rem; margin-top:.35rem; }
.badge { display:inline-block; padding:.18rem .55rem; border-radius:8px; font-size:.78rem; font-weight:600; margin-top:.6rem; } .badge.up { background:#E8FAF3; color:#047857 !important; } .badge.down { background:#FFE9ED; color:#BE123C !important; } .badge.flat { background:#F1F4F9; color:#475569 !important; }
.mini { display:flex; gap:1.3rem; margin-top:1rem; padding-top:.9rem; border-top:1px dashed #DDE2F3; } .mini div b { display:block; font-family:'JetBrains Mono',monospace; font-size:1rem; color:#0F172A !important; } .mini div span { font-size:.74rem; color:#64748B !important; }
.live { border-radius:18px; padding:1.2rem 1.3rem; background:linear-gradient(180deg,#FFFFFF,#F7F8FF); border:1px solid #E3E6FA; height:100%; }
.live .big { font-family:'JetBrains Mono',monospace; font-size:3.2rem; font-weight:600; letter-spacing:-2px; color:#0F172A !important; line-height:1; margin:.7rem 0 .2rem; } .live .big small { font-size:1.1rem; color:#64748B !important; }
.bar { margin:.55rem 0; } .bar .row { display:flex; justify-content:space-between; font-size:.8rem; color:#475569 !important; font-weight:500; } .bar .row b { font-family:'JetBrains Mono',monospace; color:#0F172A !important; }
.bar .trk { height:8px; background:#EEF1F7; border-radius:99px; margin-top:.25rem; overflow:hidden; } .bar .fil { height:100%; border-radius:99px; }
.note { border-left:3px solid #6366F1; background:#F4F5FF; padding:.7rem 1rem; border-radius:0 12px 12px 0; color:#475569 !important; font-size:.88rem; margin:.5rem 0 .9rem; } .note b, .note i { color:#1E293B !important; }
.note.amber { border-left-color:#F59E0B; background:#FFF8EB; }
.note.good { border-left-color:#10B981; background:#E8FAF3; } .note.bad { border-left-color:#F43F5E; background:#FFF1F3; } .note ul { margin:.4rem 0 0 1.1rem; padding:0; } .note li { margin:.15rem 0; }
.tblwrap { overflow-x:auto; border:1px solid #E6EAF2; border-radius:14px; background:#FFFFFF; margin:.4rem 0 1rem; }
table.tbl { border-collapse:collapse; width:100%; font-size:.84rem; } table.tbl th { text-align:right; background:#F7F9FC; color:#64748B !important; font-weight:600; padding:.55rem .8rem; border-bottom:1px solid #E6EAF2; white-space:nowrap; }
table.tbl td { text-align:right; padding:.5rem .8rem; border-bottom:1px solid #F0F3F8; color:#0F172A !important; font-family:'JetBrains Mono',monospace; font-size:.8rem; } table.tbl th:first-child, table.tbl td:first-child { text-align:left; }

/* inputs (forced light) */
[data-baseweb="select"] > div, [data-baseweb="input"], [data-baseweb="base-input"], input, [data-testid="stNumberInputContainer"] { background:#FFFFFF !important; color:#0F172A !important; border-color:#DDE2EE !important; border-radius:10px !important; }
[data-baseweb="popover"] *, ul[role="listbox"], li[role="option"] { background:#FFFFFF !important; color:#0F172A !important; } li[role="option"]:hover { background:#F1F3FF !important; }
[data-testid="stExpander"] { background:#FFFFFF; border:1px solid #E6EAF2 !important; border-radius:14px; } [data-testid="stExpander"] summary, [data-testid="stExpander"] summary p { color:#0F172A !important; font-weight:600; }
[data-testid="stBaseButton-secondary"] { background:#FFFFFF !important; color:#0F172A !important; border:1px solid #DDE2EE !important; border-radius:10px !important; font-weight:600; }
input:focus, [data-baseweb="input"]:focus-within, [data-baseweb="select"] > div:focus-within { border-color:#6366F1 !important; box-shadow:0 0 0 1px #6366F1 !important; }
[data-testid="stSlider"] [role="slider"] { background-color:#6366F1 !important; box-shadow:none !important; } [data-testid="stSliderThumbValue"], [data-testid="stSliderThumbValue"] * { color:#4F46E5 !important; }
[data-testid="stBaseButton-primary"] { background:#0F172A !important; color:#FFFFFF !important; border:0 !important; border-radius:10px !important; font-weight:600; } [data-testid="stBaseButton-primary"] p { color:#FFFFFF !important; }
</style>""", unsafe_allow_html=True)


# ------------------------------------------------------------------ UI helpers
def spark(values, color):
    v = np.asarray(values, float); v = v[~np.isnan(v)]
    if len(v) < 2: return ""
    y = 27 - (v - v.min()) / (v.max() - v.min() + 1e-9) * 22
    pts = " ".join(f"{i * 100 / (len(v) - 1):.1f},{yy:.1f}" for i, yy in enumerate(y))
    return f'<svg class="spark" viewBox="0 0 100 30" preserveAspectRatio="none"><polygon points="0,30 {pts} 100,30" fill="{color}22"/><polyline points="{pts}" fill="none" stroke="{color}" stroke-width="1.8" vector-effect="non-scaling-stroke"/></svg>'


def kpi(label, value, sub="", tone="", icon="", trend=None, trend_color=FORECAST):
    st.markdown(f'<div class="kpi {tone}"><div class="top"><div class="ic">{icon}</div><div class="l">{label}</div></div><div class="v">{value}</div><div class="s">{sub}</div>'
                f'{spark(trend, trend_color) if trend is not None else ""}</div>', unsafe_allow_html=True)


def note(text, tone=""):
    st.markdown(f'<div class="note {tone}">{text}</div>', unsafe_allow_html=True)


def table(df, nd=3):
    html = df.round(nd).to_html(index=False, classes="tbl", border=0, na_rep="–")
    st.markdown('<div class="tblwrap">' + "".join(x.strip() for x in html.splitlines()) + "</div>", unsafe_allow_html=True)


@contextmanager
def panel(title, sub="", key="p"):
    with st.container(border=True, key=f"panel_{key}"):
        st.markdown(f'<div class="ph"><b>{title}</b><span>{sub}</span></div>', unsafe_allow_html=True)
        yield


def style(fig, h=440, legend=True, legend_y=1.02, top=95):
    """Title sits at the top-left; the legend gets its own row just above the plot, so they never overlap."""
    fig.update_layout(height=h + (top - 50), template="plotly_white", paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)", hovermode="x unified",
                      font=dict(family="Inter, sans-serif", size=12.5, color=INK), margin=dict(l=10, r=10, t=top, b=10),
                      title=dict(x=0, xanchor="left", y=0.985, yanchor="top", font=dict(size=15, color=INK)),
                      hoverlabel=dict(bgcolor="#FFFFFF", bordercolor=AXIS, font=dict(family="Inter, sans-serif", size=12, color=INK)),
                      legend=dict(orientation="h", x=0, xanchor="left", y=legend_y, yanchor="bottom", bgcolor="rgba(0,0,0,0)"), showlegend=legend)
    fig.update_xaxes(gridcolor=GRID, zeroline=False, linecolor=AXIS, showspikes=False, tickfont=dict(color=MUTED), title_font=dict(color=MUTED)); fig.update_yaxes(gridcolor=GRID, zeroline=False, tickfont=dict(color=MUTED), title_font=dict(color=MUTED))
    return fig


def show(fig, key=None):
    st.plotly_chart(fig, width="stretch", config={"displayModeBar": False}, key=key)


def gaps(t, *arrays, step=1):
    """Break plot lines wherever the timestamp jumps by more than `step` minutes (1 for full data; the thinning step when rows were skipped)."""
    t = pd.DatetimeIndex(t).values; brk = np.where(np.diff(t) > np.timedelta64(step, "m"))[0] + 1
    if not len(brk): return (t, *[np.asarray(a, float) for a in arrays])
    return (np.insert(t, brk, t[brk - 1] + np.timedelta64(30, "s")), *[np.insert(np.asarray(a, float), brk, np.nan) for a in arrays])


def phase_bars(values, scale):
    out = ""
    for ph, v in zip(PHASES, values):
        w = 0 if pd.isna(v) else min(100, max(0, v / scale * 100))
        out += (f'<div class="bar"><div class="row"><span>{PNAME[ph]} phase</span><b>{"–" if pd.isna(v) else f"{v:,.0f} kW"}</b></div>'
                f'<div class="trk"><div class="fil" style="width:{w:.0f}%;background:{PHASE_COL[ph]}"></div></div></div>')
    return out


# ------------------------------------------------------------------ data (device_16_prediction_dataset.csv)
FEATS = ["V1", "V2", "V3", "I1", "I2", "I3", "PF"]
KWCOLS = PHASES
CURR, VOLT = ["I1", "I2", "I3"], ["V1", "V2", "V3"]
NEEDED = FEATS + KWCOLS
CLOCK_COLS = ("Timestamps", "Timestamp", "Timestamp_Index")
SX_STEP, SX_WINDOW_H, SX_MAXITER = 5, 24, 50                 # SARIMAX: one point every 5 min, learns from the last 24 h (288 points)
SX_WSTEPS = SX_WINDOW_H * 60 // SX_STEP
SX_CONFIGS = {"SARIMA(1,1,1)x(1,0,0,12)": dict(order=(1, 1, 1), seasonal_order=(1, 0, 0, 12)),     # s = 12 x 5 min = a 1-hour cycle
              "ARIMA(1,1,1)": dict(order=(1, 1, 1), seasonal_order=(0, 0, 0, 0)),
              "ARIMA(2,1,2)": dict(order=(2, 1, 2), seasonal_order=(0, 0, 0, 0)),
              "SARIMA(1,1,1)x(0,1,1,12)": dict(order=(1, 1, 1), seasonal_order=(0, 1, 1, 12))}


def find_data():
    for folder in (HERE, Path.cwd(), HERE / "data"):
        if (folder / DATA_NAME).exists(): return folder / DATA_NAME
    return None


def _parse(buffer, name):
    name = name.lower()
    if name.endswith(".parquet"): return pd.read_parquet(buffer)
    if name.endswith((".xlsx", ".xls")): return pd.read_excel(buffer)
    return pd.read_csv(buffer, encoding="utf-8-sig", low_memory=False)


@st.cache_data(show_spinner="Reading the file…")
def read_raw_path(path, mtime):
    return _parse(path, path)


@st.cache_data(show_spinner="Reading the file…")
def read_raw_bytes(data, name):
    return _parse(io.BytesIO(data), name)


@st.cache_data(show_spinner="Cleaning the data…")
def clean_data(raw, device, start_iso):
    """Same cleaning as the notebooks: pick one device, build the minute clock, tiny negatives -> 0, impossible meter values removed.
    Returns the cleaned rows (one per recorded minute, total KW = R + Y + B) and a few facts about the file."""
    raw = raw.copy()
    if "DeviceID" in raw.columns and raw["DeviceID"].nunique() > 1 and device is not None: raw = raw[raw["DeviceID"] == device]
    missing = [c for c in NEEDED if c not in raw.columns]
    if missing: raise ValueError(f"These columns are missing: {missing}. The file has: {list(raw.columns)[:30]}")
    tcol = next((c for c in CLOCK_COLS if c in raw.columns), None)
    if tcol in ("Timestamps", "Timestamp"):
        ts, clock = pd.to_datetime(raw[tcol], dayfirst=True, errors="coerce"), "file"
    else:                                              # Timestamp_Index or no clock at all: one row / index step = one minute from the chosen start
        idx = pd.to_numeric(raw[tcol], errors="coerce") if tcol else pd.Series(np.arange(len(raw), dtype=float), index=raw.index)
        ts, clock = pd.Timestamp(start_iso) + pd.to_timedelta((idx - idx.min()).to_numpy(float), unit="m"), "assumed"
    df = raw[NEEDED].apply(pd.to_numeric, errors="coerce").assign(Timestamps=pd.DatetimeIndex(ts))
    n_read = len(df)
    df = df.dropna(subset=NEEDED + ["Timestamps"])
    df["Timestamps"] = df["Timestamps"].dt.floor("min")
    df = df.drop_duplicates("Timestamps").sort_values("Timestamps")
    df[KWCOLS + CURR] = df[KWCOLS + CURR].clip(lower=0); df["PF"] = df["PF"].clip(-1, 1)
    df = df[(df[KWCOLS] < 5e3).all(axis=1) & (df[VOLT] < 1000).all(axis=1) & (df[CURR] < 1e4).all(axis=1)]
    df["KW"] = df[KWCOLS].sum(axis=1)
    tl = df[["Timestamps", "KW"] + KWCOLS + CURR + VOLT + ["PF"]].reset_index(drop=True)
    return tl, dict(clock=clock, tcol=tcol, rows_read=n_read, rows_used=len(tl))


def raw_devices(raw):
    return sorted(raw["DeviceID"].dropna().unique().tolist()) if "DeviceID" in raw.columns and raw["DeviceID"].nunique() > 1 else []


@st.cache_data(show_spinner="Building the minute grid…")
def make_grid(tl, smooth):
    """D_raw = every minute of the period (NaN where the meter sent nothing); D = rolling mean of the PAST `smooth` minutes only."""
    grid = pd.date_range(tl["Timestamps"].min(), tl["Timestamps"].max(), freq="min")
    D_raw = tl.drop(columns="KW").set_index("Timestamps").reindex(grid)
    D_raw["KW"] = D_raw[KWCOLS].sum(axis=1, min_count=3)
    D = D_raw.rolling(int(smooth), min_periods=1 if smooth <= 1 else max(2, int(smooth * 2 / 3))).mean()
    return D_raw, D


@st.cache_data
def timeline_summary(tl, run_kw):
    measured, running = tl.KW.notna(), tl.KW > run_kw
    span_min = int((tl.Timestamps.max() - tl.Timestamps.min()).total_seconds() // 60) + 1
    day = tl.Timestamps.dt.date
    daily = pd.DataFrame({"day": day, "kwh": tl.KW.fillna(0) / 60, "run_h": running / 60}).groupby("day").sum()
    daily["avg_run"] = tl[running].groupby(day[running]).KW.mean().reindex(daily.index)
    heat = tl.assign(day=day, hour=tl.Timestamps.dt.hour).pivot_table(index="day", columns="hour", values="KW", aggfunc="mean")
    heat = heat.reindex(index=[d.date() for d in pd.date_range(heat.index.min(), heat.index.max())], columns=range(24))   # every day x every hour; NaN = no data
    last_run = tl.Timestamps[running].max() if running.any() else tl.Timestamps.min(); last_seen = tl.Timestamps[measured].max()
    return dict(energy=float(tl.KW.fillna(0).sum() / 60), run_min=int(running.sum()), idle_min=int((measured & ~running).sum()),
                nodata_min=max(0, span_min - int(measured.sum())), peak=float(tl.KW.max()), avg_run=float(tl.KW[running].mean()) if running.any() else 0.0,
                last_run=last_run, last_seen=last_seen, daily=daily, heat=heat, util=float(running.sum() / max(measured.sum(), 1) * 100),
                now_running=bool(tl.KW[measured].iloc[-1] > run_kw))


# ------------------------------------------------------------------ shared forecast-control options (Load prediction tab)
CUSTOM = "Custom date & time"
LENGTHS = {"30 minutes": 30, "1 hour": 60, "6 hours": 360, "12 hours": 720, "24 hours": 1440, "3 days": 4320, "7 days": 10080}
STEPS = {"1 min": 1, "5 min": 5, "10 min": 10, "15 min": 15, "30 min": 30, "1 hour": 60, "3 hours": 180}
HISTS = {"1 hour": 60, "3 hours": 180, "6 hours": 360, "12 hours": 720, "24 hours": 1440}
MAX_HORIZON = 10080                                   # 7 days of one-minute steps


def thin_step(frame, max_points=4000):
    """Every how-many-th row thin() keeps."""
    return max(1, int(np.ceil(len(frame) / max_points)))


def thin(frame, max_points=4000):
    """Plot at most ~max_points rows (the full data is still used for every calculation)."""
    k = int(np.ceil(len(frame) / max_points)); return frame.iloc[::k] if k > 1 else frame


# ------------------------------------------------------------------ phase imbalance maths (the notebooks' formula)
def analyse(P, thr=None, min_kw=None, sev=None):
    """P = kW (or A, or V) of Red, Yellow, Blue in 3 columns.  imbalance % = max(|R-avg|, |Y-avg|, |B-avg|) / avg x 100,  avg = (R+Y+B) / 3.
    170 / 169 / 160 kW -> 3.81 %.  A row is 'loaded' when the average phase value is above min_kw (otherwise the machine is off)."""
    thr = ALARM_PCT if thr is None else thr
    min_kw = MIN_KW if min_kw is None else min_kw
    sev = SEVERE_PCT if sev is None else sev
    P = np.asarray(P, float).reshape(-1, 3); n = len(P)
    avg = P.mean(axis=1); loaded = avg > min_kw
    with np.errstate(divide="ignore", invalid="ignore"):
        d = (P - avg[:, None]) / avg[:, None] * 100
    d[~loaded, :] = np.nan
    worst = np.nan_to_num(np.abs(d), nan=-1).argmax(axis=1)
    wdev = np.where(loaded, d[np.arange(n), worst], np.nan)
    imb = np.abs(wdev); alarm = loaded & (imb > thr)
    status = np.where(~loaded, "NO LOAD", np.where(~alarm, "OK", np.where(imb > sev, "SEVERE", "ALARM")))
    return pd.DataFrame({"Imbalance_pct": imb, "R_dev": d[:, 0], "Y_dev": d[:, 1], "B_dev": d[:, 2], "Worst_phase": np.where(loaded, np.array(NAMES)[worst], "-"),
                         "Worst_dev": wdev, "Direction": np.where(~loaded, "-", np.where(wdev > 0, "HIGH", "LOW")), "Alarm": alarm, "Status": status, "Loaded": loaded})


def transfers(vals, names):
    """Smallest set of moves that makes the three values equal: [(from, to, amount), ...] (largest surplus -> largest deficit first)."""
    vals = np.asarray(vals, float); delta = vals - vals.mean()
    sur = sorted([[d, n] for d, n in zip(delta, names) if d > 1e-9], reverse=True)
    dfc = sorted([[-d, n] for d, n in zip(delta, names) if d < -1e-9], reverse=True)
    out, i, j = [], 0, 0
    while i < len(sur) and j < len(dfc):
        m = min(sur[i][0], dfc[j][0]); out.append((sur[i][1], dfc[j][1], m))
        sur[i][0] -= m; dfc[j][0] -= m
        if sur[i][0] <= 1e-9: i += 1
        if dfc[j][0] <= 1e-9: j += 1
    return out


@st.cache_data(show_spinner="Measuring the imbalance and looking for the cause…")
def diagnose(frame, thr, min_kw, sev):
    """Everything the 'Phase imbalance' tab shows: size, which colour, WHY (current / voltage / power factor), load-level behaviour, a move plan."""
    A = analyse(frame[KWCOLS].to_numpy(), thr, min_kw, sev); A.index = frame.index
    ok = A.Loaded.to_numpy() & (frame[CURR] > 0).all(axis=1).to_numpy() & (frame[VOLT] > 0).all(axis=1).to_numpy()
    out = dict(A=A, n_all=len(frame), n_loaded=int(A.Loaded.sum()), n_ok=int(ok.sum()), empty=int(ok.sum()) < 20)
    if out["empty"]: return out
    d = frame[ok]; K = d[KWCOLS].to_numpy(float); I = d[CURR].to_numpy(float); V = d[VOLT].to_numpy(float)
    tot = K.sum(axis=1); PFp = np.clip(K * 1000 / (V * I), 0, 1.5)            # PF of each phase = kW / (V x I)
    rows = np.arange(len(K)); w = np.abs(K - K.mean(axis=1, keepdims=True)).argmax(axis=1)
    dev = lambda X: (X[rows, w] / X.mean(axis=1) - 1) * 100                  # the worst phase vs the average of the 3, in %
    nema = lambda X: np.abs(X - X.mean(axis=1, keepdims=True)).max(axis=1) / X.mean(axis=1) * 100
    dP, dV, dI, dF = dev(K), dev(V), dev(I), dev(PFp)
    imb_kw, imb_I, imb_V = np.abs(dP), nema(I), nema(V)
    sh = pd.DataFrame({"Current": np.abs(dI), "Voltage": np.abs(dV), "Power factor": np.abs(dF)})
    share = (sh.div(sh.sum(axis=1).replace(0, np.nan), axis=0).mean() * 100).fillna(0)
    cor = lambda a, b: float(np.corrcoef(a, b)[0, 1]) if np.std(a) > 0 and np.std(b) > 0 else float("nan")
    cnt = np.bincount(w, minlength=3); main = int(cnt.argmax()); others = [i for i in range(3) if i != main]
    per = pd.DataFrame({"Worst phase in % of minutes": cnt / len(w) * 100, "of those HIGH %": [np.mean(dP[w == i] > 0) * 100 if cnt[i] else np.nan for i in range(3)],
                        "Avg current A": I.mean(axis=0), "95 % of minutes below A": np.percentile(I, 95, axis=0), "Peak A": I.max(axis=0),
                        "Avg voltage V": V.mean(axis=0), "Avg kW": K.mean(axis=0), "PF (kW / V·I)": PFp.mean(axis=0)}, index=NAMES)
    extra_A = I[:, main] - I[:, others].mean(axis=1); extra_kW = K[:, main] - K[:, others].mean(axis=1)
    levels = pd.DataFrame({"total": tot, "imb": imb_kw, "extra_A": extra_A, "extra_kW": extra_kW}).groupby(pd.qcut(tot, 5, duplicates="drop"), observed=True).mean()
    levels.index = [f"{iv.left:.0f}–{iv.right:.0f} kW" for iv in levels.index]
    lvA = levels["extra_A"]; const = len(lvA) > 2 and (lvA.iloc[1:].max() - lvA.iloc[1:].min()) / max(abs(lvA.iloc[1:].mean()), 1e-9) < 0.25
    Im, Km = I.mean(axis=0), K.mean(axis=0)
    plan = pd.DataFrame([(f, t, a, a * Km[NAMES.index(f)] / Im[NAMES.index(f)]) for f, t, a in transfers(Im, NAMES) if a >= 1.0], columns=["From", "To", "Move A", "Move kW"])
    neutral = np.sqrt(np.maximum(0, (I ** 2).sum(axis=1) - I[:, 0] * I[:, 1] - I[:, 1] * I[:, 2] - I[:, 2] * I[:, 0]))      # neutral current of a 4-wire supply (same PF angle on all phases)
    loss = ((I ** 2).sum(axis=1) / (3 * I.mean(axis=1) ** 2) - 1) * 100                                                  # extra I²R heating against a perfectly balanced load
    by_hour = pd.Series(imb_kw, index=d.index).groupby(d.index.hour).mean()
    status = A.Status[A.Loaded].value_counts().reindex(["OK", "ALARM", "SEVERE"]).fillna(0)
    return dict(out, empty=False, mean_imb=float(imb_kw.mean()), p95_imb=float(np.percentile(imb_kw, 95)), pct_above=float((A.Imbalance_pct[A.Loaded] > thr).mean() * 100),
                mean_I=float(imb_I.mean()), mean_V=float(imb_V.mean()), p95_V=float(np.percentile(imb_V, 95)), corr_I=cor(imb_kw, imb_I), corr_V=cor(imb_kw, imb_V), share=share,
                per=per, main=main, main_name=NAMES[main], extra_A=float(extra_A.mean()), extra_kW=float(extra_kW.mean()), levels=levels, const=bool(const), lvA_const=float(lvA.iloc[1:].mean()),
                ctot=cor(imb_kw, tot), plan=plan, Im=Im, Km=Km, neutral_mean=float(neutral.mean()), neutral_p95=float(np.percentile(neutral, 95)), loss=float(loss.mean()),
                by_hour=by_hour, status=status, pf_phase=float(PFp.mean()), pf_meter=float(d["PF"].mean()), high_share=float(per.iloc[main, 1]))


def imbalance_actions(dg, thr, sev, real_clock, rating):
    """Turns the measured numbers into a short action list. Returns (tone, headline, [html lines])."""
    mi, ab, nm, sh = dg["mean_imb"], dg["pct_above"], dg["main_name"], dg["share"]
    if mi > sev: tone, head = "bad", f"Severe and constant: the average imbalance is {mi:.1f} % (alarm {thr:g} %, severe {sev:g} %)"
    elif mi > thr: tone, head = "amber", f"Above the limit: the average imbalance is {mi:.1f} % (alarm {thr:g} %)"
    else: tone, head = "good", f"Within the limit: the average imbalance is {mi:.1f} % (alarm {thr:g} %)"
    acts = [f"<b>How often:</b> above the {thr:g} % limit in <b>{ab:.0f} %</b> of the loaded minutes, 95 % of minutes below {dg['p95_imb']:.1f} %. "
            + ("This is permanent, not an event - fix it in how the loads are connected to the phases, not by operating the plant differently." if ab > 50
               else "It comes and goes - look at when it happens (hour of day, which machine is on) before moving any wiring.")]
    top = sh.idxmax()
    if top == "Current":
        acts.append(f"<b>Cause = load, not supply.</b> Current explains {sh['Current']:.0f} % of how far the worst phase is from the average (voltage {sh['Voltage']:.0f} %, power factor {sh['Power factor']:.0f} %). "
                    f"The single-phase loads are not shared equally; <b>{nm}</b> is the worst phase in {dg['per'].iloc[dg['main'], 0]:.0f} % of minutes and is {'HIGH' if dg['high_share'] >= 50 else 'LOW'} in {max(dg['high_share'], 100 - dg['high_share']):.0f} % of them.")
    elif top == "Voltage":
        acts.append(f"<b>Cause = supply voltage.</b> Voltage explains {sh['Voltage']:.0f} % of the deviation (average voltage unbalance {dg['mean_V']:.1f} %). Check the incoming supply, the transformer tap and the upstream feeders, and raise it with the utility.")
    else:
        acts.append(f"<b>Cause = power factor.</b> The phases have different power factors ({sh['Power factor']:.0f} % of the deviation). Check capacitor banks and the single-phase loads (rectifiers, drives, lighting) on each phase.")
    if top != "Voltage" and dg["mean_V"] > 1:
        acts.append(f"<b>Supply is not perfect either:</b> voltage unbalance is {dg['mean_V']:.1f} % on average ({dg['p95_V']:.1f} % at the 95th percentile). Motors start losing capacity above about 1 % "
                    f"(rule of thumb: winding temperature rises about 2 × (voltage unbalance %)² - roughly {2 * dg['mean_V'] ** 2:.0f} % here). Above 2 % ask the utility to correct it.")
    elif top != "Voltage": acts.append(f"<b>Supply is fine:</b> voltage unbalance is only {dg['mean_V']:.1f} % on average, so the supply is not the reason.")
    if len(dg["plan"]):
        first = dg["plan"].iloc[0]
        steps = " and ".join(f"<b>{r['To']}</b> (≈ {r['Move A']:.0f} A, ≈ {r['Move kW']:.0f} kW)" for _, r in dg["plan"].iterrows())
        acts.append(f"<b>What to move:</b> take single-phase load off <b>{first['From']}</b> and give it to {steps}. "
                    f"Whole machines cannot be split, so pick the loads that add up closest to these amps and re-check this page - the aim is to be inside {thr:g} %.")
    if dg["const"]: acts.append(f"<b>Looks like a fixed load:</b> from the second load level upward {nm} carries about <b>{dg['lvA_const']:.0f} A</b> more than the other two phases, whatever the plant draws. "
                                f"That points to one or a few single-phase loads (lighting, heaters, welders, control or auxiliary panels) wired to {nm} - find them first.")
    if dg["ctot"] < -0.2: acts.append(f"<b>Worse at light load</b> (correlation {dg['ctot']:.2f} with total kW): {dg['levels']['imb'].iloc[0]:.1f} % at {dg['levels'].index[0]} against {dg['levels']['imb'].iloc[-1]:.1f} % at {dg['levels'].index[-1]}. "
                                      "The extra single-phase load is the same, but it is a bigger share when the plant draws less - rebalance with the machine at light load in mind.")
    elif dg["ctot"] > 0.2: acts.append(f"<b>Worse at heavy load</b> (correlation {dg['ctot']:.2f} with total kW): look for an overloaded phase at full production.")
    if real_clock and len(dg["by_hour"]) >= 6 and dg["by_hour"].max() > 1.25 * dg["by_hour"].min():
        hh, lh = int(dg["by_hour"].idxmax()), int(dg["by_hour"].idxmin())
        acts.append(f"<b>When:</b> worst around {hh:02d}:00 ({dg['by_hour'].max():.1f} %), best around {lh:02d}:00 ({dg['by_hour'].min():.1f} %) - check which machines run at {hh:02d}:00.")
    if dg["neutral_mean"] >= 1:
        acts.append(f"<b>Neutral and heating:</b> in a 4-wire supply the imbalance returns through the neutral, about <b>{dg['neutral_mean']:.0f} A</b> on average ({dg['neutral_p95']:.0f} A at the 95th percentile) - check the neutral conductor and its protection. "
                    f"Extra I²R heating against a balanced load is about {dg['loss']:.1f} % " + ("(small: the cost is low, the risk is the hot {0} phase).".format(nm) if dg["loss"] < 3 else "(significant: it also costs energy)."))
    pk = dg["per"]["Peak A"]; p95 = dg["per"]["95 % of minutes below A"]
    if rating > 0:
        u = (pk / rating * 100); hot = u.idxmax()
        acts.append(f"<b>Rating check ({rating:,.0f} A):</b> {hot} peaks at {pk[hot]:,.0f} A = <b>{u[hot]:.0f} %</b> of the rating (95 % of minutes below {p95[hot]:,.0f} A). "
                    + ("It is over the rating - reduce its load now." if u[hot] > 100 else "Little headroom - move load away before adding any." if u[hot] > 85 else "There is headroom, but the loaded phase will reach the limit first."))
    else: acts.append(f"<b>Rating check:</b> {pk.idxmax()} peaks at {pk.max():,.0f} A. Enter the breaker / cable rating in 'Data & limits' to see the headroom per phase.")
    return tone, head, acts


# ------------------------------------------------------------------ SARIMAX per phase (learns from the last 24 hours, 5-minute steps)
def _fit_window(window, steps, cfg):
    """One SARIMAX per phase on a (288, 3) window -> forecast, 95 % range and fitted values. A failed fit falls back to 'same as now'."""
    mean, lo, hi = np.zeros((steps, 3)), np.zeros((steps, 3)), np.zeros((steps, 3)); fit = np.full((len(window), 3), np.nan); ok = []
    for j in range(3):
        y = window[:, j]
        try:
            res = SARIMAX(y, enforce_stationarity=False, enforce_invertibility=False, **cfg).fit(disp=False, maxiter=SX_MAXITER, cov_type="none")
            fc = res.get_forecast(steps); ci = np.asarray(fc.conf_int(alpha=0.05), float)
            mean[:, j], lo[:, j], hi[:, j], fit[:, j] = np.asarray(fc.predicted_mean, float), ci[:, 0], ci[:, 1], np.asarray(res.fittedvalues, float); ok.append(True)
        except Exception:
            mean[:, j], lo[:, j], hi[:, j] = y[-1], np.nan, np.nan; ok.append(False)
    return dict(mean=np.clip(mean, 0, None), lo=np.clip(lo, 0, None), hi=hi, fit=fit, ok=ok)


@st.cache_data(show_spinner="Fitting SARIMAX on the last 24 hours (Red, Yellow, Blue)…")
def sx_forecast(window, steps, cfg_name):
    return _fit_window(window, steps, SX_CONFIGS[cfg_name])


def newest_window(arr):
    """The newest 24 h of one series (every 5 min). Short gaps are filled by straight lines. Returns (window, index of its last row, number of filled points) or None."""
    ok = ~np.isnan(arr).any(axis=1)
    if not ok.any(): return None
    pn = int(np.flatnonzero(ok)[-1]); ih = pn - SX_STEP * np.arange(SX_WSTEPS - 1, -1, -1)
    if ih[0] < 0: return None
    w = pd.DataFrame(arr[ih]); miss = int(w.isna().any(axis=1).sum())
    if miss > 0.1 * len(w): return None
    return w.interpolate(limit_direction="both").to_numpy(float), pn, miss


def sx_backtest(D_raw, D, train_on, cfg_name, n_origins, horizons, progress=None):
    """At n_origins moments in the last 20 % of the file: SARIMAX learns from the previous 24 h only, forecasts 15 / 30 / 60 min, and is scored against
    what really happened and against 'same as now'. Returns a table (or None when there were too few usable moments)."""
    SRC = (D_raw if train_on == "Real values" else D)[KWCOLS].to_numpy(float)
    K = max(horizons) // SX_STEP; first = int(len(SRC) * 0.8); last = len(SRC) - 1 - K * SX_STEP
    if last <= first: return None
    F, T, Z = [], [], []
    for n_i, p in enumerate(np.unique(np.linspace(first, last, n_origins).astype(int)), 1):
        ih, ifu = p - SX_STEP * np.arange(SX_WSTEPS - 1, -1, -1), p + SX_STEP * np.arange(1, K + 1)
        if ih[0] >= 0:
            w, fut = SRC[ih], SRC[ifu]
            if not (np.isnan(w).any() or np.isnan(fut).any()):
                F.append(_fit_window(w, K, SX_CONFIGS[cfg_name])["mean"]); T.append(fut); Z.append(np.repeat(w[-1:], K, axis=0))
        if progress: progress(min(n_i / n_origins, 1.0))
    if len(F) < 3: return None
    F, T, Z = np.array(F), np.array(T), np.array(Z); rows = []
    for h in horizons:
        k = h // SX_STEP - 1; truth = T[:, k, :]; ib = analyse(truth).Imbalance_pct.to_numpy()
        for name, X in (("SARIMAX", F[:, k, :]), ("Same as now", Z[:, k, :])):
            rows.append({"Minutes ahead": h, "Method": name, "Phase kW error": float(np.abs(X - truth).mean()), "Total kW error": float(np.abs(X.sum(axis=1) - truth.sum(axis=1)).mean()),
                         "Imbalance error (pts)": float(np.nanmean(np.abs(analyse(X).Imbalance_pct.to_numpy() - ib))), "Moments": len(F)})
    return pd.DataFrame(rows)


def bt_verdict(res):
    """Compare SARIMAX with 'same as now' over all horizons. Returns (tone, text)."""
    s, z = res[res.Method == "SARIMAX"].reset_index(drop=True), res[res.Method == "Same as now"].reset_index(drop=True)
    gain = lambda col: float((1 - s[col].mean() / max(z[col].mean(), 1e-9)) * 100) if np.isfinite(s[col].mean()) and np.isfinite(z[col].mean()) else 0.0
    g_tot, g_imb = gain("Total kW error"), gain("Imbalance error (pts)")
    if g_tot > 3 and g_imb > 3: return "good", f"SARIMAX adds value on your data: total kW error {g_tot:+.0f} % and imbalance error {g_imb:+.0f} % against 'same as now' (positive = better)."
    if g_tot < -3 or g_imb < -3: return "bad", f"SARIMAX is WORSE than simply repeating the last value (total kW {g_tot:+.0f} %, imbalance {g_imb:+.0f} %). Use the forecast only as a trend hint."
    return "amber", f"SARIMAX is about as good as 'same as now' (total kW {g_tot:+.0f} %, imbalance {g_imb:+.0f} %): the phases change too slowly for a model to add much - treat the forecast as 'the imbalance will stay about where it is now'."


def forecast_advice(P, rep, t_fut, chronic, verdict, long_horizon):
    """What to do about the forecast imbalance. P = forecast kW (steps x 3), rep = analyse(P). Returns (tone, headline, [html lines])."""
    n = len(P); n_al, n_sev = int(rep.Alarm.sum()), int((rep.Status == "SEVERE").sum()); acts = []
    word = {"up": "rising", "down": "falling", "flat": "steady", "idle": "idle"}
    first, last = P[0], P[-1]
    if not rep.Loaded.any(): return "", "The machine is expected to stay idle", ["No imbalance can build up while the load is below the minimum - nothing to do."]
    for j, nm in enumerate(NAMES): acts.append(f"<b>{nm}</b> {word[trend_kind(first[j], last[j])]}: {first[j]:,.1f} → {last[j]:,.1f} kW{pct_text(pct_change(first[j], last[j]))}")
    imb = rep.Imbalance_pct.to_numpy(); i_max = int(np.nanargmax(imb))
    if n_sev: tone, head = "bad", f"SEVERE imbalance expected: {n_sev} of {n} forecast points above {SEVERE_PCT:g} % (highest {imb[i_max]:.1f} %)"
    elif n_al:
        idx = int(np.flatnonzero(rep.Alarm.to_numpy())[0]); mins = int((t_fut[idx] - (t_fut[0] - pd.Timedelta(minutes=SX_STEP))).total_seconds() // 60)
        tone, head = ("bad" if n_al > n / 2 else "amber"), f"Alarm expected from {t_fut[idx]:%H:%M} (in {mins} min): {n_al} of {n} points above {ALARM_PCT:g} %, highest {imb[i_max]:.1f} %"
    else: tone, head = "good", f"No alarm expected: the imbalance stays at {np.nanmin(imb):.1f}–{np.nanmax(imb):.1f} % (limit {ALARM_PCT:g} %)"
    r = rep.iloc[i_max]
    if np.isfinite(r.Worst_dev): acts.append(f"At the worst point ({t_fut[i_max]:%H:%M}) <b>{r.Worst_phase}</b> is <b>{abs(r.Worst_dev):.1f} % {'above' if r.Worst_dev > 0 else 'below'}</b> the average of the three phases.")
    mv = transfers(P.mean(axis=0), NAMES)
    if n_al or n_sev:
        if mv: acts.append("<b>Act before it happens:</b> " + "; ".join(f"move about <b>{a:,.1f} kW</b> of single-phase load from <b>{f}</b> to <b>{t}</b>" for f, t, a in mv if a >= 0.5) + " (average of the forecast window).")
        if n_sev: acts.append("SEVERE points: check for a loose or failed connection, a single-phase fault or a tripped single-phase protection if the load really is that uneven.")
    if chronic is not None and (n_al or n_sev):
        acts.append(f"This matches the measured pattern ({chronic:.0f} % of loaded minutes are above the limit): it is the usual imbalance, not a new fault." if chronic > 50
                    else f"Only {chronic:.0f} % of the measured loaded minutes are above the limit, so an alarm here is unusual - check what changed on the plant.")
    if verdict: acts.append(verdict)
    else: acts.append("Reliability not tested yet: press <b>Run backtest</b> to see whether SARIMAX beats 'same as now' on your file (in your notebook the 15-minute model was only about equal to it).")
    if long_horizon: acts.append("Beyond about 1 hour the 95 % range widens and the forecast drifts towards a flat line - use the far end as a guide only.")
    return tone, head, acts


# ------------------------------------------------------------------ Load prediction (ARIMA on the rolling total kW)
@st.cache_data(show_spinner="Training ARIMA and forecasting…")
def predict_kw(meter_df, arima_order, train_split_pct, backtest_block_min, rolling_window_min):
    # 1) clean + rolling mean of KW (the series ARIMA models) - same steps as the notebook
    raw = (meter_df[["Timestamps", "KW"]].dropna().sort_values("Timestamps").drop_duplicates("Timestamps").set_index("Timestamps"))
    rolling_kw = raw["KW"].rolling(rolling_window_min, min_periods=rolling_window_min).mean().dropna().astype(float)

    # 2) chronological train / test split
    n_train = int(len(rolling_kw) * train_split_pct / 100)
    train_series, test_series = rolling_kw.iloc[:n_train], rolling_kw.iloc[n_train:]
    if len(train_series) < 50 or len(test_series) < 5: raise ValueError("Not enough data for this train/test split.")
    train_values = pd.Series(train_series.to_numpy(), name="KW")        # plain 0..n index so the model can be extended later

    # 3) train ARIMA on the training part only (or reuse the saved, identical parameters)
    model = ARIMA(train_values, order=arima_order, enforce_stationarity=False, enforce_invertibility=False)
    same_as_saved = False
    arima_fit = model.fit()
    train_fitted = np.asarray(arima_fit.predict(start=0, end=len(train_values) - 1, dynamic=False)).reshape(-1)
    train_table = pd.DataFrame({"Timestamp": train_series.index, "Actual": train_series.to_numpy(), "Fitted": train_fitted})
    train_table["Residual"] = train_table.Actual - train_table.Fitted

    # 4) test: forecast one block BEFORE seeing it, then reveal its actual values (parameters are never re-fitted)
    next_position, forecast_values, actual_values, block_timestamps = len(train_values), [], [], []
    running_fit = arima_fit
    for block_start in range(0, len(test_series), backtest_block_min):
        actual_block = test_series.iloc[block_start:block_start + backtest_block_min]
        forecast_values.extend(np.asarray(running_fit.forecast(steps=len(actual_block))).reshape(-1))
        actual_values.extend(actual_block.to_numpy()); block_timestamps.extend(actual_block.index)
        running_fit = running_fit.extend(pd.Series(actual_block.to_numpy(dtype=float), index=pd.RangeIndex(next_position, next_position + len(actual_block)), name="KW"))
        next_position += len(actual_block)
    test_table = pd.DataFrame({"Timestamp": block_timestamps, "Actual": actual_values, "Forecast": forecast_values}).sort_values("Timestamp").reset_index(drop=True)
    test_table["Residual"] = test_table.Actual - test_table.Forecast

    def score(actual, predicted):
        return dict(RMSE=float(np.sqrt(mean_squared_error(actual, predicted))), MAE=float(mean_absolute_error(actual, predicted)), R2=float(r2_score(actual, predicted)))

    # 5) future forecast: starts right after the last test timestamp, same parameters (up to 7 days; the page slices what you ask for)
    future = running_fit.get_forecast(steps=MAX_HORIZON); band = np.asarray(future.conf_int(alpha=0.05)); last_time = test_series.index[-1]
    return dict(train=train_table, test=test_table, m_train=score(train_table.Actual, train_table.Fitted), m_test=score(test_table.Actual, test_table.Forecast),
                t2=test_series.index[0], t3=last_time, n_train=len(train_series), n_test=len(test_series), idle_share=float((raw["KW"] < 1).mean()),
                test_constant=bool(test_series.nunique() == 1), level=float(rolling_kw.mean()), reused_saved_fit=same_as_saved,
                future=dict(times=pd.date_range(last_time + pd.Timedelta(minutes=1), periods=MAX_HORIZON, freq="min"), mean=np.asarray(future.predicted_mean),
                            lower=np.maximum(band[:, 0], 0), upper=band[:, 1], last_time=last_time, last_value=float(test_series.iloc[-1]), history=rolling_kw.tail(1440)))


# ------------------------------------------------------------------ power summary helpers (energy, first / last point, averaged blocks, advice)
AVG_BLOCKS = {"30 min": 30, "1 hour": 60, "2 hours": 120, "3 hours": 180, "6 hours": 360, "12 hours": 720, "24 hours": 1440}


def span_text(minutes):
    minutes = int(minutes)
    return f"{minutes} min" if minutes < 60 else f"{minutes // 60} h" if minutes % 60 == 0 else f"{minutes / 60:.1f} h"


def block_averages(values, last_time, minutes):
    """Average of the 1-minute forecast over consecutive blocks of `minutes` (the last block may be shorter), with the change from block to block."""
    v = pd.Series(np.asarray(values, float), index=pd.date_range(last_time + pd.Timedelta(minutes=1), periods=len(values), freq="min"))
    rows = [(v.index[i], v.index[min(i + minutes, len(v)) - 1], float(v.iloc[i:i + minutes].mean()), len(v.iloc[i:i + minutes])) for i in range(0, len(v), minutes)]
    out = pd.DataFrame(rows, columns=["Start", "End", "Avg kW", "Minutes"])
    out["Change kW"] = out["Avg kW"].diff()
    prev = out["Avg kW"].shift(1)
    out["Change %"] = np.where(prev >= 1, out["Change kW"] / prev * 100, np.nan)
    return out


def block_label(row):
    return f"{row.Start:%d %b %H:%M} → {row.End:%H:%M}"


def trend_kind(first, last):
    """idle / flat / up / down - a change counts only if it is at least 1 kW and 5 % of the larger value."""
    top = max(abs(first), abs(last))
    if top < 1: return "idle"
    if abs(last - first) < max(1.0, 0.05 * top): return "flat"
    return "up" if last > first else "down"


def pct_change(first, last):
    return (last - first) / first * 100 if first >= 1 else float("nan")


def pct_text(p):
    return "" if pd.isna(p) else f" ({p:+.1f} %)"


def advice_html(head, actions):
    return f"<b>{head}</b><ul>" + "".join(f"<li>{a}</li>" for a in actions) + "</ul>"


def pf_action(avg, pf, target=0.95):
    """Power factor advice from the actual average power factor and the forecast average load."""
    if not (pf and np.isfinite(pf) and 0.3 < pf < 1.0 and avg and avg >= 1): return None
    if pf >= target: return f"Power factor is already about <b>{pf:.2f}</b>: keep it, and check that capacitor banks do not over-compensate when the load drops."
    q = avg * (np.tan(np.arccos(pf)) - np.tan(np.arccos(target)))
    return (f"Power factor is about <b>{pf:.2f}</b>. Raising it to {target:.2f} at the forecast average of {avg:,.0f} kW needs roughly <b>{q:,.0f} kVAR</b> of capacitors "
            f"and cuts the kVA demand by about {(1 - pf / target) * 100:.0f} %. Many utilities charge below about 0.90-0.95 - check your tariff.")


def power_advice(kind, first, last, peak, long_horizon, ctx=None):
    """What to do when the total power requirement is rising / falling / steady, using the actual numbers. Returns (note tone, headline, actions)."""
    ctx = ctx or {}; avg, pf, demand, slope = ctx.get("avg"), ctx.get("pf"), ctx.get("demand", 0.0), ctx.get("slope", float("nan"))
    p = pct_text(pct_change(first, last)); acts = []
    if kind == "up":
        tone, head = "amber", f"Power requirement is rising: {first:,.1f} → {last:,.1f} kW{p}"
        if np.isfinite(slope): acts.append(f"The load is climbing at about <b>{slope:+,.1f} kW per hour</b>; the expected peak is <b>{peak:,.1f} kW</b>" + (f" around {ctx['peak_at']:%H:%M}." if ctx.get("peak_at") is not None else "."))
    elif kind == "down":
        tone, head = "good", f"Power requirement is falling: {first:,.1f} → {last:,.1f} kW{p}"
        if np.isfinite(slope): acts.append(f"The load is dropping at about <b>{slope:+,.1f} kW per hour</b>.")
    elif kind == "idle":
        return "", f"The machine is expected to stay idle (below 1 kW): {first:,.1f} → {last:,.1f} kW", ["No extra power is needed in this window.", "If the machine should be running now, check the supply, the start-up and the process."]
    else:
        tone, head = "", f"Power requirement is steady: {first:,.1f} → {last:,.1f} kW{p}"
        acts.append("No action needed - the change is within 5 %.")
    if demand > 0:
        use, room = peak / demand * 100, demand - peak
        acts.append(f"<b>Sanctioned demand {demand:,.0f} kW:</b> the expected peak uses <b>{use:.0f} %</b> of it (headroom {room:+,.0f} kW). "
                    + ("It is OVER the demand - shed or defer load before that time to avoid a penalty." if use > 100 else "Very little headroom - stagger motor starts and defer batch loads." if use > 90 else "Comfortable headroom."))
    elif kind == "up": acts.append(f"Compare the expected peak (<b>{peak:,.1f} kW</b>) with your sanctioned demand - enter it in 'Data & limits' for an automatic check.")
    if pf and np.isfinite(pf) and pf > 0.3 and peak >= 1: acts.append(f"At power factor {pf:.2f} the peak of {peak:,.1f} kW is about <b>{peak / pf:,.0f} kVA</b> - compare it with the transformer / generator rating.")
    pa = pf_action(avg, pf)
    if pa: acts.append(pa)
    if kind == "up": acts += ["Stagger the start-up of large motors and move deferrable loads (batch jobs, pumps, charging) to off-peak hours.", "If the rise continues in the next forecast, plan extra capacity or a load-shedding sequence for non-critical equipment."]
    elif kind == "down": acts += ["Switch off or park idle auxiliaries and standby equipment; use the low-load window for maintenance.", "If the machine should be producing, check for a stoppage or a process problem."]
    else: acts.append("Keep monitoring; the forecast updates as new readings arrive.")
    if long_horizon: acts.append("ARIMA is a short-term model: beyond 1-2 hours the forecast flattens towards the long-run average, so treat the far end as a guide.")
    return tone, head, acts


# ------------------------------------------------------------------ page
bar = st.empty()                                                  # the app bar is filled in once the data is known
src = find_data()
settings = st.expander("Data & limits", expanded=src is None)
with settings:
    up = st.file_uploader(f"Dataset - {DATA_NAME} (optional: overrides the file next to app.py)" if src else f"Upload {DATA_NAME}", type=["csv", "parquet", "xlsx"], key="upload")
if up is not None: raw, data_name = read_raw_bytes(up.getvalue(), up.name), up.name
elif src is not None: raw, data_name = read_raw_path(str(src), src.stat().st_mtime), src.name
else:
    st.info(f"Put **{DATA_NAME}** in the same folder as app.py, or upload it in the panel above.")
    st.stop()

devices = raw_devices(raw)
device = st.session_state.get("device_sel")
device = device if device in devices else (16 if 16 in devices else (devices[0] if devices else None))
start_iso = f"{st.session_state.get('start_date', dt.date(2024, 1, 1))} {st.session_state.get('start_time', dt.time(0, 0))}"
try: tl, info = clean_data(raw, device, start_iso)
except ValueError as err: st.error(str(err)); st.stop()

with settings:
    if devices: st.selectbox("Device in this file", devices, index=devices.index(device), key="device_sel")
    if info["clock"] == "assumed":
        c = st.columns([1, 1, 3])
        c[0].date_input("Date of the first reading", value=dt.date(2024, 1, 1), key="start_date"); c[1].time_input("Time of the first reading", value=dt.time(0, 0), step=300, key="start_time")
        c[2].caption("This file has no real timestamps, so every row is taken as one minute after the previous one, starting here. Times on the charts follow from it.")
    c = st.columns(6)
    ALARM_PCT = float(c[0].number_input("Alarm above (%)", 0.5, 50.0, 3.0, 0.5, key="lim_alarm", help="Phase imbalance % above this = ALARM."))
    SEVERE_PCT = float(c[1].number_input("Severe above (%)", 1.0, 100.0, 10.0, 0.5, key="lim_sev"))
    MIN_KW = float(c[2].number_input("Ignore below (kW per phase)", 0.0, 100.0, 5.0, 1.0, key="lim_min", help="Minutes with an average phase load below this count as 'machine off'."))
    SMOOTH = int(c[3].number_input("Smoothing (min)", 1, 120, 30, key="lim_smooth", help="Rolling mean of the PAST minutes only (1 = no smoothing)."))
    DEMAND = float(c[4].number_input("Sanctioned demand (kW, 0 = unknown)", 0.0, 100000.0, 0.0, 10.0, key="lim_demand"))
    RATING = float(c[5].number_input("Phase breaker / cable (A, 0 = unknown)", 0.0, 10000.0, 0.0, 10.0, key="lim_rating"))

RUN_KW = 3 * MIN_KW                                               # three-phase load above this = machine running
D_raw, D = make_grid(tl, SMOOTH)
S = timeline_summary(tl, RUN_KW)
Dv = D.dropna(subset=NEEDED)                                      # smoothed minutes without gaps
if len(Dv) < 200: st.error("Not enough rows: the file needs at least a few hours of one-minute readings."); st.stop()
real_clock = info["clock"] == "file"
loaded_tl = tl.KW > RUN_KW
PF_AVG = float(tl.PF[loaded_tl].mean()) if loaded_tl.any() else float(tl.PF.mean())
HEAT_MAX = max(50.0, float(np.ceil(np.nanpercentile(S["heat"].to_numpy(float), 99) / 50) * 50))
vi = (tl[VOLT].to_numpy() * tl[CURR].to_numpy()).sum(axis=1)
KW_OVER_VI = float(np.median(tl.KW.to_numpy()[loaded_tl.to_numpy() & (vi > 0)] * 1000 / vi[loaded_tl.to_numpy() & (vi > 0)])) if (loaded_tl.to_numpy() & (vi > 0)).any() else float("nan")
idle_days = (S["last_seen"] - S["last_run"]).total_seconds() / 86400
a_now = analyse(Dv.iloc[-1][KWCOLS].to_numpy())
status = (f'<span class="pill run"><span class="dot"></span>Running now</span>' if S["now_running"] else
          f'<span class="pill idle"><span class="dot"></span>Idle · last ran {S["last_run"]:%d %b %H:%M}</span>')
clock_chip = '<span class="chip">clock assumed</span>' if not real_clock else ""
bar.markdown(f"""<div class="appbar"><div class="logo">⚡</div><div><div class="t">GridSense</div><div class="s">Load forecasting · phase imbalance</div></div><div class="sp"></div>
{status}<span class="chip">{data_name}</span><span class="chip">{tl.Timestamps.min():%d %b} → {tl.Timestamps.max():%d %b %Y}</span><span class="chip">{len(tl):,} readings</span>{clock_chip}</div>""", unsafe_allow_html=True)

tab_overview, tab_pred, tab_imb, tab_fc = st.tabs(["Overview", "Load prediction", "Phase imbalance", "Phase forecast"])

# ================================================================== OVERVIEW
with tab_overview:
    daily = S["daily"]; total_h = (S["run_min"] + S["idle_min"]) / 60
    k = st.columns(5)
    with k[0]: kpi("Energy used", f"{S['energy'] / 1000:,.1f} MWh", f"{len(daily)} days of data", "", "🔋", daily.kwh.values, LOAD)
    with k[1]: kpi("Machine uptime", f"{S['util']:.1f} %", f"{S['run_min'] / 60:,.0f} h on · {S['idle_min'] / 60:,.0f} h idle", "info", "⏱️", daily.run_h.values, FORECAST)
    with k[2]: kpi("Peak load", f"{S['peak']:,.0f} kW", "highest 1-minute reading", "", "📈")
    with k[3]: kpi("Average load · running", f"{S['avg_run']:,.0f} kW", "only while the machine is on", "", "⚙️", daily.avg_run.values, CYAN)
    _st = a_now.Status.iloc[0]
    with k[4]: kpi("Phase imbalance now", "idle" if not a_now.Loaded.iloc[0] else f"{a_now.Imbalance_pct.iloc[0]:.1f} %", "machine idle" if not a_now.Loaded.iloc[0] else f"{a_now.Worst_phase.iloc[0]} {a_now.Direction.iloc[0]} · limit {ALARM_PCT:g} %",
                   "good" if _st == "OK" else "bad" if _st == "SEVERE" else "warn" if _st == "ALARM" else "info", "⚖️")
    st.write("")
    if KW_OVER_VI > 0.98 and PF_AVG < 0.97: note(f"<b>Data check:</b> the phase 'kW' columns equal V × I (kW / V·I = {KW_OVER_VI:.2f}) while the meter power factor is {PF_AVG:.2f}, so they behave like kVA. Energy and kW values here may be about {(1 / PF_AVG - 1) * 100:.0f} % too high; phase imbalance % is not affected.", "amber")
    with panel("Load timeline", "10-minute averages · gaps are missing data", key="timeline"):
        signal = st.selectbox("Signal", ["Total load (kW)", "Phase load R / Y / B (kW)", "Phase imbalance (%)", "Phase current (A)", "Voltage (V)"], label_visibility="collapsed")
        cols = {"Total load (kW)": ["KW"], "Phase load R / Y / B (kW)": PHASES, "Phase imbalance (%)": PHASES, "Phase current (A)": CURR, "Voltage (V)": VOLT}[signal]
        ten = tl.set_index("Timestamps")[cols].resample("10min").mean(); f = go.Figure()
        if signal == "Phase imbalance (%)":
            r10 = analyse(ten.to_numpy()); y10 = r10.Imbalance_pct.where(r10.Loaded).to_numpy()
            f.add_scatter(x=ten.index, y=y10, name="No data (gap)", mode="lines", line=dict(color=NODATA, width=1.6, dash="dot"), connectgaps=True, hoverinfo="skip")
            f.add_scatter(x=ten.index, y=y10, name="Imbalance %", line=dict(color=ACCENT, width=1.4), fill="tozeroy", fillcolor=tint(ACCENT, .14), connectgaps=False)
            f.add_hline(y=ALARM_PCT, line_dash="dot", line_color=WARN, annotation_text=f"alarm {ALARM_PCT:g} %", annotation_position="top left")
            f.add_hline(y=SEVERE_PCT, line_dash="dot", line_color=BAD, annotation_text=f"severe {SEVERE_PCT:g} %", annotation_position="top left")
        else:
            palette = [LOAD] if cols == ["KW"] else [PHASE_COL["RKW"], PHASE_COL["YKW"], PHASE_COL["BKW"]]
            for i_, (col_name, color) in enumerate(zip(cols, palette)):
                f.add_scatter(x=ten.index, y=ten[col_name], name="No data (gap)", mode="lines", line=dict(color=NODATA, width=1.6, dash="dot"), connectgaps=True, legendgroup="nodata", showlegend=i_ == 0, hoverinfo="skip")
                f.add_scatter(x=ten.index, y=ten[col_name], name=col_name, line=dict(color=color, width=1.4), fill="tozeroy" if len(cols) == 1 else None, fillcolor=tint(ACCENT, .14), connectgaps=False)
        f.update_xaxes(rangeslider_visible=True, rangeslider_thickness=.07); show(style(f, 400, legend=True, top=40))
    a, b = st.columns([2.1, 1])
    with a, panel("Activity calendar", "average kW per hour of day · grey = idle, red = no data", key="heat"):
        heat = S["heat"]; xs, ys = [f"{h:02d}:00" for h in heat.columns], [f"{d:%d %b}" for d in heat.index]
        f = go.Figure(go.Heatmap(z=np.where(np.isnan(heat.values), 1.0, np.nan), x=xs, y=ys, xgap=2, ygap=2, showscale=False, hoverinfo="skip", colorscale=[[0, tint(NODATA, .45)], [1, tint(NODATA, .45)]], zmin=0, zmax=1))
        f.add_trace(go.Heatmap(z=heat.values, x=xs, y=ys, zmin=0, zmax=HEAT_MAX, xgap=2, ygap=2,
                                                   colorscale=[[0, GRID], [.02, tint(ACCENT, .2)], [.5, ACCENT], [1, ACCENT_DARK]], colorbar=dict(title="kW", thickness=10, len=.6),
                                                   hoverongaps=False, hovertemplate="%{y} · %{x}<br><b>%{z:.0f} kW</b><extra></extra>"))
        f.update_yaxes(autorange="reversed", dtick=3); f.update_layout(hovermode="closest"); show(style(f, 520, False, top=20))
    with b, panel("Where the time went", "share of all minutes", key="donut"):
        f = go.Figure(go.Pie(labels=["Running", "Idle", "No data"], values=[S["run_min"], S["idle_min"], S["nodata_min"]], hole=.68, sort=False, marker=dict(colors=[GOOD, IDLE, NODATA], line=dict(color="#fff", width=2)),
                             textinfo="percent", hovertemplate="%{label}<br>%{value:,} min<extra></extra>"))
        f.update_layout(hovermode="closest", annotations=[dict(text=f"<b>{S['util']:.1f}%</b><br><span style='font-size:11px;color:{MUTED}'>uptime</span>", showarrow=False, font=dict(size=22))])
        show(style(f, 300, True, top=20).update_layout(legend=dict(orientation="h", y=-0.05, x=0)))
        st.markdown(f'<div class="note amber" style="margin-top:.2rem">"Idle" = the three-phase load is below {3 * MIN_KW:g} kW. Prediction and balance use every recorded minute.</div>', unsafe_allow_html=True)
    with panel("Running hours per day", "hover for energy and average load", key="daily"):
        f = go.Figure(go.Bar(x=daily.index, y=daily.run_h, marker_color=[LOAD if v > 0 else AXIS for v in daily.run_h], customdata=np.c_[daily.kwh, daily.avg_run],
                             hovertemplate="%{x|%d %b}<br><b>%{y:.1f} h running</b><br>%{customdata[0]:,.0f} kWh · avg %{customdata[1]:.0f} kW<extra></extra>"))
        missing = pd.date_range(daily.index.min(), daily.index.max()).difference(pd.DatetimeIndex(daily.index))
        if len(missing): f.add_bar(x=missing, y=[max(float(daily.run_h.max()), 1.0)] * len(missing), marker_color=NODATA, name="No data", hovertemplate="%{x|%d %b}<br><b>no data this day</b><extra></extra>")
        f.update_layout(hovermode="closest", yaxis_title="hours", showlegend=False); show(style(f, 300, False, top=20))

# ================================================================== LOAD PREDICTION (ARIMA on the rolling total kW)
with tab_pred:
    with st.expander("ARIMA settings", expanded=False):
        c = st.columns(6)
        p, d_, q = c[0].number_input("p (AR)", 0, 6, 2), c[1].number_input("d (diff)", 0, 2, 0), c[2].number_input("q (MA)", 0, 6, 2)
        win = c[3].number_input("Rolling window (min)", 1, 120, 30); split_pct = c[4].slider("Train split %", 50, 99, 95); block = c[5].number_input("Backtest block (min)", 5, 240, 60, 5)
    R = predict_kw(tl[["Timestamps", "KW"]], (p, d_, q), split_pct, int(block), int(win)); mt, mr = R["m_test"], R["m_train"]
    NX = R["future"]; last_time = NX["last_time"]; default_until = last_time + pd.Timedelta(minutes=30)
    if st.session_state.get("anchor_pred") != last_time:      # data changed -> reset the end time
        st.session_state.update(anchor_pred=last_time, until_date_pred=default_until.date(), until_time_pred=default_until.time(), len_pred="30 minutes")

    def length_changed_pred():                      # picking a length fills in the end date + time
        if st.session_state.len_pred != CUSTOM:
            end = last_time + pd.Timedelta(minutes=LENGTHS[st.session_state.len_pred]); st.session_state.until_date_pred, st.session_state.until_time_pred = end.date(), end.time()

    def until_changed_pred(): st.session_state.len_pred = CUSTOM   # editing the date/time switches the length to Custom

    # ---- future forecast with user-chosen end time, number of points and history
    with panel("Forecast ahead", f"ARIMA{(p, d_, q)} · parameters fixed after training · starts after the last data point · {last_time:%d %b %Y %H:%M}", key="next"):
        c = st.columns(5)
        c[0].selectbox("Forecast length", list(LENGTHS) + [CUSTOM], key="len_pred", on_change=length_changed_pred)
        step_choice = c[1].selectbox("Show a point every", list(STEPS), index=1, key="step_pred")
        hist_choice = c[2].selectbox("History shown", list(HISTS), index=1, key="hist_pred")
        until_date = c[3].date_input("Forecast until · date", min_value=last_time.date(), max_value=(last_time + pd.Timedelta(days=7)).date(), key="until_date_pred", on_change=until_changed_pred)
        until_time = c[4].time_input("Forecast until · time", step=300, key="until_time_pred", on_change=until_changed_pred)
        horizon_raw = int((pd.Timestamp.combine(until_date, until_time) - last_time).total_seconds() // 60)
        horizon = int(min(max(horizon_raw, 1), MAX_HORIZON)); step = min(STEPS[step_choice], horizon); step = max(step, int(np.ceil(horizon / 400)))   # at most ~400 points on screen
        st.caption(f"Forecasting {horizon} min ahead, until {last_time + pd.Timedelta(minutes=horizon):%a %d %b %H:%M} · one point every {step} min" + (" (capped at 7 days)" if horizon_raw > MAX_HORIZON else " (end time was before the last data point, so showing 1 min)" if horizon_raw < 1 else ""))
        idx = np.unique(np.r_[np.arange(step - 1, horizon, step), horizon - 1])           # which forecast minutes to display
        t_f, m_f, lo_f, hi_f = NX["times"][idx], NX["mean"][idx], NX["lower"][idx], NX["upper"][idx]
        end_value, lo, hi = NX["mean"][horizon - 1], NX["lower"][horizon - 1], NX["upper"][horizon - 1]
        change = f"{(end_value - NX['last_value']) / NX['last_value'] * 100:.1f} %" if NX["last_value"] >= 1 else f"{abs(end_value - NX['last_value']):.1f} kW"
        up_down = "▲" if end_value >= NX["last_value"] else "▼"
        left, right = st.columns([1, 2.4])
        with left:
            st.markdown(f"""<div class="hero-f"><div class="l">Expected load · {NX['times'][horizon - 1]:%d %b %H:%M}</div><div class="big">{end_value:,.1f}<small> kW</small></div>
<div class="r">95 % range <b>{lo:,.1f} – {hi:,.1f} kW</b></div><span class="badge flat">{up_down} {change.lstrip('-')} vs last {int(win)}-min average</span>
<div class="mini"><div><b>{NX['mean'][:horizon].mean():,.1f}</b><span>average over period kW</span></div><div><b>{len(idx)}</b><span>points shown</span></div><div><b>{horizon / 60:.1f} h</b><span>ahead</span></div></div></div>""", unsafe_allow_html=True)
        with right:
            window_start = last_time - pd.Timedelta(minutes=HISTS[hist_choice])
            hist = NX["history"][NX["history"].index > window_start]
            f = go.Figure()
            f.add_shape(type="rect", xref="x", yref="paper", x0=last_time, x1=t_f[-1], y0=0, y1=1, fillcolor=tint(ACCENT, .06), line_width=0)
            anchor_value, tt = NX["last_value"], [last_time] + list(t_f)          # start the forecast exactly at the last actual point (no gap)
            f.add_scatter(x=hist.index, y=hist.values, name=f"Actual rolling mean ({int(win)} min)", mode="lines", line=dict(color=INK, width=2.2))
            f.add_scatter(x=tt + tt[::-1], y=[anchor_value] + list(hi_f) + ([anchor_value] + list(lo_f))[::-1], fill="toself", fillcolor=tint(ACCENT, .16), mode="lines", line=dict(width=0), name="95 % range", hoverinfo="skip")
            f.add_scatter(x=tt, y=[anchor_value] + list(m_f), name=f"Forecast ({len(idx)} points)", mode="lines", line=dict(color=FORECAST, width=3))
            if len(idx) <= 120: f.add_scatter(x=t_f, y=m_f, mode="markers", marker=dict(size=6, color="#fff", line=dict(color=FORECAST, width=2)), showlegend=False, hoverinfo="skip")
            f.add_shape(type="line", xref="x", yref="paper", x0=last_time, x1=last_time, y0=0, y1=1, line=dict(color=MUTED, width=1.2, dash="dot"))
            f.add_annotation(x=last_time, yref="paper", y=1.0, text="last data point", showarrow=False, xanchor="right", yanchor="bottom", font=dict(size=11, color=MUTED))
            f.update_yaxes(title="kW"); show(style(f, 340, True, top=70))
        with st.expander(f"Forecast table · {len(idx)} points"):
            table(pd.DataFrame({"Time": [t.strftime("%d %b %H:%M") for t in t_f], "Forecast kW": m_f, "Lower 95 %": lo_f, "Upper 95 %": hi_f}).head(300), 2)
        if horizon > 120: note("ARIMA is a short-term model: beyond roughly 1–2 hours the forecast flattens towards the long-run average and the range keeps widening, so treat far-ahead points as a guide, not a prediction.")
    # ---- how much power is needed in the chosen forecast window (energy, first / last point, averaged blocks, advice)
    fv = NX["mean"][:horizon]; t_first, t_last = NX["times"][0], NX["times"][horizon - 1]
    energy_f, avg_f, peak_f = float(fv.sum() / 60), float(fv.mean()), float(fv.max())
    before = tl[(tl.Timestamps > last_time - pd.Timedelta(minutes=horizon)) & (tl.Timestamps <= last_time)]
    energy_before = float(before.KW.fillna(0).sum() / 60)
    with panel("Power needed in the selected period", f"{t_first:%d %b %H:%M} → {t_last:%d %b %H:%M} · {span_text(horizon)} · energy = average kW × hours", key="pw_pred"):
        pk = st.columns(5)
        with pk[0]: kpi("Energy needed · forecast", f"{energy_f:,.1f} kWh", f"average {avg_f:,.1f} kW over {span_text(horizon)}", "info", "🔌")
        diff_e = energy_f - energy_before
        with pk[1]: kpi("Energy used · previous period", f"{energy_before:,.1f} kWh", (f"forecast is {diff_e:+,.1f} kWh ({diff_e / energy_before * 100:+.0f} %) vs this" if energy_before >= 1 else "the machine was idle in that period") + f" · last {span_text(horizon)} of actual data", "", "📊")
        with pk[2]: kpi("Power at first point", f"{fv[0]:,.1f} kW", f"{t_first:%d %b %H:%M}", "", "▶️")
        with pk[3]: kpi("Power at last point", f"{fv[-1]:,.1f} kW", f"{t_last:%d %b %H:%M} · {fv[-1] - fv[0]:+,.1f} kW{pct_text(pct_change(fv[0], fv[-1]))} vs first", "info", "⏹️")
        with pk[4]: kpi("Average · peak", f"{avg_f:,.1f} kW", f"peak {peak_f:,.1f} kW at {NX['times'][int(np.argmax(fv))]:%H:%M}", "", "📈")
        st.write("")
        valid_avg = [name for name, m_ in AVG_BLOCKS.items() if m_ <= horizon] or [list(AVG_BLOCKS)[0]]
        sel_col, _ = st.columns([1.2, 3])
        avg_choice = sel_col.selectbox("Show average power every", valid_avg, index=min(1, len(valid_avg) - 1), key="avg_pred", help="Splits the forecast window into blocks of this length and averages the power in each block.")
        blk = block_averages(fv, last_time, min(AVG_BLOCKS[avg_choice], horizon)); labels = [block_label(r) for r in blk.itertuples()]
        two = len(blk) >= 2
        kind = trend_kind(float(blk["Avg kW"].iloc[0]), float(blk["Avg kW"].iloc[-1])) if two else trend_kind(float(fv[0]), float(fv[-1]))
        first_v, last_v = (float(blk["Avg kW"].iloc[0]), float(blk["Avg kW"].iloc[-1])) if two else (float(fv[0]), float(fv[-1]))
        slope = float(np.polyfit(np.arange(len(blk)) * (AVG_BLOCKS[avg_choice] / 60), blk["Avg kW"], 1)[0]) if two else float("nan")
        cl, cr = st.columns([2.2, 1.1])
        with cl:
            f = go.Figure(go.Bar(x=labels, y=blk["Avg kW"], marker_color=ACCENT, text=[("" if pd.isna(x) else f"{x:+.1f} %") for x in blk["Change %"]], textposition="outside", cliponaxis=False,
                                 customdata=np.c_[blk["Change kW"].fillna(0), blk["Minutes"]], hovertemplate="%{x}<br><b>%{y:,.1f} kW average</b><br>change vs previous block %{customdata[0]:+,.1f} kW<br>%{customdata[1]} min<extra></extra>"))
            f.update_layout(hovermode="closest", yaxis_title=f"average kW per {avg_choice}", title=f"Average power every {avg_choice}  ·  label = change vs previous block"); f.update_xaxes(type="category")
            show(style(f, 300, False, top=60))
            st.caption((f"Average growth: {blk['Change kW'].mean():+,.1f} kW per block · trend {slope:+,.1f} kW per hour" if two else "Only one block fits in this window - pick a shorter interval or a longer forecast.")
                       + (f" · last block is partial ({int(blk['Minutes'].iloc[-1])} min)" if blk["Minutes"].iloc[-1] < blk["Minutes"].iloc[0] else ""))
        with cr:
            tone_a, head_a, acts_a = power_advice(kind, first_v, last_v, peak_f, horizon > 120, dict(avg=avg_f, slope=slope, pf=PF_AVG, demand=DEMAND, peak_at=NX["times"][int(np.argmax(fv))]))
            note(advice_html(head_a, acts_a), tone_a)
        with st.expander(f"Average power table · every {avg_choice}"):
            table(pd.DataFrame({"Period": labels, "Avg kW": blk["Avg kW"], "Change kW": blk["Change kW"], "Change %": blk["Change %"]}), 2)

    if R["idle_share"] > .5:
        note(f"<b>Why the forecast is near zero:</b> as in your notebook, the model is trained on the full timeline, which is <b>{R['idle_share'] * 100:.0f} % idle (0 kW)</b>. The data ends in an idle period "
             f"({last_time:%d %b}), so the forecast starts near 0 kW and only drifts towards the long-run average of about {R['level']:.0f} kW.", "amber")

    st.markdown('<div class="ph" style="margin-top:.6rem"><b>Model check on past data</b><span>how well the same ARIMA predicted data it had not seen</span></div>', unsafe_allow_html=True)
    k = st.columns(4)
    with k[0]: kpi("Test RMSE", f"{mt['RMSE']:.2f} kW", f"{len(R['test']):,} forecast points", "info", "🎯")
    with k[1]: kpi("Test MAE", f"{mt['MAE']:.2f} kW", f"{mt['MAE'] / max(R['test'].Actual.mean(), 1e-9) * 100:.1f} % of mean test load" if R["test"].Actual.mean() >= 1 else "test period has ~0 kW load", "", "📏")
    with k[2]: kpi("Test R²", f"{mt['R2']:.3f}", "test period is constant → R² not meaningful" if R["test_constant"] else "1 = perfect · 0 = no better than the average", "info" if R["test_constant"] else "good" if mt["R2"] > .5 else "warn", "✅")
    with k[3]: kpi("Train fit R²", f"{mr['R2']:.3f}", f"RMSE {mr['RMSE']:.2f} kW · {len(R['train']):,} points", "", "🧪")
    st.write("")
    s1, s2, s3 = st.tabs(["Test forecast", "Training fit", "Residuals"])

    def line_pair(frame, a, b, title, name_b):
        shown = thin(frame); t, ya, yb = gaps(shown.Timestamp, shown[a], shown[b], step=thin_step(frame)); f = go.Figure()
        f.add_scatter(x=t, y=ya, name=f"Actual rolling mean ({int(win)} min)", mode="lines", line=dict(color=INK, width=1.8), connectgaps=False)
        f.add_scatter(x=t, y=yb, name=name_b, mode="lines", line=dict(color=FORECAST, width=2, dash="dash"), connectgaps=False)
        f.update_layout(title=title, yaxis_title="Rolling KW"); f.update_xaxes(rangeslider_visible=True); return f

    with s1:
        with panel("Test: actual vs forecast", f"each {int(block)}-minute block forecast before its actuals are shown · {R['t2']:%d %b %H:%M} → {R['t3']:%d %b %H:%M}", key="test"):
            show(style(line_pair(R["test"], "Actual", "Forecast", "", "ARIMA forecast"), 440, top=60))
        table(R["test"].head(10).assign(Timestamp=lambda x: x.Timestamp.dt.strftime("%Y-%m-%d %H:%M"))[["Timestamp", "Actual", "Forecast", "Residual"]], 3)
    with s2:
        with panel("Training: actual vs fitted", "how closely ARIMA reproduces the data it was trained on" + (f" · showing every {int(np.ceil(len(R['train']) / 4000))}th point" if len(R["train"]) > 4000 else ""), key="train"):
            show(style(line_pair(R["train"], "Actual", "Fitted", "", "ARIMA fitted"), 440, top=60))
    with s3:
        a, b = st.columns(2)
        for col, frame, name, color in ((a, R["train"], "Training", CYAN), (b, R["test"], "Test", ACCENT_DARK)):
            shown = thin(frame); t, r = gaps(shown.Timestamp, shown.Residual, step=thin_step(frame)); f = go.Figure(go.Scatter(x=t, y=r, mode="lines", line=dict(color=color, width=1.3), connectgaps=False))
            f.add_hline(y=0, line_dash="dash", line_color=GREY); f.update_layout(title=f"{name} residuals (actual − predicted)", yaxis_title="kW")
            with col, st.container(border=True, key=f"panel_res_{name}"): show(style(f, 340, False, top=60))

# ================================================================== PHASE IMBALANCE (measured data: how big, why, what to do)
with tab_imb:
    top = st.columns([1.3, 3.4])
    mean_label = f"{SMOOTH}-min mean"
    basis = top[0].selectbox("Measure on", ["Smoothed (rolling mean)", "Real readings"], key="imb_basis", help=f"Smoothed = rolling mean of the past {SMOOTH} min (set in 'Data & limits').")
    frame = (Dv if basis.startswith("Smoothed") else D_raw.dropna(subset=NEEDED))[NEEDED]
    DG = diagnose(frame, ALARM_PCT, MIN_KW, SEVERE_PCT)
    if DG["empty"]: note("There are not enough loaded minutes (above the 'ignore below' limit) to measure the imbalance. Lower the limit in 'Data & limits'.", "amber")
    else:
        mi, per = DG["mean_imb"], DG["per"]; tone_i = "good" if mi <= ALARM_PCT else "warn" if mi <= SEVERE_PCT else "bad"
        k = st.columns(5)
        with k[0]: kpi("Average imbalance", f"{mi:.1f} %", f"95 % of minutes below {DG['p95_imb']:.1f} % · limit {ALARM_PCT:g} %", tone_i, "⚖️")
        with k[1]: kpi("Above the limit", f"{DG['pct_above']:.0f} %", f"of {DG['n_loaded']:,} loaded minutes", "good" if DG["pct_above"] < 5 else "bad" if DG["pct_above"] > 50 else "warn", "🚨")
        with k[2]: kpi("Worst colour", DG["main_name"], f"worst in {per.iloc[DG['main'], 0]:.0f} % of minutes · {'HIGH' if DG['high_share'] >= 50 else 'LOW'} in {max(DG['high_share'], 100 - DG['high_share']):.0f} % of them", "info", "🎨")
        with k[3]: kpi("Voltage imbalance", f"{DG['mean_V']:.1f} %", "supply is fine (usual limit about 2 %)" if DG["mean_V"] <= 2 else "supply voltage is unbalanced", "good" if DG["mean_V"] <= 1 else "warn" if DG["mean_V"] <= 2 else "bad", "🔌")
        with k[4]: kpi(f"Extra current on {DG['main_name']}", f"{DG['extra_A']:+.0f} A", f"{DG['extra_kW']:+.1f} kW more than the other two phases", "", "➕")
        st.write("")
        a, b = st.columns([2.2, 1])
        with a, panel("Imbalance over time", f"{mean_label if basis.startswith('Smoothed') else basis} · 30-minute averages · lines = alarm and severe limits", key="imb_time"):
            A = DG["A"]; ser = A.Imbalance_pct.where(A.Loaded).resample("30min").mean()
            f = go.Figure(); f.add_scatter(x=ser.index, y=ser.values, name="Imbalance %", line=dict(color=ACCENT, width=1.6), fill="tozeroy", fillcolor=tint(ACCENT, .12), connectgaps=False)
            f.add_hline(y=ALARM_PCT, line_dash="dot", line_color=WARN, annotation_text=f"alarm {ALARM_PCT:g} %", annotation_position="top left")
            f.add_hline(y=SEVERE_PCT, line_dash="dot", line_color=BAD, annotation_text=f"severe {SEVERE_PCT:g} %", annotation_position="top left")
            f.update_yaxes(title="imbalance %"); f.update_xaxes(rangeslider_visible=True, rangeslider_thickness=.07); show(style(f, 380, False, top=30))
        with b, panel("Who carries the load", "average over the loaded minutes", key="imb_phase"):
            st.markdown(phase_bars(list(DG["Km"]), float(DG["Km"].max()) * 1.15), unsafe_allow_html=True)
            table(per[["Avg current A", "95 % of minutes below A", "Avg voltage V", "PF (kW / V·I)"]].rename_axis("Phase").reset_index(), 2)
        c1, c2, c3 = st.columns(3)
        with c1, panel("What explains it", "share of the worst phase's deviation", key="imb_why"):
            sh = DG["share"]; f = go.Figure(go.Bar(x=sh.values, y=sh.index, orientation="h", marker_color=[ACCENT, BLUE, WARN], text=[f"{v:.0f} %" for v in sh.values], textposition="outside", cliponaxis=False,
                                                    hovertemplate="%{y}: %{x:.0f} %<extra></extra>"))
            f.update_layout(hovermode="closest", xaxis_range=[0, 115]); show(style(f, 250, False, top=20))
            st.caption(f"kW imbalance follows the current imbalance (correlation {DG['corr_I']:.2f}) far more than the voltage imbalance ({DG['corr_V']:.2f}).")
        with c2, panel("By load level", "imbalance % (bars) and extra amps on the worst phase (line)", key="imb_level"):
            lv = DG["levels"]; f = make_subplots(specs=[[{"secondary_y": True}]])
            f.add_bar(x=lv.index, y=lv["imb"], name="Imbalance %", marker_color=ACCENT, secondary_y=False, hovertemplate="%{x}<br>%{y:.1f} %<extra></extra>")
            f.add_scatter(x=lv.index, y=lv["extra_A"], name=f"{DG['main_name']} extra A", mode="lines+markers", line=dict(color=BAD, width=2.4), secondary_y=True, hovertemplate="%{x}<br>%{y:.0f} A<extra></extra>")
            f.update_layout(hovermode="closest"); f.update_yaxes(title="%", secondary_y=False); f.update_yaxes(title="A", secondary_y=True, showgrid=False); f.update_xaxes(type="category")
            show(style(f, 250, True, top=50))
        with c3, panel("How often", "share of loaded minutes", key="imb_status"):
            stt = DG["status"]; f = go.Figure(go.Pie(labels=list(stt.index), values=list(stt.values), hole=.66, sort=False, marker=dict(colors=[GOOD, WARN, BAD], line=dict(color="#fff", width=2)), textinfo="percent",
                                                      hovertemplate="%{label}<br>%{value:,.0f} min<extra></extra>"))
            f.update_layout(hovermode="closest", annotations=[dict(text=f"<b>{DG['pct_above']:.0f}%</b><br><span style='font-size:11px;color:{MUTED}'>above limit</span>", showarrow=False, font=dict(size=20))])
            show(style(f, 250, True, top=20).update_layout(legend=dict(orientation="h", y=-0.08, x=0)))
        if real_clock and len(DG["by_hour"]) >= 6:
            with panel("By hour of day", "average imbalance % in each hour · shows when the plant is most unbalanced", key="imb_hour"):
                bh = DG["by_hour"]; f = go.Figure(go.Bar(x=[f"{h:02d}:00" for h in bh.index], y=bh.values, marker_color=ACCENT, hovertemplate="%{x}<br>%{y:.1f} %<extra></extra>"))
                f.add_hline(y=ALARM_PCT, line_dash="dot", line_color=WARN); f.update_layout(hovermode="closest", yaxis_title="imbalance %"); show(style(f, 260, False, top=20))
        with panel("What to do", "written from the measured numbers above", key="imb_act"):
            tone_a, head_a, acts_a = imbalance_actions(DG, ALARM_PCT, SEVERE_PCT, real_clock, RATING)
            l, r = st.columns([1.7, 1])
            with l: note(advice_html(head_a, acts_a), tone_a)
            with r:
                st.markdown('<div class="ph"><b>Move plan</b><span>average currents</span></div>', unsafe_allow_html=True)
                if len(DG["plan"]): table(DG["plan"], 1)
                else: st.caption("The three phase currents are already within 1 A of each other.")
                if KW_OVER_VI > 0.98 and DG["pf_meter"] < 0.97:
                    note(f"<b>Data check:</b> the phase 'kW' columns equal V × I (kW / V·I = {KW_OVER_VI:.2f}) while the meter power factor is {DG['pf_meter']:.2f}. They behave like <b>kVA</b>, not true kW - real kW would be about {DG['pf_meter'] * 100:.0f} % of them. "
                         "Imbalance % is not affected, but energy (kWh) and kW limits are over-stated by about " + f"{(1 / DG['pf_meter'] - 1) * 100:.0f} %.", "amber")
    with st.expander("Check your own readings"):
        c = st.columns(4)
        rr, yy, bb = c[0].number_input("Red kW", 0.0, 5000.0, 170.0, 1.0, key="chk_r"), c[1].number_input("Yellow kW", 0.0, 5000.0, 169.0, 1.0, key="chk_y"), c[2].number_input("Blue kW", 0.0, 5000.0, 160.0, 1.0, key="chk_b")
        res = analyse([[rr, yy, bb]]).iloc[0]
        with c[3]:
            if not res.Loaded: kpi("Result", "No load", f"average below {MIN_KW:g} kW per phase", "", "ℹ️")
            else: kpi("Imbalance", f"{res.Imbalance_pct:.2f} %", f"{res.Worst_phase} is {abs(res.Worst_dev):.1f} % {'above' if res.Worst_dev > 0 else 'below'} the average · {res.Status}", "good" if res.Status == "OK" else "bad" if res.Status == "SEVERE" else "warn", "🧮")

# ================================================================== PHASE FORECAST (SARIMAX per phase on the last 24 hours)
with tab_fc:
    mean_label = f"{SMOOTH}-min mean"
    FC_LENGTHS = {"15 minutes": 15, "30 minutes": 30, "1 hour": 60, "2 hours": 120, "3 hours": 180, "6 hours": 360}
    c = st.columns([1.1, 1.6, 1, 1, .9, .9])
    train_on = c[0].selectbox("Learn from", ["Smoothed (rolling mean)", "Real values"], key="fc_train", help=f"The series SARIMAX is trained on and forecasts. Smoothed = rolling mean of the past {SMOOTH} min.")
    cfg_name = c[1].selectbox("SARIMAX setting", list(SX_CONFIGS), key="fc_cfg")
    len_name = c[2].selectbox("Forecast length", list(FC_LENGTHS), index=2, key="fc_len")
    hist_name = c[3].selectbox("History shown", ["3 hours", "6 hours", "12 hours", "24 hours"], index=1, key="fc_hist")
    n_or = int(c[4].number_input("Backtest moments", 4, 40, 12, 2, key="fc_nor", help="More moments = firmer answer but slower (3 SARIMAX fits per moment)."))
    c[5].markdown("<div style='height:1.65rem'></div>", unsafe_allow_html=True); run_bt = c[5].button("Run backtest", type="primary", key="fc_run")
    SRC = (D_raw if train_on == "Real values" else D)[KWCOLS].to_numpy(float)
    nw = newest_window(SRC)
    if nw is None: note("The newest 24 hours have too many gaps (or the file is shorter than 24 hours), so SARIMAX cannot learn from them.", "amber")
    else:
        window, pn, filled = nw; horizon = FC_LENGTHS[len_name]; K = horizon // SX_STEP
        t0 = D.index[pn]; t_fut = pd.date_range(t0 + pd.Timedelta(minutes=SX_STEP), periods=K, freq=f"{SX_STEP}min")
        F = sx_forecast(window, K, cfg_name); P = F["mean"]; rep = analyse(P)
        DGm = diagnose(Dv[NEEDED], ALARM_PCT, MIN_KW, SEVERE_PCT); chronic = None if DGm["empty"] else DGm["pct_above"]
        bt_key = (train_on, cfg_name, n_or, ALARM_PCT, SEVERE_PCT, MIN_KW, SMOOTH, str(t0))
        if run_bt:
            pbar = st.progress(0.0, text="Testing SARIMAX at past moments - about 1-3 minutes…")
            out_bt = sx_backtest(D_raw, D, "Real values" if train_on == "Real values" else "mean", cfg_name, n_or, (15, 30, 60), progress=lambda x: pbar.progress(float(x), text="Testing SARIMAX at past moments…"))
            pbar.empty(); st.session_state["bt"] = dict(key=bt_key, res=out_bt)
        cur = st.session_state.get("bt"); res_bt = cur["res"] if cur and cur["key"] == bt_key else None
        tone_v, verdict = bt_verdict(res_bt) if res_bt is not None else ("", None)
        if not all(F["ok"]): note(f"A SARIMAX fit failed for {', '.join(n for n, ok in zip(NAMES, F['ok']) if not ok)} - that phase falls back to 'same as now'. Try another setting.", "amber")
        if filled: note(f"{filled} gap point(s) in the last 24 hours were filled with straight lines.", "amber")
        imb = rep.Imbalance_pct.to_numpy(); n_al, n_sev = int(rep.Alarm.sum()), int((rep.Status == "SEVERE").sum()); tot = P.sum(axis=1)
        k = st.columns(4)
        with k[0]: kpi(f"Imbalance at {t_fut[-1]:%H:%M}", "–" if np.isnan(imb[-1]) else f"{imb[-1]:.1f} %", f"limit {ALARM_PCT:g} % · {rep.Status.iloc[-1].lower()}", "good" if rep.Status.iloc[-1] == "OK" else "bad" if rep.Status.iloc[-1] == "SEVERE" else "warn" if rep.Status.iloc[-1] == "ALARM" else "", "⚖️", trend=np.nan_to_num(imb), trend_color=ACCENT)
        with k[1]: kpi("Alarm points", f"{n_al}", f"of {K} forecast points · {n_sev} severe", "good" if n_al == 0 else "bad", "🚨")
        with k[2]: kpi(f"Total power at {t_fut[-1]:%H:%M}", f"{tot[-1]:,.1f} kW", f"{tot[-1] - window[-1].sum():+,.1f} kW vs now ({window[-1].sum():,.1f} kW)", "info", "🔌", trend=tot)
        with k[3]: kpi("Worst colour at the end", "–" if not rep.Loaded.iloc[-1] else rep.Worst_phase.iloc[-1], "" if not rep.Loaded.iloc[-1] else f"{rep.Direction.iloc[-1]} · {abs(rep.Worst_dev.iloc[-1]):.1f} % from the average", "info", "🎨")
        st.write("")
        with panel("Phase forecast & imbalance alarms", f"SARIMAX learns from the last 24 h ({SX_WSTEPS} points, one every {SX_STEP} min) · {cfg_name} · learned from {mean_label if train_on != "Real values" else "real values"} · forecast starts after {t0:%d %b %Y %H:%M}", key="fc_main"):
            hmin = {"3 hours": 180, "6 hours": 360, "12 hours": 720, "24 hours": 1440}[hist_name]; w0 = t0 - pd.Timedelta(minutes=hmin)
            hm, hr = D.loc[w0:t0, KWCOLS], D_raw.loc[w0:t0, KWCOLS]
            f = make_subplots(rows=4, cols=1, shared_xaxes=True, vertical_spacing=.06, subplot_titles=[f"{n} phase (kW)" for n in NAMES] + ["Phase imbalance %"])
            xt = [t0] + list(t_fut)
            for j, ph in enumerate(PHASES, 1):
                f.add_scatter(x=hr.index, y=hr[ph], name="Real values", legendgroup="r", showlegend=j == 1, line=dict(color=IDLE, width=1), row=j, col=1)
                f.add_scatter(x=hm.index, y=hm[ph], name=mean_label, legendgroup="m", showlegend=j == 1, line=dict(color=INK, width=2), row=j, col=1)
                a0 = float(window[-1, j - 1]); lo_, hi_ = F["lo"][:, j - 1], F["hi"][:, j - 1]
                if np.isfinite(lo_).all(): f.add_scatter(x=xt + xt[::-1], y=[a0] + list(hi_) + ([a0] + list(lo_))[::-1], fill="toself", fillcolor=tint(PHASE_COL[ph], .16), mode="lines", line=dict(width=0), name="95 % range", legendgroup="b", showlegend=j == 1, hoverinfo="skip", row=j, col=1)
                f.add_scatter(x=xt, y=[a0] + list(P[:, j - 1]), name="Forecast", legendgroup="f", showlegend=j == 1, mode="lines+markers", line=dict(color=PHASE_COL[ph], width=2.6, dash="dash"), marker=dict(size=5), row=j, col=1)
            ih_m = analyse(hm[KWCOLS].to_numpy()).Imbalance_pct.to_numpy(); ih_r = analyse(hr[KWCOLS].to_numpy()).Imbalance_pct.to_numpy()
            f.add_scatter(x=hr.index, y=ih_r, name="Real values", legendgroup="r", showlegend=False, line=dict(color=IDLE, width=1), row=4, col=1)
            f.add_scatter(x=hm.index, y=ih_m, name=mean_label, legendgroup="m", showlegend=False, line=dict(color=INK, width=2), row=4, col=1)
            f.add_scatter(x=t_fut, y=imb, name="Forecast", legendgroup="f", showlegend=False, mode="lines+markers", line=dict(color=ACCENT, width=2.6, dash="dash"), marker=dict(size=5), row=4, col=1)
            for sev_, col_ in (("ALARM", WARN), ("SEVERE", BAD)):
                m_ = (rep.Status == sev_).to_numpy()
                if m_.any(): f.add_scatter(x=t_fut[m_], y=imb[m_], mode="markers", name=f"{sev_.title()} (forecast)", marker=dict(symbol="x", size=10, color=col_), row=4, col=1)
            f.add_hline(y=ALARM_PCT, line_dash="dot", line_color=WARN, row=4, col=1); f.add_hline(y=SEVERE_PCT, line_dash="dot", line_color=BAD, row=4, col=1)
            f.add_shape(type="line", xref="x", yref="paper", x0=t0, x1=t0, y0=0, y1=1, line=dict(color=MUTED, width=1.2, dash="dot"))
            style(f, 860, legend_y=1.0, top=110); f.update_annotations(font=dict(size=13, color=INK)); show(f)
            with st.expander(f"Forecast table · {K} points"):
                table(pd.DataFrame({"Time": [t.strftime("%d %b %H:%M") for t in t_fut], "Red kW": P[:, 0], "Yellow kW": P[:, 1], "Blue kW": P[:, 2], "Total kW": tot, "Imbalance %": imb, "Worst": rep.Worst_phase.to_numpy(), "Direction": rep.Direction.to_numpy(), "Status": rep.Status.to_numpy()}).head(300), 2)
            exp = pd.DataFrame({"Timestamp": t_fut, "Red_kW": P[:, 0], "Yellow_kW": P[:, 1], "Blue_kW": P[:, 2], "Total_kW": tot, "Imbalance_pct": imb, "Worst_phase": rep.Worst_phase.to_numpy(), "Direction": rep.Direction.to_numpy(), "Status": rep.Status.to_numpy()})
            d1, _ = st.columns([1, 3]); d1.download_button("⬇ Forecast + alarms (CSV)", exp.to_csv(index=False).encode(), "GridSense_phase_forecast.csv", "text/csv", on_click="ignore", width="stretch")
        with panel("What to do about the forecast", "written from the forecast numbers", key="fc_act"):
            tone_f, head_f, acts_f = forecast_advice(P, rep, t_fut, chronic, verdict, horizon > 60)
            note(advice_html(head_f, acts_f), tone_f)
        with panel("Is the forecast better than 'same as now'?", "backtest: SARIMAX learns from the 24 h before each past moment, forecasts 15 / 30 / 60 min, and is scored against what really happened", key="fc_bt"):
            if res_bt is None: st.caption("Not run yet for these settings - press 'Run backtest' above. It tests the last 20 % of the file.")
            else:
                note(verdict, {"good": "good", "bad": "bad", "amber": "amber"}[tone_v])
                table(res_bt.drop(columns="Moments"), 2); st.caption(f"{int(res_bt['Moments'].iloc[0])} past moments tested · errors in kW (phase, total) and percentage points (imbalance) · smaller is better")
        note(f"<b>Alarm rule:</b> imbalance % = largest deviation of a phase from the average of the three ÷ the average × 100 (170 / 169 / 160 kW → 3.81 %). Alarm above {ALARM_PCT:g} %, severe above {SEVERE_PCT:g} %. "
             "The 95 % range is the model's own interval; the backtest above is the fairer test of how far to trust it.")
