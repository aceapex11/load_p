"""GridSense - machine load intelligence (light mode). Rule-based advice, no AI service, no API key.
Data            : device_16_prediction_dataset.csv - one row per minute with V1-V3, I1-I3, PF, RKW, YKW, BKW
                  (an optional Timestamps / Timestamp / Timestamp_Index column is used when present; you can also upload the file in the app)
Overview        : load timeline (kW, phases, current, voltage, imbalance) and where the time went (uptime)
Load forecast   : hourly total kW, Exponential Smoothing  tested on the last 168 hours, power advice
Phase balance   : ONE page - measured imbalance, 7-day R / Y / B forecast (three_phase_imbalance_detection_forecasting notebook), imbalance alarms,
                  and a rule-based action plan: what is out of phase, why, what to do now, when to do what, what can happen, how to bring it in phase, future balance
Files next to app.py: gridsense_forecast.py (the notebooks' models) and gridsense_rules.py (the rule-based advisor)
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

import gridsense_forecast as gf
import gridsense_rules as gr

warnings.filterwarnings("ignore")
st.set_page_config(page_title="GridSense · Load intelligence", page_icon="⚡", layout="wide", initial_sidebar_state="collapsed")

HERE = Path(__file__).resolve().parent
DATA_NAME = "device_16_full_data.csv"       # put this file next to app.py
PHASES = ["RKW", "YKW", "BKW"]
PNAME = {"RKW": "Red", "YKW": "Yellow", "BKW": "Blue"}
NAMES = [PNAME[p] for p in PHASES]

# LIMITS - edit here.  (The old "Data & limits" panel is gone.)
ALARM_PCT = 3.0      # imbalance % above this = ALARM
SEVERE_PCT = 10.0    # imbalance % above this = SEVERE
MIN_KW = 5.0         # average kW per phase below this = machine off
SMOOTH = 30          # minutes of rolling mean (past minutes only)
DEMAND = 0.0         # sanctioned demand in kW, 0 = unknown
RATING = 0.0         # phase breaker / cable in A, 0 = unknown
LOAD_CHG_PCT = 5.0   # forecast energy must differ more than this % from the previous period to count as an INCREASE / DECREASE
CAPTIVE_KW = 0.0     # own (captive) generation capacity in kW, 0 = unknown
HAS_BATTERY = False  # True if a battery storage system is available
SHIFTABLE_PCT = 10.0 # % of the peak that could be moved to other hours (flexible loads)
TARIFF = 0.0         # money per kWh, 0 = do not show a cost
DEMAND_CHARGE = 0.0  # money per kW of peak demand, 0 = do not show a cost
CURRENCY = ""        # e.g. "INR"
MINR = 45            # an hour with fewer one-minute readings than this counts as missing (as in the notebooks)

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
.kpi { position:relative; overflow:hidden; background:#FFFFFF; border:1px solid #E6EAF2; border-radius:18px; padding:1rem 1.1rem .9rem; height:100%; min-height:9rem; box-shadow:0 1px 2px rgba(15,23,42,.04), 0 10px 28px -16px rgba(15,23,42,.12); }.kpi .top { display:flex; align-items:center; gap:.55rem; } .kpi .ic { width:30px; height:30px; border-radius:9px; display:grid; place-items:center; font-size:.95rem; background:#F3F5FB; }
.kpi .l { color:#64748B !important; font-size:.8rem; font-weight:500; }
.kpi .v { font-family:'JetBrains Mono',monospace; font-size:1.75rem; font-weight:600; letter-spacing:-1px; margin-top:.55rem; color:#0F172A !important; }
.kpi .s { font-size:.78rem; margin-top:.1rem; color:#64748B !important; }
.kpi.kb .v { color:#2563EB !important; } .kpi.kk .v { color:#0F172A !important; }
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


KPI_N = [0]                                                       # card counter: blue, black, blue, black ... (reset at the start of every row of cards)


def kreset():
    KPI_N[0] = 0


def kpi(label, value, sub="", tone="", icon="", trend=None, trend_color=FORECAST):
    """One card. Only the number colour alternates blue / black; icon, trend line and everything else keep their original colours."""
    blue = KPI_N[0] % 2 == 0; KPI_N[0] += 1
    st.markdown(f'<div class="kpi {"kb" if blue else "kk"} {tone}"><div class="top"><div class="ic">{icon}</div><div class="l">{label}</div></div><div class="v">{value}</div><div class="s">{sub}</div>'
                f'{spark(trend, trend_color) if trend is not None else ""}</div>', unsafe_allow_html=True)


def note(text, tone=""):
    st.markdown(f'<div class="note {tone}">{text}</div>', unsafe_allow_html=True)


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



# ------------------------------------------------------------------ phase imbalance maths (lives in gridsense_rules.py; these wrappers use the limits set on the page)
def analyse(P, thr=None, min_kw=None, sev=None):
    """imbalance % = max(|R-avg|, |Y-avg|, |B-avg|) / avg x 100.  170 / 169 / 160 kW -> 3.81 %."""
    return gr.analyse(P, ALARM_PCT if thr is None else thr, MIN_KW if min_kw is None else min_kw, SEVERE_PCT if sev is None else sev)


def transfers(vals, names):
    return gr.transfers(vals, names)


# ------------------------------------------------------------------ data (device_16_prediction_dataset.csv)
FEATS = ["V1", "V2", "V3", "I1", "I2", "I3", "PF"]
KWCOLS = PHASES
CURR, VOLT = ["I1", "I2", "I3"], ["V1", "V2", "V3"]
NEEDED = FEATS + KWCOLS
CLOCK_COLS = ("Timestamps", "Timestamp", "Timestamp_Index")

def find_data():
    for folder in (HERE, Path.cwd(), HERE / "data"):
        if (folder / DATA_NAME).exists(): return folder / DATA_NAME
    return None


def _parse(buffer, name):
    name = name.lower()
    if name.endswith(".parquet"): return pd.read_parquet(buffer)
    if name.endswith((".xlsx", ".xls")): return pd.read_excel(buffer)
    return pd.read_csv(buffer, encoding="utf-8-sig", low_memory=False)


def parse_clock(col):
    """ISO dates (2024-09-02 00:00:00) are read as year-month-day. Other layouts fall back to day-first (02/09/2024)."""
    s = col.astype(str).str.strip()
    t = pd.to_datetime(s, format="ISO8601", errors="coerce")
    if t.notna().mean() < 0.9:
        t = pd.to_datetime(s, dayfirst=True, errors="coerce")
    return t


@st.cache_data(show_spinner="Reading the file…")
def read_raw_path(path, mtime):
    return _parse(path, path)


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
        ts, clock = parse_clock(raw[tcol]), "file"
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
    pf_ok = float(np.median(PFp)) < 0.97                                      # kW / (V x I) near 1 everywhere = the "kW" columns are really V x I (kVA): no PF information in them
    sh = pd.DataFrame({"Current": np.abs(dI), "Voltage": np.abs(dV), "Power factor": np.abs(dF) if pf_ok else np.zeros(len(dF))})
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
                by_hour=by_hour, status=status, pf_ok=bool(pf_ok), pf_phase=float(PFp.mean()), pf_meter=float(d["PF"].mean()), high_share=float(per.iloc[main, 1]))


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
    elif kind == "up": acts.append(f"Compare the expected peak (<b>{peak:,.1f} kW</b>) with your sanctioned demand - set DEMAND at the top of app.py for an automatic check.")
    if pf and np.isfinite(pf) and pf > 0.3 and peak >= 1: acts.append(f"At power factor {pf:.2f} the peak of {peak:,.1f} kW is about <b>{peak / pf:,.0f} kVA</b> - compare it with the transformer / generator rating.")
    pa = pf_action(avg, pf)
    if pa: acts.append(pa)
    if kind == "up": acts += ["Stagger the start-up of large motors and move deferrable loads (batch jobs, pumps, charging) to off-peak hours.", "If the rise continues in the next forecast, plan extra capacity or a load-shedding sequence for non-critical equipment."]
    elif kind == "down": acts += ["Switch off or park idle auxiliaries and standby equipment; use the low-load window for maintenance.", "If the machine should be producing, check for a stoppage or a process problem."]
    else: acts.append("Keep monitoring; the forecast updates as new readings arrive.")
    if long_horizon: acts.append("Exponential Smoothing repeats the daily / weekly pattern it has learned: the far end of the forecast is a typical day, not a prediction of one specific event - use it as a guide.")
    return tone, head, acts



# ------------------------------------------------------------------ rule-based suggestion cards (no AI): LOAD forecast + 3-PHASE analysis
CARD_TONE = {"bad": ("#F43F5E", "#FFE4E8", "#BE123C"), "amber": ("#F59E0B", "#FFEBD5", "#C2410C"), "good": ("#10B981", "#D8F5E9", "#047857"), "info": ("#6366F1", "#E7E8FF", "#4338CA")}


def _badge(text, tone):
    c = CARD_TONE[tone]; return f'<span style="background:{c[1]};color:{c[2]};border-radius:99px;padding:2px 10px;font-size:.72rem;font-weight:600;white-space:nowrap">{text}</span>'


def card_html(title, badge, tone, items, headline="", numbered=True, border=None):
    """One white card: TITLE + badge, optional bold headline, then heading + one short sentence per item (single line of HTML - Streamlit markdown must not see blank lines)."""
    c = CARD_TONE[border or tone]
    rows = "".join(f'<div style="padding:9px 0;border-top:1px solid #EEF1F6"><div style="font-weight:600;color:{INK};font-size:.9rem">{f"{i}. " if numbered else ""}{h}</div><div style="color:#475569;font-size:.85rem;line-height:1.4">{t}</div></div>' for i, (h, t) in enumerate(items, 1))
    head = f'<div style="font-weight:600;color:{INK};font-size:.9rem;margin:8px 0 4px">{headline}</div>' if headline else ""
    return f'<div style="border:1px solid {c[0]};border-radius:14px;padding:14px 18px;background:#fff;margin-bottom:.6rem"><div style="display:flex;justify-content:space-between;align-items:center;gap:8px"><span style="font-weight:700;color:{INK};font-size:.8rem;letter-spacing:.03em">{title}</span>{_badge(badge, tone)}</div>{head}{rows}</div>'


def load_suggestions(fv, t_f, prev_kwh):
    """Short rule-based risks + actions for the forecast TOTAL kW.  Direction = forecast energy vs the previous period of the same length (> LOAD_CHG_PCT).
    Numbers come only from the forecast and the settings at the top of the file; a setting left at 0 is simply not mentioned.
    Returns dict(kind, badge, tone, headline, risks, actions, managed, ignored)."""
    fv = np.asarray(fv, float); n = len(fv); kwh = float(fv.sum()); avg = float(fv.mean())
    peak, pk = float(fv.max()), t_f[int(fv.argmax())]; low, lo = float(fv.min()), t_f[int(fv.argmin())]
    d_kwh = kwh - prev_kwh; d_pct = d_kwh / prev_kwh * 100 if prev_kwh >= 1 else float("nan")
    kind = "idle" if avg < MIN_KW else "up" if d_pct > LOAD_CHG_PCT else "down" if d_pct < -LOAD_CHG_PCT else "stable"
    pct = "" if not np.isfinite(d_pct) else f"{abs(d_pct):.0f} % {'above' if d_pct > 0 else 'below'} the previous {span_text(n)}" if abs(d_pct) >= 0.5 else f"same as the previous {span_text(n)}"
    mny = lambda x: f"{x:,.0f}{(' ' + CURRENCY) if CURRENCY else ''}"
    short = max(peak - CAPTIVE_KW, 0.0) if CAPTIVE_KW > 0 else 0.0; spare = max(CAPTIVE_KW - peak, 0.0) if CAPTIVE_KW > 0 else 0.0
    over = max(peak - DEMAND, 0.0) if DEMAND > 0 else 0.0; use = peak / DEMAND * 100 if DEMAND > 0 else float("nan")
    shave = peak * SHIFTABLE_PCT / 100; extra = max(d_kwh, 0) * TARIFF + over * DEMAND_CHARGE; unused = max(-d_kwh, 0.0)
    at = lambda ts: f"{ts:%a %H:%M}"; gap = max(short, over)
    if kind == "idle":
        risks = [("Machine expected idle", f"Average {avg:,.1f} kW (below {MIN_KW:g} kW). If it should be running, check supply, start-up and process."), ("Idle fixed charges", "Demand and fixed charges are still billed with no output.")]
        acts = [("Confirm the stop is planned", "If yes, switch off auxiliaries and use the time for maintenance."), ("Look for a fault", "Tripped breaker, supply problem, sensor or data problem.")]
        gain, lose = "A planned stop saves energy and gives a maintenance window.", "An unplanned stop means lost production and fixed charges with no output."
    elif kind == "up":
        risks = [("Higher electricity cost", f"{kwh:,.0f} kWh needed, {pct}. " + (f"Extra grid or backup power may cost about {mny(max(d_kwh, 0) * TARIFF)}." if TARIFF > 0 else "Extra grid or backup power may be needed.")),
                 ("Increased peak demand", f"Peak {peak:,.1f} kW around {at(pk)}" + (f", {short:,.1f} kW above own generation" if short else "") + (f", {over:,.1f} kW above sanctioned demand (penalty risk)" if over else f", {use:.0f} % of sanctioned demand" if DEMAND > 0 else "") + "."),
                 ("Missed energy storage" if HAS_BATTERY else "Missed load shifting", f"Battery not charged around {at(lo)} may be empty at the peak." if HAS_BATTERY else f"Flexible work left in the peak wastes the {peak - low:,.1f} kW gap to the lowest hour."),
                 ("Increased operational risk", f"Peak exceeds available power by {gap:,.1f} kW; backup or reduced operation may be needed." if gap else "If demand exceeds supply, production may stop."),
                 ("Higher energy management cost", f"Uncoordinated demand may add about {mny(extra)}." if extra > 0 else "Unmanaged peaks and grid imports raise avoidable cost.")]
        acts = [("Optimize internal generation", f"Plan own generation for {kwh:,.0f} kWh and arrange grid or backup for the rest."),
                ("Prepare for peak demand", f"Keep {peak:,.1f} kW capacity ready around {at(pk)}" + (f" (shortfall {short:,.1f} kW)." if short else f" (spare {spare:,.1f} kW)." if spare else ".")),
                ("Optimize energy storage" if HAS_BATTERY else "Use the low-load hours", f"Charge the battery around {at(lo)}; discharge at the peak." if HAS_BATTERY else f"Run flexible work around {at(lo)} ({low:,.1f} kW), not at the peak."),
                ("Shift flexible loads", f"Stagger motor starts away from {at(pk)}; moving {SHIFTABLE_PCT:g} % cuts the peak by about {shave:,.1f} kW."),
                ("Monitor power availability", "Compare the forecast with available generation; plan backup if short.")]
        gain = f"Peak can drop by about {shave:,.1f} kW" + (f", saving about {mny(shave * DEMAND_CHARGE)}" if DEMAND_CHARGE > 0 else "") + "; backup supply and stoppages are avoided."
        lose = (f"About {mny(extra)} extra cost" if extra > 0 else "Extra grid or backup cost") + (f", peak {gap:,.1f} kW over capacity" if gap else "") + ", and possible production disruption."
    elif kind == "down":
        risks = [("Under-utilised generation", f"{kwh:,.0f} kWh needed, {pct}; up to {unused:,.0f} kWh of usual generation may go unused" + (f" (about {mny(unused * TARIFF)})." if TARIFF > 0 else ".")),
                 ("Wasted fuel", f"Peak {peak:,.1f} kW leaves {spare:,.1f} kW of own generation idle." if spare else f"Generation sized for a higher load runs inefficiently (peak only {peak:,.1f} kW)."),
                 ("Low-load inefficiency", f"Lowest hour {low:,.1f} kW around {at(lo)}; motors and generators are least efficient at low load."),
                 ("Possible production slowdown", "Check the drop is planned, not a stoppage, fault or data problem."),
                 ("Fixed-cost pressure", f"Contract demand {DEMAND:g} kW is billed while the peak uses {use:.0f} %." if DEMAND > 0 else "Fixed and demand charges stay the same, so cost per kWh rises.")]
        acts = [("Right-size generation", f"Run fewer units or a lower setpoint to match {kwh:,.0f} kWh."),
                ("Use the surplus", "Charge the battery when generation exceeds demand." if HAS_BATTERY else f"Move flexible work into the low hours around {at(lo)}."),
                ("Consolidate loads", f"Run machines together; switch off idle equipment around {at(lo)}."),
                ("Review contract demand", f"Peak is {use:.0f} % of contract demand; ask if it can be lowered." if DEMAND > 0 and use < 80 else "If the peak stays well below contract demand, a lower contract may cut fixed charges."),
                ("Check why demand fell", "Maintenance, planned shutdown, sensor or hidden machine fault.")]
        gain = f"Matching generation to demand avoids up to {unused:,.0f} kWh" + (f" (about {mny(unused * TARIFF)})" if TARIFF > 0 else "") + " of unnecessary generation and fuel."
        lose = f"Up to {unused:,.0f} kWh" + (f" (about {mny(unused * TARIFF)})" if TARIFF > 0 else "") + " wasted, fixed charges on unused capacity, and a hidden fault may go unnoticed."
    else:
        risks = [("Demand is steady", f"{kwh:,.0f} kWh needed, {pct or 'no major change'}."), ("Peak to watch", f"{peak:,.1f} kW around {at(pk)}" + (f" ({use:.0f} % of sanctioned demand)." if DEMAND > 0 else ".")), ("Low-load hours", f"Lowest {low:,.1f} kW around {at(lo)}.")]
        acts = [("Keep the current plan", "Generation and storage settings can stay as they are."), ("Prepare for the peak", f"Keep {peak:,.1f} kW ready around {at(pk)}; stagger motor starts."), ("Keep monitoring", "The forecast updates as new readings arrive.")]
        gain, lose = "Staggered motor starts keep the peak low and the bill steady.", "A coincident start-up can still create a costly peak."
    big = kind == "down" and np.isfinite(d_pct) and d_pct < -10
    tone = "bad" if gap else "amber" if kind in ("up", "down") else "good"
    return dict(kind=kind, tone=tone, badge="Attention" if gap or big else "Watch" if kind in ("up", "down") else "Normal", risks=risks, actions=acts, managed=gain, ignored=lose, headline=f"Forecasted peak demand: {peak:,.1f} kW around {pk:%A, %H:%M}")


def phase_analysis_html(Pf, rep, dg, win=None):
    """The compact '3-phase load analysis' card: kW per colour vs the average, likely cause, actions, impact if not corrected.  Forecast average per colour + measured causes."""
    fc = Pf.mean(); m = float(fc.mean()); dev = (fc / m - 1) * 100 if m >= 1 else fc * 0
    n_al, n_sev = int(rep.Alarm.sum()), int((rep.Status == "SEVERE").sum()); worst = float(dev.abs().max())
    bad = worst > ALARM_PCT or n_al > 0; tone = "bad" if (n_sev or worst > SEVERE_PCT) else "amber" if bad else "good"
    badge = "Severe imbalance" if tone == "bad" else "Imbalance detected" if bad else "Balanced"
    tiles = ""
    for p, nm_ in zip(Pf.columns, NAMES):
        d = float(dev[p]); off = abs(d) > ALARM_PCT; lab = "Overloaded" if d > ALARM_PCT else "Underloaded" if d < -ALARM_PCT else "Near average"
        col = CARD_TONE["bad"][2] if off else INK
        tiles += (f'<div style="flex:1;min-width:110px"><div style="font-size:.72rem;color:{MUTED};letter-spacing:.04em">{nm_.upper()}</div><div style="font-size:1.55rem;font-weight:700;color:{col}">{fc[p]:,.1f} kW</div>'
                  f'<div style="font-size:.78rem;color:{col if off else MUTED}">{"↑" if d >= 0 else "↓"} {abs(d):.1f} % · {lab}</div></div>')
    sec = lambda t: f'<div style="font-weight:700;color:{INK};font-size:.9rem;margin:12px 0 4px">{t}</div>'
    ul = lambda xs: '<ul style="margin:2px 0 0 1.1rem;padding:0;color:#475569;font-size:.85rem;line-height:1.5">' + "".join(f"<li>{x}</li>" for x in xs) + "</ul>"
    hi = [nm for p, nm in zip(Pf.columns, NAMES) if dev[p] > ALARM_PCT] or [NAMES[int(np.argmax(fc.to_numpy()))]]; lo = [nm for p, nm in zip(Pf.columns, NAMES) if dev[p] < -ALARM_PCT] or [NAMES[int(np.argmin(fc.to_numpy()))]]
    cause, top = "", None
    if dg:
        top = dg["share"].idxmax(); ci, cv = dg["mean_I"] > ALARM_PCT, dg["mean_V"] > ALARM_PCT
        row = lambda lab, v, flag: f'<div style="display:flex;justify-content:space-between;padding:3px 0;color:#475569;font-size:.85rem"><span>{lab} ({v:.1f} %)</span>{_badge(("Above " if flag else "Within ") + f"{ALARM_PCT:g} %", "bad" if flag else "good")}</div>'
        cause = (sec("Likely cause analysis") + row("Current imbalance", dg["mean_I"], ci) + row("Voltage imbalance", dg["mean_V"], cv)
                 + f'<div style="font-weight:700;color:{INK};font-size:.88rem;margin-top:6px">Possible cause: ' + {"Current": "Uneven load distribution", "Voltage": "Supply voltage unbalance", "Power factor": "Unequal power factor between phases"}[top] + "</div>")
    if not bad: acts, imp = [f"No action needed - the colours are within {ALARM_PCT:g} %.", "Keep new single-phase loads spread across R, Y and B."], []
    else:
        if top == "Voltage": acts = ["Ask the utility to check the supply voltage and transformer tap.", "Do not rewire before the voltage is checked.", "Re-check the phase voltages afterwards."]
        else:
            mv = " and ".join(f"{r['To']} (about {r['Move A']:.0f} A)" for _, r in dg["plan"].iterrows()) if dg and len(dg["plan"]) else " / ".join(lo)
            src = dg["plan"].iloc[0]["From"] if dg and len(dg["plan"]) else " / ".join(hi)
            acts = [f"Inspect single-phase loads on {' / '.join(hi)}.", f"Move suitable loads from {src} to {mv}.", "Verify phase currents after one working day."]
        loss = dg["loss"] if dg else float(((Pf ** 2).sum(axis=1) / (3 * Pf.mean(axis=1) ** 2).replace(0, np.nan) - 1).mean() * 100)
        imp = [f"Increased cable losses and heating (about {max(loss, 0):.1f} % extra I²R heat on {hi[0]}).",
               f"Higher neutral current ({dg['neutral_mean']:.0f} A) and equipment stress." if dg and dg["neutral_mean"] >= 1 else "Higher neutral current and equipment stress.",
               "Increased risk of overheating and unwanted trips." + (f" {hi[0]} peaks at {dg['per']['Peak A'].max() / RATING * 100:.0f} % of the {RATING:,.0f} A rating." if dg and RATING > 0 else "")]
    first = Pf.index[int(np.flatnonzero(rep.Alarm.to_numpy())[0])] if n_al else None
    foot = (f"Forecast: {n_al} of {len(Pf)} hours in alarm" + (f" ({n_sev} severe)" if n_sev else "") + f", first {first:%a %d %b %H:%M}" + (f". Best time to rewire: {win['quiet'][0]:%a %d %b %H:%M} → {win['quiet'][1]:%H:%M}." if win else ".")) if n_al else f"Forecast: no alarm hour expected in the next {len(Pf)} hours."
    c = CARD_TONE[tone]
    return (f'<div style="border:1px solid {c[0]};border-radius:14px;padding:14px 18px;background:#fff"><div style="display:flex;justify-content:space-between;align-items:center"><span style="font-weight:700;color:{INK};font-size:.8rem;letter-spacing:.03em">3-PHASE LOAD ANALYSIS</span>{_badge(badge, tone)}</div>'
            + sec("Phase-wise load distribution") + f'<div style="display:flex;gap:12px;flex-wrap:wrap">{tiles}</div>' + cause + sec("Recommended actions") + ul(acts) + (sec("Potential impact if not corrected") + ul(imp) if imp else "")
            + f'<div style="margin-top:12px;padding-top:8px;border-top:1px solid #EEF1F6;color:{MUTED};font-size:.78rem">{foot}</div></div>')


# ------------------------------------------------------------------ hourly helpers for the two forecast tabs
AVG_BLOCKS_H = {"1 hour": 1, "3 hours": 3, "6 hours": 6, "12 hours": 12, "24 hours": 24}
LD_LENGTHS = {"24 hours": 24, "3 days": 72, "7 days": 168}
HIST_H = {"24 hours": 24, "3 days": 72, "7 days": 168, "14 days": 336}
SEASONS = {"Daily cycle (24 h)": 24, "Weekly cycle (168 h)": 168, "144 h": 144}
MAX_STEPS = 168                                                   # every forecast is made 7 days ahead once; the page slices what you ask for


def span_text(hours):
    hours = int(hours); return f"{hours} h" if hours < 48 else f"{hours / 24:.0f} days"


def block_averages(values, first_time, hours):
    """Average of the hourly forecast over consecutive blocks of `hours` (the last block may be shorter), with the change from block to block."""
    v = pd.Series(np.asarray(values, float), index=pd.date_range(first_time, periods=len(values), freq="h"))
    rows = [(v.index[i], v.index[min(i + hours, len(v)) - 1] + pd.Timedelta(hours=1), float(v.iloc[i:i + hours].mean()), len(v.iloc[i:i + hours])) for i in range(0, len(v), hours)]
    out = pd.DataFrame(rows, columns=["Start", "End", "Avg kW", "Hours"]); out["Change kW"] = out["Avg kW"].diff(); prev = out["Avg kW"].shift(1)
    out["Change %"] = np.where(prev >= 1, out["Change kW"] / prev * 100, np.nan)
    return out


def block_label(row):
    return f"{row.Start:%d %b %H:%M} → {row.End:%H:%M}"


@st.cache_data(show_spinner="Building the hourly series…")
def hourly_cached(tl_, min_readings):
    return gf.hourly_frame(tl_, ["KW"] + KWCOLS, min_readings)


@st.cache_data(show_spinner="Training the model and forecasting the total load…")
def run_total(y_tail, steps, model, season, fixed):
    return gf.forecast_series(y_tail.to_numpy(), steps, model, season, fixed)


@st.cache_data(show_spinner="Testing the model on the last 168 hours…")
def run_holdout(y_real, y_model, model, season, fixed, fit_steps):
    return gf.holdout_total(y_real, y_model, model, season, fixed, fit_steps)


@st.cache_data(show_spinner="Training the model on Red, Yellow and Blue ")
def run_phases(window, steps, model, season, method=gf.PHASE_METHODS[0]):
    return gf.forecast_phases(window, steps, model, season, None, method)


@st.cache_data(show_spinner="Testing the phase forecast on past weeks ")
def run_phase_backtest(load_real, load_model, model, season, fit_steps, n_weeks, method, alarm, sev, min_kw):
    return gf.backtest_phases(load_real, load_model, model, season, fit_steps, 168, n_weeks, lambda P: analyse(P, alarm, min_kw, sev), None, method)

PP_PARAMS = {"Load (kW)": ["RKW", "YKW", "BKW"], "Current (A)": ["I1", "I2", "I3"], "Voltage (V)": ["V1", "V2", "V3"]}
PP_TOL = {"Load (kW)": 2.0, "Current (A)": 2.0, "Voltage (V)": 0.5}     # % change that counts as rising / falling


def pp_frame(tl, run_kw, rule="1h", min_readings=10):
    """Loaded minutes only, averaged per hour / day. A period with too few loaded minutes is NaN (lines break)."""
    d = tl[tl["KW"] > run_kw].set_index("Timestamps")
    cols = [c for v in PP_PARAMS.values() for c in v]
    g = d[cols].resample(rule)
    return g.mean().where(g[cols[0]].count() >= min_readings)


def pp_dev(F, cols):
    X = F[cols].to_numpy(float); m = X.mean(axis=1, keepdims=True); ok = np.isfinite(m) & (m > 0)
    return pd.DataFrame(np.where(ok, (X / np.where(ok, m, 1.0) - 1) * 100, np.nan), index=F.index, columns=NAMES)


def pp_imb(F, cols):
    return pp_dev(F, cols).abs().max(axis=1)


def pp_halves(F):
    G = F.dropna(); h = len(G) // 2
    return (None, None) if h < 2 else (G.iloc[:h], G.iloc[h:])


def pp_trend(F):
    """Change of every phase / parameter, first half vs second half of the window."""
    a, b = pp_halves(F)
    if a is None: return None
    chg = pd.DataFrame(index=NAMES, columns=list(PP_PARAMS), dtype=float)
    for p, cols in PP_PARAMS.items():
        for n, c in zip(NAMES, cols):
            x0, x1 = a[c].mean(), b[c].mean(); chg.loc[n, p] = (x1 / x0 - 1) * 100 if x0 else np.nan
    txt = pd.DataFrame(index=NAMES, columns=list(PP_PARAMS), dtype=object)
    for p in PP_PARAMS:
        txt[p] = [("–" if pd.isna(v) else ("▲" if v > PP_TOL[p] else "▼" if v < -PP_TOL[p] else "●") + f" {v:+.1f} %") for v in chg[p]]
    return chg, txt


def pp_explain(F):
    """Rule-based reading: how the load imbalance moved, which parameter moved with it, which phase is responsible."""
    a, b = pp_halves(F)
    if a is None: return ["Not enough loaded hours in this window to explain a change."]
    imb = {p: (pp_imb(a, c).mean(), pp_imb(b, c).mean()) for p, c in PP_PARAMS.items()}
    (k0, k1), (i0, i1), (v0, v1) = imb["Load (kW)"], imb["Current (A)"], imb["Voltage (V)"]; dk = k1 - k0
    out = [f"<b>Load imbalance</b> went from {k0:.1f} % to {k1:.1f} % ({dk:+.1f} points). Current imbalance: {i0:.1f} → {i1:.1f} %. Voltage imbalance: {v0:.1f} → {v1:.1f} %."]
    G = F.dropna()
    if len(G) >= 10:
        k_, i_, v_ = (pp_imb(G, PP_PARAMS[p]) for p in PP_PARAMS)
        r = lambda x, y: float(np.corrcoef(x, y)[0, 1]) if np.std(x) > 0 and np.std(y) > 0 else float("nan")
        out.append(f"Over the whole window the load imbalance follows the <b>current</b> imbalance (r = {r(k_, i_):.2f}) and the <b>voltage</b> imbalance (r = {r(k_, v_):.2f}).")
    if abs(dk) < 0.5:
        out.append("The imbalance is <b>stable</b> in this window - no parameter is pushing it up or down."); return out
    sgn = 1 if dk > 0 else -1; s_i, s_v = (i1 - i0) * sgn, (v1 - v0) * sgn
    if max(s_i, s_v) < 0.2:
        out.append("Neither the current nor the voltage imbalance moved the same way, so they do not explain the change."); return out
    drv = "Current (A)" if s_i >= s_v else "Voltage (V)"; word = drv.split(" ")[0].lower()
    out.append(f"The imbalance is {'rising' if dk > 0 else 'falling'} <b>with {word}</b>: " + ("a load problem - the phases draw unequal current." if word == "current" else "a supply problem - the voltage between phases is changing; ask the utility."))
    d0, d1 = pp_dev(a, PP_PARAMS[drv]).mean(), pp_dev(b, PP_PARAMS[drv]).mean(); mv = (d1 - d0).abs().idxmax()
    out.append(f"Most responsible phase: <b>{mv}</b> - its {word} moved from {d0[mv]:+.1f} % to {d1[mv]:+.1f} % against the three-phase average.")
    return out


@st.cache_data(show_spinner="Building the load / current / voltage table…")
def params_cached(tl_, run_kw, rule, min_readings):
    return pp_frame(tl_, run_kw, rule, min_readings)
@st.cache_data(show_spinner=False)
def fill_hours(real, short=3, max_days=7):
    """Fill missing hours so lines have no gaps. Gap of up to `short` hours: straight line. Longer gap: average of the SAME HOUR on the nearest
    valid day before and after it (within +/- max_days), so the daily shape is kept and the level follows the neighbouring days."""
    out = real.interpolate(method="time", limit=short, limit_area="inside")
    for ts in out.index[out.isna().any(axis=1)]:
        for c in out.columns[out.loc[ts].isna()]:
            vals = []
            for sgn in (-1, 1):
                for k in range(1, max_days + 1):
                    t2 = ts + sgn * pd.Timedelta(days=k)
                    if t2 in real.index and pd.notna(real.at[t2, c]): vals.append(real.at[t2, c]); break
            if vals: out.at[ts, c] = float(np.mean(vals))
    return out.ffill().bfill()
# ------------------------------------------------------------------ page
bar = st.empty()                                                  # the app bar is filled in once the data is known
src = find_data()
if src is not None: raw, data_name = read_raw_path(str(src), src.stat().st_mtime), src.name
else:
    st.info(f"Put **{DATA_NAME}** in the same folder as app.py.")
    st.stop()

devices = raw_devices(raw)
device = 16 if 16 in devices else (devices[0] if devices else None)
start_iso = "2024-01-01 00:00:00"                                 # only used when a file has no timestamp column
try: tl, info = clean_data(raw, device, start_iso)
except ValueError as err: st.error(str(err)); st.stop()

RUN_KW = 3 * MIN_KW                                               # three-phase load above this = machine running
D_raw, D = make_grid(tl, SMOOTH)
S = timeline_summary(tl, RUN_KW)
Dv = D.dropna(subset=NEEDED)                                      # smoothed minutes without gaps
if len(Dv) < 200: st.error("Not enough rows: the file needs at least a few hours of one-minute readings."); st.stop()
real_clock = info["clock"] == "file"
loaded_tl = tl.KW > RUN_KW
PF_AVG = float(tl.PF[loaded_tl].mean()) if loaded_tl.any() else float(tl.PF.mean())
HEAT_MAX = max(50.0, float(np.ceil(np.nanpercentile(S["heat"].to_numpy(float), 99) / 50) * 50))
idle_days = (S["last_seen"] - S["last_run"]).total_seconds() / 86400
a_now = analyse(Dv.iloc[-1][KWCOLS].to_numpy())
status = (f'<span class="pill run"><span class="dot"></span>Running now</span>' if S["now_running"] else
          f'<span class="pill idle"><span class="dot"></span>Idle · last ran {S["last_run"]:%d %b %H:%M}</span>')
clock_chip = '<span class="chip">clock assumed</span>' if not real_clock else ""
bar.markdown(f"""<div class="appbar"><div class="logo">⚡</div><div><div class="t">GridSense</div><div class="s">Load forecasting · phase imbalance</div></div><div class="sp"></div>
{status}<span class="chip">{data_name}</span><span class="chip">{tl.Timestamps.min():%d %b} → {tl.Timestamps.max():%d %b %Y}</span><span class="chip">{len(tl):,} readings</span>{clock_chip}</div>""", unsafe_allow_html=True)


tab_overview, tab_load, tab_phase = st.tabs(["Overview", "Load forecast", "Phase balance"])
HF = hourly_cached(tl, MINR)                                      # hourly series for both forecast tabs (notebook method)
H_OK = HF is not None and len(HF["model"]) >= 72
if H_OK:
    H_real, H_mod = HF["real"], HF["model"]
    PH3 = gf.PH
    ph_real, ph_mod = H_real[KWCOLS].set_axis(PH3, axis=1), fill_hours(H_real[KWCOLS]).set_axis(PH3, axis=1)
    y_real, y_mod, t_last = H_real["KW"], H_mod["KW"], H_mod.index[-1]
    t_fut_all = pd.date_range(t_last + pd.Timedelta(hours=1), periods=MAX_STEPS, freq="h")

def need_more_data():
    note("<b>Not enough hourly data.</b> The forecasts need at least 3 days of hours that each have enough readings. "
         "For the weekly cycle and the backtest, about 5 weeks are better.", "amber")

# ================================================================== OVERVIEW
with tab_overview:
    daily = S["daily"]; total_h = (S["run_min"] + S["idle_min"]) / 60
    kreset(); k = st.columns(5)
    with k[0]: kpi("Energy used", f"{S['energy'] / 1000:,.1f} MWh", f"{len(daily)} days of data", "", "🔋", daily.kwh.values, LOAD)
    with k[1]: kpi("Machine uptime", f"{S['util']:.1f} %", f"{S['run_min'] / 60:,.0f} h on · {S['idle_min'] / 60:,.0f} h idle", "info", "⏱️", daily.run_h.values, FORECAST)
    with k[2]: kpi("Peak load", f"{S['peak']:,.0f} kW", "highest 1-minute reading", "", "📈")
    with k[3]: kpi("Average load · running", f"{S['avg_run']:,.0f} kW", "only while the machine is on", "", "⚙️", daily.avg_run.values, CYAN)
    _st = a_now.Status.iloc[0]
    with k[4]: kpi("Phase imbalance now", "idle" if not a_now.Loaded.iloc[0] else f"{a_now.Imbalance_pct.iloc[0]:.1f} %", "machine idle" if not a_now.Loaded.iloc[0] else f"{a_now.Worst_phase.iloc[0]} {a_now.Direction.iloc[0]} · limit {ALARM_PCT:g} %",
                   "good" if _st == "OK" else "bad" if _st == "SEVERE" else "warn" if _st == "ALARM" else "info", "⚖️")
    st.write("")
    a, b = st.columns([2.1, 1])
    with a, panel("Load timeline", "10-minute averages · gaps are missing data", key="timeline"):
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
    with b, panel("Where the time went", "share of all minutes", key="donut"):
        f = go.Figure(go.Pie(labels=["Running", "Idle", "No data"], values=[S["run_min"], S["idle_min"], S["nodata_min"]], hole=.68, sort=False, marker=dict(colors=[GOOD, IDLE, NODATA], line=dict(color="#fff", width=2)),
                             textinfo="percent", hovertemplate="%{label}<br>%{value:,} min<extra></extra>"))
        f.update_layout(hovermode="closest", annotations=[dict(text=f"<b>{S['util']:.1f}%</b><br><span style='font-size:11px;color:{MUTED}'>uptime</span>", showarrow=False, font=dict(size=22))])
        show(style(f, 400, True, top=40).update_layout(legend=dict(orientation="h", y=-0.08, x=0.1), margin=dict(l=10, r=10, t=40, b=60)))


# ================================================================== LOAD FORECAST (hourly total kW - Exponential Smoothing)
with tab_load:
  if not H_OK: need_more_data()
  else:
    c = st.columns([1, 1, 1.3, 1])
    L_model = "Exponential Smoothing"
    L_len = c[0].selectbox("Forecast length", list(LD_LENGTHS), index=2, key="ld_len")
    L_days = c[1].selectbox("Learn from", [0, 14, 28, 56, 90], index=0, key="ld_days", format_func=lambda x: "All history" if x == 0 else f"last {x} days", help="A short window (14-28 days) throws away most of the history and gives a larger error.")
    L_season = SEASONS[c[2].selectbox("Seasonal cycle", list(SEASONS), index=2, key="ld_season", help="Hourly data: 24 = daily pattern, 168 = weekly pattern.")]
    L_hist = c[3].selectbox("History shown", list(HIST_H), index=2, key="ld_hist")
    fixed = dict(smoothing_level=0.7, smoothing_trend=0.05, smoothing_seasonal=0.1, damping_trend=0.95)
    fit_steps = len(y_mod) if int(L_days) == 0 else int(L_days) * 24
    L_days_txt = "all available history" if int(L_days) == 0 else f"the last {L_days} days"
    fc_tot, lbl_tot, ok_tot = run_total(y_mod.iloc[-fit_steps:], MAX_STEPS, L_model, L_season, fixed)
    if not ok_tot: note(f"The model could not be fitted ({lbl_tot}); the forecast is a flat line. Try the other model or a different cycle.", "amber")
    horizon = LD_LENGTHS[L_len]; fv = fc_tot[:horizon]; t_f = t_fut_all[:horizon]
    last_actual = float(y_mod.iloc[-1]); end_value = float(fv[-1])
    change = f"{abs(end_value - last_actual) / last_actual * 100:.1f} %" if last_actual >= 1 else f"{abs(end_value - last_actual):.1f} kW"
    up_down = "▲" if end_value >= last_actual else "▼"
    energy_f, avg_f, peak_f = float(fv.sum()), float(fv.mean()), float(fv.max()); energy_before = float(y_mod.iloc[-horizon:].sum())
    with panel("Power needed in the selected period", f"{t_f[0]:%d %b %H:%M} → {(t_f[-1] + pd.Timedelta(hours=1)):%d %b %H:%M} · {span_text(horizon)} · energy = sum of hourly kW", key="ld_pw"):
        kreset(); pk = st.columns(5)
        with pk[0]: kpi("Energy needed · forecast", f"{energy_f:,.0f} kWh", f"average {avg_f:,.1f} kW over {span_text(horizon)}", "info", "🔌")
        diff_e = energy_f - energy_before
        with pk[1]: kpi("Energy used · previous period", f"{energy_before:,.0f} kWh", (f"forecast is {diff_e:+,.0f} kWh ({diff_e / energy_before * 100:+.0f} %) vs this" if energy_before >= 1 else "the machine was idle in that period") + f" · last {span_text(horizon)}", "", "📊")
        with pk[2]: kpi("Power at first hour", f"{fv[0]:,.1f} kW", f"{t_f[0]:%d %b %H:%M}", "", "▶️")
        with pk[3]: kpi("Power at last hour", f"{fv[-1]:,.1f} kW", f"{t_f[-1]:%d %b %H:%M} · {fv[-1] - fv[0]:+,.1f} kW{pct_text(pct_change(fv[0], fv[-1]))} vs first", "info", "⏹️")
        with pk[4]: kpi("Average · peak", f"{avg_f:,.1f} kW", f"peak {peak_f:,.1f} kW at {t_f[int(np.argmax(fv))]:%a %H:%M}", "", "📈")
        st.write("")
        valid_avg = [n for n, h in AVG_BLOCKS_H.items() if h <= horizon]
        sel_col, _ = st.columns([1.2, 3])
        avg_choice = sel_col.selectbox("Show average power every", valid_avg, index=min(2, len(valid_avg) - 1), key="ld_avg", help="Splits the forecast into blocks of this length and averages the power in each block.")
        blk = block_averages(fv, t_f[0], AVG_BLOCKS_H[avg_choice]); labels = [block_label(r) for r in blk.itertuples()]; two = len(blk) >= 2
        kind = trend_kind(float(blk["Avg kW"].iloc[0]), float(blk["Avg kW"].iloc[-1])) if two else trend_kind(float(fv[0]), float(fv[-1]))
        first_v, last_v = (float(blk["Avg kW"].iloc[0]), float(blk["Avg kW"].iloc[-1])) if two else (float(fv[0]), float(fv[-1]))
        slope = float(np.polyfit(np.arange(len(blk)) * AVG_BLOCKS_H[avg_choice], blk["Avg kW"], 1)[0]) if two else float("nan")
        cl, cr = st.columns([2.2, 1.1])
        with cl:
            f = go.Figure(go.Bar(x=labels, y=blk["Avg kW"], marker_color=ACCENT, text=[("" if pd.isna(x) else f"{x:+.1f} %") for x in blk["Change %"]], textposition="outside", cliponaxis=False,
                                 customdata=np.c_[blk["Change kW"].fillna(0), blk["Hours"]], hovertemplate="%{x}<br><b>%{y:,.1f} kW average</b><br>change vs previous block %{customdata[0]:+,.1f} kW<br>%{customdata[1]} h<extra></extra>"))
            f.update_layout(hovermode="closest", yaxis_title=f"average kW per {avg_choice}", title=f"Average power every {avg_choice}  ·  label = change vs previous block"); f.update_xaxes(type="category")
            show(style(f, 300, False, top=60))
            st.caption((f"Average growth: {blk['Change kW'].mean():+,.1f} kW per block · trend {slope:+,.1f} kW per hour" if two else "Only one block fits in this window - pick a shorter interval or a longer forecast.")
                       + (f" · last block is partial ({int(blk['Hours'].iloc[-1])} h)" if blk["Hours"].iloc[-1] < blk["Hours"].iloc[0] else ""))
        with cr:
            tone_a, head_a, acts_a = power_advice(kind, first_v, last_v, peak_f, horizon > 72, dict(avg=avg_f, slope=slope, pf=PF_AVG, demand=DEMAND, peak_at=t_f[int(np.argmax(fv))]))
            note(advice_html(head_a, acts_a), tone_a)

    ls_ = load_suggestions(fv, t_f, energy_before)
    with panel("Load forecast · risks and recommendations", f"rule-based · forecast total kW vs the previous {span_text(horizon)} · no AI is used", key="ld_sugg"):
        lc_ = st.columns(2)
        lc_[0].markdown(card_html("POTENTIAL RISK &amp; COST IMPACT", ls_["badge"], ls_["tone"], ls_["risks"]), unsafe_allow_html=True)
        lc_[1].markdown(card_html("ENERGY MANAGEMENT RECOMMENDATIONS", f"{span_text(horizon)} forecast", "info", ls_["actions"], headline=ls_["headline"]), unsafe_allow_html=True)
        lc2_ = st.columns(2)
        with lc2_[0]: note(f"<b>If managed:</b> {ls_['managed']}", "good")
        with lc2_[1]: note(f"<b>If not managed:</b> {ls_['ignored']}", ls_["tone"] if ls_["tone"] != "good" else "amber")
        st.caption("Set CAPTIVE_KW, HAS_BATTERY, SHIFTABLE_PCT, TARIFF and DEMAND_CHARGE at the top of app.py for site-specific numbers. A qualified engineer should review every action.")

    with panel("Forecast ahead", f"{L_model} · {lbl_tot} · learns from {L_days_txt} · starts after {t_last:%d %b %Y %H:%M} (hourly means)", key="ld_next"):
        left, right = st.columns([1, 2.4])
        with left:
            st.markdown(f"""<div class="hero-f"><div class="l">Expected load · {t_f[-1]:%d %b %H:%M}</div><div class="big">{end_value:,.1f}<small> kW</small></div>
<div class="r">average over the period <b>{fv.mean():,.1f} kW</b></div><span class="badge flat">{up_down} {change} vs last hour</span>
<div class="mini"><div><b>{fv.max():,.1f}</b><span>peak kW</span></div><div><b>{horizon}</b><span>hourly points</span></div><div><b>{span_text(horizon)}</b><span>ahead</span></div></div></div>""", unsafe_allow_html=True)
        with right:
            hist = y_real.iloc[-HIST_H[L_hist]:]; f = go.Figure()
            f.add_shape(type="rect", xref="x", yref="paper", x0=t_last, x1=t_f[-1], y0=0, y1=1, fillcolor=tint(ACCENT, .06), line_width=0)
            f.add_scatter(x=hist.index, y=hist.values, name="Actual hourly kW", mode="lines", line=dict(color=INK, width=2.2), connectgaps=False)
            f.add_scatter(x=[t_last] + list(t_f), y=[last_actual] + list(fv), name=f"Forecast ({L_model})", mode="lines", line=dict(color=FORECAST, width=3))
            f.add_shape(type="line", xref="x", yref="paper", x0=t_last, x1=t_last, y0=0, y1=1, line=dict(color=MUTED, width=1.2, dash="dot"))
            f.add_annotation(x=t_last, yref="paper", y=1.0, text="last data point", showarrow=False, xanchor="right", yanchor="bottom", font=dict(size=11, color=MUTED))
            f.update_yaxes(title="kW"); show(style(f, 340, True, top=70))
        st.caption("Exponential Smoothing gives one line, no confidence band. How far to trust it is shown in 'Model check' below.")

    # ---- model check on the last 168 hours (the load notebook's test)
    st.markdown('<div class="ph" style="margin-top:.6rem"><b>Model check on past data</b><span>the last 168 hours are hidden, forecast from the hours before, then compared with what really happened</span></div>', unsafe_allow_html=True)
    ho = run_holdout(y_real, y_mod, L_model, L_season, fixed, fit_steps)
    if ho is None: st.caption("Not enough history for a 7-day test.")
    else:
        sc = ho["scores"]; mean_a = float(np.nanmean(ho["actual"].to_numpy()))
        kreset(); k = st.columns(4)
        with k[0]: kpi("MAE", f"{sc['MAE']:.2f} kW", f"{sc['MAE'] / max(mean_a, 1e-9) * 100:.1f} % of the mean test load" if mean_a >= 1 else "test period has ~0 kW load", "info", "📏")
        with k[1]: kpi("RMSE", f"{sc['RMSE']:.2f} kW", f"{sc['n']} hours compared", "", "🎯")
        with k[2]: kpi("WAPE", f"{sc['WAPE']:.1f} %", "total error ÷ total load · smaller is better", "good" if sc["WAPE"] < 10 else "warn" if sc["WAPE"] < 25 else "bad", "✅")
        with k[3]: kpi("SMAPE", f"{sc['SMAPE']:.1f} %", "average % error · smaller is better", "", "🧪")
        f = go.Figure(); f.add_scatter(x=ho["actual"].index, y=ho["actual"].values, name="Actual hourly kW", line=dict(color=INK, width=2), connectgaps=False)
        f.add_scatter(x=ho["pred"].index, y=ho["pred"].values, name=f"Predicted ({L_model})", line=dict(color=FORECAST, width=2, dash="dash")); f.update_yaxes(title="kW")
        with st.container(border=True, key="panel_ld_test"): show(style(f, 340, True, top=60))


# ================================================================== PHASE BALANCE (measured + forecast + action plan, all on one page)
with tab_phase:
  if not H_OK: need_more_data()
  else:
    c = st.columns([1, 1, 1.3, 1, 1.5])
    P_model = "Exponential Smoothing"
    P_len = c[0].selectbox("Forecast length", list(LD_LENGTHS), index=2, key="ph_len")
    P_days = c[1].selectbox("Learn from last", [14, 28, 56], index=1, key="ph_days", format_func=lambda x: f"{x} days")
    P_season = {"Weekly cycle (168 h)": 168, "Daily cycle (24 h)": 24}[c[2].selectbox("Seasonal cycle", ["Weekly cycle (168 h)", "Daily cycle (24 h)"], key="ph_season")]
    P_hist = c[3].selectbox("History shown", list(HIST_H), index=2, key="ph_hist")
    P_method = c[4].selectbox("Forecast method", gf.PHASE_METHODS, key="ph_method", help="Each phase separately = forecast R, Y and B one by one. Total × phase shares = forecast the total and each phase's share, then rebuild R / Y / B. The test below shows which one does better on your data.")
    fit_steps = int(P_days) * 24; horizon = LD_LENGTHS[P_len]
    fc_arr, lbls, ok_ph = run_phases(ph_mod.iloc[-fit_steps:], MAX_STEPS, P_model, P_season, P_method)
    Pf = pd.DataFrame(fc_arr[:horizon], index=t_fut_all[:horizon], columns=PH3); rep = analyse(Pf.to_numpy())
    if not ok_ph: note(f"A model fit failed for {', '.join(p for p in PH3 if lbls[p].startswith('failed'))} - that phase repeats its last value. Try the other model or cycle.", "amber")

    DG = diagnose(Dv[NEEDED], ALARM_PCT, MIN_KW, SEVERE_PCT); dg = None if DG["empty"] else DG
    r0 = a_now.iloc[0]
    hr_ = analyse(ph_real.dropna().to_numpy()); hr_mean = float(hr_.Imbalance_pct[hr_.Loaded].mean()) if hr_.Loaded.any() else float("nan")      # same hourly basis as the forecast
    now = dict(loaded=bool(r0.Loaded), imb=float(r0.Imbalance_pct) if r0.Loaded else 0.0, worst=r0.Worst_phase, dev=float(r0.Worst_dev) if r0.Loaded else 0.0, direction=r0.Direction, status=r0.Status)

    imb = rep.Imbalance_pct.to_numpy(); n_al, n_sev = int(rep.Alarm.sum()), int((rep.Status == "SEVERE").sum())
    first_alarm = Pf.index[int(np.flatnonzero(rep.Alarm.to_numpy())[0])] if n_al else None
    worst_k = int(np.nanargmax(imb)) if np.isfinite(imb).any() else 0
    kreset(); k = st.columns(5)
    with k[0]: kpi("Imbalance now", "idle" if not now["loaded"] else f"{now['imb']:.1f} %", "machine idle" if not now["loaded"] else f"{now['worst']} {now['direction']} · limit {ALARM_PCT:g} %", "good" if now["status"] == "OK" else "bad" if now["status"] == "SEVERE" else "warn" if now["status"] == "ALARM" else "info", "⚖️")
    with k[1]:
        if dg: kpi("Average imbalance · measured", f"{dg['mean_imb']:.1f} %", f"above the limit in {dg['pct_above']:.0f} % of loaded minutes · hourly means: {hr_mean:.1f} %", "good" if dg["mean_imb"] <= ALARM_PCT else "warn" if dg["mean_imb"] <= SEVERE_PCT else "bad", "📏")
        else: kpi("Average imbalance · measured", "–", "too few loaded minutes", "", "📏")
    with k[2]: kpi("Forecast alarm hours", f"{n_al}", f"of {horizon} hours · {n_sev} severe", "", "🚨", trend=np.nan_to_num(imb), trend_color=ACCENT)
    with k[3]: kpi("First forecast alarm", "none" if first_alarm is None else f"{first_alarm:%a %H:%M}", "no alarm expected" if first_alarm is None else f"{first_alarm:%d %b} · in {(first_alarm - t_last).total_seconds() / 3600:.0f} h", "good" if first_alarm is None else "warn", "⏰")
    with k[4]: kpi("Worst colour · forecast", "–" if not np.isfinite(imb).any() else rep.Worst_phase.iloc[worst_k], "" if not np.isfinite(imb).any() else f"{rep.Direction.iloc[worst_k]} · peak {imb[worst_k]:.1f} % at {Pf.index[worst_k]:%a %H:%M}", "info", "🎨")
    st.write("")

    # ---- one chart: measured hours + forecast per phase; an X marks the hours when THAT phase is the one out of balance
    with panel("Phase load · measured and forecast", f"hourly means · {P_model} · learns from the last {P_days} days · forecast starts after {t_last:%d %b %Y %H:%M} · ✕ = alarm hour on the phase that is out of balance", key="ph_main"):
        hh = ph_mod.iloc[-HIST_H[P_hist]:]; xt = [t_last] + list(Pf.index)
        f = make_subplots(rows=3, cols=1, shared_xaxes=True, vertical_spacing=.08, subplot_titles=[f"{n} phase (kW)" for n in NAMES])
        for j, p in enumerate(PH3, 1):
            col = PHASE_COL[KWCOLS[j - 1]]
            f.add_scatter(x=hh.index, y=hh[p], name="Measured (hourly)", legendgroup="m", showlegend=j == 1, line=dict(color=INK, width=2), connectgaps=False, row=j, col=1)
            f.add_scatter(x=xt, y=[float(ph_mod[p].iloc[-1])] + list(Pf[p]), name="Forecast", legendgroup="f", showlegend=j == 1, mode="lines", line=dict(color=col, width=2.6, dash="dash"), row=j, col=1)
        worst_nm, legend_done = rep.Worst_phase.to_numpy(), set()
        for sev_, col_ in (("ALARM", WARN), ("SEVERE", BAD)):
            for j, p in enumerate(PH3, 1):
                m_ = (rep.Status == sev_).to_numpy() & (worst_nm == NAMES[j - 1])
                if not m_.any(): continue
                f.add_scatter(x=Pf.index[m_], y=Pf[p].to_numpy()[m_], mode="markers", name=f"{sev_.title()} (forecast)", legendgroup=sev_, showlegend=sev_ not in legend_done,
                              marker=dict(symbol="x", size=9, color=col_), customdata=np.c_[imb[m_], rep.Direction.to_numpy()[m_]],
                              hovertemplate="%{y:.1f} kW · imbalance %{customdata[0]:.1f} % (%{customdata[1]})<extra>" + sev_.title() + "</extra>", row=j, col=1)
                legend_done.add(sev_)
        f.add_shape(type="line", xref="x", yref="paper", x0=t_last, x1=t_last, y0=0, y1=1, line=dict(color=MUTED, width=1.2, dash="dot"))
        f.update_yaxes(title_text="kW")
        style(f, 640, legend_y=1.0, top=110); f.update_annotations(font=dict(size=13, color=INK)); show(f)

    with panel("Forecast by day", "hourly calls are noisy - the day-level picture is steadier", key="ph_days"):
        fd = Pf.assign(imb=rep.Imbalance_pct.to_numpy(), worst=rep.Worst_phase.to_numpy(), alarm=rep.Alarm.to_numpy(), sev=(rep.Status == "SEVERE").to_numpy(), day=Pf.index.date)
        rows_d = []
        for d_, g_ in fd.groupby("day"):
            ga = g_[g_["alarm"]]
            rows_d.append({"Day": f"{d_:%a %d %b}", "Hours": len(g_), "Avg imbalance %": round(float(g_["imb"].mean()), 1) if g_["imb"].notna().any() else np.nan,
                           "Alarm hours": int(g_["alarm"].sum()), "Severe hours": int(g_["sev"].sum()), "Main phase out of balance": ga["worst"].mode().iloc[0] if len(ga) else "-"})
        st.dataframe(pd.DataFrame(rows_d).set_index("Day"), width="stretch")
    if dg and not dg["pf_ok"]: note("The kW columns match voltage × current almost exactly (power factor ≈ 1), so they look like <b>kVA</b>, not kW. The power-factor share is therefore left out of 'What explains it'. Check with the meter owner.", "amber")

    # ---- 3-phase load analysis card (rule-based, short)
    with panel("3-phase load analysis", "rule-based from the measured numbers and the phase forecast · no AI is used", key="ph_plan"):
        win_ = gr.best_windows(Pf.sum(axis=1).to_numpy(), Pf.index) if n_al else None
        st.markdown(phase_analysis_html(Pf, rep, dg, win_), unsafe_allow_html=True)
        st.caption("Every action needs review by a qualified engineer before anything is changed.")

    # ---- the evidence behind the advice
        # ---- A) which parameter moves each phase: load, current, voltage per colour + the imbalance each one creates
    with panel("Phase parameters · what is rising or falling", "loaded minutes only · load, current and voltage of each colour, and the imbalance each one creates", key="ph_params"):
        pc = st.columns([1, 1, 3])
        pp_rule = pc[0].selectbox("Average by", ["Hour", "Day"], key="pp_rule")
        pp_days = pc[1].selectbox("Window", [7, 14, 28], key="pp_days", format_func=lambda x: f"last {x} days")
        PP = params_cached(tl, RUN_KW, "1h" if pp_rule == "Hour" else "1D", 10 if pp_rule == "Hour" else 60).iloc[-(pp_days * (24 if pp_rule == "Hour" else 1)):]
        if PP.dropna().shape[0] < 4: st.caption("Not enough loaded periods in this window.")
        else:
            pcol = [PHASE_COL[k_] for k_ in PHASES]
            g1, g2 = st.columns(2)
            with g1:
                f = go.Figure()
                for (pn_, cols_), col_ in zip(PP_PARAMS.items(), [ACCENT, BLUE, WARN]):
                    f.add_scatter(x=PP.index, y=pp_imb(PP, cols_), name=pn_.split(" ")[0], line=dict(color=col_, width=2), connectgaps=True)
                f.add_hline(y=ALARM_PCT, line_dash="dot", line_color=WARN, annotation_text=f"alarm {ALARM_PCT:g} %", annotation_position="top left")
                f.update_layout(title="Imbalance % of load, current and voltage", yaxis_title="%"); show(style(f, 300, True, top=70), key="pp_imb")
            with g2:
                dv = pp_dev(PP, PP_PARAMS["Load (kW)"]); f = go.Figure()
                for n_, col_ in zip(NAMES, pcol): f.add_scatter(x=dv.index, y=dv[n_], name=n_, line=dict(color=col_, width=2), connectgaps=True)
                f.add_hline(y=0, line_color=MUTED, line_width=1); f.add_hrect(y0=-ALARM_PCT, y1=ALARM_PCT, fillcolor=tint(GOOD, .08), line_width=0)
                f.update_layout(title="Which phase is off: load vs the 3-phase average", yaxis_title="% above / below average"); show(style(f, 300, True, top=70), key="pp_dev")

            tt = pp_trend(PP)
            if tt:
                st.markdown('<div class="ph"><b>What changed</b><span>first half vs second half of the window · ▲ rising · ▼ falling · ● steady (±2 % load and current, ±0.5 % voltage)</span></div>', unsafe_allow_html=True)
                st.dataframe(tt[1], width="stretch")
            note("<ul>" + "".join(f"<li>{t_}</li>" for t_ in pp_explain(PP)) + "</ul>")

      # ---- B) actual vs predicted on past weeks (follows the Forecast length / Learn from / Seasonal cycle / Forecast method chosen at the top)
    with panel("Model check · actual vs predicted on past weeks", "each week is hidden, forecast from the hours before it, then compared with what really happened", key="ph_bt"):
        bt_weeks = st.selectbox("Weeks tested", [1, 2, 3, 4], index=1, key="bt_weeks")
        ref = run_phase_backtest(ph_real, ph_mod, P_model, P_season, fit_steps, bt_weeks, P_method, ALARM_PCT, SEVERE_PCT, MIN_KW)
        if ref is None: st.caption("Not enough history: the test needs the 'Learn from last' window plus the weeks tested. Lower either one.")
        else:
            A_, F_ = ref["actual"], ref["pred"]; A_ = A_.fillna(ph_mod.reindex(A_.index))        # missing hours are drawn with the filled values; the error below uses real hours only
            ra_, rf_ = analyse(A_.to_numpy()), analyse(F_.to_numpy()); ra_.index = A_.index; rf_.index = F_.index
            f = make_subplots(rows=4, cols=1, shared_xaxes=True, vertical_spacing=.06, subplot_titles=[f"{n_} phase (kW)" for n_ in NAMES] + ["Imbalance % (calculated from the phase kW)"])
            for j, p in enumerate(PH3, 1):
                f.add_scatter(x=A_.index, y=A_[p], name="Actual", legendgroup="a", showlegend=j == 1, line=dict(color=INK, width=2), row=j, col=1)
                f.add_scatter(x=F_.index, y=F_[p], name="Predicted", legendgroup="p", showlegend=j == 1, line=dict(color=PHASE_COL[KWCOLS[j - 1]], width=2.2, dash="dash"), row=j, col=1)
            f.add_scatter(x=A_.index, y=ra_.Imbalance_pct.where(ra_.Loaded), name="Actual", legendgroup="a", showlegend=False, line=dict(color=INK, width=2), connectgaps=True, row=4, col=1)
            f.add_scatter(x=F_.index, y=rf_.Imbalance_pct.where(rf_.Loaded), name="Predicted", legendgroup="p", showlegend=False, line=dict(color=ACCENT, width=2.2, dash="dash"), connectgaps=True, row=4, col=1)
            f.add_hline(y=ALARM_PCT, line_dash="dot", line_color=WARN, row=4, col=1)
            style(f, 780, legend_y=1.0, top=110); f.update_annotations(font=dict(size=13, color=INK)); show(f, key="bt_main")
            sp_ = float(np.nanmean([F_[p].std() / A_[p].std() for p in PH3 if A_[p].std() > 0])); wp_ = float(ref["per_phase"]["WAPE"].mean())
            st.caption(f"{ref['imbalance']['weeks']} week(s) · {P_method} · {P_season} h cycle · phase kW error {wp_:.1f} % (WAPE, real hours only) · the predicted line moves {sp_ * 100:.0f} % as much as the actual line"
                       + (" - FLAT: the model is not learning the pattern; try the other seasonal cycle or a different 'Learn from last'." if sp_ < 0.5 else "."))
