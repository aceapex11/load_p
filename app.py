"""GridSense - machine load intelligence (light mode). Rule-based advice, no AI service, no API key.
Data            : device_16_full_data.csv - one row per minute with V1-V3, I1-I3, PF, RKW, YKW, BKW
                  (an optional Timestamps / Timestamp / Timestamp_Index column is used when present)
Overview        : load timeline (kW, phases, current, voltage, imbalance) and where the time went (uptime)
Load forecast   : hourly total kW, Exponential Smoothing (sktime) tested on the last 168 hours, power advice
Phase balance   : ONE page - measured imbalance, 7-day R / Y / B forecast, imbalance alarms,
                  and a rule-based action plan: what is out of phase, why, what to do now, when to do what, what can happen
Files next to app.py: gridsense_forecast.py (the notebooks' models, built on sktime) and gridsense_rules.py (the rule-based phase-balance logic)
Layout of this file : page setup -> settings -> colours + CSS -> UI helpers -> data loading -> phase maths -> advice text -> forecast helpers -> the page itself
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

# ------------------------------------------------------------------ PAGE SETUP
warnings.filterwarnings("ignore")


st.set_page_config(page_title="GridSense · Load intelligence", page_icon="⚡", layout="wide", initial_sidebar_state="collapsed")

# ------------------------------------------------------------------ SETTINGS - edit the limits here
APP_DIR = Path(__file__).resolve().parent
DATA_FILE_NAME = "device_16_full_data.csv"       # put this file next to app.py
PHASE_KW_COLS = ["RKW", "YKW", "BKW"]
PHASE_LABEL_BY_COL = {"RKW": "Red", "YKW": "Yellow", "BKW": "Blue"}
PHASE_LABELS = [PHASE_LABEL_BY_COL[p] for p in PHASE_KW_COLS]


# LIMITS - edit here.  (The old "Data & limits" panel is gone.)
ALARM_LIMIT_PCT = 5.0      # imbalance % above this = ALARM


MIN_PHASE_KW = 5.0         # average kW per phase below this = machine off
SMOOTHING_MINUTES = 30          # minutes of rolling mean (past minutes only)
SANCTIONED_DEMAND_KW = 0.0         # sanctioned demand in kW, 0 = unknown
PHASE_RATING_A = 0.0         # phase breaker / cable in A, 0 = unknown
LOAD_CHANGE_PCT = 5.0   # forecast energy must differ more than this % from the previous period to count as an INCREASE / DECREASE
OWN_GENERATION_KW = 0.0     # own (captive) generation capacity in kW, 0 = unknown
HAS_BATTERY = False  # True if a battery storage system is available
SHIFTABLE_LOAD_PCT = 10.0 # % of the peak that could be moved to other hours (flexible loads)
TARIFF_PER_KWH = 0.0         # money per kWh, 0 = do not show a cost
DEMAND_CHARGE_PER_KW = 0.0  # money per kW of peak demand, 0 = do not show a cost
CURRENCY = ""        # e.g. "INR"
MIN_READINGS_PER_HOUR = 45            # an hour with fewer one-minute readings than this counts as missing (as in the notebooks)

# ------------------------------------------------------------------ COLUMN NAMES AND PAGE OPTIONS
MEASURED_COLS = ["V1", "V2", "V3", "I1", "I2", "I3", "PF"]
CURRENT_COLS, VOLTAGE_COLS = ["I1", "I2", "I3"], ["V1", "V2", "V3"]
REQUIRED_COLS = MEASURED_COLS + PHASE_KW_COLS
TIME_COLUMN_CANDIDATES = ("Timestamps", "Timestamp", "Timestamp_Index")
AVERAGE_BLOCK_HOURS = {"1 hour": 1, "3 hours": 3, "6 hours": 6, "8 hours": 8, "12 hours": 12, "24 hours": 24}
FORECAST_LENGTH_HOURS = {f"{d} day{'s' if d > 1 else ''}": d * 24 for d in range(1, 8)}
HISTORY_SHOWN_HOURS = {"24 hours": 24, "3 days": 72, "7 days": 168, "14 days": 336}
SEASON_LENGTH_HOURS = {f"{d} day{'s' if d > 1 else ''}": d * 24 for d in range(1, 8)}
MAX_FORECAST_HOURS = 168                                                   # every forecast is made 7 days ahead once; the page slices what you ask for
PHASE_PARAMETER_COLS = {"Load (kW)": ["RKW", "YKW", "BKW"], "Current (A)": ["I1", "I2", "I3"], "Voltage (V)": ["V1", "V2", "V3"]}
TREND_TOLERANCE_PCT = {"Load (kW)": 2.0, "Current (A)": 2.0, "Voltage (V)": 0.5}     # % change that counts as rising / falling

# ------------------------------------------------------------------ COLOUR THEME
# COLOUR THEME  (one palette everywhere)
#   neutrals : slate  (INK text / actual values, MUTED secondary, GRID lines)
#   accent   : indigo (ACCENT = measured load + forecasts + selected buttons; ACCENT_LIGHT = its tint)
#   status   : green = good, amber = warning, rose = alarm / bad   (status only)
#   phases   : R rose, Y amber, B blue                              (phase charts only)
TEXT_COLOR, MUTED_COLOR, GRID_COLOR = "#0F172A", "#64748B", "#EDF0F6"


ACCENT, ACCENT_LIGHT = "#6366F1", "#818CF8"
LOAD_COLOR = FORECAST_COLOR = ACCENT
CONFIDENCE_Z, CONFIDENCE_TEXT = 1.96, "95 %"                              # width of the forecast interval
GOOD_COLOR, WARN_COLOR, BAD_COLOR = "#10B981", "#F59E0B", "#F43F5E"
BLUE_COLOR = "#3B82F6"
IDLE_COLOR, NO_DATA_COLOR, AXIS_COLOR = "#CBD5E1", BAD_COLOR, "#E2E8F0"            # idle share (slate), NO DATA (red), axis lines / empty bars


def hex_to_rgba(hex_color, alpha):
    """'#6366F1', .16  ->  'rgba(99,102,241,.16)'  (transparent fills that follow the theme colours)."""
    h = hex_color.lstrip("#"); return f"rgba({int(h[0:2], 16)},{int(h[2:4], 16)},{int(h[4:6], 16)},{alpha})"


PHASE_COLORS = {"RKW": "#F43F5E", "YKW": "#F59E0B", "BKW": "#3B82F6"}

# ------------------------------------------------------------------ DESIGN SYSTEM (light + indigo accent, forced in CSS - no config file needed)
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

# ------------------------------------------------------------------ UI HELPERS (cards, panels, charts)
def sparkline_svg(values, color):
    v = np.asarray(values, float); v = v[~np.isnan(v)]
    if len(v) < 2: return ""
    y = 27 - (v - v.min()) / (v.max() - v.min() + 1e-9) * 22
    pts = " ".join(f"{i * 100 / (len(v) - 1):.1f},{yy:.1f}" for i, yy in enumerate(y))
    return f'<svg class="spark" viewBox="0 0 100 30" preserveAspectRatio="none"><polygon points="0,30 {pts} 100,30" fill="{color}22"/><polyline points="{pts}" fill="none" stroke="{color}" stroke-width="1.8" vector-effect="non-scaling-stroke"/></svg>'


KPI_CARD_COUNTER = [0]                                                       # card counter: blue, black, blue, black ... (reset at the start of every row of cards)


def reset_kpi_counter():
    KPI_CARD_COUNTER[0] = 0


def kpi_card(label, value, sub="", tone="", icon="", trend=None, trend_color=FORECAST_COLOR):
    """One card. Only the number colour alternates blue / black; icon, trend line and everything else keep their original colours."""
    # alternate the number colour blue / black from card to card
    blue = KPI_CARD_COUNTER[0] % 2 == 0; KPI_CARD_COUNTER[0] += 1
    st.markdown(f'<div class="kpi {"kb" if blue else "kk"} {tone}"><div class="top"><div class="ic">{icon}</div><div class="l">{label}</div></div><div class="v">{value}</div><div class="s">{sub}</div>'
                f'{sparkline_svg(trend, trend_color) if trend is not None else ""}</div>', unsafe_allow_html=True)


def note_box(text, tone=""):
    st.markdown(f'<div class="note {tone}">{text}</div>', unsafe_allow_html=True)


@contextmanager
def panel(title, sub="", key="p"):
    with st.container(border=True, key=f"panel_{key}"):
        st.markdown(f'<div class="ph"><b>{title}</b></div>', unsafe_allow_html=True)             # the grey description (sub) is intentionally not shown
        yield


def style_figure(fig, h=440, legend=True, legend_y=1.02, top=95):
    """Title sits at the top-left; the legend gets its own row just above the plot, so they never overlap."""
    # one shared look for every chart
    fig.update_layout(height=h + (top - 50), template="plotly_white", paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)", hovermode="x unified",
                      font=dict(family="Inter, sans-serif", size=12.5, color=TEXT_COLOR), margin=dict(l=10, r=10, t=top, b=10),
                      title=dict(x=0, xanchor="left", y=0.985, yanchor="top", font=dict(size=15, color=TEXT_COLOR)),
                      hoverlabel=dict(bgcolor="#FFFFFF", bordercolor=AXIS_COLOR, font=dict(family="Inter, sans-serif", size=12, color=TEXT_COLOR)),
                      legend=dict(orientation="h", x=0, xanchor="left", y=legend_y, yanchor="bottom", bgcolor="rgba(0,0,0,0)"), showlegend=legend)
    fig.update_xaxes(gridcolor=GRID_COLOR, zeroline=False, linecolor=AXIS_COLOR, showspikes=False, tickfont=dict(color=MUTED_COLOR), title_font=dict(color=MUTED_COLOR)); fig.update_yaxes(gridcolor=GRID_COLOR, zeroline=False, tickfont=dict(color=MUTED_COLOR), title_font=dict(color=MUTED_COLOR))
    return fig


def show_figure(fig, key=None):
    st.plotly_chart(fig, width="stretch", config={"displayModeBar": False}, key=key)

# ------------------------------------------------------------------ DATA LOADING AND CLEANING
def find_data_file():
    for folder in (APP_DIR, Path.cwd(), APP_DIR / "data"):
        if (folder / DATA_FILE_NAME).exists(): return folder / DATA_FILE_NAME
    return None


def _read_table(buffer, name):
    name = name.lower()
    if name.endswith(".parquet"): return pd.read_parquet(buffer)
    if name.endswith((".xlsx", ".xls")): return pd.read_excel(buffer)
    return pd.read_csv(buffer, encoding="utf-8-sig", low_memory=False)


def parse_timestamps(col):
    """ISO dates (2024-09-02 00:00:00) are read as year-month-day. Other layouts fall back to day-first (02/09/2024)."""
    s = col.astype(str).str.strip()
    t = pd.to_datetime(s, format="ISO8601", errors="coerce")
    if t.notna().mean() < 0.9:
        t = pd.to_datetime(s, dayfirst=True, errors="coerce")
    return t


@st.cache_data(show_spinner="Reading the file…")
def load_raw_file(path, mtime):
    return _read_table(path, path)


@st.cache_data(show_spinner="Cleaning the data…")
def clean_meter_data(raw_table, device_id, assumed_start_time):
    """Same cleaning as the notebooks: pick one device, build the minute clock, tiny negatives -> 0, impossible meter values removed.
    Returns the cleaned rows (one per recorded minute, total KW = R + Y + B) and a few facts about the file."""
    # work on a copy so the cached raw table is never changed
    raw_table = raw_table.copy()
    # keep one device when the file holds several
    if "DeviceID" in raw_table.columns and raw_table["DeviceID"].nunique() > 1 and device_id is not None: raw_table = raw_table[raw_table["DeviceID"] == device_id]
    # stop early when a required column is absent
    missing = [c for c in REQUIRED_COLS if c not in raw_table.columns]
    if missing: raise ValueError(f"These columns are missing: {missing}. The file has: {list(raw_table.columns)[:30]}")
    tcol = next((c for c in TIME_COLUMN_CANDIDATES if c in raw_table.columns), None)
    if tcol in ("Timestamps", "Timestamp"):
        ts, clock = parse_timestamps(raw_table[tcol]), "file"
    else:                                              # Timestamp_Index or no clock at all: one row / index step = one minute from the chosen start
        idx = pd.to_numeric(raw_table[tcol], errors="coerce") if tcol else pd.Series(np.arange(len(raw_table), dtype=float), index=raw_table.index)
        ts, clock = pd.Timestamp(assumed_start_time) + pd.to_timedelta((idx - idx.min()).to_numpy(float), unit="m"), "assumed"
    df = raw_table[REQUIRED_COLS].apply(pd.to_numeric, errors="coerce").assign(Timestamps=pd.DatetimeIndex(ts))
    n_read = len(df)
    df = df.dropna(subset=REQUIRED_COLS + ["Timestamps"])
    df["Timestamps"] = df["Timestamps"].dt.floor("min")
    # one row per minute, oldest first
    df = df.drop_duplicates("Timestamps").sort_values("Timestamps")
    # tiny negative readings are noise -> 0; power factor must stay within -1..1
    df[PHASE_KW_COLS + CURRENT_COLS] = df[PHASE_KW_COLS + CURRENT_COLS].clip(lower=0); df["PF"] = df["PF"].clip(-1, 1)
    # remove impossible meter values
    df = df[(df[PHASE_KW_COLS] < 5e3).all(axis=1) & (df[VOLTAGE_COLS] < 1000).all(axis=1) & (df[CURRENT_COLS] < 1e4).all(axis=1)]
    # total kW = Red + Yellow + Blue
    df["KW"] = df[PHASE_KW_COLS].sum(axis=1)
    minute_rows = df[["Timestamps", "KW"] + PHASE_KW_COLS + CURRENT_COLS + VOLTAGE_COLS + ["PF"]].reset_index(drop=True)
    return minute_rows, dict(clock=clock, tcol=tcol, rows_read=n_read, rows_used=len(minute_rows))


def list_device_ids(raw_table):
    return sorted(raw_table["DeviceID"].dropna().unique().tolist()) if "DeviceID" in raw_table.columns and raw_table["DeviceID"].nunique() > 1 else []


@st.cache_data(show_spinner="Building the minute grid…")
def build_minute_grid(minute_rows, smooth):
    """D_raw = every minute of the period (NaN where the meter sent nothing); D = rolling mean of the PAST `smooth` minutes only."""
    # every minute of the period, so missing minutes show up as NaN
    grid = pd.date_range(minute_rows["Timestamps"].min(), minute_rows["Timestamps"].max(), freq="min")
    minute_grid_raw = minute_rows.drop(columns="KW").set_index("Timestamps").reindex(grid)
    minute_grid_raw["KW"] = minute_grid_raw[PHASE_KW_COLS].sum(axis=1, min_count=3)
    # rolling mean over PAST minutes only (no look-ahead)
    minute_grid_smoothed = minute_grid_raw.rolling(int(smooth), min_periods=1 if smooth <= 1 else max(2, int(smooth * 2 / 3))).mean()
    return minute_grid_raw, minute_grid_smoothed


@st.cache_data
def summarise_timeline(minute_rows, running_kw_threshold):
    # measured = meter sent a value; running = load above the threshold
    measured, running = minute_rows.KW.notna(), minute_rows.KW > running_kw_threshold
    span_min = int((minute_rows.Timestamps.max() - minute_rows.Timestamps.min()).total_seconds() // 60) + 1
    day = minute_rows.Timestamps.dt.date
    daily = pd.DataFrame({"day": day, "kwh": minute_rows.KW.fillna(0) / 60, "run_h": running / 60}).groupby("day").sum()
    daily["avg_run"] = minute_rows[running].groupby(day[running]).KW.mean().reindex(daily.index)
    # average kW for every day x hour (heat-map table)
    heat = minute_rows.assign(day=day, hour=minute_rows.Timestamps.dt.hour).pivot_table(index="day", columns="hour", values="KW", aggfunc="mean")
    heat = heat.reindex(index=[d.date() for d in pd.date_range(heat.index.min(), heat.index.max())], columns=range(24))   # every day x every hour; NaN = no data
    last_run = minute_rows.Timestamps[running].max() if running.any() else minute_rows.Timestamps.min(); last_seen = minute_rows.Timestamps[measured].max()
    return dict(energy=float(minute_rows.KW.fillna(0).sum() / 60), run_min=int(running.sum()), idle_min=int((measured & ~running).sum()),
                nodata_min=max(0, span_min - int(measured.sum())), peak=float(minute_rows.KW.max()), avg_run=float(minute_rows.KW[running].mean()) if running.any() else 0.0,
                last_run=last_run, last_seen=last_seen, daily=daily, heat=heat, util=float(running.sum() / max(measured.sum(), 1) * 100),
                now_running=bool(minute_rows.KW[measured].iloc[-1] > running_kw_threshold))

# ------------------------------------------------------------------ PHASE IMBALANCE MATHS (rules live in gridsense_rules.py)
def analyse_balance(phase_kw, alarm_limit_pct=None, min_phase_kw=None):
    """imbalance % = max(|R-avg|, |Y-avg|, |B-avg|) / avg x 100.  170 / 169 / 160 kW -> 3.81 %."""
    return gr.analyse_phase_balance(phase_kw, ALARM_LIMIT_PCT if alarm_limit_pct is None else alarm_limit_pct, MIN_PHASE_KW if min_phase_kw is None else min_phase_kw)


def plan_transfers(vals, names):
    return gr.plan_phase_transfers(vals, names)


@st.cache_data(show_spinner="Measuring the imbalance and looking for the cause…")
def diagnose_phase_imbalance(minute_frame, alarm_limit_pct, min_phase_kw):
    """Everything the 'Phase imbalance' tab shows: size, which colour, WHY (current / voltage / power factor), load-level behaviour, a move plan."""
    imbalance_table = analyse_balance(minute_frame[PHASE_KW_COLS].to_numpy(), alarm_limit_pct, min_phase_kw); imbalance_table.index = minute_frame.index
    valid_rows = imbalance_table.Loaded.to_numpy() & (minute_frame[CURRENT_COLS] > 0).all(axis=1).to_numpy() & (minute_frame[VOLTAGE_COLS] > 0).all(axis=1).to_numpy()
    result = dict(A=imbalance_table, n_all=len(minute_frame), n_loaded=int(imbalance_table.Loaded.sum()), n_ok=int(valid_rows.sum()), empty=int(valid_rows.sum()) < 20)
    if result["empty"]: return result
    valid_frame = minute_frame[valid_rows]; kw_values = valid_frame[PHASE_KW_COLS].to_numpy(float); current_values = valid_frame[CURRENT_COLS].to_numpy(float); voltage_values = valid_frame[VOLTAGE_COLS].to_numpy(float)
    # power factor of each phase = kW / (V x I)
    total_kw = kw_values.sum(axis=1); phase_power_factor = np.clip(kw_values * 1000 / (voltage_values * current_values), 0, 1.5)            # PF of each phase = kW / (V x I)
    row_numbers = np.arange(len(kw_values)); worst_phase_index = np.abs(kw_values - kw_values.mean(axis=1, keepdims=True)).argmax(axis=1)
    # worst phase against the average of the three, in %
    worst_deviation_pct = lambda X: (X[row_numbers, worst_phase_index] / X.mean(axis=1) - 1) * 100                  # the worst phase vs the average of the 3, in %
    # NEMA-style imbalance: largest deviation from the average, in %
    nema_imbalance_pct = lambda X: np.abs(X - X.mean(axis=1, keepdims=True)).max(axis=1) / X.mean(axis=1) * 100
    kw_deviation_pct, voltage_deviation_pct, current_deviation_pct, pf_deviation_pct = worst_deviation_pct(kw_values), worst_deviation_pct(voltage_values), worst_deviation_pct(current_values), worst_deviation_pct(phase_power_factor)
    kw_imbalance_pct, current_imbalance_pct, voltage_imbalance_pct = np.abs(kw_deviation_pct), nema_imbalance_pct(current_values), nema_imbalance_pct(voltage_values)
    # kW / (V x I) near 1 everywhere means the 'kW' columns are really kVA: no power-factor information
    pf_is_meaningful = float(np.median(phase_power_factor)) < 0.97                                      # kW / (V x I) near 1 everywhere = the "kW" columns are really V x I (kVA): no PF information in them
    # how strongly current, voltage and power factor deviate on the worst minutes
    cause_strength = pd.DataFrame({"Current": np.abs(current_deviation_pct), "Voltage": np.abs(voltage_deviation_pct), "Power factor": np.abs(pf_deviation_pct) if pf_is_meaningful else np.zeros(len(pf_deviation_pct))})
    cause_share_pct = (cause_strength.div(cause_strength.sum(axis=1).replace(0, np.nan), axis=0).mean() * 100).fillna(0)
    safe_correlation = lambda a, b: float(np.corrcoef(a, b)[0, 1]) if np.std(a) > 0 and np.std(b) > 0 else float("nan")
    worst_phase_counts = np.bincount(worst_phase_index, minlength=3); main_phase_index = int(worst_phase_counts.argmax()); other_phase_indexes = [i for i in range(3) if i != main_phase_index]
    per_phase_table = pd.DataFrame({"Worst phase in % of minutes": worst_phase_counts / len(worst_phase_index) * 100, "of those HIGH %": [np.mean(kw_deviation_pct[worst_phase_index == i] > 0) * 100 if worst_phase_counts[i] else np.nan for i in range(3)],
                        "Avg current A": current_values.mean(axis=0), "95 % of minutes below A": np.percentile(current_values, 95, axis=0), "Peak A": current_values.max(axis=0),
                        "Avg voltage V": voltage_values.mean(axis=0), "Avg kW": kw_values.mean(axis=0), "PF (kW / V·I)": phase_power_factor.mean(axis=0)}, index=PHASE_LABELS)
    main_extra_current_a = current_values[:, main_phase_index] - current_values[:, other_phase_indexes].mean(axis=1); main_extra_kw = kw_values[:, main_phase_index] - kw_values[:, other_phase_indexes].mean(axis=1)
    load_level_table = pd.DataFrame({"total": total_kw, "imb": kw_imbalance_pct, "extra_A": main_extra_current_a, "extra_kW": main_extra_kw}).groupby(pd.qcut(total_kw, 5, duplicates="drop"), observed=True).mean()
    load_level_table.index = [f"{iv.left:.0f}–{iv.right:.0f} kW" for iv in load_level_table.index]
    extra_current_by_level = load_level_table["extra_A"]; extra_current_is_constant = len(extra_current_by_level) > 2 and (extra_current_by_level.iloc[1:].max() - extra_current_by_level.iloc[1:].min()) / max(abs(extra_current_by_level.iloc[1:].mean()), 1e-9) < 0.25
    mean_current_a, mean_kw = current_values.mean(axis=0), kw_values.mean(axis=0)
    transfer_plan = pd.DataFrame([(f, t, a, a * mean_kw[PHASE_LABELS.index(f)] / mean_current_a[PHASE_LABELS.index(f)]) for f, t, a in plan_transfers(mean_current_a, PHASE_LABELS) if a >= 1.0], columns=["From", "To", "Move A", "Move kW"])
    # neutral current of a 4-wire supply (same PF angle on all phases)
    neutral_current_a = np.sqrt(np.maximum(0, (current_values ** 2).sum(axis=1) - current_values[:, 0] * current_values[:, 1] - current_values[:, 1] * current_values[:, 2] - current_values[:, 2] * current_values[:, 0]))      # neutral current of a 4-wire supply (same PF angle on all phases)
    # extra I²R heating against a perfectly balanced load
    extra_heating_pct = ((current_values ** 2).sum(axis=1) / (3 * current_values.mean(axis=1) ** 2) - 1) * 100                                                  # extra I²R heating against a perfectly balanced load
    imbalance_by_hour = pd.Series(kw_imbalance_pct, index=valid_frame.index).groupby(valid_frame.index.hour).mean()
    status_counts = imbalance_table.Status[imbalance_table.Loaded].value_counts().reindex(["OK", "ALARM"]).fillna(0)
    return dict(result, empty=False, mean_imb=float(kw_imbalance_pct.mean()), p95_imb=float(np.percentile(kw_imbalance_pct, 95)), pct_above=float((imbalance_table.Imbalance_pct[imbalance_table.Loaded] > alarm_limit_pct).mean() * 100),
                mean_I=float(current_imbalance_pct.mean()), mean_V=float(voltage_imbalance_pct.mean()), p95_V=float(np.percentile(voltage_imbalance_pct, 95)), corr_I=safe_correlation(kw_imbalance_pct, current_imbalance_pct), corr_V=safe_correlation(kw_imbalance_pct, voltage_imbalance_pct), share=cause_share_pct,
                per=per_phase_table, main=main_phase_index, main_name=PHASE_LABELS[main_phase_index], extra_A=float(main_extra_current_a.mean()), extra_kW=float(main_extra_kw.mean()), levels=load_level_table, const=bool(extra_current_is_constant), lvA_const=float(extra_current_by_level.iloc[1:].mean()),
                ctot=safe_correlation(kw_imbalance_pct, total_kw), plan=transfer_plan, Im=mean_current_a, Km=mean_kw, neutral_mean=float(neutral_current_a.mean()), neutral_p95=float(np.percentile(neutral_current_a, 95)), loss=float(extra_heating_pct.mean()),
                by_hour=imbalance_by_hour, status=status_counts, pf_ok=bool(pf_is_meaningful), pf_phase=float(phase_power_factor.mean()), pf_meter=float(valid_frame["PF"].mean()), high_share=float(per_phase_table.iloc[main_phase_index, 1]))

# ------------------------------------------------------------------ RULE-BASED ADVICE TEXT (no AI)
def classify_trend(first, last):
    """idle / flat / up / down - a change counts only if it is at least 1 kW and 5 % of the larger value."""
    top = max(abs(first), abs(last))
    if top < 1: return "idle"
    if abs(last - first) < max(1.0, 0.05 * top): return "flat"
    return "up" if last > first else "down"


def percent_change(first, last):
    return (last - first) / first * 100 if first >= 1 else float("nan")


def percent_change_text(p):
    return "" if pd.isna(p) else f" ({p:+.1f} %)"


CARD_TONE_COLORS = {"bad": ("#F43F5E", "#FFE4E8", "#BE123C"), "amber": ("#F59E0B", "#FFEBD5", "#C2410C"), "good": ("#10B981", "#D8F5E9", "#047857"), "info": ("#6366F1", "#E7E8FF", "#4338CA")}


def _status_badge_html(text, tone):
    tone_colors = CARD_TONE_COLORS[tone]; return f'<span style="background:{tone_colors[1]};color:{tone_colors[2]};border-radius:99px;padding:2px 10px;font-size:.72rem;font-weight:600;white-space:nowrap">{text}</span>'


def suggestion_card_html(title, badge, tone, items, headline="", numbered=True, border=None):
    """ONE white card: TITLE + badge, bold headline, then numbered items (bold heading, sentence below, thin line between).  Single line of HTML - Streamlit markdown must not see blank lines."""
    tone_colors = CARD_TONE_COLORS[border or tone]
    rows = "".join(f'<div style="padding:.7rem 0;border-top:1px solid #EEF0F5"><div style="font-weight:600;color:{TEXT_COLOR};font-size:.92rem;margin-bottom:.2rem">{f"{i}. " if numbered else ""}{h}</div><div style="color:#334155;font-size:.88rem;line-height:1.55">{t}</div></div>' for i, (h, t) in enumerate(items, 1))
    head = f'<div style="font-weight:700;color:{TEXT_COLOR};font-size:.95rem;margin:12px 0 8px">{headline}</div>' if headline else '<div style="height:10px"></div>'
    return f'<div style="border:1px solid {tone_colors[0]};border-radius:14px;padding:14px 20px 6px;background:#fff;margin-bottom:.8rem"><div style="display:flex;justify-content:space-between;align-items:center;gap:8px"><span style="font-weight:700;color:{TEXT_COLOR};font-size:.8rem;letter-spacing:.03em">{title}</span>{_status_badge_html(badge, tone)}</div>{head}{rows}</div>'


CARD_BLUE, CARD_BLUE_DARK, CARD_BLUE_TINT = "#2563EB", "#1D4ED8", "#DBEAFE"                  # the blue two-column advice block (load forecast + phase balance)


def _blue_badge_html(text):
    return f'<span style="background:{CARD_BLUE_TINT};color:{CARD_BLUE_DARK};border-radius:99px;padding:2px 10px;font-size:.72rem;font-weight:600;white-space:nowrap">{text}</span>'


def _blue_title_row_html(title, badge_text=None, badge_html=None):
    return f'<div style="display:flex;justify-content:space-between;align-items:center;gap:8px"><span style="font-weight:700;color:{CARD_BLUE_DARK};font-size:.85rem;letter-spacing:.03em">{title}</span>{badge_html or _blue_badge_html(badge_text)}</div>'


def _blue_item_rows_html(items, start=1):
    """Numbered rows: blue number + bold heading, one sentence below, thin blue line between."""
    return "".join(f'<div style="padding:.75rem 0;border-top:1px solid {CARD_BLUE_TINT}"><div style="font-weight:600;color:{TEXT_COLOR};font-size:.95rem;margin-bottom:.25rem"><span style="color:{CARD_BLUE}">{i}.</span> {h}</div><div style="color:#334155;font-size:.9rem;line-height:1.6">{t}</div></div>' for i, (h, t) in enumerate(items, start))


def _blue_card_html(left_html, right_html):
    """ONE blue-bordered block, the two parts side by side (they stack on a narrow screen)."""
    return (f'<div style="border:1.5px solid {CARD_BLUE};border-radius:14px;padding:16px 22px 8px;background:#fff;margin-bottom:.8rem;display:grid;grid-template-columns:repeat(auto-fit,minmax(340px,1fr));gap:0 2.2rem">'
            f'<div>{left_html}</div><div style="border-left:1px solid {CARD_BLUE_TINT};padding-left:2.2rem">{right_html}</div></div>')


def _machine_text():
    machine_no = next((p for p in Path(DATA_FILE_NAME).stem.split("_") if p.isdigit()), None)
    return f"Machine {machine_no}'s" if machine_no else "this machine's"


def forecast_advice_card_html(forecast_values, forecast_times, previous_energy_kwh, horizon_hours, badge, tone):
    """ONE card with the two reference sections: POTENTIAL RISK & COST IMPACT, then ENERGY MANAGEMENT RECOMMENDATIONS.
    The wording is fixed (copied from the reference screenshots); only the numbers come from the forecast: energy, change vs the previous period, peak kW and its time."""
    values = np.asarray(forecast_values, float); energy_kwh = float(values.sum())
    day_summary = daily_energy_summary(values, forecast_times); peak_day = day_summary["sum"].idxmax(); peak_day_kwh, peak_day_kw = float(day_summary.loc[peak_day, "sum"]), float(day_summary.loc[peak_day, "mean"]); mean_day_kwh = float(day_summary["sum"].mean())     # the PEAK DAY = the day with the most energy
    peak_day_text = f"{peak_day:%A, %d %b}"
    peak_day_vs_mean = "" if len(day_summary) < 2 or mean_day_kwh < 1 else f" That is {abs(peak_day_kwh / mean_day_kwh - 1) * 100:.0f}% {'above' if peak_day_kwh >= mean_day_kwh else 'below'} the daily average of {mean_day_kwh:,.0f} kWh."
    days = horizon_hours // 24 if horizon_hours % 24 == 0 else None
    period = (f"{days} day{'s' if days != 1 else ''}" if days else f"{horizon_hours} hours"); badge_period = (f"{days}-day" if days else f"{horizon_hours}-hour")
    change_pct = (energy_kwh - previous_energy_kwh) / previous_energy_kwh * 100 if previous_energy_kwh >= 1 else float("nan")
    change_sentence = "" if not np.isfinite(change_pct) else ", about the same as the previous period" if abs(change_pct) < 0.5 else f", approximately {abs(change_pct):.0f}% {'higher' if change_pct > 0 else 'lower'} than the previous period"
    machine_text = _machine_text()
    risks = [("Higher Electricity Procurement Cost", "If internal generation cannot meet the forecasted demand, additional grid electricity or backup generation may be required, potentially increasing operating costs."),
             ("Increased Peak Demand", f"Forecasted peak usage is on <b>{peak_day_text}</b>: <b>{peak_day_kwh:,.0f} kWh</b>, average {peak_day_kw:,.1f} kW. Without peak-load management, simultaneous machine operations may increase demand beyond available generation capacity."),
             ("Missed Energy Storage Opportunity", "If battery storage is available but not charged during surplus-generation periods, stored energy may not be available during peak-demand hours."),
             ("Increased Operational Risk", "If demand exceeds available power supply, the company may need backup supply or operational adjustments to avoid production disruption."),
             ("Higher Energy Management Cost", "Without demand and generation coordination, the company may miss opportunities to reduce unnecessary generation, grid imports and avoidable operating expenses.")]
    actions = [("Optimize Internal Power Generation", f"Forecast energy demand is {energy_kwh:,.0f} kWh over the next {period}{change_sentence}. Align captive power generation schedules with the expected demand to avoid unnecessary generation."),
               ("Prepare for Peak Demand", f"Peak usage is expected on <b>{peak_day_text}</b>, when the machine uses about <b>{peak_day_kwh:,.0f} kWh</b> (average {peak_day_kw:,.1f} kW).{peak_day_vs_mean} Keep sufficient generation capacity available on that day to handle the expected peak."),
               ("Optimize Energy Storage", "If battery storage is available, charge during low-demand periods or when internal generation exceeds consumption. Discharge during forecasted peak-demand periods."),
               ("Shift Flexible Loads", "Stagger large motor start-ups and move non-critical operations away from peak-demand hours to reduce simultaneous power demand."),
               ("Monitor Power Availability", "Compare forecasted demand with available internal generation. Plan grid support or backup capacity if a supply shortfall is expected.")]
    risk_part = _blue_title_row_html("POTENTIAL RISK &amp; COST IMPACT", badge) + f'<div style="color:{MUTED_COLOR};font-size:.78rem;margin:.5rem 0 .6rem">Based on {machine_text} {badge_period} demand forecast</div>' + _blue_item_rows_html(risks)
    action_part = _blue_title_row_html("ENERGY MANAGEMENT RECOMMENDATIONS", f"{badge_period} forecast") + f'<div style="font-weight:700;color:{TEXT_COLOR};font-size:.95rem;margin:.5rem 0 .6rem">Forecasted peak demand: {peak_day_kwh:,.0f} kWh on {peak_day_text} (average {peak_day_kw:,.1f} kW)</div>' + _blue_item_rows_html(actions)
    # all 10 sections in ONE block, the two parts side by side
    return _blue_card_html(risk_part, action_part)


def build_load_suggestions(forecast_values, forecast_times, previous_energy_kwh):
    """Short rule-based risks + actions for the forecast TOTAL kW.  Direction = forecast energy vs the previous period of the same length (> LOAD_CHG_PCT).
    Numbers come only from the forecast and the settings at the top of the file; a setting left at 0 is simply not mentioned.
    Returns dict(kind, badge, tone, headline, risks, actions, managed, ignored)."""
    forecast_values = np.asarray(forecast_values, float); n_hours = len(forecast_values); energy_kwh = float(forecast_values.sum()); average_kw = float(forecast_values.mean())
    peak, peak_time = float(forecast_values.max()), forecast_times[int(forecast_values.argmax())]; lowest_kw, lowest_time = float(forecast_values.min()), forecast_times[int(forecast_values.argmin())]
    energy_change_kwh = energy_kwh - previous_energy_kwh; energy_change_pct = energy_change_kwh / previous_energy_kwh * 100 if previous_energy_kwh >= 1 else float("nan")
    # direction = forecast energy vs the previous period of the same length
    trend_direction = "idle" if average_kw < MIN_PHASE_KW else "up" if energy_change_pct > LOAD_CHANGE_PCT else "down" if energy_change_pct < -LOAD_CHANGE_PCT else "stable"
    change_text = "" if not np.isfinite(energy_change_pct) else f"{abs(energy_change_pct):.0f} % {'above' if energy_change_pct > 0 else 'below'} the previous {hours_to_text(n_hours)}" if abs(energy_change_pct) >= 0.5 else f"same as the previous {hours_to_text(n_hours)}"
    format_money = lambda x: f"{x:,.0f}{(' ' + CURRENCY) if CURRENCY else ''}"
    shortfall_kw = max(peak - OWN_GENERATION_KW, 0.0) if OWN_GENERATION_KW > 0 else 0.0; spare_kw = max(OWN_GENERATION_KW - peak, 0.0) if OWN_GENERATION_KW > 0 else 0.0
    over_demand_kw = max(peak - SANCTIONED_DEMAND_KW, 0.0) if SANCTIONED_DEMAND_KW > 0 else 0.0; demand_use_pct = peak / SANCTIONED_DEMAND_KW * 100 if SANCTIONED_DEMAND_KW > 0 else float("nan")
    peak_cut_kw = peak * SHIFTABLE_LOAD_PCT / 100; extra_cost = max(energy_change_kwh, 0) * TARIFF_PER_KWH + over_demand_kw * DEMAND_CHARGE_PER_KW; unused_kwh = max(-energy_change_kwh, 0.0)
    format_time = lambda ts: f"{ts:%a %H:%M}"; capacity_gap_kw = max(shortfall_kw, over_demand_kw)
    if trend_direction == "idle":
        risks = [("Machine expected idle", f"Average {average_kw:,.1f} kW (below {MIN_PHASE_KW:g} kW). If it should be running, check supply, start-up and process."), ("Idle fixed charges", "Demand and fixed charges are still billed with no output.")]
        actions = [("Confirm the stop is planned", "If yes, switch off auxiliaries and use the time for maintenance."), ("Look for a fault", "Tripped breaker, supply problem, sensor or data problem.")]
        if_managed_text, if_ignored_text = "A planned stop saves energy and gives a maintenance window.", "An unplanned stop means lost production and fixed charges with no output."
    elif trend_direction == "up":
        risks = [("Higher electricity cost", f"{energy_kwh:,.0f} kWh needed, {change_text}. " + (f"Extra grid or backup power may cost about {format_money(max(energy_change_kwh, 0) * TARIFF_PER_KWH)}." if TARIFF_PER_KWH > 0 else "Extra grid or backup power may be needed.")),
                 ("Increased peak demand", f"Peak {peak:,.1f} kW around {format_time(peak_time)}" + (f", {shortfall_kw:,.1f} kW above own generation" if shortfall_kw else "") + (f", {over_demand_kw:,.1f} kW above sanctioned demand (penalty risk)" if over_demand_kw else f", {demand_use_pct:.0f} % of sanctioned demand" if SANCTIONED_DEMAND_KW > 0 else "") + "."),
                 ("Missed energy storage" if HAS_BATTERY else "Missed load shifting", f"Battery not charged around {format_time(lowest_time)} may be empty at the peak." if HAS_BATTERY else f"Flexible work left in the peak wastes the {peak - lowest_kw:,.1f} kW gap to the lowest hour."),
                 ("Increased operational risk", f"Peak exceeds available power by {capacity_gap_kw:,.1f} kW; backup or reduced operation may be needed." if capacity_gap_kw else "If demand exceeds supply, production may stop."),
                 ("Higher energy management cost", f"Uncoordinated demand may add about {format_money(extra_cost)}." if extra_cost > 0 else "Unmanaged peaks and grid imports raise avoidable cost.")]
        actions = [("Optimize internal generation", f"Forecast energy demand is {energy_kwh:,.0f} kWh over the next {hours_to_text(n_hours)}, {change_text}. Plan own generation for it and arrange grid or backup for the rest."),
                ("Prepare for peak demand", f"Keep {peak:,.1f} kW capacity ready around {format_time(peak_time)}" + (f" (shortfall {shortfall_kw:,.1f} kW)." if shortfall_kw else f" (spare {spare_kw:,.1f} kW)." if spare_kw else ".")),
                ("Optimize energy storage" if HAS_BATTERY else "Use the low-load hours", f"Charge the battery around {format_time(lowest_time)}; discharge at the peak." if HAS_BATTERY else f"Run flexible work around {format_time(lowest_time)} ({lowest_kw:,.1f} kW), not at the peak."),
                ("Shift flexible loads", f"Stagger motor starts away from {format_time(peak_time)}; moving {SHIFTABLE_LOAD_PCT:g} % cuts the peak by about {peak_cut_kw:,.1f} kW."),
                ("Monitor power availability", "Compare the forecast with available generation; plan backup if short.")]
        if_managed_text = f"Peak can drop by about {peak_cut_kw:,.1f} kW" + (f", saving about {format_money(peak_cut_kw * DEMAND_CHARGE_PER_KW)}" if DEMAND_CHARGE_PER_KW > 0 else "") + "; backup supply and stoppages are avoided."
        if_ignored_text = (f"About {format_money(extra_cost)} extra cost" if extra_cost > 0 else "Extra grid or backup cost") + (f", peak {capacity_gap_kw:,.1f} kW over capacity" if capacity_gap_kw else "") + ", and possible production disruption."
    elif trend_direction == "down":
        risks = [("Under-utilised generation", f"{energy_kwh:,.0f} kWh needed, {change_text}; up to {unused_kwh:,.0f} kWh of usual generation may go unused" + (f" (about {format_money(unused_kwh * TARIFF_PER_KWH)})." if TARIFF_PER_KWH > 0 else ".")),
                 ("Wasted fuel", f"Peak {peak:,.1f} kW leaves {spare_kw:,.1f} kW of own generation idle." if spare_kw else f"Generation sized for a higher load runs inefficiently (peak only {peak:,.1f} kW)."),
                 ("Low-load inefficiency", f"Lowest hour {lowest_kw:,.1f} kW around {format_time(lowest_time)}; motors and generators are least efficient at low load."),
                 ("Possible production slowdown", "Check the drop is planned, not a stoppage, fault or data problem."),
                 ("Fixed-cost pressure", f"Contract demand {SANCTIONED_DEMAND_KW:g} kW is billed while the peak uses {demand_use_pct:.0f} %." if SANCTIONED_DEMAND_KW > 0 else "Fixed and demand charges stay the same, so cost per kWh rises.")]
        actions = [("Right-size generation", f"Forecast energy demand is {energy_kwh:,.0f} kWh over the next {hours_to_text(n_hours)}, {change_text}. Run fewer units or a lower setpoint to match it."),
                ("Use the surplus", "Charge the battery when generation exceeds demand." if HAS_BATTERY else f"Move flexible work into the low hours around {format_time(lowest_time)}."),
                ("Consolidate loads", f"Run machines together; switch off idle equipment around {format_time(lowest_time)}."),
                ("Review contract demand", f"Peak is {demand_use_pct:.0f} % of contract demand; ask if it can be lowered." if SANCTIONED_DEMAND_KW > 0 and demand_use_pct < 80 else "If the peak stays well below contract demand, a lower contract may cut fixed charges."),
                ("Check why demand fell", "Maintenance, planned shutdown, sensor or hidden machine fault.")]
        if_managed_text = f"Matching generation to demand avoids up to {unused_kwh:,.0f} kWh" + (f" (about {format_money(unused_kwh * TARIFF_PER_KWH)})" if TARIFF_PER_KWH > 0 else "") + " of unnecessary generation and fuel."
        if_ignored_text = f"Up to {unused_kwh:,.0f} kWh" + (f" (about {format_money(unused_kwh * TARIFF_PER_KWH)})" if TARIFF_PER_KWH > 0 else "") + " wasted, fixed charges on unused capacity, and a hidden fault may go unnoticed."
    else:
        risks = [("Demand is steady", f"{energy_kwh:,.0f} kWh needed, {change_text or 'no major change'}."), ("Peak to watch", f"{peak:,.1f} kW around {format_time(peak_time)}" + (f" ({demand_use_pct:.0f} % of sanctioned demand)." if SANCTIONED_DEMAND_KW > 0 else ".")), ("Low-load hours", f"Lowest {lowest_kw:,.1f} kW around {format_time(lowest_time)}.")]
        actions = [("Keep the current plan", f"Forecast energy demand is {energy_kwh:,.0f} kWh over the next {hours_to_text(n_hours)}, {change_text or 'no major change'}. Generation and storage settings can stay as they are."), ("Prepare for the peak", f"Keep {peak:,.1f} kW ready around {format_time(peak_time)}; stagger motor starts."), ("Keep monitoring", "The forecast updates as new readings arrive.")]
        if_managed_text, if_ignored_text = "Staggered motor starts keep the peak low and the bill steady.", "A coincident start-up can still create a costly peak."
    # a drop of more than 10 % needs attention
    big_drop = trend_direction == "down" and np.isfinite(energy_change_pct) and energy_change_pct < -10
    tone = "bad" if capacity_gap_kw else "amber" if trend_direction in ("up", "down") else "good"
    return dict(kind=trend_direction, tone=tone, badge="Attention" if capacity_gap_kw or big_drop else "Watch" if trend_direction in ("up", "down") else "Normal", risks=risks, actions=actions, managed=if_managed_text, ignored=if_ignored_text, headline=f"Forecasted peak demand: {peak:,.1f} kW around {peak_time:%A, %H:%M}")


def phase_analysis_card_html(forecast_phase_kw, forecast_table, diagnosis, quiet_busy_window=None, phase_low_kw=None, phase_high_kw=None):
    """'3-phase load analysis' in the SAME blue two-column block as the load forecast.
    LEFT  : phase-wise load distribution (forecast average per colour + interval), likely cause, potential impact if not corrected.
    RIGHT : phase balance recommendations (inspect / move loads / verify).   Forecast average per colour + measured causes."""
    forecast_avg_by_phase = forecast_phase_kw.mean(); overall_avg_kw = float(forecast_avg_by_phase.mean()); deviation_pct = (forecast_avg_by_phase / overall_avg_kw - 1) * 100 if overall_avg_kw >= 1 else forecast_avg_by_phase * 0
    n_alarm_hours = int(forecast_table.Alarm.sum()); worst_deviation_pct = float(deviation_pct.abs().max())
    is_out_of_balance = worst_deviation_pct > ALARM_LIMIT_PCT or n_alarm_hours > 0; tone = "amber" if is_out_of_balance else "good"
    badge = "Imbalance detected" if is_out_of_balance else "Balanced"
    n_hours = len(forecast_phase_kw); n_days = n_hours // 24 if n_hours % 24 == 0 else None; badge_period = f"{n_days}-day" if n_days else f"{n_hours}-hour"
    label_of = dict(zip(forecast_phase_kw.columns, PHASE_LABELS))
    phase_tiles_html = ""
    for p, nm_ in zip(forecast_phase_kw.columns, PHASE_LABELS):
        d = float(deviation_pct[p]); off = abs(d) > ALARM_LIMIT_PCT; lab = "Overloaded" if d > ALARM_LIMIT_PCT else "Underloaded" if d < -ALARM_LIMIT_PCT else "Near average"
        col = CARD_TONE_COLORS["bad"][2] if off else TEXT_COLOR
        interval_html = "" if phase_low_kw is None or phase_high_kw is None else f'<div style="font-size:.72rem;color:{MUTED_COLOR};margin-top:2px">{CONFIDENCE_TEXT} range {float(phase_low_kw[p].mean()):,.1f} – {float(phase_high_kw[p].mean()):,.1f} kW</div>'
        phase_tiles_html += (f'<div style="flex:1;min-width:105px"><div style="font-size:.72rem;color:{MUTED_COLOR};letter-spacing:.04em">{nm_.upper()}</div><div style="font-size:1.4rem;font-weight:700;color:{col}">{forecast_avg_by_phase[p]:,.1f} kW</div>'
                  f'<div style="font-size:.78rem;color:{col if off else MUTED_COLOR}">{"↑" if d >= 0 else "↓"} {abs(d):.1f} % · {lab}</div>{interval_html}</div>')
    section_title_html = lambda t: f'<div style="font-weight:700;color:{TEXT_COLOR};font-size:.9rem;margin:.9rem 0 .35rem">{t}</div>'
    high_phases = [nm for p, nm in zip(forecast_phase_kw.columns, PHASE_LABELS) if deviation_pct[p] > ALARM_LIMIT_PCT] or [PHASE_LABELS[int(np.argmax(forecast_avg_by_phase.to_numpy()))]]; low_phases = [nm for p, nm in zip(forecast_phase_kw.columns, PHASE_LABELS) if deviation_pct[p] < -ALARM_LIMIT_PCT] or [PHASE_LABELS[int(np.argmin(forecast_avg_by_phase.to_numpy()))]]
    cause_html, top_cause = "", None
    if diagnosis:
        top_cause = diagnosis["share"].idxmax(); current_flagged, voltage_flagged = diagnosis["mean_I"] > ALARM_LIMIT_PCT, diagnosis["mean_V"] > ALARM_LIMIT_PCT
        cause_row_html = lambda lab, v, flag: f'<div style="display:flex;justify-content:space-between;padding:3px 0;color:#475569;font-size:.88rem"><span>{lab} ({v:.1f} %)</span>{_status_badge_html(("Above " if flag else "Within ") + f"{ALARM_LIMIT_PCT:g} %", "bad" if flag else "good")}</div>'
        cause_html = (section_title_html("Likely cause analysis") + cause_row_html("Current imbalance", diagnosis["mean_I"], current_flagged) + cause_row_html("Voltage imbalance", diagnosis["mean_V"], voltage_flagged)
                 + f'<div style="font-weight:700;color:{TEXT_COLOR};font-size:.9rem;margin-top:6px">Possible cause: ' + {"Current": "Uneven load distribution", "Voltage": "Supply voltage unbalance", "Power factor": "Unequal power factor between phases"}[top_cause] + "</div>")
    # RIGHT side: recommendations as (heading, sentence)
    if not is_out_of_balance:
        actions, impacts = [("No action needed", f"The colours are within {ALARM_LIMIT_PCT:g} %."), ("Keep the load spread", "Keep new single-phase loads spread across R, Y and B.")], []
    else:
        if top_cause == "Voltage": actions = [("Check the supply voltage", "Ask the utility to check the supply voltage and transformer tap."), ("Hold off on rewiring", "Do not rewire before the voltage is checked."), ("Re-check the voltages", "Re-check the phase voltages afterwards.")]
        else:
            move_targets = " and ".join(f"{r['To']} (about {r['Move A']:.0f} A)" for _, r in diagnosis["plan"].iterrows()) if diagnosis and len(diagnosis["plan"]) else " / ".join(low_phases)
            source_phase = diagnosis["plan"].iloc[0]["From"] if diagnosis and len(diagnosis["plan"]) else " / ".join(high_phases)
            actions = [("Inspect single-phase loads", f"Inspect single-phase loads on {' / '.join(high_phases)}."),
                       ("Move suitable loads", f"Move suitable loads from {source_phase} to {move_targets}" + (f", best done {quiet_busy_window['quiet'][0]:%a %d %b %H:%M} → {quiet_busy_window['quiet'][1]:%H:%M} (lowest load)." if quiet_busy_window else ".")),
                       ("Verify phase currents", "Verify phase currents after one working day.")]
        extra_heating_pct = diagnosis["loss"] if diagnosis else float(((forecast_phase_kw ** 2).sum(axis=1) / (3 * forecast_phase_kw.mean(axis=1) ** 2).replace(0, np.nan) - 1).mean() * 100)
        impacts = [("Increased cable losses and heating", f"About {max(extra_heating_pct, 0):.1f} % extra I²R heat on {high_phases[0]}."),
                   ("Higher neutral current and equipment stress", f"Neutral current is about {diagnosis['neutral_mean']:.0f} A." if diagnosis and diagnosis["neutral_mean"] >= 1 else "The neutral carries the unbalanced current."),
                   ("Increased risk of overheating and unwanted trips", f"{high_phases[0]} peaks at {diagnosis['per']['Peak A'].max() / PHASE_RATING_A * 100:.0f} % of the {PHASE_RATING_A:,.0f} A rating." if diagnosis and PHASE_RATING_A > 0 else "A sustained imbalance heats cables and can trip protection.")]
    worst_p = deviation_pct.idxmax(); headline = (f"Forecasted imbalance: {label_of[worst_p]} {deviation_pct[worst_p]:+.1f} % vs the 3-phase average" if is_out_of_balance else "Forecasted load is balanced across Red, Yellow and Blue")
    left_html = (_blue_title_row_html("3-PHASE LOAD ANALYSIS", badge_html=_status_badge_html(badge, tone)) + f'<div style="color:{MUTED_COLOR};font-size:.78rem;margin:.5rem 0 .2rem">Based on {_machine_text()} {badge_period} phase forecast</div>'
                 + section_title_html("Phase-wise load distribution") + f'<div style="display:flex;gap:12px;flex-wrap:wrap">{phase_tiles_html}</div>' + cause_html
                 + (section_title_html("Potential impact if not corrected") + _blue_item_rows_html(impacts) if impacts else ""))
    right_html = _blue_title_row_html("PHASE BALANCE RECOMMENDATIONS", f"{badge_period} forecast") + f'<div style="font-weight:700;color:{TEXT_COLOR};font-size:.95rem;margin:.5rem 0 .6rem">{headline}</div>' + _blue_item_rows_html(actions)
    return _blue_card_html(left_html, right_html)

# ------------------------------------------------------------------ FORECAST HELPERS (cached wrappers around gridsense_forecast.py)
def hours_to_text(hours):
    hours = int(hours); return f"{hours} h" if hours < 48 else f"{hours / 24:.0f} days"


def average_in_blocks(values, first_time, hours):
    """Average of the hourly forecast over consecutive blocks of `hours` (the last block may be shorter), with the change from block to block."""
    v = pd.Series(np.asarray(values, float), index=pd.date_range(first_time, periods=len(values), freq="h"))
    rows = [(v.index[i], v.index[min(i + hours, len(v)) - 1] + pd.Timedelta(hours=1), float(v.iloc[i:i + hours].mean()), len(v.iloc[i:i + hours])) for i in range(0, len(v), hours)]
    out = pd.DataFrame(rows, columns=["Start", "End", "Avg kW", "Hours"]); out["Change kW"] = out["Avg kW"].diff(); prev = out["Avg kW"].shift(1)
    out["Change %"] = np.where(prev >= 1, out["Change kW"] / prev * 100, np.nan)
    return out


def forecast_interval_sigma(holdout_result, filled_hours, steps=MAX_FORECAST_HOURS):
    """Typical forecast error (kW) for every hour ahead, taken from the 7-day test: RMSE per day ahead, smoothed between the days and never shrinking.
    Without a test result: week-on-week change of the hourly load is used instead."""
    if holdout_result is not None:
        err = (holdout_result["actual"] - holdout_result["pred"]).to_numpy(float); centres, rmses = [], []
        for start in range(0, len(err), 24):
            block = err[start:start + 24]; block = block[np.isfinite(block)]
            if len(block) >= 6: centres.append(start + min(24, len(err) - start) / 2); rmses.append(float(np.sqrt(np.mean(block ** 2))))
        if centres: return np.maximum.accumulate(np.interp(np.arange(steps), centres, rmses))
    weekly_change = (filled_hours - filled_hours.shift(168)).dropna()
    sigma = float(weekly_change.std() / np.sqrt(2)) if len(weekly_change) >= 24 else float(filled_hours.diff().dropna().std())
    return np.full(steps, 0.0 if not np.isfinite(sigma) else sigma)


def daily_energy_summary(values, times):
    """Energy (kWh = sum of hourly kW) and average kW of every forecast DAY; only full 24 h days when there are any, so a partial first / last day never skews it."""
    series = pd.Series(np.asarray(values, float), index=pd.DatetimeIndex(times)); days = series.groupby(series.index.normalize()).agg(["sum", "mean", "count"])
    return days[days["count"] == 24] if (days["count"] == 24).any() else days


def block_time_label(row):
    return f"{row.Start:%d %b %H:%M} → {row.End:%H:%M}"


@st.cache_data(show_spinner=False)
def fill_missing_hours(measured_hours, max_linear_gap_hours=3, max_neighbour_days=7):
    """Fill missing hours so lines have no gaps. Gap of up to `short` hours: straight line. Longer gap: average of the SAME HOUR on the nearest
    valid day before and after it (within +/- max_days), so the daily shape is kept and the level follows the neighbouring days."""
    filled = measured_hours.interpolate(method="time", limit=max_linear_gap_hours, limit_area="inside")
    for missing_time in filled.index[filled.isna().any(axis=1)]:
        for column in filled.columns[filled.loc[missing_time].isna()]:
            neighbour_values = []
            for direction in (-1, 1):
                for days_away in range(1, max_neighbour_days + 1):
                    neighbour_time = missing_time + direction * pd.Timedelta(days=days_away)
                    if neighbour_time in measured_hours.index and pd.notna(measured_hours.at[neighbour_time, column]): neighbour_values.append(measured_hours.at[neighbour_time, column]); break
            if neighbour_values: filled.at[missing_time, column] = float(np.mean(neighbour_values))
    return filled.ffill().bfill()


@st.cache_data(show_spinner="Building the hourly series…")
def load_hourly_frame(minute_table, min_readings_per_hour):
    return gf.build_hourly_frame(minute_table, ["KW"] + PHASE_KW_COLS, min_readings_per_hour)


@st.cache_data(show_spinner="Training the model and forecasting the total load…")
def forecast_total_load(training_values, steps_ahead, model_name, season_length, fixed_smoothing):
    return gf.forecast_one_series(training_values.to_numpy(), steps_ahead, model_name, season_length, fixed_smoothing)


@st.cache_data(show_spinner="Testing the model on the last 168 hours…")
def run_total_load_holdout(total_measured_hourly, total_filled_hourly, model_name, season_length, fixed_smoothing, max_training_hours):
    return gf.holdout_test_total_load(total_measured_hourly, total_filled_hourly, model_name, season_length, fixed_smoothing, max_training_hours)


@st.cache_data(show_spinner="Training the model on Red, Yellow and Blue ")
def forecast_phases_cached(history_window, steps_ahead, model_name, season_length, method=gf.PHASE_FORECAST_METHODS[0]):
    return gf.forecast_three_phases(history_window, steps_ahead, model_name, season_length, None, method)


@st.cache_data(show_spinner="Testing the phase forecast on past weeks ")
def backtest_phases_cached(measured_phase_hours, filled_phase_hours, model_name, season_length, max_training_hours, n_weeks, method, alarm_limit_pct, min_phase_kw):
    return gf.backtest_phase_forecast(measured_phase_hours, filled_phase_hours, model_name, season_length, max_training_hours, 168, n_weeks, lambda phase_kw: analyse_balance(phase_kw, alarm_limit_pct, min_phase_kw), None, method)

# ------------------------------------------------------------------ PHASE PARAMETER HELPERS (load / current / voltage per colour)
def build_phase_parameter_frame(minute_rows, running_kw_threshold, rule="1h", min_readings_per_hour=10):
    """Loaded minutes only, averaged per hour / day. A period with too few loaded minutes is NaN (lines break)."""
    d = minute_rows[minute_rows["KW"] > running_kw_threshold].set_index("Timestamps")
    cols = [c for v in PHASE_PARAMETER_COLS.values() for c in v]
    g = d[cols].resample(rule)
    return g.mean().where(g[cols[0]].count() >= min_readings_per_hour)


def deviation_from_average_pct(phase_table, cols):
    X = phase_table[cols].to_numpy(float); m = X.mean(axis=1, keepdims=True); ok = np.isfinite(m) & (m > 0)
    return pd.DataFrame(np.where(ok, (X / np.where(ok, m, 1.0) - 1) * 100, np.nan), index=phase_table.index, columns=PHASE_LABELS)


def imbalance_pct_of(phase_table, cols):
    return deviation_from_average_pct(phase_table, cols).abs().max(axis=1)


def split_into_halves(phase_table):
    G = phase_table.dropna(); h = len(G) // 2
    return (None, None) if h < 2 else (G.iloc[:h], G.iloc[h:])


def half_over_half_change(phase_table):
    """Change of every phase / parameter, first half vs second half of the window."""
    a, b = split_into_halves(phase_table)
    if a is None: return None
    chg = pd.DataFrame(index=PHASE_LABELS, columns=list(PHASE_PARAMETER_COLS), dtype=float)
    for p, cols in PHASE_PARAMETER_COLS.items():
        for n, c in zip(PHASE_LABELS, cols):
            x0, x1 = a[c].mean(), b[c].mean(); chg.loc[n, p] = (x1 / x0 - 1) * 100 if x0 else np.nan
    txt = pd.DataFrame(index=PHASE_LABELS, columns=list(PHASE_PARAMETER_COLS), dtype=object)
    for p in PHASE_PARAMETER_COLS:
        txt[p] = [("–" if pd.isna(v) else ("▲" if v > TREND_TOLERANCE_PCT[p] else "▼" if v < -TREND_TOLERANCE_PCT[p] else "●") + f" {v:+.1f} %") for v in chg[p]]
    return chg, txt


def explain_imbalance_change(phase_table):
    """Rule-based reading: how the load imbalance moved, which parameter moved with it, which phase is responsible."""
    a, b = split_into_halves(phase_table)
    if a is None: return ["Not enough loaded hours in this window to explain a change."]
    imb = {p: (imbalance_pct_of(a, c).mean(), imbalance_pct_of(b, c).mean()) for p, c in PHASE_PARAMETER_COLS.items()}
    (k0, k1), (i0, i1), (v0, v1) = imb["Load (kW)"], imb["Current (A)"], imb["Voltage (V)"]; dk = k1 - k0
    out = [f"<b>Load imbalance</b> went from {k0:.1f} % to {k1:.1f} % ({dk:+.1f} points). Current imbalance: {i0:.1f} → {i1:.1f} %. Voltage imbalance: {v0:.1f} → {v1:.1f} %."]
    G = phase_table.dropna()
    if len(G) >= 10:
        k_, i_, v_ = (imbalance_pct_of(G, PHASE_PARAMETER_COLS[p]) for p in PHASE_PARAMETER_COLS)
        r = lambda x, y: float(np.corrcoef(x, y)[0, 1]) if np.std(x) > 0 and np.std(y) > 0 else float("nan")
        out.append(f"Over the whole window the load imbalance follows the <b>current</b> imbalance (r = {r(k_, i_):.2f}) and the <b>voltage</b> imbalance (r = {r(k_, v_):.2f}).")
    if abs(dk) < 0.5:
        out.append("The imbalance is <b>stable</b> in this window - no parameter is pushing it up or down."); return out
    sgn = 1 if dk > 0 else -1; s_i, s_v = (i1 - i0) * sgn, (v1 - v0) * sgn
    if max(s_i, s_v) < 0.2:
        out.append("Neither the current nor the voltage imbalance moved the same way, so they do not explain the change."); return out
    drv = "Current (A)" if s_i >= s_v else "Voltage (V)"; word = drv.split(" ")[0].lower()
    out.append(f"The imbalance is {'rising' if dk > 0 else 'falling'} <b>with {word}</b>: " + ("a load problem - the phases draw unequal current." if word == "current" else "a supply problem - the voltage between phases is changing; ask the utility."))
    d0, d1 = deviation_from_average_pct(a, PHASE_PARAMETER_COLS[drv]).mean(), deviation_from_average_pct(b, PHASE_PARAMETER_COLS[drv]).mean(); mv = (d1 - d0).abs().idxmax()
    out.append(f"Most responsible phase: <b>{mv}</b> - its {word} moved from {d0[mv]:+.1f} % to {d1[mv]:+.1f} % against the three-phase average.")
    return out


@st.cache_data(show_spinner="Building the load / current / voltage table…")
def load_phase_parameter_frame(minute_table, running_kw_threshold, rule, min_readings_per_hour):
    return build_phase_parameter_frame(minute_table, running_kw_threshold, rule, min_readings_per_hour)


def show_not_enough_data_note():
    note_box("<b>Not enough hourly data.</b> The forecasts need at least 3 days of hours that each have enough readings. "
         "For the weekly cycle and the backtest, about 5 weeks are better.", "amber")

# ------------------------------------------------------------------ UNUSED HELPERS (kept for reference, the page does not call them today)
def break_lines_at_gaps(t, *arrays, step=1):
    """Break plot lines wherever the timestamp jumps by more than `step` minutes (1 for full data; the thinning step when rows were skipped)."""
    t = pd.DatetimeIndex(t).values; brk = np.where(np.diff(t) > np.timedelta64(step, "m"))[0] + 1
    if not len(brk): return (t, *[np.asarray(a, float) for a in arrays])
    return (np.insert(t, brk, t[brk - 1] + np.timedelta64(30, "s")), *[np.insert(np.asarray(a, float), brk, np.nan) for a in arrays])


def phase_load_bars(values, scale):
    out = ""
    for ph, v in zip(PHASE_KW_COLS, values):
        w = 0 if pd.isna(v) else min(100, max(0, v / scale * 100))
        out += (f'<div class="bar"><div class="row"><span>{PHASE_LABEL_BY_COL[ph]} phase</span><b>{"–" if pd.isna(v) else f"{v:,.0f} kW"}</b></div>'
                f'<div class="trk"><div class="fil" style="width:{w:.0f}%;background:{PHASE_COLORS[ph]}"></div></div></div>')
    return out


def advice_to_html(head, actions):
    return f"<b>{head}</b><ul>" + "".join(f"<li>{a}</li>" for a in actions) + "</ul>"


def power_factor_advice(avg, pf, target=0.95):
    """Power factor advice from the actual average power factor and the forecast average load."""
    if not (pf and np.isfinite(pf) and 0.3 < pf < 1.0 and avg and avg >= 1): return None
    if pf >= target: return f"Power factor is already about <b>{pf:.2f}</b>: keep it, and check that capacitor banks do not over-compensate when the load drops."
    q = avg * (np.tan(np.arccos(pf)) - np.tan(np.arccos(target)))
    return (f"Power factor is about <b>{pf:.2f}</b>. Raising it to {target:.2f} at the forecast average of {avg:,.0f} kW needs roughly <b>{q:,.0f} kVAR</b> of capacitors "
            f"and cuts the kVA demand by about {(1 - pf / target) * 100:.0f} %. Many utilities charge below about 0.90-0.95 - check your tariff.")


def total_power_advice(trend_direction, first, last, peak, long_horizon, ctx=None):
    """What to do when the total power requirement is rising / falling / steady, using the actual numbers. Returns (note tone, headline, actions)."""
    ctx = ctx or {}; avg, pf, demand, slope = ctx.get("avg"), ctx.get("pf"), ctx.get("demand", 0.0), ctx.get("slope", float("nan"))
    p = percent_change_text(percent_change(first, last)); acts = []
    if trend_direction == "up":
        tone, head = "amber", f"Power requirement is rising: {first:,.1f} → {last:,.1f} kW{p}"
        if np.isfinite(slope): acts.append(f"The load is climbing at about <b>{slope:+,.1f} kW per hour</b>; the expected peak is <b>{peak:,.1f} kW</b>" + (f" around {ctx['peak_at']:%H:%M}." if ctx.get("peak_at") is not None else "."))
    elif trend_direction == "down":
        tone, head = "good", f"Power requirement is falling: {first:,.1f} → {last:,.1f} kW{p}"
        if np.isfinite(slope): acts.append(f"The load is dropping at about <b>{slope:+,.1f} kW per hour</b>.")
    elif trend_direction == "idle":
        return "", f"The machine is expected to stay idle (below 1 kW): {first:,.1f} → {last:,.1f} kW", ["No extra power is needed in this window.", "If the machine should be running now, check the supply, the start-up and the process."]
    else:
        tone, head = "", f"Power requirement is steady: {first:,.1f} → {last:,.1f} kW{p}"
        acts.append("No action needed - the change is within 5 %.")
    if demand > 0:
        use, room = peak / demand * 100, demand - peak
        acts.append(f"<b>Sanctioned demand {demand:,.0f} kW:</b> the expected peak uses <b>{use:.0f} %</b> of it (headroom {room:+,.0f} kW). "
                    + ("It is OVER the demand - shed or defer load before that time to avoid a penalty." if use > 100 else "Very little headroom - stagger motor starts and defer batch loads." if use > 90 else "Comfortable headroom."))
    elif trend_direction == "up": acts.append(f"Compare the expected peak (<b>{peak:,.1f} kW</b>) with your sanctioned demand - set SANCTIONED_DEMAND_KW at the top of app.py for an automatic check.")
    if pf and np.isfinite(pf) and pf > 0.3 and peak >= 1: acts.append(f"At power factor {pf:.2f} the peak of {peak:,.1f} kW is about <b>{peak / pf:,.0f} kVA</b> - compare it with the transformer / generator rating.")
    pa = power_factor_advice(avg, pf)
    if pa: acts.append(pa)
    if trend_direction == "up": acts += ["Stagger the start-up of large motors and move deferrable loads (batch jobs, pumps, charging) to off-peak hours.", "If the rise continues in the next forecast, plan extra capacity or a load-shedding sequence for non-critical equipment."]
    elif trend_direction == "down": acts += ["Switch off or park idle auxiliaries and standby equipment; use the low-load window for maintenance.", "If the machine should be producing, check for a stoppage or a process problem."]
    else: acts.append("Keep monitoring; the forecast updates as new readings arrive.")
    if long_horizon: acts.append("Exponential Smoothing repeats the daily / weekly pattern it has learned: the far end of the forecast is a typical day, not a prediction of one specific event - use it as a guide.")
    return tone, head, acts


# ================================================================== THE PAGE: data, app bar and the three tabs
# the app bar is filled in once the data is known
app_bar_placeholder = st.empty()                                                  # the app bar is filled in once the data is known
data_path = find_data_file()

if data_path is not None: raw_table, data_file_label = load_raw_file(str(data_path), data_path.stat().st_mtime), data_path.name
else:
    st.info(f"Put **{DATA_FILE_NAME}** in the same folder as app.py.")
    st.stop()
device_ids = list_device_ids(raw_table)
# default to device 16 when the file holds several devices
device_id = 16 if 16 in device_ids else (device_ids[0] if device_ids else None)
assumed_start_time = "2024-01-01 00:00:00"                                 # only used when a file has no timestamp column

try: minute_rows, file_info = clean_meter_data(raw_table, device_id, assumed_start_time)
except ValueError as err: st.error(str(err)); st.stop()
RUNNING_KW_THRESHOLD = 3 * MIN_PHASE_KW                                               # three-phase load above this = machine running
minute_grid_raw, minute_grid_smoothed = build_minute_grid(minute_rows, SMOOTHING_MINUTES)
timeline_stats = summarise_timeline(minute_rows, RUNNING_KW_THRESHOLD)
smoothed_complete_minutes = minute_grid_smoothed.dropna(subset=REQUIRED_COLS)                                      # smoothed minutes without gaps

if len(smoothed_complete_minutes) < 200: st.error("Not enough rows: the file needs at least a few hours of one-minute readings."); st.stop()
has_real_clock = file_info["clock"] == "file"
latest_balance_table = analyse_balance(smoothed_complete_minutes.iloc[-1][PHASE_KW_COLS].to_numpy())
status_pill_html = (f'<span class="pill run"><span class="dot"></span>Running now</span>' if timeline_stats["now_running"] else
          f'<span class="pill idle"><span class="dot"></span>Idle · last ran {timeline_stats["last_run"]:%d %b %H:%M}</span>')
clock_chip_html = '<span class="chip">clock assumed</span>' if not has_real_clock else ""
app_bar_placeholder.markdown(f"""<div class="appbar"><div class="logo">⚡</div><div><div class="t">GridSense</div><div class="s">Load forecasting · phase imbalance</div></div><div class="sp"></div>
{status_pill_html}<span class="chip">{data_file_label}</span><span class="chip">{minute_rows.Timestamps.min():%d %b} → {minute_rows.Timestamps.max():%d %b %Y}</span><span class="chip">{len(minute_rows):,} readings</span>{clock_chip_html}</div>""", unsafe_allow_html=True)

# three pages: overview, load forecast, phase balance
tab_overview, tab_load, tab_phase = st.tabs(["Overview", "Load forecast", "Phase balance"])
hourly_data = load_hourly_frame(minute_rows, MIN_READINGS_PER_HOUR)                                      # hourly series for both forecast tabs (notebook method)
hourly_data_ok = hourly_data is not None and len(hourly_data["filled"]) >= 72

if hourly_data_ok:
    hourly_measured, hourly_filled = hourly_data["measured"], hourly_data["filled"]
    PHASE_CODES = gf.PHASE_CODES
    phase_measured_hourly, phase_filled_hourly = hourly_measured[PHASE_KW_COLS].set_axis(PHASE_CODES, axis=1), fill_missing_hours(hourly_measured[PHASE_KW_COLS]).set_axis(PHASE_CODES, axis=1)
    total_measured_hourly, total_filled_hourly, last_data_time = hourly_measured["KW"], hourly_filled["KW"], hourly_filled.index[-1]
    future_hours = pd.date_range(last_data_time + pd.Timedelta(hours=1), periods=MAX_FORECAST_HOURS, freq="h")

# ================================================================== OVERVIEW
with tab_overview:
    daily = timeline_stats["daily"]
    reset_kpi_counter(); kpi_columns = st.columns(5)
    with kpi_columns[0]: kpi_card("Energy used", f"{timeline_stats['energy'] / 1000:,.1f} MWh", f"{len(daily)} days of data", "", "🔋", daily.kwh.values, LOAD_COLOR)
    with kpi_columns[1]: kpi_card("Machine uptime", f"{timeline_stats['util']:.1f} %", f"{timeline_stats['run_min'] / 60:,.0f} h on · {timeline_stats['idle_min'] / 60:,.0f} h idle", "info", "⏱️", daily.run_h.values, FORECAST_COLOR)
    with kpi_columns[2]: kpi_card("Peak load", f"{timeline_stats['peak']:,.0f} kW", "highest 1-minute reading", "", "📈")
    with kpi_columns[3]: kpi_card("Average load · running", f"{timeline_stats['avg_run']:,.0f} kW", "only while the machine is on", "", "⚙️", daily.avg_run.values, ACCENT_LIGHT)
    imbalance_status = latest_balance_table.Status.iloc[0]
    with kpi_columns[4]: kpi_card("Phase imbalance now", "idle" if not latest_balance_table.Loaded.iloc[0] else f"{latest_balance_table.Imbalance_pct.iloc[0]:.1f} %", "machine idle" if not latest_balance_table.Loaded.iloc[0] else f"{latest_balance_table.Worst_phase.iloc[0]} {latest_balance_table.Direction.iloc[0]} · limit {ALARM_LIMIT_PCT:g} %",
                   "good" if imbalance_status == "OK" else "warn" if imbalance_status == "ALARM" else "info", "⚖️")
    st.write("")
    timeline_col, donut_col = st.columns([2.1, 1])
    with timeline_col, panel("Load timeline", "10-minute averages · gaps are missing data", key="timeline"):
        signal_choice = st.selectbox("Signal", ["Total load (kW)", "Phase load R / Y / B (kW)", "Phase imbalance (%)", "Phase current (A)", "Voltage (V)"], label_visibility="collapsed")
        cols = {"Total load (kW)": ["KW"], "Phase load R / Y / B (kW)": PHASE_KW_COLS, "Phase imbalance (%)": PHASE_KW_COLS, "Phase current (A)": CURRENT_COLS, "Voltage (V)": VOLTAGE_COLS}[signal_choice]
        ten_minute_means = minute_rows.set_index("Timestamps")[cols].resample("10min").mean(); fig = go.Figure()
        if signal_choice == "Phase imbalance (%)":
            balance_10min = analyse_balance(ten_minute_means.to_numpy()); imbalance_10min = balance_10min.Imbalance_pct.where(balance_10min.Loaded).to_numpy()
            fig.add_scatter(x=ten_minute_means.index, y=imbalance_10min, name="Imbalance %", line=dict(color=ACCENT, width=1.4), fill="tozeroy", fillcolor=hex_to_rgba(ACCENT, .14), connectgaps=False)
            fig.add_hline(y=ALARM_LIMIT_PCT, line_dash="dot", line_color=WARN_COLOR, annotation_text=f"alarm {ALARM_LIMIT_PCT:g} %", annotation_position="top left")
        else:
            line_colors = [LOAD_COLOR] if cols == ["KW"] else [PHASE_COLORS["RKW"], PHASE_COLORS["YKW"], PHASE_COLORS["BKW"]]
            for color_index, (col_name, color) in enumerate(zip(cols, line_colors)):
                fig.add_scatter(x=ten_minute_means.index, y=ten_minute_means[col_name], name=col_name, line=dict(color=color, width=1.4), fill="tozeroy" if len(cols) == 1 else None, fillcolor=hex_to_rgba(ACCENT, .14), connectgaps=False)
        fig.update_xaxes(rangeslider_visible=True, rangeslider_thickness=.07); show_figure(style_figure(fig, 400, legend=True, top=40))
    with donut_col, panel("Where the time went", "share of all minutes", key="donut"):
        fig = go.Figure(go.Pie(labels=["Running", "Idle", "No data"], values=[timeline_stats["run_min"], timeline_stats["idle_min"], timeline_stats["nodata_min"]], hole=.68, sort=False, marker=dict(colors=[GOOD_COLOR, IDLE_COLOR, NO_DATA_COLOR], line=dict(color="#fff", width=2)),
                             textinfo="percent", hovertemplate="%{label}<br>%{value:,} min<extra></extra>"))
        fig.update_layout(hovermode="closest", annotations=[dict(text=f"<b>{timeline_stats['util']:.1f}%</b><br><span style='font-size:11px;color:{MUTED_COLOR}'>uptime</span>", showarrow=False, font=dict(size=22))])
        show_figure(style_figure(fig, 400, True, top=40).update_layout(legend=dict(orientation="h", y=-0.08, x=0.1), margin=dict(l=10, r=10, t=40, b=60)))

# ================================================================== LOAD FORECAST (hourly total kW - Exponential Smoothing)
with tab_load:
  if not hourly_data_ok: show_not_enough_data_note()
  else:
    control_columns = st.columns([1, 1.3, 1])
    load_model_name = "Exponential Smoothing"
    load_length_choice = control_columns[0].selectbox("Forecast length", list(FORECAST_LENGTH_HOURS), index=6, key="ld_len")
    load_season_hours = SEASON_LENGTH_HOURS[control_columns[1].selectbox("Seasonal cycle", list(SEASON_LENGTH_HOURS), index=6, key="ld_season", help="Length of the repeating pattern the model learns (7 days = weekly pattern).")]
    load_history_choice = control_columns[2].selectbox("History shown", list(HISTORY_SHOWN_HOURS), index=2, key="ld_hist")
    fixed_smoothing = dict(smoothing_level=0.7, smoothing_trend=0.05, smoothing_seasonal=0.1, damping_trend=0.95)
    max_training_hours = len(total_filled_hourly)                       # learns from all history
    load_training_text = "all available history"
    # train once for 7 days ahead; the page only slices what is asked for
    total_forecast, total_forecast_label, total_fit_ok = forecast_total_load(total_filled_hourly.iloc[-max_training_hours:], MAX_FORECAST_HOURS, load_model_name, load_season_hours, fixed_smoothing)
    if not total_fit_ok: note_box(f"The model could not be fitted ({total_forecast_label}); the forecast is a flat line. Try the other model or a different cycle.", "amber")
    horizon_hours = FORECAST_LENGTH_HOURS[load_length_choice]; forecast_values = total_forecast[:horizon_hours]; forecast_times = future_hours[:horizon_hours]
    last_actual_kw = float(total_filled_hourly.iloc[-1]); end_value_kw = float(forecast_values[-1])
    change_text = f"{abs(end_value_kw - last_actual_kw) / last_actual_kw * 100:.1f} %" if last_actual_kw >= 1 else f"{abs(end_value_kw - last_actual_kw):.1f} kW"
    arrow = "▲" if end_value_kw >= last_actual_kw else "▼"
    forecast_energy_kwh, forecast_avg_kw, forecast_peak_kw = float(forecast_values.sum()), float(forecast_values.mean()), float(forecast_values.max()); previous_energy_kwh = float(total_filled_hourly.iloc[-horizon_hours:].sum())
    # 7-day test (cached): its errors give the width of the forecast interval (used by the cards, the block chart and the forecast chart; the Model check below reuses the test)
    holdout_result = run_total_load_holdout(total_measured_hourly, total_filled_hourly, load_model_name, load_season_hours, fixed_smoothing, max_training_hours)
    interval_sigma = forecast_interval_sigma(holdout_result, total_filled_hourly)[:horizon_hours]
    interval_low, interval_high = np.maximum(forecast_values - CONFIDENCE_Z * interval_sigma, 0.0), forecast_values + CONFIDENCE_Z * interval_sigma
    energy_low_kwh, energy_high_kwh = float(interval_low.sum()), float(interval_high.sum())
    with panel("Power needed in the selected period", f"{forecast_times[0]:%d %b %H:%M} → {(forecast_times[-1] + pd.Timedelta(hours=1)):%d %b %H:%M} · {hours_to_text(horizon_hours)} · energy = sum of hourly kW", key="ld_pw"):
        reset_kpi_counter(); energy_kpi_columns = st.columns(5)
        with energy_kpi_columns[0]: kpi_card("Energy needed · forecast", f"{forecast_energy_kwh:,.0f} kWh", f"average {forecast_avg_kw:,.1f} kW over {hours_to_text(horizon_hours)}<br>{CONFIDENCE_TEXT} range {energy_low_kwh:,.0f} – {energy_high_kwh:,.0f} kWh", "info", "🔌")
        energy_difference_kwh = forecast_energy_kwh - previous_energy_kwh
        with energy_kpi_columns[1]: kpi_card("Energy used · previous period", f"{previous_energy_kwh:,.0f} kWh", (f"forecast is {energy_difference_kwh:+,.0f} kWh ({energy_difference_kwh / previous_energy_kwh * 100:+.0f} %) vs this" if previous_energy_kwh >= 1 else "the machine was idle in that period") + f" · last {hours_to_text(horizon_hours)}", "", "📊")
        # busiest / quietest forecast DAY (energy of the day) - no hours anywhere
        day_summary = daily_energy_summary(forecast_values, forecast_times); busiest_day, quietest_day = day_summary["sum"].idxmax(), day_summary["sum"].idxmin(); mean_day_kwh = float(day_summary["sum"].mean())
        busiest_vs_mean_pct = (day_summary.loc[busiest_day, "sum"] / mean_day_kwh - 1) * 100 if mean_day_kwh >= 1 else float("nan")
        with energy_kpi_columns[2]: kpi_card("Most power used · day", f"{busiest_day:%a %d %b}", f"{day_summary.loc[busiest_day, 'sum']:,.0f} kWh · average {day_summary.loc[busiest_day, 'mean']:,.1f} kW", "", "📅")
        with energy_kpi_columns[3]: kpi_card("Least power used · day", f"{quietest_day:%a %d %b}", f"{day_summary.loc[quietest_day, 'sum']:,.0f} kWh · average {day_summary.loc[quietest_day, 'mean']:,.1f} kW", "info", "🌙")
        # AVERAGE 8 H (a shift): energy of every 8-hour block of the forecast; only full 8 h blocks when there are any
        shift_table = average_in_blocks(forecast_values, forecast_times[0], 8); shift_low_table, shift_high_table = average_in_blocks(interval_low, forecast_times[0], 8), average_in_blocks(interval_high, forecast_times[0], 8)
        shift_use = (shift_table["Hours"] == 8).to_numpy(); shift_use = shift_use if shift_use.any() else np.ones(len(shift_table), bool)
        shift_hours = shift_table["Hours"].to_numpy()[shift_use]; shift_kwh = shift_table["Avg kW"].to_numpy()[shift_use] * shift_hours; mean_shift_kwh = float(shift_kwh.mean())
        shift_low_kwh, shift_high_kwh = float((shift_low_table["Avg kW"].to_numpy()[shift_use] * shift_hours).mean()), float((shift_high_table["Avg kW"].to_numpy()[shift_use] * shift_hours).mean())
        busiest_shift_vs_mean_pct = (shift_kwh.max() / mean_shift_kwh - 1) * 100 if mean_shift_kwh >= 1 else float("nan")
        with energy_kpi_columns[4]: kpi_card("Average 8 h", f"{mean_shift_kwh:,.0f} kWh", f"average {float(shift_table['Avg kW'].to_numpy()[shift_use].mean()):,.1f} kW" + ("" if len(shift_kwh) < 2 or not np.isfinite(busiest_shift_vs_mean_pct) else f" · busiest 8 h block {busiest_shift_vs_mean_pct:+.0f} % vs this") + f"<br>{CONFIDENCE_TEXT} range {shift_low_kwh:,.0f} – {shift_high_kwh:,.0f} kWh", "", "🕗")
        st.write("")
        valid_block_names = [n for n, h in AVERAGE_BLOCK_HOURS.items() if h <= horizon_hours]
        block_select_column, _ = st.columns([1.2, 3])
        block_choice = block_select_column.selectbox("Show average power every", valid_block_names, index=valid_block_names.index("8 hours") if "8 hours" in valid_block_names else min(2, len(valid_block_names) - 1), key="ld_avg", help="Splits the forecast into blocks of this length and averages the power in each block.")
        # split the forecast into blocks and average each block
        block_table = average_in_blocks(forecast_values, forecast_times[0], AVERAGE_BLOCK_HOURS[block_choice]); block_labels = [block_time_label(r) for r in block_table.itertuples()]; has_two_blocks = len(block_table) >= 2
        trend_direction = classify_trend(float(block_table["Avg kW"].iloc[0]), float(block_table["Avg kW"].iloc[-1])) if has_two_blocks else classify_trend(float(forecast_values[0]), float(forecast_values[-1]))
        first_block_kw, last_block_kw = (float(block_table["Avg kW"].iloc[0]), float(block_table["Avg kW"].iloc[-1])) if has_two_blocks else (float(forecast_values[0]), float(forecast_values[-1]))
        # straight-line trend of the block averages, in kW per hour
        slope = float(np.polyfit(np.arange(len(block_table)) * AVERAGE_BLOCK_HOURS[block_choice], block_table["Avg kW"], 1)[0]) if has_two_blocks else float("nan")
        block_chart_container = st.container()
        with block_chart_container:
            block_low_kw = average_in_blocks(interval_low, forecast_times[0], AVERAGE_BLOCK_HOURS[block_choice])["Avg kW"].to_numpy(); block_high_kw = average_in_blocks(interval_high, forecast_times[0], AVERAGE_BLOCK_HOURS[block_choice])["Avg kW"].to_numpy(); block_avg_kw = block_table["Avg kW"].to_numpy()
            fig = go.Figure(go.Bar(x=block_labels, y=block_avg_kw, marker_color=ACCENT, name="Average kW", error_y=dict(type="data", symmetric=False, array=block_high_kw - block_avg_kw, arrayminus=block_avg_kw - block_low_kw, color=TEXT_COLOR, thickness=1.4, width=4),
                                 customdata=np.c_[block_table["Change kW"].fillna(0), block_table["Hours"], block_low_kw, block_high_kw],
                                 hovertemplate="%{x}<br><b>%{y:,.1f} kW average</b><br>" + CONFIDENCE_TEXT + " interval %{customdata[2]:,.1f} – %{customdata[3]:,.1f} kW<br>change vs previous block %{customdata[0]:+,.1f} kW<br>%{customdata[1]} h<extra></extra>"))
            fig.add_scatter(x=block_labels, y=block_high_kw, mode="text", text=[("" if pd.isna(x) else f"{x:+.1f} %") for x in block_table["Change %"]], textposition="top center", showlegend=False, hoverinfo="skip", cliponaxis=False)    # change label sits above the interval whisker
            fig.update_layout(hovermode="closest", yaxis_title=f"average kW per {block_choice}", title=f"Average power every {block_choice}  ·  whisker = {CONFIDENCE_TEXT} interval  ·  label = change vs previous block"); fig.update_xaxes(type="category"); fig.update_yaxes(range=[0, float(np.nanmax(block_high_kw)) * 1.15])
            show_figure(style_figure(fig, 300, False, top=60))

    # ONE suggestions block (fixed wording from the reference images, numbers from the forecast)
    load_advice = build_load_suggestions(forecast_values, forecast_times, previous_energy_kwh)                     # only used for the badge and the border colour
    st.markdown(forecast_advice_card_html(forecast_values, forecast_times, previous_energy_kwh, horizon_hours, load_advice["badge"], load_advice["tone"]), unsafe_allow_html=True)

    with panel("Forecast ahead", key="ld_next"):
        actual_history = total_measured_hourly.iloc[-HISTORY_SHOWN_HOURS[load_history_choice]:]; fig = go.Figure(); forecast_x = [last_data_time] + list(forecast_times)
        fig.add_shape(type="rect", xref="x", yref="paper", x0=last_data_time, x1=forecast_times[-1], y0=0, y1=1, fillcolor=hex_to_rgba(ACCENT, .06), line_width=0)
        fig.add_scatter(x=actual_history.index, y=actual_history.values, name="Actual hourly kW", mode="lines", line=dict(color=TEXT_COLOR, width=2.2), connectgaps=False)
        # confidence interval: upper edge first, then the lower edge filled up to it
        fig.add_scatter(x=forecast_x, y=[last_actual_kw] + list(interval_high), name="Upper", mode="lines", line=dict(width=0), showlegend=False, hoverinfo="skip")
        fig.add_scatter(x=forecast_x, y=[last_actual_kw] + list(interval_low), name=f"{CONFIDENCE_TEXT} interval", mode="lines", line=dict(width=0), fill="tonexty", fillcolor=hex_to_rgba(ACCENT, .18),
                        customdata=[last_actual_kw] + list(interval_high), hovertemplate=f"{CONFIDENCE_TEXT} interval: %{{y:,.1f}} – %{{customdata:,.1f}} kW<extra></extra>")
        fig.add_scatter(x=forecast_x, y=[last_actual_kw] + list(forecast_values), name=f"Forecast ({load_model_name})", mode="lines", line=dict(color=FORECAST_COLOR, width=3))
        fig.add_shape(type="line", xref="x", yref="paper", x0=last_data_time, x1=last_data_time, y0=0, y1=1, line=dict(color=MUTED_COLOR, width=1.2, dash="dot"))
        fig.add_annotation(x=last_data_time, yref="paper", y=1.0, text="last data point", showarrow=False, xanchor="right", yanchor="bottom", font=dict(size=11, color=MUTED_COLOR))
        fig.update_yaxes(title="kW"); show_figure(style_figure(fig, 340, True, top=70))

    # ---- model check on the last 168 hours (the load notebook's test)
    st.markdown('<div class="ph" style="margin-top:.6rem"><b>Model check on past data</b></div>', unsafe_allow_html=True)
    # (the 7-day test result `holdout_result` was computed above)
    if holdout_result is None: st.caption("Not enough history for a 7-day test.")
    else:
        holdout_scores = holdout_result["scores"]; mean_actual_kw = float(np.nanmean(holdout_result["actual"].to_numpy()))
        reset_kpi_counter(); kpi_columns = st.columns(4)
        with kpi_columns[0]: kpi_card("MAE", f"{holdout_scores['MAE']:.2f} kW", f"{holdout_scores['MAE'] / max(mean_actual_kw, 1e-9) * 100:.1f} % of the mean test load" if mean_actual_kw >= 1 else "test period has ~0 kW load", "info", "📏")
        with kpi_columns[1]: kpi_card("RMSE", f"{holdout_scores['RMSE']:.2f} kW", f"{holdout_scores['n']} hours compared", "", "🎯")
        with kpi_columns[2]: kpi_card("WAPE", f"{holdout_scores['WAPE']:.1f} %", "total error ÷ total load · smaller is better", "good" if holdout_scores["WAPE"] < 10 else "warn" if holdout_scores["WAPE"] < 25 else "bad", "✅")
        with kpi_columns[3]: kpi_card("SMAPE", f"{holdout_scores['SMAPE']:.1f} %", "average % error · smaller is better", "", "🧪")
        fig = go.Figure(); fig.add_scatter(x=holdout_result["actual"].index, y=holdout_result["actual"].values, name="Actual hourly kW", line=dict(color=TEXT_COLOR, width=2), connectgaps=False)
        fig.add_scatter(x=holdout_result["pred"].index, y=holdout_result["pred"].values, name=f"Predicted ({load_model_name})", line=dict(color=FORECAST_COLOR, width=2, dash="dash")); fig.update_yaxes(title="kW")
        with st.container(border=True, key="panel_ld_test"): show_figure(style_figure(fig, 340, True, top=60))

# ================================================================== PHASE BALANCE (measured + forecast + action plan, all on one page)
with tab_phase:
  if not hourly_data_ok: show_not_enough_data_note()
  else:
    control_columns = st.columns([1, 1.3, 1, 1.5])
    phase_model_name = "Exponential Smoothing"
    phase_length_choice = control_columns[0].selectbox("Forecast length", list(FORECAST_LENGTH_HOURS), index=6, key="ph_len")
    phase_training_days = 14                                  # fixed learning window (the dropdown was removed)
    phase_season_hours = SEASON_LENGTH_HOURS[control_columns[1].selectbox("Seasonal cycle", list(SEASON_LENGTH_HOURS), index=6, key="ph_season")]
    phase_history_choice = control_columns[2].selectbox("History shown", list(HISTORY_SHOWN_HOURS), index=2, key="ph_hist")
    phase_method = control_columns[3].selectbox("Forecast method", gf.PHASE_FORECAST_METHODS, key="ph_method", help="Each phase separately = forecast R, Y and B one by one. Total × phase shares = forecast the total and each phase's share, then rebuild R / Y / B. The test below shows which one does better on your data.")
    max_training_hours = int(phase_training_days) * 24; horizon_hours = FORECAST_LENGTH_HOURS[phase_length_choice]
    # forecast Red, Yellow and Blue 7 days ahead
    phase_forecast_array, phase_labels_by_series, phase_fit_ok = forecast_phases_cached(phase_filled_hourly.iloc[-max_training_hours:], MAX_FORECAST_HOURS, phase_model_name, phase_season_hours, phase_method)
    forecast_phase_kw = pd.DataFrame(phase_forecast_array[:horizon_hours], index=future_hours[:horizon_hours], columns=PHASE_CODES); forecast_table = analyse_balance(forecast_phase_kw.to_numpy())
    TEST_HOURS = 168                                                              # all test data = the last 168 hours (7 days)
    if not phase_fit_ok: note_box(f"A model fit failed for {', '.join(p for p in PHASE_CODES if phase_labels_by_series[p].startswith('failed'))} - that phase repeats its last value. Try the other model or cycle.", "amber")

    # forecast interval per colour: from the same 7-day test the Model check below shows (cached, so no extra training)
    phase_hold_pred = None
    if len(phase_filled_hourly) >= TEST_HOURS + 2 * phase_season_hours:
        hold_index = phase_filled_hourly.index[-TEST_HOURS:]; hold_array, _, _ = forecast_phases_cached(phase_filled_hourly.iloc[:-TEST_HOURS].iloc[-max_training_hours:], TEST_HOURS, phase_model_name, phase_season_hours, phase_method)
        phase_hold_pred = pd.DataFrame(hold_array[:TEST_HOURS], index=hold_index, columns=PHASE_CODES)
    phase_low_kw, phase_high_kw = forecast_phase_kw.copy(), forecast_phase_kw.copy()
    for p in PHASE_CODES:
        phase_sigma = forecast_interval_sigma(None if phase_hold_pred is None else {"actual": phase_measured_hourly[p].reindex(phase_hold_pred.index), "pred": phase_hold_pred[p]}, phase_filled_hourly[p])[:horizon_hours]
        phase_low_kw[p], phase_high_kw[p] = np.maximum(forecast_phase_kw[p].to_numpy() - CONFIDENCE_Z * phase_sigma, 0.0), forecast_phase_kw[p].to_numpy() + CONFIDENCE_Z * phase_sigma

    # measured imbalance, cause and move plan from the one-minute data
    diagnosis_result = diagnose_phase_imbalance(smoothed_complete_minutes[REQUIRED_COLS], ALARM_LIMIT_PCT, MIN_PHASE_KW); diagnosis = None if diagnosis_result["empty"] else diagnosis_result
    latest_row = latest_balance_table.iloc[0]
    hourly_balance_table = analyse_balance(phase_measured_hourly.dropna().to_numpy()); hourly_mean_imbalance_pct = float(hourly_balance_table.Imbalance_pct[hourly_balance_table.Loaded].mean()) if hourly_balance_table.Loaded.any() else float("nan")      # same hourly basis as the forecast
    latest_status = dict(loaded=bool(latest_row.Loaded), imb=float(latest_row.Imbalance_pct) if latest_row.Loaded else 0.0, worst=latest_row.Worst_phase, dev=float(latest_row.Worst_dev) if latest_row.Loaded else 0.0, direction=latest_row.Direction, status=latest_row.Status)

    forecast_imbalance_pct = forecast_table.Imbalance_pct.to_numpy(); n_alarm_hours = int(forecast_table.Alarm.sum())
    # first forecast hour that breaks the limit
    first_alarm = forecast_phase_kw.index[int(np.flatnonzero(forecast_table.Alarm.to_numpy())[0])] if n_alarm_hours else None
    worst_hour_position = int(np.nanargmax(forecast_imbalance_pct)) if np.isfinite(forecast_imbalance_pct).any() else 0
    reset_kpi_counter(); kpi_columns = st.columns(5)
    with kpi_columns[0]: kpi_card("Imbalance now", "idle" if not latest_status["loaded"] else f"{latest_status['imb']:.1f} %", "machine idle" if not latest_status["loaded"] else f"{latest_status['worst']} {latest_status['direction']} · limit {ALARM_LIMIT_PCT:g} %", "good" if latest_status["status"] == "OK" else "warn" if latest_status["status"] == "ALARM" else "info", "⚖️")
    with kpi_columns[1]:
        if diagnosis: kpi_card("Average imbalance · measured", f"{diagnosis['mean_imb']:.1f} %", f"above the limit in {diagnosis['pct_above']:.0f} % of loaded minutes · hourly means: {hourly_mean_imbalance_pct:.1f} %", "good" if diagnosis["mean_imb"] <= ALARM_LIMIT_PCT else "warn", "📏")
        else: kpi_card("Average imbalance · measured", "–", "too few loaded minutes", "", "📏")
    with kpi_columns[2]: kpi_card("Forecast alarm hours", f"{n_alarm_hours}", f"of {horizon_hours} hours", "", "🚨", trend=np.nan_to_num(forecast_imbalance_pct), trend_color=ACCENT)
    with kpi_columns[3]: kpi_card("First forecast alarm", "none" if first_alarm is None else f"{first_alarm:%a %H:%M}", "no alarm expected" if first_alarm is None else f"{first_alarm:%d %b} · in {(first_alarm - last_data_time).total_seconds() / 3600:.0f} h", "good" if first_alarm is None else "warn", "⏰")
    with kpi_columns[4]: kpi_card("Worst phase · forecast", "–" if not np.isfinite(forecast_imbalance_pct).any() else forecast_table.Worst_phase.iloc[worst_hour_position], "" if not np.isfinite(forecast_imbalance_pct).any() else f"{forecast_table.Direction.iloc[worst_hour_position]} · peak {forecast_imbalance_pct[worst_hour_position]:.1f} % at {forecast_phase_kw.index[worst_hour_position]:%a %H:%M}", "info", "🎨")
    st.write("")

    # ---- one chart: measured hours + forecast per phase; an X marks the hours when THAT phase is the one out of balance
    with panel("Phase load · measured and forecast", f"hourly means · {phase_model_name} · learns from the last {phase_training_days} days · forecast starts after {last_data_time:%d %b %Y %H:%M} · shaded band = {CONFIDENCE_TEXT} interval · ✕ = alarm hour on the phase that is out of balance", key="ph_main"):
        recent_phase_hours = phase_filled_hourly.iloc[-HISTORY_SHOWN_HOURS[phase_history_choice]:]; forecast_line_x = [last_data_time] + list(forecast_phase_kw.index)
        fig = make_subplots(rows=3, cols=1, shared_xaxes=True, vertical_spacing=.08, subplot_titles=[f"{n} phase (kW)" for n in PHASE_LABELS])
        for j, p in enumerate(PHASE_CODES, 1):
            col = PHASE_COLORS[PHASE_KW_COLS[j - 1]]
            fig.add_scatter(x=recent_phase_hours.index, y=recent_phase_hours[p], name="Measured (hourly)", legendgroup="m", showlegend=j == 1, line=dict(color=TEXT_COLOR, width=2), connectgaps=False, row=j, col=1)
            last_phase_kw = float(phase_filled_hourly[p].iloc[-1])
            fig.add_scatter(x=forecast_line_x, y=[last_phase_kw] + list(phase_high_kw[p]), mode="lines", line=dict(width=0), showlegend=False, hoverinfo="skip", legendgroup="ci", row=j, col=1)
            fig.add_scatter(x=forecast_line_x, y=[last_phase_kw] + list(phase_low_kw[p]), name=f"{CONFIDENCE_TEXT} interval", legendgroup="ci", showlegend=j == 1, mode="lines", line=dict(width=0), fill="tonexty", fillcolor=hex_to_rgba(col, .20),
                            customdata=[last_phase_kw] + list(phase_high_kw[p]), hovertemplate=f"{CONFIDENCE_TEXT} interval: %{{y:,.1f}} – %{{customdata:,.1f}} kW<extra></extra>", row=j, col=1)
            fig.add_scatter(x=forecast_line_x, y=[last_phase_kw] + list(forecast_phase_kw[p]), name="Forecast", legendgroup="f", showlegend=j == 1, mode="lines", line=dict(color=col, width=2.6, dash="dash"), row=j, col=1)
        worst_phase_names, legend_added = forecast_table.Worst_phase.to_numpy(), False

        def reason_txt(name):
            """Which parameter explains the imbalance, and how far this phase is from the 3-phase average (measured history)."""
            if not diagnosis: return "Load (not enough data)"
            param, col = ("Current", "Avg current A") if diagnosis["share"]["Current"] >= diagnosis["share"]["Voltage"] else ("Voltage", "Avg voltage V")
            v = diagnosis["per"][col]; d = (v[name] / v.mean() - 1) * 100
            return f"{param} {'▲ up' if d > 0 else '▼ down'} {abs(d):.1f} % vs 3-phase average"

        for j, p in enumerate(PHASE_CODES, 1):
            phase_alarm_mask = forecast_table.Alarm.to_numpy() & (worst_phase_names == PHASE_LABELS[j - 1])
            if not phase_alarm_mask.any(): continue
            fig.add_scatter(x=forecast_phase_kw.index[phase_alarm_mask], y=forecast_phase_kw[p].to_numpy()[phase_alarm_mask], mode="markers", name="Alarm (forecast)", legendgroup="alarm", showlegend=not legend_added,
                          marker=dict(symbol="x", size=9, color=WARN_COLOR),
                          customdata=np.c_[forecast_imbalance_pct[phase_alarm_mask], forecast_table.Direction.to_numpy()[phase_alarm_mask], np.full(phase_alarm_mask.sum(), reason_txt(PHASE_LABELS[j - 1]))],
                          hovertemplate="%{y:.1f} kW · imbalance %{customdata[0]:.1f} % (%{customdata[1]})<br>Reason: %{customdata[2]}<extra>Alarm</extra>", row=j, col=1)
            legend_added = True
        fig.add_shape(type="line", xref="x", yref="paper", x0=last_data_time, x1=last_data_time, y0=0, y1=1, line=dict(color=MUTED_COLOR, width=1.2, dash="dot"))
        fig.update_yaxes(title_text="kW")
        style_figure(fig, 640, legend_y=1.0, top=110); fig.update_annotations(font=dict(size=13, color=TEXT_COLOR)); show_figure(fig)

    with panel("Forecast by day", "hourly calls are noisy - the day-level picture is steadier", key="ph_days"):
        forecast_with_day = forecast_phase_kw.assign(imb=forecast_table.Imbalance_pct.to_numpy(), worst=forecast_table.Worst_phase.to_numpy(), alarm=forecast_table.Alarm.to_numpy(), day=forecast_phase_kw.index.date)
        # one summary row per forecast day
        daily_rows = []
        for forecast_day, day_rows in forecast_with_day.groupby("day"):
            alarm_rows_of_day = day_rows[day_rows["alarm"]]
            daily_rows.append({"Day": f"{forecast_day:%a %d %b}", "Hours": len(day_rows), "Avg imbalance %": round(float(day_rows["imb"].mean()), 1) if day_rows["imb"].notna().any() else np.nan,
                           "Alarm hours": int(day_rows["alarm"].sum()), "Main phase out of balance": alarm_rows_of_day["worst"].mode().iloc[0] if len(alarm_rows_of_day) else "-"})
        st.dataframe(pd.DataFrame(daily_rows).set_index("Day"), width="stretch")

    # ---- 3-phase load analysis card (rule-based, short)
    # best time to move loads = quietest 3-hour window
    quiet_busy_window = gr.find_quietest_and_busiest_windows(forecast_phase_kw.sum(axis=1).to_numpy(), forecast_phase_kw.index) if n_alarm_hours else None
    st.markdown(phase_analysis_card_html(forecast_phase_kw, forecast_table, diagnosis, quiet_busy_window, phase_low_kw, phase_high_kw), unsafe_allow_html=True)

    # ---- the evidence behind the advice
        # ---- A) which parameter moves each phase: load, current, voltage per colour + the imbalance each one creates
    with panel("Phase parameters · what is rising or falling", "loaded minutes only · load, current and voltage of each colour, and the imbalance each one creates", key="ph_params"):
        parameter_controls = st.columns([1, 1, 3])
        average_by_choice = parameter_controls[0].selectbox("Average by", ["Hour", "Day"], key="pp_rule")
        window_days_choice = parameter_controls[1].selectbox("Window", [7, 14, 28], key="pp_days", format_func=lambda x: f"last {x} days")
        phase_params_window = load_phase_parameter_frame(minute_rows, RUNNING_KW_THRESHOLD, "1h" if average_by_choice == "Hour" else "1D", 10 if average_by_choice == "Hour" else 60).iloc[-(window_days_choice * (24 if average_by_choice == "Hour" else 1)):]
        if phase_params_window.dropna().shape[0] < 4: st.caption("Not enough loaded periods in this window.")
        else:
            phase_color_list = [PHASE_COLORS[k_] for k_ in PHASE_KW_COLS]
            imbalance_chart_column, deviation_chart_column = st.columns(2)
            with imbalance_chart_column:
                fig = go.Figure()
                for (parameter_label, parameter_cols), line_color in zip(PHASE_PARAMETER_COLS.items(), [ACCENT, BLUE_COLOR, WARN_COLOR]):
                    fig.add_scatter(x=phase_params_window.index, y=imbalance_pct_of(phase_params_window, parameter_cols), name=parameter_label.split(" ")[0], line=dict(color=line_color, width=2), connectgaps=True)
                fig.add_hline(y=ALARM_LIMIT_PCT, line_dash="dot", line_color=WARN_COLOR, annotation_text=f"alarm {ALARM_LIMIT_PCT:g} %", annotation_position="top left")
                fig.update_layout(title="Imbalance % of load, current and voltage", yaxis_title="%"); show_figure(style_figure(fig, 300, True, top=70), key="pp_imb")
            with deviation_chart_column:
                load_deviation_table = deviation_from_average_pct(phase_params_window, PHASE_PARAMETER_COLS["Load (kW)"]); fig = go.Figure()
                for n_, line_color in zip(PHASE_LABELS, phase_color_list): fig.add_scatter(x=load_deviation_table.index, y=load_deviation_table[n_], name=n_, line=dict(color=line_color, width=2), connectgaps=True)
                fig.add_hline(y=0, line_color=MUTED_COLOR, line_width=1); fig.add_hrect(y0=-ALARM_LIMIT_PCT, y1=ALARM_LIMIT_PCT, fillcolor=hex_to_rgba(GOOD_COLOR, .08), line_width=0)
                fig.update_layout(title="Which phase is off: load vs the 3-phase average", yaxis_title="% above / below average"); show_figure(style_figure(fig, 300, True, top=70), key="pp_dev")

            # first half vs second half of the window
            trend_tables = half_over_half_change(phase_params_window)
            if trend_tables:
                st.markdown('<div class="ph"><b>What changed</b></div>', unsafe_allow_html=True)
                st.dataframe(trend_tables[1], width="stretch")
            note_box("<ul>" + "".join(f"<li>{t_}</li>" for t_ in explain_imbalance_change(phase_params_window)) + "</ul>")

      # ---- B) actual vs predicted on test data: last 7 days hidden, model learns only from the hours before them
    with panel("Model check · actual vs predicted on test data", "the last 7 days are hidden, the model learns from the 28 days before them, then its forecast is compared with what really happened", key="ph_bt"):

        def phase_error_scores(actual, predicted):
            """MAE (kW), MAPE (%), SMAPE (%) on the hours where both values exist."""
            a, f = np.asarray(actual, float), np.asarray(predicted, float); ok = np.isfinite(a) & np.isfinite(f); a, f = a[ok], f[ok]
            if len(a) == 0: return dict(MAE=np.nan, MAPE=np.nan, SMAPE=np.nan)
            err = np.abs(f - a); nonzero = np.abs(a) > 1e-9; den = np.abs(a) + np.abs(f)
            return dict(MAE=float(err.mean()),
                        MAPE=float((err[nonzero] / np.abs(a[nonzero])).mean() * 100) if nonzero.any() else np.nan,
                        SMAPE=float(np.mean(np.where(den > 1e-9, 2 * err / np.where(den > 1e-9, den, 1), 0)) * 100))

        if len(phase_filled_hourly) < TEST_HOURS + 2 * phase_season_hours:
            st.caption("Not enough history for a 7-day test with this seasonal cycle. Choose a shorter seasonal cycle.")
        else:
            test_index = phase_filled_hourly.index[-TEST_HOURS:]
            training_hours = phase_filled_hourly.iloc[:-TEST_HOURS].iloc[-max_training_hours:]            # only hours BEFORE the test period
            test_forecast_array, _, _ = forecast_phases_cached(training_hours, TEST_HOURS, phase_model_name, phase_season_hours, phase_method)
            predicted_kw = pd.DataFrame(test_forecast_array[:TEST_HOURS], index=test_index, columns=PHASE_CODES)
            actual_kw_drawn = phase_filled_hourly.reindex(test_index)                                       # drawn (missing hours filled)
            actual_kw_real = phase_measured_hourly.reindex(test_index)                                      # scored (real hours only)
            actual_balance_table, predicted_balance_table = analyse_balance(actual_kw_drawn.to_numpy()), analyse_balance(predicted_kw.to_numpy())
            actual_balance_table.index = test_index; predicted_balance_table.index = test_index

            fig = make_subplots(rows=4, cols=1, shared_xaxes=True, vertical_spacing=.06, subplot_titles=[f"{n_} phase (kW)" for n_ in PHASE_LABELS] + ["Imbalance % (calculated from the phase kW)"])
            for j, p in enumerate(PHASE_CODES, 1):
                fig.add_scatter(x=test_index, y=actual_kw_drawn[p], name="Actual", legendgroup="a", showlegend=j == 1, line=dict(color=TEXT_COLOR, width=2), row=j, col=1)
                fig.add_scatter(x=test_index, y=predicted_kw[p], name="Predicted", legendgroup="p", showlegend=j == 1, line=dict(color=PHASE_COLORS[PHASE_KW_COLS[j - 1]], width=2.2, dash="dash"), row=j, col=1)
            fig.add_scatter(x=test_index, y=actual_balance_table.Imbalance_pct.where(actual_balance_table.Loaded), name="Actual", legendgroup="a", showlegend=False, line=dict(color=TEXT_COLOR, width=2), connectgaps=True, row=4, col=1)
            fig.add_scatter(x=test_index, y=predicted_balance_table.Imbalance_pct.where(predicted_balance_table.Loaded), name="Predicted", legendgroup="p", showlegend=False, line=dict(color=ACCENT, width=2.2, dash="dash"), connectgaps=True, row=4, col=1)
            fig.add_hline(y=ALARM_LIMIT_PCT, line_dash="dot", line_color=WARN_COLOR, row=4, col=1)
            style_figure(fig, 780, legend_y=1.0, top=110); fig.update_annotations(font=dict(size=13, color=TEXT_COLOR)); show_figure(fig, key="bt_main")

            error_table = pd.DataFrame({name: phase_error_scores(actual_kw_real[p].to_numpy(), predicted_kw[p].to_numpy()) for name, p in zip(PHASE_LABELS, PHASE_CODES)}).T
            error_table.columns = ["MAE (kW)", "MAPE (%)", "SMAPE (%)"]
            error_table.loc["Average"] = error_table.mean()
            st.dataframe(error_table.round(2), width="stretch")
