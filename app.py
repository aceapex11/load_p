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
import html as html_lib
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
FORECAST_LENGTH_HOURS = {f"{d} day{'s' if d > 1 else ''}": d * 24 for d in range(1, 8)}
HISTORY_SHOWN_HOURS = {"24 hours": 24, "3 days": 72, "7 days": 168, "14 days": 336}
SEASON_LENGTH_HOURS = {f"{d} day{'s' if d > 1 else ''}": d * 24 for d in range(1, 8)}
MAX_FORECAST_HOURS = 168                                                   # every forecast is made 7 days ahead once; the page slices what you ask for
PHASE_PARAMETER_COLS = {"Load (kW)": ["RKW", "YKW", "BKW"], "Current (A)": ["I1", "I2", "I3"], "Voltage (V)": ["V1", "V2", "V3"]}

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
    imbalance_by_hour = pd.Series(kw_imbalance_pct, index=valid_frame.index).groupby(valid_frame.index.hour).mean()
    status_counts = imbalance_table.Status[imbalance_table.Loaded].value_counts().reindex(["OK", "ALARM"]).fillna(0)
    return dict(result, empty=False, mean_imb=float(kw_imbalance_pct.mean()), p95_imb=float(np.percentile(kw_imbalance_pct, 95)), pct_above=float((imbalance_table.Imbalance_pct[imbalance_table.Loaded] > alarm_limit_pct).mean() * 100),
                mean_I=float(current_imbalance_pct.mean()), mean_V=float(voltage_imbalance_pct.mean()), p95_V=float(np.percentile(voltage_imbalance_pct, 95)), pct_current_above=float((current_imbalance_pct > alarm_limit_pct).mean() * 100), pct_voltage_above=float((voltage_imbalance_pct > alarm_limit_pct).mean() * 100), max_current_imb=float(np.max(current_imbalance_pct)), max_voltage_imb=float(np.max(voltage_imbalance_pct)), history_points=int(len(valid_frame)), history_start=valid_frame.index.min(), history_end=valid_frame.index.max(), corr_I=safe_correlation(kw_imbalance_pct, current_imbalance_pct), corr_V=safe_correlation(kw_imbalance_pct, voltage_imbalance_pct), share=cause_share_pct,
                per=per_phase_table, main=main_phase_index, main_name=PHASE_LABELS[main_phase_index], extra_A=float(main_extra_current_a.mean()), extra_kW=float(main_extra_kw.mean()), levels=load_level_table, const=bool(extra_current_is_constant), lvA_const=float(extra_current_by_level.iloc[1:].mean()),
                ctot=safe_correlation(kw_imbalance_pct, total_kw), plan=transfer_plan, Im=mean_current_a, Km=mean_kw, neutral_mean=float(neutral_current_a.mean()), neutral_p95=float(np.percentile(neutral_current_a, 95)),
                by_hour=imbalance_by_hour, status=status_counts, pf_ok=bool(pf_is_meaningful), pf_phase=float(phase_power_factor.mean()), pf_meter=float(valid_frame["PF"].mean()), high_share=float(per_phase_table.iloc[main_phase_index, 1]))

# ------------------------------------------------------------------ RULE-BASED ADVICE TEXT (no AI)


CARD_BLUE, CARD_BLUE_DARK, CARD_BLUE_TINT = "#2563EB", "#1D4ED8", "#DBEAFE"                  # the blue two-column advice block (load forecast + phase balance)


def forecast_advice_card_html(forecast_values, forecast_times, previous_energy_kwh, horizon_hours, badge, tone, advice=None):
    'Render benefit-versus-loss cards for the Load forecast tab.'
    values = np.asarray(forecast_values, dtype=float)
    if values.size == 0:
        return ""

    energy_kwh = float(np.nansum(values))
    days = horizon_hours // 24 if horizon_hours % 24 == 0 else None
    badge_period = f"{days}-day" if days else f"{horizon_hours}-hour"
    change_pct = ((energy_kwh - previous_energy_kwh) / previous_energy_kwh * 100
                  if previous_energy_kwh >= 1 else float("nan"))
    period = f"{days} day{'s' if days != 1 else ''}" if days else f"{horizon_hours} hours"
    if not np.isfinite(change_pct):
        change_text = "No comparable previous period is available."
    elif abs(change_pct) < 0.5:
        change_text = f"About the same as the previous {period}."
    else:
        direction = "higher" if change_pct > 0 else "lower"
        change_text = f"{abs(change_pct):.0f}% {direction} than the previous {period}."

    esc = lambda value: html_lib.escape(str(value), quote=True)
    advice = advice or {}
    actions = list(advice.get("actions", []))[:3]
    risks = list(advice.get("risks", []))[:3]
    benefit_text = advice.get("managed", "Following the plan can help the site match its power use to the forecast.")
    loss_text = advice.get("ignored", "Unmanaged demand may increase operating pressure or interrupt work.")
    why_text = advice.get("why", "The recommendation is based on the selected forecast and its comparison with the previous period.")
    tone_class = "bad" if tone == "bad" else "good" if tone == "good" else "amber"
    action_icons = ("🏭", "⚡", "🔋", "🕒", "🔌")
    risk_icons = ("💸", "📈", "⚠️", "🏭", "🧾")

    def rows(items, icons):
        rendered = []
        for index, (heading, body) in enumerate(items):
            rendered.append(
                '<div class="load-summary-item">'
                f'<span class="load-summary-emoji">{icons[index % len(icons)]}</span>'
                '<div><div class="load-summary-item-title">'
                f'{esc(heading)}</div><div class="load-summary-item-copy">{esc(body)}</div></div></div>'
            )
        return "".join(rendered) if rendered else '<div class="load-summary-empty">No specific items are flagged for this forecast.</div>'

    peak_value = float(np.nanmax(values))
    peak_time = pd.Timestamp(forecast_times[int(np.nanargmax(values))])
    peak_text = f"Peak forecast: {peak_value:,.1f} kW · {peak_time:%a, %d %b at %H:%M}"
    badge_text = advice.get("badge", badge or "Watch")
    risk_tone = "bad" if tone_class == "bad" else "amber" if tone_class == "amber" else "good"

    return f'''<style>
    html, body {{ margin:0 !important; padding:0 !important; background:transparent !important; overflow:hidden; }}
    * {{ box-sizing:border-box; }}
    .load-summary-wrap {{ display:grid; grid-template-columns:repeat(2,minmax(0,1fr)); gap:12px; margin:0; font-family:Inter,Arial,sans-serif; align-items:stretch; }}
    .load-summary-card {{ position:relative; min-width:0; height:100%; align-self:stretch; background:linear-gradient(155deg,#FFFFFF 0%,#FBFCFF 100%); border:1px solid #E2E8F0; border-radius:14px; padding:11px 14px 9px 15px; box-shadow:0 2px 7px rgba(15,23,42,.045); transition:transform .2s ease,box-shadow .2s ease,border-color .2s ease; animation:loadSummaryEnter .35s ease both; }}
    .load-summary-card::before {{ content:""; position:absolute; inset:0 auto 0 0; width:3px; border-radius:14px 0 0 14px; background:#93C5FD; opacity:.95; transition:width .2s ease; }}
    .load-summary-card:hover,.load-summary-card:focus-within {{ transform:translateY(-3px); border-color:#93C5FD; box-shadow:0 10px 22px -14px rgba(37,99,235,.48),0 3px 8px rgba(15,23,42,.05); }}
    .load-summary-card:hover::before,.load-summary-card:focus-within::before {{ width:5px; }}
    .load-summary-card-head {{ display:flex; justify-content:space-between; align-items:center; gap:8px; }}
    .load-summary-title {{ color:#0F172A; font-size:.76rem; font-weight:800; letter-spacing:.035em; }}
    .load-summary-badge {{ flex:0 0 auto; padding:3px 8px; border-radius:99px; font-size:.66rem; font-weight:700; }}
    .load-summary-badge.bad {{ background:#FFE9ED; color:#BE123C; }} .load-summary-badge.amber {{ background:#FFF4DD; color:#B45309; }} .load-summary-badge.good {{ background:#E8FAF3; color:#047857; }}
    .load-summary-sub {{ margin:3px 0 7px; color:#64748B; font-size:.69rem; }}
    .load-summary-feature {{ display:flex; align-items:flex-start; gap:8px; padding:7px 9px; margin-bottom:5px; border-radius:9px; background:#F3F6FF; color:#1E293B; }}
    .load-summary-feature-icon {{ font-size:1rem; line-height:1.25; }} .load-summary-feature strong {{ display:block; font-size:.79rem; line-height:1.3; color:#1E293B; }}
    .load-summary-feature small {{ display:block; margin-top:1px; font-size:.68rem; line-height:1.35; color:#64748B; }}
    .load-summary-item {{ display:grid; grid-template-columns:19px minmax(0,1fr); gap:6px; padding:6px 0; border-top:1px solid #EEF2F8; }}
    .load-summary-emoji {{ font-size:.88rem; line-height:1.35; }} .load-summary-item-title {{ color:#172033; font-size:.74rem; font-weight:700; margin-bottom:1px; }}
    .load-summary-item-copy {{ color:#526078; font-size:.69rem; line-height:1.4; }} .load-summary-empty {{ padding:8px 0; color:#64748B; font-size:.7rem; }}
    .load-summary-outcome {{ margin-top:7px; padding:8px 10px; border-radius:9px; font-size:.69rem; line-height:1.42; }}
    .load-summary-outcome.benefit {{ background:#ECFDF5; border:1px solid #A7F3D0; color:#065F46; }}
    .load-summary-outcome.loss {{ background:#FFF7ED; border:1px solid #FED7AA; color:#9A3412; }}
    .load-summary-outcome strong {{ display:block; margin-bottom:2px; font-size:.73rem; }}
    .load-summary-reason {{ margin-top:4px; color:#526078; font-size:.67rem; line-height:1.4; }}
    @keyframes loadSummaryEnter {{ from {{ opacity:0; transform:translateY(6px); }} to {{ opacity:1; transform:translateY(0); }} }}
    @media (max-width:760px) {{ .load-summary-wrap {{ grid-template-columns:1fr; }} }}
    @media (prefers-reduced-motion:reduce) {{ .load-summary-card {{ animation:none; transition:none; }} .load-summary-card:hover {{ transform:none; }} }}
    </style>
    <div class="load-summary-wrap">
      <section class="load-summary-card" tabindex="0" aria-label="Actions and company benefits">
        <div class="load-summary-card-head"><div class="load-summary-title">✅ ACTIONS TO TAKE</div><span class="load-summary-badge good">Recommended plan</span></div>
        <div class="load-summary-sub">Based on the selected {esc(badge_period)} load forecast</div>
        <div class="load-summary-feature"><span class="load-summary-feature-icon">📊</span><div><strong>{energy_kwh:,.0f} kWh forecast energy</strong><small>{esc(change_text)}</small></div></div>
        {rows(actions, action_icons)}
        <div class="load-summary-outcome benefit"><strong>Company benefit if you follow the plan</strong>{esc(benefit_text)}<div class="load-summary-reason"><b>Why?</b> {esc(why_text)}</div></div>
      </section>
      <section class="load-summary-card" tabindex="0" aria-label="Possible company losses if no action is taken">
        <div class="load-summary-card-head"><div class="load-summary-title">⚠️ IF YOU DON'T ACT</div><span class="load-summary-badge {risk_tone}">{esc(badge_text)}</span></div>
        <div class="load-summary-sub">Possible effects of not following the forecast plan</div>
        <div class="load-summary-feature"><span class="load-summary-feature-icon">🎯</span><div><strong>{esc(peak_text)}</strong><small>Prepare for this forecast peak.</small></div></div>
        {rows(risks, risk_icons)}
        <div class="load-summary-outcome loss"><strong>Possible company loss if no action is taken</strong>{esc(loss_text)}<div class="load-summary-reason"><b>Why?</b> {esc(why_text)}</div></div>
      </section>
    </div>'''
def build_load_suggestions(forecast_values, forecast_times, previous_energy_kwh):
    """Build simple actions, reasons, benefits and risks from the total-load forecast."""
    forecast_values = np.asarray(forecast_values, dtype=float)
    n_hours = len(forecast_values)
    energy_kwh = float(np.nansum(forecast_values))
    average_kw = float(np.nanmean(forecast_values))
    peak_idx = int(np.nanargmax(forecast_values))
    low_idx = int(np.nanargmin(forecast_values))
    peak, peak_time = float(forecast_values[peak_idx]), pd.Timestamp(forecast_times[peak_idx])
    lowest_kw, lowest_time = float(forecast_values[low_idx]), pd.Timestamp(forecast_times[low_idx])

    energy_change_kwh = energy_kwh - previous_energy_kwh
    energy_change_pct = (energy_change_kwh / previous_energy_kwh * 100
                         if previous_energy_kwh >= 1 else float("nan"))
    trend_direction = (
        "idle" if average_kw < MIN_PHASE_KW else
        "up" if np.isfinite(energy_change_pct) and energy_change_pct > LOAD_CHANGE_PCT else
        "down" if np.isfinite(energy_change_pct) and energy_change_pct < -LOAD_CHANGE_PCT else
        "stable"
    )
    period_text = hours_to_text(n_hours)
    if not np.isfinite(energy_change_pct):
        change_text = "there is no matching past period to compare"
    elif abs(energy_change_pct) < 0.5:
        change_text = f"it is about the same as the previous {period_text}"
    else:
        change_text = f"it is {abs(energy_change_pct):.0f}% {'higher' if energy_change_pct > 0 else 'lower'} than the previous {period_text}"

    format_money = lambda x: f"{x:,.0f}{(' ' + CURRENCY) if CURRENCY else ''}"
    format_time = lambda ts: f"{pd.Timestamp(ts):%a %d %b at %H:%M}"
    shortfall_kw = max(peak - OWN_GENERATION_KW, 0.0) if OWN_GENERATION_KW > 0 else 0.0
    spare_kw = max(OWN_GENERATION_KW - peak, 0.0) if OWN_GENERATION_KW > 0 else 0.0
    over_demand_kw = max(peak - SANCTIONED_DEMAND_KW, 0.0) if SANCTIONED_DEMAND_KW > 0 else 0.0
    demand_use_pct = peak / SANCTIONED_DEMAND_KW * 100 if SANCTIONED_DEMAND_KW > 0 else float("nan")
    capacity_gap_kw = max(shortfall_kw, over_demand_kw)
    extra_energy_cost = max(energy_change_kwh, 0.0) * TARIFF_PER_KWH
    demand_cost = over_demand_kw * DEMAND_CHARGE_PER_KW
    extra_cost = extra_energy_cost + demand_cost
    peak_to_low_gap_kw = max(peak - lowest_kw, 0.0)

    if trend_direction == "idle":
        risks = [
            ("Very low load expected", f"The forecast average is {average_kw:,.1f} kW. Check whether the stop is planned."),
            ("Production may be affected", "If the site should be running, a supply, machine or meter problem may need attention."),
        ]
        actions = [
            ("Confirm the stop is planned", "Check the production plan before switching off equipment."),
            ("Check for a problem", "If the stop is not planned, check the supply, breakers, machines and meter readings."),
        ]
        managed_text = "Confirming a planned stop and switching off equipment that is not needed can avoid waste and create time for maintenance."
        ignored_text = "If the low load is caused by a fault, the problem may go unnoticed and production may remain stopped."
        why_text = f"The forecast average is {average_kw:,.1f} kW, below the {MIN_PHASE_KW:g} kW running level. Check whether this is a planned stop."

    elif trend_direction == "up":
        risks = [
            ("More energy may be needed", f"The forecast is {change_text}. The site may need more grid power or backup supply."),
            ("High demand at the peak", f"The forecast peak is {peak:,.1f} kW around {format_time(peak_time)}."),
            ("Peak may be harder to manage", "If many machines run together, the site may need more power at the same time."),
        ]
        actions = [
            ("Plan the power supply", f"The forecast is {change_text}. Check how much power can come from the site generator and the grid."),
            ("Prepare for the peak", f"Keep enough supply ready for about {peak:,.1f} kW around {format_time(peak_time)}" + (f". The forecast is {shortfall_kw:,.1f} kW above the configured own-generation level." if OWN_GENERATION_KW > 0 and shortfall_kw > 0 else f". This is {over_demand_kw:,.1f} kW above the configured demand limit." if over_demand_kw > 0 else ".")),
            ("Review flexible equipment times", f"The lower-load hour is {format_time(lowest_time)} at {lowest_kw:,.1f} kW. The peak is {format_time(peak_time)} at {peak:,.1f} kW. Check whether any suitable equipment planned near the peak can safely run in the lower-load period instead. The {peak_to_low_gap_kw:,.1f} kW gap is only a difference between forecast hours; it is not the amount of equipment that can be moved."),
        ]
        if HAS_BATTERY:
            actions[2] = ("Plan battery use", f"Review charging around the lower-load hour ({format_time(lowest_time)}) and discharge near the peak ({format_time(peak_time)}), within battery limits.")
        managed_text = ("Planning power supply early and, where production allows, moving suitable flexible work away from the peak may lower peak demand. "
                        "This can reduce supply pressure and may lower demand charges if the tariff includes them. Total energy use may stay the same.")
        if extra_cost > 0:
            ignored_text = f"Using more energy may add about {format_money(extra_energy_cost)}. The configured peak or demand limit may also add about {format_money(demand_cost)}. If supply is not enough, production may be affected."
        elif capacity_gap_kw > 0:
            ignored_text = f"The forecast peak is {capacity_gap_kw:,.1f} kW above the configured supply or demand limit. Extra supply may be needed, and production may be affected if it is not available."
        else:
            ignored_text = "The site may need more grid or backup power. If supply is not ready for the forecast peak, production may be affected."
        why_text = (f"Forecast energy is {change_text}. The highest load is {peak:,.1f} kW around {format_time(peak_time)}. "
                    f"The lower-load hour is {lowest_kw:,.1f} kW around {format_time(lowest_time)}. These are forecast hours, not a confirmed machine schedule. "
                    "Check which equipment can safely run at a different time before changing the schedule.")

    elif trend_direction == "down":
        risks = [
            ("Lower energy use expected", f"The forecast is {change_text}. Check whether this matches the production plan."),
            ("Generation may be higher than needed", "Running more generators than needed can waste fuel. Check the actual supply and process demand."),
            ("A fault may go unnoticed", "A sudden drop can also come from a machine, supply or meter problem."),
        ]
        actions = [
            ("Check the lower demand", f"The forecast is {change_text}. Confirm that the lower load is planned and not caused by a fault."),
            ("Match generation to demand", "If the lower demand is confirmed, review generator output and keep enough supply for the process."),
            ("Keep essential equipment ready", f"The forecast peak is {peak:,.1f} kW around {format_time(peak_time)}. Do not reduce supply below the site's needs."),
        ]
        managed_text = "If the lower demand is planned, matching generation to the real need may reduce unnecessary fuel use. Checking the reason for the drop can also help find faults early."
        ignored_text = "Generators may run above the site's needs and use extra fuel. If the drop is caused by a fault, production problems may continue without being noticed."
        why_text = f"Forecast energy is {change_text}, and the peak is {peak:,.1f} kW. First confirm that the lower demand is planned. Then adjust generation only after checking the process needs."

    else:
        risks = [
            ("Energy use is steady", f"The forecast is {change_text} over the selected period."),
            ("A peak still needs planning", f"The highest load is {peak:,.1f} kW around {format_time(peak_time)}."),
            ("Machines starting together can raise demand", "Several large machines starting at once can create a short peak."),
        ]
        actions = [
            ("Keep the current supply plan", f"Forecast energy is {energy_kwh:,.0f} kWh, and {change_text}. Keep checking that supply matches demand."),
            ("Prepare for the peak", f"Keep enough supply ready for {peak:,.1f} kW around {format_time(peak_time)}."),
            ("Review machine start times", "Where the process allows, start large machines at different times to avoid one sudden rise in demand."),
        ]
        managed_text = "Keeping supply ready and spacing out large machine starts can help avoid sudden demand peaks and support steady operation."
        ignored_text = "If several large machines start together, demand may rise above the planned level and put extra pressure on the supply."
        why_text = f"Forecast energy is {change_text}, but the peak is still {peak:,.1f} kW around {format_time(peak_time)}. The current plan can continue, with attention to the peak and machine start times."

    big_drop = trend_direction == "down" and np.isfinite(energy_change_pct) and energy_change_pct < -10
    tone = "bad" if capacity_gap_kw > 0 else "amber" if trend_direction in ("up", "down") else "good"
    return dict(
        kind=trend_direction,
        tone=tone,
        badge="Attention" if capacity_gap_kw > 0 or big_drop else "Watch" if trend_direction in ("up", "down") else "Normal",
        risks=risks,
        actions=actions,
        managed=managed_text,
        ignored=ignored_text,
        why=why_text,
        headline=f"Forecasted peak demand: {peak:,.1f} kW around {peak_time:%A, %H:%M}",
    )

def phase_analysis_card_html(forecast_phase_kw, quiet_busy_window=None, diagnosis=None):
    'Show the suggested kW phase shift, its possible company benefit, and the risk of doing nothing.'
    esc = lambda value: html_lib.escape(str(value), quote=True)
    labels = dict(zip(PHASE_CODES, PHASE_LABELS))
    phase_values = forecast_phase_kw[PHASE_CODES].astype(float)
    matrix = phase_values.to_numpy(dtype=float)
    valid = np.isfinite(matrix).all(axis=1)
    hourly_avg = np.full(len(matrix), np.nan, dtype=float)
    hourly_avg[valid] = matrix[valid].mean(axis=1)
    loaded = valid & (hourly_avg >= MIN_PHASE_KW)
    deviations = np.full_like(matrix, np.nan, dtype=float)
    deviations[loaded] = (matrix[loaded] / hourly_avg[loaded, None] - 1.0) * 100.0
    hourly_imbalance = np.full(len(matrix), np.nan, dtype=float)
    hourly_imbalance[loaded] = np.max(np.abs(deviations[loaded]), axis=1)

    n_hours = len(phase_values)
    n_alarm_hours = int(np.sum(loaded & (hourly_imbalance > ALARM_LIMIT_PCT)))
    n_days = n_hours // 24 if n_hours % 24 == 0 else None
    badge_period = f"{n_days}-day" if n_days else f"{n_hours}-hour"
    candidates = np.flatnonzero(loaded)
    worst_pos = int(candidates[np.nanargmax(hourly_imbalance[candidates])]) if len(candidates) else None

    # Past readings explain the likely reason. They are not used to calculate the future kW transfer.
    if diagnosis and not diagnosis.get("empty", False):
        current_pct = float(diagnosis.get("mean_I", float("nan")))
        voltage_pct = float(diagnosis.get("mean_V", float("nan")))
        current_above = float(diagnosis.get("pct_current_above", float("nan")))
        voltage_above = float(diagnosis.get("pct_voltage_above", float("nan")))
        max_voltage = float(diagnosis.get("max_voltage_imb", float("nan")))
        start_date, end_date = diagnosis.get("history_start"), diagnosis.get("history_end")
        date_range = (f"{pd.Timestamp(start_date):%d %b %Y} to {pd.Timestamp(end_date):%d %b %Y}"
                      if start_date is not None and end_date is not None else "the available history")
        if np.isfinite(current_pct) and np.isfinite(voltage_pct):
            if current_pct > ALARM_LIMIT_PCT and voltage_pct <= ALARM_LIMIT_PCT:
                current_share = (f"It was above the {ALARM_LIMIT_PCT:.0f}% limit in {current_above:.1f}% of valid records. "
                                 if np.isfinite(current_above) else "It often crossed the set limit. ")
                if np.isfinite(voltage_above) and voltage_above == 0 and np.isfinite(max_voltage):
                    voltage_share = f"Voltage averaged {voltage_pct:.1f}% and stayed below the limit in all valid records (highest {max_voltage:.1f}%). "
                else:
                    voltage_share = f"Voltage averaged {voltage_pct:.1f}% and stayed within the set limit. "
                diagnostic_text = (f"Past readings ({date_range}) show average current imbalance of {current_pct:.1f}%. "
                                   f"{current_share}{voltage_share}The current imbalance also closely matched the phase load imbalance. "
                                   "This suggests uneven phase loading is the main issue. Voltage variation is less likely to be the main cause.")
            elif voltage_pct > ALARM_LIMIT_PCT and current_pct <= ALARM_LIMIT_PCT:
                diagnostic_text = (f"Past readings ({date_range}) show average voltage imbalance of {voltage_pct:.1f}%, above the "
                                   f"{ALARM_LIMIT_PCT:.0f}% limit. Average current imbalance was {current_pct:.1f}%. "
                                   "Ask an electrician to check the incoming supply before moving loads.")
            elif current_pct > ALARM_LIMIT_PCT and voltage_pct > ALARM_LIMIT_PCT:
                diagnostic_text = (f"Past readings ({date_range}) show current imbalance of {current_pct:.1f}% and voltage "
                                   f"imbalance of {voltage_pct:.1f}%. Both are above the {ALARM_LIMIT_PCT:.0f}% limit. "
                                   "Check the incoming supply and the phase loads together.")
            else:
                diagnostic_text = (f"Past readings ({date_range}) show current imbalance of {current_pct:.1f}% and voltage "
                                   f"imbalance of {voltage_pct:.1f}%. Both stayed within the {ALARM_LIMIT_PCT:.0f}% limit.")
        else:
            diagnostic_text = "There are not enough past current and voltage readings to compare the three phases."
    else:
        diagnostic_text = "There are not enough valid past readings to check current and voltage balance."

    impacts = []
    if worst_pos is None:
        forecast_line = "No loaded three-phase forecast hour is available."
        load_detail = "No phase kW values are available for a shift estimate."
        badge_text, badge_class = "No loaded forecast", "good"
        actions = [
            ("No kW shift suggested", "The forecast has no loaded hour to compare across the three phases."),
            ("Keep checking the phases", "Review the Red, Yellow and Blue kW values when new meter data arrives.")
        ]
        benefit_text = "There is no clear kW shift to recommend from this forecast. Keeping new loads balanced can help prevent a future imbalance."
        loss_text = "There is not enough loaded forecast data to identify a phase risk. Missing data may hide a problem, so check the meter readings."
        reason_text = "A shift can only be estimated when all three phase forecasts have valid values and the site is expected to be running."
    else:
        stamp = phase_values.index[worst_pos]
        values = matrix[worst_pos]
        dev = deviations[worst_pos]
        avg_kw = float(hourly_avg[worst_pos])
        high_idx, low_idx = int(np.argmax(dev)), int(np.argmin(dev))
        high_phase, low_phase = labels[PHASE_CODES[high_idx]], labels[PHASE_CODES[low_idx]]
        high_kw, low_kw = float(values[high_idx]), float(values[low_idx])
        high_dev, low_dev = float(dev[high_idx]), float(dev[low_idx])
        worst_imbalance = float(hourly_imbalance[worst_pos])
        is_out_of_balance = worst_imbalance > ALARM_LIMIT_PCT
        phase_load_detail = " · ".join(f"{labels[code]} {float(values[i]):,.1f} kW" for i, code in enumerate(PHASE_CODES))
        forecast_line = (
            f"Worst forecast hour: {stamp:%a %d %b %H:%M}. {high_phase} is highest "
            f"({high_kw:,.1f} kW, {high_dev:+.1f}%). {low_phase} is lowest "
            f"({low_kw:,.1f} kW, {low_dev:+.1f}%)."
            if is_out_of_balance else
            f"Forecast phase loading at {stamp:%a %d %b %H:%M} is within the {ALARM_LIMIT_PCT:.0f}% limit."
        )
        load_detail = phase_load_detail + "."
        badge_text, badge_class = ("Review phase loading", "amber") if is_out_of_balance else ("Within limit", "good")

        # Estimate a kW shift that would bring the forecast phase values closer to their average.
        transfers = []
        if is_out_of_balance:
            donors = [[i, max(float(values[i]) - avg_kw, 0.0)] for i in range(3)]
            receivers = [[i, max(avg_kw - float(values[i]), 0.0)] for i in range(3)]
            donors.sort(key=lambda item: item[1], reverse=True)
            receivers.sort(key=lambda item: item[1], reverse=True)
            di = ri = 0
            while di < len(donors) and ri < len(receivers):
                donor_idx, surplus = donors[di]
                receiver_idx, deficit = receivers[ri]
                amount_kw = min(surplus, deficit)
                if amount_kw > 0.05:
                    transfers.append((labels[PHASE_CODES[donor_idx]], labels[PHASE_CODES[receiver_idx]], amount_kw))
                donors[di][1] -= amount_kw
                receivers[ri][1] -= amount_kw
                if donors[di][1] <= 1e-8:
                    di += 1
                if receivers[ri][1] <= 1e-8:
                    ri += 1

        reason_text = (f"At {stamp:%a %d %b %H:%M}, {high_phase} is {high_kw:,.1f} kW and {low_phase} is "
                       f"{low_kw:,.1f} kW. The estimate compares each phase with the average of all three phases. "
                       "It is a forecast estimate, not a confirmed amount of equipment that can be moved.")
        if is_out_of_balance and transfers:
            transfer_text = "; ".join(f"{donor} → {receiver}: about {amount:.1f} kW" for donor, receiver, amount in transfers)
            actions = [
                ("Check the highest phase", f"{high_phase} has the highest forecast load at {stamp:%a %d %b %H:%M}. {load_detail}"),
                ("Review suitable equipment", f"The forecast balance estimate is {transfer_text}. Check whether suitable single-phase equipment can safely be moved. This is not a confirmed amount of movable load."),
                ("Check after any change", "After an approved change, measure all three phase loads, currents and voltages again.")
            ]
            total_shift = sum(item[2] for item in transfers)
            benefit_text = (f"If suitable equipment with a total load of about {total_shift:.1f} kW can be moved safely, the phase loads may become closer at this forecast hour. "
                            "This may reduce phase imbalance. Total site energy use may stay the same.")
            loss_text = (f"If no suitable load is moved, {high_phase} is still forecast to carry more kW than the other phases. "
                         "If actual current goes above cable or breaker limits, cables may heat up or protection may trip. Check live readings and equipment ratings.")
            impacts = [
                ("Uneven phase load", f"{high_phase} is forecast at {high_kw:,.1f} kW, above the three-phase average of {avg_kw:,.1f} kW."),
                ("Cable heating risk", "If actual current is too high for the cable, it can heat up. Check the live current and cable rating."),
                ("Breaker trip risk", "A breaker can trip if actual current goes above its rating. Check the circuit before making a change.")
            ]
        elif is_out_of_balance:
            actions = [
                ("Review the forecast phase loads", forecast_line),
                ("Check the real circuits", "The forecast cannot show which circuit can be moved. Ask a qualified electrician to check the loads and ratings."),
                ("Measure after any change", "Check all three phase loads, currents and voltages after an approved change.")
            ]
            benefit_text = "Checking the real phase loads and suitable equipment may help reduce the imbalance. This forecast does not confirm a safe transfer amount."
            loss_text = f"The predicted difference between {high_phase} and {low_phase} may remain. If actual current goes above cable or breaker limits, cables may heat up or protection may trip."
            impacts = [
                ("Uneven phase load", forecast_line),
                ("Cable heating risk", "Check actual current and cable ratings before changing loads."),
                ("Breaker trip risk", "Protection may trip if actual current exceeds the breaker rating.")
            ]
        else:
            actions = [
                ("No kW shift needed now", f"The largest forecast phase difference is {worst_imbalance:.1f}%, within the {ALARM_LIMIT_PCT:.0f}% limit."),
                ("Balance new loads", "Share new single-phase loads across Red, Yellow and Blue."),
                ("Keep checking the phases", "Review the phase kW values when new meter data arrives.")
            ]
            benefit_text = "No phase transfer is needed under the selected forecast. Keeping new loads balanced can help the phases stay within the limit."
            loss_text = "There is no major phase imbalance forecast now. If new loads are added to one phase only, an imbalance may develop later."
            impacts = [
                ("No large difference forecast", f"The highest phase difference is {worst_imbalance:.1f}%, within the {ALARM_LIMIT_PCT:.0f}% limit."),
                ("Watch new connections", "Adding several loads to one phase can make it higher than the others."),
            ]

    def render_rows(items, icons):
        return "".join(
            '<div class="phase-summary-item">'
            f'<span class="phase-summary-icon">{icons[i % len(icons)]}</span>'
            f'<div><div class="phase-summary-item-title">{esc(title)}</div>'
            f'<div class="phase-summary-item-copy">{esc(copy)}</div></div></div>'
            for i, (title, copy) in enumerate(items)
        )

    summary_sub = f"Selected {badge_period} forecast · phase load in kW"
    impact_sub = f"What may happen if the phase load is not balanced"
    return f'''<style>
    .phase-summary-wrap {{ display:grid; grid-template-columns:repeat(2,minmax(0,1fr)); gap:12px; align-items:stretch; margin:2px 0 8px; font-family:Inter,Arial,sans-serif; }}
    .phase-summary-card {{ min-width:0; height:100%; display:flex; flex-direction:column; background:linear-gradient(155deg,#fff 0%,#fbfcff 100%); border:1px solid #dfe7f3; border-left:3px solid #93c5fd; border-radius:14px; padding:12px 14px 10px; box-shadow:0 2px 7px rgba(15,23,42,.045); transition:transform .2s ease,box-shadow .2s ease,border-color .2s ease; animation:phaseSummaryEnter .35s ease both; }}
    .phase-summary-card:hover,.phase-summary-card:focus-within {{ transform:translateY(-3px); border-color:#93c5fd; box-shadow:0 10px 22px -14px rgba(37,99,235,.48),0 3px 8px rgba(15,23,42,.05); }}
    .phase-summary-head {{ display:flex; justify-content:space-between; align-items:center; gap:8px; }}
    .phase-summary-title {{ color:#0f172a; font-size:.78rem; font-weight:800; letter-spacing:.035em; }}
    .phase-summary-badge {{ flex:0 0 auto; padding:3px 8px; border-radius:99px; font-size:.66rem; font-weight:700; }}
    .phase-summary-badge.amber {{ background:#fff4dd; color:#b45309; }} .phase-summary-badge.good {{ background:#e8faf3; color:#047857; }}
    .phase-summary-sub {{ margin:3px 0 7px; color:#64748b; font-size:.7rem; }}
    .phase-summary-feature {{ padding:8px 10px; border-radius:9px; background:#f3f6ff; margin-bottom:6px; }}
    .phase-summary-feature strong {{ display:block; font-size:.79rem; color:#172033; line-height:1.35; }}
    .phase-summary-feature span {{ display:block; margin-top:2px; font-size:.7rem; line-height:1.45; color:#526078; }}
    .phase-summary-context {{ padding:8px 10px; border-radius:9px; background:#f8fafc; border:1px solid #e8edf5; margin:0 0 5px; }}
    .phase-summary-context strong {{ display:block; color:#172033; font-size:.72rem; line-height:1.35; }}
    .phase-summary-context span {{ display:block; color:#526078; font-size:.68rem; line-height:1.42; margin-top:2px; }}
    .phase-summary-item {{ display:grid; grid-template-columns:20px minmax(0,1fr); gap:6px; padding:7px 0; border-top:1px solid #edf1f7; }}
    .phase-summary-icon {{ font-size:.9rem; line-height:1.35; }}
    .phase-summary-item-title {{ color:#172033; font-size:.74rem; font-weight:700; margin-bottom:2px; }}
    .phase-summary-item-copy {{ color:#526078; font-size:.69rem; line-height:1.45; }}
    .phase-summary-outcome {{ padding:8px 10px; margin-top:7px; border-radius:9px; font-size:.69rem; line-height:1.45; }}
    .phase-summary-outcome.benefit {{ background:#ecfdf5; border:1px solid #a7f3d0; color:#065f46; }}
    .phase-summary-outcome.loss {{ background:#fff7ed; border:1px solid #fed7aa; color:#9a3412; }}
    .phase-summary-outcome strong {{ display:block; margin-bottom:2px; font-size:.73rem; }}
    .phase-summary-reason {{ margin-top:4px; color:#526078; font-size:.67rem; line-height:1.42; }}
    @keyframes phaseSummaryEnter {{ from {{ opacity:0; transform:translateY(6px); }} to {{ opacity:1; transform:translateY(0); }} }}
    @media (max-width:760px) {{ .phase-summary-wrap {{ grid-template-columns:1fr; }} }}
    @media (prefers-reduced-motion:reduce) {{ .phase-summary-card {{ animation:none; transition:none; }} .phase-summary-card:hover {{ transform:none; }} }}
    </style>
    <div class="phase-summary-wrap">
      <section class="phase-summary-card" tabindex="0" aria-label="Actions and company benefits from phase balancing">
        <div class="phase-summary-head"><div class="phase-summary-title">✅ ACTIONS TO TAKE</div><span class="phase-summary-badge {badge_class}">kW forecast plan</span></div>
        <div class="phase-summary-sub">{esc(summary_sub)}</div>
        <div class="phase-summary-feature"><strong>{esc(forecast_line)}</strong><span>{esc(load_detail)} Hours above the {ALARM_LIMIT_PCT:.0f}% limit: {n_alarm_hours} of {n_hours}.</span></div>
        <div class="phase-summary-context"><strong>🔎 Why this recommendation?</strong><span>{esc(diagnostic_text)} {esc(reason_text)}</span></div>
        {render_rows(actions, ("🔎", "⚖️", "✅"))}
        <div class="phase-summary-outcome benefit"><strong>Company benefit if you follow the plan</strong>{esc(benefit_text)}</div>
      </section>
      <section class="phase-summary-card" tabindex="0" aria-label="Possible company losses if phase imbalance is not addressed">
        <div class="phase-summary-head"><div class="phase-summary-title">⚠️ IF YOU DON'T ACT</div><span class="phase-summary-badge {badge_class}">{esc(badge_text)}</span></div>
        <div class="phase-summary-sub">{esc(impact_sub)}</div>
        <div class="phase-summary-feature"><strong>{esc(forecast_line)}</strong><span>{esc(load_detail)} This is a forecast, not a live meter reading.</span></div>
        {render_rows(impacts, ("📉", "⚡", "🚨"))}
        <div class="phase-summary-outcome loss"><strong>Possible company loss if no action is taken</strong>{esc(loss_text)}<div class="phase-summary-reason"><b>Why?</b> {esc(reason_text)}</div></div>
      </section>
    </div>'''
# ------------------------------------------------------------------ FORECAST HELPERS (cached wrappers around gridsense_forecast.py)
def hours_to_text(hours):
    hours = int(hours); return f"{hours} h" if hours < 48 else f"{hours / 24:.0f} days"


def average_by_clock_hour(values, times):
    """Compare each clock-hour (00:00–01:00 ... 23:00–00:00) across all forecast days.

    Each hourly forecast point is one hour long, so its average kW is also the
    energy for that hour in kWh. The profile averages matching clock-hours,
    not consecutive blocks within the 7-day series.
    """
    series = pd.Series(np.asarray(values, dtype=float), index=pd.DatetimeIndex(times))
    rows = []
    for hour in range(24):
        points = series[series.index.hour == hour].dropna()
        if points.empty:
            continue
        day_details = "<br>".join(
            f"{stamp:%a %d %b}: {value:,.1f} kWh"
            for stamp, value in points.items()
        )
        rows.append({
            "Hour": hour,
            "Window": f"{hour:02d}:00–{(hour + 1) % 24:02d}:00",
            "Avg kW": float(points.mean()),
            "Avg kWh/day": float(points.mean()),  # one-hour window: kW × 1 hour
            "Days": int(points.count()),
            "Daily detail": day_details,
        })
    return pd.DataFrame(rows)


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
def forecast_total_load(training_values, steps_ahead, season_length, fixed_smoothing):
    return gf.forecast_one_series(training_values.to_numpy(), steps_ahead, season_length, fixed_smoothing)


@st.cache_data(show_spinner="Testing the model on the last 168 hours…")
def run_total_load_holdout(total_measured_hourly, total_filled_hourly, season_length, fixed_smoothing, max_training_hours):
    return gf.holdout_test_total_load(total_measured_hourly, total_filled_hourly, season_length, fixed_smoothing, max_training_hours)


@st.cache_data(show_spinner="Training the model on Red, Yellow and Blue ")
def forecast_phases_cached(history_window, steps_ahead, season_length):
    return gf.forecast_three_phases(history_window, steps_ahead, season_length, None)


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


@st.cache_data(show_spinner="Building the load / current / voltage table…")
def load_phase_parameter_frame(minute_table, running_kw_threshold, rule, min_readings_per_hour):
    return build_phase_parameter_frame(minute_table, running_kw_threshold, rule, min_readings_per_hour)


def show_not_enough_data_note():
    note_box("<b>Not enough hourly data.</b> The forecasts need at least 3 days of hours that each have enough readings. "
         "For the weekly cycle and the backtest, about 5 weeks are better.", "amber")

# ------------------------------------------------------------------ UNUSED HELPERS (kept for reference, the page does not call them today)


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
        signal_choice = st.selectbox("Signal", ["Total load (kW)", "Phase load R / Y / B (kW)", "Phase current (A)", "Voltage (V)"], label_visibility="collapsed")
        cols = {"Total load (kW)": ["KW"], "Phase load R / Y / B (kW)": PHASE_KW_COLS, "Phase current (A)": CURRENT_COLS, "Voltage (V)": VOLTAGE_COLS}[signal_choice]
        ten_minute_means = minute_rows.set_index("Timestamps")[cols].resample("10min").mean(); fig = go.Figure()
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
    fixed_smoothing = gf.NOTEBOOK_SMOOTHING_PARAMS
    max_training_hours = len(total_filled_hourly)                       # learns from all history
    load_training_text = "all available history"
    # train once for 7 days ahead; the page only slices what is asked for
    total_forecast, total_forecast_label, total_fit_ok = forecast_total_load(total_filled_hourly.iloc[-max_training_hours:], MAX_FORECAST_HOURS, load_season_hours, fixed_smoothing)
    if not total_fit_ok: note_box(f"The model could not be fitted ({total_forecast_label}); the forecast is a flat line. Try the other model or a different cycle.", "amber")
    horizon_hours = FORECAST_LENGTH_HOURS[load_length_choice]; forecast_values = total_forecast[:horizon_hours]; forecast_times = future_hours[:horizon_hours]
    last_actual_kw = float(total_filled_hourly.iloc[-1]); end_value_kw = float(forecast_values[-1])
    change_text = f"{abs(end_value_kw - last_actual_kw) / last_actual_kw * 100:.1f} %" if last_actual_kw >= 1 else f"{abs(end_value_kw - last_actual_kw):.1f} kW"
    arrow = "▲" if end_value_kw >= last_actual_kw else "▼"
    forecast_energy_kwh, forecast_avg_kw, forecast_peak_kw = float(forecast_values.sum()), float(forecast_values.mean()), float(forecast_values.max()); previous_energy_kwh = float(total_filled_hourly.iloc[-horizon_hours:].sum())
    # 7-day test (cached): its errors give the width of the forecast interval (used by the cards, the block chart and the forecast chart; the Model check below reuses the test)
    holdout_result = run_total_load_holdout(total_measured_hourly, total_filled_hourly, load_season_hours, fixed_smoothing, max_training_hours)
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
        # Recurring peak hour: compare the SAME clock-hour across all forecast days.
        hourly_profile = average_by_clock_hour(forecast_values, forecast_times)
        if hourly_profile.empty:
            with energy_kpi_columns[4]:
                kpi_card("Highest recurring hour", "No data", "No hourly forecast values available.", "", "🕒")
        else:
            peak_hour_row = hourly_profile.loc[hourly_profile["Avg kW"].idxmax()]
            peak_hour_window = str(peak_hour_row["Window"])
            peak_hour_kw = float(peak_hour_row["Avg kW"])
            peak_hour_days = int(peak_hour_row["Days"])
            with energy_kpi_columns[4]:
                kpi_card("Highest recurring hour", f"{peak_hour_kw:,.1f} kW",
                         f"{peak_hour_window} · {peak_hour_kw:,.1f} kWh/day average · {peak_hour_days} forecast days",
                         "warn", "⏰")

        st.markdown(
            '<div class="ph" style="margin:.45rem 0 .2rem"><b>Daily demand by time of day</b></div>',
            unsafe_allow_html=True,
        )
        st.caption("Each bar compares the same one-hour slot across all forecast days. The highlighted bar is the recurring peak hour.")
        if not hourly_profile.empty:
            peak_position = int(hourly_profile["Avg kW"].to_numpy().argmax())
            bar_colors = [WARN_COLOR if i == peak_position else ACCENT for i in range(len(hourly_profile))]
            customdata = [
                [int(row.Days), row["Daily detail"]]
                for _, row in hourly_profile.iterrows()
            ]
            fig = go.Figure(go.Bar(
                x=hourly_profile["Window"],
                y=hourly_profile["Avg kW"],
                marker_color=bar_colors,
                name="Average hourly demand",
                text=["PEAK" if i == peak_position else "" for i in range(len(hourly_profile))],
                textposition="outside",
                cliponaxis=False,
                customdata=customdata,
                hovertemplate=(
                    "<b>%{x}</b><br>"
                    "Average power across forecast days: <b>%{y:,.1f} kW</b><br>"
                    "Average energy for this 1-hour slot: <b>%{y:,.1f} kWh/day</b><br>"
                    "Forecast days included: %{customdata[0]}<br><br>"
                    "%{customdata[1]}<extra></extra>"
                ),
            ))
            peak_label = f"{peak_hour_window} ({peak_hour_kw:,.1f} kW average)"
            fig.update_layout(
                hovermode="closest",
                title=f"Recurring hourly demand · highest: {peak_label}",
            )
            fig.update_xaxes(type="category", title="Same one-hour slot on each forecast day", tickangle=-45)
            fig.update_yaxes(title="Average power across forecast days (kW)", rangemode="tozero")
            show_figure(style_figure(fig, 330, False, top=65))

    # ONE suggestions block (fixed wording from the reference images, numbers from the forecast)
    load_advice = build_load_suggestions(forecast_values, forecast_times, previous_energy_kwh)                     # only used for the badge and the border colour
    # Render this HTML summary in an isolated iframe so Streamlit never displays
    # the markup as literal code; the card CSS and hover animation remain active.
    summary_html = forecast_advice_card_html(
        forecast_values,
        forecast_times,
        previous_energy_kwh,
        horizon_hours,
        load_advice["badge"],
        load_advice["tone"],
        advice=load_advice,
    )
    # st.html renders the summary at its natural content height (no fixed iframe
    # height), removing the unused space before the forecast chart. The fallback
    # keeps compatibility with older Streamlit versions.
    if hasattr(st, "html"):
        st.html(summary_html)
    else:
        st.components.v1.html(summary_html, height=310, scrolling=False)

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
    control_columns = st.columns([1, 1.3, 1])
    phase_model_name = "Exponential Smoothing"
    phase_length_choice = control_columns[0].selectbox("Forecast length", list(FORECAST_LENGTH_HOURS), index=6, key="ph_len")
    phase_training_days = None                                # fixed learning window (the dropdown was removed)
    phase_season_hours = SEASON_LENGTH_HOURS[control_columns[1].selectbox("Seasonal cycle", list(SEASON_LENGTH_HOURS), index=6, key="ph_season")]
    phase_history_choice = control_columns[2].selectbox("History shown", list(HISTORY_SHOWN_HOURS), index=2, key="ph_hist")
    max_training_hours = len(phase_filled_hourly); horizon_hours = FORECAST_LENGTH_HOURS[phase_length_choice]
    # forecast Red, Yellow and Blue up to 7 days ahead; display slices the selected horizon
    phase_forecast_array, phase_labels_by_series, phase_fit_ok = forecast_phases_cached(phase_filled_hourly.iloc[-max_training_hours:], MAX_FORECAST_HOURS, phase_season_hours)
    forecast_phase_kw = pd.DataFrame(phase_forecast_array[:horizon_hours], index=future_hours[:horizon_hours], columns=PHASE_CODES); forecast_table = analyse_balance(forecast_phase_kw.to_numpy())

    TEST_HOURS = 168                                                              # all test data = the last 168 hours (7 days)
    if not phase_fit_ok: note_box(f"A model fit failed for {', '.join(p for p in PHASE_CODES if phase_labels_by_series[p].startswith('failed'))} - that phase repeats its last value. Try the other model or cycle.", "amber")

    # forecast interval per colour: from the same 7-day test the Model check below shows (cached, so no extra training)
    phase_hold_pred = None
    if len(phase_filled_hourly) >= TEST_HOURS + 2 * phase_season_hours:
        hold_index = phase_filled_hourly.index[-TEST_HOURS:]; hold_array, _, _ = forecast_phases_cached(phase_filled_hourly.iloc[:-TEST_HOURS].iloc[-max_training_hours:], TEST_HOURS, phase_season_hours)
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
    reset_kpi_counter(); kpi_columns = st.columns(4)
    with kpi_columns[0]: kpi_card("Imbalance now", "idle" if not latest_status["loaded"] else f"{latest_status['imb']:.1f} %", "machine idle" if not latest_status["loaded"] else f"{latest_status['worst']} {latest_status['direction']} · limit {ALARM_LIMIT_PCT:g} %", "good" if latest_status["status"] == "OK" else "warn" if latest_status["status"] == "ALARM" else "info", "⚖️")
    with kpi_columns[1]:
        if diagnosis: kpi_card("Average imbalance · measured", f"{diagnosis['mean_imb']:.1f} %","","", "📏")
        else: kpi_card("Average imbalance · measured", "–", "too few loaded minutes", "", "📏")
    with kpi_columns[2]: kpi_card("Forecast alarm hours", f"{n_alarm_hours}", f"of {horizon_hours} hours", "", "🚨", trend=np.nan_to_num(forecast_imbalance_pct), trend_color=ACCENT)
    with kpi_columns[3]: kpi_card("First forecast alarm", "none" if first_alarm is None else f"{first_alarm:%a %H:%M}", "no alarm expected" if first_alarm is None else f"{first_alarm:%d %b} · in {(first_alarm - last_data_time).total_seconds() / 3600:.0f} h", "good" if first_alarm is None else "warn", "⏰")
    st.write("")

    # ---- 3-phase action / impact cards (shown above the phase forecast chart)
    # best time to move loads = quietest 3-hour window
    quiet_busy_window = gr.find_quietest_and_busiest_windows(forecast_phase_kw.sum(axis=1).to_numpy(), forecast_phase_kw.index)
    phase_summary_html = phase_analysis_card_html(
        forecast_phase_kw,
        quiet_busy_window=quiet_busy_window,
        diagnosis=diagnosis,
    )
    try:
        st.html(phase_summary_html)
    except AttributeError:
        st.markdown(phase_summary_html, unsafe_allow_html=True)
    st.write("")

    # ---- one chart: measured hours + forecast per phase; an X marks the hours when THAT phase is the one out of balance
    with panel("Phase load · measured and forecast", f"hourly means · {phase_model_name} · learns from all available history · forecast starts after {last_data_time:%d %b %Y %H:%M} · shaded band = {CONFIDENCE_TEXT} interval · ✕ = alarm hour on the phase that is out of balance", key="ph_main"):
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

        def driver_name():
            """Whichever of current / voltage is more imbalanced in the measured history."""
            if not diagnosis: return "n/a"
            return "Current" if diagnosis["mean_I"] >= diagnosis["mean_V"] else "Voltage"
        for j, p in enumerate(PHASE_CODES, 1):
            phase_alarm_mask = forecast_table.Alarm.to_numpy() & (worst_phase_names == PHASE_LABELS[j - 1])
            if not phase_alarm_mask.any(): continue
            hour_average_kw = forecast_phase_kw.mean(axis=1).to_numpy()
            hour_dev_pct = (forecast_phase_kw[p].to_numpy() / np.where(hour_average_kw > 0, hour_average_kw, np.nan) - 1) * 100
            fig.add_scatter(x=forecast_phase_kw.index[phase_alarm_mask], y=forecast_phase_kw[p].to_numpy()[phase_alarm_mask], mode="markers", name="Alarm (forecast)", legendgroup="alarm", showlegend=not legend_added,
                          marker=dict(symbol="x", size=9, color=WARN_COLOR),
                          customdata=np.c_[forecast_imbalance_pct[phase_alarm_mask], forecast_table.Direction.to_numpy()[phase_alarm_mask], np.full(phase_alarm_mask.sum(), driver_name()), np.round(hour_dev_pct[phase_alarm_mask], 1)],
                          hovertemplate="%{y:.1f} kW · %{customdata[3]}&nbsp;% vs 3-phase average this hour<br>Imbalance %{customdata[0]:.1f} % (%{customdata[1]}) · %{customdata[2]}<extra>Alarm (forecast)</extra>", row=j, col=1)
            legend_added = True
        fig.add_shape(type="line", xref="x", yref="paper", x0=last_data_time, x1=last_data_time, y0=0, y1=1, line=dict(color=MUTED_COLOR, width=1.2, dash="dot"))
        fig.update_yaxes(title_text="kW")
        style_figure(fig, 640, legend_y=1.0, top=110); fig.update_annotations(font=dict(size=13, color=TEXT_COLOR)); show_figure(fig)

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
                forecast_deviation_table = deviation_from_average_pct(forecast_phase_kw, PHASE_CODES); fig = go.Figure()
                for n_, line_color in zip(PHASE_LABELS, phase_color_list): fig.add_scatter(x=forecast_deviation_table.index, y=forecast_deviation_table[n_], name=n_, line=dict(color=line_color, width=2), connectgaps=True)
                fig.add_hline(y=0, line_color=MUTED_COLOR, line_width=1); fig.add_hrect(y0=-ALARM_LIMIT_PCT, y1=ALARM_LIMIT_PCT, fillcolor=hex_to_rgba(GOOD_COLOR, .08), line_width=0)
                fig.update_layout(title="Forecast · which phase is off: load vs the 3-phase average", yaxis_title="% above / below average"); show_figure(style_figure(fig, 300, True, top=70), key="pp_dev")

      # ---- B) actual vs predicted on test data: last 7 days hidden, model learns only from the hours before them
    with panel("Model check · actual vs predicted on test data", "the last 7 days are hidden, the model learns from all the history before them, then its forecast is compared with what really happened", key="ph_bt"):

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
            test_forecast_array, _, _ = forecast_phases_cached(training_hours, TEST_HOURS, phase_season_hours)
            predicted_kw = pd.DataFrame(test_forecast_array[:TEST_HOURS], index=test_index, columns=PHASE_CODES)
            actual_kw_drawn = phase_filled_hourly.reindex(test_index)                                       # drawn (missing hours filled)
            actual_kw_real = phase_measured_hourly.reindex(test_index)                                      # scored (real hours only)
            # Show only the three phase-wise actual vs predicted load charts.
            # The separate imbalance-% subplot has been removed as requested.
            fig = make_subplots(rows=3, cols=1, shared_xaxes=True, vertical_spacing=.07,
                                subplot_titles=[f"{n_} phase (kW)" for n_ in PHASE_LABELS])
            for j, p in enumerate(PHASE_CODES, 1):
                fig.add_scatter(x=test_index, y=actual_kw_drawn[p], name="Actual", legendgroup="a", showlegend=j == 1, line=dict(color=TEXT_COLOR, width=2), row=j, col=1)
                fig.add_scatter(x=test_index, y=predicted_kw[p], name="Predicted", legendgroup="p", showlegend=j == 1, line=dict(color=PHASE_COLORS[PHASE_KW_COLS[j - 1]], width=2.2, dash="dash"), row=j, col=1)
            style_figure(fig, 600, legend_y=1.0, top=105); fig.update_annotations(font=dict(size=13, color=TEXT_COLOR)); show_figure(fig, key="bt_main")

            error_table = pd.DataFrame({name: phase_error_scores(actual_kw_real[p].to_numpy(), predicted_kw[p].to_numpy()) for name, p in zip(PHASE_LABELS, PHASE_CODES)}).T
            error_table.columns = ["MAE (kW)", "MAPE (%)", "SMAPE (%)"]
            st.dataframe(error_table.round(2), width="stretch")
