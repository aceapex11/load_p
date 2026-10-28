"""GridSense - machine load intelligence (light mode, ARIMA(2,0,2)).
Overview     : full minute-level timeline INCLUDING idle (zero) periods   -> SecureElite_DeviceID19_sorted
Prediction   : ARIMA(2,0,2) on the 30-min rolling KW, future forecast         -> choose dataset: with zeros / without zeros
Load balance : static ARIMA per phase R/Y/B, outlier-phase alarms, forecast    -> choose dataset: with zeros / without zeros; alert threshold in %
"""
import io
import warnings
import itertools
from contextlib import contextmanager
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from plotly.subplots import make_subplots
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from statsmodels.tsa.arima.model import ARIMA

warnings.filterwarnings("ignore")
st.set_page_config(page_title="GridSense · Load intelligence", page_icon="⚡", layout="wide", initial_sidebar_state="collapsed")

HERE = Path(__file__).resolve().parent
FILE = HERE / "SecureElite_DeviceID19_KW_gt_1.xlsx"            # running data only (KW > 1)  -> used for prediction + balance
TIMELINE_PARQUET = HERE / "SecureElite_DeviceID19_sorted.parquet"  # full minute data incl. zeros (fast)
TIMELINE_XLSX = HERE / "SecureElite_DeviceID19_sorted.xlsx"        # same data as xlsx (slow to read, used only if parquet is missing)
PHASES = ["RKW", "YKW", "BKW"]

# colour roles (used consistently everywhere)
INK, MUTED, GRID = "#0F172A", "#64748B", "#EDF0F6"
LOAD, FORECAST, GOOD, WARN, BAD, CYAN, VIOLET = "#F59E0B", "#6366F1", "#10B981", "#F59E0B", "#F43F5E", "#06B6D4", "#8B5CF6"
AMBER, RED, BLUE = LOAD, BAD, "#3B82F6"
GREY = MUTED
PHASE_COL = {"RKW": "#F43F5E", "YKW": "#F59E0B", "BKW": "#3B82F6"}

# ------------------------------------------------------------------ design system (light, forced - no config file needed)
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
.appbar .logo { width:42px; height:42px; border-radius:12px; display:grid; place-items:center; font-size:1.3rem; background:linear-gradient(135deg,#FBBF24,#F97316 55%,#F43F5E); box-shadow:0 6px 16px -6px rgba(249,115,22,.7); }
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
.live { border-radius:18px; padding:1.2rem 1.3rem; background:linear-gradient(180deg,#FFFFFF,#FFFBF2); border:1px solid #F5E6C8; height:100%; }
.live .big { font-family:'JetBrains Mono',monospace; font-size:3.2rem; font-weight:600; letter-spacing:-2px; color:#0F172A !important; line-height:1; margin:.7rem 0 .2rem; } .live .big small { font-size:1.1rem; color:#64748B !important; }
.bar { margin:.55rem 0; } .bar .row { display:flex; justify-content:space-between; font-size:.8rem; color:#475569 !important; font-weight:500; } .bar .row b { font-family:'JetBrains Mono',monospace; color:#0F172A !important; }
.bar .trk { height:8px; background:#EEF1F7; border-radius:99px; margin-top:.25rem; overflow:hidden; } .bar .fil { height:100%; border-radius:99px; }
.note { border-left:3px solid #6366F1; background:#F4F5FF; padding:.7rem 1rem; border-radius:0 12px 12px 0; color:#475569 !important; font-size:.88rem; margin:.5rem 0 .9rem; } .note b, .note i { color:#1E293B !important; }
.note.amber { border-left-color:#F59E0B; background:#FFF8EB; }
.tblwrap { overflow-x:auto; border:1px solid #E6EAF2; border-radius:14px; background:#FFFFFF; margin:.4rem 0 1rem; }
table.tbl { border-collapse:collapse; width:100%; font-size:.84rem; } table.tbl th { text-align:right; background:#F7F9FC; color:#64748B !important; font-weight:600; padding:.55rem .8rem; border-bottom:1px solid #E6EAF2; white-space:nowrap; }
table.tbl td { text-align:right; padding:.5rem .8rem; border-bottom:1px solid #F0F3F8; color:#0F172A !important; font-family:'JetBrains Mono',monospace; font-size:.8rem; } table.tbl th:first-child, table.tbl td:first-child { text-align:left; }

/* inputs (forced light) */
[data-baseweb="select"] > div, [data-baseweb="input"], [data-baseweb="base-input"], input, [data-testid="stNumberInputContainer"] { background:#FFFFFF !important; color:#0F172A !important; border-color:#DDE2EE !important; border-radius:10px !important; }
[data-baseweb="popover"] *, ul[role="listbox"], li[role="option"] { background:#FFFFFF !important; color:#0F172A !important; } li[role="option"]:hover { background:#F1F3FF !important; }
[data-testid="stExpander"] { background:#FFFFFF; border:1px solid #E6EAF2 !important; border-radius:14px; } [data-testid="stExpander"] summary, [data-testid="stExpander"] summary p { color:#0F172A !important; font-weight:600; }
[data-testid="stBaseButton-secondary"] { background:#FFFFFF !important; color:#0F172A !important; border:1px solid #DDE2EE !important; border-radius:10px !important; font-weight:600; }
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
                      font=dict(family="Inter, sans-serif", size=12.5, color="#334155"), margin=dict(l=10, r=10, t=top, b=10),
                      title=dict(x=0, xanchor="left", y=0.985, yanchor="top", font=dict(size=15, color=INK)),
                      hoverlabel=dict(bgcolor="#FFFFFF", bordercolor="#E2E8F0", font=dict(family="Inter, sans-serif", size=12, color=INK)),
                      legend=dict(orientation="h", x=0, xanchor="left", y=legend_y, yanchor="bottom", bgcolor="rgba(0,0,0,0)"), showlegend=legend)
    fig.update_xaxes(gridcolor=GRID, zeroline=False, linecolor="#E2E8F0", showspikes=False); fig.update_yaxes(gridcolor=GRID, zeroline=False)
    return fig


def show(fig, key=None):
    st.plotly_chart(fig, width="stretch", config={"displayModeBar": False}, key=key)


def gaps(t, *arrays):
    """Break plot lines wherever the timestamp jumps by more than 1 minute (as in the notebook)."""
    t = pd.DatetimeIndex(t).values; brk = np.where(np.diff(t) > np.timedelta64(1, "m"))[0] + 1
    if not len(brk): return (t, *[np.asarray(a, float) for a in arrays])
    return (np.insert(t, brk, t[brk - 1] + np.timedelta64(30, "s")), *[np.insert(np.asarray(a, float), brk, np.nan) for a in arrays])


def phase_bars(values, scale):
    out = ""
    for ph, v in zip(PHASES, values):
        w = 0 if pd.isna(v) else min(100, max(0, v / scale * 100))
        out += (f'<div class="bar"><div class="row"><span>{ph[0]} phase</span><b>{"–" if pd.isna(v) else f"{v:,.0f} kW"}</b></div>'
                f'<div class="trk"><div class="fil" style="width:{w:.0f}%;background:{PHASE_COL[ph]}"></div></div></div>')
    return out


# ------------------------------------------------------------------ data
@st.cache_data(show_spinner="Reading the full minute-level timeline…")
def load_timeline():
    """FULL data incl. zero (idle) minutes and missing minutes -> used for plotting / replay only."""
    cols = ["Timestamps", "KW", "RKW", "YKW", "BKW", "KVA", "PF", "I1", "I2", "I3", "V1", "V2", "V3", "F"]
    tl = pd.read_parquet(TIMELINE_PARQUET) if TIMELINE_PARQUET.exists() else pd.read_excel(TIMELINE_XLSX, usecols=cols)
    tl["Timestamps"] = pd.to_datetime(tl["Timestamps"])
    for c in cols[1:]: tl[c] = pd.to_numeric(tl[c], errors="coerce").astype(float) + 0.0
    return tl[cols].sort_values("Timestamps").drop_duplicates("Timestamps").reset_index(drop=True)


@st.cache_data(show_spinner="Reading running-data file…")
def load():
    """Running data only (KW > 1) -> used for prediction + load balance."""
    df = pd.read_excel(FILE)
    df.columns = df.columns.astype(str).str.strip()
    df["Timestamps"] = pd.to_datetime(df["Timestamps"], errors="coerce")
    for c in ["KW"] + PHASES: df[c] = pd.to_numeric(df[c], errors="coerce")
    return df.dropna(subset=["Timestamps", "KW"]).sort_values("Timestamps").drop_duplicates("Timestamps").reset_index(drop=True)


@st.cache_data
def timeline_summary(tl):
    measured, running = tl.KW.notna(), tl.KW > 1
    span_min = int((tl.Timestamps.max() - tl.Timestamps.min()).total_seconds() // 60) + 1
    day = tl.Timestamps.dt.date
    daily = pd.DataFrame({"day": day, "kwh": tl.KW.fillna(0) / 60, "run_h": running / 60}).groupby("day").sum()
    daily["avg_run"] = tl[running].groupby(day[running]).KW.mean().reindex(daily.index)
    heat = tl.assign(day=day, hour=tl.Timestamps.dt.hour).pivot_table(index="day", columns="hour", values="KW", aggfunc="mean")
    last_run = tl.Timestamps[running].max(); last_seen = tl.Timestamps[measured].max()
    return dict(energy=float(tl.KW.fillna(0).sum() / 60), run_min=int(running.sum()), idle_min=int((measured & ~running).sum()),
                nodata_min=max(0, span_min - int(measured.sum())), peak=float(tl.KW.max()), avg_run=float(tl.KW[running].mean()),
                last_run=last_run, last_seen=last_seen, daily=daily, heat=heat, util=float(running.sum() / measured.sum() * 100),
                now_running=bool(tl.KW[measured].iloc[-1] > 1))


# ------------------------------------------------------------------ shared forecast-control options
CUSTOM = "Custom date & time"
LENGTHS = {"30 minutes": 30, "1 hour": 60, "6 hours": 360, "12 hours": 720, "24 hours": 1440, "3 days": 4320, "7 days": 10080}
STEPS = {"1 min": 1, "5 min": 5, "10 min": 10, "15 min": 15, "30 min": 30, "1 hour": 60, "3 hours": 180}
HISTS = {"1 hour": 60, "3 hours": 180, "6 hours": 360, "12 hours": 720, "24 hours": 1440}
MAX_HORIZON = 10080                                   # 7 days of one-minute steps
MODE_FULL, MODE_RUN = "With zeros (full timeline)", "Without zeros (running only, KW > 1)"


def dataset_button(label, key):
    """Button-style switch between the two datasets. Returns MODE_FULL (with zeros) or MODE_RUN (without zeros)."""
    choice = st.segmented_control(label, [MODE_FULL, MODE_RUN], default=MODE_FULL, key=key,
                                  help="With zeros = every recorded minute incl. idle (0 kW) time.  Without zeros = only minutes when the machine is running (KW > 1).")
    return choice or MODE_FULL

# Fitted parameters of the notebook's default model (full timeline, ARIMA(2,0,2), 30-row rolling mean, 95 % train).
# Fitting that model takes ~65 s, so the app reuses these numbers through model.smooth() - which gives exactly the same
# results as .fit(). They are only used when the training series is identical (checked below); every other setting is fitted from scratch.
SAVED_FIT = dict(order=(2, 0, 2), n_train=92962, train_end="2024-11-06 07:06:00", params=[25.501715174988526, 1.9706117515295842, -0.9707191716482642, -0.6595154250758719, 0.05126904956450774, 0.42126987828231904])


def thin(frame, max_points=4000):
    """Plot at most ~max_points rows (the full data is still used for every calculation)."""
    k = int(np.ceil(len(frame) / max_points)); return frame.iloc[::k] if k > 1 else frame


# ------------------------------------------------------------------ Load prediction (Untitled20 notebook)
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
    same_as_saved = (tuple(arima_order) == SAVED_FIT["order"] and n_train == SAVED_FIT["n_train"] and str(train_series.index[-1]) == SAVED_FIT["train_end"])
    arima_fit = model.smooth(np.asarray(SAVED_FIT["params"], float)) if same_as_saved else model.fit()
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


# ------------------------------------------------------------------ Load balance (Untitled21 notebook)
@st.cache_data(show_spinner="Preparing the running data…")
def balance_features(timeline, rolling_minutes, with_zeros):
    """with_zeros=False: only rows where ALL three phases are >= 1 kW.  with_zeros=True: every recorded minute incl. idle zeros.
    Then a time-based rolling mean of R / Y / B (first window removed)."""
    kept = timeline[["Timestamps"] + PHASES].dropna().sort_values("Timestamps").drop_duplicates("Timestamps", keep="last")
    if not with_zeros: kept = kept.loc[(kept[PHASES] >= 1).all(axis=1)]
    kept = kept.set_index("Timestamps").sort_index()
    rolled = kept[PHASES].rolling(f"{rolling_minutes}min", min_periods=1).mean()
    rolled = rolled.loc[rolled.index >= rolled.index.min() + pd.Timedelta(minutes=rolling_minutes)].dropna()
    return kept, rolled


@st.cache_data(show_spinner="Training one ARIMA per phase (R, Y, B)…")
def balance_forecast(rolled, arima_order):
    """Static ARIMA per phase on the standardised rolling mean, trained on 100 % of the observations; one step = one observation (~1 min)."""
    pred, lower, upper = {}, {}, {}
    for phase in PHASES:
        values = rolled[phase].to_numpy(float); mean_value, std_value = values.mean(), values.std(ddof=0)
        scaled = pd.Series((values - mean_value) / std_value, index=pd.RangeIndex(1, len(values) + 1))
        forecast = ARIMA(scaled, order=arima_order, enforce_stationarity=False, enforce_invertibility=False).fit().get_forecast(steps=MAX_HORIZON)
        band = np.asarray(forecast.conf_int(alpha=0.05))
        pred[phase], lower[phase], upper[phase] = (np.asarray(forecast.predicted_mean) * std_value + mean_value, np.maximum(band[:, 0] * std_value + mean_value, 0), band[:, 1] * std_value + mean_value)
    last_time = rolled.index[-1]; times = pd.date_range(last_time + pd.Timedelta(minutes=1), periods=MAX_HORIZON, freq="min")
    return dict(pred=pd.DataFrame(pred, index=times), lower=pd.DataFrame(lower, index=times), upper=pd.DataFrame(upper, index=times), last_time=last_time, last_observation=len(rolled))


def phase_alarms(forecast, threshold):
    """A phase raises an alarm whenever it differs from the average of the OTHER two phases by more than `threshold` %.
    Each phase is judged on its own, so one, two or all three phases can alarm at the same time.
    Returns (gap % per phase, alarm flags per phase)."""
    gap, alarm = {}, {}
    for phase in PHASES:
        a, b = [p for p in PHASES if p != phase]
        pair_mean = (forecast[a] + forecast[b]) / 2
        gap[phase] = (forecast[phase] - pair_mean).abs() / pair_mean.abs().replace(0, np.nan) * 100
        alarm[phase] = ((gap[phase] > threshold) & (pair_mean > 0)).fillna(False)
    return pd.DataFrame(gap), pd.DataFrame(alarm)


# ------------------------------------------------------------------ page
tl, df = load_timeline(), load()
S = timeline_summary(tl)
idle_days = (S["last_seen"] - S["last_run"]).total_seconds() / 86400
status = (f'<span class="pill run"><span class="dot"></span>Running now</span>' if S["now_running"] else
          f'<span class="pill idle"><span class="dot"></span>Idle · last ran {S["last_run"]:%d %b %H:%M}</span>')
st.markdown(f"""<div class="appbar"><div class="logo">⚡</div><div><div class="t">GridSense</div><div class="s">Load forecasting · phase balance</div></div><div class="sp"></div>
{status}<span class="chip">Device 19</span><span class="chip">{tl.Timestamps.min():%d %b} → {tl.Timestamps.max():%d %b %Y}</span><span class="chip">{len(tl):,} readings</span></div>""", unsafe_allow_html=True)

tab_overview, tab_pred, tab_bal = st.tabs(["Overview", "Load prediction", "Load balance"])

# ================================================================== OVERVIEW (full data incl. zeros)
with tab_overview:
    daily = S["daily"]; total_h = (S["run_min"] + S["idle_min"]) / 60
    k = st.columns(5)
    with k[0]: kpi("Energy used", f"{S['energy'] / 1000:,.1f} MWh", f"{len(daily)} days of data", "", "🔋", daily.kwh.values, LOAD)
    with k[1]: kpi("Machine uptime", f"{S['util']:.1f} %", f"{S['run_min'] / 60:,.0f} h on · {S['idle_min'] / 60:,.0f} h idle", "info", "⏱️", daily.run_h.values, FORECAST)
    with k[2]: kpi("Peak load", f"{S['peak']:,.0f} kW", "highest 1-minute reading", "", "📈")
    with k[3]: kpi("Average load · running", f"{S['avg_run']:,.0f} kW", "only while the machine is on", "", "⚙️", daily.avg_run.values, CYAN)
    with k[4]: kpi("Last running reading", f"{S['last_run']:%d %b}", f"{S['last_run']:%H:%M} · idle for {idle_days:.0f} days since", "warn" if not S["now_running"] else "good", "🕒")
    st.write("")
    with panel("Load timeline", "10-minute averages · zeros are real idle time, gaps are missing data", key="timeline"):
        signal = st.selectbox("Signal", ["Total load (kW)", "Phase load R / Y / B (kW)", "Phase current (A)", "Voltage (V)"], label_visibility="collapsed")
        cols = {"Total load (kW)": ["KW"], "Phase load R / Y / B (kW)": PHASES, "Phase current (A)": ["I1", "I2", "I3"], "Voltage (V)": ["V1", "V2", "V3"]}[signal]
        ten = tl.set_index("Timestamps")[cols].resample("10min").mean(); f = go.Figure()
        palette = [LOAD] if cols == ["KW"] else [PHASE_COL["RKW"], PHASE_COL["YKW"], PHASE_COL["BKW"]]
        for col_name, color in zip(cols, palette):
            f.add_scatter(x=ten.index, y=ten[col_name], name=col_name, line=dict(color=color, width=1.4), fill="tozeroy" if len(cols) == 1 else None, fillcolor="rgba(245,158,11,.16)", connectgaps=False)
        f.update_xaxes(rangeslider_visible=True, rangeslider_thickness=.07); show(style(f, 400, legend=len(cols) > 1, top=40))
    a, b = st.columns([2.1, 1])
    with a, panel("Activity calendar", "average kW per hour of day · grey = idle, white = no data", key="heat"):
        heat = S["heat"]; f = go.Figure(go.Heatmap(z=heat.values, x=[f"{h:02d}:00" for h in heat.columns], y=[f"{d:%d %b}" for d in heat.index], zmin=0, zmax=650, xgap=2, ygap=2,
                                                   colorscale=[[0, "#EBEEF5"], [.02, "#FDE9BF"], [.5, "#F59E0B"], [1, "#B45309"]], colorbar=dict(title="kW", thickness=10, len=.6),
                                                   hovertemplate="%{y} · %{x}<br><b>%{z:.0f} kW</b><extra></extra>"))
        f.update_yaxes(autorange="reversed", dtick=3); f.update_layout(hovermode="closest"); show(style(f, 520, False, top=20))
    with b, panel("Where the time went", "share of all minutes", key="donut"):
        f = go.Figure(go.Pie(labels=["Running", "Idle (zero load)", "No data"], values=[S["run_min"], S["idle_min"], S["nodata_min"]], hole=.68, sort=False, marker=dict(colors=[GOOD, "#CBD5E1", "#F1F5F9"], line=dict(color="#fff", width=2)),
                             textinfo="percent", hovertemplate="%{label}<br>%{value:,} min<extra></extra>"))
        f.update_layout(hovermode="closest", annotations=[dict(text=f"<b>{S['util']:.1f}%</b><br><span style='font-size:11px;color:#64748B'>uptime</span>", showarrow=False, font=dict(size=22))])
        show(style(f, 300, True, top=20).update_layout(legend=dict(orientation="h", y=-0.05, x=0)))
        st.markdown(f'<div class="note amber" style="margin-top:.2rem">Most of the file is the machine being <b>off</b> (zeros). Prediction and balance only use the <b>running</b> minutes.</div>', unsafe_allow_html=True)
    with panel("Running hours per day", "hover for energy and average load", key="daily"):
        f = go.Figure(go.Bar(x=daily.index, y=daily.run_h, marker_color=[LOAD if v > 0 else "#E2E8F0" for v in daily.run_h], customdata=np.c_[daily.kwh, daily.avg_run],
                             hovertemplate="%{x|%d %b}<br><b>%{y:.1f} h running</b><br>%{customdata[0]:,.0f} kWh · avg %{customdata[1]:.0f} kW<extra></extra>"))
        f.update_layout(hovermode="closest", yaxis_title="hours"); show(style(f, 300, False, top=20))

# ================================================================== LOAD PREDICTION (Untitled20 notebook)
with tab_pred:
    data_mode = dataset_button("Dataset to forecast on", "ds_pred")
    with st.expander("ARIMA settings", expanded=False):
        c = st.columns(6)
        p, d_, q = c[0].number_input("p (AR)", 0, 6, 2), c[1].number_input("d (diff)", 0, 2, 0), c[2].number_input("q (MA)", 0, 6, 2)
        win = c[3].number_input("Rolling window (min)", 1, 120, 30); split_pct = c[4].slider("Train split %", 50, 99, 95); block = c[5].number_input("Backtest block (min)", 5, 240, 60, 5)
    src = tl if data_mode == MODE_FULL else df
    R = predict_kw(src[["Timestamps", "KW"]], (p, d_, q), split_pct, int(block), int(win)); mt, mr = R["m_test"], R["m_train"]
    NX = R["future"]; last_time = NX["last_time"]; default_until = last_time + pd.Timedelta(minutes=30)
    if st.session_state.get("anchor_pred") != last_time:      # data changed (e.g. other training data) -> reset the end time
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
            f.add_shape(type="rect", xref="x", yref="paper", x0=last_time, x1=t_f[-1], y0=0, y1=1, fillcolor="rgba(99,102,241,.06)", line_width=0)
            anchor_value, tt = NX["last_value"], [last_time] + list(t_f)          # start the forecast exactly at the last actual point (no gap)
            f.add_scatter(x=hist.index, y=hist.values, name=f"Actual rolling mean ({int(win)} min)", mode="lines", line=dict(color=INK, width=2.2))
            f.add_scatter(x=tt + tt[::-1], y=[anchor_value] + list(hi_f) + ([anchor_value] + list(lo_f))[::-1], fill="toself", fillcolor="rgba(99,102,241,.16)", mode="lines", line=dict(width=0), name="95 % range", hoverinfo="skip")
            f.add_scatter(x=tt, y=[anchor_value] + list(m_f), name=f"Forecast ({len(idx)} points)", mode="lines", line=dict(color=FORECAST, width=3))
            if len(idx) <= 120: f.add_scatter(x=t_f, y=m_f, mode="markers", marker=dict(size=6, color="#fff", line=dict(color=FORECAST, width=2)), showlegend=False, hoverinfo="skip")
            f.add_shape(type="line", xref="x", yref="paper", x0=last_time, x1=last_time, y0=0, y1=1, line=dict(color=MUTED, width=1.2, dash="dot"))
            f.add_annotation(x=last_time, yref="paper", y=1.0, text="last data point", showarrow=False, xanchor="right", yanchor="bottom", font=dict(size=11, color=MUTED))
            f.update_yaxes(title="kW"); show(style(f, 340, True, top=70))
        with st.expander(f"Forecast table · {len(idx)} points"):
            table(pd.DataFrame({"Time": [t.strftime("%d %b %H:%M") for t in t_f], "Forecast kW": m_f, "Lower 95 %": lo_f, "Upper 95 %": hi_f}).head(300), 2)
        if horizon > 120: note("ARIMA is a short-term model: beyond roughly 1–2 hours the forecast flattens towards the long-run average and the range keeps widening, so treat far-ahead points as a guide, not a prediction.")
    if data_mode == MODE_FULL and R["idle_share"] > .5:
        note(f"<b>Why the forecast is near zero:</b> as in your notebook, the model is trained on the full timeline, which is <b>{R['idle_share'] * 100:.0f} % idle (0 kW)</b>. The data ends in an idle period "
             f"({last_time:%d %b}), so the forecast starts near 0 kW and only drifts towards the long-run average of about {R['level']:.0f} kW. For a forecast of the machine's <i>running</i> load, switch the dataset button above to <b>Without zeros</b>.", "amber")

    st.markdown('<div class="ph" style="margin-top:.6rem"><b>Model check on past data</b><span>how well the same ARIMA predicted data it had not seen</span></div>', unsafe_allow_html=True)
    k = st.columns(4)
    with k[0]: kpi("Test RMSE", f"{mt['RMSE']:.2f} kW", f"{len(R['test']):,} forecast points", "info", "🎯")
    with k[1]: kpi("Test MAE", f"{mt['MAE']:.2f} kW", f"{mt['MAE'] / max(R['test'].Actual.mean(), 1e-9) * 100:.1f} % of mean test load" if R["test"].Actual.mean() >= 1 else "test period has ~0 kW load", "", "📏")
    with k[2]: kpi("Test R²", f"{mt['R2']:.3f}", "test period is constant → R² not meaningful" if R["test_constant"] else "1 = perfect · 0 = no better than the average", "warn" if R["test_constant"] else "good" if mt["R2"] > .5 else "warn", "✅")
    with k[3]: kpi("Train fit R²", f"{mr['R2']:.3f}", f"RMSE {mr['RMSE']:.2f} kW · {len(R['train']):,} points", "", "🧪")
    st.write("")
    s1, s2, s3 = st.tabs(["Test forecast", "Training fit", "Residuals"])

    def line_pair(frame, a, b, title, name_b):
        shown = thin(frame); t, ya, yb = gaps(shown.Timestamp, shown[a], shown[b]); f = go.Figure()
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
        for col, frame, name, color in ((a, R["train"], "Training", CYAN), (b, R["test"], "Test", VIOLET)):
            shown = thin(frame); t, r = gaps(shown.Timestamp, shown.Residual); f = go.Figure(go.Scatter(x=t, y=r, mode="lines", line=dict(color=color, width=1.3), connectgaps=False))
            f.add_hline(y=0, line_dash="dash", line_color=GREY); f.update_layout(title=f"{name} residuals (actual − predicted)", yaxis_title="kW")
            with col, st.container(border=True, key=f"panel_res_{name}"): show(style(f, 340, False, top=60))

# ================================================================== LOAD BALANCE (Untitled21 notebook)
with tab_bal:
    top = st.columns([2.4, 1.1])
    with top[0]: bal_mode = dataset_button("Dataset to forecast on", "ds_bal")
    thr = top[1].number_input("Alert threshold (%)", 0.5, 100.0, 5.0, 0.5, format="%.1f", key="thr_b",
                              help="A phase raises an alarm when it differs from the average of the other two phases by more than this percentage.")
    with st.expander("ARIMA settings", expanded=False):
        c = st.columns(4)
        pb, db, qb = c[0].number_input("p (AR)", 0, 6, 2, key="pb"), c[1].number_input("d (diff)", 0, 2, 0, key="db"), c[2].number_input("q (MA)", 0, 6, 2, key="qb")
        roll_min = c[3].number_input("Rolling window (min)", 5, 120, 30, key="roll_b")
    with_zeros = bal_mode == MODE_FULL
    raw_b, rolled = balance_features(tl, int(roll_min), with_zeros)
    if len(rolled) < 200: st.error("Not enough data in this dataset to train the phase models. Try the other dataset."); st.stop()
    BF = balance_forecast(rolled, (pb, db, qb)); last_b = BF["last_time"]; default_b = last_b + pd.Timedelta(minutes=30)
    if st.session_state.get("anchor_bal") != last_b:
        st.session_state.update(anchor_bal=last_b, until_date_bal=default_b.date(), until_time_bal=default_b.time(), len_bal="30 minutes")

    def length_changed_bal():
        if st.session_state.len_bal != CUSTOM:
            end = last_b + pd.Timedelta(minutes=LENGTHS[st.session_state.len_bal]); st.session_state.until_date_bal, st.session_state.until_time_bal = end.date(), end.time()

    def until_changed_bal(): st.session_state.len_bal = CUSTOM

    idle_share_b = float(raw_b[PHASES].lt(1).all(axis=1).mean())
    if with_zeros and idle_share_b > .5:
        note(f"<b>With zeros:</b> this dataset is <b>{idle_share_b * 100:.0f} % idle (0 kW)</b> and it ends in an idle period ({last_b:%d %b}), so the phase forecasts start near 0 kW and drift towards the average. "
             "Alarms need all phases above 0 kW, so few or none will fire. Switch to <b>Without zeros</b> to forecast the running load.", "amber")
    alarms_30, alarms_60 = (phase_alarms(BF["pred"].iloc[:n_], thr)[1] for n_ in (30, 60))
    k = st.columns(4)
    with k[0]: kpi("Observations used", f"{len(rolled):,}", f"{int(roll_min)}-min rolling mean · " + ("incl. idle zero minutes" if with_zeros else "all 3 phases ≥ 1 kW"), "info", "🧮")
    with k[1]: kpi("Training period", f"{rolled.index.min():%d %b} → {last_b:%d %b}", f"static ARIMA{(pb, db, qb)} · trained on 100 % of the data", "", "🗓️")
    with k[2]: kpi("Alarm points · next 30 min", f"{int(alarms_30.to_numpy().sum())}", " · ".join(f"{ph[0]} {int(alarms_30[ph].sum())}" for ph in PHASES), "good" if not alarms_30.to_numpy().any() else "bad", "🚨")
    with k[3]: kpi("Alarm points · next 1 hour", f"{int(alarms_60.to_numpy().sum())}", " · ".join(f"{ph[0]} {int(alarms_60[ph].sum())}" for ph in PHASES), "good" if not alarms_60.to_numpy().any() else "bad", "🚨")
    st.write("")

    # ---- phase forecast + alarms with user-chosen end time, number of points and history
    with panel("Phase forecast & imbalance alarms", f"one ARIMA per phase · one step = one observation (≈ 1 min) · starts after the last data point · {last_b:%d %b %Y %H:%M}", key="bal_next"):
        c = st.columns(5)
        c[0].selectbox("Forecast length", list(LENGTHS) + [CUSTOM], key="len_bal", on_change=length_changed_bal)
        step_choice_b = c[1].selectbox("Show a point every", list(STEPS), index=1, key="step_bal")
        hist_choice_b = c[2].selectbox("History shown", list(HISTS), index=4, key="hist_bal")
        until_date_b = c[3].date_input("Forecast until · date", min_value=last_b.date(), max_value=(last_b + pd.Timedelta(days=7)).date(), key="until_date_bal", on_change=until_changed_bal)
        until_time_b = c[4].time_input("Forecast until · time", step=300, key="until_time_bal", on_change=until_changed_bal)
        horizon_raw_b = int((pd.Timestamp.combine(until_date_b, until_time_b) - last_b).total_seconds() // 60)
        horizon_b = int(min(max(horizon_raw_b, 1), MAX_HORIZON)); step_b = max(min(STEPS[step_choice_b], horizon_b), int(np.ceil(horizon_b / 400)))
        st.caption(f"Forecasting {horizon_b} steps ahead, until {last_b + pd.Timedelta(minutes=horizon_b):%a %d %b %H:%M} · one point every {step_b} min" + (" (capped at 7 days)" if horizon_raw_b > MAX_HORIZON else " (end time was before the last data point, so showing 1 step)" if horizon_raw_b < 1 else ""))
        idx_b = np.unique(np.r_[np.arange(step_b - 1, horizon_b, step_b), horizon_b - 1])
        fc_b = BF["pred"].iloc[:horizon_b]; gap_b, alarm_b = phase_alarms(fc_b, thr); n_alarms = int(alarm_b.to_numpy().sum())
        with st.container():
            pill = '<span class="pill run"><span class="dot"></span>No imbalance alarm</span>' if n_alarms == 0 else f'<span class="pill bad"><span class="dot"></span>{n_alarms} alarm points</span>'
            end_vals = list(fc_b.iloc[-1].values)
            st.markdown(f'<div class="live">{pill}<div style="color:#64748B;font-size:.85rem;margin:.6rem 0 .3rem">Forecast at {fc_b.index[-1]:%d %b %H:%M}</div>{phase_bars(end_vals, max(end_vals) * 1.15)}'
                        f'<div class="mini"><div><b>{gap_b.max().max():.1f} %</b><span>largest phase gap</span></div><div><b>{thr:g} %</b><span>alarm threshold</span></div></div></div>', unsafe_allow_html=True)

        window_start_b = last_b - pd.Timedelta(minutes=HISTS[hist_choice_b])
        roll_hist = rolled[rolled.index > window_start_b]
        t_fb = fc_b.index[idx_b]; tt_b = [last_b] + list(t_fb)
        f = make_subplots(rows=3, cols=1, shared_xaxes=True, subplot_titles=[f"{ph[0]} phase (kW)" for ph in PHASES], vertical_spacing=.09)
        for i, ph in enumerate(PHASES, 1):
            a_v = float(rolled[ph].iloc[-1]); up_, lo_ = BF["upper"][ph].iloc[:horizon_b].iloc[idx_b], BF["lower"][ph].iloc[:horizon_b].iloc[idx_b]       # forecast starts at the last actual point (no gap)
            f.add_scatter(x=roll_hist.index, y=roll_hist[ph], name=f"Actual rolling mean ({int(roll_min)} min)", mode="lines", line=dict(color=INK, width=2), legendgroup="r", showlegend=i == 1, row=i, col=1)
            f.add_scatter(x=tt_b + tt_b[::-1], y=[a_v] + list(up_) + ([a_v] + list(lo_))[::-1], fill="toself", fillcolor="rgba(99,102,241,.14)", mode="lines", line=dict(width=0), name="95 % range", legendgroup="b", showlegend=i == 1, hoverinfo="skip", row=i, col=1)
            f.add_scatter(x=tt_b, y=[a_v] + list(fc_b[ph].iloc[idx_b]), name="Forecast", mode="lines", line=dict(color=FORECAST, width=2.6), legendgroup="f", showlegend=i == 1, row=i, col=1)
            hit = fc_b[ph][alarm_b[ph]]
            if len(hit): f.add_scatter(x=hit.index, y=hit.values, name="Imbalance alarm", mode="markers", marker=dict(symbol="x", size=10, color=BAD), legendgroup="al", showlegend=i == 1, hoverinfo="skip", row=i, col=1)
        f.add_shape(type="line", xref="x", yref="paper", x0=last_b, x1=last_b, y0=0, y1=1, line=dict(color=MUTED, width=1.2, dash="dot"))
        show(style(f, 640, legend_y=1.0, top=120))
        with st.expander(f"Alarm points · {n_alarms}"):
            if n_alarms == 0: st.caption("No phase is flagged as an outlier anywhere in the chosen forecast window.")
            else:
                rows = [(t, ph, fc_b.at[t, ph], gap_b.at[t, ph]) for ph in PHASES for t in fc_b.index[alarm_b[ph].to_numpy()]]
                table(pd.DataFrame(rows, columns=["Time", "Phase", "Forecast kW", "Gap %"]).sort_values("Time").head(300).assign(Time=lambda x: x.Time.dt.strftime("%d %b %H:%M")), 2)
        d1, d2, _ = st.columns([1, 1, 2])
        export = pd.concat([pd.DataFrame({"Timestamp": fc_b.index, "ObservationNo": np.arange(BF["last_observation"] + 1, BF["last_observation"] + horizon_b + 1), "Phase": ph, "Predicted_KW": fc_b[ph].to_numpy(), "ImbalanceAlarm": alarm_b[ph].to_numpy()}) for ph in PHASES], ignore_index=True)
        d1.download_button("⬇ Forecasts + alarms (CSV)", export.to_csv(index=False).encode(), "RYB_ARIMA_Forecasts_Imbalance_Alarms.csv", "text/csv", on_click="ignore", width="stretch")
        d2.download_button(f"⬇ {int(roll_min)}-min rolling features (CSV)", rolled.reset_index().assign(ObservationNo=np.arange(1, len(rolled) + 1)).to_csv(index=False).encode(), "RYB_30min_Rolling_Features.csv", "text/csv", on_click="ignore", width="stretch")
    note(f"<b>Alarm rule:</b> a phase is flagged whenever it differs from the average of the <i>other two</i> phases by more than {thr:g} %. Each phase is checked on its own, so one, two or all three can be flagged at the same time. "
         "Steps are observations, assumed one minute apart; the range around each forecast is the model's 95 % interval.")
