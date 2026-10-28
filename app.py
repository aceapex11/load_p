"""GridSense - machine load intelligence (light mode, single ARIMA model).
Overview     : full minute-level timeline INCLUDING idle (zero) periods   -> SecureElite_DeviceID19_sorted
Live replay  : stream the recorded data minute by minute                  -> SecureElite_DeviceID19_sorted
Prediction   : ARIMA on the 30-min rolling KW + next-30-min forecast      -> SecureElite_DeviceID19_KW_gt_1 (running data only)
Load balance : ARIMA on R/Y/B phase kW -> imbalance %, alerts, next 30 min -> SecureElite_DeviceID19_KW_gt_1
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
HORIZONS_MIN = {"30 minute": 30, "1 hour": 60, "3 hours": 180, "6 hours": 360}

# colour roles (used consistently everywhere)
INK, MUTED, GRID = "#0F172A", "#64748B", "#EDF0F6"
LOAD, FORECAST, GOOD, WARN, BAD, CYAN, VIOLET = "#F59E0B", "#6366F1", "#10B981", "#F59E0B", "#F43F5E", "#06B6D4", "#8B5CF6"
AMBER, RED, BLUE = LOAD, BAD, "#3B82F6"
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


# ------------------------------------------------------------------ Load prediction
@st.cache_data(show_spinner="Fitting ARIMA on the training data…")
def predict_kw(meter_df, arima_order, train_split_pct, forecast_block_min, rolling_window_min):
    # 1) smooth the 1-minute KW into a rolling average (the series ARIMA actually models)
    rolling_kw = (meter_df.set_index("Timestamps")["KW"]
                  .rolling(rolling_window_min, min_periods=rolling_window_min).mean().dropna().astype(float))

    # 2) chronological train / test split
    n_train = int(len(rolling_kw) * train_split_pct / 100)
    train_series, test_series = rolling_kw.iloc[:n_train], rolling_kw.iloc[n_train:]
    train_values = pd.Series(train_series.to_numpy(), name="KW")       # plain 0..n index so ARIMA can append later

    # 3) fit ARIMA on the training data and get its in-sample (fitted) values
    arima_fit = ARIMA(train_values, order=arima_order, enforce_stationarity=False, enforce_invertibility=False).fit()
    train_fitted = np.asarray(arima_fit.predict(start=0, end=len(train_values) - 1, dynamic=False)).reshape(-1)
    train_table = pd.DataFrame({"Timestamp": train_series.index, "Actual": train_series.to_numpy(), "Fitted": train_fitted})
    train_table["Residual"] = train_table.Actual - train_table.Fitted

    # 4) test: forecast one block BEFORE seeing it, then reveal the actual values to the model (no refit)
    next_position = len(train_values)
    forecast_values, actual_values, block_timestamps = [], [], []
    running_fit = arima_fit
    for block_start in range(0, len(test_series), forecast_block_min):
        actual_block = test_series.iloc[block_start:block_start + forecast_block_min]
        forecast_values.extend(np.asarray(running_fit.forecast(steps=len(actual_block))).reshape(-1))
        actual_values.extend(actual_block.to_numpy())
        block_timestamps.extend(actual_block.index)
        running_fit = running_fit.append(
            pd.Series(actual_block.to_numpy(), index=range(next_position, next_position + len(actual_block)), name="KW"), refit=False)
        next_position += len(actual_block)
    test_table = (pd.DataFrame({"Timestamp": block_timestamps, "Actual": actual_values, "Forecast": forecast_values})
                  .sort_values("Timestamp").reset_index(drop=True))
    test_table["Residual"] = test_table.Actual - test_table.Forecast

    def score(actual, predicted):
        return dict(RMSE=float(np.sqrt(mean_squared_error(actual, predicted))),
                    MAE=float(mean_absolute_error(actual, predicted)),
                    R2=float(r2_score(actual, predicted)))

    return dict(train=train_table, test=test_table,
                m_train=score(train_table.Actual, train_table.Fitted),
                m_test=score(test_table.Actual, test_table.Forecast),
                n=len(rolling_kw),
                t0=train_series.index[0], t1=train_series.index[-1],
                t2=test_series.index[0], t3=test_series.index[-1])


# ------------------------------------------------------------------ Load balance 
MIN_TRAIN_BINS, MIN_OBS_TEST, TOL_BINS = 60, 12, 1


def imbalance_pct(frame):
    mean = frame[PHASES].mean(axis=1).replace(0, np.nan)
    return frame[PHASES].sub(mean, axis=0).abs().max(axis=1) / mean * 100


def alert_flags(frame, thr, min_load, min_consec, valid):
    raw = (imbalance_pct(frame) > thr) & (frame[PHASES].mean(axis=1) >= min_load) & valid
    run_end = raw.astype(int).rolling(min_consec, min_periods=min_consec).sum() == min_consec
    sus = run_end.copy()
    for k in range(1, min_consec):
        sus = sus | run_end.shift(-k, fill_value=False)
    return sus


def select_order(s, d):
    best, best_aic = (1, d, 1), np.inf
    for p, q in itertools.product(range(3), range(3)):
        try: aic = ARIMA(s, order=(p, d, q)).fit().aic
        except Exception: continue
        if aic < best_aic: best, best_aic = (p, d, q), aic
    return best


def rolling_fc(s, n_roll, max_steps, d, manual):
    fe = len(s) - n_roll
    if s.iloc[:fe].notna().sum() < MIN_TRAIN_BINS:
        raise ValueError(f"Fewer than {MIN_TRAIN_BINS} measured training bins - shorten the test window.")
    order = manual if manual else select_order(s.iloc[:fe], d)
    r = ARIMA(s.iloc[:fe], order=order).fit()
    out = np.full((n_roll, max_steps), np.nan)
    for i in range(n_roll):
        out[i] = r.forecast(max_steps).values
        r = r.append(s.iloc[[fe + i]], refit=False)
    return out, order


@st.cache_data(show_spinner="Rolling ARIMA forecast for R / Y / B …")
def balance(df, freq, auto, p, d, q, test_hours, thr, min_consec, min_load):
    step = pd.Timedelta(freq).total_seconds() / 60
    hz = {k: max(1, round(v / step)) for k, v in HORIZONS_MIN.items()}
    x = df.set_index("Timestamps")[PHASES].apply(pd.to_numeric, errors="coerce").resample(freq).mean()
    x = x.where(x.notna().all(axis=1))
    ok = x.notna().all(axis=1)
    full = x.loc[ok[ok].index[0]:ok[ok].index[-1]]
    test_steps = int(test_hours * 60 / step); max_steps = max(hz.values()); warm = max_steps - 1; n_roll = test_steps + warm
    test = full.iloc[-test_steps:]; obs = test.notna().all(axis=1)
    base = dict(share=float(ok.mean()), latest=float(imbalance_pct(df.tail(1)).iloc[0]))
    if int(obs.sum()) < MIN_OBS_TEST:
        return dict(error=f"Only {int(obs.sum())} measured bins in the last {test_hours} h (need {MIN_OBS_TEST}). Increase the test window.", **base)

    mean_s = full[PHASES].mean(axis=1, skipna=False)
    try:
        fc_mean, o_mean = rolling_fc(mean_s, n_roll, max_steps, d, None if auto else (p, d, q))
        fc, orders = {}, {"mean": o_mean}
        for f in PHASES:
            fc[f], orders[f] = rolling_fc(full[f] - mean_s, n_roll, max_steps, 0, None if auto else (p, 0, q))
    except ValueError as e:
        return dict(error=str(e), **base)
    preds = {}
    for k, s in hz.items():
        o = warm + np.arange(test_steps) - s + 1
        dev = np.column_stack([fc[f][o, s - 1] for f in PHASES]); dev = dev - dev.mean(axis=1, keepdims=True)
        preds[k] = pd.DataFrame(fc_mean[o, s - 1][:, None] + dev, index=test.index, columns=PHASES)

    act = imbalance_pct(test); a_strict = (act > thr) & obs
    a_sus = alert_flags(test, thr, min_load, min_consec, obs)
    near = lambda v: np.convolve(v.astype(int), np.ones(2 * TOL_BINS + 1), mode="same") > 0
    rows, det = [], []
    for k, fr in preds.items():
        for f in PHASES:
            a, pr = test[f][obs], fr[f][obs]
            rows.append(dict(horizon=k, phase=f, MAE=mean_absolute_error(a, pr), RMSE=np.sqrt(mean_squared_error(a, pr)), R2=r2_score(a, pr)))
        imb = imbalance_pct(fr); p_sus = alert_flags(fr, thr, min_load, min_consec, obs); av, pv = a_sus.values, p_sus.values
        hp, ha = int((pv & near(av)).sum()), int((av & near(pv)).sum())
        prec = hp / pv.sum() if pv.sum() else np.nan; rec = ha / av.sum() if av.sum() else np.nan
        f1 = 2 * prec * rec / (prec + rec) if (prec == prec and rec == rec and prec + rec > 0) else 0.0
        s_tp = int(((imb > thr) & obs & a_strict).sum()); s_p = int(((imb > thr) & obs).sum()); s_a = int(a_strict.sum())
        det.append({"horizon": k, "mean imbalance %": imb[obs].mean(), "max imbalance %": imb[obs].max(), "actual alerts": int(av.sum()), "predicted alerts": int(pv.sum()),
                    "precision": prec, "recall": rec, "F1": f1,
                    "strict precision": s_tp / s_p if s_p else np.nan, "strict recall": s_tp / s_a if s_a else np.nan})
    return dict(test=test, obs=obs, preds=preds, metrics=pd.DataFrame(rows), det=pd.DataFrame(det), act=act, a_sus=a_sus, orders=orders,
                n_obs=int(obs.sum()), n_alerts=int(a_sus.sum()), **base)


GREY = MUTED


# ------------------------------------------------------------------ NEW: future forecasts
@st.cache_data(show_spinner="Forecasting the next 30 minutes…")
def forecast_next(meter_df, arima_order, rolling_window_min, horizon_min):
    """Fit ARIMA on ALL running data and forecast `horizon_min` minutes beyond the last running reading (95 % interval)."""
    rolling_kw = (meter_df.set_index("Timestamps")["KW"]
                  .rolling(rolling_window_min, min_periods=rolling_window_min).mean().dropna().astype(float))
    arima_fit = ARIMA(rolling_kw.to_numpy(), order=arima_order, enforce_stationarity=False, enforce_invertibility=False).fit()
    forecast = arima_fit.get_forecast(steps=horizon_min)
    band = np.asarray(forecast.conf_int(alpha=0.05))
    last_time = rolling_kw.index[-1]
    return dict(times=pd.date_range(last_time + pd.Timedelta(minutes=1), periods=horizon_min, freq="min"),
                mean=np.asarray(forecast.predicted_mean), lower=band[:, 0], upper=band[:, 1],
                last_time=last_time, last_value=float(rolling_kw.iloc[-1]), history=rolling_kw.tail(120))


@st.cache_data(show_spinner="Forecasting the next 30 minutes of phase load…")
def balance_outlook(df, freq, orders, steps):
    """Fit the (already selected) ARIMA orders on the full phase series and forecast the next `steps` bins."""
    x = df.set_index("Timestamps")[PHASES].apply(pd.to_numeric, errors="coerce").resample(freq).mean()
    x = x.where(x.notna().all(axis=1)); ok = x.notna().all(axis=1)
    full = x.loc[ok[ok].index[0]:ok[ok].index[-1]]
    mean_series = full[PHASES].mean(axis=1, skipna=False)
    mean_forecast = ARIMA(mean_series, order=orders["mean"]).fit().forecast(steps)
    deviation = np.column_stack([ARIMA(full[ph] - mean_series, order=orders[ph]).fit().forecast(steps).to_numpy() for ph in PHASES])
    deviation = deviation - deviation.mean(axis=1, keepdims=True)
    forecast = pd.DataFrame(mean_forecast.to_numpy()[:, None] + deviation, index=mean_forecast.index, columns=PHASES)
    return dict(forecast=forecast, latest=full.iloc[-1], latest_time=full.index[-1])


# ------------------------------------------------------------------ page
tl, df = load_timeline(), load()
S = timeline_summary(tl)
idle_days = (S["last_seen"] - S["last_run"]).total_seconds() / 86400
status = (f'<span class="pill run"><span class="dot"></span>Running now</span>' if S["now_running"] else
          f'<span class="pill idle"><span class="dot"></span>Idle · last ran {S["last_run"]:%d %b %H:%M}</span>')
st.markdown(f"""<div class="appbar"><div class="logo">⚡</div><div><div class="t">GridSense</div><div class="s">Load forecasting · phase balance · live replay</div></div><div class="sp"></div>
{status}<span class="chip">Device 19</span><span class="chip">{tl.Timestamps.min():%d %b} → {tl.Timestamps.max():%d %b %Y}</span><span class="chip">{len(tl):,} readings</span></div>""", unsafe_allow_html=True)

tab_overview, tab_live, tab_pred, tab_bal = st.tabs(["Overview", "Live replay", "Load prediction", "Load balance"])

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

# ================================================================== LIVE REPLAY (streams the recorded minutes)
with tab_live:
    SPEEDS = {"1 min / second": 1, "5 min / second": 5, "15 min / second": 15, "30 min / second": 30}
    WINDOWS = {"Last 1 hour": 60, "Last 3 hours": 180, "Last 6 hours": 360, "Last 12 hours": 720}
    on = tl.KW > 1; day_key = tl.Timestamps.dt.date
    first_on = tl[on].groupby(day_key[on]).apply(lambda g: g.index[0]); minutes_on = on.groupby(day_key).sum()
    jump_options = {f"{d:%a %d %b} · starts {tl.Timestamps[i]:%H:%M}": max(0, int(i) - 45) for d, i in first_on.items() if minutes_on[d] >= 60}
    if "cursor" not in st.session_state: st.session_state.cursor, st.session_state.playing = list(jump_options.values())[-1], False

    def toggle_play(): st.session_state.playing = not st.session_state.playing
    def restart(): st.session_state.cursor, st.session_state.playing = jump_options[st.session_state.jump_sel], False
    def on_jump(): st.session_state.cursor, st.session_state.playing = jump_options[st.session_state.jump_sel], False

    c = st.columns([1.1, 1, 2.3, 1.5, 1.5])
    c[0].button("⏸ Pause" if st.session_state.playing else "▶ Start replay", type="primary", on_click=toggle_play, width="stretch")
    c[1].button("⟲ Restart", on_click=restart, width="stretch")
    c[2].selectbox("Jump to a running day", list(jump_options), index=len(jump_options) - 1, key="jump_sel", on_change=on_jump)
    c[3].selectbox("Replay speed", list(SPEEDS), index=1, key="speed_sel")
    c[4].selectbox("Chart window", list(WINDOWS), index=1, key="win_sel")

    def live_view():
        cur, step, win = int(st.session_state.cursor), SPEEDS[st.session_state.speed_sel], WINDOWS[st.session_state.win_sel]
        row, w = tl.iloc[cur], tl.iloc[max(0, cur - win): cur + 1]
        kw, phases = row.KW, [row.RKW, row.YKW, row.BKW]; is_on = (not pd.isna(kw)) and kw > 1
        before = tl.KW.iloc[max(0, cur - 5)]
        delta = "" if pd.isna(kw) or pd.isna(before) else f'<span class="chip">{kw - before:+,.0f} kW vs 5 min ago</span>'
        mean_ph = np.nanmean(phases) if not np.isnan(phases).all() else np.nan
        imb = (np.nanmax(np.abs(np.array(phases) - mean_ph)) / mean_ph * 100) if is_on else np.nan
        imb_tone = "idle" if pd.isna(imb) else "run" if imb < 3 else "warn" if imb < 5 else "bad"
        pill = ('<span class="pill run"><span class="dot"></span>RUNNING</span>' if is_on else
                '<span class="pill idle"><span class="dot"></span>NO SIGNAL</span>' if pd.isna(kw) else '<span class="pill idle"><span class="dot"></span>IDLE · 0 kW</span>')
        left, right = st.columns([1, 2.6])
        with left:
            st.markdown(f'<div class="live">{pill}<div class="big">{"–" if pd.isna(kw) else f"{kw:,.0f}"}<small> kW</small></div><div style="color:#64748B;font-size:.85rem">{row.Timestamps:%a %d %b %Y · %H:%M}</div>'
                        f'<div style="margin:.5rem 0 .9rem">{delta}</div>{phase_bars(phases, 330)}'
                        f'<div style="margin-top:.9rem" class="pill {imb_tone}">Phase imbalance {"–" if pd.isna(imb) else f"{imb:.1f} %"}</div></div>', unsafe_allow_html=True)
        with right, st.container(border=True, key="panel_livechart"):
            f = go.Figure(); f.add_scatter(x=w.Timestamps, y=w.KW, name="Load", line=dict(color=LOAD, width=2.2), fill="tozeroy", fillcolor="rgba(245,158,11,.18)", connectgaps=False)
            if not pd.isna(kw): f.add_scatter(x=[row.Timestamps], y=[kw], mode="markers", marker=dict(size=13, color=LOAD, line=dict(color="#fff", width=3)), showlegend=False)
            f.update_yaxes(range=[0, 750], title="kW"); f.update_xaxes(range=[w.Timestamps.iloc[0], w.Timestamps.iloc[0] + pd.Timedelta(minutes=win)])
            show(style(f, 360, False, top=20), key="live_chart")
            st.caption(f"Replaying recorded data · minute {cur + 1:,} of {len(tl):,} · stream advances {step} min of data per second")
        k = st.columns(4)
        volt = np.nanmean([row.V1, row.V2, row.V3]); amps = np.nanmean([row.I1, row.I2, row.I3])
        with k[0]: kpi("Voltage (avg)", "–" if pd.isna(volt) else f"{volt:,.0f} V", "three-phase mean", "", "🔌", w[["V1", "V2", "V3"]].mean(axis=1).values, CYAN)
        with k[1]: kpi("Current (avg)", "–" if pd.isna(amps) else f"{amps:,.0f} A", f"I1 {row.I1:,.0f} · I2 {row.I2:,.0f} · I3 {row.I3:,.0f}" if is_on else "no load", "", "🧲", w[["I1", "I2", "I3"]].mean(axis=1).values, VIOLET)
        with k[2]: kpi("Power factor", "–" if pd.isna(row.PF) else f"{row.PF:.2f}", "1.00 = ideal", "good" if is_on and row.PF > .9 else "", "🎯")
        with k[3]: kpi("Apparent power", "–" if pd.isna(row.KVA) else f"{row.KVA:,.0f} kVA", "kW ÷ power factor", "", "⚡", w.KVA.values, LOAD)
        if st.session_state.playing:
            if cur >= len(tl) - 1: st.session_state.playing = False; st.rerun()
            st.session_state.cursor = min(cur + step, len(tl) - 1)

    st.fragment(run_every=1.0 if st.session_state.playing else None)(live_view)()

# ================================================================== LOAD PREDICTION (running data only)
with tab_pred:
    with st.expander("ARIMA settings", expanded=False):
        c = st.columns(7)
        p, d_, q = c[0].number_input("p (AR)", 0, 6, 2), c[1].number_input("d (diff)", 0, 2, 0), c[2].number_input("q (MA)", 0, 6, 2)
        win = c[3].number_input("Rolling window (min)", 1, 120, 30); split_pct = c[4].slider("Train split %", 50, 90, 80, 5); block = c[5].number_input("Backtest block (min)", 5, 240, 30, 5)
        horizon = c[6].number_input("Forecast ahead (min)", 5, 120, 30, 5)
    R = predict_kw(df, (p, d_, q), split_pct, int(block), int(win)); mt, mr = R["m_test"], R["m_train"]
    NX = forecast_next(df, (p, d_, q), int(win), int(horizon))

    # ---- next-30-minute forecast
    end_value, lo, hi = NX["mean"][-1], NX["lower"][-1], NX["upper"][-1]; chg = (end_value - NX["last_value"]) / NX["last_value"] * 100
    with panel(f"Next {int(horizon)} minutes", f"ARIMA{(p, d_, q)} forecast from the last running reading · {NX['last_time']:%d %b %Y %H:%M}", key="next"):
        left, right = st.columns([1, 2.4])
        with left:
            st.markdown(f"""<div class="hero-f"><div class="l">Expected load</div><div class="big">{end_value:,.0f}<small> kW</small></div>
<div class="r">95 % range <b>{lo:,.0f} – {hi:,.0f} kW</b> at {NX['times'][-1]:%H:%M}</div><span class="badge flat">{'▲' if chg > 0 else '▼'} {abs(chg):.1f} % vs last {int(win)}-min average</span>
<div class="mini"><div><b>{NX['last_value']:,.0f}</b><span>last {int(win)}-min avg kW</span></div><div><b>±{mt['RMSE']:.1f}</b><span>typical test error kW</span></div><div><b>{int(horizon)} min</b><span>horizon</span></div></div></div>""", unsafe_allow_html=True)
        with right:
            hist = NX["history"]; raw_tail = df[df.Timestamps > hist.index[0]]; f = go.Figure()
            f.add_shape(type="rect", xref="x", yref="paper", x0=NX["last_time"], x1=NX["times"][-1], y0=0, y1=1, fillcolor="rgba(99,102,241,.06)", line_width=0)
            f.add_scatter(x=raw_tail.Timestamps, y=raw_tail.KW, name="Recorded KW (1 min)", line=dict(color="#CBD5E1", width=1))
            f.add_scatter(x=hist.index, y=hist.values, name=f"Rolling {int(win)}-min KW", line=dict(color=INK, width=2.2))
            f.add_scatter(x=list(NX["times"]) + list(NX["times"][::-1]), y=list(NX["upper"]) + list(NX["lower"][::-1]), fill="toself", fillcolor="rgba(99,102,241,.16)", line=dict(width=0), name="95 % range", hoverinfo="skip")
            f.add_scatter(x=NX["times"], y=NX["mean"], name="Forecast", line=dict(color=FORECAST, width=3))
            f.add_shape(type="line", xref="x", yref="paper", x0=NX["last_time"], x1=NX["last_time"], y0=0, y1=1, line=dict(color=MUTED, width=1.2, dash="dot"))
            f.add_annotation(x=NX["last_time"], yref="paper", y=1.0, text="last running reading", showarrow=False, xanchor="right", yanchor="bottom", font=dict(size=11, color=MUTED))
            f.update_yaxes(title="kW"); show(style(f, 340, True, top=70))
        with st.expander(f"Forecast table · every 5 minutes"):
            table(pd.DataFrame({"Time": [t.strftime("%d %b %H:%M") for t in NX["times"][4::5]], "Forecast kW": NX["mean"][4::5], "Lower 95 %": NX["lower"][4::5], "Upper 95 %": NX["upper"][4::5]}), 1)
    if not S["now_running"]:
        note(f"The meter's last <b>running</b> reading is <b>{S['last_run']:%d %b %Y %H:%M}</b>; from then until {S['last_seen']:%d %b} it logs zero (machine off). This forecast therefore shows what the load would be "
             f"<i>if the machine kept running</i> from its last operating state. With live data, it would update from the newest reading.", "amber")

    st.markdown('<div class="ph" style="margin-top:.6rem"><b>Model check on past data</b><span>how well the same ARIMA predicted data it had not seen</span></div>', unsafe_allow_html=True)
    k = st.columns(4)
    with k[0]: kpi("Test RMSE", f"{mt['RMSE']:.1f} kW", f"{len(R['test']):,} forecast points", "info", "🎯")
    with k[1]: kpi("Test MAE", f"{mt['MAE']:.1f} kW", f"{mt['MAE'] / R['test'].Actual.mean() * 100:.1f} % of mean load", "", "📏")
    with k[2]: kpi("Test R²", f"{mt['R2']:.3f}", "1 = perfect · 0 = no better than the average", "good" if mt["R2"] > .5 else "warn", "✅")
    with k[3]: kpi("Train fit R²", f"{mr['R2']:.3f}", f"RMSE {mr['RMSE']:.1f} kW · {len(R['train']):,} points", "", "🧪")
    st.write("")
    s1, s2, s3 = st.tabs(["Test forecast", "Training fit", "Residuals"])

    def line_pair(frame, a, b, title, name_b):
        t, ya, yb = gaps(frame.Timestamp, frame[a], frame[b]); f = go.Figure()
        f.add_scatter(x=t, y=ya, name="Actual rolling KW", line=dict(color=INK, width=1.8), connectgaps=False)
        f.add_scatter(x=t, y=yb, name=name_b, line=dict(color=FORECAST, width=2, dash="dash"), connectgaps=False)
        f.update_layout(title=title, yaxis_title="Rolling KW"); f.update_xaxes(rangeslider_visible=True); return f

    with s1:
        with panel("Test: actual vs forecast", f"each {int(block)}-minute block forecast before its actuals are shown · {R['t2']:%d %b %H:%M} → {R['t3']:%d %b %H:%M}", key="test"):
            show(style(line_pair(R["test"], "Actual", "Forecast", "", "ARIMA forecast"), 440, top=60))
        table(R["test"].head(10).assign(Timestamp=lambda x: x.Timestamp.dt.strftime("%Y-%m-%d %H:%M"))[["Timestamp", "Actual", "Forecast", "Residual"]], 2)
    with s2:
        with panel("Training: actual vs fitted", "how closely ARIMA reproduces the data it was trained on", key="train"):
            show(style(line_pair(R["train"], "Actual", "Fitted", "", "ARIMA fitted"), 440, top=60))
    with s3:
        a, b = st.columns(2)
        for col, frame, name, color in ((a, R["train"], "Training", CYAN), (b, R["test"], "Test", VIOLET)):
            t, r = gaps(frame.Timestamp, frame.Residual); f = go.Figure(go.Scatter(x=t, y=r, line=dict(color=color, width=1.3), connectgaps=False))
            f.add_hline(y=0, line_dash="dash", line_color=GREY); f.update_layout(title=f"{name} residuals (actual − predicted)", yaxis_title="kW")
            with col, st.container(border=True, key=f"panel_res_{name}"): show(style(f, 340, False, top=60))

# ================================================================== LOAD BALANCE (running data only)
with tab_bal:
    c = st.columns(4)
    auto = c[0].toggle("Auto order (AIC, train only)", True)
    p2 = c[1].number_input("p (AR)", 0, 6, 1, key="p2", disabled=auto)
    d2 = c[2].number_input("d (diff)", 0, 2, 1, key="d2")
    q2 = c[3].number_input("q (MA)", 0, 6, 1, key="q2", disabled=auto)
    freq, hrs, thr, minc, minl = "30min", 24, 5.0, 2, 10.0
    B = balance(df, freq, auto, int(p2), int(d2), int(q2), hrs, thr, minc, minl)
    if "error" in B:
        st.error(B["error"]); st.stop()
    test, obs, det, met, hz = B["test"], B["obs"], B["det"], B["metrics"], list(HORIZONS_MIN)
    tone = "good" if B["latest"] < thr * .6 else ("warn" if B["latest"] < thr else "bad")

    # ---- next-30-minute phase forecast
    OUT = balance_outlook(df, freq, B["orders"], 2); nxt = OUT["forecast"].iloc[0]; imb_next = float(imbalance_pct(nxt.to_frame().T).iloc[0]); t_next = OUT["forecast"].index[0]
    gtone = "run" if imb_next < thr * .6 else "warn" if imb_next < thr else "bad"; gtxt = "Balanced" if gtone == "run" else "Watch" if gtone == "warn" else "Imbalanced"
    with panel("Next 30 minutes", f"forecast for {t_next:%d %b %H:%M} – {(t_next + pd.Timedelta(freq)):%H:%M} · from the last running data ({OUT['latest_time']:%d %b %H:%M})", key="bal_next"):
        g, pcol = st.columns([1, 1.5])
        with g:
            gmax = max(10.0, thr * 2, imb_next * 1.3)
            f = go.Figure(go.Indicator(mode="gauge+number", value=imb_next, number=dict(suffix=" %", font=dict(size=40, family="JetBrains Mono")),
                                       gauge=dict(axis=dict(range=[0, gmax]), bar=dict(color=INK, thickness=.28), bgcolor="#fff", borderwidth=0,
                                                  steps=[dict(range=[0, thr * .6], color="#D1FAE5"), dict(range=[thr * .6, thr], color="#FEF3C7"), dict(range=[thr, gmax], color="#FFE4E6")],
                                                  threshold=dict(line=dict(color=BAD, width=3), thickness=.8, value=thr))))
            f.update_layout(height=230, margin=dict(l=20, r=20, t=20, b=0), paper_bgcolor="rgba(0,0,0,0)", font=dict(family="Inter, sans-serif")); show(f, key="gauge")
            st.markdown(f'<div style="text-align:center"><span class="pill {gtone}">Forecast imbalance · {gtxt}</span></div>', unsafe_allow_html=True)
        with pcol:
            st.markdown(phase_bars(list(nxt.values), float(max(nxt.max() * 1.15, 1))), unsafe_allow_html=True)
            st.markdown("<div style='height:.3rem'></div>" + "".join(f'<span class="chip" style="margin-right:.4rem">{ph[0]}: {nxt[ph] - OUT["latest"][ph]:+.1f} kW vs last bin</span>' for ph in PHASES), unsafe_allow_html=True)
            st.caption(f"Alert threshold {thr:g} % · imbalance = largest phase deviation from the three-phase mean ÷ the mean.")
    st.write("")
    k = st.columns(4)
    with k[0]: kpi("Imbalance — latest reading", f"{B['latest']:.1f} %", "Balanced" if tone == "good" else "Watch" if tone == "warn" else "Imbalanced", tone, "⚖️")
    with k[1]: kpi("Peak actual imbalance", f"{B['act'][obs].max():.1f} %", f"{B['n_alerts']} bins in sustained alerts", "warn" if B["act"][obs].max() > thr else "good", "🚨")
    with k[2]: kpi("Mean R² (30 min ahead)", f"{met[met.horizon == hz[0]].R2.mean():.2f}", "average of R / Y / B · measured bins only", "info", "🎯")
    with k[3]: kpi("Measured bins scored", f"{B['n_obs']} of {len(test)}", f"{B['share'] * 100:.0f} % of {freq} bins in the file are measured", "warn" if B["n_obs"] < 30 else "good", "🧮")
    note(f"Orders: mean {B['orders']['mean']}, " + ", ".join(f"{ph} {B['orders'][ph]}" for ph in PHASES) + ".")
    xr = [test.index[obs.values].min(), test.index[obs.values].max()]
    u1, u2, u3, u4 = st.tabs(["Phase forecasts", "Imbalance alerts", "Accuracy", "Tables"])
    cols = [BLUE, "#D97706", "#059669", VIOLET]
    with u1:
        f = make_subplots(rows=3, cols=1, shared_xaxes=True, subplot_titles=PHASES, vertical_spacing=.09)
        for i, ph in enumerate(PHASES, 1):
            f.add_scatter(x=test.index, y=test[ph], name="Actual", mode="lines+markers", marker=dict(size=4), line=dict(color=INK, width=2), legendgroup="a", showlegend=i == 1, row=i, col=1)
            for j, h in enumerate(hz): f.add_scatter(x=test.index, y=B["preds"][h][ph].where(obs), name=f"ARIMA {h}", line=dict(color=cols[j], width=1.5), legendgroup=h, showlegend=i == 1, row=i, col=1)
        f.update_layout(title=f"{freq} smoothing · ARIMA forecast per phase (measured bins only)"); f.update_xaxes(range=xr)
        with st.container(border=True, key="panel_phases"): show(style(f, 760, legend_y=1.0, top=130))
    with u2:
        f = go.Figure(); f.add_scatter(x=test.index, y=B["act"], name="Actual", mode="lines+markers", marker=dict(size=4), line=dict(color=INK, width=2.4))
        aa = B["act"][B["a_sus"]]; f.add_scatter(x=aa.index, y=aa, mode="markers", name="actual alert", marker=dict(symbol="circle-open", size=12, color=INK))
        for j, h in enumerate(hz):
            im = imbalance_pct(B["preds"][h]).where(obs); fl = im[alert_flags(B["preds"][h], thr, float(minl), int(minc), obs)]
            f.add_scatter(x=im.index, y=im, name=f"ARIMA {h}", line=dict(color=cols[j], width=1.5)); f.add_scatter(x=fl.index, y=fl, mode="markers", name=f"{h} flagged", marker=dict(symbol="x", size=9, color=cols[j]), showlegend=False)
        f.add_hrect(y0=thr, y1=max(thr * 2, float(B["act"][obs].max()) * 1.1), fillcolor="rgba(244,63,94,.07)", line_width=0); f.add_hline(y=thr, line_dash="dash", line_color=BAD, annotation_text=f"threshold {thr:g} %")
        f.update_layout(title="Load imbalance % on actual vs forecast data (✕ = sustained alert on forecast, ○ = actual sustained alert)", yaxis_title="imbalance %"); f.update_xaxes(range=xr)
        with st.container(border=True, key="panel_alerts"): show(style(f, 480))
    with u3:
        f = go.Figure()
        for ph in PHASES:
            g_ = met[met.phase == ph]; f.add_bar(x=g_.horizon, y=g_.R2, name=ph, marker_color=PHASE_COL[ph])
        f.update_layout(barmode="group", title="R² by horizon and phase (measured bins only)", yaxis_title="R²")
        with st.container(border=True, key="panel_acc"): show(style(f, 400).update_layout(hovermode="closest"))
        st.caption("R² below 0 means worse than predicting the window's average. Scored on measured bins only.")
    with u4:
        st.markdown("**Forecast accuracy**"); table(met, 3)
        st.markdown("**Imbalance detection on forecasts**"); table(det, 3)
