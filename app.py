"""GridSense - Load Prediction + Load Balance (single ARIMA model, light mode).
Prediction  : ARIMA on the 30-min rolling KW        (from Untitled20 notebook)
Load balance: ARIMA on R/Y/B phase kW -> imbalance % (from imbalance_arima_forecasting notebook)
"""
import io
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from plotly.subplots import make_subplots
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from statsmodels.tsa.arima.model import ARIMA

warnings.filterwarnings("ignore")
st.set_page_config(page_title="GridSense", page_icon="⚡", layout="wide", initial_sidebar_state="collapsed")

FILE = Path(__file__).resolve().parent / "SecureElite_DeviceID19_KW_gt_1.xlsx"
PHASES = ["RKW", "YKW", "BKW"]
AMBER, CYAN, VIOLET, GREEN, RED, BLUE, INK, GREY = "#E8920C", "#0891B2", "#7C3AED", "#059669", "#E11D48", "#2563EB", "#0F172A", "#64748B"
PHASE_COL = {"RKW": RED, "YKW": "#D49A06", "BKW": BLUE}
HORIZONS_MIN = {"30 minute": 30, "1 hour": 60, "3 hours": 180, "6 hours": 360}

# ------------------------------------------------------------------ light theme (forced, no config file needed)
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Space+Grotesk:wght@400;500;600;700&family=JetBrains+Mono:wght@500;600&display=swap');
:root { color-scheme: light; }
html, body, .stApp, [data-testid="stAppViewContainer"], [data-testid="stHeader"], [data-testid="stMain"] { background:#F6F7FB !important; color:#0F172A !important; font-family:'Space Grotesk',sans-serif; }
[data-testid="stHeader"] { height:0; } #MainMenu, footer { visibility:hidden; }
.block-container { max-width:1240px; padding-top:1.4rem; padding-bottom:3rem; }
.stApp p, .stApp label, .stApp li, .stApp h1, .stApp h2, .stApp h3, [data-testid="stWidgetLabel"] p, [data-testid="stCaptionContainer"] { color:#0F172A !important; }
[data-testid="stCaptionContainer"] p { color:#64748B !important; }
[data-baseweb="select"] > div, [data-baseweb="input"], [data-baseweb="base-input"], input, [data-testid="stNumberInputContainer"] { background:#FFFFFF !important; color:#0F172A !important; border-color:#D9DEEA !important; }
[data-baseweb="popover"] *, ul[role="listbox"], li[role="option"] { background:#FFFFFF !important; color:#0F172A !important; }
li[role="option"]:hover { background:#FFF3DC !important; }
[data-testid="stExpander"] { background:#FFFFFF; border:1px solid #E2E6F0; border-radius:14px; }
[data-testid="stExpander"] summary, [data-testid="stExpander"] summary p { color:#0F172A !important; }
[data-testid="stFileUploader"] section { background:#FFFFFF !important; border:1px dashed #CBD3E3; }
[data-testid="stFileUploader"] section *, [data-testid="stFileUploader"] small { color:#475569 !important; }
button[kind], .stButton button, [data-testid="stBaseButton-secondary"] { background:#FFFFFF !important; color:#0F172A !important; border:1px solid #D9DEEA !important; }
.stTabs [data-baseweb="tab-list"] { gap:.4rem; border-bottom:1px solid #E2E6F0; }
.stTabs [data-baseweb="tab"] { color:#64748B !important; font-weight:500; background:transparent !important; }
.stTabs [aria-selected="true"] { color:#0F172A !important; }
.stTabs [data-baseweb="tab-highlight"] { background:#E8920C !important; }
.hero { padding:1.5rem 1.7rem; border-radius:18px; border:1px solid #E2E6F0; margin-bottom:1.1rem; box-shadow:0 1px 3px rgba(15,23,42,.05);
  background: radial-gradient(900px 260px at 92% -30%, rgba(232,146,12,.22), transparent 60%), radial-gradient(700px 240px at 0% 130%, rgba(8,145,178,.12), transparent 60%), #FFFFFF; }
.hero h1 { margin:0; font-size:2rem; font-weight:700; letter-spacing:-.5px; color:#0F172A !important; }
.hero p { margin:.4rem 0 0; color:#475569 !important; max-width:72ch; }
.chip { display:inline-block; padding:.18rem .65rem; border-radius:999px; font-size:.75rem; border:1px solid #D9DEEA; background:#F3F5FA; color:#475569 !important; margin:.8rem .4rem 0 0; }
.kpi { background:#FFFFFF; border:1px solid #E2E6F0; border-radius:14px; padding:.9rem 1.05rem; box-shadow:0 1px 3px rgba(15,23,42,.05); height:100%; }
.kpi .l { color:#64748B !important; font-size:.8rem; } .kpi .s { color:#64748B !important; font-size:.78rem; margin-top:.1rem; }
.kpi .v { font-family:'JetBrains Mono',monospace; font-size:1.65rem; font-weight:600; letter-spacing:-.5px; color:#0F172A !important; margin-top:.15rem; }
.kpi.good .v { color:#059669 !important; } .kpi.warn .v { color:#D97706 !important; } .kpi.bad .v { color:#E11D48 !important; } .kpi.info .v { color:#0891B2 !important; }
.note { border-left:3px solid #E8920C; background:#FFF8EB; padding:.7rem 1rem; border-radius:0 10px 10px 0; color:#475569 !important; font-size:.9rem; margin:.6rem 0 1rem; }
.tblwrap { overflow-x:auto; border:1px solid #E2E6F0; border-radius:12px; background:#FFFFFF; margin:.4rem 0 1rem; }
table.tbl { border-collapse:collapse; width:100%; font-size:.86rem; } table.tbl th { text-align:right; background:#F3F5FA; color:#475569 !important; font-weight:600; padding:.5rem .75rem; border-bottom:1px solid #E2E6F0; }
table.tbl td { text-align:right; padding:.45rem .75rem; border-bottom:1px solid #EEF1F7; color:#0F172A !important; font-family:'JetBrains Mono',monospace; } table.tbl th:first-child, table.tbl td:first-child { text-align:left; }
</style>""", unsafe_allow_html=True)


def kpi(label, value, sub="", tone=""):
    st.markdown(f'<div class="kpi {tone}"><div class="l">{label}</div><div class="v">{value}</div><div class="s">{sub}</div></div>', unsafe_allow_html=True)


def note(text):
    st.markdown(f'<div class="note">{text}</div>', unsafe_allow_html=True)


def table(df, nd=3):
    html = df.round(nd).to_html(index=False, classes="tbl", border=0, na_rep="–")
    st.markdown('<div class="tblwrap">' + "".join(x.strip() for x in html.splitlines()) + "</div>", unsafe_allow_html=True)


def style(fig, h=440, legend=True):
    fig.update_layout(height=h, template="plotly_white", paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)", hovermode="x unified",
                      font=dict(family="Space Grotesk, sans-serif", size=13, color="#334155"), margin=dict(l=10, r=10, t=50, b=10),
                      legend=dict(orientation="h", y=1.12, x=0, bgcolor="rgba(0,0,0,0)"), showlegend=legend)
    fig.update_xaxes(gridcolor="#E6EAF2", zeroline=False); fig.update_yaxes(gridcolor="#E6EAF2", zeroline=False)
    return fig


def gaps(t, *arrays):
    """Break plot lines wherever the timestamp jumps by more than 1 minute (as in the notebook)."""
    t = pd.DatetimeIndex(t).values; brk = np.where(np.diff(t) > np.timedelta64(1, "m"))[0] + 1
    if not len(brk): return (t, *[np.asarray(a, float) for a in arrays])
    return (np.insert(t, brk, t[brk - 1] + np.timedelta64(30, "s")), *[np.insert(np.asarray(a, float), brk, np.nan) for a in arrays])


# ------------------------------------------------------------------ data
@st.cache_data(show_spinner="Reading meter data…")
def load(upload=None):
    df = pd.read_excel(io.BytesIO(upload) if upload else FILE)
    df.columns = df.columns.astype(str).str.strip()
    df["Timestamps"] = pd.to_datetime(df["Timestamps"], errors="coerce")
    for c in ["KW"] + PHASES: df[c] = pd.to_numeric(df[c], errors="coerce")
    return df.dropna(subset=["Timestamps", "KW"]).sort_values("Timestamps").drop_duplicates("Timestamps").reset_index(drop=True)


# ------------------------------------------------------------------ Load prediction (Untitled20 notebook)
@st.cache_data(show_spinner="Fitting ARIMA on the training data…")
def predict_kw(df, order, split_pct, block, win):
    series = df.set_index("Timestamps")["KW"].rolling(win, min_periods=win).mean().dropna().astype(float)
    split = int(len(series) * split_pct / 100); train, test = series.iloc[:split], series.iloc[split:]
    train_reset = pd.Series(train.to_numpy(), name="KW")
    fit = ARIMA(train_reset, order=order, enforce_stationarity=False, enforce_invertibility=False).fit()
    fitted = np.asarray(fit.predict(start=0, end=len(train_reset) - 1, dynamic=False)).reshape(-1)
    tr = pd.DataFrame({"Timestamp": train.index, "Actual": train.to_numpy(), "Fitted": fitted}); tr["Residual"] = tr.Actual - tr.Fitted
    cur, pred, act, tim = len(train_reset), [], [], []
    res = fit
    for s in range(0, len(test), block):                    # forecast a whole block BEFORE seeing it, then append actuals (no refit)
        b = test.iloc[s:s + block]; pred.extend(np.asarray(res.forecast(steps=len(b))).reshape(-1)); act.extend(b.to_numpy()); tim.extend(b.index)
        res = res.append(pd.Series(b.to_numpy(), index=range(cur, cur + len(b)), name="KW"), refit=False); cur += len(b)
    te = pd.DataFrame({"Timestamp": tim, "Actual": act, "Forecast": pred}).sort_values("Timestamp").reset_index(drop=True); te["Residual"] = te.Actual - te.Forecast
    m = lambda a, p: dict(RMSE=float(np.sqrt(mean_squared_error(a, p))), MAE=float(mean_absolute_error(a, p)), R2=float(r2_score(a, p)))
    return dict(train=tr, test=te, m_train=m(tr.Actual, tr.Fitted), m_test=m(te.Actual, te.Forecast), n=len(series),
                t0=train.index[0], t1=train.index[-1], t2=test.index[0], t3=test.index[-1])


# ------------------------------------------------------------------ Load balance (imbalance_arima_forecasting notebook)
def imbalance_pct(frame):
    mean = frame[PHASES].mean(axis=1).replace(0, np.nan)
    return frame[PHASES].sub(mean, axis=0).abs().max(axis=1) / mean * 100


@st.cache_data(show_spinner="Rolling ARIMA forecast for R / Y / B …")
def balance(df, freq, order, test_hours, thr, measured_only):
    step = pd.Timedelta(freq).total_seconds() / 60
    hz = {k: max(1, round(v / step)) for k, v in HORIZONS_MIN.items()}
    x = df.set_index("Timestamps")[PHASES].apply(pd.to_numeric, errors="coerce").resample(freq).mean()
    measured = x.notna().all(axis=1); data = x.interpolate(limit_direction="both")
    test_steps = int(test_hours * 60 / step); max_steps = max(hz.values()); warm = max_steps - 1; n_roll = test_steps + warm
    test = data.iloc[-test_steps:]; mask = measured.iloc[-test_steps:].values

    def rolling(s):
        fe = len(s) - n_roll; r = ARIMA(s.iloc[:fe], order=order).fit(); out = np.full((n_roll, max_steps), np.nan)
        for i in range(n_roll):
            out[i] = r.forecast(max_steps).values; r = r.append(s.iloc[[fe + i]], refit=False)
        return out

    raw = {f: rolling(data[f]) for f in PHASES}
    preds = {k: pd.DataFrame({f: raw[f][warm + np.arange(test_steps) - s + 1, s - 1] for f in PHASES}, index=test.index) for k, s in hz.items()}
    use = mask if (measured_only and mask.sum() >= 8) else np.ones(len(test), bool)
    rows, det = [], []
    act = imbalance_pct(test); aflag = (act > thr).values
    for k, fr in preds.items():
        for f in PHASES:
            a, p = test[f].values[use], fr[f].values[use]
            rows.append(dict(horizon=k, phase=f, MAE=mean_absolute_error(a, p), RMSE=np.sqrt(mean_squared_error(a, p)), R2=r2_score(a, p)))
        imb = imbalance_pct(fr); flag = (imb > thr).values; tp = int((flag & aflag & use).sum()); pf = int((flag & use).sum()); af = int((aflag & use).sum())
        det.append({"horizon": k, "mean imbalance %": imb.values[use].mean(), "max imbalance %": imb.values[use].max(), "actual alerts": af, "predicted alerts": pf,
                    "hits": tp, "precision": tp / pf if pf else np.nan, "recall": tp / af if af else np.nan})
    return dict(test=test, mask=mask, use=use, preds=preds, metrics=pd.DataFrame(rows), det=pd.DataFrame(det), act=act, share=float(measured.mean()), latest=float(imbalance_pct(df.tail(1)).iloc[0]))


# ------------------------------------------------------------------ page
up = st.session_state.get("upload")
df = load(up)
st.markdown(f"""<div class="hero"><h1>⚡ GridSense</h1><p>Load prediction and three-phase load balance for one metered feeder, forecast with ARIMA.</p>
<span class="chip">{len(df):,} readings</span><span class="chip">{df.Timestamps.min():%d %b} → {df.Timestamps.max():%d %b %Y}</span><span class="chip">avg {df.KW.mean():,.0f} kW</span></div>""", unsafe_allow_html=True)

tab1, tab2 = st.tabs(["Load Prediction", "Load Balance"])

with tab1:
    with st.expander("ARIMA settings", expanded=False):
        c = st.columns(6)
        p, d_, q = c[0].number_input("p (AR)", 0, 6, 2), c[1].number_input("d (diff)", 0, 2, 0), c[2].number_input("q (MA)", 0, 6, 2)
        win = c[3].number_input("Rolling window (min)", 1, 120, 30); split_pct = c[4].slider("Train split %", 50, 90, 80, 5); block = c[5].number_input("Forecast block (min)", 5, 240, 30, 5)
    R = predict_kw(df, (p, d_, q), split_pct, int(block), int(win)); mt, mr = R["m_test"], R["m_train"]
    k = st.columns(4)
    with k[0]: kpi("Test RMSE", f"{mt['RMSE']:.1f} kW", f"{len(R['test']):,} forecast points", "info")
    with k[1]: kpi("Test MAE", f"{mt['MAE']:.1f} kW", f"{mt['MAE'] / R['test'].Actual.mean() * 100:.1f} % of mean load")
    with k[2]: kpi("Test R²", f"{mt['R2']:.3f}", "1 = perfect · 0 = no better than the average", "good" if mt["R2"] > .5 else "warn")
    with k[3]: kpi("Train fit R²", f"{mr['R2']:.3f}", f"RMSE {mr['RMSE']:.1f} kW · {len(R['train']):,} points")
    st.write("")
    s1, s2, s3 = st.tabs(["Test forecast", "Training fit", "Residuals"])

    def line_pair(frame, a, b, title, name_b):
        t, ya, yb = gaps(frame.Timestamp, frame[a], frame[b]); f = go.Figure()
        f.add_scatter(x=t, y=ya, name="Actual rolling KW", line=dict(color=INK, width=1.8), connectgaps=False)
        f.add_scatter(x=t, y=yb, name=name_b, line=dict(color=AMBER, width=2, dash="dash"), connectgaps=False)
        f.update_layout(title=title, yaxis_title="Rolling KW"); f.update_xaxes(rangeslider_visible=True); return f

    with s1:
        st.plotly_chart(style(line_pair(R["test"], "Actual", "Forecast", f"ARIMA{(p, d_, q)} — test actual vs forecast", "ARIMA forecast"), 480), width="stretch")
        note(f"Test period {R['t2']:%d %b %H:%M} → {R['t3']:%d %b %H:%M}. Each {int(block)}-minute block is forecast <i>before</i> its actual values are shown to the model; parameters are never refit.")
        table(R["test"].head(10).assign(Timestamp=lambda x: x.Timestamp.dt.strftime("%Y-%m-%d %H:%M"))[["Timestamp", "Actual", "Forecast", "Residual"]], 2)
    with s2: st.plotly_chart(style(line_pair(R["train"], "Actual", "Fitted", f"ARIMA{(p, d_, q)} — training actual vs fitted", "ARIMA fitted"), 480), width="stretch")
    with s3:
        a, b = st.columns(2)
        for col, frame, name, color in ((a, R["train"], "Training", CYAN), (b, R["test"], "Test", VIOLET)):
            t, r = gaps(frame.Timestamp, frame.Residual); f = go.Figure(go.Scatter(x=t, y=r, line=dict(color=color, width=1.3), connectgaps=False))
            f.add_hline(y=0, line_dash="dash", line_color=GREY); f.update_layout(title=f"{name} residuals (actual − predicted)", yaxis_title="kW")
            col.plotly_chart(style(f, 360, False), width="stretch")

with tab2:
    with st.expander("Balance settings", expanded=False):
        c = st.columns(6)
        freq = c[0].selectbox("Smoothing", ["30min", "15min"]); p2, d2, q2 = c[1].number_input("p (AR)", 0, 5, 1, key="p2"), c[2].number_input("d (diff)", 0, 2, 1, key="d2"), c[3].number_input("q (MA)", 0, 5, 1, key="q2")
        hrs = c[4].slider("Test window (hours)", 12, 168, 24, 12); thr = c[5].slider("Alert threshold %", 1.0, 15.0, 5.0, .5)
        mo = st.toggle("Score only on measured bins (ignore interpolated gaps)", False)
    B = balance(df, freq, (p2, d2, q2), hrs, thr, mo); test, det, met, hz = B["test"], B["det"], B["metrics"], list(HORIZONS_MIN)
    tone = "good" if B["latest"] < thr * .6 else ("warn" if B["latest"] < thr else "bad")
    k = st.columns(4)
    with k[0]: kpi("Imbalance — latest reading", f"{B['latest']:.1f} %", "Balanced" if tone == "good" else "Watch" if tone == "warn" else "Imbalanced", tone)
    with k[1]: kpi("Peak actual imbalance", f"{B['act'][B['use']].max():.1f} %", f"{int((B['act'] > thr)[B['use']].sum())} bins above {thr:g} %", "warn" if B["act"][B["use"]].max() > thr else "good")
    with k[2]: kpi("Mean R² (30 min ahead)", f"{met[met.horizon == hz[0]].R2.mean():.2f}", "average of R / Y / B", "info")
    with k[3]: kpi("Measured bins", f"{B['share'] * 100:.0f} %", f"{int(B['use'].sum())} of {len(test)} scored", "warn" if B["share"] < .5 else "good")
    if B["share"] < .5:
        note(f"Only <b>{B['share'] * 100:.0f} %</b> of {freq} bins in the file are real measurements; the rest are filled by interpolation (as in the notebook). Accuracy is flattered by interpolated bins — switch on <i>Score only on measured bins</i> for an honest figure.")
    st.write("")
    u1, u2, u3, u4 = st.tabs(["Phase forecasts", "Imbalance alerts", "Accuracy", "Tables"])
    cols = [BLUE, "#D97706", "#059669", VIOLET]
    with u1:
        f = make_subplots(rows=3, cols=1, shared_xaxes=True, subplot_titles=PHASES, vertical_spacing=.07)
        for i, ph in enumerate(PHASES, 1):
            f.add_scatter(x=test.index, y=test[ph], name="Actual", line=dict(color=INK, width=2), legendgroup="a", showlegend=i == 1, row=i, col=1)
            for j, h in enumerate(hz): f.add_scatter(x=test.index, y=B["preds"][h][ph], name=f"ARIMA {h}", line=dict(color=cols[j], width=1.5), legendgroup=h, showlegend=i == 1, row=i, col=1)
        f.update_layout(title=f"{freq} smoothing · ARIMA{(p2, d2, q2)} forecast per phase"); st.plotly_chart(style(f, 760), width="stretch")
    with u2:
        f = go.Figure(); f.add_scatter(x=test.index, y=B["act"], name="Actual", line=dict(color=INK, width=2.4))
        for j, h in enumerate(hz):
            im = imbalance_pct(B["preds"][h]); fl = im[im > thr]
            f.add_scatter(x=im.index, y=im, name=f"ARIMA {h}", line=dict(color=cols[j], width=1.5)); f.add_scatter(x=fl.index, y=fl, mode="markers", name=f"{h} flagged", marker=dict(symbol="x", size=9, color=cols[j]), showlegend=False)
        f.add_hrect(y0=thr, y1=max(thr * 2, float(B["act"].max()) * 1.1), fillcolor="rgba(225,29,72,.07)", line_width=0); f.add_hline(y=thr, line_dash="dash", line_color=RED, annotation_text=f"threshold {thr:g} %")
        f.update_layout(title="Load imbalance % on actual vs forecast data (✕ = flagged)", yaxis_title="imbalance %"); st.plotly_chart(style(f, 480), width="stretch")
        note("Imbalance % = largest phase deviation from the 3-phase mean ÷ the mean × 100. A window is flagged when it crosses the threshold.")
    with u3:
        f = go.Figure()
        for ph in PHASES:
            g = met[met.phase == ph]; f.add_bar(x=g.horizon, y=g.R2, name=ph, marker_color=PHASE_COL[ph])
        f.update_layout(barmode="group", title="R² by horizon and phase", yaxis_title="R²", hovermode="closest"); st.plotly_chart(style(f, 400), width="stretch")
        st.caption("R² falls as the horizon grows: ARIMA is reliable a few steps ahead and drifts toward the mean at 3–6 hours.")
    with u4:
        st.markdown("**Forecast accuracy**"); table(met, 3)
        st.markdown("**Imbalance detection on forecasts**"); table(det, 3)

with st.expander("Use your own meter file"):
    f_ = st.file_uploader("Upload .xlsx (same columns: Timestamps, KW, RKW, YKW, BKW)", type=["xlsx"])
    if f_ is not None and st.session_state.get("upload") != f_.getvalue(): st.session_state["upload"] = f_.getvalue(); st.rerun()
    if up is not None and st.button("Back to bundled dataset"): del st.session_state["upload"]; st.rerun()
    st.caption("Using: " + ("uploaded file" if up is not None else "bundled SecureElite · Device 19"))
