"""GridSense rule-based advisor - no AI, no API key, no internet.

Every sentence the app shows about phase balance is produced here from measured numbers and the forecast, by fixed rules:
  what is out of phase  ->  why  ->  what to do now  ->  when to do what  ->  what can happen  ->  how to bring it in phase  ->  future load balance
This module has no Streamlit import, so it can be tested on its own.
"""
import numpy as np
import pandas as pd

NAMES = ["Red", "Yellow", "Blue"]


# ------------------------------------------------------------------ imbalance maths
def analyse(P, thr, min_kw, sev):
    """P = kW (or A, or V) of Red, Yellow, Blue in 3 columns.  imbalance % = max(|R-avg|, |Y-avg|, |B-avg|) / avg x 100,  avg = (R+Y+B) / 3.
    170 / 169 / 160 kW -> 3.81 %.  A row is 'loaded' when the average phase value is above min_kw (otherwise the machine is off)."""
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


def transfers(vals, names=NAMES):
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


def level(imb, thr, sev):
    """'good' / 'amber' / 'bad' for an imbalance %."""
    if imb is None or not np.isfinite(imb): return "good"
    return "bad" if imb > sev else "amber" if imb > thr else "good"


def nema_derating(vu):
    """Motor derating factor for a voltage unbalance of vu % (NEMA MG-1 curve, straight lines between the published points)."""
    return float(np.interp(vu, [0, 1, 2, 3, 4, 5], [1.0, 1.0, 0.95, 0.88, 0.82, 0.75]))


# ------------------------------------------------------------------ when to act (from the forecast)
def best_windows(total, times, span=2, horizon=48):
    """Quietest and busiest `span`-hour window in the first `horizon` forecast hours (hourly steps).
    Returns dict(quiet=(start, stop, avg_kw), busy=(start, stop, avg_kw)) or None."""
    s = pd.Series(np.asarray(total, float)[:horizon], index=pd.DatetimeIndex(times[:horizon]))
    if len(s) < span: return None
    roll = s.rolling(span).mean().dropna()
    if roll.empty: return None
    mk = lambda end: (end - pd.Timedelta(hours=span - 1), end + pd.Timedelta(hours=1), float(roll[end]))
    return dict(quiet=mk(roll.idxmin()), busy=mk(roll.idxmax()))


def _t(t): return f"{t:%a %d %b %H:%M}"


def _moves_text(mv, unit, nd=0, min_amt=0.5):
    parts = [f"<b>{a:,.{nd}f} {unit}</b> from <b>{f}</b> to <b>{t}</b>" for f, t, a in mv if a >= min_amt]
    return "move " + " and ".join(parts) if parts else ""


# ------------------------------------------------------------------ the playbook
def playbook(dg, now, fc, thr, sev, min_kw, rating=0.0, demand=0.0, reliability=None, real_clock=True):
    """dg  = measured diagnosis (dict from the app's diagnose(), or None when there are too few loaded minutes)
    now = dict(loaded, imb, worst, direction, status) for the latest reading
    fc  = dict(P=DataFrame hourly forecast with columns R, Y, B and a time index, rep=analyse() of it) or None
    reliability = (label, text) from the backtest or None
    Returns dict(level, head, sections=[(key, title, tone, [html lines])], when=DataFrame, target=DataFrame, target_fc=DataFrame, daily=DataFrame)."""
    out = dict(sections=[], when=pd.DataFrame(), target=pd.DataFrame(), target_fc=pd.DataFrame(), daily=pd.DataFrame())
    n_al = n_sev = n_pts = 0; first_alarm = None; worst_t = None; worst_imb = np.nan; wins = None; Pf = rep = None
    if fc is not None:
        Pf, rep = fc["P"], fc["rep"]; n_pts = len(Pf); times = Pf.index
        n_al, n_sev = int(rep.Alarm.sum()), int((rep.Status == "SEVERE").sum())
        if n_al: first_alarm = times[int(np.flatnonzero(rep.Alarm.to_numpy())[0])]
        imb_f = rep.Imbalance_pct.to_numpy()
        if np.isfinite(imb_f).any(): k = int(np.nanargmax(imb_f)); worst_t, worst_imb = times[k], float(imb_f[k])
        wins = best_windows(Pf.sum(axis=1).to_numpy(), times)

    mi = dg["mean_imb"] if dg else (now["imb"] if now.get("loaded") else 0.0)
    lvl_m = level(mi, thr, sev); lvl_f = "bad" if n_sev else ("amber" if n_al else "good")
    rank = {"good": 0, "amber": 1, "bad": 2}; lvl = max(lvl_m, lvl_f, key=rank.get)
    nm = dg["main_name"] if dg else (now.get("worst") or "-")
    high = (dg["high_share"] >= 50) if dg else (now.get("direction") == "HIGH")
    word = "too much" if high else "too little"
    head = (f"Out of phase: <b>{nm}</b> carries {word} load - average imbalance {mi:.1f} % against the {thr:g} % limit" if lvl_m != "good" else
            f"Measured balance is inside the limit ({mi:.1f} % against {thr:g} %)")
    if lvl_f != "good": head += f" · forecast: {n_al} of {n_pts} hours in alarm" + (f", first at {_t(first_alarm)}" if first_alarm is not None else "")
    elif fc is not None: head += f" · forecast: no alarm in the next {n_pts} hours"
    out["level"], out["head"] = lvl, head

    # 1 ---------------- what is out of phase
    s1 = []
    if dg:
        per = dg["per"]
        s1.append(f"<b>Measured (all loaded minutes):</b> average imbalance <b>{mi:.1f} %</b>, above the {thr:g} % limit in <b>{dg['pct_above']:.0f} %</b> of the minutes (95 % of minutes below {dg['p95_imb']:.1f} %).")
        s1.append(f"<b>{nm}</b> is the worst phase in {per.iloc[dg['main'], 0]:.0f} % of the minutes and is {'HIGH' if high else 'LOW'} in {max(dg['high_share'], 100 - dg['high_share']):.0f} % of them - it carries {dg['extra_A']:+.0f} A ({dg['extra_kW']:+.1f} kW) compared with the other two.")
        s1.append("<b>Meaning:</b> the three phases should carry the same load. 'Out of phase' here means one colour is loaded more or less than the others.")
    if now.get("loaded"):
        s1.append(f"<b>Right now:</b> {now['worst']} is {abs(now['dev']):.1f} % {'above' if now['dev'] > 0 else 'below'} the average of the three phases = <b>{now['imb']:.1f} %</b> ({now['status']}).")
    else: s1.append("<b>Right now:</b> the machine is idle, so there is no imbalance to act on at this moment.")
    if fc is not None:
        if n_al: s1.append(f"<b>Forecast:</b> {n_al} of the next {n_pts} hours are above the limit" + (f" ({n_sev} severe)" if n_sev else "") + f"; the worst hour is <b>{_t(worst_t)}</b> at <b>{worst_imb:.1f} %</b>.")
        else: s1.append(f"<b>Forecast:</b> no hour above the {thr:g} % limit in the next {n_pts} hours" + (f" (highest {worst_imb:.1f} %)." if np.isfinite(worst_imb) else "."))
    out["sections"].append(("what", "What is out of phase", lvl, s1))

    # 2 ---------------- why
    s2 = []
    if dg:
        sh = dg["share"]; top = sh.idxmax()
        if top == "Current":
            s2.append(f"<b>Cause = the load, not the supply.</b> Current explains {sh['Current']:.0f} % of how far {nm} is from the average (voltage {sh['Voltage']:.0f} %, power factor {sh['Power factor']:.0f} %). Single-phase loads are not shared equally between the three colours.")
        elif top == "Voltage":
            s2.append(f"<b>Cause = the supply voltage.</b> Voltage explains {sh['Voltage']:.0f} % of the deviation (average voltage unbalance {dg['mean_V']:.1f} %). Check the incoming supply, the transformer tap and the upstream feeders, and raise it with the utility.")
        else:
            s2.append(f"<b>Cause = power factor.</b> The phases have different power factors ({sh['Power factor']:.0f} % of the deviation). Check capacitor banks and the single-phase loads (rectifiers, drives, lighting) on each phase.")
        if top != "Voltage":
            s2.append(f"<b>Supply is {'not perfect either' if dg['mean_V'] > 1 else 'fine'}:</b> voltage unbalance is {dg['mean_V']:.1f} % on average ({dg['p95_V']:.1f} % at the 95th percentile)." + (" Above 2 % ask the utility to correct it." if dg["mean_V"] > 1 else " It is not the reason."))
        if dg["const"]: s2.append(f"<b>Looks like a fixed load:</b> from the second load level upward {nm} carries about <b>{dg['lvA_const']:.0f} A</b> more than the other two whatever the plant draws - one or a few single-phase loads (lighting, heaters, welders, control or auxiliary panels) wired to {nm}. Find them first.")
        if dg["ctot"] < -0.2: s2.append(f"<b>Worse at light load</b> (correlation {dg['ctot']:.2f} with total kW): the extra single-phase load is the same but it is a bigger share when the plant draws less.")
        elif dg["ctot"] > 0.2: s2.append(f"<b>Worse at heavy load</b> (correlation {dg['ctot']:.2f} with total kW): look for an overloaded phase at full production.")
        if real_clock and len(dg["by_hour"]) >= 6 and dg["by_hour"].max() > 1.25 * dg["by_hour"].min():
            hh, lh = int(dg["by_hour"].idxmax()), int(dg["by_hour"].idxmin())
            s2.append(f"<b>When it is worst:</b> around {hh:02d}:00 ({dg['by_hour'].max():.1f} %), best around {lh:02d}:00 ({dg['by_hour'].min():.1f} %) - check which machines run at {hh:02d}:00.")
    else: s2.append("Not enough loaded minutes to separate current, voltage and power factor. Lower the 'ignore below' limit in Data & limits.")
    out["sections"].append(("why", "Why it happens", "", s2))

    # 3 ---------------- what to do now = how to rectify
    s3 = []
    if dg and len(dg["plan"]):
        steps = " and ".join(f"<b>{r['To']}</b> (about {r['Move A']:.0f} A, {r['Move kW']:.0f} kW)" for _, r in dg["plan"].iterrows())
        first = dg["plan"].iloc[0]
        s3.append(f"<b>1. Move load:</b> take single-phase load off <b>{first['From']}</b> and give it to {steps}.")
        s3.append("<b>2. Choose the loads:</b> a machine cannot be split, so pick whole single-phase loads (lighting, heaters, small motors, control panels, air-conditioners) whose currents add up closest to these amps. Read the real current with a clamp meter before and after.")
        s3.append(f"<b>3. Check the result:</b> come back to this page after one full working day. The aim is to be inside {thr:g} %; if one phase is still the worst, repeat with a smaller move.")
    elif dg: s3.append("The three average phase currents are already within 1 A of each other - there is nothing to move. Keep watching the forecast.")
    if dg and dg["share"].idxmax() == "Voltage": s3.append("<b>Do not rewire:</b> the voltage is the main cause, so moving load will not fix it. Ask the utility / check the transformer tap.")
    if dg and dg["share"].idxmax() == "Power factor": s3.append("<b>Also:</b> check that capacitor banks are balanced across the phases and are not over-compensating at light load.")
    if lvl == "bad":
        if dg and dg["pct_above"] > 50: s3.append("<b>Priority:</b> the imbalance is above the severe limit and present most of the time, so this is a wiring / load-sharing problem to fix first. If the measured amps do not match this page, also check for a loose or failed connection.")
        else: s3.append("<b>Urgent:</b> a large imbalance that comes and goes can be a loose or failed connection, a single-phase fault or a tripped single-phase protection - check these before moving anything.")
    s3.append("<b>Safety:</b> only a qualified electrician should change phase connections, with the circuit isolated and locked out.")
    out["sections"].append(("do", "What to do now (rectify)", "", s3))

    # 4 ---------------- when to do what
    rows = []
    if lvl == "bad": rows.append(("Now", "Check for a fault (loose connection, single-phase trip) and reduce load on the HIGH phase if it is near its rating.", f"imbalance is above the severe limit of {sev:g} %"))
    elif lvl_m != "good": rows.append(("Today", "Measure each phase with a clamp meter and list the single-phase loads on " + nm + ".", f"average imbalance {mi:.1f} % is above the {thr:g} % limit"))
    else: rows.append(("Today", "No rewiring needed. Keep monitoring.", "measured imbalance is inside the limit"))
    if first_alarm is not None:
        hrs = (first_alarm - pd.Timestamp(Pf.index[0])).total_seconds() / 3600 + 1
        rows.append((f"Before {_t(first_alarm)}", "Finish the load move so the forecast alarm does not happen." if lvl_m != "good" else "Check what will change on the plant at this time: the model expects the phases to drift apart.", f"first forecast alarm, about {max(hrs, 0):.0f} h after the last reading"))
    if wins is not None and (lvl_m != "good" or n_al):
        q = wins["quiet"]
        rows.append((f"{_t(q[0])} → {q[1]:%H:%M}", "Best window to move wiring: the lowest-load two hours in the next 48 h.",
                     "the machine is expected to be idle, so no production is lost" if q[2] < 1 else f"forecast average only {q[2]:,.0f} kW, so least production is affected"))
        b = wins["busy"]
        if b[2] >= 1 and b[0] != q[0]: rows.append((f"{_t(b[0])} → {b[1]:%H:%M}", "Avoid switching or rewiring loads.", f"highest-load two hours, forecast average {b[2]:,.0f} kW"))
    if lvl_m != "good" or n_al: rows.append(("1 working day after the change", "Re-open this page, press the backtest if you have not, and compare the average imbalance with the value above.", f"target: below {thr:g} %"))
    rows.append(("Every week", "Look at the daily outlook below; act on the first day that shows alarm hours.", "the forecast updates as new readings arrive"))
    if dg is not None:
        low = NAMES[int(np.argmin(dg["Im"]))]; rows.append(("Whenever a new single-phase load is added", f"Connect it to <b>{low}</b>, the phase that carries the least current today.", "keeps the three phases level in future"))
    out["when"] = pd.DataFrame(rows, columns=["When", "Do what", "Because"])

    # 5 ---------------- what can happen
    s5 = []
    if dg:
        derate = nema_derating(dg["mean_V"]); vu = dg["mean_V"]
        s5.append(f"<b>Overheated {nm} phase:</b> it carries {dg['extra_A']:+.0f} A more than the others, and heat in cables and breakers grows with the square of the current. Extra I²R heating against a balanced load is about <b>{dg['loss']:.1f} %</b>" + (" - small in energy, but the risk is local: the hot phase reaches its limit first." if dg["loss"] < 3 else " - significant, it also costs energy."))
        if dg["neutral_mean"] >= 1: s5.append(f"<b>Neutral overload:</b> in a 4-wire supply the imbalance returns through the neutral, about <b>{dg['neutral_mean']:.0f} A</b> on average ({dg['neutral_p95']:.0f} A at the 95th percentile). An undersized or loose neutral can overheat or burn.")
        s5.append(f"<b>Motors:</b> the supply voltage unbalance is only {vu:.1f} %, so motors need no derating - but unequal phase currents still heat the motor on the loaded phase." if vu <= 1 else
                  f"<b>Motors:</b> voltage unbalance of {vu:.1f} % means motors should be loaded to about <b>{derate * 100:.0f} %</b> of their rating (NEMA curve) and windings run roughly {2 * vu ** 2:.0f} % hotter - ask the utility to correct the supply.")
        pk = dg["per"]["Peak A"]
        if rating > 0:
            u = pk / rating * 100; hot = u.idxmax()
            s5.append(f"<b>Trips and overload:</b> {hot} peaks at {pk[hot]:,.0f} A = <b>{u[hot]:.0f} %</b> of the {rating:,.0f} A rating. " + ("It is over the rating: expect breaker trips, shed its load now." if u[hot] > 100 else "Little headroom: the loaded phase will trip first, stopping the machine without warning." if u[hot] > 85 else "There is headroom, but the loaded phase reaches the limit first."))
        else: s5.append(f"<b>Trips and overload:</b> {pk.idxmax()} peaks at {pk.max():,.0f} A. Enter the breaker / cable rating in Data & limits to see how close it is.")
        if dg["pct_above"] > 50: s5.append("<b>It is permanent, not an event:</b> these effects are present most of the time, so cable ageing and unplanned trips build up silently.")
    if n_sev: s5.append(f"<b>Forecast severe hours ({n_sev}):</b> a large imbalance can trip protection or damage single-phase equipment.")
    elif n_al: s5.append(f"<b>Forecast alarm hours ({n_al}):</b> the effects above will appear mostly in these hours.")
    if demand > 0 and fc is not None:
        pk_f = float(Pf.sum(axis=1).max()); s5.append(f"<b>Demand:</b> the forecast peak is {pk_f:,.0f} kW = {pk_f / demand * 100:.0f} % of the {demand:,.0f} kW sanctioned demand.")
    s5.append("<b>Tariff:</b> some utilities penalise a high power factor spread or imbalance - check your tariff.")
    out["sections"].append(("risk", "What can happen if nothing is done", "amber" if lvl != "good" else "", s5))

    # 6 ---------------- how to make it in phase (target per phase)
    if dg:
        Im, Km = np.asarray(dg["Im"], float), np.asarray(dg["Km"], float); tgt = Im.mean()
        kw_per_A = np.where(Im > 0, Km / np.where(Im > 0, Im, 1), 0)
        out["target"] = pd.DataFrame({"Phase": NAMES, "Now (A)": Im, "Target (A)": [tgt] * 3, "Change (A)": tgt - Im, "Change (kW)": (tgt - Im) * kw_per_A,
                                      "Action": ["take off" if v < -1 else "add" if v > 1 else "keep" for v in (tgt - Im)]})
        cur_imb = float(np.abs(Im - tgt).max() / tgt * 100) if tgt > 0 else 0.0
        s6 = [f"<b>Goal:</b> each phase near <b>{tgt:,.0f} A</b> (the average of the three). Today the current spread is {cur_imb:.1f} %; a perfect move brings it to 0 %, a practical one to below {thr:g} %.",
              "Work from the table: <b>take off</b> the phase with a negative change, <b>add</b> to the phase with a positive change, 'keep' means leave it."]
        out["sections"].append(("inphase", "How to bring it in phase", "", s6))
    if Pf is not None and (n_al or n_sev):
        avg = Pf.mean(axis=0).to_numpy(); mv = transfers(avg)
        out["target_fc"] = pd.DataFrame({"Phase": NAMES, "Forecast avg (kW)": avg, "Target (kW)": [avg.mean()] * 3, "Change (kW)": avg.mean() - avg})

    # 7 ---------------- future load balance
    s7 = []
    if rep is not None:
        d = Pf.copy(); d["imb"] = rep.Imbalance_pct.to_numpy(); d["alarm"] = rep.Alarm.to_numpy(); d["sev"] = (rep.Status == "SEVERE").to_numpy(); d["worst"] = rep.Worst_phase.to_numpy(); d["dir"] = rep.Direction.to_numpy(); d["day"] = d.index.date
        rows = []
        for day, g in d.groupby("day"):
            al = g[g.alarm]; mv = transfers(g[["R", "Y", "B"]].mean().to_numpy())
            culprit = f"{al.worst.mode().iloc[0]} {al.dir[al.worst == al.worst.mode().iloc[0]].mode().iloc[0]}" if len(al) else "-"
            act = ("Move " + " and ".join(f"{a:,.1f} kW from {f_} to {t_}" for f_, t_, a in mv if a >= 0.5)) if len(al) and any(a >= 0.5 for _, _, a in mv) else "No action"
            rows.append((f"{day:%a %d %b}", g.R.mean(), g.Y.mean(), g.B.mean(), g.imb.mean() if g.imb.notna().any() else np.nan, int(g.alarm.sum()), int(g.sev.sum()), culprit, act))
        out["daily"] = pd.DataFrame(rows, columns=["Day", "Red kW", "Yellow kW", "Blue kW", "Avg imbalance %", "Alarm hours", "Severe hours", "Main culprit", "Action"])
        dd = out["daily"]
        if len(dd) >= 2 and dd["Avg imbalance %"].notna().sum() >= 2:
            a, b = float(dd["Avg imbalance %"].dropna().iloc[0]), float(dd["Avg imbalance %"].dropna().iloc[-1])
            s7.append(f"<b>Trend:</b> the daily average imbalance goes from {a:.1f} % to {b:.1f} % over the forecast - " + ("getting worse, plan the move early." if b > a + 0.5 else "getting better." if b < a - 0.5 else "about steady."))
        bad_days = dd[dd["Alarm hours"] > 0]
        if len(bad_days): s7.append(f"<b>Days with alarm hours:</b> {', '.join(bad_days.Day.head(7))}. Act on the first one: {bad_days.Day.iloc[0]}.")
        else: s7.append("<b>No alarm hours</b> in the forecast: keep the present connection and review again next week.")
        culprits = d[d.alarm].worst.value_counts()
        if len(culprits) and dg and culprits.idxmax() != nm: s7.append(f"<b>Note:</b> the forecast alarms are mostly on <b>{culprits.idxmax()}</b>, not on {nm} as in the measured past - the pattern is shifting, so do not move load only by the old numbers.")
    if dg: s7.append(f"<b>Standing rule:</b> add new single-phase loads to <b>{NAMES[int(np.argmin(dg['Im']))]}</b> and keep the three phase currents within {thr:g} % of each other.")
    if reliability: s7.append(f"<b>How far to trust the forecast:</b> {reliability[0].lower()} - {reliability[1]}")
    else: s7.append("<b>How far to trust the forecast:</b> not tested yet - press 'Run backtest' to compare the model with past weeks.")
    s7.append("Beyond about 3 days the forecast drifts towards a typical weekly pattern; use the far end as a guide.")
    out["sections"].append(("future", "Future load balance", "", s7))
    return out
