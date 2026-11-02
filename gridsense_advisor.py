"""GridSense AI advisor - evidence list -> Groq LLM -> independent Python validation (fail closed).

Principles
  * Python is the source of truth for every number. The LLM never types a measured value: it writes {E003}-style placeholders and Python fills in the real value.
  * Everything the LLM returns is re-checked here (schema, evidence ids, stray numbers, priority ceiling, safety wording, missing ratings, forecast reliability).
  * If the LLM is unavailable or its answer fails a critical check, the caller falls back to the rule-based advice. Nothing is ever shown as "verified safe".
This module has no Streamlit import, so it can be tested on its own (see tests/test_advisor.py).
"""
import copy
import hashlib
import json
import os
import re
import time
import urllib.error
import urllib.request
from pathlib import Path

import numpy as np

GROQ_URL = "https://api.groq.com/openai/v1/chat/completions"
DEFAULT_MODEL = "openai/gpt-oss-120b"                # Groq shut llama-3.3-70b-versatile down on 16 Aug 2026; set GROQ_MODEL to change it
FALLBACK_MODELS = ["qwen/qwen3.6-27b", "openai/gpt-oss-20b"]   # tried automatically when Groq says the model does not exist
PRIORITIES = ["low", "medium", "high", "critical"]
CLASSES = {"confirmed", "strongly_supported", "possible", "unknown"}
CONFIDENCE = {"low", "medium", "high"}
PASSED, FAILED, INSUFFICIENT, REVIEW = "PASSED", "FAILED", "INSUFFICIENT_EVIDENCE", "HUMAN_REVIEW_REQUIRED"
REL_MARGIN = 5.0                                    # SARIMAX must beat the best baseline by more than this % to be called RELIABLE (a project setting, not a universal rule)
PH = re.compile(r"\{(E\d{3})\}")


# ------------------------------------------------------------------ API key (never hard-coded)
def load_key(environ=None, env_file=None, secret=None, typed=None):
    """Order: key typed in the app (this session only) > GROQ_API_KEY environment variable > Streamlit secret > .env file next to app.py."""
    environ = os.environ if environ is None else environ
    for k in (typed, environ.get("GROQ_API_KEY"), secret):
        if k and str(k).strip(): return str(k).strip()
    if env_file and Path(env_file).exists():
        for line in Path(env_file).read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if line and not line.startswith("#") and "=" in line:
                name, _, val = line.partition("=")
                if name.strip() == "GROQ_API_KEY" and val.strip().strip("'\""): return val.strip().strip("'\"")
    return None


# ------------------------------------------------------------------ forecast metrics and reliability
def smape(actual, forecast):
    """sMAPE in %, zero denominators counted as 0 error (both values are 0)."""
    a, f = np.asarray(actual, float), np.asarray(forecast, float); den = np.abs(a) + np.abs(f)
    with np.errstate(divide="ignore", invalid="ignore"): r = np.where(den == 0, 0.0, 2 * np.abs(a - f) / den)
    return float(np.nanmean(r) * 100)


def mase(actual, forecast, scale):
    """Mean absolute scaled error: MAE divided by the MAE of the naive one-step change (scale). < 1 = better than the naive step."""
    return float(np.mean(np.abs(np.asarray(actual, float) - np.asarray(forecast, float))) / scale) if scale and scale > 0 else float("nan")


def forecast_reliability(res, margin=REL_MARGIN, min_moments=6):
    """RELIABLE / DEGRADED / UNRELIABLE / INSUFFICIENT DATA from a backtest table (columns: Minutes ahead, Method, Total kW error, Imbalance error (pts), Moments).
    SARIMAX is compared with the BEST baseline at every horizon; it must win by `margin` % at most horizons to be called reliable."""
    if res is None or len(res) == 0: return "INSUFFICIENT DATA", "the backtest has not been run (or had too few usable moments)"
    if int(res["Moments"].iloc[0]) < min_moments: return "INSUFFICIENT DATA", f"only {int(res['Moments'].iloc[0])} backtest moments (need at least {min_moments})"
    wins = worse = 0; hs = sorted(res["Minutes ahead"].unique())
    for h in hs:
        sub = res[res["Minutes ahead"] == h].set_index("Method")
        if "SARIMAX" not in sub.index or len(sub) < 2: continue
        s, base = sub.loc["SARIMAX"], sub.drop("SARIMAX"); gains = []
        for col in ("Total kW error", "Imbalance error (pts)"):
            b = float(base[col].min())
            if np.isfinite(b) and np.isfinite(s[col]): gains.append((1 - s[col] / max(b, 1e-9)) * 100)
        if not gains: continue
        if all(g > margin for g in gains): wins += 1
        elif any(g < -margin for g in gains): worse += 1
    need = int(np.ceil(len(hs) * 2 / 3))
    if wins >= need and worse == 0: return "RELIABLE", f"beats the best baseline by more than {margin:g} % at {wins} of {len(hs)} horizons"
    if worse >= need: return "UNRELIABLE", f"worse than a simple baseline by more than {margin:g} % at {worse} of {len(hs)} horizons"
    return "DEGRADED", f"no clear gain over the best baseline (wins {wins}, worse {worse} of {len(hs)} horizons)"


def load_reliability(test_mae_pct, idle_pct, constant_test):
    """RELIABLE / DEGRADED / UNRELIABLE / INSUFFICIENT DATA for the ARIMA load forecast, from its error on data it had not seen (project rule, not a universal one)."""
    if constant_test or test_mae_pct is None or not np.isfinite(test_mae_pct): return "INSUFFICIENT DATA", "the test period is flat or empty, so the error cannot be judged"
    if test_mae_pct > 25: return "UNRELIABLE", f"test error is {test_mae_pct:.0f} % of the mean load (above 25 %)"
    if test_mae_pct > 10 or (idle_pct or 0) > 50: return "DEGRADED", f"test error is {test_mae_pct:.0f} % of the mean load" + (f" and {idle_pct:.0f} % of the training data is idle" if (idle_pct or 0) > 50 else "")
    return "RELIABLE", f"test error is only {test_mae_pct:.0f} % of the mean load"


# ------------------------------------------------------------------ evidence
def build_evidence(f):
    """Turn the numbers the app measured/calculated (dict `f`) into evidence rows: id, parameter, value, unit, window, source column, method, kind."""
    ev = []; win = f.get("window", "all loaded minutes")

    def add(parameter, value, unit, window, source, method, kind="measured"):
        if value is None or (isinstance(value, float) and not np.isfinite(value)): return
        ev.append(dict(id=f"E{len(ev) + 1:03d}", parameter=parameter, value=round(float(value), 2), unit=unit, window=window, source=source, method=method, kind=kind))

    add("average phase imbalance", f.get("mean_imb"), "%", win, "RKW,YKW,BKW", "max |phase - average| / average x 100, averaged over loaded minutes", "calculated")
    add("95th percentile phase imbalance", f.get("p95_imb"), "%", win, "RKW,YKW,BKW", "95th percentile of the same imbalance", "calculated")
    add("share of loaded minutes above the alarm limit", f.get("pct_above"), "%", win, "RKW,YKW,BKW", "count(imbalance > alarm) / loaded minutes", "calculated")
    add("average current imbalance", f.get("mean_I"), "%", win, "I1,I2,I3", "same formula on the phase currents", "calculated")
    add("average voltage imbalance", f.get("mean_V"), "%", win, "V1,V2,V3", "same formula on the phase voltages", "calculated")
    for ph, src_i, src_k in (("Red", "I1", "RKW"), ("Yellow", "I2", "YKW"), ("Blue", "I3", "BKW")):
        p = (f.get("per") or {}).get(ph, {})
        add(f"{ph} phase average current", p.get("avg_A"), "A", win, src_i, "mean of loaded minutes", "measured")
        add(f"{ph} phase peak current", p.get("peak_A"), "A", win, src_i, "maximum of loaded minutes", "measured")
        add(f"{ph} phase average power", p.get("avg_kW"), "kW", win, src_k, "mean of loaded minutes", "measured")
    add(f"extra current on {f.get('main_name', 'the worst phase')} vs the other two", f.get("extra_A"), "A", win, "I1,I2,I3", "mean difference", "calculated")
    add("estimated neutral current", f.get("neutral_mean"), "A", win, "I1,I2,I3", "sqrt(I1²+I2²+I3²-I1I2-I2I3-I3I1), same power-factor angle assumed", "calculated")
    add("extra I²R heating against a balanced load", f.get("loss"), "%", win, "I1,I2,I3", "sum(I²) / (3 x average²) - 1", "calculated")
    add("meter power factor", f.get("pf_meter"), "", win, "PF", "mean of loaded minutes", "measured")
    for mv in f.get("plan") or []: add(f"current to move from {mv['From']} to {mv['To']}", mv["Move A"], "A", win, "I1,I2,I3", "equalise the three average currents", "calculated")
    add("alarm limit", f.get("thr"), "%", "configuration", "settings", "set in Data & limits", "config")
    add("severe limit", f.get("sev"), "%", "configuration", "settings", "set in Data & limits", "config")
    add("minimum load per phase for 'machine on'", f.get("min_kw"), "kW", "configuration", "settings", "set in Data & limits", "config")
    if f.get("rating"): add("phase breaker / cable rating (user supplied)", f["rating"], "A", "configuration", "user input", "typed by the user, not verified", "config")
    if f.get("demand"): add("sanctioned demand (user supplied)", f["demand"], "kW", "configuration", "user input", "typed by the user, not verified", "config")
    ld = f.get("load")
    if ld:
        w = ld.get("window", "forecast window")
        add("forecast average load", ld.get("avg_kw"), "kW", w, "ARIMA forecast", "mean of the forecast minutes", "forecast")
        add("forecast peak load", ld.get("peak_kw"), "kW", w, "ARIMA forecast", "maximum of the forecast minutes", "forecast")
        add("forecast load at the first point", ld.get("first_kw"), "kW", w, "ARIMA forecast", "first forecast minute", "forecast")
        add("forecast load at the last point", ld.get("last_kw"), "kW", w, "ARIMA forecast", "last forecast minute", "forecast")
        add("forecast trend", ld.get("trend_kw_h"), "kW per hour", w, "ARIMA forecast", "straight-line slope of the block averages", "forecast")
        add("forecast energy needed", ld.get("energy_kwh"), "kWh", w, "ARIMA forecast", "average kW x hours", "forecast")
        add("energy used in the previous period of the same length", ld.get("energy_prev_kwh"), "kWh", "previous period", "KW", "sum of kW / 60", "measured")
        add("average power factor", ld.get("pf_avg"), "", win, "PF", "mean of loaded minutes", "measured")
        add("model test error (MAE) as share of the mean test load", ld.get("test_mae_pct"), "%", "test period", "ARIMA test", "MAE / mean actual load x 100", "calculated")
        add("share of the training timeline that is idle", ld.get("idle_pct"), "%", "training period", "KW", "idle minutes / all minutes", "calculated")
    fc = f.get("forecast")
    if fc:
        w = fc.get("window", "forecast window")
        add("forecast imbalance at the end of the window", fc.get("end_imb"), "%", w, "SARIMAX forecast", "analyse(forecast kW)", "forecast")
        add("highest forecast imbalance", fc.get("max_imb"), "%", w, "SARIMAX forecast", "analyse(forecast kW)", "forecast")
        add("forecast points above the alarm limit", fc.get("n_alarm"), "points", w, "SARIMAX forecast", "count", "forecast")
        add("forecast points in total", fc.get("n_points"), "points", w, "SARIMAX forecast", "count", "forecast")
        add("forecast total power at the end of the window", fc.get("end_total_kw"), "kW", w, "SARIMAX forecast", "sum of the three phase forecasts", "forecast")
    return ev


def missing_information(f):
    out = []
    if not f.get("rating"): out.append("breaker / cable current rating per phase")
    if not f.get("demand"): out.append("sanctioned (contract) demand")
    out += ["transformer rating and loading", "which individual loads are connected to each phase", "maintenance history and protection settings"]
    if not f.get("real_clock"): out.append("real timestamps (the clock in the file is assumed)")
    return out


def python_priority(f):
    """The highest priority the LLM is allowed to give - decided by configured rules, not by the LLM."""
    ld = f.get("load")
    if ld and f.get("mean_imb") is None:
        dem, pk = f.get("demand") or 0, ld.get("peak_kw") or 0
        return "high" if dem and pk > dem else "medium" if dem and pk > 0.85 * dem else "low"
    mi, thr, sev, rating, peak = f.get("mean_imb", 0), f.get("thr", 3), f.get("sev", 10), f.get("rating") or 0, f.get("peak_A_max") or 0
    if rating and peak > rating: return "critical"
    if mi > sev or (rating and peak > 0.85 * rating): return "high"
    if mi > thr: return "medium"
    return "low"


# ------------------------------------------------------------------ prompt
SCHEMA_TEXT = """Return ONLY one JSON object with exactly these keys:
{"what_is_happening": {"summary": str, "evidence_ids": [str]},
 "why": [{"cause": str, "classification": "confirmed|strongly_supported|possible|unknown", "supporting_evidence_ids": [str], "contradicting_evidence_ids": [str], "missing_information": [str], "confidence": "low|medium|high"}],
 "actions": [{"action": str, "reason": str, "priority": "low|medium|high|critical", "evidence_ids": [str], "benefit": str, "drawbacks": [str], "verification": str}],
 "if_no_action": {"consequences": [str], "horizon": str},
 "uncertainties": [str]}"""

SYSTEM = """You are the explanation layer of an energy-management system for a three-phase industrial machine.
STRICT RULES
1. Use ONLY the evidence rows given. Never invent readings, ratings, standards, costs, savings, maintenance history or root causes.
2. NEVER write a measured or calculated number yourself. Refer to it with its placeholder, e.g. "Blue is {E012} above the average". Python fills in the value. Words like 'two' or 'half' for measurements are also not allowed.
3. Every statement about the data must cite evidence ids. If the evidence cannot answer something, write exactly: "Insufficient evidence to determine this."
4. Separate observations from hypotheses. A cause is 'confirmed' only if the evidence directly proves it; correlation is at most 'possible' or 'strongly_supported'.
5. Do not give switching commands, do not say anything is 'safe', and do not claim any equipment is within its rating unless a rating row exists. Recommend inspection and qualified-engineer review for any electrical intervention.
6. Do not exceed the priority ceiling. Forecast rows are predictions, not facts; follow the forecast_reliability field and express uncertainty when it is not RELIABLE.
7. For every action give a benefit, the drawbacks, and how to verify it worked. Explain what happens if nothing is done (keep current faults and forecast risks separate).
8. Answer the three questions: what is happening, why, what to do."""


def build_messages(ev, ctx):
    rows = [{k: e[k] for k in ("id", "parameter", "value", "unit", "window", "kind")} for e in ev]
    user = json.dumps(dict(evidence=rows, context=ctx), ensure_ascii=False) + "\n\n" + SCHEMA_TEXT
    return [{"role": "system", "content": SYSTEM}, {"role": "user", "content": user}]


# ------------------------------------------------------------------ Groq call (timeout, retries, rate limits)
def groq_chat(messages, api_key, model=DEFAULT_MODEL, timeout=45, retries=2, opener=None):
    """POST to Groq's OpenAI-compatible endpoint with JSON mode. Returns (text, None) or (None, error message). `opener` is only for tests."""
    opener = opener or urllib.request.urlopen
    payload = dict(model=model, messages=messages, temperature=0.1, max_tokens=4000, response_format={"type": "json_object"})
    if model.startswith("openai/gpt-oss"): payload["reasoning_effort"] = "low"          # reasoning tokens would otherwise eat the answer budget
    body = json.dumps(payload).encode("utf-8")
    err = "unknown error"
    for attempt in range(retries + 1):
        req = urllib.request.Request(GROQ_URL, data=body, method="POST", headers={"Authorization": f"Bearer {api_key}", "Content-Type": "application/json", "User-Agent": "GridSense/1.0"})
        try:
            with opener(req, timeout=timeout) as r: data = json.loads(r.read().decode("utf-8"))
            return data["choices"][0]["message"]["content"], None
        except urllib.error.HTTPError as e:
            err = f"Groq returned HTTP {e.code}"
            if e.code in (401, 403): return None, err + " - the API key was rejected (check GROQ_API_KEY)."
            if e.code == 404 or e.code == 400: return None, err + f" - check the model name '{model}'."
            if e.code == 429 or e.code >= 500:
                wait = float(e.headers.get("retry-after", 2 ** attempt)) if getattr(e, "headers", None) else 2 ** attempt
                if attempt < retries: time.sleep(min(wait, 20)); continue
                err += " (rate limit or server busy)"
            return None, err
        except (urllib.error.URLError, TimeoutError, OSError) as e:
            err = f"could not reach Groq ({getattr(e, 'reason', e)})"
            if attempt < retries: time.sleep(2 ** attempt); continue
            return None, err
        except (KeyError, IndexError, ValueError) as e: return None, f"unexpected Groq response ({e})"
    return None, err


def parse_json(text):
    t = (text or "").strip(); t = re.sub(r"^```(?:json)?|```$", "", t).strip()
    try: return json.loads(t), None
    except Exception as e:
        m = re.search(r"\{.*\}", t, re.S)
        if m:
            try: return json.loads(m.group(0)), None
            except Exception: pass
        return None, f"the answer was not valid JSON ({e})"


# ------------------------------------------------------------------ independent validation (never done by the LLM)
UNSAFE = re.compile(r"\b(switch|turn|trip)\s+(off|on)\b|\bclose the (breaker|switch)\b|\bopen the (breaker|switch)\b|\bbypass\b|\bdisable\b.*\bprotection\b|\bremove the (fuse|protection)\b|\bshort[- ]?circuit\b.*\bto\b|\benergi[sz]e\b|\bis (completely |perfectly )?safe\b|\bsafe to (increase|add|connect|run)\b", re.I)
RATING_CLAIM = re.compile(r"within (the |its |their )?(rated|rating|limits?|capacity)|\bdoes not exceed\b|\bbelow the (rating|capacity)\b|\bsufficient (capacity|headroom)\b", re.I)
NEEDS_RATING = re.compile(r"\b(rating|rated|transformer|breaker|cable|sanctioned|contract demand)\b", re.I)
FORECASTY = re.compile(r"forecast|predict|expected|will (rise|increase|reach|exceed)", re.I)
NUM = re.compile(r"(?<![A-Za-z_{])(\d+(?:[.,]\d+)?)(?!\s*(?:min|mins|minutes?|h|hrs?|hours?|days?|weeks?|months?|years?)\b)(?![A-Za-z_}])")
STR_FIELDS = {"what_is_happening": ["summary"], "why": ["cause"], "actions": ["action", "reason", "benefit", "verification"]}


def _strings(o):
    if isinstance(o, str): yield o
    elif isinstance(o, dict):
        for v in o.values(): yield from _strings(v)
    elif isinstance(o, list):
        for v in o: yield from _strings(v)


def _bad_numbers(text):
    """Numbers the LLM typed itself (placeholders are removed first). The digits 1-3 are allowed ('phase 1', 'top 2')."""
    return [m for m in NUM.findall(PH.sub("", text)) if m not in {"1", "2", "3"}]


def _is_list(x, t=str): return isinstance(x, list) and all(isinstance(i, t) for i in x)


def check_schema(o):
    """Returns a list of schema problems (empty = OK)."""
    p = []
    if not isinstance(o, dict): return ["the answer is not a JSON object"]
    w = o.get("what_is_happening")
    if not (isinstance(w, dict) and isinstance(w.get("summary"), str) and _is_list(w.get("evidence_ids"))): p.append("what_is_happening is missing or malformed")
    if not isinstance(o.get("why"), list): p.append("why must be a list")
    else:
        for c in o["why"]:
            if not (isinstance(c, dict) and isinstance(c.get("cause"), str) and c.get("classification") in CLASSES and c.get("confidence") in CONFIDENCE and _is_list(c.get("supporting_evidence_ids", []))): p.append("a 'why' item is malformed"); break
    if not isinstance(o.get("actions"), list): p.append("actions must be a list")
    else:
        for a in o["actions"]:
            if not (isinstance(a, dict) and all(isinstance(a.get(k), str) for k in ("action", "reason", "benefit", "verification")) and a.get("priority") in PRIORITIES and _is_list(a.get("evidence_ids")) and _is_list(a.get("drawbacks", []))): p.append("an action is malformed"); break
    n = o.get("if_no_action")
    if not (isinstance(n, dict) and _is_list(n.get("consequences", [])) and isinstance(n.get("horizon", ""), str)): p.append("if_no_action is malformed")
    if not _is_list(o.get("uncertainties", [])): p.append("uncertainties must be a list of text")
    return p


def validate(obj, ev, ceiling, reliability, facts):
    """Check the LLM answer. Returns (clean answer or None, validation statuses, rejected claims)."""
    rejected = []; st = dict(schema=PASSED, grounding=PASSED, numbers=PASSED, priority=PASSED, safety=PASSED, forecast=PASSED, engineering=PASSED)
    problems = check_schema(obj)
    if problems:
        st.update(schema=FAILED, publication=FAILED); return None, st, [dict(where="schema", why="; ".join(problems))]
    ids = {e["id"] for e in ev}; o = copy.deepcopy(obj)
    has_rating, has_demand = bool(facts.get("rating")), bool(facts.get("demand"))

    def good_ids(lst, where):
        keep = [i for i in lst if i in ids]
        if len(keep) != len(lst): st["grounding"] = FAILED; rejected.append(dict(where=where, why=f"cited evidence ids that do not exist: {sorted(set(lst) - ids)}"))
        return keep

    def bad_text(text, where):
        """True when the text must be rejected (stray numbers / unknown placeholders / unsafe wording / unsupported rating claims)."""
        nums = _bad_numbers(text)
        if nums: st["numbers"] = FAILED; rejected.append(dict(where=where, why=f"the LLM typed numbers itself ({', '.join(nums[:4])}); only {{E###}} placeholders are allowed", text=text[:160])); return True
        unknown = [m for m in PH.findall(text) if m not in ids]
        if unknown: st["grounding"] = FAILED; rejected.append(dict(where=where, why=f"placeholder for a missing evidence row: {unknown}", text=text[:160])); return True
        if UNSAFE.search(text): st["safety"] = FAILED; rejected.append(dict(where=where, why="unsafe wording (switching instruction or a claim that something is safe)", text=text[:160])); return True
        if RATING_CLAIM.search(text) and not (has_rating or has_demand): st["engineering"] = FAILED; rejected.append(dict(where=where, why="claims something is within a rating / capacity, but no rating was supplied", text=text[:160])); return True
        return False

    w = o["what_is_happening"]; w["evidence_ids"] = good_ids(w["evidence_ids"], "what_is_happening")
    if not w["evidence_ids"] or bad_text(w["summary"], "what_is_happening"):
        w["summary"] = "Insufficient evidence to determine this (the summary was rejected by the validator - see the audit list)."; w["evidence_ids"] = []
    causes = []
    for i, c in enumerate(o["why"]):
        c["supporting_evidence_ids"] = good_ids(c.get("supporting_evidence_ids", []), f"why[{i}]"); c["contradicting_evidence_ids"] = good_ids(c.get("contradicting_evidence_ids", []) if _is_list(c.get("contradicting_evidence_ids", [])) else [], f"why[{i}]")
        c.setdefault("missing_information", [])
        if bad_text(c["cause"], f"why[{i}]"): continue
        if c["classification"] in ("confirmed", "strongly_supported") and len(c["supporting_evidence_ids"]) < (1 if c["classification"] == "confirmed" else 2):
            c["classification"] = "possible"; rejected.append(dict(where=f"why[{i}]", why="classification lowered to 'possible': not enough supporting evidence rows for that level", text=c["cause"][:160]))
        elif c["classification"] == "confirmed": c["classification"] = "strongly_supported"; rejected.append(dict(where=f"why[{i}]", why="'confirmed' is not allowed from meter data alone - lowered to 'strongly_supported'", text=c["cause"][:160]))
        causes.append(c)
    o["why"] = causes
    acts, order = [], {p: i for i, p in enumerate(PRIORITIES)}
    for i, a in enumerate(o["actions"]):
        a["evidence_ids"] = good_ids(a["evidence_ids"], f"actions[{i}]")
        blob = " ".join([a["action"], a["reason"], a["benefit"], a["verification"]] + a.get("drawbacks", []))
        if not a["evidence_ids"]: rejected.append(dict(where=f"actions[{i}]", why="the action cites no valid evidence", text=a["action"][:160])); st["grounding"] = FAILED; continue
        if any(bad_text(t, f"actions[{i}]") for t in [a["action"], a["reason"], a["benefit"], a["verification"]] + a.get("drawbacks", [])): continue
        top = ceiling
        if FORECASTY.search(blob) and reliability != "RELIABLE": top = PRIORITIES[min(order[ceiling], order["medium"])]
        if order[a["priority"]] > order[top]:
            rejected.append(dict(where=f"actions[{i}]", why=f"priority '{a['priority']}' lowered to '{top}' (Python rule ceiling{' and forecast not reliable' if top != ceiling else ''})", text=a["action"][:160])); a["priority"] = top
        if NEEDS_RATING.search(blob) and not (has_rating or has_demand): st["engineering"] = INSUFFICIENT
        a["human_approval_required"] = True; acts.append(a)
    o["actions"] = acts
    for sec in (o["if_no_action"]["consequences"], o["uncertainties"]):
        for j in range(len(sec) - 1, -1, -1):
            if bad_text(sec[j], "text"): sec.pop(j)
    if reliability == "RELIABLE": st["forecast"] = PASSED
    elif reliability in ("DEGRADED", "UNRELIABLE"): st["forecast"] = REVIEW
    else: st["forecast"] = INSUFFICIENT
    if not (has_rating or has_demand) and st["engineering"] == PASSED: st["engineering"] = INSUFFICIENT
    st["priority"] = PASSED
    bad = [k for k, v in st.items() if v == FAILED]
    st["publication"] = FAILED if bad else INSUFFICIENT if INSUFFICIENT in st.values() else REVIEW        # never PASSED: electrical actions always need a qualified human
    return o, st, rejected


def fill(o, ev):
    """Replace {E###} placeholders with the real value and unit from the evidence rows."""
    by = {e["id"]: e for e in ev}

    def sub(m):
        e = by[m.group(1)]; v = f"{e['value']:g}"; return f"{v} {e['unit']}".strip() if e["unit"] not in ("%",) else f"{v} %"
    def walk(x):
        if isinstance(x, str): return PH.sub(sub, x)
        if isinstance(x, list): return [walk(i) for i in x]
        if isinstance(x, dict): return {k: walk(v) for k, v in x.items()}
        return x
    return walk(o)


def advise(facts, reliability, api_key, model=DEFAULT_MODEL, caller=groq_chat, topic="phase imbalance"):
    """Full pipeline. Returns dict(source='llm'|'fallback', error, answer, validation, rejected, evidence, ceiling).
    source='fallback' means the app should show the rule-based advice instead (no key, API failure, invalid JSON or schema failure)."""
    ev = build_evidence(facts); ceiling = python_priority(facts)
    ctx = dict(topic=topic, forecast_reliability=reliability, priority_ceiling=ceiling, missing_information=missing_information(facts), data_clock="real" if facts.get("real_clock") else "assumed (no timestamps in the file)")
    out = dict(source="fallback", error=None, answer=None, validation=None, rejected=[], evidence=ev, ceiling=ceiling, ctx=ctx)
    if not api_key: out["error"] = "no Groq API key found (paste it into GROQ_API_KEY_EMBEDDED at the top of app.py)"; return out
    msgs = build_messages(ev, ctx); obj = None
    models = [model] + [m for m in FALLBACK_MODELS if m != model]

    def call(m):
        """Ask the models in order; move to the next one only when Groq does not know the current one (HTTP 404 / 400)."""
        while True:
            text, err = caller(m, api_key, models[0])
            if err and ("HTTP 404" in err or "HTTP 400" in err) and len(models) > 1: models.pop(0); continue
            return text, err

    for attempt in range(2):                                   # one repair attempt when the JSON is invalid
        text, err = call(msgs)
        if err: out["error"] = err; return out
        obj, perr = parse_json(text)
        if obj is not None: break
        msgs = msgs + [{"role": "assistant", "content": text or ""}, {"role": "user", "content": f"That was rejected: {perr}. Return ONLY the JSON object."}]
    if obj is None: out["error"] = perr; return out
    clean, st, rej = validate(obj, ev, ceiling, reliability, facts)
    out.update(validation=st, rejected=rej)
    if clean is None: out["error"] = "the answer did not match the required structure: " + rej[0]["why"]; return out
    out.update(source="llm", answer=fill(clean, ev)); return out


def evidence_hash(facts, reliability):
    return hashlib.md5(json.dumps([facts, reliability], sort_keys=True, default=str).encode()).hexdigest()
