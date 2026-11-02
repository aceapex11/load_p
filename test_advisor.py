import io, json, os, sys, urllib.error
import numpy as np, pandas as pd, pytest
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
import gridsense_advisor as ga

PER = {"Red": dict(avg_A=234.5, peak_A=300, avg_kW=56.3), "Yellow": dict(avg_A=229.9, peak_A=290, avg_kW=55.6), "Blue": dict(avg_A=260.8, peak_A=330, avg_kW=64.1)}
def facts(**k):
    f = dict(mean_imb=10.6, p95_imb=17.6, pct_above=96, mean_I=9.5, mean_V=1.4, per=PER, main_name="Blue", extra_A=28.6, neutral_mean=40, loss=1.2, pf_meter=0.91,
             plan=[dict(From="Blue", To="Yellow", **{"Move A": 11.8}), dict(From="Blue", To="Red", **{"Move A": 7.3})], thr=3, sev=10, min_kw=5, rating=0, demand=0, real_clock=True, peak_A_max=330, window="all loaded minutes")
    f.update(k); return f
def good_answer(**over):
    a = {"what_is_happening": {"summary": "Blue carries {E012} more current than the other phases; imbalance averages {E001}.", "evidence_ids": ["E001", "E012"]},
         "why": [{"cause": "Single-phase loads are not shared equally", "classification": "possible", "supporting_evidence_ids": ["E004"], "contradicting_evidence_ids": [], "missing_information": ["list of loads per phase"], "confidence": "low"}],
         "actions": [{"action": "Inspect which single-phase loads are on Blue", "reason": "Blue is {E012} above the others", "priority": "medium", "evidence_ids": ["E012"], "benefit": "lower imbalance", "drawbacks": ["needs an engineer's time"], "verification": "re-measure the imbalance after the change"}],
         "if_no_action": {"consequences": ["Blue keeps running hotter"], "horizon": "ongoing"}, "uncertainties": ["cause not confirmed"]}
    a.update(over); return a
def ev_for(f=None): return ga.build_evidence(f or facts())
def run(ans, f=None, rel="RELIABLE", ceil=None):
    f = f or facts(); ev = ga.build_evidence(f); return ga.validate(ans, ev, ceil or ga.python_priority(f), rel, f)

# ---- priorities decided by Python
def test_priority_rules():
    assert ga.python_priority(facts(mean_imb=1.0)) == "low"
    assert ga.python_priority(facts(mean_imb=5.0)) == "medium"
    assert ga.python_priority(facts(mean_imb=12.0)) == "high"
    assert ga.python_priority(facts(mean_imb=1.0, rating=300, peak_A_max=330)) == "critical"
def test_balanced_load_gives_low_ceiling_and_clamps_llm():
    f = facts(mean_imb=1.0); ans = good_answer(); ans["actions"][0]["priority"] = "critical"
    clean, st, rej = run(ans, f)
    assert clean["actions"][0]["priority"] == "low" and any("lowered" in r["why"] for r in rej)

# ---- evidence + grounding
def test_evidence_ids_unique_and_have_units():
    ev = ev_for(); assert len({e["id"] for e in ev}) == len(ev); assert all(e["unit"] is not None and e["source"] for e in ev)
def test_missing_evidence_id_is_rejected():
    ans = good_answer(); ans["actions"][0]["evidence_ids"] = ["E999"]
    clean, st, rej = run(ans); assert clean["actions"] == [] and st["grounding"] == ga.FAILED and st["publication"] == ga.FAILED
def test_unknown_placeholder_rejected():
    ans = good_answer(); ans["what_is_happening"]["summary"] = "Blue is {E777} high"
    clean, st, rej = run(ans); assert "Insufficient evidence" in clean["what_is_happening"]["summary"] and st["grounding"] == ga.FAILED
def test_llm_typed_numbers_rejected_but_placeholders_ok():
    ans = good_answer(); ans["actions"][0]["reason"] = "Blue is 42 A above the others"
    clean, st, rej = run(ans); assert clean["actions"] == [] and st["numbers"] == ga.FAILED
    clean, st, rej = run(good_answer()); assert st["numbers"] == ga.PASSED and len(clean["actions"]) == 1
def test_time_words_allowed():
    ans = good_answer(); ans["if_no_action"]["horizon"] = "over the next 30 days"; ans["uncertainties"] = ["re-check within 2 hours"]
    clean, st, rej = run(ans); assert st["numbers"] == ga.PASSED
def test_placeholders_filled_with_python_values():
    f = facts(); ev = ga.build_evidence(f); clean, st, rej = ga.validate(good_answer(), ev, "medium", "RELIABLE", f); filled = ga.fill(clean, ev)
    by = {e["id"]: e for e in ev}; assert f"{by['E012']['value']:g}" in filled["what_is_happening"]["summary"] and "{E" not in json.dumps(filled)

# ---- safety / engineering
@pytest.mark.parametrize("text", ["Switch off the Blue breaker now", "Bypass the protection relay", "It is safe to increase the load", "Energize the spare feeder"])
def test_unsafe_wording_blocked(text):
    ans = good_answer(); ans["actions"][0]["action"] = text
    clean, st, rej = run(ans); assert clean["actions"] == [] and st["safety"] == ga.FAILED
def test_rating_claim_without_rating_rejected():
    ans = good_answer(); ans["actions"][0]["benefit"] = "The load stays within the rated capacity"
    clean, st, rej = run(ans); assert clean["actions"] == [] and st["engineering"] == ga.FAILED
def test_no_rating_means_insufficient_engineering_and_never_passed_publication():
    clean, st, rej = run(good_answer()); assert st["engineering"] == ga.INSUFFICIENT and st["publication"] != ga.PASSED
def test_with_rating_publication_needs_human_review():
    clean, st, rej = run(good_answer(), facts(rating=400)); assert st["engineering"] == ga.PASSED and st["publication"] == ga.REVIEW
    assert all(a["human_approval_required"] for a in clean["actions"])
def test_confirmed_cause_is_lowered():
    ans = good_answer(); ans["why"][0]["classification"] = "confirmed"
    clean, st, rej = run(ans); assert clean["why"][0]["classification"] != "confirmed"

# ---- forecast gate
def test_unreliable_forecast_caps_forecast_actions():
    ans = good_answer(); ans["actions"][0].update(action="Plan a rebalance because the forecast shows a rising imbalance", priority="high")
    clean, st, rej = run(ans, facts(mean_imb=12.0), rel="UNRELIABLE"); assert clean["actions"][0]["priority"] == "medium" and st["forecast"] == ga.REVIEW
def test_untested_forecast_is_insufficient():
    clean, st, rej = run(good_answer(), rel="INSUFFICIENT DATA"); assert st["forecast"] == ga.INSUFFICIENT

# ---- schema
@pytest.mark.parametrize("mut", [lambda a: a.pop("actions"), lambda a: a["actions"][0].update(priority="urgent!"), lambda a: a["why"][0].update(classification="certain"), lambda a: a.update(what_is_happening="text")])
def test_schema_failures_fail_closed(mut):
    ans = good_answer(); mut(ans); clean, st, rej = run(ans); assert clean is None and st["schema"] == ga.FAILED

# ---- orchestration: fallbacks
def test_no_key_falls_back():
    r = ga.advise(facts(), "RELIABLE", None); assert r["source"] == "fallback" and "key" in r["error"]
def test_api_failure_falls_back():
    r = ga.advise(facts(), "RELIABLE", "k", caller=lambda *a, **k: (None, "Groq returned HTTP 500")); assert r["source"] == "fallback" and "500" in r["error"]
def test_invalid_json_falls_back_after_one_repair():
    calls = []; r = ga.advise(facts(), "RELIABLE", "k", caller=lambda m, k, md: (calls.append(1) or "not json at all", None)); assert r["source"] == "fallback" and len(calls) == 2
def test_valid_answer_goes_through():
    r = ga.advise(facts(rating=400), "RELIABLE", "k", caller=lambda m, k, md: (json.dumps(good_answer()), None))
    assert r["source"] == "llm" and r["validation"]["publication"] == ga.REVIEW and "{E" not in json.dumps(r["answer"])
def test_json_in_code_fence_is_parsed():
    obj, err = ga.parse_json("```json\n" + json.dumps(good_answer()) + "\n```"); assert err is None and "actions" in obj

# ---- Groq HTTP handling (fake opener, no network)
class Resp(io.BytesIO):
    def __enter__(self): return self
    def __exit__(self, *a): return False
def http_error(code): return urllib.error.HTTPError("u", code, "x", {"retry-after": "0"}, io.BytesIO(b"{}"))
def test_groq_401_not_retried():
    n = []
    def op(req, timeout): n.append(1); raise http_error(401)
    text, err = ga.groq_chat([{"role": "user", "content": "x"}], "k", opener=op); assert text is None and "rejected" in err and len(n) == 1
def test_groq_429_retried_then_ok(monkeypatch):
    monkeypatch.setattr(ga.time, "sleep", lambda s: None); n = []
    def op(req, timeout):
        n.append(1)
        if len(n) < 3: raise http_error(429)
        return Resp(json.dumps({"choices": [{"message": {"content": "{}"}}]}).encode())
    text, err = ga.groq_chat([{"role": "user", "content": "x"}], "k", opener=op); assert text == "{}" and err is None and len(n) == 3
def test_groq_network_error(monkeypatch):
    monkeypatch.setattr(ga.time, "sleep", lambda s: None)
    def op(req, timeout): raise urllib.error.URLError("dns")
    text, err = ga.groq_chat([{"role": "user", "content": "x"}], "k", opener=op); assert text is None and "could not reach" in err

# ---- key handling
def test_key_order(tmp_path):
    envf = tmp_path / ".env"; envf.write_text("# c\nGROQ_API_KEY='from_file'\n")
    assert ga.load_key({}, envf) == "from_file"
    assert ga.load_key({"GROQ_API_KEY": "from_env"}, envf) == "from_env"
    assert ga.load_key({"GROQ_API_KEY": "from_env"}, envf, typed="typed") == "typed"
    assert ga.load_key({}, tmp_path / "missing.env") is None

# ---- forecast metrics and reliability
def test_smape_zero_denominator_and_value():
    assert ga.smape([0, 0], [0, 0]) == 0.0
    assert ga.smape([100], [110]) == pytest.approx(200 * 10 / 210)
def test_mase():
    assert ga.mase([10, 12], [11, 11], 2.0) == pytest.approx(0.5); assert np.isnan(ga.mase([1], [1], 0))
def table(sx, base, moments=12):
    rows = []
    for h in (15, 30, 60):
        rows.append({"Minutes ahead": h, "Method": "SARIMAX", "Total kW error": sx, "Imbalance error (pts)": sx, "Moments": moments})
        rows.append({"Minutes ahead": h, "Method": "Same as now", "Total kW error": base, "Imbalance error (pts)": base, "Moments": moments})
        rows.append({"Minutes ahead": h, "Method": "Seasonal naive (24 h)", "Total kW error": base * 2, "Imbalance error (pts)": base * 2, "Moments": moments})
    return pd.DataFrame(rows)
def test_reliability_labels():
    assert ga.forecast_reliability(table(0.5, 1.0))[0] == "RELIABLE"
    assert ga.forecast_reliability(table(1.02, 1.0))[0] == "DEGRADED"
    assert ga.forecast_reliability(table(1.5, 1.0))[0] == "UNRELIABLE"          # SARIMAX worse than baseline
    assert ga.forecast_reliability(table(0.5, 1.0, moments=3))[0] == "INSUFFICIENT DATA"
    assert ga.forecast_reliability(None)[0] == "INSUFFICIENT DATA"
