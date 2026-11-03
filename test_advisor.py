# Tests for gridsense_advisor (the AI-advice validator). Helper names below were made more descriptive; the gridsense_advisor API (ga.*) is unchanged.
import io, json, os, sys, urllib.error
import numpy as np, pandas as pd, pytest
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
import gridsense_advisor as ga

PHASE_STATS = {"Red": dict(avg_A=234.5, peak_A=300, avg_kW=56.3), "Yellow": dict(avg_A=229.9, peak_A=290, avg_kW=55.6), "Blue": dict(avg_A=260.8, peak_A=330, avg_kW=64.1)}
# measured facts of a typical unbalanced machine; override any value per test
def make_facts(**k):
    f = dict(mean_imb=10.6, p95_imb=17.6, pct_above=96, mean_I=9.5, mean_V=1.4, per=PHASE_STATS, main_name="Blue", extra_A=28.6, neutral_mean=40, loss=1.2, pf_meter=0.91,
             plan=[dict(From="Blue", To="Yellow", **{"Move A": 11.8}), dict(From="Blue", To="Red", **{"Move A": 7.3})], thr=3, sev=10, min_kw=5, rating=0, demand=0, real_clock=True, peak_A_max=330, window="all loaded minutes")
    f.update(k); return f
# a model answer that passes every check; tests break one part of it at a time
def make_valid_answer(**over):
    a = {"what_is_happening": {"summary": "Blue carries {E012} more current than the other phases; imbalance averages {E001}.", "evidence_ids": ["E001", "E012"]},
         "why": [{"cause": "Single-phase loads are not shared equally", "classification": "possible", "supporting_evidence_ids": ["E004"], "contradicting_evidence_ids": [], "missing_information": ["list of loads per phase"], "confidence": "low"}],
         "actions": [{"action": "Inspect which single-phase loads are on Blue", "reason": "Blue is {E012} above the others", "priority": "medium", "evidence_ids": ["E012"], "benefit": "lower imbalance", "drawbacks": ["needs an engineer's time"], "verification": "re-measure the imbalance after the change"}],
         "if_no_action": {"consequences": ["Blue keeps running hotter"], "horizon": "ongoing"}, "uncertainties": ["cause not confirmed"]}
    a.update(over); return a
# the numbered evidence list built from the facts
def evidence_for(f=None): return ga.build_evidence(f or make_facts())
# run the full validation of one answer and return (cleaned answer, status, rejected items)
def validate_answer(answer, f=None, rel="RELIABLE", ceil=None):
    f = f or make_facts(); evidence = ga.build_evidence(f); return ga.validate(answer, evidence, ceil or ga.python_priority(f), rel, f)

# ---- priorities decided by Python
# priority comes from the numbers, not from the model
def test_priority_rules():
    assert ga.python_priority(make_facts(mean_imb=1.0)) == "low"
    assert ga.python_priority(make_facts(mean_imb=5.0)) == "medium"
    assert ga.python_priority(make_facts(mean_imb=12.0)) == "high"
    assert ga.python_priority(make_facts(mean_imb=1.0, rating=300, peak_A_max=330)) == "critical"
# a balanced load caps the priority at low, even if the model says critical
def test_balanced_load_gives_low_ceiling_and_clamps_llm():
    f = make_facts(mean_imb=1.0); answer = make_valid_answer(); answer["actions"][0]["priority"] = "critical"
    cleaned, status, rejected = validate_answer(answer, f)
    assert cleaned["actions"][0]["priority"] == "low" and any("lowered" in r["why"] for r in rejected)

# ---- evidence + grounding
# every evidence item has a unique id, a unit and a source
def test_evidence_ids_unique_and_have_units():
    evidence = evidence_for(); assert len({e["id"] for e in evidence}) == len(evidence); assert all(e["unit"] is not None and e["source"] for e in evidence)
# an action that cites an unknown evidence id is dropped
def test_missing_evidence_id_is_rejected():
    answer = make_valid_answer(); answer["actions"][0]["evidence_ids"] = ["E999"]
    cleaned, status, rejected = validate_answer(answer); assert cleaned["actions"] == [] and status["grounding"] == ga.FAILED and status["publication"] == ga.FAILED
# a {E...} placeholder with no matching evidence fails grounding
def test_unknown_placeholder_rejected():
    answer = make_valid_answer(); answer["what_is_happening"]["summary"] = "Blue is {E777} high"
    cleaned, status, rejected = validate_answer(answer); assert "Insufficient evidence" in cleaned["what_is_happening"]["summary"] and status["grounding"] == ga.FAILED
# numbers typed by the model are rejected, {E...} placeholders are fine
def test_llm_typed_numbers_rejected_but_placeholders_ok():
    answer = make_valid_answer(); answer["actions"][0]["reason"] = "Blue is 42 A above the others"
    cleaned, status, rejected = validate_answer(answer); assert cleaned["actions"] == [] and status["numbers"] == ga.FAILED
    cleaned, status, rejected = validate_answer(make_valid_answer()); assert status["numbers"] == ga.PASSED and len(cleaned["actions"]) == 1
# plain time words such as '30 days' or '2 hours' are not counted as invented numbers
def test_time_words_allowed():
    answer = make_valid_answer(); answer["if_no_action"]["horizon"] = "over the next 30 days"; answer["uncertainties"] = ["re-check within 2 hours"]
    cleaned, status, rejected = validate_answer(answer); assert status["numbers"] == ga.PASSED
# placeholders are replaced by the real measured values
def test_placeholders_filled_with_python_values():
    f = make_facts(); evidence = ga.build_evidence(f); cleaned, status, rejected = ga.validate(make_valid_answer(), evidence, "medium", "RELIABLE", f); filled = ga.fill(cleaned, evidence)
    by = {e["id"]: e for e in evidence}; assert f"{by['E012']['value']:g}" in filled["what_is_happening"]["summary"] and "{E" not in json.dumps(filled)

# ---- safety / engineering
# switching off, bypassing or energising wording is blocked
@pytest.mark.parametrize("text", ["Switch off the Blue breaker now", "Bypass the protection relay", "It is safe to increase the load", "Energize the spare feeder"])
def test_unsafe_wording_blocked(text):
    answer = make_valid_answer(); answer["actions"][0]["action"] = text
    cleaned, status, rejected = validate_answer(answer); assert cleaned["actions"] == [] and status["safety"] == ga.FAILED
# no 'within rated capacity' claim when no rating was given
def test_rating_claim_without_rating_rejected():
    answer = make_valid_answer(); answer["actions"][0]["benefit"] = "The load stays within the rated capacity"
    cleaned, status, rejected = validate_answer(answer); assert cleaned["actions"] == [] and status["engineering"] == ga.FAILED
# without a rating the engineering check stays insufficient
def test_no_rating_means_insufficient_engineering_and_never_passed_publication():
    cleaned, status, rejected = validate_answer(make_valid_answer()); assert status["engineering"] == ga.INSUFFICIENT and status["publication"] != ga.PASSED
# with a rating the result still needs a human to approve it
def test_with_rating_publication_needs_human_review():
    cleaned, status, rejected = validate_answer(make_valid_answer(), make_facts(rating=400)); assert status["engineering"] == ga.PASSED and status["publication"] == ga.REVIEW
    assert all(a["human_approval_required"] for a in cleaned["actions"])
# a cause can never be 'confirmed' from this data alone
def test_confirmed_cause_is_lowered():
    answer = make_valid_answer(); answer["why"][0]["classification"] = "confirmed"
    cleaned, status, rejected = validate_answer(answer); assert cleaned["why"][0]["classification"] != "confirmed"

# ---- forecast gate
# an unreliable forecast limits forecast-based actions to medium priority
def test_unreliable_forecast_caps_forecast_actions():
    answer = make_valid_answer(); answer["actions"][0].update(action="Plan a rebalance because the forecast shows a rising imbalance", priority="high")
    cleaned, status, rejected = validate_answer(answer, make_facts(mean_imb=12.0), rel="UNRELIABLE"); assert cleaned["actions"][0]["priority"] == "medium" and status["forecast"] == ga.REVIEW
# a forecast that was never tested counts as insufficient
def test_untested_forecast_is_insufficient():
    cleaned, status, rejected = validate_answer(make_valid_answer(), rel="INSUFFICIENT DATA"); assert status["forecast"] == ga.INSUFFICIENT

# ---- schema
# a broken answer structure is rejected completely
@pytest.mark.parametrize("break_answer", [lambda a: a.pop("actions"), lambda a: a["actions"][0].update(priority="urgent!"), lambda a: a["why"][0].update(classification="certain"), lambda a: a.update(what_is_happening="text")])
def test_schema_failures_fail_closed(break_answer):
    answer = make_valid_answer(); break_answer(answer); cleaned, status, rejected = validate_answer(answer); assert cleaned is None and status["schema"] == ga.FAILED

# ---- orchestration: fallbacks
# without an API key the rule-based fallback is used
def test_no_key_falls_back():
    r = ga.advise(make_facts(), "RELIABLE", None); assert r["source"] == "fallback" and "key" in r["error"]
# an API error falls back and keeps the error text
def test_api_failure_falls_back():
    r = ga.advise(make_facts(), "RELIABLE", "k", caller=lambda *a, **k: (None, "Groq returned HTTP 500")); assert r["source"] == "fallback" and "500" in r["error"]
# invalid JSON gets one repair attempt, then falls back
def test_invalid_json_falls_back_after_one_repair():
    call_log = []; r = ga.advise(make_facts(), "RELIABLE", "k", caller=lambda m, k, md: (call_log.append(1) or "not json at all", None)); assert r["source"] == "fallback" and len(call_log) == 2
# a valid answer is published as 'llm' and still needs review
def test_valid_answer_goes_through():
    r = ga.advise(make_facts(rating=400), "RELIABLE", "k", caller=lambda m, k, md: (json.dumps(make_valid_answer()), None))
    assert r["source"] == "llm" and r["validation"]["publication"] == ga.REVIEW and "{E" not in json.dumps(r["answer"])
# JSON wrapped in a ```json fence is still read
def test_json_in_code_fence_is_parsed():
    parsed, error = ga.parse_json("```json\n" + json.dumps(make_valid_answer()) + "\n```"); assert error is None and "actions" in parsed

# ---- Groq HTTP handling (fake opener, no network)
# tiny stand-in for the object returned by urlopen
class FakeResponse(io.BytesIO):
    def __enter__(self): return self
    def __exit__(self, *a): return False
# build an HTTPError with the given status code
def make_http_error(code): return urllib.error.HTTPError("u", code, "x", {"retry-after": "0"}, io.BytesIO(b"{}"))
# a rejected key (401) is not retried
def test_groq_401_not_retried():
    n = []
    def fake_opener(req, timeout): n.append(1); raise make_http_error(401)
    text, error = ga.groq_chat([{"role": "user", "content": "x"}], "k", opener=fake_opener); assert text is None and "rejected" in error and len(n) == 1
# rate limiting (429) is retried until it works
def test_groq_429_retried_then_ok(monkeypatch):
    monkeypatch.setattr(ga.time, "sleep", lambda s: None); n = []
    def fake_opener(req, timeout):
        n.append(1)
        if len(n) < 3: raise make_http_error(429)
        return FakeResponse(json.dumps({"choices": [{"message": {"content": "{}"}}]}).encode())
    text, error = ga.groq_chat([{"role": "user", "content": "x"}], "k", opener=fake_opener); assert text == "{}" and error is None and len(n) == 3
# a network failure gives a clear 'could not reach' message
def test_groq_network_error(monkeypatch):
    monkeypatch.setattr(ga.time, "sleep", lambda s: None)
    def fake_opener(req, timeout): raise urllib.error.URLError("dns")
    text, error = ga.groq_chat([{"role": "user", "content": "x"}], "k", opener=fake_opener); assert text is None and "could not reach" in error

# ---- key handling
# key order: typed key > environment > .env file
def test_key_order(tmp_path):
    envf = tmp_path / ".env"; envf.write_text("# c\nGROQ_API_KEY='from_file'\n")
    assert ga.load_key({}, envf) == "from_file"
    assert ga.load_key({"GROQ_API_KEY": "from_env"}, envf) == "from_env"
    assert ga.load_key({"GROQ_API_KEY": "from_env"}, envf, typed="typed") == "typed"
    assert ga.load_key({}, tmp_path / "missing.env") is None

# ---- forecast metrics and reliability
# SMAPE handles zero values and gives the expected percentage
def test_smape_zero_denominator_and_value():
    assert ga.smape([0, 0], [0, 0]) == 0.0
    assert ga.smape([100], [110]) == pytest.approx(200 * 10 / 210)
# MASE against a naive error scale
def test_mase():
    assert ga.mase([10, 12], [11, 11], 2.0) == pytest.approx(0.5); assert np.isnan(ga.mase([1], [1], 0))
# forecast error table: SARIMAX vs the two baselines at 15 / 30 / 60 minutes
def make_forecast_table(sarimax_error, baseline_error, moments=12):
    rows = []
    for h in (15, 30, 60):
        rows.append({"Minutes ahead": h, "Method": "SARIMAX", "Total kW error": sarimax_error, "Imbalance error (pts)": sarimax_error, "Moments": moments})
        rows.append({"Minutes ahead": h, "Method": "Same as now", "Total kW error": baseline_error, "Imbalance error (pts)": baseline_error, "Moments": moments})
        rows.append({"Minutes ahead": h, "Method": "Seasonal naive (24 h)", "Total kW error": baseline_error * 2, "Imbalance error (pts)": baseline_error * 2, "Moments": moments})
    return pd.DataFrame(rows)
# RELIABLE / DEGRADED / UNRELIABLE / INSUFFICIENT DATA labels
def test_reliability_labels():
    assert ga.forecast_reliability(make_forecast_table(0.5, 1.0))[0] == "RELIABLE"
    assert ga.forecast_reliability(make_forecast_table(1.02, 1.0))[0] == "DEGRADED"
    assert ga.forecast_reliability(make_forecast_table(1.5, 1.0))[0] == "UNRELIABLE"          # SARIMAX worse than baseline
    assert ga.forecast_reliability(make_forecast_table(0.5, 1.0, moments=3))[0] == "INSUFFICIENT DATA"
    assert ga.forecast_reliability(None)[0] == "INSUFFICIENT DATA"
