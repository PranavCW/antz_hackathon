"""§G safeguards that don't need the LLM: verbatim check, lexicon + negation, cache key."""

from src import notes_llm as nl


def test_verbatim_quote_must_occur_in_note():
    text = "Tiger ate only half of the meat today, seemed dull."
    assert nl.verbatim_ok("ate only half of the meat", text, 90)
    assert nl.verbatim_ok("Ate only half of the meat", text, 90)            # case-insensitive
    assert not nl.verbatim_ok("refused all food", text, 90)                 # invented quote
    assert not nl.verbatim_ok("", text, 90)


def test_lexicon_hits_and_negation():
    sig = {(s["category"], s["direction"]) for s in nl.lexicon_signals("Limping badly, no vomiting, coughing.")}
    assert ("mobility", "worse") in sig and ("respiratory", "worse") in sig
    assert ("gi", "normal") in sig and ("gi", "worse") not in sig


def test_lexicon_hinglish():
    sig = {s["category"] for s in nl.lexicon_signals("Aaj khana nahi khaya, sust hai") if s["direction"] == "worse"}
    assert sig == {"appetite", "behaviour"}


def test_check_signals_flags_rejected_quotes():
    res = {"signals": [{"category": "gi", "direction": "worse", "severity": 2, "evidence_quote": "loose stool"},
                       {"category": "mobility", "direction": "worse", "severity": 2, "evidence_quote": "limping"}]}
    out = nl.check_signals(res, "Loose stool since morning.", 90)
    assert [s["verbatim_ok"] for s in out] == [True, False]


def test_cache_key_changes_with_prompt_and_model():
    m1 = nl.build_messages("Not eating", "tiger", "", "2026-01-01")
    m2 = nl.build_messages("Not eating", "lion", "", "2026-01-01")
    assert nl.cache_key("a", m1) == nl.cache_key("a", m1)
    assert nl.cache_key("a", m1) != nl.cache_key("a", m2) != nl.cache_key("b", m1)
