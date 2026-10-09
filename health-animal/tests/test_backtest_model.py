"""Alerting/backtest maths (§I), model helpers (§H) and explanations."""

import numpy as np
import pandas as pd
import pytest

from src import backtest as B
from src import config, explain
from src import model as M

D0 = pd.Timestamp("2026-01-01")


def rows(animal_id, risks, excl=None, start=D0):
    n = len(risks)
    return pd.DataFrame({"animal_id": animal_id, "date": pd.date_range(start, periods=n), "risk": risks,
                         "excl_reason": excl or [""] * n})


def day(n):
    return int((D0 + pd.Timedelta(days=n)).to_datetime64().astype("datetime64[D]").astype("int64"))


# ── Alerting ─────────────────────────────────────────────────────────────────────────────────────────────

def test_cooldown_limits_repeat_alerts():
    a = B.generate_alerts(rows(1, [0.9] * 20), "risk", 0.5, cooldown=7, rise=0.5)
    assert a["day"].tolist() == [day(0), day(7), day(14)]


def test_risk_rise_overrides_cooldown():
    a = B.generate_alerts(rows(1, [0.5, 0.6, 0.8, 0.8]), "risk", 0.5, cooldown=7, rise=0.5)
    assert a["day"].tolist() == [day(0), day(2)]          # 0.6 < 0.75 suppressed; 0.8 ≥ 0.75 re-alerts


def test_no_alerts_under_treatment_and_cooldown_is_per_animal():
    r = pd.concat([rows(1, [0.9, 0.9], excl=["treatment", ""]), rows(2, [0.9, 0.9])]).reset_index(drop=True)
    a = B.generate_alerts(r, "risk", 0.5, cooldown=7, rise=0.5)
    assert sorted(zip(a["animal_id"], a["day"])) == [(1, day(1)), (2, day(0))]


# ── Evaluation ───────────────────────────────────────────────────────────────────────────────────────────

def test_caught_lead_time_and_precision():
    alerts = pd.DataFrame({"animal_id": [1, 1, 2], "day": [day(100), day(130), day(10)], "risk": 0.9})
    cases = pd.DataFrame({"animal_id": [1], "t0_day": [day(150)]})
    m = B.evaluate(alerts, cases, cases, n_weeks=10, early=60, clear=60)
    assert m["caught"] == 1 and m["mean_lead"] == 50                     # first alert in window = day 100
    assert m["precision"] == pytest.approx(2 / 3) and m["alerts_per_week"] == 0.3


def test_window_boundaries_alert_on_t0_day_or_too_early_does_not_count():
    cases = pd.DataFrame({"animal_id": [1], "t0_day": [day(100)]})
    on_t0 = pd.DataFrame({"animal_id": [1], "day": [day(100)], "risk": 1.0})
    too_early = pd.DataFrame({"animal_id": [1], "day": [day(39)], "risk": 1.0})
    just_in = pd.DataFrame({"animal_id": [1], "day": [day(40)], "risk": 1.0})
    assert B.evaluate(on_t0, cases, cases, 1, 60, 60)["caught"] == 0       # vet is already aware
    assert B.evaluate(too_early, cases, cases, 1, 60, 60)["caught"] == 0   # > 60 d ahead
    assert B.evaluate(just_in, cases, cases, 1, 60, 60)["caught"] == 1


def test_random_baseline_and_empty_alerts():
    assert B.random_baseline_caught_pct(10, 1000, 60) == pytest.approx(100 * (1 - 0.99 ** 60))
    m = B.evaluate(pd.DataFrame(columns=["animal_id", "day", "risk"]), pd.DataFrame({"animal_id": [1], "t0_day": [5]}),
                   pd.DataFrame({"animal_id": [1], "t0_day": [5]}), 4, 60, 60)
    assert m["caught"] == 0 and m["alerts_per_week"] == 0 and np.isnan(m["precision"])


def test_choose_threshold_respects_budget_and_falls_back():
    sw = pd.DataFrame({"threshold": [0.2, 0.5, 0.8], "alerts_per_week": [9.0, 2.5, 0.5], "caught": [5, 3, 1],
                       "mean_lead": [20.0, 30.0, 40.0]})
    best, ok = B.choose_threshold(sw, budget=3)
    assert ok and best["threshold"] == 0.5
    best, ok = B.choose_threshold(sw, budget=0.1)
    assert not ok and best["threshold"] == 0.8                            # strictest available, flagged


def test_miss_reasons():
    cols = {"risk": [0.1], "w_fired": [0.0], "n_neg_cats_7d": [0.0], "i_cusum_max_7": [0.0], "w_pct_change": [0.0]}
    assert B.miss_reason(pd.DataFrame(columns=list(cols)), 0.5, "risk") == "no_data"
    assert B.miss_reason(pd.DataFrame(cols), 0.5, "risk") == "no_signal"
    assert B.miss_reason(pd.DataFrame({**cols, "n_neg_cats_7d": [1.0]}), 0.5, "risk") == "below_threshold"


# ── Model helpers ────────────────────────────────────────────────────────────────────────────────────────

@pytest.fixture(scope="module")
def cfg():
    return config.load()


def feature_row(**kw):
    base = {"w_fired": 0, "w_confirmed": 0, "i_cusum_max_7": 0, "n_neg_cats_7d": 0, "n_max_priority_7d": 0}
    return pd.DataFrame([{**base, **kw}])


def test_rule_score_monotone_and_capped(cfg):
    none = M.rule_score(feature_row(), cfg).iloc[0]
    one = M.rule_score(feature_row(w_fired=1), cfg).iloc[0]
    more = M.rule_score(feature_row(w_fired=1, w_confirmed=1, n_neg_cats_7d=2, n_max_priority_7d=4), cfg).iloc[0]
    everything = M.rule_score(feature_row(w_fired=1, w_confirmed=1, n_neg_cats_7d=3, n_max_priority_7d=4,
                                          i_cusum_max_7=9), cfg).iloc[0]
    assert none == 0 < one < more <= everything == 1.0
    # priority alone (no note says anything worse) must not fire
    assert M.rule_score(feature_row(n_max_priority_7d=4), cfg).iloc[0] == 0


def test_time_split_purges_label_horizon():
    d = pd.DataFrame({"date": pd.date_range("2026-01-01", periods=100), "y": 0.0})
    split, train, purge, test = M.time_split(d, "y", test_fraction=0.5, purge_days=10)
    assert d.loc[train, "date"].max() < split - pd.Timedelta(days=10) + pd.Timedelta(days=1)
    assert d.loc[test, "date"].min() == split and not (train & purge).any() and not (purge & test).any()
    assert train.sum() + purge.sum() + test.sum() == 100


def test_pr_auc_and_bootstrap():
    y = np.array([0, 0, 0, 1, 0, 1, 0, 0] * 50)
    perfect, noise = y.astype(float), np.random.default_rng(0).random(len(y))
    assert M.pr_auc(y, perfect) == pytest.approx(1.0) and np.isnan(M.pr_auc(np.zeros(5), np.ones(5)))
    groups = np.repeat(np.arange(40), 10)
    d = M.bootstrap_diff(y, perfect, noise, groups, n=60)
    assert d["lo"] > 0 and d["p_better"] == 1.0


def test_standardise_clips_rare_flags():
    z = M.standardise(np.array([[1.0]]), np.array([0.0]), np.array([M.SCALE_FLOOR]))
    assert z[0, 0] == pytest.approx(10.0)                              # not 1,000-sigma


# ── Explanations ─────────────────────────────────────────────────────────────────────────────────────────

def test_logistic_reasons_rank_by_contribution():
    model = {"features": ["w_pct_change", "n_gi_7d", "ctx_age_years"], "coef": [-0.5, 0.8, 0.1],
             "mean": [0, 0, 5], "scale": [5, 1, 3]}
    row = pd.Series({"w_pct_change": -10.0, "n_gi_7d": 2.0, "ctx_age_years": 5.0, "w_confirmed": 1})
    r = explain.logistic_reasons(row, model)
    # contributions: weight −0.5×(−2)=+1.0, gi 0.8×2=+1.6, age 0 → gi first, weight second, age dropped
    assert r == ["Keeper notes: gi worse (2 notes this week)", "Weight -10.0% vs 180-day median (confirmed)"]


def test_rule_reasons_and_quotes_and_confidence():
    comps = pd.Series({"weight_drop": 0.3, "weight_confirmed": 0.0, "notes_one": 0.2})
    assert explain.rule_reasons(comps) == ["Weight dropped vs its 180-day median", "Keeper notes report something worse"]
    sig = pd.DataFrame({"animal_id": [1, 1, 1, 2],
                        "available_date": pd.to_datetime(["2026-03-01", "2026-03-10", "2026-03-20", "2026-03-10"]),
                        "quote": ["seemed dull", "ate half", "future note", "other animal"]})
    now = pd.Timestamp("2026-03-12")
    # newest first; the 20 Mar note (future) and animal 2's note are never shown
    assert explain.recent_quotes(sig, 1, now) == ["“ate half” (10 Mar)", "“seemed dull” (01 Mar)"]
    assert explain.recent_quotes(sig, 1, now, days=5) == ["“ate half” (10 Mar)"]
    assert explain.recent_quotes(sig, 1, now, n=1) == ["“ate half” (10 Mar)"]
    row = pd.Series({"cov_sources": 2.0, "w_has_baseline": 1.0, "n_notes_30d": 3.0, "i_logged_30d": 0.0})
    assert explain.confidence(row).startswith("high")
    assert explain.confidence(row.copy().set_axis(row.index).where(row.index != "cov_sources", 0.0)).startswith("low")
    assert explain.alert_text(0.41, ["A", "B"], ["“q”"], "high").startswith("Risk 0.41 — A · B · Notes: “q”")
