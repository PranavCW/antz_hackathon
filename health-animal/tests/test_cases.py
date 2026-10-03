"""§I.1–2 / B5 episode rules: severity normalisation, 14-day merge, T0, onset, death attachment, euthanasia."""

import numpy as np
import pandas as pd

from src import cases

D = pd.Timestamp
LCFG = {"episode_merge_days": 14, "euthanasia_lookback_days": 60}


def records(rows):
    """rows: (animal_id, record_id, t0, max_severity, serious, duration_days)"""
    df = pd.DataFrame(rows, columns=["animal_id", "medical_record_id", "t0", "max_severity", "serious", "duration_days"])
    return df.assign(t0_source="record", worst_prognosis=np.nan, hospital_serious=False, critical=False, date_flag="")


def deaths(rows):
    return pd.DataFrame(rows, columns=["animal_id", "death_at", "manner_class"]).assign(is_estimate=0)


def test_severity_and_prognosis_normalised():
    s = pd.Series(["Mild", "Mildd", " high", "Extreme", "Medium", "Low", "", None, "Undetermined"])
    assert cases.severity_rank(s).tolist()[:6] == [1, 1, 3, 4, 2, 1]
    assert cases.severity_rank(s).iloc[6:].isna().all()
    assert cases.prognosis_rank(pd.Series(["favorable", "Poor", "grave", "undetermined"])).tolist()[:3] == [0, 3, 4]


def test_duration_to_days():
    d = cases.duration_days(pd.Series(["3", "2", "1", "", "x"]), pd.Series(["Days", "Weeks", "Months", "Days", "Days"]))
    assert d.tolist()[:3] == [3, 14, 30] and d.iloc[3:].isna().all()


def test_records_within_14_days_merge_and_t0_is_earliest():
    ep = cases.build_episodes(records([
        (1, 10, D("2026-01-01"), 1, False, 5), (1, 11, D("2026-01-10"), 3, True, np.nan),  # one episode, serious
        (1, 12, D("2026-03-01"), 1, False, np.nan),                                       # separate, mild
    ]), deaths([]), LCFG)
    assert len(ep) == 2
    first = ep.iloc[0]
    assert first["t0"] == D("2026-01-01") and first["n_records"] == 2 and first["is_case"]
    assert first["est_onset"] == D("2025-12-27")          # T0 − 5-day complaint duration
    assert ep.iloc[1]["case_class"] == "not_serious"


def test_illness_death_makes_mild_episode_a_case():
    ep = cases.build_episodes(records([(1, 10, D("2026-01-01"), 1, False, np.nan)]),
                              deaths([(1, D("2026-02-10"), "illness")]), LCFG)
    assert len(ep) == 1 and ep.iloc[0]["is_case"] and ep.iloc[0]["outcome"] == "died_illness"


def test_euthanasia_needs_prior_illness_record():
    ep = cases.build_episodes(records([(1, 10, D("2026-01-01"), 1, False, np.nan)]),
                              deaths([(1, D("2026-02-15"), "euthanasia"), (2, D("2026-02-15"), "euthanasia")]), LCFG)
    by = ep.set_index("animal_id")
    assert by.loc[1, "is_case"] and by.loc[1, "outcome"] == "euthanised"
    assert not by.loc[2, "is_case"]                       # no record before → not a case


def test_trauma_death_reported_separately():
    ep = cases.build_episodes(records([(1, 10, D("2026-01-01"), 4, True, np.nan)]),
                              deaths([(1, D("2026-01-05"), "trauma")]), LCFG)
    assert ep.iloc[0]["case_class"] == "trauma" and not ep.iloc[0]["is_case"]


def test_death_without_record_is_its_own_episode():
    ep = cases.build_episodes(records([(1, 10, D("2025-01-01"), 1, False, np.nan)]),
                              deaths([(1, D("2026-01-01"), "illness")]), LCFG)   # record > 60 d before death
    assert len(ep) == 2
    lone = ep[ep["t0_source"] == "death"].iloc[0]
    assert lone["t0"] == D("2026-01-01") and lone["is_case"]
