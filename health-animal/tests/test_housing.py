"""§1.6 stay rebuild and single-occupant enclosure-days."""

import numpy as np
import pandas as pd

from src import housing

NOW = pd.Timestamp("2026-01-31")


def animals(rows):
    return pd.DataFrame(rows, columns=["animal_id", "type", "total_animal", "death_at"])


def logs(rows):
    df = pd.DataFrame(rows, columns=["animal_id", "enclosure_id", "in_date", "out_date"])
    df["in_date"], df["out_date"] = pd.to_datetime(df["in_date"]), pd.to_datetime(df["out_date"])
    return df.assign(id=range(len(df)), is_deleted=0)


def test_stay_ends_at_next_stay_and_death():
    a = animals([(1, "single", 1, pd.Timestamp("2026-01-20"))])
    # out_date overwritten to "now" by ON UPDATE → must end at the next stay; open stay capped at death
    s = housing.build_stays(logs([(1, 10, "2026-01-01", "2026-01-30"), (1, 11, "2026-01-10", None)]), a, NOW)
    assert s["end"].tolist() == [pd.Timestamp("2026-01-10"), pd.Timestamp("2026-01-20")]
    assert "end_from_next_stay" in s.iloc[0]["flags"] and "capped_at_death" in s.iloc[1]["flags"]


def test_repeated_visit_is_low_confidence():
    a = animals([(1, "single", 1, pd.NaT)])
    s = housing.build_stays(logs([(1, 10, "2026-01-01", "2026-01-05"), (1, 11, "2026-01-05", "2026-01-08"),
                                  (1, 10, "2026-01-08", None)]), a, NOW)
    assert s["confidence"].tolist() == ["low", "high", "low"]


def test_single_occupant_days():
    a = animals([(1, "single", 1, pd.NaT), (2, "single", 1, pd.NaT), (3, "group", 5, pd.NaT)])
    s = housing.build_stays(logs([
        (1, 10, "2026-01-01", "2026-01-10"),   # alone 1–4, shares 5–10 with animal 2
        (2, 10, "2026-01-05", "2026-01-12"),   # alone 11–12
        (3, 20, "2026-01-01", "2026-01-03"),   # a group of 5 is never "solo"
    ]), a, NOW)
    d = housing.enclosure_days(s, "UTC").set_index(["enclosure_id", "date"])
    e10 = d.loc[10]
    assert e10.loc["2026-01-01":"2026-01-04", "animal_id_if_solo"].eq(1).all()
    assert e10.loc["2026-01-05":"2026-01-10", "n_animals"].eq(2).all()
    assert e10.loc["2026-01-05":"2026-01-10", "animal_id_if_solo"].isna().all()
    assert e10.loc["2026-01-11":"2026-01-12", "animal_id_if_solo"].eq(2).all()
    assert len(e10) == 12
    assert d.loc[20, "n_animals"].eq(5).all() and d.loc[20, "animal_id_if_solo"].isna().all()


def test_low_confidence_propagates_to_days():
    a = animals([(1, "single", 1, pd.NaT)])
    s = housing.build_stays(logs([(1, 10, "2026-01-01", "2026-01-03"), (1, 11, "2026-01-03", "2026-01-04"),
                                  (1, 10, "2026-01-04", "2026-01-05")]), a, NOW)
    d = housing.enclosure_days(s, "UTC")
    assert set(d.loc[d["enclosure_id"] == 10, "confidence"]) == {"low"}
    assert set(d.loc[d["enclosure_id"] == 11, "confidence"]) == {"high"}
    assert np.isnan(d["animal_id_if_solo"]).sum() == 0
