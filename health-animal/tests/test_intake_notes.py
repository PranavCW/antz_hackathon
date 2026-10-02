"""§1.3/B3 intake and §1.4/B4 note rules."""

import numpy as np
import pandas as pd

from src import clean_notes, intake

D = pd.Timestamp


def enclosure_day(rows):
    df = pd.DataFrame(rows, columns=["enclosure_id", "date", "n_animals", "animal_id_if_solo", "confidence"])
    return df.astype({"date": "datetime64[ns]", "animal_id_if_solo": float})


# ── Intake ───────────────────────────────────────────────────────────────────────────────────────────────

def wastage(rows):
    df = pd.DataFrame(rows, columns=["id", "enclosure_id", "date", "leftover_g", "notes"])
    return df.assign(drop_reason=None)


def test_missing_wastage_day_is_unknown_not_zero():
    ed = enclosure_day([(10, D("2026-01-0%d" % d), 1, 7.0, "high") for d in range(1, 6)])
    w = wastage([(1, 10, D("2026-01-01"), 200.0, None), (2, 10, D("2026-01-03"), 0.0, None)])
    changes = pd.DataFrame(columns=["animal_id", "date"])
    daily, cov, _ = intake.daily_intake(w, ed, changes, 0.3)
    assert daily["date"].tolist() == [D("2026-01-01"), D("2026-01-03")]   # days 2, 4, 5 absent, not 0
    assert daily["leftover_g"].tolist() == [200.0, 0.0]                    # a logged 0 stays a real 0
    assert cov.iloc[0]["solo_days"] == 5 and cov.iloc[0]["logging_rate"] == 0.4


def test_wastage_only_on_solo_high_confidence_days():
    ed = enclosure_day([(10, D("2026-01-01"), 2, np.nan, "high"), (11, D("2026-01-01"), 1, 8.0, "low")])
    w = wastage([(1, 10, D("2026-01-01"), 50.0, None), (2, 11, D("2026-01-01"), 50.0, None),
                 (3, 12, D("2026-01-01"), 50.0, None)])
    daily, _, rows = intake.daily_intake(w, ed, pd.DataFrame(columns=["animal_id", "date"]), 0.3)
    assert daily.empty
    assert rows["drop_reason"].tolist() == ["not_single_occupant", "low_confidence_stay", "no_occupant_on_day"]


def test_diet_change_starts_new_segment():
    ed = enclosure_day([(10, D("2026-01-0%d" % d), 1, 7.0, "high") for d in range(1, 5)])
    w = wastage([(i, 10, D("2026-01-0%d" % d), 100.0, None) for i, d in enumerate([1, 2, 3, 4])])
    changes = pd.DataFrame({"animal_id": [7], "date": [D("2026-01-03")]})
    daily, cov, _ = intake.daily_intake(w, ed, changes, 0.3)
    assert daily["diet_segment"].tolist() == [0, 0, 1, 1] and len(cov) == 2


# ── Notes ────────────────────────────────────────────────────────────────────────────────────────────────

TYPES = pd.DataFrame({"id": [1, 51], "parent_id": [np.nan, np.nan], "type_name": ["Medical", "Plumbing Issues"],
                      "string_id": ["medical", "plumbing_issues"]})
ANIMALS = pd.DataFrame({"animal_id": [1, 2, 3]})


def notes(obs_rows, mapping_rows, type_rows=(), comment_rows=(), ed=None):
    obs = pd.DataFrame(obs_rows, columns=["observation_id", "observation_name", "created_for", "created_at",
                                          "modified_at"]).assign(priority="Low", is_deleted=0, active=1)
    mapping = pd.DataFrame(mapping_rows, columns=["observation_id", "ref_type", "ref_id"]).assign(is_deleted=0)
    tm = pd.DataFrame(list(type_rows), columns=["observation_id", "parent_observation_type_id"]).assign(
        child_observation_type_id=np.nan, deleted_on=pd.NaT)
    comments = pd.DataFrame(list(comment_rows), columns=["observation_id", "observation", "created_at"]).assign(active=1)
    ed = ed if ed is not None else enclosure_day([])
    rows = clean_notes.observation_rows(obs, mapping, comments, tm, TYPES, ed, ANIMALS, "UTC", {"plumbing_issues"})
    return clean_notes.finalise(rows, {"all ok"}, 3, 3)


def test_note_attribution_and_weight():
    ed = enclosure_day([(10, D("2026-01-05"), 1, 3.0, "high"), (11, D("2026-01-05"), 2, np.nan, "high")])
    r = notes([(1, "Both tigers ate half", D("2026-01-05"), D("2026-01-05 08:00"), pd.NaT),
               (2, "Enclosure note", D("2026-01-05"), D("2026-01-05 08:00"), pd.NaT),
               (3, "Shared enclosure note", D("2026-01-05"), D("2026-01-05 08:00"), pd.NaT),
               (4, "Section note", D("2026-01-05"), D("2026-01-05 08:00"), pd.NaT)],
              [(1, "animal", 1), (1, "animal", 2), (2, "enclosure", 10), (3, "enclosure", 11), (4, "section", 5)],
              ed=ed)
    kept = r[r["kept"]].set_index(["source_id", "animal_id"])
    assert kept.loc[(1, 1), "weight"] == 0.5 and kept.loc[(1, 2), "weight"] == 0.5
    assert kept.loc[(2, 3), "attribution"] == "solo_enclosure"
    assert set(r.loc[r["source_id"].isin([3, 4]), "drop_reason"]) == {"not_attributable"}


def test_created_for_edit_reset_fixed():
    # note written on Jan 5, edited Jan 20 without resending created_for → created_for reset to Jan 20
    r = notes([(1, "Limping on left leg", D("2026-01-20"), D("2026-01-05 09:00"), D("2026-01-20 10:00"))],
              [(1, "animal", 1)])
    assert r.iloc[0]["date"] == D("2026-01-05") and r.iloc[0]["date_fixed"]


def test_comments_appended_and_filters():
    r = notes([(1, "Not eating", D("2026-01-01"), D("2026-01-01"), pd.NaT),
               (2, "Not eating", D("2026-01-02"), D("2026-01-02"), pd.NaT),     # copy-paste within 3 days
               (3, "All OK.", D("2026-01-02"), D("2026-01-02"), pd.NaT),        # template
               (4, "Pipe leaking", D("2026-01-02"), D("2026-01-02"), pd.NaT),   # maintenance-only type
               (5, "Pipe leaking, tiger stressed", D("2026-01-02"), D("2026-01-02"), pd.NaT)],
              [(i, "animal", 1) for i in range(1, 6)],
              type_rows=[(4, 51), (5, 51), (5, 1)],
              comment_rows=[(1, "still not eating", D("2026-01-02"))])
    by = r.set_index("source_id")
    assert by.loc[1, "text"] == "Not eating\n[comments]\nstill not eating" and by.loc[1, "kept"]
    assert by.loc[2, "drop_reason"] is None and by.loc[2, "kept"]   # different text (no thread) → kept
    assert by.loc[3, "drop_reason"] == "template_text"
    assert by.loc[4, "drop_reason"] == "maintenance_type"
    assert by.loc[5, "kept"]                                        # has a non-maintenance type too


def test_duplicate_text_counted_once():
    r = notes([(1, "Dull, lying in corner", D("2026-01-01"), D("2026-01-01"), pd.NaT),
               (2, "dull,  lying in corner", D("2026-01-03"), D("2026-01-03"), pd.NaT),
               (3, "Dull, lying in corner", D("2026-01-10"), D("2026-01-10"), pd.NaT)],
              [(1, "animal", 1), (2, "animal", 1), (3, "animal", 1)])
    assert r.set_index("source_id")["drop_reason"].to_dict() == {1: None, 2: "duplicate_text", 3: None}
