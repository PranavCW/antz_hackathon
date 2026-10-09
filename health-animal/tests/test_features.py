"""Label rules (§H) and the §6 leakage guarantees, on small synthetic inputs."""

import numpy as np
import pandas as pd
import pytest

from src import config
from src import features as F

D = pd.Timestamp
LCFG = {"horizon_days": 14, "negative_clear_days": 60, "treatment_buffer_days": 7}


# ── Labels ───────────────────────────────────────────────────────────────────────────────────────────────

def grid(animal_id, start, end):
    return pd.DataFrame({"animal_id": animal_id, "date": pd.date_range(start, end)})


def episodes(rows):
    df = pd.DataFrame(rows, columns=["animal_id", "t0", "last_t", "n_records", "is_case", "case_class", "t0_source"])
    return df.assign(t0=pd.to_datetime(df["t0"]), last_t=pd.to_datetime(df["last_t"]))


def at(lab, g, day):
    return lab.loc[g.index[g["date"] == D(day)][0]]


def test_label_horizon_grey_zone_treatment_and_clear():
    g = grid(1, "2025-12-01", "2026-03-31")
    e = episodes([(1, "2026-02-20", "2026-02-25", 2, True, "illness", "record")])
    lab = F.label_days(g, e, LCFG)
    assert at(lab, g, "2026-02-19")["label_vet"] == 1 and at(lab, g, "2026-02-06")["label_vet"] == 1   # d+14 inclusive
    r = at(lab, g, "2026-02-05")
    assert np.isnan(r["label_vet"]) and r["excl_reason"] == "grey_zone"                                # 15 d out
    # 50 days out is inside the 60 d clear window → excluded as grey zone, not a negative
    assert np.isnan(at(lab, g, "2026-01-01")["label_vet"]) and at(lab, g, "2026-01-01")["excl_reason"] == "grey_zone"
    assert at(lab, g, "2025-12-15")["label_vet"] == 0                                                  # 67 d out
    for day in ("2026-02-20", "2026-02-25", "2026-03-04"):                                             # under treatment
        assert np.isnan(at(lab, g, day)["label_vet"]) and at(lab, g, day)["excl_reason"] == "treatment"
    assert at(lab, g, "2026-03-05")["label_vet"] == 0                                                  # buffer over


def test_death_only_case_positive_only_for_label_all():
    g = grid(2, "2025-12-01", "2026-02-28")                  # grid stops the day before death (2026-03-01)
    e = episodes([(2, "2026-03-01", "2026-03-01", 0, True, "illness", "death")])
    lab = F.label_days(g, e, LCFG)
    r = at(lab, g, "2026-02-20")
    assert r["label_all"] == 1 and np.isnan(r["label_vet"]) and r["excl_reason"] == "death_only_case"
    assert at(lab, g, "2025-12-15")["label_vet"] == 0


def test_trauma_death_window_excluded_and_mild_episode_not_a_case():
    g = grid(3, "2025-12-01", "2026-02-28")
    e = episodes([(3, "2026-03-01", "2026-03-01", 0, False, "trauma", "death"),
                  (3, "2025-12-20", "2025-12-22", 1, False, "not_serious", "record")])
    lab = F.label_days(g, e, LCFG)
    assert at(lab, g, "2026-02-10")["excl_reason"] == "other_event"
    assert at(lab, g, "2025-12-21")["excl_reason"] == "treatment"
    assert at(lab, g, "2025-12-10")["label_all"] == 0        # a mild episode ahead is not a positive


# ── Features: replay rules + leakage ─────────────────────────────────────────────────────────────────────

@pytest.fixture(scope="module")
def cfg():
    c = config.load()
    return {**c, "species": [100]}


def synthetic_inputs():
    t = pd.date_range("2026-01-01", periods=9, freq="14D")
    vals = [180, 181, 180, 150, 179, 180, 160, 158, 157]
    w = pd.DataFrame({"animal_id": 1, "recorded_at": t + pd.Timedelta(hours=10), "value_g": np.array(vals) * 1000.0,
                      "kept": True, "drop_reason": None, "is_juvenile": False})
    w.loc[3, ["kept", "drop_reason"]] = [False, "spike"]            # 150 is a one-off spike, known only later
    w["created_at"] = w["recorded_at"] + pd.Timedelta(minutes=5)
    w.loc[6, "created_at"] = w.loc[6, "recorded_at"] + pd.Timedelta(days=5)   # entered 5 days late
    w["local_date"] = w["recorded_at"].dt.normalize()
    animals = pd.DataFrame({"animal_id": [1, 2], "taxonomy_id": 100, "birth_date": D("2015-01-01"),
                            "sex": "male", "death_at": pd.NaT})
    intake = pd.DataFrame(columns=["animal_id", "date", "available_date", "leftover_g", "diet_segment",
                                   "enclosure_id", "low_confidence"])
    notes = pd.DataFrame({"animal_id": [1, 1], "available_date": [D("2026-02-10"), D("2026-02-20")],
                          "weight": 1.0, "priority": ["High", "Low"]})
    signals = pd.DataFrame({"animal_id": [1], "available_date": [D("2026-02-10")], "category": "gi", "weight": 1.0,
                            "quote": "loose stool"})
    stays = pd.DataFrame({"animal_id": [1, 2], "start": [D("2025-12-01"), D("2025-12-01")]})
    return F.Inputs(animals=animals, weights=w, intake=intake, notes=notes, signals=signals, stays=stays,
                    species=[100], tz="UTC")


def feats(inp, cfg, end="2026-04-30"):
    return F.build_features(inp, cfg, D("2026-01-01"), D(end)).set_index(["animal_id", "date"])


def test_backdated_note_visible_only_from_entry_day(cfg):
    inp = synthetic_inputs()
    inp.signals["available_date"] = D("2026-02-18")        # about Feb 10, entered Feb 18
    f = feats(inp, cfg)
    assert f.loc[(1, D("2026-02-17")), "n_gi_7d"] == 0
    assert f.loc[(1, D("2026-02-18")), "n_gi_7d"] == 1
    assert f.loc[(1, D("2026-02-24")), "n_gi_7d"] == 1 and f.loc[(1, D("2026-02-25")), "n_gi_7d"] == 0   # 7 days incl. today


def test_late_entered_weight_not_used_before_entry(cfg):
    f = feats(synthetic_inputs(), cfg)
    # reading #6 (160 kg, recorded Mar 26) was entered Mar 31
    assert f.loc[(1, D("2026-03-28")), "w_days_since_last"] >= 0
    assert f.loc[(1, D("2026-03-30")), "w_pct_change"] > -5            # drop not visible yet
    assert f.loc[(1, D("2026-03-31")), "w_pct_change"] <= -5 and f.loc[(1, D("2026-03-31")), "w_fired"] == 1


def test_spike_is_pending_then_leaves_baseline(cfg):
    f = feats(synthetic_inputs(), cfg)
    assert f.loc[(1, D("2026-02-12")), "w_pct_change"] <= -5           # spike visible on its own day (pending)
    # next reading (179 kg) is scored vs the 180 kg median: (179 − 180) / 180
    assert f.loc[(1, D("2026-02-26")), "w_pct_change"] == pytest.approx(-0.556, abs=0.01)


def test_features_identical_when_future_is_removed(cfg):
    inp = synthetic_inputs()
    full = feats(inp, cfg)
    cols = F.feature_columns(full.reset_index())
    for as_of in ["2026-02-05", "2026-02-12", "2026-02-27", "2026-03-31", "2026-04-15"]:
        t = F.build_features(F.truncate_inputs(inp, D(as_of)), cfg, D("2026-01-01"), D(as_of)).set_index(
            ["animal_id", "date"])
        common = t.index.intersection(full.index)
        assert len(common) > 0
        assert np.allclose(t.loc[common, cols].to_numpy(float), full.loc[common, cols].to_numpy(float),
                           equal_nan=True), f"features differ when data after {as_of} is removed"
    assert F.leakage_check(inp, cfg, D("2026-01-01"), D("2026-04-30"), full.reset_index(), n_animals=1, n_dates=3) > 0


def test_leakage_check_catches_a_peeking_feature(cfg, monkeypatch):
    """Sanity of the check itself: a feature that looks at the series' final value must fail it."""
    inp = synthetic_inputs()
    real = F.build_features

    def peeking(i, c, s, e):
        out = real(i, c, s, e)
        out["w_pct_change"] = out["w_pct_change"] + float(i.weights["value_g"].iloc[-1]) / 1e6   # uses the last reading
        return out

    monkeypatch.setattr(F, "build_features", peeking)
    full = peeking(inp, cfg, D("2026-01-01"), D("2026-04-30"))
    with pytest.raises(AssertionError, match="Leakage"):
        F.leakage_check(inp, cfg, D("2026-01-01"), D("2026-04-30"), full, n_animals=1, n_dates=3)


def test_no_medical_inputs_or_label_like_columns(cfg):
    import inspect
    assert set(inspect.signature(F.build_features).parameters) == {"inp", "cfg", "grid_start", "grid_end"}
    assert set(F.Inputs.__dataclass_fields__) == {"animals", "weights", "intake", "notes", "signals", "stays",
                                                  "species", "tz"}
    cols = F.feature_columns(feats(synthetic_inputs(), cfg).reset_index())
    assert cols and not [c for c in cols if any(w in c for w in F.FORBIDDEN_IN_FEATURES)]
