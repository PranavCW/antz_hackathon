"""One test (or more) per B1 row of PLAN.md, plus the §1.2 unit, parse and dedup rules."""

import numpy as np
import pandas as pd
import pytest

from src import clean_weights as cw

UNITS = pd.DataFrame([
    # id, unit_name, string_id, zoo_id, measurement_type, base_uom_id, conversion_factor
    (1, "gram", "gram", np.nan, "weight", 1, 1.0),
    (2, "kilogram", "kilogram", np.nan, "weight", 1, 1000.0),
    (3, "milligram", "milligram", np.nan, "weight", 1, 0.001),
    (5, "pound", "pound", np.nan, "weight", 1, 453.592),
    (8, "centimetre", "centimetre", np.nan, "length", 11, 0.01),
    (24, "zoo self-based", "zsb", 11, "weight", 24, 1.0),
    (49, "zoo kilo", "zkilo", 11, "weight", 2, 1.0),
    (38, "zero factor", "zf", 11, "weight", np.nan, 0.0),
], columns=["id", "unit_name", "string_id", "zoo_id", "measurement_type", "base_uom_id", "conversion_factor"])

WCFG = {"min_date": "2000-01-01", "near_duplicate_minutes": 1, "unit_slip_tolerance": 0.05,
        "species_band_pct": [1, 99], "species_band_factor": [0.5, 1.5], "juvenile_days": 365,
        "band_min_readings": 10, "band_min_animals": 3, "hampel_window": 3, "hampel_sigmas": 3,
        "hampel_mad_floor_pct": 0.01}
NOW = pd.Timestamp("2026-10-01")


def animals(ids=(1,), taxonomy_id=100, type_="single"):
    return pd.DataFrame({"animal_id": list(ids), "type": type_, "taxonomy_id": taxonomy_id,
                         "birth_date": pd.Timestamp("2015-01-01"), "death_at": pd.NaT})


def readings(values, unit_id=2, animal_id=1, start="2026-01-01", step_days=14, **extra):
    n = len(values)
    df = pd.DataFrame({
        "animal_id": animal_id, "source": "assessment", "source_id": range(n), "value_raw": [str(v) for v in values],
        "unit_id": unit_id, "recorded_at": pd.date_range(start, periods=n, freq=f"{step_days}D"),
        "time_imputed": False, "base_uom_value": np.nan, "created_at": pd.NaT, "record_date": pd.NaT, "comments": None,
    })
    for k, v in extra.items():
        df[k] = v
    return df


def run(values, **kw):
    a = kw.pop("animals_df", None)
    df, _ = cw.clean(readings(values, **kw), UNITS, a if a is not None else animals(), WCFG, NOW, "UTC")
    return df.sort_values("recorded_at").reset_index(drop=True)


# ── §1.2 parse + units ───────────────────────────────────────────────────────────────────────────────────

@pytest.mark.parametrize("raw, expected", [
    ("12.5", (12.5, None)), ("12,5", (12.5, None)), ("1,234.5", (1234.5, None)), ("12 kg", (12.0, "kilogram")),
    ("12kg", (12.0, "kilogram")), ("350 G", (350.0, "gram")), ("4 lbs", (4.0, "pound")), (" 7 ", (7.0, None)),
    ("abc", (None, None)), ("", (None, None)), ("12 apples", (None, None)), (None, (None, None)),
])
def test_parse_value(raw, expected):
    assert cw.parse_value(raw) == expected


def test_unit_factors_resolve_base_chain():
    f = cw.unit_factors(UNITS)
    assert f.at[2, "to_g"] == 1000 and f.at[3, "to_g"] == 0.001
    assert f.at[49, "to_g"] == 1000                           # one hop: zoo unit based on kg
    assert f.at[24, "unit_reason"] == "unit_unresolvable"     # zoo unit that is its own base
    assert f.at[38, "unit_reason"] == "unit_unresolvable"     # zero factor
    assert f.at[8, "unit_reason"] == "unit_not_weight"


def test_non_numeric_and_bad_units_dropped():
    df = cw.normalise(pd.concat([readings(["abc"]), readings([5], unit_id=8), readings([5], unit_id=24),
                                 readings([0]), readings([5], unit_id=np.nan)]), UNITS)
    assert list(df["drop_reason"]) == ["value_not_numeric", "unit_not_weight", "unit_unresolvable",
                                       "value_not_positive", "unit_missing"]


def test_suffix_overrides_selected_unit():
    df = cw.normalise(readings(["180 kg"], unit_id=1), UNITS)
    assert df.at[0, "value_g"] == 180_000 and "unit_from_suffix" in df.at[0, "flags"]


def test_base_uom_cross_check_flags():
    df = cw.normalise(readings([10, 10], base_uom_value=[0, 9000]), UNITS)
    assert "base_zero" in df.at[0, "flags"] and "base_mismatch" in df.at[1, "flags"]


# ── B1 rows ──────────────────────────────────────────────────────────────────────────────────────────────

def test_kg_g_slip_corrected():
    # 180 kg tiger; one reading entered in grams under the kg unit → 180000 kg
    df = run([180, 181, 180_500, 179])
    assert df.at[2, "value_g"] == pytest.approx(180_500)
    assert "corrected_unit_x1000" in df.at[2, "flags"] and df.at[2, "kept"]


def test_g_kg_slip_corrected():
    df = run([180_000, 181_000, 180, 179_000], unit_id=1)
    assert df.at[2, "value_g"] == pytest.approx(180_000)
    assert "corrected_unit_x0.001" in df.at[2, "flags"]


def test_kg_lb_slip_corrected():
    # reading typed in pounds under the kg unit: 180 kg ≈ 396.8 lb
    df = run([180, 181, 396.8, 179])
    assert df.at[2, "value_g"] == pytest.approx(180_000, rel=0.01)
    assert "corrected_unit_lb_as_kg" in df.at[2, "flags"]


def test_no_slip_fix_against_mixed_unit_series():
    # reference readings are themselves a mix of units → median is meaningless, nothing is "corrected"
    df = run([24, 0.024, 24, 0.024, 24, 0.024])
    assert not df["flags"].str.contains("corrected_unit").any()


def test_typo_outside_species_band_dropped():
    # 3 tigers ~180 kg → band ≈ [89, 272] kg. 18 kg typo dropped (not a ÷10 slip we can fix).
    parts = [readings([178, 180, 182, 181], animal_id=i, start=f"2026-01-0{i}") for i in (1, 2, 3)]
    parts.append(readings([18], animal_id=1, start="2026-03-15"))
    df, bands = cw.clean(pd.concat(parts), UNITS, animals((1, 2, 3)), WCFG, NOW, "UTC")
    assert set(df.loc[df["value_g"] == 18_000, "drop_reason"]) == {"outside_species_band"}
    assert df["kept"].sum() == 12 and len(bands) == 1


def test_one_off_spike_dropped():
    df = run([180, 181, 150, 179])
    assert df.at[2, "drop_reason"] == "spike"
    assert df["kept"].tolist() == [True, True, False, True]


def test_confirmed_drop_kept():
    df = run([180, 181, 165, 163])
    assert df["kept"].all()
    assert df.at[2, "confirmation"] == "confirmed" and df.at[3, "confirmation"] == "confirmed"


def test_latest_only_drop_pending():
    df = run([180, 181, 180, 165])
    assert df.at[3, "kept"] and df.at[3, "confirmation"] == "pending_confirmation"


def test_steady_series_untouched():
    df = run([180, 181, 180, 179, 180, 181])
    assert df["kept"].all() and df["confirmation"].isna().all()


def test_exact_and_near_duplicates_dropped():
    r = readings([180, 180, 180, 182])
    r.loc[1, "recorded_at"] = r.loc[0, "recorded_at"]                              # exact duplicate
    r.loc[2, "recorded_at"] = r.loc[0, "recorded_at"] + pd.Timedelta(seconds=40)   # medical + normal write
    df, _ = cw.clean(r, UNITS, animals(), WCFG, NOW, "UTC")
    assert sorted(df["drop_reason"].dropna()) == ["duplicate", "near_duplicate"]


def test_legacy_assessment_dedup():
    a = readings([180, 181]).assign(created_at=pd.Timestamp("2024-01-01"), record_date=pd.Timestamp("2024-01-01"))
    legacy = pd.DataFrame({
        "id": [10, 11], "animal_id": 1, "measurement_type_id": 1, "measurement_unit_id": 2,
        "measurement_value": [180.0, 175.0], "record_date": pd.Timestamp("2024-01-01"),
        "record_time": pd.to_timedelta(["10:00:00", "11:00:00"]), "created_at": pd.Timestamp("2024-01-01"),
        "comments": None,
    })
    kept = cw.legacy_readings(legacy, a)
    assert kept["source_id"].tolist() == [11]   # 180 kg copy dropped, 175 kg (only in legacy) kept


def test_impossible_dates_dropped():
    r = pd.concat([readings([180], start="2027-01-01"), readings([180], start="1970-01-01"),
                   readings([180], start="2014-06-01"), readings([180], start="2026-02-01")])
    a = animals().assign(death_at=pd.Timestamp("2026-01-15"))
    df, _ = cw.clean(r, UNITS, a, WCFG, NOW, "UTC")
    assert sorted(df["drop_reason"]) == ["after_death", "before_birth", "impossible_date", "impossible_date"]


def test_groups_excluded():
    df = run([180, 181, 182], animals_df=animals(type_="group"))
    assert set(df["drop_reason"]) == {"group_animal"}


def test_parse_value_dotted_text_is_not_a_weight():
    from src.clean_weights import parse_value
    assert parse_value("1.2.3") == (None, None)
    assert parse_value("12.05.23") == (None, None)
    assert parse_value("1,234.5") == (1234.5, None)
