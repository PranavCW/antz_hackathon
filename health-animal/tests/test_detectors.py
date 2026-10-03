"""§C–F detector maths: the PLAN.md worked examples, plus no-future-data behaviour."""

import numpy as np
import pandas as pd
import pytest

from src import detectors

WCFG = {"baseline_days": 180, "min_readings_in_baseline": 3, "min_readings_last_days": 60}
DCFG = {"weight_alert_pct": -5, "weight_alert_slope_pct_month": -3, "juvenile_min_slope_pct_month": 0,
        "intake_weekday_min_points": 3, "intake_mad_floor_g": 10, "mad_floor": 1e-6, "z_threshold": 3}
ICFG = {"baseline_days": 30, "min_valid_days": 10, "cusum_k": 0.5, "cusum_h": 4, "cusum_max_missing_days": 7}


def test_cusum_plan_example_flags_on_day_5():
    z = pd.Series([1.5, 1.2, 1.8, 1.4, 1.6])
    r = detectors.cusum(pd.Series(pd.date_range("2026-01-01", periods=5)), z, k=0.5, h=4, max_missing=7)
    assert r["cusum"].round(1).tolist() == [1.0, 1.7, 3.0, 3.9, 5.0]
    assert r["cusum_flag"].tolist() == [False, False, False, False, True]


def test_cusum_holds_on_missing_and_resets_after_long_gap():
    dates = pd.Series(pd.to_datetime(["2026-01-01", "2026-01-02", "2026-01-05", "2026-01-20"]))
    r = detectors.cusum(dates, pd.Series([2.0, np.nan, 1.5, 1.5]), k=0.5, h=4, max_missing=7)
    # day 2 missing z → held at 1.5; Jan 5 after 2 missing days → continues; Jan 20 after 14 missing → reset
    assert r["cusum"].tolist() == [1.5, 1.5, 2.5, 1.0]


def test_median_ignores_outlier_plan_example():
    x = np.array([96, 94, 97, 95, 20, 96, 93, 95, 97, 94])
    assert x.mean() == pytest.approx(87.7) and np.median(x) == 95


def test_robust_z_scales_with_animals_own_wobble():
    steady, swingy = np.array([100, 101, 99, 100, 100.5]), np.array([80, 120, 95, 110, 100])
    assert detectors.robust_z(94, steady, 100, 1e-6) < -3      # unusual for a steady animal
    assert detectors.robust_z(94, swingy, 100, 1e-6) > -1      # normal for a swingy one


def test_theil_sen_resists_one_bad_point():
    t = np.arange(6, dtype=float)
    y = np.array([100, 99, 98, 40, 96, 95], dtype=float)   # one bad weighing
    assert detectors.theil_sen(t, y) == pytest.approx(-1.0)


def weights(values, days, juvenile=False):
    t = pd.Timestamp("2026-01-01") + pd.to_timedelta(days, unit="D")
    return pd.DataFrame({"animal_id": 1, "recorded_at": t, "local_date": t.normalize(), "value_g": values,
                         "kept": True, "is_juvenile": juvenile})


def test_weight_detector_fires_and_confirms_on_sustained_drop():
    d = detectors.weight_detector(weights([180, 181, 180, 179, 170, 168], [0, 14, 28, 42, 56, 70]), WCFG, DCFG)
    assert d["status"].tolist()[:3] == ["insufficient_data"] * 3       # < 3 previous readings
    assert d["fired"].tolist()[3:] == [False, True, True]
    assert d["confirmed"].tolist()[3:] == [False, False, True]         # first drop pending, second confirms


def test_weight_baseline_uses_only_past_readings():
    # a later recovery must not change the score of the drop day
    a = detectors.weight_detector(weights([180, 181, 180, 165], [0, 14, 28, 42]), WCFG, DCFG)
    b = detectors.weight_detector(weights([180, 181, 180, 165, 200, 200], [0, 14, 28, 42, 56, 70]), WCFG, DCFG)
    assert a.iloc[3]["pct_change"] == b.iloc[3]["pct_change"]


def test_weight_insufficient_without_recent_reading():
    d = detectors.weight_detector(weights([180, 181, 180, 170], [0, 10, 20, 120]), WCFG, DCFG)
    assert d.iloc[3]["status"] == "insufficient_data"                  # nothing in the last 60 days


def test_juvenile_fires_on_stalled_growth_not_level():
    growing = detectors.weight_detector(weights([10, 12, 14, 16], [0, 14, 28, 42], juvenile=True), WCFG, DCFG)
    stalled = detectors.weight_detector(weights([10, 12, 14, 13], [0, 14, 28, 42], juvenile=True), WCFG, DCFG)
    assert not growing.iloc[3]["fired"]
    assert not stalled.iloc[3]["fired"]          # still trending up over the window → not stalled yet
    flat = detectors.weight_detector(weights([14, 14, 14, 13.5], [0, 14, 28, 42], juvenile=True), WCFG, DCFG)
    assert flat.iloc[3]["fired"]


def test_intake_detector_needs_10_days_and_flags_rising_leftovers():
    dates = pd.date_range("2026-01-01", periods=20)
    left = [200, 210, 190, 205, 195, 200, 210, 190, 205, 195, 200, 200, 600, 650, 700, 700, 650, 700, 700, 700]
    daily = pd.DataFrame({"animal_id": 1, "date": dates, "enclosure_id": 10, "diet_segment": 0,
                          "leftover_g": left, "low_confidence": False})
    d = detectors.intake_detector(daily, ICFG, DCFG)
    assert (d["status"].iloc[:10] == "insufficient_data").all() and (d["status"].iloc[10:] == "ok").all()
    assert not d["cusum_flag"].iloc[:12].any() and d["cusum_flag"].iloc[12]
