"""Intake from food wastage (PLAN.md §1.3, B3): leftovers in grams per single-occupant animal-day.

Missing wastage day = unknown (never "ate everything") → only logged days are written. Non-solo and
low-confidence-stay days are skipped. Diet changes split the series into segments (baseline restarts there).
"""

import numpy as np
import pandas as pd

from src.animals import local_day
from src.clean_weights import unit_factors


def clean_wastage(wastage: pd.DataFrame, units: pd.DataFrame, tz: str) -> pd.DataFrame:
    """All wastage rows with grams recomputed from the entered unit, a local date and a drop_reason."""
    w = wastage.copy()
    w["drop_reason"] = None
    factors = unit_factors(units)
    w = w.join(factors["to_g"], on="wastage_quantity_uom")
    w["leftover_g"] = w["wastage_quantity"] * w["to_g"]
    # The stored base quantity is a cross-check only (column comments are swapped in the schema)
    w["base_mismatch"] = (w["base_wastage_quantity"] - w["leftover_g"]).abs() > 0.01 * w["leftover_g"].abs() + 0.5
    when = pd.to_datetime(w["wastage_date"]) + pd.to_timedelta(w["wastage_time"]).fillna(pd.Timedelta(0))
    w["date"] = local_day(when, tz)  # wastage_date/time are stored in UTC

    for mask, reason in [
        ((w["is_deleted"] == 1) | (w["active"] == 0), "deleted"),
        (w["to_g"].isna(), "unit_unresolvable"),
        (w["leftover_g"] < 0, "negative"),
        (w["date"].isna(), "no_date"),
    ]:
        w.loc[mask & w["drop_reason"].isna(), "drop_reason"] = reason
    return w.drop(columns=["to_g"])


def diet_change_dates(diet_logs: pd.DataFrame, animals: pd.DataFrame, tz: str) -> pd.DataFrame:
    """(animal_id, change_date): animal-level diet changes plus species-level changes for its taxonomy."""
    d = diet_logs.assign(date=local_day(diet_logs["created_at"], tz))
    per_animal = d[d["entity_type"] == "animal"].rename(columns={"entity_id": "animal_id"})[["animal_id", "date"]]
    species = d[d["entity_type"] == "species"].merge(
        animals[["animal_id", "taxonomy_id"]], left_on="entity_id", right_on="taxonomy_id")[["animal_id", "date"]]
    return pd.concat([per_animal, species]).drop_duplicates().sort_values(["animal_id", "date"])


def _diet_segment(df: pd.DataFrame, changes: pd.DataFrame) -> pd.Series:
    """Number of diet changes on or before each row's date (segment 0 = before the first known change)."""
    seg = np.zeros(len(df), dtype=int)
    by_animal = {k: np.sort(g.to_numpy()) for k, g in changes.groupby("animal_id")["date"]}
    for i, (animal_id, date) in enumerate(zip(df["animal_id"].to_numpy(), df["date"].to_numpy())):
        dates = by_animal.get(animal_id)
        if dates is not None:
            seg[i] = int(np.searchsorted(dates, date, side="right"))
    return pd.Series(seg, index=df.index)


def daily_intake(w: pd.DataFrame, enclosure_day: pd.DataFrame, changes: pd.DataFrame, min_logging_rate: float,
                 tz: str = "UTC"):
    """Returns (intake_daily per solo animal-day, coverage per animal × diet segment, wastage rows with reasons)."""
    w = w.merge(enclosure_day, on=["enclosure_id", "date"], how="left")
    live = w["drop_reason"].isna()
    w.loc[live & w["n_animals"].isna(), "drop_reason"] = "no_occupant_on_day"
    w.loc[w["drop_reason"].isna() & w["animal_id_if_solo"].isna(), "drop_reason"] = "not_single_occupant"
    w.loc[w["drop_reason"].isna() & (w["confidence"] == "low"), "drop_reason"] = "low_confidence_stay"

    ok = w[w["drop_reason"].isna()].rename(columns={"animal_id_if_solo": "animal_id"})
    daily = (ok.groupby(["animal_id", "enclosure_id", "date"])
               .agg(leftover_g=("leftover_g", "sum"), n_entries=("leftover_g", "size"),
                    has_note=("notes", lambda s: s.fillna("").str.strip().ne("").any()),
                    created_at=("created_at", "max"))
               .reset_index())
    daily["animal_id"] = daily["animal_id"].astype(int)
    # A leftover day is visible to a replay from the day its last entry was inserted (entries can be backdated)
    daily["available_date"] = pd.concat([daily["date"], local_day(daily["created_at"], tz)], axis=1).max(axis=1)
    daily = daily.drop(columns="created_at")
    daily["diet_segment"] = _diet_segment(daily, changes)

    # Coverage: logged days / solo high-confidence days, per animal and diet segment
    solo = enclosure_day[enclosure_day["animal_id_if_solo"].notna() & (enclosure_day["confidence"] == "high")]
    solo = solo.rename(columns={"animal_id_if_solo": "animal_id"}).astype({"animal_id": int})
    solo = solo[solo["animal_id"].isin(daily["animal_id"])]
    solo = solo.assign(diet_segment=_diet_segment(solo, changes))
    cov = solo.groupby(["animal_id", "diet_segment"]).agg(solo_days=("date", "size"), first=("date", "min"),
                                                           last=("date", "max"))
    cov = cov.join(daily.groupby(["animal_id", "diet_segment"]).size().rename("logged_days")).fillna(
        {"logged_days": 0}).reset_index()
    cov["logging_rate"] = cov["logged_days"] / cov["solo_days"]
    cov["low_confidence"] = cov["logging_rate"] < min_logging_rate
    daily = daily.merge(cov[["animal_id", "diet_segment", "low_confidence"]], on=["animal_id", "diet_segment"],
                        how="left")
    return daily, cov, w
