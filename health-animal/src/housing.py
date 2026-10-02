"""Housing history (PLAN.md §1.6): rebuild stays from antz_enclosure_logs and derive who was alone where.

Stays: out_date is unreliable (ON UPDATE CURRENT_TIMESTAMP, closures matched on (enclosure, animal), death
doesn't close it), so a stay ends at the earliest of out_date, the animal's next stay, death, or extract time.
Repeated visits of the same animal to the same enclosure → `low_confidence` (out_dates may be overwritten).

Enclosure-days: an enclosure-day is single-occupant iff SUM(total_animal) over animals present that day = 1.
Computed with an interval sweep per enclosure (exploding every stay per day would be ~30M rows).
"""

import numpy as np
import pandas as pd

from src.animals import local_day


def build_stays(logs: pd.DataFrame, animals: pd.DataFrame, now: pd.Timestamp) -> pd.DataFrame:
    """One row per stay: animal_id, enclosure_id, start, end (UTC), confidence, flags."""
    s = logs[logs["is_deleted"] == 0].merge(
        animals[["animal_id", "type", "total_animal", "death_at"]], on="animal_id", how="inner")
    s = s.dropna(subset=["in_date"]).sort_values(["animal_id", "in_date", "id"]).reset_index(drop=True)
    s["flags"] = ""

    def flag(mask, name):
        s.loc[mask, "flags"] = (s.loc[mask, "flags"] + ";" + name).str.strip(";")

    next_in = s.groupby("animal_id")["in_date"].shift(-1)
    end = s["out_date"].copy()
    flag(end.notna() & (end < s["in_date"]), "out_before_in")
    overrun = next_in.notna() & (end.isna() | (end > next_in) | (end < s["in_date"]))
    flag(overrun & end.notna(), "end_from_next_stay")
    end = end.where(~overrun, next_in)
    flag(end.isna(), "open")
    end = end.fillna(now)
    capped = s["death_at"].notna() & (end > s["death_at"])
    flag(capped, "capped_at_death")
    end = end.where(~capped, s["death_at"])
    s["start"], s["end"] = s["in_date"], end

    flag(s["start"] > s["end"], "starts_after_death")
    repeated = s.duplicated(["animal_id", "enclosure_id"], keep=False)
    flag(repeated, "repeated_visit")
    s["confidence"] = np.where(repeated | s["flags"].str.contains("out_before_in"), "low", "high")
    s = s[s["start"] <= s["end"]]
    return s[["id", "animal_id", "enclosure_id", "type", "total_animal", "start", "end", "confidence", "flags"]]


def movement_check(stays: pd.DataFrame, movements: pd.DataFrame, tolerance_days: int = 2) -> dict:
    """Cross-check: share of movements whose destination has a stay starting within ±N days of transferred_on."""
    m = movements.dropna(subset=["transferred_on", "destination_id"])
    if m.empty:
        return {"movements": 0, "matched": 0}
    j = m.merge(stays[["animal_id", "enclosure_id", "start"]], left_on=["entity_id", "destination_id"],
                right_on=["animal_id", "enclosure_id"], how="left")
    j["ok"] = (j["start"] - j["transferred_on"]).abs() <= pd.Timedelta(days=tolerance_days)
    matched = j.groupby("id")["ok"].any()
    return {"movements": int(len(m)), "matched": int(matched.sum())}


def enclosure_days(stays: pd.DataFrame, tz: str) -> pd.DataFrame:
    """enclosure_day(enclosure_id, date, n_animals, animal_id_if_solo, confidence) for every occupied day.

    An animal counts on every local day its stay touches (move-day counts in both enclosures).
    """
    s = stays.assign(d0=local_day(stays["start"], tz), d1=local_day(stays["end"], tz))
    s["low"] = (s["confidence"] == "low").astype(int)
    s["single"] = (s["type"] == "single").astype(int)
    one = pd.Timedelta(days=1)

    # Sweep: +1 at the first day, −1 the day after the last day; cumulative sums give occupancy per segment.
    # animal_id is summed with sign too: when exactly one stay is open, the running sum *is* its animal_id.
    cols = ["enclosure_id", "total_animal", "low", "animal_id", "single"]
    starts = s[cols].assign(date=s["d0"], n_stays=1)
    ends = (-s[cols[1:]]).assign(enclosure_id=s["enclosure_id"], date=s["d1"] + one, n_stays=-1)
    ev = pd.concat([starts, ends])
    agg = ev.groupby(["enclosure_id", "date"])[["total_animal", "low", "animal_id", "single", "n_stays"]].sum()
    seg = agg.groupby(level="enclosure_id").cumsum().reset_index()
    seg["seg_end"] = seg.groupby("enclosure_id")["date"].shift(-1) - one
    seg = seg.dropna(subset=["seg_end"])
    seg = seg[seg["n_stays"] > 0]

    # Explode segments into days (bounded: enclosures × days)
    lengths = ((seg["seg_end"] - seg["date"]).dt.days + 1).to_numpy()
    rep = seg.loc[seg.index.repeat(lengths)].copy()
    rep["date"] = rep["date"] + pd.to_timedelta(np.concatenate([np.arange(n) for n in lengths]), unit="D")
    solo = (rep["total_animal"] == 1) & (rep["n_stays"] == 1) & (rep["single"] == 1)
    out = pd.DataFrame({
        "enclosure_id": rep["enclosure_id"].to_numpy(), "date": rep["date"].to_numpy(),
        "n_animals": rep["total_animal"].to_numpy(),
        "animal_id_if_solo": np.where(solo, rep["animal_id"], np.nan),
        "confidence": np.where(rep["low"] > 0, "low", "high"),
    })
    return out
