"""Shared helpers for clean-stage modules: animal master with species name, death times, UTC → local dates."""

import pandas as pd


def local_day(ts_utc: pd.Series, tz: str) -> pd.Series:
    """Naive UTC timestamps → zoo-local calendar day (as midnight datetime64, so it stays mergeable)."""
    ts = pd.to_datetime(ts_utc)
    return ts.dt.tz_localize("UTC").dt.tz_convert(tz).dt.tz_localize(None).dt.normalize()


def death_times(mortality: pd.DataFrame) -> pd.Series:
    """animal_id → death time (UTC) from approved, active mortality rows (§1.5; superseded rows excluded)."""
    m = mortality[(mortality["is_active"] == 1) & (mortality["status"] == "APPROVED")
                  & (mortality["entity_type"] == "animal")]
    t = m["date_of_death"].fillna(m["discovered_date"])
    return t.groupby(m["entity_id"]).min().dropna().rename("death_at")


def species_names(taxonomy: pd.DataFrame) -> pd.Series:
    """tsn → common name. tsn is not unique (zoos override names); prefer the global row, then the zoo row."""
    t = taxonomy.assign(_global=taxonomy["zoo_id"].fillna(0).eq(0)).sort_values("_global", ascending=False)
    t = t.drop_duplicates("tsn")
    return t.set_index("tsn")["default_common_name"].fillna(t.set_index("tsn")["complete_name"]).rename("species")


def animal_table(animals: pd.DataFrame, taxonomy: pd.DataFrame, mortality: pd.DataFrame) -> pd.DataFrame:
    """Non-deleted animals with species name, single/group, birth and death times."""
    a = animals[animals["is_deleted"] == 0][
        ["animal_id", "taxonomy_id", "type", "total_animal", "sex", "birth_date", "approximate_dob", "is_alive"]
    ].copy()
    a["total_animal"] = a["total_animal"].fillna(1)
    a = a.join(species_names(taxonomy), on="taxonomy_id").join(death_times(mortality), on="animal_id")
    return a.reset_index(drop=True)
