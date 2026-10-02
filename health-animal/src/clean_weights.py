"""Weight cleaning (PLAN.md §1.2, B1–B2): unit normalisation to grams, safe parse, timestamp rule, dedup, and
the B1 per-animal rules (unit slips, species band, Hampel spike filter with confirmation).

Every row is kept in the output: dropped rows carry `drop_reason`, fixes and caveats go into `flags`
(';'-separated). Pure functions on DataFrames so each rule can be unit-tested without the DB.
"""

import re

import numpy as np
import pandas as pd

GRAM = "gram"
# Unit suffixes keepers type into the value field → global unit string_id
SUFFIXES = {
    "g": "gram", "gm": "gram", "gms": "gram", "gr": "gram", "gram": "gram", "grams": "gram",
    "kg": "kilogram", "kgs": "kilogram", "kilo": "kilogram", "kilos": "kilogram", "kilogram": "kilogram",
    "kilograms": "kilogram", "mg": "milligram", "lb": "pound", "lbs": "pound", "pound": "pound",
    "pounds": "pound", "oz": "ounce", "st": "stone", "t": "tonne", "ton": "tonne", "tonne": "tonne",
}
_NUMBER = re.compile(r"([0-9]+(?:[.,][0-9]+)*|[.,][0-9]+)\s*([a-z]+)?\.?")

READING_COLUMNS = ["animal_id", "source", "source_id", "value_raw", "unit_id", "recorded_at", "time_imputed",
                   "base_uom_value", "created_at", "record_date", "comments"]


# ── Parsing and units ────────────────────────────────────────────────────────────────────────────────────

def parse_value(raw) -> tuple:
    """Safe-parse a weight value typed as text. Returns (number or None, suffix unit string_id or None).

    "12,5" → 12.5; "1,234.5" → 1234.5; "12 kg" → (12, "kilogram"); "abc" / "" → (None, None).
    """
    if raw is None or (isinstance(raw, float) and np.isnan(raw)):
        return None, None
    if isinstance(raw, (int, float)):
        return float(raw), None
    m = _NUMBER.fullmatch(str(raw).strip().lower())
    if not m:
        return None, None
    num, suffix = m.group(1), m.group(2)
    if suffix and suffix not in SUFFIXES:
        return None, None
    if "," in num and "." in num:
        num = num.replace(",", "")           # thousands separator
    elif num.count(",") == 1:
        num = num.replace(",", ".")          # decimal comma
    else:
        num = num.replace(",", "")
    return float(num), SUFFIXES.get(suffix)


def unit_factors(units: pd.DataFrame) -> pd.DataFrame:
    """unit id → grams-per-unit (`to_g`) or a reason why it can't be resolved (§1.2 normalisation rule).

    to_g = factor if the unit's base is gram, else factor × base unit's factor (one hop), else unresolvable.
    """
    u = units.set_index("id")
    gram_ids = set(u.index[(u["string_id"] == GRAM) & u["zoo_id"].isna()])
    out = []
    for uid, row in u.iterrows():
        factor, base = row["conversion_factor"], row["base_uom_id"]
        to_g, reason = None, None
        if row["measurement_type"] != "weight":
            reason = "unit_not_weight"
        elif pd.isna(factor) or factor <= 0:
            reason = "unit_unresolvable"
        elif uid in gram_ids or base in gram_ids:
            to_g = float(factor)
        elif pd.notna(base) and base != uid and base in u.index:
            b = u.loc[base]
            if b["measurement_type"] == "weight" and b["base_uom_id"] in gram_ids and b["conversion_factor"] > 0:
                to_g = float(factor) * float(b["conversion_factor"])
            else:
                reason = "unit_unresolvable"
        else:
            reason = "unit_unresolvable"   # no base, or a zoo unit that is its own base
        out.append({"unit_id": uid, "unit_name": row["unit_name"], "to_g": to_g, "unit_reason": reason})
    return pd.DataFrame(out).set_index("unit_id")


def global_unit_ids(units: pd.DataFrame) -> dict:
    """string_id → id for the global (non-zoo) units, used to resolve typed suffixes."""
    g = units[units["zoo_id"].isna()]
    return dict(zip(g["string_id"], g["id"]))


# ── Building the reading table ───────────────────────────────────────────────────────────────────────────

def _add_flag(df: pd.DataFrame, mask, flag: str) -> None:
    mask = pd.Series(mask, index=df.index).fillna(False).astype(bool)
    df.loc[mask, "flags"] = (df.loc[mask, "flags"] + ";" + flag).str.strip(";")


def _drop(df: pd.DataFrame, mask, reason: str) -> None:
    """Set drop_reason on rows not already dropped (first reason wins)."""
    mask = pd.Series(mask, index=df.index).fillna(False).astype(bool) & df["drop_reason"].isna()
    df.loc[mask, "drop_reason"] = reason


def assessment_readings(assessments: pd.DataFrame, values: pd.DataFrame, weight_type_ids) -> pd.DataFrame:
    """Weight rows from antz_animal_assessments; value/unit from the child values table when present."""
    a = assessments[assessments["assessment_type_id"].isin(weight_type_ids)].copy()
    v = (values[values["is_deleted"] == 0].sort_values("id")
         .drop_duplicates("assessment_id")[["assessment_id", "assessment_value", "assessment_unit_id"]]
         .rename(columns={"assessment_value": "v_value", "assessment_unit_id": "v_unit"}))
    a = a.merge(v, left_on="id", right_on="assessment_id", how="left")
    a["value_raw"] = a["v_value"].where(a["v_value"].notna(), a["assessment_value"])
    a["unit_id"] = a["v_unit"].where(a["v_unit"].notna(), a["assessment_unit_id"])
    # Timestamp rule (§1.2): recorded_date_time, else record_date + record_time, else created_at
    date_time = pd.to_datetime(a["record_date"]) + pd.to_timedelta(a["record_time"])
    a["recorded_at"] = a["recorded_date_time"].fillna(date_time).fillna(a["created_at"])
    a["time_imputed"] = a["recorded_date_time"].isna() & a["record_time"].isna()
    a = a[~a["is_deleted"].fillna(0).astype(bool)]
    return a.assign(source="assessment", source_id=a["id"])[READING_COLUMNS]


def legacy_readings(legacy: pd.DataFrame, assessment_rows: pd.DataFrame) -> pd.DataFrame:
    """Weight rows from antz_animal_measurement not already copied into assessments (§1.2 dedup rule)."""
    m = legacy[legacy["measurement_type_id"] == 1].copy()
    date_time = pd.to_datetime(m["record_date"]) + pd.to_timedelta(m["record_time"])
    m["recorded_at"] = date_time.fillna(m["created_at"])
    m["time_imputed"] = m["record_time"].isna()
    m = m.assign(source="legacy", source_id=m["id"], value_raw=m["measurement_value"],
                 unit_id=m["measurement_unit_id"], base_uom_value=np.nan)

    # Legacy kept only if no assessment row has the same animal, unit, value, record_date and created_at
    def keys(df):
        return pd.DataFrame({
            "animal_id": pd.to_numeric(df["animal_id"]), "unit_id": pd.to_numeric(df["unit_id"]),
            "v": pd.to_numeric(df["value_raw"], errors="coerce").round(2),
            "record_date": pd.to_datetime(df["record_date"]), "created_at": pd.to_datetime(df["created_at"]),
        }, index=df.index)

    copied_keys = keys(assessment_rows).drop_duplicates().assign(_copied=True)
    copied = keys(m).merge(copied_keys, how="left", on=list(copied_keys.columns[:-1]))["_copied"].notna()
    return m.loc[~copied.to_numpy(), READING_COLUMNS]


def anaesthesia_readings(pre: pd.DataFrame, hospital_animals: pd.DataFrame) -> pd.DataFrame:
    """Actual (not estimated) weights from pre-anaesthesia forms, linked to the animal via the hospital case."""
    p = pre[(pre["weight_type"] == "Actual") & pre["weight"].notna() & (pre["anaesthesia_is_deleted"] == 0)]
    p = p.merge(hospital_animals[["hospital_case_id", "animal_id"]].drop_duplicates(), on="hospital_case_id")
    p = p.assign(source="pre_anaesthesia", source_id=p["id"],
                 value_raw=p["weight"].astype(str) + " " + p["weight_unit"].fillna(""),
                 unit_id=np.nan, recorded_at=p["anaesthesia_datetime"].fillna(p["created_at"]),
                 time_imputed=p["anaesthesia_datetime"].isna(), base_uom_value=np.nan,
                 record_date=pd.NaT, comments=None)
    return p[READING_COLUMNS]


def normalise(readings: pd.DataFrame, units: pd.DataFrame) -> pd.DataFrame:
    """Parse values and convert to grams; drop what can't be trusted (§1.2, B1 rows 1–2)."""
    df = readings.reset_index(drop=True).copy()
    df["flags"], df["drop_reason"] = "", None
    factors, by_string = unit_factors(units), global_unit_ids(units)

    parsed = df["value_raw"].map(parse_value)
    df["value_num"] = parsed.str[0]
    suffix_unit = parsed.str[1].map(by_string)
    # A typed suffix overrides the selected unit when they disagree
    use_suffix = suffix_unit.notna() & (suffix_unit != df["unit_id"])
    _add_flag(df, use_suffix & df["unit_id"].notna(), "unit_from_suffix")
    df["unit_id"] = suffix_unit.where(use_suffix, df["unit_id"])

    df = df.join(factors[["to_g", "unit_reason"]], on="unit_id")
    df["value_g"] = df["value_num"] * df["to_g"]

    _drop(df, df["value_num"].isna(), "value_not_numeric")
    _drop(df, df["unit_id"].isna() | ~df["unit_id"].isin(factors.index), "unit_missing")
    unit_bad = df["drop_reason"].isna() & df["unit_reason"].notna()
    df.loc[unit_bad, "drop_reason"] = df.loc[unit_bad, "unit_reason"]
    _drop(df, df["value_g"] <= 0, "value_not_positive")

    # Stored base_uom_value kept only as a cross-check (§1.2)
    base = df["base_uom_value"]
    _add_flag(df, base == 0, "base_zero")
    rel = (base - df["value_g"]).abs() / df["value_g"]
    _add_flag(df, (base > 0) & (rel > 0.01), "base_mismatch")
    return df.drop(columns=["to_g", "unit_reason"])


def row_rules(df: pd.DataFrame, animals: pd.DataFrame, now: pd.Timestamp, min_date: str) -> pd.DataFrame:
    """Impossible dates and groups (B1 rows 'impossible date', 'groups')."""
    df = df.merge(animals[["animal_id", "type", "taxonomy_id", "birth_date", "death_at"]], on="animal_id", how="left")
    _drop(df, df["type"].isna(), "animal_not_in_zoo")
    _drop(df, df["type"] == "group", "group_animal")
    t = pd.to_datetime(df["recorded_at"])
    _drop(df, t.isna(), "no_date")
    _drop(df, (t > now + pd.Timedelta(days=1)) | (t < pd.Timestamp(min_date)), "impossible_date")
    _drop(df, t < df["birth_date"] - pd.Timedelta(days=1), "before_birth")
    _drop(df, t > df["death_at"] + pd.Timedelta(days=1), "after_death")
    return df


def dedup(df: pd.DataFrame, near_minutes: float) -> pd.DataFrame:
    """Exact duplicates on (animal, recorded_at, value_g), then same weighing written twice within ±N minutes."""
    live = df["drop_reason"].isna()
    exact = df[live].duplicated(["animal_id", "recorded_at", "value_g"])
    _drop(df, exact.reindex(df.index, fill_value=False), "duplicate")

    live = df[df["drop_reason"].isna()].sort_values(["animal_id", "recorded_at"])
    prev_t = live.groupby("animal_id")["recorded_at"].shift()
    prev_v = live.groupby("animal_id")["value_g"].shift()
    near = ((live["recorded_at"] - prev_t) <= pd.Timedelta(minutes=near_minutes)) & \
           ((live["value_g"] - prev_v).abs() <= 0.01 * live["value_g"])
    _drop(df, near.reindex(df.index, fill_value=False), "near_duplicate")
    return df


# ── Per-animal series rules (B1) ─────────────────────────────────────────────────────────────────────────

def fix_unit_slips(df: pd.DataFrame, tolerance: float, window_days: int = 180, stable_share: float = 0.75,
                   stable_band: float = 0.3) -> pd.DataFrame:
    """kg↔g and kg↔lb slips: ratio to the median of the animal's *other* readings within ±window_days
    ≈ 1000 / 0.001 / 2.2046 / 0.4536.

    Only corrected against a stable reference (≥ stable_share of those readings within ±stable_band of their
    median): when the series itself is a mix of units, its median means nothing and a "fix" would invent data.
    """
    slips = [(1000, "corrected_unit_x1000"), (0.001, "corrected_unit_x0.001"),
             (2.20462, "corrected_unit_lb_as_kg"), (1 / 2.20462, "corrected_unit_kg_as_lb")]
    live = df[df["drop_reason"].isna()]
    window = pd.Timedelta(days=window_days)
    for animal_id, g in live.groupby("animal_id"):
        if len(g) < 3:
            continue
        times = pd.to_datetime(g["recorded_at"])
        for idx in g.index:
            near = (times - times[idx]).abs() <= window
            others = g.loc[near & (g.index != idx), "value_g"]
            if len(others) < 2:
                continue
            ref = others.median()
            if ((others / ref - 1).abs() <= stable_band).mean() < stable_share:
                continue
            ratio = df.at[idx, "value_g"] / ref
            for factor, flag in slips:
                if abs(ratio / factor - 1) <= tolerance:
                    df.at[idx, "value_g"] = df.at[idx, "value_g"] / factor
                    df.at[idx, "flags"] = (df.at[idx, "flags"] + ";" + flag).strip(";")
                    break
    return df


def species_band(df: pd.DataFrame, pct, factor, juvenile_days: int, min_readings: int, min_animals: int):
    """Typos outside the species band (1st–99th pct of adult readings in the zoo × 0.5–1.5) are dropped.

    Returns (df, band table). Species with too little adult data get no band (no rows dropped).
    """
    age_days = (pd.to_datetime(df["recorded_at"]) - df["birth_date"]).dt.days
    df["is_juvenile"] = age_days < juvenile_days
    adult = df[df["drop_reason"].isna() & ~df["is_juvenile"]]
    stats = adult.groupby("taxonomy_id").agg(
        n=("value_g", "size"), animals=("animal_id", "nunique"),
        lo=("value_g", lambda s: s.quantile(pct[0] / 100)), hi=("value_g", lambda s: s.quantile(pct[1] / 100)))
    bands = stats[(stats["n"] >= min_readings) & (stats["animals"] >= min_animals)].copy()
    bands["band_lo_g"], bands["band_hi_g"] = bands["lo"] * factor[0], bands["hi"] * factor[1]

    b = df[["taxonomy_id"]].join(bands[["band_lo_g", "band_hi_g"]], on="taxonomy_id")
    outside = (df["value_g"] < b["band_lo_g"]) | (df["value_g"] > b["band_hi_g"])
    _drop(df, outside & ~df["is_juvenile"], "outside_species_band")
    return df, bands.reset_index()


def hampel_confirm(df: pd.DataFrame, window: int, sigmas: float, mad_floor_pct: float) -> pd.DataFrame:
    """One-off spikes vs real changes (B1): Hampel candidate = > N robust SD from the median of up to `window`
    neighbours each side. Then:
      next reading back to normal → spike, dropped;  next/previous reading deviates the same way → `confirmed`;
      latest reading, nothing to compare → `pending_confirmation` (kept, weak signal).
    """
    df["confirmation"] = None
    live = df[df["drop_reason"].isna()].sort_values(["animal_id", "recorded_at"])
    for _, g in live.groupby("animal_id"):
        v, idx = g["value_g"].to_numpy(), g.index.to_numpy()
        n = len(v)
        if n < 4:
            continue
        cand, sign = np.zeros(n, bool), np.zeros(n)
        for i in range(n):
            nb = np.r_[v[max(0, i - window):i], v[i + 1:i + 1 + window]]
            med = np.median(nb)
            scale = max(1.4826 * np.median(np.abs(nb - med)), mad_floor_pct * med)
            if abs(v[i] - med) > sigmas * scale:
                cand[i], sign[i] = True, np.sign(v[i] - med)
        for i in np.flatnonzero(cand):
            same_prev = i > 0 and cand[i - 1] and sign[i - 1] == sign[i]
            same_next = i < n - 1 and cand[i + 1] and sign[i + 1] == sign[i]
            if same_prev or same_next:
                df.at[idx[i], "confirmation"] = "confirmed"
            elif i == n - 1:
                df.at[idx[i], "confirmation"] = "pending_confirmation"
            else:
                df.at[idx[i], "drop_reason"] = "spike"
    return df


def clean(readings: pd.DataFrame, units: pd.DataFrame, animals: pd.DataFrame, wcfg: dict, now: pd.Timestamp,
          tz: str):
    """Full weight pipeline. Returns (all rows with drop_reason/flags, species bands)."""
    from src.animals import local_day

    df = normalise(readings, units)
    df = row_rules(df, animals, now, wcfg["min_date"])
    df = dedup(df, wcfg["near_duplicate_minutes"])
    df = fix_unit_slips(df, wcfg["unit_slip_tolerance"])
    df, bands = species_band(df, wcfg["species_band_pct"], wcfg["species_band_factor"], wcfg["juvenile_days"],
                             wcfg["band_min_readings"], wcfg["band_min_animals"])
    df = hampel_confirm(df, wcfg["hampel_window"], wcfg["hampel_sigmas"], wcfg["hampel_mad_floor_pct"])
    df["local_date"] = local_day(df["recorded_at"], tz)
    df["kept"] = df["drop_reason"].isna()
    df["value_raw"] = df["value_raw"].astype("string")  # text from assessments, numbers from legacy
    return df.drop(columns=["birth_date", "death_at"]), bands
