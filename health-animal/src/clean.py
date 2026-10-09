"""Stage clean: weights (§1.2, B1–B2), housing (§1.6), intake (§1.3, B3) and notes (§1.4, B4) → data/processed/.

Every dropped or corrected row keeps a reason; counts per source × reason go to outputs/data_quality_rows.csv,
which the profile stage turns into the data-quality report (B6).
"""

import pandas as pd

from src import clean_notes, clean_weights, config, housing, intake, store
from src.animals import animal_table


def _reason_counts(df: pd.DataFrame, source: str) -> pd.DataFrame:
    """Rows per drop_reason plus rows per flag (flags are ';'-joined) for one source."""
    reasons = df["drop_reason"].fillna("kept").value_counts().rename_axis("reason").reset_index(name="rows")
    out = [reasons.assign(kind="drop_reason")]
    if "flags" in df:
        flags = df["flags"].fillna("").str.split(";").explode()
        flags = flags[flags.ne("")].value_counts().rename_axis("reason").reset_index(name="rows")
        out.append(flags.assign(kind="flag"))
    return pd.concat(out).assign(source=source, total=len(df))


def run() -> None:
    cfg = config.load()
    tz, wcfg = cfg["timezone"], cfg["weight"]
    raw = lambda name: store.read("raw", name)  # noqa: E731
    now = pd.to_datetime(pd.read_csv(store.output_path("extract_log.csv"))["extracted_at_utc"].iloc[0])

    animals = animal_table(raw("animals"), raw("taxonomy"), raw("mortality"))
    units = raw("measurement_units")

    # 1. Weights
    types = raw("assessment_types")
    weight_types = set(types.loc[(types["string_id"] == "weight") & (types["zoo_id"] == 0), "assessment_type_id"])
    weight_types |= set(wcfg["extra_assessment_type_ids"])
    assessments = raw("assessments")
    a_rows = clean_weights.assessment_readings(assessments, raw("assessment_values"), weight_types)
    readings = pd.concat([
        a_rows,
        clean_weights.legacy_readings(raw("measurements_legacy"), a_rows),
        clean_weights.anaesthesia_readings(raw("pre_anaesthesia"), raw("hospital_animals")),
    ], ignore_index=True)
    weights, bands = clean_weights.clean(readings, units, animals, wcfg, now, tz)
    store.write("processed", "weights_clean", weights)
    store.write("processed", "species_weight_bands", bands)

    # 2. Housing
    stays = housing.build_stays(raw("enclosure_logs"), animals, now)
    enclosure_day = housing.enclosure_days(stays, tz)
    moves = housing.movement_check(stays, raw("movements"))
    store.write("processed", "stays", stays)
    store.write("processed", "enclosure_day", enclosure_day)

    # 3. Intake
    wastage = intake.clean_wastage(raw("food_wastage"), units, tz)
    changes = intake.diet_change_dates(raw("diet_assign_logs"), animals, tz)
    intake_daily, coverage, wastage = intake.daily_intake(wastage, enclosure_day, changes,
                                                          cfg["intake"]["min_logging_rate"], tz)
    store.write("processed", "wastage_clean", wastage)
    store.write("processed", "intake_daily", intake_daily)
    store.write("processed", "intake_coverage", coverage)

    # 4. Notes
    ncfg = cfg["notes"]
    obs = clean_notes.observation_rows(
        raw("observations"), raw("observation_mapping"), raw("observation_comments"), raw("observation_type_mapping"),
        raw("observation_types"), enclosure_day, animals, tz, set(ncfg["skip_note_types"]))
    other = clean_notes.other_rows(wastage, assessments, raw("incidents"), animals, tz)
    notes = clean_notes.finalise(pd.concat([obs, other], ignore_index=True), set(ncfg["template_texts"]),
                                 ncfg["min_chars"], ncfg["duplicate_window_days"])
    store.write("processed", "notes_clean", notes)

    # 5. Row-level data-quality counts
    dq = pd.concat([_reason_counts(weights, "weights"), _reason_counts(stays.assign(drop_reason=None), "stays"),
                    _reason_counts(wastage, "food_wastage"), _reason_counts(notes, "notes")])
    dq.to_csv(store.output_path("data_quality_rows.csv"), index=False)

    solo_days = enclosure_day["animal_id_if_solo"].notna().sum()
    print(f"Weights:  {weights['kept'].sum():,}/{len(weights):,} readings kept, "
          f"{weights['flags'].str.contains('corrected_unit').sum()} unit slips corrected, "
          f"{len(bands)} species bands, {weights['animal_id'][weights['kept']].nunique()} animals")
    print(f"Housing:  {len(stays):,} stays ({(stays['confidence'] == 'low').sum():,} low confidence), "
          f"{len(enclosure_day):,} enclosure-days, {solo_days:,} single-occupant; "
          f"movements matched to a stay: {moves['matched']}/{moves['movements']}")
    print(f"Intake:   {wastage['drop_reason'].isna().sum()}/{len(wastage)} wastage rows usable → "
          f"{len(intake_daily)} animal-days for {intake_daily['animal_id'].nunique()} animals")
    print(f"Notes:    {notes['kept'].sum():,}/{len(notes):,} note rows ready for the LLM "
          f"({notes.loc[notes['kept'], 'animal_id'].nunique()} animals)")
    print("Drop reasons → outputs/data_quality_rows.csv")
