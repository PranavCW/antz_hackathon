"""Stage extract: pull the PLAN.md §1 source tables for ZOO_ID into data/raw/ as parquet (one file per source).

Raw = as stored, scoped to the zoo, soft-deleted rows included (clean decides). Only type fixes are applied so
parquet can hold the data: MySQL zero/invalid dates become NaT (counted in the log), DECIMAL → float, JSON → str.
"""

import datetime as dt
import decimal
import time

import pandas as pd
from sqlalchemy import text

from src import config, store
from src.db import get_engine

ANIMALS = "SELECT animal_id FROM antz_animals WHERE zoo_id = :z"
OBS = "SELECT observation_id FROM antz_observation WHERE zoo_id = :z"
RECORDS = "SELECT id FROM antz_medical_record WHERE zoo_id = :z"
HOSP_CASES = "SELECT hospital_case_id FROM antz_hospital_medical_animal_mapping WHERE zoo_id = :z"

# name → SQL. All scoped to :z, either directly or through the zoo's animals / notes / records.
SOURCES = {
    # §1.1 animal master
    "animals": "SELECT * FROM antz_animals WHERE zoo_id = :z",
    "taxonomy": "SELECT tsn, parent_tsn, rank_id, complete_name, default_common_name, zoo_id FROM antz_taxonomic_units "
                "WHERE tsn IN (SELECT taxonomy_id FROM antz_animals WHERE zoo_id = :z) AND zoo_id IN (0, :z)",
    "enclosures": "SELECT * FROM antz_zoo_enclosure WHERE zoo_id = :z",
    # §1.2 weight
    "assessment_types": "SELECT * FROM antz_master_assessments_types WHERE zoo_id IN (0, :z)",
    "measurement_units": "SELECT * FROM antz_master_measurement_units",
    "assessments": f"SELECT * FROM antz_animal_assessments WHERE animal_id IN ({ANIMALS})",
    "assessment_values": "SELECT * FROM antz_animal_assessment_values WHERE assessment_id IN "
                         f"(SELECT id FROM antz_animal_assessments WHERE animal_id IN ({ANIMALS}))",
    "measurements_legacy": f"SELECT * FROM antz_animal_measurement WHERE animal_id IN ({ANIMALS})",
    "pre_anaesthesia": "SELECT p.*, a.hospital_case_id, a.medical_record_id, a.anaesthesia_datetime, "
                       "  a.is_deleted AS anaesthesia_is_deleted "
                       "FROM antz_hospital_pre_anaesthesia p JOIN antz_hospital_anaesthesia a ON a.id = p.anaesthesia_id "
                       "WHERE a.zoo_id = :z",
    # §1.3 feeding
    "food_wastage": "SELECT * FROM antz_food_wastage WHERE zoo_id = :z",
    "diet_entity_assign": "SELECT * FROM antz_diet_entity_assign WHERE zoo_id = :z",
    "diet_assign_logs": "SELECT id, zoo_id, old_diet_id, new_diet_id, entity_type, entity_id, created_by, created_at "
                        "FROM antz_diet_assign_logs WHERE zoo_id = :z",
    # §1.4 keeper notes
    "observations": "SELECT * FROM antz_observation WHERE zoo_id = :z",
    "observation_mapping": f"SELECT * FROM antz_observation_mapping WHERE observation_id IN ({OBS})",
    "observation_type_mapping": f"SELECT * FROM antz_observation_type_mapping WHERE observation_id IN ({OBS})",
    "observation_types": "SELECT * FROM antz_observation_type_master WHERE zoo_id IN (0, :z)",
    "observation_comments": f"SELECT * FROM antz_observation_note WHERE observation_id IN ({OBS})",
    "incidents": "SELECT i.animal_id, i.incident_code, i.is_deleted AS incident_is_deleted, d.* "
                 "FROM antz_animal_incident i JOIN antz_animal_incident_details d ON d.incident_id = i.incident_id "
                 "WHERE i.zoo_id = :z",
    # §1.5 labels (never model inputs)
    "medical_records": "SELECT * FROM antz_medical_record WHERE zoo_id = :z",
    "medical_case_types": "SELECT * FROM antz_master_medical_case_type",
    "medical_record_animals": f"SELECT * FROM antz_medical_record_animal_mapping WHERE medical_record_id IN ({RECORDS})",
    "complaints": "SELECT * FROM antz_medical_complaints WHERE zoo_id = :z",
    "complaint_master": "SELECT * FROM antz_master_animal_complaints",
    "medical_categories": "SELECT * FROM antz_master_medical_category",
    "complaint_notes": "SELECT id, complaint, medical_record_id, hospital_case_id, type, status, is_system_generated, "
                       f"  time, date, is_deleted, created_at FROM antz_complaints_notes WHERE medical_record_id IN ({RECORDS})",
    "diagnoses": "SELECT * FROM antz_medical_diagnosis WHERE zoo_id = :z",
    "diagnosis_master": "SELECT * FROM antz_master_diagnosis",
    "diagnosis_notes": "SELECT id, diagnosis, medical_record_id, hospital_case_id, type, status, is_system_generated, "
                       f"  time, date, is_deleted, created_at FROM antz_diagnosis_notes WHERE medical_record_id IN ({RECORDS})",
    "prescriptions": "SELECT id, medical_record_id, created_for, animal_id, zoo_id, medicine_name, start_date, stop_date, "
                     "  end_date, is_deleted, created_at FROM antz_medical_prescriptions WHERE zoo_id = :z",
    "medical_tests": "SELECT id, medical_record_id, animal_id, test_name, status, is_deleted, created_at "
                     f"FROM antz_medical_tests WHERE medical_record_id IN ({RECORDS})",
    "hospital_cases": f"SELECT * FROM antz_hospital_case_master WHERE id IN ({HOSP_CASES})",
    "hospital_animals": "SELECT * FROM antz_hospital_medical_animal_mapping WHERE zoo_id = :z",
    "hospital_log": "SELECT * FROM antz_hospital_log WHERE zoo_id = :z",
    "hospital_discharges": f"SELECT * FROM antz_hospital_animal_discharge WHERE hospital_case_id IN ({HOSP_CASES})",
    "mortality": f"SELECT * FROM antz_animals_mortality WHERE entity_type = 'animal' AND entity_id IN ({ANIMALS})",
    "manner_of_death": "SELECT * FROM antz_master_disposition_manner_of_death",
    "necropsy": f"SELECT * FROM antz_animal_necropsy WHERE animal_id IN ({ANIMALS})",
    # §1.6 housing
    "enclosure_logs": "SELECT * FROM antz_enclosure_logs WHERE zoo_id = :z",
    "movements": "SELECT * FROM antz_movement_animal_mapping WHERE entity_type = 'animal' "
                 f"AND entity_id IN ({ANIMALS})",
}


def to_parquet_safe(df: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    """Make object columns parquet-typed; return (df, {column: invalid dates turned into NaT})."""
    invalid = {}
    for col in df.columns[df.dtypes == object]:
        values = df[col].dropna()
        if values.empty:
            continue
        kinds = {type(v) for v in values}
        if kinds & {dt.date, dt.datetime, pd.Timestamp}:
            # pymysql hands back MySQL zero dates ('0000-00-00') as strings next to real dates
            parsed = pd.to_datetime(df[col].where(df[col].map(lambda v: not isinstance(v, str))), errors="coerce")
            n_bad = int(df[col].notna().sum() - parsed.notna().sum())
            if n_bad:
                invalid[col] = n_bad
            df[col] = parsed
        elif kinds <= {dt.timedelta, pd.Timedelta}:
            df[col] = pd.to_timedelta(df[col])
        elif kinds <= {decimal.Decimal}:
            df[col] = df[col].astype(float)
        elif kinds - {str}:
            # mixed types or JSON/bytes columns: keep as text
            df[col] = df[col].map(lambda v: v if v is None or isinstance(v, str) else
                                  v.decode("utf-8", "replace") if isinstance(v, (bytes, bytearray)) else str(v))
    return df, invalid


def run() -> None:
    """Extract every SOURCES entry for ZOO_ID; write parquet + outputs/extract_log.csv with row counts."""
    zoo_id = config.load()["zoo_id"]
    if zoo_id is None:
        raise RuntimeError("Set ZOO_ID in .env")

    log = []
    extracted_at = pd.Timestamp.now(tz="UTC").tz_localize(None)
    with get_engine() as engine, engine.connect() as conn:
        for name, sql in SOURCES.items():
            start = time.time()
            df = pd.read_sql(text(sql), conn, params={"z": zoo_id})
            df, invalid_dates = to_parquet_safe(df)
            store.write("raw", name, df)
            log.append({"source": name, "rows": len(df),
                        "invalid_dates_to_nat": "; ".join(f"{c}={n}" for c, n in invalid_dates.items()),
                        "seconds": round(time.time() - start, 1)})
            print(f"  {name:<26} {len(df):>8,} rows" + (f"  (invalid dates → NaT: {log[-1]['invalid_dates_to_nat']})" if invalid_dates else ""))

    log_df = pd.DataFrame(log).assign(zoo_id=zoo_id, extracted_at_utc=extracted_at)
    log_df.to_csv(store.output_path("extract_log.csv"), index=False)
    print(f"{len(log)} sources, {log_df['rows'].sum():,} rows → data/raw/ (log: outputs/extract_log.csv)")
