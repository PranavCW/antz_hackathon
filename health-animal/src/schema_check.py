"""Stage schema_check: verify the live schema against the PLAN.md §1 dictionary and audit coded values.

Writes outputs/schema_check.md (column diff + DISTINCT audits) and outputs/schema/create_tables.sql
(raw SHOW CREATE TABLE for every table). Run before extract: migrations ≠ live schema.
"""

import pandas as pd
from sqlalchemy import text

from src import config, store
from src.db import get_engine

# Columns the pipeline relies on, per table (PLAN.md §1). Extra live columns are fine; missing ones are not.
EXPECTED = {
    "antz_animals": "animal_id taxonomy_id type total_animal sex birth_date approximate_dob enclosure_id zoo_id "
                    "is_alive active is_deleted",
    "antz_taxonomic_units": "tsn default_common_name complete_name zoo_id",
    "antz_animal_assessments": "id animal_id assessment_type_id assessment_unit_id assessment_value base_uom_value "
                               "base_uom_name record_date record_time recorded_date_time created_at modified_at "
                               "medical_record_id hospital_case_id comments is_deleted",
    "antz_animal_assessment_values": "assessment_id assessment_value assessment_unit_id base_uom_value base_uom_name "
                                     "is_deleted",
    "antz_animal_measurement": "id animal_id measurement_type_id measurement_unit_id measurement_value record_date "
                               "record_time created_at",
    "antz_master_assessments_types": "assessment_type_id label string_id zoo_id measurement_type response_type active",
    "antz_master_measurement_units": "id unit_name string_id zoo_id uom_abbr measurement_type base_uom_id "
                                     "conversion_factor conversion_offset",
    "antz_hospital_pre_anaesthesia": "anaesthesia_id weight weight_unit weight_type created_at",
    "antz_hospital_anaesthesia": "id hospital_case_id medical_record_id anaesthesia_datetime zoo_id is_deleted",
    "antz_diet_entity_assign": "zoo_id diet_id entity_id entity_type start_date end_date is_active is_delete",
    "antz_diet_assign_logs": "zoo_id old_diet_id new_diet_id entity_type entity_id created_at",
    "antz_food_wastage": "zoo_id enclosure_id wastage_quantity wastage_quantity_uom base_wastage_quantity "
                         "base_wastage_quantity_uom wastage_date wastage_time notes active is_deleted",
    "antz_observation": "observation_id observation_name priority zoo_id created_by created_at created_for "
                        "modified_at active is_deleted",
    "antz_observation_mapping": "observation_id ref_type ref_id enclosure_id species_id is_deleted",
    "antz_observation_type_mapping": "observation_id parent_observation_type_id child_observation_type_id deleted_on",
    "antz_observation_type_master": "id parent_id type_name string_id zoo_id",
    "antz_observation_note": "observation_id observation created_at active",
    "antz_animal_incident": "animal_id incident_id zoo_id is_deleted",
    "antz_animal_incident_details": "incident_id incident_type incident_date notes is_deleted",
    "antz_medical_record": "id animal_id zoo_id case_type medical_record_type created_for created_at "
                           "system_generated_record_date parent_medical_id group_entry is_deleted",
    "antz_master_medical_case_type": "id label string_id",
    "antz_medical_record_animal_mapping": "medical_record_id animal_id added_date detach_date active total_animal "
                                          "is_hospitalized",
    "antz_medical_complaints": "complaint medical_record_id animal_id severity duration duration_unit status "
                               "recorded_date_time created_at is_deleted",
    "antz_master_animal_complaints": "id label category_id",
    "antz_master_medical_category": "med_cat_id label string_id",
    "antz_complaints_notes": "complaint medical_record_id status date time notes_dump",
    "antz_medical_diagnosis": "diagnosis_type medical_record_id animal_id severity status closed_at is_cronical "
                              "confidence_level clinical_assessment prognosis recorded_date_time created_at is_deleted",
    "antz_master_diagnosis": "id label category_id is_contagious",
    "antz_diagnosis_notes": "diagnosis medical_record_id status date time",
    "antz_medical_prescriptions": "medical_record_id animal_id start_date stop_date end_date is_deleted",
    "antz_medical_tests": "medical_record_id animal_id status is_deleted",
    "antz_hospital_case_master": "id visit_type treatment_type admitted_at opd_to_inpatient_date discharge_at status",
    "antz_hospital_medical_animal_mapping": "hospital_case_id medical_record_id animal_id zoo_id health_status",
    "antz_hospital_log": "hospital_case_id action old_data new_data action_at",
    "antz_hospital_animal_discharge": "hospital_case_id animal_id discharge_type discharge_date",
    "antz_animals_mortality": "entity_id entity_type total_animal date_of_death discovered_date is_estimate "
                              "manner_of_death history_of_illness is_active status",
    "antz_master_disposition_manner_of_death": "id name string_id zoo_id",
    "antz_animal_necropsy": "animal_id death_date death_time suspected_cause_of_death confirmed_cause_of_death "
                            "is_system_generated is_active",
    "antz_enclosure_logs": "zoo_id enclosure_id animal_id in_date out_date is_deleted",
    "antz_movement_animal_mapping": "entity_type entity_id from_enclosure destination_id transferred_on",
    "antz_zoo_enclosure": "enclosure_id zoo_id user_enclosure_name is_deleted",
}

ZOO_ANIMALS = "SELECT animal_id FROM antz_animals WHERE zoo_id = :z"

# DISTINCT audits on coded values the pipeline maps by string (§1 note, §3).
AUDITS = {
    "Weight assessment types (global + zoo)":
        "SELECT t.assessment_type_id, t.label, t.string_id, t.zoo_id, t.response_type, t.active, COUNT(s.id) AS n_zoo "
        "FROM antz_master_assessments_types t LEFT JOIN antz_animal_assessments s "
        f"  ON s.assessment_type_id = t.assessment_type_id AND s.animal_id IN ({ZOO_ANIMALS}) "
        "WHERE t.measurement_type = 'weight' AND t.zoo_id IN (0, :z) GROUP BY 1,2,3,4,5,6 ORDER BY n_zoo DESC",
    "Units used on weight-type assessments":
        "SELECT s.assessment_unit_id, u.unit_name, u.measurement_type, u.base_uom_id, u.conversion_factor, "
        "  COUNT(*) AS n, SUM(s.base_uom_value IS NULL) AS base_null, SUM(s.base_uom_value = 0) AS base_zero "
        "FROM antz_animal_assessments s JOIN antz_master_assessments_types t ON t.assessment_type_id = s.assessment_type_id "
        "LEFT JOIN antz_master_measurement_units u ON u.id = s.assessment_unit_id "
        f"WHERE s.animal_id IN ({ZOO_ANIMALS}) AND t.measurement_type = 'weight' GROUP BY 1,2,3,4,5 ORDER BY n DESC",
    "Weight units master (zoo-custom units, base chain)":
        "SELECT id, unit_name, zoo_id, uom_abbr, measurement_type, base_uom_id, conversion_factor, conversion_offset "
        "FROM antz_master_measurement_units WHERE measurement_type = 'weight' AND (zoo_id IS NULL OR zoo_id IN (0, :z))",
    "Legacy measurement types/units (zoo)":
        "SELECT m.measurement_type_id, m.measurement_unit_id, COUNT(*) AS n, MIN(m.record_date) AS first, "
        "  MAX(m.record_date) AS last FROM antz_animal_measurement m "
        f"WHERE m.animal_id IN ({ZOO_ANIMALS}) GROUP BY 1,2 ORDER BY n DESC",
    "Weight record_date range (assessments)":
        "SELECT SUM(record_date IS NULL) AS null_date, SUM(record_date < '2000-01-01') AS before_2000, "
        "  SUM(record_date > CURDATE()) AS future, MIN(record_date) AS min_date, MAX(record_date) AS max_date "
        f"FROM antz_animal_assessments WHERE animal_id IN ({ZOO_ANIMALS})",
    "Complaint severity":
        "SELECT severity, COUNT(*) AS n FROM antz_medical_complaints WHERE zoo_id = :z GROUP BY 1 ORDER BY n DESC",
    "Complaint duration_unit":
        "SELECT duration_unit, COUNT(*) AS n FROM antz_medical_complaints WHERE zoo_id = :z GROUP BY 1 ORDER BY n DESC",
    "Diagnosis severity":
        "SELECT severity, COUNT(*) AS n FROM antz_medical_diagnosis WHERE zoo_id = :z GROUP BY 1 ORDER BY n DESC",
    "Diagnosis prognosis":
        "SELECT prognosis, COUNT(*) AS n FROM antz_medical_diagnosis WHERE zoo_id = :z GROUP BY 1 ORDER BY n DESC",
    "Medical record case_type × created_for":
        "SELECT r.case_type, ct.string_id AS case_type_string_id, r.created_for, COUNT(*) AS n "
        "FROM antz_medical_record r LEFT JOIN antz_master_medical_case_type ct ON ct.id = r.case_type "
        "WHERE r.zoo_id = :z AND r.is_deleted = 0 GROUP BY 1,2,3 ORDER BY n DESC",
    "Hospital visit_type / treatment_type / status":
        "SELECT c.visit_type, c.treatment_type, c.status, COUNT(*) AS n FROM antz_hospital_case_master c "
        "WHERE c.id IN (SELECT hospital_case_id FROM antz_hospital_medical_animal_mapping WHERE zoo_id = :z) "
        "GROUP BY 1,2,3 ORDER BY n DESC",
    "Hospital health_status":
        "SELECT health_status, COUNT(*) AS n FROM antz_hospital_medical_animal_mapping WHERE zoo_id = :z "
        "GROUP BY 1 ORDER BY n DESC",
    "Manner of death (string_id) on approved active mortality":
        "SELECT md.id, md.name, md.string_id, md.zoo_id, md.active, COUNT(m._id) AS n_zoo "
        "FROM antz_master_disposition_manner_of_death md LEFT JOIN antz_animals_mortality m "
        "  ON m.manner_of_death = md.id AND m.is_active = 1 AND m.status = 'APPROVED' "
        f"  AND m.entity_type = 'animal' AND m.entity_id IN ({ZOO_ANIMALS}) "
        "WHERE md.zoo_id IN (0, :z) GROUP BY 1,2,3,4,5 ORDER BY n_zoo DESC",
    "Mortality status × is_active":
        "SELECT status, is_active, entity_type, COUNT(*) AS n FROM antz_animals_mortality "
        f"WHERE entity_id IN ({ZOO_ANIMALS}) GROUP BY 1,2,3 ORDER BY n DESC",
    "Observation priority":
        "SELECT priority, COUNT(*) AS n FROM antz_observation WHERE zoo_id = :z GROUP BY 1 ORDER BY n DESC",
    "Observation mapping ref_type × is_deleted":
        "SELECT m.ref_type, m.is_deleted, COUNT(*) AS n FROM antz_observation_mapping m "
        "WHERE m.observation_id IN (SELECT observation_id FROM antz_observation WHERE zoo_id = :z) "
        "GROUP BY 1,2 ORDER BY n DESC",
    "Observation parent types in use":
        "SELECT p.id, p.type_name, p.string_id, p.zoo_id, COUNT(tm.id) AS n_notes "
        "FROM antz_observation_type_mapping tm JOIN antz_observation_type_master p ON p.id = tm.parent_observation_type_id "
        "WHERE tm.deleted_on IS NULL AND tm.observation_id IN (SELECT observation_id FROM antz_observation WHERE zoo_id = :z) "
        "GROUP BY 1,2,3,4 ORDER BY n_notes DESC",
    "Observation text duplication":
        "SELECT COUNT(*) AS notes, COUNT(DISTINCT observation_name) AS distinct_texts, "
        "  SUM(observation_name IS NULL OR TRIM(observation_name) = '') AS empty_text, "
        "  SUM(LOWER(TRIM(observation_name)) LIKE 'test%') AS starts_with_test "
        "FROM antz_observation WHERE zoo_id = :z AND is_deleted = 0",
    "Food wastage units":
        "SELECT wastage_quantity_uom, base_wastage_quantity_uom, COUNT(*) AS n, MIN(wastage_date) AS first, "
        "  MAX(wastage_date) AS last FROM antz_food_wastage WHERE zoo_id = :z GROUP BY 1,2 ORDER BY n DESC",
    "Enclosure log stays":
        "SELECT COUNT(*) AS rows_, SUM(is_deleted) AS deleted, SUM(out_date IS NULL) AS open_stays, "
        "  SUM(out_date < in_date) AS out_before_in, COUNT(DISTINCT enclosure_id) AS enclosures, "
        "  MIN(in_date) AS first_in FROM antz_enclosure_logs WHERE zoo_id = :z",
    "Taxonomy rows per tsn (zoo-specific names duplicate tsn)":
        "SELECT n_rows, COUNT(*) AS tsns FROM (SELECT tsn, COUNT(*) AS n_rows FROM antz_taxonomic_units "
        f"  WHERE tsn IN (SELECT taxonomy_id FROM antz_animals WHERE zoo_id = :z) GROUP BY tsn) x GROUP BY 1",
}


def _md_table(df: pd.DataFrame, max_rows: int = 60) -> str:
    if df.empty:
        return "_no rows_"
    shown = df.head(max_rows).astype(str).replace({"None": "NULL", "nan": "NULL", "NaT": "NULL"})
    lines = ["| " + " | ".join(shown.columns) + " |", "|" + "---|" * len(shown.columns)]
    lines += ["| " + " | ".join(v.replace("|", "\\|").replace("\n", " ")[:80] for v in row) + " |"
              for row in shown.itertuples(index=False)]
    if len(df) > max_rows:
        lines.append(f"\n_… {len(df) - max_rows} more rows_")
    return "\n".join(lines)


def run() -> None:
    """Diff live columns vs EXPECTED, dump SHOW CREATE TABLE, run DISTINCT audits; write the report."""
    zoo_id = config.load()["zoo_id"]
    if zoo_id is None:
        raise RuntimeError("Set ZOO_ID in .env")

    with get_engine() as engine, engine.connect() as conn:
        # 1. Column diff against the dictionary
        live = pd.read_sql(text(
            "SELECT TABLE_NAME AS t, COLUMN_NAME AS c, COLUMN_TYPE AS type FROM information_schema.COLUMNS "
            "WHERE TABLE_SCHEMA = DATABASE()"), conn)
        live_cols = live.groupby("t")["c"].apply(set).to_dict()
        diff_rows, create_sql = [], []
        for table, cols in EXPECTED.items():
            have = live_cols.get(table)
            missing = sorted(set(cols.split()) - have) if have else ["(table missing)"]
            diff_rows.append({"table": table, "status": "ok" if not missing else "MISMATCH",
                              "missing_columns": ", ".join(missing)})
            if have:
                ddl = conn.execute(text(f"SHOW CREATE TABLE `{table}`")).one()[1]
                create_sql.append(f"-- {table}\n{ddl};\n")
        diff = pd.DataFrame(diff_rows)

        # 2. DISTINCT audits
        audits = {}
        for title, sql in AUDITS.items():
            audits[title] = pd.read_sql(text(sql), conn, params={"z": zoo_id})

    # 3. Write outputs
    schema_dir = store.output_path("schema")
    schema_dir.mkdir(exist_ok=True)
    (schema_dir / "create_tables.sql").write_text("\n".join(create_sql))

    n_bad = int((diff["status"] != "ok").sum())
    parts = [f"# Schema check — zoo {zoo_id}\n",
             f"**{len(diff) - n_bad}/{len(diff)} tables match the dictionary.** Raw DDL: `outputs/schema/create_tables.sql`.\n",
             "## Column diff vs PLAN.md §1\n", _md_table(diff, max_rows=200), "\n## Value audits\n"]
    for title, df in audits.items():
        parts += [f"### {title}\n", _md_table(df), ""]
    store.output_path("schema_check.md").write_text("\n".join(parts))

    print(f"Schema: {len(diff) - n_bad}/{len(diff)} tables match.")
    for row in diff[diff["status"] != "ok"].itertuples():
        print(f"  ✗ {row.table}: {row.missing_columns}")
    print(f"{len(audits)} value audits → {store.output_path('schema_check.md')}")
