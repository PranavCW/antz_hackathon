"""Stage db_check: confirm we can reach the DB, see what we're connected to, and that key tables exist."""

from sqlalchemy import text

from src.db import env, get_engine

# Core tables from PLAN.md §1 — just existence + row counts here; full schema diff is schema_check.
KEY_TABLES = [
    "antz_animals",
    "antz_animal_assessments",
    "antz_animal_measurement",
    "antz_food_wastage",
    "antz_observation",
    "antz_observation_mapping",
    "antz_medical_record",
    "antz_hospital_case_master",
    "antz_animals_mortality",
    "antz_enclosure_logs",
]


def run() -> None:
    """Print a connection sanity report: server/user identity, write privileges, key tables, zoo animal count."""
    with get_engine() as engine, engine.connect() as conn:
        # 1. Identity: which server, schema and account we actually landed on
        version, db, user = conn.execute(text("SELECT VERSION(), DATABASE(), CURRENT_USER()")).one()
        print(f"Connected: MySQL {version}, database '{db}' as {user}")

        # 2. Privileges: flag any grant that allows writes (substring match on the GRANT statements)
        grants = [row[0] for row in conn.execute(text("SHOW GRANTS"))]
        writable = [g for g in grants if any(p in g.upper() for p in ("ALL PRIVILEGES", "INSERT", "UPDATE", "DELETE"))]
        if writable:
            print("⚠ This user can write. Sessions are forced read-only, but a read-only user is safer.")
        else:
            print("User is read-only ✓")

        # 3. Tables: confirm each KEY_TABLES entry exists and show its approximate size
        existing = {row[0] for row in conn.execute(text("SHOW TABLES"))}
        print(f"\n{len(existing)} tables in schema. Key tables:")
        for table in KEY_TABLES:
            if table not in existing:
                print(f"  ✗ {table:<30} MISSING")
                continue
            # Estimated counts from information_schema: instant, fine for a sanity check
            n = conn.execute(
                text("SELECT TABLE_ROWS FROM information_schema.TABLES WHERE TABLE_SCHEMA = DATABASE() AND TABLE_NAME = :t"),
                {"t": table},
            ).scalar()
            print(f"  ✓ {table:<30} ~{n:,} rows")

        # 4. Scope: if ZOO_ID is set, exact count of that zoo's live (non-soft-deleted) animals
        zoo_id = env("ZOO_ID")
        if zoo_id and "antz_animals" in existing:
            n_animals = conn.execute(
                text("SELECT COUNT(*) FROM antz_animals WHERE zoo_id = :z AND is_deleted = 0"), {"z": zoo_id}
            ).scalar()
            print(f"\nZOO_ID={zoo_id}: {n_animals:,} non-deleted animals")
