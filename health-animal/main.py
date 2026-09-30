"""Entry point for the Antz Early Health Warning prototype.

Stages are placeholders for now; see docs/PLAN.md for what each one will do.
"""

import argparse

STAGES = [
    ("db_check", "Test DB connection from .env; list key tables and row counts"),
    ("schema_check", "Verify dump schema against the data dictionary"),
    ("extract", "Pull source tables from the DB dump into data/raw"),
    ("clean", "Normalise units, apply cleaning rules, data-quality report"),
    ("profile", "Per-species data density; choose 3-5 species"),
    ("cases", "Build illness episodes and vet-awareness date (T0)"),
    ("detectors", "Weight and leftover baselines, z-scores, CUSUM"),
    ("notes_llm", "Decode keeper notes with a self-hosted open-source LLM"),
    ("features", "Animal-day feature table (past data only)"),
    ("model", "Rule baseline, logistic regression, LightGBM"),
    ("backtest", "Replay history, lead time, alerts per week"),
]


def main() -> None:
    parser = argparse.ArgumentParser(description="Antz Early Health Warning")
    parser.add_argument("stage", nargs="?", choices=[name for name, _ in STAGES],
                        help="pipeline stage to run (omit to list stages)")
    args = parser.parse_args()

    if args.stage is None:
        print("Pipeline stages:")
        for name, desc in STAGES:
            print(f"  {name:<13} {desc}")
        return

    if args.stage == "db_check":
        from src import db_check
        db_check.run()
        return

    print(f"Stage '{args.stage}' is not implemented yet.")


if __name__ == "__main__":
    main()
