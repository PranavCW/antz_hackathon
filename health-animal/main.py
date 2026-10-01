"""Entry point for the Antz Early Health Warning prototype.

Each stage is a module in src/ with a run() function; see docs/PLAN.md for what each one does.
"""

import argparse
import importlib

STAGES = [
    ("db_check", "Test DB connection from .env; list key tables and row counts"),
    ("llm_check", "Test Ollama models and schema-forced JSON on a sample note"),
    ("schema_check", "Verify live schema against the data dictionary"),
    ("extract", "Pull source tables for ZOO_ID into data/raw"),
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

    importlib.import_module(f"src.{args.stage}").run()


if __name__ == "__main__":
    main()
