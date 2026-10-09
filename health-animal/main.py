"""Entry point for the Antz Early Health Warning prototype.

Each stage is a module in src/ with a run() function; see docs/PLAN.md for what each one does.
"""

import argparse
import importlib
import inspect

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
    ("notes_gold", "Write a 50-note hand-labelling sheet for the LLM audit"),
    ("notes_audit", "Audit LLM and lexicon against the hand-labelled gold notes"),
    ("features", "Animal-day feature table (past data only)"),
    ("model", "Rule baseline, logistic regression, LightGBM"),
    ("backtest", "Replay history, lead time, alerts per week"),
]


def main() -> None:
    parser = argparse.ArgumentParser(description="Antz Early Health Warning")
    parser.add_argument("stage", nargs="?", choices=[name for name, _ in STAGES],
                        help="pipeline stage to run (omit to list stages)")
    parser.add_argument("--limit", type=int, help="notes_llm: decode only the first N notes")
    parser.add_argument("--sample", action="store_true", help="notes_llm: decode the built-in 20-note check set")
    args = parser.parse_args()

    if args.stage is None:
        print("Pipeline stages:")
        for name, desc in STAGES:
            print(f"  {name:<13} {desc}")
        return

    run = importlib.import_module(f"src.{args.stage}").run
    options = {"limit": args.limit, "sample": args.sample or None}
    accepted = inspect.signature(run).parameters
    unsupported = [k for k, v in options.items() if v is not None and k not in accepted]
    if unsupported:
        parser.error(f"stage {args.stage} does not take --{', --'.join(unsupported)}")
    run(**{k: v for k, v in options.items() if v is not None and k in accepted})


if __name__ == "__main__":
    main()
