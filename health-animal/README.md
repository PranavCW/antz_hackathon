# Health Animal — Early Health Warning (Antz Hackathon)

Offline prototype that compares each zoo animal against its own baseline (weight, feeding leftovers,
keeper notes decoded by a self-hosted open-source LLM) and raises explainable early-warning alerts,
validated by backtesting on historical data. Full design: [docs/PLAN.md](docs/PLAN.md).

## Setup
```bash
cp .env.example .env      # then fill in DB credentials, ZOO_ID, timezone, LLM settings
```

## Run
```bash
python main.py                 # list pipeline stages
python main.py db_check        # connection sanity
python main.py schema_check    # live schema vs PLAN.md §1 → outputs/schema_check.md
python main.py extract         # zoo tables → data/raw/*.parquet (only stage besides the checks that reads the DB)
python main.py clean           # weights, housing, intake, notes → data/processed/
python main.py profile         # data-quality report + species choice → outputs/data_quality.md, config.yaml
python -m pytest tests         # cleaning-rule and read-only-guard tests
```
Stages after `extract` work offline from parquet.

## Layout
- `config/` — settings (zoo, species, thresholds)
- `data/raw/`, `data/processed/` — extracts from the DB dump and cleaned outputs
- `src/` — pipeline modules
- `notebooks/` — exploration
- `outputs/` — reports, backtest results
- `docs/` — plan and notes
