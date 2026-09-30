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
python main.py            # list pipeline stages
python main.py <stage>    # run a stage (not implemented yet)
```

## Layout
- `config/` — settings (zoo, species, thresholds)
- `data/raw/`, `data/processed/` — extracts from the DB dump and cleaned outputs
- `src/` — pipeline modules
- `notebooks/` — exploration
- `outputs/` — reports, backtest results
- `docs/` — plan and notes
