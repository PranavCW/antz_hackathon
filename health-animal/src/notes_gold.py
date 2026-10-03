"""Stage notes_gold: write a 50-note hand-labelling sheet for the LLM audit (PLAN.md §6, Sprint 3 S3-1).

Writes data/processed/notes_gold.csv. A person fills `health_related` (1/0) and one column per category
(1 = the note says this got worse, blank/0 = not). Never overwrites a sheet that already has labels.
Notes with content (lexicon hit or ≥ 30 characters) are sampled first; the rest at random.
"""

import pandas as pd

from src import config
from src.notes_llm import CATEGORIES, lexicon_signals, notes_to_decode

N = 50


def run() -> None:
    cfg = config.load()
    path = cfg["paths"]["processed"] / "notes_gold.csv"
    if path.exists():
        existing = pd.read_csv(path)
        if existing[["health_related"] + CATEGORIES].notna().any().any():
            print(f"{path} already has labels — not overwriting. Delete it to resample.")
            return

    notes = notes_to_decode(cfg)
    notes["content"] = notes["text"].str.len().ge(30) | notes["text"].map(lambda t: bool(lexicon_signals(t)))
    picked = pd.concat([
        notes[notes["content"]].sample(frac=1, random_state=cfg["model"]["random_seed"]),
        notes[~notes["content"]].sample(frac=1, random_state=cfg["model"]["random_seed"]),
    ]).head(N)
    sheet = picked[["source", "source_id", "text_hash", "species", "date", "text"]].copy()
    sheet["health_related"] = None
    for cat in CATEGORIES:
        sheet[cat] = None
    sheet["labeller_notes"] = None
    sheet.to_csv(path, index=False)
    print(f"{len(sheet)} notes ({int(picked['content'].sum())} with content) → {path}")
    print("Fill health_related (1/0) and 1 in each category the note says got worse.")
