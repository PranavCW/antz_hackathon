"""Stage notes_audit: LLM audit against the hand-labelled gold notes (PLAN.md §6, S3-1).

Per category precision/recall of the LLM and of the keyword lexicon on "worse" signals, agreement on
is_health_related, and the verbatim-quote pass rate over the whole batch. Reads data/processed/notes_gold.csv
(filled by a person; if its `labeller_notes` start with `claude-provisional` the report says so) and
notes_decoded/notes_signals from `notes_llm`. Writes outputs/notes_audit.md.
"""

import pandas as pd

from src import config, store
from src.notes_llm import CATEGORIES
from src.schema_check import _md_table


def split_cats(value) -> set:
    return {c for c in str(value or "").split(";") if c and c != "nan"}


def prf(tp: int, fp: int, fn: int) -> dict:
    return {"tp": tp, "fp": fp, "fn": fn,
            "precision": tp / (tp + fp) if tp + fp else float("nan"),
            "recall": tp / (tp + fn) if tp + fn else float("nan")}


def audit_table(joined: pd.DataFrame, pred_col: str) -> pd.DataFrame:
    """One row per category (+ overall) for predictions in `pred_col` against gold columns of the same names."""
    rows, totals = [], [0, 0, 0]
    for cat in CATEGORIES:
        gold = joined[cat].fillna(0).astype(int) == 1
        pred = joined[pred_col].map(lambda v: cat in split_cats(v))
        tp, fp, fn = int((gold & pred).sum()), int((~gold & pred).sum()), int((gold & ~pred).sum())
        totals = [totals[0] + tp, totals[1] + fp, totals[2] + fn]
        if tp + fp + fn:
            rows.append({"category": cat, **prf(tp, fp, fn)})
    rows.append({"category": "ALL", **prf(*totals)})
    return pd.DataFrame(rows)


def run() -> None:
    cfg = config.load()
    gold = pd.read_csv(cfg["paths"]["processed"] / "notes_gold.csv")
    labelled = gold[["health_related"] + CATEGORIES].notna().any(axis=1)
    if not labelled.any():
        raise RuntimeError("notes_gold.csv has no labels yet — fill it in first")
    gold = gold[labelled]
    dec = store.read("processed", "notes_decoded")
    sig = store.read("processed", "notes_signals")
    j = gold.merge(dec, on=["source", "source_id"], how="left", suffixes=("", "_llm"))
    missing = int(j["llm_worse"].isna().sum())
    j[["llm_worse", "lexicon_worse"]] = j[["llm_worse", "lexicon_worse"]].fillna("")

    llm, lex = audit_table(j, "llm_worse"), audit_table(j, "lexicon_worse")
    agree = (j["health_related"].astype(int) == j["is_health_related"].fillna(False).astype(int)).mean()
    llm_sig = sig[sig["method"] == "llm"]
    verbatim = llm_sig["verbatim_ok"].mean() if len(llm_sig) else float("nan")
    provisional = gold["labeller_notes"].fillna("").str.startswith("claude-provisional").mean()
    first = int(gold["source_id"].iloc[0]) if len(gold) else 0  # noqa: F841

    parts = [f"# LLM audit — zoo {cfg['zoo_id']}\n",
             f"Gold notes: {len(gold)} ({int(gold['health_related'].sum())} health-related); decoded batch: {len(dec)} notes."
             + (f" {missing} gold notes were not decoded (not in the chosen species)." if missing else ""),
             f"- is_health_related agreement with gold: **{agree:.0%}**",
             f"- Verbatim-quote pass rate over the whole batch: **{verbatim:.0%}** of {len(llm_sig)} LLM signals "
             f"(failed quotes are never used)",
             f"- LLM and lexicon disagreed on {int(dec['disagree'].sum())} of {len(dec)} notes\n"]
    if provisional > 0.5:
        parts.insert(1, "> ⚠️ **These gold labels are provisional (written by Claude on Dev test data).** The audit "
                        "checks the machinery; a person must relabel a fresh sheet on real notes before the numbers "
                        "mean anything (README §7 step 9).\n")
    parts += ["## LLM (qwen2.5) — 'worse' signals per category\n", _md_table(llm.round(3)),
              "\n## Keyword lexicon — 'worse' signals per category\n", _md_table(lex.round(3))]
    store.output_path("notes_audit.md").write_text("\n".join(parts))
    print("\n".join(parts[:8]))
    print("→ outputs/notes_audit.md")
