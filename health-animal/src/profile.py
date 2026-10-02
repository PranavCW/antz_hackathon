"""Stage profile: per-species data density + data-quality report (B6); choose 3–5 species → config.yaml.

Reads data/processed/ (clean stage) and the raw label tables. Writes outputs/data_quality.md,
outputs/profile_species.csv, outputs/profile_animals.csv, and the `species:` line of config/config.yaml.
The case count here is a rough proxy for checkpoint M1; Sprint 2 (cases stage) builds real episodes.
"""

import re

import numpy as np
import pandas as pd

from src import config, store
from src.animals import animal_table
from src.db import PROJECT_ROOT
from src.schema_check import _md_table


def max_in_window(dates: pd.Series, days: int) -> int:
    """Largest number of readings falling inside any `days`-long window."""
    t = np.sort(pd.to_datetime(dates).to_numpy())
    if len(t) == 0:
        return 0
    right = np.searchsorted(t, t + np.timedelta64(days, "D"), side="right")
    return int((right - np.arange(len(t))).max())


def candidate_cases(raw, animals: pd.DataFrame, cfg: dict, since: pd.Timestamp) -> pd.DataFrame:
    """Rough (animal_id, event) list of serious illness events since `since` — the M1 go/no-go proxy (§I.1)."""
    rec = raw("medical_records")
    ct = raw("medical_case_types")
    standard_ids = set(ct.loc[ct["string_id"] == "standard", "id"].astype(str))
    rec = rec[(rec["is_deleted"] == 0) & rec["case_type"].astype(str).isin(standard_ids)
              & (rec["created_for"] == "medical") & (rec["created_at"] >= since)]

    sev = {"high", "extreme"}
    comp, diag = raw("complaints"), raw("diagnoses")
    serious = set(comp.loc[(comp["is_deleted"] == 0) & comp["severity"].str.lower().isin(sev), "medical_record_id"])
    serious |= set(diag.loc[(diag["is_deleted"] == 0) & (diag["severity"].str.lower().isin(sev)
                            | diag["prognosis"].str.lower().isin({"poor", "grave"})), "medical_record_id"])
    hosp = raw("hospital_cases").merge(raw("hospital_animals"), left_on="id", right_on="hospital_case_id")
    hosp = hosp[(hosp["status"] != "rejected") & ((hosp["visit_type"] == "emergency")
                | (hosp["treatment_type"] == "inpatient") | (hosp["health_status"] == "critical"))]
    serious |= set(hosp["medical_record_id"].dropna())

    rec = rec[rec["id"].isin(serious)]
    mapped = raw("medical_record_animals")
    mapped = mapped[mapped["active"] == 1][["medical_record_id", "animal_id"]]
    rec_animals = pd.concat([rec[["id", "animal_id"]].rename(columns={"id": "medical_record_id"}),
                             mapped[mapped["medical_record_id"].isin(rec["id"])]]).dropna().drop_duplicates()
    events = rec_animals.assign(event="medical_" + rec_animals["medical_record_id"].astype(int).astype(str))

    # Illness / euthanasia deaths (manner_of_death matched case-insensitively; live string_ids vary in case)
    mod = raw("manner_of_death").assign(sid=lambda m: m["string_id"].str.lower())
    classes = cfg["manner_of_death"]
    illness_ids = set(mod.loc[mod["sid"].isin(classes["illness"] + classes["euthanasia"]), "id"])
    m = raw("mortality")
    m = m[(m["is_active"] == 1) & (m["status"] == "APPROVED") & m["manner_of_death"].isin(illness_ids)
          & (m["date_of_death"].fillna(m["discovered_date"]) >= since)]
    deaths = pd.DataFrame({"animal_id": m["entity_id"], "event": "death_" + m["_id"].astype(str)})
    out = pd.concat([events[["animal_id", "event"]], deaths])
    return out[out["animal_id"].isin(animals["animal_id"])]


def write_species_to_config(tsns, names) -> None:
    path = PROJECT_ROOT / "config" / "config.yaml"
    text = path.read_text()
    line = f"species: [{', '.join(str(t) for t in tsns)}]  # {', '.join(names)}"
    path.write_text(re.sub(r"^species:.*$", line, text, count=1, flags=re.MULTILINE))


def run() -> None:
    cfg = config.load()
    pcfg = cfg["profile"]
    raw = lambda name: store.read("raw", name)  # noqa: E731
    proc = lambda name: store.read("processed", name)  # noqa: E731
    now = pd.to_datetime(pd.read_csv(store.output_path("extract_log.csv"))["extracted_at_utc"].iloc[0])
    since = now - pd.Timedelta(days=pcfg["window_days"])
    weeks = pcfg["window_days"] / 7

    animals = animal_table(raw("animals"), raw("taxonomy"), raw("mortality"))
    singles = animals[animals["type"] == "single"]
    weights, notes = proc("weights_clean"), proc("notes_clean")
    enclosure_day, intake_daily = proc("enclosure_day"), proc("intake_daily")

    # 1. Per animal
    w = weights[weights["animal_id"].isin(singles["animal_id"])]
    w_kept = w[w["kept"] & (w["recorded_at"] >= since)]
    per = pd.DataFrame(index=pd.Index(singles["animal_id"], name="animal_id"))
    per["weights_kept"] = w_kept.groupby("animal_id").size()
    per["max_weights_180d"] = w_kept.groupby("animal_id")["recorded_at"].apply(lambda s: max_in_window(s, 180))
    per["weights_dropped"] = w[~w["kept"]].groupby("animal_id").size()
    per["weights_corrected"] = w[w["flags"].str.contains("corrected_unit")].groupby("animal_id").size()
    solo = enclosure_day[enclosure_day["animal_id_if_solo"].notna() & (enclosure_day["confidence"] == "high")
                         & (enclosure_day["date"] >= since)]
    per["solo_days"] = solo.groupby("animal_id_if_solo").size()
    per["leftover_days"] = intake_daily[intake_daily["date"] >= since].groupby("animal_id").size()
    n_kept = notes[notes["kept"] & (notes["date"] >= since)]
    per["notes"] = n_kept.groupby("animal_id").size()
    cases = candidate_cases(raw, singles, cfg, since)
    per["candidate_cases"] = cases.groupby("animal_id")["event"].nunique()
    per = per.fillna(0).astype(int)
    per["weight_monitorable"] = per["max_weights_180d"] >= cfg["weight"]["min_readings_in_baseline"]
    per["intake_monitorable"] = (per["leftover_days"] >= cfg["intake"]["min_valid_days"]) & \
        (per["leftover_days"] >= cfg["intake"]["min_logging_rate"] * per["solo_days"].clip(lower=1))
    per["monitorable"] = per["weight_monitorable"] | per["intake_monitorable"]
    per["cases_if_monitorable"] = per["candidate_cases"].where(per["monitorable"], 0)
    per = per.join(singles.set_index("animal_id")[["taxonomy_id", "species"]])
    per.reset_index().to_csv(store.output_path("profile_animals.csv"), index=False)

    # 2. Per species
    g = per.groupby(["taxonomy_id", "species"], dropna=False)
    sp = pd.DataFrame({
        "animals": g.size(),
        "weighed": g["weights_kept"].apply(lambda s: (s > 0).sum()),
        "weight_monitorable": g["weight_monitorable"].sum(),
        "intake_monitorable": g["intake_monitorable"].sum(),
        "weights_per_animal_90d": g["weights_kept"].sum() / g["weights_kept"].apply(lambda s: max((s > 0).sum(), 1))
                                  / (pcfg["window_days"] / 90),
        "pct_weights_dropped": 100 * g["weights_dropped"].sum()
                               / (g["weights_dropped"].sum() + g["weights_kept"].sum()).clip(lower=1),
        "pct_weights_corrected": 100 * g["weights_corrected"].sum() / g["weights_kept"].sum().clip(lower=1),
        "pct_solo_days_logged": 100 * g["leftover_days"].sum() / g["solo_days"].sum().clip(lower=1),
        "notes_per_100_animal_weeks": 100 * g["notes"].sum() / g.size() / weeks,
        "candidate_cases": g["candidate_cases"].sum(),
        # cases on animals we could actually have watched — the number that matters for the backtest
        "cases_on_monitorable": g["cases_if_monitorable"].sum(),
    }).round(2).reset_index()
    sp["monitorable"] = sp["weight_monitorable"] >= pcfg["min_weight_monitorable_animals"]
    sp = sp.sort_values(["monitorable", "cases_on_monitorable", "weight_monitorable"], ascending=False)
    sp.to_csv(store.output_path("profile_species.csv"), index=False)

    # 3. Choose species and write them to config
    lo, hi = pcfg["n_species"]
    chosen = sp[sp["monitorable"]].head(hi)
    write_species_to_config(chosen["taxonomy_id"].astype(int).tolist(), chosen["species"].fillna("?").tolist())
    total_cases = int(chosen["cases_on_monitorable"].sum())
    m1_ok = len(chosen) >= lo and total_cases >= 10

    # 4. Data-quality report
    dq = pd.read_csv(store.output_path("data_quality_rows.csv"))
    dq["pct"] = (100 * dq["rows"] / dq["total"]).round(1)
    test_like = notes["text"].fillna("").str.strip().str.lower().str.match(r"^(test|testing|asdf|qwerty)")
    parts = [
        f"# Data-quality report — zoo {cfg['zoo_id']}\n",
        f"History window for density metrics: last {pcfg['window_days']} days up to extract "
        f"({since:%Y-%m-%d} → {now:%Y-%m-%d}).\n",
        "## Checkpoint M1\n",
        f"- Species chosen (written to `config/config.yaml`): **{len(chosen)}** — "
        + (", ".join(f"{r.species} ({int(r.taxonomy_id)})" for r in chosen.itertuples()) or "none") + "\n"
        f"- Candidate serious cases on monitorable animals of those species (rough proxy): **{total_cases}** "
        f"(all animals of those species: {int(chosen['candidate_cases'].sum())})\n"
        f"- Go/no-go (≥ {lo} species with ≥ 10 cases total): **{'GO' if m1_ok else 'NO-GO'}**\n",
        "## Data reality\n",
        f"- Single animals: {len(singles):,}; weighed in window: {(per['weights_kept'] > 0).sum():,}; "
        f"weight-monitorable (≥3 kept readings within 180 d): {per['weight_monitorable'].sum():,}\n"
        f"- Intake-monitorable animals (≥{cfg['intake']['min_valid_days']} leftover days, logging ≥ "
        f"{cfg['intake']['min_logging_rate']:.0%}): {per['intake_monitorable'].sum():,} → leftover detector "
        f"{'viable' if per['intake_monitorable'].sum() >= 5 else '**not viable — cut order #3**'}\n"
        f"- Note rows whose text starts with test/testing/asdf/qwerty: {test_like.sum():,} of {len(notes):,} "
        f"({100 * test_like.mean():.0f}%)\n",
        "## Species profile (top 30)\n", _md_table(sp.head(30)),
        "\n## Rows dropped / flagged per source\n", _md_table(dq[["source", "kind", "reason", "rows", "pct"]], 200),
    ]
    store.output_path("data_quality.md").write_text("\n".join(parts))

    print(sp.head(10).to_string(index=False))
    print(f"\nChosen {len(chosen)} species → config.yaml; {total_cases} candidate cases → M1 {'GO' if m1_ok else 'NO-GO'}")
    print("Report → outputs/data_quality.md")
