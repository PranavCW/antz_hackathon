"""Stage cases: illness episodes, T0 (vet aware) and estimated onset (PLAN.md §1.5, §I.1–2, B5). Labels only —
nothing here is ever a model input.

A record counts for animal A if it is a standard-case medical record for A (legacy `animal_id` or an active
mapping). Its items (complaints, diagnoses) apply to A if `item.animal_id = A` or the item has no animal.
Records ≤ 14 days apart merge into one episode. An episode is a **case** if any record is serious (max
severity ≥ High, prognosis ≥ poor, hospital emergency/inpatient/critical) or it ends in an illness death;
euthanasia counts only after an illness record within 60 days. Trauma deaths are reported separately.

Outputs: data/processed/case_records.parquet (one row per record × animal), cases.parquet (episodes),
outputs/cases_summary.md.
"""

import numpy as np
import pandas as pd

from src import config, store
from src.animals import animal_table
from src.schema_check import _md_table

SEVERITY = {"mild": 1, "mildd": 1, "milds": 1, "low": 1, "moderate": 2, "medium": 2, "high": 3, "extreme": 4}
PROGNOSIS = {"favourable": 0, "favorable": 0, "guarded": 1, "doubtful": 2, "poor": 3, "grave": 4}
SEVERITY_NAMES = {1: "Mild", 2: "Moderate", 3: "High", 4: "Extreme"}
PROGNOSIS_NAMES = {0: "favourable", 1: "guarded", 2: "doubtful", 3: "poor", 4: "grave"}
DURATION_DAYS = {"day": 1, "days": 1, "week": 7, "weeks": 7, "month": 30, "months": 30, "year": 365, "years": 365}


def severity_rank(s: pd.Series) -> pd.Series:
    return s.fillna("").str.strip().str.lower().map(SEVERITY)


def prognosis_rank(s: pd.Series) -> pd.Series:
    return s.fillna("").str.strip().str.lower().map(PROGNOSIS)


def duration_days(value: pd.Series, unit: pd.Series) -> pd.Series:
    """Complaint duration before the vet saw it, in days (NaN when missing/unparseable)."""
    n = pd.to_numeric(value.astype(str).str.extract(r"([0-9]+(?:\.[0-9]+)?)")[0], errors="coerce")
    return n * unit.fillna("").str.strip().str.lower().map(DURATION_DAYS)


def manner_class(manner: pd.DataFrame, classes: dict) -> pd.Series:
    """manner_of_death id → class (illness/euthanasia/trauma/unknown/junk/unmapped), string_id case-insensitive."""
    lookup = {sid: cls for cls, sids in classes.items() for sid in sids}
    sid = manner["string_id"].fillna("").str.strip().str.lower()
    return pd.Series(sid.map(lookup).fillna("unmapped").to_numpy(), index=manner["id"])


def record_animals(records: pd.DataFrame, mapping: pd.DataFrame, max_animals: int) -> pd.DataFrame:
    """(medical_record_id, animal_id, n_animals) — legacy animal_id ∪ active mappings; batch records dropped."""
    legacy = records[["id", "animal_id"]].dropna().rename(columns={"id": "medical_record_id"})
    mapped = mapping.loc[mapping["active"] == 1, ["medical_record_id", "animal_id"]]
    ra = pd.concat([legacy, mapped]).dropna().drop_duplicates().astype(int)
    ra["n_animals"] = ra.groupby("medical_record_id")["animal_id"].transform("size")
    return ra[ra["n_animals"] <= max_animals]


def item_times(items: pd.DataFrame, ra: pd.DataFrame) -> pd.DataFrame:
    """Attach complaint/diagnosis items to (record, animal): item.animal_id = A, or no animal on the item."""
    it = items[items["is_deleted"] == 0].copy()
    it["t"] = it["recorded_date_time"].fillna(it["created_at"])
    own = it[it["animal_id"].notna()].merge(ra[["medical_record_id", "animal_id"]], on=["medical_record_id", "animal_id"])
    shared = it[it["animal_id"].isna()].drop(columns="animal_id").merge(
        ra[["medical_record_id", "animal_id"]], on="medical_record_id")
    return pd.concat([own, shared], ignore_index=True)


def build_records(raw, cfg: dict) -> pd.DataFrame:
    """One row per standard-case medical record × animal with T0 candidates and seriousness."""
    lcfg = cfg["labels"]
    rec = raw("medical_records")
    ct = raw("medical_case_types")
    standard = set(ct.loc[ct["string_id"] == "standard", "id"].astype(str))
    rec = rec[(rec["is_deleted"] == 0) & rec["case_type"].astype(str).isin(standard) & (rec["created_for"] == "medical")]
    ra = record_animals(rec, raw("medical_record_animals"), lcfg["max_animals_per_record"])
    r = ra.merge(rec[["id", "created_at", "system_generated_record_date", "medical_record_type"]],
                 left_on="medical_record_id", right_on="id").drop(columns="id")

    # Complaints: severity, earliest time, longest pre-visit duration
    comp = item_times(raw("complaints"), ra)
    comp["sev"] = severity_rank(comp["severity"])
    comp["dur_days"] = duration_days(comp["duration"], comp["duration_unit"])
    cg = comp.groupby(["medical_record_id", "animal_id"]).agg(
        complaint_t=("t", "min"), complaint_sev=("sev", "max"), duration_days=("dur_days", "max"),
        n_complaints=("t", "size"))

    # Diagnoses: severity, prognosis, earliest time
    diag = item_times(raw("diagnoses"), ra)
    diag["sev"] = severity_rank(diag["severity"])
    diag["prog"] = prognosis_rank(diag["prognosis"])
    dg = diag.groupby(["medical_record_id", "animal_id"]).agg(
        diagnosis_t=("t", "min"), diagnosis_sev=("sev", "max"), worst_prognosis=("prog", "max"),
        n_diagnoses=("t", "size"))

    # Hospital (rejected excluded)
    h = raw("hospital_animals").merge(raw("hospital_cases"), left_on="hospital_case_id", right_on="id",
                                      suffixes=("", "_case"))
    h = h[h["status"] != "rejected"]
    h["hospital_serious"] = (h["visit_type"] == "emergency") | (h["treatment_type"] == "inpatient")
    h["critical"] = h["health_status"] == "critical"
    hg = h.groupby(["medical_record_id", "animal_id"]).agg(
        admitted_t=("admitted_at", "min"), hospital_serious=("hospital_serious", "any"), critical=("critical", "any"))

    r = r.join(cg, on=["medical_record_id", "animal_id"]).join(dg, on=["medical_record_id", "animal_id"]) \
         .join(hg, on=["medical_record_id", "animal_id"])
    for col in ("hospital_serious", "critical"):
        r[col] = r[col].eq(True)

    # T0 = LEAST(record created_at, earliest complaint/diagnosis, hospital admitted_at)
    t_cols = {"created_at": "record", "complaint_t": "complaint", "diagnosis_t": "diagnosis", "admitted_t": "hospital"}
    times = r[list(t_cols)]
    r["t0"] = times.min(axis=1)
    r["t0_source"] = times.idxmin(axis=1).map(t_cols)
    r["max_severity"] = r[["complaint_sev", "diagnosis_sev"]].max(axis=1)
    r["serious"] = (r["max_severity"] >= SEVERITY[lcfg["min_severity"].lower()]) \
        | (r["worst_prognosis"] >= PROGNOSIS[lcfg["min_prognosis"].lower()]) | r["hospital_serious"] | r["critical"]
    # B5: T0 uses the user-entered date; flag records whose entered date is far from the real insert time
    lag = (r["system_generated_record_date"] - r["created_at"]).dt.days
    r["date_flag"] = np.select([lag > 1, lag < -1], ["backdated", "dated_after_insert"], default="")
    return r.dropna(subset=["t0"])


def death_rows(raw, cfg: dict) -> pd.DataFrame:
    """Approved, active animal mortality with death time and manner class (B5)."""
    m = raw("mortality")
    m = m[(m["is_active"] == 1) & (m["status"] == "APPROVED") & (m["entity_type"] == "animal")].copy()
    cls = manner_class(raw("manner_of_death"), cfg["manner_of_death"])
    m["death_at"] = m["date_of_death"].fillna(m["discovered_date"])
    m["manner_class"] = m["manner_of_death"].map(cls).fillna("unmapped")
    m = m.dropna(subset=["death_at"]).sort_values("death_at").drop_duplicates("entity_id")
    return m.rename(columns={"entity_id": "animal_id"})[["animal_id", "death_at", "is_estimate", "manner_class"]]


def build_episodes(records: pd.DataFrame, deaths: pd.DataFrame, lcfg: dict) -> pd.DataFrame:
    """Merge records ≤ merge_days apart per animal; attach deaths; decide which episodes are cases."""
    r = records.sort_values(["animal_id", "t0"]).copy()
    gap = r.groupby("animal_id")["t0"].diff()
    r["episode_n"] = (gap.isna() | (gap > pd.Timedelta(days=lcfg["episode_merge_days"]))).groupby(r["animal_id"]).cumsum()
    first = r.groupby(["animal_id", "episode_n"]).head(1).set_index(["animal_id", "episode_n"])
    g = r.groupby(["animal_id", "episode_n"])
    ep = pd.DataFrame({
        "t0": g["t0"].min(), "last_t": g["t0"].max(), "n_records": g.size(),
        "record_ids": g["medical_record_id"].apply(lambda s: ",".join(map(str, s))),
        "max_severity": g["max_severity"].max(), "worst_prognosis": g["worst_prognosis"].max(),
        "hospital_serious": g["hospital_serious"].any(), "critical": g["critical"].any(),
        "serious_record": g["serious"].any(), "date_flags": g["date_flag"].apply(lambda s: ";".join(sorted(set(s) - {""}))),
    })
    ep["t0_source"] = first["t0_source"]
    # Estimated onset = T0 − how long the first record's symptoms had existed before the vet saw them
    ep["est_onset"] = ep["t0"] - pd.to_timedelta(first["duration_days"].fillna(0), unit="D")
    ep = ep.reset_index()

    # Attach each death to the animal's latest episode within the lookback window before it
    d = deaths.assign(death_at=pd.to_datetime(deaths["death_at"]))
    lookback = pd.Timedelta(days=lcfg["euthanasia_lookback_days"])
    cand = d.merge(ep[["animal_id", "episode_n", "last_t", "t0"]], on="animal_id", how="left")
    cand = cand[(cand["t0"] <= cand["death_at"]) & (cand["last_t"] >= cand["death_at"] - lookback)]
    attached = cand.sort_values("last_t").drop_duplicates("animal_id", keep="last")
    ep = ep.merge(attached[["animal_id", "episode_n", "death_at", "is_estimate", "manner_class"]],
                  on=["animal_id", "episode_n"], how="left")

    # Deaths without a preceding record become their own episode (T0 = death; no early-warning lead possible
    # from vet data, but the animal may still have had keeper-visible signs)
    lone = d[~d["animal_id"].isin(attached["animal_id"])]
    lone = lone.assign(t0=lone["death_at"], last_t=lone["death_at"], n_records=0, t0_source="death",
                       est_onset=lone["death_at"], serious_record=False, hospital_serious=False, critical=False)
    ep = pd.concat([ep, lone], ignore_index=True)

    mc = ep["manner_class"]
    ep["outcome"] = np.select(
        [mc.isna(), mc == "illness", mc == "euthanasia", mc == "trauma", mc == "unknown", mc == "junk"],
        ["alive", "died_illness", "euthanised", "died_trauma", "died_unknown", "junk"], default="died_unmapped")
    ill_death = (mc == "illness") | ((mc == "euthanasia") & (ep["n_records"] > 0))
    ep["case_class"] = np.select(
        [mc == "junk", mc == "trauma", ep["serious_record"].fillna(False) | ill_death,
         (mc == "unknown") | (mc == "unmapped"), ep["n_records"] > 0],
        ["junk", "trauma", "illness", "unknown_death", "not_serious"], default="not_serious")
    ep["is_case"] = ep["case_class"] == "illness"
    ep["severity"] = ep["max_severity"].map(SEVERITY_NAMES)
    ep["prognosis"] = ep["worst_prognosis"].map(PROGNOSIS_NAMES)
    ep = ep.sort_values(["animal_id", "t0"]).reset_index(drop=True)
    ep["episode_id"] = np.arange(1, len(ep) + 1)
    return ep[["episode_id", "animal_id", "t0", "t0_source", "est_onset", "last_t", "n_records", "record_ids",
               "severity", "prognosis", "hospital_serious", "critical", "serious_record", "death_at", "is_estimate",
               "outcome", "case_class", "is_case", "date_flags"]]


def run() -> None:
    cfg = config.load()
    raw = lambda name: store.read("raw", name)  # noqa: E731
    animals = animal_table(raw("animals"), raw("taxonomy"), raw("mortality"))

    records = build_records(raw, cfg)
    records = records[records["animal_id"].isin(animals["animal_id"])]
    deaths = death_rows(raw, cfg)
    deaths = deaths[deaths["animal_id"].isin(animals["animal_id"])]
    episodes = build_episodes(records, deaths, cfg["labels"])
    episodes = episodes.merge(animals[["animal_id", "taxonomy_id", "species", "type"]], on="animal_id", how="left")
    store.write("processed", "case_records", records)
    store.write("processed", "cases", episodes)

    chosen = set(cfg["species"])
    in_species = episodes["taxonomy_id"].isin(chosen)
    summary = episodes.groupby(["case_class", "outcome"]).agg(
        episodes=("episode_id", "size"), in_chosen_species=("taxonomy_id", lambda s: s.isin(chosen).sum())
    ).reset_index()
    cases = episodes[episodes["is_case"]]
    spot = cases[in_species[episodes["is_case"]]].sort_values("t0", ascending=False).head(3)
    parts = [
        f"# Cases — zoo {cfg['zoo_id']}\n",
        f"- Standard medical record × animal rows: {len(records):,} → episodes (≤ {cfg['labels']['episode_merge_days']} d "
        f"merge): {int((episodes['n_records'] > 0).sum()):,}; deaths: {len(deaths):,}",
        f"- **Cases: {len(cases):,}** ({int(in_species[episodes['is_case']].sum())} in chosen species, "
        f"{int((cases['type'] == 'single').sum())} single animals)",
        f"- T0 source: " + ", ".join(f"{k} {v}" for k, v in cases["t0_source"].value_counts().items()),
        f"- Records with suspicious dates: " + ", ".join(
            f"{k} {v}" for k, v in records["date_flag"].replace("", "ok").value_counts().items()) + "\n",
        "## Episodes by class and outcome\n", _md_table(summary),
        "\n## Spot-check in Antz UI (3 most recent cases in chosen species)\n",
        _md_table(spot[["episode_id", "animal_id", "species", "t0", "t0_source", "record_ids", "severity", "prognosis",
                        "hospital_serious", "outcome"]]),
    ]
    store.output_path("cases_summary.md").write_text("\n".join(parts))
    print("\n".join(parts[1:5]))
    print("→ data/processed/cases.parquet, outputs/cases_summary.md")
