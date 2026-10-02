"""Keeper notes for the LLM (PLAN.md §1.4, B4): one row per (text source, animal) with date, text and reasons.

Sources: observations (+ their comment thread), food-wastage notes, weighing comments, incident notes.
Attribution: animal tags → that animal, weight 1/n animals tagged; enclosure tags → the single occupant that
day only; section/site tags ignored. Rows that must not reach the LLM carry `drop_reason`.
"""

import hashlib
import re

import numpy as np
import pandas as pd

from src.animals import local_day


def text_hash(text: str) -> str:
    norm = re.sub(r"\s+", " ", (text or "").strip().lower())
    return hashlib.sha1(norm.encode()).hexdigest()[:16]


def note_types(type_mapping: pd.DataFrame, types: pd.DataFrame) -> pd.DataFrame:
    """observation_id → parent/child type string_ids (live mappings only)."""
    tm = type_mapping[type_mapping["deleted_on"].isna()]
    key = types.set_index("id")["string_id"].fillna(types.set_index("id")["type_name"].str.lower())
    tm = tm.assign(parent=tm["parent_observation_type_id"].map(key), child=tm["child_observation_type_id"].map(key))
    return tm.groupby("observation_id").agg(
        parent_types=("parent", lambda s: ";".join(sorted(set(s.dropna())))),
        child_types=("child", lambda s: ";".join(sorted(set(s.dropna())))))


def observation_rows(obs, mapping, comments, type_mapping, types, enclosure_day, animals, tz, skip_types):
    """Observation notes attributed to animals (§1.4)."""
    o = obs[(obs["is_deleted"] == 0) & (obs["active"] == 1)].copy()
    # created_for edit bug: edits without resending it reset it to the edit date (§1.4)
    created_day = local_day(o["created_at"], tz)
    edited = o["modified_at"].notna() & (o["modified_at"] > o["created_at"])
    o["date"] = pd.to_datetime(o["created_for"]).fillna(created_day)
    fixed = edited & (created_day < o["date"])
    o.loc[fixed, "date"] = created_day[fixed]
    o["date_fixed"] = fixed

    # Comment thread, flat, in time order, appended to the note body
    c = comments[comments["active"] == 1].sort_values("created_at")
    thread = c.groupby("observation_id")["observation"].apply(
        lambda s: "\n".join(t.strip() for t in s.dropna() if t.strip()))
    o = o.join(thread.rename("thread"), on="observation_id").join(note_types(type_mapping, types), on="observation_id")
    o["text"] = o["observation_name"].fillna("").str.strip()
    has_thread = o["thread"].fillna("").ne("")
    o.loc[has_thread, "text"] = o.loc[has_thread, "text"] + "\n[comments]\n" + o.loc[has_thread, "thread"]

    # Attribution
    m = mapping[mapping["is_deleted"] == 0][["observation_id", "ref_type", "ref_id"]]
    m = m.merge(o[["observation_id", "date"]], on="observation_id")
    by_animal = m[m["ref_type"] == "animal"].assign(animal_id=lambda x: x["ref_id"], attribution="animal")
    encl = m[m["ref_type"] == "enclosure"].merge(
        enclosure_day, left_on=["ref_id", "date"], right_on=["enclosure_id", "date"], how="left")
    encl_ok = encl[encl["animal_id_if_solo"].notna() & (encl["confidence"] == "high")]
    by_encl = encl_ok.assign(animal_id=encl_ok["animal_id_if_solo"].astype(int), attribution="solo_enclosure")
    att = pd.concat([by_animal, by_encl])[["observation_id", "animal_id", "attribution"]]
    att = att[att["animal_id"].isin(animals["animal_id"])]
    att = att.sort_values("attribution").drop_duplicates(["observation_id", "animal_id"])
    att["weight"] = 1 / att.groupby("observation_id")["animal_id"].transform("size")

    unattributed = o[~o["observation_id"].isin(att["observation_id"])][["observation_id"]].assign(
        animal_id=np.nan, attribution=None, weight=np.nan)
    rows = o.merge(pd.concat([att, unattributed]), on="observation_id")
    rows["drop_reason"] = None
    rows.loc[rows["animal_id"].isna(), "drop_reason"] = "not_attributable"

    # Maintenance-only notes skipped before the LLM
    parents = rows["parent_types"].fillna("")
    maintenance = parents.ne("") & parents.str.split(";").map(lambda ps: all(p in skip_types for p in ps))
    rows.loc[rows["drop_reason"].isna() & maintenance, "drop_reason"] = "maintenance_type"
    return rows.assign(source="observation", source_id=rows["observation_id"])[
        ["source", "source_id", "animal_id", "date", "text", "priority", "parent_types", "child_types",
         "attribution", "weight", "date_fixed", "drop_reason"]]


def other_rows(wastage_rows, assessments, incidents, animals, tz):
    """Free text from wastage (solo enclosure-days only), weighing comments and incidents."""
    w = wastage_rows[wastage_rows["notes"].fillna("").str.strip().ne("")]
    w = w.assign(source="food_wastage", source_id=w["id"], animal_id=w["animal_id_if_solo"], text=w["notes"].str.strip(),
                 attribution=np.where(w["animal_id_if_solo"].notna(), "solo_enclosure", None),
                 drop_reason=w["drop_reason"])
    w.loc[w["drop_reason"].isna() & w["animal_id"].isna(), "drop_reason"] = "not_attributable"

    a = assessments[assessments["comments"].fillna("").str.strip().ne("") & (assessments["is_deleted"] == 0)]
    a = a.assign(source="assessment_comment", source_id=a["id"], text=a["comments"].str.strip(),
                 date=local_day(a["recorded_date_time"].fillna(a["created_at"]), tz), attribution="animal",
                 drop_reason=None)

    i = incidents[incidents["notes"].fillna("").str.strip().ne("") & (incidents["is_deleted"] == 0)]
    i = i.assign(source="incident", source_id=i["id"], text=i["notes"].str.strip(),
                 date=local_day(i["incident_date"].fillna(i["created_at"]), tz), attribution="animal",
                 drop_reason=None)

    out = pd.concat([w, a, i])[["source", "source_id", "animal_id", "date", "text", "attribution", "drop_reason"]]
    out["weight"] = np.where(out["animal_id"].notna(), 1.0, np.nan)
    out.loc[out["drop_reason"].isna() & ~out["animal_id"].isin(animals["animal_id"]), "drop_reason"] = "animal_not_in_zoo"
    return out


def finalise(rows: pd.DataFrame, template_texts, min_chars: int, duplicate_days: int) -> pd.DataFrame:
    """Empty/template texts and copy-paste duplicates (same animal, same text within N days) — B4."""
    rows = rows.reset_index(drop=True)
    rows["text_hash"] = rows["text"].map(text_hash)
    norm = rows["text"].fillna("").str.strip().str.lower().str.replace(r"[\s.!]+$", "", regex=True)
    live = rows["drop_reason"].isna()
    rows.loc[live & (norm.str.len() < min_chars), "drop_reason"] = "empty"
    rows.loc[rows["drop_reason"].isna() & norm.isin(template_texts), "drop_reason"] = "template_text"

    live = rows[rows["drop_reason"].isna()].sort_values(["animal_id", "text_hash", "date"])
    gap = live.groupby(["animal_id", "text_hash"])["date"].diff()
    dup = gap.notna() & (gap <= pd.Timedelta(days=duplicate_days))
    rows.loc[dup[dup].index, "drop_reason"] = "duplicate_text"
    rows["kept"] = rows["drop_reason"].isna()
    return rows
