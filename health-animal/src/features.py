"""Stage features: animal-day feature table (only data ≤ d) + labels (PLAN.md §H, §6 leakage test).

One row per (animal, local day) for single animals of the chosen species, from the animal's first signal
(or the start of vet-data coverage) to death / extract. Model inputs are the columns starting with
`w_` (weight), `i_` (intake), `n_` (notes), `ctx_` (context), `cov_` (coverage).

Replay rules — a value is visible from the day it was *entered*, never earlier:
  weights  : known_at = max(recorded_at, created_at); a spike is scored as pending at its own time and only
             leaves the baseline once the next reading exists (detectors.weight_detector(causal=True))
  notes    : available_date (created_for can be backdated)
  leftovers: available_date of the day's last entry
Medical/hospital/mortality tables are labels only: `build_features` has no parameter that could carry them, and
`leakage_check` re-derives features from inputs truncated at random dates and requires identical values.
`days since last vet case` (PLAN §H context) is deliberately omitted — it would need medical tables.

Labels (`label_days`, from cases only): label_vet = 1 if a vet-aware case starts in (d, d+14], 0 if no case of any
kind within 60 d; NaN (excluded) for days under treatment, the 15–60 d grey zone, near trauma/unknown deaths and
near death-only cases. label_all counts death-only cases (T0 = death, no vet record) as positives too.

Outputs: data/processed/animal_day.parquet, outputs/features_summary.md.
"""

from dataclasses import dataclass, replace

import numpy as np
import pandas as pd

from src import config, store
from src.animals import animal_table, local_day
from src.detectors import intake_detector, weight_detector
from src.notes_llm import CATEGORIES

PRIORITY = {"low": 1, "moderate": 2, "high": 3, "critical": 4}
MODEL_PREFIXES = ("w_", "i_", "n_", "ctx_", "cov_")
# Words that must never appear in a model-input column name (label/medical leakage guard)
FORBIDDEN_IN_FEATURES = ("label", "case", "t0", "medical", "hospital", "diagnos", "mortality", "death", "vet_")


@dataclass
class Inputs:
    animals: pd.DataFrame        # animal_id, taxonomy_id, birth_date, sex, death_at
    weights: pd.DataFrame        # weights_clean rows (kept + dropped, with drop_reason, created_at)
    intake: pd.DataFrame         # intake_daily (with available_date)
    notes: pd.DataFrame          # kept notes: animal_id, available_date, weight, priority
    signals: pd.DataFrame        # animal_id, available_date, category, weight, quote (LLM, verbatim ok, worse)
    stays: pd.DataFrame          # animal_id, start (UTC)
    species: list                # chosen taxonomy ids
    tz: str


def feature_columns(df: pd.DataFrame) -> list:
    return [c for c in df.columns if c.startswith(MODEL_PREFIXES)]


# ── Windowed helpers on a dense, sorted (animal_id, date) grid with a RangeIndex ─────────────────────────

def _roll_sum(grid: pd.DataFrame, col: str, days: int) -> pd.Series:
    """Sum of `col` over the last `days` days including today."""
    cs = grid.groupby("animal_id")[col].cumsum()
    return cs - cs.groupby(grid["animal_id"]).shift(days).fillna(0)


def _roll_max(grid: pd.DataFrame, col: str, days: int) -> pd.Series:
    r = grid.groupby("animal_id")[col].rolling(days, min_periods=1).max()
    return r.reset_index(level=0, drop=True).reindex(grid.index)


def _add_daily(grid: pd.DataFrame, events: pd.DataFrame, date_col: str, **aggs) -> pd.DataFrame:
    """Left-join per-(animal, day) aggregates of `events` onto the grid (missing → 0)."""
    names = list(aggs)
    if events.empty:
        for n in names:
            grid[n] = 0.0
        return grid
    daily = events.groupby(["animal_id", date_col]).agg(**aggs).reset_index().rename(columns={date_col: "date"})
    grid = grid.merge(daily, on=["animal_id", "date"], how="left")
    grid[names] = grid[names].fillna(0.0)
    return grid


# ── Feature build ────────────────────────────────────────────────────────────────────────────────────────

def weight_events(inp: Inputs, cfg: dict) -> pd.DataFrame:
    """Causal weight detector output, indexed on the day each reading became known."""
    w = inp.weights[inp.weights["animal_id"].isin(inp.animals["animal_id"])].copy()
    known = w[["recorded_at", "created_at"]].max(axis=1)
    w["recorded_at"] = known
    w["local_date"] = local_day(known, inp.tz)
    det = weight_detector(w, cfg["weight"], cfg["detectors"], causal=True)
    return det if len(det) else det.assign(local_date=pd.NaT)


def first_signal_dates(inp: Inputs, det: pd.DataFrame, intake: pd.DataFrame) -> pd.Series:
    parts = [det.groupby("animal_id")["local_date"].min(), inp.notes.groupby("animal_id")["available_date"].min(),
             intake.groupby("animal_id")["available_date"].min()]
    first = pd.concat([pd.to_datetime(p) for p in parts], axis=1).min(axis=1).dropna()
    first.index = first.index.astype("int64")
    return first


def make_grid(first: pd.Series, animals: pd.DataFrame, grid_end: pd.Timestamp, tz: str) -> pd.DataFrame:
    """Dense (animal_id, date) from the first signal to min(day before death, grid_end)."""
    a = animals.set_index("animal_id")
    death_day = local_day(a["death_at"], tz) if a["death_at"].notna().any() else a["death_at"]
    end = pd.concat([pd.Series(grid_end, index=a.index), death_day - pd.Timedelta(days=1)], axis=1).min(axis=1)
    span = pd.concat([first.rename("start"), end.rename("end")], axis=1, join="inner").dropna()
    span = span[span["start"] <= span["end"]]
    lengths = ((span["end"] - span["start"]).dt.days + 1).to_numpy()
    ids = np.repeat(span.index.to_numpy().astype("int64"), lengths)
    offsets = np.concatenate([np.arange(n) for n in lengths]) if len(lengths) else np.array([], dtype=int)
    starts = np.repeat(span["start"].to_numpy(), lengths)
    grid = pd.DataFrame({"animal_id": ids, "date": starts + offsets.astype("timedelta64[D]")})
    return grid.sort_values(["animal_id", "date"]).reset_index(drop=True)


def build_features(inp: Inputs, cfg: dict, grid_start: pd.Timestamp, grid_end: pd.Timestamp) -> pd.DataFrame:
    """The animal-day feature table. Pure function of `inp`: row (a, d) only depends on entries visible by d."""
    wcfg, icfg, dcfg = cfg["weight"], cfg["intake"], cfg["detectors"]
    max_age = cfg["features"]["weight_max_age_days"]

    det = weight_events(inp, cfg)
    intake = inp.intake[inp.intake["animal_id"].isin(inp.animals["animal_id"])]
    first = first_signal_dates(inp, det, intake)
    grid = make_grid(first, inp.animals, grid_end, inp.tz)
    if grid.empty:
        return grid

    # ── weight: latest reading known by d (as-of join), then freshness / coverage
    # Latest reading of the day wins: sort on the exact time so same-day ties are deterministic
    ev = det.sort_values(["local_date", "recorded_at"], kind="mergesort")[
        ["animal_id", "local_date", "status", "pct_change", "slope_pct_month", "confirmed", "fired"]]
    g = pd.merge_asof(grid.sort_values("date", kind="mergesort"), ev, left_on="date", right_on="local_date", by="animal_id",
                      direction="backward").sort_values(["animal_id", "date"]).reset_index(drop=True)
    age = (g["date"] - g["local_date"]).dt.days
    fresh = g["status"].eq("ok") & age.le(max_age)
    g["w_has_baseline"] = fresh.astype(float)
    g["w_pct_change"] = g["pct_change"].where(fresh, 0.0).clip(-50, 50)
    g["w_slope_pct_month"] = g["slope_pct_month"].where(fresh, 0.0).clip(-50, 50)
    g["w_fired"] = (g["fired"].eq(True) & fresh).astype(float)
    g["w_confirmed"] = (g["confirmed"].eq(True) & fresh).astype(float)
    g["w_pending"] = ((g["w_fired"] == 1) & (g["w_confirmed"] == 0)).astype(float)
    g["w_days_since_last"] = age.fillna(365).clip(upper=365)
    g = g.drop(columns=["local_date", "status", "pct_change", "slope_pct_month", "confirmed", "fired"])
    g = _add_daily(g, det.assign(n=1), "local_date", _w_count=("n", "sum"))
    g["w_n_180d"] = _roll_sum(g, "_w_count", wcfg["baseline_days"])

    # ── intake: leftover detector on the days as they became visible
    di = intake_detector(intake.assign(date=intake["available_date"]), icfg, dcfg) if len(intake) else pd.DataFrame()
    if len(di):
        di = di.assign(z_pos=di["z"].fillna(0).clip(lower=0), logged=1.0, low=di["low_confidence"].astype(float),
                       high=di["high_day"].astype(float))
        g = _add_daily(g, di, "date", _z=("z_pos", "max"), _cusum=("cusum", "max"), _high=("high", "sum"),
                       _logged=("logged", "max"), _low=("low", "max"))
    else:
        for c in ("_z", "_cusum", "_high", "_logged", "_low"):
            g[c] = 0.0
    g["i_z_max_7"] = _roll_max(g, "_z", 7)
    g["i_cusum_max_7"] = _roll_max(g, "_cusum", 7)
    g["i_high_days_7"] = _roll_sum(g, "_high", 7)
    g["i_logged_30d"] = _roll_sum(g, "_logged", 30) / 30
    g["i_low_confidence"] = _roll_max(g, "_low", 30)

    # ── notes: LLM "worse" signals per category + volume + priority, by entry date
    sig = inp.signals[inp.signals["animal_id"].isin(inp.animals["animal_id"])]
    for cat in CATEGORIES:
        g = _add_daily(g, sig[sig["category"] == cat], "available_date", **{f"_s_{cat}": ("weight", "sum")})
        g[f"n_{cat}_7d"] = _roll_sum(g, f"_s_{cat}", 7)
    g["n_neg_cats_7d"] = (g[[f"n_{c}_7d" for c in CATEGORIES]] > 0).sum(axis=1).astype(float)
    notes = inp.notes[inp.notes["animal_id"].isin(inp.animals["animal_id"])]
    g = _add_daily(g, notes.assign(prio=notes["priority"].astype(str).str.lower().map(PRIORITY).fillna(0)),
                   "available_date", _n=("weight", "sum"), _prio=("prio", "max"))
    g["n_notes_30d"] = _roll_sum(g, "_n", 30)
    prior = _roll_sum(g, "_n", 180) - g["n_notes_30d"]
    g["n_volume_ratio"] = (g["n_notes_30d"] + 1) / (prior / 5 + 1)
    g["n_max_priority_7d"] = _roll_max(g, "_prio", 7)

    # ── context
    a = inp.animals.set_index("animal_id")
    age_years = (g["date"] - g["animal_id"].map(a["birth_date"])).dt.days / 365.25
    g["ctx_age_known"] = age_years.notna().astype(float)
    g["ctx_age_years"] = age_years.fillna(0).clip(0, 80)
    g["ctx_juvenile"] = (age_years < 1).astype(float)
    sex = g["animal_id"].map(a["sex"])
    g["ctx_male"], g["ctx_female"] = (sex == "male").astype(float), (sex == "female").astype(float)
    tsn = g["animal_id"].map(a["taxonomy_id"])
    for sp in inp.species:
        g[f"ctx_sp_{int(sp)}"] = (tsn == sp).astype(float)
    st = inp.stays.assign(date=local_day(inp.stays["start"], inp.tz), animal_id=inp.stays["animal_id"].astype("int64"))[
        ["animal_id", "date"]].rename(
        columns={"date": "move_date"}).sort_values("move_date", kind="mergesort")
    moved = pd.merge_asof(g[["animal_id", "date"]].sort_values("date"), st, left_on="date", right_on="move_date",
                          by="animal_id", direction="backward").sort_values(["animal_id", "date"]).reset_index(drop=True)
    g["ctx_days_since_move"] = (g["date"] - moved["move_date"]).dt.days.fillna(365).clip(0, 365)

    # ── coverage → data confidence
    g["cov_sources"] = (g["w_has_baseline"] + (g["i_logged_30d"] >= icfg["min_logging_rate"])
                        + (g["n_notes_30d"] > 0)).astype(float)

    g = g[(g["date"] >= grid_start) & (g["date"] <= grid_end)].reset_index(drop=True)
    keep = ["animal_id", "date"] + feature_columns(g)
    return g[keep].astype({c: "float32" for c in feature_columns(g)})


# ── Labels ───────────────────────────────────────────────────────────────────────────────────────────────

def _next_gap(sorted_days: np.ndarray, d: np.ndarray) -> np.ndarray:
    """Days from d to the first event strictly after d (inf if none)."""
    i = np.searchsorted(sorted_days, d, side="right")
    nxt = np.where(i < len(sorted_days), sorted_days[np.minimum(i, max(len(sorted_days) - 1, 0))], 0) if len(sorted_days) else d
    return np.where(i < len(sorted_days), nxt - d, np.inf)


def label_days(grid: pd.DataFrame, episodes: pd.DataFrame, lcfg: dict) -> pd.DataFrame:
    """label_vet / label_all (1, 0 or NaN) and excl_reason for every animal-day. Uses cases only."""
    h, clear, buf = lcfg["horizon_days"], lcfg["negative_clear_days"], lcfg["treatment_buffer_days"]
    day = lambda s: s.dt.normalize().to_numpy().astype("datetime64[D]").astype(np.int64)  # noqa: E731
    ep = episodes.assign(t0_day=day(episodes["t0"]), last_day=day(episodes["last_t"]))
    by_animal = {k: v for k, v in ep.groupby("animal_id")}

    n = len(grid)
    label_vet, label_all = np.full(n, np.nan), np.full(n, np.nan)
    reason = np.full(n, "", dtype=object)
    dates = day(grid["date"])
    for animal_id, idx in grid.groupby("animal_id").indices.items():
        d = dates[idx]
        e = by_animal.get(animal_id)
        if e is None:
            gap_all = gap_vet = np.full(len(d), np.inf)
            treat = other = np.zeros(len(d), bool)
        else:
            cases = e[e["is_case"]]
            gap_all = _next_gap(np.sort(cases["t0_day"].to_numpy()), d)
            gap_vet = _next_gap(np.sort(cases.loc[cases["t0_source"] != "death", "t0_day"].to_numpy()), d)
            treat = np.zeros(len(d), bool)
            for r in e[e["n_records"] > 0].itertuples():
                treat |= (d >= r.t0_day) & (d <= r.last_day + buf)
            other = np.zeros(len(d), bool)
            for r in e[e["case_class"].isin(["trauma", "unknown_death"])].itertuples():
                other |= (d >= r.t0_day - clear) & (d <= r.t0_day)

        lv = np.where(gap_vet <= h, 1.0, np.where(gap_all > clear, 0.0, np.nan))
        la = np.where(gap_all <= h, 1.0, np.where(gap_all > clear, 0.0, np.nan))
        why = np.where(np.isnan(lv), np.where(gap_vet <= clear, "grey_zone", "death_only_case"), "")
        why = np.where(np.isnan(la) & (why == ""), "grey_zone", why)
        blocked = treat | other
        lv[blocked], la[blocked] = np.nan, np.nan
        why = np.where(treat, "treatment", np.where(other, "other_event", why))
        label_vet[idx], label_all[idx], reason[idx] = lv, la, why
    return pd.DataFrame({"label_vet": label_vet, "label_all": label_all, "excl_reason": reason}, index=grid.index)


# ── Leakage check ────────────────────────────────────────────────────────────────────────────────────────

def truncate_inputs(inp: Inputs, as_of: pd.Timestamp, animal_ids=None) -> Inputs:
    """What the system knew at the end of day `as_of`: drop every entry that became visible later."""
    end = as_of + pd.Timedelta(days=1)
    keep_animal = (lambda df: df[df["animal_id"].isin(animal_ids)]) if animal_ids is not None else (lambda df: df)
    w = keep_animal(inp.weights)
    known = w[["recorded_at", "created_at"]].max(axis=1)
    return replace(
        inp,
        animals=keep_animal(inp.animals),
        weights=w[known < end],
        intake=keep_animal(inp.intake)[lambda d: d["available_date"] <= as_of],
        notes=keep_animal(inp.notes)[lambda d: d["available_date"] <= as_of],
        signals=keep_animal(inp.signals)[lambda d: d["available_date"] <= as_of],
        stays=keep_animal(inp.stays)[lambda d: d["start"] < end],
    )


def leakage_check(inp: Inputs, cfg: dict, grid_start, grid_end, full: pd.DataFrame, n_animals=4, n_dates=4,
                  seed=0) -> int:
    """For random (animal, date D): features rebuilt from inputs truncated at D must equal the full-data features
    for every row ≤ D. Raises AssertionError on the first difference; returns the number of rows compared."""
    rng = np.random.default_rng(seed)
    cols = feature_columns(full)
    rich = full.groupby("animal_id")[["w_has_baseline", "n_notes_30d", "i_logged_30d"]].max().sum(axis=1)
    pool = rich[rich > 0].index.to_numpy()
    animals = rng.choice(pool if len(pool) else full["animal_id"].unique(), size=min(n_animals, len(full["animal_id"].unique())),
                         replace=False)
    full_a = full[full["animal_id"].isin(animals)].set_index(["animal_id", "date"])
    dates = full_a.index.get_level_values("date").unique().sort_values()
    compared = 0
    for as_of in rng.choice(dates[len(dates) // 4:], size=min(n_dates, len(dates)), replace=False):
        as_of = pd.Timestamp(as_of)
        trunc = build_features(truncate_inputs(inp, as_of, animals), cfg, grid_start, as_of)
        if trunc.empty:
            continue
        t = trunc.set_index(["animal_id", "date"])
        f = full_a.loc[t.index.intersection(full_a.index), cols]
        if not np.allclose(t.loc[f.index, cols].to_numpy(float), f.to_numpy(float), equal_nan=True, atol=1e-4):
            bad = (~np.isclose(t.loc[f.index, cols].to_numpy(float), f.to_numpy(float), equal_nan=True, atol=1e-4))
            col = cols[int(np.argwhere(bad.any(axis=0))[0][0])]
            raise AssertionError(f"Leakage: feature {col} changes when data after {as_of:%Y-%m-%d} is removed")
        compared += len(f)
    return compared


# ── Stage ────────────────────────────────────────────────────────────────────────────────────────────────

def load_inputs(cfg: dict) -> Inputs:
    raw = lambda n: store.read("raw", n)  # noqa: E731
    proc = lambda n: store.read("processed", n)  # noqa: E731
    animals = animal_table(raw("animals"), raw("taxonomy"), raw("mortality"))
    animals = animals[(animals["type"] == "single") & animals["taxonomy_id"].isin(cfg["species"])]
    ids = set(animals["animal_id"])

    notes = proc("notes_clean")
    notes = notes[notes["kept"] & notes["animal_id"].isin(ids)].assign(animal_id=lambda d: d["animal_id"].astype(int))
    sig = proc("notes_signals")
    sig = sig[(sig["method"] == "llm") & sig["verbatim_ok"] & (sig["direction"] == "worse")]
    sig = notes.merge(sig[["source", "source_id", "category", "evidence_quote"]], on=["source", "source_id"])
    sig = sig.rename(columns={"evidence_quote": "quote"})[["animal_id", "available_date", "category", "weight", "quote"]]
    stays = proc("stays")
    return Inputs(
        animals=animals[["animal_id", "taxonomy_id", "birth_date", "sex", "death_at"]],
        weights=proc("weights_clean")[lambda d: d["animal_id"].isin(ids)],
        intake=proc("intake_daily")[lambda d: d["animal_id"].isin(ids)],
        notes=notes[["animal_id", "available_date", "weight", "priority"]],
        signals=sig, stays=stays[stays["animal_id"].isin(ids)][["animal_id", "start"]],
        species=list(cfg["species"]), tz=cfg["timezone"])


def vet_data_start(cfg: dict) -> pd.Timestamp:
    start = cfg["labels"].get("start_date")
    if start:
        return pd.Timestamp(start)
    rec = store.read("processed", "case_records")
    return rec["t0"].min().normalize()


def run() -> None:
    cfg = config.load()
    if not cfg["species"]:
        raise RuntimeError("config species is empty — run `profile` first")
    inp = load_inputs(cfg)
    now = pd.to_datetime(pd.read_csv(store.output_path("extract_log.csv"))["extracted_at_utc"].iloc[0])
    grid_end = local_day(pd.Series([now]), cfg["timezone"]).iloc[0]
    grid_start = vet_data_start(cfg)

    feats = build_features(inp, cfg, grid_start, grid_end)
    episodes = store.read("processed", "cases")
    labels = label_days(feats, episodes, cfg["labels"])
    table = pd.concat([feats, labels], axis=1)

    compared = leakage_check(inp, cfg, grid_start, grid_end, feats)
    cols = feature_columns(table)
    bad = [c for c in cols if any(w in c for w in FORBIDDEN_IN_FEATURES)]
    assert not bad, f"label/medical-looking feature names: {bad}"
    store.write("processed", "animal_day", table)

    lab = table["label_vet"].notna()
    parts = [
        f"# Features — zoo {cfg['zoo_id']}\n",
        f"- Grid: {len(table):,} animal-days, {table['animal_id'].nunique():,} animals, "
        f"{table['date'].min():%Y-%m-%d} → {table['date'].max():%Y-%m-%d} (vet-data coverage starts {grid_start:%Y-%m-%d})",
        f"- Features: {len(cols)} model inputs ({', '.join(sorted({c.split('_')[0] + '_' for c in cols}))})",
        f"- **Leakage check passed**: features rebuilt from data truncated at random dates matched on "
        f"{compared:,} rows; no label/medical-looking column names",
        f"- Labelled rows (label_vet): {int(lab.sum()):,}; positives {int((table['label_vet'] == 1).sum()):,} "
        f"({100 * (table['label_vet'] == 1).sum() / max(lab.sum(), 1):.2f}% of labelled); "
        f"label_all positives {int((table['label_all'] == 1).sum()):,}",
        "- Excluded days: " + ", ".join(f"{k or 'labelled'} {v:,}" for k, v in table["excl_reason"].value_counts().items()),
        f"- Feature coverage: weight baseline on {100 * table['w_has_baseline'].mean():.1f}% of days, "
        f"notes in last 30 d on {100 * (table['n_notes_30d'] > 0).mean():.1f}%, "
        f"leftovers logged on {100 * (table['i_logged_30d'] > 0).mean():.2f}%",
    ]
    store.output_path("features_summary.md").write_text("\n".join(parts))
    print("\n".join(parts[1:]))
    print("→ data/processed/animal_day.parquet, outputs/features_summary.md")
