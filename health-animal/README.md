# Health Animal — Early Health Warning (Antz Hackathon)

Offline Python prototype that flags zoo animals drifting from **their own normal** (weight, feeding leftovers,
keeper notes) **before a vet gets involved**, with alerts that explain themselves, validated by replaying
history: *"flags X of Y past cases, on average Z days earlier, at ≤ N alerts/week/zoo."*

- Design and data dictionary: [docs/PLAN.md](docs/PLAN.md) (section refs like §1.2, B1, H point there)
- Sprint plan, decisions log: [docs/SPRINT_PLAN.md](docs/SPRINT_PLAN.md)

---

## 1. Status

| Sprint | Scope | State |
|---|---|---|
| 0 | DB connection (read-only), config, LLM check, stage stubs | ✅ |
| 1 | Schema check, extract, cleaning (weights, housing, intake, notes), data-quality report, species choice | ✅ |
| 2 | Cases + T0, weight & leftover detectors, notes LLM batch, gold sheet | ✅ |
| 3 | Feature table (+ leakage test), rule baseline vs logistic regression, backtest | ⏳ next |
| 4 | Streamlit timeline demo, slides | ⏳ |

> ⚠️ **Currently running on Dev test data.** `admin_zoolive` zoo 11 is the QA zoo: test weight types/units,
> lorem-ipsum notes, random weights. The code is real and tested; **numbers produced from this DB are not
> meaningful**. Everything is built so that pointing it at a real zoo snapshot is a config + rerun job
> (see [§7](#7-running-on-real-world-data)).

> 📝 **The 50 gold labels for the LLM audit are provisional.** `data/processed/notes_gold.csv` was labelled
> by Claude (every row's `labeller_notes` starts with `claude-provisional`) because the Dev notes are test
> text. With real data, a **person** must relabel a fresh sheet — see step 9 of §7.

---

## 2. Architecture

```
            ┌──────────────── Antz MySQL (read-only) ────────────────┐
            │  only db_check · schema_check · extract connect here   │
            └───────────────────────────┬────────────────────────────┘
                                        │ extract (once per zoo)
                                        ▼
                           data/raw/*.parquet  (39 sources, as stored)
                                        │ clean
            ┌──────────────┬────────────┼──────────────┬────────────────┐
            ▼              ▼            ▼              ▼                ▼
      weights_clean     stays +     intake_daily   notes_clean     (raw labels)
      (§1.2, B1)     enclosure_day  (§1.3, B3)    (§1.4, B4)           │
            │          (§1.6)          │              │                │ cases
            │                          │              │ notes_llm      ▼
            │ detectors                │ detectors    ▼             cases.parquet
            ▼                          ▼          notes_signals    (episodes, T0)
      detector_weight            detector_intake  (LLM + lexicon)       │
            └──────────────┬───────────┴──────────────┘                │
                           ▼  features (Sprint 3: only data ≤ day d)    │
                    animal-day table ──► model (rules / logistic) ◄─────┘ labels
                                                 │
                                                 ▼ backtest (replay day by day)
                               X of Y cases · lead days · alerts/week
```

**Two layers** (PLAN §A):
1. **Detectors** — statistics per animal against its own history ("Raja usually leaves ~200 g; this week 1.5 kg").
   Self-comparison makes a tiger and a flamingo comparable.
2. **Prediction model** — takes the detector outputs for an animal-day and returns *P(vet case starts in the next
   14 days)*. One zoo has too few cases to learn "normal" per animal from raw data, but enough to learn how much
   each kind of deviation matters.

**Design rules**
- **The DB is never written.** Three independent guards: (1) `src/readonly.py` rejects anything that isn't a single
  SELECT/SHOW/DESCRIBE/EXPLAIN/WITH before it is sent (incl. `INTO OUTFILE`, locking reads, `/*! */` comments,
  multi-statements); (2) every session is `SET SESSION TRANSACTION READ ONLY`, verified on connect; (3) ask for a
  read-only DB user. All access goes through `src.db.get_engine()`. Everything after `extract` runs from parquet.
- **Never silently trust, never silently invent.** Every dropped or corrected row keeps a `drop_reason` / `flags`;
  missing stays missing (a day without a leftover entry is unknown, never "ate everything").
- **No future data.** Every detector value for time *t* uses only rows before *t*.
- **Medical tables are labels only**, never model inputs.
- **No proprietary LLM API.** Notes are decoded by an open-source model on the local machine (Ollama).

---

## 3. Pipeline stages

Run with `python main.py <stage>` (from `health-animal/`, inside `.venv`). `python main.py` lists them.

| Stage | Reads | Writes | What it does |
|---|---|---|---|
| `db_check` | DB | — | Who/where am I connected, write grants, key tables, zoo animal count |
| `llm_check` | Ollama | — | Models pulled, schema-forced JSON works |
| `schema_check` | DB | `outputs/schema_check.md`, `outputs/schema/create_tables.sql` | Live columns vs PLAN §1 dictionary + 21 DISTINCT audits of coded values |
| `extract` | DB | `data/raw/*.parquet`, `outputs/extract_log.csv` | 39 sources scoped to `ZOO_ID`; zero dates → NaT (counted per column) |
| `clean` | raw | `data/processed/{weights_clean, species_weight_bands, stays, enclosure_day, wastage_clean, intake_daily, intake_coverage, notes_clean}`, `outputs/data_quality_rows.csv` | All §1/B cleaning rules |
| `profile` | raw + processed | `outputs/data_quality.md`, `profile_species.csv`, `profile_animals.csv`, `species:` in `config.yaml` | Data density per species, monitorability, M1 go/no-go, picks 3–5 species |
| `cases` | raw | `data/processed/{cases, case_records}`, `outputs/cases_summary.md` | Illness episodes, T0, estimated onset, outcome |
| `detectors` | processed | `data/processed/{detector_weight, detector_intake}`, `outputs/detector_cases.{md,csv}`, `outputs/detectors/*.png` | Layer 1 + did it fire before T0 on known cases (M2) |
| `notes_llm` | processed + Ollama | `data/processed/{llm_cache.jsonl, notes_decoded, notes_signals}`, `outputs/notes_llm_progress.log` | Decode notes of chosen species; `--sample` = 20-note correctness check; `--limit N` |
| `notes_gold` | processed | `data/processed/notes_gold.csv` | 50-note hand-labelling sheet (never overwrites a labelled sheet) |
| `features`, `model`, `backtest` | — | — | Sprint 3 (stubs) |

`data/` and `outputs/` are git-ignored: they hold zoo data and must never be committed.

---

## 4. How each part works

### 4.1 Weights (`src/clean_weights.py`, PLAN §1.2, B1–B2)
Sources: `antz_animal_assessments` (value/unit from the child `antz_animal_assessment_values` when present),
legacy `antz_animal_measurement` (rows not already copied into assessments), pre-anaesthesia *Actual* weights.
1. **Parse** text values safely (`"12,5"` → 12.5, `"12 kg"` → 12 + unit kg; a typed unit overrides the selected one).
2. **Grams ourselves**: unit factor if its base is gram, else one hop through the base unit; otherwise unresolvable
   → dropped. The stored `base_uom_value` is only a cross-check (`base_zero`, `base_mismatch` flags).
3. **Timestamp**: `recorded_date_time` → `record_date + record_time` → `created_at`; converted to the zoo-local day.
4. **Drop**: groups, impossible dates (future, < 2000, before birth, after death), exact duplicates, same weighing
   written twice within ±1 min.
5. **Unit slips** (kg↔g ×1000, kg↔lb ×2.2046): ratio to the median of the animal's other readings within ±180 d,
   ±5% — only when that reference is stable (≥75% of it within ±30%), otherwise we'd "fix" toward noise.
6. **Species band**: outside [1st pct × 0.5, 99th pct × 1.5] of adult readings in the zoo → typo, dropped
   (needs ≥10 readings from ≥3 animals; juveniles exempt).
7. **Hampel filter + confirmation**: > 3 robust SD from the median of ±3 neighbours is a candidate. Next reading
   back to normal → **spike, dropped**; neighbour deviates the same way → **confirmed (real change, kept)**;
   last reading → **pending confirmation**.

### 4.2 Housing (`src/housing.py`, §1.6)
`antz_enclosure_logs.out_date` is unreliable (ON UPDATE timestamp, not closed at death). A stay ends at the earliest
of out_date, the animal's next stay, death, or extract time. Repeated visits to the same enclosure → low confidence.
An **interval sweep** per enclosure gives `enclosure_day(enclosure, date, n_animals, animal_id_if_solo,
confidence)`; an enclosure-day is single-occupant iff the summed `total_animal` present that day is 1.

### 4.3 Intake (`src/intake.py`, §1.3, B3)
There is no historical planned diet, so we use **leftover grams relative to the animal's own baseline**:
food-wastage entries (grams recomputed from the entered unit, UTC → local day) on single-occupant, high-confidence
enclosure-days only. Diet changes (`antz_diet_assign_logs`, animal or species level) start a new segment (baseline
restarts). Coverage per animal × segment; logging rate < 30% → `low_confidence`.

### 4.4 Notes (`src/clean_notes.py`, §1.4, B4)
Sources: observations + their comment thread, food-wastage notes, weighing comments, incident notes.
Attribution: animal tag → that animal (weight 1/n animals tagged); enclosure tag → only the single occupant that
day; section/site tags ignored. `created_for` edit bug fixed (edits reset it to the edit date). Dropped: empty,
template ("all ok"), copy-paste duplicates (same animal + text within 3 days), notes whose types are all maintenance.

### 4.5 Cases and T0 (`src/cases.py`, §1.5, §I) — labels only
- Standard-case medical records (`case_type` = standard, `created_for = medical`) per animal (legacy `animal_id` or
  active mapping; records on > 5 animals are batch treatments and skipped). Complaint/diagnosis items apply to the
  animal if they name it or name no animal.
- Records ≤ 14 days apart merge into one **episode**. **T0** = earliest of record date, complaint, diagnosis,
  hospital admission. **Estimated onset** = T0 − complaint duration ("since 3 days").
- **Case** if serious: severity ≥ High, prognosis ≥ poor, emergency/inpatient/critical, or an illness death.
  Euthanasia counts only after an illness record within 60 days; trauma deaths are reported separately; deaths
  without any record become their own episode (T0 = death).
- Coded values are normalised (severity `Mildd`/`Medium`/`Low`…, prognosis `favorable`, manner-of-death matched
  case-insensitively).

### 4.6 Detectors (`src/detectors.py`, §C–F)
| Signal | Algorithm | Fires when |
|---|---|---|
| Weight level | **Median** of the previous 180 d (≥3 readings, ≥1 in last 60 d, same day excluded) → % change | ≤ −5% (adults) |
| Weight trend | **Theil-Sen** slope (median of pairwise slopes; one bad weighing can't fake a trend) in %/month | ≤ −3 %/month; juveniles: growth ≤ 0 |
| Confirmation | previous reading deviated the same way | `confirmed` vs `pending` |
| Leftovers today | **Robust z** = (x − median) / (1.4826 × MAD) over the previous 30 d of the same diet segment; weekday median when ≥3 same-weekday days (absorbs fasting days); MAD floor 10 g | z > 3 = high day |
| Leftovers sustained | **CUSUM** S = max(0, S + z − 0.5); missing day holds S, > 7 missing days resets | S ≥ 4 |

Why median/MAD: one bad value can't move them (`96,94,97,95,20,…` → mean 87.7, median 95).

### 4.7 Notes LLM (`src/notes_llm.py`, §G)
- **Model**: Qwen2.5-7B-Instruct via **Ollama** on the local machine (fallback Qwen2.5-3B), temperature 0.
- **Output** forced by a JSON schema: `signals[{category, direction (worse/normal/better), severity 1–3,
  evidence_quote}]`, `is_health_related`. Categories: appetite, mobility, behaviour, gi, respiratory, skin_injury,
  body_condition, other.
- **Prompt** (`PROMPT_VERSION = "v1"`): rules + 6 few-shots (negation "no limping" → normal, Hinglish, maintenance
  → nothing, improving → better). Context per note: species, note type, date.
- **Safeguards**: verbatim-quote check (rapidfuzz partial ratio ≥ 90, else the signal is not used); keyword
  lexicon in parallel (fallback + disagreement log; negation only within the same clause); cache keyed on the
  exact prompt + model (append-only `llm_cache.jsonl` → resumable, reruns free). The LLM only extracts, never
  diagnoses.
- Measured: 19/20 on the synthetic check set (`tests/data/notes_sample.json`); Dev batch 464 notes in ~5 min.

### 4.8 Prediction model and backtest (Sprint 3–4, planned — PLAN §H–I)
- **Rows**: one per animal-day of the chosen species. **Label** 1 if a case's T0 falls in (d, d+14]; 0 if none
  within 60 d; days under treatment / grey zone / after death excluded.
- **Features**: weight % change, slope, confirmed, days since last weight; leftover z, CUSUM, high days in 7,
  coverage; note signals per category over 7 d, # negative categories, note volume, max priority; species,
  age class, sex, days since move. **Leakage test**: every feature for day d uses rows ≤ d; no medical tables.
- **Models**: (1) rule-based scorer (baseline); (2) **regularised logistic regression** with class weights (main —
  readable contributions → "top 3 reasons"); (3) **LightGBM with monotonic constraints** + SHAP (stretch).
- **Validation**: time-based split (older → train, newer → test) + leave-one-case-out; metric **PR-AUC**.
  The model must beat the rules on the held-out period.
- **Backtest**: replay day by day, alert at risk ≥ threshold with a 7-day cooldown; lead time = T0 − first alert in
  [T0 − 60 d, T0); false alerts → alerts/week/zoo; threshold sweep under the alert budget; missed-case analysis.

---

## 5. Setup

```bash
cd health-animal
python3 -m venv .venv && .venv/bin/pip install -r requirements.txt
cp .env.example .env                       # DB credentials, ZOO_ID, ZOO_TIMEZONE, LLM settings
brew install ollama && ollama serve        # separate terminal
ollama pull qwen2.5:7b-instruct && ollama pull qwen2.5:3b
```
Tuning (species, windows, thresholds) lives in `config/config.yaml`; connection details and the zoo live in `.env`.

## 6. Running (current pipeline)

```bash
.venv/bin/python main.py db_check          # connection sanity
.venv/bin/python main.py llm_check         # Ollama sanity
.venv/bin/python main.py schema_check      # review outputs/schema_check.md
.venv/bin/python main.py extract           # the only data pull from the DB
.venv/bin/python main.py clean
.venv/bin/python main.py profile           # review outputs/data_quality.md; species → config.yaml
.venv/bin/python main.py cases             # review outputs/cases_summary.md
.venv/bin/python main.py detectors         # review outputs/detector_cases.md + outputs/detectors/*.png
.venv/bin/python main.py notes_llm --sample
.venv/bin/python main.py notes_llm         # batch; tail -f outputs/notes_llm_progress.log
.venv/bin/python main.py notes_gold
.venv/bin/python -m pytest tests           # 127 tests: cleaning rules, housing, intake, notes, cases, detectors, LLM helpers, read-only guard
```

---

## 7. Running on real-world data

A checklist for switching from the Dev test zoo to a real zoo. Steps 1–2 are prerequisites; the rest is a rerun
with a review at each checkpoint.

1. **Access** — confirm we're allowed to pull keeper/medical data locally, and get a **read-only DB user** (or a
   restored dump). Put it in `.env`; set `ZOO_ID` and `ZOO_TIMEZONE`. `db_check` should print "User is read-only ✓".
2. **Start clean** — move the old Dev outputs away so nothing mixes:
   `mv data/raw data/raw_dev && mv data/processed data/processed_dev && mv outputs outputs_dev` then
   `mkdir -p data/raw data/processed outputs`. (The LLM cache is keyed on the prompt, so it's safe either way.)
3. **Schema** — `schema_check`; any `MISMATCH` or new coded values (severity, prognosis, manner of death,
   units, note types) → fix the mapping in code/config **and** update PLAN.md §1 before extracting.
4. **Config review** (`config/config.yaml`), using `outputs/schema_check.md`:
   - `weight.extra_assessment_type_ids`: add real zoo-created weight types (e.g. "Body Weight").
   - `notes.skip_note_types`: maintenance parent types for this zoo.
   - `manner_of_death`: map any new string_ids.
   - `species: []` — clear it so `profile` picks again.
5. **Extract + clean** — `extract`, `clean`. Check `outputs/data_quality_rows.csv`: high shares of
   `unit_unresolvable`, `spike` or `outside_species_band` usually mean a unit/type mapping to fix, not bad data.
6. **Profile (M1)** — `profile`; read `outputs/data_quality.md`. Needs ≥3 species with ≥10 cases on monitorable
   animals. If not: widen species or lower `labels.min_severity`. If intake-monitorable animals ≈ 0, drop the
   leftover detector (cut order #3).
7. **Cases** — `cases`; **spot-check 3–5 episodes in the Antz UI** (T0 date, severity, outcome).
8. **Detectors (M2)** — `detectors`; look at the plots: do weights drop before T0 on known cases? Tune thresholds
   (`detectors.*`) only after checking coverage.
9. **Notes** — `notes_llm --sample` (expect ≥ 18/20), then `notes_llm`. Then the LLM audit with **human** labels:
   `mv data/processed/notes_gold.csv data/processed/notes_gold_provisional.csv`, run `notes_gold`, and have a
   keeper/vet-literate person fill the 50 rows. If the real notes are in other languages, add few-shots and bump
   `PROMPT_VERSION`.
10. **Sprint 3–4** — features (leakage test must pass), model vs rules on the held-out period, backtest, demo.
    Agree the alert budget (`alerting.max_alerts_per_week`) with the vets before the threshold sweep.

---

## 8. Configuration reference (`config/config.yaml`)

| Section | Key knobs |
|---|---|
| `species` | taxonomy tsn list, written by `profile` |
| `weight` | baseline 180 d / ≥3 readings / ≥1 in 60 d; species band; Hampel window & sigmas; unit-slip tolerance; `extra_assessment_type_ids` |
| `intake` | baseline 30 d / ≥10 days; min logging rate 0.3; CUSUM k 0.5, h 4, reset after 7 missing days |
| `detectors` | z threshold 3; weight alert −5% / −3 %/month; juvenile slope; leftover MAD floor |
| `notes` | categories; skip types; template texts; duplicate window; evidence fuzzy minimum 90 |
| `labels` | horizon 14 d; negatives clear 60 d; episode merge 14 d; euthanasia lookback 60 d; min severity/prognosis; max animals per record |
| `manner_of_death` | string_id → illness / euthanasia / trauma / unknown / junk |
| `profile` | window 730 d; min monitorable animals; 3–5 species |
| `alerting`, `model` | cooldown, alert budget; time-split fraction, seed |

## 9. Known limitations (from the Dev data)
- Dev DB is QA data: ~⅓ of notes are test strings; zoo-created weight types/units are junk; weights jump between
  units at random. M1/M2 "GO" results on it are mechanical, not evidence.
- Food wastage is nearly empty (15 of 246 rows usable) → the leftover detector cannot be evaluated here.
- 826 of 1,611 cases are deaths with no prior vet record (T0 = death) — no vet-data lead time is possible; they'll
  be reported separately in the backtest.
- Clean drops spikes using the *next* reading; in the backtest replay that reading wasn't known yet, so a spike
  should appear as "pending" until then — handled in the Sprint 3 leakage work.
- One zoo has few cases → wide uncertainty; the production path is a multi-zoo pooled model (PLAN §5).

## 10. Layout
```
health-animal/
├── main.py              stage dispatcher (--limit, --sample for notes_llm)
├── config/config.yaml   tuning
├── src/
│   ├── db.py, readonly.py              read-only engine + SQL guard
│   ├── config.py, store.py, animals.py  config/paths, parquet IO, shared animal helpers
│   ├── db_check.py, llm_check.py, schema_check.py, extract.py
│   ├── clean.py → clean_weights.py, housing.py, intake.py, clean_notes.py
│   ├── profile.py, cases.py, detectors.py, notes_llm.py, notes_gold.py
│   └── features.py, model.py, backtest.py     (Sprint 3 stubs)
├── tests/               pytest suites + data/notes_sample.json (synthetic LLM check set)
├── docs/                PLAN.md, SPRINT_PLAN.md
├── data/raw, data/processed, outputs/   (git-ignored)
└── notebooks/
```
