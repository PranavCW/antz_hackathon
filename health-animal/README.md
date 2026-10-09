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
| 3 | Feature table (+ leakage test), rule baseline vs logistic regression, backtest, alert explanations, LLM audit | ✅ (verified on Dev; numbers not meaningful) |
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
| `features` | processed + raw labels | `data/processed/animal_day.parquet`, `outputs/features_summary.md` | Animal-day feature table + labels; runs the **leakage test** and aborts if it fails |
| `model` | `animal_day` | `predictions.parquet`, `model_logreg.json`, `model_summary.json`, `outputs/model_report.md`, `model_coefficients.csv` | Rule baseline vs logistic regression; time split + grouped CV + ablation; M3 verdict |
| `backtest` | predictions + cases | `outputs/backtest_report.md`, `backtest_{sweep,cases,alerts}.csv` | Day-by-day replay: cases caught, lead time, alerts/week, missed-case reasons, alerts with explanations |
| `notes_audit` | gold + decoded notes | `outputs/notes_audit.md` | LLM and lexicon precision/recall per category vs the gold labels |

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

### 4.8 Features and labels (`src/features.py`, §H)
One row per (single animal of the chosen species, local day), 37 model inputs (`w_` weight, `i_` intake, `n_` notes,
`ctx_` context, `cov_` coverage). **Replay rule: a value exists only from the day it was entered** — weights from
`max(recorded_at, created_at)`, notes/leftovers from `available_date`; a spike is scored as *pending* on its own day
and only leaves later baselines once the next reading exists. **Leakage test** (runs on every `features` run): rebuild
the features from inputs truncated at random dates and require identical rows; the test is itself tested with a
deliberately peeking feature. No medical table can reach `build_features` (its signature has no parameter for one).
Labels (cases only): `label_vet` = 1 if a vet-aware case starts in (d, d+14], 0 if no case within 60 d, excluded
(NaN) for days under treatment (+7 d), the 15–60 d grey zone, near trauma/unknown deaths and death-only cases;
`label_all` also counts death-only cases (T0 = death, no vet record).

### 4.9 Model (`src/model.py`, §H)
- **Rule scorer (baseline)**: capped sum of firing rules with weights in `model.rule_weights`
  (weight drop, confirmed, leftover CUSUM, notes in 1 / 2+ categories, High/Critical priority *with* a worse note).
- **Logistic regression**: standardised features (clipped ±10, scale floor), L2, `class_weight=balanced`, C chosen by
  grouped CV on the training period.
- **Validation**: time split (newest 50% held out, 60-day purge so no training label looks into the test period);
  grouped 5-fold CV by animal over all history; bootstrap over animals for the PR-AUC difference vs the rules.
  Metric PR-AUC (prevalence shown as the random baseline).
- **Ablation + verdict**: the same model on *context/coverage only* vs *health signals only*. The logistic model is
  the primary scorer only if it beats the rules (95% interval above 0) **and** the health signals beat context
  (`NOT CREDIBLE` otherwise — it would be learning who gets observed, not who gets sick). The backtest reads
  `model_summary.json` to know which scorer is the result.
- Explanations (`src/explain.py`): logistic → top-3 positive contributions (coef × standardised value) as readable
  clauses; rules → the rules that fired; plus recent keeper quotes and a data-confidence line.
- Stretch (not built): LightGBM with monotonic constraints + SHAP.

### 4.10 Backtest (`src/backtest.py`, §I)
Replay: alert when risk ≥ threshold, 7-day cooldown per animal (re-alert if risk rose ≥ 50%), no alerts under
treatment. **Caught** = an alert in [T0 − 60 d, T0); **lead** = T0 − first such alert; **false alert** = no case
within 60 d → alerts/week/zoo. The **threshold is chosen on the older period with out-of-fold scores to maximise
cases caught under the alert budget, then applied unchanged to the newer period**. Reports a random-alert baseline at
the same volume, a missed-case analysis (`no_data` / `no_signal` / `below_threshold`), vet-aware vs death-only cases
separately, and every alert with its explanation. The headline sentence is generated from these numbers.

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
.venv/bin/python main.py features        # prints the leakage-test result; fails loudly on leakage
.venv/bin/python main.py model            # read the M3 verdict in outputs/model_report.md
.venv/bin/python main.py backtest         # headline in outputs/backtest_report.md
.venv/bin/python main.py notes_audit
.venv/bin/python -m pytest tests           # 151 tests: cleaning, housing, intake, notes, cases, detectors, LLM helpers, features/labels/leakage, model, backtest, explanations, read-only guard
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
10. **Features → model → backtest** — `features` (leakage test must pass), `model`, `backtest`. Read the M3
    verdict first: only `GO` means the logistic model is the result; `NOT CREDIBLE`/`NO-GO` means present the rule
    scorer and say the model needs more data. Agree the alert budget (`alerting.max_alerts_per_week`) with the vets
    *before* looking at the sweep. If `labels.start_date` is null the grid starts at the first vet record in the
    extract — set it if the zoo only started recording medical data later than animals.
11. **Sprint 4** — Streamlit timeline demo + slides (planned).

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
| `alerting` | cooldown 7 d, rise factor, alert budget (placeholder 3/week/zoo) |
| `features`, `model` | weight freshness 60 d; target, test fraction, min training positives, `rule_weights` |

## 9. Known limitations (from the Dev data)
- Dev DB is QA data: ~⅓ of notes are test strings; zoo-created weight types/units are junk; weights jump between
  units at random. M1/M2 "GO" results on it are mechanical, not evidence.
- Food wastage is nearly empty (15 of 246 rows usable) → the leftover detector cannot be evaluated here.
- 826 of 1,611 cases are deaths with no prior vet record (T0 = death) — no vet-data lead time is possible; they'll
  be reported separately in the backtest.
- Clean drops spikes using the *next* reading; in the backtest replay that reading wasn't known yet, so a spike
  should appear as "pending" until then — handled in the Sprint 3 leakage work.
- **Backtest on Dev**: 0 of 42 vet-aware held-out cases flagged — 33 have no weight/note/leftover data in their
  60-day early window, 9 had data but no signal. The M3 verdict is `NOT CREDIBLE` (the logistic model's edge comes
  from context/coverage, not health signals). These are the machinery working correctly on data that has no signal.
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
│   ├── profile.py, cases.py, detectors.py, notes_llm.py, notes_gold.py, notes_audit.py
│   └── features.py, model.py, explain.py, backtest.py
├── tests/               pytest suites + data/notes_sample.json (synthetic LLM check set)
├── docs/                PLAN.md, SPRINT_PLAN.md
├── data/raw, data/processed, outputs/   (git-ignored)
└── notebooks/
```
