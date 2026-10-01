# Early Health Warning — Sprint Plan

Companion to [PLAN.md](PLAN.md). PLAN.md says **what** we build and **why**; this file says **who does what,
in which order, and when it counts as done**. Section references (§1.2, B1, H …) point into PLAN.md.

## 0. Frame

| Item | Value |
|---|---|
| Duration | 2 days, split into 1 setup sprint + 4 half-day sprints (~4 h each) |
| Team | 1–2 people. **Track A** = data & model (critical path). **Track B** = notes LLM & demo app |
| Solo mode | One person runs Track A in full. Track B shrinks to: kick off the LLM batch (S2), then the Streamlit timeline (S4) |
| Input | Dev DB (read-only), one zoo (`ZOO_ID` in `.env`) extracted once to parquet. Offline Python only, no proprietary LLM API |
| Headline deliverable | *"Flags X of Y past cases, on average Z days earlier, at ≤ N alerts/week/zoo"* + a timeline demo + slides |
| Sprint goal = | the one thing that must be true at the end of the sprint. If it is at risk, cut scope, not the goal |

### Milestones & checkpoints
| # | When | Checkpoint | Go / no-go question |
|---|---|---|---|
| M0 | Day 1, start | Dump restored, schema check green | Can we query every §1 table? |
| M1 | Day 1, lunch | Cleaned series + data-quality report, species picked | Is there enough data for ≥ 3 species with ≥ 10 cases total? |
| M2 | Day 1, end | Cases/T0 + weight detector running; LLM batch running overnight | Do detectors fire before T0 on ≥ 1 known case? |
| M3 | Day 2, lunch | Model trained, first backtest number | Does the model beat the rule baseline on the held-out period? |
| M4 | Day 2, end | Demo + slides | Can we tell the story in 5 minutes? |

---

## 1. Sprint 0 — Setup (Day 1, first ~1 h)
**Goal:** everyone can run `python main.py` against the Dev DB.

| ID | Task | Track | Est. | Done when |
|---|---|---|---|---|
| S0-1 | ~~Docker MySQL + dump restore~~ → connect to Dev DB via `.env` (read-only, UTC sessions; optional SSH tunnel) | A | 30 m | `python main.py db_check` lists key tables + zoo animal count |
| S0-2 | `requirements.txt`, `.venv`, `config/config.yaml` (windows, thresholds, alert budget) + `src/config.py` loader (merges `.env`) | A | 20 m | `pip install -r requirements.txt` clean; config loads |
| S0-3 | Install Ollama, pull `qwen2.5:7b-instruct` (fallback `qwen2.5:3b`); JSON-schema smoke call | B | 30 m | `python main.py llm_check` returns valid JSON with verbatim quotes |
| S0-4 | `main.py` dispatches stage names → `src/<stage>.py` `run()` (stubs OK) | B | 15 m | `python main.py schema_check` reaches the stub |

Setup for a new machine: `python3 -m venv .venv && .venv/bin/pip install -r requirements.txt`,
`cp .env.example .env` (fill in), `brew install ollama && ollama serve`, `ollama pull qwen2.5:7b-instruct`.

**Open from S0:** confirm which environment `admin_zoolive` is and that pulling keeper/medical data locally
is allowed; get a read-only DB user (current one can write).

---

## 2. Sprint 1 — Trustworthy data (Day 1 AM, ~4 h) → **M1**
**Goal:** cleaned per-animal daily series + data-quality report, and a chosen species list.

| ID | Task | PLAN ref | Track | Est. | Done when |
|---|---|---|---|---|---|
| S1-1 | `00_schema_check`: `SHOW CREATE TABLE` for all §1 tables, diff vs dictionary; `SELECT DISTINCT` audits (severity, prognosis, manner_of_death `string_id`, weight units, note types) | §1 note, §3 | A | 45 m | Diff report in `outputs/`; PLAN.md updated for any mismatch |
| S1-2 | `01_extract`: SQL → parquet in `data/raw/` for animals, weights (+legacy), units, wastage, diet logs, enclosure logs + movements, notes (+comments, mapping, types), medical, hospital, mortality | §1.1–1.6 | A | 60 m | One parquet per source, row counts logged |
| S1-3 | `02_clean` weights: unit normalisation to grams, safe-parse, timestamp rule, legacy/assessment dedup, B1 rules (kg↔g, kg↔lb, species band, Hampel + confirmation) | §1.2, B1–B2 | A | 75 m | Unit tests for each B1 row pass (§6) |
| S1-4 | Housing: rebuild stays, cap at death, single-occupant enclosure-days, low-confidence stays | §1.6 | B | 45 m | Table `enclosure_day(enclosure, date, n_animals, animal_id_if_solo, confidence)` |
| S1-5 | `02_clean` intake + notes: wastage on solo days, diet-change segments, logging-rate confidence; note attribution (animal / solo enclosure), `created_for` fix, mapping `is_deleted` filter, dedup by text hash, maintenance types skipped | §1.3–1.4, B3–B4 | B | 60 m | Clean notes table ready for the LLM; intake daily series per solo animal |
| S1-6 | Data-quality report + `03_profile`: per species weights/90 d, % solo days logged, notes/week, % dropped/corrected, monitorable yes/no | B6, §3 | A | 30 m | Report in `outputs/`; **3–5 species chosen and written to config** |

**Checkpoint M1.** If < 10 usable cases overall, widen species or lower case-severity bar (§I.1) now, not later.

---

## 3. Sprint 2 — Cases & signals (Day 1 PM, ~4–5 h) → **M2**
**Goal:** labelled episodes with T0, working weight + leftover detectors, and the LLM batch running overnight.

| ID | Task | PLAN ref | Track | Est. | Done when |
|---|---|---|---|---|---|
| S2-1 | `04_cases`: illness episodes (case filters, 14-day merge), T0 = LEAST(...), estimated onset, outcome; mortality row selection + manner-of-death classes; euthanasia rule | §1.5, §I.1–2, B5 | A | 75 m | Episode table; spot-check 3 cases in Antz UI |
| S2-2 | `05_detectors` weight: % change vs 180 d median, Theil-Sen slope %/month, confirmed/pending, `insufficient_data` | C, F | A | 60 m | Plot for 3 known cases shows the detector firing (or not) before T0 |
| S2-3 | `05_detectors` leftovers: robust z (MAD floor, weekday baseline), CUSUM (k 0.5, h 4, hold/reset rules), coverage flag | C–E, B3 | A | 60 m | Unit test reproduces the PLAN.md CUSUM example (flags on day 5) |
| S2-4 | `06_notes_llm`: prompt + few-shots (incl. negation), JSON schema, temperature 0, verbatim-evidence check (rapidfuzz ≥ 90), cache by `observation_id` + text hash, keyword lexicon in parallel | G | B | 90 m | 20 sample notes decode correctly; cache hit on rerun |
| S2-5 | **Start LLM batch** on all notes of chosen species (resumable, logs progress) | G | B | 10 m | Batch running before leaving; ETA logged |
| S2-6 | Hand-label ~50 notes for the LLM audit (can run while batch runs) | §6 | B | 45 m | `data/processed/notes_gold.csv` |

**Checkpoint M2.** If no detector fires before T0 on any known case, inspect data coverage for those animals
before tuning thresholds.

---

## 4. Sprint 3 — Model & backtest (Day 2 AM, ~4 h) → **M3**
**Goal:** the headline number from a model that beats the rule baseline on held-out data.

| ID | Task | PLAN ref | Track | Est. | Done when |
|---|---|---|---|---|---|
| S3-1 | Check LLM batch: failures, verbatim pass rate, precision/recall vs the 50 gold notes | G, §6 | B | 30 m | Audit table per category |
| S3-2 | `07_features`: animal-day table (weight, intake, notes, context, coverage); labels (1 / 0 / excluded grey zone and treatment days) | H | A | 75 m | **Leakage test passes**: every feature for day d uses rows ≤ d, no medical tables in features |
| S3-3 | `08_model`: rule-based scorer; logistic regression (regularised, class weights); time split + leave-one-case-out; PR-AUC | H | A | 60 m | Held-out PR-AUC logistic vs rules reported |
| S3-4 | `09_backtest`: day-by-day replay, alert cooldown, lead time, false alerts → alerts/week; threshold sweep; missed-case analysis | I | A | 60 m | Headline sentence filled in with X, Y, Z, N |
| S3-5 | Explanations: top-3 reasons (logistic contributions) + note quotes + data confidence per alert | H | B | 45 m | Reason strings generated for every backtest alert |
| S3-6 | *Stretch:* LightGBM with monotonic constraints + SHAP | H | A | 45 m | Only if S3-4 done by 11:30 |

**Checkpoint M3.** If the model doesn't beat the rules, present the rule scorer as the result and the model as
"needs more zoos" (honest-limits slide). Don't spend Day 2 PM tuning.

---

## 5. Sprint 4 — Demo & story (Day 2 PM, ~4 h) → **M4**
**Goal:** a 5-minute demo that shows one caught case end to end and the headline number.

| ID | Task | PLAN ref | Track | Est. | Done when |
|---|---|---|---|---|---|
| S4-1 | `app.py` Streamlit: animal picker → timeline (weight, leftovers, notes, risk score, alerts, T0 marker), alert card with reasons | §3 | B | 120 m | Runs locally from parquet; no DB needed |
| S4-2 | Pick 2 hero cases (1 caught early, 1 missed with reason) + 1 false alert | I.6 | A | 30 m | Screens captured |
| S4-3 | Slides: problem, data reality (data-quality findings), method (2 layers), headline, hero cases, honest limits, production path (§5), product gaps | §4–5 | A | 90 m | Deck reviewed once end to end |
| S4-4 | Verification sweep (§6): unit tests green, spot-check 3–5 animals vs Antz UI, README run instructions | §6 | A+B | 30 m | Checklist below ticked |
| S4-5 | Dry run + timing | — | A+B | 20 m | ≤ 5 min |

---

## 6. Cut order (when behind)
Cut from the top; never cut the backtest.
1. LightGBM + SHAP (S3-6)
2. Note student model (TF-IDF / small transformer, PLAN §G stretch — not scheduled)
3. Leftover detector (S2-3) → features become weight + notes only
4. LLM on notes → keep keyword lexicon only
5. Streamlit app → static matplotlib timelines in slides

## 7. Risks
| Risk | Likelihood | Impact | Mitigation | Owner |
|---|---|---|---|---|
| Live schema differs from dictionary | High | High | S1-1 first; update PLAN.md before extracting | A |
| Too few illness cases (< 10) | Medium | High | Decide at M1: widen species / severity bar; report with wide CIs | A |
| LLM too slow or bad JSON on laptop | Medium | Medium | 3B model fallback; volume filters (§G); lexicon fallback | B |
| Leftover data too sparse on solo enclosures | High | Low | `low_confidence` flag; cut order #3 | A |
| Leakage inflates results | Medium | High | Leakage test is a done-criterion of S3-2 | A |
| DB is live/shared (`admin_zoolive`, write-capable user) | Medium | High | Sessions forced read-only; extract once to parquet; ask for read-only user | A |
| Feeding-wastage data near-empty (~130 rows all zoos) | High | Medium | Decide at M1 whether to cut the leftover detector (cut order #3) | A |

## 8. Definition of done (every task)
- Code in `src/`, runnable via `main.py --stage <name>`, reads `config/config.yaml`.
- Outputs written to `data/processed/` or `outputs/`, never back to the DB.
- **All DB access goes through `src.db.get_engine()`** — it is read-only by construction (`src/readonly.py`
  guard + server-side READ ONLY session). Never open a pymysql/SQLAlchemy connection any other way.
- Every dropped or corrected row carries a reason flag (feeds the data-quality report).
- Any finding that contradicts PLAN.md is fixed in PLAN.md the same sprint.

## 9. Final verification checklist (from PLAN §6)
- [ ] Schema check passes (or dictionary updated)
- [ ] Unit tests: kg/g fix, lb fix, spike dropped, confirmed drop kept, missing ≠ 0 leftover, legacy/assessment dedup, CUSUM example
- [ ] 3–5 animals' cleaned weight and note timelines spot-checked against Antz UI
- [ ] Leakage test green
- [ ] LLM audit: precision/recall per category, verbatim pass rate
- [ ] Held-out PR-AUC: model vs rule baseline
- [ ] Backtest: X/Y cases, mean/median lead days, alerts/week/zoo

## 10. Daily stand-up (5 min, start of each sprint)
Done since last sprint · plan for this sprint · blocked on · cut anything? Record decisions below.

### Decision log
| Date | Decision | Why |
|---|---|---|
| | | |
