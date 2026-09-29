# Early Health Warning — Hackathon Plan (1–2 days, 1–2 people, Python offline, DB dump)

## ▶ Immediate step (only this, nothing else)
Outside the web-app-ci4 repo, create the skeleton at `/Users/pranav/Code/antz-hackathon/health-animal/`:
```
antz-hackathon/
└── health-animal/
    ├── main.py              entry point (argparse stub: lists pipeline stages, no logic yet)
    ├── README.md            one-paragraph description + how to run main.py
    ├── requirements.txt     empty placeholder
    ├── config/              (.gitkeep)
    ├── data/raw/            (.gitkeep)   DB dump extracts go here
    ├── data/processed/      (.gitkeep)
    ├── src/                 (.gitkeep)   pipeline modules later
    ├── notebooks/           (.gitkeep)
    ├── outputs/             (.gitkeep)
    └── docs/
        └── PLAN.md          copy of this plan
```
No pipeline code, no dependencies installed, no DB access.

## Context
Goal: flag animals drifting from their own normal (weight, feeding leftovers, keeper notes) **before a vet
gets involved**, with alerts that explain themselves, validated by replaying history:
"flags X of Y past cases Z days earlier at ≤ N alerts/week/zoo".
Constraints: offline Python on a DB dump of one zoo; **no proprietary LLM API** — self-hosted open-source LLM
decodes keeper notes, and we train **our own prediction model**. CI4 integration = path to production only.

> ⚠️ Migrations ≠ live schema (the team's own tracker records wrong conclusions drawn from migrations).
> Step 0 of the build runs `SHOW CREATE TABLE` on every table below against the dump and diffs it with this
> dictionary before any extraction.

---

## 1. Data dictionary & field mapping (what's stored, how, and how we use it)

Conventions across Antz: server time is **UTC** (`app/Config/App.php:144`); user-entered dates are converted
with `convertToUTC()` (`app/Common.php:63`) *except where noted*. Soft delete is a per-table flag
(`is_deleted` / `active` / `is_active`), never CI4 soft-deletes. Declared FKs are unreliable (some point to a
non-existent `antz_animal`) — join on columns as listed.

### 1.1 Animal master — `antz_animals`
| Column | Meaning | Use |
|---|---|---|
| `animal_id` | PK | key of everything |
| `taxonomy_id` → `antz_taxonomic_units.tsn` (`default_common_name`, `complete_name`) | species | species selection, model context |
| `type` ('single' / 'group'), `total_animal` | one row may be a group of N | **groups excluded** from weight & intake; notes only |
| `sex`, `birth_date`, `approximate_dob` | demographics | age class (juvenile/adult/geriatric) |
| `enclosure_id` | **current** enclosure only | history comes from 1.6 |
| `is_alive`, `active`, `is_deleted`, `animal_status`, `health_status`, `is_hospitalized` | **current state only** (overwritten) | never used as historical features |

### 1.2 Weight — how it's stored and **how units are handled**
**Tables**
- `antz_animal_assessments` (parent: one row per recording) — `id`, `animal_id`, `assessment_type_id`,
  `assessment_unit_id`, `assessment_value` **TEXT**, `base_uom_value` DOUBLE, `base_uom_name`,
  `record_date`, `record_time`, `recorded_date_time`, `created_at`, `medical_record_id`, `hospital_case_id`,
  `comments`, `is_deleted`.
- `antz_animal_assessment_values` (child: one row per answer; source of truth since 2026-08; backfilled 1:1)
  — `assessment_id`, `assessment_value`, `assessment_unit_id`, `base_uom_value`, `base_uom_name`, `is_deleted`.
- `antz_animal_measurement` (**legacy, still written** by `POST animal/measurement/add` and RNSync) —
  `measurement_type_id` (1 = Weight), `measurement_unit_id`, `measurement_value` DECIMAL(10,2),
  `record_date`, `record_time`. Its rows up to 2024-03-07 were **copied** into assessments (same `id`,
  `created_at`); after that both tables receive different rows.
- Weight type: `antz_master_assessments_types` — global `string_id='weight'` (id 1 live) **plus zoo-created
  types** like "Body Weight" (`zoo_id=<zoo>`, `measurement_type='weight'`) that
  `getAssessmentWeightID()` (app/Common.php:2005) misses.

**Units (`antz_master_measurement_units`, `measurement_type='weight'`)** — base unit = gram:
| unit | factor → g | | unit | factor → g |
|---|---|---|---|---|
| mg | 0.001 | | lb | 453.592 |
| g (gm) | 1 | | st | 6350.29 |
| oz | 28.3495 | | t | 1,000,000 |
| kg | 1000 | | zoo-custom | whatever the zoo set (may even have itself as base) |

**Why we recompute grams ourselves (don't trust `base_uom_value`):**
- 2025-03-19 backfill rounded to 2 dp and wrote **0** (not NULL) when factor/unit missing.
- Medical-record + CSV import paths wrote **NULL** until 2026-08-05.
- Zoo-custom units may have a non-gram base (e.g. "Kilo" as its own base → value in kilos).
- Legacy "add unit" endpoint creates units with `measurement_type='length'` default and NULL factor.
- Write path doesn't check unit category → a "weight" can be recorded in cm.
- `assessment_value` is TEXT, validated only as `required` → "12,5", "12 kg" possible.
- CSV importer maps units by abbreviation across **all zoos** → may point to another zoo's unit.

**Normalisation rule we apply:**
`value_g = numeric(assessment_value) × to_gram(unit) + offset`, where `to_gram` = unit factor if base is
gram, else unit factor × base unit's factor (one hop), else **unresolvable → row flagged, excluded**.
Non-numeric values → try safe parse (comma→dot, strip "kg"/"g" suffix and use that suffix as the unit if it
disagrees) → else excluded. Unit category ≠ weight → excluded. Stored `base_uom_value` kept only as a
cross-check column (`mismatch` flag).

**Timestamp rule:** `recorded_at = COALESCE(recorded_date_time, TIMESTAMP(record_date, record_time), created_at)`;
flag `time_imputed` when only a date exists; convert to zoo-local date for daily bucketing.

**Dedup rule:** union assessments + legacy (legacy row kept only if no assessment row with same animal,
unit, value, record_date, created_at); then drop exact duplicates on (animal, recorded_at, value_g) and
near-duplicates within ±1 minute (medical-record write + normal write of the same weighing).

**Extra weight sources:** `antz_hospital_pre_anaesthesia.weight` (+ `weight_unit`, `weight_type`
Actual/Estimated — use Actual only), `antz_medical_record.body_weight` (free text VARCHAR — ignored, it's
at vet time anyway). Groups: weight semantics undefined (total vs sample) → excluded.

Other assessment types worth checking per zoo: `temperature` (normalised to °C; only correct if the
2026-08-05 temperature backfill ran on the dumped env), zoo-created body-condition-score types
(`response_type='numeric_scale'`, `assessment_rank` = score).

### 1.3 Feeding — what exists, and the pivot
| Source | Grain | Reality |
|---|---|---|
| `antz_feed_log` | no animal/enclosure | dead table, ignore |
| `antz_diet_report_for_ingredients_{zoo_id}` (planned qty per animal/day, `ingredient_qty` already × `total_animal`, in `base_uom_name`) | animal × day × ingredient | **TRUNCATED and regenerated daily** for today…today+7 (`AntzCronModel.php:1446-1504`) → **no historical planned diet** |
| `antz_diet_entity_assign` (+ `antz_diet_assign_logs` old/new diet id + `created_at`) | animal or species | assignment history available → marks **diet change dates** |
| `antz_food_wastage` — `enclosure_id`, `base_wastage_quantity` (g; the column comments are swapped), `wastage_date`/`wastage_time` (**UTC**), `notes` TEXT, `active`, `is_deleted` | enclosure × event | the **only actual-consumption signal** |
| Feeding completion / "meal served" | — | doesn't exist |

**Pivot (replaces "planned − wastage"):** for enclosures with a **single occupant** (1.6), track
**daily leftover grams vs that enclosure's own baseline**, segmented at diet-change dates
(`antz_diet_assign_logs`). Rising leftovers relative to baseline ⇒ eating less. Denominator (planned) is
unavailable historically, so we use *relative change* (robust z of daily leftover within a stable-diet
segment). Days without a wastage entry = **unknown**. Enclosures whose logging rate is erratic
(< ~30% of days) → intake detector marked `low_confidence`.
Also: `antz_food_wastage.notes` text goes to the LLM too ("tiger left most of chicken").
Production recommendation: start snapshotting the daily diet report + per-animal intake logging.

### 1.4 Keeper notes — what the LLM decodes
**`antz_observation`** (the note): `observation_id`; **`observation_name` TEXT = the full note body**
(not a title; plain text, may contain emojis, max 64 KB, nullable); `priority`
ENUM(Critical, High, Moderate, Low) chosen by keeper; `zoo_id`; `created_by`; `created_at` (UTC insert time);
**`created_for` DATE** = day the note is about (can be backdated; ⚠️ edits without resending it reset it to
edit date — so use `LEAST(created_for, DATE(created_at))` if `modified_at` > `created_at`); `active`,
`is_deleted`. No open/closed status. Legacy `ref_type/ref_id/observation_type_id` not used by V2.

**`antz_observation_mapping`** (what the note is about): `observation_id`, `ref_type`
ENUM(animal, enclosure, section, site), `ref_id`, snapshot `enclosure_id/section_id/site_id/species_id`
(NULL after edits; pre-Sep-2025 snapshots are backfilled, not historical), `is_deleted` (**must filter** —
edits soft-delete + re-insert; built-in reports forget this and double-count).
One note → many animals/enclosures. Attribution rule:
- `ref_type='animal'` → that animal; **weight = 1 / n_animals tagged** (a note about 1 animal is stronger).
- `ref_type='enclosure'` → animals present on that day via `antz_enclosure_logs`; used **only if single
  occupant**.
- section/site tags → ignored (too broad).

**`antz_observation_type_mapping` + `antz_observation_type_master`** (category): parent/child types; seeded
parents are maintenance-oriented (Enclosure, Plantation, **Diet**, Plumbing, Electrical, Carpentry,
**Medical**, Pharmacy, Civil, Fabrication, HR, House Keeping, Other); zoos add their own
(e.g. Behaviour, Enrichment, Feeding). Filter `deleted_on IS NULL`, `zoo_id IN (0, :zoo)`. Used as LLM
context + to skip pure maintenance notes (Plumbing/Electrical/Civil/…) before the LLM (saves time).

**`antz_observation_note`** (comment thread): `observation` LONGTEXT = comment text, `created_at`, `active`.
Flat thread → concatenated in time order and decoded with the parent note (keepers often add
"still not eating" as a comment).

**Other free text also decoded:** `antz_food_wastage.notes` (enclosure), `antz_animal_assessments.comments`
(at weighing), `antz_animal_incident_details.notes`. **Not used as input** (vet side = outcome):
`antz_medical_notes`, `antz_complaints_notes`, `antz_diagnosis_notes`, hospital discharge notes.

### 1.5 Medical, hospital, mortality — the **labels** (never model inputs)
**Record header `antz_medical_record`:** `id`, `zoo_id`, `case_type` → `antz_master_medical_case_type`
(`standard` = illness; also quarantine/vaccination/deworming/supplement), `medical_record_type`
(SINGLE/BATCH/LAB_BATCH/GROUP_LAB), **`created_for`** (keep only `medical`; others are routine
vaccination/deworming/supplement), **`created_at` = encounter date the user entered** (UTC),
`system_generated_record_date` = real insert time, `parent_medical_id`, `group_entry`, `is_deleted`.
⚠️ `status` is **stale** (v2 resolve flow never updates it) → derive open/closed from items.

**Record → animal:** `antz_medical_record_animal_mapping` (`medical_record_id`, `animal_id`, `added_date`,
`detach_date`, `active`, `total_animal`, `is_hospitalized`). Legacy records: `antz_medical_record.animal_id`.
Item applies to animal A if `item.animal_id = A` OR (`item.animal_id IS NULL` AND A mapped to the record).

**Complaints (symptoms) `antz_medical_complaints`:** `complaint` → `antz_master_animal_complaints.label`
→ `category_id` → `antz_master_medical_category` (Behavioural Changes, Digestive, **Feeding Abnormalities**,
Respiratory, Physical Injuries…); **`severity`** ∈ {Mild, Moderate, High, Extreme, NULL};
**`duration` + `duration_unit`** = how long the symptom existed **before** the vet saw it (→ estimated
onset); `status` active/closed; time = `COALESCE(recorded_date_time, created_at)`. Status history in
`antz_complaints_notes` (`status`, `date`/`time` in **local** time, `notes_dump` JSON).

**Diagnoses `antz_medical_diagnosis`:** `diagnosis_type` → `antz_master_diagnosis` (`label`,
`is_contagious`, `category_id` → e.g. Infectious, Digestive, Respiratory, **Traumatic and Physical Injuries**,
**Toxicological**…); `severity` (same scale, may be `''`); `status` active/cured/closed; `closed_at` (**local**
time); `is_cronical`; `confidence_level` (Not Sure/Probable/Presumptive/Confirmed); `clinical_assessment`
(tentative/diagnosis); `prognosis` (favourable < guarded < doubtful < poor < grave; undetermined).
History in `antz_diagnosis_notes`.

**Prescriptions / tests:** `antz_medical_prescriptions` (open if `stop_date IS NULL` and not ended),
`antz_medical_tests.status` (completed_positive / completed_detected → severity support).

**Hospital:** `antz_hospital_case_master` — `visit_type` (checkup/**emergency**/opd/follow_up/planned),
`treatment_type` (inpatient/opd), **`admitted_at`** (UTC), `opd_to_inpatient_date`, `discharge_at`,
`status` (exclude `rejected`). `antz_hospital_medical_animal_mapping` — `health_status`
(stable/improving/recovering/**critical**; overwritten → history in `antz_hospital_log`).
`antz_hospital_animal_discharge.discharge_type` = Mortality / TransferHospital / TransferEnclosure.

**Mortality `antz_animals_mortality`:** `entity_id` (+ `entity_type='animal'`), `total_animal`,
**death time = `COALESCE(date_of_death, discovered_date)`**, `is_estimate`, `manner_of_death` →
`antz_master_disposition_manner_of_death` (**overwritten by necropsy confirmed cause**),
`history_of_illness` TEXT. Row selection: `is_active=1 AND status='APPROVED'` (approval inserts a **new**
APPROVED row and marks the original `NO_ACTION_REQUIRED`; exclude that and `REVOKED_*`).
Necropsy `antz_animal_necropsy`: `death_date/time`, `suspected_/confirmed_cause_of_death`; use
`is_system_generated=0 AND is_active=1`.

**Manner-of-death grouping** (by `string_id`, verified against `SELECT DISTINCT` on the dump):
| Class | values | In backtest? |
|---|---|---|
| illness | disease, malnutrition, natural, old_age/oldage, congential_types | ✅ positive |
| euthanasia | euthanasia | ✅ if preceded by an illness medical record within 60 days |
| trauma/external | attacked, stray_attack, murder, foul_play, poisoned, thunder_attack, thunder_blast | ❌ reported separately (no warning expected) |
| unknown | indeterminate, undetermined, suspicious | ✅ secondary analysis |
| junk | test, test2 | drop |

### 1.6 Housing history — who lived where (needed for "single occupant")
`antz_enclosure_logs` — `enclosure_id`, `animal_id`, `in_date`, `out_date`, `is_deleted` (from 2023-11-17).
Quirks: `out_date` has `ON UPDATE CURRENT_TIMESTAMP` and closures match on (enclosure, animal) not row id →
repeated stays get **overwritten out_dates**; death doesn't close the stay; RNSync hard-deletes.
Rule: rebuild stays, cross-check with `antz_movement_animal_mapping.transferred_on`, cap at death time;
enclosure-day is **single-occupant** iff `SUM(total_animal)=1` over animals present that day.
Low-confidence stays (repeated enclosure visits) → excluded from intake and enclosure-tagged notes.

### 1.7 Field → model mapping summary
| Model component | Built from | Never from |
|---|---|---|
| Weight features | 1.2 cleaned `value_g` series | medical_record.body_weight |
| Intake features | 1.3 wastage in single-occupant enclosure-days, diet-change segments | planned diet history (doesn't exist) |
| Notes features | 1.4 note + comments + wastage notes + assessment comments, via LLM | medical/diagnosis/complaint notes |
| Context | 1.1 species, sex, age class; 1.6 days since move | current `health_status` etc. |
| **Labels / T0** | 1.5 medical, hospital, mortality | — |

---

## 2. How it works — explained from scratch

### A. Two layers
**Layer 1 — Detectors (statistics, per animal):** compare each animal to **its own normal**, not to other
animals ("Raja usually leaves ~200 g and weighs ~180 kg; this week he leaves 1.5 kg and weighed 171 kg —
unusual **for Raja**"). Self-comparison makes a tiger and a flamingo comparable automatically.
**Layer 2 — Our own prediction model:** takes Layer-1 numbers for an animal-day and outputs a
**risk score = probability the animal needs veterinary attention in the next 14 days** (not death).
Trained on this zoo's history; a rule-based scorer is kept as a baseline the model must beat.
Why two layers: one zoo has ~20–60 past cases — too few to learn "normal for each of 300 animals" from raw
data, but enough to learn **how much each kind of deviation matters**.
```
raw tables ─► CLEAN & NORMALISE (§1 rules) ─► daily animal series ─► baseline ─► deviation features ─►
            ─► OPEN-SOURCE LLM on notes ─► structured note signals ─┘                OUR MODEL ─► risk + reasons
```

### B. Handling badly-recorded data (never silently trust, never silently invent)
1. **Validate** (reject impossible) → 2. **Clean** (fix safe errors, flag them) → 3. **Missing stays
missing** (no imputation for scoring) → 4. **Coverage** (every alert carries data confidence; animals with
too little data are reported "not monitorable", not "healthy").

**B1. Erroneous weights**
| Problem | Detection | Action |
|---|---|---|
| ≤0 / empty / non-numeric | parse fails | safe-parse or drop |
| Unit not weight / unresolvable | unit category / base chain (§1.2) | drop, flag |
| kg↔g slip | ratio to animal median ≈ 1000 / 0.001 | ÷/×1000, flag `corrected_unit` |
| kg↔lb slip | ratio ≈ 2.2046 / 0.4536 | convert, flag |
| Typo (18 or 1800 for a 180 kg tiger) | outside species band (1st–99th pct of adults in zoo × 0.5–1.5) | drop |
| One-off spike (180,181,**150**,179) | **Hampel filter** (>3 robust SD from neighbours' median) **and** next reading back to normal | drop |
| Real drop (180,181,**165**,**163**) | next reading confirms | **keep — signal** |
| Latest-only drop | can't confirm yet | "pending confirmation" (weak) + suggest re-weigh |
| Duplicates (two tables, double submit) | §1.2 dedup rule | keep one |
| Impossible date | future / before birth / after death | drop |
| Groups | `type='group'` | exclude |
| Juveniles | age class | alert on growth **slowing**, not level |
**B2. Missing weights:** no interpolation; detector runs only with ≥3 valid readings in 180 days (≥1 in 60);
`days_since_last_weight` is a feature so the model learns how much to trust stale weights. Weights taken at
or after T0 (vet already involved) are never used for that episode.
**B3. Intake data:** missing wastage day = unknown (never "ate everything"); wastage in non-solo days →
skip; diet change → baseline restart; erratic logging enclosures → low confidence; planned fast days — unknown
historically, so a day-of-week baseline (leftover median per weekday) absorbs regular fasting patterns.
**B4. Notes:** empty/template ("all ok") → no signal; copy-paste duplicates (same text hash within 3 days)
→ counted once; multi-animal enclosure notes → skipped; `created_for` edit bug handled (§1.4); maintenance
note types skipped.
**B5. Labels:** mortality superseded rows removed; estimated death dates flagged; trauma deaths separate;
backdated medical records: T0 uses the user-entered event date (`created_at`), not insert time.
**B6. Data-quality report (deliverable):** per species/animal: weights per 90d, % solo days with wastage
logged, notes/week, % values dropped/corrected and why, "monitorable" yes/no.

### C. Baseline (median, not average)
Median of the recent window (weight 180 d, leftovers 30 d), **excluding the days being tested**, from
cleaned data only; needs ≥10 valid days (leftovers) / ≥3 readings (weight) else `insufficient_data`.
Median ignores outliers: `96,94,97,95,20,96,93,95,97,94` → average 87.7 (wrong), median 95 (right).

### D. Robust z-score — how unusual is today
`z = (today − median) / (1.4826 × MAD)`, MAD = median distance from the median (the animal's normal wobble).
Steady animal (MAD small) → a modest change is unusual; swingy animal → same change is normal. Floor on MAD
avoids divide-by-zero. |z| > 3 ≈ unusual. (For leftovers, *positive* z = more left = worse.)

### E. CUSUM — slow, sustained changes
`S_today = max(0, S_yesterday + z_today − k)`, flag when `S ≥ h` (k≈0.5, h≈4). Five slightly-bad days
(z = 1.5,1.2,1.8,1.4,1.6) → S = 1.0,1.7,3.0,3.9,**5.0 ⇒ flag**, though no single day was alarming.
Missing day → S held; >7 missing days → reset.

### F. Weight trend
On each new valid reading: (1) % change vs 180-day median; (2) **Theil-Sen** slope (median of pairwise
slopes — one bad weighing can't fake a trend) in %/month; (3) confirmed vs pending.

### G. Decoding keeper notes with an open-source LLM (self-hosted; data never leaves Antz)
**Runtime:** Ollama (laptop/server) — or vLLM on GPU for volume.
**Model:** **Qwen2.5-7B-Instruct** (strong JSON, multilingual incl. Hindi) — fallback Llama-3.1-8B-Instruct,
or Qwen2.5-3B on weak hardware. 4-bit ≈ 5–6 GB RAM; ≈ 2–5 s/note on Apple silicon.
**Input per call:** species common name, note type (parent/child), date, note text + its comment thread.
**Output (forced by Ollama JSON-schema `format`, temperature 0):**
```json
{"signals":[
 {"category":"appetite","direction":"worse","severity":2,"evidence_quote":"ate half of the meat"},
 {"category":"behaviour","direction":"worse","severity":2,"evidence_quote":"seemed dull, lying in corner"}],
 "is_health_related": true}
```
Categories: appetite, mobility (limping, stiffness), behaviour (lethargy, isolation, aggression),
gi (vomit, diarrhoea, stool), respiratory, skin_injury, body_condition, other.
Direction: worse / normal / better. Severity 1–3.
**Safeguards:** (1) verbatim evidence check — quote must occur in the text (fuzzy ≥90%) or the signal is
dropped; (2) negation few-shots ("no limping" → normal); (3) keyword lexicon runs in parallel as fallback and
disagreement log; (4) cache by `observation_id` + text hash; (5) the LLM only **extracts**, never diagnoses.
**Volume control:** only animals of the chosen species; skip maintenance note types; run overnight Day 1.
**Our own note model (stretch):** LLM outputs + ~200 hand-checked notes → TF-IDF + logistic regression per
category (hackathon) → later a fine-tuned small multilingual transformer (MuRIL / multilingual MiniLM).
LLM = teacher, small model = student we own, runs on CPU in ms.
**Note features:** 7-day weighted count of `worse` per category; # distinct negative categories;
note volume vs animal's usual (keepers writing more than usual is itself a signal); max keeper `priority`.

### H. Our own prediction model (Layer 2)
**Question:** for animal *a* on day *d*, using only data ≤ *d*: P(a vet case starts within d+1…d+14)?
**Rows:** one per animal-day (chosen species). **Label 1** if T0 ∈ (d, d+14]; **0** if no T0 within 60 days;
excluded: days under active treatment/hospitalisation, the 15–60-day grey zone, after death.
**Features:**
| Group | Features |
|---|---|
| Weight | % change vs baseline, slope %/month, confirmed/pending, days since last weight |
| Intake (solo enclosures) | leftover robust z today, CUSUM, # high-leftover days in 7, coverage |
| Notes | per-category 7-day `worse` score, # negative categories, note volume z, max priority |
| Context | species group, age class, sex, days since enclosure move, days since last vet case |
| Coverage | weights in 180 d, % logged leftover days (30 d), notes in 30 d |
**Models:** (1) rule-based scorer (baseline); (2) **logistic regression** (main; readable weights, strong
regularisation, class weights); (3) **LightGBM with monotonic constraints** if enough cases (more weight
loss can never lower risk).
**Validation:** **time-based split** (train older, test newer) + leave-one-case-out; metric **PR-AUC**
(not accuracy); the model must beat the rule baseline on the held-out period.
**Explanations:** logistic contributions or SHAP → top 3 reasons + note quotes + data confidence, e.g.
*"Risk 0.41 — Leftovers above normal 5 of 7 days · Weight −6.1% in 5 weeks (confirmed) · Notes: 'seemed
dull' (12 Mar), 'ate half' (14 Mar) · Data confidence: high."*
**Alerting:** risk ≥ threshold, 7-day cooldown per animal unless risk rises.
**Honest limits:** one zoo's cases are few; production path is a **multi-zoo pooled model** (features are
per-animal normalised, so pooling works).

### I. Backtest ("time machine")
Replay day by day with data ≤ D only.
1. **Cases:** illness episodes from §1.5 — case_type standard + `created_for='medical'`, and at least one of:
   max severity ≥ High, hospital inpatient/emergency, `critical` health status, worst prognosis ≥ poor, or
   illness/euthanasia death. Consecutive records ≤14 days apart merged into one episode.
2. **T0 (vet aware)** = `LEAST(record created_at, earliest complaint/diagnosis time, hospital admitted_at)`.
   **Estimated onset** = T0 − complaint `duration` (tells us how early a sign *could* have been seen).
3. Early alert window [T0−60 d, T0); **lead time** = T0 − first alert.
4. **False alerts** = alerts not followed by a case in 60 d → **alerts/week/zoo**.
5. Threshold sweep → *cases caught vs alerts/week* → pick point under budget →
   **"flags X of Y past cases, on average Z days earlier, at ≤ N alerts/week/zoo."**
6. Missed-case analysis: no data / no signal / model miss (links back to the data-quality report).
Caveats stated: small N → wide uncertainty; tune on older half, report on newer half; trauma excluded.

### J. Glossary
Baseline · Median · MAD / robust z · CUSUM · Theil-Sen · Hampel filter · Leakage (using future data) ·
Logistic regression · LightGBM · Monotonic constraint · PR-AUC · SHAP · Recall · Lead time · T0.

---

## 3. Project layout (separate folder, e.g. `ml/early_warning/`)
```
docker-compose.yml   MySQL 8 to restore the dump
config.yaml          zoo_id, timezone, species, windows, thresholds, alert budget, LLM model
00_schema_check.py   SHOW CREATE TABLE for §1 tables → diff vs expected columns; DISTINCT value audits
                     (severity, prognosis, manner_of_death string_id, weight units, note types)
01_extract.py        SQL → parquet: animals, weights(+legacy), units, wastage, diet_assign_logs,
                     enclosure_logs+movements, notes(+comments, mapping, types), medical items,
                     hospital, mortality
02_clean.py          §1.2 unit normalisation + B1–B5 rules → cleaned series + data-quality report
03_profile.py        per-species density → choose 3–5 species
04_cases.py          episodes, T0, estimated onset, outcome
05_detectors.py      weight (%Δ, Theil-Sen, Hampel), leftovers (solo days, z, CUSUM)
06_notes_llm.py      Ollama + JSON schema, verbatim/negation checks, lexicon, cache
07_features.py       animal-day feature table (only data ≤ d)
08_model.py          rules baseline, logistic regression, LightGBM; time split + LOCO CV
09_backtest.py       replay, metrics, threshold sweep, missed-case analysis
app.py               Streamlit timeline per animal
```
Libraries: pandas, numpy, scipy, scikit-learn, lightgbm, shap, ollama, sqlalchemy+pymysql, streamlit,
plotly, pyarrow, rapidfuzz.

## 4. Schedule
- **Day 1 AM:** restore dump → schema check + value audits → extract → clean + data-quality report → profile → pick species.
- **Day 1 PM:** cases/T0 → weight + leftover detectors → **start LLM batch overnight**.
- **Day 2 AM:** features → rule baseline + logistic regression → backtest + sweep.
- **Day 2 PM:** Streamlit timeline → slides (headline, data-quality findings, production path).
- **Cut order:** LightGBM → note student model → leftover detector.

## 5. Path to production (slides only)
Daily timezone-aware `BaseTask` per zoo (`app/Config/Tasks.php`) → scoring service; `HealthAlert` case in
`NotificationModel::getNotificationData()` → `persistAndQueueNotification()`; recipients: keepers
(`antz_animal_staff_map`, `antz_entity_incharge`) + vets (`medical_records_access` + site access).
Product gaps to raise: per-animal intake logging, daily diet-report snapshot, weight unit validation at
input, fix `created_for` edit reset and `out_date ON UPDATE`, multi-zoo pooled model.

## 6. Verification
- Schema check passes (or dictionary updated) before extraction.
- Unit tests per cleaning rule (kg/g fix, lb fix, spike dropped, confirmed drop kept, missing ≠ 0 leftover,
  legacy/assessment dedup).
- Spot-check 3–5 animals' cleaned weight and note timelines against the Antz UI.
- Leakage test: every feature for day d uses rows ≤ d; no medical tables in features.
- LLM audit: ~50 notes hand-reviewed → precision/recall per category, verbatim pass rate.
- Model: held-out PR-AUC vs rule baseline; backtest X/Y, mean/median lead days, alerts/week/zoo.
