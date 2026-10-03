"""Stage notes_llm: decode keeper notes into structured signals with a self-hosted LLM (PLAN.md §G).

The LLM only *extracts* what the note says — never diagnoses. Safeguards: JSON schema forced by Ollama,
temperature 0, negation few-shots, verbatim-evidence check (quote must fuzzy-match the note, else the signal
is dropped), a keyword lexicon run in parallel (fallback + disagreement log), and a cache keyed on the exact
prompt so reruns and resumed batches skip work already done.

    python main.py notes_llm             # batch: kept notes of the chosen species (resumable)
    python main.py notes_llm --limit 20  # first 20 only
    python main.py notes_llm --sample    # 20 built-in synthetic notes with expected answers (correctness check)

Outputs: data/processed/llm_cache.jsonl (append-only), notes_decoded.parquet (one row per note),
notes_signals.parquet (one row per signal, method llm|lexicon), outputs/notes_llm_progress.log.
"""

import hashlib
import json
import re
import time
from pathlib import Path

import pandas as pd
from rapidfuzz import fuzz

from src import config, store

PROMPT_VERSION = "v1"
CATEGORIES = ["appetite", "mobility", "behaviour", "gi", "respiratory", "skin_injury", "body_condition", "other"]

SCHEMA = {
    "type": "object",
    "properties": {
        "signals": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "category": {"type": "string", "enum": CATEGORIES},
                    "direction": {"type": "string", "enum": ["worse", "normal", "better"]},
                    "severity": {"type": "integer", "minimum": 1, "maximum": 3},
                    "evidence_quote": {"type": "string"},
                },
                "required": ["category", "direction", "severity", "evidence_quote"],
            },
        },
        "is_health_related": {"type": "boolean"},
    },
    "required": ["signals", "is_health_related"],
}

SYSTEM = """You read zoo keeper notes and extract animal health signals as JSON. Rules:
- Only extract what the note states about the animal. Never diagnose, never guess causes.
- Categories: appetite (eating, leftovers), mobility (limping, stiffness, unable to stand), behaviour (lethargy,
  dullness, isolation, aggression, restlessness), gi (vomiting, diarrhoea, stool), respiratory (cough, sneeze,
  nasal discharge, breathing), skin_injury (wounds, swelling, bleeding, feather/hair loss), body_condition
  (thin, weight loss, obese), other (any other health sign).
- direction: worse = a problem is present; normal = explicitly fine or negated ("no limping"); better = improving.
- severity: 1 slight, 2 clear, 3 severe (not eating at all, unable to stand, heavy bleeding).
- evidence_quote: copy the exact words from the note (verbatim, same language), the shortest span that shows it.
- Maintenance, enclosure, staff or admin notes, and meaningless text: no signals, is_health_related false.
- The note may be English, Hindi or Hinglish."""

FEW_SHOTS = [
    ("Ate only half of the chicken, seemed dull and lying in corner.",
     {"signals": [{"category": "appetite", "direction": "worse", "severity": 2, "evidence_quote": "Ate only half of the chicken"},
                  {"category": "behaviour", "direction": "worse", "severity": 2, "evidence_quote": "seemed dull and lying in corner"}],
      "is_health_related": True}),
    ("Active and alert. No limping today, ate full ration.",
     {"signals": [{"category": "mobility", "direction": "normal", "severity": 1, "evidence_quote": "No limping today"},
                  {"category": "appetite", "direction": "normal", "severity": 1, "evidence_quote": "ate full ration"}],
      "is_health_related": True}),
    ("Water pipe leaking near night shelter, plumber informed.", {"signals": [], "is_health_related": False}),
    ("Loose stool since morning, slight nasal discharge.",
     {"signals": [{"category": "gi", "direction": "worse", "severity": 2, "evidence_quote": "Loose stool since morning"},
                  {"category": "respiratory", "direction": "worse", "severity": 1, "evidence_quote": "slight nasal discharge"}],
      "is_health_related": True}),
    ("Khana nahi khaya, sust hai",
     {"signals": [{"category": "appetite", "direction": "worse", "severity": 2, "evidence_quote": "Khana nahi khaya"},
                  {"category": "behaviour", "direction": "worse", "severity": 1, "evidence_quote": "sust hai"}],
      "is_health_related": True}),
    ("Limping improved, now walking normally.",
     {"signals": [{"category": "mobility", "direction": "better", "severity": 1, "evidence_quote": "Limping improved"}],
      "is_health_related": True}),
]

# Keyword lexicon (§G safeguard 3): fallback + disagreement log. English + common Hinglish.
LEXICON = {
    "appetite": r"not eat\w*|off (?:food|feed)|refus\w+ (?:food|feed)|(?:ate|eaten) (?:only )?(?:half|little|less)|"
                r"left (?:most|half|all)|poor appetite|no appetite|anorexi\w*|reduced (?:feed|food|intake)|khana nahi",
    "mobility": r"limp\w*|lame\w*|stiff\w*|unable to (?:stand|walk|move)|reluctant to move|dragging",
    "behaviour": r"letharg\w*|dull|depress\w*|isolat\w*|hiding|aggress\w*|restless\w*|inactive|weak\w*|sust",
    "gi": r"vomit\w*|diarrh\w*|loose (?:stool|motion|dropping)s?|constipat\w*|blood in (?:stool|dropping)s?|bloat\w*",
    "respiratory": r"cough\w*|sneez\w*|nasal discharge|wheez\w*|laboured breath\w*|breathing (?:difficult\w*|heavy)",
    "skin_injury": r"wound\w*|injur\w*|lacerat\w*|bleed\w*|swell\w*|swollen|abscess\w*|lesion\w*|bite mark\w*|"
                   r"feather (?:loss|pluck\w*)|hair loss",
    "body_condition": r"emaciat\w*|weight loss|losing weight|underweight|bony|thin\b|obese",
}
# Negation word up to 2 words before the hit, within the same clause (no punctuation in between)
NEGATION = re.compile(r"\b(?:no|not|nil|without|absent|never|normal|nahi)\b(?:\s+\w+){0,2}\s*$", re.IGNORECASE)

SAMPLE_PATH = Path(__file__).resolve().parent.parent / "tests" / "data" / "notes_sample.json"


# ── Pure helpers (unit-tested) ───────────────────────────────────────────────────────────────────────────

def verbatim_ok(quote: str, text: str, min_ratio: float) -> bool:
    """Quote must occur in the note (fuzzy partial match ≥ min_ratio, case-insensitive)."""
    q = (quote or "").strip().lower()
    return bool(q) and fuzz.partial_ratio(q, text.lower()) >= min_ratio


def lexicon_signals(text: str) -> list:
    """Keyword hits as signals; a negation word just before the hit, in the same clause, makes it 'normal'."""
    out = []
    for cat, pattern in LEXICON.items():
        for m in re.finditer(rf"\b(?:{pattern})", text, flags=re.IGNORECASE):
            negated = bool(NEGATION.search(text[max(0, m.start() - 40):m.start()]))
            out.append({"category": cat, "direction": "normal" if negated else "worse", "severity": 1,
                        "evidence_quote": m.group(0)})
    return out


def build_messages(text: str, species: str, note_types: str, date: str) -> list:
    msgs = [{"role": "system", "content": SYSTEM}]
    for note, answer in FEW_SHOTS:
        msgs += [{"role": "user", "content": f"Species: tiger\nNote type: -\nDate: -\nNote: {note}"},
                 {"role": "assistant", "content": json.dumps(answer)}]
    msgs.append({"role": "user", "content": f"Species: {species or 'unknown'}\nNote type: {note_types or '-'}\n"
                                            f"Date: {date or '-'}\nNote: {text}"})
    return msgs


def cache_key(model: str, messages: list) -> str:
    return hashlib.sha1(json.dumps([PROMPT_VERSION, model, messages], ensure_ascii=False).encode()).hexdigest()


def check_signals(result: dict, text: str, min_ratio: float) -> list:
    """Attach verbatim_ok to every LLM signal (failed ones are kept in the table but flagged, not used)."""
    return [{**s, "verbatim_ok": verbatim_ok(s.get("evidence_quote", ""), text, min_ratio)}
            for s in result.get("signals", [])]


# ── LLM + cache ──────────────────────────────────────────────────────────────────────────────────────────

class Decoder:
    def __init__(self, cfg: dict):
        import ollama

        self.llm = cfg["llm"]
        self.client = ollama.Client(host=self.llm["host"], timeout=self.llm["timeout_seconds"])
        self.cache_path = cfg["paths"]["processed"] / "llm_cache.jsonl"
        self.cache = {}
        if self.cache_path.exists():
            for line in self.cache_path.read_text().splitlines():
                if line.strip():
                    entry = json.loads(line)
                    self.cache[entry["key"]] = entry

    def decode(self, messages: list, meta: dict) -> tuple:
        """Returns (cache entry, was_cached). Tries the main model, then the fallback on error."""
        for model in (self.llm["model"], self.llm["fallback_model"]):
            key = cache_key(model, messages)
            if key in self.cache:
                return self.cache[key], True
        error = None
        for model in (self.llm["model"], self.llm["fallback_model"]):
            start = time.time()
            try:
                resp = self.client.chat(model=model, messages=messages, format=SCHEMA,
                                        options={"temperature": self.llm["temperature"]})
                result = json.loads(resp.message.content)
            except Exception as e:  # timeout, bad JSON, server down → try fallback, then record the error
                error = f"{model}: {type(e).__name__}: {e}"[:300]
                continue
            entry = {"key": cache_key(model, messages), "model": model, "prompt_version": PROMPT_VERSION,
                     "seconds": round(time.time() - start, 2), "result": result, **meta}
            with self.cache_path.open("a") as f:
                f.write(json.dumps(entry, ensure_ascii=False) + "\n")
            self.cache[entry["key"]] = entry
            return entry, False
        return {"error": error, **meta}, False


# ── Stage ────────────────────────────────────────────────────────────────────────────────────────────────

def notes_to_decode(cfg: dict) -> pd.DataFrame:
    """Unique kept notes (one per source row) of the chosen species, with species + type context."""
    notes = store.read("processed", "notes_clean")
    animals = store.read("raw", "animals")[["animal_id", "taxonomy_id"]]
    from src.animals import species_names
    names = species_names(store.read("raw", "taxonomy"))
    n = notes[notes["kept"]].merge(animals, on="animal_id")
    if cfg["species"]:
        n = n[n["taxonomy_id"].isin(cfg["species"])]
    n["species"] = n["taxonomy_id"].map(names)
    g = n.groupby(["source", "source_id"])
    out = g.agg(text=("text", "first"), date=("date", "min"), text_hash=("text_hash", "first"),
                parent_types=("parent_types", "first"), child_types=("child_types", "first"),
                species=("species", lambda s: ", ".join(sorted(set(s.dropna()))[:3])),
                n_animals=("animal_id", "nunique")).reset_index()
    return out.sort_values("date").reset_index(drop=True)


def run_sample(decoder: Decoder, min_ratio: float) -> None:
    """Decode the 20 built-in notes and compare the set of 'worse' categories (+ is_health_related) to expected."""
    sample = json.loads(SAMPLE_PATH.read_text())
    passed = 0
    for i, s in enumerate(sample, 1):
        entry, cached = decoder.decode(build_messages(s["text"], s["species"], "", ""), {"sample": i})
        if "error" in entry:
            print(f"  ✗ {i:>2} ERROR {entry['error']}")
            continue
        sig = [x for x in check_signals(entry["result"], s["text"], min_ratio) if x["verbatim_ok"]]
        got = sorted({x["category"] for x in sig if x["direction"] == "worse"})
        ok = got == sorted(s["expected_worse"]) and entry["result"]["is_health_related"] == s["health_related"]
        passed += ok
        timing = "(cached)" if cached else f"{entry['seconds']:4.1f}s"
        print(f"  {'✓' if ok else '✗'} {i:>2} {timing:>8} worse={got} expected={sorted(s['expected_worse'])}  "
              f"{s['text'][:60]!r}")
    print(f"\nSample: {passed}/{len(sample)} decoded correctly")


def run(limit: int = None, sample: bool = False) -> None:
    cfg = config.load()
    min_ratio = cfg["notes"]["evidence_fuzzy_min"]
    decoder = Decoder(cfg)
    if sample:
        run_sample(decoder, min_ratio)
        return

    todo = notes_to_decode(cfg)
    if limit:
        todo = todo.head(limit)
    log_path = store.output_path("notes_llm_progress.log")
    log = log_path.open("a")
    say = lambda msg: (print(msg), log.write(f"{pd.Timestamp.now():%Y-%m-%d %H:%M:%S} {msg}\n"), log.flush())  # noqa: E731
    say(f"Start: {len(todo)} notes, model {cfg['llm']['model']} (fallback {cfg['llm']['fallback_model']}), "
        f"prompt {PROMPT_VERSION}")

    decoded, signals, new_seconds, n_new = [], [], 0.0, 0
    start = time.time()
    for i, note in enumerate(todo.itertuples(), 1):
        types = ";".join(t for t in (note.parent_types, note.child_types) if isinstance(t, str) and t)
        date = f"{note.date:%Y-%m-%d}" if pd.notna(note.date) else ""
        meta = {"source": note.source, "source_id": int(note.source_id), "text_hash": note.text_hash}
        entry, cached = decoder.decode(build_messages(note.text, note.species, types, date), meta)
        llm_sig = check_signals(entry["result"], note.text, min_ratio) if "result" in entry else []
        lex_sig = lexicon_signals(note.text)
        decoded.append({**meta, "date": note.date, "model": entry.get("model"), "error": entry.get("error"),
                        "is_health_related": entry.get("result", {}).get("is_health_related"),
                        "n_llm_signals": sum(s["verbatim_ok"] for s in llm_sig),
                        "n_quotes_rejected": sum(not s["verbatim_ok"] for s in llm_sig),
                        "n_lexicon_signals": len(lex_sig),
                        "llm_worse": ";".join(sorted({s["category"] for s in llm_sig if s["verbatim_ok"] and s["direction"] == "worse"})),
                        "lexicon_worse": ";".join(sorted({s["category"] for s in lex_sig if s["direction"] == "worse"}))})
        signals += [{**meta, "method": "llm", **s} for s in llm_sig]
        signals += [{**meta, "method": "lexicon", **s, "verbatim_ok": True} for s in lex_sig]
        if not cached and "result" in entry:
            n_new += 1
            new_seconds += entry["seconds"]
        if i % 10 == 0 or i == len(todo):
            rate = new_seconds / n_new if n_new else 0
            eta = pd.Timedelta(seconds=rate * (len(todo) - i))
            say(f"{i}/{len(todo)} done ({n_new} new, {i - n_new} cached), {rate:.1f}s/note, ETA {str(eta).split('.')[0]}")

    d = pd.DataFrame(decoded)
    d["disagree"] = d["llm_worse"] != d["lexicon_worse"]
    store.write("processed", "notes_decoded", d)
    store.write("processed", "notes_signals", pd.DataFrame(signals))
    say(f"Done in {time.time() - start:.0f}s: {d['error'].notna().sum()} errors, "
        f"{int(d['n_quotes_rejected'].sum())} quotes rejected, {int(d['disagree'].sum())} LLM/lexicon disagreements, "
        f"{int(d['is_health_related'].fillna(False).sum())} health-related notes")
    log.close()
