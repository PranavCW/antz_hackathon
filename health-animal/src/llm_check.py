"""Stage llm_check: confirm Ollama is up, the models are pulled, and schema-forced JSON decoding works."""

import json
import time

import ollama

from src import config

# Minimal version of the §G output schema; the full one lives in notes_llm.
SCHEMA = {
    "type": "object",
    "properties": {
        "signals": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "category": {"type": "string", "enum": config.load()["notes"]["categories"]},
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

SAMPLE_NOTE = "Tiger ate only half of the meat today, seemed dull and was lying in the corner. No limping."

PROMPT = (
    "You extract health signals from zoo keeper notes. Only extract what the note states; never diagnose. "
    "Negated findings (e.g. 'no limping') are direction 'normal'. evidence_quote must be copied verbatim "
    "from the note.\n\nSpecies: Bengal tiger\nNote: {note}"
)


def run() -> None:
    """Print which models are available, then decode one sample note with each and time it."""
    llm = config.load()["llm"]
    client = ollama.Client(host=llm["host"], timeout=llm["timeout_seconds"])

    # 1. Server reachable and models pulled
    available = {m.model for m in client.list().models}
    print(f"Ollama at {llm['host']}: {', '.join(sorted(available)) or 'no models'}")

    for model in (llm["model"], llm["fallback_model"]):
        if not any(a == model or a.startswith(model + ":") for a in available):
            print(f"  ✗ {model}: not pulled (ollama pull {model})")
            continue

        # 2. Schema-forced decode of the sample note
        start = time.time()
        resp = client.chat(
            model=model,
            messages=[{"role": "user", "content": PROMPT.format(note=SAMPLE_NOTE)}],
            format=SCHEMA,
            options={"temperature": llm["temperature"]},
        )
        result = json.loads(resp.message.content)

        # 3. Verbatim-evidence check, the §G safeguard in its simplest form
        quotes_ok = all(s["evidence_quote"].lower() in SAMPLE_NOTE.lower() for s in result["signals"])
        print(f"  ✓ {model}: {time.time() - start:.1f}s, verbatim quotes {'ok' if quotes_ok else 'FAILED'}")
        print("    " + json.dumps(result, indent=2).replace("\n", "\n    "))
