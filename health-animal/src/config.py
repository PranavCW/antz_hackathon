"""Load config/config.yaml and resolve paths/scope from .env."""

from functools import lru_cache

import yaml

from src.db import PROJECT_ROOT, env


@lru_cache(maxsize=1)
def load() -> dict:
    """Return config.yaml merged with .env-driven scope and resolved data paths."""
    with open(PROJECT_ROOT / "config" / "config.yaml") as f:
        cfg = yaml.safe_load(f)

    cfg["zoo_id"] = int(env("ZOO_ID")) if env("ZOO_ID") else None
    cfg["timezone"] = env("ZOO_TIMEZONE", "UTC")
    cfg["llm"] = {
        "host": env("OLLAMA_HOST", "http://localhost:11434"),
        "model": env("LLM_MODEL", "qwen2.5:7b-instruct"),
        "fallback_model": env("LLM_FALLBACK_MODEL", "qwen2.5:3b"),
        "timeout_seconds": int(env("LLM_TIMEOUT_SECONDS", "60")),
        "temperature": cfg["notes"]["llm_temperature"],
    }
    cfg["paths"] = {
        name: PROJECT_ROOT / env(key, default)
        for name, key, default in [
            ("raw", "DATA_RAW_DIR", "data/raw"),
            ("processed", "DATA_PROCESSED_DIR", "data/processed"),
            ("outputs", "OUTPUTS_DIR", "outputs"),
        ]
    }
    for path in cfg["paths"].values():
        path.mkdir(parents=True, exist_ok=True)
    return cfg
