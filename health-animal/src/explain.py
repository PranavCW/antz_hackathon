"""Alert explanations (PLAN.md §H): top-3 reasons + keeper quotes + data confidence, for both scorers.

Logistic reasons are the largest positive contributions coef × standardised value (the exact quantity the model
adds up), rendered with the raw feature value. Rule reasons are the rules that fired. Pure functions.
"""

import numpy as np
import pandas as pd

from src.notes_llm import CATEGORIES

RULE_TEXT = {
    "weight_drop": "Weight dropped vs its 180-day median",
    "weight_confirmed": "Weight drop confirmed by a second reading",
    "intake_cusum": "Leftovers rising for several days (CUSUM)",
    "notes_one": "Keeper notes report something worse",
    "notes_two_plus": "Keeper notes report several kinds of problems",
    "priority_high": "A keeper marked a note High/Critical",
}


def describe_feature(name: str, row: pd.Series) -> str:
    """One human-readable clause for a feature, using the animal-day's raw values."""
    v = float(row[name])
    cat = next((c for c in CATEGORIES if name == f"n_{c}_7d"), None)
    if cat:
        return f"Keeper notes: {cat.replace('_', ' ')} worse ({v:g} note{'s' if v != 1 else ''} this week)"
    if name == "w_pct_change":
        state = "confirmed" if row.get("w_confirmed", 0) else "pending confirmation"
        return f"Weight {v:+.1f}% vs 180-day median ({state})"
    if name == "w_slope_pct_month":
        return f"Weight trend {v:+.1f}%/month"
    if name in ("w_fired", "w_pending", "w_confirmed"):
        return "Weight-drop detector fired"
    if name == "i_high_days_7":
        return f"Leftovers above normal on {v:g} of the last 7 days"
    if name == "i_z_max_7":
        return f"Leftovers {v:.1f} robust SD above normal this week"
    if name == "i_cusum_max_7":
        return f"Leftovers rising for several days (CUSUM {v:.1f})"
    if name == "n_neg_cats_7d":
        return f"{v:g} different problem types in keeper notes this week"
    if name == "n_max_priority_7d":
        return f"Highest keeper note priority this week: {['-', 'Low', 'Moderate', 'High', 'Critical'][int(min(v, 4))]}"
    if name == "n_volume_ratio":
        return f"Keepers writing {v:.1f}× their usual volume of notes"
    if name == "n_notes_30d":
        return f"{v:g} keeper note{'s' if v != 1 else ''} in the last 30 days"
    if name == "w_days_since_last":
        return f"No recent weight ({v:.0f} days since the last one)"
    if name == "ctx_days_since_move":
        return f"Recently moved enclosure ({v:.0f} days ago)"
    if name.startswith("ctx_"):
        return f"Context: {name[4:].replace('_', ' ')} = {v:.3g}"
    return f"{name} = {v:.3g}"


def logistic_reasons(row: pd.Series, model: dict, top: int = 3) -> list:
    """Top positive contributions of the fitted model for this animal-day, as readable clauses."""
    from src.model import standardise
    feats = model["features"]
    x = row[feats].to_numpy(float)
    contrib = np.array(model["coef"]) * standardise(x, np.array(model["mean"]), np.array(model["scale"]))
    order = np.argsort(contrib)[::-1]
    return [describe_feature(feats[i], row) for i in order[:top] if contrib[i] > 0]


def rule_reasons(components: pd.Series, top: int = 3) -> list:
    fired = components[components > 0].sort_values(ascending=False)
    return [RULE_TEXT[k] for k in fired.index[:top]]


def confidence(row: pd.Series) -> str:
    """high / medium / low from how many of weights, leftovers, notes had recent data (cov_sources 0–3)."""
    n = row["cov_sources"]
    label = "high" if n >= 2 else "medium" if n >= 1 else "low"
    parts = [f"weight baseline {'yes' if row['w_has_baseline'] else 'no'}",
             f"notes in 30 d {row['n_notes_30d']:g}", f"leftovers logged {100 * row['i_logged_30d']:.0f}% of 30 d"]
    return f"{label} ({', '.join(parts)})"


def recent_quotes(signals: pd.DataFrame, animal_id: int, date: pd.Timestamp, days: int = 14, n: int = 2) -> list:
    """Evidence quotes of 'worse' note signals entered in the last `days` days (never later than `date`)."""
    s = signals[(signals["animal_id"] == animal_id) & (signals["available_date"] <= date)
                & (signals["available_date"] > date - pd.Timedelta(days=days))]
    s = s.sort_values("available_date", ascending=False).drop_duplicates("quote").head(n)
    return [f"“{q}” ({d:%d %b})" for q, d in zip(s["quote"], s["available_date"])]


def alert_text(risk: float, reasons: list, quotes: list, conf: str) -> str:
    parts = [f"Risk {risk:.2f}"] + (reasons or ["no single dominant factor"])
    if quotes:
        parts.append("Notes: " + ", ".join(quotes))
    parts.append(f"Data confidence: {conf}")
    return " — ".join([parts[0], " · ".join(parts[1:])])
