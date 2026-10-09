"""Stage backtest: replay history day by day → cases caught, lead time, alerts/week (PLAN.md §I).

For each scorer (rules, logistic): alert on a day when risk ≥ threshold, with a per-animal cooldown (a new alert
inside the cooldown only if risk rose by `rise_factor`); no alerts while the animal is under treatment.
  caught    : ≥ 1 alert in [T0 − 60 d, T0);  lead time = T0 − first such alert
  false     : an alert with no case for that animal in the next 60 d  →  alerts/week/zoo
The threshold is **chosen on the older (train) period with out-of-fold scores** to maximise cases caught under the
alert budget, then applied unchanged to the newer (test) period — numbers are never tuned on what they report.
Reported with a random-alert baseline at the same volume, and a missed-case analysis (no data / no signal /
below threshold). Headline cases are vet-aware; death-only cases (T0 = death, no vet record) are reported apart.

Outputs: outputs/backtest_report.md, backtest_sweep.csv, backtest_cases.csv, backtest_alerts.csv (with reasons).
"""

import json

import numpy as np
import pandas as pd

from src import config, explain, store
from src.features import feature_columns, load_inputs
from src.model import rule_components, rule_score
from src.schema_check import _md_table

SIGNAL_COLS = ["w_fired", "n_neg_cats_7d", "i_cusum_max_7"]


def _days(s: pd.Series) -> np.ndarray:
    return pd.to_datetime(s).dt.normalize().to_numpy().astype("datetime64[D]").astype("int64")


# ── Alerting and scoring (pure) ──────────────────────────────────────────────────────────────────────────

def generate_alerts(rows: pd.DataFrame, score_col: str, thr: float, cooldown: int, rise: float) -> pd.DataFrame:
    """Alerts from rows sorted by (animal_id, date). Cooldown applies per animal; a risk rise ≥ `rise` overrides."""
    r = rows[(rows[score_col] >= thr) & (rows["excl_reason"] != "treatment")]
    ids, days, risk = r["animal_id"].to_numpy(), _days(r["date"]), r[score_col].to_numpy(float)
    keep, cur, last_day, last_risk = [], None, -10 ** 9, 0.0
    for i in range(len(r)):
        if ids[i] != cur:
            cur, last_day, last_risk = ids[i], -10 ** 9, 0.0
        if days[i] - last_day >= cooldown or risk[i] >= last_risk * (1 + rise):
            keep.append(i)
            last_day, last_risk = days[i], risk[i]
    return pd.DataFrame({"animal_id": ids[keep], "day": days[keep], "risk": risk[keep]})


def evaluate(alerts: pd.DataFrame, cases: pd.DataFrame, all_cases: pd.DataFrame, n_weeks: float, early: int,
             clear: int) -> dict:
    """Metrics for one alert set. `cases` = the cases being scored (animal_id, t0_day); `all_cases` = every case,
    used to decide whether an alert was a false alarm."""
    out = {"alerts": len(alerts), "alerts_per_week": len(alerts) / n_weeks if n_weeks else np.nan, "cases": len(cases)}
    if len(alerts) and len(cases):
        j = cases.reset_index().merge(alerts, on="animal_id")
        j = j[(j["day"] >= j["t0_day"] - early) & (j["day"] < j["t0_day"])]
        first = j.groupby("index")["day"].min()
        lead = (cases.loc[first.index, "t0_day"] - first).to_numpy()
    else:
        lead = np.array([])
    out.update(caught=len(lead), caught_pct=100 * len(lead) / len(cases) if len(cases) else np.nan,
               mean_lead=float(lead.mean()) if len(lead) else np.nan,
               median_lead=float(np.median(lead)) if len(lead) else np.nan)
    if len(alerts):
        k = alerts.reset_index().merge(all_cases[["animal_id", "t0_day"]], on="animal_id")
        hit = k[(k["t0_day"] > k["day"]) & (k["t0_day"] - k["day"] <= clear)]["index"].nunique()
        out["precision"] = hit / len(alerts)
    else:
        out["precision"] = np.nan
    return out


def random_baseline_caught_pct(n_alerts: int, n_rows: int, early: int) -> float:
    """Expected % of cases caught if the same number of alerts were spread uniformly over animal-days."""
    a = min(n_alerts / n_rows, 1.0) if n_rows else 0.0
    return 100 * (1 - (1 - a) ** early)


def candidate_thresholds(scores: np.ndarray) -> np.ndarray:
    """Distinct thresholds from the 50th to the 99.99th percentile (and every distinct value if there are few)."""
    s = scores[~np.isnan(scores)]
    uniq = np.unique(s)
    if len(uniq) <= 60:
        return uniq[uniq > 0] if (uniq > 0).any() else uniq
    return np.unique(np.quantile(s, np.r_[np.linspace(0.5, 0.99, 50), 0.995, 0.998, 0.999, 0.9995, 0.9999]))


def sweep(rows, score_col, cases, all_cases, cfg, thresholds) -> pd.DataFrame:
    acfg, early, clear = cfg["alerting"], cfg["labels"]["early_window_days"], cfg["labels"]["negative_clear_days"]
    n_weeks = max((rows["date"].max() - rows["date"].min()).days, 1) / 7
    n_rows = int((rows["excl_reason"] != "treatment").sum())
    res = []
    for thr in thresholds:
        a = generate_alerts(rows, score_col, thr, acfg["cooldown_days"], acfg["rise_factor"])
        m = evaluate(a, cases, all_cases, n_weeks, early, clear)
        m["threshold"] = float(thr)
        m["random_caught_pct"] = random_baseline_caught_pct(m["alerts"], n_rows, early)
        res.append(m)
    return pd.DataFrame(res)


def choose_threshold(sw: pd.DataFrame, budget: float):
    """Most cases caught within the alert budget (ties → longer lead, then higher threshold). (row, within_budget)."""
    ok = sw[sw["alerts_per_week"] <= budget]
    if ok.empty:
        return sw.sort_values("threshold", ascending=False).iloc[0], False
    best = ok.sort_values(["caught", "mean_lead", "threshold"], ascending=[False, False, False],
                          na_position="last").iloc[0]
    return best, True


def miss_reason(window: pd.DataFrame, thr: float, score_col: str) -> str:
    """Why a case was missed: no rows in its early window / rows but no detector or note signal / signal below thr."""
    if window.empty:
        return "no_data"
    sig = (window[SIGNAL_COLS] > 0).any().any() or (window["w_pct_change"] <= -2).any()
    return "below_threshold" if sig and window[score_col].max() < thr else "no_signal"


# ── Stage ────────────────────────────────────────────────────────────────────────────────────────────────

def _case_frame(cases: pd.DataFrame, animal_ids: set) -> pd.DataFrame:
    c = cases[cases["is_case"] & cases["animal_id"].isin(animal_ids)].copy()
    c["t0_day"] = _days(c["t0"])
    c["kind"] = np.where(c["t0_source"] == "death", "death_only", "vet_aware")
    return c[["episode_id", "animal_id", "species", "t0", "t0_day", "t0_source", "kind"]].reset_index(drop=True)


def run() -> None:
    cfg = config.load()
    early, clear, acfg = cfg["labels"]["early_window_days"], cfg["labels"]["negative_clear_days"], cfg["alerting"]
    pred = store.read("processed", "predictions").sort_values(["animal_id", "date"]).reset_index(drop=True)
    summary = json.loads((cfg["paths"]["processed"] / "model_summary.json").read_text())
    inp = load_inputs(cfg)
    cases = _case_frame(store.read("processed", "cases"), set(inp.animals["animal_id"]))
    split = pd.Timestamp(summary["split_date"])
    scorers = ["risk_rules"] + (["risk_logreg"] if pred["risk_logreg"].notna().any() else [])
    primary = "risk_" + summary["primary_scorer"]

    # Replay every grid day, not just labelled ones: labels drop the 15–60 d grey zone before each case, which is
    # exactly where early alerts have to be visible. Same boundaries as the model's time split (purge gap between).
    purge = pd.Timedelta(days=clear)
    train, test = pred[pred["date"] < split - purge], pred[pred["date"] >= split]
    t_start, t_end = test["date"].min(), test["date"].max()
    tr_start, tr_end = train["date"].min(), train["date"].max()

    def eligible(kind, start, end):
        lo, hi = _days(pd.Series([start]))[0] + early, _days(pd.Series([end]))[0]
        c = cases[(cases["kind"] == kind) & cases["t0_day"].between(lo, hi)]
        return c.reset_index(drop=True)

    train_cases, test_vet, test_death = eligible("vet_aware", tr_start, tr_end), eligible("vet_aware", t_start, t_end), \
        eligible("death_only", t_start, t_end)
    all_cases = cases
    n_weeks_test = max((t_end - t_start).days, 1) / 7

    sweep_rows, case_rows, alert_rows, report, headlines = [], [], [], [], []
    model = None
    if "risk_logreg" in scorers:
        model = json.loads((cfg["paths"]["processed"] / "model_logreg.json").read_text())
    for col in scorers:
        name = col.replace("risk_", "")
        # 1. Threshold on the older period (out-of-fold scores), under the alert budget
        sw_tr = sweep(train, col, train_cases, all_cases, cfg, candidate_thresholds(train[col].to_numpy(float)))
        best, within = choose_threshold(sw_tr, acfg["max_alerts_per_week"])
        thr = float(best["threshold"])
        # 2. Apply unchanged to the newer period; also the full curve there (for transparency, not for tuning)
        sw_te = sweep(test, col, test_vet, all_cases, cfg, candidate_thresholds(test[col].to_numpy(float)))
        sweep_rows += [sw_tr.assign(scorer=name, period="train (threshold chosen here)"),
                       sw_te.assign(scorer=name, period="test (curve, not used for tuning)")]
        alerts = generate_alerts(test, col, thr, acfg["cooldown_days"], acfg["rise_factor"])
        m = evaluate(alerts, test_vet, all_cases, n_weeks_test, early, clear)
        m_death = evaluate(alerts, test_death, all_cases, n_weeks_test, early, clear)
        n_rows = int((test["excl_reason"] != "treatment").sum())
        rand = random_baseline_caught_pct(m["alerts"], n_rows, early)

        # 3. Per-case outcomes + missed-case analysis
        for kind, cs in (("vet_aware", test_vet), ("death_only", test_death)):
            for c in cs.itertuples():
                a = alerts[(alerts["animal_id"] == c.animal_id) & (alerts["day"] >= c.t0_day - early)
                           & (alerts["day"] < c.t0_day)]
                win = test[(test["animal_id"] == c.animal_id) & (_days(test["date"]) >= c.t0_day - early)
                           & (_days(test["date"]) < c.t0_day)]
                caught = len(a) > 0
                case_rows.append({"scorer": name, "episode_id": c.episode_id, "animal_id": c.animal_id,
                                  "species": c.species, "t0": c.t0, "kind": kind, "caught": caught,
                                  "lead_days": int(c.t0_day - a["day"].min()) if caught else np.nan,
                                  "window_rows": len(win), "max_risk_in_window": win[col].max() if len(win) else np.nan,
                                  "miss_reason": "" if caught else miss_reason(win, thr, col)})

        # 4. Alerts with reasons (S3-5)
        if len(alerts):
            key = alerts.assign(date=pd.to_datetime(alerts["day"], unit="D"))
            rows_a = key.merge(test, on=["animal_id", "date"], how="left", suffixes=("", "_p"))
            comps = rule_components(rows_a, cfg) if name == "rules" else None
            hit = alerts.reset_index().merge(all_cases[["animal_id", "t0_day"]], on="animal_id")
            followed = set(hit[(hit["t0_day"] > hit["day"]) & (hit["t0_day"] - hit["day"] <= clear)]["index"])
            texts = []
            for i, r in rows_a.iterrows():
                reasons = explain.rule_reasons(comps.loc[i]) if comps is not None else explain.logistic_reasons(r, model)
                quotes = explain.recent_quotes(inp.signals, int(r["animal_id"]), r["date"])
                texts.append(explain.alert_text(r["risk"], reasons, quotes, explain.confidence(r)))
            alert_rows.append(pd.DataFrame({"scorer": name, "animal_id": alerts["animal_id"],
                                            "date": key["date"], "risk": alerts["risk"],
                                            "followed_by_case_within_60d": [i in followed for i in range(len(alerts))],
                                            "explanation": texts}))

        miss = pd.DataFrame([r for r in case_rows if r["scorer"] == name and r["kind"] == "vet_aware" and not r["caught"]])
        reasons_cnt = miss["miss_reason"].value_counts().to_dict() if len(miss) else {}
        mon = sum(1 for r in case_rows if r["scorer"] == name and r["kind"] == "vet_aware" and r["window_rows"] > 0)
        headlines.append({
            "scorer": name, "threshold": round(thr, 4), "within_budget_on_train": within,
            "line": (f"**{name}**: flags **{m['caught']} of {m['cases']}** vet-aware cases "
                     f"({m['caught_pct']:.0f}%; random alerts at the same volume would catch ≈ {rand:.0f}%), "
                     + (f"on average **{m['mean_lead']:.0f} days earlier** (median {m['median_lead']:.0f}), "
                        if m["caught"] else "no lead time (nothing caught), ")
                     + f"at **{m['alerts_per_week']:.1f} alerts/week/zoo** (budget {acfg['max_alerts_per_week']}); "
                     + (f"{100 * m['precision']:.1f}% of alerts were followed by a case within {clear} d. "
                        if m["alerts"] else "No alerts fired. ")
                     + f"{mon} of {m['cases']} cases had any data in their early window. "
                       f"Death-only cases: {m_death['caught']} of {m_death['cases']} flagged."),
            "missed": reasons_cnt})

    sweeps = pd.concat(sweep_rows, ignore_index=True)
    sweeps.to_csv(store.output_path("backtest_sweep.csv"), index=False)
    pd.DataFrame(case_rows).to_csv(store.output_path("backtest_cases.csv"), index=False)
    alerts_df = pd.concat(alert_rows, ignore_index=True) if alert_rows else pd.DataFrame()
    alerts_df.to_csv(store.output_path("backtest_alerts.csv"), index=False)

    report = [f"# Backtest — zoo {cfg['zoo_id']}\n",
              f"Replay of **{t_start:%Y-%m-%d} → {t_end:%Y-%m-%d}** ({n_weeks_test:.0f} weeks, held out); thresholds chosen on "
              f"{tr_start:%Y-%m-%d} → {tr_end:%Y-%m-%d} (out-of-fold scores) under a budget of "
              f"{acfg['max_alerts_per_week']} alerts/week/zoo; cooldown {acfg['cooldown_days']} d; early window {early} d.\n",
              f"Model verdict (M3): **{summary['verdict']}** → primary scorer: **{summary['primary_scorer']}**\n",
              "## Headline\n"]
    for h in sorted(headlines, key=lambda h: "risk_" + h["scorer"] != primary):
        report += [f"- {'**(primary)** ' if 'risk_' + h['scorer'] == primary else ''}{h['line']}",
                   f"  - missed vet-aware cases: {h['missed'] or 'none'}; threshold {h['threshold']}"
                   + ("" if h["within_budget_on_train"] else " (⚠️ even the highest threshold exceeded the budget on train)")]
    if not summary["verdict"].startswith("GO"):
        report.append("\n> Read with the model verdict above: the logistic model did not clearly beat the rules "
                      "(see `outputs/model_report.md`), so treat these numbers as a demonstration that the machinery "
                      "works end to end, not as proof the system works.")
    report += [
        "\n## Case outcomes (test period)\n",
        _md_table(pd.DataFrame(case_rows).groupby(["scorer", "kind", "caught"]).size().rename("cases").reset_index()),
        f"\nAlerts with reasons: `outputs/backtest_alerts.csv` ({len(alerts_df):,} rows). Sweep curves: `outputs/backtest_sweep.csv`.",
    ]
    if len(alerts_df):
        sample = alerts_df[alerts_df["scorer"] == primary.replace("risk_", "")].sort_values("risk", ascending=False).head(5)
        report += ["\n## Example alerts (highest risk, primary scorer)\n"] + [f"- animal {r.animal_id}, {r.date:%Y-%m-%d}: {r.explanation}"
                                                                           for r in sample.itertuples()]
    store.output_path("backtest_report.md").write_text("\n".join(report))
    print("\n".join(report[1:8]))
    print("→ outputs/backtest_report.md")
