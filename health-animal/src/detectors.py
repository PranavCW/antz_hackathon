"""Stage detectors: Layer 1 — each animal against its own normal (PLAN.md §C–F, B2–B3).

Weight (on every kept reading): % change vs the median of the previous 180 days (same local day excluded),
Theil-Sen slope in %/month, confirmed (previous reading deviated the same way) vs pending, insufficient_data.
Leftovers (on every logged solo day): robust z vs the previous 30 days of the same diet segment (weekday
median when there are enough same-weekday days), CUSUM over calendar days with hold/reset on missing days.

Every value for time t uses only rows before t (≤ t for the slope's own point) — no future data.
Outputs: data/processed/detector_weight.parquet, detector_intake.parquet, outputs/detector_cases.md,
outputs/detectors/case_<episode_id>.png.
"""

import numpy as np
import pandas as pd

from src import config, store

DAY = pd.Timedelta(days=1)


# ── Building blocks ──────────────────────────────────────────────────────────────────────────────────────

def robust_z(x: float, window: np.ndarray, center: float, mad_floor: float) -> float:
    """z = (x − center) / (1.4826 × MAD of window), MAD floored to avoid divide-by-zero (§D)."""
    mad = np.median(np.abs(window - np.median(window)))
    return (x - center) / (1.4826 * max(mad, mad_floor))


def theil_sen(t_days: np.ndarray, y: np.ndarray) -> float:
    """Median of pairwise slopes (y per day); one bad point can't fake a trend (§F). NaN if < 2 distinct times."""
    i, j = np.triu_indices(len(t_days), k=1)
    dt = t_days[j] - t_days[i]
    ok = dt != 0
    return float(np.median((y[j] - y[i])[ok] / dt[ok])) if ok.any() else np.nan


def cusum(dates: pd.Series, z: pd.Series, k: float, h: float, max_missing: int) -> pd.DataFrame:
    """S_today = max(0, S_yesterday + z − k); flag when S ≥ h (§E).

    `dates` are the logged days (sorted). Days not in `dates`, or with z NaN, are missing → S held;
    a run of more than `max_missing` missing days resets S to 0.
    """
    s, out, last_valid = 0.0, [], None
    for d, zi in zip(pd.to_datetime(dates), z):
        if last_valid is not None and (d - last_valid).days - 1 > max_missing:
            s = 0.0
        if not np.isnan(zi):
            s = max(0.0, s + zi - k)
            last_valid = d
        out.append(s)
    out = np.array(out)
    return pd.DataFrame({"cusum": out, "cusum_flag": out >= h}, index=z.index)


# ── Weight ───────────────────────────────────────────────────────────────────────────────────────────────

def weight_detector(weights: pd.DataFrame, wcfg: dict, dcfg: dict, causal: bool = False) -> pd.DataFrame:
    """One row per reading with baseline, % change, slope, confirmation and fired flag.

    causal=True is the replay view: a one-off spike is only recognised once the *next* reading arrives, so at its
    own time it is scored as a pending reading (kept in the output) but never enters any later baseline.
    """
    keep = weights["kept"] | (weights["drop_reason"].eq("spike") if causal else False)
    w = weights[keep].sort_values(["animal_id", "recorded_at"])
    window, recent = pd.Timedelta(days=wcfg["baseline_days"]), pd.Timedelta(days=wcfg["min_readings_last_days"])
    rows = []
    for animal_id, g in w.groupby("animal_id"):
        t = pd.to_datetime(g["recorded_at"]).to_numpy()
        day = g["local_date"].to_numpy()
        v = g["value_g"].to_numpy(float)
        juvenile = g["is_juvenile"].fillna(False).to_numpy(bool)
        usable = g["kept"].to_numpy(bool)          # spikes (causal mode) are scored but never form a baseline
        prev_dev = 0.0
        for i in range(len(v)):
            base = (t >= t[i] - window) & (day < day[i]) & usable   # previous 180 d, excluding today
            n_base = int(base.sum())
            has_recent = bool(((t >= t[i] - recent) & base).any())
            row = {"animal_id": animal_id, "recorded_at": t[i], "local_date": day[i], "value_g": v[i],
                   "is_juvenile": juvenile[i], "n_baseline": n_base, "baseline_g": np.nan, "pct_change": np.nan,
                   "slope_pct_month": np.nan, "status": "insufficient_data", "confirmed": False, "fired": False}
            if n_base >= wcfg["min_readings_in_baseline"] and has_recent:
                b = float(np.median(v[base]))
                pts = base | (np.arange(len(v)) == i)
                t_days = (t[pts] - t[pts][0]) / np.timedelta64(1, "D")
                slope = theil_sen(t_days.astype(float), v[pts])
                pct = 100 * (v[i] - b) / b
                drop = pct <= dcfg["weight_alert_pct"]
                if juvenile[i]:
                    fired = slope * 30.44 / b * 100 < dcfg["juvenile_min_slope_pct_month"]   # growth stalled
                else:
                    fired = drop or (slope * 30.44 / b * 100 <= dcfg["weight_alert_slope_pct_month"])
                row.update(baseline_g=b, pct_change=pct, slope_pct_month=slope * 30.44 / b * 100, status="ok",
                           confirmed=bool(drop and prev_dev <= dcfg["weight_alert_pct"]), fired=bool(fired))
                prev_dev = pct
            else:
                prev_dev = 0.0
            rows.append(row)
    return pd.DataFrame(rows)


# ── Leftovers ────────────────────────────────────────────────────────────────────────────────────────────

def intake_detector(daily: pd.DataFrame, icfg: dict, dcfg: dict) -> pd.DataFrame:
    """Robust z of daily leftover grams vs the animal's previous 30 days in the same diet segment + CUSUM."""
    out = []
    window = pd.Timedelta(days=icfg["baseline_days"])
    for (animal_id, seg), g in daily.sort_values("date").groupby(["animal_id", "diet_segment"]):
        d = pd.to_datetime(g["date"]).to_numpy()
        x = g["leftover_g"].to_numpy(float)
        wd = pd.to_datetime(g["date"]).dt.weekday.to_numpy()
        z = np.full(len(x), np.nan)
        base = np.full(len(x), np.nan)
        for i in range(len(x)):
            win = (d >= d[i] - window) & (d < d[i])
            if win.sum() < icfg["min_valid_days"]:
                continue
            same_wd = win & (wd == wd[i])
            center = np.median(x[same_wd]) if same_wd.sum() >= dcfg["intake_weekday_min_points"] else np.median(x[win])
            base[i] = center
            z[i] = robust_z(x[i], x[win], center, max(dcfg["intake_mad_floor_g"], dcfg["mad_floor"]))
        res = g[["animal_id", "date", "enclosure_id", "diet_segment", "leftover_g", "low_confidence"]].copy()
        res["baseline_g"], res["z"] = base, z
        res["status"] = np.where(np.isnan(z), "insufficient_data", "ok")
        res = res.join(cusum(res["date"], res["z"], icfg["cusum_k"], icfg["cusum_h"], icfg["cusum_max_missing_days"]))
        res["high_day"] = res["z"] > dcfg["z_threshold"]
        out.append(res)
    cols = ["animal_id", "date", "enclosure_id", "diet_segment", "leftover_g", "low_confidence", "baseline_g", "z",
            "status", "cusum", "cusum_flag", "high_day"]
    return pd.concat(out, ignore_index=True) if out else pd.DataFrame(columns=cols)


# ── Case review (M2) ─────────────────────────────────────────────────────────────────────────────────────

def case_hits(cases: pd.DataFrame, det_w: pd.DataFrame, det_i: pd.DataFrame, early_days: int) -> pd.DataFrame:
    """Per case: did a detector fire in [T0 − early_days, T0)? Lead = T0 − first firing."""
    c = cases[cases["is_case"]][["episode_id", "animal_id", "species", "t0", "t0_source"]]
    out = c.copy()
    for name, det, tcol, fcol in [("weight", det_w, "recorded_at", "fired"), ("intake", det_i, "date", "cusum_flag")]:
        j = c.merge(det[["animal_id", tcol, fcol, "status"]], on="animal_id")
        j = j[(j[tcol] < j["t0"]) & (j[tcol] >= j["t0"] - pd.Timedelta(days=early_days))]
        cov = j[j["status"] == "ok"].groupby("episode_id").size()
        first = j[j[fcol].astype(bool)].groupby("episode_id")[tcol].min()
        out[f"{name}_points_in_window"] = out["episode_id"].map(cov).fillna(0).astype(int)
        out[f"{name}_lead_days"] = (out["t0"] - out["episode_id"].map(first)).dt.days
    return out


def plot_case(episode: pd.Series, weights: pd.DataFrame, det_w: pd.DataFrame, path) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    t0 = episode["t0"]
    lo, hi = t0 - pd.Timedelta(days=240), t0 + pd.Timedelta(days=60)
    a = episode["animal_id"]
    raw = weights[(weights["animal_id"] == a) & weights["recorded_at"].between(lo, hi)]
    det = det_w[(det_w["animal_id"] == a) & det_w["recorded_at"].between(lo, hi)]
    fig, ax = plt.subplots(figsize=(9, 3.6))
    kept, dropped = raw[raw["kept"]], raw[~raw["kept"]]
    ax.plot(kept["recorded_at"], kept["value_g"] / 1000, "o-", color="#3b6ea5", ms=4, lw=1, label="weight (kept)")
    if len(dropped):
        ax.plot(dropped["recorded_at"], dropped["value_g"] / 1000, "x", color="#999", label="dropped by clean")
    ax.plot(det["recorded_at"], det["baseline_g"] / 1000, "--", color="#888", lw=1, label="180 d median")
    f = det[det["fired"]]
    ax.plot(f["recorded_at"], f["value_g"] / 1000, "o", mfc="none", mec="#c0392b", ms=10, mew=2, label="detector fired")
    ax.axvline(t0, color="#c0392b", lw=1.5)
    ax.axvspan(t0 - pd.Timedelta(days=60), t0, color="#f5b7b1", alpha=0.25, label="early window (T0−60 d)")
    ax.set_title(f"Episode {episode['episode_id']} — animal {a} ({episode['species']}) — T0 {t0:%Y-%m-%d}", fontsize=10)
    ax.set_ylabel("kg")
    ax.legend(fontsize=7, loc="best")
    fig.tight_layout()
    fig.savefig(path, dpi=110)
    plt.close(fig)


def run() -> None:
    cfg = config.load()
    proc = lambda name: store.read("processed", name)  # noqa: E731
    weights, daily, cases = proc("weights_clean"), proc("intake_daily"), proc("cases")

    det_w = weight_detector(weights, cfg["weight"], cfg["detectors"])
    det_i = intake_detector(daily, cfg["intake"], cfg["detectors"])
    store.write("processed", "detector_weight", det_w)
    store.write("processed", "detector_intake", det_i)

    # M2: which known cases had a detector fire before T0?
    hits = case_hits(cases, det_w, det_i, cfg["labels"]["early_window_days"])
    covered = hits[(hits["weight_points_in_window"] > 0) | (hits["intake_points_in_window"] > 0)]
    fired = covered[covered["weight_lead_days"].notna() | covered["intake_lead_days"].notna()]
    hits.to_csv(store.output_path("detector_cases.csv"), index=False)

    # Plots: 2 cases where a detector fired + 1 covered case where it didn't (most weight data first)
    plot_dir = store.output_path("detectors")
    plot_dir.mkdir(exist_ok=True)
    by_data = lambda df: df.sort_values("weight_points_in_window", ascending=False)  # noqa: E731
    missed = covered[~covered["episode_id"].isin(fired["episode_id"])]
    picks = pd.concat([by_data(fired).head(2), by_data(missed).head(1)])
    by_id = cases.set_index("episode_id", drop=False)
    for episode_id in picks["episode_id"]:
        plot_case(by_id.loc[episode_id], weights, det_w, plot_dir / f"case_{episode_id}.png")

    from src.schema_check import _md_table
    n_cases = int(cases["is_case"].sum())
    parts = [
        f"# Detectors — zoo {cfg['zoo_id']}\n",
        f"- Weight: {len(det_w):,} readings scored, {(det_w['status'] == 'ok').mean():.0%} with enough baseline, "
        f"{int(det_w['fired'].sum())} fired ({int(det_w['confirmed'].sum())} confirmed)",
        f"- Leftovers: {len(det_i):,} days scored, {int((det_i['status'] == 'ok').sum()) if len(det_i) else 0} "
        f"with enough baseline, {int(det_i['cusum_flag'].sum()) if len(det_i) else 0} CUSUM flags",
        f"- Cases: {n_cases:,}; with detector data in [T0−{cfg['labels']['early_window_days']} d, T0): {len(covered)}; "
        f"**detector fired before T0: {len(fired)}**",
        f"- Checkpoint M2 (fires before T0 on ≥ 1 case): **{'GO' if len(fired) else 'NO-GO'}**\n",
        "## Cases with detector coverage\n",
        _md_table(covered.sort_values("weight_lead_days", ascending=False)),
        "\n## Plots\n", "\n".join(f"- `outputs/detectors/case_{e}.png`" for e in picks["episode_id"]),
    ]
    store.output_path("detector_cases.md").write_text("\n".join(parts))
    print("\n".join(parts[1:5]))
    print("→ outputs/detector_cases.md, outputs/detectors/*.png")
