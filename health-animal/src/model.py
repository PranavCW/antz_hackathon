"""Stage model: rule-based baseline vs regularised logistic regression on the animal-day table (PLAN.md §H).

Target (config model.target, default label_vet): P(a vet-aware case starts in the next 14 days).
Validation — all by time or by animal, never random rows:
  * time split: train on days older than the split date (minus a purge gap = the 60-day label horizon, so no
    training label looks into the test period), test on the newer part; metric PR-AUC;
  * grouped 5-fold CV by animal over the whole history (a case belongs to one animal, so this is the
    leave-cases-out version of the plan's leave-one-case-out);
  * bootstrap over animals for the model-vs-rules PR-AUC difference (few cases → wide intervals).
Scores for rows the model was trained on are out-of-fold, so nothing downstream sees in-sample risk.

Outputs: data/processed/predictions.parquet, model_logreg.json; outputs/model_report.md, model_coefficients.csv.
"""

import json
import warnings

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, roc_auc_score
from sklearn.model_selection import GroupKFold

from src import config, store
from src.features import feature_columns
from src.schema_check import _md_table

# macOS Accelerate BLAS raises spurious floating-point flags inside matmul; run() asserts inputs are finite and
# fit_logreg clips them, so these warnings carry no information.
warnings.filterwarnings("ignore", message=".*encountered in matmul", category=RuntimeWarning)

RULE_COMPONENTS = ["weight_drop", "weight_confirmed", "intake_cusum", "notes_one", "notes_two_plus", "priority_high"]
C_GRID = [0.001, 0.01, 0.1, 1.0]
# Columns that describe how much data an animal has, not how it is doing
COVERAGE_ONLY = {"w_days_since_last", "w_n_180d", "w_has_baseline", "n_notes_30d", "n_volume_ratio", "i_logged_30d",
                 "i_low_confidence"}


# ── Rule baseline ────────────────────────────────────────────────────────────────────────────────────────

def rule_components(df: pd.DataFrame, cfg: dict) -> pd.DataFrame:
    """Hand-set contribution of each firing rule (config model.rule_weights); the score is their capped sum."""
    rw = cfg["model"]["rule_weights"]
    c = pd.DataFrame(index=df.index)
    c["weight_drop"] = rw["weight_fired"] * (df["w_fired"] > 0)
    c["weight_confirmed"] = rw["weight_confirmed"] * (df["w_confirmed"] > 0)
    c["intake_cusum"] = rw["intake_cusum"] * (df["i_cusum_max_7"] >= cfg["intake"]["cusum_h"])
    c["notes_one"] = rw["notes_one_category"] * (df["n_neg_cats_7d"] >= 1)
    c["notes_two_plus"] = rw["notes_two_categories"] * (df["n_neg_cats_7d"] >= 2)
    # a High/Critical priority only counts when a note also says something got worse
    c["priority_high"] = rw["priority_high"] * ((df["n_max_priority_7d"] >= 3) & (df["n_neg_cats_7d"] >= 1))
    return c[RULE_COMPONENTS]


def rule_score(df: pd.DataFrame, cfg: dict) -> pd.Series:
    return rule_components(df, cfg).sum(axis=1).clip(upper=1.0)


# ── Logistic regression ──────────────────────────────────────────────────────────────────────────────────

SCALE_FLOOR, Z_CLIP = 0.1, 10.0


def standardise(X: np.ndarray, mean: np.ndarray, scale: np.ndarray) -> np.ndarray:
    """(x − mean) / scale clipped to ±10: a flag that is almost never set must not become a 1,000-sigma outlier."""
    return np.clip((X - mean) / scale, -Z_CLIP, Z_CLIP)


class Scaler:
    def __init__(self, X: np.ndarray):
        self.mean_ = X.mean(axis=0)
        self.scale_ = np.maximum(X.std(axis=0), SCALE_FLOOR)

    def transform(self, X: np.ndarray) -> np.ndarray:
        return standardise(X, self.mean_, self.scale_)


def fit_logreg(X: np.ndarray, y: np.ndarray, C: float, seed: int):
    scaler = Scaler(X)
    clf = LogisticRegression(C=C, class_weight="balanced", max_iter=2000, random_state=seed)
    clf.fit(scaler.transform(X), y)
    return scaler, clf


def predict(scaler, clf, X: np.ndarray) -> np.ndarray:
    return clf.predict_proba(scaler.transform(X))[:, 1]


def grouped_oof(X, y, groups, C, seed, n_splits=5, X_extra=None, g_extra=None):
    """Out-of-fold probabilities with whole animals held out together.

    With X_extra / g_extra (unlabelled rows of the same period) returns (oof, oof_extra): each extra row is scored
    by the fold model that did not see its animal; animals absent from `groups` stay NaN.
    """
    oof = np.full(len(y), np.nan)
    oof_extra = np.full(0 if X_extra is None else len(X_extra), np.nan)
    n_splits = min(n_splits, len(np.unique(groups)))
    for tr, te in GroupKFold(n_splits=n_splits).split(X, y, groups):
        if y[tr].sum() < 2 or y[tr].sum() == len(tr):
            continue
        scaler, clf = fit_logreg(X[tr], y[tr], C, seed)
        oof[te] = predict(scaler, clf, X[te])
        if X_extra is not None:
            hit = np.isin(g_extra, np.unique(groups[te]))
            if hit.any():
                oof_extra[hit] = predict(scaler, clf, X_extra[hit])
    return oof if X_extra is None else (oof, oof_extra)


def pr_auc(y, s) -> float:
    ok = ~np.isnan(s)
    return float(average_precision_score(y[ok], s[ok])) if ok.any() and y[ok].sum() > 0 else float("nan")


def bootstrap_diff(y, s_a, s_b, groups, n=200, seed=0) -> dict:
    """Resample animals with replacement; PR-AUC(a) − PR-AUC(b). Returns mean, 95% interval, P(a > b)."""
    rng = np.random.default_rng(seed)
    idx = pd.Series(np.arange(len(y))).groupby(groups).apply(np.array)
    diffs = []
    for _ in range(n):
        pick = np.concatenate(idx.sample(len(idx), replace=True, random_state=int(rng.integers(1 << 31))).to_list())
        if y[pick].sum() == 0:
            continue
        diffs.append(average_precision_score(y[pick], s_a[pick]) - average_precision_score(y[pick], s_b[pick]))
    d = np.array(diffs)
    if len(d) < 20:
        return {"mean": np.nan, "lo": np.nan, "hi": np.nan, "p_better": np.nan, "reps": len(d)}
    return {"mean": float(d.mean()), "lo": float(np.percentile(d, 2.5)), "hi": float(np.percentile(d, 97.5)),
            "p_better": float((d > 0).mean()), "reps": len(d)}


def time_split(df: pd.DataFrame, target: str, test_fraction: float, purge_days: int):
    """(split_date, train mask, purge mask, test mask) over rows with a label."""
    lab = df[target].notna()
    split = df.loc[lab, "date"].quantile(1 - test_fraction, interpolation="nearest")
    train = lab & (df["date"] < split - pd.Timedelta(days=purge_days))
    purge = lab & (df["date"] >= split - pd.Timedelta(days=purge_days)) & (df["date"] < split)
    test = lab & (df["date"] >= split)
    return split, train, purge, test


# ── Stage ────────────────────────────────────────────────────────────────────────────────────────────────

def run() -> None:
    cfg = config.load()
    mcfg = cfg["model"]
    target = mcfg["target"]
    df = store.read("processed", "animal_day").sort_values(["animal_id", "date"]).reset_index(drop=True)
    cols = feature_columns(df)
    X = df[cols].to_numpy(float)
    assert np.isfinite(X).all(), "non-finite feature values"
    y_all = df[target].to_numpy(float)

    split, tr, pu, te = time_split(df, target, mcfg["test_fraction"], cfg["labels"]["negative_clear_days"])
    df["period"] = np.select([tr, pu, te], ["train", "purge", "test"], default="unlabelled")
    ytr, yte = y_all[tr].astype(int), y_all[te].astype(int)
    gtr, gte = df.loc[tr, "animal_id"].to_numpy(), df.loc[te, "animal_id"].to_numpy()

    df["risk_rules"] = rule_score(df, cfg)
    n_pos_tr = int(ytr.sum())
    report = [f"# Model — zoo {cfg['zoo_id']}\n",
              f"Target `{target}`; time split at **{split:%Y-%m-%d}** (purge {cfg['labels']['negative_clear_days']} d). "
              f"Train {int(tr.sum()):,} rows / {n_pos_tr} positives / {len(np.unique(gtr))} animals; "
              f"test {int(te.sum()):,} rows / {int(yte.sum())} positives / {len(np.unique(gte))} animals.\n"]

    if n_pos_tr < mcfg["min_train_positives"]:
        df["risk_logreg"] = np.nan
        report.append(f"**Logistic regression skipped: only {n_pos_tr} training positives "
                      f"(< {mcfg['min_train_positives']}). The rule scorer is the result.**")
        store.write("processed", "predictions", _predictions(df))
        _write_summary(cfg, target, split, "rules", "SKIPPED — too few training positives")
        store.output_path("model_report.md").write_text("\n".join(report))
        print("\n".join(report[1:]))
        return

    # 1. Pick C by grouped CV on the training period
    scores = {}
    for C in C_GRID:
        scores[C] = pr_auc(ytr, grouped_oof(X[tr], ytr, gtr, C, mcfg["random_seed"]))
    best_C = max(scores, key=lambda c: (np.nan_to_num(scores[c], nan=-1), -c))

    # 2. Final train-period model; out-of-fold scores for train rows, model scores for everything else
    scaler, clf = fit_logreg(X[tr], ytr, best_C, mcfg["random_seed"])
    risk = predict(scaler, clf, X)
    # Every row in the training period gets an out-of-fold score, labelled or not (the backtest replays them all)
    early = (df["date"] < split - pd.Timedelta(days=cfg["labels"]["negative_clear_days"])).to_numpy()
    extra = early & ~tr.to_numpy()
    risk[tr.to_numpy()], risk[extra] = grouped_oof(
        X[tr], ytr, gtr, best_C, mcfg["random_seed"], X_extra=X[extra], g_extra=df.loc[extra, "animal_id"].to_numpy())
    df["risk_logreg"] = risk

    # 3. Held-out metrics
    prev = float(yte.mean()) if len(yte) else float("nan")
    s_rules, s_lr = df.loc[te, "risk_rules"].to_numpy(), df.loc[te, "risk_logreg"].to_numpy()
    boot = bootstrap_diff(yte, s_lr, s_rules, gte, seed=mcfg["random_seed"])
    rows = [{"scorer": "prevalence (random)", "test_pr_auc": prev},
            {"scorer": "rules", "test_pr_auc": pr_auc(yte, s_rules)},
            {"scorer": f"logistic (C={best_C})", "test_pr_auc": pr_auc(yte, s_lr)}]
    if yte.sum() > 0 and yte.sum() < len(yte):
        rows[1]["test_auroc"] = float(roc_auc_score(yte, s_rules))
        rows[2]["test_auroc"] = float(roc_auc_score(yte, s_lr))
    other = "label_all" if target == "label_vet" else "label_vet"
    te_o = te & df[other].notna()
    if df.loc[te_o, other].sum() > 0:
        yo = df.loc[te_o, other].to_numpy(int)
        rows[1][f"test_pr_auc_{other}"] = pr_auc(yo, df.loc[te_o, "risk_rules"].to_numpy())
        rows[2][f"test_pr_auc_{other}"] = pr_auc(yo, df.loc[te_o, "risk_logreg"].to_numpy())

    # 3b. Ablation: same model on coverage/context columns only. If it scores like the full model, the model is
    #     reading *who gets watched*, not *who is getting sick* — the headline would then be an artefact.
    signal_cols = [i for i, c in enumerate(cols) if c.startswith(("w_", "i_", "n_")) and c not in COVERAGE_ONLY]
    context_cols = [i for i in range(len(cols)) if i not in signal_cols]
    ablation = {}
    for name, sel in (("health signals only", signal_cols), ("context + coverage only", context_cols)):
        sc, cl = fit_logreg(X[tr][:, sel], ytr, best_C, mcfg["random_seed"])
        ablation[name] = pr_auc(yte, predict(sc, cl, X[te][:, sel]))

    # 4. Grouped CV over the whole history (leave-cases-out)
    lab = df[target].notna().to_numpy()
    yl, gl = y_all[lab].astype(int), df.loc[lab, "animal_id"].to_numpy()
    cv_lr = pr_auc(yl, grouped_oof(X[lab], yl, gl, best_C, mcfg["random_seed"]))
    cv_rules = pr_auc(yl, df.loc[lab, "risk_rules"].to_numpy())

    pa_lr, pa_rules = rows[2]["test_pr_auc"], rows[1]["test_pr_auc"]
    pa_sig, pa_ctx = ablation["health signals only"], ablation["context + coverage only"]
    if np.isnan(pa_lr) or np.isnan(boot["lo"]):
        verdict = "INCONCLUSIVE — too few held-out positives"
    elif boot["lo"] > 0 and not (pa_sig > pa_ctx and pa_sig > prev):
        verdict = ("NOT CREDIBLE — logistic beats the rules, but a context/coverage-only model does as well or better "
                   "than the health-signal model: it is learning who gets observed, not who gets sick")
    elif boot["lo"] > 0:
        verdict = "GO — logistic beats the rules on held-out data (95% interval above 0) and health signals carry it"
    elif pa_lr > pa_rules:
        verdict = "NOT DISTINGUISHABLE — logistic is ahead but the interval includes 0; present both"
    else:
        verdict = "NO-GO — logistic does not beat the rules; present the rule scorer as the result"

    coef = pd.DataFrame({"feature": cols, "coef": clf.coef_[0], "mean": scaler.mean_, "scale": scaler.scale_})
    coef = coef.reindex(coef["coef"].abs().sort_values(ascending=False).index)
    coef.to_csv(store.output_path("model_coefficients.csv"), index=False)
    (cfg["paths"]["processed"] / "model_logreg.json").write_text(json.dumps({
        "target": target, "features": cols, "mean": scaler.mean_.tolist(), "scale": scaler.scale_.tolist(),
        "coef": clf.coef_[0].tolist(), "intercept": float(clf.intercept_[0]), "C": best_C,
        "split_date": f"{split:%Y-%m-%d}", "train_rows": int(tr.sum()), "train_positives": n_pos_tr}))
    store.write("processed", "predictions", _predictions(df))
    _write_summary(cfg, target, split, "logreg" if verdict.startswith("GO") else "rules", verdict)

    report += [
        f"## Checkpoint M3: {verdict}\n",
        "### Held-out period (time split)\n", _md_table(pd.DataFrame(rows).round(4)),
        f"\nPR-AUC difference logistic − rules (bootstrap over animals, {boot['reps']} reps): "
        f"**{boot['mean']:+.4f}** [95% {boot['lo']:+.4f}, {boot['hi']:+.4f}], P(logistic better) = {boot['p_better']:.2f}\n",
        "### Grouped 5-fold CV over the whole history (animals held out together)\n",
        f"PR-AUC logistic **{cv_lr:.4f}** vs rules **{cv_rules:.4f}** (prevalence {yl.mean():.4f}, "
        f"{int(yl.sum())} positives)\n",
        "### C selection (grouped CV on train period, PR-AUC)\n", ", ".join(f"C={c}: {v:.4f}" for c, v in scores.items()),
        "\n### Ablation (held-out PR-AUC, same C)\n", _md_table(pd.DataFrame(
            [{"features": k, "test_pr_auc": round(v, 4)} for k, v in ablation.items()])),
        "\n### Strongest coefficients (standardised features)\n", _md_table(coef.head(12).round(4)),
    ]
    if int(yte.sum()) < 30:
        report.append(f"\n> ⚠️ Only {int(yte.sum())} held-out positives — treat every number above as indicative.")
    store.output_path("model_report.md").write_text("\n".join(report))
    print("\n".join(report[1:11]))
    print("→ outputs/model_report.md, data/processed/predictions.parquet")


def _write_summary(cfg: dict, target: str, split, primary: str, verdict: str) -> None:
    """The backtest reads which scorer is the result: logistic only if it genuinely beat the rules."""
    (cfg["paths"]["processed"] / "model_summary.json").write_text(json.dumps(
        {"target": target, "split_date": f"{split:%Y-%m-%d}", "primary_scorer": primary, "verdict": verdict}))


def _predictions(df: pd.DataFrame) -> pd.DataFrame:
    keep = ["animal_id", "date", "period", "label_vet", "label_all", "excl_reason", "risk_rules", "risk_logreg"]
    return df[keep + feature_columns(df)]
