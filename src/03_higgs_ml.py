"""Part 3 - Finding the Higgs boson (H -> tau tau) with gradient-boosted trees.

Input : data/raw/atlas-higgs-challenge-2014-v2.csv.gz
        (ATLAS Higgs Boson ML Challenge 2014, CERN Open Data record 328, 818k
        simulated events, labelled signal 's' / background 'b')

Protocol follows the original Kaggle challenge exactly so the score is
comparable to the 2014 leaderboard:
  * train on KaggleSet == 't' (250k events)
  * pick the selection threshold on a held-out validation split of 't'
  * report AMS on the public ('b', 100k) and private ('v', 450k) test sets,
    using the official KaggleWeight and AMS definition (b_reg = 10).

Output: figures/higgs_*.png, results/higgs_ml.json
"""
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score, roc_curve
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

ROOT = Path(__file__).resolve().parent.parent
RAW = ROOT / "data/raw/atlas-higgs-challenge-2014-v2.csv.gz"
FIG, RES = ROOT / "figures", ROOT / "results"
SEED = 42


def ams(s, b, b_reg=10.0):
    """Approximate Median Significance - the challenge's official metric."""
    return np.sqrt(2 * ((s + b + b_reg) * np.log(1 + s / (b + b_reg)) - s))


def ams_curve(score, y, w, fractions):
    """AMS when selecting the top `f` fraction of events by score."""
    order = np.argsort(-score)
    y, w = y[order], w[order]
    s_cum, b_cum = np.cumsum(w * y), np.cumsum(w * (1 - y))
    idx = np.clip((fractions * len(score)).astype(int) - 1, 0, len(score) - 1)
    return ams(s_cum[idx], b_cum[idx])


def ams_at_fraction(score, y, w, frac):
    return float(ams_curve(score, y, w, np.array([frac]))[0])


def add_features(df):
    """Physics-motivated extras: azimuthal angle differences are rotation
    invariant, the raw phi angles are not."""
    def dphi(a, b):
        d = np.abs(df[a] - df[b])
        return np.where(d > np.pi, 2 * np.pi - d, d)

    out = df.copy()
    out["DPHI_tau_lep"] = dphi("PRI_tau_phi", "PRI_lep_phi")
    out["DPHI_tau_met"] = dphi("PRI_tau_phi", "PRI_met_phi")
    out["DPHI_lep_met"] = dphi("PRI_lep_phi", "PRI_met_phi")
    jet = df["PRI_jet_leading_phi"].notna()
    out["DPHI_jet_met"] = np.where(jet, dphi("PRI_jet_leading_phi", "PRI_met_phi"), np.nan)
    return out.drop(columns=[c for c in df.columns if c.endswith("_phi")])


def main():
    FIG.mkdir(exist_ok=True)
    RES.mkdir(exist_ok=True)
    df = pd.read_csv(RAW).replace(-999.0, np.nan)
    df["y"] = (df.Label == "s").astype(int)
    feats_raw = [c for c in df.columns if c.startswith(("DER_", "PRI_"))]
    X_all = add_features(df[feats_raw])
    feats = list(X_all.columns)

    train = df.KaggleSet == "t"
    X, y, w = X_all[train].values, df.y[train].values, df.KaggleWeight[train].values
    X_tr, X_va, y_tr, y_va, w_tr, w_va = train_test_split(
        X, y, w, test_size=0.2, random_state=SEED, stratify=y)
    w_va_scaled = w_va / 0.2  # re-normalise so the split represents the full luminosity

    # balance classes for training (signal weights are tiny vs background)
    bal = np.where(y_tr == 1, w_tr / w_tr[y_tr == 1].sum(), w_tr / w_tr[y_tr == 0].sum())
    bal *= len(bal) / bal.sum()
    bal_va = np.where(y_va == 1, w_va / w_va[y_va == 1].sum(), w_va / w_va[y_va == 0].sum())
    bal_va *= len(bal_va) / bal_va.sum()

    model = xgb.XGBClassifier(
        n_estimators=3000, learning_rate=0.03, max_depth=6, min_child_weight=5,
        subsample=0.9, colsample_bytree=0.8, reg_lambda=1.0, tree_method="hist",
        eval_metric="auc", early_stopping_rounds=150, random_state=SEED, n_jobs=-1)
    model.fit(X_tr, y_tr, sample_weight=bal, eval_set=[(X_va, y_va)],
              sample_weight_eval_set=[bal_va], verbose=200)

    baseline = make_pipeline(SimpleImputer(strategy="median"), StandardScaler(),
                             LogisticRegression(max_iter=2000))
    baseline.fit(X_tr, y_tr, logisticregression__sample_weight=bal)

    # threshold = fraction of events selected, tuned on validation only
    fracs = np.linspace(0.05, 0.30, 251)
    va_xgb = model.predict_proba(X_va)[:, 1]
    va_lr = baseline.predict_proba(X_va)[:, 1]
    curve_xgb = ams_curve(va_xgb, y_va, w_va_scaled, fracs)
    curve_lr = ams_curve(va_lr, y_va, w_va_scaled, fracs)
    # smooth the noisy curve before picking the maximum
    k = 15
    smooth = np.convolve(curve_xgb, np.ones(k) / k, mode="same")
    best_frac = float(fracs[k:-k][np.argmax(smooth[k:-k])])
    best_frac_lr = float(fracs[np.argmax(curve_lr)])

    results = {"features": len(feats), "best_iteration": int(model.best_iteration),
               "selected_fraction": round(best_frac, 4)}
    for name, mask in {"public": df.KaggleSet == "b", "private": df.KaggleSet == "v"}.items():
        Xs, ys, ws = X_all[mask].values, df.y[mask].values, df.KaggleWeight[mask].values
        p_x, p_l = model.predict_proba(Xs)[:, 1], baseline.predict_proba(Xs)[:, 1]
        results[name] = {
            "n_events": int(mask.sum()),
            "xgboost_AMS": round(ams_at_fraction(p_x, ys, ws, best_frac), 4),
            "xgboost_AUC": round(roc_auc_score(ys, p_x, sample_weight=ws), 4),
            "logreg_AMS": round(ams_at_fraction(p_l, ys, ws, best_frac_lr), 4),
            "logreg_AUC": round(roc_auc_score(ys, p_l, sample_weight=ws), 4),
        }
        if name == "private":
            priv = (p_x, p_l, ys, ws)
    # Source: Adam-Bourdarios et al., "The Higgs boson machine learning
    # challenge", PMLR 42 (2015), private leaderboard table. 1785 teams took part.
    results["reference_2014_leaderboard_private_AMS"] = {
        "rank 1, G. Melis (DNN ensemble)": 3.80581,
        "rank 45, Crowwork (XGBoost)": 3.71885,
        "MultiBoost benchmark": 3.40488,
        "TMVA benchmark": 3.19956}

    # --- figures -------------------------------------------------------------
    p_x, p_l, ys, ws = priv
    fig, ax = plt.subplots(figsize=(6, 6))
    for p, lab, c in [(p_x, "XGBoost", "#d62728"), (p_l, "Logistic regression", "#7f7f7f")]:
        fpr, tpr, _ = roc_curve(ys, p, sample_weight=ws)
        ax.plot(fpr, tpr, color=c, lw=2,
                label=f"{lab} (AUC={roc_auc_score(ys, p, sample_weight=ws):.3f})")
    ax.plot([0, 1], [0, 1], "k:", lw=1)
    ax.set_xlabel("Background efficiency")
    ax.set_ylabel("Signal efficiency")
    ax.set_title("ROC - private test set (450k events)")
    ax.legend(loc="lower right")
    fig.tight_layout()
    fig.savefig(FIG / "higgs_roc.png", dpi=150)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(8, 5))
    bins = np.linspace(0, 1, 51)
    ax.hist(p_x[ys == 0], bins=bins, weights=ws[ys == 0], histtype="stepfilled",
            alpha=0.6, color="#1f77b4", label="Background (Z, ttbar, W)")
    ax.hist(p_x[ys == 1], bins=bins, weights=ws[ys == 1] * 50, histtype="step",
            lw=2, color="#d62728", label=r"Higgs signal $\times 50$")
    cut = np.quantile(p_x, 1 - best_frac)
    ax.axvline(cut, color="k", ls="--", label=f"Selection (top {best_frac:.0%})")
    ax.set_yscale("log")
    ax.set_xlabel("XGBoost signal score")
    ax.set_ylabel("Expected events (weighted)")
    ax.set_title(r"$H\to\tau\tau$ classifier output - private test set")
    ax.legend()
    fig.tight_layout()
    fig.savefig(FIG / "higgs_score.png", dpi=150)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(fracs * 100, curve_xgb, color="#d62728", alpha=0.35)
    ax.plot(fracs[k:-k] * 100, smooth[k:-k], color="#d62728", lw=2, label="XGBoost")
    ax.plot(fracs * 100, curve_lr, color="#7f7f7f", lw=2, label="Logistic regression")
    ax.axvline(best_frac * 100, color="k", ls="--", lw=1)
    ax.set_xlabel("Events selected [% of sample]")
    ax.set_ylabel("AMS (discovery significance)")
    ax.set_title("Choosing the selection threshold (validation split)")
    ax.legend()
    fig.tight_layout()
    fig.savefig(FIG / "higgs_ams_threshold.png", dpi=150)
    plt.close(fig)

    imp = pd.Series(model.get_booster().get_score(importance_type="gain"))
    imp.index = [feats[int(i[1:])] for i in imp.index]
    imp = imp.sort_values().tail(15)
    fig, ax = plt.subplots(figsize=(8, 6))
    ax.barh(imp.index, imp.values / imp.values.sum(), color="#1f4e8c")
    ax.set_xlabel("Relative importance (gain)")
    ax.set_title("Top 15 features")
    fig.tight_layout()
    fig.savefig(FIG / "higgs_feature_importance.png", dpi=150)
    plt.close(fig)

    results["top_features"] = list(imp.index[::-1][:5])
    (RES / "higgs_ml.json").write_text(json.dumps(results, indent=2))
    print(json.dumps(results, indent=2))


if __name__ == "__main__":
    main()
