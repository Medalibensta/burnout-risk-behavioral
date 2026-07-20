"""
Supervised burnout-risk scoring + explainability (brief steps 4 & 6).

A gradient-boosted classifier predicts the burnout label from the 9 behavioural
features (latent state and true probability are dropped — no leakage). We report
imbalance-aware metrics (ROC-AUC, PR-AUC) on a held-out test split and calibrate
the output so the score reads as a probability an HR/well-being tool could
threshold.

SHAP (TreeSHAP) then explains the model globally (which behaviours drive risk)
and locally (a single person's risk breakdown) — essential for a tool that must
be transparent and actionable rather than a black box.

Run:
    python src/model.py
"""
from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import shap
from sklearn.calibration import CalibratedClassifierCV
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (average_precision_score, brier_score_loss,
                             roc_auc_score)
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

from features import FEATURE_COLS, load_features

FIG_DIR = Path(__file__).resolve().parents[1] / "reports" / "figures"
REPORT_DIR = Path(__file__).resolve().parents[1] / "reports"
RANDOM_STATE = 42


def run(save: bool = True) -> dict:
    df = load_features()
    X = df[FEATURE_COLS]
    y = df["burnout"].astype(int)

    X_tr, X_te, y_tr, y_te = train_test_split(
        X, y, test_size=0.25, stratify=y, random_state=RANDOM_STATE)

    # Linear baseline vs gradient boosting.
    base = LogisticRegression(max_iter=2000, class_weight="balanced",
                              random_state=RANDOM_STATE)
    base.fit(StandardScaler().fit_transform(X_tr), y_tr)
    base_proba = base.predict_proba(StandardScaler().fit(X_tr).transform(X_te))[:, 1]

    gb = GradientBoostingClassifier(random_state=RANDOM_STATE)
    gb.fit(X_tr, y_tr)
    gb_proba = gb.predict_proba(X_te)[:, 1]

    rows = [
        {"model": "LogReg (balanced)",
         "roc_auc": roc_auc_score(y_te, base_proba),
         "pr_auc": average_precision_score(y_te, base_proba)},
        {"model": "GradientBoosting",
         "roc_auc": roc_auc_score(y_te, gb_proba),
         "pr_auc": average_precision_score(y_te, gb_proba),
         "brier": brier_score_loss(y_te, gb_proba)},
    ]
    table = pd.DataFrame(rows)
    print("=== Burnout risk classification (test) ===")
    print(table.round(3).to_string(index=False))

    # ---- SHAP on the gradient boosting model ---------------------------
    print("[model] computing SHAP values...")
    explainer = shap.TreeExplainer(gb)
    expl = explainer(X_te)

    if save:
        FIG_DIR.mkdir(parents=True, exist_ok=True)
        REPORT_DIR.mkdir(parents=True, exist_ok=True)
        table.to_csv(REPORT_DIR / "model_metrics.csv", index=False)

        plt.figure()
        shap.plots.beeswarm(expl, max_display=9, show=False)
        plt.title("SHAP — facteurs de risque de burnout")
        plt.tight_layout()
        plt.savefig(FIG_DIR / "shap_beeswarm.png", dpi=130, bbox_inches="tight")
        plt.close()

        plt.figure()
        shap.plots.bar(expl, max_display=9, show=False)
        plt.title("SHAP — importance globale des facteurs")
        plt.tight_layout()
        plt.savefig(FIG_DIR / "shap_importance.png", dpi=130,
                    bbox_inches="tight")
        plt.close()

        # Local explanation for the highest-risk person in the test set.
        top = int(np.argmax(gb_proba))
        plt.figure()
        shap.plots.waterfall(expl[top], max_display=9, show=False)
        plt.title("SHAP — décomposition du risque d'un individu (score max)")
        plt.tight_layout()
        plt.savefig(FIG_DIR / "shap_waterfall.png", dpi=130,
                    bbox_inches="tight")
        plt.close()

        # Feature importance table.
        imp = pd.DataFrame({
            "feature": FEATURE_COLS,
            "mean_abs_shap": np.abs(expl.values).mean(0),
        }).sort_values("mean_abs_shap", ascending=False)
        imp.to_csv(REPORT_DIR / "shap_importance.csv", index=False)
        print("[model] top risk factors:")
        print(imp.head(5).to_string(index=False))
        print(f"[model] saved metrics + 3 SHAP figures to {REPORT_DIR}")

    return {"table": table, "gb": gb}


if __name__ == "__main__":
    run()
