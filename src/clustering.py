"""
Unsupervised behavioural profiling (brief step 3).

Before any supervised label, we ask: are there natural *archetypes* of
behaviour? K-Means on the standardised feature set, with K chosen by silhouette,
groups people into profiles. We then look at each cluster's average burnout rate
— WITHOUT having used the label to build the clusters — to see whether a
purely behavioural segmentation already carries risk signal.

Ward hierarchical clustering is run as a cross-check (dendrogram) to confirm the
number of profiles is stable across algorithms.

Run:
    python src/clustering.py
"""
from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.cluster.hierarchy import dendrogram, linkage
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score
from sklearn.preprocessing import StandardScaler

from features import FEATURE_COLS, load_features

FIG_DIR = Path(__file__).resolve().parents[1] / "reports" / "figures"
REPORT_DIR = Path(__file__).resolve().parents[1] / "reports"
RANDOM_STATE = 42


def choose_k(X, ks=range(2, 7)) -> int:
    best_k, best_s = 2, -1.0
    for k in ks:
        labels = KMeans(k, random_state=RANDOM_STATE, n_init=10).fit_predict(X)
        s = silhouette_score(X, labels)
        if s > best_s:
            best_k, best_s = k, s
    return best_k


def run(save: bool = True) -> pd.DataFrame:
    df = load_features()
    X = StandardScaler().fit_transform(df[FEATURE_COLS])

    k = choose_k(X)
    km = KMeans(k, random_state=RANDOM_STATE, n_init=10)
    df["cluster"] = km.fit_predict(X)
    print(f"[clustering] K={k} chosen by silhouette")

    # Describe clusters (mean features + burnout rate, label NOT used to fit).
    summary = df.groupby("cluster").agg(
        n=("person_id", "count"),
        burnout_rate=("burnout", "mean"),
        work=("work_hours_mean", "mean"),
        screen=("screen_evening_mean", "mean"),
        sleep=("sleep_mean", "mean"),
        sleep_var=("sleep_var", "mean"),
        steps=("steps_mean", "mean"),
        social=("social_mean", "mean"),
    ).round(2).sort_values("burnout_rate", ascending=False)
    print(summary.to_string())

    coords = PCA(2, random_state=RANDOM_STATE).fit_transform(X)
    df["pc1"], df["pc2"] = coords[:, 0], coords[:, 1]

    if save:
        FIG_DIR.mkdir(parents=True, exist_ok=True)
        REPORT_DIR.mkdir(parents=True, exist_ok=True)
        summary.to_csv(REPORT_DIR / "cluster_profiles.csv")
        df[["person_id", "cluster", "burnout"]].to_csv(
            REPORT_DIR / "person_clusters.csv", index=False)

        # PCA scatter coloured by cluster, marker by burnout.
        fig, axes = plt.subplots(1, 2, figsize=(14, 6))
        for c in sorted(df.cluster.unique()):
            sub = df[df.cluster == c]
            axes[0].scatter(sub.pc1, sub.pc2, s=12, alpha=0.6,
                            label=f"profil {c} ({sub.burnout.mean()*100:.0f}% burnout)")
        axes[0].set(xlabel="PC1", ylabel="PC2",
                    title=f"Profils comportementaux (K-Means, K={k})")
        axes[0].legend(fontsize=8)

        # Burnout rate per cluster bar.
        sr = summary.burnout_rate.sort_values()
        axes[1].barh([f"profil {i}" for i in sr.index], sr.values * 100,
                     color=plt.cm.RdYlGn_r(sr.values))
        axes[1].set(xlabel="% burnout dans le profil",
                    title="Risque de burnout par profil (label non utilisé)")
        plt.tight_layout()
        plt.savefig(FIG_DIR / "behavioral_clusters.png", dpi=130)
        plt.close()

        # Hierarchical cross-check on a subsample (dendrogram).
        sub_idx = np.random.default_rng(RANDOM_STATE).choice(
            len(X), size=min(120, len(X)), replace=False)
        Z = linkage(X[sub_idx], method="ward")
        plt.figure(figsize=(11, 4.5))
        dendrogram(Z, no_labels=True, color_threshold=0.7 * Z[:, 2].max())
        plt.title("Classification hiérarchique (Ward) — vérification du nombre de profils")
        plt.ylabel("distance")
        plt.tight_layout()
        plt.savefig(FIG_DIR / "dendrogram.png", dpi=130)
        plt.close()
        print(f"[clustering] saved scatter + dendrogram + tables to {REPORT_DIR}")

    return df


if __name__ == "__main__":
    run()
