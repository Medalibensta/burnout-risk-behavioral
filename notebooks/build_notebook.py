"""
Build and execute notebooks/06_burnout_risk.ipynb.

Light aggregates run live; heavy artefacts are displayed from src/ outputs,
keeping the notebook fast and single-sourced.

Run:
    python notebooks/build_notebook.py
"""
from __future__ import annotations

from pathlib import Path

import nbformat as nbf
from nbclient import NotebookClient

HERE = Path(__file__).resolve().parent
NB_PATH = HERE / "06_burnout_risk.ipynb"


def md(t): return nbf.v4.new_markdown_cell(t.strip())
def code(t): return nbf.v4.new_code_cell(t.strip())


cells = [
    md("""
# Prédiction du risque de burnout à partir de données comportementales

> ⚠️ **Cadre éthique** — outil d'**alerte précoce** (RH / bien-être personnel),
> pas de diagnostic médical. Données **entièrement synthétiques**. Un tel
> système soulève des enjeux forts de **vie privée** et de **biais** discutés en
> fin de notebook — ils font partie du livrable.

**Angle décisionnel** — repérer, à partir de signaux comportementaux (temps
d'écran, travail, sommeil, mobilité, sociabilité), des profils à risque de
surcharge **avant** qu'ils ne deviennent critiques.

**Données** — aucun jeu réel de traces comportementales brutes labellisées
n'est éthiquement exploitable publiquement. On **simule** 1 200 personnes ×
56 jours avec une structure causale explicite : un stress latent caché génère
les comportements, et le **label de burnout est tiré des comportements
réellement observés** (pas du latent) — le modèle doit donc apprendre un signal
*récupérable*, comme un vrai système d'alerte.

**Plan** : données → features → profils (non supervisé) → score de risque
(supervisé) → SHAP → limites & éthique.
"""),
    code("""
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from IPython.display import Image, display

ROOT = Path.cwd().parent if Path.cwd().name == "notebooks" else Path.cwd()
sys.path.insert(0, str(ROOT / "src"))

from features import load_features, FEATURE_COLS

feats = load_features()
print(f"{len(feats)} personnes | {len(FEATURE_COLS)} features")
print(f"prévalence burnout : {feats.burnout.mean()*100:.1f}%")
feats[["person_id"] + FEATURE_COLS + ["burnout"]].head()
"""),
    md("""
## 1. Des features interprétables, en trois familles

**Surcharge** (travail, écran du soir, travail le week-end), **récupération**
(sommeil, pas, messages sociaux), **régularité** (variabilité du sommeil et de
l'heure de coucher, rupture de routine). On regarde leur corrélation brute avec
l'issue — diagnostic, pas modèle.
"""),
    code("""
corr = feats[FEATURE_COLS + ["burnout"]].corr()["burnout"].drop("burnout")
display(corr.sort_values(key=abs, ascending=False).round(3).to_frame(
    "corrélation avec burnout"))
"""),
    md("""
**Lecture** — l'écran du soir (+0,71), le retrait social (−0,70), l'irrégularité
du coucher (+0,67) et les heures de travail (+0,65) sont les plus liés. La
*rupture de routine* est faible ici (fenêtre courte, sans tendance forte) — une
feature honnêtement peu informative, conservée pour transparence.
"""),
    md("""
## 2. Profils comportementaux (non supervisé)

K-Means (K par silhouette) sur les features standardisées, **sans utiliser le
label**. On regarde ensuite le taux de burnout par profil : une segmentation
purement comportementale porte-t-elle déjà le risque ?
"""),
    code("""
prof = pd.read_csv(ROOT / "reports" / "cluster_profiles.csv")
display(prof.round(2))
display(Image(ROOT / "reports" / "figures" / "behavioral_clusters.png", width=880))
display(Image(ROOT / "reports" / "figures" / "dendrogram.png", width=720))
"""),
    md("""
**Lecture** — deux archétypes émergent : un profil **« surchargé »** (travail et
écran élevés, sommeil bas, peu de pas/social) avec **~59 % de burnout**, et un
profil **« équilibré »** à **~2 %**. La segmentation comportementale seule
sépare donc déjà fortement le risque — et le dendrogramme Ward confirme la
stabilité du nombre de profils.
"""),
    md("""
## 3. Score de risque supervisé

Classifieur (Gradient Boosting vs régression logistique) sur les 9 features
(le stress latent et la probabilité vraie sont **retirés** — pas de fuite).
Métriques adaptées : ROC-AUC et PR-AUC sur un test tenu à l'écart.
"""),
    code("""
metrics = pd.read_csv(ROOT / "reports" / "model_metrics.csv")
display(metrics.round(3))
"""),
    md("""
**Lecture** — le risque est bien prédictible (ROC-AUC ≈ 0,95-0,96 ; PR-AUC
≈ 0,89-0,91). La régression logistique fait jeu égal, ce qui est rassurant : le
signal est largement **linéaire et interprétable**, pas un artefact d'un modèle
complexe.
"""),
    md("""
## 4. Explicabilité (SHAP) — pourquoi ce score ?

Un outil RH/bien-être doit être **transparent et actionnable**. SHAP explique le
modèle globalement (quels comportements pèsent) et individuellement (le
décompte de risque d'une personne).
"""),
    code("""
for fig in ["shap_beeswarm.png", "shap_waterfall.png"]:
    display(Image(ROOT / "reports" / "figures" / fig, width=640))
imp = pd.read_csv(ROOT / "reports" / "shap_importance.csv")
display(imp.round(3))
"""),
    md("""
**Lecture** — les facteurs dominants (écran du soir, heures de travail, manque
de sommeil, travail le week-end, retrait social) sont **cohérents et
actionnables** : ils suggèrent des leviers concrets (hygiène numérique du soir,
charge de travail, récupération le week-end). Le *waterfall* fournit la vue
individuelle indispensable à un accompagnement personnalisé.
"""),
    md("""
## 5. Limites & éthique *(centrales pour ce projet)*

**Données**
- **Entièrement synthétiques** : les performances élevées reflètent en partie la
  structure causale que nous avons encodée. Sur données réelles, bruitées et
  confondues, les scores seraient plus bas et les biais bien réels.
- Un vrai déploiement devrait combiner un jeu proche (ex. *Mental Health in Tech
  Survey*) et une collecte consentie, jamais des traces à l'insu des personnes.

**Éthique & vie privée (non négociable)**
- **Surveillance** : suivre écran/mobilité/sommeil est hautement intrusif. Un
  tel outil n'est acceptable qu'avec **consentement explicite**, **contrôle par
  la personne**, minimisation et traitement local/chiffré.
- **Finalité** : à visée de **prévention et de soutien**, jamais de sanction, de
  tri RH ou de décision d'emploi. Un score de risque ne doit pas devenir un
  outil de contrôle.
- **Biais & faux positifs** : un modèle appris sur une population peut mal
  généraliser (âge, culture, neurodiversité) ; un faux positif peut stigmatiser.
  Score explicable, seuil transparent, et **humain dans la boucle** obligatoires.
- **Pas un diagnostic** : le burnout est clinique ; cet outil signale un
  *risque comportemental*, à confronter à un accompagnement humain qualifié.
"""),
]


def main() -> None:
    nb = nbf.v4.new_notebook()
    nb.metadata.kernelspec = {"display_name": "Python 3", "language": "python",
                              "name": "python3"}
    nb.cells = cells
    print(f"[notebook] executing {len(cells)} cells...")
    NotebookClient(nb, timeout=600, kernel_name="python3",
                   resources={"metadata": {"path": str(HERE)}}).execute()
    nbf.write(nb, NB_PATH)
    print(f"[notebook] written -> {NB_PATH}")


if __name__ == "__main__":
    main()
