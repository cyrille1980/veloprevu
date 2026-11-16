"""Entraînement, comparaison et enregistrement des modèles dans MLflow.

Trois familles sont comparées sur le jeu de validation :
- base saisonnière naïve (la demande de la même heure, 7 jours plus tôt) ;
- régression Ridge ;
- LightGBM (plusieurs jeux d'hyperparamètres, dont un objectif de Poisson).

Le meilleur est réentraîné sur train + validation, évalué sur le test, enregistré dans
le Model Registry, puis promu « champion » seulement s'il franchit la barrière de qualité.

Usage : python -m src.training.entrainement
"""

import json
from datetime import UTC, datetime

import mlflow
import numpy as np
import pandas as pd
from lightgbm import LGBMRegressor
from mlflow.models import infer_signature
from sklearn.base import BaseEstimator, RegressorMixin, clone
from sklearn.linear_model import Ridge
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from src.common.config import charger_config, dossier, uri_mlflow
from src.common.journal import obtenir_journal
from src.evaluation.evaluation import analyse_erreurs, metriques
from src.features.construction import CIBLE, FEATURES
from src.training import registre

journal = obtenir_journal("training.entrainement")


class BaseSaisonniere(RegressorMixin, BaseEstimator):
    """Modèle de référence : prédit la demande observée 7 jours plus tôt à la même heure."""

    def fit(self, X, y=None):
        self.est_ajuste_ = True
        return self

    def predict(self, X):
        return np.asarray(X["demande_h_168"], dtype=float)


def candidats(cfg: dict) -> list[tuple[str, object, dict]]:
    graine = cfg["projet"]["graine"]
    liste = [("base_saisonniere", BaseSaisonniere(), {})]
    for alpha in cfg["modeles"]["ridge"]["alpha"]:
        liste.append(("ridge", make_pipeline(StandardScaler(), Ridge(alpha=alpha)), {"alpha": alpha}))
    for params in cfg["modeles"]["lightgbm"]:
        modele = LGBMRegressor(random_state=graine, verbose=-1, n_jobs=2, **params)
        liste.append(("lightgbm", modele, dict(params)))
    return liste


def predire(modele, X: pd.DataFrame) -> np.ndarray:
    """Une demande ne peut pas être négative."""
    return np.clip(modele.predict(X[FEATURES]), 0, None)


def _jeux(cfg: dict, supplement: pd.DataFrame | None):
    """Retourne (train, validation, test). Avec un supplément de données récentes
    (réentraînement après dérive), celui-ci est découpé chronologiquement 60/20/20 :
    la validation et le test portent alors sur la période récente uniquement."""
    df = pd.read_parquet(dossier(cfg, "features") / "features.parquet")
    train, val, test = (df[df["jeu"] == j] for j in ("train", "validation", "test"))
    if supplement is not None:
        n = len(supplement)
        a, b = int(n * 0.6), int(n * 0.8)
        train = pd.concat([train, val, test, supplement.iloc[:a]], ignore_index=True)
        val, test = supplement.iloc[a:b], supplement.iloc[b:]
    return train, val, test


def entrainer(cfg: dict, supplement: pd.DataFrame | None = None) -> dict:
    journal.info("entrainement_demarre", reentrainement=supplement is not None)
    mlflow.set_tracking_uri(uri_mlflow(cfg))
    mlflow.set_experiment(cfg["mlflow"]["experience"])
    train, val, test = _jeux(cfg, supplement)
    rapports = dossier(cfg, "rapports")

    # 1. Comparaison sur la validation : un run MLflow par candidat.
    essais = []
    for nom, modele, params in candidats(cfg):
        with mlflow.start_run(run_name=nom):
            modele.fit(train[FEATURES], train[CIBLE])
            m_train = metriques(train[CIBLE], predire(modele, train))
            m_val = metriques(val[CIBLE], predire(modele, val))
            mlflow.set_tag("type_modele", nom)
            mlflow.log_params(params)
            mlflow.log_metrics({f"train_{k}": v for k, v in m_train.items()})
            mlflow.log_metrics({f"val_{k}": v for k, v in m_val.items()})
            # Écart train/validation : indicateur de surapprentissage.
            mlflow.log_metric("ecart_surapprentissage", m_val["mae"] - m_train["mae"])
        essais.append(
            {
                "type": nom,
                "params": json.dumps(params),
                "modele": modele,
                "train_mae": m_train["mae"],
                "val_mae": m_val["mae"],
                "val_rmse": m_val["rmse"],
                "val_wape": m_val["wape"],
            }
        )
        journal.info("candidat_evalue", type=nom, params=params, val_mae=round(m_val["mae"], 4))

    comparaison = pd.DataFrame(essais).drop(columns="modele").sort_values("val_mae")
    comparaison.to_csv(rapports / "comparaison_modeles.csv", index=False)
    meilleur = min(essais, key=lambda e: e["val_mae"])

    # 2. Réentraînement du meilleur sur train + validation, évaluation finale sur le test.
    train_val = pd.concat([train, val], ignore_index=True)
    final = clone(meilleur["modele"]).fit(train_val[FEATURES], train_val[CIBLE])
    pred_test = predire(final, test)
    m_test = metriques(test[CIBLE], pred_test)
    m_base = metriques(test[CIBLE], predire(BaseSaisonniere(), test))
    erreurs = analyse_erreurs(test, pred_test)
    for nom, table in erreurs.items():
        table.to_csv(rapports / f"erreurs_{nom}.csv", index=False)

    # 3. Barrière de qualité : battre la base naïve, et ne pas faire pire que le champion.
    gain = 1 - m_test["mae"] / m_base["mae"] if m_base["mae"] > 0 else 0.0
    mae_champion = None
    try:
        champion, _ = registre.charger_champion(cfg)
        mae_champion = metriques(test[CIBLE], predire(champion, test))["mae"]
    except Exception:
        journal.info("aucun_champion_a_comparer")
    promu = gain >= cfg["modeles"]["gain_minimal_vs_base"] and (mae_champion is None or m_test["mae"] <= mae_champion)

    # 4. Run final : modèle, métriques de test, artefacts, enregistrement au registre.
    nom_modele = cfg["mlflow"]["nom_modele"]
    with mlflow.start_run(run_name=f"final_{meilleur['type']}") as run:
        mlflow.set_tags({"type_modele": meilleur["type"], "etape": "final"})
        mlflow.log_params(json.loads(meilleur["params"]))
        mlflow.log_param("reentrainement", supplement is not None)
        mlflow.log_metrics({f"test_{k}": v for k, v in m_test.items()})
        mlflow.log_metrics({"test_mae_base": m_base["mae"], "gain_vs_base": gain})
        if mae_champion is not None:
            mlflow.log_metric("test_mae_champion_precedent", mae_champion)
        mlflow.log_artifacts(str(rapports), artifact_path="rapports")
        info = mlflow.sklearn.log_model(
            final,
            name="modele",
            signature=infer_signature(test[FEATURES].head(50), pred_test[:50]),
            registered_model_name=nom_modele,
            # Modèles produits par notre propre pipeline : la sérialisation pickle est acceptable.
            serialization_format="cloudpickle",
        )
    version = str(info.registered_model_version)
    client = registre.client_mlflow(cfg)
    for cle, valeur in {
        "type_modele": meilleur["type"],
        "mae_test": m_test["mae"],
        "wape_test": m_test["wape"],
        "promu": promu,
    }.items():
        client.set_model_version_tag(nom_modele, version, cle, str(valeur))

    if promu:
        registre.promouvoir(cfg, version)
        # Jeu de référence pour la détection de dérive : ce que le modèle a vu à l'entraînement.
        reference = train_val[["station", "heure", *FEATURES, CIBLE]]
        taille = min(len(reference), cfg["derive"]["taille_reference"])
        reference.sample(taille, random_state=cfg["projet"]["graine"]).to_parquet(
            dossier(cfg, "features") / "reference.parquet", index=False
        )

    resultat = {
        "version": version,
        "run_id": run.info.run_id,
        "type": meilleur["type"],
        "params": json.loads(meilleur["params"]),
        "metriques_test": m_test,
        "mae_base": m_base["mae"],
        "gain_vs_base": gain,
        "mae_champion_precedent": mae_champion,
        "promu": promu,
        "date": datetime.now(UTC).isoformat(timespec="seconds"),
    }
    (rapports / "dernier_entrainement.json").write_text(json.dumps(resultat, indent=2), encoding="utf-8")
    journal.info(
        "entrainement_termine",
        version=version,
        type=meilleur["type"],
        promu=promu,
        test_mae=round(m_test["mae"], 4),
        gain_vs_base=round(gain, 4),
    )
    return resultat


if __name__ == "__main__":
    print(json.dumps(entrainer(charger_config()), indent=2))
