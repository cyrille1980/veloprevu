"""Chargement du modèle de production : registre MLflow d'abord, copie locale en secours."""

import json

import joblib

from src.common.config import dossier
from src.common.journal import obtenir_journal
from src.training import registre

journal = obtenir_journal("serving.chargement")


def charger_modele(cfg: dict) -> dict:
    """Retourne {"modele", "version", "source", "wape_reference", "mae_reference"}.

    Si MLflow est injoignable (panne, CI sans serveur), la copie locale models/modele.joblib,
    écrite à chaque promotion, prend le relais : l'API reste disponible.
    """
    try:
        modele, version = registre.charger_champion(cfg)
        meta = registre.client_mlflow(cfg).get_model_version(cfg["mlflow"]["nom_modele"], version)
        infos = {
            "modele": modele,
            "version": version,
            "source": "mlflow",
            "mae_reference": float(meta.tags.get("mae_test", "nan")),
            "wape_reference": float(meta.tags.get("wape_test", "nan")),
        }
    except Exception as e:
        journal.warning("registre_indisponible", erreur=str(e)[:200])
        rep = dossier(cfg, "modeles")
        meta = json.loads((rep / "metadata.json").read_text(encoding="utf-8"))
        infos = {
            "modele": joblib.load(rep / "modele.joblib"),
            "version": str(meta["version"]),
            "source": "local",
            "mae_reference": meta.get("mae_reference", float("nan")),
            "wape_reference": meta.get("wape_reference", float("nan")),
        }
    journal.info("modele_charge", version=infos["version"], source=infos["source"])
    return infos
