"""Réentraînement automatique après détection d'une dérive.

Règle de décision :
- performance dégradée (valeurs réelles disponibles)  -> réentraînement sur les données récentes ;
- dérive des entrées seule                            -> alerte, pas de réentraînement ;
- sinon                                               -> rien.
Le nouveau modèle ne remplace le champion que s'il franchit la barrière de qualité sur
la période récente. En cas de regret : python -m src.training.registre rollback

Usage : python -m src.monitoring.reentrainement --scenario hiver
"""

import json
import os

import httpx
import pandas as pd

from src.common.config import charger_config
from src.common.journal import obtenir_journal
from src.features.construction import CIBLE, FEATURES
from src.monitoring.derive import analyseur_arguments, charger_courant, detecter
from src.training.entrainement import entrainer

journal = obtenir_journal("monitoring.reentrainement")


def notifier_api() -> bool:
    """Demande à l'API de recharger le champion (si VELOPREVU_API_URL est définie)."""
    url = os.getenv("VELOPREVU_API_URL")
    if not url:
        return False
    try:
        entetes = {"X-API-Key": os.getenv("VELOPREVU_API_KEY", "")}
        httpx.post(f"{url}/admin/reload", headers=entetes, timeout=30).raise_for_status()
        return True
    except httpx.HTTPError as e:
        journal.warning("rechargement_api_echec", erreur=str(e))
        return False


def verifier_et_reentrainer(cfg: dict, courant: pd.DataFrame, origine: str) -> dict:
    rapport = detecter(cfg, courant, origine)
    resultat = {"rapport": rapport, "action": "aucune", "entrainement": None}
    if not rapport["alerte_performance"]:
        if rapport["alerte_donnees"]:
            resultat["action"] = "alerte_sans_reentrainement"
        journal.info("reentrainement_non_requis", action=resultat["action"])
        return resultat

    colonnes = ["station", "heure", *FEATURES, CIBLE]
    supplement = courant.dropna(subset=[CIBLE])[colonnes].sort_values("heure").reset_index(drop=True)
    journal.warning("reentrainement_declenche", lignes_recentes=len(supplement), origine=origine)
    entrainement = entrainer(cfg, supplement=supplement)
    resultat["entrainement"] = entrainement
    resultat["action"] = "modele_remplace" if entrainement["promu"] else "candidat_rejete"
    if entrainement["promu"]:
        resultat["api_rechargee"] = notifier_api()
    journal.info("reentrainement_termine", action=resultat["action"], version=entrainement["version"])
    return resultat


if __name__ == "__main__":
    args = analyseur_arguments("Détection de dérive puis réentraînement si nécessaire").parse_args()
    config = charger_config()
    donnees, source = charger_courant(config, args.scenario, args.courant, args.journal)
    r = verifier_et_reentrainer(config, donnees, source)
    print(
        json.dumps(
            {"action": r["action"], "diagnostic": r["rapport"]["diagnostic"], "entrainement": r["entrainement"]},
            indent=2,
            ensure_ascii=False,
        )
    )
