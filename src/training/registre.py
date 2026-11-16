"""Model Registry MLflow : version en production (alias « champion »), promotion, rollback.

Usage :
    python -m src.training.registre lister
    python -m src.training.registre promouvoir --version 3
    python -m src.training.registre rollback            # revient à la version précédente
    python -m src.training.registre rollback --version 1
"""

import argparse
import json
from datetime import UTC, datetime

import joblib
import mlflow
from mlflow import MlflowClient
from mlflow.exceptions import MlflowException

from src.common.config import charger_config, dossier, uri_mlflow
from src.common.journal import obtenir_journal

journal = obtenir_journal("training.registre")


def client_mlflow(cfg: dict) -> MlflowClient:
    mlflow.set_tracking_uri(uri_mlflow(cfg))
    return MlflowClient()


def version_champion(cfg: dict) -> str | None:
    """Numéro de la version qui porte l'alias de production, ou None s'il n'y en a pas."""
    m = cfg["mlflow"]
    try:
        return client_mlflow(cfg).get_model_version_by_alias(m["nom_modele"], m["alias"]).version
    except MlflowException:
        return None


def charger_champion(cfg: dict):
    """Retourne (modèle, version) depuis le registre ; lève une exception s'il est absent."""
    m = cfg["mlflow"]
    version = version_champion(cfg)
    if version is None:
        raise LookupError("Aucune version ne porte l'alias " + m["alias"])
    return mlflow.sklearn.load_model(f"models:/{m['nom_modele']}@{m['alias']}"), str(version)


def lister(cfg: dict) -> list[dict]:
    m = cfg["mlflow"]
    champion = version_champion(cfg)
    versions = client_mlflow(cfg).search_model_versions(f"name='{m['nom_modele']}'")
    return [
        {
            "version": v.version,
            "run_id": v.run_id,
            "champion": str(v.version) == str(champion),
            "type": v.tags.get("type_modele"),
            "mae_test": v.tags.get("mae_test"),
        }
        for v in sorted(versions, key=lambda v: int(v.version))
    ]


def synchroniser_local(cfg: dict) -> dict:
    """Copie le champion dans models/ : l'API s'en sert si MLflow est injoignable."""
    modele, version = charger_champion(cfg)
    m = cfg["mlflow"]
    v = client_mlflow(cfg).get_model_version(m["nom_modele"], version)
    meta = {
        "nom_modele": m["nom_modele"],
        "version": version,
        "run_id": v.run_id,
        "type": v.tags.get("type_modele"),
        "mae_reference": float(v.tags.get("mae_test", "nan")),
        "wape_reference": float(v.tags.get("wape_test", "nan")),
        "date_synchronisation": datetime.now(UTC).isoformat(timespec="seconds"),
    }
    rep = dossier(cfg, "modeles")
    joblib.dump(modele, rep / "modele.joblib")
    (rep / "metadata.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
    return meta


def promouvoir(cfg: dict, version: str | int) -> dict:
    m = cfg["mlflow"]
    ancienne = version_champion(cfg)
    client_mlflow(cfg).set_registered_model_alias(m["nom_modele"], m["alias"], str(version))
    journal.info("modele_promu", version=str(version), version_precedente=ancienne)
    return synchroniser_local(cfg)


def rollback(cfg: dict, version: str | int | None = None) -> dict:
    """Replace l'alias de production sur une version antérieure (par défaut : la précédente)."""
    actuelle = version_champion(cfg)
    if actuelle is None:
        raise LookupError("Aucun champion : rien à annuler")
    if version is None:
        anterieures = [int(v["version"]) for v in lister(cfg) if int(v["version"]) < int(actuelle)]
        if not anterieures:
            raise LookupError("Aucune version antérieure à la version " + str(actuelle))
        version = max(anterieures)
    journal.warning("rollback", de=str(actuelle), vers=str(version))
    return promouvoir(cfg, version)


if __name__ == "__main__":
    analyseur = argparse.ArgumentParser(description="Gestion du Model Registry")
    analyseur.add_argument("action", choices=["lister", "promouvoir", "rollback"])
    analyseur.add_argument("--version")
    args = analyseur.parse_args()
    config = charger_config()
    if args.action == "lister":
        resultat = lister(config)
    elif args.action == "promouvoir":
        resultat = promouvoir(config, args.version)
    else:
        resultat = rollback(config, args.version)
    print(json.dumps(resultat, indent=2, ensure_ascii=False))
