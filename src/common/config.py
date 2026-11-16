"""Chargement de la configuration YAML et accès aux répertoires du projet."""

import os
from pathlib import Path

import yaml


def charger_config(chemin: str | Path | None = None) -> dict:
    """Charge configs/config.yaml (ou le fichier désigné par VELOPREVU_CONFIG)."""
    chemin = Path(chemin or os.getenv("VELOPREVU_CONFIG", "configs/config.yaml"))
    with open(chemin, encoding="utf-8") as f:
        return yaml.safe_load(f)


def dossier(cfg: dict, cle: str) -> Path:
    """Retourne le répertoire demandé en le créant au besoin."""
    p = Path(cfg["chemins"][cle])
    p.mkdir(parents=True, exist_ok=True)
    return p


def uri_mlflow(cfg: dict) -> str:
    """La variable d'environnement a priorité sur le fichier de configuration."""
    return os.getenv("MLFLOW_TRACKING_URI", cfg["mlflow"]["uri"])
