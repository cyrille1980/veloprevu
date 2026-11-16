"""Pipeline complet en une seule commande : de la donnée brute au modèle enregistré.

Usage :
    python -m src.pipeline --source reel
    python -m src.pipeline --source echantillon     # sans réseau (tests, CI, première prise en main)
"""

import argparse
import json

from src.common.config import charger_config
from src.common.journal import obtenir_journal
from src.features.construction import construire
from src.ingestion.extraction import extraire
from src.preprocessing.preparation import preparer
from src.training.entrainement import entrainer

journal = obtenir_journal("pipeline")


def executer(cfg: dict, source: str = "reel", sans_extraction: bool = False) -> dict:
    journal.info("pipeline_demarre", source=source)
    if not sans_extraction:
        extraire(cfg, source)
    preparer(cfg)
    construire(cfg)
    resultat = entrainer(cfg)
    journal.info("pipeline_termine", version=resultat["version"], promu=resultat["promu"])
    return resultat


if __name__ == "__main__":
    analyseur = argparse.ArgumentParser(description="Pipeline VéloPrévu")
    analyseur.add_argument("--source", choices=["reel", "echantillon"], default="reel")
    analyseur.add_argument("--sans-extraction", action="store_true", help="réutilise les données brutes déjà présentes")
    args = analyseur.parse_args()
    print(json.dumps(executer(charger_config(), args.source, args.sans_extraction), indent=2))
