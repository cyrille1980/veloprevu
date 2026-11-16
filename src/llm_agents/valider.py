"""Validation humaine des propositions de l'agent.

Usage :
    python -m src.llm_agents.valider lister
    python -m src.llm_agents.valider approuver <id>     # lance réellement le réentraînement
    python -m src.llm_agents.valider rejeter <id>
"""

import argparse
import getpass
import json
from datetime import UTC, datetime

import pandas as pd

from src.common.config import charger_config, dossier
from src.common.journal import obtenir_journal
from src.monitoring.derive import lire_journal_api
from src.monitoring.reentrainement import verifier_et_reentrainer

journal = obtenir_journal("llm_agents.valider")


def _repertoire(cfg: dict):
    rep = dossier(cfg, "rapports") / "propositions"
    rep.mkdir(parents=True, exist_ok=True)
    return rep


def lister(cfg: dict) -> list[dict]:
    return [json.loads(f.read_text(encoding="utf-8")) for f in sorted(_repertoire(cfg).glob("*.json"))]


def _donnees(cfg: dict, origine: str) -> pd.DataFrame:
    if origine == "journal_api":
        return lire_journal_api(cfg)
    if origine.startswith("simulation_"):
        scenario = origine.removeprefix("simulation_")
        return pd.read_parquet(dossier(cfg, "features") / f"courant_{scenario}.parquet")
    return pd.read_parquet(origine)


def decider(cfg: dict, identifiant: str, approuver: bool) -> dict:
    fichier = _repertoire(cfg) / f"{identifiant}.json"
    proposition = json.loads(fichier.read_text(encoding="utf-8"))
    if proposition["statut"] != "en_attente_validation_humaine":
        raise ValueError(f"Proposition déjà traitée : {proposition['statut']}")
    proposition.update(
        statut="approuvee" if approuver else "rejetee",
        decide_par=getpass.getuser(),
        date_decision=datetime.now(UTC).isoformat(timespec="seconds"),
    )
    journal.warning(
        "proposition_decidee", id=identifiant, statut=proposition["statut"], decide_par=proposition["decide_par"]
    )
    if approuver:
        origine = proposition["origine_donnees"]
        resultat = verifier_et_reentrainer(cfg, _donnees(cfg, origine), origine)
        proposition["resultat"] = resultat["action"]
    fichier.write_text(json.dumps(proposition, indent=2, ensure_ascii=False), encoding="utf-8")
    return proposition


if __name__ == "__main__":
    a = argparse.ArgumentParser(description="Validation humaine des propositions de l'agent")
    a.add_argument("action", choices=["lister", "approuver", "rejeter"])
    a.add_argument("id", nargs="?")
    args = a.parse_args()
    config = charger_config()
    sortie = lister(config) if args.action == "lister" else decider(config, args.id, args.action == "approuver")
    print(json.dumps(sortie, indent=2, ensure_ascii=False))
