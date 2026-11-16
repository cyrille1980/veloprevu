"""Agent : boucle d'appels d'outils, garde-fous et échecs maîtrisés (LLM simulé)."""

import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from src.llm_agents import outils
from src.llm_agents.agent import executer_outil, repondre
from src.llm_agents.evaluation import evaluer
from src.monitoring.derive import detecter
from src.monitoring.simulation import simuler


class FauxLLM:
    """Rejoue une suite de réponses : listes d'appels d'outils, puis un texte final."""

    def __init__(self, reponses):
        self._reponses = iter(reponses)
        self.messages_recus = []
        self.chat = SimpleNamespace(completions=SimpleNamespace(create=self._creer))

    def _creer(self, model, messages, tools, temperature):
        self.messages_recus = messages
        suivante = next(self._reponses)
        if isinstance(suivante, Exception):
            raise suivante
        if isinstance(suivante, str):
            message = SimpleNamespace(content=suivante, tool_calls=None)
        else:
            appels = [
                SimpleNamespace(id=f"appel_{i}", function=SimpleNamespace(name=nom, arguments=json.dumps(args)))
                for i, (nom, args) in enumerate(suivante)
            ]
            message = SimpleNamespace(content=None, tool_calls=appels)
        return SimpleNamespace(choices=[SimpleNamespace(message=message)])


@pytest.fixture()
def rapport(cfg, pipeline):
    return detecter(cfg, simuler(cfg, "hiver"), "simulation_hiver")


def test_parcours_en_plusieurs_etapes(cfg, rapport):
    llm = FauxLLM(
        [
            [("lire_rapport_derive", {})],
            [("lire_etat_modele", {})],
            "Constat : dérive des données [lire_rapport_derive].",
        ]
    )
    r = repondre(cfg, "Fais le point", client=llm)
    assert r["statut"] == "ok" and r["etapes"] == 3
    assert [a["outil"] for a in r["outils"]] == ["lire_rapport_derive", "lire_etat_modele"]
    # le résultat de l'outil a bien été transmis au modèle
    resultats = [m for m in llm.messages_recus if m["role"] == "tool"]
    assert "simulation_hiver" in resultats[0]["content"]


def test_outil_inconnu_refuse(cfg):
    assert "non autorisé" in executer_outil(cfg, "supprimer_modele", "{}")["erreur"]


def test_arguments_inattendus_refuses(cfg, rapport):
    assert "invalides" in executer_outil(cfg, "lire_rapport_derive", '{"chemin": "/etc/passwd"}')["erreur"]


def test_metrique_hors_liste_blanche(cfg):
    assert "non autorisée" in outils.interroger_metrique(cfg, "up{job='x'}")["erreur"]


def test_prometheus_arrete_echec_maitrise(cfg):
    config = {**cfg, "agent": {**cfg["agent"], "url_prometheus": "http://127.0.0.1:9"}}
    assert "indisponible" in outils.interroger_metrique(config, "mae_glissante")["erreur"]


def test_proposition_sans_execution(cfg, rapport):
    from src.training import registre

    version_avant = registre.version_champion(cfg)
    llm = FauxLLM(
        [
            [("proposer_reentrainement", {"justification": "WAPE doublé"})],
            "Proposition déposée ; validation humaine requise.",
        ]
    )
    repondre(cfg, "Faut-il réentraîner ?", client=llm)
    propositions = list((Path(cfg["chemins"]["rapports"]) / "propositions").glob("*.json"))
    contenu = json.loads(propositions[-1].read_text())
    assert contenu["statut"] == "en_attente_validation_humaine"
    assert contenu["origine_donnees"] == "simulation_hiver"
    assert registre.version_champion(cfg) == version_avant  # rien n'a été entraîné ni promu


def test_nombre_d_etapes_borne(cfg, rapport):
    boucle = [[("lire_rapport_derive", {})]] * 20
    r = repondre(cfg, "Boucle", client=FauxLLM(boucle))
    assert r["statut"] == "etapes_max" and r["etapes"] == cfg["agent"]["etapes_max"]


def test_llm_indisponible(cfg):
    r = repondre(cfg, "Question", client=FauxLLM([ConnectionError("refusé")]))
    assert r["statut"] == "echec_llm" and r["outils"] == []


def test_evaluation_des_scenarios(cfg, rapport, tmp_path):
    scenarios = tmp_path / "scenarios.yaml"
    scenarios.write_text(
        "- nom: ok\n  question: q\n  outils_attendus: [lire_rapport_derive]\n  mots_attendus: [dérive]\n"
        "- nom: interdit\n  question: q\n  outils_interdits: [proposer_reentrainement]\n",
        encoding="utf-8",
    )
    llm = FauxLLM(
        [
            [("lire_rapport_derive", {})],
            "Il y a une dérive.",
            [("proposer_reentrainement", {"justification": "x"})],
            "Fait.",
        ]
    )
    bilan = evaluer(cfg, client=llm, fichier=str(scenarios))
    assert bilan["reussis"] == 1 and bilan["total"] == 2
    assert bilan["scenarios"][1]["outils_interdits"] is False
