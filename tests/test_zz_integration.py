"""Intégration de bout en bout : dérive -> réentraînement -> rechargement de l'API -> rollback.

Ce fichier s'exécute en dernier (ordre alphabétique) car il modifie le champion.
"""

from src.llm_agents import outils, valider
from src.monitoring.reentrainement import verifier_et_reentrainer
from src.monitoring.simulation import simuler
from src.training import registre


def test_pas_de_reentrainement_sans_degradation(cfg, pipeline):
    avant = registre.version_champion(cfg)
    r = verifier_et_reentrainer(cfg, simuler(cfg, "aucune"), "simulation_aucune")
    assert r["action"] == "aucune" and r["entrainement"] is None
    assert registre.version_champion(cfg) == avant


def test_cycle_complet_derive_reentrainement_rollback(cfg, client, demande_valide):
    v1 = str(registre.version_champion(cfg))
    assert client.get("/health").json()["version_modele"] == v1

    # 1. La dérive « hiver » dégrade la performance : un nouveau modèle est entraîné et promu.
    r = verifier_et_reentrainer(cfg, simuler(cfg, "hiver"), "simulation_hiver")
    assert r["action"] == "modele_remplace"
    v2 = str(registre.version_champion(cfg))
    assert int(v2) > int(v1)
    assert r["entrainement"]["metriques_test"]["mae"] <= r["entrainement"]["mae_champion_precedent"]

    # 2. L'API sert le nouveau modèle après rechargement, sans redémarrage.
    assert client.post("/admin/reload").json() == {"version_avant": v1, "version_apres": v2}
    assert client.post("/predict", json=demande_valide).json()["version_modele"] == v2

    # 3. Rollback : retour à la version précédente, visible dans l'API.
    assert registre.rollback(cfg)["version"] == v1
    assert client.post("/admin/reload").json()["version_apres"] == v1
    assert [v["champion"] for v in registre.lister(cfg)].count(True) == 1


def test_validation_humaine_d_une_proposition(cfg, pipeline):
    from src.monitoring.derive import detecter

    detecter(cfg, simuler(cfg, "concept"), "simulation_concept")
    identifiant = outils.proposer_reentrainement(cfg, "WAPE triplé sans dérive des entrées")["id"]
    assert any(p["id"] == identifiant for p in valider.lister(cfg))
    rejetee = valider.decider(cfg, identifiant, approuver=False)
    assert rejetee["statut"] == "rejetee" and "resultat" not in rejetee

    identifiant = outils.proposer_reentrainement(cfg, "Deuxième demande")["id"]
    approuvee = valider.decider(cfg, identifiant, approuver=True)
    assert approuvee["statut"] == "approuvee"
    assert approuvee["resultat"] in ("modele_remplace", "candidat_rejete")
