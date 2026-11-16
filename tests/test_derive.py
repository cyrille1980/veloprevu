"""Dérive : PSI, scénarios simulés, distinction données / performance, rapports."""

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from src.monitoring.derive import analyser, detecter, lire_journal_api, psi
from src.monitoring.simulation import simuler
from src.serving.chargement import charger_modele


def test_psi_proche_de_zero_si_meme_distribution():
    rng = np.random.default_rng(0)
    assert psi(rng.normal(0, 1, 5000), rng.normal(0, 1, 5000)) < 0.05


def test_psi_eleve_si_distribution_decalee():
    rng = np.random.default_rng(0)
    assert psi(rng.normal(0, 1, 5000), rng.normal(2, 1, 5000)) > 0.2


def test_psi_variable_presque_constante():
    reference = np.zeros(1000)
    reference[:20] = 5.0
    assert psi(reference, reference.copy()) == pytest.approx(0, abs=1e-6)
    assert psi(reference, np.full(1000, 5.0)) > 0.2


@pytest.fixture(scope="module")
def contexte(cfg, pipeline):
    reference = pd.read_parquet(Path(cfg["chemins"]["features"]) / "reference.parquet")
    infos = charger_modele(cfg)
    return reference, infos["modele"], infos["wape_reference"]


def test_scenario_aucune_pas_d_alerte_performance(cfg, contexte):
    reference, modele, wape = contexte
    r = analyser(reference, simuler(cfg, "aucune"), cfg, modele, wape)
    assert r["alerte_performance"] is False
    assert r["performance"]["degradation"] == pytest.approx(0, abs=1e-6)


def test_scenario_hiver_derive_des_donnees_et_de_la_performance(cfg, contexte):
    reference, modele, wape = contexte
    r = analyser(reference, simuler(cfg, "hiver"), cfg, modele, wape)
    assert r["alerte_donnees"] is True
    assert {"temperature", "demande_h_1"} <= set(r["variables_derivees"])
    assert r["alerte_performance"] is True
    assert "réentraînement" in r["diagnostic"]


def test_scenario_concept_invisible_dans_les_entrees(cfg, contexte):
    reference, modele, wape = contexte
    sans_derive = analyser(reference, simuler(cfg, "aucune"), cfg, modele, wape)
    r = analyser(reference, simuler(cfg, "concept"), cfg, modele, wape)
    assert r["variables_derivees"] == sans_derive["variables_derivees"]  # mêmes entrées
    assert r["alerte_performance"] is True
    assert "concept" in r["diagnostic"]


def test_sans_valeurs_reelles_pas_de_verdict_de_performance(cfg, contexte):
    reference, modele, wape = contexte
    courant = simuler(cfg, "hiver").drop(columns="demande")
    r = analyser(reference, courant, cfg, modele, wape)
    assert r["alerte_donnees"] is True
    assert r["performance"] is None and r["alerte_performance"] is False
    assert "impact inconnu" in r["diagnostic"]


def test_scenario_inconnu(cfg, pipeline):
    with pytest.raises(ValueError):
        simuler(cfg, "canicule")


def test_detecter_ecrit_les_rapports(cfg, pipeline):
    r = detecter(cfg, simuler(cfg, "hiver"), "simulation_hiver")
    rep = Path(cfg["chemins"]["rapports"]) / "derive"
    assert json.loads((rep / "dernier.json").read_text())["alerte_donnees"] is True
    assert (rep / "historique.jsonl").exists()
    assert Path(r["rapport_html"]).stat().st_size > 10_000


def test_derive_exposee_dans_les_metriques(cfg, client):
    detecter(cfg, simuler(cfg, "hiver"), "simulation_hiver")
    assert "veloprevu_derive_alerte 1.0" in client.get("/metrics").text


def test_journal_api_joint_predictions_et_retours(client, demande_valide, cfg):
    p = client.post("/predict", json=demande_valide).json()
    client.post("/feedback", json={"id_prediction": p["id_prediction"], "valeur_reelle": 6})
    journal = lire_journal_api(cfg)
    ligne = journal[journal["id_prediction"] == p["id_prediction"]].iloc[0]
    assert ligne["demande"] == 6 and ligne["prediction"] == pytest.approx(p["prediction"], abs=0.01)
