"""API : prédiction, validation des entrées, cas d'erreur, authentification, métriques."""

import copy
import json
import sys
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from src.serving.api import creer_app


def test_health(client, pipeline):
    r = client.get("/health")
    assert r.status_code == 200
    assert r.json()["statut"] == "ok"
    assert r.json()["version_modele"] is not None


def test_model_expose_version_et_stations(client):
    corps = client.get("/model").json()
    assert corps["version"] and "Vieux-Port" in corps["stations"]


def test_predict_nominal(client, demande_valide):
    r = client.post("/predict", json=demande_valide)
    assert r.status_code == 200
    corps = r.json()
    assert corps["prediction"] >= 0
    assert corps["version_modele"] == client.get("/health").json()["version_modele"]
    assert "X-Request-ID" in r.headers


def test_predict_reagit_a_la_pluie(client, demande_valide):
    sec = client.post("/predict", json=demande_valide).json()["prediction"]
    pluie = client.post("/predict", json={**demande_valide, "precipitation": 8.0}).json()["prediction"]
    assert pluie < sec


@pytest.mark.parametrize(
    "modification",
    [
        {"temperature": 999},  # hors bornes
        {"precipitation": -1},  # négatif
        {"demande_h_1": "beaucoup"},  # mauvais type
        {"horodatage": "pas une date"},
        {"station": ""},
    ],
)
def test_predict_entree_invalide(client, demande_valide, modification):
    assert client.post("/predict", json={**demande_valide, **modification}).status_code == 422


def test_predict_champ_manquant(client, demande_valide):
    del demande_valide["vent"]
    assert client.post("/predict", json=demande_valide).status_code == 422


def test_predict_station_inconnue(client, demande_valide):
    r = client.post("/predict", json={**demande_valide, "station": "Station Fantôme"})
    assert r.status_code == 422
    assert "inconnue" in r.json()["detail"]


def test_feedback_met_a_jour_la_performance(client, demande_valide):
    p = client.post("/predict", json=demande_valide).json()
    r = client.post("/feedback", json={"id_prediction": p["id_prediction"], "valeur_reelle": p["prediction"] + 2})
    assert r.status_code == 200
    assert r.json()["erreur_absolue"] == pytest.approx(2.0, abs=0.01)
    ligne = next(x for x in client.get("/metrics").text.splitlines() if x.startswith("veloprevu_mae_glissante "))
    assert float(ligne.split()[1]) == pytest.approx(2.0, abs=0.01)
    # un même identifiant ne peut pas être évalué deux fois
    r2 = client.post("/feedback", json={"id_prediction": p["id_prediction"], "valeur_reelle": 1})
    assert r2.status_code == 404


def test_metrics_expose_les_indicateurs(client, demande_valide):
    client.post("/predict", json=demande_valide)
    client.post("/predict", json={**demande_valide, "temperature": 999})
    texte = client.get("/metrics").text
    assert 'veloprevu_requetes_total{code="200",route="/predict"} 1.0' in texte
    assert 'veloprevu_requetes_total{code="422",route="/predict"} 1.0' in texte
    for nom in (
        "veloprevu_predictions_total",
        "veloprevu_latence_secondes_bucket",
        "veloprevu_modele_info",
        "veloprevu_entree_temperature_bucket",
    ):
        assert nom in texte
    # Les métriques CPU et mémoire du processus ne sont exposées que sous Linux (donc dans Docker).
    if sys.platform.startswith("linux"):
        assert "process_cpu_seconds_total" in texte


def test_prediction_journalisee(client, demande_valide, cfg):
    p = client.post("/predict", json=demande_valide, headers={"X-Request-ID": "corr-123"}).json()
    lignes = (Path(cfg["chemins"]["journaux"]) / "predictions.jsonl").read_text().splitlines()
    derniere = json.loads(lignes[-1])
    assert derniere["id_prediction"] == p["id_prediction"]
    assert derniere["id_requete"] == "corr-123"
    assert derniere["temperature"] == demande_valide["temperature"]


def test_authentification_par_cle(client, demande_valide, monkeypatch):
    monkeypatch.setenv("VELOPREVU_API_KEY", "cle-de-test")
    assert client.post("/predict", json=demande_valide).status_code == 401
    assert client.post("/predict", json=demande_valide, headers={"X-API-Key": "mauvaise"}).status_code == 401
    assert client.post("/predict", json=demande_valide, headers={"X-API-Key": "cle-de-test"}).status_code == 200
    assert client.post("/admin/reload").status_code == 401
    assert client.get("/health").status_code == 200  # la sonde reste publique


def test_reload(client):
    r = client.post("/admin/reload")
    assert r.status_code == 200
    assert r.json()["version_apres"] == r.json()["version_avant"]


def test_sans_modele_le_service_repond_503(cfg, tmp_path, demande_valide):
    vide = copy.deepcopy(cfg)
    vide["chemins"] = {cle: str(tmp_path / cle) for cle in cfg["chemins"]}
    vide["mlflow"]["uri"] = f"sqlite:///{tmp_path / 'vide.db'}"
    with TestClient(creer_app(vide)) as c:
        assert c.get("/health").status_code == 503
        assert c.post("/predict", json=demande_valide).status_code == 503
