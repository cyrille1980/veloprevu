"""Variables, découpage temporel, métriques, entraînement et registre."""

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from src.evaluation.evaluation import analyse_erreurs, metriques
from src.features.construction import CIBLE, FEATURES, ajouter_variables, decouper, variables_calendrier
from src.training import registre
from src.training.entrainement import BaseSaisonniere


@pytest.fixture()
def serie():
    """Deux stations, 10 jours ; la demande de A vaut le rang de l'heure (0, 1, 2, ...)."""
    heures = pd.date_range("2025-06-02", periods=240, freq="h")
    blocs = [
        pd.DataFrame(
            {
                "station": s,
                "heure": heures,
                "demande": np.arange(240.0) * k,
                "temperature": 20.0,
                "precipitation": 0.0,
                "vent": 5.0,
            }
        )
        for s, k in (("A", 1), ("B", 10))
    ]
    return pd.concat(blocs, ignore_index=True)


def test_retards_ne_regardent_que_le_passe(serie):
    df = ajouter_variables(serie, {"A": 0, "B": 1})
    a = df[df["station"] == "A"]
    assert (a["demande_h_1"] == a[CIBLE] - 1).all()
    assert (a["demande_h_24"] == a[CIBLE] - 24).all()
    assert (a["demande_h_168"] == a[CIBLE] - 168).all()
    # moyenne des 24 heures précédentes de la suite 0,1,2,... = valeur - 12,5
    assert np.allclose(a["moyenne_24h"], a[CIBLE] - 12.5)


def test_retards_ne_melangent_pas_les_stations(serie):
    df = ajouter_variables(serie, {"A": 0, "B": 1})
    b = df[df["station"] == "B"]
    assert (b["demande_h_1"] == b[CIBLE] - 10).all()


def test_calendrier_identique_entrainement_et_service(serie):
    df = ajouter_variables(serie, {"A": 0, "B": 1})
    ligne = df.iloc[5]
    attendu = variables_calendrier(ligne["heure"].to_pydatetime())
    for cle, valeur in attendu.items():
        assert ligne[cle] == pytest.approx(valeur)


def test_decoupage_chronologique(cfg, pipeline):
    df = pd.read_parquet(Path(cfg["chemins"]["features"]) / "features.parquet")
    bornes = df.groupby("jeu")["heure"].agg(["min", "max"])
    assert bornes.loc["train", "max"] < bornes.loc["validation", "min"]
    assert bornes.loc["validation", "max"] < bornes.loc["test", "min"]
    assert not df[FEATURES].isna().any().any()


def test_decouper_respecte_les_durees(serie, cfg):
    config = {"decoupage": {"validation_jours": 2, "test_jours": 1}}
    df = decouper(serie.copy(), config)
    assert df[df["jeu"] == "test"]["heure"].nunique() == 24
    assert df[df["jeu"] == "validation"]["heure"].nunique() == 48


def test_metriques_valeurs_connues():
    m = metriques([0, 10, 20], [0, 8, 24])
    assert m["mae"] == pytest.approx(2.0)
    assert m["wape"] == pytest.approx(6 / 30)
    assert m["rmse"] == pytest.approx(np.sqrt(20 / 3))


def test_wape_indefini_si_demande_nulle():
    assert np.isnan(metriques([0, 0], [1, 1])["wape"])


def test_analyse_erreurs_par_heure_et_station():
    df = pd.DataFrame({"station": ["A", "A", "B"], "heure_jour": [8, 9, 8], "demande": [10.0, 10.0, 4.0]})
    tables = analyse_erreurs(df, [12, 10, 1])
    par_station = tables["par_station"].set_index("station")
    assert par_station.loc["A", "mae"] == pytest.approx(1.0)
    assert par_station.loc["B", "biais"] == pytest.approx(-3.0)


def test_base_saisonniere():
    X = pd.DataFrame({"demande_h_168": [3.0, 7.0]})
    assert BaseSaisonniere().fit(X).predict(X).tolist() == [3.0, 7.0]


def test_pipeline_enregistre_et_promeut(cfg, pipeline):
    assert pipeline["promu"] is True
    assert pipeline["gain_vs_base"] >= cfg["modeles"]["gain_minimal_vs_base"]
    assert pipeline["type"] != "base_saisonniere"
    assert registre.version_champion(cfg) is not None
    modeles = Path(cfg["chemins"]["modeles"])
    assert (modeles / "modele.joblib").exists()
    meta = json.loads((modeles / "metadata.json").read_text())
    assert meta["version"] == str(registre.version_champion(cfg))
    assert (Path(cfg["chemins"]["features"]) / "reference.parquet").exists()
    assert (Path(cfg["chemins"]["rapports"]) / "comparaison_modeles.csv").exists()


def test_champion_charge_et_predit(cfg, pipeline):
    modele, version = registre.charger_champion(cfg)
    df = pd.read_parquet(Path(cfg["chemins"]["features"]) / "features.parquet").head(20)
    predictions = modele.predict(df[FEATURES])
    assert len(predictions) == 20 and np.isfinite(predictions).all()
