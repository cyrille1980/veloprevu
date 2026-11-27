"""Préparation : nettoyage, agrégation horaire, lecture des deux schémas BIXI, validation."""

import io

import pandas as pd
import pandera.errors
import pytest

from src.preprocessing import preparation as prep


def _deplacements(lignes):
    return pd.DataFrame(lignes, columns=["debut", "station"]).assign(debut=lambda d: pd.to_datetime(d["debut"]))


def test_nettoyer_retire_nuls_doublons_et_hors_periode(cfg):
    cfg_test = {**cfg, "periode": {"debut": "2025-06-01", "fin": "2025-06-30"}}
    df = _deplacements(
        [
            ("2025-06-01 08:10", "A"),
            ("2025-06-01 08:10", "A"),  # doublon exact
            ("2025-06-01 09:00", None),
            (None, "A"),  # valeurs manquantes
            ("2019-01-01 09:00", "A"),  # hors période
            ("2025-06-01 10:30", "B"),
        ]
    )
    propre = prep.nettoyer(df, cfg_test)
    assert len(propre) == 2
    assert set(propre["station"]) == {"A", "B"}


def test_agreger_reconstruit_la_grille_et_met_zero(cfg):
    df = _deplacements(
        [("2025-06-01 08:10", "A"), ("2025-06-01 08:50", "A"), ("2025-06-01 11:05", "A"), ("2025-06-01 08:20", "B")]
    )
    demande = prep.agreger_horaire(df, {**cfg, "preparation": {**cfg["preparation"], "plafond_quantile": 1.0}})
    assert len(demande) == 2 * 4  # 2 stations x 4 heures (8 h à 11 h)
    a = demande[demande["station"] == "A"].set_index("heure")["demande"]
    assert a[pd.Timestamp("2025-06-01 08:00")] == 2
    assert a[pd.Timestamp("2025-06-01 09:00")] == 0
    assert not demande.duplicated(["station", "heure"]).any()


def test_lecture_nouveau_schema_millisecondes(cfg):
    # 2025-07-01 12:00 UTC = 08:00 à Montréal (heure d'été)
    ms = int(pd.Timestamp("2025-07-01 12:00", tz="UTC").timestamp() * 1000)
    csv = io.StringIO(f"STARTSTATIONNAME,STARTTIMEMS,ENDSTATIONNAME\n Laurier ,{ms},X\n")
    df = prep._lire_csv(csv, cfg)
    assert df.loc[0, "debut"] == pd.Timestamp("2025-07-01 08:00")
    assert df.loc[0, "station"] == "Laurier"


def test_lecture_ancien_schema_dates_texte(cfg):
    csv = io.StringIO("start_date,emplacement_pk_start,duration_sec\n2021-07-01 08:15,6001,300\n")
    df = prep._lire_csv(csv, cfg)
    assert df.loc[0, "debut"] == pd.Timestamp("2021-07-01 08:15")
    assert df.loc[0, "station"] == "6001"


def test_colonnes_inconnues_erreur_explicite(cfg):
    with pytest.raises(ValueError, match="Aucune des colonnes"):
        prep._lire_csv(io.StringIO("a,b\n1,2\n"), cfg)


def test_jointure_meteo_interpole_les_trous():
    heures = pd.date_range("2025-06-01", periods=4, freq="h")
    demande = pd.DataFrame({"station": "A", "heure": heures, "demande": 1.0})
    meteo = pd.DataFrame(
        {"heure": heures[[0, 3]], "temperature": [10.0, 16.0], "precipitation": [0.0, 0.0], "vent": [5.0, 5.0]}
    )
    resultat = prep.joindre_meteo(demande, meteo)
    assert resultat["temperature"].tolist() == [10.0, 12.0, 14.0, 16.0]


def test_schema_rejette_une_demande_negative():
    df = pd.DataFrame(
        {
            "station": ["A"],
            "heure": [pd.Timestamp("2025-06-01")],
            "demande": [-1.0],
            "temperature": [10.0],
            "precipitation": [0.0],
            "vent": [1.0],
        }
    )
    with pytest.raises(pandera.errors.SchemaError):
        prep.SCHEMA.validate(df)


def test_sortie_du_pipeline(cfg, pipeline):
    from pathlib import Path

    df = pd.read_parquet(Path(cfg["chemins"]["traite"]) / "demande_horaire.parquet")
    assert df["station"].nunique() == cfg["preparation"]["nb_stations"]
    assert not df.isna().any().any()
    assert (df["demande"] >= 0).all()
