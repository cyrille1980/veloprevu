"""Extraction : nouvelles tentatives, échec maîtrisé, traçabilité, échantillon."""

import hashlib
import json

import httpx
import pytest

from src.common.journal import FormatJson, masquer
from src.ingestion import extraction
from src.ingestion.extraction import ErreurExtraction, telecharger, tracer


def _client(reponses: list[int]) -> httpx.Client:
    """Client factice : renvoie successivement les codes HTTP donnés."""
    file = iter(reponses)

    def repondre(requete: httpx.Request) -> httpx.Response:
        return httpx.Response(next(file), content=b"contenu")

    return httpx.Client(transport=httpx.MockTransport(repondre))


def test_telecharger_reussit_apres_deux_echecs(tmp_path):
    fichier = telecharger(
        "https://exemple.test/f", tmp_path / "f.bin", tentatives=3, delai_initial_s=0, client=_client([500, 503, 200])
    )
    assert fichier.read_bytes() == b"contenu"


def test_telecharger_abandonne_apres_les_tentatives(tmp_path):
    with pytest.raises(ErreurExtraction):
        telecharger(
            "https://exemple.test/f", tmp_path / "f.bin", tentatives=2, delai_initial_s=0, client=_client([500, 500])
        )
    assert not (tmp_path / "f.bin").exists()


def test_tracer_ecrit_le_manifeste(tmp_path):
    cfg = {"chemins": {"brut": str(tmp_path)}}
    fichier = tmp_path / "a.csv"
    fichier.write_text("x")
    tracer(cfg, "source_test", "https://exemple.test", fichier)
    tracer(cfg, "source_test", "https://exemple.test", fichier)
    manifeste = json.loads((tmp_path / "manifeste.json").read_text())
    assert len(manifeste) == 2
    assert manifeste[0]["sha256"] == hashlib.sha256(b"x").hexdigest()


def test_extraire_bixi_sans_url_explique_quoi_faire(cfg):
    with pytest.raises(ErreurExtraction, match="url_historique"):
        extraction.extraire_bixi({**cfg, "sources": {"bixi": {"url_historique": ""}}})


def test_extraire_meteo_passe_la_periode(cfg, tmp_path):
    vues = {}

    def repondre(requete: httpx.Request) -> httpx.Response:
        vues.update(requete.url.params)
        return httpx.Response(200, json={"hourly": {"time": []}})

    config = {**cfg, "chemins": {**cfg["chemins"], "brut": str(tmp_path)}}
    extraction.extraire_meteo(config, client=httpx.Client(transport=httpx.MockTransport(repondre)))
    assert vues["start_date"] == cfg["periode"]["debut"]
    assert (tmp_path / "meteo" / "meteo.json").exists()


def test_echantillon_au_format_des_sources(cfg, pipeline):
    from pathlib import Path

    brut = Path(cfg["chemins"]["brut"])
    entete = (brut / "bixi" / "echantillon.csv").read_text().splitlines()[0]
    assert "STARTSTATIONNAME" in entete and "STARTTIMEMS" in entete
    meteo = json.loads((brut / "meteo" / "meteo.json").read_text())
    assert {"time", "temperature_2m", "precipitation", "wind_speed_10m"} <= set(meteo["hourly"])


def test_journal_masque_les_secrets():
    assert masquer({"api_key": "abc", "station": "X"}) == {"api_key": "***", "station": "X"}


def test_journal_structure_json():
    import logging

    enregistrement = logging.LogRecord("composant.test", logging.INFO, "", 0, "evenement_test", None, None)
    enregistrement.contexte = {"n": 1, "token": "secret"}
    ligne = json.loads(FormatJson().format(enregistrement))
    assert {"date", "niveau", "composant", "evenement", "contexte"} <= set(ligne)
    assert ligne["contexte"] == {"n": 1, "token": "***"}
