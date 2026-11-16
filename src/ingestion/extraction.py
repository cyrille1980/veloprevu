"""Extraction des données brutes : historique BIXI, météo Open-Meteo, état des stations GBFS.

Chaque extraction est journalisée et tracée dans data/raw/manifeste.json
(source, URL, date, taille, empreinte SHA-256) pour assurer la traçabilité.

Usage :
    python -m src.ingestion.extraction --source reel
    python -m src.ingestion.extraction --source echantillon
"""

import argparse
import hashlib
import json
import time
from datetime import UTC, datetime
from pathlib import Path

import httpx

from src.common.config import charger_config, dossier
from src.common.journal import obtenir_journal
from src.ingestion.echantillon import generer_echantillon

journal = obtenir_journal("ingestion.extraction")


class ErreurExtraction(RuntimeError):
    """Levée quand une source reste inaccessible après toutes les tentatives."""


def telecharger(
    url: str,
    destination: Path,
    params: dict | None = None,
    tentatives: int = 3,
    delai_initial_s: float = 2,
    timeout_s: float = 120,
    client: httpx.Client | None = None,
) -> Path:
    """Télécharge une URL vers un fichier, avec nouvelles tentatives et attente exponentielle."""
    destination.parent.mkdir(parents=True, exist_ok=True)
    client = client or httpx.Client(timeout=timeout_s, follow_redirects=True)
    for essai in range(1, tentatives + 1):
        try:
            reponse = client.get(url, params=params)
            reponse.raise_for_status()
            destination.write_bytes(reponse.content)
            journal.info("telechargement_reussi", url=url, octets=len(reponse.content), essai=essai)
            return destination
        except httpx.HTTPError as e:
            journal.warning("telechargement_echec", url=url, essai=essai, erreur=str(e))
            if essai < tentatives:
                time.sleep(delai_initial_s * 2 ** (essai - 1))
    raise ErreurExtraction(f"Source inaccessible après {tentatives} tentatives : {url}")


def tracer(cfg: dict, source: str, url: str, fichier: Path) -> dict:
    """Ajoute une entrée au manifeste des extractions."""
    entree = {
        "source": source,
        "url": url,
        "fichier": str(fichier),
        "date_extraction": datetime.now(UTC).isoformat(timespec="seconds"),
        "octets": fichier.stat().st_size,
        "sha256": hashlib.sha256(fichier.read_bytes()).hexdigest(),
    }
    manifeste = dossier(cfg, "brut") / "manifeste.json"
    entrees = json.loads(manifeste.read_text(encoding="utf-8")) if manifeste.exists() else []
    entrees.append(entree)
    manifeste.write_text(json.dumps(entrees, indent=2, ensure_ascii=False), encoding="utf-8")
    return entree


def _options(cfg: dict) -> dict:
    e = cfg["extraction"]
    return {
        "tentatives": e["tentatives"],
        "delai_initial_s": e["delai_initial_s"],
        "timeout_s": e["timeout_s"],
    }


def extraire_bixi(cfg: dict, client: httpx.Client | None = None) -> Path:
    url = cfg["sources"]["bixi"]["url_historique"]
    if not url:
        raise ErreurExtraction(
            "sources.bixi.url_historique est vide : copier le lien du fichier annuel "
            "depuis https://bixi.com/fr/donnees-ouvertes/ dans configs/config.yaml"
        )
    nom = url.split("/")[-1].split("?")[0] or "bixi.zip"
    fichier = telecharger(url, dossier(cfg, "brut") / "bixi" / nom, client=client, **_options(cfg))
    tracer(cfg, "bixi_historique", url, fichier)
    return fichier


def extraire_meteo(cfg: dict, client: httpx.Client | None = None) -> Path:
    m = cfg["sources"]["meteo"]
    params = {
        "latitude": m["latitude"],
        "longitude": m["longitude"],
        "start_date": cfg["periode"]["debut"],
        "end_date": cfg["periode"]["fin"],
        "hourly": ",".join(m["variables"]),
        "timezone": cfg["preparation"]["fuseau"],
    }
    fichier = telecharger(
        m["url"],
        dossier(cfg, "brut") / "meteo" / "meteo.json",
        params=params,
        client=client,
        **_options(cfg),
    )
    tracer(cfg, "meteo_open_meteo", m["url"], fichier)
    return fichier


def extraire_gbfs(cfg: dict, client: httpx.Client | None = None) -> Path:
    """Instantané de l'état des stations (vélos disponibles), horodaté."""
    url = cfg["sources"]["bixi"]["url_gbfs"] + "/station_status.json"
    horodatage = datetime.now(UTC).strftime("%Y%m%dT%H%M%S")
    fichier = telecharger(
        url,
        dossier(cfg, "brut") / "gbfs" / f"station_status_{horodatage}.json",
        client=client,
        **_options(cfg),
    )
    tracer(cfg, "bixi_gbfs", url, fichier)
    return fichier


def extraire(cfg: dict, source: str = "reel") -> None:
    journal.info("extraction_demarree", source=source)
    try:
        if source == "echantillon":
            generer_echantillon(cfg)
        else:
            extraire_bixi(cfg)
            extraire_meteo(cfg)
    except Exception:
        journal.exception("extraction_erreur", source=source)
        raise
    journal.info("extraction_terminee", source=source)


if __name__ == "__main__":
    analyseur = argparse.ArgumentParser(description="Extraction des données brutes")
    analyseur.add_argument("--source", choices=["reel", "echantillon", "gbfs"], default="reel")
    args = analyseur.parse_args()
    config = charger_config()
    if args.source == "gbfs":
        extraire_gbfs(config)
    else:
        extraire(config, args.source)
