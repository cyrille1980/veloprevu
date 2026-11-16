"""Génère un échantillon synthétique aux mêmes formats que les sources réelles.

Sert aux tests automatisés et à la CI (pas d'accès réseau), jamais à remplacer les
données réelles : le fichier de déplacements imite le CSV BIXI et le fichier météo
imite la réponse JSON d'Open-Meteo, de sorte que le reste du pipeline est identique.
"""

import json
from pathlib import Path

import numpy as np
import pandas as pd

from src.common.config import dossier
from src.common.journal import obtenir_journal

journal = obtenir_journal("ingestion.echantillon")

STATIONS = [
    "Métro Mont-Royal",
    "Parc La Fontaine",
    "Métro Laurier",
    "Square Victoria",
    "Marché Jean-Talon",
    "Vieux-Port",
    "Université McGill",
    "Canal de Lachine",
    "Métro Berri-UQAM",
    "Parc Jarry",
    "Place des Arts",
    "Marché Atwater",
]


def _profil_horaire(heure: np.ndarray, fin_semaine: np.ndarray) -> np.ndarray:
    """Deux pointes en semaine (8 h et 17 h), une bosse l'après-midi en fin de semaine."""
    semaine = 0.15 + np.exp(-((heure - 8) ** 2) / 4) + 1.2 * np.exp(-((heure - 17) ** 2) / 6)
    weekend = 0.15 + 1.1 * np.exp(-((heure - 14) ** 2) / 18)
    return np.where(fin_semaine, weekend, semaine)


def generer_echantillon(cfg: dict, nb_stations: int = 8, nb_jours: int = 70) -> dict:
    rng = np.random.default_rng(cfg["projet"]["graine"])
    fuseau = cfg["preparation"]["fuseau"]
    heures = pd.date_range(cfg["periode"]["debut"], periods=nb_jours * 24, freq="h")
    jour = np.arange(len(heures)) / 24

    # Météo : cycle saisonnier lent + cycle jour/nuit + bruit ; averses aléatoires.
    temperature = (
        18 + 6 * np.sin(2 * np.pi * jour / 120) + 5 * np.sin(2 * np.pi * (heures.hour - 9) / 24)
    ) + rng.normal(0, 2, len(heures))
    pluie = np.where(rng.random(len(heures)) < 0.08, rng.gamma(2.0, 1.5, len(heures)), 0.0)
    vent = np.clip(rng.normal(12, 5, len(heures)), 0, None)

    meteo = {
        "hourly": {
            "time": [h.strftime("%Y-%m-%dT%H:%M") for h in heures],
            "temperature_2m": np.round(temperature, 1).tolist(),
            "precipitation": np.round(pluie, 1).tolist(),
            "wind_speed_10m": np.round(vent, 1).tolist(),
        }
    }

    # Demande : Poisson dont l'intensité dépend de la station, de l'heure et de la météo.
    fin_semaine = np.asarray(heures.dayofweek >= 5)
    effet_meteo = np.clip(1 + (temperature - 18) * 0.03, 0.2, 1.6) * np.where(pluie > 0, 0.45, 1.0)
    lignes = []
    for i, station in enumerate(STATIONS[:nb_stations]):
        intensite = (4 + 2.5 * i) * _profil_horaire(np.asarray(heures.hour), fin_semaine)
        nb = rng.poisson(intensite * effet_meteo)
        debut_heure = np.repeat(heures.values, nb)
        decalage = rng.integers(0, 3600, len(debut_heure)).astype("timedelta64[s]")
        debut = pd.Series(debut_heure + decalage).dt.tz_localize(fuseau, ambiguous="NaT", nonexistent="NaT")
        ms = (debut.dt.tz_convert("UTC").dt.tz_localize(None) - pd.Timestamp(0)) // pd.Timedelta("1ms")
        lignes.append(
            pd.DataFrame(
                {
                    "STARTSTATIONNAME": station,
                    "STARTTIMEMS": ms,
                    "ENDSTATIONNAME": rng.choice(STATIONS[:nb_stations], len(ms)),
                    "ENDTIMEMS": ms + rng.integers(120, 3600, len(ms)) * 1000,
                }
            )
        )
    deplacements = pd.concat(lignes, ignore_index=True).dropna()

    brut = dossier(cfg, "brut")
    (brut / "bixi").mkdir(parents=True, exist_ok=True)
    (brut / "meteo").mkdir(parents=True, exist_ok=True)
    chemin_bixi = Path(brut / "bixi" / "echantillon.csv")
    deplacements.to_csv(chemin_bixi, index=False)
    (brut / "meteo" / "meteo.json").write_text(json.dumps(meteo), encoding="utf-8")
    journal.info("echantillon_genere", deplacements=len(deplacements), heures=len(heures))
    return {"deplacements": len(deplacements), "heures": len(heures)}
