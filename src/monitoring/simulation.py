"""Simulation contrôlée de dérive à partir du jeu de test.

Scénarios :
    aucune    le jeu de test tel quel (contrôle : aucune alerte attendue)
    hiver     températures négatives et demande divisée par ~3 (dérive des données)
    capteur   capteur de température en panne, bloqué à 0 °C (problème de qualité)
    concept   mêmes entrées, mais demande réelle divisée par 2 (dérive de concept,
              par exemple une piste cyclable fermée) : invisible dans les entrées

Les données simulées complètent les données réelles ; dès que l'historique d'hiver
réel est disponible, il peut être passé à `derive --courant` à la place.
"""

import numpy as np
import pandas as pd

from src.common.config import dossier
from src.features.construction import CIBLE

SCENARIOS = ("aucune", "hiver", "capteur", "concept")
RETARDS = ["demande_h_1", "demande_h_2", "demande_h_24", "demande_h_168"]


def simuler(cfg: dict, scenario: str) -> pd.DataFrame:
    if scenario not in SCENARIOS:
        raise ValueError(f"Scénario inconnu : {scenario} (choix : {SCENARIOS})")
    rng = np.random.default_rng(cfg["projet"]["graine"])
    df = pd.read_parquet(dossier(cfg, "features") / "features.parquet")
    df = df[df["jeu"] == "test"].drop(columns="jeu").sort_values("heure").reset_index(drop=True)

    if scenario == "hiver":
        df["temperature"] = df["temperature"] - 26 + rng.normal(0, 2, len(df))
        df["vent"] = df["vent"] * 1.3
        # Amincissement binomial : chaque départ de l'été a 30 % de chances d'exister en hiver.
        for colonne in [*RETARDS, CIBLE]:
            df[colonne] = rng.binomial(df[colonne].round().astype(int), 0.3).astype(float)
        df["moyenne_24h"] = df["moyenne_24h"] * 0.3
    elif scenario == "capteur":
        df["temperature"] = 0.0
    elif scenario == "concept":
        df[CIBLE] = rng.binomial(df[CIBLE].round().astype(int), 0.5).astype(float)

    df.to_parquet(dossier(cfg, "features") / f"courant_{scenario}.parquet", index=False)
    return df
