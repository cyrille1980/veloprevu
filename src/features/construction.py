"""Construction des variables (features) et découpage temporel train / validation / test.

Prévention de la fuite de données :
- les retards (lags) et la moyenne mobile n'utilisent que des heures strictement passées ;
- le découpage est chronologique, jamais aléatoire ;
- la même fonction `variables_calendrier` sert à l'entraînement et à l'API.

Le mois n'est volontairement pas une variable : un modèle entraîné sur l'été ne l'aurait
jamais vu varier et ne pourrait rien en tirer en hiver.

Usage : python -m src.features.construction
"""

import json
import math
from datetime import datetime

import numpy as np
import pandas as pd

from src.common.config import charger_config, dossier
from src.common.journal import obtenir_journal

journal = obtenir_journal("features.construction")

CIBLE = "demande"
FEATURES = [
    "station_code",
    "heure_jour",
    "jour_semaine",
    "fin_semaine",
    "heure_sin",
    "heure_cos",
    "temperature",
    "precipitation",
    "vent",
    "demande_h_1",
    "demande_h_2",
    "demande_h_24",
    "demande_h_168",
    "moyenne_24h",
]
RETARDS = {"demande_h_1": 1, "demande_h_2": 2, "demande_h_24": 24, "demande_h_168": 168}


def variables_calendrier(horodatage: datetime) -> dict:
    """Variables dérivées de la date ; partagées par l'entraînement et le service."""
    h = horodatage.hour
    return {
        "heure_jour": h,
        "jour_semaine": horodatage.weekday(),
        "fin_semaine": int(horodatage.weekday() >= 5),
        "heure_sin": math.sin(2 * math.pi * h / 24),
        "heure_cos": math.cos(2 * math.pi * h / 24),
    }


def ajouter_variables(df: pd.DataFrame, stations: dict[str, int]) -> pd.DataFrame:
    df = df.sort_values(["station", "heure"]).reset_index(drop=True)
    df["station_code"] = df["station"].map(stations).astype(int)
    df["heure_jour"] = df["heure"].dt.hour
    df["jour_semaine"] = df["heure"].dt.dayofweek
    df["fin_semaine"] = (df["jour_semaine"] >= 5).astype(int)
    df["heure_sin"] = np.sin(2 * np.pi * df["heure_jour"] / 24)
    df["heure_cos"] = np.cos(2 * np.pi * df["heure_jour"] / 24)
    par_station = df.groupby("station")[CIBLE]
    for nom, decalage in RETARDS.items():
        df[nom] = par_station.shift(decalage)
    # shift(1) avant la moyenne : l'heure à prédire n'entre jamais dans sa propre moyenne.
    df["moyenne_24h"] = par_station.transform(lambda s: s.shift(1).rolling(24).mean())
    return df.dropna(subset=FEATURES).reset_index(drop=True)


def decouper(df: pd.DataFrame, cfg: dict) -> pd.DataFrame:
    """Ajoute la colonne `jeu` : les derniers jours vont au test, les précédents à la validation."""
    fin = df["heure"].max()
    debut_test = fin - pd.Timedelta(days=cfg["decoupage"]["test_jours"])
    debut_validation = debut_test - pd.Timedelta(days=cfg["decoupage"]["validation_jours"])
    df["jeu"] = np.select(
        [df["heure"] > debut_test, df["heure"] > debut_validation],
        ["test", "validation"],
        default="train",
    )
    return df


def construire(cfg: dict) -> pd.DataFrame:
    journal.info("features_demarrees")
    df = pd.read_parquet(dossier(cfg, "traite") / "demande_horaire.parquet")
    stations = {nom: i for i, nom in enumerate(sorted(df["station"].unique()))}
    df = decouper(ajouter_variables(df, stations), cfg)
    sortie = dossier(cfg, "features")
    df.to_parquet(sortie / "features.parquet", index=False)
    (sortie / "stations.json").write_text(json.dumps(stations, ensure_ascii=False, indent=2), encoding="utf-8")
    journal.info("features_terminees", lignes=len(df), **df["jeu"].value_counts().to_dict())
    return df


if __name__ == "__main__":
    construire(charger_config())
