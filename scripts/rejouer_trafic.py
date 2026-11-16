"""Rejoue des données vers l'API pour alimenter les tableaux de bord et simuler une dérive.

Pour chaque ligne : POST /predict, puis POST /feedback avec la demande réellement observée
(dans la réalité, cette valeur est connue une heure plus tard).

Exemples :
    python scripts/rejouer_trafic.py --scenario aucune --n 300
    python scripts/rejouer_trafic.py --scenario hiver --n 300      # la dérive apparaît dans Grafana
"""

import argparse
import os
import sys
import time
from pathlib import Path

import httpx
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.common.config import charger_config  # noqa: E402
from src.monitoring.simulation import SCENARIOS, simuler  # noqa: E402

CHAMPS = [
    "temperature",
    "precipitation",
    "vent",
    "demande_h_1",
    "demande_h_2",
    "demande_h_24",
    "demande_h_168",
    "moyenne_24h",
]


def rejouer(url: str, donnees: pd.DataFrame, n: int, pause: float, cle: str | None) -> dict:
    entetes = {"X-API-Key": cle} if cle else {}
    bilan = {"envoyees": 0, "refusees": 0, "erreur_abs_totale": 0.0}
    with httpx.Client(base_url=url, headers=entetes, timeout=30) as client:
        for _, ligne in donnees.head(n).iterrows():
            corps = {"station": ligne["station"], "horodatage": ligne["heure"].isoformat()}
            corps.update({c: float(ligne[c]) for c in CHAMPS})
            reponse = client.post("/predict", json=corps)
            if reponse.status_code != 200:
                bilan["refusees"] += 1
                continue
            retour = client.post(
                "/feedback",
                json={"id_prediction": reponse.json()["id_prediction"], "valeur_reelle": float(ligne["demande"])},
            )
            bilan["envoyees"] += 1
            bilan["erreur_abs_totale"] += retour.json().get("erreur_absolue", 0.0)
            time.sleep(pause)
    bilan["mae"] = round(bilan.pop("erreur_abs_totale") / max(bilan["envoyees"], 1), 3)
    return bilan


if __name__ == "__main__":
    a = argparse.ArgumentParser(description="Rejoue du trafic vers l'API")
    a.add_argument("--url", default=os.getenv("VELOPREVU_API_URL", "http://localhost:8000"))
    a.add_argument("--scenario", choices=SCENARIOS, default="aucune")
    a.add_argument("--fichier", help="fichier parquet à rejouer à la place d'un scénario simulé")
    a.add_argument("--n", type=int, default=300)
    a.add_argument("--pause", type=float, default=0.05, help="secondes entre deux requêtes")
    args = a.parse_args()
    cfg = charger_config()
    df = pd.read_parquet(args.fichier) if args.fichier else simuler(cfg, args.scenario)
    # Mélange reproductible : toutes les stations et toutes les heures sont représentées.
    df = df.sample(frac=1, random_state=cfg["projet"]["graine"])
    print(rejouer(args.url, df, args.n, args.pause, os.getenv("VELOPREVU_API_KEY")))
