"""Test de fumée après déploiement : le service répond-il correctement ?

Attend que /health soit prêt, puis vérifie une prédiction valide et le rejet d'une entrée
invalide. Code de sortie 0 si tout va bien, 1 sinon (utilisé par la CI/CD).

Usage : python scripts/test_fumee.py [http://localhost:8000]
"""

import os
import sys
import time

import httpx


def principal(url: str, delai_max_s: int = 90) -> int:
    entetes = {"X-API-Key": os.getenv("VELOPREVU_API_KEY", "")}
    limite = time.time() + delai_max_s
    while True:
        try:
            sante = httpx.get(f"{url}/health", timeout=5)
            if sante.status_code == 200:
                break
            etat = f"HTTP {sante.status_code} {sante.text[:100]}"
        except httpx.HTTPError as e:
            etat = str(e)
        if time.time() > limite:
            print(f"ÉCHEC : service non prêt après {delai_max_s} s ({etat})")
            return 1
        time.sleep(3)
    print("health :", sante.json())

    station = httpx.get(f"{url}/model", timeout=10).json()["stations"][0]
    demande = {
        "station": station,
        "horodatage": "2025-07-15T17:00:00",
        "temperature": 24.0,
        "precipitation": 0.0,
        "vent": 10.0,
        "demande_h_1": 8,
        "demande_h_2": 6,
        "demande_h_24": 9,
        "demande_h_168": 10,
        "moyenne_24h": 4.0,
    }
    reponse = httpx.post(f"{url}/predict", json=demande, headers=entetes, timeout=30)
    print("predict :", reponse.status_code, reponse.text[:200])
    if reponse.status_code != 200 or reponse.json()["prediction"] < 0:
        return 1

    invalide = httpx.post(f"{url}/predict", json={**demande, "temperature": 999}, headers=entetes, timeout=30)
    print("entrée invalide :", invalide.status_code)
    if invalide.status_code != 422:
        return 1

    if "veloprevu_predictions_total" not in httpx.get(f"{url}/metrics", timeout=10).text:
        print("ÉCHEC : métriques absentes")
        return 1
    print("Test de fumée réussi")
    return 0


if __name__ == "__main__":
    sys.exit(principal(sys.argv[1] if len(sys.argv) > 1 else "http://localhost:8000"))
