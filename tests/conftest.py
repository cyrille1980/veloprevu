"""Fixtures partagées : configuration isolée dans un dossier temporaire et pipeline exécuté une fois."""

import os
import tempfile
from pathlib import Path

# À définir avant l'import des modules src : les journaux des tests ne polluent pas le dépôt.
os.environ["VELOPREVU_LOGS"] = tempfile.mkdtemp(prefix="veloprevu_logs_")
for variable in ("MLFLOW_TRACKING_URI", "VELOPREVU_API_KEY", "VELOPREVU_API_URL"):
    os.environ.pop(variable, None)

import pytest  # noqa: E402
from fastapi.testclient import TestClient  # noqa: E402

from src.common.config import charger_config  # noqa: E402

RACINE = Path(__file__).resolve().parents[1]


@pytest.fixture(scope="session")
def cfg(tmp_path_factory) -> dict:
    base = tmp_path_factory.mktemp("veloprevu")
    config = charger_config(RACINE / "configs" / "config.yaml")
    config["chemins"] = {cle: str(base / valeur) for cle, valeur in config["chemins"].items()}
    config["mlflow"]["uri"] = f"sqlite:///{base / 'mlflow.db'}"
    config["preparation"]["nb_stations"] = 6
    # Un seul petit LightGBM : les tests doivent rester rapides.
    config["modeles"]["lightgbm"] = [{"n_estimators": 80, "learning_rate": 0.1, "num_leaves": 15}]
    config["modeles"]["ridge"]["alpha"] = [1.0]
    return config


@pytest.fixture(scope="session")
def pipeline(cfg) -> dict:
    """Exécute le pipeline complet sur l'échantillon synthétique (une fois par session)."""
    import mlflow

    from src.pipeline import executer

    resultat = executer(cfg, source="echantillon")
    mlflow.set_tracking_uri(cfg["mlflow"]["uri"])
    return resultat


@pytest.fixture()
def client(cfg, pipeline):
    from src.serving.api import creer_app

    with TestClient(creer_app(cfg)) as c:
        yield c


@pytest.fixture()
def demande_valide() -> dict:
    return {
        "station": "Vieux-Port",
        "horodatage": "2025-07-15T17:00:00",
        "temperature": 24.5,
        "precipitation": 0.0,
        "vent": 12.0,
        "demande_h_1": 9,
        "demande_h_2": 7,
        "demande_h_24": 10,
        "demande_h_168": 11,
        "moyenne_24h": 4.2,
    }
