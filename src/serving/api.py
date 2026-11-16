"""API REST de prédiction (FastAPI).

Points d'accès :
    GET  /health         état du service et version du modèle
    GET  /model          détails du modèle servi
    POST /predict        prévision de la demande pour une station et une heure
    POST /feedback       valeur réellement observée (permet de suivre la performance)
    POST /admin/reload   recharge le champion (après promotion, réentraînement ou rollback)
    GET  /metrics        métriques Prometheus

Si la variable VELOPREVU_API_KEY est définie, /predict, /feedback et /admin/reload exigent
l'en-tête X-API-Key. Lancement : uvicorn src.serving.api:creer_app --factory --port 8000
"""

import json
import os
import secrets
import time
import uuid
from collections import OrderedDict, deque
from contextlib import asynccontextmanager
from datetime import UTC, datetime
from pathlib import Path

import pandas as pd
from fastapi import Depends, FastAPI, Header, HTTPException, Request, Response
from prometheus_client import (
    CONTENT_TYPE_LATEST,
    CollectorRegistry,
    Counter,
    Gauge,
    Histogram,
    ProcessCollector,
    generate_latest,
)
from pydantic import BaseModel, Field

from src.common.config import charger_config, dossier
from src.common.journal import obtenir_journal
from src.features.construction import FEATURES, variables_calendrier
from src.serving.chargement import charger_modele

journal = obtenir_journal("serving.api")
ROUTES_SUIVIES = {"/health", "/model", "/predict", "/feedback", "/admin/reload"}


class DemandePrediction(BaseModel):
    station: str = Field(min_length=1, examples=["Métro Mont-Royal"])
    horodatage: datetime = Field(description="Début de l'heure à prédire (heure de Montréal)")
    temperature: float = Field(ge=-50, le=50, description="°C")
    precipitation: float = Field(ge=0, le=200, description="mm dans l'heure")
    vent: float = Field(ge=0, le=200, description="km/h")
    demande_h_1: float = Field(ge=0, le=2000, description="Départs 1 h plus tôt")
    demande_h_2: float = Field(ge=0, le=2000, description="Départs 2 h plus tôt")
    demande_h_24: float = Field(ge=0, le=2000, description="Départs 24 h plus tôt")
    demande_h_168: float = Field(ge=0, le=2000, description="Départs 7 jours plus tôt")
    moyenne_24h: float = Field(ge=0, le=2000, description="Moyenne des 24 dernières heures")


class Retour(BaseModel):
    id_prediction: str
    valeur_reelle: float = Field(ge=0, le=2000)


class Metriques:
    """Métriques Prometheus, dans un registre propre à l'application."""

    def __init__(self):
        r = self.registre = CollectorRegistry()
        ProcessCollector(registry=r)  # CPU, mémoire, descripteurs du processus
        self.requetes = Counter("veloprevu_requetes_total", "Requêtes HTTP", ["route", "code"], registry=r)
        self.latence = Histogram("veloprevu_latence_secondes", "Temps de réponse", ["route"], registry=r)
        self.erreurs = Counter("veloprevu_erreurs_total", "Erreurs internes", ["route"], registry=r)
        self.predictions = Counter("veloprevu_predictions_total", "Prédictions servies", ["version"], registry=r)
        self.valeur = Histogram(
            "veloprevu_prediction_valeur",
            "Distribution des prédictions",
            buckets=(0, 1, 2, 5, 10, 20, 40, 80, 160),
            registry=r,
        )
        self.temperature = Histogram(
            "veloprevu_entree_temperature",
            "Distribution de la température reçue",
            buckets=(-30, -20, -10, 0, 10, 20, 30, 40),
            registry=r,
        )
        self.demande_h1 = Histogram(
            "veloprevu_entree_demande_h_1",
            "Distribution du retard 1 h reçu",
            buckets=(0, 1, 2, 5, 10, 20, 40, 80, 160),
            registry=r,
        )
        self.modele = Gauge("veloprevu_modele_info", "Modèle servi (valeur 1)", ["version", "source"], registry=r)
        self.retours = Counter("veloprevu_retours_total", "Valeurs réelles reçues", registry=r)
        self.mae = Gauge("veloprevu_mae_glissante", "MAE sur les 500 derniers retours", registry=r)
        self.wape = Gauge("veloprevu_wape_glissant", "WAPE sur les 500 derniers retours", registry=r)
        self.derive_part = Gauge("veloprevu_derive_part_variables", "Part des variables ayant dérivé", registry=r)
        self.derive_alerte = Gauge("veloprevu_derive_alerte", "1 si une alerte de dérive est active", registry=r)


def verifier_cle(x_api_key: str | None = Header(default=None)):
    attendue = os.getenv("VELOPREVU_API_KEY")
    if attendue and not secrets.compare_digest(x_api_key or "", attendue):
        raise HTTPException(status_code=401, detail="Clé d'API absente ou invalide")


def creer_app(cfg: dict | None = None) -> FastAPI:
    cfg = cfg or charger_config()
    m = Metriques()
    etat: dict = {"modele": None, "version": None, "source": None, "stations": {}}
    en_attente: OrderedDict[str, float] = OrderedDict()  # id_prediction -> prédiction
    erreurs_recentes: deque[tuple[float, float]] = deque(maxlen=500)  # (|erreur|, réel)
    journaux = dossier(cfg, "journaux")

    def charger():
        try:
            infos = charger_modele(cfg)
            stations = json.loads((dossier(cfg, "features") / "stations.json").read_text(encoding="utf-8"))
        except Exception as e:
            journal.error("modele_absent", erreur=str(e)[:200])
            return
        etat.update(modele=infos["modele"], version=infos["version"], source=infos["source"], stations=stations)
        m.modele.clear()
        m.modele.labels(version=infos["version"], source=infos["source"]).set(1)

    @asynccontextmanager
    async def cycle_de_vie(app: FastAPI):
        journal.info("service_demarre")
        charger()
        yield
        journal.info("service_arrete")

    app = FastAPI(
        title="VéloPrévu",
        version="1.0.0",
        lifespan=cycle_de_vie,
        description="Prévision de la demande horaire BIXI par station",
    )

    @app.middleware("http")
    async def instrumenter(request: Request, call_next):
        """Identifiant de corrélation, latence et compteur pour chaque requête."""
        id_requete = request.headers.get("X-Request-ID", uuid.uuid4().hex[:16])
        request.state.id_requete = id_requete
        route = request.url.path if request.url.path in ROUTES_SUIVIES else "autre"
        debut = time.perf_counter()
        try:
            reponse = await call_next(request)
        except Exception:
            m.erreurs.labels(route=route).inc()
            m.requetes.labels(route=route, code="500").inc()
            journal.exception("erreur_interne", id_requete=id_requete, route=request.url.path)
            return Response(
                json.dumps({"detail": "Erreur interne", "id_requete": id_requete}),
                status_code=500,
                media_type="application/json",
            )
        duree = time.perf_counter() - debut
        if request.url.path != "/metrics":
            m.requetes.labels(route=route, code=str(reponse.status_code)).inc()
            m.latence.labels(route=route).observe(duree)
            journal.info(
                "requete",
                id_requete=id_requete,
                methode=request.method,
                route=request.url.path,
                code=reponse.status_code,
                duree_ms=round(duree * 1000, 2),
            )
        reponse.headers["X-Request-ID"] = id_requete
        return reponse

    @app.get("/health")
    def sante(response: Response):
        pret = etat["modele"] is not None
        if not pret:
            response.status_code = 503
        return {"statut": "ok" if pret else "modele_absent", "version_modele": etat["version"]}

    @app.get("/model")
    def modele():
        return {
            "nom": cfg["mlflow"]["nom_modele"],
            "version": etat["version"],
            "source": etat["source"],
            "variables": FEATURES,
            "stations": sorted(etat["stations"]),
        }

    @app.post("/predict", dependencies=[Depends(verifier_cle)])
    def predire(demande: DemandePrediction, request: Request):
        if etat["modele"] is None:
            raise HTTPException(status_code=503, detail="Aucun modèle chargé")
        if demande.station not in etat["stations"]:
            raise HTTPException(status_code=422, detail=f"Station inconnue : {demande.station}")
        ligne = demande.model_dump(exclude={"station", "horodatage"})
        ligne.update(variables_calendrier(demande.horodatage), station_code=etat["stations"][demande.station])
        prediction = max(0.0, float(etat["modele"].predict(pd.DataFrame([ligne])[FEATURES])[0]))

        id_prediction = uuid.uuid4().hex
        en_attente[id_prediction] = prediction
        if len(en_attente) > 50_000:
            en_attente.popitem(last=False)
        m.predictions.labels(version=etat["version"]).inc()
        m.valeur.observe(prediction)
        m.temperature.observe(demande.temperature)
        m.demande_h1.observe(demande.demande_h_1)
        trace = {
            "id_prediction": id_prediction,
            "id_requete": request.state.id_requete,
            "date": datetime.now(UTC).isoformat(timespec="seconds"),
            "version_modele": etat["version"],
            "station": demande.station,
            "heure": demande.horodatage.isoformat(),
            **ligne,
            "prediction": prediction,
        }
        with open(Path(journaux) / "predictions.jsonl", "a", encoding="utf-8") as f:
            f.write(json.dumps(trace, ensure_ascii=False) + "\n")
        journal.info(
            "prediction",
            id_requete=request.state.id_requete,
            id_prediction=id_prediction,
            station=demande.station,
            prediction=round(prediction, 2),
            version=etat["version"],
        )
        return {
            "id_prediction": id_prediction,
            "prediction": round(prediction, 2),
            "version_modele": etat["version"],
            "unite": "départs par heure",
        }

    @app.post("/feedback", dependencies=[Depends(verifier_cle)])
    def retour(r: Retour):
        prediction = en_attente.pop(r.id_prediction, None)
        if prediction is None:
            raise HTTPException(status_code=404, detail="Prédiction inconnue ou déjà évaluée")
        erreurs_recentes.append((abs(prediction - r.valeur_reelle), r.valeur_reelle))
        m.retours.inc()
        m.mae.set(sum(e for e, _ in erreurs_recentes) / len(erreurs_recentes))
        total = sum(v for _, v in erreurs_recentes)
        if total > 0:
            m.wape.set(sum(e for e, _ in erreurs_recentes) / total)
        with open(Path(journaux) / "retours.jsonl", "a", encoding="utf-8") as f:
            f.write(json.dumps({"id_prediction": r.id_prediction, "demande": r.valeur_reelle}) + "\n")
        return {"erreur_absolue": round(abs(prediction - r.valeur_reelle), 2)}

    @app.post("/admin/reload", dependencies=[Depends(verifier_cle)])
    def recharger():
        avant = etat["version"]
        charger()
        journal.info("modele_recharge", version_avant=avant, version_apres=etat["version"])
        return {"version_avant": avant, "version_apres": etat["version"]}

    @app.get("/metrics")
    def metriques():
        # Le dernier rapport de dérive (produit par lot) est exposé comme jauges.
        dernier = dossier(cfg, "rapports") / "derive" / "dernier.json"
        if dernier.exists():
            try:
                d = json.loads(dernier.read_text(encoding="utf-8"))
                m.derive_part.set(d["part_variables_derivees"])
                m.derive_alerte.set(int(d["alerte_donnees"] or d["alerte_performance"]))
            except (ValueError, KeyError):
                journal.warning("rapport_derive_illisible")
        return Response(generate_latest(m.registre), media_type=CONTENT_TYPE_LATEST)

    return app
