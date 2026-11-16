"""Détection de dérive : données d'entrée, performance, et diagnostic qui les distingue.

- Dérive des données : PSI (indice de stabilité de population) par variable entre le jeu
  de référence (entraînement) et le jeu récent ; test de Kolmogorov-Smirnov en complément.
- Dérive de performance : WAPE du modèle sur le jeu récent comparé au WAPE de référence,
  quand les valeurs réelles sont disponibles.
- Une dérive des entrées ne prouve pas une baisse de performance : le diagnostic croise
  les deux signaux (stable, entrées seules, performance seule = concept, ou les deux).

Sorties : reports/derive/dernier.json, historique.jsonl et un rapport HTML Evidently.

Usage :
    python -m src.monitoring.derive --scenario hiver
    python -m src.monitoring.derive --courant chemin/vers/recent.parquet
    python -m src.monitoring.derive --journal        # prédictions réellement servies par l'API
"""

import argparse
import json
from datetime import UTC, datetime

import numpy as np
import pandas as pd
from scipy.stats import ks_2samp

from src.common.config import charger_config, dossier
from src.common.journal import obtenir_journal
from src.evaluation.evaluation import metriques
from src.features.construction import CIBLE, FEATURES
from src.monitoring.simulation import SCENARIOS, simuler

journal = obtenir_journal("monitoring.derive")

VARIABLES_SURVEILLEES = [
    "temperature",
    "precipitation",
    "vent",
    "demande_h_1",
    "demande_h_24",
    "demande_h_168",
    "moyenne_24h",
]


def psi(reference: np.ndarray, courant: np.ndarray, classes: int = 10) -> float:
    """Indice de stabilité de population, classes définies par les quantiles de la référence.

    Repères usuels : < 0,1 stable ; 0,1 à 0,2 à surveiller ; > 0,2 dérive significative.
    """
    bornes = np.unique(np.quantile(reference, np.linspace(0, 1, classes + 1)))
    if len(bornes) < 3:  # variable presque constante (ex. pluie nulle la plupart du temps)
        bas, haut = float(np.min(reference)), float(np.max(reference))
        bornes = np.linspace(bas, haut + 1e-9, 4) if haut > bas else np.array([bas - 0.5, bas + 0.5])
    bornes = bornes.astype(float)
    bornes[0], bornes[-1] = -np.inf, np.inf
    p_ref = np.histogram(reference, bornes)[0] / len(reference)
    p_cur = np.histogram(courant, bornes)[0] / len(courant)
    p_ref, p_cur = np.clip(p_ref, 1e-4, None), np.clip(p_cur, 1e-4, None)
    return float(np.sum((p_cur - p_ref) * np.log(p_cur / p_ref)))


def lire_journal_api(cfg: dict) -> pd.DataFrame:
    """Prédictions servies par l'API, jointes aux valeurs réelles reçues par /feedback."""
    rep = dossier(cfg, "journaux")
    courant = pd.read_json(rep / "predictions.jsonl", lines=True)
    retours = rep / "retours.jsonl"
    if retours.exists():
        courant = courant.merge(pd.read_json(retours, lines=True), on="id_prediction", how="left")
    courant["heure"] = pd.to_datetime(courant["heure"])
    return courant


def analyser(
    reference: pd.DataFrame,
    courant: pd.DataFrame,
    cfg: dict,
    modele=None,
    wape_reference: float | None = None,
    origine: str = "",
) -> dict:
    seuils = cfg["derive"]
    variables = {}
    for v in VARIABLES_SURVEILLEES:
        ref, cur = reference[v].dropna().to_numpy(float), courant[v].dropna().to_numpy(float)
        valeur_psi = psi(ref, cur)
        variables[v] = {
            "psi": round(valeur_psi, 4),
            "ks_pvalue": float(ks_2samp(ref, cur).pvalue),
            "moyenne_reference": round(float(ref.mean()), 3),
            "moyenne_courante": round(float(cur.mean()), 3),
            "derive": bool(valeur_psi > seuils["seuil_psi"]),
        }
    derivees = [v for v, r in variables.items() if r["derive"]]
    part = len(derivees) / len(variables)
    alerte_donnees = part >= seuils["part_variables_alerte"]

    # Performance : seulement si les valeurs réelles sont connues pour le jeu récent.
    performance, alerte_performance = None, False
    etiquetes = courant.dropna(subset=[CIBLE]) if CIBLE in courant else courant.iloc[0:0]
    if modele is not None and len(etiquetes) > 0:
        if "prediction" in etiquetes and etiquetes["prediction"].notna().all():
            predictions = etiquetes["prediction"]  # ce que l'API a réellement répondu
        else:
            predictions = np.clip(modele.predict(etiquetes[FEATURES]), 0, None)
        performance = metriques(etiquetes[CIBLE], predictions)
        performance["nb_etiquetes"] = len(etiquetes)
        if wape_reference and not np.isnan(wape_reference):
            performance["wape_reference"] = wape_reference
            performance["degradation"] = performance["wape"] / wape_reference - 1
            alerte_performance = performance["degradation"] > seuils["degradation_max"]

    if alerte_donnees and alerte_performance:
        diagnostic = "Dérive des données avec dégradation de la performance : réentraînement recommandé."
    elif alerte_performance:
        diagnostic = (
            "Performance dégradée sans dérive marquée des entrées : dérive de concept probable "
            "(la relation entre les variables et la demande a changé)."
        )
    elif alerte_donnees and performance is None:
        diagnostic = "Dérive des données ; impact inconnu tant que les valeurs réelles ne sont pas disponibles."
    elif alerte_donnees:
        diagnostic = "Dérive des données sans dégradation mesurée : à surveiller, pas de réentraînement requis."
    elif derivees:
        diagnostic = (
            f"Dérive limitée à {', '.join(derivees)} : sous le seuil d'alerte, "
            "sans dégradation mesurée de la performance."
        )
    else:
        diagnostic = "Stable : ni dérive des données ni dégradation de la performance."

    return {
        "date": datetime.now(UTC).isoformat(timespec="seconds"),
        "origine": origine,
        "taille_reference": len(reference),
        "taille_courant": len(courant),
        "variables": variables,
        "variables_derivees": derivees,
        "part_variables_derivees": round(part, 3),
        "alerte_donnees": bool(alerte_donnees),
        "performance": performance,
        "alerte_performance": bool(alerte_performance),
        "diagnostic": diagnostic,
    }


def rapport_html(reference: pd.DataFrame, courant: pd.DataFrame, sortie) -> bool:
    """Rapport visuel Evidently (distributions comparées). Non bloquant s'il échoue."""
    try:
        from evidently import Report
        from evidently.presets import DataDriftPreset

        colonnes = [c for c in [*VARIABLES_SURVEILLEES, CIBLE] if c in courant and courant[c].notna().any()]
        Report([DataDriftPreset()]).run(courant[colonnes], reference[colonnes]).save_html(str(sortie))
        return True
    except Exception as e:
        journal.warning("rapport_evidently_echec", erreur=str(e)[:200])
        return False


def detecter(cfg: dict, courant: pd.DataFrame, origine: str) -> dict:
    """Analyse complète, écriture des rapports et journalisation de l'événement."""
    from src.serving.chargement import charger_modele

    reference = pd.read_parquet(dossier(cfg, "features") / "reference.parquet")
    try:
        infos = charger_modele(cfg)
        modele, wape_ref = infos["modele"], infos["wape_reference"]
    except Exception:
        journal.warning("derive_sans_modele")
        modele, wape_ref = None, None
    resultat = analyser(reference, courant, cfg, modele, wape_ref, origine)

    rep = dossier(cfg, "rapports") / "derive"
    rep.mkdir(parents=True, exist_ok=True)
    nom = origine.replace("/", "_").replace(".", "_") or "courant"
    if rapport_html(reference, courant, rep / f"rapport_{nom}.html"):
        resultat["rapport_html"] = str(rep / f"rapport_{nom}.html")
    (rep / "dernier.json").write_text(json.dumps(resultat, indent=2, ensure_ascii=False), encoding="utf-8")
    with open(rep / "historique.jsonl", "a", encoding="utf-8") as f:
        f.write(json.dumps(resultat, ensure_ascii=False) + "\n")

    niveau = journal.warning if resultat["alerte_donnees"] or resultat["alerte_performance"] else journal.info
    niveau(
        "derive_analysee",
        origine=origine,
        variables_derivees=resultat["variables_derivees"],
        part=resultat["part_variables_derivees"],
        alerte_donnees=resultat["alerte_donnees"],
        alerte_performance=resultat["alerte_performance"],
        diagnostic=resultat["diagnostic"],
    )
    return resultat


def charger_courant(cfg: dict, scenario: str | None, courant: str | None, depuis_journal: bool):
    if depuis_journal:
        return lire_journal_api(cfg), "journal_api"
    if courant:
        return pd.read_parquet(courant), courant
    scenario = scenario or "aucune"
    return simuler(cfg, scenario), f"simulation_{scenario}"


def analyseur_arguments(description: str) -> argparse.ArgumentParser:
    a = argparse.ArgumentParser(description=description)
    a.add_argument("--scenario", choices=SCENARIOS, help="simulation contrôlée")
    a.add_argument("--courant", help="fichier parquet de données récentes (features + demande)")
    a.add_argument("--journal", action="store_true", help="utilise les prédictions servies par l'API")
    return a


if __name__ == "__main__":
    args = analyseur_arguments("Détection de dérive").parse_args()
    config = charger_config()
    donnees, source = charger_courant(config, args.scenario, args.courant, args.journal)
    r = detecter(config, donnees, source)
    print(json.dumps({k: v for k, v in r.items() if k != "variables"}, indent=2, ensure_ascii=False))
    print(pd.DataFrame(r["variables"]).T.to_string())
