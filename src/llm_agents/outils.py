"""Outils que l'agent de surveillance a le droit d'appeler (principe du moindre privilège).

Tous sont en lecture seule, sauf `proposer_reentrainement` qui se limite à écrire une
proposition en attente : seul un humain peut l'approuver (src.llm_agents.valider).
"""

import json
import uuid
from datetime import UTC, datetime

import httpx

from src.common.config import dossier
from src.training import registre

# Requêtes PromQL autorisées : l'agent choisit un nom, jamais une requête libre.
REQUETES_PROMETHEUS = {
    "taux_erreurs": 'sum(rate(veloprevu_requetes_total{code=~"5.."}[5m])) / sum(rate(veloprevu_requetes_total[5m]))',
    "latence_p95_secondes": "histogram_quantile(0.95, sum(rate(veloprevu_latence_secondes_bucket[5m])) by (le))",
    "predictions_par_minute": "sum(rate(veloprevu_predictions_total[5m])) * 60",
    "mae_glissante": "veloprevu_mae_glissante",
    "wape_glissant": "veloprevu_wape_glissant",
}


def lire_rapport_derive(cfg: dict) -> dict:
    fichier = dossier(cfg, "rapports") / "derive" / "dernier.json"
    if not fichier.exists():
        return {"erreur": "Aucun rapport de dérive disponible"}
    r = json.loads(fichier.read_text(encoding="utf-8"))
    cles = [
        "date",
        "origine",
        "variables_derivees",
        "part_variables_derivees",
        "alerte_donnees",
        "performance",
        "alerte_performance",
        "diagnostic",
    ]
    resume = {k: r.get(k) for k in cles}
    resume["psi_par_variable"] = {v: d["psi"] for v, d in r["variables"].items()}
    return resume


def lire_etat_modele(cfg: dict) -> dict:
    etat = {"versions": registre.lister(cfg)}
    dernier = dossier(cfg, "rapports") / "dernier_entrainement.json"
    if dernier.exists():
        etat["dernier_entrainement"] = json.loads(dernier.read_text(encoding="utf-8"))
    return etat


def interroger_metrique(cfg: dict, nom: str) -> dict:
    if nom not in REQUETES_PROMETHEUS:
        return {"erreur": f"Métrique non autorisée. Choix : {sorted(REQUETES_PROMETHEUS)}"}
    try:
        reponse = httpx.get(
            f"{cfg['agent']['url_prometheus']}/api/v1/query", params={"query": REQUETES_PROMETHEUS[nom]}, timeout=10
        )
        reponse.raise_for_status()
        resultats = reponse.json()["data"]["result"]
        return {"metrique": nom, "valeur": float(resultats[0]["value"][1]) if resultats else None}
    except (httpx.HTTPError, KeyError, ValueError) as e:
        return {"erreur": f"Prometheus indisponible : {str(e)[:150]}"}


def proposer_reentrainement(cfg: dict, justification: str) -> dict:
    """N'entraîne rien : dépose une proposition que l'humain approuvera ou rejettera."""
    rapport = lire_rapport_derive(cfg)
    proposition = {
        "id": uuid.uuid4().hex[:8],
        "date": datetime.now(UTC).isoformat(timespec="seconds"),
        "statut": "en_attente_validation_humaine",
        "justification": justification[:1000],
        "origine_donnees": rapport.get("origine"),
    }
    rep = dossier(cfg, "rapports") / "propositions"
    rep.mkdir(parents=True, exist_ok=True)
    (rep / f"{proposition['id']}.json").write_text(
        json.dumps(proposition, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    return {
        "id": proposition["id"],
        "statut": proposition["statut"],
        "suite": f"Un humain doit exécuter : python -m src.llm_agents.valider approuver {proposition['id']}",
    }


OUTILS = {
    "lire_rapport_derive": (
        lire_rapport_derive,
        "Lit le dernier rapport de dérive (variables, PSI, performance, diagnostic).",
        {},
    ),
    "lire_etat_modele": (lire_etat_modele, "Liste les versions du modèle, le champion et le dernier entraînement.", {}),
    "interroger_metrique": (
        interroger_metrique,
        "Lit une métrique de production dans Prometheus.",
        {"nom": {"type": "string", "enum": sorted(REQUETES_PROMETHEUS)}},
    ),
    "proposer_reentrainement": (
        proposer_reentrainement,
        "Dépose une proposition de réentraînement soumise à validation humaine. N'exécute rien.",
        {"justification": {"type": "string", "description": "Raisons chiffrées de la proposition"}},
    ),
}


def schemas_outils() -> list[dict]:
    """Description des outils au format « function calling » compatible OpenAI."""
    return [
        {
            "type": "function",
            "function": {
                "name": nom,
                "description": description,
                "parameters": {
                    "type": "object",
                    "properties": parametres,
                    "required": list(parametres),
                    "additionalProperties": False,
                },
            },
        }
        for nom, (_, description, parametres) in OUTILS.items()
    ]
