"""Évaluation de l'agent sur des scénarios de test (configs/scenarios_agent.yaml).

Pour chaque scénario : outils attendus appelés, outils interdits non appelés, mots-clés
attendus présents dans la réponse, nombre d'étapes et latence.
Résultat : reports/agent/evaluation.json

Usage : python -m src.llm_agents.evaluation     (nécessite un LLM joignable)
"""

import json

import yaml

from src.common.config import charger_config, dossier
from src.llm_agents.agent import repondre


def evaluer(cfg: dict, client=None, fichier: str = "configs/scenarios_agent.yaml") -> dict:
    with open(fichier, encoding="utf-8") as f:
        scenarios = yaml.safe_load(f)
    resultats = []
    for s in scenarios:
        r = repondre(cfg, s["question"], client=client)
        appeles = {a["outil"] for a in r["outils"]}
        texte = r["reponse"].lower()
        verifications = {
            "outils_attendus": set(s.get("outils_attendus", [])) <= appeles,
            "outils_interdits": not (set(s.get("outils_interdits", [])) & appeles),
            "mots_attendus": all(m.lower() in texte for m in s.get("mots_attendus", [])),
            "statut": r["statut"] == s.get("statut_attendu", "ok"),
        }
        resultats.append(
            {
                "scenario": s["nom"],
                "reussi": all(verifications.values()),
                **verifications,
                "etapes": r["etapes"],
                "latence_s": r["latence_s"],
                "outils": sorted(appeles),
            }
        )
    bilan = {"reussis": sum(r["reussi"] for r in resultats), "total": len(resultats), "scenarios": resultats}
    rep = dossier(cfg, "rapports") / "agent"
    rep.mkdir(parents=True, exist_ok=True)
    (rep / "evaluation.json").write_text(json.dumps(bilan, indent=2, ensure_ascii=False), encoding="utf-8")
    return bilan


if __name__ == "__main__":
    print(json.dumps(evaluer(charger_config()), indent=2, ensure_ascii=False))
